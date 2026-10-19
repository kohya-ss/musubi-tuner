"""Measure input retention without changing primary renders or existing metrics."""

from copy import deepcopy
import csv
import json
import random

import numpy as np
import pytest
import torch

from musubi_tuner.dlssnr import control_scan
from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.evaluation import evaluate
from musubi_tuner.training.dlssnr_services import capture_rng, evaluation_mode
from musubi_tuner.training.dlssnr_trainer import _datasets
from test_dlssnr_content_metrics import tiny_content_metric
from test_dlssnr_control_scan import ControlResponseNR, source_manifest
from test_dlssnr_training import SmallNR, make_args


@pytest.mark.parametrize("native", [False, True])
def test_content_evaluation_compares_to_input_not_training_target(tmp_path, native):
    args = make_args(tmp_path, evaluate=True)
    _, datasets = _datasets(build_train_config(args))
    model = SmallNR().eval()
    model.blocks["70"].head.rgb.weight.data.zero_()
    metric = tiny_content_metric()
    report = evaluate(model, datasets, 4, torch.device("cpu"), compare_native=native, content_metric=metric)
    case = report["validation"][0]
    for value in (case, case["native"]) if native else (case,):
        assert value["rgb_mae"] > 0.3
        content = value["content_preservation"]
        assert content["dinov3_patch_mse"] == 0
        assert content["protocol"] == metric.identity
        assert content["valid_rgb_values"] == 48 * 48 * 3


def test_temporal_content_evaluation_does_not_change_other_metrics_or_histories(tmp_path):
    args = make_args(tmp_path, mode="temporal", evaluate=True)
    _, datasets = _datasets(build_train_config(args))
    model = SmallNR().eval()
    options = {"compare_native": True, "detail_diagnostics": True}
    plain = evaluate(model, datasets, 4, torch.device("cpu"), **options)
    metric = tiny_content_metric()
    report = evaluate(model, datasets, 4, torch.device("cpu"), content_metric=metric, **options)
    stripped = deepcopy(report)
    for case in stripped["validation"]:
        for runtime in (case, case["native"]):
            content = runtime.pop("content_preservation")
            assert content["valid_rgb_values"] == 3 * 3 * 48 * 48
            assert content["dinov3_patch_mse"] >= 0
    assert stripped == plain
    assert metric.feature_loss.backend.grad_modes == [False] * 12


def test_content_failure_does_not_leak_rng_or_native_runtime_state(tmp_path, monkeypatch):
    args = make_args(tmp_path, evaluate=True)
    _, datasets = _datasets(build_train_config(args))
    model, metric = SmallNR(), tiny_content_metric()
    backend = metric.feature_loss.backend
    original = backend.dinov3_fwd
    calls = []

    def fail_on_native(image):
        calls.append(1)
        if len(calls) == 3:
            torch.rand(1)
            random.random()
            np.random.rand()
            raise RuntimeError("content features failed")
        return original(image)

    monkeypatch.setattr(backend, "dinov3_fwd", fail_on_native)
    before = capture_rng()
    with pytest.raises(RuntimeError, match="content features failed"), evaluation_mode(model):
        evaluate(model, datasets, 4, torch.device("cpu"), compare_native=True, content_metric=metric)
    assert model.training and not hasattr(model, "native_weight_qat")
    assert all(not child.training for child in metric.modules())
    after = capture_rng()
    torch.testing.assert_close(after.pop("torch"), before.pop("torch"), rtol=0, atol=0)
    torch.testing.assert_close(after.pop("cuda"), before.pop("cuda"), rtol=0, atol=0)
    assert after == before


def test_source_only_scan_adds_content_scores_without_changing_pixels_or_other_metrics(tmp_path):
    model = ControlResponseNR()
    manifest = source_manifest(tmp_path / "data")
    config = control_scan.build_scan_config(48, 48, tone_values=[0, 1], structure_values=[0], compare_native=True)
    plain = control_scan.scan_controls(model, manifest, tmp_path / "plain", config)
    config = control_scan.build_scan_config(
        48, 48, tone_values=[0, 1], structure_values=[0], compare_native=True, content_preservation=True
    )
    metric = tiny_content_metric()
    report = control_scan.scan_controls(model, manifest, tmp_path / "content", config, content_metric=metric)
    assert report["content_preservation"] == metric.identity
    assert "content_metrics.py" in report["implementation"] and "dino_loss.py" in report["implementation"]
    assert report["data_identity"] == plain["data_identity"]
    for row, original in zip(report["measurements"], plain["measurements"]):
        stripped = dict(row)
        assert stripped.pop("dinov3_patch_mse_vs_input") >= 0
        assert stripped == original
        assert (tmp_path / "content" / row["image"]).read_bytes() == (tmp_path / "plain" / original["image"]).read_bytes()
    with (tmp_path / "content/scan_metrics.csv").open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    assert float(records[0]["dinov3_patch_mse_vs_input"]) == report["measurements"][0]["dinov3_patch_mse_vs_input"]
    assert len(metric.feature_loss.backend.inputs) == 8  # Four primary outputs; alternate-noise probes are not scored.


def test_scan_never_silently_skips_a_missing_enabled_content_backend(tmp_path):
    manifest = source_manifest(tmp_path / "data")
    config = control_scan.build_scan_config(48, 48, content_preservation=True)
    with pytest.raises(ValueError, match="content.*backend"):
        control_scan.scan_controls(ControlResponseNR(), manifest, tmp_path / "out", config)
    assert not (tmp_path / "out").exists()


def test_scan_cli_only_loads_content_backend_on_opt_in(tmp_path, monkeypatch):
    from musubi_tuner.dlssnr import infer
    from musubi_tuner.dlssnr.artifacts import save_canonical
    from test_dlssnr_control_scan_cli import arguments, cli

    module = cli()
    monkeypatch.setattr(infer, "NRModel", SmallNR)
    save_canonical(SmallNR(), tmp_path / "model")

    def unexpected():
        pytest.fail("disabled content evaluation loaded a feature model")

    monkeypatch.setattr(module, "create_content_metric", unexpected, raising=False)
    base_args = arguments(tmp_path, ["--tone_values", "0", "--structure_values", "0"])
    plain = module.run(module.setup_parser().parse_args(base_args))
    assert "content_preservation" not in plain
    monkeypatch.setattr(module, "create_content_metric", tiny_content_metric)
    argv = [*base_args, "--output_dir", str(tmp_path / "content"), "--eval_content_preservation"]
    report = module.run(module.setup_parser().parse_args(argv))
    assert "dinov3_patch_mse_vs_input" in report["measurements"][0]
    assert json.loads((tmp_path / "content/scan_report.json").read_text()) == report
