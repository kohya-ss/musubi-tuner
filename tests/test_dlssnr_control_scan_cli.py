"""Exercise CLI parsing, canonical artifact loading and scan publication."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from musubi_tuner.dlssnr import infer
from musubi_tuner.dlssnr.artifacts import save_canonical
from musubi_tuner.dlssnr.identity import file_sha256
from musubi_tuner.dlssnr.runtime import default_runtime_policy
from test_dlssnr_control_scan import source_manifest
from test_dlssnr_training import SmallNR


SCRIPT = Path(__file__).resolve().parents[1] / "tools/scan_dlssnr_controls.py"


def cli():
    spec = importlib.util.spec_from_file_location("scan_dlssnr_controls", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arguments(tmp_path, extra=()):
    manifest = source_manifest(tmp_path / "data")
    return [
        "--model_dir",
        str(tmp_path / "model"),
        "--sample_manifest",
        str(manifest),
        "--output_dir",
        str(tmp_path / "scan"),
        "--bucket_width",
        "48",
        "--bucket_height",
        "48",
        "--device",
        "cpu",
        *extra,
    ]


def test_command_help_runs_without_weights_or_optional_feature_models():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        env={**os.environ, "PYTHONPATH": str(SCRIPT.parents[1] / "src")},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "--tone_values" in result.stdout and "--eval_native" in result.stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--tone_values", "nan"],
        ["--structure_values", "1.01"],
        ["--tone_values", "0.5", "0.50001"],
        ["--bucket_width", "0"],
        ["--lowpass_sigma", "0"],
        ["--seed", "-1"],
        ["--nr_style", "-1"],
    ],
)
def test_invalid_cli_settings_fail_before_trying_to_load_missing_weights(tmp_path, extra):
    module = cli()
    args = module.setup_parser().parse_args(arguments(tmp_path, extra))
    with pytest.raises(ValueError):
        module.run(args)
    assert not args.output_dir.exists()


def test_existing_scan_or_distributed_launcher_is_rejected_before_loading_weights(tmp_path, monkeypatch):
    module = cli()
    args = module.setup_parser().parse_args(arguments(tmp_path))
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single.process"):
        module.run(args)
    monkeypatch.setenv("WORLD_SIZE", "1")
    args.output_dir.mkdir()
    with pytest.raises(FileExistsError):
        module.run(args)


def test_missing_manifest_fails_before_loading_model_weights(tmp_path):
    module = cli()
    args = module.setup_parser().parse_args(arguments(tmp_path, ["--sample_manifest", str(tmp_path / "missing.jsonl")]))
    with pytest.raises(FileNotFoundError, match="sample manifest"):
        module.run(args)
    assert not args.output_dir.exists()


@pytest.mark.parametrize("override", [False, True])
@pytest.mark.parametrize("save_precision", ["native", "fp32"])
def test_cli_uses_real_canonical_loader_and_records_effective_runtime(tmp_path, monkeypatch, capsys, override, save_precision):
    monkeypatch.setattr(infer, "NRModel", SmallNR)
    torch.manual_seed(5)
    policy = {**default_runtime_policy(), "numerics_profile": "train_experimental", "attention_backend": "sdpa"}
    model_dir = tmp_path / "model"
    save_canonical(SmallNR(), model_dir, metadata={"runtime_policy": policy}, save_precision=save_precision)
    source_sha = file_sha256(model_dir / "model.safetensors")
    extra = [
        "--tone_values",
        "0",
        "0.5",
        "--structure_values",
        "0.25",
        "--eval_native",
        "--nr_style",
        "2",
        "--nr_skin",
        "0.25",
        "--no-nr_auto_mask",
        "--seed",
        "21",
        "--lowpass_sigma",
        "4",
    ]
    if override:
        extra += ["--numerics_profile", "train_surrogate", "--attention_backend", "native"]
    cli().main(arguments(tmp_path, extra))
    folder = tmp_path / "scan"
    report = json.loads((folder / "scan_report.json").read_text())
    assert len(report["measurements"]) == 4
    assert report["config"]["seed"] == 21 and report["config"]["lowpass_sigma"] == 4
    assert report["source_identity"]["model_sha256"] == source_sha == file_sha256(model_dir / "model.safetensors")
    assert report["runtime_provenance"]["saved_policy"] == policy
    assert report["runtime_provenance"]["matches_saved_policy"] is not override
    assert report["runtimes"]["configured"] == (default_runtime_policy() if override else policy)
    assert report["runtimes"]["native"]["native_weight_qat"] is True
    assert report["runtimes"]["native"]["attention_backend"] == "native"
    assert all(row["auto_mask"] is False and row["style"] == 2 for row in report["measurements"])
    configured = [row["rgb_mae_vs_input"] for row in report["measurements"] if row["runtime"] == "configured"]
    native = [row["rgb_mae_vs_input"] for row in report["measurements"] if row["runtime"] == "native"]
    if save_precision == "native":
        assert configured == native
    else:
        assert any(abs(left - right) > 1e-9 for left, right in zip(configured, native))
    assert "scan_report.json" in capsys.readouterr().out
