"""Paired-seed diagnostics must not alter the primary rollout or its runtime."""

import copy
import json

import pytest
import torch

from musubi_tuner.dlssnr import evaluation
from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy
from musubi_tuner.training.dlssnr_services import evaluation_mode
from musubi_tuner.training.dlssnr_trainer import _datasets
from test_dlssnr_training import SmallNR, make_args


class NoiseProbeNR(SmallNR):
    """Known noise gain through real preprocessing, clipping, blending and history."""

    def __init__(self):
        super().__init__()
        self.noise_gain = 0.005

    def forward(self, features, geometry):
        rgb = 4 * self.noise_gain * features[:, :3] + 0.05 * features[:, 7:10]
        return torch.cat((rgb, torch.zeros_like(rgb[:, :1])), dim=1)


@pytest.mark.parametrize("native", [False, True])
def test_paired_probe_reuses_history_without_advancing_it_or_changing_primary_metrics(tmp_path, monkeypatch, native):
    args = make_args(tmp_path, mode="temporal", evaluate=True)
    manifest = tmp_path / "validation.jsonl"
    row = json.loads(manifest.read_text())
    row["frames"][-1]["reset"] = True
    manifest.write_text(json.dumps(row), encoding="utf-8")
    _, datasets = _datasets(build_train_config(args))
    model = NoiseProbeNR().eval()
    configure_model_runtime(model, default_runtime_policy())
    before = {name: value.clone() for name, value in model.state_dict().items()}
    expected = evaluation.evaluate(model, datasets, 42, torch.device("cpu"), compare_native=native)
    records = []
    real_forward = evaluation.forward_frame

    def observed(model, source, controls, seed, **kwargs):
        result = real_forward(model, source, controls, seed, **kwargs)
        records.append((seed, kwargs["history"], result["next_history"], model.native_weight_qat))
        return result

    monkeypatch.setattr(evaluation, "forward_frame", observed)
    rng = torch.get_rng_state().clone()
    actual = evaluation.evaluate(model, datasets, 42, torch.device("cpu"), compare_native=native, detail_diagnostics=True)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    assert model.runtime_policy == default_runtime_policy()
    stripped = copy.deepcopy(actual)
    for kind, cases in stripped.items():
        for case in cases:
            details = case.pop("detail_diagnostics")
            assert details["noise_sensitivity"]["rgb_mae"] > 0
            assert details["protocol"]["noise_history"] == "shared_primary_history_current_frame_only"
            if native:
                assert case["native"].pop("detail_diagnostics")["protocol"] == details["protocol"]
        assert cases == expected[kind]
    modes = 2 if native else 1
    assert len(records) == 3 * modes * 2
    for frame in range(3):
        for mode in range(modes):
            primary, alternate = records[(frame * modes + mode) * 2 : (frame * modes + mode) * 2 + 2]
            assert primary[0] != alternate[0]
            assert primary[0] ^ alternate[0] == details["protocol"]["noise_seed_xor"]
            assert primary[3] == alternate[3] == bool(mode)
            assert primary[1] is alternate[1]
            if frame in (0, 2):
                assert primary[1] is None
            else:
                previous = records[((frame - 1) * modes + mode) * 2]
                assert primary[1].data_ptr() == previous[2].data_ptr()
    assert evaluation.evaluate(model, datasets, 42, torch.device("cpu"), compare_native=native, detail_diagnostics=True) == actual


def test_noise_sensitivity_tracks_known_gain_and_detects_noise_channel_collapse(tmp_path):
    args = make_args(tmp_path, evaluate=True)
    _, datasets = _datasets(build_train_config(args))
    model = NoiseProbeNR().eval()
    configure_model_runtime(model, default_runtime_policy())

    def measure():
        return evaluation.evaluate(model, datasets, 29, torch.device("cpu"), detail_diagnostics=True)["validation"][0][
            "detail_diagnostics"
        ]

    original = measure()["noise_sensitivity"]
    model.noise_gain *= 0.5
    reduced = measure()["noise_sensitivity"]
    assert original["rgb_mae"] > 0 and original["preclamp_mae"] > 0
    for name in ("rgb_mae", "preclamp_mae"):
        assert reduced[name] == pytest.approx(original[name] * 0.5, rel=1e-4)
    for band, rms in original["high_frequency_rms"].items():
        assert rms > 0 and reduced["high_frequency_rms"][band] == pytest.approx(rms * 0.5, rel=1e-4)
    model.noise_gain = 0
    collapsed = measure()["noise_sensitivity"]
    assert collapsed["rgb_mae"] == collapsed["preclamp_mae"] == 0
    assert all(value == 0 for value in collapsed["high_frequency_rms"].values())


def test_native_detail_metrics_match_actually_requantized_weights(tmp_path):
    from musubi_tuner.dlssnr.native import quantize_tensor
    from musubi_tuner.dlssnr.weight_quantization import native_storage_kinds
    from test_dlssnr_fp8 import TinyFP8NR

    args = make_args(tmp_path, evaluate=True, mode="temporal")
    _, datasets = _datasets(build_train_config(args))
    model = TinyFP8NR().eval()
    configure_model_runtime(model, default_runtime_policy())
    exported = TinyFP8NR().eval()
    kinds = native_storage_kinds()
    exported.load_state_dict(
        {name: torch.from_numpy(quantize_tensor(value.numpy(), kinds[name])) for name, value in model.state_dict().items()}
    )
    expected = evaluation.evaluate(exported, datasets, 42, torch.device("cpu"), detail_diagnostics=True)
    actual = evaluation.evaluate(model, datasets, 42, torch.device("cpu"), compare_native=True, detail_diagnostics=True)
    assert actual["validation"][0]["native"] == expected["validation"][0]


def test_invalid_alternate_output_aborts_and_restores_runtime_and_mode(tmp_path, monkeypatch):
    args = make_args(tmp_path, evaluate=True)
    _, datasets = _datasets(build_train_config(args))
    model = NoiseProbeNR().train()
    configure_model_runtime(model, default_runtime_policy())
    real_forward = evaluation.forward_frame
    calls = 0

    def invalid(model, *args, **kwargs):
        nonlocal calls
        calls += 1
        result = real_forward(model, *args, **kwargs)
        if calls == 4:
            result["raw_head"] = torch.full_like(result["raw_head"], float("nan"))
        return result

    monkeypatch.setattr(evaluation, "forward_frame", invalid)
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="non-finite evaluation output.*alternate"):
        with evaluation_mode(model):
            evaluation.evaluate(model, datasets, 42, torch.device("cpu"), compare_native=True, detail_diagnostics=True)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert model.training and not model.native_weight_qat
    assert model.runtime_policy == default_runtime_policy()


def test_disabled_diagnostics_do_not_add_forwards_or_report_fields(tmp_path, monkeypatch):
    args = make_args(tmp_path, evaluate=True)
    _, datasets = _datasets(build_train_config(args))
    model = NoiseProbeNR().eval()
    calls = []
    real_forward = evaluation.forward_frame

    def observed(*args, **kwargs):
        calls.append(1)
        return real_forward(*args, **kwargs)

    monkeypatch.setattr(evaluation, "forward_frame", observed)
    report = evaluation.evaluate(model, datasets, 42, torch.device("cpu"), detail_diagnostics=False)
    assert len(calls) == 1
    assert "detail_diagnostics" not in report["validation"][0]
