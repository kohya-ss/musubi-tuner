"""Randomized-control targets and counters across real CPU/Gloo workers."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_control_randomization_training import assert_nested_equal
from test_dlssnr_distributed import _args, _data, _launch
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import small_math  # noqa: F401


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora,temporal", [(False, True), (True, False)])
def test_control_ddp_matches_global_accumulation_metrics_and_resume(tmp_path, monkeypatch, lora, temporal):
    (tmp_path / "control_randomization").touch()
    _data(tmp_path, temporal=temporal)
    for marker in ("shuffle", "ema", "qat"):
        (tmp_path / marker).touch()
    if temporal:
        from test_dlssnr_dino import tiny_dino_loss

        monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)
        for marker in ("base_anchor", "frequency", "dino"):
            (tmp_path / marker).touch()
        for index in range(5):
            motion = np.load(tmp_path / f"motion{index}.npy")
            motion[0], motion[1] = 0.5, -0.25
            np.save(tmp_path / f"motion{index}.npy", motion)
    np.save(tmp_path / "loss2.npy", np.load(tmp_path / "loss2.npy") * 0.25)
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        (tmp_path / "fp8").touch()
        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    codes, logs = _launch(tmp_path, lora=lora)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["ema"] == ranks[1]["ema"]
    assert all(len(rank["augmentation"]) == 4 for rank in ranks)
    assert any(len(batch) == 1 for rank in ranks for batch in rank["indices"])
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(_args(tmp_path, "reference", lora=lora))
    reference = tmp_path / "output/reference"
    torch.testing.assert_close(load_file(reference / "final" / filename), raw, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(load_file(reference / "final/ema" / filename), averaged, rtol=1e-5, atol=2e-7)
    actual_rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    expected_rows = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
    for actual, expected in zip(actual_rows, expected_rows):
        assert actual["consumed_samples"] == expected["consumed_samples"]
        assert "control/tone_mean" in actual and not any(key.startswith("control/_") for key in actual)
        for key in expected:
            if key.startswith(("control/", "loss/")) or key == "loss":
                assert actual[key] == pytest.approx(expected[key], rel=1e-5, abs=2e-7)
    codes, logs = _launch(tmp_path, lora=lora, resume=True)
    assert codes == [0, 0], logs
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    assert_nested_equal(resumed["optimizer"], state["optimizer"])
    assert_nested_equal(resumed["rank_states"], state["rank_states"])
    for rank in range(2):
        assert json.loads((tmp_path / f"rank{rank}.json").read_text())["augmentation"] == ranks[rank]["augmentation"][2:]


def test_control_ddp_dropout_resume_keeps_rank_rng(tmp_path):
    (tmp_path / "control_randomization").touch()
    _data(tmp_path, temporal=True)
    for marker in ("shuffle", "ema"):
        (tmp_path / marker).touch()
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    raw = load_file(folder / "final/adapter.safetensors")
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3, resume=True)
    assert codes == [0, 0], logs
    torch.testing.assert_close(load_file(folder / "final/adapter.safetensors"), raw, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    assert_nested_equal(resumed["rank_states"], state["rank_states"])


@pytest.mark.parametrize("marker", ["fail_control_reference", "control_reference_mismatch"])
def test_rank_control_reference_failure_aborts_without_a_checkpoint(tmp_path, marker):
    (tmp_path / "control_randomization").touch()
    _data(tmp_path)
    (tmp_path / marker).touch()
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    for rank in range(2):
        error = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        assert "identity" in error if marker.endswith("mismatch") else "control reference failed" in error and "rank 1" in error
    assert not (tmp_path / "output/ddp/state-step000001").exists()
    assert not (tmp_path / "output/ddp/final").exists()
