"""Pixel-weighted frequency losses across bucket tails and CPU/Gloo ranks."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_distributed import _args, _data, _launch
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import small_math  # noqa: F401


@pytest.mark.parametrize(
    "lora,temporal,qat", [(False, False, False), (False, True, True), (True, False, True), (True, True, False)]
)
@pytest.mark.usefixtures("small_math")
def test_frequency_ddp_matches_accumulation_and_resumes_with_ema(tmp_path, lora, temporal, qat):
    _data(tmp_path, temporal=temporal)
    for marker in ("frequency", "shuffle", "ema"):
        (tmp_path / marker).touch()
    if qat:
        (tmp_path / "qat").touch()
    for index in range(5):
        target = np.load(tmp_path / f"target{index}.npy")
        np.save(tmp_path / f"target{index}.npy", target[:, ::-1, :] * 0.8 + 0.1)
        if temporal:
            motion = np.load(tmp_path / f"motion{index}.npy")
            motion[0], motion[1] = 0.5, -0.25
            np.save(tmp_path / f"motion{index}.npy", motion)
    codes, logs = _launch(tmp_path, lora=lora)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["ema"] == ranks[1]["ema"]
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(_args(tmp_path, "reference", lora=lora))
    reference = tmp_path / "output/reference"
    torch.testing.assert_close(load_file(reference / "final" / filename), raw, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(load_file(reference / "final/ema" / filename), averaged, rtol=1e-5, atol=2e-7)
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    expected = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
    for actual, wanted in zip(records, expected):
        assert actual["consumed_samples"] == wanted["consumed_samples"]
        assert "loss/lowpass_pre" in actual and "loss/input_edge" in actual
        if temporal:
            assert "loss/lowpass_temporal" in actual
        for name in wanted:
            if name == "loss" or name.startswith("loss/"):
                assert actual[name] == pytest.approx(wanted[name], rel=1e-5, abs=2e-7)
    codes, logs = _launch(tmp_path, lora=lora, resume=True)
    assert codes == [0, 0], logs
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    for rank in range(2):
        actual_rank = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert actual_rank["indices"] == ranks[rank]["indices"][2:]
        assert actual_rank["ema"] == ranks[rank]["ema"][1:]
        torch.testing.assert_close(resumed["rank_states"][rank]["rng"]["torch"], state["rank_states"][rank]["rng"]["torch"])
