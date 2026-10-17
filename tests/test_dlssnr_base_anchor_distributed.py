"""Frozen-base anchoring over real CPU/Gloo ranks and uneven bucket tails."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_distributed import _args, _data, _launch
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import small_math  # noqa: F401


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora,temporal", [(False, True), (True, False)])
def test_anchor_ddp_matches_global_accumulation_and_exact_resume(tmp_path, monkeypatch, lora, temporal):
    _data(tmp_path, temporal=temporal)
    for marker in ("base_anchor", "frequency", "shuffle", "ema", "qat"):
        (tmp_path / marker).touch()
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
    assert state["identity"]["base_anchor"]["schema"] == "dlssnr_base_anchor_v1"
    assert not any("base_anchor" in name for name in state["ema"]["shadow"])
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["ema"] == ranks[1]["ema"]
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(_args(tmp_path, "reference", lora=lora))
    reference = tmp_path / "output/reference"
    torch.testing.assert_close(load_file(reference / "final" / filename), raw, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(load_file(reference / "final/ema" / filename), averaged, rtol=1e-5, atol=2e-7)
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    expected = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
    assert "loss/base_anchor" in records[-1]
    for actual, wanted in zip(records, expected):
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
        torch.testing.assert_close(
            resumed["rank_states"][rank]["rng"]["torch"], state["rank_states"][rank]["rng"]["torch"], rtol=0, atol=0
        )


def test_anchor_ddp_dropout_resume_restores_each_rank_rng(tmp_path):
    _data(tmp_path, temporal=True)
    for marker in ("base_anchor", "shuffle", "ema"):
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
    for rank in range(2):
        torch.testing.assert_close(
            resumed["rank_states"][rank]["rng"]["torch"], state["rank_states"][rank]["rng"]["torch"], rtol=0, atol=0
        )


@pytest.mark.parametrize("marker", ["fail_anchor_load", "anchor_rank_mismatch", "fail_anchor_forward"])
def test_anchor_rank_failures_do_not_hang(tmp_path, marker):
    _data(tmp_path)
    for name in ("base_anchor", marker):
        (tmp_path / name).touch()
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    for rank in range(2):
        message = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        if marker == "anchor_rank_mismatch":
            assert "identity" in message
        else:
            assert "base anchor" in message and "failed" in message and "rank 1" in message
    assert not (tmp_path / "output/ddp/final").exists()
