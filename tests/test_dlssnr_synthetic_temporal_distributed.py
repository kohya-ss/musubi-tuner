"""Mixed real/synthetic training, weighted controls and coordinated Gloo failures."""

import json

import numpy as np
import pytest
import toml
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_buckets import write_pairs
from test_dlssnr_control_randomization_training import assert_nested_equal
from test_dlssnr_distributed import _args, _data, _launch
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import small_math  # noqa: F401


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora,randomize", [(False, False), (True, True)])
def test_mixed_synthetic_ddp_matches_accumulation_and_resumes_across_tails(tmp_path, monkeypatch, lora, randomize):
    for marker in ("synthetic_temporal", "shuffle", "ema"):
        (tmp_path / marker).touch()
    if randomize:
        (tmp_path / "control_randomization").touch()
    _data(tmp_path)
    np.save(tmp_path / "loss2.npy", np.load(tmp_path / "loss2.npy") * 0.25)
    real = tmp_path / "real"
    real.mkdir()
    write_pairs(real, [(64, 48), (64, 48)], frames=3)
    data = toml.load(tmp_path / "dataset.toml")
    data["datasets"].append(
        {"train_manifest": "real/pairs.jsonl", "batch_size": 1, **({"nr_controls_mode": "fixed"} if randomize else {})}
    )
    (tmp_path / "dataset.toml").write_text(toml.dumps(data), encoding="utf-8")
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject
        from test_dlssnr_dino import tiny_dino_loss

        for marker in ("fp8", "qat", "frequency", "base_anchor", "dino"):
            (tmp_path / marker).touch()
        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
        monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)
    codes, logs = _launch(tmp_path, lora=lora)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["ema"] == ranks[1]["ema"]
    assert all(len(bucket["resolution"]) == 2 and bucket["frames"] == 3 for bucket in state["identity"]["bucket_plan"]["buckets"])
    assert any(len(indices) == 1 for rank in ranks for indices in rank["indices"])
    observed = [value for rank in ranks for value in rank["augmentation"]]
    assert {item["epoch"] for item in observed} == {0, 1}
    if not randomize:
        assert {item["temporal_support"] for item in observed} == {None, "joint_loss_mask"}
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(_args(tmp_path, "reference", lora=lora))
    reference = tmp_path / "output/reference"
    torch.testing.assert_close(load_file(reference / "final" / filename), raw, rtol=1e-5, atol=2e-7)
    torch.testing.assert_close(load_file(reference / "final/ema" / filename), averaged, rtol=1e-5, atol=2e-7)
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    expected = [json.loads(line) for line in (reference / "metrics.jsonl").read_text().splitlines()]
    for actual, wanted in zip(records, expected):
        assert actual["consumed_samples"] == wanted["consumed_samples"]
        for name in wanted:
            if name.startswith(("control/", "loss/")) or name == "loss":
                assert actual[name] == pytest.approx(wanted[name], rel=1e-5, abs=2e-7)
    codes, logs = _launch(tmp_path, lora=lora, resume=True)
    assert codes == [0, 0], logs
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    assert_nested_equal(resumed["optimizer"], state["optimizer"])
    assert_nested_equal(resumed["rank_states"], state["rank_states"])
    for rank in range(2):
        actual = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert actual["augmentation"] == ranks[rank]["augmentation"][2:]


@pytest.mark.parametrize("marker", ["fail_synthetic_sample", "fail_control_reference"])
def test_synthetic_rank_failure_aborts_both_workers_without_complete_state(tmp_path, marker):
    for name in ("synthetic_temporal", "control_randomization", marker):
        (tmp_path / name).touch()
    _data(tmp_path)
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    expected = "synthetic sample failed" if marker == "fail_synthetic_sample" else "control reference failed"
    for rank in range(2):
        error = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        assert expected in error and "rank 1" in error
    assert not (tmp_path / "output/ddp/state-step000001").exists()
    assert not (tmp_path / "output/ddp/final").exists()
