"""Two-process CPU/Gloo EMA coverage; no real-data or multi-GPU quality claim."""

import json

import pytest
import torch
from safetensors.torch import load_file

from test_dlssnr_distributed import _args, _data, _launch
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import small_math  # noqa: F401


@pytest.mark.parametrize(
    "lora,temporal,fp8,qat,dropout",
    [(False, True, False, True, 0), (True, False, False, False, 0.3), (True, False, False, True, 0), (True, False, True, True, 0)],
)
@pytest.mark.usefixtures("small_math")
def test_distributed_ema_is_identical_across_ranks_and_resumes_exactly(tmp_path, lora, temporal, fp8, qat, dropout):
    from musubi_tuner.training import dlssnr_trainer as trainer

    _data(tmp_path, temporal=temporal)
    for name, enabled in (("ema", True), ("shuffle", True), ("fp8", fp8), ("qat", qat)):
        if enabled:
            (tmp_path / name).touch()
    codes, logs = _launch(tmp_path, lora=lora, dropout=dropout)
    assert codes == [0, 0], logs
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["ema"] == ranks[1]["ema"]
    assert [item["num_updates"] for item in ranks[0]["ema"]] == [1, 2]
    assert ranks[0]["ema"][0]["sha256"] != ranks[0]["ema"][1]["sha256"]
    folder = tmp_path / "output/ddp"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw = load_file(folder / "final" / filename)
    averaged = load_file(folder / "final/ema" / filename)
    original = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert original["ema"]["num_updates"] == original["global_update"] == 2
    if not lora:
        trainer.train_from_args(_args(tmp_path, "reference"))
        torch.testing.assert_close(load_file(tmp_path / "output/reference/final/ema" / filename), averaged, rtol=1e-5, atol=2e-7)
    codes, logs = _launch(tmp_path, lora=lora, dropout=dropout, resume=True)
    assert codes == [0, 0], logs
    resumed_ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    for rank in range(2):
        assert resumed_ranks[rank]["ema"] == ranks[rank]["ema"][1:]
        assert resumed_ranks[rank]["indices"] == ranks[rank]["indices"][2:]
        assert resumed_ranks[rank]["rng"] == ranks[rank]["rng"]
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], original["ema"])
    for actual, expected in zip(resumed["rank_states"], original["rank_states"]):
        torch.testing.assert_close(actual["rng"]["torch"], expected["rng"]["torch"], rtol=0, atol=0)


def test_ema_output_failure_reaches_all_ranks_without_publishing_complete_state(tmp_path):
    _data(tmp_path)
    (tmp_path / "ema").touch()
    (tmp_path / "fail_ema_save").touch()
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    for rank in range(2):
        error = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        assert "EMA output failed" in error and "rank 0" in error
    assert not (tmp_path / "output/ddp/state-step000001/checkpoint_manifest.json").exists()
