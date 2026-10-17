"""Only the evaluating rank loads the frozen content model; every rank can resume."""

import json

import pytest
import torch
from safetensors.torch import load_file

from test_dlssnr_detail_distributed import detail_data
from test_dlssnr_distributed import _launch
from test_dlssnr_ema import assert_ema_state


@pytest.mark.parametrize("shared", [False, True])
def test_content_model_is_main_rank_only_and_exact_resume_preserves_all_ranks(tmp_path, shared):
    detail_data(tmp_path)
    (tmp_path / "content_preservation").touch()
    if shared:
        for marker in ("dino", "fp8", "qat"):
            (tmp_path / marker).touch()
    codes, logs = _launch(tmp_path, lora=shared)
    assert codes == [0, 0], logs
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["content_created"] == [{"shared": shared}]
    assert ranks[1]["content_created"] == []
    assert ranks[0]["ema"] == ranks[1]["ema"]
    folder = tmp_path / "output/ddp"
    filename = "adapter.safetensors" if shared else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    report = json.loads((folder / "evaluation/step000002.json").read_text())
    assert state["identity"]["content_preservation"] == report["content_preservation"]
    assert not any("content" in name for name in state["ema"]["shadow"])
    for variant in ("baseline", "candidate", "ema_candidate"):
        case = report[variant]["validation"][0]
        for runtime in (case, case["native"]):
            assert runtime["content_preservation"]["protocol"] == report["content_preservation"]
            assert runtime["content_preservation"]["dinov3_patch_mse"] >= 0
    codes, logs = _launch(tmp_path, lora=shared, resume=True)
    assert codes == [0, 0], logs
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == report
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    for left, right in zip(state["rank_states"], resumed["rank_states"]):
        torch.testing.assert_close(left["rng"]["torch"], right["rng"]["torch"], rtol=0, atol=0)


@pytest.mark.parametrize("marker", ["fail_content_load", "fail_content_eval"])
def test_main_rank_content_failures_reach_peer_without_hanging(tmp_path, marker):
    detail_data(tmp_path)
    (tmp_path / "content_preservation").touch()
    (tmp_path / marker).touch()
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    for rank in range(2):
        message = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        assert "rank 0" in message and "content metric" in message and "failed" in message
    assert not (tmp_path / "output/ddp/state-step000001/checkpoint_manifest.json").exists()
