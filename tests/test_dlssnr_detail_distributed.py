"""Main-rank detail evaluation must not desynchronize training or block peer ranks."""

import json

import torch
import toml
from safetensors.torch import load_file

from test_dlssnr_distributed import _data, _launch
from test_dlssnr_ema import assert_ema_state


def detail_data(root):
    _data(root)
    row = json.loads((root / "pairs.jsonl").read_text().splitlines()[0])
    row["sequence_id"] = "validation_scene"
    (root / "validation.jsonl").write_text(json.dumps(row), encoding="utf-8")
    config = toml.load(root / "dataset.toml")
    config["datasets"][0]["validation_manifest"] = "validation.jsonl"
    (root / "dataset.toml").write_text(toml.dumps(config), encoding="utf-8")
    for marker in ("ema", "shuffle", "detail_diagnostics"):
        (root / marker).touch()


def test_main_rank_diagnostics_with_dropout_and_ema_resume_exactly(tmp_path):
    detail_data(tmp_path)
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3)
    assert codes == [0, 0], logs
    folder = tmp_path / "output/ddp"
    raw = load_file(folder / "final/adapter.safetensors")
    averaged = load_file(folder / "final/ema/adapter.safetensors")
    expected = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    report = json.loads((folder / "evaluation/step000002.json").read_text())
    assert "detail_diagnostics" in report
    for variant in ("baseline", "candidate", "ema_candidate"):
        case = report[variant]["validation"][0]
        assert "detail_diagnostics" in case and "detail_diagnostics" in case["native"]
    ranks = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert ranks[0]["ema"] == ranks[1]["ema"]
    codes, logs = _launch(tmp_path, lora=True, dropout=0.3, resume=True)
    assert codes == [0, 0], logs
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == report
    torch.testing.assert_close(load_file(folder / "final/adapter.safetensors"), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema/adapter.safetensors"), averaged, rtol=0, atol=0)
    actual = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(actual["ema"], expected["ema"])
    for left, right in zip(actual["rank_states"], expected["rank_states"]):
        torch.testing.assert_close(left["rng"]["torch"], right["rng"]["torch"], rtol=0, atol=0)


def test_main_rank_diagnostic_failure_reaches_both_ranks(tmp_path):
    detail_data(tmp_path)
    (tmp_path / "fail_detail_eval").touch()
    codes, logs = _launch(tmp_path, failure=True)
    assert codes == [0, 0], logs
    for rank in range(2):
        message = json.loads((tmp_path / f"rank{rank}.json").read_text())["error"]
        assert "rank 0" in message and "diagnostic probe failed" in message
    assert not (tmp_path / "output/ddp/state-step000001/checkpoint_manifest.json").exists()
