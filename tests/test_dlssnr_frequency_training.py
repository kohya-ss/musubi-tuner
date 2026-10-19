"""Frequency objectives through the real loader, optimizer and checkpoint lifecycle."""

import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import make_args, small_math  # noqa: F401


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
@pytest.mark.usefixtures("small_math")
def test_frequency_profile_with_qat_and_ema_resumes_and_records_its_objective(tmp_path, monkeypatch, lora, mode):
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.loss_profile, args.loss_lowpass_sigma, args.loss_edge = "frequency_split", 4, 0.2
    args.native_weight_qat = args.eval_native = True
    args.ema_decay = 0.5
    if lora:
        args.numerics_profile, args.fp8_base, args.fp8_scaled, args.network_dropout = "train_experimental", True, True, 0
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    assert all("loss/lowpass_pre" in row and "loss/input_edge" in row for row in records), "The runner ignored the loss profile"
    for row in records:
        expected = row["loss/lowpass_pre"] * args.loss_pre + row["loss/lowpass_out"] * args.loss_out
        expected += row["loss/input_edge"] * args.loss_edge
        if mode == "temporal":
            expected += row["loss/lowpass_temporal"] * args.loss_temporal
        assert row["loss"] == pytest.approx(expected, rel=1e-6, abs=1e-7)
        assert "loss/edge" not in row and "loss/out" not in row
    config = json.loads((folder / "run_config.json").read_text())["config"]
    assert config["loss_profile"] == {"name": "frequency_split", "lowpass_sigma": 4}
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    evaluation = json.loads((folder / "evaluation/step000002.json").read_text())
    assert "ema_candidate" in evaluation
    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    torch.testing.assert_close(resumed["optimizer"]["state"], state["optimizer"]["state"], rtol=0, atol=0)
    assert resumed["scheduler"] == state["scheduler"]
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == evaluation
    args.loss_lowpass_sigma = 6
    with pytest.raises(ValueError, match="identity"):
        train(args)
    args.loss_profile, args.loss_lowpass_sigma = "pixel", None
    with pytest.raises(ValueError, match="identity"):
        train(args)
