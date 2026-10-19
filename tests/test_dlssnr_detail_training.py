"""Optional diagnostics preserve the trained weights, EMA and exact resume."""

import copy
import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_dino import tiny_dino_loss
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import make_args, small_math  # noqa: F401


def strip_details(report):
    stripped = copy.deepcopy(report)
    for cases in stripped.values():
        for case in cases:
            case.pop("detail_diagnostics")
            if "native" in case:
                case["native"].pop("detail_diagnostics")
    return stripped


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora,mode,qat", [(False, "temporal", True), (True, "single_frame", False), (True, "single_frame", True)])
def test_detail_reports_keep_training_and_ema_unchanged_and_resume_exactly(tmp_path, monkeypatch, lora, mode, qat):
    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.native_weight_qat, args.eval_native, args.ema_decay = qat, True, 0.5
    args.loss_profile, args.loss_lowpass_sigma = "frequency_split", 4
    if lora and qat:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
        monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)
        args.numerics_profile, args.fp8_base, args.fp8_scaled, args.network_dropout = "train_experimental", True, True, 0
        args.dino_loss_weight = 0.1
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    plain = args.output_dir / args.output_name
    args.output_name, args.eval_detail_diagnostics = "with_details", True
    train(args)
    folder = args.output_dir / args.output_name
    metadata = json.loads((folder / "run_config.json").read_text())
    assert metadata["config"]["evaluation"]["detail_diagnostics"] is True
    assert "detail_metrics.py" in metadata["identity"]["implementation"]
    report = json.loads((folder / "evaluation/step000002.json").read_text())
    original = json.loads((plain / "evaluation/step000002.json").read_text())
    assert metadata["identity"]["detail_diagnostics"] == report["detail_diagnostics"]
    for variant in ("baseline", "candidate", "ema_candidate"):
        assert strip_details(report[variant]) == original[variant]
        for case in report[variant]["validation"]:
            for runtime in (case, case["native"]):
                details = runtime["detail_diagnostics"]
                assert details["protocol"] == report["detail_diagnostics"]
                assert details["noise_sensitivity"]["rgb_mae"] >= 0
                assert details["valid_rgb_values"] > 0
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    torch.testing.assert_close(raw, load_file(plain / "final" / filename), rtol=0, atol=0)
    torch.testing.assert_close(averaged, load_file(plain / "final/ema" / filename), rtol=0, atol=0)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    original_state = torch.load(plain / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(state["ema"], original_state["ema"])
    torch.testing.assert_close(state["optimizer"]["state"], original_state["optimizer"]["state"], rtol=0, atol=0)
    torch.testing.assert_close(
        state["rank_states"][0]["rng"]["torch"], original_state["rank_states"][0]["rng"]["torch"], rtol=0, atol=0
    )
    args.resume = folder / "state-step000001"
    train(args)
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == report
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    args.eval_detail_diagnostics = False
    with pytest.raises(ValueError, match="identity"):
        train(args)


@pytest.mark.usefixtures("small_math")
def test_detail_diagnostics_work_without_baseline_or_periodic_evaluation(tmp_path):
    args = make_args(tmp_path, evaluate=True)
    args.eval_detail_diagnostics, args.compare_baseline, args.sample_every_n_steps = True, False, 0
    trainer.train_from_args(args)
    folder = args.output_dir / args.output_name / "evaluation"
    assert not (folder / "step000001.json").exists()
    report = json.loads((folder / "step000002.json").read_text())
    assert report["baseline"] is None
    assert "detail_diagnostics" in report["candidate"]["validation"][0]
