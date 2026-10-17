"""Content evaluation must not perturb training, EMA, exports or exact resume."""

from copy import deepcopy
import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_content_metrics import CONTENT_SETTINGS, content_module, tiny_content_metric
from test_dlssnr_dino import tiny_dino_loss
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_training import make_args, small_math  # noqa: F401


def strip_content(report):
    stripped = deepcopy(report)
    for cases in stripped.values():
        for case in cases:
            case.pop("content_preservation")
            if "native" in case:
                case["native"].pop("content_preservation")
    return stripped


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora,mode,fp8", [(False, "temporal", False), (True, "temporal", False), (True, "single_frame", True)])
def test_content_evaluation_keeps_weights_ema_rng_and_exact_resume_unchanged(tmp_path, monkeypatch, lora, mode, fp8):
    created = []
    module = content_module()
    monkeypatch.setattr(module, "create_dino_loss", tiny_dino_loss)
    monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)

    def create(dino_loss=None):
        metric = module.create_content_metric(dino_loss)
        created.append((metric, deepcopy(metric.state_dict()), dino_loss))
        return metric

    monkeypatch.setattr(trainer, "create_content_metric", create, raising=False)
    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.native_weight_qat, args.eval_native, args.ema_decay = not lora or fp8, True, 0.5
    args.eval_detail_diagnostics, args.base_anchor_weight = True, 0.5
    args.loss_profile, args.loss_lowpass_sigma = "frequency_split", 4
    if fp8:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
        args.numerics_profile, args.fp8_base, args.fp8_scaled, args.network_dropout = "train_experimental", True, True, 0
        args.dino_loss_weight = 0.1
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    plain = args.output_dir / args.output_name
    assert not created
    args.output_name, args.eval_content_preservation = "with_content", True
    train(args)
    folder = args.output_dir / args.output_name
    metadata = json.loads((folder / "run_config.json").read_text())
    assert metadata["config"]["evaluation"]["content_preservation"] is True
    assert metadata["content_preservation"] == metadata["identity"]["content_preservation"] == created[0][0].identity
    assert "content_metrics.py" in metadata["identity"]["implementation"]
    assert "dino_loss.py" in metadata["identity"]["implementation"]
    assert not any("content" in name for name in metadata["trainable_parameters"])
    report = json.loads((folder / "evaluation/step000002.json").read_text())
    original = json.loads((plain / "evaluation/step000002.json").read_text())
    assert report["content_preservation"] == metadata["content_preservation"]
    for variant in ("baseline", "candidate", "ema_candidate"):
        assert strip_content(report[variant]) == original[variant]
        case = report[variant]["validation"][0]
        assert case["content_preservation"]["protocol"] == report["content_preservation"]
        assert case["native"]["content_preservation"]["dinov3_patch_mse"] >= 0
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    torch.testing.assert_close(raw, load_file(plain / "final" / filename), rtol=0, atol=0)
    torch.testing.assert_close(averaged, load_file(plain / "final/ema" / filename), rtol=0, atol=0)
    assert not any("content" in name or "feature_loss" in name for name in raw)
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
    for metric, before, loss_fn in created:
        torch.testing.assert_close(metric.state_dict(), before, rtol=0, atol=0)
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in metric.parameters())
        if fp8:
            assert metric.feature_loss.backend is loss_fn.backend
            assert loss_fn.settings["use_gram"] is True
    args.eval_content_preservation = False
    with pytest.raises(ValueError, match="identity"):
        train(args)
    args.eval_content_preservation = True

    def changed(dino_loss=None):
        from musubi_tuner.dlssnr.dino_loss import NRDinoLoss

        backend = tiny_dino_loss(CONTENT_SETTINGS).backend
        backend.mean.add_(0.1)
        return module.NRContentMetric(NRDinoLoss(backend, CONTENT_SETTINGS, provenance={"provider": "changed"}))

    monkeypatch.setattr(trainer, "create_content_metric", changed)
    with pytest.raises(ValueError, match="identity"):
        train(args)


@pytest.mark.usefixtures("small_math")
def test_content_evaluation_is_disabled_without_loading_and_can_run_final_only(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("content evaluation is disabled")

    monkeypatch.setattr(trainer, "create_content_metric", unexpected, raising=False)
    args = make_args(tmp_path, evaluate=True)
    trainer.train_from_args(args)
    monkeypatch.setattr(trainer, "create_content_metric", tiny_content_metric)
    args.output_name, args.eval_content_preservation = "final_only", True
    args.compare_baseline, args.sample_every_n_steps = False, 0
    trainer.train_from_args(args)
    folder = args.output_dir / args.output_name / "evaluation"
    assert not (folder / "step000001.json").exists()
    report = json.loads((folder / "step000002.json").read_text())
    assert report["baseline"] is None
    assert "content_preservation" in report["candidate"]["validation"][0]
