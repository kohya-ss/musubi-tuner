"""DINO supervision through NR accumulation, temporal clips, QAT and checkpoints."""

import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dlssnr import training_step
from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_dino import dino_module, tiny_dino_loss
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_frequency_loss import PROFILE, analytical_forward, clip_batch, still_batch  # noqa: F401
from test_dlssnr_training import make_args, small_math  # noqa: F401


@pytest.mark.usefixtures("analytical_forward")
@pytest.mark.parametrize("profile", [None, PROFILE])
def test_dino_uses_global_valid_rgb_mass_across_microbatches(profile):
    generator = torch.Generator().manual_seed(29)
    prediction = torch.rand(3, 3, 16, 24, generator=generator, requires_grad=True)
    target = torch.rand(3, 3, 16, 24, generator=generator)
    mask = torch.ones(3, 1, 16, 24)
    mask[0] = 0
    mask[1, :, :8] = 0.2
    batch = still_batch(target * 0.9, target, mask)
    weights = {"pre": 1, "out": 0.5, "edge": 0.1, "temporal": 0, "dino": 2}
    dino = tiny_dino_loss()
    counts = training_step.loss_denominators(batch, 0, loss_profile=profile, include_dino=True)
    assert counts["dino"] == counts["pre"]
    assert counts["dino"] == pytest.approx(3 * float(mask.sum()))
    actual, metrics = training_step.training_loss(prediction, batch, [4] * 3, weights, loss_profile=profile, dino_loss=dino)
    base, _ = training_step.training_loss(prediction, batch, [4] * 3, {**weights, "dino": 0}, loss_profile=profile)
    expected_dino = (dino(prediction, target, mask) * mask.sum((1, 2, 3))).sum() / mask.sum()
    torch.testing.assert_close(actual, base + weights["dino"] * expected_dino)
    assert metrics["loss/dino"] == pytest.approx(float(expected_dino.detach()))
    assert metrics["loss/dino_weighted"] == pytest.approx(float(expected_dino.detach()) * weights["dino"])
    expected_grad = torch.autograd.grad(actual, prediction)[0]
    micro_loss = 0
    for index in range(3):
        micro = {name: value[index : index + 1] for name, value in batch.items()}
        value, _ = training_step.training_loss(
            prediction[index : index + 1], micro, [4], weights, normalizers=counts, loss_profile=profile, dino_loss=dino
        )
        micro_loss = micro_loss + value
    torch.testing.assert_close(micro_loss, actual)
    torch.testing.assert_close(torch.autograd.grad(micro_loss, prediction)[0], expected_grad)


@pytest.mark.usefixtures("analytical_forward")
def test_temporal_dino_uses_only_supervised_frames_in_matching_order():
    generator = torch.Generator().manual_seed(29)
    prediction = torch.rand(2, 3, 3, 16, 24, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 3, 16, 24, generator=generator)
    mask = torch.rand(2, 3, 1, 16, 24, generator=generator)
    batch = clip_batch(target * 0.9, target, mask)
    dino = tiny_dino_loss()
    loss, metrics = training_step.training_loss(prediction, batch, [[1, 2, 3]] * 2, {"dino": 1}, 1, dino_loss=dino)
    masses = mask[:, 1:].sum((2, 3, 4))
    expected = (
        sum((dino(prediction[:, index], target[:, index], mask[:, index]) * masses[:, index - 1]).sum() for index in (1, 2))
        / masses.sum()
    )
    torch.testing.assert_close(loss, expected)
    assert metrics["loss/dino"] > 0
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert gradient[:, 0].eq(0).all() and gradient[:, 1:].abs().sum() > 0


@pytest.mark.usefixtures("analytical_forward")
def test_enabled_dino_never_silently_skips_a_missing_backend():
    image = torch.zeros(1, 3, 16, 24)
    with pytest.raises(ValueError, match="DINO"):
        training_step.training_loss(image, still_batch(image, image), [4], {"out": 1, "dino": 0.1})


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_dino_qat_ema_resumes_without_exporting_or_training_teacher(tmp_path, monkeypatch, lora, mode):
    created = []

    def create(settings):
        loss_fn = tiny_dino_loss(settings)
        created.append(loss_fn)
        return loss_fn

    monkeypatch.setattr(trainer, "create_dino_loss", create)
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.loss_profile, args.loss_lowpass_sigma = "frequency_split", 4
    args.native_weight_qat = args.eval_native = True
    args.ema_decay, args.dino_loss_weight = 0.5, 0.1
    if lora:
        args.numerics_profile, args.fp8_base, args.fp8_scaled, args.network_dropout = "train_experimental", True, True, 0
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    for row in records:
        assert row["loss/dino"] > 0
        assert row["loss/dino_weighted"] == pytest.approx(row["loss/dino"] * args.dino_loss_weight)
        expected = row["loss/lowpass_pre"] * args.loss_pre + row["loss/lowpass_out"] * args.loss_out
        expected += row["loss/input_edge"] * args.loss_edge + row["loss/dino_weighted"]
        if mode == "temporal":
            expected += row["loss/lowpass_temporal"] * args.loss_temporal
        assert row["loss"] == pytest.approx(expected, rel=1e-6, abs=1e-7)
    metadata = json.loads((folder / "run_config.json").read_text())
    assert metadata["identity"]["dino_loss"] == metadata["dino_loss"] == created[0].identity
    assert "dino_loss.py" in metadata["identity"]["implementation"]
    assert not any("dino" in name for name in metadata["trainable_parameters"])
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    assert not any("dino" in name for name in raw)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert not any("dino" in name for name in state["ema"]["shadow"])
    evaluation = json.loads((folder / "evaluation/step000002.json").read_text())
    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    torch.testing.assert_close(resumed["optimizer"]["state"], state["optimizer"]["state"], rtol=0, atol=0)
    assert resumed["scheduler"] == state["scheduler"]
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == evaluation
    expected_teacher = tiny_dino_loss().state_dict()
    for loss_fn in created:
        assert loss_fn.backend.inputs
        assert all(not module.training for module in loss_fn.modules())
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in loss_fn.parameters())
        torch.testing.assert_close(loss_fn.state_dict(), expected_teacher, rtol=0, atol=0)
    args.dino_loss_resize = 384
    with pytest.raises(ValueError, match="identity"):
        train(args)
    args.dino_loss_resize = None

    def changed_weights(settings):
        backend = tiny_dino_loss(settings).backend
        with torch.no_grad():
            backend.mean.add_(0.1)
        return dino_module().NRDinoLoss(backend, settings, provenance={"provider": "synthetic_test_only"})

    monkeypatch.setattr(trainer, "create_dino_loss", changed_weights)
    with pytest.raises(ValueError, match="identity"):
        train(args)


@pytest.mark.usefixtures("small_math")
def test_default_training_does_not_load_the_optional_dino_backend(tmp_path, monkeypatch):
    def unexpected(settings):
        pytest.fail("Disabled DINO must not load weights or optional dependencies")

    monkeypatch.setattr(trainer, "create_dino_loss", unexpected)
    args = make_args(tmp_path)
    trainer.train_from_args(args)
    metadata = json.loads((args.output_dir / args.output_name / "run_config.json").read_text())
    assert "dino_loss" not in metadata and "dino_loss" not in metadata["identity"]
