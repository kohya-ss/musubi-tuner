"""Anchor lifetime, normalization, export ownership and exact resume."""

import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dlssnr import training_step
from musubi_tuner.dlssnr.base_anchor import NRBaseAnchor
from musubi_tuner.dlssnr.fp8 import canonical_tensor_sha256, iter_canonical_tensors
from musubi_tuner.dlssnr.losses import _rho
from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_base_anchor import batch_and_seeds
from test_dlssnr_dino import tiny_dino_loss
from test_dlssnr_ema import assert_ema_state
from test_dlssnr_frequency_loss import PROFILE, analytical_forward, clip_batch, still_batch  # noqa: F401
from test_dlssnr_training import SmallNR, make_args, small_inject, small_math  # noqa: F401


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_anchor_qat_dino_ema_resume_retains_original_reference(tmp_path, monkeypatch, lora, mode):
    created = []

    def create(model, network):
        reference = NRBaseAnchor(model, network)
        created.append((reference, model))
        return reference

    monkeypatch.setattr(trainer, "NRBaseAnchor", create, raising=False)
    monkeypatch.setattr(trainer, "create_dino_loss", tiny_dino_loss)
    if lora:
        from musubi_tuner.networks import lora_dlssnr
        from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

        monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
        monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.loss_profile, args.loss_lowpass_sigma = "frequency_split", 4
    args.base_anchor_weight, args.ema_decay, args.dino_loss_weight = 2.0, 0.5, 0.1
    args.native_weight_qat = args.eval_native = True
    if lora:
        args.numerics_profile, args.fp8_base, args.fp8_scaled, args.network_dropout = "train_experimental", True, True, 0
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    records = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
    for row in records:
        assert row["loss/base_anchor"] >= 0
        assert row["loss/base_anchor_weighted"] == pytest.approx(args.base_anchor_weight * row["loss/base_anchor"])
        expected = args.loss_pre * row["loss/lowpass_pre"] + args.loss_out * row["loss/lowpass_out"]
        expected += args.loss_edge * row["loss/input_edge"] + row["loss/dino_weighted"] + row["loss/base_anchor_weighted"]
        if mode == "temporal":
            expected += args.loss_temporal * row["loss/lowpass_temporal"]
        assert row["loss"] == pytest.approx(expected, rel=1e-6)
    assert records[0]["loss/base_anchor"] == 0
    metadata = json.loads((folder / "run_config.json").read_text())
    assert metadata["base_anchor"] == metadata["identity"]["base_anchor"] == created[0][0].identity
    assert "base_anchor.py" in metadata["identity"]["implementation"]
    assert not any("base_anchor" in name for name in metadata["trainable_parameters"])
    filename = "adapter.safetensors" if lora else "model.safetensors"
    raw, averaged = load_file(folder / "final" / filename), load_file(folder / "final/ema" / filename)
    assert not any("reference" in name or "base_anchor" in name for name in raw)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert not any("base_anchor" in name for name in state["ema"]["shadow"])
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
    assert len(created) == 2
    for reference, model in created:
        assert not reference.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in reference.parameters())
        base = model if lora else reference.reference
        assert canonical_tensor_sha256(iter_canonical_tensors(base)) == reference.identity["base_parameters_sha256"]
        assert reference.identity == metadata["base_anchor"]
    args.base_anchor_weight = 3
    with pytest.raises(ValueError, match="identity"):
        train(args)
    args.base_anchor_weight = 2

    def changed_reference(model, network):
        reference = create(model, network)
        reference.identity["base_parameters_sha256"] = "changed"
        return reference

    monkeypatch.setattr(trainer, "NRBaseAnchor", changed_reference)
    with pytest.raises(ValueError, match="identity"):
        train(args)


@pytest.mark.usefixtures("small_math")
def test_anchor_disabled_does_not_construct_or_execute_a_reference(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A disabled anchor must not allocate or run a teacher")

    monkeypatch.setattr(trainer, "NRBaseAnchor", unexpected, raising=False)
    args = make_args(tmp_path)
    trainer.train_from_args(args)
    metadata = json.loads((args.output_dir / args.output_name / "run_config.json").read_text())
    assert "base_anchor" not in metadata and "base_anchor" not in metadata["identity"]
    batch, seeds = batch_and_seeds()
    module = trainer.NRTrainModule(SmallNR(), {"out": 1}, base_anchor=unexpected)
    module(batch, seeds)


@pytest.mark.usefixtures("small_math")
def test_lora_dropout_anchor_resume_is_bit_exact(tmp_path):
    args = make_args(tmp_path, lora=True, mode="temporal")
    args.base_anchor_weight = 2.0
    trainer.train_lora_from_args(args)
    folder = args.output_dir / args.output_name
    expected = load_file(folder / "final/adapter.safetensors")
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    args.resume = folder / "state-step000001"
    trainer.train_lora_from_args(args)
    torch.testing.assert_close(load_file(folder / "final/adapter.safetensors"), expected, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    torch.testing.assert_close(resumed["rank_states"][0]["rng"]["torch"], state["rank_states"][0]["rng"]["torch"], rtol=0, atol=0)


@pytest.mark.usefixtures("analytical_forward")
@pytest.mark.parametrize("profile", [None, PROFILE])
def test_anchor_global_mass_matches_microbatch_gradients_with_empty_local_mask(profile):
    prediction = torch.linspace(0.1, 0.9, 3 * 3 * 12 * 16).reshape(3, 3, 12, 16).requires_grad_()
    reference = prediction.detach().roll(2, -1)
    mask = torch.ones(3, 1, 12, 16)
    mask[0] = 0
    mask[1, :, :, :8] = 0.2
    batch = still_batch(reference, reference, mask)
    weights = {"pre": 1, "out": 0.5, "edge": 0.1, "temporal": 0, "base_anchor": 2}
    counts = training_step.loss_denominators(batch, 0, loss_profile=profile, include_base_anchor=True)
    loss, _ = training_step.training_loss(prediction, batch, [4] * 3, weights, loss_profile=profile, base_reference=reference)
    expected_gradient = torch.autograd.grad(loss, prediction)[0]
    micro_loss = 0
    for index in range(3):
        micro = {name: value[index : index + 1] for name, value in batch.items()}
        value, _ = training_step.training_loss(
            prediction[index : index + 1],
            micro,
            [4],
            weights,
            normalizers=counts,
            loss_profile=profile,
            base_reference=reference[index : index + 1],
        )
        micro_loss = micro_loss + value
    torch.testing.assert_close(micro_loss, loss)
    torch.testing.assert_close(torch.autograd.grad(micro_loss, prediction)[0], expected_gradient)


@pytest.mark.usefixtures("analytical_forward")
def test_temporal_anchor_excludes_burn_in_and_preserves_time_major_order():
    prediction = torch.linspace(0.1, 0.9, 2 * 3 * 3 * 12 * 16).reshape(2, 3, 3, 12, 16).requires_grad_()
    reference = prediction.detach().roll(2, -1)
    mask = torch.ones(2, 3, 1, 12, 16)
    mask[0, 1] = 0.1
    batch = clip_batch(reference, reference, mask)
    rendered_reference = torch.cat(list(reference[:, 1:].unbind(1)))
    loss, _ = training_step.training_loss(
        prediction, batch, [[1, 2, 3]] * 2, {"base_anchor": 1}, 1, base_reference=rendered_reference
    )
    expected = (_rho(prediction[:, 1:] - reference[:, 1:]) * mask[:, 1:]).sum() / (3 * mask[:, 1:].sum())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert prediction.grad[:, 0].eq(0).all()
    assert prediction.grad[:, 1:].abs().sum() > 0


def test_checkpoint_replay_keeps_adapters_enabled_after_reference_pass():
    from musubi_tuner.dlssnr.model import _run_window_stage
    from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA
    from test_dlssnr_checkpointing import _blocks

    blocks = _blocks()
    network = DLSSNRLoRA()
    for index in range(2):
        network.add(f"{index}.ffn.fc1.weight", blocks[str(index)].ffn.fc1, 2, 2, 0.3)
    for adapter in network.adapters:
        torch.nn.init.normal_(adapter.lora_up, std=0.02)
    blocks.requires_grad_(False)
    source = torch.rand(1, 32, 9, 11)
    results = []
    for checkpointing in (False, True):
        network.zero_grad(set_to_none=True)
        torch.manual_seed(71)
        with torch.no_grad(), network.disable_adapters():
            reference = _run_window_stage(blocks, source, range(2), checkpointing=checkpointing)
        output = _run_window_stage(blocks, source, range(2), checkpointing=checkpointing)
        (output - reference).square().mean().backward()
        gradients = {name: parameter.grad.clone() for name, parameter in network.named_parameters()}
        assert any(value.abs().sum() > 0 for value in gradients.values())
        assert network.enabled
        results.append((output.detach(), gradients, torch.random.get_rng_state().clone()))
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
