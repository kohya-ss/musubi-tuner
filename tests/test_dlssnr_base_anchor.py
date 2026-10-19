"""Base retention uses matched conditioning and a genuinely frozen reference."""

from copy import deepcopy
import random

import numpy as np
import pytest
import torch

from musubi_tuner.dlssnr import training_step
from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from musubi_tuner.dlssnr.losses import _rho
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy, with_native_weight_qat
from musubi_tuner.training.dlssnr_services import capture_rng
from test_dlssnr_config import make_args
from test_dlssnr_frequency_loss import PROFILE, analytical_forward, still_batch  # noqa: F401
from test_dlssnr_training import SmallNR, small_inject


def anchor(model, network=None):
    from musubi_tuner.dlssnr.base_anchor import NRBaseAnchor

    return NRBaseAnchor(model, network)


@pytest.mark.parametrize("lora", [False, True])
def test_anchor_is_opt_in_and_its_weight_is_bound_to_config(tmp_path, lora):
    args = make_args(tmp_path, lora=lora)
    assert getattr(args, "base_anchor_weight", None) == 0
    original = build_train_config(args, lora=lora)
    assert "base_anchor" not in original["loss"]
    args.base_anchor_weight = 0.2
    enabled = build_train_config(args, lora=lora)
    assert enabled["loss"]["base_anchor"] == 0.2
    assert config_sha256(enabled) != config_sha256(original)
    parsed = make_args(tmp_path, ["--base_anchor_weight", "0.2"], lora=lora)
    assert build_train_config(parsed, lora=lora) == enabled


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_anchor_weight_fails_configuration(tmp_path, value):
    args = make_args(tmp_path)
    args.base_anchor_weight = value
    with pytest.raises(ValueError, match="base_anchor_weight"):
        build_train_config(args)


def batch_and_seeds(temporal=False):
    source = torch.linspace(0.2, 0.8, 2 * 3 * 48 * 48).reshape(2, 3, 48, 48)
    controls = torch.ones(2, 5, 48, 48)
    if not temporal:
        return {"source": source, "target": source * 0.5, "controls": controls}, [7, 19]
    return {
        "source": torch.stack([source, source.roll(1, -1), source.roll(2, -2)], dim=1),
        "target": torch.stack([source * 0.5] * 3, dim=1),
        "controls": torch.stack([controls, controls * 0.9, controls * 0.8], dim=1),
        "motion": torch.zeros(2, 3, 2, 48, 48),
        "history_valid": torch.ones(2, 3, 1, 48, 48),
        "temporal_valid": torch.ones(2, 3, 1, 48, 48),
        "reset": torch.tensor([[True, False, False], [True, False, True]]),
    }, [[7, 8, 9], [19, 20, 21]]


@pytest.mark.parametrize("qat", [False, True])
def test_disabling_lora_is_nested_exception_safe_and_skips_dropout(qat):
    model = SmallNR()
    configure_model_runtime(model, with_native_weight_qat(default_runtime_policy(), qat), training=True)
    network = small_inject(model, {"dropout": 0.5})
    network.adapters[0].lora_up.data.fill_(0.1)
    module = model.blocks["70"].head.rgb
    x = torch.ones(2, 32, 4, 4)
    expected = module.project(module.published_weight(module.materialized_weight()), x)
    before = torch.random.get_rng_state().clone()
    original_keys = set(network.state_dict())
    with pytest.raises(RuntimeError, match="injected failure"):
        with network.disable_adapters():
            assert not network.enabled
            torch.testing.assert_close(module(x), expected, rtol=0, atol=0)
            with network.disable_adapters():
                assert not network.enabled
            assert not network.enabled
            raise RuntimeError("injected failure")
    assert network.enabled
    assert set(network.state_dict()) == original_keys
    assert torch.equal(before, torch.random.get_rng_state())
    network.eval()
    assert not torch.equal(module(x), expected)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("qat", [False, True])
def test_reference_is_initial_model_with_same_controls_seeds_and_own_history(lora, temporal, qat):
    torch.manual_seed(17)
    model = SmallNR()
    configure_model_runtime(model, with_native_weight_qat(default_runtime_policy(), qat), training=True)
    original = deepcopy(model).eval()
    network = small_inject(model, {"dropout": 0.2 if not qat else 0}) if lora else None
    reference = anchor(model, network)
    reference.train()
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    if lora:
        assert reference.reference is None and not list(reference.parameters())
        network.adapters[0].lora_up.data.fill_(0.2)
    else:
        assert reference.reference is not model
        model.blocks["70"].head.rgb.weight.data.add_(0.15)
    batch, seeds = batch_and_seeds(temporal)
    burn_in = int(temporal)
    modes = [child.training for child in model.modules()]
    rng = torch.random.get_rng_state().clone()
    actual = reference(model, network, batch, seeds, burn_in)
    with torch.no_grad():
        frames = (
            training_step._clip_outputs(original, batch, seeds, burn_in)
            if temporal
            else [forward_frame(original, batch["source"], batch["controls"], seeds)]
        )
    expected = torch.cat([frame["rendered_proxy"] for frame in frames])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    snapshot = reference.predict(model, network, batch, seeds, burn_in)
    torch.testing.assert_close(snapshot["rendered_proxy"], actual, rtol=0, atol=0)
    torch.testing.assert_close(
        snapshot["neural_preclamp"], torch.cat([frame["neural_preclamp"] for frame in frames]), rtol=0, atol=0
    )
    assert all(not value.requires_grad for value in snapshot.values())
    assert "loss" not in reference.reference_identity
    assert reference.reference_identity["base_parameters_sha256"] == reference.identity["base_parameters_sha256"]
    assert not actual.requires_grad
    assert [child.training for child in model.modules()] == modes
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert network is None or network.enabled
    assert reference.identity["conditioning"] == "same_inputs_controls_seeds"
    assert reference.identity["history"] == "independent_reference_rollout"
    assert reference.identity["runtime_policy"] == model.runtime_policy


@pytest.mark.parametrize("scaled", [False, True])
def test_fp8_lora_anchor_is_the_effective_quantized_base_without_a_copy(scaled):
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base
    from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

    model = TinyFP8NR()
    network = tiny_fp8_inject(model, {})
    report = quantize_frozen_base(model, scaled=scaled)
    reference = anchor(model, network)
    assert not list(reference.parameters()) and not list(reference.buffers())
    assert reference.identity["base_parameters_sha256"] == report["effective_base_sha256"]
    batch, seeds = batch_and_seeds()
    expected = forward_frame(model, batch["source"], batch["controls"], seeds)["rendered_proxy"].detach()
    network.adapters[0].lora_up.data.fill_(0.1)
    torch.testing.assert_close(reference(model, network, batch, seeds, 0), expected, rtol=0, atol=0)


@pytest.mark.parametrize("method", ["forward", "predict"])
def test_reference_failure_restores_lora_model_modes_and_all_rng(monkeypatch, method):
    model = SmallNR()
    network = small_inject(model, {"dropout": 0.3})
    reference = anchor(model, network)
    model.blocks["0"].eval()
    modes = [child.training for child in model.modules()]
    before = capture_rng()

    def fail(*args, **kwargs):
        assert not torch.is_grad_enabled() and not network.enabled
        torch.rand(1)
        np.random.rand()
        random.random()
        raise RuntimeError("reference failed")

    monkeypatch.setattr(training_step, "forward_frame", fail)
    batch, seeds = batch_and_seeds()
    with pytest.raises(RuntimeError, match="reference failed"):
        getattr(reference, method)(model, network, batch, seeds, 0)
    assert [child.training for child in model.modules()] == modes
    after = capture_rng()
    assert torch.equal(before.pop("torch"), after.pop("torch"))
    torch.testing.assert_close(before.pop("cuda"), after.pop("cuda"), rtol=0, atol=0)
    assert before == after
    assert network.enabled and network.training


def test_lora_reference_rejects_a_mutable_base():
    model = SmallNR()
    network = small_inject(model, {})
    model.blocks["0"].input_adapter.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        anchor(model, network)


@pytest.mark.usefixtures("analytical_forward")
@pytest.mark.parametrize("profile", [None, PROFILE])
def test_anchor_is_full_band_masked_charbonnier_with_detached_reference(profile):
    prediction = torch.linspace(0.1, 0.9, 2 * 3 * 12 * 16).reshape(2, 3, 12, 16).requires_grad_()
    reference = prediction.detach().roll(1, -1).requires_grad_()
    mask = torch.ones(2, 1, 12, 16)
    mask[0, :, :, :8] = 0
    mask[1] *= 0.25
    batch = still_batch(torch.zeros_like(prediction), torch.zeros_like(prediction), mask)
    weights = {"base_anchor": 0.3}
    total, metrics = training_step.training_loss(prediction, batch, [5, 9], weights, loss_profile=profile, base_reference=reference)
    expected = (_rho(prediction - reference.detach()) * mask).sum() / (3 * mask.sum())
    torch.testing.assert_close(total, 0.3 * expected)
    assert metrics["loss/base_anchor"] == pytest.approx(float(expected.detach()))
    assert metrics["loss/base_anchor_weighted"] == pytest.approx(float(total.detach()))
    total.backward()
    assert reference.grad is None
    assert prediction.grad[mask.expand_as(prediction) == 0].eq(0).all()
    assert prediction.grad[mask.expand_as(prediction) > 0].abs().sum() > 0
    counts = training_step.loss_denominators(batch, 0, include_base_anchor=True)
    assert counts["base_anchor"] == counts["out"]


@pytest.mark.usefixtures("analytical_forward")
def test_anchor_requires_a_matching_reference_but_disabled_anchor_does_not():
    prediction = torch.ones(2, 3, 4, 4, requires_grad=True)
    batch = still_batch(prediction, prediction)
    with pytest.raises(ValueError, match="reference"):
        training_step.training_loss(prediction, batch, [1, 2], {"base_anchor": 1})
    with pytest.raises(ValueError, match="shape"):
        training_step.training_loss(prediction, batch, [1, 2], {"base_anchor": 1}, base_reference=prediction[:1])
    _, metrics = training_step.training_loss(prediction, batch, [1, 2], {"out": 1})
    assert "loss/base_anchor" not in metrics


def test_reference_rejects_nonfinite_outputs_even_when_rendered_values_are_clamped(monkeypatch):
    model = SmallNR()
    reference = anchor(model)
    original = training_step.forward_frame

    def nonfinite(*args, **kwargs):
        frame = original(*args, **kwargs)
        frame["raw_head"].fill_(float("inf"))
        return frame

    monkeypatch.setattr(training_step, "forward_frame", nonfinite)
    batch, seeds = batch_and_seeds()
    with pytest.raises(RuntimeError, match="non-finite base anchor"):
        reference(model, None, batch, seeds, 0)
