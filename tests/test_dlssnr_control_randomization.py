"""Effective controls, detached targets and augmented temporal supervision."""

import random
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr import losses, training_step
from test_dlssnr_frequency_loss import PROFILE, analytical_forward, clip_batch, still_batch  # noqa: F401


SETTINGS = {"anchor_probability": 0.25, "corner_probability": 0.25, "residual_sigma": 6.0}


def encode(reference, ratios):
    from musubi_tuner.dlssnr.control_randomization import encode_control_point

    return encode_control_point(reference, ratios)


def sample(settings=SETTINGS, *, epoch=2, sample_id="d0-r0-a", crop_id=0):
    from musubi_tuner.dlssnr.control_randomization import sample_control_point

    return sample_control_point({}, settings, seed=4, epoch=epoch, sample_id=sample_id, crop_id=crop_id)


def test_effective_control_alias_uses_exact_endpoint():
    point = encode({"nr_tone": 1, "nr_structure": 1}, (0.9999, 0.9999))
    assert point["ratios"].tolist() == [1.0, 1.0]
    assert torch.equal(point["sampled_lanes"], point["reference_lanes"])


@pytest.mark.parametrize("name", ["nr_tone", "nr_structure"])
@pytest.mark.parametrize("value", [0, 1e-12])
def test_reference_tone_that_rounds_to_zero_is_rejected(name, value):
    with pytest.raises(ValueError, match="reference"):
        encode({name: value}, (1, 1))


@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("skin", [-1, 0.75])
def test_encoding_keeps_style_and_skin_rule_and_uses_effective_ratios(auto, skin):
    point = encode({"nr_style": 128, "nr_auto_mask": auto, "nr_skin": skin, "nr_tone": 0.7, "nr_structure": 0.9}, (0.25, 0.5))
    expected = torch.tensor([0.7 * 0.25, 0.9 * 0.5]).half().float()
    assert point["values"].tolist() == expected.tolist()
    torch.testing.assert_close(point["ratios"], expected / torch.tensor([0.7, 0.9]).half().float(), rtol=0, atol=0)
    expected_lanes = (
        [1, expected[0], 1, expected[1] if skin == -1 else skin, expected[1]] if auto else [1, expected[0], expected[1], -1, -1]
    )
    assert point["sampled_lanes"].tolist() == expected_lanes
    assert all(value.dtype == torch.float32 and value.device.type == "cpu" for value in point.values())


@pytest.mark.parametrize(
    "ratios", [(-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5), (0.5, float("inf")), (True, 0), (0.5,), (0, 0, 0)]
)
def test_invalid_requested_ratios_are_rejected(ratios):
    with pytest.raises(ValueError, match="ratios"):
        encode({}, ratios)


@pytest.mark.parametrize(
    "updates",
    [
        {"anchor_probability": -0.1},
        {"corner_probability": 1.1},
        {"anchor_probability": 0.8},
        {"corner_probability": float("nan")},
        {"anchor_probability": True},
        {"residual_sigma": 0},
        {"residual_sigma": 33},
    ],
)
def test_invalid_sampling_settings_are_rejected(updates):
    with pytest.raises(ValueError):
        sample({**SETTINGS, **updates})


def test_sampling_branches_and_default_endpoint_mass():
    assert sample({**SETTINGS, "anchor_probability": 1, "corner_probability": 0})["ratios"].tolist() == [1, 1]
    corners = {
        tuple(sample({**SETTINGS, "anchor_probability": 0, "corner_probability": 1}, sample_id=str(i))["ratios"].tolist())
        for i in range(100)
    }
    assert corners == {(0, 0), (0, 1), (1, 0), (1, 1)}
    continuous = sample({**SETTINGS, "anchor_probability": 0, "corner_probability": 0})["ratios"]
    assert ((continuous > 0) & (continuous < 1)).all()
    draws = torch.stack([sample(sample_id=str(i))["ratios"] for i in range(1000)])
    assert float((draws == 1).all(dim=1).float().mean()) == pytest.approx(0.3125, abs=0.05)
    assert float((draws == 0).all(dim=1).float().mean()) == pytest.approx(0.0625, abs=0.025)


def test_seeds_vary_by_epoch_repeat_crop_and_domain_without_global_rng():
    from musubi_tuner.dlssnr.temporal import augmentation_seed

    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    settings = {**SETTINGS, "anchor_probability": 0, "corner_probability": 0}
    point = sample(settings)
    assert augmentation_seed(4, 2, "d0-r0-a", 0, domain="controls") != augmentation_seed(
        4, 2, "d0-r0-a", 0, domain="synthetic_jitter"
    )
    for kwargs in ({"epoch": 3}, {"sample_id": "d0-r1-a"}, {"crop_id": 1}):
        assert not torch.equal(point["sampled_lanes"], sample(settings, **kwargs)["sampled_lanes"])
    assert torch.equal(point["sampled_lanes"], sample(settings)["sampled_lanes"])
    assert random.getstate() == before[0]
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before[1][0] and after_numpy[2:] == before[1][2:]
    np.testing.assert_array_equal(after_numpy[1], before[1][1])
    assert torch.equal(torch.random.get_rng_state(), before[2])


@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_control_sampling_does_not_depend_on_ambient_device_or_dtype(device):
    settings = {**SETTINGS, "anchor_probability": 0, "corner_probability": 0}
    expected = sample(settings)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device(device):
            actual = sample(settings)
    finally:
        torch.set_default_dtype(previous_dtype)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def targets(source, target, mask, reference, sampled, ratios, *, sigma=6, input_edge=True):
    from musubi_tuner.dlssnr.control_randomization import control_targets

    return control_targets(source, target, mask, reference, sampled, ratios, sigma=sigma, input_edge=input_edge)


def target_inputs():
    source = torch.full((1, 3, 9, 13), 0.2)
    target = torch.full_like(source, 0.6)
    mask = torch.ones_like(source[:, :1])
    reference = {"neural_preclamp": torch.full_like(source, 1.3), "rendered_proxy": torch.full_like(source, 0.8)}
    return source, target, mask, reference


def test_zero_point_preserves_preclamp_and_frequency_edge():
    source, target, mask, reference = target_inputs()
    actual = targets(source, target, mask, reference, reference, torch.zeros(1, 2))
    torch.testing.assert_close(actual["preclamp_target"], reference["neural_preclamp"], rtol=0, atol=0)
    for name in ("target", "edge_target"):
        torch.testing.assert_close(actual[name], reference["rendered_proxy"], rtol=0, atol=0)
    assert not actual["rgb_clipped"].any() and not actual["edge_clipped"].any()


@pytest.mark.parametrize("input_edge", [False, True])
def test_reference_endpoint_is_exact_on_soft_support_and_fills_excluded_with_teacher(input_edge):
    source, target, mask, reference = target_inputs()
    mask *= 0.3
    mask[..., 2:4, 3:7] = 0
    actual = targets(source, target, mask, reference, reference, torch.ones(1, 2), input_edge=input_edge)
    for key, original, field in (
        ("target", target, "rendered_proxy"),
        ("preclamp_target", target, "neural_preclamp"),
        ("edge_target", source if input_edge else target, "rendered_proxy"),
    ):
        torch.testing.assert_close(actual[key], torch.where(mask > 0, original, reference[field]), rtol=0, atol=0)


def direct_gaussian(value, mask, sigma):
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64)
    yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
    kernel /= kernel.sum()
    numerator = F.conv2d(
        F.pad((value * mask).double(), (radius,) * 4, mode="replicate"), kernel[None, None].expand(3, 1, -1, -1), groups=3
    )
    denominator = F.conv2d(F.pad(mask.double(), (radius,) * 4, mode="replicate"), kernel[None, None])
    return numerator / denominator.clamp_min(1e-30)


@pytest.mark.parametrize("input_edge", [False, True])
def test_target_bands_match_independent_2d_gaussian_and_are_detached(input_edge):
    generator = torch.Generator().manual_seed(19)
    source = torch.rand(3, 3, 9, 13, generator=generator).requires_grad_()
    target = torch.rand(source.shape, generator=generator).requires_grad_()
    mask = torch.rand(3, 1, 9, 13, generator=generator)
    mask[..., 1:4, 2:8] = 0
    reference = {"neural_preclamp": source * 1.5, "rendered_proxy": source * 0.6}
    sampled = {"neural_preclamp": source * 1.8, "rendered_proxy": source * 0.8}
    ratios = torch.tensor([[0.8, 0.2], [0.1, 0.75], [0.7, 0.7]])
    tensors = [source, target, mask, *reference.values(), *sampled.values(), ratios]
    copies = [item.detach().clone() for item in tensors]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = targets(source, target, mask, reference, sampled, ratios, sigma=1.5, input_edge=input_edge)
    for key, field, original in (
        ("target", "rendered_proxy", target),
        ("preclamp_target", "neural_preclamp", target),
        ("edge_target", "rendered_proxy", source if input_edge else target),
    ):
        residual = original.detach() - reference[field].detach()
        a_t, a_s = ratios[:, 0, None, None, None], ratios[:, 1, None, None, None]
        expected = sampled[field].detach() + a_s * residual + (a_t - a_s) * direct_gaussian(residual, mask, 1.5)
        expected = torch.where(mask > 0, expected, sampled[field])
        if key != "preclamp_target":
            clipped = (expected < 0) | (expected > 1)
            assert torch.equal(actual["rgb_clipped" if key == "target" else "edge_clipped"], clipped)
            expected = expected.clamp(0, 1)
        torch.testing.assert_close(actual[key], expected.float(), rtol=2e-6, atol=2e-7)
        assert actual[key].dtype == torch.float32
    assert not any(item.requires_grad for item in actual.values())
    for tensor, copy in zip(tensors, copies):
        torch.testing.assert_close(tensor, copy, rtol=0, atol=0)


def test_target_masks_exclude_unknown_labels_and_allow_empty_local_support():
    source, target, mask, reference = target_inputs()
    mask[..., 2:7, 4:9] = 0
    ratios = torch.tensor([[0.2, 0.7]])
    expected = targets(source, target, mask, reference, reference, ratios)
    unknown = target.masked_fill(mask.expand_as(target) == 0, 2000)
    changed_source = source.masked_fill(mask.expand_as(source) == 0, -1000)
    actual = targets(changed_source, unknown, mask, reference, reference, ratios)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    empty = targets(source, target, torch.zeros_like(mask), reference, reference, ratios)
    torch.testing.assert_close(empty["target"], reference["rendered_proxy"], rtol=0, atol=0)
    torch.testing.assert_close(empty["preclamp_target"], reference["neural_preclamp"], rtol=0, atol=0)
    assert expected["preclamp_target"].max() > 1


@pytest.mark.parametrize("field", ["source", "target", "mask", "ratios", "reference", "sampled"])
def test_nonfinite_target_inputs_are_not_hidden_by_clipping(field):
    source, target, mask, reference = target_inputs()
    args = {
        "source": source,
        "target": target,
        "mask": mask,
        "reference": reference,
        "sampled": {key: value.clone() for key, value in reference.items()},
        "ratios": torch.tensor([[0.5, 0.5]]),
    }
    tensor = args[field]["neural_preclamp"] if field in ("reference", "sampled") else args[field]
    tensor.flatten()[0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        targets(**args)


@pytest.mark.parametrize("field", ["target", "mask", "ratios", "reference"])
def test_target_kernel_rejects_broadcastable_shapes(field):
    source, target, mask, reference = target_inputs()
    args = {
        "source": source,
        "target": target,
        "mask": mask,
        "reference": reference,
        "sampled": reference,
        "ratios": torch.tensor([[0.5, 0.5]]),
    }
    if field == "reference":
        args[field] = {**reference, "neural_preclamp": reference["neural_preclamp"][:, :1]}
    else:
        args[field] = args[field][..., :1]
    with pytest.raises(ValueError, match="shape"):
        targets(**args)


@pytest.mark.parametrize("profile", [None, PROFILE])
def test_zero_point_has_zero_loss_in_both_profiles(monkeypatch, profile):
    source, target, mask, _ = target_inputs()
    render = torch.linspace(0.1, 0.8, source.numel()).reshape_as(source).requires_grad_()
    preclamp = torch.full_like(source, 1.3, requires_grad=True)
    teacher = {"neural_preclamp": preclamp.detach(), "rendered_proxy": render.detach()}
    generated = targets(source, target, mask, teacher, teacher, torch.zeros(1, 2), input_edge=profile is not None)
    batch = {
        **still_batch(source, generated["target"], mask),
        "preclamp_target": generated["preclamp_target"],
        "edge_target": generated["edge_target"],
    }

    def forward(*args, **kwargs):
        return {"neural_preclamp": preclamp.clone(), "rendered_proxy": render.clone(), "blend_weight": render.new_zeros(1)}

    monkeypatch.setattr(training_step, "forward_frame", forward)
    total, metrics = training_step.training_loss(None, batch, [3], {"pre": 1, "out": 1, "edge": 1}, loss_profile=profile)
    assert total == 0
    assert metrics["loss/control_edge"] == 0 and "loss/input_edge" not in metrics
    total.backward()
    assert preclamp.grad.eq(0).all() and render.grad.eq(0).all()


@pytest.mark.parametrize("clip", [False, True])
@pytest.mark.parametrize("field", ["preclamp_target", "edge_target", "temporal_support"])
def test_loss_rejects_unknown_support_and_broadcastable_target_overrides(clip, field):
    source = torch.ones((1, 3, 3, 5, 7) if clip else (1, 3, 5, 7))
    batch = clip_batch(source, source) if clip else still_batch(source, source)
    batch[field] = "unknown" if field == "temporal_support" else source[..., :1, :, :]
    with pytest.raises(ValueError, match="shape|temporal_support"):
        training_step.loss_denominators(batch, int(clip))
    with pytest.raises(ValueError, match="shape|temporal_support"):
        training_step.training_loss(None, batch, [[1, 2, 3]] if clip else [1], {"out": 1}, int(clip))


def test_supervised_values_are_time_major_and_exclude_burn_in():
    value = torch.arange(2 * 4 * 3 * 2 * 2).reshape(2, 4, 3, 2, 2)
    actual = training_step.supervised_values(value, 1)
    expected = torch.stack([value[0, 1], value[1, 1], value[0, 2], value[1, 2], value[0, 3], value[1, 3]])
    assert torch.equal(actual, expected)
    assert training_step.supervised_values(value[:, 0], 0) is not None
    assert torch.equal(training_step.supervised_values(value[:, 0], 0), value[:, 0])


def soft_temporal_batch():
    generator = torch.Generator().manual_seed(25)
    source = torch.rand(3, 3, 3, 5, 7, generator=generator)
    target = torch.rand(source.shape, generator=generator)
    mask = torch.ones(3, 3, 1, 5, 7)
    mask[0, 1, :, 1:3, 2:4] = 0
    mask[1, 1] *= 0.25
    mask[1, 2] *= 0.4
    mask[2] = 0
    batch = clip_batch(source, target, mask)
    batch["motion"][:, 2, 0] = 0.5
    batch["temporal_valid"][0, 2, :, 3:, :2] = False
    batch["temporal_support"] = "joint_loss_mask"
    return batch


def half_shift_oracle(value):
    return F.pad((value[..., :-1] + value[..., 1:]) * 0.5, (0, 1))


@pytest.mark.parametrize("profile", [None, PROFILE])
@pytest.mark.usefixtures("analytical_forward")
def test_joint_temporal_denominator_matches_residual_with_soft_masks(profile):
    batch = soft_temporal_batch()
    mask, motion, target = batch["loss_mask"], batch["motion"], batch["target"]
    prediction = (batch["source"] * 0.8).requires_grad_()
    previous_coverage = half_shift_oracle(mask[:, 1])
    valid = mask[:, 2] * previous_coverage * batch["temporal_valid"][:, 2]
    previous_error = (prediction[:, 1] - target[:, 1]) * mask[:, 1]
    previous_error = half_shift_oracle(previous_error) / torch.where(previous_coverage > 0, previous_coverage, 1)
    expected_error = prediction[:, 2] - target[:, 2] - previous_error
    actual_error, actual_valid = losses.masked_temporal_residual(
        prediction[:, 2],
        prediction[:, 1],
        target[:, 2],
        target[:, 1],
        mask[:, 2],
        mask[:, 1],
        motion[:, 2],
        batch["temporal_valid"][:, 2],
        batch["reset"][:, 2],
    )
    torch.testing.assert_close(actual_error, expected_error, rtol=1e-5, atol=5e-7)
    torch.testing.assert_close(actual_valid, valid, rtol=1e-5, atol=2e-7)
    counts = training_step.loss_denominators(batch, 1, loss_profile=profile)
    assert counts["temporal"] == pytest.approx(float(3 * valid.sum()))
    if profile is not None:
        expected_error = direct_gaussian(expected_error, valid, profile["lowpass_sigma"]).float()
    expected = (losses._rho(expected_error) * valid).sum() / counts["temporal"]
    weights = {"temporal": 1}
    seeds = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    full, _ = training_step.training_loss(prediction, batch, seeds, weights, 1, loss_profile=profile)
    torch.testing.assert_close(full, expected, rtol=1e-5, atol=5e-7)
    gradient = torch.autograd.grad(full, prediction)[0]
    partial = []
    for index in range(3):
        part = {name: value[index : index + 1] if isinstance(value, torch.Tensor) else value for name, value in batch.items()}
        loss, _ = training_step.training_loss(
            prediction[index : index + 1], part, seeds[index : index + 1], weights, 1, counts, loss_profile=profile
        )
        partial.append(loss)
    accumulated = sum(partial)
    torch.testing.assert_close(accumulated, full, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(torch.autograd.grad(accumulated, prediction)[0], gradient, rtol=1e-5, atol=1e-8)
    altered = {**batch, "target": target.masked_fill(mask.expand_as(target) == 0, 500)}
    other, _ = training_step.training_loss(prediction, altered, seeds, weights, 1, loss_profile=profile)
    torch.testing.assert_close(other, full, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(other, prediction)[0], gradient, rtol=0, atol=0)


@pytest.mark.usefixtures("analytical_forward")
def test_joint_policy_does_not_square_frequency_support():
    batch = soft_temporal_batch()
    counts = training_step.loss_denominators(batch, 1, loss_profile=PROFILE)
    with_policy, _ = training_step.training_loss(batch["source"], batch, [[1, 2, 3]] * 3, {"temporal": 1}, 1, loss_profile=PROFILE)
    ordinary = {key: value for key, value in batch.items() if key != "temporal_support"}
    old_counts = training_step.loss_denominators(ordinary, 1, loss_profile=PROFILE)
    original, _ = training_step.training_loss(batch["source"], ordinary, [[1, 2, 3]] * 3, {"temporal": 1}, 1, loss_profile=PROFILE)
    assert counts == old_counts
    torch.testing.assert_close(with_policy, original, rtol=0, atol=0)


@pytest.mark.parametrize("inactive", ["reset", "outside", "previous_mask", "current_mask"])
@pytest.mark.usefixtures("analytical_forward")
def test_joint_support_reset_outside_or_missing_labels_has_zero_gradient(inactive):
    batch = soft_temporal_batch()
    if inactive == "reset":
        batch["reset"][:, 2] = True
    elif inactive == "outside":
        batch["motion"][:, 2] = 100
    else:
        batch["loss_mask"][:, 1 if inactive == "previous_mask" else 2] = 0
    prediction = batch["source"].clone().requires_grad_()
    total, _ = training_step.training_loss(prediction, batch, [[1, 2, 3]] * 3, {"temporal": 1}, 1)
    assert training_step.loss_denominators(batch, 1)["temporal"] == 0
    assert total == 0
    total.backward()
    assert prediction.grad.eq(0).all()


@pytest.mark.usefixtures("analytical_forward")
def test_legacy_pixel_temporal_support_remains_geometric_not_label_masked():
    batch = soft_temporal_batch()
    del batch["temporal_support"]
    previous = half_shift_oracle(batch["source"][:, 1])
    previous_target = half_shift_oracle(batch["target"][:, 1])
    error = batch["source"][:, 2] - previous - (batch["target"][:, 2] - previous_target)
    valid = batch["temporal_valid"][:, 2].clone()
    valid[..., -1] = False
    expected = (losses._rho(error) * valid).sum() / (3 * valid.sum())
    actual, _ = training_step.training_loss(batch["source"], batch, [[1, 2, 3]] * 3, {"temporal": 1}, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
