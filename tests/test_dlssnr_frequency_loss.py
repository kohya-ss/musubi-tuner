"""Analytical frequency/mask tests, independent of the expensive NR forward."""

import math

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr import losses, training_step
from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from test_dlssnr_config import make_args


PROFILE = {"name": "frequency_split", "lowpass_sigma": 6.0}
WEIGHTS = {"pre": 1.0, "out": 1.0, "edge": 0.0, "temporal": 0.0}


def lowpass(error, mask, sigma):
    function = getattr(losses, "masked_gaussian_lowpass", None)
    assert callable(function), "Mask-aware low-frequency filtering is not implemented"
    return function(error, mask, sigma)


@pytest.mark.parametrize("lora", [False, True])
def test_frequency_loss_is_opt_in_and_preserves_default_config(tmp_path, lora):
    args = make_args(tmp_path, lora=lora)
    assert getattr(args, "loss_profile", None) == "pixel", "The existing pixel loss must remain the default"
    assert args.loss_lowpass_sigma is None
    default = build_train_config(args, lora=lora)
    explicit = build_train_config(make_args(tmp_path, ["--loss_profile", "pixel"], lora=lora), lora=lora)
    assert default == explicit
    assert "loss_profile" not in default
    split = build_train_config(make_args(tmp_path, ["--loss_profile", "frequency_split"], lora=lora), lora=lora)
    assert split["loss_profile"] == PROFILE
    assert split["loss"] == default["loss"]
    assert config_sha256(split) != config_sha256(default)
    custom = build_train_config(
        make_args(tmp_path, ["--loss_profile", "frequency_split", "--loss_lowpass_sigma", "4"], lora=lora), lora=lora
    )
    assert custom["loss_profile"]["lowpass_sigma"] == 4
    assert config_sha256(custom) != config_sha256(split)


@pytest.mark.parametrize("value", [0, -1, 33, float("nan"), float("inf"), True])
def test_invalid_lowpass_sigma_is_rejected_at_configuration(tmp_path, value):
    args = make_args(tmp_path)
    args.loss_profile, args.loss_lowpass_sigma = "frequency_split", value
    with pytest.raises(ValueError, match="loss_lowpass_sigma"):
        build_train_config(args)


def test_pixel_loss_rejects_an_ignored_sigma_or_unknown_profile(tmp_path):
    args = make_args(tmp_path)
    args.loss_lowpass_sigma = 4
    with pytest.raises(ValueError, match="frequency_split"):
        build_train_config(args)
    args.loss_profile, args.loss_lowpass_sigma = "unknown", None
    with pytest.raises(ValueError, match="loss_profile"):
        build_train_config(args)


@pytest.mark.parametrize("sigma", [0.5, 2, 6, 32])
def test_separable_lowpass_matches_a_direct_normalized_gaussian(sigma):
    generator = torch.Generator().manual_seed(4)
    error = torch.rand(2, 3, 9, 13, generator=generator, dtype=torch.float64)
    mask = torch.rand(2, 1, 9, 13, generator=generator, dtype=torch.float64)
    mask[..., 3:7, 5:9] = 0
    radius = math.ceil(3 * sigma)
    coordinates = torch.arange(-radius, radius + 1, dtype=error.dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    kernel = kernel / kernel.sum()
    kernel_2d = (kernel[:, None] * kernel[None, :])[None, None]
    padding = (radius, radius, radius, radius)
    numerator = F.conv2d(F.pad(error * mask, padding, mode="replicate"), kernel_2d.expand(3, 1, -1, -1), groups=3)
    denominator = F.conv2d(F.pad(mask, padding, mode="replicate"), kernel_2d)
    actual = lowpass(error, mask, sigma)
    torch.testing.assert_close(actual, numerator / denominator, rtol=1e-12, atol=1e-12)
    assert actual.shape == error.shape


def test_lowpass_does_not_read_masked_pixels_or_dilute_constants_at_mask_edges():
    mask = torch.ones(2, 1, 17, 21)
    mask[0, :, 3:11, 8:13] = 0
    mask[1] *= 0.25
    error = torch.full((2, 3, 17, 21), 0.2)
    expected = lowpass(error, mask, 3)
    altered = error.masked_fill(mask.expand_as(error) == 0, 1000).requires_grad_()
    actual = lowpass(altered, mask, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual, error, rtol=1e-6, atol=1e-7)
    actual.sum().backward()
    assert torch.isfinite(altered.grad).all()
    assert altered.grad[mask.expand_as(error) == 0].eq(0).all()
    assert altered.grad[mask.expand_as(error) > 0].ne(0).any()


def test_zero_mask_lowpass_is_zero_with_zero_finite_gradient():
    error = torch.ones(1, 3, 7, 11, requires_grad=True)
    actual = lowpass(error, torch.zeros(1, 1, 7, 11), 6)
    assert actual.eq(0).all()
    actual.sum().backward()
    assert error.grad.eq(0).all()


@pytest.mark.parametrize("sigma", [1e-50, 1e-300])
def test_positive_subnormal_sigma_has_a_finite_discrete_delta_kernel(sigma):
    error = torch.linspace(0.1, 0.9, 12).reshape(1, 1, 3, 4).requires_grad_()
    mask = torch.ones_like(error)
    actual = lowpass(error, mask, sigma)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, error, rtol=0, atol=0)
    actual.sum().backward()
    torch.testing.assert_close(error.grad, torch.ones_like(error), rtol=0, atol=0)


def test_lowpass_preserves_fp32_inside_autocast_and_has_correct_derivatives():
    error = torch.linspace(0.1, 0.9, 12).reshape(1, 1, 3, 4)
    mask = torch.ones_like(error)
    mask[..., 1, 2] = 0
    expected = lowpass(error, mask, 1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = lowpass(error, mask, 1)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    differentiable = error.double().requires_grad_()
    assert torch.autograd.gradcheck(lambda value: lowpass(value, mask.double(), 1), (differentiable,))


@pytest.fixture
def analytical_forward(monkeypatch):
    def forward(prediction, source, controls, seeds, **kwargs):
        index = int(controls[0, 0, 0, 0])
        value = (prediction if prediction.ndim == 4 else prediction[:, index]).clone()
        return {
            "neural_preclamp": value,
            "rendered_proxy": value,
            "next_history": value,
            "blend_weight": value.new_zeros(1),
        }

    monkeypatch.setattr(training_step, "forward_frame", forward)


def still_batch(source, target, mask=None):
    return {
        "source": source,
        "target": target,
        "controls": source.new_zeros(source.shape[0], 5, *source.shape[-2:]),
        "loss_mask": torch.ones_like(source[:, :1]) if mask is None else mask,
    }


@pytest.mark.usefixtures("analytical_forward")
@pytest.mark.parametrize("shift", [1, 2])
def test_lowpass_reduces_shifted_texture_penalty_without_discarding_tone(shift):
    x = torch.arange(96).float()[None, :]
    y = torch.arange(64).float()[:, None]
    source = (0.5 + 0.2 * torch.sin(2 * math.pi * x / 5) * torch.cos(2 * math.pi * y / 7))[None, None].expand(1, 3, -1, -1)
    target = source.roll(shift, dims=-1)
    batch = still_batch(source, target)
    prediction = source.clone().requires_grad_()
    legacy, _ = training_step.training_loss(prediction, batch, [4], WEIGHTS)
    split, metrics = training_step.training_loss(prediction, batch, [4], WEIGHTS, loss_profile=PROFILE)
    assert split < legacy * 0.2
    assert "loss/lowpass_pre" in metrics and "loss/lowpass_out" in metrics
    split.backward()
    assert torch.isfinite(prediction.grad).all()
    tone_batch = still_batch(torch.zeros_like(source), torch.full_like(source, 0.2))
    tone, _ = training_step.training_loss(torch.zeros_like(source), tone_batch, [4], WEIGHTS, loss_profile=PROFILE)
    torch.testing.assert_close(tone, 2 * losses._rho(torch.tensor(0.2)), rtol=1e-5, atol=1e-6)


@pytest.mark.usefixtures("analytical_forward")
def test_structure_loss_anchors_input_edges_not_redrawn_target_edges():
    source = torch.zeros(1, 3, 48, 64)
    source[..., 20:44] = 0.5
    target = source.roll(2, dims=-1)
    batch = still_batch(source, target)
    weights = {**WEIGHTS, "pre": 0, "out": 0, "edge": 1}
    old, _ = training_step.training_loss(source, batch, [4], weights)
    anchored, metrics = training_step.training_loss(source, batch, [4], weights, loss_profile=PROFILE)
    assert old > 0
    assert anchored == 0 and metrics["loss/input_edge"] == 0
    blurred = lowpass(source, batch["loss_mask"], 2).requires_grad_()
    structure, _ = training_step.training_loss(blurred, batch, [4], weights, loss_profile=PROFILE)
    assert structure > 0
    structure.backward()
    assert torch.isfinite(blurred.grad).all() and blurred.grad.ne(0).any()


@pytest.mark.usefixtures("analytical_forward")
def test_frequency_loss_is_invariant_to_ignored_target_values():
    source = torch.linspace(0, 0.6, 48 * 64).reshape(1, 1, 48, 64).expand(1, 3, -1, -1)
    mask = torch.ones(1, 1, 48, 64)
    mask[..., 13:25, 22:39] = 0
    batch = still_batch(source, source * 0.8, mask)
    prediction = source.clone().requires_grad_()
    loss, _ = training_step.training_loss(prediction, batch, [4], WEIGHTS, loss_profile=PROFILE)
    expected_grad = torch.autograd.grad(loss, prediction)[0]
    batch["target"] = batch["target"].masked_fill(mask.expand_as(source) == 0, 500)
    altered, _ = training_step.training_loss(prediction, batch, [4], WEIGHTS, loss_profile=PROFILE)
    gradient = torch.autograd.grad(altered, prediction)[0]
    torch.testing.assert_close(altered, loss, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_grad, rtol=0, atol=0)
    assert gradient[mask.expand_as(source) == 0].eq(0).all()


@pytest.mark.usefixtures("analytical_forward")
def test_frequency_loss_accumulation_matches_global_valid_pixel_weighting():
    generator = torch.Generator().manual_seed(9)
    source = torch.rand(3, 3, 48, 64, generator=generator)
    target = torch.rand(3, 3, 48, 64, generator=generator)
    mask = torch.ones(3, 1, 48, 64)
    mask[1, :, :30] = 0
    mask[2, :, :, :20] = 0.25
    batch = still_batch(source, target, mask)
    prediction = (source * 0.9).requires_grad_()
    weights = {**WEIGHTS, "edge": 0.2}
    full, _ = training_step.training_loss(prediction, batch, [1, 2, 3], weights, loss_profile=PROFILE)
    gradient = torch.autograd.grad(full, prediction)[0]
    counts = training_step.loss_denominators(batch, 0, loss_profile=PROFILE)
    accumulated = prediction.new_zeros(())
    for index in range(3):
        part = {name: value[index : index + 1] for name, value in batch.items()}
        loss, _ = training_step.training_loss(
            prediction[index : index + 1], part, [index + 1], weights, normalizers=counts, loss_profile=PROFILE
        )
        accumulated = accumulated + loss
    torch.testing.assert_close(accumulated, full, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(torch.autograd.grad(accumulated, prediction)[0], gradient, rtol=1e-5, atol=1e-8)


def clip_batch(source, target, mask=None):
    batch, frames, _, height, width = source.shape
    controls = source.new_zeros(batch, frames, 5, height, width)
    for index in range(frames):
        controls[:, index, 0] = index
    reset = torch.zeros(batch, frames, dtype=torch.bool)
    reset[:, 0] = True
    return {
        "source": source,
        "target": target,
        "controls": controls,
        "loss_mask": torch.ones_like(source[:, :, :1]) if mask is None else mask,
        "motion": source.new_zeros(batch, frames, 2, height, width),
        "history_valid": torch.ones(batch, frames, 1, height, width, dtype=torch.bool),
        "temporal_valid": torch.ones(batch, frames, 1, height, width, dtype=torch.bool),
        "reset": reset,
    }


@pytest.mark.usefixtures("analytical_forward")
def test_temporal_lowpass_respects_both_loss_masks_even_with_subpixel_motion():
    from musubi_tuner.dlssnr.temporal import warp_bilinear

    source = torch.zeros(1, 3, 3, 48, 64)
    mask = torch.ones(1, 3, 1, 48, 64)
    mask[:, 1, :, 12:22, 20:30] = 0
    mask[:, 2, :, :8] = 0.25
    batch = clip_batch(source, source.clone(), mask)
    batch["motion"][:, 2, 0] = 0.5
    batch["temporal_valid"][:, 2, :, 30:40, 5:10] = False
    previous_mask, inside = warp_bilinear(mask[:, 1], batch["motion"][:, 2])
    valid = mask[:, 2] * previous_mask * inside * batch["temporal_valid"][:, 2]
    counts = training_step.loss_denominators(batch, 1, loss_profile=PROFILE)
    assert counts["temporal"] == pytest.approx(float(valid.sum() * 3))
    prediction = source.clone()
    prediction[:, 1], prediction[:, 2] = 0.05, 0.2
    prediction.requires_grad_()
    weights = {"pre": 0, "out": 0, "edge": 0, "temporal": 1}
    expected, metrics = training_step.training_loss(prediction, batch, [[1, 2, 3]], weights, 1, loss_profile=PROFILE)
    assert "loss/lowpass_temporal" in metrics
    torch.testing.assert_close(expected, losses._rho(torch.tensor(0.15)), rtol=1e-5, atol=1e-6)
    expected_gradient = torch.autograd.grad(expected, prediction)[0]
    changed = batch["target"].clone()
    changed[:, 1] = changed[:, 1].masked_fill(mask[:, 1].expand_as(changed[:, 1]) == 0, 500)
    batch["target"] = changed
    actual, _ = training_step.training_loss(prediction, batch, [[1, 2, 3]], weights, 1, loss_profile=PROFILE)
    gradient = torch.autograd.grad(actual, prediction)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    assert gradient[:, 0].eq(0).all()
    assert gradient[:, 1][mask[:, 1].expand_as(gradient[:, 1]) == 0].eq(0).all()


@pytest.mark.usefixtures("analytical_forward")
def test_temporal_term_cannot_reintroduce_full_band_target_regression():
    x = torch.arange(64).float()[None, :]
    y = torch.arange(48).float()[:, None]
    image = (0.5 + 0.2 * torch.sin(2 * math.pi * x / 5) * torch.cos(2 * math.pi * y / 7))[None, None]
    source = image[:, None].expand(1, 3, 3, -1, -1).clone()
    target = source.clone()
    target[:, 2] = source[:, 2].roll(1, dims=-1)
    batch = clip_batch(source, target)
    weights = {"pre": 0, "out": 0, "edge": 0, "temporal": 1}
    legacy, _ = training_step.training_loss(source, batch, [[1, 2, 3]], weights, 1)
    split, _ = training_step.training_loss(source, batch, [[1, 2, 3]], weights, 1, loss_profile=PROFILE)
    assert split < legacy * 0.2


@pytest.mark.parametrize("inactive", ["reset", "outside", "masked"])
@pytest.mark.usefixtures("analytical_forward")
def test_inactive_temporal_support_has_zero_loss_and_finite_zero_gradient(inactive):
    source = torch.ones(1, 3, 3, 48, 64) * 0.3
    batch = clip_batch(source, source * 0.8)
    if inactive == "reset":
        batch["reset"][:, 2] = True
    elif inactive == "outside":
        batch["motion"][:, 2] = 1000
    else:
        batch["loss_mask"][:, 1] = 0
    prediction = source.clone().requires_grad_()
    weights = {"pre": 0, "out": 0, "edge": 0, "temporal": 1}
    loss, _ = training_step.training_loss(prediction, batch, [[1, 2, 3]], weights, 1, loss_profile=PROFILE)
    assert training_step.loss_denominators(batch, 1, loss_profile=PROFILE)["temporal"] == 0
    assert loss == 0
    loss.backward()
    assert prediction.grad.eq(0).all()


@pytest.mark.usefixtures("analytical_forward")
def test_temporal_frequency_accumulation_uses_the_same_joint_support_as_the_loss():
    generator = torch.Generator().manual_seed(15)
    source = torch.rand(2, 3, 3, 48, 64, generator=generator)
    target = torch.rand(2, 3, 3, 48, 64, generator=generator)
    batch = clip_batch(source, target)
    batch["loss_mask"][1, 1, :, :20] = 0
    batch["loss_mask"][1, 2, :, :, 10:30] = 0.25
    batch["motion"][:, 2, 0] = 0.5
    prediction = (source * 0.9).requires_grad_()
    weights = {**WEIGHTS, "edge": 0.2, "temporal": 0.5}
    counts = training_step.loss_denominators(batch, 1, loss_profile=PROFILE)
    seeds = [[1, 2, 3], [4, 5, 6]]
    full, _ = training_step.training_loss(prediction, batch, seeds, weights, 1, loss_profile=PROFILE)
    gradient = torch.autograd.grad(full, prediction)[0]
    partial = []
    for index in range(2):
        part = {name: value[index : index + 1] for name, value in batch.items()}
        loss, _ = training_step.training_loss(
            prediction[index : index + 1], part, seeds[index : index + 1], weights, 1, counts, loss_profile=PROFILE
        )
        partial.append(loss)
    accumulated = sum(partial)
    torch.testing.assert_close(accumulated, full, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(torch.autograd.grad(accumulated, prediction)[0], gradient, rtol=1e-5, atol=1e-8)
