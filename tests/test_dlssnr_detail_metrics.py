"""Analytical, mask-aware detail diagnostics without pretrained models or downloads."""

import importlib
import importlib.util
import json

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from test_dlssnr_config import make_args


def detail_module():
    name = "musubi_tuner.dlssnr.detail_metrics"
    assert importlib.util.find_spec(name) is not None, "Detail diagnostics are not implemented"
    return importlib.import_module(name)


def output(rgb, preclamp=None):
    return {"rendered_proxy": rgb, "neural_preclamp": rgb if preclamp is None else preclamp}


def measure(source, target, rendered, alternate=None, mask=None):
    totals = detail_module().DetailDiagnostics()
    totals.add(
        output(rendered),
        output(rendered if alternate is None else alternate),
        {"source": source, "target": target, "loss_mask": torch.ones_like(source[:, :1]) if mask is None else mask},
    )
    return totals.metrics()


def checkerboard(height=32, width=40):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    return ((x + y) % 2 * 2 - 1).float()[None, None].expand(1, 3, -1, -1)


@pytest.mark.parametrize("lora", [False, True])
def test_detail_diagnostics_are_opt_in_and_require_held_out_data(tmp_path, lora):
    args = make_args(tmp_path, lora=lora)
    assert getattr(args, "eval_detail_diagnostics", None) is False
    plain = build_train_config(args, lora=lora)
    assert "detail_diagnostics" not in plain["evaluation"]
    args.eval_detail_diagnostics = True
    with pytest.raises(ValueError, match="evaluation requires"):
        build_train_config(args, lora=lora)
    dataset = {"datasets": [{"train_manifest": "train.jsonl", "validation_manifest": "held_out.jsonl"}]}
    plain = build_train_config(make_args(tmp_path, lora=lora, dataset=dataset), lora=lora)
    enabled = build_train_config(make_args(tmp_path, ["--eval_detail_diagnostics"], lora=lora, dataset=dataset), lora=lora)
    assert enabled["evaluation"]["detail_diagnostics"] is True
    assert enabled["evaluation"]["sample_every_n_steps"] == 0
    assert config_sha256(plain) != config_sha256(enabled)


def test_high_frequency_energy_ignores_dc_and_tracks_squared_detail_amplitude():
    texture = checkerboard()
    report = measure(0.5 + 0.2 * texture, 0.5 + 0.4 * texture, 0.7 + 0.1 * texture)
    assert report["valid_rgb_values"] == 3 * 32 * 40
    assert report["protocol"]["highpass_sigmas_px"] == [1.0, 4.0]
    for band in report["high_frequency"].values():
        assert band["input_energy"] > 0
        assert band["output_to_input"] == pytest.approx(0.25, rel=1e-5)
        assert band["target_to_input"] == pytest.approx(4.0, rel=1e-5)
        assert band["output_to_target"] == pytest.approx(1 / 16, rel=1e-5)
    assert report["noise_sensitivity"]["rgb_mae"] == 0
    assert report["noise_sensitivity"]["preclamp_mae"] == 0
    assert all(value == 0 for value in report["noise_sensitivity"]["high_frequency_rms"].values())


def test_highpass_matches_a_direct_2d_mask_normalized_gaussian():
    generator = torch.Generator().manual_seed(12)
    source = torch.rand(1, 3, 11, 17, generator=generator)
    mask = torch.rand(1, 1, 11, 17, generator=generator)
    mask[..., 2:8, 4:9] = 0
    report = measure(source, source, source, mask=mask)
    for sigma in (1, 4):
        radius = 3 * sigma
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64)
        kernel = torch.exp(-0.5 * (coordinates / sigma).square())
        kernel = kernel / kernel.sum()
        kernel = (kernel[:, None] * kernel[None, :])[None, None]
        padding = (radius, radius, radius, radius)
        numerator = F.conv2d(F.pad(source.double() * mask, padding, mode="replicate"), kernel.expand(3, 1, -1, -1), groups=3)
        support = F.conv2d(F.pad(mask.double(), padding, mode="replicate"), kernel)
        expected = ((source - numerator / support).square() * mask).sum() / (3 * mask.sum())
        assert report["high_frequency"][f"sigma_{sigma}px"]["input_energy"] == pytest.approx(float(expected), rel=1e-5)


def test_mask_holes_do_not_create_false_detail_and_ratios_handle_flat_inputs():
    image = torch.full((1, 3, 16, 24), 0.5)
    mask = torch.ones_like(image[:, :1])
    mask[..., 4:10, 7:13] = 0
    report = measure(image, image + 0.1, image - 0.1, mask=mask)
    for band in report["high_frequency"].values():
        assert band["input_energy"] < 1e-12
        assert band["target_energy"] < 1e-12
        assert band["output_energy"] < 1e-12
        assert band["output_to_input"] is None
        assert band["target_to_input"] is None
        assert band["output_to_target"] is None
    json.dumps(report, allow_nan=False)


def test_excluded_values_cannot_leak_into_any_detail_or_noise_metric():
    generator = torch.Generator().manual_seed(25)
    images = [torch.rand(2, 3, 16, 24, generator=generator) for _ in range(4)]
    mask = torch.ones_like(images[0][:, :1])
    mask[0, :, 3:11, 8:15] = 0
    mask[0, :, :2] = 0.25
    mask[1] = 0
    expected = measure(*images, mask=mask)
    changed = [value.masked_fill(mask.expand_as(value) == 0, float("nan")) for value in images]
    assert measure(*changed, mask=mask) == expected
    assert expected["valid_rgb_values"] == float(mask.sum()) * 3


def test_noise_reports_post_clipping_and_preclamp_changes_separately():
    image = torch.full((1, 3, 16, 24), 0.5)
    totals = detail_module().DetailDiagnostics()
    totals.add(
        output(image + 0.5, image + 0.75),
        output(image + 0.5, image + 1.0),
        {"source": image, "target": image, "loss_mask": torch.ones_like(image[:, :1])},
    )
    report = totals.metrics()["noise_sensitivity"]
    assert report["rgb_mae"] == 0
    assert report["preclamp_mae"] == 0.25
    assert all(value == 0 for value in report["high_frequency_rms"].values())


def test_accumulation_weights_valid_pixels_not_frame_means():
    texture = checkerboard()
    image = 0.5 + 0.1 * texture
    original = measure(image, image, image)
    totals = detail_module().DetailDiagnostics()
    for amplitude, coverage in ((0.1, 1.0), (0.2, 0.25)):
        value = 0.5 + amplitude * texture
        tensors = {"source": value, "target": value, "loss_mask": torch.full_like(value[:, :1], coverage)}
        totals.add(output(value), output(value + amplitude), tensors)
    actual = totals.metrics()
    assert actual["valid_rgb_values"] == original["valid_rgb_values"] * 1.25
    for band, values in actual["high_frequency"].items():
        assert values["input_energy"] == pytest.approx(original["high_frequency"][band]["input_energy"] * 1.6, rel=1e-5)
    assert actual["noise_sensitivity"]["rgb_mae"] == pytest.approx(0.12, rel=1e-5)


def test_empty_diagnostics_reject_missing_supervision():
    totals = detail_module().DetailDiagnostics()
    with pytest.raises(ValueError, match="supervised"):
        totals.metrics()


def test_diagnostics_are_fp32_under_autocast_and_never_build_gradients():
    image = (0.5 + 0.1 * checkerboard()).requires_grad_()
    expected = measure(image, image, image)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = measure(image, image, image)
    assert actual == expected
    assert image.grad is None
    json.dumps(actual, allow_nan=False)


@pytest.mark.parametrize("enabled", [False, True])
def test_diagnostics_preserve_external_tf32_runtime_flags(monkeypatch, enabled):
    module = detail_module()
    real_lowpass = module.masked_gaussian_lowpass
    flags = []

    def observed(*args, **kwargs):
        flags.append((torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32))
        return real_lowpass(*args, **kwargs)

    monkeypatch.setattr(module, "masked_gaussian_lowpass", observed)
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = enabled
        image = 0.5 + 0.1 * checkerboard()
        measure(image, image, image)
        assert flags and all(value == (enabled, enabled) for value in flags)
        assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == (enabled, enabled)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def test_nonfinite_supervised_values_are_not_silently_reported():
    image = torch.ones(1, 3, 8, 12)
    corrupted = image.clone()
    corrupted[..., 2, 3] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite detail"):
        measure(corrupted, image, image)
