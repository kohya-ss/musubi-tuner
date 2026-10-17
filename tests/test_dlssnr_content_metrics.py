"""Content retention is a spatial DINOv3 measurement, never another training loss."""

from copy import deepcopy
import importlib

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from test_dlssnr_config import make_args
from test_dlssnr_dino import SETTINGS, tiny_dino_loss


CONTENT_SETTINGS = {"model_type": "small", "layer": -4, "resize": 224, "use_gram": False, "use_norm": True}


def content_module():
    return importlib.import_module("musubi_tuner.dlssnr.content_metrics")


def tiny_content_metric(dino_loss=None):
    return content_module().NRContentMetric(tiny_dino_loss(CONTENT_SETTINGS))


@pytest.mark.parametrize("lora", [False, True])
def test_content_evaluation_is_opt_in_and_requires_validation_without_enabling_loss(tmp_path, lora):
    args = make_args(tmp_path, lora=lora)
    assert getattr(args, "eval_content_preservation", None) is False
    original = build_train_config(args, lora=lora)
    assert "content_preservation" not in original["evaluation"]
    args.eval_content_preservation = True
    with pytest.raises(ValueError, match="evaluation requires"):
        build_train_config(args, lora=lora)
    dataset = {"general": {"resolution": 48}, "datasets": [{"train_manifest": "train.jsonl", "validation_manifest": "val.jsonl"}]}
    plain = build_train_config(make_args(tmp_path, lora=lora, dataset=dataset), lora=lora)
    enabled = build_train_config(make_args(tmp_path, ["--eval_content_preservation"], lora=lora, dataset=dataset), lora=lora)
    assert enabled["evaluation"]["content_preservation"] is True
    assert enabled["loss"] == plain["loss"]
    assert "dino_loss" not in enabled
    assert config_sha256(enabled) != config_sha256(plain)


def test_content_score_detects_patch_displacement_that_gram_ignores():
    generator = torch.Generator().manual_seed(13)
    source = torch.rand(1, 3, 24, 24, generator=generator)
    displaced = source.roll(4, dims=-1)
    mask = torch.ones(1, 1, 24, 24)
    metric = tiny_content_metric()
    assert metric(source, source, mask).item() == 0
    assert metric(displaced, source, mask).item() > 1e-4
    assert tiny_dino_loss()(displaced, source, mask).item() < 1e-6


def test_content_metric_matches_mask_weighted_normalized_spatial_patch_mse():
    metric = tiny_content_metric()
    backend = metric.feature_loss.backend
    generator = torch.Generator().manual_seed(31)
    output = torch.rand(2, 3, 8, 12, generator=generator)
    source = torch.rand(2, 3, 8, 12, generator=generator)
    mask = torch.ones(2, 1, 8, 12)
    mask[0, :, :4, :4] = 0.25
    mask[1, :, 4:, 4:] = 0.5
    features = []
    for image in (output, source):
        patches = F.avg_pool2d((image - backend.mean) / backend.std, 4)
        feature = backend.model.projection(patches).flatten(2).transpose(1, 2)
        features.append(F.normalize(feature, dim=-1))
    weights = F.avg_pool2d(mask, 4).flatten(1)
    expected = ((features[0] - features[1]).square().mean(-1) * weights).sum(1) / weights.sum(1)
    torch.testing.assert_close(metric(output, source, mask), expected)


def test_content_metric_is_frozen_no_grad_and_does_not_change_modes_rng_or_tf32():
    metric = tiny_content_metric()
    metric.train()
    output = torch.full((1, 3, 16, 24), 0.8, requires_grad=True)
    source = torch.full_like(output, 0.2, requires_grad=True)
    mask = torch.ones_like(source[:, :1])
    before = torch.random.get_rng_state().clone()
    flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with torch.autocast("cpu", dtype=torch.bfloat16):
        score = metric(output, source, mask)
    assert score.dtype == torch.float32 and not score.requires_grad
    assert all(not child.training for child in metric.modules())
    assert all(not parameter.requires_grad and parameter.grad is None for parameter in metric.parameters())
    assert output.grad is source.grad is None
    assert metric.feature_loss.backend.grad_modes == [False, False]
    torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == flags


def test_content_metric_masks_before_resizing_and_ignores_empty_support():
    metric = content_module().NRContentMetric(tiny_dino_loss({**CONTENT_SETTINGS, "resize": 16}))
    generator = torch.Generator().manual_seed(18)
    output = torch.rand(2, 3, 32, 48, generator=generator)
    source = torch.rand(2, 3, 32, 48, generator=generator)
    mask = torch.ones(2, 1, 32, 48)
    mask[0, :, 8:20, 12:32] = 0
    mask[1] = 0
    expected = metric(output, source, mask)
    changed_output = output.masked_fill(mask.expand_as(output) == 0, float("nan"))
    changed_source = source.masked_fill(mask.expand_as(source) == 0, float("inf"))
    torch.testing.assert_close(metric(changed_output, changed_source, mask), expected, rtol=0, atol=0)
    assert expected[1] == 0


@pytest.mark.parametrize("change", ["gram", "unnormalized", "bad_mask", "bad_source", "bad_backend"])
def test_incorrect_content_protocol_or_nonfinite_measurements_are_rejected(change):
    module = content_module()
    if change in ("gram", "unnormalized"):
        settings = {**CONTENT_SETTINGS, "use_gram": change == "gram", "use_norm": change != "unnormalized"}
        with pytest.raises(ValueError, match="spatial|normalized"):
            module.NRContentMetric(tiny_dino_loss(settings))
        return
    metric = tiny_content_metric()
    source = torch.full((1, 3, 8, 12), 0.5)
    mask = torch.ones_like(source[:, :1])
    if change == "bad_mask":
        mask[..., 0, 0] = -1
    elif change == "bad_source":
        source[..., 0, 0] = float("inf")
    else:
        metric.feature_loss.backend.model.projection.weight.data.fill_(float("nan"))
    with pytest.raises((ValueError, RuntimeError), match="mask|finite"):
        metric(source, source, mask)


def test_content_summary_weights_original_valid_rgb_mass_not_frame_means():
    summary = content_module().ContentMetrics()
    mask = torch.ones(3, 1, 2, 2)
    mask[1] *= 0.5
    mask[2] = 0
    summary.add(torch.tensor([1.0, 3.0, 100.0]), mask)
    measured = summary.metrics()
    assert measured["valid_rgb_values"] == 18
    assert measured["dinov3_patch_mse"] == pytest.approx(5 / 3)
    with pytest.raises(ValueError, match="pixels"):
        content_module().ContentMetrics().metrics()


def test_factory_reuses_matching_training_backend_without_changing_its_gram_objective(monkeypatch):
    module = content_module()
    training_loss = tiny_dino_loss({**SETTINGS, "resize": 384, "use_norm": False})
    identity = deepcopy(training_loss.identity)
    tensors = deepcopy(training_loss.state_dict())
    source = torch.linspace(0.1, 0.9, 3 * 8 * 12).reshape(1, 3, 8, 12)
    target, mask = source.roll(4, -1), torch.ones_like(source[:, :1])
    expected = training_loss(source, target, mask).detach()

    def unexpected(*args, **kwargs):
        pytest.fail("A compatible, frozen training backbone should not be loaded twice")

    monkeypatch.setattr(module, "create_dino_loss", unexpected)
    metric = module.create_content_metric(training_loss)
    assert metric.feature_loss.backend is training_loss.backend
    assert metric.feature_loss.settings == CONTENT_SETTINGS
    assert metric(source, target, mask).item() > 0
    assert training_loss.identity == identity
    torch.testing.assert_close(training_loss(source, target, mask), expected, rtol=0, atol=0)
    torch.testing.assert_close(training_loss.state_dict(), tensors, rtol=0, atol=0)


def test_factory_does_not_reuse_a_different_feature_layer(monkeypatch):
    module = content_module()
    monkeypatch.setattr(module, "create_dino_loss", tiny_dino_loss)
    training_loss = tiny_dino_loss({**SETTINGS, "layer": -2})
    metric = module.create_content_metric(training_loss)
    assert metric.feature_loss.backend is not training_loss.backend
    assert metric.identity["feature_model"]["settings"] == CONTENT_SETTINGS


def test_content_protocol_binds_frozen_weights_and_input_reference():
    metric = tiny_content_metric()
    assert metric.identity["reference"] == "input_proxy_rgb"
    assert metric.identity["metric"] == "normalized_spatial_patch_mse"
    teacher = tiny_dino_loss(CONTENT_SETTINGS)
    teacher.backend.mean.add_(0.1)
    from musubi_tuner.dlssnr.dino_loss import NRDinoLoss

    changed = content_module().NRContentMetric(NRDinoLoss(teacher.backend, CONTENT_SETTINGS, provenance={"provider": "synthetic"}))
    assert changed.identity["feature_model"]["weights_sha256"] != metric.identity["feature_model"]["weights_sha256"]
