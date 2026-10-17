"""Mask-aware DINO loss contracts, without downloading pretrained weights."""

import importlib
import importlib.util
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from test_dlssnr_config import make_args


SETTINGS = {"model_type": "small", "layer": -4, "resize": 224, "use_gram": True, "use_norm": True}


def dino_module():
    name = "musubi_tuner.dlssnr.dino_loss"
    assert importlib.util.find_spec(name) is not None, "NR DINO integration is not implemented"
    return importlib.import_module(name)


class TinyDinoBackend(torch.nn.Module):
    """The SenseCraft feature interface, not a pretrained perceptual model."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.config = SimpleNamespace(patch_size=4, num_register_tokens=2)
        self.model.projection = torch.nn.Conv2d(3, 4, 1)
        self.model.dropout = torch.nn.Dropout(0.7)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))
        self.input_range = (0, 1)
        self.inputs = []
        self.grad_modes = []

    def normalize_input(self, image):
        self.inputs.append(image.detach().clone())
        return (image - self.mean) / self.std

    def dinov3_fwd(self, image):
        self.grad_modes.append(torch.is_grad_enabled())
        patches = F.avg_pool2d(image, 4)
        features = self.model.dropout(self.model.projection(patches)).flatten(2).transpose(1, 2)
        prefix = features.new_full((len(image), 3, 4), 900)
        return torch.cat((prefix, features), dim=1)

    def gram_matrix(self, features, normalize=True):
        gram = features.transpose(1, 2).bmm(features)
        return gram / features.shape[1] if normalize else gram


def tiny_dino_loss(settings=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        return dino_module().NRDinoLoss(
            TinyDinoBackend(), SETTINGS if settings is None else settings, provenance={"provider": "synthetic_test_only"}
        )


@pytest.mark.parametrize("lora", [False, True])
def test_dino_is_optional_and_gram_patch_loss_is_the_enabled_default(tmp_path, lora):
    args = make_args(tmp_path, lora=lora)
    assert getattr(args, "dino_loss_weight", None) == 0
    plain = build_train_config(args, lora=lora)
    assert "dino_loss" not in plain and "dino" not in plain["loss"]
    enabled = build_train_config(make_args(tmp_path, ["--dino_loss_weight", "0.1"], lora=lora), lora=lora)
    assert enabled["dino_loss"] == SETTINGS
    assert enabled["loss"]["dino"] == 0.1
    assert config_sha256(plain) != config_sha256(enabled)
    custom = make_args(
        tmp_path,
        [
            "--dino_loss_weight",
            "0.1",
            "--dino_loss_model_type",
            "base",
            "--dino_loss_layer",
            "-2",
            "--dino_loss_resize",
            "384",
            "--no-dino_loss_use_gram",
            "--no-dino_loss_use_norm",
        ],
        lora=lora,
    )
    assert build_train_config(custom, lora=lora)["dino_loss"] == {
        "model_type": "base",
        "layer": -2,
        "resize": 384,
        "use_gram": False,
        "use_norm": False,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("dino_loss_weight", -1),
        ("dino_loss_weight", float("nan")),
        ("dino_loss_weight", float("inf")),
        ("dino_loss_model_type", "unknown"),
        ("dino_loss_layer", True),
        ("dino_loss_layer", 0.5),
        ("dino_loss_resize", 0),
        ("dino_loss_resize", 225),
        ("dino_loss_resize", 2048),
        ("dino_loss_use_gram", "false"),
        ("dino_loss_use_norm", 1),
    ],
)
def test_dino_configuration_rejects_invalid_settings_before_model_load(tmp_path, field, value):
    args = make_args(tmp_path)
    args.dino_loss_weight = 0.1
    setattr(args, field, value)
    with pytest.raises(ValueError, match="dino_loss"):
        build_train_config(args)


def test_disabled_dino_rejects_ignored_options_and_cannot_replace_all_base_losses(tmp_path):
    args = make_args(tmp_path)
    args.dino_loss_resize = 384
    with pytest.raises(ValueError, match="dino_loss_weight"):
        build_train_config(args)
    args.dino_loss_weight = 0.1
    args.loss_pre = args.loss_out = args.loss_edge = args.loss_temporal = 0
    with pytest.raises(ValueError, match="loss weight"):
        build_train_config(args)


def test_dino_freezes_eval_mode_and_gradients_but_preserves_prediction_gradients():
    loss_fn = tiny_dino_loss()
    loss_fn.train()
    assert all(not module.training for module in loss_fn.modules())
    assert all(not parameter.requires_grad for parameter in loss_fn.parameters())
    generator = torch.Generator().manual_seed(4)
    prediction = torch.rand(2, 3, 16, 24, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 16, 24, generator=generator, requires_grad=True)
    mask = torch.ones(2, 1, 16, 24)
    rng = torch.get_rng_state().clone()
    result = loss_fn(prediction, target, mask)
    assert result.shape == (2,)
    result.sum().backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert target.grad is None
    assert all(parameter.grad is None for parameter in loss_fn.parameters())
    assert loss_fn.backend.grad_modes == [True, False]
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert all(0 <= image.min() <= image.max() <= 1 for image in loss_fn.backend.inputs)


@pytest.mark.parametrize("gram", [False, True])
def test_dino_ignores_excluded_values_before_resize_and_handles_empty_masks(gram):
    loss_fn = tiny_dino_loss({**SETTINGS, "resize": 16, "use_gram": gram})
    generator = torch.Generator().manual_seed(18)
    prediction = torch.rand(2, 3, 32, 48, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 32, 48, generator=generator)
    mask = torch.ones(2, 1, 32, 48)
    mask[0, :, 7:20, 11:35] = 0
    mask[0, :, :5] = 0.25
    mask[1] = 0
    original = loss_fn(prediction, target, mask)
    gradient = torch.autograd.grad(original.sum(), prediction)[0]
    changed_target = target.masked_fill(mask.expand_as(target) == 0, 1000)
    changed_prediction = prediction.detach().masked_fill(mask.expand_as(target) == 0, -1000).requires_grad_()
    changed = loss_fn(changed_prediction, changed_target, mask)
    torch.testing.assert_close(changed, original, rtol=0, atol=0)
    actual_gradient = torch.autograd.grad(changed.sum(), changed_prediction)[0]
    torch.testing.assert_close(actual_gradient, gradient, rtol=0, atol=0)
    assert gradient[mask.expand_as(gradient) == 0].eq(0).all()
    assert original[1] == 0 and torch.isfinite(gradient).all()


def test_gram_comparison_is_patch_order_invariant_not_positionwise_regression():
    generator = torch.Generator().manual_seed(13)
    prediction = torch.rand(1, 3, 24, 24, generator=generator)
    target = prediction.roll(4, dims=-1)
    mask = torch.ones(1, 1, 24, 24)
    gram = tiny_dino_loss()(prediction, target, mask)
    feature = tiny_dino_loss({**SETTINGS, "use_gram": False})(prediction, target, mask)
    assert gram.max() < 1e-6
    assert feature.min() > 1e-4


def test_dino_resize_preserves_aspect_ratio_and_only_pads_to_patch_grid():
    loss_fn = tiny_dino_loss({**SETTINGS, "resize": 32})
    source = torch.zeros(1, 3, 40, 80)
    loss_fn(source, source, torch.ones_like(source[:, :1]))
    assert loss_fn.backend.inputs[0].shape[-2:] == (16, 32)
    source = torch.zeros(1, 3, 13, 21)
    loss_fn(source, source, torch.ones_like(source[:, :1]))
    image = loss_fn.backend.inputs[-1]
    assert image.shape[-2:] == (16, 24)
    assert image[..., 13:, :].eq(0.5).all() and image[..., :, 21:].eq(0.5).all()


def test_dino_identity_binds_frozen_weights_and_normalization_buffers():
    original = tiny_dino_loss()
    same = tiny_dino_loss()
    assert original.identity == same.identity
    with torch.no_grad():
        same.backend.mean.add_(0.01)
    changed = dino_module().NRDinoLoss(same.backend, SETTINGS, provenance={"provider": "synthetic_test_only"})
    assert original.identity["weights_sha256"] != changed.identity["weights_sha256"]


def test_dino_stays_fp32_under_autocast():
    loss_fn = tiny_dino_loss()
    generator = torch.Generator().manual_seed(19)
    prediction = torch.rand(1, 3, 16, 24, generator=generator, requires_grad=True)
    target = torch.rand(1, 3, 16, 24, generator=generator)
    mask = torch.ones_like(prediction[:, :1])
    expected = loss_fn(prediction, target, mask)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = loss_fn(prediction, target, mask)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("gram", [False, True])
def test_dino_matches_weighted_patch_formula_without_special_tokens(gram):
    loss_fn = tiny_dino_loss({**SETTINGS, "use_gram": gram, "use_norm": False})
    generator = torch.Generator().manual_seed(31)
    prediction = torch.rand(2, 3, 8, 12, generator=generator)
    target = torch.rand(2, 3, 8, 12, generator=generator)
    mask = torch.ones(2, 1, 8, 12)
    mask[0, :, :4, :4] = 0.25
    mask[1, :, 4:, 4:] = 0.5
    features = []
    for image in (prediction, target):
        normalized = (image - loss_fn.backend.mean) / loss_fn.backend.std
        patches = F.avg_pool2d(normalized, 4)
        features.append(loss_fn.backend.model.projection(patches).flatten(2).transpose(1, 2))
    pred, wanted = features
    weights = F.avg_pool2d(mask, 4).flatten(1)
    mass = weights.sum(1)
    if gram:
        pred_gram = pred.transpose(1, 2).bmm(pred * weights[..., None]) / mass[:, None, None]
        target_gram = wanted.transpose(1, 2).bmm(wanted * weights[..., None]) / mass[:, None, None]
        expected = (pred_gram - target_gram).abs().mean((1, 2))
    else:
        expected = ((pred - wanted).square().mean(-1) * weights).sum(1) / mass
    torch.testing.assert_close(loss_fn(prediction, target, mask), expected)


def test_missing_optional_dependency_has_an_actionable_error(monkeypatch):
    import builtins

    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("sensecraft"):
            raise ModuleNotFoundError("test missing SenseCraft")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match=r'uv pip install ".\[dinov3\]"'):
        dino_module().create_dino_loss(SETTINGS)


@pytest.mark.parametrize("layer", [-4, 0, -1, 6, -7])
def test_real_sensecraft_api_without_pretrained_downloads(monkeypatch, layer):
    pytest.importorskip("sensecraft.loss")
    from transformers import DINOv3ViTConfig, DINOv3ViTModel
    from musubi_tuner.training.dlssnr_services import capture_rng

    calls = []

    def pretrained(model_name):
        calls.append(model_name)
        return DINOv3ViTModel(
            DINOv3ViTConfig(hidden_size=8, intermediate_size=16, num_hidden_layers=6, num_attention_heads=2, num_register_tokens=2)
        )

    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", pretrained)
    before = capture_rng()
    if layer in (6, -7):
        with pytest.raises(ValueError, match="dino_loss_layer"):
            dino_module().create_dino_loss({**SETTINGS, "layer": layer})
    else:
        loss_fn = dino_module().create_dino_loss({**SETTINGS, "layer": layer})
        resolved = layer if layer >= 0 else 6 + layer
        assert loss_fn.backend.loss_layer == resolved
        assert loss_fn.backend.num_layers == len(loss_fn.backend.model.layer) == resolved + 1
        provenance = loss_fn.identity["provenance"]
        assert provenance["resolved_layer"] == resolved
        assert provenance["sensecraft_version"] == "0.3.11"
        assert len(provenance["sensecraft_source_sha256"]) == 64
        loss_fn.train()
        image = torch.linspace(0.1, 0.9, 3 * 32 * 48).reshape(1, 3, 32, 48).requires_grad_()
        mask = torch.ones_like(image[:, :1])
        actual = loss_fn(image, image.detach().flip(-1), mask)
        actual.sum().backward()
        assert image.grad is not None and image.grad.abs().sum() > 0
        assert torch.isfinite(image.grad).all()
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in loss_fn.parameters())
        # Compare with the actual selected hidden state, not just the layer counter.
        inputs = loss_fn.backend.normalize_input(image.detach())
        with torch.no_grad():
            expected = loss_fn.backend.model(inputs, output_hidden_states=True).hidden_states[-1]
            features = loss_fn.backend.dinov3_fwd(inputs)
        torch.testing.assert_close(features, expected)
    assert calls == ["facebook/dinov3-vits16-pretrain-lvd1689m"]
    assert capture_rng()["python"] == before["python"]
    torch.testing.assert_close(capture_rng()["torch"], before["torch"], rtol=0, atol=0)


def test_dino_load_failure_restores_training_rng(monkeypatch):
    pytest.importorskip("sensecraft.loss")
    import random
    import numpy as np
    from transformers import DINOv3ViTModel
    from musubi_tuner.training.dlssnr_services import capture_rng

    def unavailable(*args, **kwargs):
        torch.rand(5)
        random.random()
        np.random.rand(5)
        raise OSError("test pretrained weights unavailable")

    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", unavailable)
    before = capture_rng()
    with pytest.raises(OSError, match="pretrained weights unavailable"):
        dino_module().create_dino_loss(SETTINGS)
    assert capture_rng()["python"] == before["python"]
    assert capture_rng()["numpy"] == before["numpy"]
    torch.testing.assert_close(capture_rng()["torch"], before["torch"], rtol=0, atol=0)
