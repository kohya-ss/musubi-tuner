"""Tests for the pluggable loss function resolver (training/loss.py)."""

import sys
import types
from dataclasses import dataclass, field

import pytest
import torch

from musubi_tuner.training.loss import (
    BUILTIN_LOSS_FNS,
    mse_loss,
    normalize_loss_output,
    parse_loss_fn_args,
    resolve_loss_fn,
)


@dataclass
class _FakeOutput:
    pred: torch.Tensor
    target: torch.Tensor
    extra: dict = field(default_factory=dict)


def _make_args(**overrides):
    ns = types.SimpleNamespace(weighting_scheme="none", loss_fn="mse", loss_fn_args=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


# --- parse_loss_fn_args ---


def test_parse_none_and_empty():
    assert parse_loss_fn_args(None) == {}
    assert parse_loss_fn_args([]) == {}


def test_parse_literal_values():
    kwargs = parse_loss_fn_args(["weight=0.1", "level=2", "normalize=True"])
    assert kwargs == {"weight": 0.1, "level": 2, "normalize": True}


def test_parse_bare_string_falls_back():
    # transform=swt: "swt" is not a Python literal -> kept as raw string
    assert parse_loss_fn_args(["transform=swt"]) == {"transform": "swt"}


def test_parse_value_containing_equals():
    assert parse_loss_fn_args(["expr=a=b"]) == {"expr": "a=b"}


def test_parse_missing_equals_raises():
    with pytest.raises(ValueError, match="key=value"):
        parse_loss_fn_args(["justakey"])


# --- built-in mse ---


def test_mse_matches_manual_computation():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    output = _FakeOutput(pred=pred, target=target)
    timesteps = torch.tensor([100.0, 500.0])

    loss, metrics = mse_loss(_make_args(), output, timesteps, None, torch.float32, torch.float32, 0)

    expected = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(loss, expected)
    assert metrics == {}


# --- resolve_loss_fn ---


def test_resolve_builtin_mse():
    fn = resolve_loss_fn("mse")
    assert fn is BUILTIN_LOSS_FNS["mse"]


def test_resolve_unknown_builtin_lists_valid_names():
    with pytest.raises(ValueError, match="mse"):
        resolve_loss_fn("nope")


def test_resolve_bad_import_path():
    with pytest.raises(ImportError, match="no_such_module_xyz"):
        resolve_loss_fn("no_such_module_xyz.Loss")


def test_resolve_missing_attribute():
    with pytest.raises(ImportError, match="no_such_attr"):
        resolve_loss_fn("musubi_tuner.training.loss.no_such_attr")


def test_resolve_dotted_function_binds_kwargs():
    mod = types.ModuleType("fake_losses")

    def fn_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step, scale=1.0):
        return torch.nn.functional.mse_loss(output.pred, output.target) * scale

    mod.fn_loss = fn_loss
    sys.modules["fake_losses"] = mod
    try:
        fn = resolve_loss_fn("fake_losses.fn_loss", ["scale=2.0"])
        output = _FakeOutput(pred=torch.ones(2, 2), target=torch.zeros(2, 2))
        loss = fn(_make_args(), output, torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0)
        assert torch.allclose(loss, torch.tensor(2.0))
    finally:
        del sys.modules["fake_losses"]


def test_resolve_dotted_class_instantiates_with_kwargs():
    mod = types.ModuleType("fake_losses_cls")

    class ClassLoss(torch.nn.Module):
        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = scale

        def forward(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
            loss = torch.nn.functional.mse_loss(output.pred, output.target) * self.scale
            return loss, {"loss/scaled": float(loss)}

    mod.ClassLoss = ClassLoss
    sys.modules["fake_losses_cls"] = mod
    try:
        fn = resolve_loss_fn("fake_losses_cls.ClassLoss", ["scale=3.0"])
        assert isinstance(fn, torch.nn.Module)
        assert fn.scale == 3.0
    finally:
        del sys.modules["fake_losses_cls"]


# --- normalize_loss_output ---


def test_normalize_bare_tensor():
    t = torch.tensor(1.5)
    loss, metrics = normalize_loss_output(t)
    assert loss is t
    assert metrics == {}


def test_normalize_tuple():
    t = torch.tensor(1.5)
    loss, metrics = normalize_loss_output((t, {"loss/aux": 0.5}))
    assert loss is t
    assert metrics == {"loss/aux": 0.5}
