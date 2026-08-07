import json
from dataclasses import FrozenInstanceError

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from musubi_tuner.modules.convrot_int8_utils import (
    ConvRotInt8Artifact,
    ConvRotInt8LayerSpec,
    canonicalize_convrot_int8_key,
    convrot_int8_linear_forward_patch,
    inspect_convrot_int8_artifact,
    prepare_convrot_int8_model,
)


def _payload(groupsize: int, *, whitespace: bool = False, **overrides) -> torch.Tensor:
    values = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
    values.update(overrides)
    separators = None if whitespace else (",", ":")
    raw = json.dumps(values, separators=separators).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _triple(module: str = "linear", *, groupsize: int = 4, in_features: int = 16, out_features: int = 8):
    return {
        f"{module}.weight": torch.zeros(out_features, in_features, dtype=torch.int8),
        f"{module}.weight_scale": torch.ones(out_features, 1, dtype=torch.float32),
        f"{module}.comfy_quant": _payload(groupsize),
    }


def _save(path, tensors):
    save_file(tensors, str(path))
    return path


def test_inspector_canonicalizes_comfy_scales_and_keeps_per_layer_groups(tmp_path):
    path = _save(
        tmp_path / "artifact.safetensors",
        {
            **_triple("root.a", groupsize=4, in_features=16, out_features=8),
            **_triple("root.b", groupsize=16, in_features=64, out_features=12),
        },
    )

    artifact = inspect_convrot_int8_artifact([path], key_normalizer=lambda key: key.removeprefix("root."))

    assert artifact is not None
    assert artifact.layers["a"].scale_key == "a.scale_weight"
    assert artifact.layers["a"].groupsize == 4
    assert artifact.layers["b"].groupsize == 16
    assert artifact.weight_keys == frozenset({"a.weight", "b.weight"})
    assert artifact.scale_keys == frozenset({"a.scale_weight", "b.scale_weight"})
    assert artifact.control_keys == frozenset({"a.comfy_quant", "b.comfy_quant"})


def test_inspector_accepts_json_whitespace(tmp_path):
    tensors = _triple()
    tensors["linear.comfy_quant"] = _payload(4, whitespace=True)
    path = _save(tmp_path / "whitespace.safetensors", tensors)

    assert inspect_convrot_int8_artifact([path]) is not None


def test_inspector_returns_none_for_an_ordinary_checkpoint(tmp_path):
    path = _save(tmp_path / "ordinary.safetensors", {"linear.weight": torch.zeros(8, 16, dtype=torch.bfloat16)})

    assert inspect_convrot_int8_artifact([path]) is None


@pytest.mark.parametrize(
    ("control", "match"),
    [
        (torch.tensor(list(b"{"), dtype=torch.uint8), "JSON"),
        (torch.tensor(list(b"[]"), dtype=torch.uint8), "object"),
        (_payload(4, format="float8_e4m3fn"), "format"),
        (_payload(4, convrot=False), "convrot"),
        (_payload(128), "power of four"),
        (_payload(512), "power of four"),
    ],
)
def test_inspector_rejects_invalid_control_payloads_with_context(tmp_path, control, match):
    tensors = _triple()
    tensors["linear.comfy_quant"] = control
    path = _save(tmp_path / "bad-control.safetensors", tensors)

    with pytest.raises(ValueError, match=match) as error:
        inspect_convrot_int8_artifact([path])

    assert str(path) in str(error.value)
    assert "linear" in str(error.value)


@pytest.mark.parametrize("missing_key", ["linear.weight", "linear.weight_scale"])
def test_inspector_rejects_missing_siblings(tmp_path, missing_key):
    tensors = _triple()
    del tensors[missing_key]
    path = _save(tmp_path / "missing.safetensors", tensors)

    with pytest.raises(ValueError, match="linear") as error:
        inspect_convrot_int8_artifact([path])

    assert str(path) in str(error.value)


def test_inspector_rejects_int8_weight_without_a_declared_triple(tmp_path):
    path = _save(tmp_path / "orphan.safetensors", {"orphan.weight": torch.zeros(8, 16, dtype=torch.int8)})

    with pytest.raises(ValueError, match="orphan") as error:
        inspect_convrot_int8_artifact([path])

    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("linear.weight", torch.zeros(8, 16, dtype=torch.bfloat16), "INT8"),
        ("linear.weight_scale", torch.ones(8, 1, dtype=torch.float16), "F32"),
        ("linear.comfy_quant", torch.zeros(16, dtype=torch.int8), "U8"),
    ],
)
def test_inspector_rejects_triple_dtype_mismatches(tmp_path, key, replacement, match):
    tensors = _triple()
    tensors[key] = replacement
    path = _save(tmp_path / "bad-dtype.safetensors", tensors)

    with pytest.raises(ValueError, match=match):
        inspect_convrot_int8_artifact([path])


@pytest.mark.parametrize("scale_shape", [(8,), (1, 8), (7, 1), (8, 2)])
def test_inspector_rejects_scale_shape_mismatches(tmp_path, scale_shape):
    tensors = _triple()
    tensors["linear.weight_scale"] = torch.ones(scale_shape, dtype=torch.float32)
    path = _save(tmp_path / "bad-scale-shape.safetensors", tensors)

    with pytest.raises(ValueError, match="scale"):
        inspect_convrot_int8_artifact([path])


def test_inspector_rejects_group_that_does_not_divide_input_width(tmp_path):
    path = _save(tmp_path / "indivisible.safetensors", _triple(groupsize=16, in_features=24))

    with pytest.raises(ValueError, match="divide"):
        inspect_convrot_int8_artifact([path])


def test_artifact_records_are_frozen():
    layer = ConvRotInt8LayerSpec("linear", "linear.weight", "linear.scale_weight", 4)
    artifact = ConvRotInt8Artifact({"linear": layer}, frozenset({"linear.comfy_quant"}))

    with pytest.raises(FrozenInstanceError):
        layer.groupsize = 16
    with pytest.raises(FrozenInstanceError):
        artifact.control_keys = frozenset()
    assert canonicalize_convrot_int8_key("linear.weight_scale") == "linear.scale_weight"
    assert canonicalize_convrot_int8_key("linear.weight") == "linear.weight"


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 8, bias=False)
        self.nested = nn.ModuleDict({"b": nn.Linear(64, 12, bias=True)})


def _artifact(*module_groups):
    layers = {
        module: ConvRotInt8LayerSpec(module, f"{module}.weight", f"{module}.scale_weight", groupsize)
        for module, groupsize in module_groups
    }
    return ConvRotInt8Artifact(layers, frozenset(f"{module}.comfy_quant" for module, _ in module_groups))


def test_prepare_model_installs_frozen_int8_weights_and_fp32_scales():
    with torch.device("meta"):
        model = _TinyModel()
    artifact = _artifact(("a", 4), ("nested.b", 16))

    result = prepare_convrot_int8_model(model, artifact, bwd_mode="bf16")

    assert result is model
    assert type(model.a) is nn.Linear
    assert model.a.weight.device.type == "meta"
    assert model.a.weight.dtype is torch.int8
    assert not model.a.weight.requires_grad
    assert model.a.scale_weight.device.type == "meta"
    assert model.a.scale_weight.dtype is torch.float32
    assert model.a.scale_weight.shape == (8, 1)
    assert model.a._convrot_groupsize == 4
    assert model.nested["b"]._convrot_groupsize == 16
    assert model.a._convrot_bwd_mode == "bf16"
    assert model.a.forward.__func__ is convrot_int8_linear_forward_patch
    assert model.is_convrot_int8 is True
    assert model.convrot_int8_layer_count == 2


def test_prepare_model_rejects_a_missing_module():
    with pytest.raises(ValueError, match="missing"):
        prepare_convrot_int8_model(_TinyModel(), _artifact(("missing", 4)))


def test_prepare_model_rejects_linear_subclasses():
    class LinearSubclass(nn.Linear):
        pass

    model = _TinyModel()
    model.a = LinearSubclass(16, 8, bias=False)

    with pytest.raises(TypeError, match="exact nn.Linear"):
        prepare_convrot_int8_model(model, _artifact(("a", 4)))
