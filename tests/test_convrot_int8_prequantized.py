import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import (
    ConvRotInt8Quantizer,
    apply_convrot_int8_monkey_patch,
    canonicalize_convrot_int8_key,
    convrot_int8_linear_forward_patch,
    has_comfy_quant_tensors,
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


def _load_prequantized(*paths, weight_hook=None):
    """Conversion-only load: an empty target list disables dynamic quantization."""
    quantizer = ConvRotInt8Quantizer(target_layer_keys=[])
    state_dict = quantizer.load_and_quantize([str(path) for path in paths], None, weight_hook=weight_hook)
    return state_dict, quantizer


def test_probe_detects_comfy_quant_tensors(tmp_path):
    prequantized = _save(tmp_path / "artifact.safetensors", _triple())
    ordinary = _save(tmp_path / "ordinary.safetensors", {"linear.weight": torch.zeros(8, 16, dtype=torch.bfloat16)})

    assert has_comfy_quant_tensors([prequantized]) is True
    assert has_comfy_quant_tensors([ordinary]) is False
    assert has_comfy_quant_tensors([ordinary, prequantized]) is True


def test_conversion_canonicalizes_comfy_scales_and_keeps_per_layer_groups(tmp_path):
    path = _save(
        tmp_path / "artifact.safetensors",
        {
            **_triple("a", groupsize=4, in_features=16, out_features=8),
            **_triple("b", groupsize=16, in_features=64, out_features=12),
        },
    )

    state_dict, quantizer = _load_prequantized(path)

    assert state_dict["a.weight"].dtype is torch.int8
    assert state_dict["a.scale_weight"].dtype is torch.float32
    assert state_dict["a.scale_weight"].shape == (8, 1)
    assert "a.weight_scale" not in state_dict
    assert "a.comfy_quant" not in state_dict
    assert quantizer.module_groupsizes == {"a": 4, "b": 16}


def test_conversion_accepts_json_whitespace(tmp_path):
    tensors = _triple()
    tensors["linear.comfy_quant"] = _payload(4, whitespace=True)
    path = _save(tmp_path / "whitespace.safetensors", tensors)

    state_dict, _quantizer = _load_prequantized(path)

    assert state_dict["linear.weight"].dtype is torch.int8


@pytest.mark.parametrize(
    ("control", "match"),
    [
        (torch.tensor(list(b"{"), dtype=torch.uint8), "JSON"),
        (torch.tensor(list(b"[]"), dtype=torch.uint8), "object"),
        (_payload(4, format="float8_e4m3fn"), "format"),
        (_payload(4, convrot=False), "convrot"),
        (_payload(128), "power of 4"),
        (_payload(512), "power of 4"),
    ],
)
def test_conversion_rejects_invalid_control_payloads_with_context(tmp_path, control, match):
    tensors = _triple()
    tensors["linear.comfy_quant"] = control
    path = _save(tmp_path / "bad-control.safetensors", tensors)

    with pytest.raises(ValueError, match=match) as error:
        _load_prequantized(path)

    assert "linear" in str(error.value)


@pytest.mark.parametrize("missing_key", ["linear.weight", "linear.weight_scale"])
def test_conversion_rejects_missing_siblings(tmp_path, missing_key):
    tensors = _triple()
    del tensors[missing_key]
    path = _save(tmp_path / "missing.safetensors", tensors)

    with pytest.raises(ValueError, match=r"linear is missing tensors"):
        _load_prequantized(path)


def test_conversion_rejects_int8_weight_without_a_declared_triple(tmp_path):
    path = _save(tmp_path / "orphan.safetensors", {"orphan.weight": torch.zeros(8, 16, dtype=torch.int8)})

    with pytest.raises(ValueError, match="comfy_quant"):
        _load_prequantized(path)


def test_conversion_rejects_scale_without_a_declared_triple(tmp_path):
    tensors = {"linear.weight_scale": torch.ones(8, 1, dtype=torch.float32)}
    path = _save(tmp_path / "orphan-scale.safetensors", tensors)

    with pytest.raises(ValueError, match="without a matching"):
        _load_prequantized(path)


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("linear.weight", torch.zeros(8, 16, dtype=torch.bfloat16), "must be int8"),
        ("linear.weight_scale", torch.ones(8, 1, dtype=torch.float16), "must be F32"),
        ("linear.comfy_quant", torch.zeros(16, dtype=torch.int8), "uint8"),
    ],
)
def test_conversion_rejects_triple_dtype_mismatches(tmp_path, key, replacement, match):
    tensors = _triple()
    tensors[key] = replacement
    path = _save(tmp_path / "bad-dtype.safetensors", tensors)

    with pytest.raises(ValueError, match=match):
        _load_prequantized(path)


@pytest.mark.parametrize("scale_shape", [(8,), (1, 8), (7, 1), (8, 2)])
def test_conversion_rejects_scale_shape_mismatches(tmp_path, scale_shape):
    tensors = _triple()
    tensors["linear.weight_scale"] = torch.ones(scale_shape, dtype=torch.float32)
    path = _save(tmp_path / "bad-scale-shape.safetensors", tensors)

    with pytest.raises(ValueError, match="scale shape"):
        _load_prequantized(path)


def test_conversion_rejects_group_that_does_not_divide_input_width(tmp_path):
    path = _save(tmp_path / "indivisible.safetensors", _triple(groupsize=16, in_features=24))

    with pytest.raises(ValueError, match="divisible"):
        _load_prequantized(path)


def test_conversion_rejects_lora_merge_into_prequantized_weights(tmp_path):
    path = _save(tmp_path / "artifact.safetensors", _triple())

    with pytest.raises(ValueError, match="Cannot merge LoRA"):
        _load_prequantized(path, weight_hook=lambda key, value, keep_on_calc_device=False: value)


def test_canonicalize_maps_comfy_scale_suffix_only():
    assert canonicalize_convrot_int8_key("linear.weight_scale") == "linear.scale_weight"
    assert canonicalize_convrot_int8_key("linear.weight") == "linear.weight"


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(16, 8, bias=False)
        self.nested = nn.ModuleDict({"b": nn.Linear(64, 12, bias=True)})


def _quantized_state(model, module_groups):
    state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for module_path, groupsize in module_groups:
        weight = state_dict.pop(f"{module_path}.weight")
        quantized_weight, scale = quantize_int8_convrot_weight(weight.float(), groupsize)
        state_dict[f"{module_path}.weight"] = quantized_weight
        state_dict[f"{module_path}.scale_weight"] = scale
    return state_dict


def _patch_and_load(model, state_dict, groupsize_map, bwd_mode="bf16"):
    apply_convrot_int8_monkey_patch(model, state_dict, bwd_mode=bwd_mode, groupsize_map=groupsize_map)
    model.requires_grad_(False)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def test_apply_patch_installs_frozen_int8_weights_and_fp32_scales():
    model = _TinyModel()
    state_dict = _quantized_state(model, (("a", 4), ("nested.b", 16)))

    _patch_and_load(model, state_dict, {"a": 4, "nested.b": 16})

    assert type(model.a) is nn.Linear
    assert model.a.weight.dtype is torch.int8
    assert not model.a.weight.requires_grad
    assert model.a.scale_weight.dtype is torch.float32
    assert model.a.scale_weight.shape == (8, 1)
    assert model.a._convrot_groupsize == 4
    assert model.nested["b"]._convrot_groupsize == 16
    assert model.a._convrot_bwd_mode == "bf16"
    assert model.a.forward.__func__ is convrot_int8_linear_forward_patch
    assert model.is_convrot_int8 is True
    assert model.convrot_int8_layer_count == 2


def test_apply_patch_rejects_a_scale_for_a_missing_module():
    model = _TinyModel()
    state_dict = _quantized_state(model, (("a", 4),))
    state_dict["ghost.scale_weight"] = torch.ones(8, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="missing module ghost"):
        apply_convrot_int8_monkey_patch(model, state_dict, groupsize_map={"a": 4})


def test_int8_backward_on_cpu_raises_a_clear_cuda_error(monkeypatch):
    from musubi_tuner.modules import convrot_int8_utils

    monkeypatch.setattr(convrot_int8_utils, "HAS_TRITON", True)
    model = _TinyModel()
    state_dict = _quantized_state(model, (("a", 4), ("nested.b", 16)))
    _patch_and_load(model, state_dict, {"a": 4, "nested.b": 16}, bwd_mode="int8")
    inputs = torch.randn(2, 16, requires_grad=True)

    with pytest.raises(RuntimeError, match=r"int8.*CUDA"):
        model.a(inputs).sum().backward()
