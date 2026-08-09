import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from musubi_tuner.modules.comfy_quant_utils import (
    FORMAT_CONVROT_INT8,
    FORMAT_INT8_TENSORWISE,
    FORMAT_NVFP4,
    classify_comfy_quant_spec,
    detect_comfy_quant_formats,
)
from musubi_tuner.modules import nvfp4_utils
from musubi_tuner.modules.nvfp4_utils import (
    NvFp4Quantizer,
    apply_nvfp4_monkey_patch,
    dequantize_nvfp4,
    from_blocked,
    quantize_nvfp4_activation,
    to_blocked,
)


def _payload(**values) -> torch.Tensor:
    raw = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _nvfp4_payload() -> torch.Tensor:
    return _payload(format="nvfp4", full_precision_matrix_mult=True)


def _int8_payload() -> torch.Tensor:
    return _payload(format="int8_tensorwise")


def _quantize_linear_weight(weight: torch.Tensor):
    """Produce the ComfyUI on-disk triple for a float weight (rows must be a multiple of 16)."""
    packed, block_scale, scale, orig_rows = quantize_nvfp4_activation(weight.float())
    assert orig_rows == weight.shape[0]
    return packed, block_scale, scale


def _nvfp4_module(module: str, weight: torch.Tensor, *, pre_quant_scale: torch.Tensor = None) -> dict:
    packed, block_scale, scale = _quantize_linear_weight(weight)
    tensors = {
        f"{module}.weight": packed,
        f"{module}.weight_scale": block_scale,
        f"{module}.weight_scale_2": scale,
        f"{module}.comfy_quant": _nvfp4_payload(),
    }
    if pre_quant_scale is not None:
        tensors[f"{module}.pre_quant_scale"] = pre_quant_scale
    return tensors


def _int8_embedding_module(module: str, weight: torch.Tensor) -> dict:
    scale = weight.abs().amax(dim=1, keepdim=True).float() / 127.0
    quantized = torch.round(weight.float() / scale).clamp(-127, 127).to(torch.int8)
    return {
        f"{module}.weight": quantized,
        f"{module}.weight_scale": scale,
        f"{module}.comfy_quant": _int8_payload(),
    }


def _save(path, tensors):
    save_file(tensors, str(path))
    return path


def _load(*paths, weight_hook=None):
    quantizer = NvFp4Quantizer()
    state_dict = quantizer.load_and_quantize([str(path) for path in paths], None, weight_hook=weight_hook)
    return state_dict, quantizer


# region format classification and probing


def test_classify_comfy_quant_spec_labels():
    assert classify_comfy_quant_spec({"format": "int8_tensorwise", "convrot": True}) == FORMAT_CONVROT_INT8
    assert classify_comfy_quant_spec({"format": "int8_tensorwise"}) == FORMAT_INT8_TENSORWISE
    assert classify_comfy_quant_spec({"format": "nvfp4"}) == FORMAT_NVFP4
    assert classify_comfy_quant_spec({"format": "something_else"}) == "something_else"


def test_detect_comfy_quant_formats(tmp_path):
    nvfp4_file = _save(
        tmp_path / "nvfp4.safetensors",
        {
            **_nvfp4_module("linear", torch.randn(16, 32)),
            **_int8_embedding_module("embed", torch.randn(8, 32)),
        },
    )
    plain_file = _save(tmp_path / "plain.safetensors", {"linear.weight": torch.zeros(8, 16, dtype=torch.bfloat16)})

    assert detect_comfy_quant_formats([nvfp4_file]) == {FORMAT_NVFP4, FORMAT_INT8_TENSORWISE}
    assert detect_comfy_quant_formats([plain_file]) == set()


# endregion

# region quantization roundtrip


def test_swizzle_roundtrip_with_padding():
    for rows, cols in ((8, 2), (128, 4), (200, 10)):
        matrix = torch.randn(rows, cols)
        assert torch.equal(from_blocked(to_blocked(matrix), rows, cols), matrix)


def test_nvfp4_quantize_dequantize_roundtrip():
    torch.manual_seed(0)
    weight = torch.randn(32, 64)
    packed, block_scale, scale = _quantize_linear_weight(weight)

    assert packed.dtype is torch.uint8
    assert packed.shape == (32, 32)
    assert block_scale.dtype is torch.float8_e4m3fn

    restored = dequantize_nvfp4(packed, block_scale, scale, (32, 64), torch.float32)
    rel_err = (restored - weight).norm() / weight.norm()
    assert rel_err < 0.15  # NVFP4 quantization noise is ~10% relative


def test_nvfp4_nibble_order_is_high_first():
    # one block of 16 values; the first value must land in the high nibble of byte 0
    weight = torch.zeros(16, 16)
    weight[0, 0] = 6.0  # E2M1 max, exactly representable
    packed, _block_scale, _scale = _quantize_linear_weight(weight)
    assert packed[0, 0] >> 4 == 0b0111  # +6.0 code
    assert packed[0, 0] & 0x0F == 0


# endregion

# region streaming loader


def test_conversion_produces_musubi_layout(tmp_path):
    weight = torch.randn(16, 32)
    embed_weight = torch.randn(8, 32)
    path = _save(
        tmp_path / "artifact.safetensors",
        {
            **_nvfp4_module("proj", weight, pre_quant_scale=torch.rand(32, dtype=torch.bfloat16) + 0.5),
            **_int8_embedding_module("embed", embed_weight),
            "norm.weight": torch.ones(32, dtype=torch.bfloat16),
        },
    )

    state_dict, quantizer = _load(path)

    assert state_dict["proj.weight"].dtype is torch.uint8
    assert state_dict["proj.nvfp4_block_scale"].dtype is torch.float8_e4m3fn
    assert state_dict["proj.nvfp4_scale"].dtype is torch.float32
    assert state_dict["proj.pre_quant_scale"].shape == (32,)
    assert "proj.weight_scale" not in state_dict
    assert "proj.weight_scale_2" not in state_dict
    assert "proj.comfy_quant" not in state_dict
    assert state_dict["embed.weight"].dtype is torch.int8
    assert state_dict["embed.scale_weight"].dtype is torch.float32
    assert state_dict["norm.weight"].dtype is torch.bfloat16
    assert quantizer.nvfp4_module_shapes == {"proj": (16, 32)}
    assert quantizer.int8_embedding_modules == ["embed"]


def test_conversion_rejects_lora_merge_hook(tmp_path):
    path = _save(tmp_path / "artifact.safetensors", _nvfp4_module("proj", torch.randn(16, 32)))

    with pytest.raises(ValueError, match="Cannot merge LoRA"):
        _load(path, weight_hook=lambda key, value, keep_on_calc_device=False: value)


def test_conversion_rejects_convrot_spec(tmp_path):
    tensors = _nvfp4_module("proj", torch.randn(16, 32))
    tensors["proj.comfy_quant"] = _payload(format="int8_tensorwise", convrot=True, convrot_groupsize=256)
    path = _save(tmp_path / "convrot.safetensors", tensors)

    with pytest.raises(ValueError, match="convrot_int8"):
        _load(path)


def test_conversion_rejects_orphan_quantized_weight(tmp_path):
    path = _save(tmp_path / "orphan.safetensors", {"orphan.weight": torch.zeros(8, 16, dtype=torch.uint8)})

    with pytest.raises(ValueError, match="comfy_quant"):
        _load(path)


def test_conversion_rejects_scale_without_spec(tmp_path):
    path = _save(tmp_path / "orphan-scale.safetensors", {"orphan.weight_scale": torch.ones(8, 1)})

    with pytest.raises(ValueError, match="without a matching"):
        _load(path)


@pytest.mark.parametrize("missing_suffix", [".weight", ".weight_scale", ".weight_scale_2"])
def test_conversion_rejects_missing_nvfp4_siblings(tmp_path, missing_suffix):
    tensors = _nvfp4_module("proj", torch.randn(16, 32))
    del tensors["proj" + missing_suffix]
    path = _save(tmp_path / "missing.safetensors", tensors)

    with pytest.raises(ValueError, match="proj is missing tensors"):
        _load(path)


@pytest.mark.parametrize(
    ("key", "replacement", "match"),
    [
        ("proj.weight", torch.zeros(16, 16, dtype=torch.int8), "must be 2D uint8"),
        ("proj.weight_scale", torch.ones(128, 4, dtype=torch.float16), "must be F8_E4M3"),
        ("proj.weight_scale_2", torch.ones(1, dtype=torch.float32), "F32 scalar"),
    ],
)
def test_conversion_rejects_dtype_mismatches(tmp_path, key, replacement, match):
    tensors = _nvfp4_module("proj", torch.randn(16, 32))
    tensors[key] = replacement
    path = _save(tmp_path / "bad-dtype.safetensors", tensors)

    with pytest.raises(ValueError, match=match):
        _load(path)


def test_conversion_rejects_block_scale_size_mismatch(tmp_path):
    tensors = _nvfp4_module("proj", torch.randn(16, 32))
    tensors["proj.weight_scale"] = torch.zeros(128, 8, dtype=torch.float8_e4m3fn)
    path = _save(tmp_path / "bad-scale-size.safetensors", tensors)

    with pytest.raises(ValueError, match="block scale"):
        _load(path)


def test_conversion_rejects_pre_quant_scale_length_mismatch(tmp_path):
    tensors = _nvfp4_module("proj", torch.randn(16, 32), pre_quant_scale=torch.ones(16, dtype=torch.bfloat16))
    path = _save(tmp_path / "bad-pqs.safetensors", tensors)

    with pytest.raises(ValueError, match="pre_quant_scale"):
        _load(path)


def test_conversion_rejects_int8_embedding_scale_shape(tmp_path):
    tensors = _int8_embedding_module("embed", torch.randn(8, 32))
    tensors["embed.weight_scale"] = torch.ones(8, dtype=torch.float32)
    path = _save(tmp_path / "bad-embed-scale.safetensors", tensors)

    with pytest.raises(ValueError, match="scale shape"):
        _load(path)


# endregion

# region monkey patch and forward


class _TinyTextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 32)
        self.proj = nn.Linear(32, 16, bias=False)
        self.head = nn.Linear(16, 4, bias=True)  # stays BF16

    def forward(self, ids):
        return self.head(self.proj(self.embed(ids)))


def _build_patched_model(tmp_path, *, use_scaled_mm=False, pre_quant_scale=None):
    torch.manual_seed(0)
    proj_weight = torch.randn(16, 32)
    embed_weight = torch.randn(8, 32)
    tensors = {
        **_nvfp4_module("proj", proj_weight, pre_quant_scale=pre_quant_scale),
        **_int8_embedding_module("embed", embed_weight),
        "head.weight": torch.randn(4, 16, dtype=torch.bfloat16),
        "head.bias": torch.zeros(4, dtype=torch.bfloat16),
    }
    path = _save(tmp_path / "artifact.safetensors", tensors)
    state_dict, quantizer = _load(path)

    model = _TinyTextModel()
    apply_nvfp4_monkey_patch(
        model,
        state_dict,
        quantizer.nvfp4_module_shapes,
        quantizer.int8_embedding_modules,
        use_scaled_mm=use_scaled_mm,
    )
    model.requires_grad_(False)
    for key in state_dict:
        if state_dict[key].is_floating_point() and not key.endswith((".scale_weight", ".nvfp4_scale", ".nvfp4_block_scale")):
            state_dict[key] = state_dict[key].to(torch.bfloat16)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model, proj_weight, embed_weight


def test_patch_installs_quantized_tensors_and_strict_assign_load(tmp_path):
    model, _proj_weight, _embed_weight = _build_patched_model(tmp_path)

    assert type(model.proj) is nn.Linear
    assert model.proj.weight.dtype is torch.uint8
    assert not model.proj.weight.requires_grad
    assert model.proj.nvfp4_block_scale.dtype is torch.float8_e4m3fn
    assert model.proj.nvfp4_scale.dtype is torch.float32
    assert model.proj._nvfp4_orig_shape == (16, 32)
    assert type(model.embed) is nn.Embedding
    assert model.embed.weight.dtype is torch.int8
    assert model.embed.scale_weight.shape == (8, 1)
    assert model.is_nvfp4 is True
    assert model.nvfp4_layer_count == 1


def test_patched_forward_matches_dequantized_reference(tmp_path):
    model, proj_weight, embed_weight = _build_patched_model(tmp_path)
    ids = torch.tensor([[0, 3, 7, 1]])

    output = model(ids)

    expected_embed = (model.embed.weight[ids].float() * model.embed.scale_weight[ids]).to(torch.bfloat16)
    dequantized = dequantize_nvfp4(
        model.proj.weight, model.proj.nvfp4_block_scale, model.proj.nvfp4_scale, (16, 32), torch.bfloat16
    )
    expected = model.head(F.linear(expected_embed, dequantized))
    assert torch.equal(output, expected)

    # and the whole pipeline stays close to the unquantized reference
    reference = F.linear(F.embedding(ids, embed_weight.to(torch.bfloat16)), proj_weight.to(torch.bfloat16))
    approx = F.linear(expected_embed, dequantized)
    rel_err = (approx.float() - reference.float()).norm() / reference.float().norm()
    assert rel_err < 0.2


def test_patched_forward_applies_pre_quant_scale(tmp_path):
    pre_quant_scale = torch.rand(32, dtype=torch.bfloat16) + 0.5
    model, _proj_weight, _embed_weight = _build_patched_model(tmp_path, pre_quant_scale=pre_quant_scale)
    x = torch.randn(2, 32, dtype=torch.bfloat16)

    output = model.proj(x)

    dequantized = dequantize_nvfp4(
        model.proj.weight, model.proj.nvfp4_block_scale, model.proj.nvfp4_scale, (16, 32), torch.bfloat16
    )
    expected = F.linear(x * model.proj.pre_quant_scale, dequantized)
    assert torch.equal(output, expected)


def test_patch_rejects_nvfp4_on_non_linear(tmp_path):
    path = _save(tmp_path / "artifact.safetensors", _nvfp4_module("embed", torch.randn(16, 32)))
    state_dict, quantizer = _load(path)
    model = _TinyTextModel()

    with pytest.raises(ValueError, match="not an nn.Linear"):
        apply_nvfp4_monkey_patch(model, state_dict, quantizer.nvfp4_module_shapes, [])


def test_patch_rejects_int8_on_linear(tmp_path):
    path = _save(tmp_path / "artifact.safetensors", _int8_embedding_module("proj", torch.randn(16, 32)))
    state_dict, quantizer = _load(path)
    model = _TinyTextModel()

    with pytest.raises(ValueError, match="only\\s+supported for embeddings"):
        apply_nvfp4_monkey_patch(model, state_dict, {}, quantizer.int8_embedding_modules)


def test_patch_scaled_mm_requires_torch_support(tmp_path, monkeypatch):
    path = _save(tmp_path / "artifact.safetensors", _nvfp4_module("proj", torch.randn(16, 32)))
    state_dict, quantizer = _load(path)
    model = _TinyTextModel()
    monkeypatch.setattr(nvfp4_utils, "nvfp4_scaled_mm_available", lambda: False)

    with pytest.raises(ValueError, match="PyTorch 2.10"):
        apply_nvfp4_monkey_patch(model, state_dict, quantizer.nvfp4_module_shapes, [], use_scaled_mm=True)


# endregion
