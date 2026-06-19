"""Tests for the Ideogram 4 pre-quantized (ComfyUI) FP8 loading path.

These exercise the shared monkey-patch FP8 loader (with the `allow_prequantized_fp8`
opt-in) plus the Ideogram4-local split hook that normalizes the Comfy scale layout.
Everything runs on CPU using torch.float8_e4m3fn for storage only.
"""

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="float8_e4m3fn not available in this torch build")

from safetensors.torch import save_file

from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.ideogram4_quantized_loading import (
    COMFY_FP8_MARKER_SUFFIX,
    FP8_SCALE_SUFFIX,
)
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

# e4m3 max; quantize a 2-D Linear weight to float8 with per-row scales, mirroring the
# layout a ComfyUI FP8 exporter produces (weight_fp8.to(dtype) * scale[:, None] ≈ weight).
_FP8_E4M3_MAX = 448.0


def _quantize_weight_to_fp8(weight):
    w = weight.detach().to(torch.float32)
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / _FP8_E4M3_MAX
    q = (w / scale).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(1).to(torch.float32)


def _write_comfy_fp8_checkpoint(path, *, out_features=8, in_features=16):
    """Write a tiny ComfyUI-style pre-quantized FP8 checkpoint.

    Layout: a quantized "layers.0.attention.qkv" Linear (fp8 weight + per-row weight_scale +
    comfy_quant marker) and a non-quantized bf16 "input_proj" Linear.
    """
    torch.manual_seed(0)
    ref_weight = torch.randn(out_features, in_features, dtype=torch.float32)
    weight_fp8, scale = _quantize_weight_to_fp8(ref_weight)  # scale: [out]

    other_weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)

    sd = {
        "layers.0.attention.qkv.weight": weight_fp8,
        "layers.0.attention.qkv" + FP8_SCALE_SUFFIX: scale,
        "layers.0.attention.qkv" + COMFY_FP8_MARKER_SUFFIX: torch.tensor([1], dtype=torch.uint8),
        "input_proj.weight": other_weight,
    }
    save_file(sd, str(path))
    # dequantized reference for the quantized layer
    dequant_ref = weight_fp8.to(torch.float32) * scale.unsqueeze(1)
    return ref_weight, dequant_ref, scale, other_weight


def _load_prequant_fp8(path, compute_dtype=torch.bfloat16):
    hooks = safetensors_utils.WeightTransformHooks(split_hook=ideogram4_utils._make_ideogram4_comfy_fp8_split_hook(compute_dtype))
    return load_safetensors_with_lora_and_fp8(
        model_files=str(path),
        lora_weights_list=None,
        lora_multipliers=None,
        fp8_optimization=True,
        calc_device=torch.device("cpu"),
        move_to_device=True,
        dit_weight_dtype=None,
        target_keys=ideogram4_utils.IDEOGRAM4_FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=ideogram4_utils.IDEOGRAM4_FP8_OPTIMIZATION_EXCLUDE_KEYS,
        weight_transform_hooks=hooks,
        allow_prequantized_fp8=True,
    )


def test_split_hook_normalizes_comfy_layout(tmp_path):
    path = tmp_path / "dit.safetensors"
    _, _, scale, _ = _write_comfy_fp8_checkpoint(path)

    sd = _load_prequant_fp8(path)

    # fp8 weight kept as-is (not re-quantized, still 1-byte dtype)
    assert sd["layers.0.attention.qkv.weight"].dtype == torch.float8_e4m3fn
    # weight_scale renamed to scale_weight, reshaped [out] -> [out, 1], cast to compute dtype
    assert "layers.0.attention.qkv" + FP8_SCALE_SUFFIX not in sd
    new_scale = sd["layers.0.attention.qkv.scale_weight"]
    assert new_scale.shape == (scale.shape[0], 1)
    assert new_scale.dtype == torch.bfloat16
    # comfy marker dropped
    assert "layers.0.attention.qkv" + COMFY_FP8_MARKER_SUFFIX not in sd
    # non-target bf16 weight passed through untouched (no scale generated)
    assert sd["input_proj.weight"].dtype == torch.bfloat16
    assert "input_proj.scale_weight" not in sd


def test_monkey_patch_forward_matches_dequant(tmp_path):
    path = tmp_path / "dit.safetensors"
    _, dequant_ref, _, other_weight = _write_comfy_fp8_checkpoint(path)
    out_features, in_features = dequant_ref.shape

    sd = _load_prequant_fp8(path)

    # Minimal model mirroring the checkpoint module paths.
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Module()
            self.attention.qkv = nn.Linear(in_features, out_features, bias=False)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block()])
            self.input_proj = nn.Linear(in_features, out_features, bias=False)

    model = Model().to(torch.bfloat16)
    apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval()

    x = torch.randn(2, in_features, dtype=torch.bfloat16)
    with torch.no_grad():
        got = model.layers[0].attention.qkv(x)
    # Reference: dequantized fp8 weight (bf16 compute) matmul.
    expected = torch.nn.functional.linear(x, dequant_ref.to(torch.bfloat16))
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


def test_prequantized_fp8_rejected_without_optin(tmp_path):
    """Without allow_prequantized_fp8, an already-FP8 target weight still raises (shared default)."""
    from musubi_tuner.modules.fp8_optimization_utils import load_safetensors_with_fp8_optimization

    path = tmp_path / "dit.safetensors"
    _write_comfy_fp8_checkpoint(path)
    hooks = safetensors_utils.WeightTransformHooks(split_hook=ideogram4_utils._make_ideogram4_comfy_fp8_split_hook(torch.bfloat16))
    with pytest.raises(ValueError, match="already in"):
        load_safetensors_with_fp8_optimization(
            [str(path)],
            calc_device=torch.device("cpu"),
            target_layer_keys=ideogram4_utils.IDEOGRAM4_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_layer_keys=ideogram4_utils.IDEOGRAM4_FP8_OPTIMIZATION_EXCLUDE_KEYS,
            move_to_device=True,
            weight_transform_hooks=hooks,
            allow_prequantized_fp8=False,
        )
