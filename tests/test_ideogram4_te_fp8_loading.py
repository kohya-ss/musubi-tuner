"""Probe for the Ideogram 4 text encoder pre-quantized FP8 loading path.

The Qwen3-VL text encoder loads pre-quantized FP8 weights through Musubi's shared
monkey-patch FP8 path (``_prepare_qwen3_vl_fp8_state_dict`` + ``apply_fp8_monkey_patch``),
the same mechanism the DiT uses.

This test loads a synthetic FP8 checkpoint through that path and asserts the forward output
matches a manual dequantization reference, for both scale layouts seen in the wild:

- per-channel ``weight_scale`` shape ``[out]``  (the official weights)
- per-tensor  ``weight_scale`` shape ``[]``     (the ComfyUI weights)

It also checks the ``.comfy_quant`` marker key is dropped and the scales are renamed and
reshaped. CPU only, no real model.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.ideogram4.ideogram4_quantized_loading import (
    COMFY_FP8_MARKER_SUFFIX,
    require_fp8_dtype,
)
from musubi_tuner.ideogram4.ideogram4_utils import _prepare_qwen3_vl_fp8_state_dict
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch

try:
    FP8_DTYPE = require_fp8_dtype()
    FP8_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the torch build
    FP8_AVAILABLE = False

pytestmark = pytest.mark.skipif(not FP8_AVAILABLE, reason="float8_e4m3fn not available in this torch build")

COMPUTE_DTYPE = torch.bfloat16


class _TinyQwenLike(nn.Module):
    """Two FP8-target Linears (one per-channel, one per-tensor) plus a non-FP8 norm."""

    def __init__(self):
        super().__init__()
        self.a = nn.Linear(8, 8, bias=True)  # quantized with a per-channel scale
        self.b = nn.Linear(8, 8, bias=False)  # quantized with a per-tensor scale
        self.norm = nn.LayerNorm(8)  # stays in compute dtype

    def forward(self, x):
        return self.b(self.a(self.norm(x)))


def _quantize(weight: torch.Tensor, *, per_channel: bool):
    fp8_max = torch.finfo(FP8_DTYPE).max
    w = weight.float()
    if per_channel:
        amax = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)  # [out, 1]
        scale = amax / fp8_max
        q = (w / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
        return q, scale.squeeze(1)  # scale shape [out]
    amax = w.abs().amax().clamp_min(1e-12)
    scale = amax / fp8_max
    q = (w / scale).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
    return q, scale.reshape(())  # scale shape []


def _build_fp8_checkpoint():
    torch.manual_seed(0)
    src = _TinyQwenLike()
    qa, sa = _quantize(src.a.weight.data, per_channel=True)
    qb, sb = _quantize(src.b.weight.data, per_channel=False)
    state_dict = {
        "a.weight": qa,
        "a.weight_scale": sa,  # [out]
        "a.bias": src.a.bias.data.to(COMPUTE_DTYPE),
        "a.comfy_quant": torch.zeros(64, dtype=torch.uint8),  # marker, must be dropped
        "b.weight": qb,
        "b.weight_scale": sb,  # []
        "norm.weight": src.norm.weight.data.to(COMPUTE_DTYPE),
        "norm.bias": src.norm.bias.data.to(COMPUTE_DTYPE),
    }
    return state_dict


def _load_via_monkey_patch(state_dict):
    model = _TinyQwenLike()
    prepared = _prepare_qwen3_vl_fp8_state_dict(state_dict, device=torch.device("cpu"), dtype=COMPUTE_DTYPE)
    apply_fp8_monkey_patch(model, prepared, use_scaled_mm=False)
    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=True)
    assert not unexpected, unexpected
    assert not missing, missing
    model.eval()
    return model, prepared


def test_prepare_drops_comfy_marker_and_renames_scales():
    state_dict = _build_fp8_checkpoint()
    prepared = _prepare_qwen3_vl_fp8_state_dict(state_dict, device=torch.device("cpu"), dtype=COMPUTE_DTYPE)

    assert not any(k.endswith(COMFY_FP8_MARKER_SUFFIX) for k in prepared)  # marker dropped
    assert "a.weight_scale" not in prepared and "b.weight_scale" not in prepared  # renamed
    assert prepared["a.scale_weight"].shape == (8, 1)  # per-channel [out] -> [out, 1]
    assert prepared["b.scale_weight"].shape == (1,)  # per-tensor [] -> [1]
    assert prepared["a.scale_weight"].dtype == COMPUTE_DTYPE  # cast for in-compute-dtype dequant
    assert prepared["a.weight"].dtype == FP8_DTYPE  # weight kept FP8
    assert prepared["norm.weight"].dtype == COMPUTE_DTYPE  # non-FP8 float cast to compute dtype


def test_monkey_patch_matches_manual_dequant_reference():
    state_dict = _build_fp8_checkpoint()
    new_model, prepared = _load_via_monkey_patch(state_dict)

    x = torch.randn(2, 8, dtype=COMPUTE_DTYPE)
    # Manual reference: dequantize each weight in the compute dtype, then the same forward.
    wa = prepared["a.weight"].to(COMPUTE_DTYPE) * prepared["a.scale_weight"]
    wb = prepared["b.weight"].to(COMPUTE_DTYPE) * prepared["b.scale_weight"]
    with torch.no_grad():
        h = F.layer_norm(x, (8,), prepared["norm.weight"], prepared["norm.bias"])
        h = F.linear(h, wa, prepared["a.bias"])
        ref = F.linear(h, wb)
        y_new = new_model(x)
    assert torch.equal(y_new, ref)
