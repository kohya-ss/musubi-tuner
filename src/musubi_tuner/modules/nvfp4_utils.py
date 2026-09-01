# Low-level NVFP4 routines (E2M1 encode/decode, cuBLAS swizzle, scaled_mm call) are
# adapted from the exp-nvfp4-support-for-torch-2-10 branch, which was inspired by
# comfy-kitchen (https://github.com/Comfy-Org/comfy-kitchen, Apache License 2.0).

"""Loading ComfyUI pre-quantized NVFP4 (+AWQ) checkpoints for frozen text encoders.

Layout of a ComfyUI NVFP4 artifact (verified bit-level against the BF16 reference of
the MiniMax-H3 Qwen3-VL text encoder):

- NVFP4 Linear: ``.weight`` U8 [N, K/2] with two E2M1 codes per byte (element 0 in the
  HIGH nibble), ``.weight_scale`` F8_E4M3 per-16-block scales stored in the cuBLAS
  128x4 tiled ("swizzled") layout, ``.weight_scale_2`` F32 per-tensor scale, and a
  ``.comfy_quant`` spec ``{"format": "nvfp4", ...}``. Modules quantized with AWQ carry
  a ``.pre_quant_scale`` [K] tensor that is multiplied into the *input* at runtime;
  for the remaining modules the AWQ scale is folded into the preceding norm weights,
  so the checkpoint is self-consistent and needs no special handling here.
- INT8 embedding: ``.weight`` I8 [V, D] + ``.weight_scale`` F32 [V, 1]
  (``{"format": "int8_tensorwise"}``, effectively per-row); dequant = weight * scale.
- Everything else (norms, vision tower, biases) stays BF16.

The state dict is converted to the Musubi runtime layout: ``.weight_scale`` becomes
``.nvfp4_block_scale`` (kept swizzled: ``torch.nn.functional.scaled_mm`` consumes the
swizzled layout directly, the dequantizing fallback unswizzles on the fly) and
``.weight_scale_2`` becomes ``.nvfp4_scale``; the embedding scale becomes
``.scale_weight`` (same dequant semantics as the ConvRot INT8 layout).

Inference only: NVFP4 modules are frozen, the patched forwards have no autograd
support. Dynamic (on-the-fly) NVFP4 quantization is deliberately not offered — the
published artifacts are AWQ-calibrated, which cannot be reproduced without
calibration data, and dynamically quantizing BF16 weights would silently produce a
lower-quality model than ConvRot INT8.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.modules.comfy_quant_utils import (
    COMFY_QUANT_SUFFIX,
    COMFY_WEIGHT_SCALE_SUFFIX,
    FORMAT_INT8_TENSORWISE,
    FORMAT_NVFP4,
    classify_comfy_quant_spec,
    decode_comfy_quant_spec,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks

logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16

F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0

COMFY_WEIGHT_SCALE_2_SUFFIX = ".weight_scale_2"
COMFY_PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"

# E2M1 code -> value (codes 0..7 positive, 8..15 negative)
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# byte -> (high nibble value, low nibble value), cached per (device, dtype)
_BYTE_PAIR_LUT_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _byte_pair_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lut = _BYTE_PAIR_LUT_CACHE.get((device, dtype))
    if lut is None:
        values = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
        codes = torch.arange(256, dtype=torch.int64)
        lut = torch.stack([values[codes >> 4], values[codes & 0x0F]], dim=1).to(device=device, dtype=dtype)
        _BYTE_PAIR_LUT_CACHE[(device, dtype)] = lut
    return lut


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _roundup(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


# region cuBLAS 128x4 tiled block-scale layout
# https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Rearrange (H, W) block scales into the cuBLAS tiled layout (padded to 128x4)."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Reverse the cuBLAS tiled layout back to a row-major (num_rows, num_cols) matrix."""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


# endregion


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    orig_shape: Tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed NVFP4 data (block_scale in the swizzled layout) to ``orig_shape``."""
    stored_rows = packed.shape[0]
    stored_cols = packed.shape[1] * 2
    lut = _byte_pair_lut(packed.device, out_dtype)
    # one embedding lookup decodes both nibbles of each byte: [R, C/2] -> [R, C/2, 2] -> [R, C]
    weight = F.embedding(packed.reshape(-1).int(), lut).reshape(stored_rows, stored_cols)
    scales = from_blocked(block_scale, stored_rows, stored_cols // NVFP4_BLOCK_SIZE)
    total = (per_tensor_scale.float() * scales.float()).to(out_dtype)
    weight = (weight.reshape(stored_rows, -1, NVFP4_BLOCK_SIZE) * total.unsqueeze(-1)).reshape(stored_rows, stored_cols)
    return weight[: orig_shape[0], : orig_shape[1]]


# region runtime activation quantization for the scaled_mm (W4A4) path


def _n_ones(n: int) -> int:
    return (1 << n) - 1


_EBITS_F32, _MBITS_F32 = 8, 23
_F32_EXP_BIAS = _n_ones(_EBITS_F32 - 1)


def _f32_to_e2m1_unpacked(x: torch.Tensor) -> torch.Tensor:
    """Convert FP32 to E2M1 codes stored one-per-byte in uint8 (round-to-nearest-even)."""
    ebits, mbits = 2, 1
    assert x.dtype == torch.float
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(_MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32, device=x.device).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = (x ^ sign).view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = (x + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    normal_x = normal_x + (((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder) + mant_odd
    normal_x = (normal_x >> (_MBITS_F32 - mbits)).to(torch.uint8)

    result = torch.full_like(x, max_int, dtype=torch.uint8)
    result = torch.where(denormal_mask, denormal_x, result)
    result = torch.where(normal_mask, normal_x, result)

    sign_lp = (sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)).to(torch.uint8) & sign_mask
    return result | sign_lp


_E2M1_MAGNITUDE_TABLE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e2m1_stochastic_magnitude_code(x_pos: torch.Tensor) -> torch.Tensor:
    """x_pos: non-negative fp32 tensor, values in [0, F4_E2M1_MAX]. Returns uint8 magnitude
    code 0..7, stochastically rounded to one of the two nearest representable E2M1 magnitudes
    with probability proportional to inverse distance (unbiased in expectation:
    E[decoded] == x_pos). Degenerate cases (x_pos exactly representable, including the 0 and
    6.0 endpoints) fall out naturally as p_up == 0 -- no special-casing needed.
    """
    table = torch.tensor(_E2M1_MAGNITUDE_TABLE, dtype=torch.float32, device=x_pos.device)
    n = table.numel()
    raw_hi_idx = torch.searchsorted(table, x_pos, right=True)
    hi_idx = torch.clamp(raw_hi_idx, max=n - 1)
    lo_idx = torch.clamp(raw_hi_idx - 1, min=0, max=n - 1)

    lo = table[lo_idx]
    hi = table[hi_idx]
    span = hi - lo
    span_safe = torch.where(span == 0, torch.ones_like(span), span)
    p_up = torch.where(span == 0, torch.zeros_like(span), (x_pos - lo) / span_safe)

    r = torch.rand_like(x_pos)
    round_up = r < p_up
    return torch.where(round_up, hi_idx, lo_idx).to(torch.uint8)


def _e2m1_stochastic_code(x: torch.Tensor) -> torch.Tensor:
    """x: signed fp32 tensor, values already clamped to [-F4_E2M1_MAX, F4_E2M1_MAX] by the
    caller (see _quantize_nvfp4_2d_prepare). Returns uint8 E2M1 code 0..15, stochastically
    rounded, sign in bit 3 (same encoding as _f32_to_e2m1_unpacked: codes 0-7 positive, 8-15
    negative, negative code = positive code | 8).
    """
    neg = x < 0
    mag_code = _e2m1_stochastic_magnitude_code(x.abs())
    return torch.where(neg, mag_code | 8, mag_code)


def pack_uint4(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit codes (one per byte) into bytes, element 0 in the HIGH nibble."""
    shape = codes.shape
    assert shape[-1] % 2 == 0
    codes = codes.contiguous().view(-1)
    return (codes[::2] << 4 | codes[1::2]).view(*shape[:-1], shape[-1] // 2)


def _quantize_nvfp4_2d_prepare(x: torch.Tensor, per_tensor_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Shared block-scale computation and normalize/clamp step for NVFP4 quantization, used by
    both the deterministic (_quantize_nvfp4_2d) and stochastic-rounding
    (quantize_nvfp4_activation_stochastic) E2M1 conversion paths -- everything through this
    point is identical between the two; only the final magnitude-conversion step differs.

    Returns (data, scaled_f8, per_tensor_scale, orig_rows): ``data`` is the row-padded,
    normalized, [-F4_E2M1_MAX, F4_E2M1_MAX]-clamped fp32 tensor [Mp, K] ready for E2M1
    conversion; ``scaled_f8`` is the not-yet-swizzled per-block F8_E4M3 scale [Mp, K/16].

    ``per_tensor_scale``, when given, is used instead of computing ``amax(x)`` internally --
    lets a caller quantize row-chunks of a larger tensor against one shared, tensor-wide scale
    (see ``_quantize_nvfp4_2d_chunked``), which keeps the chunked result numerically identical
    to calling this function once on the whole tensor.
    """
    orig_rows, cols = x.shape
    if cols % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"NVFP4 quantization width must be a multiple of {NVFP4_BLOCK_SIZE}, got {cols}")
    padded_rows = _roundup(orig_rows, 16)
    if padded_rows != orig_rows:
        x = F.pad(x, (0, 0, 0, padded_rows - orig_rows))

    if per_tensor_scale is None:
        per_tensor_scale = (torch.amax(x.abs()).float() / (F8_E4M3_MAX * F4_E2M1_MAX)).reshape(())

    blocks = x.reshape(padded_rows, -1, NVFP4_BLOCK_SIZE)
    block_scale = torch.amax(blocks.abs(), dim=-1).float() / F4_E2M1_MAX
    scaled = torch.clamp(block_scale / torch.clamp(per_tensor_scale, min=torch.finfo(torch.float32).tiny), max=F8_E4M3_MAX)
    scaled_f8 = scaled.to(torch.float8_e4m3fn)
    total = per_tensor_scale * scaled_f8.float()
    total_safe = torch.where(total == 0, torch.ones_like(total), total)

    data = blocks.float() / total_safe.unsqueeze(-1)
    data = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(data), data)
    data = torch.clamp(data, -F4_E2M1_MAX, F4_E2M1_MAX).reshape(padded_rows, cols)
    return data, scaled_f8, per_tensor_scale, orig_rows


def _quantize_nvfp4_2d(x: torch.Tensor, per_tensor_scale: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2D tensor to NVFP4, grouping blocks along the last axis.

    Returns (packed uint8 [Mp, K/2], swizzled block scales F8_E4M3, per-tensor scale F32,
    original row count). Rows are padded to a multiple of 16 as scaled_mm requires. Shared by
    ``quantize_nvfp4_activation`` (grouping activations along their feature axis) and
    ``quantize_nvfp4_weight_columnwise`` (re-grouping a frozen weight along its out_features
    axis for the backward GEMM).
    """
    data, scaled_f8, per_tensor_scale, orig_rows = _quantize_nvfp4_2d_prepare(x, per_tensor_scale)
    packed = pack_uint4(_f32_to_e2m1_unpacked(data))
    return packed, to_blocked(scaled_f8), per_tensor_scale, orig_rows


def _quantize_nvfp4_2d_chunked(x: torch.Tensor, chunk_rows: int = 1024) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Row-chunked ``_quantize_nvfp4_2d`` for large 2D tensors.

    ``_f32_to_e2m1_unpacked`` allocates roughly ten full-size fp32/int32/bool temporaries, so
    quantizing a large weight (e.g. Krea2's 24576x6144 mlp gate/up) in one call transiently
    peaks around 5GB for a 72MB packed result. Chunking the pack step over row groups bounds
    the transient peak to a few hundred MB regardless of input size.

    Correctness requires two things this function gets right where a naive "call
    _quantize_nvfp4_2d per chunk independently" would not:
      - the per-tensor scale must be shared across all chunks (computed once, up front, as a
        single cheap global reduction) -- otherwise each chunk would get its own local
        amax-based scale and the result would not match a single unchunked call;
      - ``chunk_rows`` must be a multiple of 128 (the cuBLAS block-scale tile height used by
        ``to_blocked``) so no chunk boundary falls inside a scale tile except possibly at the
        very last chunk, which is padded identically to how the unchunked path pads the
        tensor's own tail.

    Returns the same 4-tuple as ``_quantize_nvfp4_2d`` and is numerically bit-identical to it.
    """
    if chunk_rows <= 0 or chunk_rows % 128 != 0:
        raise ValueError(f"chunk_rows must be a positive multiple of 128 (cuBLAS block-scale tile height), got {chunk_rows}")
    orig_rows, _cols = x.shape
    per_tensor_scale = (torch.amax(x.abs()).float() / (F8_E4M3_MAX * F4_E2M1_MAX)).reshape(())

    packed_chunks = []
    scale_chunks = []
    for start in range(0, orig_rows, chunk_rows):
        chunk = x[start : start + chunk_rows]
        packed, block_scale, _chunk_scale, _chunk_orig_rows = _quantize_nvfp4_2d(chunk, per_tensor_scale=per_tensor_scale)
        packed_chunks.append(packed)
        scale_chunks.append(block_scale)

    return torch.cat(packed_chunks, dim=0), torch.cat(scale_chunks, dim=0), per_tensor_scale, orig_rows


def quantize_nvfp4_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2D activation to NVFP4 for scaled_mm. See ``_quantize_nvfp4_2d``."""
    return _quantize_nvfp4_2d(x)


def quantize_nvfp4_activation_stochastic(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2D tensor to NVFP4 using stochastic rounding for the E2M1 magnitude
    conversion. Block-scale computation and the F8_E4M3 scale cast are identical to, and stay
    deterministic like, quantize_nvfp4_activation -- only the final magnitude conversion
    differs (see _e2m1_stochastic_code).

    Used exclusively by NvFp4LinearFn.backward for grad_out, per arXiv:2509.25149's finding
    that stochastic rounding is needed for gradient tensors specifically (to avoid the
    directional bias deterministic rounding introduces) but is detrimental for forward-pass
    tensors and unnecessary for weights -- do not use this for forward activations
    (quantize_nvfp4_activation) or weight quantization (quantize_nvfp4_weight_columnwise).
    """
    data, scaled_f8, per_tensor_scale, orig_rows = _quantize_nvfp4_2d_prepare(x)
    packed = pack_uint4(_e2m1_stochastic_code(data))
    return packed, to_blocked(scaled_f8), per_tensor_scale, orig_rows


def quantize_nvfp4_weight_columnwise(
    weight_packed: torch.Tensor,
    block_scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    orig_shape: Tuple[int, int],
    chunk_rows: int = 1024,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-quantize a K-grouped (row-wise) NVFP4 weight along its N axis (out_features).

    NVFP4 block scales are computed along the forward contraction axis (K); the backward
    GEMM (``grad_x = grad_out @ W``) contracts over N instead, so the existing block scale
    cannot be reused via a plain transpose the way ConvRot INT8's single-byte-per-element
    weight can (``wq.t().contiguous()``) -- a transposed *view* of nibble-packed FP4 data
    would pair up the wrong elements. This produces a second, independently computed NVFP4
    quantization grouped along N (mirrors Transformer Engine's rowwise/columnwise tensor
    pattern). Call once per frozen weight at load time; the result never changes afterward.

    Quantizes in ``chunk_rows``-sized row chunks of the transposed weight (see
    ``_quantize_nvfp4_2d_chunked``) to bound the transient GPU memory peak instead of
    materializing all of ``_f32_to_e2m1_unpacked``'s temporaries for the full weight at once.
    """
    n, k = orig_shape
    if n % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"NVFP4 columnwise requant needs out_features {n} to be a multiple of {NVFP4_BLOCK_SIZE}, got {n}"
        )
    weight_bf16 = dequantize_nvfp4(weight_packed, block_scale, per_tensor_scale, orig_shape, torch.bfloat16)
    weight_t = weight_bf16.t().contiguous()  # [K, N]
    packed_t, block_scale_t, tensor_scale_t, _ = _quantize_nvfp4_2d_chunked(weight_t, chunk_rows=chunk_rows)
    return packed_t, block_scale_t, tensor_scale_t


def nvfp4_scaled_mm_available() -> bool:
    return hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch.nn.functional, "scaled_mm")


def nvfp4_scaled_mm_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_block_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    orig_out_features: int,
) -> torch.Tensor:
    """W4A4 linear via torch.nn.functional.scaled_mm (torch 2.10+, Blackwell)."""
    from torch.nn.functional import ScalingType, SwizzleType

    x_packed, x_block_scale, x_scale, orig_rows = quantize_nvfp4_activation(x)
    result = torch.nn.functional.scaled_mm(
        x_packed.view(torch.float4_e2m1fn_x2),
        weight_packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[x_block_scale.view(-1), x_scale],
        scale_b=[weight_block_scale.view(-1), weight_scale],
        bias=bias,
        output_dtype=x.dtype,
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    )
    return result[:orig_rows, :orig_out_features]


class NvFp4LinearFn(torch.autograd.Function):
    """True FP4x4 tensor-core Linear for a frozen, pre-quantized NVFP4 weight.

    Forward uses the row-wise (K-grouped) weight via ``nvfp4_scaled_mm_linear``. Backward
    computes only ``grad_x`` (the base is frozen, never ``grad_weight``) using the columnwise
    (N-grouped) weight copy -- see ``quantize_nvfp4_weight_columnwise`` for why a second
    quantization, not a transpose, is required. Structurally mirrors ``ConvRotInt8LinearFn``.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, weight_packed, block_scale, tensor_scale, weight_t_packed, block_scale_t, tensor_scale_t, bias, orig_out_features):
        if torch.is_autocast_enabled(x.device.type):
            cast_dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(cast_dtype)
            if bias is not None:
                bias = bias.to(cast_dtype)
        x_2d = x.reshape(-1, x.shape[-1])
        out = nvfp4_scaled_mm_linear(x_2d, weight_packed, block_scale, tensor_scale, bias, orig_out_features)
        ctx.save_for_backward(weight_t_packed, block_scale_t, tensor_scale_t)
        ctx.in_features = x.shape[-1]
        ctx.bias_needs_grad = bias is not None and bias.requires_grad
        return out.reshape(*x.shape[:-1], out.shape[-1])

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        weight_t_packed, block_scale_t, tensor_scale_t = ctx.saved_tensors
        g2d = grad_out.reshape(-1, grad_out.shape[-1])  # [M, N]

        grad_x = None
        if ctx.needs_input_grad[0]:
            # grad_x = grad_out @ W, computed as the "linear" from N -> K using the
            # columnwise-quantized weight (packed as [K, N/2], i.e. a virtual Linear with
            # in_features=N, out_features=K).
            gx = nvfp4_scaled_mm_linear(g2d, weight_t_packed, block_scale_t, tensor_scale_t, None, weight_t_packed.shape[0])
            grad_x = gx.reshape(*grad_out.shape[:-1], ctx.in_features)

        grad_bias = g2d.sum(dim=0) if ctx.bias_needs_grad else None
        return grad_x, None, None, None, None, None, None, grad_bias, None


def nvfp4_swap_tensor_selector(block: nn.Module) -> List[Tuple[nn.Module, str]]:
    """Block-swap tensor selector for blocks containing NVFP4-patched Linears.

    The offloader's default selector only tracks each Linear's ``weight``. An NVFP4-patched
    Linear also carries ``nvfp4_block_scale``/``nvfp4_scale`` (forward) and, under
    ``training=True``, ``nvfp4_weight_t``/``nvfp4_block_scale_t``/``nvfp4_scale_t`` (backward) --
    the columnwise copy is a second full-size weight matrix, so leaving it out of the selector
    does not just skip a small scale vector: the offloader's per-block ``.to(device)`` call still
    drags it onto the device once and, since it is never part of the ring/master swap machinery,
    it never comes back off -- every block ends up pinning its full columnwise copy resident,
    silently defeating block swap's memory savings. Pass this selector (instead of the default)
    whenever any Linear in the block list has been NVFP4-patched.
    """
    jobs = []
    for _, module in block.named_modules():
        if not (hasattr(module, "weight") and module.weight is not None and module.__class__.__name__.endswith("Linear")):
            continue
        jobs.append((module, "weight"))
        for name in ("nvfp4_block_scale", "nvfp4_scale", "nvfp4_weight_t", "nvfp4_block_scale_t", "nvfp4_scale_t"):
            if name in module._buffers:
                jobs.append((module, name))
    return jobs


def block_has_nvfp4_patched_linear(block: nn.Module) -> bool:
    """True if any Linear in ``block`` was patched by ``apply_nvfp4_monkey_patch``."""
    return any("nvfp4_block_scale" in module._buffers for module in block.modules())


# endregion


# region pre-quantized state dict loading


class NvFp4Quantizer:
    """Streams a ComfyUI pre-quantized NVFP4 (+ INT8 embedding) checkpoint.

    Same protocol as ``ConvRotInt8Quantizer``: passed as ``quantizer`` to
    ``load_safetensors_with_lora_and_fp8``. Loading only converts key names and
    validates the tensors — there is no dynamic quantization, the file dictates the
    quantized layers. ``nvfp4_module_shapes`` maps module paths to their original
    (out_features, in_features) after loading; ``int8_embedding_modules`` lists the
    per-row INT8 modules (embeddings). Pass both to ``apply_nvfp4_monkey_patch``.
    """

    def __init__(self):
        self.nvfp4_module_shapes: Dict[str, Tuple[int, int]] = {}
        self.int8_embedding_modules: List[str] = []

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
    ) -> dict:
        state_dict = {}
        module_formats: Dict[str, str] = {}  # spans all shards
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

                keys = f.keys()

                # pre-scan the tiny spec tensors so each module's format is known before
                # the (possibly earlier-iterated) weight/scale keys arrive
                for key in keys:
                    if key.endswith(COMFY_QUANT_SUFFIX):
                        module_path = key[: -len(COMFY_QUANT_SUFFIX)]
                        spec_format = classify_comfy_quant_spec(decode_comfy_quant_spec(key, f.get_tensor(key)))
                        if spec_format not in (FORMAT_NVFP4, FORMAT_INT8_TENSORWISE):
                            raise ValueError(
                                f"Unsupported comfy_quant format for {key}: {spec_format}. The NVFP4 loader supports"
                                ' "nvfp4" Linear layers and "int8_tensorwise" embeddings only.'
                                f" / {key} の comfy_quant 形式 {spec_format} はNVFP4ローダーではサポートされていません。"
                            )
                        module_formats[module_path] = spec_format
                if module_formats and weight_hook is not None:
                    raise ValueError(
                        f"Cannot merge LoRA weights into pre-quantized NVFP4 checkpoint {model_file}."
                        " Use the original BF16 weights instead."
                        f" / 事前量子化済みNVFP4チェックポイント {model_file} にはLoRAをマージできません。"
                        "BF16の元重みを使用してください。"
                    )

                for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                    if key.endswith(COMFY_QUANT_SUFFIX):
                        continue  # consumed in the pre-scan, not a model tensor

                    value = f.get_tensor(key)
                    original_device = value.device  # usually cpu
                    passthrough_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    converted_key = self._convert_key(key, value, module_formats)
                    state_dict[converted_key] = value.to(passthrough_device)

        self._validate_completeness(state_dict, module_formats)
        logger.info(
            f"Number of pre-quantized layers: {len(self.nvfp4_module_shapes)} NVFP4 Linear,"
            f" {len(self.int8_embedding_modules)} INT8 embedding"
        )
        return state_dict

    def _convert_key(self, key: str, value: torch.Tensor, module_formats: Dict[str, str]) -> str:
        """Validate a tensor against its module's declared format and return the Musubi key."""
        for suffix in (COMFY_WEIGHT_SCALE_SUFFIX, COMFY_WEIGHT_SCALE_2_SUFFIX, COMFY_PRE_QUANT_SCALE_SUFFIX, ".weight"):
            if key.endswith(suffix):
                module_path = key[: -len(suffix)]
                break
        else:
            return key  # bias, norm, etc.: passthrough
        spec_format = module_formats.get(module_path)
        if spec_format is None:
            if key.endswith(".weight") and value.dtype.itemsize == 1:
                raise ValueError(
                    f"Layer {key} is already in {value.dtype} format but has no {COMFY_QUANT_SUFFIX} spec."
                    f" / レイヤー {key} は既に{value.dtype}形式ですが {COMFY_QUANT_SUFFIX} がありません。"
                )
            if key.endswith((COMFY_WEIGHT_SCALE_SUFFIX, COMFY_WEIGHT_SCALE_2_SUFFIX, COMFY_PRE_QUANT_SCALE_SUFFIX)):
                raise ValueError(f"Found {key} without a matching {module_path}{COMFY_QUANT_SUFFIX} spec")
            return key

        if spec_format == FORMAT_NVFP4:
            if key.endswith(COMFY_WEIGHT_SCALE_SUFFIX):
                if value.dtype is not torch.float8_e4m3fn:
                    raise ValueError(f"NVFP4 block scale {key} must be F8_E4M3, got {value.dtype}")
                return module_path + ".nvfp4_block_scale"
            if key.endswith(COMFY_WEIGHT_SCALE_2_SUFFIX):
                if value.dtype is not torch.float32 or value.ndim != 0:
                    raise ValueError(f"NVFP4 per-tensor scale {key} must be a F32 scalar, got {value.dtype} {tuple(value.shape)}")
                return module_path + ".nvfp4_scale"
            if key.endswith(COMFY_PRE_QUANT_SCALE_SUFFIX):
                if not value.is_floating_point() or value.ndim != 1:
                    raise ValueError(f"AWQ pre_quant_scale {key} must be a 1D float tensor, got {value.dtype} {tuple(value.shape)}")
                return key
            # .weight
            if value.dtype is not torch.uint8 or value.ndim != 2:
                raise ValueError(f"NVFP4 weight {key} must be 2D uint8 (packed), got {value.dtype} ndim={value.ndim}")
            in_features = value.shape[1] * 2
            if in_features % NVFP4_BLOCK_SIZE != 0:
                raise ValueError(f"NVFP4 weight {key}: in_features {in_features} not a multiple of {NVFP4_BLOCK_SIZE}")
            self.nvfp4_module_shapes[module_path] = (value.shape[0], in_features)
            return key

        # FORMAT_INT8_TENSORWISE: per-row INT8, embeddings only (validated at patch time)
        if key.endswith(COMFY_WEIGHT_SCALE_SUFFIX):
            if value.dtype is not torch.float32:
                raise ValueError(f"INT8 per-row scale {key} must be F32, got {value.dtype}")
            return module_path + ".scale_weight"
        if key.endswith((COMFY_WEIGHT_SCALE_2_SUFFIX, COMFY_PRE_QUANT_SCALE_SUFFIX)):
            raise ValueError(f"Unexpected tensor {key} for int8_tensorwise module {module_path}")
        if value.dtype is not torch.int8:
            raise ValueError(f"INT8 weight {key} must be int8, got {value.dtype}")
        self.int8_embedding_modules.append(module_path)
        return key

    def _validate_completeness(self, state_dict: dict, module_formats: Dict[str, str]) -> None:
        for module_path, spec_format in module_formats.items():
            if spec_format == FORMAT_NVFP4:
                required = (".weight", ".nvfp4_block_scale", ".nvfp4_scale")
            else:
                required = (".weight", ".scale_weight")
            missing = [module_path + suffix for suffix in required if module_path + suffix not in state_dict]
            if missing:
                raise ValueError(f"Pre-quantized module {module_path} is missing tensors {missing}")

            if spec_format == FORMAT_NVFP4:
                rows, in_features = self.nvfp4_module_shapes[module_path]
                expected_scale_numel = _roundup(rows, 128) * _roundup(in_features // NVFP4_BLOCK_SIZE, 4)
                block_scale = state_dict[module_path + ".nvfp4_block_scale"]
                if block_scale.numel() != expected_scale_numel:
                    raise ValueError(
                        f"NVFP4 module {module_path}: block scale has {block_scale.numel()} elements,"
                        f" expected {expected_scale_numel} for weight shape ({rows}, {in_features})"
                    )
                pre_quant_scale = state_dict.get(module_path + COMFY_PRE_QUANT_SCALE_SUFFIX)
                if pre_quant_scale is not None and pre_quant_scale.shape[0] != in_features:
                    raise ValueError(
                        f"NVFP4 module {module_path}: pre_quant_scale has {pre_quant_scale.shape[0]} elements,"
                        f" expected in_features {in_features}"
                    )
            else:
                weight = state_dict[module_path + ".weight"]
                scale = state_dict[module_path + ".scale_weight"]
                expected_scale_shape = (weight.shape[0], 1)
                if tuple(scale.shape) != expected_scale_shape:
                    raise ValueError(
                        f"INT8 module {module_path}: scale shape must be {expected_scale_shape}, got {tuple(scale.shape)}"
                    )


# endregion


# region monkey patch


def nvfp4_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    pre_quant_scale = getattr(self, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        x = x * pre_quant_scale
    if self._nvfp4_use_scaled_mm:
        x_2d = x.reshape(-1, x.shape[-1])
        out = nvfp4_scaled_mm_linear(
            x_2d, self.weight, self.nvfp4_block_scale, self.nvfp4_scale, self.bias, self._nvfp4_orig_shape[0]
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
    weight = dequantize_nvfp4(self.weight, self.nvfp4_block_scale, self.nvfp4_scale, self._nvfp4_orig_shape, x.dtype)
    return F.linear(x, weight, self.bias)


def nvfp4_linear_forward_patch_autograd(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    pre_quant_scale = getattr(self, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        x = x * pre_quant_scale
    return NvFp4LinearFn.apply(
        x,
        self.weight,
        self.nvfp4_block_scale,
        self.nvfp4_scale,
        self.nvfp4_weight_t,
        self.nvfp4_block_scale_t,
        self.nvfp4_scale_t,
        self.bias,
        self._nvfp4_orig_shape[0],
    )


def int8_embedding_forward_patch(self: nn.Embedding, input: torch.Tensor) -> torch.Tensor:
    rows = self.weight[input]  # index_select works on int8; padding_idx etc. only affect training
    return (rows.float() * self.scale_weight[input]).to(self._int8_dequant_dtype)


def apply_nvfp4_monkey_patch(
    model: nn.Module,
    optimized_state_dict: dict,
    nvfp4_module_shapes: Dict[str, Tuple[int, int]],
    int8_embedding_modules: List[str],
    use_scaled_mm: bool = False,
    embedding_dtype: torch.dtype = torch.bfloat16,
    training: bool = False,
    calc_device: Optional[Union[str, torch.device]] = None,
    columnwise_chunk_rows: int = 1024,
) -> nn.Module:
    """Patch NVFP4 Linear and INT8 embedding modules so a strict assign load can follow.

    The patched modules get placeholder parameters/buffers with the quantized shapes and
    dtypes (on the meta device); ``model.load_state_dict(state_dict, strict=True,
    assign=True)`` then installs the real tensors. Modules stay ``nn.Linear`` /
    ``nn.Embedding`` (patched forward), mirroring the ConvRot INT8 approach.

    ``training=True`` additionally computes and caches a columnwise (N-grouped) NVFP4
    requantization of each weight (see ``quantize_nvfp4_weight_columnwise``) and routes the
    forward through ``NvFp4LinearFn`` for a real backward. Requires ``use_scaled_mm=True`` --
    the dequantize-fallback forward has no matching backward.

    ``calc_device``, when set, is where that columnwise requantization actually runs: each
    weight is moved there before quantizing and the result is moved back to the weight's own
    device afterward. This matters when ``optimized_state_dict`` lives on CPU (e.g. block swap
    keeps the whole state dict off-GPU), since the requant math is dequant/bit-pack heavy and
    slow on CPU across the full block count -- pass the accelerator's GPU device here to make
    it fast while leaving the resulting tensors on CPU for the block-swap offloader.

    ``columnwise_chunk_rows`` is forwarded to ``quantize_nvfp4_weight_columnwise`` (see there for
    the numerical-equivalence contract) and bounds that call's transient GPU memory peak. The
    default (1024) is a comfortable value for typical shapes (~1.3GB measured on Krea2's largest
    real Linear, 24576 out_features) -- it is a plain fixed row count, not a computed bound, so an
    unusually large out_features could still push the peak higher. Lower this via
    ``--nvfp4_columnwise_chunk_rows`` for tighter control on memory-constrained GPUs or unusually
    large models, at the cost of more quantization passes at load time (a one-time cost, not a
    per-step training cost).
    """
    if use_scaled_mm and not nvfp4_scaled_mm_available():
        raise ValueError(
            "NVFP4 scaled_mm requires PyTorch 2.10+ (torch.float4_e2m1fn_x2 and torch.nn.functional.scaled_mm)."
            " Omit the scaled_mm option to use the dequantize fallback."
            " / NVFP4 scaled_mm には PyTorch 2.10 以降が必要です。scaled_mm オプションを外すと dequantize フォールバックで動作します。"
        )
    if training and not use_scaled_mm:
        raise ValueError(
            "NVFP4 training requires use_scaled_mm=True (the dequantize fallback forward has no backward)."
        )
    if columnwise_chunk_rows <= 0 or columnwise_chunk_rows % 128 != 0:
        raise ValueError(
            f"columnwise_chunk_rows must be a positive multiple of 128 (cuBLAS block-scale tile height),"
            f" got {columnwise_chunk_rows}"
        )

    modules_by_name = dict(model.named_modules())
    patched_count = 0

    for name, (out_features, in_features) in nvfp4_module_shapes.items():
        module = modules_by_name.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"NVFP4 state dict declares {name}, which is not an nn.Linear in the model")
        weight_key = name + ".weight"
        module.weight = nn.Parameter(torch.empty_like(optimized_state_dict[weight_key], device="meta"), requires_grad=False)
        module.register_buffer(
            "nvfp4_block_scale", torch.empty_like(optimized_state_dict[name + ".nvfp4_block_scale"], device="meta")
        )
        module.register_buffer("nvfp4_scale", torch.empty((), dtype=torch.float32, device="meta"))
        pre_quant_key = name + COMFY_PRE_QUANT_SCALE_SUFFIX
        if pre_quant_key in optimized_state_dict:
            module.register_buffer("pre_quant_scale", torch.empty_like(optimized_state_dict[pre_quant_key], device="meta"))
        if training:
            if not use_scaled_mm:
                raise ValueError(
                    "NVFP4 training requires use_scaled_mm=True (the dequantize fallback forward has no backward)."
                )
            orig_weight_device = optimized_state_dict[weight_key].device
            if calc_device is not None:
                weight_for_calc = optimized_state_dict[weight_key].to(calc_device)
                block_scale_for_calc = optimized_state_dict[name + ".nvfp4_block_scale"].to(calc_device)
                tensor_scale_for_calc = optimized_state_dict[name + ".nvfp4_scale"].to(calc_device)
            else:
                weight_for_calc = optimized_state_dict[weight_key]
                block_scale_for_calc = optimized_state_dict[name + ".nvfp4_block_scale"]
                tensor_scale_for_calc = optimized_state_dict[name + ".nvfp4_scale"]
            weight_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(
                weight_for_calc,
                block_scale_for_calc,
                tensor_scale_for_calc,
                (out_features, in_features),
                chunk_rows=columnwise_chunk_rows,
            )
            if calc_device is not None:
                weight_t = weight_t.to(orig_weight_device)
                block_scale_t = block_scale_t.to(orig_weight_device)
                tensor_scale_t = tensor_scale_t.to(orig_weight_device)
            optimized_state_dict[name + ".nvfp4_weight_t"] = weight_t
            optimized_state_dict[name + ".nvfp4_block_scale_t"] = block_scale_t
            optimized_state_dict[name + ".nvfp4_scale_t"] = tensor_scale_t
            module.register_buffer("nvfp4_weight_t", torch.empty_like(weight_t, device="meta"))
            module.register_buffer("nvfp4_block_scale_t", torch.empty_like(block_scale_t, device="meta"))
            module.register_buffer("nvfp4_scale_t", torch.empty((), dtype=torch.float32, device="meta"))
        module._nvfp4_orig_shape = (out_features, in_features)
        module._nvfp4_use_scaled_mm = use_scaled_mm
        forward_fn = nvfp4_linear_forward_patch_autograd if training else nvfp4_linear_forward_patch
        module.forward = forward_fn.__get__(module, type(module))
        patched_count += 1

    for name in int8_embedding_modules:
        module = modules_by_name.get(name)
        if not isinstance(module, nn.Embedding):
            raise ValueError(
                f"int8_tensorwise module {name} is not an nn.Embedding; INT8 per-row quantization is only"
                " supported for embeddings (use ConvRot INT8 checkpoints for Linear layers)"
            )
        module.weight = nn.Parameter(torch.empty_like(optimized_state_dict[name + ".weight"], device="meta"), requires_grad=False)
        module.register_buffer("scale_weight", torch.empty_like(optimized_state_dict[name + ".scale_weight"], device="meta"))
        module._int8_dequant_dtype = embedding_dtype
        module.forward = int8_embedding_forward_patch.__get__(module, type(module))
        patched_count += 1

    if not use_scaled_mm and nvfp4_module_shapes:
        logger.info("NVFP4 runs in weight-only mode (transient dequantization per forward)")
    model.is_nvfp4 = True
    model.nvfp4_layer_count = len(nvfp4_module_shapes)
    logger.info(f"Number of NVFP4/INT8 monkey-patched modules: {patched_count}")
    return model


# endregion
