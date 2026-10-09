"""cuBLAS INT8 backend for ConvRot INT8 linear layers.

Alternative to the Triton GEMM in `convrot_int8_kernels` that runs the INT8 matrix multiplication with
`torch._int_mm` (cuBLASLt IMMA kernels). On consumer Ampere GPUs this is considerably faster than the Triton GEMM
and has no per-shape autotuning warm-up.

Two entry points:

* `int8_linear_cublas`: same signature and semantics as `convrot_int8_kernels.int8_linear`:

      y = quantize_rowwise(rotate(x)) @ weight.T, dequantized with the per-row activation scale and the weight
      scale (per-channel vector or scalar), plus optional bias.

* `int8_grad_input_rot_cublas`: input gradient of such a layer, in the rotated (ConvRot) basis:

      grad_x_rot = quantize_rowwise(g * w_scale) @ wq

  The caller rotates the result back to the original basis.

The activation rotation and the row-wise INT8 quantization use the same functions as the Triton path, so the
quantized forward activations are bit-identical; only the GEMM and its dequantization epilogue differ. The
per-channel weight scale in the backward is applied in fp32 inside the quantizer instead of being rounded to the
activation dtype first.

The int32 accumulator returned by `torch._int_mm` is rows * N * 4 bytes. `CHUNK_ROWS` limits the number of rows
processed at once to bound that buffer (results are identical, rows are independent; 0 disables chunking). It can
be set with the environment variable CONVROT_CUBLAS_CHUNK or at runtime.

Requires Triton (for the quantization and epilogue kernels) and a GPU supported by `torch._int_mm` (sm_75+).
Shapes `torch._int_mm` cannot handle (rows <= 16, dimensions not a multiple of 8) fall back to the Triton path.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from musubi_tuner.modules import convrot_int8_kernels as K

CHUNK_ROWS = int(os.environ.get("CONVROT_CUBLAS_CHUNK", "0"))

_EPILOGUE_BLOCK_N = 1024


@triton.jit
def _dequant_epilogue_kernel(
    acc_ptr,  # int32 [m, n] accumulator from torch._int_mm
    xs_ptr,  # fp32 [m] (or [m, 1]) per-row activation scale
    ws_ptr,  # fp32 [n] per-channel weight scale, or [1] scalar (unused if not HAS_WS)
    bias_ptr,  # [n] bias, any float dtype (unused if not HAS_BIAS)
    out_ptr,  # bf16/fp16/fp32 [m, n]
    n_cols,
    stride_acc,
    stride_out,
    HAS_WS: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    acc = tl.load(acc_ptr + row * stride_acc + cols, mask=mask, other=0).to(tl.float32)
    y = acc * tl.load(xs_ptr + row)
    if HAS_WS:
        if PER_CHANNEL:
            y = y * tl.load(ws_ptr + cols, mask=mask, other=0.0)
        else:
            y = y * tl.load(ws_ptr)
    if HAS_BIAS:
        y = y + tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_out + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def _dequant_epilogue(acc, xs, ws, per_channel, bias, out):
    """out = acc * xs [* ws] [+ bias], one pass over the int32 accumulator. `ws=None` skips the weight scale."""
    m, n = acc.shape
    grid = (m, triton.cdiv(n, _EPILOGUE_BLOCK_N))
    _dequant_epilogue_kernel[grid](
        acc,
        xs,
        ws if ws is not None else acc,
        bias if bias is not None else acc,
        out,
        n,
        acc.stride(0),
        out.stride(0),
        HAS_WS=ws is not None,
        PER_CHANNEL=per_channel,
        HAS_BIAS=bias is not None,
        BLOCK_N=_EPILOGUE_BLOCK_N,
        num_warps=4,
    )


def _prep_scale(weight_scale, device) -> tuple[torch.Tensor, bool]:
    """Returns the weight scale as fp32 and whether it is per-channel (vector) or a scalar."""
    if not isinstance(weight_scale, torch.Tensor):
        return torch.tensor([float(weight_scale)], device=device, dtype=torch.float32), False
    if weight_scale.numel() == 1:
        return weight_scale.reshape(1).to(torch.float32), False
    return weight_scale.reshape(-1).to(torch.float32).contiguous(), True


def _chunk_bounds(m: int, chunk: int) -> list[tuple[int, int]]:
    """Row ranges of at most `chunk` rows; a tail of <= 16 rows is merged into the previous chunk (_int_mm needs > 16)."""
    if chunk <= 0 or m <= chunk:
        return [(0, m)]
    bounds = list(range(0, m, chunk)) + [m]
    if m - bounds[-2] <= 16:
        del bounds[-2]
    return list(zip(bounds[:-1], bounds[1:]))


def int8_linear_cublas(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
) -> torch.Tensor:
    """INT8 linear layer using cuBLAS. Same signature and semantics as `convrot_int8_kernels.int8_linear`.

    x: [..., K] activations, weight: [N, K] int8, weight_scale: per-channel [N] / [N, 1] or scalar (fp32).
    """
    orig_shape = x.shape
    x2d = x.reshape(-1, orig_shape[-1])
    m, k = x2d.shape
    n = weight.shape[0]

    # torch._int_mm needs rows > 16 and dimensions that are multiples of 8
    if m <= 16 or k % 8 != 0 or n % 8 != 0:
        return K.int8_linear(x, weight, weight_scale, bias, out_dtype, convrot, convrot_groupsize)

    if convrot:
        h = K._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x2d = K._rotate_activation(x2d, h, convrot_groupsize)
        x8, xs = K.triton_quantize_rowwise(x2d)  # int8 [m, k], fp32 [m, 1]
        del x2d  # free the rotated copy before the GEMM; the caller's x is untouched
    else:
        x8, xs = K.triton_quantize_rowwise(x2d)

    ws, per_channel = _prep_scale(weight_scale, x.device)
    wt = weight.t()  # [k, n] column-major view of the row-major [n, k] weight, the layout cuBLAS wants

    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    for r0, r1 in _chunk_bounds(m, CHUNK_ROWS):
        acc = torch._int_mm(x8[r0:r1], wt)  # int32 [rows, n], freed after each chunk
        _dequant_epilogue(acc, xs[r0:r1], ws, per_channel, bias, out[r0:r1])
        del acc

    return out.reshape(*orig_shape[:-1], n)


@triton.jit
def _quantize_rowwise_colscale_kernel(x_ptr, cs_ptr, y_ptr, s_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    v = x * tl.load(cs_ptr + cols, mask=mask, other=0.0)
    scale = tl.maximum(tl.max(tl.abs(v), axis=0) / 127.0, 1e-30)
    q = tl.clamp(libdevice.rint(v / scale), -128.0, 127.0).to(tl.int32)
    tl.store(y_ptr + row * n_cols + cols, q.to(tl.int8), mask=mask)
    tl.store(s_ptr + row, scale)


def _quantize_rowwise_colscale(x2d: torch.Tensor, col_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise INT8 quantization of x2d * col_scale (scale applied in fp32, no intermediate copy).

    Returns int8 [m, n] and fp32 [m, 1] row scales.
    """
    m, n = x2d.shape
    x2d = x2d.contiguous()
    y = torch.empty((m, n), device=x2d.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=x2d.device, dtype=torch.float32)
    block = max(triton.next_power_of_2(n), 128)
    _quantize_rowwise_colscale_kernel[(m,)](x2d, col_scale, y, s, n, BLOCK=block, num_warps=8 if block >= 8192 else 4)
    return y, s


def int8_grad_input_rot_cublas(
    g2d: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    out_dtype: torch.dtype,
    groupsize: int = 256,
) -> torch.Tensor:
    """Input gradient of a ConvRot INT8 linear layer, in the rotated basis: (g * w_scale) @ wq.

    g2d: [m, n] output gradient, wq: [n, k] int8 weight, w_scale: [n, 1] or [n] fp32.
    Returns [m, k]; the caller rotates it back to the original basis.
    """
    m, n = g2d.shape
    k = wq.shape[1]
    cs = w_scale.reshape(-1).to(torch.float32).contiguous()

    if m <= 16 or n % 8 != 0 or k % 8 != 0:
        one = torch.ones(1, device=g2d.device, dtype=torch.float32)
        g_scaled = g2d * cs.reshape(1, -1).to(g2d.dtype)
        return K.int8_linear(g_scaled, wq.t().contiguous(), one, None, out_dtype, False, groupsize)

    g8, gs = _quantize_rowwise_colscale(g2d, cs)

    # The GEMM needs wq as a column-major [n, k] operand, i.e. a row-major [k, n] copy viewed transposed.
    # Transient (1 byte/param), freed after the GEMMs.
    wt = wq.t().contiguous()

    out = torch.empty((m, k), device=g2d.device, dtype=out_dtype)
    for r0, r1 in _chunk_bounds(m, CHUNK_ROWS):
        acc = torch._int_mm(g8[r0:r1], wt.t())  # int32 [rows, k], freed after each chunk (the double-transpose is intended)
        _dequant_epilogue(acc, gs[r0:r1], None, False, None, out[r0:r1])
        del acc

    return out
