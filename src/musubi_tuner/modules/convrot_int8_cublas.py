"""cuBLAS INT8 backend for ConvRot INT8 linear layers.

Alternative to the Triton GEMM in `convrot_int8_kernels` that runs the INT8 matrix multiplication with
`torch._int_mm` (cuBLASLt IMMA kernels). On consumer Ampere GPUs this is considerably faster than the Triton GEMM
and has no per-shape autotuning warm-up.

Two entry points:

* `int8_linear_cublas`: same signature and semantics as `convrot_int8_kernels.int8_linear`:

      y = quantize_rowwise(rotate(x)) @ weight.T, dequantized with the per-row activation scale and the weight
      scale (per-channel vector or scalar), plus optional bias.

* `int8_grad_input_rot_cublas`: input gradient of such a layer, in the rotated (ConvRot) basis:

      grad_x_rot = quantize_rowwise(g * weight_scale) @ weight

  The caller rotates the result back to the original basis.

The activation rotation and the row-wise INT8 quantization use the same functions as the Triton path, so the
quantized forward activations are bit-identical; only the GEMM and its dequantization epilogue differ. The
per-channel weight scale in the backward is applied in fp32 inside the quantizer instead of being rounded to the
activation dtype first.

The int32 accumulator returned by `torch._int_mm` is rows * N * 4 bytes. `CHUNK_ROWS` limits the number of rows
processed at once to bound that buffer (results are identical, rows are independent; 0 disables chunking). It can
be set with the environment variable CONVROT_CUBLAS_CHUNK or at runtime.

Requires Triton (for the quantization and epilogue kernels) and a GPU supported by `torch._int_mm` (sm_75+).
Shapes `torch._int_mm` cannot handle (rows <= 16, dimensions not a multiple of 8) must fall back to the Triton path.
The functions don't check the shape; a caller must check with `supports_shape()` before calling the functions.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from itertools import pairwise

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from musubi_tuner.modules import convrot_int8_kernels as K

_EPILOGUE_BLOCK_N = 1024


def ensure_available(device: torch.device | None = None):
    """Raises RuntimeError unless the cuBLAS INT8 backend can run on `device` (default: the current CUDA device).

    The backend is built on `torch._int_mm`, PyTorch's INT8 matrix multiplication (int8 x int8 -> int32, run on
    NVIDIA tensor cores through cuBLASLt IMMA kernels). It is a private API: the leading underscore means PyTorch
    gives no stability guarantee, so its signature, shape constraints or availability can change or disappear
    between releases, and it may be missing or broken for some GPUs, CUDA builds or ROCm.

    Rather than comparing version numbers, this runs a tiny GEMM once per device in both layouts used here and checks
    the result exactly. That fails early with a clear message (use the Triton backend instead) and avoids a crash
    or wrong results deep into training. The result is cached per device.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("ConvRot INT8 cuBLAS backend needs a CUDA GPU")

    if device is not None and device.type != "cuda":
        raise RuntimeError(f"ConvRot INT8 cuBLAS backend needs a CUDA device, got {device}")
    if device is not None and device.index is not None:
        dev_index = device.index
        if dev_index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {device} does not exist ({torch.cuda.device_count()} available)")
    else:
        dev_index = torch.cuda.current_device()

    reason = _probe_int_mm(dev_index)
    if reason is not None:
        raise RuntimeError(f"{reason}.\nUse '--convrot_int8_backend triton' instead or try another torch version.")


@lru_cache(maxsize=None)
def _probe_int_mm(dev_index: int) -> str | None:
    """Runs torch._int_mm once on a tiny problem in both production layouts.
    Returns None if it works on this device, otherwise a human-readable reason.
    Results are cached per device index.
    """
    if not hasattr(torch, "_int_mm"):
        return f"torch {torch.__version__} has no torch._int_mm"

    dev = torch.device("cuda", dev_index)
    dev_name = torch.cuda.get_device_name(dev)

    # m=17 is the minimum supports_shape() allows
    # n and k are multiples of 8; n != k so a transposition mistake shows up as a wrong shape
    m, n, k = 17, 32, 48
    gen = torch.Generator().manual_seed(0)
    act    = torch.randint(-127, 128, (m, k), dtype=torch.int8, generator=gen)  # forward input  [m, k]
    grad   = torch.randint(-127, 128, (m, n), dtype=torch.int8, generator=gen)  # backward input [m, n]
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=gen)  # weight [n, k]

    try:
        w = weight.to(dev)
        fwd = torch._int_mm(act.to(dev), w.t())  # forward layout: transposed view of the row-major weight
        bwd = torch._int_mm(grad.to(dev), w.t().contiguous().t())  # backward layout: [k, n] copy viewed as [n, k]
        torch.cuda.synchronize(dev)
    except Exception as e:  # unsupported GPU / CUDA build / ROCm: the exception type varies
        return f"torch._int_mm failed on {dev_name}:\n{type(e).__name__}: {e}"

    # Integer GEMM is exact: compare bit for bit against an int64 matmul on the CPU (|sum| < 2**31, no overflow).
    for name, result, a, b in (("forward", fwd, act, weight.t()), ("backward", bwd, grad, weight)):
        expected = a.to(torch.int64) @ b.to(torch.int64)
        if result.dtype != torch.int32 or result.shape != expected.shape:
            return f"torch._int_mm {name} returned {result.dtype} {tuple(result.shape)} on {dev_name}, " \
                   f"expected int32 {tuple(expected.shape)}"
        if not torch.equal(result.cpu(), expected.to(torch.int32)):
            return f"torch._int_mm {name} returned unexpected values on {dev_name}"
    return None


def supports_shape(activation: torch.Tensor, weight: torch.Tensor) -> bool:
    """True if torch._int_mm can run `activation @ weight` (forward) or `activation @ weight.T`-style contractions
    with this weight (backward): needs more than 16 rows, and both weight dimensions multiples of 8.

    activation: [..., K] (forward, x), or [m, N] (backward, g2d)
    weight:     [N, K] int8.
    """
    n, k = weight.shape
    rows = activation.numel() // activation.shape[-1]
    return rows > 16 and n % 8 == 0 and k % 8 == 0


@triton.jit
def _dequant_epilogue_kernel(
    acc_ptr,    # int32 [m, n] raw accumulator from torch._int_mm; rows are `stride_acc` apart, columns contiguous
    xs_ptr,     # fp32 [m] (or [m, 1]) per-row activation scale from the row-wise quantizer
    ws_ptr,     # fp32 [n] per-channel weight scale, or [1] scalar (only read if HAS_WS)
    bias_ptr,   # [n] bias, any float dtype (only read if HAS_BIAS)
    out_ptr,    # bf16/fp16/fp32 [m, n] result; rows are `stride_out` apart, columns contiguous (may be a row slice)
    n_cols,     # n: number of valid columns; the mask disables the padding lanes of the last column block
    stride_acc, # row stride of acc, in elements
    stride_out, # row stride of out, in elements
    HAS_WS: tl.constexpr,       # multiply by the weight scale (False in the backward: it is already folded into the gradient)
    PER_CHANNEL: tl.constexpr,  # weight scale is a vector over columns (True) or a single scalar (False); needs HAS_WS
    HAS_BIAS: tl.constexpr,     # add the bias
    BLOCK_N: tl.constexpr,      # columns handled per program
):
    """Dequantization epilogue of an INT8 GEMM: out[r, c] = acc[r, c] * xs[r] [* ws[c]] [+ bias[c]].

    `torch._int_mm` returns the raw int32 products. They still have to be scaled back to real values with the per-row
    activation scale and the weight scale, optionally biased, and converted to the output dtype. The Triton GEMM
    does this inside the matmul; with cuBLAS it is a separate pass. Doing it in one kernel avoids the chain of
    elementwise torch ops (fp32 temporaries, several passes over the data).

    One program handles BLOCK_N columns of one row (grid: rows x column blocks). The kernel is memory-bound
    (reads 4 bytes, writes 2 bytes per element). The constexpr flags are resolved at compile time, so each
    parameter combination is its own specialization without runtime branches.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = (tl.program_id(1) * BLOCK_N) + tl.arange(0, BLOCK_N)
    mask = cols < n_cols

    acc_offsets = (row * stride_acc) + cols
    acc = tl.load(acc_ptr + acc_offsets, mask=mask, other=0).to(tl.float32)

    # Apply row activation scale
    y = acc * tl.load(xs_ptr + row)

    # Apply weight scale
    if HAS_WS:
        if PER_CHANNEL:
            y *= tl.load(ws_ptr + cols, mask=mask, other=0.0)
        else:
            y *= tl.load(ws_ptr)

    # Apply bias
    if HAS_BIAS:
        y += tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    out_offsets = (row * stride_out) + cols
    tl.store(out_ptr + out_offsets, y.to(out_ptr.dtype.element_ty), mask=mask)


def _dequant_epilogue(
    acc: torch.Tensor,
    xs: torch.Tensor,
    ws: torch.Tensor | None,
    per_channel: bool,
    bias: torch.Tensor | None,
    out: torch.Tensor,
):
    """Dequantize the int32 accumulator of `torch._int_mm` into `out`: out = acc * xs [* ws] [+ bias].

    Launches `_dequant_epilogue_kernel` in one pass over the accumulator, writing into the preallocated `out`
    (rows must be contiguous, but `out` can be a row slice of a larger tensor, which is how the chunked GEMM loops use it).

    Used by `int8_linear_cublas` (forward: per-channel or scalar weight scale, optional bias), and by
    `int8_grad_input_rot_cublas` (backward: `ws=None`, because the weight scale is already folded into the gradient
                                  before it is quantized).

    acc:  int32 [m, n]
    xs:   fp32 per-row scale [m] or [m, 1]
    ws:   fp32 [n] (per_channel=True) or [1] (per_channel=False), or None to skip the weight scale
    bias: [n] or None
    out:  [m, n] in the output dtype.
    """
    m, n = acc.shape
    grid = (m, triton.cdiv(n, _EPILOGUE_BLOCK_N))
    _dequant_epilogue_kernel[grid](
        acc,
        xs,
        # Unused pointers still need a valid tensor; the constexpr flags below remove the corresponding loads
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


def _chunk_bounds(m: int, chunk: int) -> Iterator[tuple[int, int]]:
    """Row ranges of at most `chunk` rows; a tail of <= 16 rows is merged into the previous chunk (_int_mm needs > 16)."""
    if chunk <= 0 or m <= chunk:
        yield (0, m)
        return

    num_full_chunks, remainder = divmod(m, chunk)
    if 0 < remainder <= 16:
        num_full_chunks -= 1
        remainder += chunk

    yield from pairwise(i * chunk for i in range(num_full_chunks + 1))  # Include right bound (+1)

    if remainder > 0:
        yield (num_full_chunks * chunk, m)


def int8_linear_cublas(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    convrot: bool = False,
    convrot_groupsize: int = 256,
    chunk_size: int = 0,
) -> torch.Tensor:
    """INT8 linear layer using cuBLAS. Same signature and semantics as `convrot_int8_kernels.int8_linear`.

    x:              [..., K] activations
    weight:         [N, K] int8
    weight_scale:   per-channel [N] / [N, 1] or scalar [1].
    """
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    m, k = x_2d.shape
    n = weight.shape[0]

    if weight_scale.numel() not in (1, n):
        raise ValueError(f"weight_scale must be scalar or have {n} elements, got {tuple(weight_scale.shape)}")

    if convrot:
        h = K._build_hadamard(convrot_groupsize, device=x.device, dtype=x.dtype)
        x_rotated = K._rotate_activation(x_2d, h, convrot_groupsize)
        x_int8, x_scale = K.triton_quantize_rowwise(x_rotated)  # int8 [m, k], fp32 [m, 1]
        del x_rotated  # free the transient tensor before the GEMM, 'h' is cached
    else:
        x_int8, x_scale = K.triton_quantize_rowwise(x_2d)

    is_per_channel = weight_scale.numel() > 1
    ws = weight_scale.reshape(-1).to(torch.float32).contiguous()
    wt = weight.t()  # [k, n] column-major view of the row-major [n, k] weight, the layout cuBLAS wants

    if bias is not None:
        bias = bias.contiguous()

    out = torch.empty((m, n), device=x.device, dtype=out_dtype)
    for r0, r1 in _chunk_bounds(m, chunk_size):
        acc = torch._int_mm(x_int8[r0:r1], wt)  # int32 [rows, n]
        _dequant_epilogue(acc, x_scale[r0:r1], ws, is_per_channel, bias, out[r0:r1])
        del acc  # free it explicitly so it's gone while the next _int_mm runs - don't wait for the assignment to free it

    return out.reshape(*orig_shape[:-1], n)


# Why the backward has its own quantizer instead of reusing int8_linear_cublas:
# The int8 input gradient needs (g * w_scale) quantized row-wise. Computing it as a separate torch op materializes a full bf16 copy
# of the gradient which stays alive for the whole call because the caller holds the reference, so int8_linear_cublas cannot free it.
# This kernel applies the per-column scale in fp32 registers while quantizing each row, so the copy never exists.
# It also gave two other benefits:
# - It saved about one elementwise pass, which showed up as a ~5% backward speedup.
# - The scale product is no longer rounded to bf16 before quantization, which lowered the backward error slightly.
@triton.jit
def _quantize_rowwise_colscale_kernel(
    x_ptr,      # [m, n] values to quantize, any float dtype (e.g. the bf16 output gradient), contiguous
    cs_ptr,     # fp32 [n] per-column scale, multiplied into x in fp32 before quantizing (the weight scale)
    out_y_ptr,  # int8 [m, n] quantized output
    out_s_ptr,  # fp32 [m] (or [m, 1]) per-row dequantization scale: max(|x * cs|) / 127 of each row
    n_cols,     # n, the row length; the mask disables the padding lanes when BLOCK is larger than n
    BLOCK: tl.constexpr,  # power of two >= n_cols; one program holds a whole row, as the row maximum is needed first
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    offsets = (row * n_cols) + cols

    # Padding lanes (cols >= n_cols) load as 0 with the mask, so they don't affect the row maximum below
    mask = cols < n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Fold the per-column weight scale into the gradient in fp32 registers
    xs = x * tl.load(cs_ptr + cols, mask=mask, other=0.0)

    # The row's largest magnitude, scalar
    absmax = tl.max(tl.abs(xs), axis=0)

    # The stored dequantization scale (resolution): Real value of one int8 step for this row.
    # Floor to avoid 0 / 0 = NaN for all-zero rows.
    step = tl.maximum(absmax / 127.0, 1e-30)

    # Scale the row, mapping 'absmax' to +/-127, then round to nearest int (exact halves round to even).
    # Max rounding error: step/2
    q = libdevice.rint(xs / step)

    # Saturate to the int8 range. Normally a no-op: values are within +/-127, so -128 is never produced.
    q = tl.clamp(q, -128.0, 127.0)

    tl.store(out_y_ptr + offsets, q.to(tl.int8), mask=mask)
    tl.store(out_s_ptr + row, step)


def _quantize_rowwise_colscale(x_2d: torch.Tensor, col_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise INT8 quantization of x_2d * col_scale, fused into one kernel.

    The product is formed in fp32 registers and never written to memory, unlike `x_2d * col_scale` followed by a
    separate quantizer, which allocates an extra full-size bf16 tensor. One program handles a whole row, because the
    quantization scale needs the row's absolute maximum before any element can be quantized.

    Returns int8 [m, n] and fp32 [m, 1] row scales.
    """
    m, n = x_2d.shape
    x_2d = x_2d.contiguous()
    y = torch.empty((m, n), device=x_2d.device, dtype=torch.int8)
    s = torch.empty((m, 1), device=x_2d.device, dtype=torch.float32)
    block = max(triton.next_power_of_2(n), 128)
    warps = 8 if block >= 8192 else 4
    _quantize_rowwise_colscale_kernel[(m,)](x_2d, col_scale, y, s, n, BLOCK=block, num_warps=warps)
    return y, s


def int8_grad_input_rot_cublas(
    g_2d: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Input gradient of a ConvRot INT8 linear layer, in the rotated basis: (g * weight_scale) @ weight.

    g_2d:           [m, n] output gradient
    weight:         [n, k] int8
    weight_scale:   per-channel [n, 1] or [n] fp32

    Returns [m, k]; the caller rotates it back to the original basis.
    """
    m, n = g_2d.shape
    k = weight.shape[1]

    col_scale = weight_scale.reshape(-1).to(torch.float32).contiguous()
    if col_scale.numel() != n:
        raise ValueError(f"weight_scale must have {n} elements, got {tuple(weight_scale.shape)}")

    # torch._int_mm wants the weights as a column-major [n, k] operand:
    # Make a row-major [k, n] copy to make it contiguous over n, then view it transposed.
    # Passing 'weight' directly is accepted and gives the same result, but is 1.5-2.5x slower in the backward
    # (RTX 4090, 4352 tokens), so the copy (1 byte/param, transient, freed on return) is worth it.
    wt = weight.t().contiguous().t()  # [n, k] view of the [k, n] copy: transposing twice is intended

    g_int8, gs = _quantize_rowwise_colscale(g_2d, col_scale)

    out = torch.empty((m, k), device=g_2d.device, dtype=out_dtype)
    for r0, r1 in _chunk_bounds(m, chunk_size):
        acc = torch._int_mm(g_int8[r0:r1], wt)  # int32 [rows, k]
        _dequant_epilogue(acc, gs[r0:r1], None, False, None, out[r0:r1])
        del acc  # free it explicitly so it's gone while the next _int_mm runs - don't wait for the assignment to free it

    return out
