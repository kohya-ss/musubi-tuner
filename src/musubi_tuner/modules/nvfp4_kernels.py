"""Fused Triton kernel for NVFP4 activation quantization.

quantize_nvfp4_activation (nvfp4_utils.py) is called on both the forward path's `x` and the
backward path's `grad_out` for every quantized Linear -- 224 Linears (8 per block x 28 blocks)
in both directions for Krea2 -- so it dominates --nvfp4's per-step cost. The unfused, pure-
PyTorch pipeline it replaces runs five separate eager passes over the input (per-block amax,
F8_E4M3 scale cast, elementwise normalize+clamp, E2M1 bit conversion, nibble packing) plus a
to_blocked permute for the cuBLAS swizzled scale layout -- each allocating its own full-size
temporaries. This module fuses all of that into a single Triton kernel launch (one program per
row), mirroring how convrot_int8_kernels.py's triton_quantize_rowwise fuses row-wise amax +
INT8 quantize into one kernel. If Triton isn't importable (HAS_TRITON is False), callers fall
back to nvfp4_utils._quantize_nvfp4_2d's original unfused implementation -- this module never
changes quantize_nvfp4_activation's signature or return contract, only its internal dispatch.

The E2M1 bit-conversion constants below are ported from nvfp4_utils._f32_to_e2m1_unpacked
with ebits=2, mbits=1 fixed (E2M1 is the only format this kernel handles). Verified against
that reference by tests/test_nvfp4_kernels.py: bit-exact for small/edge-case inputs, and at
real Krea2 tensor sizes matching to within a documented, bounded rounding-tie tolerance (an
input value a few ULPs from an exact rounding boundary can land on opposite sides of it
between Triton's in-kernel fp32 arithmetic and PyTorch/ATen's eager ops -- an unavoidable
floating-point divergence between two independently-ordered implementations, not a logic
bug; see the test file's _compare helper for the adjacency check that distinguishes this
from an actual decoding error).
"""

import struct

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

NVFP4_BLOCK_SIZE = 16
F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0

_E2M1_MAX_INT = 7
_E2M1_DENORM_MASK_INT = 1249902592  # 149 << 23
_E2M1_ADDER_CONST = -1054867457  # ((1-127) << 23) + ((1 << 21) - 1)
_E2M1_SIGN_SHIFT = 28
_E2M1_DENORM_MASK_F32 = struct.unpack("f", struct.pack("i", _E2M1_DENORM_MASK_INT))[0]


if HAS_TRITON:

    @triton.jit
    def _e2m1_magnitude_code(
        x_pos, DENORM_MASK_F32: tl.constexpr, DENORM_MASK_INT: tl.constexpr, ADDER_CONST: tl.constexpr, MAX_INT: tl.constexpr
    ):
        """x_pos: non-negative fp32 tensor. Returns uint8 magnitude code 0..7."""
        saturate_mask = x_pos >= 6.0
        denormal_mask = (~saturate_mask) & (x_pos < 1.0)
        normal_mask = ~(saturate_mask | denormal_mask)

        denorm_added = x_pos + DENORM_MASK_F32
        denorm_bits = denorm_added.to(tl.int32, bitcast=True) - DENORM_MASK_INT
        denormal_code = denorm_bits.to(tl.uint8)

        x_bits = x_pos.to(tl.int32, bitcast=True)
        mant_odd = (x_bits >> 22) & 1
        normal_bits = x_bits + ADDER_CONST + mant_odd
        normal_code = (normal_bits >> 22).to(tl.uint8)

        result = tl.where(denormal_mask, denormal_code, MAX_INT)
        result = tl.where(normal_mask, normal_code, result)
        return result

    @triton.jit
    def _e2m1_code(
        x,
        DENORM_MASK_F32: tl.constexpr,
        DENORM_MASK_INT: tl.constexpr,
        ADDER_CONST: tl.constexpr,
        MAX_INT: tl.constexpr,
        SIGN_SHIFT: tl.constexpr,
    ):
        """x: signed fp32 tensor. Returns uint8 E2M1 code 0..15 (sign in bit 3)."""
        x_bits = x.to(tl.int32, bitcast=True)
        sign_bit = x_bits & tl.full(x_bits.shape, -2147483648, tl.int32)
        x_pos_bits = x_bits ^ sign_bit
        x_pos = x_pos_bits.to(tl.float32, bitcast=True)
        mag_code = _e2m1_magnitude_code(x_pos, DENORM_MASK_F32, DENORM_MASK_INT, ADDER_CONST, MAX_INT)
        sign_code = ((sign_bit >> SIGN_SHIFT) & 8).to(tl.uint8)
        return mag_code | sign_code

    @triton.jit
    def _e2m1_stochastic_magnitude_code(x_pos, r):
        """x_pos: non-negative fp32 tensor, values in [0, 6.0]. r: uniform [0, 1) fp32 tensor of
        the same shape (caller-supplied via tl.rand so the same random stream can be reused for
        the sign-independent magnitude draw). Returns uint8 magnitude code 0..7, stochastically
        rounded to one of the two nearest representable E2M1 magnitudes with probability
        proportional to inverse distance -- mirrors nvfp4_utils._e2m1_stochastic_magnitude_code's
        searchsorted-based bracket selection via a chained comparison (equivalent for this fixed,
        monotonically increasing 8-entry table): the bracket [lo, hi) chosen this way exactly
        reproduces searchsorted(table, x_pos, right=True)'s index for every x_pos, including the
        exact-table-value and saturating (x_pos==6.0) degenerate cases (both resolve to lo==hi,
        i.e. span==0, forcing a deterministic round to that value).
        """
        lo = tl.where(x_pos >= 0.5, 0.5, 0.0)
        lo = tl.where(x_pos >= 1.0, 1.0, lo)
        lo = tl.where(x_pos >= 1.5, 1.5, lo)
        lo = tl.where(x_pos >= 2.0, 2.0, lo)
        lo = tl.where(x_pos >= 3.0, 3.0, lo)
        lo = tl.where(x_pos >= 4.0, 4.0, lo)
        lo = tl.where(x_pos >= 6.0, 6.0, lo)

        lo_idx = tl.where(x_pos >= 0.5, 1, 0)
        lo_idx = tl.where(x_pos >= 1.0, 2, lo_idx)
        lo_idx = tl.where(x_pos >= 1.5, 3, lo_idx)
        lo_idx = tl.where(x_pos >= 2.0, 4, lo_idx)
        lo_idx = tl.where(x_pos >= 3.0, 5, lo_idx)
        lo_idx = tl.where(x_pos >= 4.0, 6, lo_idx)
        lo_idx = tl.where(x_pos >= 6.0, 7, lo_idx)
        hi_idx = tl.minimum(lo_idx + 1, 7)

        hi = tl.where(hi_idx == 0, 0.0, 6.0)
        hi = tl.where(hi_idx == 1, 0.5, hi)
        hi = tl.where(hi_idx == 2, 1.0, hi)
        hi = tl.where(hi_idx == 3, 1.5, hi)
        hi = tl.where(hi_idx == 4, 2.0, hi)
        hi = tl.where(hi_idx == 5, 3.0, hi)
        hi = tl.where(hi_idx == 6, 4.0, hi)

        span = hi - lo
        span_is_zero = span == 0.0
        span_safe = tl.where(span_is_zero, 1.0, span)
        p_up = tl.where(span_is_zero, 0.0, (x_pos - lo) / span_safe)

        round_up = r < p_up
        return tl.where(round_up, hi_idx, lo_idx).to(tl.uint8)

    @triton.jit
    def _e2m1_stochastic_code(x, r):
        """x: signed fp32 tensor. r: uniform [0, 1) fp32 tensor, same shape as x. Returns uint8
        E2M1 code 0..15, stochastically rounded, sign in bit 3 (same encoding as _e2m1_code)."""
        neg = x < 0.0
        mag_code = _e2m1_stochastic_magnitude_code(tl.abs(x), r)
        return tl.where(neg, mag_code | 8, mag_code)

    @triton.jit
    def _quantize_nvfp4_row_kernel(
        x_ptr,
        packed_ptr,
        scale_ptr,
        per_tensor_scale_ptr,
        DENORM_MASK_F32: tl.constexpr,
        DENORM_MASK_INT: tl.constexpr,
        ADDER_CONST: tl.constexpr,
        MAX_INT: tl.constexpr,
        SIGN_SHIFT: tl.constexpr,
        K: tl.constexpr,
        n_groups: tl.constexpr,
        n_col_blocks: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        per_tensor_scale = tl.load(per_tensor_scale_ptr)

        group_idx = tl.arange(0, BLOCK_GROUPS)  # [BLOCK_GROUPS]
        group_mask = group_idx < n_groups
        pair_idx = tl.arange(0, 8)  # [8]
        group2 = group_idx[:, None]
        pair2 = pair_idx[None, :]
        mask2 = group_mask[:, None]

        high_col = group2 * 16 + pair2 * 2
        low_col = high_col + 1

        row_ptr = x_ptr + row * K
        x_high = tl.load(row_ptr + high_col, mask=mask2, other=0.0)  # [BLOCK_GROUPS, 8]
        x_low = tl.load(row_ptr + low_col, mask=mask2, other=0.0)  # [BLOCK_GROUPS, 8]

        block_amax = tl.max(tl.maximum(tl.abs(x_high), tl.abs(x_low)), axis=1)  # [BLOCK_GROUPS]
        block_scale = block_amax / 6.0  # F4_E2M1_MAX

        denom = tl.maximum(per_tensor_scale, 1.1754943508222875e-38)
        scaled = tl.minimum(block_scale / denom, 448.0)  # F8_E4M3_MAX
        scaled_f8 = scaled.to(tl.float8e4nv)
        scaled_f8_f32 = scaled_f8.to(tl.float32)

        total = per_tensor_scale * scaled_f8_f32  # [BLOCK_GROUPS]
        total_is_zero = total == 0.0
        total_safe = tl.where(total_is_zero, 1.0, total)
        total_safe_2d = total_safe[:, None]

        data_high = tl.where(total_is_zero[:, None], 0.0, x_high / total_safe_2d)
        data_low = tl.where(total_is_zero[:, None], 0.0, x_low / total_safe_2d)
        data_high = tl.clamp(data_high, -6.0, 6.0)
        data_low = tl.clamp(data_low, -6.0, 6.0)

        code_high = _e2m1_code(data_high, DENORM_MASK_F32, DENORM_MASK_INT, ADDER_CONST, MAX_INT, SIGN_SHIFT)
        code_low = _e2m1_code(data_low, DENORM_MASK_F32, DENORM_MASK_INT, ADDER_CONST, MAX_INT, SIGN_SHIFT)
        packed_byte = (code_high << 4) | code_low  # [BLOCK_GROUPS, 8] uint8, element 0 = high nibble

        packed_row_ptr = packed_ptr + row * (n_groups * 8)
        packed_col = group2 * 8 + pair2
        tl.store(packed_row_ptr + packed_col, packed_byte, mask=mask2)

        # Swizzled (cuBLAS 128x4 tiled) scale write, computed in closed form instead of via a
        # separate to_blocked permute pass. The scale tensor is tiled into 128-row x 4-col
        # blocks, each tile flattened to 512 contiguous elements: row_block/col_block pick the
        # tile; within it, r_in_tile (0..127) splits into a 32-row sub-block index `a` (0..3)
        # and a row-within-sub-block `b` (0..31), c_in_tile (0..3) is the column within the
        # tile, `n` is the tile's linear index among all tiles, and `d = a*4 + c_in_tile`
        # combines the sub-block and column into the tile's inner 16-wide axis -- so
        # flat = n*512 + b*16 + d lands each (row, col) scale at the same offset
        # nvfp4_utils.to_blocked's view/permute/reshape chain would produce.
        row_block = (row // 128).to(tl.int32)
        r_in_tile = (row % 128).to(tl.int32)
        a = r_in_tile // 32
        b = r_in_tile % 32
        col_block = group_idx // 4
        c_in_tile = group_idx % 4
        n = row_block * n_col_blocks + col_block
        d = a * 4 + c_in_tile
        flat = n * 512 + b * 16 + d

        scale_bits = scaled_f8.to(tl.uint8, bitcast=True)  # [BLOCK_GROUPS]
        tl.store(scale_ptr + flat, scale_bits, mask=group_mask)

    @triton.jit
    def _quantize_nvfp4_row_kernel_stochastic(
        x_ptr,
        packed_ptr,
        scale_ptr,
        per_tensor_scale_ptr,
        seed,
        K: tl.constexpr,
        n_groups: tl.constexpr,
        n_col_blocks: tl.constexpr,
        BLOCK_GROUPS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        per_tensor_scale = tl.load(per_tensor_scale_ptr)

        group_idx = tl.arange(0, BLOCK_GROUPS)  # [BLOCK_GROUPS]
        group_mask = group_idx < n_groups
        pair_idx = tl.arange(0, 8)  # [8]
        group2 = group_idx[:, None]
        pair2 = pair_idx[None, :]
        mask2 = group_mask[:, None]

        high_col = group2 * 16 + pair2 * 2
        low_col = high_col + 1

        row_ptr = x_ptr + row * K
        x_high = tl.load(row_ptr + high_col, mask=mask2, other=0.0)  # [BLOCK_GROUPS, 8]
        x_low = tl.load(row_ptr + low_col, mask=mask2, other=0.0)  # [BLOCK_GROUPS, 8]

        block_amax = tl.max(tl.maximum(tl.abs(x_high), tl.abs(x_low)), axis=1)  # [BLOCK_GROUPS]
        block_scale = block_amax / 6.0  # F4_E2M1_MAX

        denom = tl.maximum(per_tensor_scale, 1.1754943508222875e-38)
        scaled = tl.minimum(block_scale / denom, 448.0)  # F8_E4M3_MAX
        scaled_f8 = scaled.to(tl.float8e4nv)
        scaled_f8_f32 = scaled_f8.to(tl.float32)

        total = per_tensor_scale * scaled_f8_f32  # [BLOCK_GROUPS]
        total_is_zero = total == 0.0
        total_safe = tl.where(total_is_zero, 1.0, total)
        total_safe_2d = total_safe[:, None]

        data_high = tl.where(total_is_zero[:, None], 0.0, x_high / total_safe_2d)
        data_low = tl.where(total_is_zero[:, None], 0.0, x_low / total_safe_2d)
        data_high = tl.clamp(data_high, -6.0, 6.0)
        data_low = tl.clamp(data_low, -6.0, 6.0)

        # Global per-element offsets so every element in this launch draws from an independent
        # Philox stream, regardless of program (row) id -- offset = flat element index.
        offs_high = row * K + high_col
        offs_low = row * K + low_col
        r_high = tl.rand(seed, offs_high)
        r_low = tl.rand(seed, offs_low)

        code_high = _e2m1_stochastic_code(data_high, r_high)
        code_low = _e2m1_stochastic_code(data_low, r_low)
        packed_byte = (code_high << 4) | code_low  # [BLOCK_GROUPS, 8] uint8, element 0 = high nibble

        packed_row_ptr = packed_ptr + row * (n_groups * 8)
        packed_col = group2 * 8 + pair2
        tl.store(packed_row_ptr + packed_col, packed_byte, mask=mask2)

        # Swizzled (cuBLAS 128x4 tiled) scale write -- identical to _quantize_nvfp4_row_kernel's,
        # since block-scale computation doesn't depend on the magnitude-rounding mode.
        row_block = (row // 128).to(tl.int32)
        r_in_tile = (row % 128).to(tl.int32)
        a = r_in_tile // 32
        b = r_in_tile % 32
        col_block = group_idx // 4
        c_in_tile = group_idx % 4
        n = row_block * n_col_blocks + col_block
        d = a * 4 + c_in_tile
        flat = n * 512 + b * 16 + d

        scale_bits = scaled_f8.to(tl.uint8, bitcast=True)  # [BLOCK_GROUPS]
        tl.store(scale_ptr + flat, scale_bits, mask=group_mask)


def triton_quantize_nvfp4(x: torch.Tensor, per_tensor_scale: torch.Tensor):
    """Fused NVFP4 quantize for a row-padded 2D fp32 tensor.

    Matches nvfp4_utils._quantize_nvfp4_2d for the same (x, per_tensor_scale) to within a
    bounded rounding-tie tolerance at scale (bit-exact for small/edge-case inputs) -- see
    test_nvfp4_kernels.py.

    Args:
        x: fp32 [rows, K]. rows must already be a multiple of 16 and K a multiple of 16
            (both invariants are enforced by the caller, quantize_nvfp4_activation).
        per_tensor_scale: 0-dim fp32 tensor (already computed via torch.amax upstream).

    Returns:
        (packed uint8 [rows, K/2], swizzled block scale float8_e4m3fn).
    """
    rows, k = x.shape
    n_groups = k // NVFP4_BLOCK_SIZE
    n_col_blocks = -(-n_groups // 4)
    n_row_blocks = -(-rows // 128)
    block_groups = triton.next_power_of_2(n_groups)

    packed = torch.empty((rows, n_groups * 8), device=x.device, dtype=torch.uint8)
    scale_bytes = torch.zeros((n_row_blocks * 128, n_col_blocks * 4), device=x.device, dtype=torch.uint8)

    grid = (rows,)
    _quantize_nvfp4_row_kernel[grid](
        x,
        packed,
        scale_bytes,
        per_tensor_scale,
        DENORM_MASK_F32=_E2M1_DENORM_MASK_F32,
        DENORM_MASK_INT=_E2M1_DENORM_MASK_INT,
        ADDER_CONST=_E2M1_ADDER_CONST,
        MAX_INT=_E2M1_MAX_INT,
        SIGN_SHIFT=_E2M1_SIGN_SHIFT,
        K=k,
        n_groups=n_groups,
        n_col_blocks=n_col_blocks,
        BLOCK_GROUPS=block_groups,
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn)


def triton_quantize_nvfp4_stochastic(x: torch.Tensor, per_tensor_scale: torch.Tensor, seed: int):
    """Fused NVFP4 quantize with stochastic-rounding E2M1 conversion for a row-padded 2D fp32
    tensor. Statistically unbiased vs. nvfp4_utils._quantize_nvfp4_2d(magnitude_code_fn=
    _e2m1_stochastic_code) for the same (x, per_tensor_scale) -- not bit-exact by design (both
    draw independent randomness) -- see test_nvfp4_kernels.py.

    Args:
        x: fp32 [rows, K]. rows must already be a multiple of 16 and K a multiple of 16
            (both invariants are enforced by the caller, quantize_nvfp4_activation_stochastic).
        per_tensor_scale: 0-dim fp32 tensor (already computed via torch.amax upstream).
        seed: int, varies per call so consecutive quantizations (e.g. consecutive backward
            steps) don't reuse the same random draws.

    Returns:
        (packed uint8 [rows, K/2], swizzled block scale float8_e4m3fn).
    """
    x = x.contiguous()  # kernel indexes x_ptr + row * K + col, which assumes row-major layout
    rows, k = x.shape
    n_groups = k // NVFP4_BLOCK_SIZE
    n_col_blocks = -(-n_groups // 4)
    n_row_blocks = -(-rows // 128)
    block_groups = triton.next_power_of_2(n_groups)

    packed = torch.empty((rows, n_groups * 8), device=x.device, dtype=torch.uint8)
    scale_bytes = torch.zeros((n_row_blocks * 128, n_col_blocks * 4), device=x.device, dtype=torch.uint8)

    grid = (rows,)
    _quantize_nvfp4_row_kernel_stochastic[grid](
        x,
        packed,
        scale_bytes,
        per_tensor_scale,
        seed,
        K=k,
        n_groups=n_groups,
        n_col_blocks=n_col_blocks,
        BLOCK_GROUPS=block_groups,
    )
    return packed, scale_bytes.view(torch.float8_e4m3fn)
