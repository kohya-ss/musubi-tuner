"""Fused Triton kernel for NVFP4 activation quantization.

Collapses nvfp4_utils._quantize_nvfp4_2d's five separate eager passes (per-block amax,
F8_E4M3 scale cast, elementwise normalize+clamp, E2M1 bit conversion, nibble packing) plus
the to_blocked swizzle permute into a single kernel launch, one Triton program per row. See
docs/superpowers/specs/2026-09-01-nvfp4-fused-activation-quant-kernel-design.md.

The E2M1 bit-conversion constants below are ported from nvfp4_utils._f32_to_e2m1_unpacked
with ebits=2, mbits=1 fixed (E2M1 is the only format this kernel handles), verified
bit-exact against that function by test_nvfp4_kernels.py::test_e2m1_code_matches_reference.
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
    def _e2m1_magnitude_code(x_pos, DENORM_MASK_F32: tl.constexpr, DENORM_MASK_INT: tl.constexpr,
                              ADDER_CONST: tl.constexpr, MAX_INT: tl.constexpr):
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
    def _e2m1_code(x, DENORM_MASK_F32: tl.constexpr, DENORM_MASK_INT: tl.constexpr,
                    ADDER_CONST: tl.constexpr, MAX_INT: tl.constexpr, SIGN_SHIFT: tl.constexpr):
        """x: signed fp32 tensor. Returns uint8 E2M1 code 0..15 (sign in bit 3)."""
        x_bits = x.to(tl.int32, bitcast=True)
        sign_bit = x_bits & tl.full(x_bits.shape, -2147483648, tl.int32)
        x_pos_bits = x_bits ^ sign_bit
        x_pos = x_pos_bits.to(tl.float32, bitcast=True)
        mag_code = _e2m1_magnitude_code(x_pos, DENORM_MASK_F32, DENORM_MASK_INT, ADDER_CONST, MAX_INT)
        sign_code = ((sign_bit >> SIGN_SHIFT) & 8).to(tl.uint8)
        return mag_code | sign_code

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

        # Swizzled (cuBLAS 128x4 tiled) scale write: closed-form index derived from
        # nvfp4_utils.to_blocked's view/permute/reshape chain, verified in
        # test_nvfp4_swizzle_index.py. Writing here directly avoids a separate to_blocked
        # permute pass over the scale tensor.
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

    Bit-exact with nvfp4_utils._quantize_nvfp4_2d for the same (x, per_tensor_scale) --
    see test_nvfp4_kernels.py.

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
