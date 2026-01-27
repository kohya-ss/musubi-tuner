# This file is inspired by the implementation in comfy-kitchen.
# https://github.com/Comfy-Org/comfy-kitchen
# Special thanks to the comfy-kitchen developers.
# Original license: Apache License 2.0

# This code was written by Claude Code (AI agent) and is maintained by the Musubi Tuner developers.

"""
NVFP4 (4-bit floating point) optimization utilities for memory-efficient inference.

This module provides functions for quantizing model weights to NVFP4 format,
which reduces memory usage by approximately 4x compared to FP16.

Requirements:
- PyTorch 2.6+ (for float4_e2m1fn_x2 dtype)
- PyTorch 2.10+ recommended (for scaled_mm API support)
- Blackwell or newer GPU for hardware acceleration (older GPUs use dequantize fallback)

Note: This module is for inference only. Backward pass is not supported.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)


# region Constants and LUT


def _n_ones(n: int) -> int:
    return (1 << n) - 1


# FP32 format constants
EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

# NVFP4 (E2M1) format constants
F4_E2M1_MAX = 6.0
F4_E2M1_EPS = 0.5

# FP8 (E4M3) format constants for block scales
F8_E4M3_MAX = 448.0
F8_E4M3_EPS = 0.125

# Block size for NVFP4 quantization
BLOCK_SIZE = 16

# E2M1 lookup table for dequantization
# Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (positive)
#        -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0 (negative)
E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]).unsqueeze(1)

# Cache for E2M1 LUT on different devices/dtypes
E2M1_LUT_CACHE = {}

# endregion


# region Version Check


def check_nvfp4_support() -> Tuple[bool, bool, str]:
    """Check if NVFP4 is supported in the current PyTorch version.

    Returns:
        Tuple[bool, bool, str]: (has_dtype, has_scaled_mm, message)
            - has_dtype: True if float4_e2m1fn_x2 dtype is available (PyTorch 2.6+)
            - has_scaled_mm: True if scaled_mm with NVFP4 is available (PyTorch 2.10+)
            - message: Description of the support status
    """
    has_dtype = False
    has_scaled_mm = False
    messages = []

    # Check for float4_e2m1fn_x2 dtype (PyTorch 2.6+)
    if hasattr(torch, "float4_e2m1fn_x2"):
        has_dtype = True
        messages.append("float4_e2m1fn_x2 dtype available")
    else:
        messages.append("float4_e2m1fn_x2 dtype NOT available (requires PyTorch 2.6+)")

    # Check for scaled_mm function (Pytorch 2.10+)
    if hasattr(torch.nn.functional, "scaled_mm"):
        has_scaled_mm = True
        messages.append("scaled_mm function available")
    else:
        messages.append("scaled_mm function NOT available (requires PyTorch 2.10+)")

    # Kept for reference, but commented out:
    # # Check for scaled_mm with NVFP4 support (PyTorch 2.10+)
    # # This is harder to check without actually calling it, so we check version
    # version_parts = torch.__version__.split(".")
    # try:
    #     major = int(version_parts[0])
    #     minor = int(version_parts[1].split("+")[0].split("a")[0].split("b")[0].split("rc")[0])
    #     if major > 2 or (major == 2 and minor >= 10):
    #         has_scaled_mm = True
    #         messages.append("scaled_mm with NVFP4 likely available")
    #     else:
    #         messages.append("scaled_mm with NVFP4 NOT available (requires PyTorch 2.10+)")
    # except (ValueError, IndexError):
    #     messages.append("Could not determine PyTorch version for scaled_mm check")

    return has_dtype, has_scaled_mm, "; ".join(messages)


def get_nvfp4_compute_mode() -> str:
    """Get the compute mode for NVFP4 operations.

    Returns:
        str: "scaled_mm" if hardware-accelerated path is available, otherwise "dequantize"
    """
    has_dtype, has_scaled_mm, _ = check_nvfp4_support()

    if not has_dtype:
        return "unsupported"

    if has_scaled_mm:
        return "scaled_mm"

    return "dequantize"


# endregion


# region Low-level Functions


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 numbers to sub-byte floating point numbers.

    Args:
        x: Input tensor of dtype torch.float
        ebits: Number of exponent bits
        mbits: Number of mantissa bits

    Returns:
        torch.Tensor of dtype torch.uint8 with the encoding in least significant bits
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    # Calculate constants
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    # All E bits and M bits are 1s
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    # E bits = 1, M bits = 0
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (
        # exp bias conversion between formats
        (F32_EXP_BIAS - exp_bias)
        # mantissa length difference between formats
        + (MBITS_F32 - mbits)
        # add one to encoded exponent for denormalized numbers
        + 1
    )
    denorm_mask_int = denorm_exp << MBITS_F32

    # Reinterpret int32 as float32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    # Save the sign
    x = x.view(torch.int32)
    sign = x & 0x80000000

    # Set everything to positive
    x = x ^ sign

    # Convert to float
    x = x.view(torch.float)

    # Create masks for different cases
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    # Branch 2: denormal conversion
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Branch 3: normal range
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    # Combine branches
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # Add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _down_size(size):
    """Calculate the size after packing two uint4 values into one uint8."""
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)


def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack two uint4 values into one uint8.

    Args:
        uint8_data: Tensor with uint4 values stored in uint8 (one value per byte)

    Returns:
        Packed tensor with two uint4 values per byte
    """
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(_down_size(shape))


def _ceil_div(a, b):
    """Ceiling division."""
    return (a + b - 1) // b


def roundup(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple


def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """Rearrange a matrix into cuBLAS tiled layout for NVFP4 block scales.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        flatten: If True, return flattened tensor

    Returns:
        Rearranged tensor
    """
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)

    # Calculate padded shape
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix

    # Rearrange the blocks
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    if flatten:
        return rearranged.flatten()

    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Reverse the cuBLAS tiled layout back to normal (H, W) layout.

    Args:
        blocked_matrix: Swizzled tensor from cuBLAS layout
        num_rows: Desired output rows (unpadded)
        num_cols: Desired output cols (unpadded)

    Returns:
        Unswizzled tensor of shape (num_rows, num_cols)
    """
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


def _float8_round(x: torch.Tensor) -> torch.Tensor:
    """Round to FP8 E4M3 precision."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)


# endregion


# region Quantization Parameters


@dataclass
class NvFp4Params:
    """Parameters for NVFP4 quantized tensor.

    Attributes:
        scale: Per-tensor scale factor (float32)
        block_scale: Per-block scale factors in swizzled format (float8_e4m3fn)
        orig_dtype: Original tensor dtype before quantization
        orig_shape: Original tensor shape before quantization
    """

    scale: torch.Tensor
    block_scale: torch.Tensor
    orig_dtype: torch.dtype
    orig_shape: Tuple[int, ...]

    def __post_init__(self):
        if isinstance(self.scale, torch.Tensor):
            self.scale = self.scale.to(dtype=torch.float32, non_blocking=True)

    def to_device(self, device: torch.device) -> "NvFp4Params":
        """Move all tensors to the specified device."""
        return NvFp4Params(
            scale=self.scale.to(device=device),
            block_scale=self.block_scale.to(device=device),
            orig_dtype=self.orig_dtype,
            orig_shape=self.orig_shape,
        )

    def clone(self) -> "NvFp4Params":
        """Clone all tensors."""
        return NvFp4Params(
            scale=self.scale.clone(), block_scale=self.block_scale.clone(), orig_dtype=self.orig_dtype, orig_shape=self.orig_shape
        )


def get_padded_shape(orig_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Get padded shape for NVFP4 quantization (must be multiple of 16)."""
    if len(orig_shape) != 2:
        raise ValueError(f"NVFP4 requires 2D shape, got {len(orig_shape)}D")
    rows, cols = orig_shape
    return (roundup(rows, 16), roundup(cols, 16))


# endregion


# region Quantization Functions


def quantize_nvfp4_core(
    x: torch.Tensor, per_tensor_scale: torch.Tensor, pad_16x: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Core NVFP4 quantization function.

    Args:
        x: Input tensor to quantize (2D)
        per_tensor_scale: Per-tensor scale factor
        pad_16x: Whether to pad to 16x multiple

    Returns:
        Tuple of (packed_data, blocked_scales)
    """
    orig_shape = x.shape

    # Handle padding
    if pad_16x:
        rows, cols = x.shape
        padded_rows = roundup(rows, 16)
        padded_cols = roundup(cols, 16)
        if padded_rows != rows or padded_cols != cols:
            x = torch.nn.functional.pad(x, (0, padded_cols - cols, 0, padded_rows - rows))
            orig_shape = x.shape

    block_size = BLOCK_SIZE

    x = x.reshape(orig_shape[0], -1, block_size)
    max_abs = torch.amax(torch.abs(x), dim=-1)
    block_scale = max_abs.to(torch.float32) / F4_E2M1_MAX
    scaled_block_scales = block_scale / per_tensor_scale
    scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
    scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
    total_scale = per_tensor_scale * scaled_block_scales_fp32

    # Handle zero blocks (from padding): avoid 0/0 NaN
    zero_scale_mask = total_scale == 0
    total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)

    data_scaled = x.float() / total_scale_safe.unsqueeze(-1)
    data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)

    out_scales = scaled_block_scales_fp8

    data_scaled = torch.clamp(data_scaled, -F4_E2M1_MAX, F4_E2M1_MAX)
    data_scaled = data_scaled.view(orig_shape)

    data_lp = _f32_to_floatx_unpacked(data_scaled, 2, 1)
    data_lp = pack_uint4(data_lp)
    blocked_scales = to_blocked(out_scales.to(torch.float8_e4m3fn), flatten=False)

    return data_lp, blocked_scales


# Create compiled version for faster input quantization
_quantize_nvfp4_compiled = None


def get_compiled_quantize_fn():
    """Get or create the compiled quantization function."""
    global _quantize_nvfp4_compiled
    if _quantize_nvfp4_compiled is None:
        _quantize_nvfp4_compiled = torch.compile(quantize_nvfp4_core)
    return _quantize_nvfp4_compiled


def quantize_nvfp4_weight(tensor: torch.Tensor) -> Tuple[torch.Tensor, NvFp4Params]:
    """Quantize a weight tensor to NVFP4 format.

    This function is used for weight quantization (pre-processing).

    Args:
        tensor: Weight tensor to quantize (must be 2D)

    Returns:
        Tuple of (quantized_data, params)
    """
    if tensor.dim() != 2:
        raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

    orig_dtype = tensor.dtype
    orig_shape = tuple(tensor.shape)

    # scale calculation: scale is always recalculated for weights
    scale = torch.amax(tensor.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)

    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale)
    scale = scale.to(device=tensor.device, dtype=torch.float32)

    padded_shape = get_padded_shape(orig_shape)
    needs_padding = padded_shape != orig_shape

    # Use non-compiled version for weight quantization: one-time operation
    qdata, block_scale = quantize_nvfp4_core(tensor, scale, pad_16x=needs_padding)

    params = NvFp4Params(scale=scale, block_scale=block_scale, orig_dtype=orig_dtype, orig_shape=orig_shape)
    return qdata, params


def quantize_nvfp4_input(tensor: torch.Tensor, use_compile: bool = True) -> Tuple[torch.Tensor, NvFp4Params]:
    """Quantize an input tensor to NVFP4 format.

    This function is used for input quantization (runtime).
    Uses torch.compile for better performance when use_compile=True.

    Args:
        tensor: Input tensor to quantize (must be 2D)
        use_compile: Whether to use torch.compile for faster quantization

    Returns:
        Tuple of (quantized_data, params)
    """
    if tensor.dim() != 2:
        raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

    orig_dtype = tensor.dtype
    orig_shape = tuple(tensor.shape)

    # scale calculation: scale is always recalculated for inputs
    scale = torch.amax(tensor.abs()) / (F8_E4M3_MAX * F4_E2M1_MAX)

    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale)
    scale = scale.to(device=tensor.device, dtype=torch.float32)

    padded_shape = get_padded_shape(orig_shape)
    needs_padding = padded_shape != orig_shape

    if use_compile:
        quantize_fn = get_compiled_quantize_fn()
    else:
        quantize_fn = quantize_nvfp4_core

    qdata, block_scale = quantize_fn(tensor, scale, pad_16x=needs_padding)

    params = NvFp4Params(scale=scale, block_scale=block_scale, orig_dtype=orig_dtype, orig_shape=orig_shape)
    return qdata, params


# endregion


# region Dequantization


def dequantize_nvfp4(
    qx: torch.Tensor,
    params: NvFp4Params,
    output_type: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize NVFP4 data back to floating point.

    Args:
        qx: Quantized data (packed uint8)
        params: Quantization parameters
        output_type: Output dtype

    Returns:
        Dequantized tensor
    """
    per_tensor_scale = params.scale
    block_scales = params.block_scale

    lut = E2M1_LUT_CACHE.get((qx.device, output_type))
    if lut is None:
        lut = E2M1_LUT.to(qx.device, output_type)
        E2M1_LUT_CACHE[(qx.device, output_type)] = lut

    lo = qx & 0x0F
    hi = qx >> 4
    out = torch.stack([hi, lo], dim=-1).view(*qx.shape[:-1], -1)
    out = torch.nn.functional.embedding(out.int(), lut).squeeze(-1)

    # Get original shape (packed tensor has half the columns)
    orig_shape = out.shape
    block_size = BLOCK_SIZE

    # Reshape to blocks for scaling
    out = out.reshape(orig_shape[0], -1, block_size)

    # Unswizzle block_scales from cuBLAS tiled layout
    num_blocks_per_row = orig_shape[1] // block_size

    # Use from_blocked to unswizzle the tiled layout
    block_scales_unswizzled = from_blocked(block_scales, num_rows=orig_shape[0], num_cols=num_blocks_per_row)

    # Compute total decode scale: per_tensor_scale * block_scale_fp8
    total_scale = per_tensor_scale.to(output_type) * block_scales_unswizzled.to(output_type)

    # Apply scaling to dequantize
    data_dequantized = out * total_scale.unsqueeze(-1)

    # Reshape back to original shape and convert to output type
    result = data_dequantized.view(orig_shape).to(output_type)

    return result


# endregion


# region Linear Forward Functions

warn_float32_output_logged = False


def nvfp4_linear_forward_scaled_mm(
    input_qdata: torch.Tensor,
    input_params: NvFp4Params,
    weight_qdata: torch.Tensor,
    weight_params: NvFp4Params,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """NVFP4 linear using scaled_mm (hardware-accelerated path).

    Args:
        input_qdata: Quantized input data
        input_params: Input quantization parameters
        weight_qdata: Quantized weight data
        weight_params: Weight quantization parameters
        bias: Optional bias tensor
        out_dtype: Output dtype

    Returns:
        Result of linear transformation
    """
    from torch.nn.functional import ScalingType, SwizzleType

    global warn_float32_output_logged
    if out_dtype == torch.float32 and warn_float32_output_logged is False:
        logger.warning("Output dtype is float32. This may be an internal bug, as NVFP4 is typically used with bfloat16 or float16.")
        warn_float32_output_logged = True

    scale_a, block_scale_a = input_params.scale, input_params.block_scale
    scale_b, block_scale_b = weight_params.scale, weight_params.block_scale

    result = torch.nn.functional.scaled_mm(
        input_qdata.view(torch.float4_e2m1fn_x2),
        weight_qdata.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[block_scale_a.view(-1), scale_a],
        scale_b=[block_scale_b.view(-1), scale_b],
        bias=bias,
        output_dtype=out_dtype,
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    )

    # Slice output to original (non-padded) shape
    orig_m = input_params.orig_shape[0]
    orig_n = weight_params.orig_shape[0]  # weight is (out_features, in_features)
    return result[:orig_m, :orig_n]


def nvfp4_linear_forward_dequantize(
    x: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_params: NvFp4Params,
    bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """NVFP4 linear using dequantization fallback.

    This path is used when scaled_mm is not available.

    Args:
        x: Input tensor (not quantized)
        weight_qdata: Quantized weight data
        weight_params: Weight quantization parameters
        bias: Optional bias tensor

    Returns:
        Result of linear transformation
    """
    # Dequantize weight
    dequantized_weight = dequantize_nvfp4(weight_qdata, weight_params, x.dtype)

    # Slice to original shape (remove padding)
    orig_rows, orig_cols = weight_params.orig_shape
    dequantized_weight = dequantized_weight[:orig_rows, :orig_cols]

    # Standard linear
    return torch.nn.functional.linear(x, dequantized_weight, bias)


# endregion


# region State Dict Operations


def validate_weight_for_nvfp4(key: str, tensor: torch.Tensor) -> bool:
    """Validate if a weight tensor can be quantized to NVFP4.

    Args:
        key: Weight key name
        tensor: Weight tensor

    Returns:
        True if the tensor can be quantized
    """
    # Only 2D Linear weights
    if tensor.ndim != 2:
        return False

    # Skip already quantized weights (1-byte dtypes like fp8, int8)
    if tensor.dtype.itemsize == 1:
        return False

    return True


def load_safetensors_with_nvfp4_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[List[str]] = None,
    exclude_layer_keys: Optional[List[str]] = None,
    move_to_device: bool = False,
    weight_hook=None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict:
    """Load weight tensors from safetensors files with NVFP4 optimization.

    Args:
        model_files: List of model files to load
        calc_device: Device to perform quantization on
        target_layer_keys: Layer key patterns to target
        exclude_layer_keys: Layer key patterns to exclude
        move_to_device: Whether to keep optimized tensors on calc_device
        weight_hook: Optional function to apply to each weight tensor
        disable_numpy_memmap: Disable numpy memmap when loading
        weight_transform_hooks: Hooks for weight transformation during loading

    Returns:
        NVFP4 optimized state dict
    """
    logger.info(
        f"Loading state dict with NVFP4 optimization. Hook enabled: {weight_hook is not None}"
    )

    def is_target_key(key):
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        return is_target and not is_excluded

    optimized_count = 0
    state_dict = {}

    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)
                original_device = value.device

                if weight_hook is not None:
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                if not is_target_key(key) or not validate_weight_for_nvfp4(key, value):
                    target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    value = value.to(target_device)
                    state_dict[key] = value
                    continue

                # Move to calculation device
                if calc_device is not None:
                    value = value.to(calc_device)

                original_dtype = value.dtype
                if original_dtype.itemsize == 1:
                    raise ValueError(
                        f"Layer {key} is already in {original_dtype} format. NVFP4 optimization should not be applied."
                    )

                # Quantize weight
                quantized_weight, params = quantize_nvfp4_weight(value)

                # Keys for state dict
                weight_key = key
                scale_key = key.replace(".weight", ".nvfp4_scale")
                block_scale_key = key.replace(".weight", ".nvfp4_block_scale")
                orig_shape_key = key.replace(".weight", ".nvfp4_orig_shape")

                target_device = calc_device if move_to_device else original_device

                state_dict[weight_key] = quantized_weight.to(target_device)
                state_dict[scale_key] = params.scale.to(device=target_device)
                state_dict[block_scale_key] = params.block_scale.to(device=target_device)
                state_dict[orig_shape_key] = torch.tensor(params.orig_shape, dtype=torch.int64, device=target_device)

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of NVFP4 optimized Linear layers: {optimized_count}")
    return state_dict


# endregion


# region Monkey Patch


def nvfp4_linear_forward_patch(
    self: nn.Linear, x: torch.Tensor, use_scaled_mm: bool = False, use_torch_compile: bool = True
) -> torch.Tensor:
    """Patched forward method for Linear layers with NVFP4 weights.

    Args:
        self: Linear layer instance
        x: Input tensor
        use_scaled_mm: Whether to use scaled_mm (requires PyTorch 2.10+)
        use_torch_compile: Whether to use torch.compile for input quantization

    Returns:
        Result of linear transformation
    """
    # Reconstruct NvFp4Params from stored tensors
    weight_params = NvFp4Params(
        scale=self.nvfp4_scale,
        block_scale=self.nvfp4_block_scale,
        orig_dtype=x.dtype,
        orig_shape=tuple(self.nvfp4_orig_shape.tolist()),
    )

    if use_scaled_mm:
        # Quantize input
        input_dtype = x.dtype
        original_shape = x.shape

        # Flatten to 2D for quantization
        x_2d = x.reshape(-1, x.shape[-1])

        input_qdata, input_params = quantize_nvfp4_input(x_2d, use_compile=use_torch_compile)

        # Perform NVFP4 matmul
        result = nvfp4_linear_forward_scaled_mm(input_qdata, input_params, self.weight, weight_params, self.bias, input_dtype)

        # Reshape back to original batch dimensions
        if len(original_shape) > 2:
            result = result.reshape(*original_shape[:-1], -1)

        return result
    else:
        # Dequantize fallback
        return nvfp4_linear_forward_dequantize(x, self.weight, weight_params, self.bias)


def apply_nvfp4_monkey_patch(
    model: nn.Module,
    optimized_state_dict: dict,
    use_scaled_mm: bool = False,
    use_torch_compile: bool = True,
) -> nn.Module:
    """Apply monkey patching to a model using NVFP4 optimized state dict.

    Args:
        model: Model instance to patch
        optimized_state_dict: NVFP4 optimized state dict
        use_scaled_mm: Whether to use scaled_mm for NVFP4 Linear layers
        use_torch_compile: Whether to use torch.compile for input quantization

    Returns:
        The patched model (same instance, modified in-place)
    """
    # Find all scale keys to identify NVFP4-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".nvfp4_scale")]

    # Enumerate patched layers
    patched_module_paths = set()
    for scale_key in scale_keys:
        module_path = scale_key.rsplit(".nvfp4_scale", 1)[0]
        patched_module_paths.add(module_path)

    patched_count = 0

    # Apply monkey patch to each layer with NVFP4 weights
    for name, module in model.named_modules():
        has_scale = name in patched_module_paths

        if isinstance(module, nn.Linear) and has_scale:
            # Register buffers for NVFP4 parameters
            scale_key = f"{name}.nvfp4_scale"
            block_scale_key = f"{name}.nvfp4_block_scale"
            orig_shape_key = f"{name}.nvfp4_orig_shape"

            # Register buffers
            module.register_buffer("nvfp4_scale", optimized_state_dict[scale_key])
            module.register_buffer("nvfp4_block_scale", optimized_state_dict[block_scale_key])
            module.register_buffer("nvfp4_orig_shape", optimized_state_dict[orig_shape_key])

            # Replace weight with quantized shape
            module.weight = nn.Parameter(optimized_state_dict[f"{name}.weight"], requires_grad=False)

            # Create patched forward method
            def new_forward(self, x, _use_scaled_mm=use_scaled_mm, _use_torch_compile=use_torch_compile):
                return nvfp4_linear_forward_patch(self, x, _use_scaled_mm, _use_torch_compile)

            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of NVFP4 monkey-patched Linear layers: {patched_count}")
    return model


# endregion

