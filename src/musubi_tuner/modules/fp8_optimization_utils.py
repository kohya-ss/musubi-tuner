import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.utils.device_utils import clean_memory_on_device


warn_float32_output_logged = False


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """Rearrange a matrix into cuBLAS tiled layout for SWIZZLE_32_4_4 scales."""
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

    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    Calculate the maximum representable value in FP8 format.
    Default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign). Only supports E4M3 and E5M2 with sign bit.

    Args:
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits (0 or 1)

    Returns:
        float: Maximum value representable in FP8 format
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"
    if exp_bits == 4 and mantissa_bits == 3 and sign_bits == 1:
        return torch.finfo(torch.float8_e4m3fn).max
    elif exp_bits == 5 and mantissa_bits == 2 and sign_bits == 1:
        return torch.finfo(torch.float8_e5m2).max
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits} with sign_bits={sign_bits}")


# The following is a manual calculation method (wrong implementation for E5M2), kept for reference.
"""
# Calculate exponent bias
bias = 2 ** (exp_bits - 1) - 1

# Calculate maximum mantissa value
mantissa_max = 1.0
for i in range(mantissa_bits - 1):
    mantissa_max += 2 ** -(i + 1)

# Calculate maximum value
max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

return max_value
"""


def quantize_fp8(tensor, scale, fp8_dtype, max_value, min_value):
    """
    Quantize a tensor to FP8 format using PyTorch's native FP8 dtype support.

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        fp8_dtype (torch.dtype): Target FP8 dtype (torch.float8_e4m3fn or torch.float8_e5m2)
        max_value (float): Maximum representable value in FP8
        min_value (float): Minimum representable value in FP8

    Returns:
        torch.Tensor: Quantized tensor in FP8 format
    """
    tensor = tensor.to(torch.float32)  # ensure tensor is in float32 for division

    # Create scaled tensor
    tensor = torch.div(tensor, scale).nan_to_num_(0.0)  # handle NaN values, equivalent to nonzero_mask in previous function

    # Clamp tensor to range
    tensor = tensor.clamp_(min=min_value, max=max_value)

    # Convert to FP8 dtype
    tensor = tensor.to(fp8_dtype)

    return tensor


def optimize_state_dict_with_fp8(
    state_dict: dict,
    calc_device: Union[str, torch.device],
    target_layer_keys: Optional[list[str]] = None,
    exclude_layer_keys: Optional[list[str]] = None,
    exp_bits: int = 4,
    mantissa_bits: int = 3,
    move_to_device: bool = False,
    quantization_mode: str = "block",
    block_size: Optional[int] = 64,
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format. The state dict is modified in-place.
    This function is a static version of load_safetensors_with_fp8_optimization without loading from files.

    Args:
        state_dict (dict): State dict to optimize, replaced in-place
        calc_device (str): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Create optimized state dict
    optimized_count = 0

    # Enumerate tarket keys
    target_state_dict_keys = []
    for key in state_dict.keys():
        # Check if it's a weight key and matches target patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    # Process each key
    for key in tqdm(target_state_dict_keys):
        value = state_dict[key]

        # Save original device and dtype
        original_device = value.device
        original_dtype = value.dtype

        # Move to calculation device
        if calc_device is not None:
            value = value.to(calc_device)

        quantized_weight, scale_tensor, tensor_scale = quantize_weight(
            key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
        )

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")
        scale_tensor_key = key.replace(".weight", ".scale_weight_tensor")

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
        scale_tensor = scale_tensor.to(dtype=torch.float32, device=quantized_weight.device)

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor
        if tensor_scale is not None:
            state_dict[scale_tensor_key] = tensor_scale.to(dtype=torch.float32, device=quantized_weight.device)

        optimized_count += 1

        if calc_device is not None:  # optimized_count % 10 == 0 and
            # free memory on calculation device
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def quantize_weight(
    key: str,
    tensor: torch.Tensor,
    fp8_dtype: torch.dtype,
    max_value: float,
    min_value: float,
    quantization_mode: str = "block",
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    original_shape = tensor.shape

    # Determine quantization mode
    if quantization_mode == "block":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor
        else:
            out_features, in_features = tensor.shape
            if in_features % block_size != 0:
                quantization_mode = "tensor"  # fallback to per-tensor
                logger.warning(
                    f"Layer {key} with shape {tensor.shape} is not divisible by block_size {block_size}, fallback to per-tensor quantization."
                )
            else:
                num_blocks = in_features // block_size
                tensor = tensor.contiguous().view(out_features, num_blocks, block_size)  # [out, num_blocks, block_size]
    elif quantization_mode == "channel":
        if tensor.ndim != 2:
            quantization_mode = "tensor"  # fallback to per-tensor

    # Calculate scale factor (per-tensor or per-output-channel with percentile or max)
    # value shape is expected to be [out_features, in_features] for Linear weights
    if quantization_mode == "channel" or quantization_mode == "block":
        # row-wise percentile to avoid being dominated by outliers
        # result shape: [out_features, 1] or [out_features, num_blocks, 1]
        scale_dim = 1 if quantization_mode == "channel" else 2
        abs_w = torch.abs(tensor)

        # shape: [out_features, 1] or [out_features, num_blocks, 1]
        row_max = torch.max(abs_w, dim=scale_dim, keepdim=True).values
        scale = row_max / max_value

        # TensorWise + BlockWise 2-level scaling for better scaled_mm block-wise accuracy.
        # total_scale = tensor_scale * block_scale
        if quantization_mode == "block":
            tensor_scale = torch.max(scale)
            tensor_scale = torch.clamp(tensor_scale, min=1e-8).to(torch.float32)
            scale = scale / tensor_scale

            # # BlockWise1x32 with swizzle expects e8m0fnu scale on many backends.
            # if block_size == 32:
            #     scale = scale.to(torch.float8_e8m0fnu).to(torch.float32)
            scale = torch.clamp(scale, min=1e-8)
            total_scale = scale * tensor_scale
        else:
            tensor_scale = None
            total_scale = scale

    else:
        # per-tensor
        tensor_max = torch.max(torch.abs(tensor).view(-1))
        scale = tensor_max / max_value
        tensor_scale = None
        total_scale = scale

    # print(f"Optimizing {key} with scale: {scale}")

    # numerical safety
    scale = torch.clamp(scale, min=1e-8)
    scale = scale.to(torch.float32)  # ensure scale is in float32 for division
    total_scale = torch.clamp(total_scale, min=1e-8).to(torch.float32)

    # Quantize weight to FP8 (scale can be scalar or [out,1], broadcasting works)
    quantized_weight = quantize_fp8(tensor, total_scale, fp8_dtype, max_value, min_value)

    # If block-wise, restore original shape
    if quantization_mode == "block":
        quantized_weight = quantized_weight.view(original_shape)  # restore to original shape [out, in]

    return quantized_weight, scale, tensor_scale


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    fp8_fast_quantization_mode: Optional[str] = None,
    block_size: Optional[int] = None,
    disable_numpy_memmap: bool = False,
    weight_transform_hooks: Optional[WeightTransformHooks] = None,
) -> dict:
    """
    Load weight tensors from safetensors files and merge LoRA weights into the state dict with explicit FP8 optimization.

    Args:
        model_files (list[str]): List of model files to load
        calc_device (str or torch.device): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target for optimization (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude from optimization
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device
        weight_hook (callable, optional): Function to apply to each weight tensor before optimization
        fp8_fast_quantization_mode (str, optional): Quantization mode, "tensor", "channel", or "block". If None, defaults to "block" for 2D weights with block size 64 and "tensor" for others.
        block_size (int, optional): Block size for block-wise quantization (used if quantization_mode is "block"). If None, defaults to 64 for non fp8_fast_quantization_mode, 16 for fp8_fast_quantization_mode.
        disable_numpy_memmap (bool): Disable numpy memmap when loading safetensors
        weight_transform_hooks (WeightTransformHooks, optional): Hooks for weight transformation during loading

    Returns:
        dict: FP8 optimized state dict
    """
    if block_size is None:
        if fp8_fast_quantization_mode is None:
            block_size = 64
        else:
            block_size = 16
    quantization_mode = fp8_fast_quantization_mode or ("block" if block_size and block_size > 1 else "tensor")

    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")

    # Calculate FP8 max value
    max_value = calculate_fp8_maxval(exp_bits, mantissa_bits)
    min_value = -max_value  # this function supports only signed FP8

    # Define function to determine if a key is a target key. target means fp8 optimization, not for weight hook.
    def is_target_key(key):
        # Check if weight key matches target patterns and does not match exclude patterns
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        return is_target and not is_excluded

    # Create optimized state dict
    optimized_count = 0

    # Process each file
    state_dict = {}
    for model_file in model_files:
        with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
            f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

            keys = f.keys()
            for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                value = f.get_tensor(key)

                # Save original device
                original_device = value.device  # usually cpu

                if weight_hook is not None:
                    # Apply weight hook if provided
                    value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                if not is_target_key(key):
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
                        f"Layer {key} is already in {original_dtype} format. `--fp8_scaled` optimization should not be applied. Please use fp16/bf16/float32 model weights."
                        + f" / レイヤー {key} は既に{original_dtype}形式です。`--fp8_scaled` 最適化は適用できません。FP16/BF16/Float32のモデル重みを使用してください。"
                    )
                quantized_weight, scale_tensor, tensor_scale = quantize_weight(
                    key, value, fp8_dtype, max_value, min_value, quantization_mode, block_size
                )

                # Add to state dict using original key for weight and new key for scale
                fp8_key = key  # Maintain original key
                scale_key = key.replace(".weight", ".scale_weight")
                scale_tensor_key = key.replace(".weight", ".scale_weight_tensor")
                assert fp8_key != scale_key, "FP8 key and scale key must be different"

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
                scale_tensor = scale_tensor.to(dtype=torch.float32, device=quantized_weight.device)

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor
                if tensor_scale is not None:
                    state_dict[scale_tensor_key] = tensor_scale.to(dtype=torch.float32, device=quantized_weight.device)

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    # free memory on calculation device
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def _get_fp8_blockwise_scaling_type(scaling_type_enum, block_size: Optional[int] = None):
    """Resolve a block-wise scaling enum value supported by the current PyTorch."""
    candidates = [
        ("BlockWise", None),
        ("BlockWise1x128", 128),
        ("BlockWise1x64", 64),
        ("BlockWise1x32", 32),
        ("BlockWise1x16", 16),
    ]

    if block_size is not None:
        # Prefer exact fixed-size match first.
        for name, size in candidates:
            if size is not None and size == block_size and hasattr(scaling_type_enum, name):
                return getattr(scaling_type_enum, name), size

    # Then prefer generic BlockWise.
    if hasattr(scaling_type_enum, "BlockWise"):
        return getattr(scaling_type_enum, "BlockWise"), None

    # Fallback to any available fixed-size variant.
    for name, size in candidates:
        if size is not None and hasattr(scaling_type_enum, name):
            return getattr(scaling_type_enum, name), size

    return None, None


def fp8_linear_forward_scaled_mm(
    input_qdata: torch.Tensor,
    input_scale: torch.Tensor,
    weight_qdata: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    input_tensor_scale: Optional[torch.Tensor] = None,
    weight_tensor_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 linear using scaled_mm (hardware-accelerated path).

    Args:
        input_qdata: Quantized input data
        input_scale: Input scale tensor
        weight_qdata: Quantized weight data
        weight_scale: Weight scale tensor
        bias: Optional bias tensor
        out_dtype: Output dtype

    Returns:
        Result of linear transformation
    """
    from torch.nn.functional import ScalingType, SwizzleType

    global warn_float32_output_logged
    if out_dtype == torch.float32 and warn_float32_output_logged is False:
        logger.warning("Output dtype is float32. This may be an internal bug, as FP8 is typically used with bfloat16 or float16.")
        warn_float32_output_logged = True

    if input_qdata.ndim != 2:
        raise ValueError(f"input_qdata must be 2D for scaled_mm, got shape {tuple(input_qdata.shape)}")

    if weight_qdata.ndim != 2:
        raise ValueError(f"weight_qdata must be 2D for scaled_mm, got shape {tuple(weight_qdata.shape)}")

    out_features, _ = weight_qdata.shape

    # BlockWise scaling for mat_a (M, K): scale_a shape should be (M, num_blocks)
    # and for mat_b (K, N): scale_b shape should be (num_blocks, N).
    if input_scale.ndim == 2 and weight_scale.ndim == 3 and weight_scale.shape[0] == out_features and weight_scale.shape[2] == 1:
        num_blocks = weight_scale.shape[1]
        if input_qdata.shape[1] % num_blocks != 0:
            raise ValueError(
                f"input hidden dimension ({input_qdata.shape[1]}) must be divisible by num_blocks ({num_blocks}) for block-wise scaled_mm."
            )
        block_size = input_qdata.shape[1] // num_blocks
        if input_scale.shape[0] != input_qdata.shape[0] or input_scale.shape[1] != num_blocks:
            raise ValueError(
                "Block-wise scaled_mm requires input_scale shape (M, num_blocks). "
                f"Got input_scale={tuple(input_scale.shape)}, expected M={input_qdata.shape[0]}, num_blocks={num_blocks}."
            )

        blockwise_scaling_type, required_block_size = _get_fp8_blockwise_scaling_type(ScalingType, block_size=block_size)
        if blockwise_scaling_type is None:
            raise ValueError("This PyTorch build does not expose block-wise ScalingType for scaled_mm.")
        if required_block_size is not None and required_block_size != block_size:
            raise ValueError(
                f"scaled_mm block-wise ScalingType expects block size {required_block_size}, but got {block_size}. "
                "Adjust quantization block_size to match the backend requirement."
            )

        scale_a_matrix = input_scale
        scale_b_matrix = weight_scale
        swizzle_type = None
        if block_size == 32:
            # BlockWise1x32 commonly requires e8m0fnu scales with SWIZZLE_32_4_4.
            scale_a_matrix = scale_a_matrix * input_tensor_scale if input_tensor_scale is not None else scale_a_matrix
            scale_b_matrix = scale_b_matrix * weight_tensor_scale if weight_tensor_scale is not None else scale_b_matrix
            scale_a_dtype = torch.float8_e8m0fnu
            scale_b_dtype = torch.float8_e8m0fnu
            swizzle_type = SwizzleType.SWIZZLE_32_4_4
            input_tensor_scale = None
            weight_tensor_scale = None
        else:
            scale_a_dtype = torch.float32
            scale_b_dtype = torch.float32
        scale_a_matrix = scale_a_matrix.to(dtype=scale_a_dtype, device=input_qdata.device).contiguous()
        scale_b_matrix = scale_b_matrix.to(dtype=scale_b_dtype, device=weight_qdata.device).squeeze(2).contiguous()
        mat_b = weight_qdata.contiguous().t()

        use_two_level_scaling = input_tensor_scale is not None and weight_tensor_scale is not None

        if swizzle_type is not None:
            scale_a_block = to_blocked(scale_a_matrix, flatten=True).contiguous()
            scale_b_block = to_blocked(scale_b_matrix, flatten=True).contiguous()
        else:
            scale_a_block = scale_a_matrix
            scale_b_block = scale_b_matrix

        if use_two_level_scaling:
            scale_a_tensor = input_tensor_scale.to(dtype=torch.float32, device=input_qdata.device).reshape(1).contiguous()
            scale_b_tensor = weight_tensor_scale.to(dtype=torch.float32, device=weight_qdata.device).reshape(1).contiguous()
            scale_a = [scale_a_block, scale_a_tensor]
            scale_b = [scale_b_block, scale_b_tensor]
            scale_recipe_a = [blockwise_scaling_type, ScalingType.TensorWise]
            scale_recipe_b = [blockwise_scaling_type, ScalingType.TensorWise]
            swizzle_a = [swizzle_type, SwizzleType.NO_SWIZZLE] if swizzle_type is not None else None
            swizzle_b = [swizzle_type, SwizzleType.NO_SWIZZLE] if swizzle_type is not None else None
        else:
            scale_a = scale_a_block
            scale_b = scale_b_block
            scale_recipe_a = blockwise_scaling_type
            scale_recipe_b = blockwise_scaling_type
            swizzle_a = swizzle_type
            swizzle_b = swizzle_type

        # print(f"input_qdata shape: {input_qdata.shape}, weight_qdata shape: {weight_qdata.shape}")
        # print(
        #     f"scale_a shape: {scale_a.shape}, scale_b shape: {scale_b.shape}, scale_recipe_a: {scale_recipe_a}, scale_recipe_b: {scale_recipe_b}, swizzle_a: {swizzle_a}, swizzle_b: {swizzle_b}"
        # )
        # print(f"scale_a dtype: {scale_a.dtype}, scale_b dtype: {scale_b.dtype}")
        result = torch.nn.functional.scaled_mm(
            input_qdata,
            mat_b,
            scale_a=scale_a,
            scale_recipe_a=scale_recipe_a,
            scale_b=scale_b,
            scale_recipe_b=scale_recipe_b,
            swizzle_a=swizzle_a,
            swizzle_b=swizzle_b,
            bias=bias,
            output_dtype=out_dtype,
        )
    # RowWise scaling for mat_a (M, K): scale_a shape must be (M, 1)
    elif input_scale.ndim == 2 and input_scale.shape[0] == input_qdata.shape[0]:
        # RowWise scaling for mat_b (K, N): scale_b shape must be (1, N)
        if weight_scale.numel() == 1:
            scale_b = (
                weight_scale.to(dtype=torch.float32, device=weight_qdata.device).reshape(1, 1).expand(1, out_features).contiguous()
            )
        elif weight_scale.ndim == 1 and weight_scale.shape[0] == out_features:
            scale_b = weight_scale.to(dtype=torch.float32, device=weight_qdata.device).reshape(1, out_features).contiguous()
        elif weight_scale.ndim == 2 and weight_scale.shape[0] == out_features and weight_scale.shape[1] == 1:
            scale_b = weight_scale.to(dtype=torch.float32, device=weight_qdata.device).transpose(0, 1).contiguous()
        else:
            raise ValueError(
                "scaled_mm (RowWise) currently supports scale_weight shapes [1], [out_features], or [out_features, 1]. "
                f"Got {tuple(weight_scale.shape)}. Please quantize weights in tensor mode when using --use_scaled_mm RowWise."
            )

        scale_a = input_scale.to(dtype=torch.float32, device=input_qdata.device).contiguous()

        # F.scaled_mm expects mat_b in column-major layout for cuBLASLt kernels.
        mat_b = weight_qdata.contiguous().t()

        result = torch.nn.functional.scaled_mm(
            input_qdata,
            mat_b,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.RowWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.RowWise,
            swizzle_a=None,
            swizzle_b=None,
            bias=bias,
            output_dtype=out_dtype,
        )
    else:
        # Tensor scaling for mat_a (M, K): scale_a shape must be (1,) and mat_b (K, N): scale_b shape must be (1,)
        if input_scale.numel() != 1:
            raise ValueError(f"input_scale must be a scalar for Tensor scaled_mm. Got shape {tuple(input_scale.shape)}")
        if weight_scale.numel() != 1:
            raise ValueError(f"weight_scale must be a scalar for Tensor scaled_mm. Got shape {tuple(weight_scale.shape)}")

        scale_a = input_scale.to(dtype=torch.float32, device=input_qdata.device).reshape(1).contiguous()
        scale_b = weight_scale.to(dtype=torch.float32, device=weight_qdata.device).reshape(1).contiguous()
        mat_b = weight_qdata.contiguous().t()
        result = torch.nn.functional.scaled_mm(
            input_qdata,
            mat_b,
            scale_a=scale_a,
            scale_recipe_a=ScalingType.TensorWise,
            scale_b=scale_b,
            scale_recipe_b=ScalingType.TensorWise,
            swizzle_a=None,
            swizzle_b=None,
            bias=bias,
            output_dtype=out_dtype,
        )
    return result


def quantize_fp8_for_scaled_mm(
    x: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    mode: str = "tensor",
    num_blocks: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Quantize input activations for `torch.nn.functional.scaled_mm` Tensor/Block mode.

    This produces:
      - quantized activation: shape (M, K), dtype fp8
      - scale_a:
        - Tensor mode: shape (1,)
        - Block mode: shape (M, num_blocks)
      - tensor_scale:
        - Tensor mode: None
        - Block mode: shape (1,)

    where M is the flattened batch/token dimension and K is hidden size.
    """
    if x.ndim < 2:
        raise ValueError(f"Input must be at least 2D, got shape {tuple(x.shape)}")

    if fp8_dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        raise ValueError(f"Unsupported fp8_dtype for scaled_mm: {fp8_dtype}")

    hidden_size = x.shape[-1]
    block_size = None

    x_2d = x.reshape(-1, hidden_size).contiguous()

    fp8_max = torch.finfo(fp8_dtype).max
    fp8_min = -fp8_max

    if mode == "block":
        if num_blocks is None or num_blocks <= 0:
            raise ValueError(f"num_blocks must be a positive integer for block mode. Got: {num_blocks}")
        if hidden_size % num_blocks != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by num_blocks ({num_blocks}) for block mode activation quantization."
            )
        block_size = hidden_size // num_blocks

        x_blocks = x_2d.contiguous().view(x_2d.shape[0], num_blocks, block_size)
        block_absmax = torch.max(torch.abs(x_blocks), dim=2, keepdim=True).values
        block_scale = torch.clamp(block_absmax / fp8_max, min=1e-8).to(dtype=torch.float32)

        if block_size != 32:
            tensor_scale = torch.max(block_scale)
            tensor_scale = torch.clamp(tensor_scale, min=1e-8).to(dtype=torch.float32)
            block_scale = block_scale / tensor_scale
            block_scale = torch.clamp(block_scale, min=1e-8)
        else:
            block_scale = block_scale.to(torch.float8_e8m0fnu).to(torch.float32)
            tensor_scale = torch.tensor(1.0, dtype=torch.float32, device=block_scale.device)

        total_scale = block_scale * tensor_scale
        qdata = quantize_fp8(x_blocks, total_scale, fp8_dtype, fp8_max, fp8_min).contiguous().view_as(x_2d)
        return qdata, block_scale.squeeze(2).contiguous(), tensor_scale.reshape(1).contiguous()

    elif mode == "channel":
        row_absmax = torch.max(torch.abs(x_2d), dim=1, keepdim=True).values
        scale = torch.clamp(row_absmax / fp8_max, min=1e-8).to(dtype=torch.float32)
        qdata = quantize_fp8(x_2d, scale, fp8_dtype, fp8_max, fp8_min).contiguous()
        return qdata, scale.contiguous(), None

    elif mode == "tensor":
        absmax = torch.max(torch.abs(x_2d))
        scale = torch.clamp(absmax / fp8_max, min=1e-8).to(dtype=torch.float32)
        qdata = quantize_fp8(x_2d, scale, fp8_dtype, fp8_max, fp8_min).contiguous()
        return qdata, scale.contiguous(), None
    else:
        raise ValueError(f"Unsupported quantization mode for scaled_mm activation quantization: {mode}")

    # not reached


def fp8_linear_forward_patch(self: nn.Linear, x, use_scaled_mm=False, max_value=None):
    """
    Patched forward method for Linear layers with FP8 weights.

    Args:
        self: Linear layer instance
        x (torch.Tensor): Input tensor
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)
        max_value (float): Maximum value for FP8 quantization. If None, no quantization is applied for input tensor.

    Returns:
        torch.Tensor: Result of linear transformation
    """
    if use_scaled_mm:
        # Quantize input
        input_dtype = x.dtype
        original_shape = x.shape

        # Flatten to 2D for quantization
        input_tensor_scale = None
        if self.scale_weight.numel() == 1:
            input_qdata, input_scale, input_tensor_scale = quantize_fp8_for_scaled_mm(x, mode="tensor")
        elif self.scale_weight.ndim == 3 and self.scale_weight.shape[2] == 1:
            input_qdata, input_scale, input_tensor_scale = quantize_fp8_for_scaled_mm(
                x,
                mode="block",
                num_blocks=self.scale_weight.shape[1],
            )
        elif self.scale_weight.ndim == 2 and self.scale_weight.shape[1] == 1:
            input_qdata, input_scale, input_tensor_scale = quantize_fp8_for_scaled_mm(x, mode="channel")
        else:
            raise ValueError(
                f"Unsupported scale_weight shape for scaled_mm: {tuple(self.scale_weight.shape)}. "
                "Expected scalar for tensor mode or [out_features, num_blocks, 1] for block mode."
            )

        weight_tensor_scale = getattr(self, "scale_weight_tensor", None)
        if (
            self.scale_weight.ndim == 3
            and self.scale_weight.shape[2] == 1
            and weight_tensor_scale is None
            and input_tensor_scale is not None
        ):
            # Backward compatibility: older checkpoints may only store total block scales.
            # Convert input scales from 2-level representation to a single total block scale.
            input_scale = input_scale * input_tensor_scale.to(dtype=input_scale.dtype, device=input_scale.device)
            input_tensor_scale = None

        # Perform fast FP8 matmul
        result = fp8_linear_forward_scaled_mm(
            input_qdata,
            input_scale,
            self.weight,
            self.scale_weight,
            self.bias,
            input_dtype,
            input_tensor_scale=input_tensor_scale,
            weight_tensor_scale=weight_tensor_scale,
        )

        # Reshape back to original batch dimensions
        if len(original_shape) > 2:
            result = result.reshape(*original_shape[:-1], -1)

        return result

        """ Old version using torch._scaled_mm, which seems to have some issues with bias and per-channel scale. Keeping it here for reference and future debugging.
        # **not tested**
        # _scaled_mm only works for per-tensor scale for now (per-channel scale does not work in certain cases)
        if self.scale_weight.ndim != 1:
            raise ValueError("scaled_mm only supports per-tensor scale_weight for now.")

        input_dtype = x.dtype
        original_weight_dtype = self.scale_weight.dtype
        target_dtype = self.weight.dtype
        # assert x.ndim == 3, "Input tensor must be 3D (batch_size, seq_len, hidden_dim)"

        if max_value is None:
            # no input quantization
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # calculate scale factor for input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # quantize input tensor to FP8: this seems to consume a lot of memory
            fp8_max_value = torch.finfo(target_dtype).max
            fp8_min_value = torch.finfo(target_dtype).min
            x = quantize_fp8(x, scale_x, target_dtype, fp8_max_value, fp8_min_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            # float32 is not supported with bias in scaled_mm
            o = torch._scaled_mm(x, weight, out_dtype=original_weight_dtype, bias=self.bias, scale_a=scale_x, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        o = o.reshape(original_shape[0], original_shape[1], -1) if len(original_shape) == 3 else o.reshape(original_shape[0], -1)
        return o.to(input_dtype)
        """

    else:
        # Dequantize the weight
        original_dtype = self.scale_weight.dtype
        weight_tensor_scale = getattr(self, "scale_weight_tensor", None)
        if self.scale_weight.ndim < 3:
            # per-tensor or per-channel quantization, we can broadcast
            dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
        else:
            # block-wise quantization, need to reshape weight to match scale shape for broadcasting
            out_features, num_blocks, _ = self.scale_weight.shape
            dequantized_weight = self.weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
            dequantized_weight = dequantized_weight * self.scale_weight
            if weight_tensor_scale is not None:
                dequantized_weight = dequantized_weight * weight_tensor_scale.to(dtype=dequantized_weight.dtype)
            dequantized_weight = dequantized_weight.view(self.weight.shape)

        # Perform linear transformation
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias)
        else:
            output = F.linear(x, dequantized_weight)

        return output


def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    # # Calculate FP8 float8_e5m2 max value
    # max_value = calculate_fp8_maxval(5, 2)
    max_value = None  # do not quantize input tensor

    # Find all scale keys to identify FP8-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]
    scale_tensor_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight_tensor")]

    # Enumerate patched layers
    patched_module_paths = set()
    scale_shape_info = {}
    scale_tensor_shape_info = {}
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

        # Store scale shape information
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape
    for scale_tensor_key in scale_tensor_keys:
        module_path = scale_tensor_key.rsplit(".scale_weight_tensor", 1)[0]
        scale_tensor_shape_info[module_path] = optimized_state_dict[scale_tensor_key].shape

    patched_count = 0

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # register the scale_weight as a buffer to load the state_dict
            # module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))
            scale_shape = scale_shape_info[name]
            module.register_buffer("scale_weight", torch.ones(scale_shape, dtype=module.weight.dtype))
            if name in scale_tensor_shape_info:
                scale_tensor_shape = scale_tensor_shape_info[name]
                module.register_buffer("scale_weight_tensor", torch.ones(scale_tensor_shape, dtype=module.weight.dtype))

            # Create a new forward method with the patched version.
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model


# Example usage
def example_usage():
    # Test scaled_mm
    from torch.nn.functional import ScalingType, SwizzleType

    block_size = 32
    input_data = torch.randn(8192, 3072, dtype=torch.float32, device="cuda") * 0.1
    weight_data = torch.randn(3072, 3072, dtype=torch.float32, device="cuda") * 0.1

    input_qdata, scale_a, tensor_scale_a, _ = quantize_fp8_for_scaled_mm(
        input_data, mode="block", num_blocks=input_data.shape[1] // block_size
    )
    weight_qdata, scale_b, tensor_scale_b, _ = quantize_fp8_for_scaled_mm(
        weight_data, mode="block", num_blocks=weight_data.shape[1] // block_size
    )
    scale_a = scale_a * tensor_scale_a
    scale_b = scale_b * tensor_scale_b

    scale_a = to_blocked(scale_a.to(dtype=torch.float8_e8m0fnu)).contiguous()
    scale_b = to_blocked(scale_b.to(dtype=torch.float8_e8m0fnu)).contiguous()
    # scale_a = to_blocked(scale_a, flatten=True).to(dtype=torch.float8_e8m0fnu).contiguous()
    # scale_b = to_blocked(scale_b, flatten=True).to(dtype=torch.float8_e8m0fnu).contiguous()
    # scale_a = scale_a.to(dtype=torch.float8_e8m0fnu).view(-1).contiguous()
    # scale_b = scale_b.to(dtype=torch.float8_e8m0fnu).view(-1).contiguous()
    mat_b = weight_qdata.contiguous().t()
    bias = torch.randn(3072, dtype=torch.float16, device="cuda")
    out_dtype = torch.float16

    print(f"input_qdata shape: {input_qdata.shape}, dtype: {input_qdata.dtype}")
    print(f"weight_qdata shape: {weight_qdata.shape}, dtype: {weight_qdata.dtype}")
    print(f"scale_a shape: {scale_a.shape}, dtype: {scale_a.dtype}")
    print(f"scale_b shape: {scale_b.shape}, dtype: {scale_b.dtype}")

    result_scaled_mm = torch.nn.functional.scaled_mm(
        input_qdata,
        mat_b,
        scale_a=scale_a,  # , tensor_scale_a.to(dtype=torch.float32)],
        scale_recipe_a=ScalingType.BlockWise1x32,  # , ScalingType.TensorWise],
        scale_b=scale_b,  # , tensor_scale_b.to(dtype=torch.float32)],
        scale_recipe_b=ScalingType.BlockWise1x32,  # , ScalingType.TensorWise],
        swizzle_a=SwizzleType.SWIZZLE_32_4_4,  # , SwizzleType.NO_SWIZZLE],
        swizzle_b=SwizzleType.SWIZZLE_32_4_4,  # , SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=out_dtype,
    )
    print("scaled_mm result shape:", result_scaled_mm.shape)

    # Compare with regular matmul
    # expected_result = F.linear(input_data, weight_data, bias.to(input_data.dtype)).to(out_dtype).to(torch.float32)
    expected_result = torch.matmul(input_data, weight_data.t()) + bias
    expected_result = expected_result.to(out_dtype).to(torch.float32)
    error = torch.mean(torch.abs(result_scaled_mm.to(torch.float32) - expected_result))
    print(f"Mean absolute error between scaled_mm and expected result: {error.item()}")

    import sys

    sys.exit(0)

    # Small test model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            fc1 = nn.Linear(768, 3072)
            act1 = nn.GELU()
            fc2 = nn.Linear(3072, 768)
            act2 = nn.GELU()
            fc3 = nn.Linear(768, 768)

            # Set layer names for testing
            self.single_blocks = nn.ModuleList([fc1, act1, fc2, act2, fc3])

            self.fc4 = nn.Linear(768, 128)

        def forward(self, x):
            for layer in self.single_blocks:
                x = layer(x)
            x = self.fc4(x)
            return x

    # Instantiate model
    test_model = TestModel()
    test_model.to(torch.float16)  # convert to FP16 for testing

    # Test input tensor
    test_input = torch.randn(1, 768, dtype=torch.float16)

    # Calculate output before optimization
    with torch.no_grad():
        original_output = test_model(test_input)
        print("original output", original_output[0, :5])

    # Get state dict
    state_dict = test_model.state_dict()

    # Apply FP8 optimization to state dict
    cuda_device = torch.device("cuda")
    optimized_state_dict = optimize_state_dict_with_fp8(state_dict, cuda_device, ["single_blocks"], ["2"])

    # Apply monkey patching to the model
    optimized_model = TestModel()  # re-instantiate model
    optimized_model.to(torch.float16)  # convert to FP16 for testing
    apply_fp8_monkey_patch(optimized_model, optimized_state_dict)

    # Load optimized state dict
    optimized_model.load_state_dict(optimized_state_dict, strict=True, assign=True)  # assign=True to load buffer

    # Calculate output after optimization
    with torch.no_grad():
        optimized_output = optimized_model(test_input)
        print("optimized output", optimized_output[0, :5])

    # Compare accuracy
    error = torch.mean(torch.abs(original_output - optimized_output))
    print(f"Mean absolute error: {error.item()}")

    # Check memory usage
    original_params = sum(p.nelement() * p.element_size() for p in test_model.parameters()) / (1024 * 1024)
    print(f"Model parameter memory: {original_params:.2f} MB")
    optimized_params = sum(p.nelement() * p.element_size() for p in optimized_model.parameters()) / (1024 * 1024)
    print(f"Optimized model parameter memory: {optimized_params:.2f} MB")

    return test_model


if __name__ == "__main__":
    example_usage()
