import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.utils.device_utils import clean_memory_on_device


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


def quantize_tensor_to_fp8(tensor, scale, exp_bits=4, mantissa_bits=3, sign_bits=1, max_value=None, min_value=None):
    """
    Quantize a tensor to FP8 format.
    **Not used in the current implementation. Kept for reference.**

    Args:
        tensor (torch.Tensor): Tensor to quantize
        scale (float or torch.Tensor): Scale factor
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits

    Returns:
        tuple: (quantized_tensor, scale_factor)
    """
    # Create scaled tensor
    scaled_tensor = tensor / scale

    # Calculate FP8 parameters
    bias = 2 ** (exp_bits - 1) - 1

    if max_value is None:
        # Calculate max and min values
        max_value = calculate_fp8_maxval(exp_bits, mantissa_bits, sign_bits)
        min_value = -max_value if sign_bits > 0 else 0.0

    # Clamp tensor to range
    clamped_tensor = torch.clamp(scaled_tensor, min_value, max_value)

    # Quantization process
    abs_values = torch.abs(clamped_tensor)
    nonzero_mask = abs_values > 0

    # Calculate log scales (only for non-zero elements)
    log_scales = torch.zeros_like(clamped_tensor)
    if nonzero_mask.any():
        log_scales[nonzero_mask] = torch.floor(torch.log2(abs_values[nonzero_mask]) + bias).detach()

    # Limit log scales and calculate quantization factor
    log_scales = torch.clamp(log_scales, min=1.0)
    quant_factor = 2.0 ** (log_scales - mantissa_bits - bias)

    # Quantize and dequantize
    quantized = torch.round(clamped_tensor / quant_factor) * quant_factor

    return quantized, scale


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
    state_dict, calc_device, target_layer_keys=None, exclude_layer_keys=None, exp_bits=4, mantissa_bits=3, move_to_device=False
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format.

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

        # Calculate scale factor
        scale = torch.max(torch.abs(value.flatten())) / max_value
        # print(f"Optimizing {key} with scale: {scale}")

        # Quantize weight to FP8
        # quantized_weight, _ = quantize_tensor_to_fp8(value, scale, exp_bits, mantissa_bits, 1, max_value, min_value)
        quantized_weight = quantize_fp8(value, scale, fp8_dtype, max_value, min_value)

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")

        # quantized_weight = quantized_weight.to(fp8_dtype)

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        scale_tensor = torch.tensor([scale], dtype=original_dtype, device=quantized_weight.device)

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        if calc_device is not None:  # optimized_count % 10 == 0 and
            # free memory on calculation device
            clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


def load_safetensors_with_fp8_optimization(
    model_files: List[str],
    calc_device: Union[str, torch.device],
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False,
    weight_hook=None,
    quantization_mode: str = "tensor",  # "tensor" , "channel", "block"
    block_size: Optional[int] = None,  # used only when quantization_mode is "block"
    percentile: Optional[float] = 0.999,
):
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
        quantization_mode (str): Quantization mode: "tensor" (per-tensor), "channel" (per-output-channel), "block" (block-wise)
        block_size (int, optional): Block size for block-wise quantization (used only when quantization_mode is "block")
        percentile (float, optional): Percentile for scale calculation (None for max value)

    Returns:
        dict: FP8 optimized state dict
    """
    if exp_bits == 4 and mantissa_bits == 3:
        fp8_dtype = torch.float8_e4m3fn
    elif exp_bits == 5 and mantissa_bits == 2:
        fp8_dtype = torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported FP8 format: E{exp_bits}M{mantissa_bits}")
    if block_size is None and quantization_mode == "block":
        block_size = 128  # default block size
    logger.info(
        f"FP8 optimization format: {fp8_dtype}, quantization_mode={quantization_mode}, block_size={block_size}, percentile={percentile}"
    )

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
        with MemoryEfficientSafeOpen(model_file) as f:
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

                # Determine quantization mode
                original_shape = value.shape
                current_quantization_mode = quantization_mode
                if quantization_mode == "block":
                    if value.ndim != 2:
                        current_quantization_mode = "tensor"  # fallback to per-tensor
                    else:
                        out_features, in_features = value.shape
                        if in_features % block_size != 0:
                            current_quantization_mode = "channel"  # fallback to per-channel
                            logger.warning(
                                f"Layer {key} with shape {value.shape} is not divisible by block_size {block_size}, fallback to per-channel quantization."
                            )
                        else:
                            num_blocks = in_features // block_size
                            value = value.contiguous().view(out_features, num_blocks, block_size)  # [out, num_blocks, block_size]
                elif quantization_mode == "channel":
                    if value.ndim != 2:
                        current_quantization_mode = "tensor"  # fallback to per-tensor

                # Save original dtype
                original_dtype = value.dtype

                # Move to calculation device
                if calc_device is not None:
                    value = value.to(calc_device)

                # Calculate scale factor (per-tensor or per-output-channel with percentile or max)
                # value shape is expected to be [out_features, in_features] for Linear weights
                if current_quantization_mode == "channel" or current_quantization_mode == "block":
                    # row-wise percentile to avoid being dominated by outliers
                    # result shape: [out_features, 1] or [out_features, num_blocks, 1]
                    scale_dim = 1 if current_quantization_mode == "channel" else 2
                    abs_w = torch.abs(value)

                    if percentile is None:
                        # shape: [out_features, 1] or [out_features, num_blocks, 1]
                        row_max = torch.max(abs_w, dim=scale_dim, keepdim=True).values
                        scale = row_max / max_value
                    else:
                        # torch.quantile supports keepdim per-dim from PyTorch 2.1+, works only for float/double dtype
                        # channel-wise quantile calculation may not exceed memory limit for quantile calculation, so we do not chunk it here
                        row_q = torch.quantile(abs_w.to(torch.float32), q=percentile, dim=scale_dim, keepdim=True)
                        scale = row_q / max_value

                else:
                    # per-tensor
                    if percentile is None:
                        tensor_max = torch.max(torch.abs(value).view(-1))
                        scale = tensor_max / max_value
                    else:
                        if value.numel() <= 8192 * 4096 // value.dtype.itemsize:  # limit to 16M elements with float16/bfloat16
                            tensor_q = torch.quantile(torch.abs(value).view(-1).to(torch.float32), q=percentile)
                        else:
                            # above code raises error for bigger tensors, use chunked processing instead
                            abs_w = torch.abs(value).view(-1).to(torch.float32)
                            num_chunks = 1
                            while abs_w.numel() / num_chunks > 8192 * 4096 // value.dtype.itemsize:
                                num_chunks *= 2
                            chunked_abs = torch.chunk(abs_w, num_chunks)
                            chunked_q = [torch.quantile(chunk, q=percentile) for chunk in chunked_abs]
                            tensor_q = torch.max(torch.stack(chunked_q, dim=0))

                        scale = tensor_q / max_value

                # numerical safety
                scale = torch.clamp(scale, min=1e-8)
                scale = scale.to(torch.float32)  # ensure scale is in float32 for division

                # Quantize weight to FP8 (scale can be scalar or [out,1], broadcasting works)
                quantized_weight = quantize_fp8(value, scale, fp8_dtype, max_value, min_value)

                # If block-wise, restore original shape
                if current_quantization_mode == "block":
                    quantized_weight = quantized_weight.view(original_shape)  # restore to original shape [out, in]

                # Add to state dict using original key for weight and new key for scale
                fp8_key = key  # Maintain original key
                scale_key = key.replace(".weight", ".scale_weight")
                assert fp8_key != scale_key, "FP8 key and scale key must be different"

                if not move_to_device:
                    quantized_weight = quantized_weight.to(original_device)

                # keep scale shape: [1] or [out,1] or [out, num_blocks, 1]. We can determine the quantization mode from the shape of scale_weight in the patched model.
                scale_tensor = scale.to(dtype=original_dtype, device=quantized_weight.device)

                state_dict[fp8_key] = quantized_weight
                state_dict[scale_key] = scale_tensor
                # print(
                #     f"Optimized {key} with scale shape {scale_tensor.shape}, dtype {scale_tensor.dtype}, scale min {scale_tensor.min().item():.4e}, max {scale_tensor.max().item():.4e}"
                # )

                optimized_count += 1

                if calc_device is not None and optimized_count % 10 == 0:
                    # free memory on calculation device
                    clean_memory_on_device(calc_device)

    logger.info(f"Number of optimized Linear layers: {optimized_count}")
    return state_dict


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
        # scaled_mm causes significant accuracy drop in current test, disable for now
        raise NotImplementedError("use_scaled_mm is not implemented in this patch function.")
        """
        # Kept for reference, not used currently, and not compatible with current quantization function
        input_dtype = x.dtype
        original_weight_dtype = self.scale_weight.dtype
        weight_dtype = self.weight.dtype
        target_dtype = torch.float8_e5m2
        assert weight_dtype == torch.float8_e4m3fn, "Only FP8 E4M3FN format is supported"
        assert x.ndim == 3, "Input tensor must be 3D (batch_size, seq_len, hidden_dim)"

        if max_value is None:
            # no input quantization
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # calculate scale factor for input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # quantize input tensor to FP8: this seems to consume a lot of memory
            x, _ = quantize_tensor_to_fp8(x, scale_x, 5, 2, 1, max_value, -max_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[2]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            # float32 is not supported with bias in scaled_mm
            o = torch._scaled_mm(x, weight, out_dtype=original_weight_dtype, bias=self.bias, scale_a=scale_x, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        return o.reshape(original_shape[0], original_shape[1], -1).to(input_dtype)
        """

    else:
        # Dequantize the weight
        original_dtype = self.scale_weight.dtype
        if self.scale_weight.ndim < 3:
            # per-tensor or per-channel quantization, we can broadcast
            dequantized_weight = self.weight.to(original_dtype) * self.scale_weight
        else:
            # block-wise quantization, need to reshape weight to match scale shape for broadcasting
            out_features, num_blocks, _ = self.scale_weight.shape
            dequantized_weight = self.weight.to(original_dtype).contiguous().view(out_features, num_blocks, -1)
            dequantized_weight = dequantized_weight * self.scale_weight
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

    # Enumerate patched layers
    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

        # Store scale shape information
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

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
