import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from utils.device_utils import clean_memory_on_device


def calculate_fp8_maxval(exp_bits=4, mantissa_bits=3, sign_bits=1):
    """
    Calculate the maximum representable value in FP8 format.
    Default is E4M3 format (4-bit exponent, 3-bit mantissa, 1-bit sign).

    Args:
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        sign_bits (int): Number of sign bits (0 or 1)

    Returns:
        float: Maximum value representable in FP8 format
    """
    assert exp_bits + mantissa_bits + sign_bits == 8, "Total bits must be 8"

    # Calculate exponent bias
    bias = 2 ** (exp_bits - 1) - 1

    # Calculate maximum mantissa value
    mantissa_max = 1.0
    for i in range(mantissa_bits - 1):
        mantissa_max += 2 ** -(i + 1)

    # Calculate maximum value
    max_value = mantissa_max * (2 ** (2**exp_bits - 1 - bias))

    return max_value


def quantize_tensor_to_fp8(tensor, scale, exp_bits=4, mantissa_bits=3, sign_bits=1, max_value=None, min_value=None):
    """
    Quantize a tensor to FP8 format.

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
        quantized_weight, _ = quantize_tensor_to_fp8(value, scale, exp_bits, mantissa_bits, 1, max_value, min_value)

        # Add to state dict using original key for weight and new key for scale
        fp8_key = key  # Maintain original key
        scale_key = key.replace(".weight", ".scale_weight")

        quantized_weight = quantized_weight.to(fp8_dtype)

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


def fp8_linear_forward_patch(self, x):
    """
    Patched forward method for Linear layers with FP8 weights.

    Args:
        self: Linear layer instance
        x (torch.Tensor): Input tensor
        original_forward: Original forward method

    Returns:
        torch.Tensor: Result of linear transformation
    """
    # Dequantize the weight
    original_dtype = self.scale_weight.dtype
    dequantized_weight = self.weight.to(original_dtype) * self.scale_weight

    # Perform linear transformation
    if self.bias is not None:
        output = F.linear(x, dequantized_weight, self.bias)
    else:
        output = F.linear(x, dequantized_weight)

    return output


def apply_fp8_monkey_patch(model, optimized_state_dict):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    # Find all scale keys to identify FP8-optimized layers
    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    # Enumerate patched layers
    patched_module_paths = set()
    for scale_key in scale_keys:
        # Extract module path from scale key (remove .scale_weight)
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)

    patched_count = 0

    # Apply monkey patch to each layer with FP8 weights
    for name, module in model.named_modules():
        # Check if this module has a corresponding scale_weight
        has_scale = name in patched_module_paths

        # Apply patch if it's a Linear layer with FP8 scale
        if isinstance(module, nn.Linear) and has_scale:
            # register the scale_weight as a buffer to load the state_dict
            module.register_buffer("scale_weight", torch.tensor(1.0, dtype=module.weight.dtype))

            # Create a new forward method with the patched version
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x)

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
