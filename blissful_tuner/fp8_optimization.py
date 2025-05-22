import torch
import torch.nn as nn
import torch.nn.functional as F

from blissful_tuner.blissful_logger import BlissfulLogger

from tqdm import tqdm

logger = BlissfulLogger(__name__, "#8e00ed")


# based on ComfyUI's and MinusZoneAI's fp8_linear optimization
def fp8_linear_forward(cls, original_dtype, input):
    weight_dtype = cls.weight.dtype
    if weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
        if len(input.shape) == 3:
            target_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
            inn = input.reshape(-1, input.shape[2]).to(target_dtype)
            w = cls.weight.t()

            scale = torch.ones((1), device=input.device, dtype=torch.float32)
            bias = cls.bias.to(original_dtype) if cls.bias is not None else None

            if bias is not None:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, bias=bias, scale_a=scale, scale_b=scale)
            else:
                o = torch._scaled_mm(inn, w, out_dtype=original_dtype, scale_a=scale, scale_b=scale)

            if isinstance(o, tuple):
                o = o[0]

            return o.reshape((-1, input.shape[1], cls.weight.shape[0]))
        else:
            return cls.original_forward(input.to(original_dtype))
    else:
        return cls.original_forward(input)


def convert_fp8_linear(module, original_dtype, params_to_keep={}):
    setattr(module, "fp8_matmul_enabled", True)

    for name, module in module.named_modules():
        if not any(keyword in name for keyword in params_to_keep):
            if isinstance(module, nn.Linear):
                original_forward = module.forward
                setattr(module, "original_forward", original_forward)
                setattr(module, "forward", lambda input, m=module: fp8_linear_forward(m, original_dtype, input))


# Below has been ported from https://github.com/kohya-ss/musubi-tuner/
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
    if exp_bits == 5 and mantissa_bits == 2:
        return 57344
    if exp_bits == 4 and mantissa_bits == 3:
        return 448  # Hard code to save calculations
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
    state_dict,
    calc_device,
    target_layer_keys=None,
    exclude_layer_keys=None,
    exp_bits=4,
    mantissa_bits=3,
    move_to_device=False
):
    """
    Optimize Linear layer weights in a model's state dict to FP8 format

    Args:
        state_dict (dict): State dict to optimize, replaced in-place
        calc_device (str): Device to quantize tensors on
        target_layer_keys (list, optional): Layer key patterns to target (None for all Linear layers)
        exclude_layer_keys (list, optional): Layer key patterns to exclude
        exp_bits (int): Number of exponent bits
        mantissa_bits (int): Number of mantissa bits
        move_to_device (bool): Move optimized tensors to the calculating device

    Returns:
        dict: FP8 optimized state dict with FP8 quantized weights and corresponding scale values
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

    optimized_count = 0
    average_quantization_error = 0.0

    # Find target keys for Linear layer weights
    target_state_dict_keys = []
    for key in state_dict.keys():
        is_target = (target_layer_keys is None or any(pattern in key for pattern in target_layer_keys)) and key.endswith(".weight")
        is_excluded = exclude_layer_keys is not None and any(pattern in key for pattern in exclude_layer_keys)
        is_target = is_target and not is_excluded

        if is_target and isinstance(state_dict[key], torch.Tensor):
            target_state_dict_keys.append(key)

    # Process each target weight tensor
    for key in tqdm(target_state_dict_keys):
        value = state_dict[key]

        # Save original device and dtype
        original_device = value.device
        original_dtype = value.dtype

        # Move to calculation device if provided
        if calc_device is not None:
            value = value.to(calc_device)

        # Calculate scale factor based on the maximum absolute value in the tensor
        scale = torch.max(torch.abs(value.flatten())) / max_value

        # Quantize weight to FP8 format
        quantized_weight, _ = quantize_tensor_to_fp8(value, scale, exp_bits, mantissa_bits, 1, max_value, min_value)

        # Otherwise, store the quantized weight and corresponding scale value.
        fp8_key = key  # Use the original key for the quantized weight
        scale_key = key.replace(".weight", ".scale_weight")

        quantized_weight = quantized_weight.to(fp8_dtype)

        # Reconstruct tensor by scaling back up
        reconstructed = quantized_weight.to(original_dtype) * scale

        # Calculate the mean relative error (in percent)
        average_quantization_error += (torch.mean(torch.abs(value - reconstructed)) / (torch.mean(torch.abs(value)) + 1e-8)) * 100  # Adding a small epsilon to avoid division by zero issues if necessary.

        if not move_to_device:
            quantized_weight = quantized_weight.to(original_device)

        scale_tensor = torch.tensor([scale], dtype=original_dtype, device=quantized_weight.device)

        state_dict[fp8_key] = quantized_weight
        state_dict[scale_key] = scale_tensor

        optimized_count += 1

        # Optionally free memory on the calculation device every 16 optimizations.
        if calc_device is not None and optimized_count % 16 == 0:
            torch.cuda.empty_cache()
    if optimized_count > 0:
        average_quantization_error /= optimized_count
        logger.info(f"Number of optimized Linear layers: {optimized_count}")
        logger.info(f"Mean quantization error: {average_quantization_error:.2f}%")
    else:
        logger.info("optimize_state_dict_with_fp8 didn't optimize any layers! Maybe check your include/exclude keys?")
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
    if use_scaled_mm and x.ndim == 3:
        input_dtype = x.dtype
        original_weight_dtype = self.scale_weight.dtype
        weight_dtype = self.weight.dtype
        assert weight_dtype in [torch.float8_e4m3fn, torch.float8_e5m2], "Only FP8 E4M3FN/E5M2 format is supported"
        target_dtype = torch.float8_e5m2 if weight_dtype == torch.float8_e4m3fn else torch.float8_e4m3fn
        e_bits = 5 if target_dtype == torch.float8_e5m2 else 4
        m_bits = 2 if target_dtype == torch.float8_e5m2 else 3

        if max_value is None:
            # no input quantization
            scale_x = torch.tensor(1.0, dtype=torch.float32, device=x.device)
        else:
            # calculate scale factor for input tensor
            scale_x = (torch.max(torch.abs(x.flatten())) / max_value).to(torch.float32)

            # quantize input tensor to FP8: this seems to consume a lot of memory
            x, _ = quantize_tensor_to_fp8(x, scale_x, e_bits, m_bits, 1, max_value, -max_value)

        original_shape = x.shape
        x = x.reshape(-1, x.shape[2]).to(target_dtype)

        weight = self.weight.t()
        scale_weight = self.scale_weight.to(torch.float32)

        if self.bias is not None:
            # float32 is not supported with bias in scaled_mm
            o = torch._scaled_mm(x, weight, out_dtype=original_weight_dtype, bias=self.bias, scale_a=scale_x, scale_b=scale_weight)
        else:
            o = torch._scaled_mm(x, weight, out_dtype=input_dtype, scale_a=scale_x, scale_b=scale_weight)

        return o.reshape(original_shape[0], original_shape[1], -1)

    else:
        # Dequantize the weight
        original_dtype = self.scale_weight.dtype
        dequantized_weight = self.weight.to(original_dtype) * self.scale_weight

        # Perform linear transformation
        if self.bias is not None:
            output = F.linear(x, dequantized_weight, self.bias)
        else:
            output = F.linear(x, dequantized_weight)

        return output


def apply_fp8_monkey_patch(model, optimized_state_dict, use_scaled_mm=False, scale_input_tensor=None):
    """
    Apply monkey patching to a model using FP8 optimized state dict.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): FP8 optimized state dict
        use_scaled_mm (bool): Use scaled_mm for FP8 Linear layers, requires SM 8.9+ (RTX 40 series)

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    max_value = None
    if use_scaled_mm:
        setattr(model, "fp8_matmul_enabled", True)
        if scale_input_tensor is not None:
            max_value = calculate_fp8_maxval(4, 3) if "e4m3" in scale_input_tensor else calculate_fp8_maxval(5, 2) if "e5m2" in scale_input_tensor else None

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

            # Create a new forward method with the patched version.
            def new_forward(self, x):
                return fp8_linear_forward_patch(self, x, use_scaled_mm, max_value)

            # Bind method to module
            module.forward = new_forward.__get__(module, type(module))

            patched_count += 1

    logger.info(f"Number of monkey-patched Linear layers: {patched_count}")
    return model
