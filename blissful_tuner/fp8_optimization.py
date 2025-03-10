#based on ComfyUI's and MinusZoneAI's fp8_linear optimization
#further borrowed from HunyuanVideoWrapper for Musubi Tuner
import torch
import torch.nn as nn

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
