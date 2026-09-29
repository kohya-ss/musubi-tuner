"""FP8 weight conversion for Qwen-Image 2.1."""

from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F


def _linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))


def store_linears_in_fp8(module: nn.Module) -> None:
    """Store Linear weights in FP8 and cast to the input dtype during forward.

    Norms, embeddings, and biases keep their dtype. F.linear preserves input
    gradients so LoRA layers before these frozen layers can be trained.
    """
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            layer.weight = nn.Parameter(layer.weight.to(torch.float8_e4m3fn), requires_grad=False)
            layer.forward = MethodType(_linear_forward, layer)
