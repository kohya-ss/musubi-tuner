# LoHa (Low-rank Hadamard Product) network module
# Linear layers only (no Conv2d/Tucker decomposition)
# Reference: https://arxiv.org/abs/2108.06098
#
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

import ast
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lora as lora_module
from .network_arch import detect_arch_config

logger = logging.getLogger(__name__)


class HadaWeight(torch.autograd.Function):
    """Efficient Hadamard product forward/backward for LoHa.

    Computes ((w1a @ w1b) * (w2a @ w2b)) * scale with custom backward
    that recomputes intermediates instead of storing them.
    """

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=None):
        if scale is None:
            scale = torch.tensor(1, device=w1a.device, dtype=w1a.dtype)
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class LoHaModule(torch.nn.Module):
    """LoHa module for training. Replaces forward method of the original Linear."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("LoHa Conv2d is not supported in this implementation")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # Hadamard product parameters: ΔW = (w1a @ w1b) * (w2a @ w2b)
        self.hada_w1_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, in_dim))
        self.hada_w2_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, in_dim))

        # Initialization: w1_a normal(0.1), w1_b normal(1.0), w2_a = 0, w2_b normal(1.0)
        # Ensures ΔW = 0 at init since w2_a = 0
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1.0)
        torch.nn.init.constant_(self.hada_w2_a, 0)
        torch.nn.init.normal_(self.hada_w2_b, std=1.0)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def get_diff_weight(self):
        """Return materialized weight delta."""
        scale = torch.tensor(self.scale, dtype=self.hada_w1_a.dtype, device=self.hada_w1_a.device)
        return HadaWeight.apply(self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b, scale)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()

        # rank dropout
        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            drop = drop.view(-1, 1)
            diff_weight = diff_weight * drop
            # scaling for rank dropout
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale


class LoHaInfModule(LoHaModule):
    """LoHa module for inference. Supports merge_to and get_weight."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: lora_module.LoRANetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        # get LoHa weights
        w1a = sd["hada_w1_a"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w1b = sd["hada_w1_b"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w2a = sd["hada_w2_a"].to(device, dtype=torch.float, non_blocking=non_blocking)
        w2b = sd["hada_w2_b"].to(device, dtype=torch.float, non_blocking=non_blocking)

        # compute ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * self.scale

        # merge
        weight = weight + self.multiplier * diff_weight

        org_sd["weight"] = weight.to(org_device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        w1a = self.hada_w1_a.to(torch.float)
        w1b = self.hada_w1_b.to(torch.float)
        w2a = self.hada_w2_a.to(torch.float)
        w2b = self.hada_w2_b.to(torch.float)

        weight = ((w1a @ w1b) * (w2a @ w2b)) * self.scale * multiplier
        return weight

    def default_forward(self, x):
        diff_weight = self.get_diff_weight()
        return self.org_forward(x) + F.linear(x, diff_weight) * self.multiplier

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create a LoHa network with auto-detected architecture."""
    target_replace_modules, default_excludes = detect_arch_config(unet)

    # merge user exclude_patterns with defaults
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
    exclude_patterns.extend(default_excludes)
    kwargs["exclude_patterns"] = exclude_patterns

    return lora_module.create_network(
        target_replace_modules,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        module_class=LoHaModule,
        **kwargs,
    )


def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora_module.LoRANetwork:
    """Create LoHa network from saved weights (internal)."""
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "hada_w1_b" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim

    module_class = LoHaInfModule if for_inference else LoHaModule

    network = lora_module.LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
    )
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora_module.LoRANetwork:
    """Create LoHa network from saved weights with auto-detected architecture."""
    target_replace_modules, _ = detect_arch_config(unet)
    return create_network_from_weights(target_replace_modules, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs)


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoHa weights directly into a model weight tensor.

    No Module/Network creation needed. Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoHa keys found.
    """
    w1a_key = lora_name + ".hada_w1_a"
    w1b_key = lora_name + ".hada_w1_b"
    w2a_key = lora_name + ".hada_w2_a"
    w2b_key = lora_name + ".hada_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1a_key not in lora_weight_keys:
        return model_weight

    w1a = lora_sd[w1a_key].to(calc_device)
    w1b = lora_sd[w1b_key].to(calc_device)
    w2a = lora_sd[w2a_key].to(calc_device)
    w2b = lora_sd[w2b_key].to(calc_device)

    dim = w1b.shape[0]
    alpha = lora_sd.get(alpha_key, torch.tensor(dim))
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / dim

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        w1a, w1b, w2a, w2b = w1a.to(torch.float16), w1b.to(torch.float16), w2a.to(torch.float16), w2b.to(torch.float16)

    # ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
    diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
    model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    # remove consumed keys
    for key in [w1a_key, w1b_key, w2a_key, w2b_key, alpha_key]:
        lora_weight_keys.discard(key)

    return model_weight
