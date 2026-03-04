# LoKr (Low-rank Kronecker Product) network module
# Linear layers only (no Conv2d/Tucker decomposition)
# Reference: https://arxiv.org/abs/2309.14859
#
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

import ast
import logging
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import lora as lora_module
from .network_arch import detect_arch_config

logger = logging.getLogger(__name__)


def factorization(dimension: int, factor: int = -1) -> tuple:
    """Return a tuple of two values whose product equals dimension,
    optimized for balanced factors.

    In LoKr, the first value is for the weight scale (smaller),
    and the second value is for the weight (larger).

    Examples:
        factor=-1: 128 -> (8, 16), 512 -> (16, 32), 1024 -> (32, 32)
        factor=4:  128 -> (4, 32), 512 -> (4, 128)
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


def make_kron(w1, w2, scale):
    """Compute Kronecker product of w1 and w2, scaled by scale."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


class LoKrModule(torch.nn.Module):
    """LoKr module for training. Replaces forward method of the original Linear."""

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
        factor=-1,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            raise ValueError("LoKr Conv2d is not supported in this implementation")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        factor = int(factor)
        self.use_w2 = False

        # Factorize dimensions
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # w1 is always a full matrix (the "scale" factor, small)
        self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))

        # w2: low-rank decomposition if rank is small enough, otherwise full matrix
        if lora_dim < max(out_k, in_n) / 2:
            self.lokr_w2_a = nn.Parameter(torch.empty(out_k, lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
        else:
            self.use_w2 = True
            self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n))
            if lora_dim >= max(out_k, in_n) / 2:
                logger.warning(
                    f"LoKr: lora_dim {lora_dim} is large for dim={max(in_dim, out_dim)} "
                    f"and factor={factor}, using full matrix mode."
                )

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        # if both w1 and w2 are full matrices, use scale = 1
        if self.use_w2:
            alpha = lora_dim
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialization
        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)
        # Ensures ΔW = kron(w1, 0) = 0 at init

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
        w1 = self.lokr_w1
        if self.use_w2:
            w2 = self.lokr_w2
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b
        return make_kron(w1, w2, self.scale)

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
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale


class LoKrInfModule(LoKrModule):
    """LoKr module for inference. Supports merge_to and get_weight."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference; pass factor from kwargs if present
        factor = kwargs.pop("factor", -1)
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, factor=factor)

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

        # get LoKr weights
        w1 = sd["lokr_w1"].to(device, dtype=torch.float, non_blocking=non_blocking)

        if "lokr_w2" in sd:
            w2 = sd["lokr_w2"].to(device, dtype=torch.float, non_blocking=non_blocking)
        else:
            w2a = sd["lokr_w2_a"].to(device, dtype=torch.float, non_blocking=non_blocking)
            w2b = sd["lokr_w2_b"].to(device, dtype=torch.float, non_blocking=non_blocking)
            w2 = w2a @ w2b

        # compute ΔW via Kronecker product
        diff_weight = make_kron(w1, w2, self.scale)

        # merge
        weight = weight + self.multiplier * diff_weight

        org_sd["weight"] = weight.to(org_device, dtype=dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        w1 = self.lokr_w1.to(torch.float)
        if self.use_w2:
            w2 = self.lokr_w2.to(torch.float)
        else:
            w2 = (self.lokr_w2_a @ self.lokr_w2_b).to(torch.float)

        weight = make_kron(w1, w2, self.scale) * multiplier
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
    """Create a LoKr network with auto-detected architecture."""
    target_replace_modules, default_excludes = detect_arch_config(unet)

    # merge user exclude_patterns with defaults
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
    exclude_patterns.extend(default_excludes)
    kwargs["exclude_patterns"] = exclude_patterns

    # extract factor from kwargs
    factor = kwargs.pop("factor", -1)
    factor = int(factor)

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
        module_class=LoKrModule,
        module_kwargs={"factor": factor},
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
    """Create LoKr network from saved weights (internal)."""
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lokr_w2_a" in key:
            # low-rank mode: dim = w2_a.shape[1]
            dim = value.shape[1]
            modules_dim[lora_name] = dim
        elif "lokr_w2" in key and "lokr_w2_a" not in key and "lokr_w2_b" not in key:
            # full matrix mode: set dim large enough to trigger full-matrix path
            if lora_name not in modules_dim:
                modules_dim[lora_name] = max(value.shape)

    # extract factor for LoKr (user must specify via --network_args factor=N if different from default)
    factor = int(kwargs.pop("factor", -1))

    module_class = LoKrInfModule if for_inference else LoKrModule
    module_kwargs = {"factor": factor}

    network = lora_module.LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
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
    """Create LoKr network from saved weights with auto-detected architecture."""
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
    """Merge LoKr weights directly into a model weight tensor.

    No Module/Network creation needed. Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoKr keys found.
    """
    w1_key = lora_name + ".lokr_w1"
    w2_key = lora_name + ".lokr_w2"
    w2a_key = lora_name + ".lokr_w2_a"
    w2b_key = lora_name + ".lokr_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1_key not in lora_weight_keys:
        return model_weight

    w1 = lora_sd[w1_key].to(calc_device)

    # determine low-rank vs full matrix mode
    if w2a_key in lora_weight_keys:
        # low-rank: w2 = w2_a @ w2_b
        w2a = lora_sd[w2a_key].to(calc_device)
        w2b = lora_sd[w2b_key].to(calc_device)
        dim = w2a.shape[1]
        consumed_keys = [w1_key, w2a_key, w2b_key, alpha_key]
    elif w2_key in lora_weight_keys:
        # full matrix mode
        w2a = None
        w2b = None
        dim = None  # will use scale=1.0
        consumed_keys = [w1_key, w2_key, alpha_key]
    else:
        return model_weight

    alpha = lora_sd.get(alpha_key, None)
    if alpha is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    # compute scale
    if w2a is not None:
        # low-rank mode
        if alpha is None:
            alpha = dim
        scale = alpha / dim
    else:
        # full matrix mode: scale = 1.0
        scale = 1.0

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        w1 = w1.to(torch.float16)
        if w2a is not None:
            w2a, w2b = w2a.to(torch.float16), w2b.to(torch.float16)

    # compute w2
    if w2a is not None:
        w2 = w2a @ w2b
    else:
        w2 = lora_sd[w2_key].to(calc_device)
        if original_dtype.itemsize == 1:
            w2 = w2.to(torch.float16)

    # ΔW = kron(w1, w2) * scale
    diff_weight = make_kron(w1, w2, scale)
    model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    # remove consumed keys
    for key in consumed_keys:
        lora_weight_keys.discard(key)

    return model_weight
