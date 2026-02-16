# NOTE: LoKr reuses LoRANetwork from lora.py via module_class/module_kwargs injection.
# LoHa (loha.py) uses its own LoHaNetwork class for historical reasons (Conv2d support).
# This asymmetry is tracked for future convergence (Slice 8).

"""Native LoKr (Low-Rank Kronecker Product) network module.

LoKr v1 is Linear-only. Conv2d candidates are skipped with a counted warning.
Factor persistence uses dual state-dict buffer + safetensors metadata.
"""

import ast
import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO
from musubi_tuner.networks.network_arch import get_arch_config

logger = logging.getLogger(__name__)
# Note: logging.basicConfig removed to avoid conflicts with BlissfulLogger - configure at entry points


def factorization(dimension: int, factor: int = -1) -> Tuple[int, int]:
    """Find factors of dimension for Kronecker product decomposition.

    If factor <= 0, finds the pair (a, b) with a*b == dimension
    that minimizes |a - b| (most square-like factorization).
    If factor > 0, returns (dimension // factor, factor) if it divides evenly,
    otherwise falls back to automatic factorization.
    """
    if factor > 0 and dimension % factor == 0:
        return dimension // factor, factor
    if factor > 0:
        logger.warning(f"factor={factor} does not divide dimension={dimension} evenly, falling back to automatic factorization")

    # Find pair (a, b) with a*b == dimension that minimizes |a - b|
    best_a, best_b = 1, dimension
    for a in range(2, int(math.isqrt(dimension)) + 1):
        if dimension % a == 0:
            b = dimension // a
            if abs(a - b) < abs(best_a - best_b):
                best_a, best_b = a, b
    return best_b, best_a  # larger first


def make_kron(w1: torch.Tensor, w2: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Compute Kronecker product of two 2D matrices with optional scaling."""
    if w2.dim() == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    w2 = w2.contiguous()
    return torch.kron(w1, w2) * scale


class LoKrModule(nn.Module):
    """LoKr training module. Linear-only — Conv2d raises ValueError."""

    _kron_warning_emitted = False
    _dropout_warning_emitted = False

    def __init__(
        self,
        lora_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1.0,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        factor: int = -1,
        **kwargs,  # absorbs use_rslora, use_dora from LoRANetwork
    ):
        super().__init__()
        self.lora_name = lora_name
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.module_dropout = module_dropout
        self.org_module = org_module  # temporary reference, deleted in apply_to()
        self.enabled = True

        if not LoKrModule._dropout_warning_emitted and (
            (dropout is not None and dropout > 0) or (rank_dropout is not None and rank_dropout > 0)
        ):
            logger.warning("LoKr v1 does not implement dropout or rank_dropout; these arguments are ignored.")
            LoKrModule._dropout_warning_emitted = True

        if org_module.__class__.__name__ != "Linear":
            raise ValueError(f"LoKr v1 only supports Linear modules, got {org_module.__class__.__name__}")

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        # Factorize input and output dimensions
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # w1: (out_l, in_m) — always full-rank (small)
        self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))

        # w2: low-rank if dim > 0, else full matrix
        shape = (out_k, in_n)
        if lora_dim > 0 and lora_dim < max(shape):
            # Low-rank factorization: w2 = w2_a @ w2_b
            self.lokr_w2_a = nn.Parameter(torch.empty(shape[0], lora_dim))
            self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1]))
            self.lokr_w2 = None
        else:
            # Full matrix
            self.lokr_w2 = nn.Parameter(torch.empty(*shape))
            self.lokr_w2_a = None
            self.lokr_w2_b = None

        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialize weights
        if self.lokr_w2 is not None:
            nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
        else:
            nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            nn.init.zeros_(self.lokr_w2_b)
        nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))

    def get_weight(self, multiplier=None) -> torch.Tensor:
        if multiplier is None:
            multiplier = self.multiplier

        w1 = self.lokr_w1
        if self.lokr_w2 is not None:
            w2 = self.lokr_w2
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b

        # Compute scale from alpha / dim
        dim = self.lokr_w2_a.shape[1] if self.lokr_w2_a is not None else max(self.lokr_w2.shape)
        alpha = self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
        scale = alpha / dim

        if not LoKrModule._kron_warning_emitted:
            expected_numel = w1.shape[0] * w2.shape[0] * w1.shape[1] * w2.shape[1]
            if expected_numel > 16_000_000:
                logger.warning(
                    f"LoKr v1 materializes full Kronecker product ({expected_numel:,} elements) in forward pass. "
                    "This uses significantly more memory than LoRA. Consider reducing target modules or using LoRA for large layers."
                )
                LoKrModule._kron_warning_emitted = True

        diff_weight = make_kron(w1, w2, scale)
        return diff_weight * multiplier

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module  # prevent base module from appearing in state_dict

    def forward(self, x, *args, **kwargs):
        if not self.enabled:
            return self.org_forward(x, *args, **kwargs)

        result = self.org_forward(x, *args, **kwargs)

        # Apply LoKr delta
        diff_weight = self.get_weight()

        # Apply dropout if training
        if self.training and self.module_dropout is not None:
            if torch.rand(1).item() < self.module_dropout:
                return result

        if diff_weight.dim() == 2:
            delta = torch.nn.functional.linear(x, diff_weight)
        else:
            delta = torch.nn.functional.linear(x, diff_weight.view(diff_weight.shape[0], -1))

        return result + delta


class LoKrInfModule(LoKrModule):
    """LoKr inference module with merge_to and get_weight support."""

    def __init__(self, lora_name, org_module, *args, **kwargs):
        super().__init__(lora_name, org_module, *args, **kwargs)
        self.org_module_ref = [org_module]  # non-module container to avoid state_dict bloat
        self.enabled = False  # disabled by default for inference (will be merged)

    def apply_to(self):
        # Override parent: keep org_module_ref intact (needed for merge_to)
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_sd = self.org_module_ref[0].state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        merge_device = org_device if device is None else device

        # Read weights from sd (matching LoRA/LoHa merge_to contract)
        w1 = sd["lokr_w1"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)

        if "lokr_w2" in sd:
            w2 = sd["lokr_w2"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
            dim = max(w2.shape)
        else:
            w2_a = sd["lokr_w2_a"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
            w2_b = sd["lokr_w2_b"].to(device=merge_device, dtype=torch.float, non_blocking=non_blocking)
            w2 = w2_a @ w2_b
            dim = w2_a.shape[1]

        alpha = sd.get("alpha", torch.tensor(dim, dtype=torch.float)).to(device=merge_device, dtype=torch.float)
        scale = alpha.item() / dim

        diff_weight = make_kron(w1, w2, scale) * self.multiplier
        if weight.shape != diff_weight.shape:
            diff_weight = diff_weight.view(weight.shape)

        weight = weight.to(device=merge_device, dtype=torch.float, non_blocking=non_blocking) + diff_weight
        org_sd["weight"] = weight.to(device=org_device, dtype=dtype if dtype is not None else org_dtype)
        self.org_module_ref[0].load_state_dict(org_sd)


def _resolve_factor(
    weights_sd: Dict[str, torch.Tensor],
    explicit_factor: Optional[int] = None,
    metadata_factor: Optional[int] = None,
) -> Tuple[int, bool]:
    """Resolve LoKr factor with precedence: explicit > persisted buffer > metadata > default(-1).

    Args:
        weights_sd: State dict (may contain lokr_factor tensor buffer).
        explicit_factor: Factor from CLI/network_args (highest precedence).
        metadata_factor: Factor from safetensors metadata (ss_lokr_factor), used as
            fallback when lokr_factor tensor buffer is absent. Callers that load
            from safetensors should read metadata via safe_open().metadata() and
            pass ss_lokr_factor here. Note: the primary merge path
            (merge_nonlora_to_model → merge_weights_to_tensor) does not need
            factor since it operates on actual weight shapes directly.

    Returns (factor: int, had_mismatch_warning: bool).
    """
    persisted_factor = None
    if "lokr_factor" in weights_sd:
        persisted_factor = int(weights_sd["lokr_factor"].item())
    elif metadata_factor is not None:
        persisted_factor = int(metadata_factor)
        logger.info(f"Recovered lokr_factor={persisted_factor} from safetensors metadata (buffer was absent)")

    if explicit_factor is not None:
        factor = int(explicit_factor)
        if persisted_factor is not None and factor != persisted_factor:
            logger.warning(f"Explicit factor={factor} differs from persisted factor={persisted_factor}. Using explicit.")
            return factor, True
        return factor, False
    elif persisted_factor is not None:
        return persisted_factor, False
    else:
        return -1, False


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
    architecture = kwargs.get("architecture", ARCHITECTURE_HUNYUAN_VIDEO)
    config = get_arch_config(architecture)

    # Warn and pop conv_dim/conv_alpha if provided (LoKr v1 is Linear-only)
    if "conv_dim" in kwargs:
        logger.warning("conv_dim is not supported by LoKr v1 (Linear-only). Ignoring.")
        kwargs.pop("conv_dim")
    if "conv_alpha" in kwargs:
        logger.warning("conv_alpha is not supported by LoKr v1 (Linear-only). Ignoring.")
        kwargs.pop("conv_alpha")

    # Add default exclude patterns (always additive — see network_arch.py contract)
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)

    exclude_patterns.extend(config["exclude_patterns"])

    # Qwen exclude_mod support (parity with lora_qwen_image.py)
    if "exclude_mod_patterns" in config:
        exclude_mod = kwargs.get("exclude_mod", True)
        if isinstance(exclude_mod, str):
            exclude_mod = ast.literal_eval(exclude_mod)
        if exclude_mod:
            exclude_patterns.extend(config["exclude_mod_patterns"])

    kwargs["exclude_patterns"] = exclude_patterns

    # Kandinsky include_patterns support
    if "include_patterns" in config and "include_patterns" not in kwargs:
        kwargs["include_patterns"] = config["include_patterns"]

    factor = int(kwargs.pop("factor", -1))

    network = lora.create_network(
        config["target_modules"],
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
        enable_conv2d=False,
        **kwargs,
    )

    # Persist factor as network-level buffer for save/load round-trip
    network.register_buffer("lokr_factor", torch.tensor(factor, dtype=torch.int64))
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    architecture = kwargs.get("architecture", ARCHITECTURE_HUNYUAN_VIDEO)
    config = get_arch_config(architecture)

    explicit_factor = kwargs.pop("factor", None)
    if explicit_factor is not None:
        explicit_factor = int(explicit_factor)
    metadata_factor = kwargs.pop("metadata_factor", None)
    if metadata_factor is not None:
        metadata_factor = int(metadata_factor)
    factor, _ = _resolve_factor(weights_sd, explicit_factor, metadata_factor)

    module_class = LoKrInfModule if for_inference else LoKrModule
    module_kwargs = {"factor": factor}

    network = lora.create_network_from_weights(
        config["target_modules"],
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        module_class=module_class,
        module_kwargs=module_kwargs,
        enable_conv2d=False,
        **kwargs,
    )

    # Re-register factor buffer on reconstructed network
    network.register_buffer("lokr_factor", torch.tensor(factor, dtype=torch.int64))
    return network


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoKr weights directly into a model weight tensor.

    Supports both low-rank (w2_a + w2_b) and full-matrix (w2) modes.
    Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoKr keys found.
    """
    w1_key = lora_name + ".lokr_w1"
    w2_key = lora_name + ".lokr_w2"
    w2_a_key = lora_name + ".lokr_w2_a"
    w2_b_key = lora_name + ".lokr_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1_key not in lora_weight_keys:
        return model_weight

    w1 = lora_sd[w1_key].to(calc_device)

    # Determine w2 mode: low-rank (w2_a @ w2_b) or full matrix (w2)
    if w2_a_key in lora_weight_keys:
        w2 = lora_sd[w2_a_key].to(calc_device) @ lora_sd[w2_b_key].to(calc_device)
        dim = lora_sd[w2_a_key].shape[1]
    elif w2_key in lora_weight_keys:
        w2 = lora_sd[w2_key].to(calc_device)
        dim = max(w2.shape)
    else:
        return model_weight

    alpha = lora_sd.get(alpha_key, torch.tensor(dim))
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / dim

    org_device = model_weight.device
    original_dtype = model_weight.dtype
    compute_dtype = torch.float16 if original_dtype.itemsize == 1 else torch.float32
    model_weight = model_weight.to(calc_device, dtype=compute_dtype)
    w1 = w1.to(compute_dtype)
    w2 = w2.to(compute_dtype)

    diff_weight = make_kron(w1, w2, scale)

    # Reshape if needed
    if model_weight.shape != diff_weight.shape:
        diff_weight = diff_weight.view(model_weight.shape)

    model_weight = model_weight + multiplier * diff_weight

    model_weight = model_weight.to(device=org_device, dtype=original_dtype)

    # Remove consumed keys
    for key in [w1_key, w2_key, w2_a_key, w2_b_key, alpha_key]:
        lora_weight_keys.discard(key)

    return model_weight
