# LoRA module for Ideogram 4

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora


IDEOGRAM4_TARGET_REPLACE_MODULES = ["Ideogram4TransformerBlock"]
IDEOGRAM4_TARGET_INCLUDE_PATTERNS = [
    r".*attention\.(qkv|o)",
    r".*feed_forward\.w[123]",
]
IDEOGRAM4_LINEAR_CLASS_NAMES = ("Linear", "Fp8Linear")


def _parse_patterns(value):
    if value is None:
        return []
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


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
    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns", None))
    include_patterns = _parse_patterns(kwargs.get("include_patterns", None))

    exclude_patterns.append(r".*")
    include_patterns.extend(IDEOGRAM4_TARGET_INCLUDE_PATTERNS)

    kwargs["exclude_patterns"] = exclude_patterns
    kwargs["include_patterns"] = include_patterns

    network = lora.create_network(
        IDEOGRAM4_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        linear_module_class_names=IDEOGRAM4_LINEAR_CLASS_NAMES,
        **kwargs,
    )
    if len(network.unet_loras) == 0:
        raise RuntimeError("Ideogram 4 LoRA found zero target modules. Check Fp8Linear discovery and include patterns.")
    return network


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        IDEOGRAM4_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        linear_module_class_names=IDEOGRAM4_LINEAR_CLASS_NAMES,
        **kwargs,
    )
