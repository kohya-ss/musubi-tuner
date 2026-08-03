import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora


MINIMAX_H3_TARGET_REPLACE_MODULES = ["DiTBlock"]
MINIMAX_H3_DEFAULT_TARGET_PATTERN = r"blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))"
_DEFAULT_EXCLUDE_PATTERN = rf"(?!{MINIMAX_H3_DEFAULT_TARGET_PATTERN}$).*"


def _pattern_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return list(value)


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
    exclude_patterns = _pattern_list(kwargs.get("exclude_patterns"))
    exclude_patterns.append(_DEFAULT_EXCLUDE_PATTERN)
    kwargs["exclude_patterns"] = exclude_patterns
    network = lora.create_network(
        MINIMAX_H3_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )
    if not network.unet_loras:
        raise RuntimeError("MiniMax-H3 LoRA found zero target modules; check the target include/exclude patterns")
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
        MINIMAX_H3_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
