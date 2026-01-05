# LoRA module for Kandinsky 5 DiT

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


KANDINSKY5_TARGET_REPLACE_MODULES = ["TransformerEncoderBlock", "TransformerDecoderBlock"]


def _prepare_include_patterns(include_patterns: Optional[str]) -> List[str]:
    if include_patterns is None:
        # Default to the same modules targeted in the reference trainer
        return [
            r".*self_attention\.to_query.*",
            r".*self_attention\.to_key.*",
            r".*self_attention\.to_value.*",
            r".*self_attention\.out_layer.*",
            r".*cross_attention\.to_query.*",
            r".*cross_attention\.to_key.*",
            r".*cross_attention\.to_value.*",
            r".*cross_attention\.out_layer.*",
            r".*feed_forward\.in_layer.*",
            r".*feed_forward\.out_layer.*",
        ]
    return ast.literal_eval(include_patterns)


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
    include_patterns = _prepare_include_patterns(kwargs.get("include_patterns", None))
    kwargs["include_patterns"] = include_patterns

    # Exclude modulation layers by default to mirror the reference targeting
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = [r".*modulation.*"]
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        exclude_patterns.append(r".*modulation.*")
    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        KANDINSKY5_TARGET_REPLACE_MODULES,
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


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    return lora.create_network_from_weights(
        KANDINSKY5_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
