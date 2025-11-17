# LoRA module for LongCat video transformer

from __future__ import annotations

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


LONGCAT_TARGET_REPLACE_MODULES = [
    "Attention",
    "MultiHeadCrossAttention",
    "FeedForwardSwiGLU",
]


def _build_exclude_patterns(raw_patterns: Optional[str]) -> List[str]:
    if raw_patterns is None:
        patterns: List[str] = []
    else:
        patterns = ast.literal_eval(raw_patterns)
        if not isinstance(patterns, list):
            raise ValueError("exclude_patterns must evaluate to a list")
    patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
    return patterns


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
    kwargs["exclude_patterns"] = _build_exclude_patterns(kwargs.get("exclude_patterns"))

    return lora.create_network(
        LONGCAT_TARGET_REPLACE_MODULES,
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
        LONGCAT_TARGET_REPLACE_MODULES,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
