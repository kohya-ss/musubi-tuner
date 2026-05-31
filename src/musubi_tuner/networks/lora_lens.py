# LoRA module for Lens

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora


LENS_TARGET_REPLACE_MODULES = ["LensTransformerBlock"]
DEFAULT_EXCLUDE_PATTERNS = [
    r".*(img_mod|txt_mod).*",
    r".*(norm|pos_embed|time_text_embed|proj_out).*",
]


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
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = list(DEFAULT_EXCLUDE_PATTERNS)
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        exclude_patterns.extend(DEFAULT_EXCLUDE_PATTERNS)

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is None:
        include_patterns = [
            r".*attn\.(img_qkv|txt_qkv|to_add_out).*",
            r".*attn\.to_out\.0.*",
            r".*(img_mlp|txt_mlp)\.(w1|w2|w3).*",
        ]

    kwargs["exclude_patterns"] = exclude_patterns
    kwargs["include_patterns"] = include_patterns

    return lora.create_network(
        LENS_TARGET_REPLACE_MODULES,
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
        LENS_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
