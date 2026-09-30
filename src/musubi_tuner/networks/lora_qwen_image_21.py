# LoRA module for Qwen-Image 2.1

from typing import Optional

import torch
from torch import nn

from musubi_tuner.networks import lora

QWEN_IMAGE_21_TARGET_REPLACE_MODULES = ["QwenImage21TransformerBlock"]


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: list[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create LoRA modules for the attention and MLP projections in each block."""
    return lora.create_network(
        QWEN_IMAGE_21_TARGET_REPLACE_MODULES,
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
    weights_sd: dict[str, torch.Tensor],
    text_encoders: Optional[list[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
):
    """Restore LoRA modules from a saved adapter state dict."""
    return lora.create_network_from_weights(
        QWEN_IMAGE_21_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
