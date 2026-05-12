# LoRA module for HiDream-O1-Image

import ast
from typing import Dict, List, Optional

import torch
import torch.nn as nn

import musubi_tuner.networks.lora as lora


HIDREAM_O1_DECODER_TARGET_REPLACE_MODULES = ["Qwen3VLTextDecoderLayer"]
HIDREAM_O1_PIXEL_TARGET_REPLACE_MODULES = ["BottleneckPatchEmbed", "FinalLayer"]
HIDREAM_O1_TIMESTEP_TARGET_REPLACE_MODULES = ["TimestepEmbedder"]
HIDREAM_O1_VISUAL_TARGET_REPLACE_MODULES = [
    "Qwen3VLVisionPatchEmbed",
    "Qwen3VLVisionBlock",
    "Qwen3VLVisionPatchMerger",
]

HIDREAM_O1_T2I_TARGET_REPLACE_MODULES = HIDREAM_O1_DECODER_TARGET_REPLACE_MODULES + HIDREAM_O1_PIXEL_TARGET_REPLACE_MODULES
HIDREAM_O1_I2I_TARGET_REPLACE_MODULES = HIDREAM_O1_T2I_TARGET_REPLACE_MODULES + HIDREAM_O1_VISUAL_TARGET_REPLACE_MODULES
HIDREAM_O1_TARGET_REPLACE_MODULES = (
    HIDREAM_O1_I2I_TARGET_REPLACE_MODULES
    + HIDREAM_O1_TIMESTEP_TARGET_REPLACE_MODULES
)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


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
    has_control = _to_bool(kwargs.pop("hidream_has_control", False))

    target_replace_modules = (
        HIDREAM_O1_I2I_TARGET_REPLACE_MODULES if has_control else HIDREAM_O1_T2I_TARGET_REPLACE_MODULES
    )

    if has_control and kwargs.get("conv_dim", None) is None:
        kwargs["conv_dim"] = 4
        kwargs.setdefault("conv_alpha", 4)

    exclude_patterns = kwargs.get("exclude_patterns", None) or []
    if isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    exclude_patterns.append(r".*(embed_tokens|lm_head).*")
    kwargs["exclude_patterns"] = exclude_patterns

    return lora.create_network(
        target_replace_modules,
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
        HIDREAM_O1_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
