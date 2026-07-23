from __future__ import annotations

from collections.abc import Mapping
import logging
import re
from typing import Optional

import torch
import torch.nn as nn

from musubi_tuner.networks import lora


logger = logging.getLogger(__name__)

_MAGE_FLOW_INCLUDE = r"transformer_blocks\.\d+\.(?:attn|img_mlp|txt_mlp)\..*"
_MAGE_FLOW_WEIGHT_NAME = re.compile(r"^lora_unet_transformer_blocks_\d+_(?:attn|img_mlp|txt_mlp)_")


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
    if kwargs.get("include_patterns") is not None or kwargs.get("exclude_patterns") is not None:
        logger.warning("Mage-Flow ignores include/exclude patterns because its supported LoRA scope is fixed")
    kwargs["exclude_patterns"] = [r".*"]
    kwargs["include_patterns"] = [_MAGE_FLOW_INCLUDE]
    return lora.create_network(
        None,
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


def _validate_weight_scope(weights_sd: Mapping[str, torch.Tensor]) -> None:
    invalid = sorted(
        {key.split(".", 1)[0] for key in weights_sd if key.startswith("lora_unet_") and not _MAGE_FLOW_WEIGHT_NAME.match(key)}
    )
    if invalid:
        preview = ", ".join(invalid[:10])
        raise ValueError(f"LoRA weights outside the supported Mage-Flow scope: {preview}")


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: dict[str, torch.Tensor],
    text_encoders: Optional[list[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> lora.LoRANetwork:
    _validate_weight_scope(weights_sd)
    return lora.create_network_from_weights(
        None,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )


__all__ = ["create_arch_network", "create_arch_network_from_weights"]
