# LoRA module for Wan2.1

import ast
from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]


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
    # Handle LORAID parameter filtering and value storage
    loraid_include_pattern = kwargs.pop("loraid_include_pattern", None)
    loraid_exclude_pattern = kwargs.pop("loraid_exclude_pattern", None)
    loraid_value = kwargs.pop("loraid_value", None)
    
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        # Handle both string and list representations
        if isinstance(exclude_patterns, str):
            try:
                exclude_patterns = ast.literal_eval(exclude_patterns)
            except (ValueError, SyntaxError):
                # If it's not a valid list representation, treat as single pattern
                exclude_patterns = [exclude_patterns]
        elif not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    # exclude if 'img_mod', 'txt_mod' or 'modulation' in the name
    exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
    
    # Add LORAID exclude pattern if specified
    if loraid_exclude_pattern:
        exclude_patterns.append(loraid_exclude_pattern)
        logger.info(f"Added LORAID exclude pattern: {loraid_exclude_pattern}")

    kwargs["exclude_patterns"] = exclude_patterns
    
    # Handle LORAID include patterns
    if loraid_include_pattern:
        include_patterns = kwargs.get("include_patterns", None)
        if include_patterns is None:
            include_patterns = []
        else:
            # Handle both string and list representations
            if isinstance(include_patterns, str):
                try:
                    include_patterns = ast.literal_eval(include_patterns)
                except (ValueError, SyntaxError):
                    # If it's not a valid list representation, treat as single pattern
                    include_patterns = [include_patterns]
            elif not isinstance(include_patterns, list):
                include_patterns = [include_patterns]
        
        include_patterns.append(loraid_include_pattern)
        kwargs["include_patterns"] = include_patterns
        logger.info(f"Added LORAID include pattern: {loraid_include_pattern}")
    
    # Pass LORAID value to the network for metadata storage
    if loraid_value is not None:
        kwargs["loraid_value"] = loraid_value
        logger.info(f"LORAID value {loraid_value} will be stored in network metadata")

    network = lora.create_network(
        WAN_TARGET_REPLACE_MODULES,
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
    
    # Store LORAID value in the network for later metadata injection
    if loraid_value is not None:
        network._loraid_value = loraid_value
    
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
        WAN_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
