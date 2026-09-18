import ast
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from musubi_tuner import convert_lora
import musubi_tuner.networks.lora as lora


logger = logging.getLogger(__name__)

MINIMAX_H3_TARGET_REPLACE_MODULES = ["DiTBlock"]
MINIMAX_H3_DEFAULT_TARGET_PATTERN = r"blocks\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))"
_DEFAULT_EXCLUDE_PATTERN = rf"(?!{MINIMAX_H3_DEFAULT_TARGET_PATTERN}$).*"
# Diffusers-format H3 LoRA keys, the format of the third-party training adapters and of
# ai-toolkit / diffusion-pipe LoRAs: `diffusion_model.blocks.N.attn.qkv_proj.lora_A.weight`
_DIFFUSERS_KEY_PREFIXES = ("diffusion_model.", "transformer.")


def convert_lora_state_dict(weights_sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return ``weights_sd`` in the native ``lora_unet_*`` key format.

    Native state dicts pass through untouched. Diffusers-format ones (``diffusion_model.`` or
    ``transformer.`` prefix, ``lora_A``/``lora_B``, no alpha) are renamed to the Musubi keys
    with ``alpha = rank`` filled in, the same treatment ``hv_train_network`` gives HunyuanVideo
    LoRAs. Every H3 LoRA entry point (``--base_weights``, ``--lora_weight`` on each of its
    routes) goes through here so the third-party adapters load without a manual conversion.
    """
    if not weights_sd:
        return weights_sd
    first_key = next(iter(weights_sd))
    if first_key.startswith("lora_"):
        return weights_sd
    if first_key.startswith(_DIFFUSERS_KEY_PREFIXES):
        logger.info("Converting MiniMax-H3 LoRA weights from the Diffusers key format to the native format")
        return convert_lora.convert_from_diffusers("lora_unet_", weights_sd)
    return weights_sd


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
    # Search every Linear of the model, not just the DiTBlock targets of the training default:
    # the modules are created from the weights, so a saved LoRA decides its own coverage, and
    # third-party LoRAs (ai-toolkit and the training adapters) also carry token_refiner modules.
    return lora.create_network_from_weights(
        None,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
