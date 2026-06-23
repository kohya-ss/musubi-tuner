# LoRA module for Krea 2 (K2)

from typing import Dict, List, Optional
import torch
import torch.nn as nn

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import musubi_tuner.networks.lora as lora


# Target ALL Linear layers in the DiT. This is None (rather than a list of module
# class names) so the generic LoRA walker wraps every Linear in the model — which,
# for K2, is exactly the set of layers the model authors recommend training by
# default (rank 32 / alpha 32): first, last.linear, the per-block attention
# (wq/wk/wv/wo/gate) and SwiGLU MLP (gate/up/down), the text-fusion transformer
# (its attn/mlp blocks + projector), and the time/text projection MLPs
# (tmlp/txtmlp/tproj). All 264 Linears, verified to coincide with the recommended list.
#
# The modulation (DoubleSharedModulation / SimpleModulation) and all RMSNorm hold
# raw nn.Parameter tensors, not Linear modules, so they are never wrapped — no
# explicit exclude needed (unlike Qwen-Image, whose modulation is a Linear).
#
# Because the default targets everything, both exclude_patterns and include_patterns
# stay free for the user. To reproduce the authors' "long training run" config
# (increase rank, focus on the attention projections to preserve prompt adherence),
# pass --network_args excluding the rest, e.g.:
#   exclude_patterns=['.*\.mlp\..*','first','last\.linear','tmlp\..*','txtmlp\..*','tproj\.1','txtfusion\..*']
# which keeps only the per-block attention wq/wk/wv/wo/gate. For an arbitrary subset,
# use exclude_patterns=['.*'] to drop everything, then include_patterns=[...] to add back.
KREA2_TARGET_REPLACE_MODULES = None


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
    return lora.create_network(
        KREA2_TARGET_REPLACE_MODULES,
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
        KREA2_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )
