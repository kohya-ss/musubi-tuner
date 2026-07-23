from __future__ import annotations

from collections.abc import Mapping
import logging
from pathlib import Path
import re
from typing import Optional

from safetensors import safe_open
import torch
import torch.nn as nn

from musubi_tuner.networks import lora


logger = logging.getLogger(__name__)

_MAGE_FLOW_INCLUDE = r"transformer_blocks\.\d+\.(?:attn|img_mlp|txt_mlp)\..*"
_MAGE_FLOW_WEIGHT_NAME = re.compile(r"^lora_unet_transformer_blocks_\d+_(?:attn|img_mlp|txt_mlp)_")
_MAGE_ARCHITECTURES = {"mage_flow", "mage_flow_edit"}


def validate_adapter_architecture(
    path: str | Path,
    *,
    expected: str,
    allow_mismatch: bool,
) -> None:
    if expected not in _MAGE_ARCHITECTURES:
        raise ValueError(f"unknown expected Mage-Flow architecture {expected!r}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
    except Exception as exc:
        raise ValueError(f"cannot inspect Mage-Flow LoRA safetensors metadata from {path}: {exc}") from exc
    actual = metadata.get("ss_base_model_version")
    if actual is None:
        logger.warning("LoRA %s has no ss_base_model_version metadata; architecture cannot be verified", path)
        return
    if actual not in _MAGE_ARCHITECTURES:
        raise ValueError(f"LoRA architecture {actual!r} is not a Mage-Flow architecture")
    if actual != expected and not allow_mismatch:
        raise ValueError(
            f"LoRA architecture mismatch: checkpoint is {actual}, current mode is {expected}; "
            "use the explicit architecture-mismatch override only if this is intentional"
        )


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
    requested = {key.split(".", 1)[0] for key in weights_sd if key.endswith(".lora_down.weight")}
    network = lora.create_network_from_weights(
        None,
        multiplier,
        weights_sd,
        text_encoders,
        unet,
        for_inference,
        **kwargs,
    )
    created = {module.lora_name for module in network.unet_loras}
    missing = sorted(requested - created)
    if missing:
        raise ValueError("LoRA weights do not map to modules in this Mage-Flow transformer: " + ", ".join(missing[:10]))
    if not requested:
        raise ValueError("Mage-Flow LoRA weights do not contain any lora_down tensors")
    return network


__all__ = ["create_arch_network", "create_arch_network_from_weights", "validate_adapter_architecture"]
