"""Utilities for Lens loading, caching, and sampling."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from einops import rearrange
from huggingface_hub import hf_hub_download
from torch import Tensor

from musubi_tuner.utils.safetensors_utils import load_split_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

COMFY_LENS_REPO_ID = "Comfy-Org/Lens"
OFFICIAL_LENS_REPO_ID = "microsoft/Lens"

COMFY_LENS_FILES = (
    "diffusion_models/lens_bf16.safetensors",
    "text_encoders/gpt_oss_20b_nvfp4.safetensors",
    "vae/flux2-vae.safetensors",
)
OFFICIAL_METADATA_FILES = (
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def download_lens_components(
    output_dir: Union[str, Path],
    repo_id: str = COMFY_LENS_REPO_ID,
    metadata_repo_id: str = OFFICIAL_LENS_REPO_ID,
    dry_run: bool = False,
    token: Optional[str] = None,
) -> list[tuple[str, str]]:
    output_dir = Path(output_dir)
    downloads: list[tuple[str, str]] = []
    for file in COMFY_LENS_FILES:
        downloads.append((repo_id, file))
    for file in OFFICIAL_METADATA_FILES:
        downloads.append((metadata_repo_id, file))

    if dry_run:
        return downloads

    for source_repo, filename in downloads:
        logger.info(f"Downloading {source_repo}/{filename}")
        local_dir = output_dir
        hf_hub_download(
            repo_id=source_repo,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token,
        )
    return downloads


def normalize_lens_state_dict(sd: dict[str, Tensor]) -> dict[str, Tensor]:
    prefixes = ("transformer.", "model.diffusion_model.", "diffusion_model.", "model.", "module.")
    out = {}
    for key, value in sd.items():
        if key.startswith("__"):
            continue
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        out[new_key] = value
    return out


def _unsqueeze_attention_weight_if_needed(key: str, value: Tensor) -> Tensor:
    if (
        ".attn_1." in key
        and key.endswith(".weight")
        and value.ndim == 2
        and any(f".{name}.weight" in key for name in ("q", "k", "v", "proj_out"))
    ):
        return value[:, :, None, None]
    return value


def normalize_lens_vae_state_dict(sd: dict[str, Tensor], num_levels: int = 4) -> dict[str, Tensor]:
    """Convert diffusers AutoencoderKLFlux2 keys to the local FLUX.2 AutoEncoder keys."""
    has_diffusers_keys = any(
        key.startswith(("quant_conv.", "post_quant_conv.", "encoder.down_blocks.", "decoder.up_blocks."))
        for key in sd.keys()
    )
    if not has_diffusers_keys:
        return sd

    out: dict[str, Tensor] = {}
    for key, value in sd.items():
        new_key = key
        if new_key.startswith("__"):
            continue
        if new_key.startswith("quant_conv."):
            new_key = f"encoder.{new_key}"
        elif new_key.startswith("post_quant_conv."):
            new_key = f"decoder.{new_key}"
        else:
            new_key = new_key.replace(".conv_shortcut.", ".nin_shortcut.")

            if new_key.startswith("encoder."):
                new_key = new_key.replace("encoder.conv_norm_out.", "encoder.norm_out.")
                new_key = new_key.replace("encoder.mid_block.resnets.0.", "encoder.mid.block_1.")
                new_key = new_key.replace("encoder.mid_block.resnets.1.", "encoder.mid.block_2.")
                new_key = new_key.replace("encoder.mid_block.attentions.0.", "encoder.mid.attn_1.")
                new_key = new_key.replace(".group_norm.", ".norm.")
                new_key = new_key.replace(".to_q.", ".q.")
                new_key = new_key.replace(".to_k.", ".k.")
                new_key = new_key.replace(".to_v.", ".v.")
                new_key = new_key.replace(".to_out.0.", ".proj_out.")
                new_key = re.sub(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.", r"encoder.down.\1.block.\2.", new_key)
                new_key = re.sub(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.", r"encoder.down.\1.downsample.conv.", new_key)
            elif new_key.startswith("decoder."):
                new_key = new_key.replace("decoder.conv_norm_out.", "decoder.norm_out.")
                new_key = new_key.replace("decoder.mid_block.resnets.0.", "decoder.mid.block_1.")
                new_key = new_key.replace("decoder.mid_block.resnets.1.", "decoder.mid.block_2.")
                new_key = new_key.replace("decoder.mid_block.attentions.0.", "decoder.mid.attn_1.")
                new_key = new_key.replace(".group_norm.", ".norm.")
                new_key = new_key.replace(".to_q.", ".q.")
                new_key = new_key.replace(".to_k.", ".k.")
                new_key = new_key.replace(".to_v.", ".v.")
                new_key = new_key.replace(".to_out.0.", ".proj_out.")

                resnet_match = re.match(r"decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.*)", new_key)
                if resnet_match is not None:
                    local_level = num_levels - 1 - int(resnet_match.group(1))
                    new_key = f"decoder.up.{local_level}.block.{resnet_match.group(2)}.{resnet_match.group(3)}"
                upsample_match = re.match(r"decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.*)", new_key)
                if upsample_match is not None:
                    local_level = num_levels - 1 - int(upsample_match.group(1))
                    new_key = f"decoder.up.{local_level}.upsample.conv.{upsample_match.group(2)}"

        out[new_key] = _unsqueeze_attention_weight_if_needed(new_key, value)

    logger.info("Converted Lens FLUX.2 VAE diffusers state dict keys to local AutoEncoder format")
    return out


def load_lens_transformer(
    dit_path: Union[str, Path],
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    disable_mmap: bool = False,
    *,
    calc_device: Optional[Union[str, torch.device]] = None,
    loading_device: Optional[Union[str, torch.device]] = None,
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
):
    from musubi_tuner.lens.lens_model import (
        DEFAULT_TRANSFORMER_CONFIG,
        FP8_OPTIMIZATION_EXCLUDE_KEYS,
        FP8_OPTIMIZATION_TARGET_KEYS,
        LensTransformer2DModel,
    )

    if dit_weight_dtype is None and not fp8_scaled:
        dit_weight_dtype = dtype
    if loading_device is None:
        loading_device = device if device is not None else "cpu"
    if calc_device is None:
        calc_device = loading_device

    calc_device = torch.device(calc_device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        model = LensTransformer2DModel(**DEFAULT_TRANSFORMER_CONFIG)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading Lens DiT weights from {dit_path}, device={loading_device}, fp8_scaled={fp8_scaled}")
    if fp8_scaled:
        from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
        from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

        sd = load_safetensors_with_lora_and_fp8(
            model_files=str(dit_path),
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=calc_device,
            move_to_device=(loading_device == calc_device),
            dit_weight_dtype=None,
            target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
            disable_numpy_memmap=disable_mmap,
        )
    else:
        sd = load_split_weights(str(dit_path), device=loading_device, disable_mmap=disable_mmap, dtype=dit_weight_dtype)
    sd = normalize_lens_state_dict(sd)

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            logger.info(f"Moving Lens fp8-scaled weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Lens DiT: {info}")
    if not fp8_scaled:
        model.to(loading_device)
        if dit_weight_dtype is not None and dit_weight_dtype.itemsize > 1:
            model.to(dit_weight_dtype)
    return model


def load_lens_vae(vae_path: Union[str, Path], dtype: torch.dtype, device: Union[str, torch.device]):
    from musubi_tuner.flux_2 import flux2_models

    logger.info("Building Lens FLUX.2 AutoEncoder")
    params = flux2_models.AutoEncoderParams()
    with init_empty_weights():
        ae = flux2_models.AutoEncoder(params).to(dtype)

    logger.info(f"Loading Lens VAE state dict from {vae_path}")
    sd = load_split_weights(str(vae_path), device=str(device), disable_mmap=True, dtype=dtype)
    sd = normalize_lens_vae_state_dict(sd, num_levels=len(params.ch_mult))
    info = ae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Lens VAE: {info}")
    return ae



def pack_latents(latents: Tensor) -> Tensor:
    return rearrange(latents, "b c h w -> b (h w) c")


def unpack_latents(latents: Tensor, height: int, width: int) -> Tensor:
    return rearrange(latents, "b (h w) c -> b c h w", h=height, w=width)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return float(a * num_steps + b)


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lens_sigmas(num_steps: int, image_seq_len: int, device: Union[str, torch.device]) -> Tensor:
    sigmas = torch.from_numpy(np.linspace(1.0, 1.0 / num_steps, num_steps)).to(device=device, dtype=torch.float32)
    mu = compute_empirical_mu(image_seq_len, num_steps)
    sigmas = time_shift(mu, 1.0, sigmas)
    sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device, dtype=sigmas.dtype)])
    return sigmas


def euler_step(model_output: Tensor, sample: Tensor, sigmas: Tensor, step_index: int) -> Tensor:
    sample = sample.to(torch.float32)
    dt = sigmas[step_index + 1] - sigmas[step_index]
    return (sample + model_output.to(torch.float32) * dt).to(model_output.dtype)


def pad_lens_text_features(features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    if not features:
        raise ValueError("features must not be empty")
    lengths = [x.shape[0] for x in features]
    max_len = max(lengths)
    if max_len == 0:
        dim = features[0].shape[-1]
        return features[0].new_zeros((len(features), 0, dim)), torch.zeros((len(features), 0), dtype=torch.bool)

    padded = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in features]
    tensor = torch.stack(padded, dim=0)
    mask = torch.zeros((len(features), max_len), dtype=torch.bool)
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    return tensor, mask


def align_text_feature_lists(
    pos_features: list[Tensor],
    pos_mask: Tensor,
    neg_features: list[Tensor],
    neg_mask: Tensor,
) -> tuple[list[Tensor], Tensor, list[Tensor], Tensor]:
    target = max(pos_mask.shape[1], neg_mask.shape[1])

    def pad(features: list[Tensor], mask: Tensor) -> tuple[list[Tensor], Tensor]:
        if mask.shape[1] == target:
            return features, mask
        pad_len = target - mask.shape[1]
        features = [torch.cat([feat, feat.new_zeros((feat.shape[0], pad_len, feat.shape[-1]))], dim=1) for feat in features]
        mask = torch.cat([mask, torch.zeros((mask.shape[0], pad_len), dtype=torch.bool, device=mask.device)], dim=1)
        return features, mask

    pos_features, pos_mask = pad(pos_features, pos_mask)
    neg_features, neg_mask = pad(neg_features, neg_mask)
    return pos_features, pos_mask, neg_features, neg_mask
