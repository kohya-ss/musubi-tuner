from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK_FULL,
    ARCHITECTURE_FLUX_KONTEXT_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    ARCHITECTURE_KANDINSKY5_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_WAN_FULL,
    ARCHITECTURE_Z_IMAGE_FULL,
)
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


def save_latent_cache(item_info: ItemInfo, latent: torch.Tensor):
    """HunyuanVideo architecture. HunyuanVideo doesn't support I2V and control latents"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_latent_cache_wan(
    item_info: ItemInfo,
    latent: torch.Tensor,
    clip_embed: Optional[torch.Tensor],
    image_latent: Optional[torch.Tensor],
    control_latent: Optional[torch.Tensor],
    f_indices: Optional[list[int]] = None,
):
    """Wan architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if clip_embed is not None:
        sd[f"clip_{dtype_str}"] = clip_embed.detach().cpu()

    if image_latent is not None:
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if control_latent is not None:
        sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu()

    if f_indices is not None:
        dtype_str = dtype_to_str(torch.int32)
        sd[f"f_indices_{dtype_str}"] = torch.tensor(f_indices, dtype=torch.int32)

    save_latent_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_latent_cache_framepack(
    item_info: ItemInfo,
    latent: torch.Tensor,
    latent_indices: torch.Tensor,
    clean_latents: torch.Tensor,
    clean_latent_indices: torch.Tensor,
    clean_latents_2x: torch.Tensor,
    clean_latent_2x_indices: torch.Tensor,
    clean_latents_4x: torch.Tensor,
    clean_latent_4x_indices: torch.Tensor,
    image_embeddings: torch.Tensor,
):
    """FramePack architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    # `latents_xxx` must have {F, H, W} suffix
    indices_dtype_str = dtype_to_str(latent_indices.dtype)
    sd[f"image_embeddings_{dtype_str}"] = image_embeddings.detach().cpu()  # image embeddings dtype is same as latents dtype
    sd[f"latent_indices_{indices_dtype_str}"] = latent_indices.detach().cpu()
    sd[f"clean_latent_indices_{indices_dtype_str}"] = clean_latent_indices.detach().cpu()
    sd[f"latents_clean_{F}x{H}x{W}_{dtype_str}"] = clean_latents.detach().cpu().contiguous()
    if clean_latent_2x_indices is not None:
        sd[f"clean_latent_2x_indices_{indices_dtype_str}"] = clean_latent_2x_indices.detach().cpu()
    if clean_latents_2x is not None:
        sd[f"latents_clean_2x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_2x.detach().cpu().contiguous()
    if clean_latent_4x_indices is not None:
        sd[f"clean_latent_4x_indices_{indices_dtype_str}"] = clean_latent_4x_indices.detach().cpu()
    if clean_latents_4x is not None:
        sd[f"latents_clean_4x_{F}x{H}x{W}_{dtype_str}"] = clean_latents_4x.detach().cpu().contiguous()

    # for key, value in sd.items():
    #     print(f"{key}: {value.shape}")
    save_latent_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_latent_cache_flux_kontext(
    item_info: ItemInfo,
    latent: torch.Tensor,
    control_latent: torch.Tensor,
):
    """FLUX.1 Kontext architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    _, H, W = control_latent.shape
    F = 1
    sd[f"latents_control_{F}x{H}x{W}_{dtype_str}"] = control_latent.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_latent_cache_flux_2(
    item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]], arch_full: str
):
    """Flux 2 architecture"""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"
    assert control_latent is None or all(cl.dim() == 3 for cl in control_latent), (
        "control_latent should be 3D tensor (channel, height, width) or None"
    )

    _, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, H, W = cl.shape
            sd[f"latents_control_{i}_{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, arch_full)


def save_latent_cache_qwen_image(item_info: ItemInfo, latent: torch.Tensor, control_latent: Optional[list[torch.Tensor]]):
    """Qwen-Image architecture"""
    assert latent.dim() == 4, "latent should be 4D tensor (frame, channel, height, width)"
    assert control_latent is None or all(cl.dim() == 4 for cl in control_latent), (
        "control_latent should be 4D tensor (frame, channel, height, width) or None"
    )

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    if control_latent is not None:
        for i, cl in enumerate(control_latent):
            _, F, H, W = cl.shape
            sd[f"latents_control_{i}_{F}x{H}x{W}_{dtype_str}"] = cl.detach().cpu().contiguous()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_latent_cache_kandinsky5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor] = None,
    control_latent: Optional[torch.Tensor] = None,
    scaling_factor: Optional[float] = None,
):
    """Kandinsky 5 architecture (image/video), with optional source/control latents for i2v/control."""
    assert latent.dim() == 3 or latent.dim() == 4, "latent should be 3D (C,H,W) or 4D (F,C,H,W) tensor"

    if latent.dim() == 4:
        _, F, H, W = latent.shape
    else:
        F, H, W = 1, latent.shape[1], latent.shape[2]
        latent = latent.unsqueeze(0)
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous().clone()}

    if image_latent is not None:
        _, F_img, H_img, W_img = image_latent.shape
        sd[f"latents_image_{F_img}x{H_img}x{W_img}_{dtype_str}"] = image_latent.detach().cpu().contiguous().clone()

    if control_latent is not None:
        _, F_ctrl, H_ctrl, W_ctrl = control_latent.shape
        sd[f"latents_control_{F_ctrl}x{H_ctrl}x{W_ctrl}_{dtype_str}"] = control_latent.detach().cpu().contiguous().clone()

    if scaling_factor is not None:
        sd["vae_scaling_factor"] = torch.tensor(float(scaling_factor))

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_latent_cache_hunyuan_video_1_5(
    item_info: ItemInfo,
    latent: torch.Tensor,
    image_latent: Optional[torch.Tensor],
    vision_feature: Optional[torch.Tensor],
):
    """HunyuanVideo 1.5 architecture"""
    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd: dict[str, torch.Tensor] = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu()}

    if image_latent is not None:
        dtype_str = dtype_to_str(image_latent.dtype)
        _, F, H, W = image_latent.shape
        sd[f"latents_image_{F}x{H}x{W}_{dtype_str}"] = image_latent.detach().cpu()

    if vision_feature is not None:
        dtype_str = dtype_to_str(vision_feature.dtype)
        sd[f"siglip_{dtype_str}"] = vision_feature.detach().cpu()

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_latent_cache_z_image(item_info: ItemInfo, latent: torch.Tensor):
    """Z-Image architecture. No control latent is supported."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    C, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_latent_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    metadata = {
        "architecture": arch_fullname,
        "width": f"{item_info.original_size[0]}",
        "height": f"{item_info.original_size[1]}",
        "format_version": "1.0.1",
    }
    if item_info.frame_count is not None:
        metadata["frame_count"] = f"{item_info.frame_count}"

    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    latent_dir = os.path.dirname(item_info.latent_cache_path)
    os.makedirs(latent_dir, exist_ok=True)

    save_file(sd, item_info.latent_cache_path, metadata=metadata)


def save_text_encoder_output_cache(item_info: ItemInfo, embed: torch.Tensor, mask: Optional[torch.Tensor], is_llm: bool):
    """HunyuanVideo architecture"""
    assert embed.dim() == 1 or embed.dim() == 2, (
        f"embed should be 2D tensor (feature, hidden_size) or (hidden_size,), got {embed.shape}"
    )
    assert mask is None or mask.dim() == 1, f"mask should be 1D tensor (feature), got {mask.shape}"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "llm" if is_llm else "clipL"
    sd[f"{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()
    if mask is not None:
        sd[f"{text_encoder_type}_mask"] = mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_FULL)


def save_text_encoder_output_cache_wan(item_info: ItemInfo, embed: torch.Tensor):
    """Wan architecture. Wan2.1 only has a single text encoder"""

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    text_encoder_type = "t5"
    sd[f"varlen_{text_encoder_type}_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_WAN_FULL)


def save_text_encoder_output_cache_framepack(
    item_info: ItemInfo, llama_vec: torch.Tensor, llama_attention_mask: torch.Tensor, clip_l_pooler: torch.Tensor
):
    """FramePack architecture."""
    sd = {}
    dtype_str = dtype_to_str(llama_vec.dtype)
    sd[f"llama_vec_{dtype_str}"] = llama_vec.detach().cpu()
    sd["llama_attention_mask"] = llama_attention_mask.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FRAMEPACK_FULL)


def save_text_encoder_output_cache_flux_kontext(item_info: ItemInfo, t5_vec: torch.Tensor, clip_l_pooler: torch.Tensor):
    """Flux Kontext architecture."""

    sd = {}
    dtype_str = dtype_to_str(t5_vec.dtype)
    sd[f"t5_vec_{dtype_str}"] = t5_vec.detach().cpu()
    dtype_str = dtype_to_str(clip_l_pooler.dtype)
    sd[f"clip_l_pooler_{dtype_str}"] = clip_l_pooler.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_FLUX_KONTEXT_FULL)


def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, ctx_vec: torch.Tensor, arch_full: str):
    """Flux 2 architecture."""

    sd = {}
    dtype_str = dtype_to_str(ctx_vec.dtype)
    sd[f"ctx_vec_{dtype_str}"] = ctx_vec.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, arch_full)


def save_text_encoder_output_cache_qwen_image(item_info: ItemInfo, embed: torch.Tensor):
    """Qwen-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_QWEN_IMAGE_FULL)


def save_text_encoder_output_cache_kandinsky5(
    item_info: ItemInfo, text_embeds: torch.Tensor, pooled_embed: torch.Tensor, attention_mask: torch.Tensor
):
    """Kandinsky 5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(text_embeds.dtype)
    sd[f"text_embeds_{dtype_str}"] = text_embeds.detach().cpu()
    dtype_str = dtype_to_str(pooled_embed.dtype)
    sd[f"pooled_embed_{dtype_str}"] = pooled_embed.detach().cpu()
    sd["attention_mask"] = attention_mask.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KANDINSKY5_FULL)


def save_text_encoder_output_cache_hunyuan_video_1_5(item_info: ItemInfo, embed: torch.Tensor, byt5_embed: torch.Tensor):
    """Hunyuan-Video 1.5 architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_vl_embed_{dtype_str}"] = embed.detach().cpu()
    dtype_str = dtype_to_str(byt5_embed.dtype)
    sd[f"varlen_byt5_embed_{dtype_str}"] = byt5_embed.detach().cpu()
    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL)


def save_text_encoder_output_cache_z_image(item_info: ItemInfo, embed: torch.Tensor):
    """Z-Image architecture."""
    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_llm_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_Z_IMAGE_FULL)


def save_text_encoder_output_cache_common(item_info: ItemInfo, sd: dict[str, torch.Tensor], arch_fullname: str):
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = {
        "architecture": arch_fullname,
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }

    if os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                if key not in sd:  # avoid overwriting by existing cache, we keep the new one
                    sd[key] = f.get_tensor(key)

        assert existing_metadata["architecture"] == metadata["architecture"], "architecture mismatch"
        if existing_metadata["caption1"] != metadata["caption1"]:
            logger.warning(f"caption mismatch: existing={existing_metadata['caption1']}, new={metadata['caption1']}, overwrite")
        # TODO verify format_version

        existing_metadata.pop("caption1", None)
        existing_metadata.pop("format_version", None)
        metadata.update(existing_metadata)  # copy existing metadata except caption and format_version
    else:
        text_encoder_output_dir = os.path.dirname(item_info.text_encoder_output_cache_path)
        os.makedirs(text_encoder_output_dir, exist_ok=True)

    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)
