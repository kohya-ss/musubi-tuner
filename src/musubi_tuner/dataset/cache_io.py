from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING, Union

import torch
from safetensors.torch import save_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK_FULL,
    ARCHITECTURE_FLUX_KONTEXT_FULL,
    ARCHITECTURE_HIDREAM_O1_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_FULL,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5_FULL,
    ARCHITECTURE_IDEOGRAM4_FULL,
    ARCHITECTURE_KANDINSKY5_FULL,
    ARCHITECTURE_KREA2_FULL,
    ARCHITECTURE_MINIMAX_H3_FULL,
    ARCHITECTURE_QWEN_IMAGE_FULL,
    ARCHITECTURE_WAN_FULL,
    ARCHITECTURE_Z_IMAGE_FULL,
)
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str, remove_dtype_suffix

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)


# We use simple if-else approach to support multiple architectures.
# Maybe we can use a plugin system in the future.

# the keys of the dict are `<content_type>_FxHxW_<dtype>` for latents
# and `<content_type>_<dtype|mask>` for other tensors


# Common audio conventions (audio-capable architectures):
# - the target audio latent is stored as `latents_audio_<shape>_<dtype>` (shape layout is
#   architecture-specific) in the same latent cache file
# - AUDIO_PRESENT_KEY holds a scalar 0/1 float32 tensor recording whether the item had real
#   audio (0: silence placeholder was encoded). This is a fact about the data; supervision
#   policy (loss weights, video-only training) is decided at training time.
AUDIO_PRESENT_KEY = "audio_present_float32"

# - ONE_FRAME_TARGET_INDEX_KEY holds a scalar int64 tensor with the one-frame target's 24 fps
#   pixel-frame index. It travels as a tensor (not metadata) because the bucket collator only
#   loads tensors; the trainer converts it to a RoPE time override at layout-build time.
ONE_FRAME_TARGET_INDEX_KEY = "one_frame_target_index_int64"

# - ONE_FRAME_CONTROL_INDICES_KEY holds an int64 [K] tensor (K = 1..2) with the 24 fps
#   pixel-frame indices of the one-frame visual conditions, in packed (first, last) order.
#   Present only when the cache carries condition latents; same tensor-not-metadata rationale.
ONE_FRAME_CONTROL_INDICES_KEY = "one_frame_control_indices_int64"


def append_audio_present_entry(sd: dict[str, torch.Tensor], audio_present: bool):
    sd[AUDIO_PRESENT_KEY] = torch.tensor(1.0 if audio_present else 0.0, dtype=torch.float32)


def append_one_frame_target_index_entry(sd: dict[str, torch.Tensor], target_index: int):
    if target_index < 0:
        raise ValueError(f"MiniMax-H3 one-frame target index must be nonnegative, got {target_index}")
    sd[ONE_FRAME_TARGET_INDEX_KEY] = torch.tensor(target_index, dtype=torch.int64)


def append_one_frame_control_indices_entry(sd: dict[str, torch.Tensor], control_indices: list[int]):
    if not 1 <= len(control_indices) <= 2:
        raise ValueError(f"MiniMax-H3 one-frame control indices must have 1 or 2 entries, got {len(control_indices)}")
    if any(index < 0 for index in control_indices):
        raise ValueError(f"MiniMax-H3 one-frame control indices must be nonnegative, got {control_indices}")
    sd[ONE_FRAME_CONTROL_INDICES_KEY] = torch.tensor(list(control_indices), dtype=torch.int64)


def validate_audio_present_entry(sd: dict[str, torch.Tensor]) -> float:
    """Validates the audio_present entry of a latent cache dict and returns its value."""
    tensor = sd.get(AUDIO_PRESENT_KEY)
    if not isinstance(tensor, torch.Tensor) or tensor.shape != torch.Size([]) or tensor.dtype != torch.float32:
        raise ValueError(f"Audio latent cache requires a scalar float32 {AUDIO_PRESENT_KEY} tensor")
    value = tensor.item()
    if value not in (0.0, 1.0):
        raise ValueError(f"{AUDIO_PRESENT_KEY} must be exactly 0.0 or 1.0, got {value}")
    return value


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


def save_latent_cache_krea2(item_info: ItemInfo, latent: torch.Tensor):
    """Krea 2 (K2) architecture. Single image (F=1), Qwen-Image VAE latents (normalized).

    The latent uses the *same* normalization as the Qwen-Image VAE
    (`(raw - mean) / std`), which is exactly what K2's decoder inverts, so the
    Qwen-Image latent caching is reused as-is. No control latent for plain t2i.
    """
    assert latent.dim() == 4, "latent should be 4D tensor (channel, frame, height, width)"

    _, F, H, W = latent.shape
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)


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


def save_pixel_cache_hidream_o1(
    item_info: ItemInfo, pixel_tokens: torch.Tensor, control_pixel_tokens: Optional[Union[list[torch.Tensor], torch.Tensor]] = None
):
    """HiDream-O1 architecture. Cache normalized 32x32 pixel patch tokens."""
    assert pixel_tokens.dim() == 3, "pixel_tokens should be 3D tensor (height_patches, width_patches, patch_dim)"

    height_patches, width_patches, _ = pixel_tokens.shape
    dtype_str = dtype_to_str(pixel_tokens.dtype)
    sd = {f"latents_1x{height_patches}x{width_patches}_{dtype_str}": pixel_tokens.detach().cpu().contiguous()}

    if control_pixel_tokens is not None:
        if torch.is_tensor(control_pixel_tokens):
            assert control_pixel_tokens.dim() == 4, (
                "control_pixel_tokens should be 4D tensor (num_controls, height_patches, width_patches, patch_dim)"
            )
            control_pixel_tokens = list(control_pixel_tokens)
        assert all(cl.dim() == 3 for cl in control_pixel_tokens), (
            "control_pixel_tokens should contain 3D tensors (height_patches, width_patches, patch_dim)"
        )
        for i, cl in enumerate(control_pixel_tokens):
            control_height_patches, control_width_patches, _ = cl.shape
            control_dtype_str = dtype_to_str(cl.dtype)
            sd[f"latents_control_{i}_{control_height_patches}x{control_width_patches}_{control_dtype_str}"] = (
                cl.detach().cpu().contiguous()
            )

    save_latent_cache_common(item_info, sd, ARCHITECTURE_HIDREAM_O1_FULL)


def save_latent_cache_ideogram4(item_info: ItemInfo, latent: torch.Tensor):
    """Ideogram 4 architecture."""
    assert latent.dim() == 3, "latent should be 3D tensor (channel, height, width)"

    _, H, W = latent.shape
    F = 1
    dtype_str = dtype_to_str(latent.dtype)
    sd = {f"latents_{F}x{H}x{W}_{dtype_str}": latent.detach().cpu().contiguous()}

    save_latent_cache_common(item_info, sd, ARCHITECTURE_IDEOGRAM4_FULL)


def _merge_cache_metadata(required: dict[str, str], additional: Optional[dict[str, str]]) -> dict[str, str]:
    metadata = dict(additional or {})
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items()):
        raise ValueError("Safetensors metadata keys and values must be strings")
    metadata.update(required)
    return metadata


def save_latent_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    additional_metadata: Optional[dict[str, str]] = None,
):
    metadata = _merge_cache_metadata(
        {
            "architecture": arch_fullname,
            "width": f"{item_info.original_size[0]}",
            "height": f"{item_info.original_size[1]}",
            "format_version": "1.0.1",
        },
        additional_metadata,
    )
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


def save_text_encoder_output_cache_krea2(item_info: ItemInfo, embed: torch.Tensor):
    """Krea 2 (K2) architecture.

    `embed` is the per-item stack of *selected* Qwen3-VL hidden-state layers for the
    valid (non-padding) tokens only: shape (valid_len, num_select_layers, hidden).
    Stored varlen (no padding, no mask): K2 gives text tokens zero RoPE position and
    masks padding in attention, so dropping padding is lossless for the image outputs.
    The layerwise fusion (TextFusionTransformer) is trainable and lives in the DiT, so
    the raw selected-layer stack is what gets cached.
    """
    assert embed.dim() == 3, "embed should be 3D tensor (valid_len, num_select_layers, hidden)"

    sd = {}
    dtype_str = dtype_to_str(embed.dtype)
    sd[f"varlen_krea2_vl_embed_{dtype_str}"] = embed.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_KREA2_FULL)


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


def save_text_encoder_output_cache_ideogram4(item_info: ItemInfo, features: torch.Tensor):
    """Ideogram 4 architecture."""
    sd = {}
    dtype_str = dtype_to_str(features.dtype)
    sd[f"varlen_i4_llm_features_{dtype_str}"] = features.detach().cpu()

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_IDEOGRAM4_FULL)


def save_text_encoder_output_cache_hidream_o1(
    item_info: ItemInfo,
    input_ids: torch.Tensor,
    input_embeds: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    token_types: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
):
    """HiDream-O1 architecture. Cache tokenized prompt and optional initial text token embeddings."""
    # The dtype suffix is parsed back on load (see bucket.py), so it must be built per tensor here; absent optionals
    # are simply skipped. HiDream-O1 writes its full key set in a single pass, so the cache is overwritten fresh
    # (merge_existing=False) instead of merged, dropping any stale optional/dtype keys left from a previous run.
    tensors = {
        "varlen_input_ids": input_ids,
        "varlen_input_embeds": input_embeds,
        "varlen_position_ids": position_ids,
        "varlen_token_types": token_types,
        "varlen_pixel_values": pixel_values,
        "varlen_image_grid_thw": image_grid_thw,
    }
    sd = {f"{name}_{dtype_to_str(t.dtype)}": t.detach().cpu() for name, t in tensors.items() if t is not None}

    save_text_encoder_output_cache_common(item_info, sd, ARCHITECTURE_HIDREAM_O1_FULL, merge_existing=False)


def _h3_dtype_matches(tensor: torch.Tensor, dtype_name: str) -> bool:
    return dtype_to_str(tensor.dtype) == dtype_name


def save_latent_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    metadata: Optional[dict[str, str]] = None,
):
    import re

    target_pattern = re.compile(r"^latents_(\d+)x(\d+)x(\d+)_(.+)$")
    audio_pattern = re.compile(r"^latents_audio_32x2x(\d+)_(.+)$")
    visual_condition_pattern = re.compile(r"^latents_(?:first|last|ref_\d{3}_(?:image|video))_(\d+)x(\d+)x(\d+)_(.+)$")
    audio_condition_pattern = re.compile(r"^latents_ref_\d{3}_audio_32x2x(\d+)_(.+)$")

    target_count = 0
    audio_count = 0
    normalized = {}
    for key, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"MiniMax-H3 cache value must be a tensor: {key}")
        if key == AUDIO_PRESENT_KEY:
            normalized[key] = tensor.detach().cpu().contiguous()
            continue
        if key == ONE_FRAME_TARGET_INDEX_KEY:
            if tensor.shape != torch.Size([]) or tensor.dtype != torch.int64 or tensor.item() < 0:
                raise ValueError(f"MiniMax-H3 {ONE_FRAME_TARGET_INDEX_KEY} must be a nonnegative scalar int64 tensor")
            normalized[key] = tensor.detach().cpu().contiguous()
            continue
        if key == ONE_FRAME_CONTROL_INDICES_KEY:
            if tensor.ndim != 1 or not 1 <= tensor.shape[0] <= 2 or tensor.dtype != torch.int64 or bool((tensor < 0).any()):
                raise ValueError(
                    f"MiniMax-H3 {ONE_FRAME_CONTROL_INDICES_KEY} must be a nonnegative int64 [K] tensor with K in 1..2"
                )
            normalized[key] = tensor.detach().cpu().contiguous()
            continue

        match = target_pattern.fullmatch(key)
        if match is not None:
            frames, height, width = (int(match.group(index)) for index in range(1, 4))
            if tensor.shape != (24, frames, height, width):
                raise ValueError(f"MiniMax-H3 target video latent must be [24,F,H,W], got {tuple(tensor.shape)}")
            if not _h3_dtype_matches(tensor, match.group(4)):
                raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
            target_count += 1
        else:
            match = audio_pattern.fullmatch(key)
            if match is not None:
                audio_frames = int(match.group(1))
                if tensor.shape != (32, 2, audio_frames):
                    raise ValueError(f"MiniMax-H3 audio latent [32,2,A] required, got {tuple(tensor.shape)}")
                if not _h3_dtype_matches(tensor, match.group(2)):
                    raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
                audio_count += 1
            else:
                match = visual_condition_pattern.fullmatch(key)
                if match is not None:
                    frames, height, width = (int(match.group(index)) for index in range(1, 4))
                    if tensor.shape != (24, frames, height, width):
                        raise ValueError(f"MiniMax-H3 visual condition latent must be [24,F,H,W], got {tuple(tensor.shape)}")
                    if not _h3_dtype_matches(tensor, match.group(4)):
                        raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
                else:
                    match = audio_condition_pattern.fullmatch(key)
                    if match is None:
                        raise ValueError(f"Unsupported MiniMax-H3 latent cache key: {key}")
                    audio_frames = int(match.group(1))
                    if tensor.shape != (32, 2, audio_frames):
                        raise ValueError(f"MiniMax-H3 audio latent [32,2,A] required, got {tuple(tensor.shape)}")
                    if not _h3_dtype_matches(tensor, match.group(2)):
                        raise ValueError(f"MiniMax-H3 cache key dtype does not match tensor: {key}")
        normalized[key] = tensor.detach().cpu().contiguous()

    if target_count != 1:
        raise ValueError(f"MiniMax-H3 cache requires exactly one target video latent, found {target_count}")
    if audio_count != 1:
        raise ValueError(f"MiniMax-H3 cache requires exactly one target audio latent, found {audio_count}")
    validate_audio_present_entry(normalized)
    save_latent_cache_common(item_info, normalized, ARCHITECTURE_MINIMAX_H3_FULL, metadata)


def save_text_encoder_output_cache_minimax_h3(
    item_info: ItemInfo,
    tensors: dict[str, torch.Tensor],
    metadata: Optional[dict[str, str]] = None,
):
    # the teacher prefixes must be split off before matching the student prefix, because
    # "varlen_mmh3_teacher[_ref]_hidden_states_*" does not share the student prefix; the two
    # teacher kinds (FL2VA "first,last" vs Ref2VA "ref") use distinct keys so the trainer can
    # hard-fail on a cache/flag mode mismatch instead of silently misreading the rows
    student_hidden_keys = [key for key in tensors if key.startswith("varlen_mmh3_hidden_states_")]
    teacher_hidden_keys = [key for key in tensors if key.startswith("varlen_mmh3_teacher_hidden_states_")]
    teacher_ref_hidden_keys = [key for key in tensors if key.startswith("varlen_mmh3_teacher_ref_hidden_states_")]
    if len(student_hidden_keys) != 1:
        raise ValueError(f"MiniMax-H3 text cache requires exactly one hidden-state tensor, found {len(student_hidden_keys)}")
    tags_key = "varlen_mmh3_token_tags_int64"
    teacher_tags_key = "varlen_mmh3_teacher_token_tags_int64"
    teacher_ref_tags_key = "varlen_mmh3_teacher_ref_token_tags_int64"

    has_fl_teacher = bool(teacher_hidden_keys) or teacher_tags_key in tensors
    has_ref_teacher = bool(teacher_ref_hidden_keys) or teacher_ref_tags_key in tensors
    if has_fl_teacher and has_ref_teacher:
        raise ValueError("MiniMax-H3 text cache cannot mix first,last and ref teacher rows")

    pairs = [(student_hidden_keys[0], "varlen_mmh3_hidden_states_", tags_key)]
    expected_keys = {student_hidden_keys[0], tags_key}
    if has_fl_teacher:
        if len(teacher_hidden_keys) != 1 or teacher_tags_key not in tensors:
            raise ValueError("MiniMax-H3 teacher text rows require exactly one hidden-state tensor and its token tags")
        pairs.append((teacher_hidden_keys[0], "varlen_mmh3_teacher_hidden_states_", teacher_tags_key))
        expected_keys |= {teacher_hidden_keys[0], teacher_tags_key}
    if has_ref_teacher:
        if len(teacher_ref_hidden_keys) != 1 or teacher_ref_tags_key not in tensors:
            raise ValueError("MiniMax-H3 teacher text rows require exactly one hidden-state tensor and its token tags")
        pairs.append((teacher_ref_hidden_keys[0], "varlen_mmh3_teacher_ref_hidden_states_", teacher_ref_tags_key))
        expected_keys |= {teacher_ref_hidden_keys[0], teacher_ref_tags_key}
    if set(tensors) != expected_keys:
        raise ValueError(f"MiniMax-H3 text cache requires exactly the keys {sorted(expected_keys)}")

    normalized = {}
    for hidden_key, hidden_prefix, pair_tags_key in pairs:
        hidden_states = tensors[hidden_key]
        token_tags = tensors[pair_tags_key]
        if hidden_states.ndim != 2 or hidden_states.shape[1] != 5120:
            raise ValueError(f"MiniMax-H3 hidden states must be [L,5120], got {tuple(hidden_states.shape)}")
        if not _h3_dtype_matches(hidden_states, hidden_key.removeprefix(hidden_prefix)):
            raise ValueError(f"MiniMax-H3 hidden-state key dtype does not match tensor: {hidden_key}")
        if hidden_states.shape[0] > 32768:
            raise ValueError(f"MiniMax-H3 text cache exceeds 32768 rows: {hidden_states.shape[0]}")
        if token_tags.dtype != torch.int64 or token_tags.shape != (hidden_states.shape[0],):
            raise ValueError("MiniMax-H3 token tags must be int64 [L]")
        if not torch.all((token_tags == 0) | (token_tags == 1)):
            raise ValueError("MiniMax-H3 token tags may contain only 0 and 1")
        normalized[hidden_key] = hidden_states.detach().cpu().contiguous()
        normalized[pair_tags_key] = token_tags.detach().cpu().contiguous()
    save_text_encoder_output_cache_common(
        item_info,
        normalized,
        ARCHITECTURE_MINIMAX_H3_FULL,
        merge_existing=False,
        additional_metadata=metadata,
    )


def save_text_encoder_output_cache_common(
    item_info: ItemInfo,
    sd: dict[str, torch.Tensor],
    arch_fullname: str,
    merge_existing: bool = True,
    additional_metadata: Optional[dict[str, str]] = None,
):
    # merge_existing keeps keys written by previous passes (e.g. HunyuanVideo caches LLM and CLIP separately).
    # Single-pass architectures that write their full key set at once should pass merge_existing=False so the
    # cache is overwritten fresh, dropping any stale keys (e.g. optionals/dtypes) left from an earlier run.
    for key, value in sd.items():
        # NaN check and show warning, replace NaN with 0
        if torch.isnan(value).any():
            logger.warning(f"{key} tensor has NaN: {item_info.item_key}, replace NaN with 0")
            value[torch.isnan(value)] = 0

    metadata = _merge_cache_metadata(
        {
            "architecture": arch_fullname,
            "caption1": item_info.caption,
            "format_version": "1.0.1",
        },
        additional_metadata,
    )
    if merge_existing and os.path.exists(item_info.text_encoder_output_cache_path):
        # load existing cache and update metadata
        new_key_bases = {remove_dtype_suffix(key) for key in sd}  # logical keys (dtype stripped) just written
        with safetensors_utils.MemoryEfficientSafeOpen(item_info.text_encoder_output_cache_path) as f:
            existing_metadata = f.metadata()
            for key in f.keys():
                # Skip any existing key superseded by a freshly written one. Comparing on the dtype-stripped base
                # (not the exact key) also drops a stale copy written in another precision, e.g. re-caching after
                # toggling fp8; otherwise both dtype variants would survive and collide under one key on load.
                if remove_dtype_suffix(key) in new_key_bases:
                    continue
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
