from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from accelerate import Accelerator

if TYPE_CHECKING:
    import argparse


@dataclass
class SamplePrompt:
    # required
    prompt: str
    width: int
    height: int
    # common optional
    frame_count: int = 1
    sample_steps: int = 20
    seed: int | None = None
    guidance_scale: float = 5.0
    discrete_flow_shift: float | None = None
    cfg_scale: float | None = None
    negative_prompt: str | None = None
    enum: int = 0
    # precomputed text embeddings (populated by process_sample_prompts)
    ctx_vec: torch.Tensor | None = None
    negative_ctx_vec: torch.Tensor | None = None
    # architecture-specific (flat, all optional)
    image_path: str | None = None
    control_video_path: str | None = None
    control_image_path: list[str] | None = None
    # HunyuanVideo-specific embeddings
    llm_embeds: torch.Tensor | None = None
    llm_mask: torch.Tensor | None = None
    clipL_embeds: torch.Tensor | None = None
    clipL_mask: torch.Tensor | None = None
    negative_llm_embeds: torch.Tensor | None = None
    negative_llm_mask: torch.Tensor | None = None
    negative_clipL_embeds: torch.Tensor | None = None
    negative_clipL_mask: torch.Tensor | None = None
    # Wan-specific embeddings
    t5_embeds: torch.Tensor | None = None
    negative_t5_embeds: torch.Tensor | None = None
    clip_embeds: torch.Tensor | None = None
    end_image_clip_embeds: torch.Tensor | None = None
    # Wan-specific fields
    one_frame: str | None = None
    # ZImage-specific embeddings
    cap_feats: torch.Tensor | None = None
    cap_mask: torch.Tensor | None = None
    negative_cap_feats: torch.Tensor | None = None
    negative_cap_mask: torch.Tensor | None = None
    # Qwen-Image-specific embeddings
    vl_embed: torch.Tensor | None = None
    negative_vl_embed: torch.Tensor | None = None
    control_image_tensors: list[torch.Tensor] | None = None
    # FLUX Kontext-specific embeddings
    t5_vec: torch.Tensor | None = None
    clip_l_pooler: torch.Tensor | None = None
    # FramePack-specific embeddings
    llama_vec: torch.Tensor | None = None
    llama_attention_mask: torch.Tensor | None = None
    negative_llama_vec: torch.Tensor | None = None
    negative_llama_attention_mask: torch.Tensor | None = None
    negative_clip_l_pooler: torch.Tensor | None = None
    image_encoder_last_hidden_state: torch.Tensor | None = None


@dataclass
class SamplingContext:
    accelerator: Accelerator
    args: argparse.Namespace
    epoch: int | None
    steps: int
    vae: torch.nn.Module
    transformer: torch.nn.Module
    network: torch.nn.Module
    sample_prompts: list[SamplePrompt]
    dit_dtype: torch.dtype
