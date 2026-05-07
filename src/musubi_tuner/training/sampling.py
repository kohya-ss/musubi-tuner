from __future__ import annotations

from dataclasses import dataclass, field
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
    discrete_flow_shift: float = 1.0
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
