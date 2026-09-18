"""Argument groups shared by the MiniMax-H3 entry points (cache scripts, trainer, generation CLI),
so one model artifact or sampler option is spelled and documented once."""

from __future__ import annotations

import argparse

from musubi_tuner.minimax_h3.sampling import (
    DEFAULT_AUDIO_CONDITION_CLEAN,
    DEFAULT_AUDIO_SHIFT,
    DEFAULT_VIDEO_SHIFT,
    DEFAULT_VISUAL_CONDITION_CLEAN,
)


def add_h3_vae_args(parser: argparse.ArgumentParser, *, required: bool = False, note: str = "") -> None:
    """``--video_vae`` / ``--audio_vae``; ``note`` is appended to both help strings (e.g. when they are needed)."""
    suffix = f" ({note})" if note else ""
    parser.add_argument(
        "--video_vae", type=str, required=required, default=None, help=f"MiniMax-H3 video VAE safetensors path or directory{suffix}"
    )
    parser.add_argument(
        "--audio_vae", type=str, required=required, default=None, help=f"MiniMax-H3 audio VAE safetensors path or directory{suffix}"
    )


def add_h3_text_encoder_args(parser: argparse.ArgumentParser, *, required: bool = False, note: str = "") -> None:
    """``--text_encoder`` and its loading options (quantized matmul, layer streaming, attention backend)."""
    suffix = f" ({note})" if note else ""
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=required,
        default=None,
        help=f"MiniMax-H3 Qwen3-VL safetensors path (BF16, ConvRot INT8 or NVFP4, auto-detected){suffix}",
    )
    parser.add_argument(
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use W4A4 scaled_mm for an NVFP4 text encoder (requires PyTorch 2.10+ and Blackwell; default is weight-only dequantization)",
    )
    parser.add_argument(
        "--text_encoder_blocks_to_swap",
        type=int,
        default=0,
        help="number of the 50 Qwen3-VL decoder layers to stream from CPU instead of keeping them on the GPU"
        " (0 = disabled, 50 = minimum VRAM; requires CUDA; unrelated to the transformer's --blocks_to_swap)",
    )
    parser.add_argument(
        "--text_encoder_attn_mode",
        choices=("sdpa", "flash_attention_2", "eager"),
        default=None,
        help="attention implementation for the text encoder (default: transformers default, sdpa)."
        " Use flash_attention_2 for long presentations: sdpa falls back to the O(L^2) math kernel and can OOM",
    )


def add_h3_sampling_args(parser: argparse.ArgumentParser) -> None:
    """The joint AV sampler's schedule and condition-augmentation options (generation and training samples)."""
    parser.add_argument("--h3_shift_video", type=float, default=DEFAULT_VIDEO_SHIFT, help="MiniMax-H3 target-video flow shift")
    parser.add_argument("--h3_shift_audio", type=float, default=DEFAULT_AUDIO_SHIFT, help="MiniMax-H3 target-audio flow shift")
    parser.add_argument(
        "--h3_visual_cond_clean",
        type=float,
        default=DEFAULT_VISUAL_CONDITION_CLEAN,
        help="clean coefficient of the MiniMax-H3 visual condition augmentation (clean*x + (1-clean)*noise)",
    )
    parser.add_argument(
        "--h3_audio_cond_clean",
        type=float,
        default=DEFAULT_AUDIO_CONDITION_CLEAN,
        help="clean coefficient of the MiniMax-H3 audio condition augmentation (clean*x + (1-clean)*noise)",
    )
