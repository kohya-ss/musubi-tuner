from __future__ import annotations

import argparse

import torch


def add_audio_train_args(parser: argparse.ArgumentParser):
    """Adds common training arguments for audio-capable architectures (opt-in per trainer)."""
    parser.add_argument(
        "--video_only",
        action="store_true",
        help="disable audio supervision entirely (audio loss weight is 0 for all samples)",
    )
    parser.add_argument(
        "--audio_loss_weight",
        type=float,
        default=1.0,
        help="scale for the audio loss term; applies only to samples cached with real audio",
    )


def effective_audio_loss_weights(audio_present: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Per-sample audio loss weights: user policy x cached audio presence.

    Samples cached from silence placeholders (audio_present=0) are never supervised, which
    prevents training audio generation toward silence.
    """
    if not torch.isfinite(audio_present).all().item():
        raise ValueError("audio_present must be finite")
    if not ((audio_present == 0.0) | (audio_present == 1.0)).all().item():
        raise ValueError("audio_present must be exactly 0.0 or 1.0 per sample")
    if args.video_only:
        return torch.zeros_like(audio_present)
    if args.audio_loss_weight < 0:
        raise ValueError(f"audio_loss_weight must be nonnegative, got {args.audio_loss_weight}")
    return args.audio_loss_weight * audio_present
