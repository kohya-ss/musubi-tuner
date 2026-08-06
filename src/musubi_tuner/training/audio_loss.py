from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from safetensors import safe_open
import torch

from musubi_tuner.dataset.cache_io import AUDIO_PRESENT_KEY, validate_audio_present_entry

import logging

logger = logging.getLogger(__name__)


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


def _read_audio_present(path: Path) -> float:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if AUDIO_PRESENT_KEY not in handle.keys():
            raise ValueError(f"Latent cache {path} has no {AUDIO_PRESENT_KEY} entry; re-run latent caching")
        tensor = handle.get_tensor(AUDIO_PRESENT_KEY)
    return validate_audio_present_entry({AUDIO_PRESENT_KEY: tensor})


def scan_audio_supervised_fraction(dataset_group) -> float:
    """Fraction of training items (num_repeats-weighted) whose cache holds real audio."""
    cache_counts: Counter[Path] = Counter()
    for dataset in dataset_group.datasets:
        for bucket in dataset.batch_manager.buckets.values():
            for item in bucket:
                cache_path = getattr(item, "latent_cache_path", None)
                if not cache_path:
                    raise ValueError("Training item is missing latent_cache_path")
                cache_counts[Path(cache_path).resolve()] += 1
    if not cache_counts:
        raise ValueError("Audio supervision scan found no latent cache files")

    supervised = 0
    total = sum(cache_counts.values())
    for path, repeats in cache_counts.items():
        supervised += repeats * int(_read_audio_present(path))
    return supervised / total


def log_audio_supervision_summary(supervised_fraction: float, args: argparse.Namespace):
    logger.info(f"supervised_audio_fraction={supervised_fraction:.6f}")
    if args.video_only:
        logger.info("audio supervision disabled by --video_only")
    elif args.audio_loss_weight > 0 and supervised_fraction == 0.0:
        logger.warning(
            "No training item has real audio, so the audio loss is always 0; "
            "if this is intended, consider passing --video_only explicitly"
        )
