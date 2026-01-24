from __future__ import annotations

import argparse
from typing import Any, Literal

import torch


MaskLossLayout = Literal["video", "layered"]


def add_mask_loss_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--use_mask_loss",
        action="store_true",
        help="Enable mask-weighted loss training. Requires alpha_mask and/or mask_directory in dataset config. "
        "White regions (255) get full training weight, black regions (0) are ignored. "
        "/ マスク重み付き損失学習を有効にする。データセット設定でalpha_maskおよび/またはmask_directoryが必要。"
        "白い領域(255)は完全な学習重みを取得し、黒い領域(0)は無視される。",
    )
    parser.add_argument(
        "--mask_min_weight",
        type=float,
        default=0.0,
        help="Minimum weight for masked-out regions (default: 0.0). Set to 0.1-0.2 to give some training signal to background regions. "
        "/ マスク外領域の最小重み（デフォルト：0.0）。0.1-0.2に設定すると背景領域にもある程度の学習シグナルを与える。",
    )
    parser.add_argument(
        "--mask_gamma",
        type=float,
        default=1.0,
        help="Gamma correction for mask weights (default: 1.0). Values < 1.0 soften the mask (more midtones, gradual falloff). "
        "Values > 1.0 sharpen the mask (more binary, stronger face focus). Try 0.5-0.7 for softer or 1.5-2.0 for sharper. "
        "/ マスク重みのガンマ補正（デフォルト：1.0）。1.0未満はマスクを柔らかくし、1.0超はマスクを鋭くして顔への集中を強める。",
    )


def validate_mask_loss_args(args: argparse.Namespace) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    # Back-compat: old configs may still include this key via read_config_from_file()'s Namespace merge.
    if hasattr(args, "mask_loss_scale") and args.mask_loss_scale is not None:
        try:
            mask_loss_scale = float(args.mask_loss_scale)
        except Exception as e:  # noqa: BLE001
            raise ValueError("--mask_loss_scale must be a number") from e

        if mask_loss_scale != 1.0:
            raise ValueError(
                "--mask_loss_scale has been removed (it had no effect with weighted-mean normalization). "
                "Use --mask_gamma and/or --mask_min_weight instead."
            )
        # mask_loss_scale == 1.0 is treated as a no-op for back-compat with old configs.

    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    if mask_gamma <= 0:
        raise ValueError("--mask_gamma must be > 0")

    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
    if mask_min_weight < 0 or mask_min_weight >= 1.0:
        raise ValueError("--mask_min_weight must be in range [0, 1)")


def log_mask_loss_banner(logger: Any, args: argparse.Namespace, cache_hint: str | None = None) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    logger.info("=" * 60)
    logger.info("MASK-WEIGHTED LOSS TRAINING ENABLED")
    logger.info("=" * 60)
    logger.info(f"  mask_min_weight: {getattr(args, 'mask_min_weight', None)}")
    logger.info(f"  mask_gamma: {getattr(args, 'mask_gamma', None)}")
    logger.info("-" * 60)
    logger.info("IMPORTANT: Masks must be baked into latent cache!")
    if cache_hint:
        logger.info(cache_hint)
    logger.info("=" * 60)


def require_mask_weights_if_enabled(batch: dict[str, Any], args: argparse.Namespace, cache_hint: str | None = None) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    if batch.get("mask_weights", None) is not None:
        return

    message = [
        "FATAL: --use_mask_loss is enabled but batch has no mask_weights!",
        "This means masks were NOT baked into your latent cache.",
        "To fix:",
        "  1. Add 'alpha_mask = true' and/or 'mask_directory = \"/path/to/masks\"' in dataset TOML",
        "  2. Use a FRESH cache_directory (masks are stored in cache)",
    ]
    if cache_hint:
        message.append(f"  3. {cache_hint}")
    else:
        message.append("  3. Recache latents with the appropriate cache script")
    message.append("  4. Then re-run training")

    raise ValueError("\n".join(message))


def apply_masked_loss(
    loss: torch.Tensor,
    mask_weights: torch.Tensor | None,
    *,
    args: argparse.Namespace,
    layout: MaskLossLayout = "video",
    drop_base_frame: bool = False,
    accelerator: Any | None = None,
) -> torch.Tensor:
    del accelerator  # reserved for future global reduction support

    if mask_weights is None or not getattr(args, "use_mask_loss", False):
        # Return fp32 for consistency with masked path (avoids dtype depending on --use_mask_loss)
        return loss.float().mean()

    # Handle both 4D (FLUX.2 images) and 5D (video) loss tensors
    if loss.ndim == 4:
        # FLUX.2 produces per-image loss (B, C, H, W); treat it as F=1 for layout='video'
        if layout != "video":
            raise ValueError("4D loss is only supported for layout='video'")
        loss = loss.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
    elif loss.ndim != 5:
        raise ValueError(f"Expected loss to be 4D or 5D, got {loss.ndim}D: {tuple(loss.shape)}")

    if drop_base_frame and layout != "layered":
        raise ValueError("drop_base_frame=True is only valid with layout='layered'")

    if mask_weights.ndim == 4:
        # (B, F, H, W) -> (B, 1, F, H, W)
        mask_weights = mask_weights.unsqueeze(1)
    elif mask_weights.ndim != 5:
        raise ValueError(f"Unexpected mask_weights shape: {tuple(mask_weights.shape)}")

    mask_weights = mask_weights.to(loss.device, dtype=loss.dtype)

    if layout == "video":
        # loss: (B, C, F, H, W), mask: (B, 1, F, H, W)
        if mask_weights.shape[0] != loss.shape[0] or mask_weights.shape[2:] != loss.shape[2:]:
            raise ValueError(
                "mask_weights shape does not match loss shape for layout='video': "
                f"mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)}"
            )
        mask_weights = mask_weights.expand_as(loss)
    elif layout == "layered":
        # loss: (B, L, C, H, W), mask: (B, 1, F, H, W) where F == (base + layers) or F == L
        if drop_base_frame:
            mask_weights = mask_weights[:, :, 1:, :, :]
        if mask_weights.shape[0] != loss.shape[0] or mask_weights.shape[2] != loss.shape[1] or mask_weights.shape[3:] != loss.shape[3:]:
            raise ValueError(
                "mask_weights shape does not match loss shape for layout='layered': "
                f"mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)} drop_base_frame={drop_base_frame}"
            )
        mask_weights = mask_weights.permute(0, 2, 1, 3, 4)  # (B, L, 1, H, W)
        mask_weights = mask_weights.expand_as(loss)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
    if mask_gamma <= 0:
        raise ValueError("--mask_gamma must be > 0")
    if mask_min_weight < 0 or mask_min_weight >= 1.0:
        raise ValueError("--mask_min_weight must be in range [0, 1)")

    # Ensure numeric stability before pow. Masks are expected to be [0, 1], but interpolation/IO can introduce tiny drift.
    mask_weights = mask_weights.clamp(0.0, 1.0)
    if mask_gamma != 1.0:
        mask_weights = mask_weights**mask_gamma
    if mask_min_weight > 0:
        mask_weights = mask_weights * (1.0 - mask_min_weight) + mask_min_weight

    weighted_loss = loss * mask_weights

    # Compute sums in float32 for numerical stability (1e-8 rounds to 0 in fp16)
    # Return float32 loss to avoid precision loss in the scalar result
    loss_sum = weighted_loss.sum(dtype=torch.float32)
    weight_sum = mask_weights.sum(dtype=torch.float32).clamp_min(1e-8)
    return loss_sum / weight_sum
