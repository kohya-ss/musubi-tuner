from __future__ import annotations

import argparse
import logging
from typing import Any, Literal

import torch


MaskLossLayout = Literal["video", "layered"]

# Module-level logger for validation warnings
_logger = logging.getLogger(__name__)


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
        "NOTE: When using --prior_preservation_weight, recommend 0.0 for best results. "
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

    # Prior preservation arguments
    parser.add_argument(
        "--prior_preservation_weight",
        type=float,
        default=0.0,
        help="Weight for prior preservation loss in unmasked regions (default: 0.0 = disabled). "
        "When enabled, unmasked regions are trained to match base model predictions, preventing "
        "phantom limbs and background hallucinations. Recommended: 0.5-1.0. "
        "NOTE: Recommend --mask_min_weight 0.0 when using this. Requires LoRA training. "
        "/ マスク外領域での事前保存損失の重み（デフォルト：0.0=無効）。",
    )
    parser.add_argument(
        "--prior_mask_threshold",
        type=float,
        default=None,
        help="Optional: Apply prior preservation only where RAW mask < threshold (before gamma/min_weight). "
        "Default: None (continuous mode - prior preservation scales with inverse mask). "
        "Set to 0.05-0.1 to preserve only true background while body/hair still train to target. "
        "/ 事前保存を適用するマスクしきい値（オプション）。",
    )
    parser.add_argument(
        "--normalize_per_sample",
        action="store_true",
        help="Normalize loss per-sample before averaging across batch (default: global normalization). "
        "Recommended when prior preservation is enabled for more predictable behavior. "
        "/ サンプルごとに損失を正規化してからバッチ全体で平均する。",
    )


def validate_mask_loss_args(args: argparse.Namespace) -> None:
    use_mask_loss = bool(getattr(args, "use_mask_loss", False))

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

    # Prior preservation validation
    prior_preservation_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    if prior_preservation_weight < 0:
        raise ValueError("--prior_preservation_weight must be >= 0")

    if prior_preservation_weight > 0:
        if not use_mask_loss:
            _logger.warning(
                "--prior_preservation_weight > 0 but --use_mask_loss is not enabled. "
                "Prior preservation requires masked training."
            )

    prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
    if prior_mask_threshold is not None:
        if prior_mask_threshold <= 0 or prior_mask_threshold >= 1:
            raise ValueError("--prior_mask_threshold must be in range (0, 1)")
        if prior_preservation_weight <= 0:
            _logger.warning(
                f"--prior_mask_threshold={prior_mask_threshold} has no effect without --prior_preservation_weight > 0"
            )

    if not use_mask_loss:
        return

    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    if mask_gamma <= 0:
        raise ValueError("--mask_gamma must be > 0")

    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
    if mask_min_weight < 0 or mask_min_weight >= 1.0:
        raise ValueError("--mask_min_weight must be in range [0, 1)")

    if prior_preservation_weight > 0 and mask_min_weight > 0:
        _logger.warning(
            f"--prior_preservation_weight={prior_preservation_weight} with --mask_min_weight={mask_min_weight}: "
            "Non-zero mask_min_weight reduces prior preservation effect. Recommend --mask_min_weight 0.0"
        )


def log_mask_loss_banner(logger: Any, args: argparse.Namespace, cache_hint: str | None = None) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    prior_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    prior_threshold = getattr(args, "prior_mask_threshold", None)
    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
    normalize_per_sample = getattr(args, "normalize_per_sample", False)

    logger.info("=" * 60)
    if prior_weight > 0:
        logger.info("MASKED PRIOR PRESERVATION TRAINING ENABLED")
    else:
        logger.info("MASK-WEIGHTED LOSS TRAINING ENABLED")
    logger.info("=" * 60)
    logger.info(f"  mask_min_weight: {mask_min_weight}")
    logger.info(f"  mask_gamma: {getattr(args, 'mask_gamma', 1.0)}")
    if prior_weight > 0:
        logger.info(f"  prior_preservation_weight: {prior_weight}")
        if prior_threshold is not None:
            logger.info(f"  prior_mask_threshold: {prior_threshold} (threshold mode)")
        else:
            logger.info("  prior_mask_threshold: None (continuous mode)")
        logger.info(f"  normalize_per_sample: {normalize_per_sample}")
        logger.info("-" * 60)
        logger.info("PRIOR PRESERVATION: Unmasked regions will match base model.")
        logger.info("                    Expect ~1.3-1.7x training time.")
        if mask_min_weight > 0:
            logger.warning(f"  NOTE: mask_min_weight={mask_min_weight} reduces prior effect.")
            logger.warning("        Recommend --mask_min_weight 0.0 with prior preservation.")
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
        if (
            mask_weights.shape[0] != loss.shape[0]
            or mask_weights.shape[2] != loss.shape[1]
            or mask_weights.shape[3:] != loss.shape[3:]
        ):
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


def _prepare_tensors(
    loss: torch.Tensor,
    mask_weights: torch.Tensor,
    layout: MaskLossLayout,
    drop_base_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare loss and mask tensors for computation.
    Handles 4D/5D tensors and layout-specific transformations.

    Args:
        loss: Loss tensor (B, C, H, W) or (B, C, F, H, W) for video, or (B, L, C, H, W) for layered
        mask_weights: Mask tensor (B, F, H, W) or (B, 1, F, H, W)
        layout: "video" or "layered"
        drop_base_frame: Whether to drop base frame for layered layout

    Returns:
        Tuple of (loss, mask_weights) both as 5D tensors expanded to match
    """
    # Handle 4D vs 5D loss
    if loss.ndim == 4:
        if layout != "video":
            raise ValueError("4D loss is only supported for layout='video'")
        loss = loss.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
    elif loss.ndim != 5:
        raise ValueError(f"Expected loss to be 4D or 5D, got {loss.ndim}D: {tuple(loss.shape)}")

    if drop_base_frame and layout != "layered":
        raise ValueError("drop_base_frame=True is only valid with layout='layered'")

    # Handle mask dimensions
    if mask_weights.ndim == 4:
        mask_weights = mask_weights.unsqueeze(1)  # (B, F, H, W) -> (B, 1, F, H, W)
    elif mask_weights.ndim != 5:
        raise ValueError(f"Unexpected mask_weights shape: {tuple(mask_weights.shape)}")

    mask_weights = mask_weights.to(loss.device, dtype=loss.dtype)

    # Layout-specific handling
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
        if (
            mask_weights.shape[0] != loss.shape[0]
            or mask_weights.shape[2] != loss.shape[1]
            or mask_weights.shape[3:] != loss.shape[3:]
        ):
            raise ValueError(
                "mask_weights shape does not match loss shape for layout='layered': "
                f"mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)} drop_base_frame={drop_base_frame}"
            )
        mask_weights = mask_weights.permute(0, 2, 1, 3, 4)  # (B, L, 1, H, W)
        mask_weights = mask_weights.expand_as(loss)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    return loss, mask_weights


def apply_masked_loss_with_prior(
    loss: torch.Tensor,
    mask_weights: torch.Tensor | None,
    *,
    prior_loss_unreduced: torch.Tensor | None = None,
    args: argparse.Namespace,
    layout: MaskLossLayout = "video",
    drop_base_frame: bool = False,
) -> torch.Tensor:
    """
    Apply masked loss with optional prior preservation.

    Uses region-normalized means + explicit weighting:
        L_target = weighted_mean(mse, mask_processed)
        L_prior  = weighted_mean(mse, prior_mask) * w_prior
        loss = L_target + L_prior

    This ensures w_prior acts as a true independent knob.

    Args:
        loss: Unreduced loss tensor (B, C, F, H, W) or (B, C, H, W)
        mask_weights: Mask weights tensor, or None to use uniform weights
        prior_loss_unreduced: Unreduced prior loss tensor (same shape as loss), or None
        args: Namespace with mask_gamma, mask_min_weight, prior_preservation_weight,
              prior_mask_threshold, normalize_per_sample
        layout: "video" or "layered"
        drop_base_frame: Whether to drop base frame for layered layout

    Returns:
        Scalar loss tensor (float32)
    """
    prior_preservation_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    normalize_per_sample = getattr(args, "normalize_per_sample", False)

    # If no mask or mask loss disabled, fall back to simple mean
    if mask_weights is None or not getattr(args, "use_mask_loss", False):
        return loss.float().mean()

    # Handle tensor shapes
    loss, mask_weights = _prepare_tensors(loss, mask_weights, layout, drop_base_frame)

    # Keep raw mask for thresholding (before gamma/min_weight)
    mask_raw = mask_weights.clamp(0.0, 1.0)

    # Apply gamma and min_weight to get processed mask for target loss
    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))

    if mask_gamma <= 0:
        raise ValueError("--mask_gamma must be > 0")
    if mask_min_weight < 0 or mask_min_weight >= 1.0:
        raise ValueError("--mask_min_weight must be in range [0, 1)")

    mask_processed = mask_raw.clone()
    if mask_gamma != 1.0:
        mask_processed = mask_processed**mask_gamma
    if mask_min_weight > 0:
        mask_processed = mask_processed * (1.0 - mask_min_weight) + mask_min_weight

    # Compute prior mask (complement of processed mask)
    prior_mask = 1 - mask_processed

    # Optional: threshold on RAW mask (before gamma/min_weight)
    prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
    if prior_mask_threshold is not None:
        # Binarize: full prior preservation where raw mask < threshold
        prior_mask = (mask_raw < prior_mask_threshold).float()
        # Prevent target/prior overlap: zero out target where prior applies
        mask_processed = mask_processed * (1 - prior_mask)

    # === Target Loss (inside mask) ===
    target_loss_weighted = loss * mask_processed
    # All sums in float32 for numerical stability (1e-8 meaningless in fp16)
    target_weight_sum = mask_processed.sum(dtype=torch.float32)

    if normalize_per_sample:
        # Per-sample weighted mean, then average over batch
        # Reduce over C, F, H, W dimensions (keep batch)
        reduce_dims = tuple(range(1, loss.ndim))
        target_sum = target_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)
        target_weight = mask_processed.sum(dim=reduce_dims, dtype=torch.float32)
        # Handle samples with zero target weight: treat as 0 contribution
        valid_target = target_weight > 1e-8
        per_sample_target = torch.where(valid_target, target_sum / target_weight.clamp_min(1e-8), torch.zeros_like(target_sum))
        L_target = per_sample_target.mean()
    else:
        # Global weighted mean
        if target_weight_sum < 1e-8:
            L_target = loss.new_zeros((), dtype=torch.float32)
        else:
            L_target = target_loss_weighted.sum(dtype=torch.float32) / target_weight_sum

    # === Prior Loss (outside mask) ===
    if prior_preservation_weight > 0 and prior_loss_unreduced is not None:
        # Check if prior mask is effectively all zeros (skip computation)
        # Use float32 for the sum so clamp_min(1e-8) is meaningful under fp16/bf16
        prior_mask_sum = prior_mask.sum(dtype=torch.float32)
        if prior_mask_sum < 1e-8:
            # No prior loss contribution (e.g., unmasked step or mask=1 everywhere)
            L_prior = loss.new_zeros((), dtype=torch.float32)
        else:
            # Prepare prior loss tensor
            prior_loss_unreduced = prior_loss_unreduced.to(loss.device, dtype=loss.dtype)
            if prior_loss_unreduced.ndim == 4:
                prior_loss_unreduced = prior_loss_unreduced.unsqueeze(2)

            prior_loss_weighted = prior_loss_unreduced * prior_mask

            if normalize_per_sample:
                reduce_dims = tuple(range(1, loss.ndim))
                prior_sum = prior_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)
                prior_weight = prior_mask.sum(dim=reduce_dims, dtype=torch.float32)
                # Handle samples with zero prior weight: treat as 0 contribution
                valid_prior = prior_weight > 1e-8
                per_sample_prior = torch.where(valid_prior, prior_sum / prior_weight.clamp_min(1e-8), torch.zeros_like(prior_sum))
                L_prior = per_sample_prior.mean() * prior_preservation_weight
            else:
                L_prior = (prior_loss_weighted.sum(dtype=torch.float32) / prior_mask_sum) * prior_preservation_weight
    else:
        L_prior = loss.new_zeros((), dtype=torch.float32)

    # === Combine: region-normalized means + explicit weighting ===
    return L_target + L_prior
