"""Surrogate losses. Temporal loss compares motion-compensated changes, not raw frame differences."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.temporal import warp_bilinear


def _rho(error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(error * error + 1e-6) - 1e-3


def masked_gaussian_lowpass(error: torch.Tensor, mask: torch.Tensor, sigma: float) -> torch.Tensor:
    """Normalized Gaussian convolution; excluded pixels never contribute to nearby supervision."""
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or not 0 < sigma <= 32:
        raise ValueError("lowpass sigma must be finite and satisfy 0 < sigma <= 32")
    if error.ndim != 4 or mask.shape != (error.shape[0], 1, *error.shape[-2:]):
        raise ValueError("lowpass expects BCHW errors and a matching B1HW mask")
    radius = math.ceil(3 * sigma)
    with torch.autocast(error.device.type, enabled=False):
        coordinates = torch.arange(-radius, radius + 1, device=error.device, dtype=error.dtype)
        kernel = torch.exp(-0.5 * (coordinates / sigma).square())
        # The center is exactly one even when a positive sigma underflows in this dtype.
        kernel[radius] = 1
        kernel = kernel / kernel.sum()
        mask = mask.to(error)
        weighted = torch.where(mask > 0, error, 0) * mask
        values = torch.cat((weighted, mask), dim=1)
        channels = values.shape[1]
        horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
        vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
        values = F.conv2d(F.pad(values, (radius, radius, 0, 0), mode="replicate"), horizontal, groups=channels)
        values = F.conv2d(F.pad(values, (0, 0, radius, radius), mode="replicate"), vertical, groups=channels)
        numerator, support = values[:, :-1], values[:, -1:]
        return numerator / torch.where(support > 0, support, 1)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    if float(weight.sum()) == 0:
        raise ValueError("loss mask has no supervised pixels")
    return (value * weight).sum() / (weight.sum() * value.shape[1])


def masked_rho(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(_rho(prediction - target), mask)


def edge_rho(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Charbonnier error of horizontal and vertical first differences, over jointly valid pixels."""
    numer = prediction.new_zeros(())
    denom = prediction.new_zeros(())
    for dimension in (-1, -2):
        pred_d = prediction.diff(dim=dimension)
        target_d = target.diff(dim=dimension)
        if dimension == -1:
            valid = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        else:
            valid = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        weight = valid.to(dtype=prediction.dtype)
        error = _rho(pred_d - target_d)
        numer = numer + (error * weight).sum()
        denom = denom + weight.sum() * prediction.shape[1]
    if float(denom) == 0:
        raise ValueError("edge loss has no supervised differences")
    return numer / denom


def single_frame_loss(
    preclamp: torch.Tensor,
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    pre_weight: float = 1.0,
    out_weight: float = 1.0,
    edge_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    pre = masked_rho(preclamp, target, mask)
    out = masked_rho(output, target, mask)
    edge = edge_rho(output, target, mask)
    total = pre_weight * pre + out_weight * out + edge_weight * edge
    return total, {
        "loss/pre": float(pre.detach()),
        "loss/out": float(out.detach()),
        "loss/edge": float(edge.detach()),
        "loss": float(total.detach()),
    }


def joint_temporal_support(current_mask, previous_mask, motion, temporal_valid, reset) -> tuple[torch.Tensor, torch.Tensor]:
    warped_mask, inside = warp_bilinear(previous_mask.float(), motion)
    active = (~reset).view(-1, 1, 1, 1)
    valid = temporal_valid.float() * inside * active * current_mask * warped_mask
    return valid, warped_mask


def masked_temporal_residual(
    current, previous, target, previous_target, current_mask, previous_mask, motion, temporal_valid, reset
) -> tuple[torch.Tensor, torch.Tensor]:
    valid, warped_mask = joint_temporal_support(current_mask, previous_mask, motion, temporal_valid, reset)
    # Normalize before differencing: excluded bilinear neighbors are not observations.
    previous_error = torch.where(previous_mask > 0, previous - previous_target, 0) * previous_mask
    warped_error, _ = warp_bilinear(previous_error, motion)
    warped_error = warped_error / torch.where(warped_mask > 0, warped_mask, 1)
    return current - target - warped_error, valid


def temporal_rho(
    current: torch.Tensor,
    previous: torch.Tensor,
    current_target: torch.Tensor,
    previous_target: torch.Tensor,
    motion: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Rho of (output change along motion) minus (target change along the same motion)."""
    warped_output, inside_output = warp_bilinear(previous, motion)
    warped_target, inside_target = warp_bilinear(previous_target, motion)
    valid = mask * inside_output.to(dtype=mask.dtype) * inside_target.to(dtype=mask.dtype)
    if float(valid.sum()) == 0:
        return current.new_zeros(())
    return masked_rho(current - warped_output, current_target - warped_target, valid)
