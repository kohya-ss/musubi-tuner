"""Surrogate losses. Temporal loss compares motion-compensated changes, not raw frame differences."""

from __future__ import annotations

import torch

from musubi_tuner.dlssnr.temporal import warp_bilinear


def _rho(error: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(error * error + 1e-6) - 1e-3


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
