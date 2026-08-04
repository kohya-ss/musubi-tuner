from __future__ import annotations

from collections.abc import Sequence

import torch

from .utils import PackedMageFlowInputs


def sigma_from_training_timesteps(timesteps: torch.Tensor, *, one_based_offset: bool = True) -> torch.Tensor:
    """Convert the trainer's 1..1000 convention back to Mage-Flow sigmas."""
    if timesteps.ndim != 1:
        raise ValueError(f"training timesteps must be rank 1, got {tuple(timesteps.shape)}")
    offset = 1.0 if one_based_offset else 0.0
    sigmas = (timesteps.to(torch.float32) - offset) / 1000.0
    if torch.any((sigmas < 0) | (sigmas > 1)):
        raise ValueError("training timesteps must map to denoising sigmas in [0, 1]")
    return sigmas


def unpack_target_predictions(
    prediction: torch.Tensor,
    packed: PackedMageFlowInputs,
) -> list[torch.Tensor]:
    """Return target frames as individual ``[C,H,W]`` tensors."""
    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise ValueError(f"packed prediction must have shape [1,total,C], got {tuple(prediction.shape)}")
    if prediction.shape[1] != packed.image_tokens.shape[1]:
        raise ValueError("prediction token count does not match the packed image token count")

    predictions: list[torch.Tensor] = []
    cumulative = packed.image_cu_seqlens.detach().cpu().tolist()
    for sample_index, start in enumerate(cumulative[:-1]):
        _, height, width = packed.image_shapes[sample_index][0]
        target_length = height * width
        target = prediction[0, start : start + target_length]
        predictions.append(target.reshape(height, width, prediction.shape[-1]).permute(2, 0, 1))
    return predictions


def stack_bucketed_targets(targets: Sequence[torch.Tensor]) -> torch.Tensor:
    """Stack current bucket-mode targets while keeping heterogeneous packing explicit."""
    items = list(targets)
    if not items:
        raise ValueError("target predictions must not be empty")
    shape = tuple(items[0].shape)
    if any(tuple(item.shape) != shape for item in items[1:]):
        raise ValueError(
            "heterogeneous target shapes reached the bucket training path; "
            "native-resolution loss reduction is not enabled in this release"
        )
    return torch.stack(items)


__all__ = ["sigma_from_training_timesteps", "stack_bucketed_targets", "unpack_target_predictions"]
