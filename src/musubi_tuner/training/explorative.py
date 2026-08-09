"""Pure mechanics shared by sequential best-of-K training paths."""

import torch


def create_candidate_generator(reference: torch.Tensor) -> torch.Generator:
    seed = torch.randint(
        0,
        torch.iinfo(torch.int64).max,
        (),
        device=reference.device,
        dtype=torch.int64,
    ).item()
    return torch.Generator(device=reference.device).manual_seed(seed)


def draw_candidate_noise(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def update_winners(
    best_losses: torch.Tensor,
    winner_noise: torch.Tensor,
    winner_indices: torch.Tensor,
    candidate_losses: torch.Tensor,
    candidate_noise: torch.Tensor,
    candidate_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if best_losses.ndim != 1:
        raise ValueError("best-of-K best loss must have shape [B]")
    batch_size = best_losses.shape[0]
    if candidate_losses.shape != (batch_size,):
        raise ValueError(f"best-of-K candidate loss must have shape [{batch_size}], got {tuple(candidate_losses.shape)}")
    if candidate_losses.device != best_losses.device or candidate_losses.dtype != best_losses.dtype:
        raise ValueError("best-of-K candidate and best losses must share dtype and device")
    if winner_indices.shape != (batch_size,) or winner_indices.dtype != torch.long or winner_indices.device != best_losses.device:
        raise ValueError("best-of-K winner indices must be int64 on the loss device with shape [B]")
    if candidate_noise.shape != winner_noise.shape or candidate_noise.shape[0] != batch_size:
        raise ValueError("best-of-K candidate and winner noise shapes must match [B, ...]")
    if candidate_noise.dtype != winner_noise.dtype or candidate_noise.device != winner_noise.device:
        raise ValueError("best-of-K candidate and winner noise must share dtype and device")
    nonfinite = (~torch.isfinite(candidate_losses)).nonzero(as_tuple=False).flatten()
    if nonfinite.numel():
        raise ValueError(f"candidate {candidate_index} has non-finite loss for sample indices {nonfinite.tolist()}")

    improved = candidate_losses < best_losses
    noise_mask = improved.reshape(batch_size, *([1] * (candidate_noise.ndim - 1)))
    return (
        torch.where(improved, candidate_losses, best_losses),
        torch.where(noise_mask, candidate_noise, winner_noise),
        torch.where(improved, torch.full_like(winner_indices, candidate_index), winner_indices),
    )
