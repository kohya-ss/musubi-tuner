from __future__ import annotations

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def build_scheduler(
    num_steps: int,
    *,
    device: str | torch.device | None = None,
    shift: float = 6.0,
) -> FlowMatchEulerDiscreteScheduler:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=shift,
        use_dynamic_shifting=False,
    )
    # diffusers 0.32.1 annotates this as a list but performs NumPy arithmetic.
    base_sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32).numpy()
    scheduler.set_timesteps(sigmas=base_sigmas, device=device)
    return scheduler


def euler_step(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Advance ``x`` along Mage-Flow's ``epsilon - z`` velocity field."""
    return latent + (next_sigma - sigma) * velocity


__all__ = ["build_scheduler", "euler_step"]
