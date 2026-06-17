"""Timestep sampling density and loss weighting utilities (SD3-style)."""

import logging
import math

import torch


logger = logging.getLogger(__name__)


def compute_ideogram4_shift_timestep(
    uniform_samples: torch.Tensor,
    token_grid_height: int,
    token_grid_width: int,
    *,
    image_patch_size: int = 16,
    base_mean: float = 0.0,
    std: float = 1.5,
) -> torch.Tensor:
    """Map uniform samples to Ideogram 4's resolution-aware logit-normal t."""
    eps = 1e-7
    u = torch.clamp(uniform_samples.to(torch.float64), eps, 1.0 - eps)
    image_pixels = token_grid_height * image_patch_size * token_grid_width * image_patch_size
    mean = base_mean + 0.5 * math.log(image_pixels / (512 * 512))
    z = torch.special.ndtri(u)
    # musubi convention: t=1 is pure noise, t=0 is clean. Higher resolution -> larger
    # ``mean`` -> ``t`` skewed toward 1 (more noise). The trainer feeds the model
    # ``model_t = 1 - t``, so this reproduces the inference schedule's
    # ``model_t = 1 - sigmoid(mean + std * z)`` instead of mirroring it.
    t = torch.special.expit(mean + std * z)
    t_min = 1.0 / (1 + math.exp(0.5 * 18.0))
    t_max = 1.0 / (1 + math.exp(0.5 * -15.0))
    return t.clamp(1.0 - t_max, 1.0 - t_min).to(dtype=uniform_samples.dtype)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)

    # if sum([(schedule_timesteps == t) for t in timesteps]) < len(timesteps):
    if any([(schedule_timesteps == t).sum() == 0 for t in timesteps]):
        # raise ValueError("Some timesteps are not in the schedule / 一部のtimestepsがスケジュールに含まれていません")
        # round to nearest timestep
        logger.warning("Some timesteps are not in the schedule / 一部のtimestepsがスケジュールに含まれていません")
        step_indices = [torch.argmin(torch.abs(schedule_timesteps - t)).item() for t in timesteps]
    else:
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_loss_weighting_for_sd3(weighting_scheme: str, noise_scheduler, timesteps, device, dtype):
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt" or weighting_scheme == "cosmap":
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=5, dtype=dtype)
        if weighting_scheme == "sigma_sqrt":
            weighting = (sigmas**-2.0).float()
        else:
            bot = 1 - 2 * sigmas + 2 * sigmas**2
            weighting = 2 / (math.pi * bot)
    else:
        weighting = None  # torch.ones_like(sigmas)
    return weighting
