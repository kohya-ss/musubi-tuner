"""Timestep sampling functions for flow matching training.

Each function returns a tensor of shape (batch_size,) with values in [0, 1]
representing the interpolation parameter t for the flow matching process.
"""
from __future__ import annotations
import math
import torch


def sample_plateau_logit_normal(
    batch_size: int,
    shift: float,   # discrete_flow_shift / alpha (e.g. 10.0)
    sigma: float,   # sigmoid_scale
    device: torch.device,
) -> torch.Tensor:
    """Sample from a plateau logit-normal distribution via inverse CDF.

    The distribution follows a logit-normal up to its mode, then holds at
    the mode's PDF value (creating a flat plateau) up to t=1. This concentrates
    samples near the mode timestep while maintaining coverage of the full range.

    Args:
        batch_size: Number of samples to draw.
        shift: Controls the mode location. Mode = sigmoid(log(shift)).
               Higher values push the mode toward t=1 (noisier timesteps).
        sigma: Width of the logit-normal component. Larger = broader distribution.
        device: Target device for the output tensor.

    Returns:
        Tensor of shape (batch_size,) with values in [0, 1].

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy.stats import norm as sp_norm
    except ImportError:
        raise ImportError(
            "scipy is required for --timestep_sampling=plateau_logit_normal. "
            "Install it with: pip install scipy"
        )

    mu_shifted = math.log(shift)
    mode_t = torch.sigmoid(torch.tensor(mu_shifted)).item()

    # PDF at mode (for normalization)
    logit_mode = math.log(mode_t / (1 - mode_t))
    pdf_at_mode = (
        1.0
        / (sigma * math.sqrt(2 * math.pi))
        * 1.0
        / (mode_t * (1 - mode_t))
        * math.exp(-((logit_mode - mu_shifted) ** 2) / (2 * sigma ** 2))
    )

    # CDF has two regions:
    # [0, mode_t]: logit-normal CDF
    # [mode_t, 1]: flat plateau at pdf_at_mode
    cdf_at_mode = sp_norm.cdf((logit_mode - mu_shifted) / sigma)
    plateau_mass = pdf_at_mode * (1 - mode_t)
    total_mass = cdf_at_mode + plateau_mass

    # Sample via inverse CDF
    u = torch.rand(batch_size, device=device) * total_mass
    t = torch.zeros(batch_size, device=device)

    in_logit_normal = u < cdf_at_mode
    if in_logit_normal.any():
        z = torch.tensor(
            sp_norm.ppf((u[in_logit_normal] / 1.0).cpu().numpy()),
            device=device,
        )
        t[in_logit_normal] = torch.sigmoid(z.float() * sigma + mu_shifted)

    in_plateau = ~in_logit_normal
    if in_plateau.any():
        t[in_plateau] = mode_t + (u[in_plateau] - cdf_at_mode) / pdf_at_mode

    return t
