"""Standalone timestep samplers extracted from hv_train_network."""

import math

import torch


def _norm_cdf(x: torch.Tensor) -> torch.Tensor:
    """CDF of the standard normal distribution using torch.erf.

    Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: torch.Tensor) -> torch.Tensor:
    """Approximate inverse CDF of the standard normal distribution.

    Uses the Beasley-Springer-Moro algorithm (rational approximation).
    Absolute error < 1.15e-3 across all p in (0, 1).
    """
    # Coefficients for the left tail (p <= 0.5) -- Abramowitz & Stegun 26.2.23
    a = [
        -3.969683028665376e01,
        2.209460984243202e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.133813401623706e01,
        2.851581188374611e00,
    ]
    b = [
        5.889062427331844e01,
        -3.042144363395251e02,
        4.317211893245512e02,
        -3.110736311814853e02,
        9.768010603293793e01,
        -1.350935116613564e01,
        1.336884614464596e00,
    ]

    # Coefficients for the right tail (p > 0.5) -- Abramowitz & Stegun 26.2.24
    c = [
        -3.526242715890073e00,
        -4.348446955440186e00,
        -3.056500844560795e01,
        -2.506628277459230e02,
        -8.440194439059601e02,
        -1.892648651189089e03,
        -1.306267905943714e03,
    ]
    d = [
        1.857704105406095e01,
        3.435533513212732e02,
        3.403092756485882e03,
        1.647639469823287e04,
        4.687572465593002e04,
        6.452508381500880e04,
        3.322706406576854e04,
    ]

    pp = p.clone()
    result = torch.zeros_like(pp)

    left_mask = pp <= 0.5
    right_mask = ~left_mask

    # Left tail
    q = torch.sqrt(-2.0 * torch.log(pp[left_mask]))
    numerator = (
        (((((a[0] * q + a[1]) * q + a[2]) * q + a[3]) * q + a[4]) * q + a[5]) * q
    )
    denominator = (
        (((((b[0] * q + b[1]) * q + b[2]) * q + b[3]) * q + b[4]) * q + b[5]) * q
        + 1.0
    )
    result[left_mask] = q - numerator / denominator

    # Right tail
    qr = torch.sqrt(-2.0 * torch.log(1.0 - pp[right_mask]))
    c_num = (
        (((((c[0] * qr + c[1]) * qr + c[2]) * qr + c[3]) * qr + c[4]) * qr + c[5])
        * qr
        + c[6]
    )
    d_den = (
        (((((d[0] * qr + d[1]) * qr + d[2]) * qr + d[3]) * qr + d[4]) * qr + d[5])
        * qr
        + d[6]
    )
    result[right_mask] = qr - c_num / d_den

    return result


def sample_plateau_logit_normal(
    batch_size: int,
    shift: float,  # discrete_flow_shift (alpha)
    sigma: float,  # sigmoid_scale
    device: torch.device,
) -> torch.Tensor:
    """Sample timesteps from the plateau logit-normal distribution.

    This combines a logit-normal region [0, mode_t] with a flat plateau
    region [mode_t, 1], sampled via inverse CDF.

    Args:
        batch_size: Number of samples to generate.
        shift: Flow shift parameter (alpha), e.g. 10.0.
        sigma: Standard deviation for the underlying normal distribution.
        device: Torch device for tensor allocation.

    Returns:
        Tensor of shape (batch_size,) with values in [0, 1].
    """
    mu_shifted = math.log(shift)
    sigma_t = torch.tensor(sigma, device=device)

    mode_t = torch.sigmoid(torch.tensor(mu_shifted, device=device)).item()

    # Logit of the mode and PDF at mode (for normalization)
    logit_mode = math.log(mode_t / (1.0 - mode_t))

    pdf_at_mode = (
        1.0
        / (sigma * math.sqrt(2 * math.pi))
        * 1.0
        / (mode_t * (1.0 - mode_t))
        * math.exp(-((logit_mode - mu_shifted) ** 2) / (2 * sigma**2))
    )

    # CDF of the underlying normal at (logit_mode - mu_shifted) / sigma
    z_at_mode = torch.tensor((logit_mode - mu_shifted) / sigma, device=device)
    cdf_at_mode = _norm_cdf(z_at_mode).item()

    plateau_mass = pdf_at_mode * (1.0 - mode_t)
    total_mass = cdf_at_mode + plateau_mass

    # Sample via inverse CDF
    u = torch.rand(batch_size, device=device) * total_mass
    t = torch.zeros(batch_size, device=device)

    in_logit_normal = u < cdf_at_mode
    if in_logit_normal.any():
        # Inverse CDF of logit-normal region
        z = _norm_ppf(u[in_logit_normal])
        t[in_logit_normal] = torch.sigmoid(z * sigma_t + mu_shifted)

    in_plateau = ~in_logit_normal
    if in_plateau.any():
        t[in_plateau] = mode_t + (u[in_plateau] - cdf_at_mode) / pdf_at_mode

    return t
