import math

import pytest
import torch

scipy = pytest.importorskip("scipy", reason="scipy required for plateau_logit_normal tests")

from musubi_tuner.timestep_samplers import sample_plateau_logit_normal


def test_output_shape():
    t = sample_plateau_logit_normal(16, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    assert t.shape == (16,)


def test_values_in_range():
    t = sample_plateau_logit_normal(1000, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    assert t.min().item() >= 0.0
    assert t.max().item() <= 1.0


def test_no_nan_or_inf():
    t = sample_plateau_logit_normal(1000, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    assert torch.all(torch.isfinite(t))


def test_output_is_float_tensor():
    t = sample_plateau_logit_normal(16, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    assert t.is_floating_point()


def test_higher_shift_higher_mean():
    torch.manual_seed(0)
    t_low = sample_plateau_logit_normal(2000, shift=1.0, sigma=1.0, device=torch.device("cpu"))
    torch.manual_seed(0)
    t_high = sample_plateau_logit_normal(2000, shift=20.0, sigma=1.0, device=torch.device("cpu"))
    assert t_high.mean() > t_low.mean()


def test_numerical_accuracy_against_scipy():
    """Verify the sampler produces t-values matching a direct scipy reference to within tolerance."""
    from scipy.stats import norm as sp_norm

    def _reference(batch_size, shift, sigma, seed=42):
        torch.manual_seed(seed)
        mu_shifted = math.log(shift)
        mode_t = torch.sigmoid(torch.tensor(mu_shifted)).item()
        logit_mode = math.log(mode_t / (1 - mode_t))
        pdf_at_mode = (
            1.0
            / (sigma * math.sqrt(2 * math.pi))
            * 1.0
            / (mode_t * (1 - mode_t))
            * math.exp(-((logit_mode - mu_shifted) ** 2) / (2 * sigma ** 2))
        )
        cdf_at_mode = sp_norm.cdf((logit_mode - mu_shifted) / sigma)
        plateau_mass = pdf_at_mode * (1 - mode_t)
        total_mass = cdf_at_mode + plateau_mass
        u = torch.rand(batch_size) * total_mass
        t = torch.zeros(batch_size)
        in_logit_normal = u < cdf_at_mode
        if in_logit_normal.any():
            z = torch.tensor(sp_norm.ppf(u[in_logit_normal].numpy()))
            t[in_logit_normal] = torch.sigmoid(z.float() * sigma + mu_shifted)
        in_plateau = ~in_logit_normal
        if in_plateau.any():
            t[in_plateau] = mode_t + (u[in_plateau] - cdf_at_mode) / pdf_at_mode
        return t

    # Each function sets its own seed so they are independently reproducible
    ref = _reference(1000, shift=10.0, sigma=1.0, seed=42)
    torch.manual_seed(42)
    got = sample_plateau_logit_normal(1000, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    assert torch.allclose(got, ref, atol=1e-6), f"Max diff: {(got - ref).abs().max()}"


def test_stochastic():
    torch.manual_seed(0)
    t1 = sample_plateau_logit_normal(10, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    torch.manual_seed(1)
    t2 = sample_plateau_logit_normal(10, shift=10.0, sigma=1.0, device=torch.device("cpu"))
    assert not torch.allclose(t1, t2)
