# tests/test_plateau_logit_normal_sampler.py
import pytest
import torch

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
