"""Tests for NVFP4 stochastic-rounding E2M1 conversion (nvfp4_utils._e2m1_stochastic_code),
per docs/superpowers/specs/2026-09-01-nvfp4-dgrad-stochastic-rounding-design.md.
"""

import pytest
import torch

from musubi_tuner.modules.nvfp4_utils import F4_E2M1_MAX, _e2m1_stochastic_code


def test_stochastic_code_always_produces_valid_e2m1_codes():
    torch.manual_seed(0)
    x = (torch.rand(10000) * 2 * F4_E2M1_MAX) - F4_E2M1_MAX  # uniform in [-6, 6]
    codes = _e2m1_stochastic_code(x)
    assert codes.dtype == torch.uint8
    assert torch.all(codes <= 15)
    assert torch.all(codes >= 0)


def test_stochastic_code_unbiased_in_expectation_midpoint():
    # 3.5 is exactly the midpoint between representable values 3.0 (code 5) and 4.0 (code 6);
    # unbiased rounding should land on each with ~50% probability, averaging back to 3.5.
    torch.manual_seed(0)
    x = torch.full((100000,), 3.5)
    codes = _e2m1_stochastic_code(x).float()
    magnitude_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    decoded = magnitude_table[codes.long()]
    assert decoded.mean().item() == pytest.approx(3.5, abs=0.05)


def test_stochastic_code_unbiased_in_expectation_near_zero_end():
    # 0.1 is close to 0.0 (code 0) and far from 0.5 (code 1); expected decode should be close
    # to 0.1, not exactly 0 or 0.5 -- checks the inverse-distance weighting, not just a coin flip.
    torch.manual_seed(0)
    x = torch.full((100000,), 0.1)
    codes = _e2m1_stochastic_code(x).float()
    magnitude_table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    decoded = magnitude_table[codes.long()]
    assert decoded.mean().item() == pytest.approx(0.1, abs=0.02)


def test_stochastic_code_exact_table_values_are_deterministic():
    # Exactly-representable values (including the 0.0 and 6.0 endpoints, both signs) must
    # always round to themselves -- no random draw should ever move them, since lo == hi
    # (the degenerate p_up == 0 case) at these points.
    torch.manual_seed(0)
    exact_values = torch.tensor([0.0, -0.0, 0.5, -0.5, 1.0, 2.0, 3.0, 4.0, 6.0, -6.0])
    expected_codes = torch.tensor([0, 0, 1, 9, 2, 4, 5, 6, 7, 15], dtype=torch.uint8)
    x = exact_values.repeat(1000)
    codes = _e2m1_stochastic_code(x)
    assert torch.equal(codes, expected_codes.repeat(1000))


def test_stochastic_code_matches_sign_convention():
    x = torch.tensor([1.5, -1.5])
    codes = _e2m1_stochastic_code(x)
    assert codes[0].item() == 3   # positive 1.5 -> code 3
    assert codes[1].item() == 11  # negative 1.5 -> code 3 | 8 = 11
