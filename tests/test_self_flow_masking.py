"""Tests for self-flow per-token masking (R_M <= 0.5 constraint)."""

import torch
import pytest
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    """Create a NetworkTrainer instance to access helper methods."""
    return NetworkTrainer()


def test_mask_ratio_applied_correctly_4d(trainer):
    """Verify mask_ratio controls fraction of masked tokens for 4D."""
    torch.manual_seed(42)

    student = torch.ones(2, 4, 16, 16)   # (B, C, H, W)
    teacher = torch.zeros(2, 4, 16, 16)  # Different values
    mask_ratio = 0.25

    result = trainer._apply_per_token_mask(student, teacher, mask_ratio, torch.device('cpu'))

    # Count masked tokens (where result == 0, i.e., teacher value)
    # Should be approximately 25% of H*W tokens per sample
    for b in range(2):
        masked_count = (result[b, 0] == 0).sum().item()  # Check one channel
        total_tokens = 16 * 16
        actual_ratio = masked_count / total_tokens
        # Allow some randomness tolerance
        assert 0.15 < actual_ratio < 0.35, f"Expected ~25%, got {actual_ratio:.2%}"


def test_mask_ratio_applied_correctly_5d(trainer):
    """Verify mask_ratio controls fraction of masked tokens for 5D."""
    torch.manual_seed(42)

    student = torch.ones(2, 4, 8, 16, 16)   # (B, C, T, H, W)
    teacher = torch.zeros(2, 4, 8, 16, 16)
    mask_ratio = 0.25

    result = trainer._apply_per_token_mask(student, teacher, mask_ratio, torch.device('cpu'))

    for b in range(2):
        masked_count = (result[b, 0] == 0).sum().item()
        total_tokens = 8 * 16 * 16
        actual_ratio = masked_count / total_tokens
        assert 0.20 < actual_ratio < 0.30


def test_masked_tokens_get_teacher_values(trainer):
    """Verify masked positions use teacher input."""
    torch.manual_seed(42)

    student = torch.randn(1, 4, 8, 8)
    teacher = torch.randn(1, 4, 8, 8)

    result = trainer._apply_per_token_mask(student, teacher, 0.5, torch.device('cpu'))

    # Every pixel should be either from student or teacher
    mask_from_teacher = torch.isclose(result, teacher, atol=1e-6)
    mask_from_student = torch.isclose(result, student, atol=1e-6)

    # Every position should match one or the other
    assert torch.all(mask_from_teacher | mask_from_student)


def test_mask_ratio_zero_returns_student(trainer):
    """With mask_ratio=0, no masking occurs."""
    torch.manual_seed(42)
    student = torch.randn(2, 4, 8, 8)
    teacher = torch.randn(2, 4, 8, 8)

    result = trainer._apply_per_token_mask(student, teacher, 0.0, torch.device('cpu'))

    assert torch.equal(result, student)


def test_mask_broadcasts_across_channels(trainer):
    """Verify mask applies to all channels identically."""
    torch.manual_seed(42)

    student = torch.randn(1, 4, 8, 8)
    teacher = torch.zeros(1, 4, 8, 8)

    result = trainer._apply_per_token_mask(student, teacher, 0.3, torch.device('cpu'))

    # If pixel (h,w) is masked in channel 0, should be masked in all channels
    masked_c0 = (result[0, 0] == 0)  # (H, W)
    for c in range(1, 4):
        masked_c = (result[0, c] == 0)
        assert torch.equal(masked_c0, masked_c), "Mask should be same across channels"
