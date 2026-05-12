"""Tests for self-flow per-token masking (R_M <= 0.5 constraint)."""

import argparse
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

    student = torch.ones(2, 4, 16, 16)  # (B, C, H, W)
    teacher = torch.zeros(2, 4, 16, 16)  # Different values
    mask_ratio = 0.25

    result, _ = trainer._apply_per_token_mask(student, teacher, mask_ratio, torch.device("cpu"))

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

    student = torch.ones(2, 4, 8, 16, 16)  # (B, C, T, H, W)
    teacher = torch.zeros(2, 4, 8, 16, 16)
    mask_ratio = 0.25

    result, _ = trainer._apply_per_token_mask(student, teacher, mask_ratio, torch.device("cpu"))

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

    result, _ = trainer._apply_per_token_mask(student, teacher, 0.5, torch.device("cpu"))

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

    result, mask = trainer._apply_per_token_mask(student, teacher, 0.0, torch.device("cpu"))

    assert torch.equal(result, student)
    assert not mask.any(), "No tokens should be masked when mask_ratio=0"


def test_mask_broadcasts_across_channels(trainer):
    """Verify mask applies to all channels identically."""
    torch.manual_seed(42)

    student = torch.randn(1, 4, 8, 8)
    teacher = torch.zeros(1, 4, 8, 8)

    result, _ = trainer._apply_per_token_mask(student, teacher, 0.3, torch.device("cpu"))

    # If pixel (h,w) is masked in channel 0, should be masked in all channels
    masked_c0 = result[0, 0] == 0  # (H, W)
    for c in range(1, 4):
        masked_c = result[0, c] == 0
        assert torch.equal(masked_c0, masked_c), "Mask should be same across channels"


def test_grid_mask_block_structure(trainer):
    """Grid mode produces block_size x block_size aligned blocks."""
    torch.manual_seed(42)
    student = torch.ones(1, 4, 16, 16)
    teacher = torch.zeros(1, 4, 16, 16)

    args = argparse.Namespace(
        self_flow_patch_locality_mode="grid",
        self_flow_patch_block_size=4,
        mask_ratio=0.25,
    )
    result, mask_flat = trainer._apply_per_token_mask(
        student, teacher, args.mask_ratio, torch.device("cpu"), args=args
    )

    # Reshape mask to spatial dims and check block alignment
    mask_2d = mask_flat.view(1, 16, 16)
    # Each 4x4 block should be all-True or all-False
    for bh in range(0, 16, 4):
        for bw in range(0, 16, 4):
            block = mask_2d[0, bh : bh + 4, bw : bw + 4]
            assert block.all() or not block.any(), (
                f"Block at ({bh},{bw}) is partially masked: {block.sum()}/{block.numel()}"
            )


def test_grid_mask_preserves_ratio(trainer):
    """Grid mode achieves approximately the target mask ratio."""
    torch.manual_seed(0)
    student = torch.ones(4, 4, 32, 32)
    teacher = torch.zeros(4, 4, 32, 32)

    args = argparse.Namespace(
        self_flow_patch_locality_mode="grid",
        self_flow_patch_block_size=2,
        mask_ratio=0.25,
    )
    _, mask_flat = trainer._apply_per_token_mask(
        student, teacher, args.mask_ratio, torch.device("cpu"), args=args
    )
    actual = mask_flat.float().mean().item()
    assert 0.15 < actual < 0.35, f"Expected ~0.25, got {actual:.3f}"


def test_grid_mask_remainder_handling(trainer):
    """Grid mode handles H,W not divisible by block_size."""
    torch.manual_seed(42)
    # 17 is not divisible by 4
    student = torch.ones(1, 4, 17, 17)
    teacher = torch.zeros(1, 4, 17, 17)

    args = argparse.Namespace(
        self_flow_patch_locality_mode="grid",
        self_flow_patch_block_size=4,
        mask_ratio=0.25,
    )
    result, mask_flat = trainer._apply_per_token_mask(
        student, teacher, args.mask_ratio, torch.device("cpu"), args=args
    )
    assert mask_flat.shape == (1, 17 * 17)
    assert result.shape == student.shape


def test_seed_mask_exact_count(trainer):
    """Seed mode masks exactly floor(mask_ratio * H * W) tokens."""
    torch.manual_seed(42)
    student = torch.ones(2, 4, 16, 16)
    teacher = torch.zeros(2, 4, 16, 16)

    args = argparse.Namespace(
        self_flow_patch_locality_mode="seed",
        self_flow_patch_seed_count=3,
        self_flow_patch_seed_shape="square",
        mask_ratio=0.25,
    )
    _, mask_flat = trainer._apply_per_token_mask(
        student, teacher, args.mask_ratio, torch.device("cpu"), args=args
    )
    expected_count = int(0.25 * 16 * 16)  # 64
    for b in range(2):
        actual_count = mask_flat[b].sum().item()
        assert actual_count == expected_count, (
            f"Batch {b}: expected {expected_count} masked, got {int(actual_count)}"
        )


def test_seed_mask_spatial_locality(trainer):
    """Seed mode produces spatially clustered masks, not scattered."""
    torch.manual_seed(42)
    student = torch.ones(1, 4, 32, 32)
    teacher = torch.zeros(1, 4, 32, 32)

    args = argparse.Namespace(
        self_flow_patch_locality_mode="seed",
        self_flow_patch_seed_count=2,
        self_flow_patch_seed_shape="square",
        mask_ratio=0.25,
    )
    _, mask_flat = trainer._apply_per_token_mask(
        student, teacher, args.mask_ratio, torch.device("cpu"), args=args
    )
    mask_2d = mask_flat.view(1, 32, 32)[0]

    # Count horizontal adjacency: masked tokens next to other masked tokens
    # Locality should produce more adjacency than random
    h_adj = (mask_2d[:, :-1] & mask_2d[:, 1:]).sum().item()
    v_adj = (mask_2d[:-1, :] & mask_2d[1:, :]).sum().item()
    total_adj = h_adj + v_adj
    masked_count = mask_2d.sum().item()

    # For truly random 25% mask on 32x32, expected adjacency is ~0.25 * masked_count * 2
    # For clustered masks, adjacency should be much higher
    random_expected_adj = masked_count * 2 * 0.25
    assert total_adj > random_expected_adj * 1.5, (
        f"Adjacency {total_adj} not significantly above random expectation {random_expected_adj:.0f}"
    )


@pytest.mark.parametrize("shape", ["square", "circle", "diamond"])
def test_seed_mask_shapes(trainer, shape):
    """All seed shapes produce valid masks with correct count."""
    torch.manual_seed(42)
    student = torch.ones(1, 4, 16, 16)
    teacher = torch.zeros(1, 4, 16, 16)

    args = argparse.Namespace(
        self_flow_patch_locality_mode="seed",
        self_flow_patch_seed_count=2,
        self_flow_patch_seed_shape=shape,
        mask_ratio=0.25,
    )
    _, mask_flat = trainer._apply_per_token_mask(
        student, teacher, args.mask_ratio, torch.device("cpu"), args=args
    )
    expected_count = int(0.25 * 16 * 16)
    actual_count = mask_flat[0].sum().item()
    assert actual_count == expected_count
