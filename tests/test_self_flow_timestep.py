"""Tests for self-flow dual-timestep scheduling (paper Eq. 4-5)."""

import torch
import pytest

# Import will work after we refactor _self_flow_step to use helpers
# For now, we'll test the methods via NetworkTrainer instance
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    """Create a NetworkTrainer instance to access helper methods."""
    return NetworkTrainer()


def test_per_sample_timestep_assignment(trainer):
    """Verify teacher gets min(t_a, t_b) per sample, student gets max."""
    # Sample 0: t_a=200 < t_b=800 → teacher=200, student=800
    # Sample 1: t_a=900 > t_b=100 → teacher=100, student=900
    timesteps_a = torch.tensor([200.0, 900.0])
    timesteps_b = torch.tensor([800.0, 100.0])

    teacher, student = trainer._assign_teacher_student_timesteps(timesteps_a, timesteps_b)

    assert teacher.tolist() == [200.0, 100.0]
    assert student.tolist() == [800.0, 900.0]
    assert teacher.shape == (2,)
    assert student.shape == (2,)


def test_teacher_always_cleaner_than_student(trainer):
    """Property: teacher timestep ≤ student timestep for all samples."""
    torch.manual_seed(42)
    timesteps_a = torch.randint(1, 1001, (10,)).float()
    timesteps_b = torch.randint(1, 1001, (10,)).float()

    teacher, student = trainer._assign_teacher_student_timesteps(timesteps_a, timesteps_b)

    assert torch.all(teacher <= student)


def test_reconstruct_noisy_input_4d(trainer):
    """Verify flow formula for 4D image latents."""
    torch.manual_seed(42)
    latents = torch.randn(2, 4, 16, 16)  # (B, C, H, W)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([1.0, 1001.0])  # Min and max timesteps

    noisy = trainer._reconstruct_noisy_input(latents, noise, timesteps)

    assert noisy.shape == (2, 4, 16, 16)

    # At timestep=1, t≈0 → noisy ≈ latents
    assert torch.allclose(noisy[0], latents[0], atol=1e-3)

    # At timestep=1001, t=1 → noisy ≈ noise
    assert torch.allclose(noisy[1], noise[1], atol=1e-3)


def test_reconstruct_noisy_input_5d(trainer):
    """Verify flow formula for 5D video latents."""
    torch.manual_seed(42)
    latents = torch.randn(2, 4, 8, 16, 16)  # (B, C, T, H, W)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([500.0, 800.0])

    noisy = trainer._reconstruct_noisy_input(latents, noise, timesteps)

    assert noisy.shape == (2, 4, 8, 16, 16)

    # Verify it's a blend of latents and noise
    # At t=0.499: noisy ≈ 0.501*latents + 0.499*noise
    t0 = (500.0 - 1.0) / 1000.0
    expected_0 = (1 - t0) * latents[0] + t0 * noise[0]
    assert torch.allclose(noisy[0], expected_0, atol=1e-5)


def test_timestep_range_validation():
    """Verify timesteps are in [1, 1001] range."""
    torch.manual_seed(42)
    timesteps = torch.randint(1, 1001, (10,))
    assert torch.all(timesteps >= 1)
    assert torch.all(timesteps <= 1001)


def test_per_sample_vs_batch_mean(trainer):
    """Verify per-sample logic differs from incorrect batch-mean approach."""
    t_a = torch.tensor([200.0, 900.0])
    t_b = torch.tensor([800.0, 100.0])

    # Correct: per-sample
    teacher_correct, _ = trainer._assign_teacher_student_timesteps(t_a, t_b)

    # Incorrect (old bug): batch mean comparison
    if t_a.mean() < t_b.mean():
        teacher_wrong = t_a
    else:
        teacher_wrong = t_b

    # They should differ for this example
    assert not torch.equal(teacher_correct, teacher_wrong)
