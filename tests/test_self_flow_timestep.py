import torch

from musubi_tuner.flux_2_train_network_self_flow import (
    assign_teacher_student_timesteps,
    reconstruct_noisy_input,
)


def test_teacher_gets_min_student_gets_max():
    t_a = torch.tensor([100.0, 900.0, 500.0])
    t_b = torch.tensor([300.0, 200.0, 500.0])
    teacher, student = assign_teacher_student_timesteps(t_a, t_b)
    assert torch.equal(teacher, torch.tensor([100.0, 200.0, 500.0]))
    assert torch.equal(student, torch.tensor([300.0, 900.0, 500.0]))


def test_assign_preserves_dtype():
    t_a = torch.tensor([100, 900], dtype=torch.long)
    t_b = torch.tensor([300, 200], dtype=torch.long)
    teacher, student = assign_teacher_student_timesteps(t_a, t_b)
    assert teacher.dtype == torch.long and student.dtype == torch.long


def test_reconstruct_noisy_input_4d():
    latents = torch.zeros(2, 4, 8, 8)
    noise = torch.ones(2, 4, 8, 8)
    # timesteps in [1, 1001]; t=1 -> 0% noise, t=1001 -> 100% noise
    timesteps = torch.tensor([1.0, 1001.0])
    out = reconstruct_noisy_input(latents, noise, timesteps)
    assert torch.allclose(out[0], torch.zeros(4, 8, 8))
    assert torch.allclose(out[1], torch.ones(4, 8, 8))


def test_reconstruct_noisy_input_5d():
    latents = torch.zeros(1, 4, 2, 8, 8)
    noise = torch.ones(1, 4, 2, 8, 8)
    timesteps = torch.tensor([501.0])  # t=0.5
    out = reconstruct_noisy_input(latents, noise, timesteps)
    assert torch.allclose(out, torch.full_like(out, 0.5))
