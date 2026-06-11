import torch

from musubi_tuner.flux_2_train_network_self_flow import apply_per_token_mask


def test_mask_ratio_zero_returns_student_unchanged():
    student = torch.randn(2, 4, 8, 8)
    teacher = torch.randn(2, 4, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.0, torch.device("cpu"))
    assert torch.equal(out, student)
    assert mask.shape == (2, 64)
    assert not mask.any()


def test_mask_ratio_one_returns_teacher():
    # ratio 1.0 is invalid for training (paper caps at 0.5) but pins the boundary
    student = torch.zeros(2, 4, 8, 8)
    teacher = torch.ones(2, 4, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 1.0, torch.device("cpu"))
    assert torch.equal(out, teacher)
    assert mask.all()


def test_masked_positions_take_teacher_values():
    torch.manual_seed(0)
    student = torch.zeros(2, 4, 8, 8)
    teacher = torch.ones(2, 4, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.25, torch.device("cpu"))
    mask_spatial = mask.view(2, 1, 8, 8).expand_as(student)
    assert torch.equal(out[mask_spatial], teacher[mask_spatial])
    assert torch.equal(out[~mask_spatial], student[~mask_spatial])


def test_mask_applies_across_all_channels():
    torch.manual_seed(0)
    student = torch.zeros(1, 4, 8, 8)
    teacher = torch.ones(1, 4, 8, 8)
    out, _ = apply_per_token_mask(student, teacher, 0.5, torch.device("cpu"))
    # every (h, w) position is either all-teacher or all-student across channels
    per_pos = out.sum(dim=1)  # (1, 8, 8)
    assert ((per_pos == 0) | (per_pos == 4)).all()


def test_5d_independent_draw_shape():
    torch.manual_seed(0)
    student = torch.zeros(1, 4, 2, 8, 8)
    teacher = torch.ones(1, 4, 2, 8, 8)
    out, mask = apply_per_token_mask(student, teacher, 0.25, torch.device("cpu"))
    assert mask.shape == (1, 2 * 8 * 8)
    assert out.shape == student.shape


def test_approximate_ratio():
    torch.manual_seed(0)
    student = torch.zeros(8, 4, 32, 32)
    teacher = torch.ones(8, 4, 32, 32)
    _, mask = apply_per_token_mask(student, teacher, 0.25, torch.device("cpu"))
    assert abs(mask.float().mean().item() - 0.25) < 0.03
