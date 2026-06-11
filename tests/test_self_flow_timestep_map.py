import torch

from musubi_tuner.flux_2_train_network_self_flow import build_per_token_timestep_map


def _setup():
    teacher = torch.tensor([100.0, 200.0])
    student = torch.tensor([800.0, 900.0])
    mask = torch.tensor([[True, False, True, False], [False, False, True, True]])
    return teacher, student, mask


def test_masked_tokens_get_teacher_timestep():
    teacher, student, mask = _setup()
    per_token, mismatch = build_per_token_timestep_map(teacher, student, mask)
    expected = torch.tensor([[100.0, 800.0, 100.0, 800.0], [900.0, 900.0, 200.0, 200.0]])
    assert torch.equal(per_token, expected)
    assert not mismatch.any()


def test_full_mismatch_gives_student_everywhere():
    teacher, student, mask = _setup()
    per_token, mismatch = build_per_token_timestep_map(teacher, student, mask, mismatch_prob=1.0)
    expected = student.unsqueeze(1).expand(2, 4)
    assert torch.equal(per_token, expected)
    assert torch.equal(mismatch, mask)


def test_mismatch_only_hits_masked_tokens():
    torch.manual_seed(0)
    teacher, student, mask = _setup()
    _, mismatch = build_per_token_timestep_map(teacher, student, mask, mismatch_prob=0.5)
    assert not (mismatch & ~mask).any()
