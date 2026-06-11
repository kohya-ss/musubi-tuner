import torch

from musubi_tuner.flux_2_train_network_self_flow import (
    compute_representation_loss,
    effective_gamma,
)


def test_rep_loss_identical_features_is_minus_one():
    feats = torch.randn(2, 4, 8)
    loss = compute_representation_loss(feats, feats, torch.nn.Identity())
    assert torch.allclose(loss, torch.tensor(-1.0), atol=1e-6)


def test_rep_loss_opposite_features_is_plus_one():
    feats = torch.randn(2, 4, 8)
    loss = compute_representation_loss(feats, -feats, torch.nn.Identity())
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_rep_loss_grad_flows_through_projection():
    proj = torch.nn.Linear(8, 8)
    student = torch.randn(2, 4, 8, requires_grad=True)
    teacher = torch.randn(2, 4, 8)
    loss = compute_representation_loss(student, teacher, proj)
    loss.backward()
    assert student.grad is not None
    assert proj.weight.grad is not None


def test_effective_gamma_no_warmup():
    assert effective_gamma(0.8, global_step=0, warmup_steps=0) == 0.8


def test_effective_gamma_ramps_linearly():
    assert effective_gamma(0.8, global_step=50, warmup_steps=100) == 0.8 * 0.5
    assert effective_gamma(0.8, global_step=100, warmup_steps=100) == 0.8
    assert effective_gamma(0.8, global_step=500, warmup_steps=100) == 0.8
