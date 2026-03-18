"""Tests for self-flow loss computation (L_gen, L_rep, combined - Eq. 6-7)."""

import torch
import pytest
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    """Create a NetworkTrainer instance to access helper methods."""
    return NetworkTrainer()


@pytest.fixture
def mock_rep_proj():
    """Simple linear projection for testing."""

    class SimpleProj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(128, 128, bias=False)
            torch.nn.init.eye_(self.linear.weight)

        def forward(self, x):
            return self.linear(x)

    return SimpleProj()


def test_representation_loss_identical_features(trainer, mock_rep_proj):
    """When student=teacher features, L_rep should be -1 (perfect similarity)."""
    torch.manual_seed(42)
    features = torch.randn(2, 64, 128)  # (B, N, D)

    l_rep = trainer._compute_representation_loss(features, features, mock_rep_proj)

    # Cosine similarity of identical vectors = 1, so L_rep = -1
    assert torch.isclose(l_rep, torch.tensor(-1.0), atol=1e-5)


def test_representation_loss_opposite_features(trainer, mock_rep_proj):
    """When student=-teacher, L_rep should be 1 (worst similarity)."""
    torch.manual_seed(42)
    student = torch.randn(2, 64, 128)
    teacher = -student  # Opposite direction

    l_rep = trainer._compute_representation_loss(student, teacher, mock_rep_proj)

    # Cosine similarity of opposite vectors = -1, so L_rep = 1
    assert torch.isclose(l_rep, torch.tensor(1.0), atol=1e-5)


def test_representation_loss_orthogonal_features(trainer):
    """When features are orthogonal, L_rep should be ~0."""
    # Create orthogonal features
    student = torch.zeros(1, 2, 4)
    student[0, 0, :2] = 1.0  # [1, 1, 0, 0]
    student[0, 1, 2:] = 1.0  # [0, 0, 1, 1]

    teacher = torch.zeros(1, 2, 4)
    teacher[0, 0, 2:] = 1.0  # [0, 0, 1, 1]
    teacher[0, 1, :2] = 1.0  # [1, 1, 0, 0]

    simple_proj = torch.nn.Identity()
    l_rep = trainer._compute_representation_loss(student, teacher, simple_proj)

    # Orthogonal vectors have cosine similarity ≈ 0
    assert torch.isclose(l_rep, torch.tensor(0.0), atol=1e-5)


def test_representation_loss_shape_requirements(trainer):
    """Verify L_rep works with correct shapes."""
    torch.manual_seed(42)
    student = torch.randn(4, 100, 256)  # (B=4, N=100, D=256)
    teacher = torch.randn(4, 100, 256)

    proj = torch.nn.Linear(256, 256)
    l_rep = trainer._compute_representation_loss(student, teacher, proj)

    assert l_rep.shape == torch.Size([])  # Scalar
    assert l_rep.requires_grad  # Should be differentiable


def test_combined_loss_with_rep(trainer):
    """Test combined loss formula: L_gen + gamma * L_rep."""
    l_gen = torch.tensor(2.0)
    l_rep = torch.tensor(0.5)
    gamma = 0.8

    loss = trainer._compute_combined_loss(l_gen, l_rep, gamma)

    # Expected: 2.0 + 0.8 * 0.5 = 2.4
    assert torch.isclose(loss, torch.tensor(2.4), atol=1e-6)


def test_combined_loss_without_rep(trainer):
    """When L_rep is None, should return only L_gen."""
    l_gen = torch.tensor(2.0)
    l_rep = None
    gamma = 0.8

    loss = trainer._compute_combined_loss(l_gen, l_rep, gamma)

    assert torch.equal(loss, l_gen)


def test_combined_loss_gamma_zero(trainer):
    """With gamma=0, L_rep should be ignored."""
    l_gen = torch.tensor(2.0)
    l_rep = torch.tensor(100.0)  # Large value
    gamma = 0.0

    loss = trainer._compute_combined_loss(l_gen, l_rep, gamma)

    assert torch.equal(loss, l_gen)


def test_combined_loss_gamma_one(trainer):
    """With gamma=1, both losses weighted equally."""
    l_gen = torch.tensor(3.0)
    l_rep = torch.tensor(2.0)
    gamma = 1.0

    loss = trainer._compute_combined_loss(l_gen, l_rep, gamma)

    assert torch.isclose(loss, torch.tensor(5.0), atol=1e-6)


def test_representation_loss_gradient_flow(trainer, mock_rep_proj):
    """Verify gradients flow through projection network."""
    torch.manual_seed(42)
    student = torch.randn(2, 10, 128, requires_grad=True)
    teacher = torch.randn(2, 10, 128)

    # Enable gradients on projection
    for param in mock_rep_proj.parameters():
        param.requires_grad = True

    l_rep = trainer._compute_representation_loss(student, teacher, mock_rep_proj)
    l_rep.backward()

    # Check gradients exist
    assert student.grad is not None
    for param in mock_rep_proj.parameters():
        assert param.grad is not None
