"""Shared test fixtures for self-flow tests."""

import argparse

import pytest
import torch


@pytest.fixture
def mock_args():
    """Mock args namespace with self-flow defaults."""
    args = argparse.Namespace()
    args.self_flow = True
    args.self_flow_gamma = 0.8
    args.mask_ratio = 0.25
    args.ema_decay = 0.9999
    args.student_feature_layer = 5
    args.teacher_feature_layer = 15
    return args


@pytest.fixture
def synthetic_latents_4d():
    """4D latents for image models (B, C, H, W)."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 16, 16)


@pytest.fixture
def synthetic_latents_5d():
    """5D latents for video models (B, C, T, H, W)."""
    torch.manual_seed(42)
    return torch.randn(2, 4, 8, 16, 16)


@pytest.fixture
def mock_flux2_model():
    """Mock Flux2 model that returns features."""

    class MockFlux2:
        def __init__(self):
            self.hidden_size = 128

        def forward(self, x, hidden_features=False, feature_layer=None, **kwargs):
            torch.manual_seed(42)
            if hidden_features:
                pred = torch.randn_like(x)
                features = torch.randn(x.shape[0], 64, self.hidden_size)
                return pred, features
            return torch.randn_like(x)

    return MockFlux2()


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
