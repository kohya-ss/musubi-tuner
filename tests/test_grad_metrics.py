"""Tests for gradient metrics collection (grad/norm, grad/mean_norm, grad/max)."""

import torch
import torch.nn as nn
import pytest

from musubi_tuner.training.trainer_base import NetworkTrainer


@pytest.fixture
def trainer():
    return NetworkTrainer()


def _params_with_grads(values: list[list[float]]) -> list[nn.Parameter]:
    """Create parameters with fixed gradient values for testing."""
    params = []
    for vals in values:
        p = nn.Parameter(torch.zeros(len(vals)))
        p.grad = torch.tensor(vals)
        params.append(p)
    return params


def test_grad_norm_single_param(trainer):
    """L2 norm of a single parameter's gradient is computed correctly."""
    # grads = [3, 4] → norm = 5
    params = _params_with_grads([[3.0, 4.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/norm"] == pytest.approx(5.0)


def test_grad_norm_multiple_params(trainer):
    """Total L2 norm across multiple parameters matches torch.nn.utils.clip_grad_norm_."""
    params = _params_with_grads([[1.0, 0.0], [0.0, 1.0]])
    metrics = trainer.collect_grad_metrics(params)
    # norm of [1,0,0,1] = sqrt(2)
    assert metrics["grad/norm"] == pytest.approx(2.0**0.5, rel=1e-5)


def test_grad_max_single_param(trainer):
    """Max absolute gradient value is the largest element."""
    params = _params_with_grads([[-5.0, 2.0, 1.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/max"] == pytest.approx(5.0)


def test_grad_max_across_params(trainer):
    """Max absolute gradient value is found across all parameters."""
    params = _params_with_grads([[1.0, 2.0], [3.0, -9.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/max"] == pytest.approx(9.0)


def test_empty_metrics_when_no_grads(trainer):
    """Returns empty dict when no parameters have gradients."""
    params = [nn.Parameter(torch.zeros(4))]  # grad is None
    metrics = trainer.collect_grad_metrics(params)
    assert metrics == {}


def test_grad_mean_norm_single_param(trainer):
    """Mean norm with one param equals its own norm."""
    params = _params_with_grads([[3.0, 4.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/mean_norm"] == pytest.approx(5.0)


def test_grad_mean_norm_multiple_params(trainer):
    """Mean norm is the average of per-parameter norms, not the total norm."""
    # param0 norm = 5, param1 norm = 5, mean = 5
    params = _params_with_grads([[3.0, 4.0], [4.0, 3.0]])
    metrics = trainer.collect_grad_metrics(params)
    assert metrics["grad/mean_norm"] == pytest.approx(5.0)


def test_skips_params_without_grad(trainer):
    """Parameters without .grad set are excluded from the computation."""
    p_with = nn.Parameter(torch.zeros(2))
    p_with.grad = torch.tensor([3.0, 4.0])
    p_without = nn.Parameter(torch.zeros(2))  # no grad
    metrics = trainer.collect_grad_metrics([p_with, p_without])
    assert metrics["grad/norm"] == pytest.approx(5.0)
    assert metrics["grad/max"] == pytest.approx(4.0)


def test_log_grad_metrics_flag_default_off():
    """--log_grad_metrics exists in the common parser and defaults to False."""
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args([])
    assert args.log_grad_metrics is False
    args, _ = parser.parse_known_args(["--log_grad_metrics"])
    assert args.log_grad_metrics is True
