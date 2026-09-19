"""Tests for NetworkTrainer.compute_loss's optional per-example reduction mode."""

import pytest
import torch

from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.training import parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


def _args(**overrides):
    parser = parser_common.setup_parser_common()
    args, _ = parser.parse_known_args([])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_reduction_none_returns_per_example_shape():
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="none")
    # batch of 2, each example (1, 1, 2) elements
    pred = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 3.0]]]])
    target = torch.zeros_like(pred)
    output = DiTOutput(pred=pred, target=target)
    timesteps = torch.tensor([500.0, 500.0])

    per_example, metrics = trainer.compute_loss(args, output, timesteps, None, torch.float32, torch.float32, 0, reduction="none")

    assert per_example.shape == (2,)
    assert per_example[0].item() == pytest.approx((2.0**2 + 0.0**2) / 2)
    assert per_example[1].item() == pytest.approx((0.0**2 + 3.0**2) / 2)
    assert metrics == {}


def test_reduction_mean_matches_batch_average_of_none():
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="none")
    pred = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 3.0]]]])
    target = torch.zeros_like(pred)
    output = DiTOutput(pred=pred, target=target)
    timesteps = torch.tensor([500.0, 500.0])

    scalar, _ = trainer.compute_loss(args, output, timesteps, None, torch.float32, torch.float32, 0)
    per_example, _ = trainer.compute_loss(args, output, timesteps, None, torch.float32, torch.float32, 0, reduction="none")

    assert scalar.item() == pytest.approx(per_example.mean().item())


def test_reduction_defaults_to_mean_and_returns_scalar():
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="none")
    pred = torch.ones(2, 1, 1, 2)
    target = torch.zeros_like(pred)
    output = DiTOutput(pred=pred, target=target)
    timesteps = torch.tensor([500.0, 500.0])

    loss, _ = trainer.compute_loss(args, output, timesteps, None, torch.float32, torch.float32, 0)

    assert loss.ndim == 0
    assert loss.item() == pytest.approx(1.0)


def test_reduction_none_with_weighting_scheme_is_per_example_not_batch_mean():
    # Regression test: `compute_loss_weighting_for_sd3` hardcodes n_dim=5 when building
    # `weighting` (shape (B,1,1,1,1)), but `pred`/`target` here are 4-dim (as e.g. Z-Image's
    # `call_dit` returns after squeezing). Raw `loss * weighting` broadcasts these as an
    # accidental (B,B,...) outer product instead of per-example elementwise weighting, which
    # collapses `reduction="none"`'s output to `weighting[i] * batch_mean(loss)` for every i
    # -- defeating per-example best-of-K selection (candidate-invariant argmin). Trainer must
    # reshape `weighting` to match `loss.ndim` before multiplying.
    trainer = NetworkTrainer()
    args = _args(weighting_scheme="sigma_sqrt")
    noise_scheduler = FlowMatchDiscreteScheduler()

    # 4-dim pred/target: (B, C, H, W) -- not the 5-dim shape `weighting` is hardcoded for.
    pred = torch.tensor([[[[2.0, 0.0]]], [[[0.0, 3.0]]]])
    target = torch.zeros_like(pred)
    assert pred.ndim == 4
    output = DiTOutput(pred=pred, target=target)

    # On-schedule timesteps (members of noise_scheduler.timesteps) so get_sigmas doesn't warn
    # and round. Different per-example timesteps so weighting[0] != weighting[1], which would
    # make the bug (batch-mean collapse) indistinguishable from correct behavior otherwise.
    timesteps = torch.tensor([300.0, 700.0])

    per_example, _ = trainer.compute_loss(
        args, output, timesteps, noise_scheduler, torch.float32, torch.float32, 0, reduction="none"
    )

    from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3

    weighting = compute_loss_weighting_for_sd3("sigma_sqrt", noise_scheduler, timesteps, timesteps.device, torch.float32)
    weighting_flat = weighting.flatten()

    mse_per_example = torch.tensor([torch.nn.functional.mse_loss(pred[i], target[i]).item() for i in range(pred.shape[0])])
    expected = weighting_flat * mse_per_example

    assert per_example.shape == (2,)
    assert torch.allclose(per_example, expected, rtol=1e-4)
    # And explicitly NOT the buggy batch-mean-collapsed value (same winner for every example).
    batch_mean_bug = weighting_flat * mse_per_example.mean()
    assert not torch.allclose(per_example, batch_mean_bug, rtol=1e-4)
