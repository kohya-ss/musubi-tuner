"""Tests for gamma warmup: linearly ramp L_rep weight from 0 to gamma over N steps.

Motivation: early in training the EMA teacher hasn't diverged from the student yet,
so applying full gamma immediately trains rep_proj on a near-identity signal.
Warmup delays full L_rep influence until the teacher is meaningful.
"""
import pytest
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    return NetworkTrainer()


def test_no_warmup_returns_gamma_immediately(trainer):
    """When warmup_steps=0, effective gamma equals gamma at all steps."""
    assert trainer.effective_gamma(0.8, global_step=0, warmup_steps=0) == pytest.approx(0.8)
    assert trainer.effective_gamma(0.8, global_step=1, warmup_steps=0) == pytest.approx(0.8)
    assert trainer.effective_gamma(0.8, global_step=100, warmup_steps=0) == pytest.approx(0.8)


def test_warmup_starts_at_zero(trainer):
    """At step 0, effective gamma is 0 regardless of configured gamma."""
    assert trainer.effective_gamma(0.8, global_step=0, warmup_steps=100) == pytest.approx(0.0)


def test_warmup_reaches_full_gamma_at_warmup_steps(trainer):
    """At step == warmup_steps, effective gamma equals the configured gamma."""
    assert trainer.effective_gamma(0.8, global_step=100, warmup_steps=100) == pytest.approx(0.8)


def test_warmup_is_linear(trainer):
    """Halfway through warmup, effective gamma is half of configured gamma."""
    assert trainer.effective_gamma(0.8, global_step=50, warmup_steps=100) == pytest.approx(0.4)


def test_warmup_clamps_after_warmup_steps(trainer):
    """Past warmup_steps, effective gamma stays at configured gamma (no overshoot)."""
    assert trainer.effective_gamma(0.8, global_step=200, warmup_steps=100) == pytest.approx(0.8)


def test_warmup_with_gamma_zero(trainer):
    """gamma=0 always returns 0 regardless of warmup."""
    assert trainer.effective_gamma(0.0, global_step=50, warmup_steps=100) == pytest.approx(0.0)
    assert trainer.effective_gamma(0.0, global_step=100, warmup_steps=0) == pytest.approx(0.0)
