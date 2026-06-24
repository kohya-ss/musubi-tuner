"""Tests for handle_model_specific_args validation in Flux2SelfFlowNetworkTrainer."""

import pytest

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer

from .test_self_flow_call_dit import make_args


def test_num_timestep_buckets_raises_with_self_flow():
    """Fix 2: --num_timestep_buckets with --self_flow must raise ValueError.

    Bucketed timesteps force the two self-flow draws to the same bucket value
    (t == s), collapsing dual-timestep scheduling to a no-op.
    """
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(num_timestep_buckets=4)
    with pytest.raises(ValueError, match="num_timestep_buckets"):
        trainer.handle_model_specific_args(args)


def test_num_timestep_buckets_one_does_not_raise():
    """num_timestep_buckets=1 is treated as disabled (same as None) — should not raise."""
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(num_timestep_buckets=1)
    # Should not raise
    trainer.handle_model_specific_args(args)


def test_num_timestep_buckets_none_does_not_raise():
    """num_timestep_buckets=None (default) must not raise."""
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(num_timestep_buckets=None)
    # Should not raise
    trainer.handle_model_specific_args(args)


def test_num_timestep_buckets_without_self_flow_does_not_raise():
    """num_timestep_buckets should only be blocked when --self_flow is active."""
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False, num_timestep_buckets=4)
    # Should not raise — self_flow is off
    trainer.handle_model_specific_args(args)


def test_student_ge_teacher_raises():
    """Existing guard: student_feature_layer >= teacher_feature_layer must raise."""
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(student_feature_layer=5, teacher_feature_layer=3)
    with pytest.raises(ValueError, match="student_feature_layer"):
        trainer.handle_model_specific_args(args)


def test_mask_ratio_gt_half_raises():
    """Existing guard: mask_ratio > 0.5 must raise."""
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(mask_ratio=0.6)
    with pytest.raises(ValueError, match="mask_ratio"):
        trainer.handle_model_specific_args(args)


def test_negative_mask_ratio_raises():
    """Guard: mask_ratio < 0 must raise (would yield 1 - R_M > 1 on coin tails)."""
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(mask_ratio=-0.1)
    with pytest.raises(ValueError, match="mask_ratio"):
        trainer.handle_model_specific_args(args)
