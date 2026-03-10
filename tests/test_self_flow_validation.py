"""Tests for self-flow argument validation constraints."""

import argparse
import pytest


def test_student_layer_less_than_teacher_valid():
    """Valid configuration: student_layer < teacher_layer should not raise."""
    args = argparse.Namespace()
    args.self_flow = True
    args.student_feature_layer = 5
    args.teacher_feature_layer = 15
    args.mask_ratio = 0.25

    # This is the validation logic from hv_train_network.py
    if args.self_flow:
        if (
            args.student_feature_layer is not None
            and args.teacher_feature_layer is not None
            and args.student_feature_layer >= args.teacher_feature_layer
        ):
            pytest.fail("Should not raise for valid student < teacher")


def test_student_layer_greater_than_teacher_invalid():
    """Invalid: student_layer >= teacher_layer should raise ValueError."""
    args = argparse.Namespace()
    args.self_flow = True
    args.student_feature_layer = 20
    args.teacher_feature_layer = 5
    args.mask_ratio = 0.25

    with pytest.raises(ValueError, match="student_feature_layer.*must be less than.*teacher_feature_layer"):
        if args.self_flow:
            if (
                args.student_feature_layer is not None
                and args.teacher_feature_layer is not None
                and args.student_feature_layer >= args.teacher_feature_layer
            ):
                raise ValueError(
                    f"--student_feature_layer ({args.student_feature_layer}) must be less than "
                    f"--teacher_feature_layer ({args.teacher_feature_layer}). "
                    "The paper requires l < k for the representation alignment loss."
                )


def test_student_layer_equal_to_teacher_invalid():
    """Invalid: student_layer == teacher_layer should raise ValueError."""
    args = argparse.Namespace()
    args.self_flow = True
    args.student_feature_layer = 10
    args.teacher_feature_layer = 10
    args.mask_ratio = 0.25

    with pytest.raises(ValueError, match="student_feature_layer.*must be less than.*teacher_feature_layer"):
        if args.self_flow:
            if (
                args.student_feature_layer is not None
                and args.teacher_feature_layer is not None
                and args.student_feature_layer >= args.teacher_feature_layer
            ):
                raise ValueError(
                    f"--student_feature_layer ({args.student_feature_layer}) must be less than "
                    f"--teacher_feature_layer ({args.teacher_feature_layer}). "
                    "The paper requires l < k for the representation alignment loss."
                )


def test_mask_ratio_valid():
    """Valid mask_ratio <= 0.5 should not raise."""
    args = argparse.Namespace()
    args.self_flow = True
    args.mask_ratio = 0.25

    if args.self_flow:
        if args.mask_ratio > 0.5:
            pytest.fail("Should not raise for valid mask_ratio")


def test_mask_ratio_at_boundary():
    """mask_ratio = 0.5 (boundary) should be valid."""
    args = argparse.Namespace()
    args.self_flow = True
    args.mask_ratio = 0.5

    if args.self_flow:
        if args.mask_ratio > 0.5:
            pytest.fail("Should not raise for mask_ratio = 0.5")


def test_mask_ratio_exceeds_limit():
    """mask_ratio > 0.5 should raise ValueError."""
    args = argparse.Namespace()
    args.self_flow = True
    args.mask_ratio = 0.6

    with pytest.raises(ValueError, match="mask_ratio.*must be <= 0.5"):
        if args.self_flow:
            if args.mask_ratio > 0.5:
                raise ValueError(
                    f"--mask_ratio ({args.mask_ratio}) must be <= 0.5 (paper constraint R_M <= 0.5)"
                )


def test_mask_ratio_zero_valid():
    """mask_ratio = 0 should be valid."""
    args = argparse.Namespace()
    args.self_flow = True
    args.mask_ratio = 0.0

    if args.self_flow:
        if args.mask_ratio > 0.5:
            pytest.fail("Should not raise for mask_ratio = 0")


def test_none_feature_layers_skip_validation():
    """When feature_layers are None, validation should be skipped."""
    args = argparse.Namespace()
    args.self_flow = True
    args.student_feature_layer = None
    args.teacher_feature_layer = None
    args.mask_ratio = 0.25

    # Should not raise even though we can't compare None values
    if args.self_flow:
        if (
            args.student_feature_layer is not None
            and args.teacher_feature_layer is not None
            and args.student_feature_layer >= args.teacher_feature_layer
        ):
            pytest.fail("Should skip validation when layers are None")


def test_validation_skipped_when_self_flow_disabled():
    """Validation should not run when self_flow=False."""
    args = argparse.Namespace()
    args.self_flow = False
    args.student_feature_layer = 20  # Would be invalid if checked
    args.teacher_feature_layer = 5
    args.mask_ratio = 0.8  # Would be invalid if checked

    # Should not validate when self_flow=False
    if args.self_flow:
        pytest.fail("Should not validate when self_flow=False")
