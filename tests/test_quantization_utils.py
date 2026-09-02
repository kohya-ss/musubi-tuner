"""Tests for shared, architecture-agnostic quantization-scheme validation."""

import pytest

from musubi_tuner.modules.quantization_utils import (
    validate_nvfp4_requirements,
    validate_quantization_scheme,
)


def test_validate_quantization_scheme_allows_none():
    validate_quantization_scheme(False, False, False)  # must not raise


@pytest.mark.parametrize(
    "fp8_scaled,convrot_int8,nvfp4",
    [(True, False, False), (False, True, False), (False, False, True)],
)
def test_validate_quantization_scheme_allows_exactly_one(fp8_scaled, convrot_int8, nvfp4):
    validate_quantization_scheme(fp8_scaled, convrot_int8, nvfp4)  # must not raise


@pytest.mark.parametrize(
    "fp8_scaled,convrot_int8,nvfp4",
    [(True, True, False), (True, False, True), (True, True, True)],
)
def test_validate_quantization_scheme_rejects_fp8_scaled_combined_with_anything(fp8_scaled, convrot_int8, nvfp4):
    with pytest.raises(ValueError, match="exclusive"):
        validate_quantization_scheme(fp8_scaled, convrot_int8, nvfp4)


def test_validate_quantization_scheme_allows_convrot_and_nvfp4_together():
    validate_quantization_scheme(False, True, True)  # must not raise: this is mixed-format mode


def test_validate_quantization_scheme_message_names_all_three_flags():
    with pytest.raises(ValueError, match="--fp8_scaled") as exc_info:
        validate_quantization_scheme(True, True, False)
    assert "--convrot_int8" in str(exc_info.value)
    assert "--nvfp4" in str(exc_info.value)


def test_validate_nvfp4_requirements_noop_when_nvfp4_false():
    # Would raise for any of these reasons if nvfp4 were True; must be a no-op when False.
    validate_nvfp4_requirements(False, scaled_mm_available=False, cuda_available=True, device_capability=(8, 9))


def test_validate_nvfp4_requirements_rejects_missing_scaled_mm():
    with pytest.raises(ValueError, match="PyTorch 2.10"):
        validate_nvfp4_requirements(True, scaled_mm_available=False, cuda_available=True, device_capability=(10, 0))


def test_validate_nvfp4_requirements_rejects_non_blackwell_gpu():
    with pytest.raises(ValueError, match="Blackwell"):
        validate_nvfp4_requirements(True, scaled_mm_available=True, cuda_available=True, device_capability=(8, 9))


def test_validate_nvfp4_requirements_allows_blackwell_gpu():
    validate_nvfp4_requirements(True, scaled_mm_available=True, cuda_available=True, device_capability=(10, 0))  # must not raise


def test_validate_nvfp4_requirements_allows_when_cuda_not_yet_available():
    # CLI validation can run before accelerate has placed the process on a GPU.
    validate_nvfp4_requirements(True, scaled_mm_available=True, cuda_available=False, device_capability=None)  # must not raise
