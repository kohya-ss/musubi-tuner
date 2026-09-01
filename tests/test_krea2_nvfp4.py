"""Tests for wiring NVFP4 pre-quantized loading into load_krea2_dit."""

import pytest

from musubi_tuner.krea2.krea2_utils import load_krea2_dit


def test_load_krea2_dit_rejects_multiple_quantizations():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        load_krea2_dit("unused.safetensors", fp8_scaled=True, nvfp4=True)


def test_load_krea2_dit_rejects_convrot_and_nvfp4_together():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        load_krea2_dit("unused.safetensors", convrot_int8=True, nvfp4=True)
