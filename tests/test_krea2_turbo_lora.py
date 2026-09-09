"""Tests for --turbo_lora: a Turbo LoRA composed live (as a second LoRA hook) on top of
RAW weights, alongside the LoRA being trained. Never merges into or mutates base weights."""

import pytest

from musubi_tuner.krea2 import krea2_utils


def _validate(**overrides):
    base = dict(
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=False,
        nvfp4_columnwise_chunk_rows=1024,
        turbo_dit=None,
        turbo_lora=None,
        scaled_mm_available=True,
        cuda_available=False,
        device_capability=None,
    )
    base.update(overrides)
    krea2_utils.validate_krea2_quantization_args(**base)


def test_validate_rejects_turbo_dit_and_turbo_lora_together():
    with pytest.raises(ValueError, match="turbo_dit.*turbo_lora|turbo_lora.*turbo_dit"):
        _validate(turbo_dit="turbo.safetensors", turbo_lora="turbo_lora.safetensors")


def test_validate_allows_turbo_lora_with_convrot_int8():
    _validate(convrot_int8=True, turbo_lora="turbo_lora.safetensors")  # must not raise


def test_validate_allows_turbo_lora_with_nvfp4():
    _validate(nvfp4=True, turbo_lora="turbo_lora.safetensors")  # must not raise


def test_validate_allows_turbo_lora_alone():
    _validate(turbo_lora="turbo_lora.safetensors")  # must not raise


def test_validate_still_rejects_convrot_with_turbo_dit():
    with pytest.raises(ValueError, match="turbo_dit"):
        _validate(convrot_int8=True, turbo_dit="turbo.safetensors")


def test_validate_still_rejects_nvfp4_with_turbo_dit():
    with pytest.raises(ValueError, match="turbo_dit"):
        _validate(nvfp4=True, turbo_dit="turbo.safetensors")
