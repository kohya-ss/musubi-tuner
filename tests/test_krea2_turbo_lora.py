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


from types import SimpleNamespace


def _trainer_args(**overrides):
    base = dict(
        fp8_base=False,
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=False,
        turbo_dit=None,
        turbo_dit_cache=False,
        turbo_lora=None,
        turbo_lora_multiplier=1.0,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _handle_args(args):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    Krea2NetworkTrainer().handle_model_specific_args(args)


def test_parser_has_turbo_lora_flags():
    import argparse

    from musubi_tuner.krea2_train_network import krea2_setup_parser

    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args([])
    assert args.turbo_lora is None
    assert args.turbo_lora_multiplier == 1.0


def test_trainer_rejects_turbo_dit_and_turbo_lora_together():
    with pytest.raises(ValueError, match="turbo_dit.*turbo_lora|turbo_lora.*turbo_dit"):
        _handle_args(_trainer_args(turbo_dit="turbo.safetensors", turbo_lora="turbo_lora.safetensors"))


def test_trainer_rejects_turbo_dit_cache_without_turbo_dit():
    with pytest.raises(ValueError, match="turbo_dit_cache"):
        _handle_args(_trainer_args(turbo_dit_cache=True, turbo_lora="turbo_lora.safetensors"))


def test_trainer_rejects_turbo_dit_cache_with_neither_turbo_source():
    with pytest.raises(ValueError, match="turbo_dit_cache"):
        _handle_args(_trainer_args(turbo_dit_cache=True))


def test_trainer_accepts_turbo_lora_with_blocks_to_swap():
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", blocks_to_swap=4, sample_prompts="p.txt"))


def test_trainer_accepts_turbo_lora_with_convrot_int8():
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", convrot_int8=True, sample_prompts="p.txt"))


def test_trainer_still_rejects_turbo_dit_with_blocks_to_swap():
    with pytest.raises(ValueError, match="blocks_to_swap"):
        _handle_args(_trainer_args(turbo_dit="turbo.safetensors", blocks_to_swap=4))


def test_trainer_accepts_turbo_lora_alone():
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", sample_prompts="p.txt"))


def test_trainer_warns_turbo_lora_without_sample_prompts(caplog):
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", sample_prompts=None))
    assert "turbo_dit" in caplog.text.lower() or "turbo_lora" in caplog.text.lower()
