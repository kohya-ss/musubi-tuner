"""CLI-level tests for --nvfp4/--convrot_int8 wiring in krea2_generate_image.py."""

import argparse

import pytest

from musubi_tuner import krea2_generate_image
from musubi_tuner.krea2 import krea2_utils


def _parser():
    parser = argparse.ArgumentParser()
    return krea2_generate_image.parse_args_setup(parser)  # see Step 3 for this helper's introduction


def test_parser_has_nvfp4_and_convrot_int8_flags():
    parser = _parser()
    args = parser.parse_args(
        [
            "--dit",
            "unused.safetensors",
            "--vae",
            "unused.safetensors",
            "--text_encoder",
            "unused.safetensors",
            "--save_path",
            "unused",
            "--nvfp4",
            "--convrot_int8",
        ]
    )
    assert args.nvfp4 is True
    assert args.convrot_int8 is True
    assert args.convrot_int8_bwd == "bf16"
    assert args.nvfp4_columnwise_chunk_rows == 1024


def test_validate_rejects_nvfp4_and_convrot_together(monkeypatch):
    monkeypatch.setattr(krea2_generate_image, "nvfp4_scaled_mm_available", lambda: True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        krea2_utils.validate_krea2_quantization_args(
            fp8_scaled=False,
            convrot_int8=True,
            convrot_int8_bwd="bf16",
            nvfp4=True,
            nvfp4_columnwise_chunk_rows=1024,
            turbo_dit=None,
            scaled_mm_available=True,
            cuda_available=False,
            device_capability=None,
            require_block_swap_h2d_only_with_nvfp4=False,
        )


def test_validate_allows_nvfp4_with_block_swap_and_no_h2d_only_at_inference():
    # This is the point of require_block_swap_h2d_only_with_nvfp4=False: the training-only
    # requirement must not be inherited by the inference caller.
    krea2_utils.validate_krea2_quantization_args(
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=True,
        nvfp4_columnwise_chunk_rows=1024,
        turbo_dit=None,
        scaled_mm_available=True,
        cuda_available=False,
        device_capability=None,
        blocks_to_swap=4,
        block_swap_h2d_only=False,
        require_block_swap_h2d_only_with_nvfp4=False,
    )  # must not raise


def test_main_rejects_nvfp4_with_lora_weight(monkeypatch, tmp_path):
    monkeypatch.setattr(krea2_generate_image, "nvfp4_scaled_mm_available", lambda: True)
    save_path = tmp_path / "out"
    lora_path = tmp_path / "lora.safetensors"
    lora_path.write_bytes(b"")  # existence only, never opened -- rejected before load
    args = argparse.Namespace(
        prompt="a cat",
        dit="unused.safetensors",
        vae="unused.safetensors",
        text_encoder="unused.safetensors",
        device="cpu",
        text_encoder_cpu=False,
        attn_mode="torch",
        split_attn=False,
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=True,
        nvfp4_columnwise_chunk_rows=1024,
        blocks_to_swap=0,
        use_pinned_memory_for_block_swap=False,
        block_swap_h2d_only=False,
        block_swap_ring_size=2,
        save_path=str(save_path),
        lora_weight=[str(lora_path)],
        lora_multiplier=None,
        from_file=None,
        interactive=False,
        bell=False,
    )
    with pytest.raises(ValueError, match="--lora_weight"):
        krea2_generate_image.main(args)  # see Step 3: main() accepts a pre-built args for testability
