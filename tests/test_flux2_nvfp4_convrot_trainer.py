"""Tests for --convrot_int8/--nvfp4 CLI wiring in flux_2_train_network.py.

Mirrors tests/test_krea2_nvfp4.py's / tests/test_krea2_convrot_int8.py's trainer-flag-
validation sections (the kernel-level ConvRot/NVFP4 math is already covered by
tests/test_krea2_convrot_int8.py and tests/test_nvfp4_training.py -- both architecture-
agnostic modules under modules/).
"""

import argparse
from types import SimpleNamespace

import pytest

from musubi_tuner import flux_2_train_network
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser


def test_parser_has_convrot_int8_flag():
    parser = argparse.ArgumentParser()
    flux2_setup_parser(parser)
    args = parser.parse_args(["--convrot_int8"])
    assert args.convrot_int8 is True


def test_parser_convrot_int8_bwd_defaults_to_bf16():
    parser = argparse.ArgumentParser()
    flux2_setup_parser(parser)
    args = parser.parse_args([])
    assert args.convrot_int8_bwd == "bf16"


def test_parser_convrot_int8_bwd_accepts_int8():
    parser = argparse.ArgumentParser()
    flux2_setup_parser(parser)
    args = parser.parse_args(["--convrot_int8_bwd", "int8"])
    assert args.convrot_int8_bwd == "int8"


def test_parser_has_nvfp4_flag():
    parser = argparse.ArgumentParser()
    flux2_setup_parser(parser)
    args = parser.parse_args(["--nvfp4"])
    assert args.nvfp4 is True


def test_parser_nvfp4_columnwise_chunk_rows_defaults_to_1024():
    parser = argparse.ArgumentParser()
    flux2_setup_parser(parser)
    args = parser.parse_args([])
    assert args.nvfp4_columnwise_chunk_rows == 1024


def _trainer_args(**overrides):
    base = dict(
        model_version="klein-base-9b",
        mixed_precision="bf16",
        fp8_base=False,
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=False,
        nvfp4_columnwise_chunk_rows=1024,
        blocks_to_swap=0,
        block_swap_h2d_only=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _handle_args(args):
    trainer = Flux2NetworkTrainer()
    trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_rejects_nvfp4_with_fp8():
    with pytest.raises(ValueError, match="exclusive"):
        _handle_args(_trainer_args(nvfp4=True, fp8_scaled=True))


def test_handle_model_specific_args_rejects_int8_bwd_without_convrot():
    with pytest.raises(ValueError, match="convrot_int8_bwd"):
        _handle_args(_trainer_args(convrot_int8_bwd="int8"))


def test_handle_model_specific_args_allows_nvfp4_with_convrot(monkeypatch):
    # nvfp4 + convrot_int8 together select mixed-format prequantized loading; must not raise.
    monkeypatch.setattr(flux_2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(flux_2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_2_train_network.torch.cuda, "get_device_capability", lambda: (10, 0))
    _handle_args(_trainer_args(nvfp4=True, convrot_int8=True))  # must not raise


def test_handle_model_specific_args_rejects_nvfp4_with_block_swap_without_h2d_only(monkeypatch):
    monkeypatch.setattr(flux_2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    with pytest.raises(ValueError, match="block_swap_h2d_only"):
        _handle_args(_trainer_args(nvfp4=True, blocks_to_swap=4, block_swap_h2d_only=False))


def test_handle_model_specific_args_allows_nvfp4_with_block_swap_h2d_only(monkeypatch):
    monkeypatch.setattr(flux_2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    _handle_args(_trainer_args(nvfp4=True, blocks_to_swap=4, block_swap_h2d_only=True))  # must not raise


def test_handle_model_specific_args_rejects_nvfp4_on_non_blackwell_gpu(monkeypatch):
    monkeypatch.setattr(flux_2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(flux_2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_2_train_network.torch.cuda, "get_device_capability", lambda: (8, 9))
    with pytest.raises(ValueError, match="Blackwell"):
        _handle_args(_trainer_args(nvfp4=True))


def test_handle_model_specific_args_allows_nvfp4_on_blackwell_gpu(monkeypatch):
    monkeypatch.setattr(flux_2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(flux_2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(flux_2_train_network.torch.cuda, "get_device_capability", lambda: (10, 0))
    _handle_args(_trainer_args(nvfp4=True))  # must not raise


def test_handle_model_specific_args_rejects_nvfp4_without_scaled_mm_support(monkeypatch):
    monkeypatch.setattr(flux_2_train_network, "nvfp4_scaled_mm_available", lambda: False)
    with pytest.raises(ValueError, match="PyTorch 2.10"):
        _handle_args(_trainer_args(nvfp4=True))


def test_compile_transformer_disables_linear_for_convrot(monkeypatch):
    from musubi_tuner.utils import model_utils

    captured = {}

    def fake_compile_transformer(args, transformer, block_lists, disable_linear=False):
        captured["disable_linear"] = disable_linear
        return transformer

    monkeypatch.setattr(model_utils, "compile_transformer", fake_compile_transformer)
    trainer = Flux2NetworkTrainer()
    trainer.blocks_to_swap = 0
    fake_transformer = SimpleNamespace(double_blocks=[], single_blocks=[])
    trainer.compile_transformer(_trainer_args(convrot_int8=True), fake_transformer)
    assert captured["disable_linear"] is True


def test_compile_transformer_disables_linear_for_nvfp4(monkeypatch):
    from musubi_tuner.utils import model_utils

    captured = {}

    def fake_compile_transformer(args, transformer, block_lists, disable_linear=False):
        captured["disable_linear"] = disable_linear
        return transformer

    monkeypatch.setattr(model_utils, "compile_transformer", fake_compile_transformer)
    trainer = Flux2NetworkTrainer()
    trainer.blocks_to_swap = 0
    fake_transformer = SimpleNamespace(double_blocks=[], single_blocks=[])
    trainer.compile_transformer(_trainer_args(nvfp4=True), fake_transformer)
    assert captured["disable_linear"] is True


def test_compile_transformer_allows_linear_without_quantization_or_block_swap(monkeypatch):
    from musubi_tuner.utils import model_utils

    captured = {}

    def fake_compile_transformer(args, transformer, block_lists, disable_linear=False):
        captured["disable_linear"] = disable_linear
        return transformer

    monkeypatch.setattr(model_utils, "compile_transformer", fake_compile_transformer)
    trainer = Flux2NetworkTrainer()
    trainer.blocks_to_swap = 0
    fake_transformer = SimpleNamespace(double_blocks=[], single_blocks=[])
    trainer.compile_transformer(_trainer_args(), fake_transformer)
    assert captured["disable_linear"] is False
