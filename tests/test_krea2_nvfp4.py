"""Tests for wiring NVFP4 pre-quantized loading into load_krea2_dit."""

import pytest

from musubi_tuner.krea2.krea2_utils import load_krea2_dit


def test_load_krea2_dit_rejects_multiple_quantizations():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        load_krea2_dit("unused.safetensors", fp8_scaled=True, nvfp4=True)


def test_load_krea2_dit_rejects_convrot_and_nvfp4_together():
    with pytest.raises(AssertionError, match="mutually exclusive"):
        load_krea2_dit("unused.safetensors", convrot_int8=True, nvfp4=True)


def test_load_krea2_dit_rejects_nvfp4_with_lora_weights():
    with pytest.raises(AssertionError, match="lora_weights"):
        load_krea2_dit("unused.safetensors", nvfp4=True, lora_weights=[{}])


import argparse
from types import SimpleNamespace

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser


def _base_args(**overrides):
    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args([])
    defaults = dict(
        fp8_base=False, fp8_scaled=False, convrot_int8=False, convrot_int8_bwd="bf16",
        nvfp4=False, turbo_dit=False, turbo_dit_cache=False, blocks_to_swap=0,
        block_swap_h2d_only=False,
    )
    for key, value in defaults.items():
        setattr(args, key, value)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_parser_has_nvfp4_flag():
    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args(["--nvfp4"])
    assert args.nvfp4 is True


def test_handle_model_specific_args_rejects_nvfp4_with_fp8():
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, fp8_base=True, fp8_scaled=True)
    with pytest.raises(ValueError, match="--nvfp4"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_rejects_nvfp4_with_convrot():
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, convrot_int8=True)
    with pytest.raises(ValueError, match="--nvfp4"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_rejects_nvfp4_with_turbo_dit():
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, turbo_dit=True)
    with pytest.raises(ValueError, match="turbo_dit"):
        trainer.handle_model_specific_args(args)


from musubi_tuner import krea2_train_network


def test_handle_model_specific_args_rejects_nvfp4_with_block_swap_without_h2d_only(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=4, block_swap_h2d_only=False)
    with pytest.raises(ValueError, match="block_swap_h2d_only"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_nvfp4_with_block_swap_h2d_only(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=4, block_swap_h2d_only=True)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_allows_nvfp4_without_block_swap(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, block_swap_h2d_only=False)
    trainer.handle_model_specific_args(args)  # must not raise


def test_parser_has_nvfp4_columnwise_chunk_rows_flag():
    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args([])
    assert args.nvfp4_columnwise_chunk_rows == 1024


def test_handle_model_specific_args_rejects_non_128_multiple_chunk_rows(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, nvfp4_columnwise_chunk_rows=1000)
    with pytest.raises(ValueError, match="nvfp4_columnwise_chunk_rows"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_128_multiple_chunk_rows(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, nvfp4_columnwise_chunk_rows=512)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_rejects_non_positive_chunk_rows(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, nvfp4_columnwise_chunk_rows=0)
    with pytest.raises(ValueError, match="nvfp4_columnwise_chunk_rows"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_rejects_nvfp4_on_non_blackwell_gpu(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "get_device_capability", lambda: (8, 9))
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    with pytest.raises(ValueError, match="Blackwell"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_nvfp4_on_blackwell_gpu(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "get_device_capability", lambda: (10, 0))
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_allows_nvfp4_when_cuda_not_yet_available(monkeypatch):
    # CLI validation can run before accelerate has placed the process on a GPU (e.g. a
    # multi-process launch's early arg-parsing phase) -- must not hard-fail just because CUDA
    # isn't visible yet at this point.
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: False)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    trainer.handle_model_specific_args(args)  # must not raise
