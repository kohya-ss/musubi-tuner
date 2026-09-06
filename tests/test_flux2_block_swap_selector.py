"""Tests for Flux2.enable_block_swap's NVFP4 swap-tensor-selector override.

Mirrors Krea2's SingleStreamDiT.enable_block_swap wiring (see
modules/nvfp4_utils.py's quantized_linear_swap_tensor_selector docstring for why this override is
needed): an NVFP4-patched Linear carries a second full-size weight copy
(nvfp4_weight_t) that the default swap-tensor selector doesn't know about.

These tests monkeypatch create_offloader to capture the BlockSwapConfig each block list
would be given, rather than exercising the real offloader machinery (LoRAStreamOffloader,
which the selector only matters for, requires CUDA -- see modules/custom_offloading_utils.py).
"""

import torch

from musubi_tuner.flux_2 import flux2_models
from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.modules.nvfp4_utils import quantized_linear_swap_tensor_selector


def _tiny_flux2_params():
    # Flux2.enable_block_swap's assert requires num_double_blocks >= 2 and
    # num_single_blocks >= 2 unconditionally (even when only one side is being swapped) --
    # depth=4/depth_single_blocks=4 keeps both comfortably above that floor.
    return Flux2Params(
        in_channels=8,
        context_in_dim=8,
        hidden_size=32,
        num_heads=2,
        depth=4,
        depth_single_blocks=4,
        axes_dim=[4, 4, 4, 4],
        theta=2000,
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )


def _capture_configs(monkeypatch):
    captured = []

    def fake_create_offloader(block_type, blocks, num_blocks, blocks_to_swap, config):
        captured.append(config)
        return object()

    monkeypatch.setattr(flux2_models, "create_offloader", fake_create_offloader)
    return captured


def test_enable_block_swap_default_selector_without_nvfp4(monkeypatch):
    captured = _capture_configs(monkeypatch)
    model = Flux2(_tiny_flux2_params())
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=False)

    model.enable_block_swap(1, config)

    assert len(captured) == 2  # double + single offloader configs
    assert all(c.swap_tensor_selector is None for c in captured)


def test_enable_block_swap_nvfp4_selector_when_single_block_patched(monkeypatch):
    captured = _capture_configs(monkeypatch)
    model = Flux2(_tiny_flux2_params())
    model.single_blocks[0].register_buffer("nvfp4_block_scale", torch.zeros(1))
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=False)

    model.enable_block_swap(1, config)

    assert len(captured) == 2
    assert all(c.swap_tensor_selector is quantized_linear_swap_tensor_selector for c in captured)


def test_enable_block_swap_nvfp4_selector_when_double_block_patched(monkeypatch):
    captured = _capture_configs(monkeypatch)
    model = Flux2(_tiny_flux2_params())
    model.double_blocks[0].register_buffer("nvfp4_block_scale", torch.zeros(1))
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=False)

    model.enable_block_swap(1, config)

    assert len(captured) == 2
    assert all(c.swap_tensor_selector is quantized_linear_swap_tensor_selector for c in captured)


def test_enable_block_swap_does_not_override_explicit_selector(monkeypatch):
    captured = _capture_configs(monkeypatch)

    def custom_selector(block):
        return []

    model = Flux2(_tiny_flux2_params())
    model.single_blocks[0].register_buffer("nvfp4_block_scale", torch.zeros(1))
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=False, swap_tensor_selector=custom_selector)

    model.enable_block_swap(1, config)

    assert all(c.swap_tensor_selector is custom_selector for c in captured)
