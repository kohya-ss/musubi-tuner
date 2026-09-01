"""Regression tests for NVFP4 + block swap.

Task 7's smoke test uncovered a real bug: the block-swap offloader's default swap-tensor
selector only tracks each Linear's ``.weight``. An NVFP4-patched Linear (``training=True``)
also carries a full second (columnwise) weight copy in ``nvfp4_weight_t`` for the backward
GEMM -- leaving it out of the selector doesn't just skip a small scale vector, it means the
offloader's per-block ``.to(device)`` call drags that full-size buffer onto the GPU once and
never streams it back off, silently pinning every block's columnwise copy resident and
defeating block swap's memory savings. ``nvfp4_swap_tensor_selector`` (nvfp4_utils.py) fixes
this by routing it through the same ring/master swap machinery as ``.weight``.
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.modules.nvfp4_utils import (
    NvFp4Quantizer,
    _quantize_nvfp4_2d,
    apply_nvfp4_monkey_patch,
    block_has_nvfp4_patched_linear,
    nvfp4_swap_tensor_selector,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap requires CUDA")


class _TinyBlock(nn.Module):
    def __init__(self, n=64, k=64):
        super().__init__()
        self.proj = nn.Linear(k, n, bias=False)

    def forward(self, x):
        return self.proj(x)


def _build_patched_blocks(tmp_path: Path, num_blocks=4, n=64, k=64):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    tensors = {}
    for i in range(num_blocks):
        weight = torch.randn(n, k) * 0.02
        packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(weight)
        payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
        tensors[f"{i}.proj.weight"] = packed
        tensors[f"{i}.proj.weight_scale"] = block_scale
        tensors[f"{i}.proj.weight_scale_2"] = tensor_scale
        tensors[f"{i}.proj.comfy_quant"] = payload

    path = tmp_path / "artifact.safetensors"
    save_file(tensors, str(path))

    quantizer = NvFp4Quantizer()
    state_dict = quantizer.load_and_quantize([str(path)], None)  # stays on CPU
    blocks = nn.ModuleList([_TinyBlock(n, k) for _ in range(num_blocks)])
    apply_nvfp4_monkey_patch(
        blocks, state_dict, quantizer.nvfp4_module_shapes, [], use_scaled_mm=True, training=True,
    )
    blocks.requires_grad_(False)
    blocks.load_state_dict(state_dict, strict=True, assign=True)
    return blocks


def test_block_has_nvfp4_patched_linear(tmp_path):
    blocks = _build_patched_blocks(tmp_path)
    assert all(block_has_nvfp4_patched_linear(b) for b in blocks)
    assert not block_has_nvfp4_patched_linear(nn.Linear(4, 4))


@requires_cuda
def test_default_selector_leaves_nvfp4_weight_t_gpu_resident_for_every_block(tmp_path):
    """Documents the bug: with the offloader's default selector, every streaming block's
    columnwise buffer ends up GPU-resident, not just the ring's worth."""
    blocks = _build_patched_blocks(tmp_path, num_blocks=4)
    device = torch.device("cuda")
    config = BlockSwapConfig(
        device=device, supports_backward=True, use_pinned_memory=True, h2d_only=True, ring_size=1,
    )  # swap_tensor_selector left at its default (None) -- reproduces the bug
    offloader = create_offloader("test", list(blocks), 4, 2, config)
    offloader.prepare_block_devices_before_forward(list(blocks))

    assert all(b.proj.nvfp4_weight_t.device.type == "cuda" for b in blocks)


@requires_cuda
def test_nvfp4_swap_tensor_selector_keeps_non_resident_columnwise_buffers_on_cpu(tmp_path):
    """The fix: with nvfp4_swap_tensor_selector, nvfp4_weight_t streams through the same
    ring/master machinery as .weight -- only ring_size streaming blocks are ever GPU-resident
    at once, so with ring_size=1 and 2 streaming blocks, at least one must stay on CPU."""
    blocks = _build_patched_blocks(tmp_path, num_blocks=4)
    device = torch.device("cuda")
    config = BlockSwapConfig(
        device=device,
        supports_backward=True,
        use_pinned_memory=True,
        h2d_only=True,
        ring_size=1,
        swap_tensor_selector=nvfp4_swap_tensor_selector,
    )
    offloader = create_offloader("test", list(blocks), 4, 2, config)
    offloader.prepare_block_devices_before_forward(list(blocks))

    devices = [b.proj.nvfp4_weight_t.device.type for b in blocks]
    assert devices.count("cpu") >= 1, f"expected at least one streaming block's columnwise buffer to stay on CPU, got {devices}"
