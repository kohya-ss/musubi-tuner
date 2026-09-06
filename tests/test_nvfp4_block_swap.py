"""Regression tests for NVFP4 + block swap.

Task 7's smoke test uncovered a real bug: the block-swap offloader's default swap-tensor
selector only tracks each Linear's ``.weight``. An NVFP4-patched Linear (``training=True``)
also carries a full second (columnwise) weight copy in ``nvfp4_weight_t`` for the backward
GEMM -- leaving it out of the selector doesn't just skip a small scale vector, it means the
offloader's per-block ``.to(device)`` call drags that full-size buffer onto the GPU once and
never streams it back off, silently pinning every block's columnwise copy resident and
defeating block swap's memory savings. ``quantized_linear_swap_tensor_selector`` (nvfp4_utils.py) fixes
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
    quantized_linear_swap_tensor_selector,
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
        blocks,
        state_dict,
        quantizer.nvfp4_module_shapes,
        [],
        use_scaled_mm=True,
        training=True,
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
        device=device,
        supports_backward=True,
        use_pinned_memory=True,
        h2d_only=True,
        ring_size=1,
    )  # swap_tensor_selector left at its default (None) -- reproduces the bug
    offloader = create_offloader("test", list(blocks), 4, 2, config)
    offloader.prepare_block_devices_before_forward(list(blocks))

    assert all(b.proj.nvfp4_weight_t.device.type == "cuda" for b in blocks)


@requires_cuda
def test_quantized_linear_swap_tensor_selector_keeps_non_resident_columnwise_buffers_on_cpu(tmp_path):
    """The fix: with quantized_linear_swap_tensor_selector, nvfp4_weight_t streams through the same
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
        swap_tensor_selector=quantized_linear_swap_tensor_selector,
    )
    offloader = create_offloader("test", list(blocks), 4, 2, config)
    offloader.prepare_block_devices_before_forward(list(blocks))

    devices = [b.proj.nvfp4_weight_t.device.type for b in blocks]
    assert devices.count("cpu") >= 1, f"expected at least one streaming block's columnwise buffer to stay on CPU, got {devices}"


@requires_cuda
def test_model_offloader_ignores_swap_tensor_selector_h2d_only_false(tmp_path):
    """Documents the gap issue 1 in the NVFP4 punch list left open: create_offloader only
    forwards swap_tensor_selector on the h2d_only branch (LoRAStreamOffloader). The default
    ModelOffloader path (h2d_only=False) silently drops it, so even a caller that supplies
    quantized_linear_swap_tensor_selector still leaks the full columnwise buffer GPU-resident for every
    block. Krea2NetworkTrainer.handle_model_specific_args now rejects this combination at the
    CLI level (see tests/test_krea2_nvfp4.py), but ModelOffloader/create_offloader remain
    reachable directly, so this guards the underlying limitation until punch-list option (a)
    (teaching ModelOffloader to honor the selector) is implemented."""
    blocks = _build_patched_blocks(tmp_path, num_blocks=4)
    device = torch.device("cuda")
    config = BlockSwapConfig(
        device=device,
        supports_backward=True,
        use_pinned_memory=True,
        h2d_only=False,
        swap_tensor_selector=quantized_linear_swap_tensor_selector,
    )
    offloader = create_offloader("test", list(blocks), 4, 2, config)
    offloader.prepare_block_devices_before_forward(list(blocks))

    assert all(b.proj.nvfp4_weight_t.device.type == "cuda" for b in blocks), (
        "expected the still-unfixed ModelOffloader path to leave every block's columnwise buffer"
        " GPU-resident (swap_tensor_selector ignored) -- if this now fails, ModelOffloader has"
        " started honoring swap_tensor_selector and this test (and its docstring) should be updated"
        " to assert the fixed behavior instead."
    )


def _build_mixed_blocks(tmp_path: Path, num_plain=2, num_quantized=4, n=64, k=64):
    """num_plain leading blocks are plain bf16 nn.Linear; the rest are NVFP4-quantized --
    mirrors a partially pre-quantized checkpoint (e.g. Krea 2's unquantized leading blocks)."""
    quantized_blocks = _build_patched_blocks(tmp_path, num_blocks=num_quantized, n=n, k=k)
    plain_blocks = nn.ModuleList([_TinyBlock(n, k) for _ in range(num_plain)])
    return list(plain_blocks) + list(quantized_blocks)


@requires_cuda
def test_lora_stream_offloader_excludes_structurally_different_leading_blocks(tmp_path):
    """The bug: 2 plain bf16 blocks + 4 NVFP4-quantized blocks, blocks_to_swap=5 (more than the
    4 homogeneous quantized blocks) used to spread stream_idx across all 6 indices via the naive
    midpoint formula, picking up a plain block alongside quantized ones and crashing on the
    ring's shared-layout assertion the first time prepare_block_devices_before_forward runs.
    The fix: construction itself must reject this (clear error), not crash deep in prepare()."""
    blocks = _build_mixed_blocks(tmp_path, num_plain=2, num_quantized=4)
    device = torch.device("cuda")
    config = BlockSwapConfig(
        device=device,
        supports_backward=True,
        use_pinned_memory=True,
        h2d_only=True,
        ring_size=2,
        swap_tensor_selector=quantized_linear_swap_tensor_selector,
    )
    with pytest.raises(ValueError, match="eligible") as exc_info:
        create_offloader("test", blocks, 6, 5, config)
    message = str(exc_info.value)
    assert "Eligible (majority layout): blocks 2-5" in message
    assert "Ineligible (different layout, always GPU-resident): blocks 0-1" in message


def test_format_block_ranges_handles_interleaved_indices():
    """Covers the user-raised concern: eligibility isn't always a clean leading/trailing split --
    if quantized/unquantized blocks alternate, the error message must still be legible rather than
    implying a simple prefix/suffix pattern."""
    from musubi_tuner.modules.custom_offloading_utils import _format_block_ranges

    assert _format_block_ranges([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]) == "2-25"
    assert _format_block_ranges([0, 1, 26, 27]) == "0-1, 26-27"
    assert _format_block_ranges([0, 2, 4, 6]) == "0, 2, 4, 6"  # fully interleaved: no ranges to collapse
    assert _format_block_ranges([]) == "(none)"
    assert _format_block_ranges([5]) == "5"


@requires_cuda
def test_lora_stream_offloader_streams_only_eligible_blocks_when_request_fits(tmp_path):
    """With blocks_to_swap<=4 (the homogeneous quantized count), construction succeeds and
    stream_idx only ever contains indices from the quantized block range [2, 3, 4, 5] --
    the plain blocks 0/1 are never selected, and prepare_block_devices_before_forward runs
    without the shape/dtype assertion firing."""
    blocks = _build_mixed_blocks(tmp_path, num_plain=2, num_quantized=4)
    device = torch.device("cuda")
    config = BlockSwapConfig(
        device=device,
        supports_backward=True,
        use_pinned_memory=True,
        h2d_only=True,
        ring_size=2,
        swap_tensor_selector=quantized_linear_swap_tensor_selector,
    )
    offloader = create_offloader("test", blocks, 6, 4, config)
    assert all(idx >= 2 for idx in offloader.stream_idx), (
        f"expected only quantized blocks (index >= 2) to be swap-eligible, got {offloader.stream_idx}"
    )
    offloader.prepare_block_devices_before_forward(blocks)  # must not raise
    assert blocks[0].proj.weight.device.type == "cuda"  # plain block: always resident
    assert blocks[1].proj.weight.device.type == "cuda"  # plain block: always resident
