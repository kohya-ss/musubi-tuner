"""Tests for MiniMax-H3 text encoder layer streaming.

Covers the generalized multi-tensor streaming of ``LoRAStreamOffloader`` (swap-tensor
selectors, flat-buffer layout, hook-driven forwarding) on the CPU, plus a CUDA
end-to-end check of the offloader itself. The numerical gate (streaming on/off produces
bit-identical hidden states on the real model) runs in the machine smoke tests.
"""

from pathlib import Path
import sys

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.text_encoder import _te_swap_tensor_selector
from musubi_tuner.modules.custom_offloading_utils import (
    LoRAStreamOffloader,
    attach_forward_streaming_hooks,
    default_swap_tensor_selector,
)


class _QuantLinear(nn.Linear):
    """Class name ends with Linear, like the fp8/quantization monkey-patch targets."""


def test_default_selector_matches_legacy_linear_weight_rule():
    block = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4), _QuantLinear(4, 4), nn.Conv1d(1, 1, 1))

    jobs = default_swap_tensor_selector(block)

    assert [(id(module), name) for module, name in jobs] == [(id(block[0]), "weight"), (id(block[2]), "weight")]


def _tiny_te_block(seed: int) -> nn.Module:
    generator = torch.Generator().manual_seed(seed)

    block = nn.Module()
    block.quant = nn.Linear(8, 4, bias=False)
    block.quant.weight = nn.Parameter(torch.randint(-128, 128, (4, 8), dtype=torch.int8, generator=generator), requires_grad=False)
    block.quant.register_buffer("scale_weight", torch.rand(4, 1, generator=generator))
    block.quant.register_buffer("nvfp4_scale", torch.rand((), generator=generator))
    block.quant.register_buffer("running_stat", torch.rand(4, generator=generator))  # not in the allowlist
    block.plain = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        block.plain.weight.copy_(torch.rand(4, 4, generator=generator))
    block.plain.weight.requires_grad_(False)
    block.norm = nn.LayerNorm(4)
    return block


def test_te_selector_streams_weights_and_allowlisted_quant_buffers_only():
    block = _tiny_te_block(0)

    jobs = _te_swap_tensor_selector(block)

    assert [(id(module), name) for module, name in jobs] == [
        (id(block.quant), "weight"),
        (id(block.quant), "scale_weight"),
        (id(block.quant), "nvfp4_scale"),
        (id(block.plain), "weight"),
    ]


def test_flat_layout_views_roundtrip_mixed_dtypes():
    tensors = [
        torch.randn(3, 5, dtype=torch.bfloat16),
        torch.randint(-128, 128, (4, 8), dtype=torch.int8),
        torch.rand((), dtype=torch.float32),
        torch.randint(0, 256, (2, 6), dtype=torch.uint8),
        torch.randn(2, 4).to(torch.float8_e4m3fn),
    ]

    layout = LoRAStreamOffloader._compute_layout(tensors)
    flat = torch.zeros(layout[1], dtype=torch.uint8)
    views = LoRAStreamOffloader._flat_views(flat, tensors, layout)

    for view, tensor in zip(views, tensors):
        view.copy_(tensor)
    for view, tensor in zip(views, tensors):
        assert view.dtype == tensor.dtype
        assert view.shape == tensor.shape
        assert torch.equal(view.reshape(-1).view(torch.uint8), tensor.reshape(-1).view(torch.uint8))


class _RecordingOffloader:
    def __init__(self, log: list):
        self.log = log

    def wait_for_block(self, block_idx: int):
        self.log.append(("wait", block_idx))

    def submit_move_blocks_forward(self, blocks, block_idx: int):
        self.log.append(("submit", block_idx))


class _LoggingBlock(nn.Module):
    def __init__(self, log: list, tag: int):
        super().__init__()
        self.log = log
        self.tag = tag

    def forward(self, x):
        self.log.append(("compute", self.tag))
        return x + 1


def test_attach_forward_streaming_hooks_waits_before_and_submits_after_each_block():
    log = []
    blocks = [_LoggingBlock(log, index) for index in range(3)]
    handles = attach_forward_streaming_hooks(_RecordingOffloader(log), blocks)

    x = torch.zeros(1)
    for block in blocks:
        x = block(x)

    assert x.item() == 3.0  # a post-hook must not replace the block output
    assert log == [
        ("wait", 0),
        ("compute", 0),
        ("submit", 0),
        ("wait", 1),
        ("compute", 1),
        ("submit", 1),
        ("wait", 2),
        ("compute", 2),
        ("submit", 2),
    ]
    for handle in handles:
        handle.remove()


def test_selector_returning_unknown_attribute_is_rejected():
    block = _tiny_te_block(0)
    offloader = LoRAStreamOffloader.__new__(LoRAStreamOffloader)  # job resolution only, no CUDA
    offloader._blocks = [block]
    offloader._job_cache = {}
    offloader.swap_tensor_selector = lambda b: [(b.quant, "not_a_tensor")]

    with pytest.raises(ValueError, match="neither a parameter nor a registered buffer"):
        offloader._jobs(0)


def _snapshot(block: nn.Module) -> dict[str, torch.Tensor]:
    return {
        "quant.weight": block.quant.weight.detach().clone(),
        "quant.scale_weight": block.quant.scale_weight.clone(),
        "quant.nvfp4_scale": block.quant.nvfp4_scale.clone(),
        "plain.weight": block.plain.weight.detach().clone(),
    }


def _assert_block_matches(block: nn.Module, expected: dict[str, torch.Tensor], device_type: str):
    for key, value in expected.items():
        owner_name, attr = key.split(".")
        actual = getattr(getattr(block, owner_name), attr)
        assert actual.device.type == device_type, f"{key} on {actual.device}, expected {device_type}"
        assert torch.equal(actual.cpu(), value), f"{key} does not match its master"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for LoRAStreamOffloader")
def test_lora_stream_offloader_streams_quantized_blocks_from_cpu_masters():
    device = torch.device("cuda")
    blocks = [_tiny_te_block(seed) for seed in range(4)]
    expected = [_snapshot(block) for block in blocks]

    offloader = LoRAStreamOffloader(
        "te-test",
        blocks,
        num_blocks=4,
        blocks_to_swap=4,
        supports_backward=False,
        device=device,
        ring_size=2,
        use_pinned_memory=False,
        swap_tensor_selector=_te_swap_tensor_selector,
    )
    offloader.prepare_block_devices_before_forward(blocks)
    torch.cuda.synchronize()

    # non-swap parts of every block are resident on the device (CPU-direct first prepare)
    for block in blocks:
        assert block.norm.weight.device.type == "cuda"
        assert block.quant.running_stat.device.type == "cuda"  # not allowlisted -> resident
    # ring size 2: blocks 0/1 are preloaded, blocks 2/3 sit on their CPU masters
    _assert_block_matches(blocks[0], expected[0], "cuda")
    _assert_block_matches(blocks[2], expected[2], "cpu")
    _assert_block_matches(blocks[3], expected[3], "cpu")

    for _pass in range(2):  # second pass exercises the wrap-around preload
        for index in range(4):
            offloader.wait_for_block(index)
            torch.cuda.synchronize()
            _assert_block_matches(blocks[index], expected[index], "cuda")
            offloader.submit_move_blocks_forward(blocks, index)
    torch.cuda.synchronize()
