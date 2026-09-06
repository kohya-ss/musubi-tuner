"""Regression tests for ConvRot INT8 + block swap.

A GPU smoke test of a downstream extension (see boo-musubi-tuner's
notes/tdm-distill-smoke-test.md) found that `--convrot_int8 --blocks_to_swap N` (without
`--block_swap_h2d_only`, i.e. the ModelOffloader path) crashed inside the Triton kernel with
`ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)`. Root cause: ConvRot's
per-layer `scale_weight` buffer was never part of ModelOffloader's swap-tensor tracking, so it
went stale/wrong-device once blocks started physically swapping between CPU and GPU object
positions. Mirrors tests/test_nvfp4_block_swap.py's structure for the analogous NVFP4 bug.
"""

import pytest
import torch
import torch.nn as nn

from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import (
    CONVROT_GROUPSIZE,
    apply_convrot_int8_monkey_patch,
)
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.modules.nvfp4_utils import quantized_linear_swap_tensor_selector

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap requires CUDA")

GS = CONVROT_GROUPSIZE  # 256
N, K = 64, 256  # out_features, in_features (must be divisible by GS)


class _TinyBlock(nn.Module):
    def __init__(self, n=N, k=K):
        super().__init__()
        self.proj = nn.Linear(k, n, bias=False)

    def forward(self, x):
        return self.proj(x)


def _build_convrot_patched_blocks(num_blocks=4, n=N, k=K, seed=0):
    torch.manual_seed(seed)
    blocks = nn.ModuleList([_TinyBlock(n, k) for _ in range(num_blocks)])
    state_dict = {}
    for i in range(num_blocks):
        w = torch.randn(n, k) * 0.02
        wq, scale = quantize_int8_convrot_weight(w, GS)
        state_dict[f"{i}.proj.weight"] = wq
        state_dict[f"{i}.proj.scale_weight"] = scale
    apply_convrot_int8_monkey_patch(blocks, state_dict, bwd_mode="bf16", groupsize=GS)
    blocks.requires_grad_(False)
    blocks.load_state_dict(state_dict, strict=True, assign=True)
    return blocks


@requires_cuda
def test_default_selector_leaves_convrot_scale_weight_stale_after_model_offloader_swap():
    """Documents the bug: with the offloader's default selector (swap_tensor_selector=None),
    scale_weight never moves during a runtime swap, so it ends up bound to whichever block
    object it started on -- not the logical block whose (correctly swapped) .weight now sits
    there."""
    blocks = _build_convrot_patched_blocks(num_blocks=4)
    device = torch.device("cuda")
    config = BlockSwapConfig(device=device, supports_backward=True, use_pinned_memory=True, h2d_only=False)
    offloader = create_offloader("test", list(blocks), 4, 2, config)
    offloader.prepare_block_devices_before_forward(list(blocks))

    # blocks[2:4] are placed on CPU by prepare(); scale_weight (not selected by the default
    # selector) stays wherever `.to(device)` put it during prepare -- GPU -- even though .weight
    # is correctly on CPU. This mismatch is exactly what crashed the Triton kernel downstream.
    assert blocks[2].proj.weight.device.type == "cpu"
    assert blocks[2].proj.scale_weight.device.type == "cuda"
    assert blocks[3].proj.weight.device.type == "cpu"
    assert blocks[3].proj.scale_weight.device.type == "cuda"


@requires_cuda
def test_quantized_linear_swap_tensor_selector_keeps_scale_weight_with_weight_in_model_offloader():
    """The fix: with quantized_linear_swap_tensor_selector wired in, scale_weight tracks .weight
    through both prepare_block_devices_before_forward and a real runtime swap round submitted via
    _submit_move_blocks (what the backward hook calls during actual training)."""
    blocks = _build_convrot_patched_blocks(num_blocks=4)
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

    for i in (0, 1):
        assert blocks[i].proj.weight.device.type == "cuda"
        assert blocks[i].proj.scale_weight.device.type == "cuda"
    for i in (2, 3):
        assert blocks[i].proj.weight.device.type == "cpu"
        assert blocks[i].proj.scale_weight.device.type == "cpu"

    # simulate a runtime swap round: block 1 (currently CUDA) leaves the resident window and moves
    # to CPU, while block 3 (currently CPU) enters the window and moves to CUDA -- the same
    # (block_idx_to_cpu, block_idx_to_cuda) argument order the backward hook uses in
    # _wait_blocks_move / submit_move_blocks_forward.
    offloader._submit_move_blocks(list(blocks), 1, 3)
    offloader._wait_blocks_move(3)

    assert blocks[1].proj.weight.device.type == "cpu"
    assert blocks[1].proj.scale_weight.device.type == "cpu"
    assert blocks[3].proj.weight.device.type == "cuda"
    assert blocks[3].proj.scale_weight.device.type == "cuda"
