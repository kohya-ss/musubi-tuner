"""Regression tests for block swap under a backward that walks more than one forward graph.

ModelOffloader's backward hooks were built around one backward traversal per forward traversal:
each hook hands the next block down to the GPU, and the traversal ends with the ring parked back at
its start-of-forward layout. A loss that sums terms from *separate* forward passes through the same
blocks (distillation objectives, auxiliary/regularizer terms, multi-sample rollouts) makes autograd
traverse the stack once per graph -- and every traversal after the first ran against blocks the ring
had already moved back to host.

Found by a downstream extension's GPU smoke test (boo-musubi-tuner's
notes/tdm-distill-smoke-test.md): with gradient checkpointing the failure surfaces inside the
recompute as a kernel receiving a host pointer, which points nowhere near the real cause.
"""

import pytest
import torch
import torch.nn as nn

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="block swap requires CUDA")

NUM_BLOCKS = 8
BLOCKS_TO_SWAP = 6  # leaves 2 resident, the tight ratio where a stale traversal fails immediately
DIM = 32


class _Block(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.lin = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return torch.relu(self.lin(x))


class _Stack(nn.Module):
    """Minimal stand-in for a model's per-block loop, including the offloader wait/submit calls."""

    def __init__(self, num_blocks=NUM_BLOCKS, dim=DIM):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(num_blocks)])
        self.offloader = None
        # Mirrors real training: the swapped base weights are frozen (quantized base / LoRA setups),
        # so backward needs them to compute grad w.r.t. the activations but never grads them.
        self.blocks.requires_grad_(False)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if self.offloader is not None:
                self.offloader.wait_for_block(i)
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            if self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.blocks, i)
        return x


def _build_stack_with_offloader():
    torch.manual_seed(0)
    stack = _Stack().to("cuda")
    config = BlockSwapConfig(device=torch.device("cuda"), supports_backward=True)
    stack.offloader = create_offloader("test", stack.blocks, NUM_BLOCKS, BLOCKS_TO_SWAP, config)
    stack.offloader.prepare_block_devices_before_forward(stack.blocks)
    return stack


@requires_cuda
def test_backward_over_two_forward_graphs():
    """One backward, two independently-built graphs over the same swapped blocks."""
    stack = _build_stack_with_offloader()

    x1 = torch.randn(2, DIM, device="cuda", requires_grad=True)
    out1 = stack(x1)

    stack.offloader.prepare_block_devices_before_forward(stack.blocks)
    x2 = torch.randn(2, DIM, device="cuda", requires_grad=True)
    out2 = stack(x2)

    # Autograd walks out2's graph top-to-bottom, then out1's -- the second traversal is the one that
    # used to find the top of the stack back on the host.
    (out1.sum() + out2.sum()).backward()

    assert x1.grad is not None and torch.isfinite(x1.grad).all()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()


@requires_cuda
def test_backward_over_two_forward_graphs_matches_no_swap_gradients():
    """The re-seated ring must not just avoid crashing -- it must produce the same gradients."""
    torch.manual_seed(0)
    reference = _Stack().to("cuda")
    x1 = torch.randn(2, DIM, device="cuda")
    x2 = torch.randn(2, DIM, device="cuda")

    ref_x1, ref_x2 = x1.clone().requires_grad_(True), x2.clone().requires_grad_(True)
    (reference(ref_x1).sum() + reference(ref_x2).sum()).backward()

    stack = _build_stack_with_offloader()
    swap_x1, swap_x2 = x1.clone().requires_grad_(True), x2.clone().requires_grad_(True)
    out1 = stack(swap_x1)
    stack.offloader.prepare_block_devices_before_forward(stack.blocks)
    out2 = stack(swap_x2)
    (out1.sum() + out2.sum()).backward()

    torch.testing.assert_close(swap_x1.grad, ref_x1.grad)
    torch.testing.assert_close(swap_x2.grad, ref_x2.grad)


@requires_cuda
def test_single_graph_backward_never_reseats_the_ring():
    """The residency guard must stay a no-op on the ordinary one-forward-one-backward path.

    Re-seating moves every swapped block, so a guard that fired on the common case would quietly
    cost far more than block swap saves.
    """
    stack = _build_stack_with_offloader()
    calls = []
    original = stack.offloader._ensure_resident_for_backward
    stack.offloader._ensure_resident_for_backward = lambda blocks, idx: (calls.append(idx), original(blocks, idx))[1]

    x = torch.randn(2, DIM, device="cuda", requires_grad=True)
    stack(x).sum().backward()

    assert calls == [], f"ring was re-seated on a single-graph backward at blocks {calls}"


@requires_cuda
def test_forward_after_forward_without_reset_is_recovered():
    """A second forward with no explicit reset in between must also work.

    Trainers that call the model several times per step (rather than once) hit the same stale-ring
    problem on the forward side; the guard covers it for free, since it re-seats to the
    start-of-forward window when it trips at block 0.
    """
    stack = _build_stack_with_offloader()

    x1 = torch.randn(2, DIM, device="cuda", requires_grad=True)
    out1 = stack(x1)
    x2 = torch.randn(2, DIM, device="cuda", requires_grad=True)  # no prepare_block_devices call here
    out2 = stack(x2)

    (out1.sum() + out2.sum()).backward()

    assert x1.grad is not None and torch.isfinite(x1.grad).all()
    assert x2.grad is not None and torch.isfinite(x2.grad).all()
