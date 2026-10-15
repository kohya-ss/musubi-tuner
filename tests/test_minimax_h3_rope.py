import sys
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.model import _apply_rope_split_half


def _rope_matmul_reference(hidden_states: torch.Tensor, rotation_table: torch.Tensor) -> torch.Tensor:
    """The original formulation: one 2x2 @ 2x1 product per (position, head, pair)."""
    pairs = rotation_table.shape[-3]
    rotary = torch.stack((hidden_states[..., :pairs], hidden_states[..., pairs : 2 * pairs]), dim=-1)
    rotary = torch.matmul(rotation_table, rotary.unsqueeze(-1)).squeeze(-1)
    rotary = torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)
    return torch.cat((rotary, hidden_states[..., 2 * pairs :]), dim=-1)


def _rotation_table(batch: int, length: int, pairs: int, dtype: torch.dtype) -> torch.Tensor:
    """Same layout as MiniMaxH3Model._rotation_table: [B, L, 1, pairs, 2, 2] of (cos, -sin, sin, cos)."""
    angles = torch.randn(batch, length, pairs) * 50
    cosine, sine = torch.cos(angles), torch.sin(angles)
    return torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(batch, length, 1, pairs, 2, 2).to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("head_dim", "pairs"), [(128, 48), (128, 64), (16, 4)])
def test_elementwise_rope_matches_the_matmul_formulation(dtype, head_dim, pairs):
    torch.manual_seed(0)
    table = _rotation_table(2, 37, pairs, dtype)
    hidden = torch.randn(2, 37, 5, head_dim, dtype=dtype, requires_grad=True)

    expected = _rope_matmul_reference(hidden, table)
    actual = _apply_rope_split_half(hidden, table)

    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # the channels past 2 * pairs are passed through untouched
    torch.testing.assert_close(actual[..., 2 * pairs :], hidden[..., 2 * pairs :], rtol=0, atol=0)
    (expected_grad,) = torch.autograd.grad(expected.float().square().sum(), hidden)
    (actual_grad,) = torch.autograd.grad(actual.float().square().sum(), hidden)
    if dtype == torch.bfloat16:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


class _OpRecorder(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def test_elementwise_rope_dispatches_no_matmul():
    """A batched 2x2 matmul here runs as GEMV launches on CUDA and dominates the training step."""
    table = _rotation_table(1, 8, 4, torch.bfloat16)
    hidden = torch.randn(1, 8, 3, 16, dtype=torch.bfloat16, requires_grad=True)
    with _OpRecorder() as recorder:
        _apply_rope_split_half(hidden, table).float().sum().backward()
    assert not recorder.ops & {"matmul", "bmm", "mm", "baddbmm", "addmm", "mv"}
