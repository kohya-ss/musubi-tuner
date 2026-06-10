"""Block-level tests: SingleStreamBlock / DoubleStreamBlock with
fused_qknorm_rope on vs off must match (outputs and grads).

Run with:
  uv run --no-sync pytest tests/test_fused_qknorm_rope_blocks.py -v
"""

import copy

import pytest
import torch

from musubi_tuner.flux_2.flux2_models import DoubleStreamBlock, SingleStreamBlock, rope
from musubi_tuner.modules.attention import AttentionParams

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

HIDDEN = 512
HEADS = 4  # head_dim = 128, matching FLUX.2
DEVICE = "cuda"


def make_pe(B, L, D, device, theta=2000):
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
    return rope(pos, D, theta).unsqueeze(1)


def make_mod(B, hidden, device, seed):
    gen = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(B, 1, hidden, device=device, generator=gen) * 0.1 for _ in range(3))


def run_single(block, x, pe, mod, attn_params):
    x = x.detach().clone().requires_grad_(True)
    out = block(x, pe, mod, attn_params)
    out.sum().backward()
    grads = {n: p.grad.detach().clone() for n, p in block.named_parameters() if p.grad is not None}
    return out.detach(), x.grad.detach().clone(), grads


@pytest.mark.parametrize("L", [33, 64])
def test_single_stream_block_fused_matches_eager(L):
    torch.manual_seed(0)
    B = 2
    head_dim = HIDDEN // HEADS
    block = SingleStreamBlock(HIDDEN, HEADS).to(DEVICE).float()
    block_fused = copy.deepcopy(block)
    block_fused.fused_qknorm_rope = True

    x = torch.randn(B, L, HIDDEN, device=DEVICE)
    pe = make_pe(B, L, head_dim, DEVICE)
    mod = make_mod(B, HIDDEN, DEVICE, seed=1)
    attn_params = AttentionParams.create_attention_params("torch", False)

    out_e, gx_e, grads_e = run_single(block, x, pe, mod, attn_params)
    out_f, gx_f, grads_f = run_single(block_fused, x, pe, mod, attn_params)

    tol = dict(rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(out_f, out_e, **tol)
    torch.testing.assert_close(gx_f, gx_e, **tol)
    assert grads_e.keys() == grads_f.keys()
    for name in grads_e:
        torch.testing.assert_close(grads_f[name], grads_e[name], **tol, msg=f"grad mismatch: {name}")


def test_single_stream_block_fused_constructor_flag():
    block = SingleStreamBlock(HIDDEN, HEADS, fused_qknorm_rope=True)
    assert block.fused_qknorm_rope is True
    assert SingleStreamBlock(HIDDEN, HEADS).fused_qknorm_rope is False
