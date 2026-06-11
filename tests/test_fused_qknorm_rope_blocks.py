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


def make_pe(B, L, D, device, theta=2000, offset: int = 0):
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
    pos = pos + offset
    return rope(pos, D, theta).unsqueeze(1)


def make_mod(B, hidden, device, seed, dtype=torch.float32):
    gen = torch.Generator(device=device).manual_seed(seed)
    return tuple(torch.randn(B, 1, hidden, device=device, dtype=dtype, generator=gen) * 0.1 for _ in range(3))


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


def run_double(block, img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params):
    img = img.detach().clone().requires_grad_(True)
    txt = txt.detach().clone().requires_grad_(True)
    out_img, out_txt = block(img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params)
    (out_img.sum() + out_txt.sum()).backward()
    grads = {n: p.grad.detach().clone() for n, p in block.named_parameters() if p.grad is not None}
    return out_img.detach(), out_txt.detach(), img.grad.detach().clone(), txt.grad.detach().clone(), grads


def test_double_stream_block_fused_matches_eager():
    torch.manual_seed(0)
    B, L_img, L_txt = 2, 33, 11
    head_dim = HIDDEN // HEADS
    block = DoubleStreamBlock(HIDDEN, HEADS, mlp_ratio=4.0).to(DEVICE).float()
    block_fused = copy.deepcopy(block)
    block_fused.fused_qknorm_rope = True

    img = torch.randn(B, L_img, HIDDEN, device=DEVICE)
    txt = torch.randn(B, L_txt, HIDDEN, device=DEVICE)
    # img positions offset by 100 to mirror disjoint id spaces in real model,
    # keeping the test sensitive to pe/pe_ctx swaps even if lengths are equalized.
    pe = make_pe(B, L_img, head_dim, DEVICE, offset=100)
    pe_ctx = make_pe(B, L_txt, head_dim, DEVICE)
    mod_img = (make_mod(B, HIDDEN, DEVICE, seed=2), make_mod(B, HIDDEN, DEVICE, seed=3))
    mod_txt = (make_mod(B, HIDDEN, DEVICE, seed=4), make_mod(B, HIDDEN, DEVICE, seed=5))
    attn_params = AttentionParams.create_attention_params("torch", False)

    res_e = run_double(block, img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params)
    res_f = run_double(block_fused, img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params)

    tol = dict(rtol=1e-4, atol=1e-4)
    for a, b, what in [
        (res_f[0], res_e[0], "img out"),
        (res_f[1], res_e[1], "txt out"),
        (res_f[2], res_e[2], "img grad"),
        (res_f[3], res_e[3], "txt grad"),
    ]:
        torch.testing.assert_close(a, b, **tol, msg=f"mismatch: {what}")
    assert res_e[4].keys() == res_f[4].keys()
    for name in res_e[4]:
        torch.testing.assert_close(res_f[4][name], res_e[4][name], **tol, msg=f"grad mismatch: {name}")


def test_double_stream_block_fused_constructor_flag():
    block = DoubleStreamBlock(HIDDEN, HEADS, mlp_ratio=4.0, fused_qknorm_rope=True)
    assert block.fused_qknorm_rope is True
    assert DoubleStreamBlock(HIDDEN, HEADS, mlp_ratio=4.0).fused_qknorm_rope is False


def test_single_stream_block_fused_with_checkpointing():
    """Regression guard: fused path must be numerically equivalent under gradient checkpointing.

    autograd.Function composes with torch.utils.checkpoint; this test guards against
    regressions where recompute changes the result.
    """
    torch.manual_seed(0)
    B, L = 2, 33
    head_dim = HIDDEN // HEADS
    block = SingleStreamBlock(HIDDEN, HEADS, fused_qknorm_rope=True).to(DEVICE).float()
    block_ckpt = copy.deepcopy(block)
    block_ckpt.enable_gradient_checkpointing()
    block.train()
    block_ckpt.train()

    x = torch.randn(B, L, HIDDEN, device=DEVICE)
    pe = make_pe(B, L, head_dim, DEVICE)
    mod = make_mod(B, HIDDEN, DEVICE, seed=1)
    attn_params = AttentionParams.create_attention_params("torch", False)

    out_a, gx_a, grads_a = run_single(block, x, pe, mod, attn_params)
    out_b, gx_b, grads_b = run_single(block_ckpt, x, pe, mod, attn_params)

    tol = dict(rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_b, out_a, **tol)
    torch.testing.assert_close(gx_b, gx_a, **tol)
    assert grads_a.keys() == grads_b.keys()
    for name in grads_a:
        # norm scale grads use a looser tolerance because dscale accumulates via atomic adds
        # (non-deterministic ordering)
        if "norm" in name and "scale" in name:
            scale_tol = dict(rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(grads_b[name], grads_a[name], **scale_tol, msg=f"grad mismatch: {name}")
        else:
            torch.testing.assert_close(grads_b[name], grads_a[name], **tol, msg=f"grad mismatch: {name}")


def test_double_stream_block_fused_with_checkpointing():
    """Regression guard: fused DoubleStreamBlock path must be numerically equivalent
    under gradient checkpointing.

    autograd.Function composes with torch.utils.checkpoint; this test guards against
    regressions where recompute changes the result.
    """
    torch.manual_seed(0)
    B, L_img, L_txt = 2, 33, 11
    head_dim = HIDDEN // HEADS
    block = DoubleStreamBlock(HIDDEN, HEADS, mlp_ratio=4.0, fused_qknorm_rope=True).to(DEVICE).float()
    block_ckpt = copy.deepcopy(block)
    block_ckpt.enable_gradient_checkpointing()
    block.train()
    block_ckpt.train()

    img = torch.randn(B, L_img, HIDDEN, device=DEVICE)
    txt = torch.randn(B, L_txt, HIDDEN, device=DEVICE)
    # img positions offset by 100 to mirror disjoint id spaces in real model.
    pe = make_pe(B, L_img, head_dim, DEVICE, offset=100)
    pe_ctx = make_pe(B, L_txt, head_dim, DEVICE)
    mod_img = (make_mod(B, HIDDEN, DEVICE, seed=2), make_mod(B, HIDDEN, DEVICE, seed=3))
    mod_txt = (make_mod(B, HIDDEN, DEVICE, seed=4), make_mod(B, HIDDEN, DEVICE, seed=5))
    attn_params = AttentionParams.create_attention_params("torch", False)

    res_a = run_double(block, img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params)
    res_b = run_double(block_ckpt, img, txt, pe, pe_ctx, mod_img, mod_txt, attn_params)

    tol = dict(rtol=1e-5, atol=1e-5)
    for a, b, what in [
        (res_b[0], res_a[0], "img out"),
        (res_b[1], res_a[1], "txt out"),
        (res_b[2], res_a[2], "img grad"),
        (res_b[3], res_a[3], "txt grad"),
    ]:
        torch.testing.assert_close(a, b, **tol, msg=f"mismatch: {what}")
    assert res_a[4].keys() == res_b[4].keys()
    for name in res_a[4]:
        # norm scale grads use a looser tolerance because dscale accumulates via atomic adds
        # (non-deterministic ordering)
        if "norm" in name and "scale" in name:
            scale_tol = dict(rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(res_b[4][name], res_a[4][name], **scale_tol, msg=f"grad mismatch: {name}")
        else:
            torch.testing.assert_close(res_b[4][name], res_a[4][name], **tol, msg=f"grad mismatch: {name}")


def _flash_available():
    from musubi_tuner.modules.attention import flash_attn_qkvpacked_func

    return flash_attn_qkvpacked_func is not None


@pytest.mark.skipif(not _flash_available(), reason="flash_attn not available")
def test_single_stream_block_fused_flash_matches_eager_bf16():
    """Production path: bf16 fused block with attn_mode=flash (packed handoff to
    flash_attn_qkvpacked_func) vs the eager block on the torch backend."""
    torch.manual_seed(0)
    B, L = 2, 33
    head_dim = HIDDEN // HEADS
    block = SingleStreamBlock(HIDDEN, HEADS).to(DEVICE).to(torch.bfloat16)
    block_fused = copy.deepcopy(block)
    block_fused.fused_qknorm_rope = True

    x = torch.randn(B, L, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    pe = make_pe(B, L, head_dim, DEVICE)
    mod = make_mod(B, HIDDEN, DEVICE, seed=1, dtype=torch.bfloat16)

    out_e, gx_e, _ = run_single(block, x, pe, mod, AttentionParams.create_attention_params("torch", False))
    out_f, gx_f, _ = run_single(block_fused, x, pe, mod, AttentionParams.create_attention_params("flash", False))

    # bf16 + different attention backends: param grads are checked by the fp32
    # torch-backend tests; here outputs and input grads pin the flash packed path.
    tol = dict(rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(out_f.float(), out_e.float(), **tol)
    torch.testing.assert_close(gx_f.float(), gx_e.float(), **tol)


@pytest.mark.skipif(not _flash_available(), reason="flash_attn not available")
def test_double_stream_block_fused_flash_matches_eager_bf16():
    """Production path: bf16 fused DoubleStreamBlock with attn_mode=flash vs eager/torch."""
    torch.manual_seed(0)
    B, L_img, L_txt = 2, 33, 11
    head_dim = HIDDEN // HEADS
    block = DoubleStreamBlock(HIDDEN, HEADS, mlp_ratio=4.0).to(DEVICE).to(torch.bfloat16)
    block_fused = copy.deepcopy(block)
    block_fused.fused_qknorm_rope = True

    img = torch.randn(B, L_img, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    txt = torch.randn(B, L_txt, HIDDEN, device=DEVICE, dtype=torch.bfloat16)
    pe = make_pe(B, L_img, head_dim, DEVICE, offset=100)
    pe_ctx = make_pe(B, L_txt, head_dim, DEVICE)
    mod_img = (make_mod(B, HIDDEN, DEVICE, seed=2, dtype=torch.bfloat16), make_mod(B, HIDDEN, DEVICE, seed=3, dtype=torch.bfloat16))
    mod_txt = (make_mod(B, HIDDEN, DEVICE, seed=4, dtype=torch.bfloat16), make_mod(B, HIDDEN, DEVICE, seed=5, dtype=torch.bfloat16))

    res_e = run_double(block, img, txt, pe, pe_ctx, mod_img, mod_txt, AttentionParams.create_attention_params("torch", False))
    res_f = run_double(block_fused, img, txt, pe, pe_ctx, mod_img, mod_txt, AttentionParams.create_attention_params("flash", False))

    tol = dict(rtol=2e-2, atol=2e-2)
    for a, b, what in [
        (res_f[0], res_e[0], "img out"),
        (res_f[1], res_e[1], "txt out"),
        (res_f[2], res_e[2], "img grad"),
        (res_f[3], res_e[3], "txt grad"),
    ]:
        torch.testing.assert_close(a.float(), b.float(), **tol, msg=f"mismatch: {what}")


def test_flux2_model_threads_fused_flag():
    # construction-only; no CUDA needed
    from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params

    params = Flux2Params(
        in_channels=64,
        context_in_dim=512,
        hidden_size=256,
        mlp_ratio=4.0,
        num_heads=2,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[32, 32, 32, 32],
        theta=2000,
        use_guidance_embed=False,
    )
    model = Flux2(params, attn_mode="torch", split_attn=False, fused_qknorm_rope=True)
    assert all(b.fused_qknorm_rope for b in model.double_blocks)
    assert all(b.fused_qknorm_rope for b in model.single_blocks)

    model_off = Flux2(params, attn_mode="torch", split_attn=False)
    assert not any(b.fused_qknorm_rope for b in model_off.single_blocks)
