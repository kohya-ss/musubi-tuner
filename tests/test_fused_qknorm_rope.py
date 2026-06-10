"""Kernel-level parity tests: fused_qknorm_rope vs eager QKNorm + apply_rope.

Run with:
  uv run --extra cu132 pytest tests/test_fused_qknorm_rope.py -v
"""

import pytest
import torch
from einops import rearrange

from musubi_tuner.flux_2.flux2_models import apply_rope, rope

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOLS = {
    torch.float32: dict(rtol=1e-5, atol=1e-5),
    torch.float16: dict(rtol=2e-3, atol=2e-3),
    torch.bfloat16: dict(rtol=2e-2, atol=2e-2),
}


def make_pe(B: int, L: int, D: int, device, theta: int = 2000) -> torch.Tensor:
    """Build pe matching EmbedND output: (B, 1, L, D//2, 2, 2), fp32.

    Rows differ per batch element so a kernel bug that always reads b=0 is caught.
    """
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
    # shift each batch element's positions so pe[b] != pe[0] for b > 0
    pos = pos + torch.arange(B, device=device)[:, None] * 100
    return rope(pos, D, theta).unsqueeze(1)


def eager_rms(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # mirrors flux2_models.RMSNorm.forward
    x_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x * rrms).to(dtype=x_dtype) * scale


def eager_reference(qkv: torch.Tensor, q_scale, k_scale, pe, eps: float = 1e-6) -> torch.Tensor:
    """Eager QKNorm + RoPE on packed (B, L, 3, H, D); returns same packed layout."""
    q, k, v = rearrange(qkv, "B L K H D -> K B H L D")
    q = eager_rms(q, q_scale, eps).to(v.dtype)
    k = eager_rms(k, k_scale, eps).to(v.dtype)
    q, k = apply_rope(q, k, pe)
    out = torch.stack([q, k, v], dim=0)  # K B H L D
    return rearrange(out, "K B H L D -> B L K H D")


def make_inputs(B, L, H, D, dtype, device="cuda", seed=0, requires_grad=False):
    gen = torch.Generator(device=device).manual_seed(seed)
    qkv = torch.randn(B, L, 3, H, D, dtype=dtype, device=device, generator=gen)
    q_scale = torch.randn(D, dtype=dtype, device=device, generator=gen) * 0.1 + 1.0
    k_scale = torch.randn(D, dtype=dtype, device=device, generator=gen) * 0.1 + 1.0
    if requires_grad:
        qkv.requires_grad_(True)
        q_scale.requires_grad_(True)
        k_scale.requires_grad_(True)
    pe = make_pe(B, L, D, device)
    return qkv, q_scale, k_scale, pe


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 33, 4, 128), (1, 64, 1, 128), (2, 17, 3, 64), (2, 17, 3, 96)])
def test_forward_parity(dtype, shape):
    from musubi_tuner.modules.fused_qknorm_rope import fused_qknorm_rope

    B, L, H, D = shape
    qkv, q_scale, k_scale, pe = make_inputs(B, L, H, D, dtype)
    out = fused_qknorm_rope(qkv, q_scale, k_scale, pe)
    ref = eager_reference(qkv, q_scale, k_scale, pe)
    assert out.shape == (B, L, 3, H, D)
    assert out.dtype == dtype
    assert out.is_contiguous()
    torch.testing.assert_close(out.float(), ref.float(), **TOLS[dtype])


def test_forward_strided_input_view():
    """Input as a non-contiguous view of (B, L, 3*H*D), like the linear output."""
    from musubi_tuner.modules.fused_qknorm_rope import fused_qknorm_rope

    B, L, H, D = 2, 33, 4, 128
    flat = torch.randn(B, L, 3 * H * D + 16, dtype=torch.float32, device="cuda")
    qkv_f = flat[..., : 3 * H * D].reshape(B, L, 3, H, D).clone().requires_grad_(True)
    qkv_e = qkv_f.detach().clone().requires_grad_(True)
    q_scale = torch.ones(D, device="cuda")
    k_scale = torch.ones(D, device="cuda")
    pe = make_pe(B, L, D, "cuda")
    grad_out = torch.randn(B, L, 3, H, D, device="cuda")

    out_f = fused_qknorm_rope(qkv_f, q_scale, k_scale, pe)
    ref = eager_reference(qkv_e, q_scale, k_scale, pe)
    torch.testing.assert_close(out_f, ref, **TOLS[torch.float32])

    out_f.backward(grad_out)
    ref.backward(grad_out)
    torch.testing.assert_close(qkv_f.grad, qkv_e.grad, **TOLS[torch.float32])


def test_forward_pe_broadcast_batch1():
    """Batch-1 pe with B=2 qkv exercises the stride-0 broadcast path."""
    from musubi_tuner.modules.fused_qknorm_rope import fused_qknorm_rope

    B, L, H, D = 2, 16, 2, 64
    qkv_f, q_scale, k_scale, _ = make_inputs(B, L, H, D, torch.float32, requires_grad=True)
    qkv_e = qkv_f.detach().clone().requires_grad_(True)
    # build a batch-1 pe and broadcast manually for the reference
    pe1 = make_pe(1, L, D, "cuda")  # shape (1, 1, L, D//2, 2, 2)
    # reference: expand pe1 to B so eager uses the same values for all batches
    pe_expanded = pe1.expand(B, -1, -1, -1, -1, -1)
    grad_out = torch.randn(B, L, 3, H, D, device="cuda")

    out_f = fused_qknorm_rope(qkv_f, q_scale, k_scale, pe1)
    ref = eager_reference(qkv_e, q_scale, k_scale, pe_expanded)
    torch.testing.assert_close(out_f.float(), ref.float(), **TOLS[torch.float32])

    out_f.backward(grad_out)
    ref.backward(grad_out)
    torch.testing.assert_close(qkv_f.grad, qkv_e.grad, **TOLS[torch.float32])


@pytest.mark.parametrize("dtype,tol,scale_tol", [
    (torch.float32, dict(rtol=1e-4, atol=1e-4), dict(rtol=1e-4, atol=1e-4)),
    # dscale sums B*H*L bf16 values via fp32 atomics; when scale grads are near
    # zero the relative error is meaningless — use absolute tolerance only.
    # Max abs diff is ~0.25 (1 bf16 ULP at value ~16). atol=0.5 catches real bugs.
    # per plan: bf16 dscale is the noisiest comparison, loosen beyond rtol=5e-2.
    (torch.bfloat16, dict(rtol=3e-2, atol=3e-2), dict(rtol=0.0, atol=0.5)),
])
@pytest.mark.parametrize("shape", [(2, 33, 4, 128), (2, 17, 3, 64), (2, 17, 3, 96)])
def test_backward_parity(dtype, tol, scale_tol, shape):
    from musubi_tuner.modules.fused_qknorm_rope import fused_qknorm_rope

    B, L, H, D = shape
    qkv_f, qs_f, ks_f, pe = make_inputs(B, L, H, D, dtype, seed=1, requires_grad=True)
    qkv_e = qkv_f.detach().clone().requires_grad_(True)
    qs_e = qs_f.detach().clone().requires_grad_(True)
    ks_e = ks_f.detach().clone().requires_grad_(True)

    grad_out = torch.randn(B, L, 3, H, D, dtype=dtype, device="cuda")

    out_f = fused_qknorm_rope(qkv_f, qs_f, ks_f, pe)
    out_f.backward(grad_out)

    out_e = eager_reference(qkv_e, qs_e, ks_e, pe)
    out_e.backward(grad_out)

    torch.testing.assert_close(qkv_f.grad.float(), qkv_e.grad.float(), **tol)
    torch.testing.assert_close(qs_f.grad.float(), qs_e.grad.float(), **scale_tol)
    torch.testing.assert_close(ks_f.grad.float(), ks_e.grad.float(), **scale_tol)


def test_backward_frozen_scales():
    """LoRA training case: scales frozen, only qkv needs grad."""
    from musubi_tuner.modules.fused_qknorm_rope import fused_qknorm_rope

    B, L, H, D = 2, 9, 2, 64
    qkv, qs, ks, pe = make_inputs(B, L, H, D, torch.float32, seed=2)
    qkv.requires_grad_(True)
    out = fused_qknorm_rope(qkv, qs, ks, pe)
    out.sum().backward()
    assert qkv.grad is not None
    assert qs.grad is None and ks.grad is None
