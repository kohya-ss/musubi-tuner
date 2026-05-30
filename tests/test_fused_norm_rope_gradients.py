"""Gradient diagnostics for fused QKNorm + RoPE kernel.

Confirms that:
  1. reference_packed_qk_norm_rope produces correct non-zero gradients (baseline).
  2. fused_packed_qk_norm_rope (with autograd.Function) matches reference gradients.
  3. Gradient flow reaches both qkv (→ linear weights) and norm scale weights.

Run with:
  uv run --extra cu128 pytest tests/test_fused_norm_rope_gradients.py -v
"""

import pytest
import torch
from musubi_tuner.kernels.fused_norm_rope import (
    HAS_TRITON,
    fused_packed_qk_norm_rope,
    reference_packed_qk_norm_rope,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(B, L, H, D, dtype, device, seed=42):
    torch.manual_seed(seed)
    D2 = D // 2
    qkv = torch.randn(B, L, 3, H, D, dtype=dtype, device=device)
    q_weight = (torch.rand(D, device=device) + 0.5).to(dtype)
    k_weight = (torch.rand(D, device=device) + 0.5).to(dtype)
    angles = torch.rand(B, L, D2, device=device) * 2 * torch.pi
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return qkv, q_weight, k_weight, cos, sin


def _clone_with_grad(*tensors):
    """Return clones with requires_grad=True for each tensor."""
    return [t.detach().clone().requires_grad_(True) for t in tensors]


# ---------------------------------------------------------------------------
# Test 1: reference produces non-zero gradients (sanity check)
# ---------------------------------------------------------------------------


def test_reference_gradients_are_nonzero():
    """Baseline: reference_packed_qk_norm_rope must produce valid grads."""
    B, L, H, D = 1, 16, 8, 64
    qkv, q_w, k_w, cos, sin = _make_inputs(B, L, H, D, torch.float32, "cpu")
    qkv_r, q_w_r, k_w_r = _clone_with_grad(qkv, q_w, k_w)

    out = reference_packed_qk_norm_rope(qkv_r, q_w_r, k_w_r, cos, sin)
    out.sum().backward()

    assert qkv_r.grad is not None, "qkv grad should exist"
    assert q_w_r.grad is not None, "q_weight grad should exist"
    assert k_w_r.grad is not None, "k_weight grad should exist"

    assert qkv_r.grad.norm() > 0, "qkv grad should be non-zero"
    assert q_w_r.grad.norm() > 0, "q_weight grad should be non-zero"
    assert k_w_r.grad.norm() > 0, "k_weight grad should be non-zero"


# ---------------------------------------------------------------------------
# Test 2: diagnose whether fused produces gradients at all
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires triton + CUDA",
)
def test_fused_gradients_exist():
    """fused_packed_qk_norm_rope must produce non-None, non-zero gradients."""
    B, L, H, D = 1, 16, 8, 64
    qkv, q_w, k_w, cos, sin = _make_inputs(B, L, H, D, torch.bfloat16, "cuda")
    qkv_f, q_w_f, k_w_f = _clone_with_grad(qkv.cuda(), q_w.cuda(), k_w.cuda())

    out = fused_packed_qk_norm_rope(qkv_f, q_w_f, k_w_f, cos, sin)
    out.sum().backward()

    assert qkv_f.grad is not None, (
        "qkv.grad is None — Triton kernel is not wrapped in autograd.Function; "
        "gradients cannot flow to linear1.weight / LoRA adapters"
    )
    assert q_w_f.grad is not None, (
        "q_weight.grad is None — query_norm.scale will not be updated"
    )
    assert k_w_f.grad is not None, (
        "k_weight.grad is None — key_norm.scale will not be updated"
    )

    assert qkv_f.grad.norm() > 0, "qkv grad is zero"
    assert q_w_f.grad.norm() > 0, "q_weight grad is zero"
    assert k_w_f.grad.norm() > 0, "k_weight grad is zero"


# ---------------------------------------------------------------------------
# Test 3: fused gradients match reference gradients
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires triton + CUDA",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B, L, H, D",
    [
        (1, 16, 8, 64),
        (2, 32, 16, 128),
        (1, 64, 24, 128),
    ],
)
def test_fused_gradients_match_reference(dtype, B, L, H, D):
    """Fused kernel gradients must numerically match the reference PyTorch path."""
    qkv, q_w, k_w, cos, sin = _make_inputs(B, L, H, D, dtype, "cuda")
    cos_cuda = cos.cuda()
    sin_cuda = sin.cuda()

    # Reference
    qkv_r, q_w_r, k_w_r = _clone_with_grad(qkv, q_w, k_w)
    out_r = reference_packed_qk_norm_rope(qkv_r, q_w_r, k_w_r, cos_cuda, sin_cuda)
    out_r.sum().backward()

    # Fused
    qkv_f, q_w_f, k_w_f = _clone_with_grad(qkv, q_w, k_w)
    out_f = fused_packed_qk_norm_rope(qkv_f, q_w_f, k_w_f, cos_cuda, sin_cuda)
    out_f.sum().backward()

    # Compare gradients (float32 comparison, loose tolerance for bfloat16)
    atol, rtol = (5e-2, 1e-2) if dtype == torch.bfloat16 else (1e-2, 1e-2)

    torch.testing.assert_close(
        qkv_f.grad.float(), qkv_r.grad.float(), atol=atol, rtol=rtol,
        msg="qkv gradient mismatch between fused and reference"
    )
    torch.testing.assert_close(
        q_w_f.grad.float(), q_w_r.grad.float(), atol=atol, rtol=rtol,
        msg="q_weight gradient mismatch"
    )
    torch.testing.assert_close(
        k_w_f.grad.float(), k_w_r.grad.float(), atol=atol, rtol=rtol,
        msg="k_weight gradient mismatch"
    )


# ---------------------------------------------------------------------------
# Test 4: V gradient passthrough (V is unchanged, grad must be identity)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires triton + CUDA",
)
def test_v_gradient_passthrough():
    """Gradient w.r.t. V slice must pass through unchanged (no norm/rope applied to V)."""
    B, L, H, D = 1, 16, 8, 64
    qkv, q_w, k_w, cos, sin = _make_inputs(B, L, H, D, torch.bfloat16, "cuda")
    qkv_f, q_w_f, k_w_f = _clone_with_grad(qkv, q_w, k_w)

    out = fused_packed_qk_norm_rope(qkv_f, q_w_f, k_w_f, cos, sin)

    # Backprop a gradient that only has signal in the V slice
    grad = torch.zeros_like(out)
    grad[:, :, 2] = 1.0  # only V
    out.backward(grad)

    # Q and K slices of qkv.grad should be zero (no gradient from V output)
    assert qkv_f.grad[:, :, 0].norm() == 0, "Q slice got unexpected grad from V output"
    assert qkv_f.grad[:, :, 1].norm() == 0, "K slice got unexpected grad from V output"
    # V slice should get the identity gradient (cast to bf16)
    torch.testing.assert_close(
        qkv_f.grad[:, :, 2].float(),
        grad[:, :, 2].float(),
        atol=1e-3, rtol=0,
        msg="V gradient should be identity passthrough"
    )


# ---------------------------------------------------------------------------
# Test 5: CPU / no-Triton fallback also has correct gradients
# ---------------------------------------------------------------------------


def test_reference_fallback_gradients_cpu():
    """reference_packed_qk_norm_rope on CPU must have valid grads (fallback path)."""
    B, L, H, D = 1, 8, 4, 32
    qkv, q_w, k_w, cos, sin = _make_inputs(B, L, H, D, torch.float32, "cpu")
    qkv_r, q_w_r, k_w_r = _clone_with_grad(qkv, q_w, k_w)

    # This exercises the PyTorch reference path (same as fused fallback when no Triton)
    out = reference_packed_qk_norm_rope(qkv_r, q_w_r, k_w_r, cos, sin)
    out.sum().backward()

    assert qkv_r.grad is not None and qkv_r.grad.norm() > 0
    assert q_w_r.grad is not None and q_w_r.grad.norm() > 0
    assert k_w_r.grad is not None and k_w_r.grad.norm() > 0
