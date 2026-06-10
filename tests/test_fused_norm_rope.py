"""Tests for the fused QKNorm + RoPE kernel.

Compares the Triton fused kernel against a pure-PyTorch reference
implementation, and validates helper utilities.
"""

import pytest
import torch
from musubi_tuner.kernels.fused_norm_rope import (
    fused_norm_rope,
    fused_packed_qk_norm_rope,
    reference_norm_rope,
    reference_packed_qk_norm_rope,
    reference_norm_rope_fp32,
    reference_packed_qk_norm_rope_fp32,
    extract_cos_sin,
    HAS_TRITON,
)


# ---------------------------------------------------------------------------
# Test 1: reference vs triton kernel (skip if no triton or no CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires triton + CUDA",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_norm_rope_matches_reference(dtype):
    B, L, H, D = 2, 64, 16, 128
    torch.manual_seed(42)
    x = torch.randn(B, L, H, D, dtype=dtype, device="cuda")
    weight = torch.randn(D, device="cuda")
    # simple cos/sin for testing
    D2 = D // 2
    cos = torch.ones(B, L, D2, device="cuda")
    sin = torch.zeros(B, L, D2, device="cuda")

    # The kernel stays in float32, so compare against the float32 oracle.
    ref = reference_norm_rope_fp32(x, weight, cos, sin)
    out = fused_norm_rope(x, weight, cos, sin)

    assert out.shape == ref.shape
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 2: extract_cos_sin roundtrip (CPU, no triton needed)
# ---------------------------------------------------------------------------


def test_extract_cos_sin():
    B, L, D2 = 2, 64, 32
    # Build a freqs_cis with known cos/sin values
    cos_expected = torch.rand(B, L, D2)
    sin_expected = torch.rand(B, L, D2)
    # freqs_cis shape: [B, 1, L, D2, 2, 2]; the '1' is the num_heads broadcast dim.
    # Assign via the squeezed view to avoid the singleton-dimension mismatch.
    fc = torch.zeros(B, 1, L, D2, 2, 2)
    fc[:, 0, :, :, 0, 0] = cos_expected
    fc[:, 0, :, :, 0, 1] = -sin_expected
    fc[:, 0, :, :, 1, 0] = sin_expected
    fc[:, 0, :, :, 1, 1] = cos_expected
    cos_out, sin_out = extract_cos_sin(fc)
    torch.testing.assert_close(cos_out, cos_expected)
    torch.testing.assert_close(sin_out, sin_expected)


# ---------------------------------------------------------------------------
# Test 3: reference is correct vs manual computation (CPU)
# ---------------------------------------------------------------------------


def test_reference_correctness():
    B, L, H, D = 1, 4, 2, 8
    torch.manual_seed(0)
    x = torch.randn(B, L, H, D)
    weight = torch.ones(D)
    D2 = D // 2
    # identity RoPE (cos=1, sin=0) -> should just return normed x
    cos = torch.ones(B, L, D2)
    sin = torch.zeros(B, L, D2)
    out = reference_norm_rope(x, weight, cos, sin)
    # verify it's normalized per head
    norms = out.pow(2).mean(-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 4: non-trivial RoPE rotations (CUDA, requires triton)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires triton + CUDA",
)
@pytest.mark.parametrize(
    "B, L, H, D",
    [
        (2, 256, 24, 128),
        (1, 64, 8, 64),
        (2, 128, 16, 128),
    ],
)
def test_fused_norm_rope_nontrivial_rope(B, L, H, D):
    """Kernel matches reference for random (non-identity) cos/sin values."""
    torch.manual_seed(99)
    D2 = D // 2
    x = torch.randn(B, L, H, D, dtype=torch.bfloat16, device="cuda")
    weight = torch.rand(D, device="cuda") + 0.5  # avoid zeros
    angles = torch.rand(B, L, D2, device="cuda") * 2 * torch.pi
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # The kernel stays in float32, so compare against the float32 oracle.
    ref = reference_norm_rope_fp32(x, weight, cos, sin)
    out = fused_norm_rope(x, weight, cos, sin)

    assert out.shape == ref.shape
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# Test 5: fused_packed_qk_norm_rope reference correctness (CPU)
# ---------------------------------------------------------------------------


def test_fused_packed_qk_norm_rope_reference_correctness():
    """reference_packed_qk_norm_rope matches applying reference_norm_rope per Q and K."""
    B, L, H, D = 1, 4, 2, 8
    torch.manual_seed(7)
    D2 = D // 2
    qkv = torch.randn(B, L, 3, H, D)
    q_weight = torch.rand(D) + 0.5
    k_weight = torch.rand(D) + 0.5
    angles = torch.rand(B, L, D2) * 2 * torch.pi
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    out = reference_packed_qk_norm_rope(qkv, q_weight, k_weight, cos, sin)

    q_ref = reference_norm_rope(qkv[:, :, 0], q_weight, cos, sin)
    k_ref = reference_norm_rope(qkv[:, :, 1], k_weight, cos, sin)
    v_ref = qkv[:, :, 2].float()

    assert out.shape == (B, L, 3, H, D)
    torch.testing.assert_close(out[:, :, 0].float(), q_ref.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, :, 1].float(), k_ref.float(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[:, :, 2].float(), v_ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Test 6: fused_packed_qk_norm_rope Triton kernel matches reference (CUDA)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="requires triton + CUDA",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "B, L, H, D",
    [
        (2, 64, 16, 128),
        (1, 32, 8, 64),
        (2, 128, 24, 128),
    ],
)
def test_fused_packed_qk_norm_rope_matches_reference(dtype, B, L, H, D):
    """Packed QKV kernel matches reference for each Q/K/V slice."""
    torch.manual_seed(42)
    D2 = D // 2
    qkv = torch.randn(B, L, 3, H, D, dtype=dtype, device="cuda")
    q_weight = (torch.rand(D, device="cuda") + 0.5).to(dtype)
    k_weight = (torch.rand(D, device="cuda") + 0.5).to(dtype)
    angles = torch.rand(B, L, D2, device="cuda") * 2 * torch.pi
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # The kernel stays in float32, so compare against the float32 oracle.
    ref = reference_packed_qk_norm_rope_fp32(qkv, q_weight, k_weight, cos, sin)
    out = fused_packed_qk_norm_rope(qkv, q_weight, k_weight, cos, sin)

    assert out.shape == (B, L, 3, H, D)
    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
