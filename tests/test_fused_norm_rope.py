"""Tests for the fused QKNorm + RoPE Triton kernel.

Compares the Triton fused kernel against a pure-PyTorch reference
implementation using realistic Flux2-like tensor shapes.
"""

import pytest
import torch

# Skip the entire module if Triton is not available (CPU-only environments)
triton = pytest.importorskip("triton", reason="triton not installed")

from musubi_tuner.kernels.fused_norm_rope import (
    extract_cos_sin,
    fused_norm_rope,
    qk_norm_rope_reference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_freqs_cis(B: int, L: int, D_half: int, device: torch.device) -> torch.Tensor:
    """Build a synthetic freqs_cis tensor with shape [B, 1, L, D_half, 2, 2]."""
    # Use random angles so the rotation matrices are non-trivial
    angles = torch.rand(B, L, D_half, device=device, dtype=torch.float32) * 2 * torch.pi
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    # Build rotation matrices [B, L, D_half, 2, 2]
    rot = torch.stack(
        [
            torch.stack([cos_a, -sin_a], dim=-1),   # row 0: [cos, -sin]
            torch.stack([sin_a,  cos_a], dim=-1),   # row 1: [sin,  cos]
        ],
        dim=-2,
    )
    # Add the head dimension (=1 in the actual model)
    return rot.unsqueeze(1)  # [B, 1, L, D_half, 2, 2]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ]
)
def device(request):
    return torch.device(request.param)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B, L, H, D",
    [
        (2, 256, 24, 128),  # Flux2 Klein-4B-ish
        (1, 64,  8,  64),   # smaller sanity check
        (2, 128, 16, 128),  # another realistic shape
    ],
)
def test_fused_norm_rope_matches_reference(B, L, H, D, device):
    """Kernel output must match the PyTorch reference within bfloat16 tolerance."""
    torch.manual_seed(42)
    D_half = D // 2

    # Input tensor in bfloat16 on the target device
    x = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=device)

    # RMSNorm weight (learned scale), one per head-dimension element
    weight = torch.rand(D, dtype=torch.bfloat16, device=device) + 0.5  # avoid zeros

    # Build freqs_cis and extract cos/sin
    freqs_cis = _make_freqs_cis(B, L, D_half, device=device)
    cos, sin = extract_cos_sin(freqs_cis)

    # Reference (pure PyTorch)
    ref = qk_norm_rope_reference(x, weight, cos, sin, eps=1e-6)

    # Fused Triton kernel
    out = fused_norm_rope(x, weight, cos, sin, eps=1e-6)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    assert out.dtype == ref.dtype, f"Dtype mismatch: {out.dtype} vs {ref.dtype}"

    # bfloat16 has limited precision — use a generous but reasonable tolerance
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_norm_rope_dtypes(dtype, device):
    """Kernel handles bf16, fp16, and fp32 inputs."""
    torch.manual_seed(0)
    B, L, H, D = 1, 32, 4, 64
    D_half = D // 2

    x = torch.randn(B, L, H, D, dtype=dtype, device=device)
    weight = torch.ones(D, dtype=dtype, device=device)
    freqs_cis = _make_freqs_cis(B, L, D_half, device=device)
    cos, sin = extract_cos_sin(freqs_cis)

    ref = qk_norm_rope_reference(x, weight, cos, sin)
    out = fused_norm_rope(x, weight, cos, sin)

    assert out.dtype == dtype
    torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)


def test_extract_cos_sin_shape(device):
    """extract_cos_sin returns tensors of the expected shape."""
    B, L, D_half = 2, 64, 32
    freqs_cis = _make_freqs_cis(B, L, D_half, device=device)
    cos, sin = extract_cos_sin(freqs_cis)
    assert cos.shape == (B, L, D_half)
    assert sin.shape == (B, L, D_half)


def test_extract_cos_sin_values(device):
    """cos and sin values extracted from freqs_cis are correct."""
    B, L, D_half = 1, 4, 8
    angles = torch.rand(B, L, D_half, device=device, dtype=torch.float32) * 2 * torch.pi
    cos_expected = torch.cos(angles)
    sin_expected = torch.sin(angles)

    rot = torch.stack(
        [
            torch.stack([cos_expected, -sin_expected], dim=-1),
            torch.stack([sin_expected,  cos_expected], dim=-1),
        ],
        dim=-2,
    ).unsqueeze(1)  # [B, 1, L, D_half, 2, 2]

    cos_got, sin_got = extract_cos_sin(rot)
    torch.testing.assert_close(cos_got, cos_expected)
    torch.testing.assert_close(sin_got, sin_expected)


def test_identity_weight_no_rope(device):
    """With unit weight and zero sin (no rotation), output == RMSNorm(x)."""
    torch.manual_seed(1)
    B, L, H, D = 1, 16, 2, 32
    D_half = D // 2

    x = torch.randn(B, L, H, D, dtype=torch.float32, device=device)
    weight = torch.ones(D, device=device)

    # Build freqs_cis with zero rotation (identity matrix)
    cos_const = torch.ones(B, L, D_half, device=device)
    sin_const = torch.zeros(B, L, D_half, device=device)

    ref = qk_norm_rope_reference(x, weight, cos_const, sin_const)
    out = fused_norm_rope(x, weight, cos_const, sin_const)
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


def test_output_dtype_preserved(device):
    """Output dtype must match input dtype."""
    for dtype in [torch.bfloat16, torch.float16, torch.float32]:
        B, L, H, D = 1, 8, 2, 16
        x = torch.randn(B, L, H, D, dtype=dtype, device=device)
        weight = torch.ones(D, dtype=dtype, device=device)
        cos = torch.ones(B, L, D // 2, device=device)
        sin = torch.zeros(B, L, D // 2, device=device)
        out = fused_norm_rope(x, weight, cos, sin)
        assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"
