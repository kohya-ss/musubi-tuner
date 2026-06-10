"""Compare original (separate Q/K/V) norm+rope path against fused packed path.

The original Flux2 path:
  1. rearrange qkv to [K, B, H, L, D]
  2. QKNorm: RMSNorm(q), RMSNorm(k), then cast to v's dtype (bf16)
  3. apply_rope in float32, cast back to input dtype
  4. transpose to [B, L, H, D]

The fused packed path:
  1. reshape qkv to [B, L, 3, H, D]
  2. extract_cos_sin from rotation-matrix pe
  3. fused_packed_qk_norm_rope: RMSNorm + RoPE in float32, cast back

These tests verify:
  - The PyTorch reference reproduces main's original numerics exactly (it is the
    production fallback when Triton is unavailable), in both fp32 and bf16.
  - The fused Triton kernel matches its own float32 oracle
    (reference_packed_qk_norm_rope_fp32), since the kernel stays in float32.
  - The fused path doesn't introduce training-breaking gradient divergence vs.
    the original (loose tolerance, bf16 + cross-layout accumulation only).

Run with:
  uv run --extra cu128 pytest tests/test_packed_vs_original.py -v
"""

import pytest
import torch
from torch import Tensor

from musubi_tuner.flux_2.flux2_models import apply_rope, rope
from musubi_tuner.kernels.fused_norm_rope import (
    HAS_TRITON,
    fused_packed_qk_norm_rope,
    reference_packed_qk_norm_rope,
    reference_packed_qk_norm_rope_fp32,
    extract_cos_sin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pe(B: int, L: int, D: int, device: torch.device, theta: int = 256):
    """Build rotation-matrix pe identical to Flux2's rope() output.

    Returns shape [B, 1, L, D//2, 2, 2].
    """
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
    pe = rope(pos, D, theta)  # [B, L, D//2, 2, 2]
    return pe.unsqueeze(1)  # [B, 1, L, D//2, 2, 2]


def _original_path(
    qkv_flat: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    pe: Tensor,
    num_heads: int,
) -> Tensor:
    """Original Flux2 path: separate Q/K/V → QKNorm → apply_rope → transpose.

    Args:
        qkv_flat: [B, L, 3*H*D] — contiguous projection output.
        q_scale, k_scale: [D] — RMSNorm scale parameters.
        pe: [B, 1, L, D//2, 2, 2] — rotation matrix.
        num_heads: number of attention heads.

    Returns:
        Packed [B, L, 3, H, D] with Q normed+rotated, K normed+rotated, V unchanged.
    """
    B, L, _ = qkv_flat.shape
    D = qkv_flat.shape[-1] // (3 * num_heads)
    # rearrange "B L (K H D) -> K B H L D"
    qkv_5d = qkv_flat.reshape(B, L, 3, num_heads, D).permute(2, 0, 3, 1, 4)  # [3, B, H, L, D]
    q, k, v = qkv_5d.unbind(0)

    # QKNorm exactly as in main's flux2_models.py RMSNorm.forward:
    #   (x.float() * rrms).to(x_dtype) * scale  — round to input dtype BEFORE scale
    def _rmsnorm(x: Tensor, scale: Tensor) -> Tensor:
        x_dtype = x.dtype
        xf = x.float()
        rrms = torch.rsqrt(torch.mean(xf**2, dim=-1, keepdim=True) + 1e-6)
        return (xf * rrms).to(x_dtype) * scale

    # QKNorm.forward then casts q,k to v's dtype.
    q = _rmsnorm(q, q_scale).to(v.dtype)
    k = _rmsnorm(k, k_scale).to(v.dtype)

    # apply_rope in float32, cast back to input dtype
    q, k = apply_rope(q, k, pe)

    # Transpose to [B, L, H, D] and stack
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    return torch.stack([q, k, v], dim=2)  # [B, L, 3, H, D]


def _fused_path(
    qkv_flat: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    pe: Tensor,
    num_heads: int,
) -> Tensor:
    """New fused packed path: reshape → extract_cos_sin → fused_packed_qk_norm_rope.

    Same inputs/outputs as _original_path.
    """
    B, L = qkv_flat.shape[:2]
    qkv = qkv_flat.contiguous().reshape(B, L, 3, num_heads, -1)
    cos, sin = extract_cos_sin(pe)
    return fused_packed_qk_norm_rope(qkv, q_scale, k_scale, cos, sin)


def _reference_path(
    qkv_flat: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    pe: Tensor,
    num_heads: int,
) -> Tensor:
    """Reference (PyTorch) packed path — always available, no Triton needed.

    Reproduces main's original numerics exactly (bf16 intermediate casts), so
    it is the oracle for the production fallback.
    """
    B, L = qkv_flat.shape[:2]
    qkv = qkv_flat.contiguous().reshape(B, L, 3, num_heads, -1)
    cos, sin = extract_cos_sin(pe)
    return reference_packed_qk_norm_rope(qkv, q_scale, k_scale, cos, sin)


def _reference_fp32_path(
    qkv_flat: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    pe: Tensor,
    num_heads: int,
) -> Tensor:
    """Float32 reference packed path — the oracle for the fused Triton kernel.

    The fused kernel keeps float32 throughout (no intermediate bf16 casts), so
    it is verified against this, not against the original-matching reference.
    """
    B, L = qkv_flat.shape[:2]
    qkv = qkv_flat.contiguous().reshape(B, L, 3, num_heads, -1)
    cos, sin = extract_cos_sin(pe)
    return reference_packed_qk_norm_rope_fp32(qkv, q_scale, k_scale, cos, sin)


def _make_inputs(B, L, H, D, dtype, device, seed=42):
    """Create shared inputs for both paths."""
    torch.manual_seed(seed)
    qkv_flat = torch.randn(B, L, 3 * H * D, dtype=dtype, device=device)
    q_scale = (torch.rand(D, device=device) + 0.5).to(torch.float32)
    k_scale = (torch.rand(D, device=device) + 0.5).to(torch.float32)
    pe = _make_pe(B, L, D, device)
    return qkv_flat, q_scale, k_scale, pe


def _clone_with_grad(*tensors):
    return [t.detach().clone().requires_grad_(True) for t in tensors]


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B, L, H, D",
    [
        (1, 16, 8, 64),
        (2, 32, 16, 128),
        (1, 64, 24, 128),
    ],
)
class TestForwardComparison:
    """Forward output comparison between original and fused paths."""

    def test_reference_matches_original_fp32(self, B, L, H, D):
        """In float32 both paths should be nearly identical (no bf16 intermediate cast)."""
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.float32, "cpu")

        out_orig = _original_path(qkv, q_s, k_s, pe, H)
        out_ref = _reference_path(qkv, q_s, k_s, pe, H)

        torch.testing.assert_close(
            out_ref.float(), out_orig.float(),
            atol=1e-5, rtol=1e-5,
            msg="In float32, reference and original should match exactly",
        )

    def test_reference_vs_original_bf16(self, B, L, H, D):
        """In bf16 the reference must match the original exactly: both apply the
        same intermediate bf16 cast in RMSNorm ((x*rrms).to(dtype)*scale) and the
        same QKNorm .to(v) cast, so they are byte-for-byte identical."""
        device = "cpu"
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.bfloat16, device)

        out_orig = _original_path(qkv, q_s, k_s, pe, H)
        out_ref = _reference_path(qkv, q_s, k_s, pe, H)

        torch.testing.assert_close(
            out_ref.float(), out_orig.float(),
            atol=1e-6, rtol=1e-6,
            msg="bf16 reference must reproduce original exactly",
        )

    @pytest.mark.skipif(
        not HAS_TRITON or not torch.cuda.is_available(),
        reason="requires triton + CUDA",
    )
    def test_fused_vs_original_cuda(self, B, L, H, D):
        """Fused Triton kernel vs original path on CUDA."""
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.bfloat16, "cuda")

        out_orig = _original_path(qkv, q_s, k_s, pe, H)
        out_fused = _fused_path(qkv, q_s, k_s, pe, H)

        torch.testing.assert_close(
            out_fused.float(), out_orig.float(),
            atol=5e-2, rtol=1e-2,
            msg="Fused CUDA forward vs original exceeds tolerance",
        )

    @pytest.mark.skipif(
        not HAS_TRITON or not torch.cuda.is_available(),
        reason="requires triton + CUDA",
    )
    def test_fused_matches_reference_cuda(self, B, L, H, D):
        """Fused Triton kernel must match its own float32 reference.

        The kernel stays in float32 throughout, so its oracle is the fp32
        reference (not the original-matching reference, which adds bf16 casts).
        """
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.bfloat16, "cuda")

        out_ref = _reference_fp32_path(qkv, q_s, k_s, pe, H)
        out_fused = _fused_path(qkv, q_s, k_s, pe, H)

        torch.testing.assert_close(
            out_fused.float(), out_ref.float(),
            atol=1e-2, rtol=1e-2,
            msg="Fused kernel should match its own float32 reference",
        )


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "B, L, H, D",
    [
        (1, 16, 8, 64),
        (2, 32, 16, 128),
        (1, 64, 24, 128),
    ],
)
class TestBackwardComparison:
    """Backward gradient comparison between original and reference/fused paths."""

    def test_gradients_reference_vs_original_fp32(self, B, L, H, D):
        """In float32, gradients should match tightly."""
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.float32, "cpu")

        # Original path
        qkv_o, q_s_o, k_s_o = _clone_with_grad(qkv, q_s, k_s)
        out_o = _original_path(qkv_o, q_s_o, k_s_o, pe, H)
        out_o.sum().backward()

        # Reference path
        qkv_r, q_s_r, k_s_r = _clone_with_grad(qkv, q_s, k_s)
        out_r = _reference_path(qkv_r, q_s_r, k_s_r, pe, H)
        out_r.sum().backward()

        torch.testing.assert_close(
            qkv_r.grad.float(), qkv_o.grad.float(),
            atol=1e-5, rtol=1e-5,
            msg="qkv gradient mismatch (fp32, reference vs original)",
        )
        torch.testing.assert_close(
            q_s_r.grad.float(), q_s_o.grad.float(),
            atol=1e-5, rtol=1e-5,
            msg="q_scale gradient mismatch (fp32)",
        )
        torch.testing.assert_close(
            k_s_r.grad.float(), k_s_o.grad.float(),
            atol=1e-5, rtol=1e-5,
            msg="k_scale gradient mismatch (fp32)",
        )

    def test_gradients_reference_vs_original_bf16(self, B, L, H, D):
        """In bf16 the reference reproduces the original op-for-op (same
        intermediate bf16 cast in RMSNorm and same QKNorm .to(v)), so autograd
        builds the identical graph and all gradients match exactly.
        """
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.bfloat16, "cpu")

        # Original path
        qkv_o, q_s_o, k_s_o = _clone_with_grad(qkv, q_s, k_s)
        out_o = _original_path(qkv_o, q_s_o, k_s_o, pe, H)
        out_o.sum().backward()

        # Reference path
        qkv_r, q_s_r, k_s_r = _clone_with_grad(qkv, q_s, k_s)
        out_r = _reference_path(qkv_r, q_s_r, k_s_r, pe, H)
        out_r.sum().backward()

        torch.testing.assert_close(
            qkv_r.grad.float(), qkv_o.grad.float(),
            atol=1e-5, rtol=1e-5,
            msg="qkv gradient mismatch (bf16, reference vs original)",
        )
        torch.testing.assert_close(
            q_s_r.grad.float(), q_s_o.grad.float(),
            atol=1e-5, rtol=1e-5,
            msg="q_scale gradient mismatch (bf16, reference vs original)",
        )
        torch.testing.assert_close(
            k_s_r.grad.float(), k_s_o.grad.float(),
            atol=1e-5, rtol=1e-5,
            msg="k_scale gradient mismatch (bf16, reference vs original)",
        )

    @pytest.mark.skipif(
        not HAS_TRITON or not torch.cuda.is_available(),
        reason="requires triton + CUDA",
    )
    def test_gradients_fused_vs_original_cuda(self, B, L, H, D):
        """Fused Triton kernel gradients vs original path."""
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.bfloat16, "cuda")

        # Original path
        qkv_o, q_s_o, k_s_o = _clone_with_grad(qkv, q_s, k_s)
        out_o = _original_path(qkv_o, q_s_o, k_s_o, pe, H)
        out_o.sum().backward()

        # Fused path
        qkv_f, q_s_f, k_s_f = _clone_with_grad(qkv, q_s, k_s)
        out_f = _fused_path(qkv_f, q_s_f, k_s_f, pe, H)
        out_f.sum().backward()

        torch.testing.assert_close(
            qkv_f.grad.float(), qkv_o.grad.float(),
            atol=5e-2, rtol=1e-2,
            msg="qkv gradient mismatch (fused vs original, CUDA)",
        )
        # Weight scale gradients are reductions over all (B, L, H) positions.
        # The original path operates on a [B, H, L, D] layout while the fused
        # path uses [B, L, H, D], so float32 accumulation order differs and
        # produces noise proportional to sqrt(B*L*H). Fused matches its own
        # reference exactly (same layout); this looser tolerance only covers the
        # cross-layout accumulation difference.
        torch.testing.assert_close(
            q_s_f.grad.float(), q_s_o.grad.float(),
            atol=0.3, rtol=1e-2,
            msg="q_scale gradient mismatch (fused vs original, CUDA)",
        )
        torch.testing.assert_close(
            k_s_f.grad.float(), k_s_o.grad.float(),
            atol=0.3, rtol=1e-2,
            msg="k_scale gradient mismatch (fused vs original, CUDA)",
        )

    @pytest.mark.skipif(
        not HAS_TRITON or not torch.cuda.is_available(),
        reason="requires triton + CUDA",
    )
    def test_gradients_fused_matches_reference_cuda(self, B, L, H, D):
        """Fused kernel gradients must match its own float32 reference."""
        qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, torch.bfloat16, "cuda")

        # Float32 reference path (the kernel's oracle)
        qkv_r, q_s_r, k_s_r = _clone_with_grad(qkv, q_s, k_s)
        out_r = _reference_fp32_path(qkv_r, q_s_r, k_s_r, pe, H)
        out_r.sum().backward()

        # Fused path
        qkv_f, q_s_f, k_s_f = _clone_with_grad(qkv, q_s, k_s)
        out_f = _fused_path(qkv_f, q_s_f, k_s_f, pe, H)
        out_f.sum().backward()

        torch.testing.assert_close(
            qkv_f.grad.float(), qkv_r.grad.float(),
            atol=5e-2, rtol=1e-2,
            msg="qkv gradient: fused kernel vs reference",
        )
        torch.testing.assert_close(
            q_s_f.grad.float(), q_s_r.grad.float(),
            atol=5e-2, rtol=1e-2,
            msg="q_scale gradient: fused kernel vs reference",
        )
        torch.testing.assert_close(
            k_s_f.grad.float(), k_s_r.grad.float(),
            atol=5e-2, rtol=1e-2,
            msg="k_scale gradient: fused kernel vs reference",
        )


# ---------------------------------------------------------------------------
# Diagnostic: measure and report the actual gap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_gap_diagnostic(dtype):
    """Measure the actual forward gap between paths — prints stats for debugging."""
    B, L, H, D = 2, 64, 16, 128
    device = "cpu"
    qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, dtype, device)

    out_orig = _original_path(qkv, q_s, k_s, pe, H)
    out_ref = _reference_path(qkv, q_s, k_s, pe, H)

    diff = (out_ref.float() - out_orig.float()).abs()
    rel_diff = diff / (out_orig.float().abs() + 1e-8)

    print(f"\n{'='*60}")
    print(f"Forward gap diagnostic — dtype={dtype}")
    print(f"  Max abs diff:  {diff.max().item():.6e}")
    print(f"  Mean abs diff: {diff.mean().item():.6e}")
    print(f"  Max rel diff:  {rel_diff.max().item():.6e}")
    print(f"  Mean rel diff: {rel_diff.mean().item():.6e}")
    print(f"  Q max diff:    {diff[:,:,0].max().item():.6e}")
    print(f"  K max diff:    {diff[:,:,1].max().item():.6e}")
    print(f"  V max diff:    {diff[:,:,2].max().item():.6e}")
    print(f"{'='*60}")

    # Reference now reproduces the original op-for-op, so both fp32 and bf16
    # should be near-zero (the reference is no longer fp32-simplified).
    if dtype == torch.float32:
        assert diff.max() < 1e-4, f"fp32 gap too large: {diff.max().item()}"
    else:
        assert diff.max() < 1e-4, f"bf16 gap too large: {diff.max().item()}"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backward_gap_diagnostic(dtype):
    """Measure actual backward gradient gap — prints stats for debugging."""
    B, L, H, D = 2, 32, 16, 128
    device = "cpu"
    qkv, q_s, k_s, pe = _make_inputs(B, L, H, D, dtype, device)

    # Original
    qkv_o, q_s_o, k_s_o = _clone_with_grad(qkv, q_s, k_s)
    out_o = _original_path(qkv_o, q_s_o, k_s_o, pe, H)
    out_o.sum().backward()

    # Reference
    qkv_r, q_s_r, k_s_r = _clone_with_grad(qkv, q_s, k_s)
    out_r = _reference_path(qkv_r, q_s_r, k_s_r, pe, H)
    out_r.sum().backward()

    for name, g_ref, g_orig in [
        ("qkv", qkv_r.grad, qkv_o.grad),
        ("q_scale", q_s_r.grad, q_s_o.grad),
        ("k_scale", k_s_r.grad, k_s_o.grad),
    ]:
        diff = (g_ref.float() - g_orig.float()).abs()
        rel_diff = diff / (g_orig.float().abs() + 1e-8)
        print(f"\n--- Backward gap: {name} (dtype={dtype}) ---")
        print(f"  Max abs diff:  {diff.max().item():.6e}")
        print(f"  Mean abs diff: {diff.mean().item():.6e}")
        print(f"  Max rel diff:  {rel_diff.max().item():.6e}")
        print(f"  Mean rel diff: {rel_diff.mean().item():.6e}")

        if dtype == torch.float32:
            assert diff.max() < 1e-4, f"fp32 {name} gradient gap too large: {diff.max().item()}"
        else:
            # Reference reproduces the original op-for-op → identical autograd graph.
            assert diff.max() < 1e-3, f"bf16 {name} gradient gap too large: {diff.max().item()}"
