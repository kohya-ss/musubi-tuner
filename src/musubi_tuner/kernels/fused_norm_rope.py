"""Fused RMSNorm + RoPE kernel for Q and K in Flux2-style attention."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    if os.environ.get("MUSUBI_DISABLE_TRITON", "0") == "1":
        HAS_TRITON = False
        logger.info("Triton disabled (MUSUBI_DISABLE_TRITON=1) — using PyTorch reference")
    else:
        HAS_TRITON = True
        logger.info("Triton fused kernel enabled")
except ImportError:
    HAS_TRITON = False
    logger.info("Triton not installed — using PyTorch reference")

import torch


# ---------------------------------------------------------------------------
# Reference implementation (always available, used as fallback)
# ---------------------------------------------------------------------------


def reference_norm_rope(
    x: torch.Tensor,       # [B, L, H, D]
    weight: torch.Tensor,  # [D]
    cos: torch.Tensor,     # [B, L, D//2]
    sin: torch.Tensor,     # [B, L, D//2]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference: RMSNorm then RoPE, pure PyTorch (no Triton).

    Matches the original Flux2 precision path:
      1. RMSNorm: (x * rrms).to(x_dtype) * weight  — intermediate cast
      2. QKNorm .to(v): cast to input dtype
      3. RoPE in float32, output cast to input dtype
    """
    orig_dtype = x.dtype
    x = x.float()
    # RMSNorm over last dim
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    x = x * rrms * weight.float()
    # RoPE: reshape to expose pairs
    x = x.reshape(*x.shape[:-1], -1, 2)   # [B, L, H, D//2, 2]
    x1, x2 = x[..., 0], x[..., 1]         # each [B, L, H, D//2]
    # broadcast cos/sin over H dim: [B, L, 1, D//2]
    c = cos.unsqueeze(2).float()
    s = sin.unsqueeze(2).float()
    o1 = c * x1 - s * x2
    o2 = s * x1 + c * x2
    out = torch.stack([o1, o2], dim=-1).reshape(*o1.shape[:-1], -1)  # [B,L,H,D]
    return out.to(orig_dtype)


def reference_packed_qk_norm_rope(
    qkv: torch.Tensor,      # [B, L, 3, H, D]
    q_weight: torch.Tensor, # [D]
    k_weight: torch.Tensor, # [D]
    cos: torch.Tensor,      # [B, L, D//2]
    sin: torch.Tensor,      # [B, L, D//2]
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference: packed QKV input — RMSNorm+RoPE on Q and K, V copied unchanged."""
    q = reference_norm_rope(qkv[:, :, 0], q_weight, cos, sin, eps)
    k = reference_norm_rope(qkv[:, :, 1], k_weight, cos, sin, eps)
    v = qkv[:, :, 2].to(q.dtype)
    return torch.stack([q, k, v], dim=2)


def extract_cos_sin(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract cos and sin arrays from the freqs_cis rotation-matrix tensor.

    Parameters
    ----------
    freqs_cis:
        Shape ``[B, 1, L, D//2, 2, 2]`` (Flux2 format from pe_embedder).
        The ``[2, 2]`` block is a rotation matrix::

            [[cos, -sin],
             [sin,  cos]]

    Returns
    -------
    cos : torch.Tensor, shape ``[B, L, D//2]``, float32.
    sin : torch.Tensor, shape ``[B, L, D//2]``, float32.
    """
    fc = freqs_cis[:, 0]               # [B, L, D//2, 2, 2]
    cos = fc[..., 0, 0].contiguous()   # [B, L, D//2]
    sin = fc[..., 1, 0].contiguous()   # [B, L, D//2]
    return cos.float(), sin.float()


# ---------------------------------------------------------------------------
# Backward helpers (pure PyTorch, float32)
# ---------------------------------------------------------------------------


def _rope_backward(
    dout: torch.Tensor,   # [B, L, H, D] float32
    cos: torch.Tensor,    # [B, L, D//2] float32
    sin: torch.Tensor,    # [B, L, D//2] float32
) -> torch.Tensor:
    """Backward through RoPE rotation: apply the transpose (inverse) rotation."""
    B, L, H, D = dout.shape
    D2 = D // 2
    c = cos.unsqueeze(2)  # [B, L, 1, D//2]
    s = sin.unsqueeze(2)
    # dout pairs
    dout_pairs = dout.reshape(B, L, H, D2, 2)
    do1 = dout_pairs[..., 0]  # [B, L, H, D//2]
    do2 = dout_pairs[..., 1]
    # Inverse rotation: R^T @ [do1, do2]
    dn1 = c * do1 + s * do2
    dn2 = -s * do1 + c * do2
    return torch.stack([dn1, dn2], dim=-1).reshape(B, L, H, D)


def _rmsnorm_backward(
    x: torch.Tensor,       # [B, L, H, D] float32 — original input to RMSNorm
    weight: torch.Tensor,  # [D] float32
    dnormed: torch.Tensor, # [B, L, H, D] float32 — grad w.r.t. RMSNorm output
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward through RMSNorm: y = x * rsqrt(mean(x²) + eps) * w. Returns (dx, dw)."""
    D = x.shape[-1]
    rrms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)  # [B, L, H, 1]
    normed = x * rrms
    dw = (dnormed * normed).sum(dim=(0, 1, 2))  # [D]
    dot = (dnormed * weight * x).sum(-1, keepdim=True)  # [B, L, H, 1]
    dx = rrms * (dnormed * weight - (rrms**2 / D) * x * dot)
    return dx, dw


# ---------------------------------------------------------------------------
# Triton kernels (only if HAS_TRITON)
# ---------------------------------------------------------------------------

if HAS_TRITON:
    @triton.jit
    def fused_norm_rope_kernel(
        x_ptr, out_ptr, w_ptr, cos_ptr, sin_ptr,
        H, L, D, D2,        # D2 = D // 2
        eps,
        BLOCK_D2: tl.constexpr,
        STORE_DTYPE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        l = (pid // H) % L  # noqa: E741
        b = pid // (H * L)

        x_base   = b * (L * H * D) + l * (H * D) + h * D
        cos_base = b * (L * D2) + l * D2

        cols = tl.arange(0, BLOCK_D2)
        mask = cols < D2

        # load pairs
        x1 = tl.load(x_ptr + x_base + cols * 2,     mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_base + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)

        # RMSNorm reduction
        ss = tl.sum(x1 * x1 + x2 * x2, axis=0)
        rrms = tl.rsqrt(ss / D + eps)

        # load weights for each element of the pair
        w1 = tl.load(w_ptr + cols * 2,     mask=mask, other=1.0).to(tl.float32)
        w2 = tl.load(w_ptr + cols * 2 + 1, mask=mask, other=1.0).to(tl.float32)

        n1 = x1 * rrms * w1
        n2 = x2 * rrms * w2

        # RoPE
        c = tl.load(cos_ptr + cos_base + cols, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + cos_base + cols, mask=mask, other=0.0).to(tl.float32)

        o1 = c * n1 - s * n2
        o2 = s * n1 + c * n2

        tl.store(out_ptr + x_base + cols * 2,     o1.to(STORE_DTYPE), mask=mask)
        tl.store(out_ptr + x_base + cols * 2 + 1, o2.to(STORE_DTYPE), mask=mask)

    @triton.jit
    def fused_packed_qk_norm_rope_kernel(
        qkv_ptr, out_ptr,
        wq_ptr, wk_ptr,     # [D] norm weights for Q and K respectively
        cos_ptr, sin_ptr,   # [B, L, D//2]
        H, L, D, D2,        # D2 = D // 2
        eps,
        BLOCK_D2: tl.constexpr,
        STORE_DTYPE: tl.constexpr,
    ):
        """One program per (b, l, h) row. Processes Q and K, copies V.

        Input/output layout: [B, L, 3, H, D] contiguous.
        Strides: (L*3*H*D, 3*H*D, H*D, D, 1).
        """
        pid = tl.program_id(0)
        h = pid % H
        l = (pid // H) % L  # noqa: E741
        b = pid // (H * L)

        # Base offsets into [B, L, 3, H, D]
        row_base  = b * (L * 3 * H * D) + l * (3 * H * D) + h * D
        q_base    = row_base + 0 * H * D   # Q index 0
        k_base    = row_base + 1 * H * D   # K index 1
        v_base    = row_base + 2 * H * D   # V index 2
        cos_base  = b * (L * D2) + l * D2

        cols = tl.arange(0, BLOCK_D2)
        mask = cols < D2

        # Load cos/sin (shared by Q and K)
        c = tl.load(cos_ptr + cos_base + cols, mask=mask, other=1.0).to(tl.float32)
        s = tl.load(sin_ptr + cos_base + cols, mask=mask, other=0.0).to(tl.float32)

        # --- Process Q ---
        q1 = tl.load(qkv_ptr + q_base + cols * 2,     mask=mask, other=0.0).to(tl.float32)
        q2 = tl.load(qkv_ptr + q_base + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        ss_q  = tl.sum(q1 * q1 + q2 * q2, axis=0)
        rrms_q = tl.rsqrt(ss_q / D + eps)
        wq1 = tl.load(wq_ptr + cols * 2,     mask=mask, other=1.0).to(tl.float32)
        wq2 = tl.load(wq_ptr + cols * 2 + 1, mask=mask, other=1.0).to(tl.float32)
        nq1 = q1 * rrms_q * wq1
        nq2 = q2 * rrms_q * wq2
        oq1 = c * nq1 - s * nq2
        oq2 = s * nq1 + c * nq2
        tl.store(out_ptr + q_base + cols * 2,     oq1.to(STORE_DTYPE), mask=mask)
        tl.store(out_ptr + q_base + cols * 2 + 1, oq2.to(STORE_DTYPE), mask=mask)

        # --- Process K ---
        k1 = tl.load(qkv_ptr + k_base + cols * 2,     mask=mask, other=0.0).to(tl.float32)
        k2 = tl.load(qkv_ptr + k_base + cols * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        ss_k  = tl.sum(k1 * k1 + k2 * k2, axis=0)
        rrms_k = tl.rsqrt(ss_k / D + eps)
        wk1 = tl.load(wk_ptr + cols * 2,     mask=mask, other=1.0).to(tl.float32)
        wk2 = tl.load(wk_ptr + cols * 2 + 1, mask=mask, other=1.0).to(tl.float32)
        nk1 = k1 * rrms_k * wk1
        nk2 = k2 * rrms_k * wk2
        ok1 = c * nk1 - s * nk2
        ok2 = s * nk1 + c * nk2
        tl.store(out_ptr + k_base + cols * 2,     ok1.to(STORE_DTYPE), mask=mask)
        tl.store(out_ptr + k_base + cols * 2 + 1, ok2.to(STORE_DTYPE), mask=mask)

        # --- Copy V unchanged ---
        v1 = tl.load(qkv_ptr + v_base + cols * 2,     mask=mask, other=0.0)
        v2 = tl.load(qkv_ptr + v_base + cols * 2 + 1, mask=mask, other=0.0)
        tl.store(out_ptr + v_base + cols * 2,     v1, mask=mask)
        tl.store(out_ptr + v_base + cols * 2 + 1, v2, mask=mask)


# ---------------------------------------------------------------------------
# autograd.Function wrappers (Triton forward, PyTorch backward)
# ---------------------------------------------------------------------------

if HAS_TRITON:
    class _FusedNormRopeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, cos, sin, eps):
            # x: [B, L, H, D]
            ctx.save_for_backward(x, weight, cos, sin)
            ctx.eps = eps

            B, L, H, D = x.shape
            D2 = D // 2
            BLOCK_D2 = triton.next_power_of_2(D2)
            out = torch.empty(B, L, H, D, dtype=x.dtype, device=x.device)
            fused_norm_rope_kernel[(B * L * H,)](
                x, out, weight.float(), cos, sin,
                H, L, D, D2, eps,
                BLOCK_D2=BLOCK_D2,
                STORE_DTYPE=tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16 if x.dtype == torch.float16 else tl.float32,
            )
            return out

        @staticmethod
        def backward(ctx, grad_output):
            x, weight, cos, sin = ctx.saved_tensors
            eps = ctx.eps

            dnormed = _rope_backward(grad_output.float(), cos, sin)
            dx_f, dw_f = _rmsnorm_backward(x.float(), weight.float(), dnormed, eps)

            grad_x = dx_f.to(x.dtype) if ctx.needs_input_grad[0] else None
            grad_w = dw_f.to(weight.dtype) if ctx.needs_input_grad[1] else None
            return grad_x, grad_w, None, None, None  # cos, sin, eps: no grad

    class _FusedPackedQKNormRopeFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, qkv, q_weight, k_weight, cos, sin, eps):
            # qkv: [B, L, 3, H, D]
            ctx.save_for_backward(qkv, q_weight, k_weight, cos, sin)
            ctx.eps = eps

            B, L, _, H, D = qkv.shape
            D2 = D // 2
            BLOCK_D2 = triton.next_power_of_2(D2)
            out = torch.empty(B, L, 3, H, D, dtype=qkv.dtype, device=qkv.device)
            fused_packed_qk_norm_rope_kernel[(B * L * H,)](
                qkv, out, q_weight.float(), k_weight.float(), cos, sin,
                H, L, D, D2, eps,
                BLOCK_D2=BLOCK_D2,
                STORE_DTYPE=tl.bfloat16 if qkv.dtype == torch.bfloat16 else tl.float16 if qkv.dtype == torch.float16 else tl.float32,
            )
            return out

        @staticmethod
        def backward(ctx, grad_output):
            qkv, q_weight, k_weight, cos, sin = ctx.saved_tensors
            eps = ctx.eps

            need_grad_qkv = ctx.needs_input_grad[0]
            need_grad_wq  = ctx.needs_input_grad[1]
            need_grad_wk  = ctx.needs_input_grad[2]

            grad_qkv = None
            grad_wq  = None
            grad_wk  = None
            dq_f     = None
            dk_f     = None

            _nan_debug = os.environ.get("MUSUBI_NAN_DEBUG", "0") == "1"
            if _nan_debug:
                def _has_nan(t, label):
                    if torch.isnan(t).any():
                        print(f"[PackedQKNorm bwd] NaN in {label}  shape={tuple(t.shape)}")
                        return True
                    return False
                _has_nan(grad_output[:, :, 0], "grad_output[Q]")
                _has_nan(grad_output[:, :, 1], "grad_output[K]")
                _has_nan(grad_output[:, :, 2], "grad_output[V]")
                _has_nan(qkv[:, :, 0], "saved qkv[Q]")
                _has_nan(qkv[:, :, 1], "saved qkv[K]")

            if need_grad_qkv or need_grad_wq:
                dnormed_q = _rope_backward(grad_output[:, :, 0].float(), cos, sin)
                if _nan_debug: _has_nan(dnormed_q, "dnormed_q (after rope bwd)")
                dq_f, dw_q_f = _rmsnorm_backward(qkv[:, :, 0].float(), q_weight.float(), dnormed_q, eps)
                if _nan_debug: _has_nan(dq_f, "dq_f (after rmsnorm bwd)")
                if need_grad_wq:
                    grad_wq = dw_q_f.to(q_weight.dtype)

            if need_grad_qkv or need_grad_wk:
                dnormed_k = _rope_backward(grad_output[:, :, 1].float(), cos, sin)
                if _nan_debug: _has_nan(dnormed_k, "dnormed_k (after rope bwd)")
                dk_f, dw_k_f = _rmsnorm_backward(qkv[:, :, 1].float(), k_weight.float(), dnormed_k, eps)
                if _nan_debug: _has_nan(dk_f, "dk_f (after rmsnorm bwd)")
                if need_grad_wk:
                    grad_wk = dw_k_f.to(k_weight.dtype)

            if need_grad_qkv:
                grad_qkv = torch.stack(
                    [dq_f.to(qkv.dtype), dk_f.to(qkv.dtype), grad_output[:, :, 2]], dim=2
                )

            return grad_qkv, grad_wq, grad_wk, None, None, None  # cos, sin, eps


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fused_norm_rope(
    x: torch.Tensor,        # [B, L, H, D] contiguous, any float dtype
    weight: torch.Tensor,   # [D] RMSNorm scale
    cos: torch.Tensor,      # [B, L, D//2] float32
    sin: torch.Tensor,      # [B, L, D//2] float32
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm + RoPE for one of Q or K. Call twice with different weights.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``[B, L, H, D]``.  bfloat16, float16, or float32.  Must be
        contiguous.
    weight : torch.Tensor
        RMSNorm scale (``query_norm.scale`` or ``key_norm.scale``).
        Shape ``[D]``.
    cos : torch.Tensor
        Shape ``[B, L, D//2]``, float32.  From :func:`extract_cos_sin`.
    sin : torch.Tensor
        Shape ``[B, L, D//2]``, float32.  From :func:`extract_cos_sin`.
    eps : float
        RMSNorm epsilon (default ``1e-6``).

    Returns
    -------
    torch.Tensor
        Same shape and dtype as *x*.
    """
    if not HAS_TRITON or not x.is_cuda:
        return reference_norm_rope(x, weight, cos, sin, eps)

    assert x.is_contiguous(), "x must be contiguous"
    B, L, H, D = x.shape
    assert D % 2 == 0, f"Head dim D={D} must be even for RoPE"

    return _FusedNormRopeFunction.apply(x, weight, cos, sin, eps)


def fused_packed_qk_norm_rope(
    qkv: torch.Tensor,      # [B, L, 3, H, D] contiguous, any float dtype
    q_weight: torch.Tensor, # [D] RMSNorm scale for Q
    k_weight: torch.Tensor, # [D] RMSNorm scale for K
    cos: torch.Tensor,      # [B, L, D//2] float32
    sin: torch.Tensor,      # [B, L, D//2] float32
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused RMSNorm + RoPE on packed QKV in a single kernel launch.

    Applies RMSNorm + RoPE to Q (index 0) and K (index 1) with separate norm
    weights.  V (index 2) is copied unchanged.  One GPU kernel launch handles
    all three components, avoiding any Q/K separation in Python.

    The backward pass is computed in pure PyTorch (float32), preserving full
    gradient flow to qkv and to the norm scale weights.

    Parameters
    ----------
    qkv : torch.Tensor
        Shape ``[B, L, 3, H, D]``.  Must be contiguous.  Can be obtained from
        a linear projection output via ``proj_out.reshape(B, L, 3, H, D)``.
    q_weight : torch.Tensor
        RMSNorm scale for Q (``query_norm.scale``).  Shape ``[D]``.
    k_weight : torch.Tensor
        RMSNorm scale for K (``key_norm.scale``).  Shape ``[D]``.
    cos : torch.Tensor
        Shape ``[B, L, D//2]``, float32.  From :func:`extract_cos_sin`.
    sin : torch.Tensor
        Shape ``[B, L, D//2]``, float32.  From :func:`extract_cos_sin`.
    eps : float
        RMSNorm epsilon (default ``1e-6``).

    Returns
    -------
    torch.Tensor
        Shape ``[B, L, 3, H, D]``, same dtype as *qkv*.  Q and K are
        normalised and rotated; V is unchanged.
    """
    if not HAS_TRITON or not qkv.is_cuda:
        return reference_packed_qk_norm_rope(qkv, q_weight, k_weight, cos, sin, eps)

    assert qkv.is_contiguous(), "qkv must be contiguous"
    assert qkv.shape[2] == 3, f"Expected packed QKV with dim-2 == 3, got {qkv.shape}"
    B, L, _, H, D = qkv.shape
    assert D % 2 == 0, f"Head dim D={D} must be even for RoPE"

    return _FusedPackedQKNormRopeFunction.apply(qkv, q_weight, k_weight, cos, sin, eps)
