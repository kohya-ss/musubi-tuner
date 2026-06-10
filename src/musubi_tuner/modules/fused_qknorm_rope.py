"""Fused QKNorm (RMSNorm on q/k) + RoPE for packed QKV, in Triton.

Input: packed qkv (B, L, 3, H, D) — typically a view of the attention qkv
projection output. Applies per-head-dim RMSNorm (learnable scale, eps 1e-6
semantics matching flux2_models.RMSNorm) to q and k, then rotary embedding
using the FLUX-style pe tensor (B, 1, L, D//2, 2, 2); v is copied through.
Returns a new contiguous (B, L, 3, H, D) tensor suitable for
flash_attn_qkvpacked_func.

Numerics: all math in fp32 with a single rounding to the input dtype on store
(the eager path rounds at intermediate casts; agreement is within dtype
tolerance, see tests).

Determinism: backward accumulates RMSNorm scale gradients with atomic adds, so
scale grads are bitwise non-deterministic run-to-run (numerically correct).
"""

import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _norm_rope_row(
        in_base, out_base, scale_ptr, cos, sin, offs, mask,
        stride_in_d, stride_out_d, eps,
        D: tl.constexpr,
    ):
        """RMSNorm + rotate one (b, l, h) row of q or k. Pairs are (x[2i], x[2i+1])."""
        x_even = tl.load(in_base + (2 * offs) * stride_in_d, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(in_base + (2 * offs + 1) * stride_in_d, mask=mask, other=0.0).to(tl.float32)
        meansq = (tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)) / D
        rrms = 1.0 / tl.sqrt(meansq + eps)
        s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
        xn_even = x_even * rrms * s_even
        xn_odd = x_odd * rrms * s_odd
        o_even = cos * xn_even - sin * xn_odd
        o_odd = sin * xn_even + cos * xn_odd
        out_ty = out_base.dtype.element_ty
        tl.store(out_base + (2 * offs) * stride_out_d, o_even.to(out_ty), mask=mask)
        tl.store(out_base + (2 * offs + 1) * stride_out_d, o_odd.to(out_ty), mask=mask)

    @triton.jit
    def _fused_qknorm_rope_fwd_kernel(
        x_ptr, out_ptr, q_scale_ptr, k_scale_ptr, pe_ptr,
        L, H,
        stride_xb, stride_xl, stride_xk, stride_xh, stride_xd,
        stride_ob, stride_ol, stride_ok, stride_oh, stride_od,
        stride_pb, stride_pl, stride_pd, stride_pi,
        eps,
        D: tl.constexpr,
        BLOCK_DHALF: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        l = (pid // H) % L
        b = pid // (H * L)

        offs = tl.arange(0, BLOCK_DHALF)
        mask = offs < (D // 2)

        # pe is (B or 1, L, D//2, 2, 2): cos at [..., 0, 0], sin at [..., 1, 0]
        pe_base = pe_ptr + b * stride_pb + l * stride_pl + offs * stride_pd
        cos = tl.load(pe_base, mask=mask, other=0.0)
        sin = tl.load(pe_base + stride_pi, mask=mask, other=0.0)

        x_base = x_ptr + b * stride_xb + l * stride_xl + h * stride_xh
        out_base = out_ptr + b * stride_ob + l * stride_ol + h * stride_oh

        # q (index 0 along K) and k (index 1)
        _norm_rope_row(x_base, out_base, q_scale_ptr, cos, sin, offs, mask, stride_xd, stride_od, eps, D)
        _norm_rope_row(
            x_base + stride_xk, out_base + stride_ok, k_scale_ptr, cos, sin, offs, mask, stride_xd, stride_od, eps, D
        )

        # v passthrough (index 2 along K)
        offs_d = tl.arange(0, 2 * BLOCK_DHALF)
        mask_d = offs_d < D
        v = tl.load(x_base + 2 * stride_xk + offs_d * stride_xd, mask=mask_d, other=0.0)
        tl.store(out_base + 2 * stride_ok + offs_d * stride_od, v, mask=mask_d)


class _FusedQKNormRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv: Tensor, q_scale: Tensor, k_scale: Tensor, pe: Tensor, eps: float):
        B, L, K, H, D = qkv.shape
        out = torch.empty((B, L, 3, H, D), dtype=qkv.dtype, device=qkv.device)
        stride_pb = pe.stride(0) if pe.shape[0] != 1 else 0
        grid = (B * L * H,)
        _fused_qknorm_rope_fwd_kernel[grid](
            qkv, out, q_scale, k_scale, pe,
            L, H,
            qkv.stride(0), qkv.stride(1), qkv.stride(2), qkv.stride(3), qkv.stride(4),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
            stride_pb, pe.stride(1), pe.stride(2), pe.stride(3),
            eps,
            D=D,
            BLOCK_DHALF=triton.next_power_of_2(D // 2),
            num_warps=2,
        )
        ctx.save_for_backward(qkv, q_scale, k_scale, pe)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        raise NotImplementedError("backward implemented in Task 2")


def fused_qknorm_rope(qkv: Tensor, q_scale: Tensor, k_scale: Tensor, pe: Tensor, eps: float = 1e-6) -> Tensor:
    """Apply RMSNorm (q/k) + RoPE to packed qkv (B, L, 3, H, D); v passes through.

    pe: (B or 1, 1, L, D//2, 2, 2) or (B or 1, L, D//2, 2, 2), fp32, as produced
    by flux2_models.EmbedND / rope().
    """
    if not HAS_TRITON:
        raise RuntimeError("fused_qknorm_rope requires triton; install triton or disable the fused path")
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError(f"expected packed qkv of shape (B, L, 3, H, D), got {tuple(qkv.shape)}")
    D = qkv.shape[-1]
    if D % 2 != 0:
        raise ValueError(f"head dim must be even, got {D}")
    if pe.ndim == 6:
        pe = pe.squeeze(1)  # drop broadcast head dim
    if pe.ndim != 5 or pe.shape[-3] != D // 2 or pe.shape[-2:] != (2, 2):
        raise ValueError(f"expected pe of shape (B, [1,] L, {D // 2}, 2, 2), got {tuple(pe.shape)}")
    if pe.shape[1] != qkv.shape[1]:
        raise ValueError(f"pe seq len {pe.shape[1]} != qkv seq len {qkv.shape[1]}")
    return _FusedQKNormRoPE.apply(qkv, q_scale.contiguous(), k_scale.contiguous(), pe.float(), eps)
