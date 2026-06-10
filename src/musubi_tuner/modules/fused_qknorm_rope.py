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

    @triton.jit
    def _bwd_row(
        go_base, x_base, gx_base, scale_ptr, dscale_ptr, cos, sin, offs, mask,
        stride_go_d, stride_x_d, stride_gx_d, eps,
        D: tl.constexpr,
        NEEDS_DSCALE: tl.constexpr,
    ):
        """Backward for one q or k row: un-rotate, RMSNorm backward, dscale atomics."""
        go_even = tl.load(go_base + (2 * offs) * stride_go_d, mask=mask, other=0.0).to(tl.float32)
        go_odd = tl.load(go_base + (2 * offs + 1) * stride_go_d, mask=mask, other=0.0).to(tl.float32)
        # RoPE backward = transposed rotation
        g_even = cos * go_even + sin * go_odd
        g_odd = -sin * go_even + cos * go_odd

        x_even = tl.load(x_base + (2 * offs) * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        x_odd = tl.load(x_base + (2 * offs + 1) * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        meansq = (tl.sum(x_even * x_even, axis=0) + tl.sum(x_odd * x_odd, axis=0)) / D
        rrms = 1.0 / tl.sqrt(meansq + eps)

        # dscale += g * x_hat (fp32 atomics; non-deterministic ordering)
        if NEEDS_DSCALE:
            tl.atomic_add(dscale_ptr + 2 * offs, g_even * (x_even * rrms), mask=mask)
            tl.atomic_add(dscale_ptr + 2 * offs + 1, g_odd * (x_odd * rrms), mask=mask)

        s_even = tl.load(scale_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
        s_odd = tl.load(scale_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
        gs_even = g_even * s_even
        gs_odd = g_odd * s_odd
        dot = (tl.sum(gs_even * x_even, axis=0) + tl.sum(gs_odd * x_odd, axis=0)) / D
        coef = rrms * rrms * rrms * dot
        gx_even = rrms * gs_even - x_even * coef
        gx_odd = rrms * gs_odd - x_odd * coef
        ty = gx_base.dtype.element_ty
        tl.store(gx_base + (2 * offs) * stride_gx_d, gx_even.to(ty), mask=mask)
        tl.store(gx_base + (2 * offs + 1) * stride_gx_d, gx_odd.to(ty), mask=mask)

    @triton.jit
    def _fused_qknorm_rope_bwd_kernel(
        go_ptr, x_ptr, gx_ptr, q_scale_ptr, k_scale_ptr, dq_scale_ptr, dk_scale_ptr, pe_ptr,
        L, H,
        stride_gob, stride_gol, stride_gok, stride_goh, stride_god,
        stride_xb, stride_xl, stride_xk, stride_xh, stride_xd,
        stride_gxb, stride_gxl, stride_gxk, stride_gxh, stride_gxd,
        stride_pb, stride_pl, stride_pd, stride_pi,
        eps,
        D: tl.constexpr,
        BLOCK_DHALF: tl.constexpr,
        NEEDS_DSCALE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        h = pid % H
        l = (pid // H) % L
        b = pid // (H * L)

        offs = tl.arange(0, BLOCK_DHALF)
        mask = offs < (D // 2)

        pe_base = pe_ptr + b * stride_pb + l * stride_pl + offs * stride_pd
        cos = tl.load(pe_base, mask=mask, other=0.0)
        sin = tl.load(pe_base + stride_pi, mask=mask, other=0.0)

        go_base = go_ptr + b * stride_gob + l * stride_gol + h * stride_goh
        x_base = x_ptr + b * stride_xb + l * stride_xl + h * stride_xh
        gx_base = gx_ptr + b * stride_gxb + l * stride_gxl + h * stride_gxh

        _bwd_row(go_base, x_base, gx_base, q_scale_ptr, dq_scale_ptr, cos, sin, offs, mask,
                 stride_god, stride_xd, stride_gxd, eps, D, NEEDS_DSCALE)
        _bwd_row(go_base + stride_gok, x_base + stride_xk, gx_base + stride_gxk,
                 k_scale_ptr, dk_scale_ptr, cos, sin, offs, mask,
                 stride_god, stride_xd, stride_gxd, eps, D, NEEDS_DSCALE)

        # grad_v passthrough
        offs_d = tl.arange(0, 2 * BLOCK_DHALF)
        mask_d = offs_d < D
        gv = tl.load(go_base + 2 * stride_gok + offs_d * stride_god, mask=mask_d, other=0.0)
        tl.store(gx_base + 2 * stride_gxk + offs_d * stride_gxd, gv, mask=mask_d)


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
        qkv, q_scale, k_scale, pe = ctx.saved_tensors
        B, L, K, H, D = qkv.shape
        grad_out = grad_out.contiguous()
        grad_qkv = torch.empty((B, L, 3, H, D), dtype=qkv.dtype, device=qkv.device)
        needs_dscale = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        if needs_dscale:
            dq_scale = torch.zeros(D, dtype=torch.float32, device=qkv.device)
            dk_scale = torch.zeros(D, dtype=torch.float32, device=qkv.device)
        else:
            # Pass valid pointers the kernel won't touch when NEEDS_DSCALE=False
            dq_scale = grad_qkv
            dk_scale = grad_qkv
        stride_pb = pe.stride(0) if pe.shape[0] != 1 else 0
        grid = (B * L * H,)
        _fused_qknorm_rope_bwd_kernel[grid](
            grad_out, qkv, grad_qkv, q_scale, k_scale, dq_scale, dk_scale, pe,
            L, H,
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3), grad_out.stride(4),
            qkv.stride(0), qkv.stride(1), qkv.stride(2), qkv.stride(3), qkv.stride(4),
            grad_qkv.stride(0), grad_qkv.stride(1), grad_qkv.stride(2), grad_qkv.stride(3), grad_qkv.stride(4),
            stride_pb, pe.stride(1), pe.stride(2), pe.stride(3),
            ctx.eps,
            D=D,
            BLOCK_DHALF=triton.next_power_of_2(D // 2),
            num_warps=2,
            NEEDS_DSCALE=needs_dscale,
        )
        grad_q_scale = dq_scale.to(q_scale.dtype) if ctx.needs_input_grad[1] else None
        grad_k_scale = dk_scale.to(k_scale.dtype) if ctx.needs_input_grad[2] else None
        return grad_qkv, grad_q_scale, grad_k_scale, None, None


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
    if pe.shape[0] not in (1, qkv.shape[0]):
        raise ValueError(f"pe batch dim {pe.shape[0]} must be 1 or match qkv batch dim {qkv.shape[0]}")
    return _FusedQKNormRoPE.apply(qkv, q_scale.contiguous(), k_scale.contiguous(), pe.float(), eps)
