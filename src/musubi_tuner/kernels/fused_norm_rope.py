"""Fused QKNorm (RMSNorm per head) + RoPE Triton kernel — forward only.

The kernel fuses two sequential operations applied to Q and K tensors before
flash attention in the Flux2 model:

1. RMSNorm (per-head, with a learned scale weight)
2. RoPE rotation using cos/sin extracted from freqs_cis

Usage
-----
Extract cos/sin from freqs_cis once, then call fused_norm_rope for each of Q and K::

    cos, sin = extract_cos_sin(freqs_cis)                   # [B, L, D//2]
    q = fused_norm_rope(q, query_norm_scale, cos, sin)      # [B, L, H, D]
    k = fused_norm_rope(k, key_norm_scale,   cos, sin)      # [B, L, H, D]

Input tensor layout expected by the wrapper: **[B, L, H, D]** (after the
transpose that is normally done before flash-attention, but *before* the
QKNorm + RoPE steps).  This avoids the non-view rearrange otherwise needed
when working in [B, H, L, D] layout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------


@triton.jit
def fused_qk_norm_rope_kernel(
    x_ptr,       # input  [B, L, H, D] – bf16 / fp16 / fp32
    out_ptr,     # output [B, L, H, D] – same dtype as input
    w_ptr,       # RMSNorm scale weight [D]
    cos_ptr,     # cos values [B, L, D//2] – float32
    sin_ptr,     # sin values [B, L, D//2] – float32
    # Strides (in elements, not bytes)
    stride_x_b,    # x stride along batch dim
    stride_x_l,    # x stride along sequence dim
    stride_x_h,    # x stride along head dim (D is stride-1)
    stride_cos_b,  # cos stride along batch
    stride_cos_l,  # cos stride along sequence (D//2 is stride-1)
    # Runtime scalars
    H,  # num_heads
    L,  # seq_len
    # Compile-time constants
    D: tl.constexpr,        # head dimension
    D_HALF: tl.constexpr,   # D // 2
    eps: tl.constexpr,      # RMSNorm epsilon
    BLOCK_D: tl.constexpr,  # next_power_of_2(D), >= D
):
    """One Triton program handles one (b, l, h) row of D elements.

    Algorithm
    ---------
    1. Load x[b, l, h, :] → fp32.
    2. Compute RMSNorm with scale weight → x_norm.
    3. Load cos/sin for position l (shape D//2).
    4. Expand cos/sin to full D: pair d → positions 2d and 2d+1 share the same
       cos[d]/sin[d].
    5. Compute the RoPE rotation in a vectorised, fully register-based pass:
          out[2d]   = cos[d]*x_norm[2d]   - sin[d]*x_norm[2d+1]
          out[2d+1] = sin[d]*x_norm[2d]   + cos[d]*x_norm[2d+1]
    6. Store output (cast back to input dtype).
    """
    row = tl.program_id(0)

    # --- Decode (b, l, h) from flat row index ---
    LH = L * H
    b = row // LH
    rem = row - b * LH
    l = rem // H  # noqa: E741
    h = rem - l * H

    x_offset = b * stride_x_b + l * stride_x_l + h * stride_x_h
    cos_offset = b * stride_cos_b + l * stride_cos_l

    # --- Step 1: Load D elements → fp32 ---
    d_idx = tl.arange(0, BLOCK_D)
    mask_d = d_idx < D
    x_vals = tl.load(x_ptr + x_offset + d_idx, mask=mask_d, other=0.0).to(tl.float32)

    # --- Step 2: RMSNorm ---
    w_vals = tl.load(w_ptr + d_idx, mask=mask_d, other=1.0).to(tl.float32)
    sum_sq = tl.sum(x_vals * x_vals, axis=0)
    mean_sq = sum_sq / D
    rrms = 1.0 / tl.sqrt(mean_sq + eps)
    x_norm = x_vals * rrms * w_vals  # [BLOCK_D], fp32

    # --- Step 3: Load cos/sin [D_HALF] ---
    half_idx = tl.arange(0, BLOCK_D // 2)
    mask_half = half_idx < D_HALF
    cos_half = tl.load(cos_ptr + cos_offset + half_idx, mask=mask_half, other=1.0)
    sin_half = tl.load(sin_ptr + cos_offset + half_idx, mask=mask_half, other=0.0)

    # --- Step 4: Expand cos/sin from D_HALF → D using pair-wise indexing.
    #
    # For pair d:
    #   even position 2d   → cos_half[d], sin_half[d]
    #   odd  position 2d+1 → cos_half[d], sin_half[d]
    #
    # We achieve this with an index map: pair_idx[i] = i // 2.
    # Then cos_full[i] = cos_half[pair_idx[i]], etc.
    #
    pair_idx = d_idx // 2  # [0, 0, 1, 1, 2, 2, ...]  shape [BLOCK_D]
    mask_pair = pair_idx < D_HALF
    cos_full = tl.load(cos_ptr + cos_offset + pair_idx, mask=mask_pair, other=1.0)
    sin_full = tl.load(sin_ptr + cos_offset + pair_idx, mask=mask_pair, other=0.0)

    # --- Step 5: Compute rotation for all D positions simultaneously.
    #
    # For each position i:
    #   - If i is even (i = 2d): partner is i+1.
    #   - If i is odd  (i = 2d+1): partner is i-1.
    #
    # Build the partner value (x_norm shifted by ±1 within each pair).
    # Triton cannot do dynamic gather from registers, so we use the
    # "interleave-shift" trick:
    #   - x_even[d] = x_norm[2d]   → expand to [BLOCK_D] by doubling: index (d_idx & ~1)
    #   - x_odd [d] = x_norm[2d+1] → expand to [BLOCK_D] by doubling: index (d_idx | 1)
    # Then:
    #   is_even_mask = (d_idx % 2 == 0)
    #   out = where(is_even,
    #               cos*x_even - sin*x_odd,   # even output
    #               sin*x_even + cos*x_odd)   # odd  output
    #
    even_partner_idx = d_idx & ~1   # for pos i: 2*(i//2)   → [0, 0, 2, 2, 4, 4, ...]
    odd_partner_idx = d_idx | 1    # for pos i: 2*(i//2)+1 → [1, 1, 3, 3, 5, 5, ...]

    x_even_partner = tl.load(
        x_ptr + x_offset + even_partner_idx, mask=even_partner_idx < D, other=0.0
    ).to(tl.float32)
    x_odd_partner = tl.load(
        x_ptr + x_offset + odd_partner_idx, mask=odd_partner_idx < D, other=0.0
    ).to(tl.float32)

    # Apply RMSNorm to partners (same rrms, but different weight positions)
    w_even = tl.load(w_ptr + even_partner_idx, mask=even_partner_idx < D, other=1.0).to(tl.float32)
    w_odd = tl.load(w_ptr + odd_partner_idx, mask=odd_partner_idx < D, other=1.0).to(tl.float32)

    x_even_norm = x_even_partner * rrms * w_even
    x_odd_norm = x_odd_partner * rrms * w_odd

    # even position output: cos*x_even - sin*x_odd
    out_even = cos_full * x_even_norm - sin_full * x_odd_norm
    # odd  position output: sin*x_even + cos*x_odd
    out_odd = sin_full * x_even_norm + cos_full * x_odd_norm

    # Select based on whether position is even or odd
    is_even = (d_idx % 2) == 0
    out_vals = tl.where(is_even, out_even, out_odd)

    # --- Step 6: Store (auto-cast fp32 → output dtype) ---
    tl.store(out_ptr + x_offset + d_idx, out_vals, mask=mask_d)


# ---------------------------------------------------------------------------
# Helper: extract cos/sin from freqs_cis
# ---------------------------------------------------------------------------


def extract_cos_sin(freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract cos and sin arrays from the freqs_cis rotation-matrix tensor.

    Parameters
    ----------
    freqs_cis:
        Shape ``[B, 1, L, D//2, 2, 2]`` (float32).  The last two dims are the
        2×2 rotation matrix::

            [[cos, -sin],
             [sin,  cos]]

    Returns
    -------
    cos : torch.Tensor, shape ``[B, L, D//2]``, float32.
    sin : torch.Tensor, shape ``[B, L, D//2]``, float32.
    """
    # freqs_cis[b, 0, l, d, 0, 0] = cos
    # freqs_cis[b, 0, l, d, 1, 0] = sin
    cos = freqs_cis[:, 0, :, :, 0, 0].contiguous()  # [B, L, D//2]
    sin = freqs_cis[:, 0, :, :, 1, 0].contiguous()  # [B, L, D//2]
    return cos, sin


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


def fused_norm_rope(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply fused RMSNorm + RoPE to a single Q or K tensor.

    Parameters
    ----------
    x : torch.Tensor
        Shape ``[B, L, H, D]``.  bfloat16, float16, or float32.
    weight : torch.Tensor
        RMSNorm scale (``query_norm.scale`` or ``key_norm.scale``).
        Shape ``[D]``.  Same dtype as *x*.
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
    B, L, H, D = x.shape
    assert D % 2 == 0, f"Head dim D={D} must be even for RoPE"
    D_HALF = D // 2

    x = x.contiguous()
    weight = weight.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    out = torch.empty_like(x)

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B * L * H,)

    fused_qk_norm_rope_kernel[grid](
        x,
        out,
        weight,
        cos,
        sin,
        x.stride(0),    # stride_x_b
        x.stride(1),    # stride_x_l
        x.stride(2),    # stride_x_h
        cos.stride(0),  # stride_cos_b
        cos.stride(1),  # stride_cos_l
        H,
        L,
        D=D,
        D_HALF=D_HALF,
        eps=eps,
        BLOCK_D=BLOCK_D,
    )
    return out


# ---------------------------------------------------------------------------
# Pure-PyTorch reference implementation (for testing / fallback)
# ---------------------------------------------------------------------------


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm over last dim, with learned scale ``weight``."""
    x_fp32 = x.float()
    rrms = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_fp32 * rrms).to(x.dtype) * weight


def _apply_rope_ref(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE given pre-extracted cos/sin.

    Parameters
    ----------
    x   : ``[B, L, H, D]``
    cos : ``[B, L, D//2]``
    sin : ``[B, L, D//2]``
    """
    B, L, H, D = x.shape
    D_HALF = D // 2
    cos = cos.unsqueeze(2).expand(B, L, H, D_HALF)  # [B, L, H, D//2]
    sin = sin.unsqueeze(2).expand(B, L, H, D_HALF)

    x1 = x[..., 0::2].float()   # even positions → [B, L, H, D//2]
    x2 = x[..., 1::2].float()   # odd  positions → [B, L, H, D//2]

    out1 = cos * x1 - sin * x2
    out2 = sin * x1 + cos * x2

    out = torch.stack([out1, out2], dim=-1)  # [B, L, H, D//2, 2]
    return out.reshape(B, L, H, D).to(x.dtype)


def qk_norm_rope_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference: RMSNorm then RoPE, pure PyTorch (no Triton)."""
    x_normed = _rms_norm_ref(x, weight, eps)
    return _apply_rope_ref(x_normed, cos, sin)
