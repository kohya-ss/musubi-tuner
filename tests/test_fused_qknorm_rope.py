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
    """Build pe matching EmbedND output: (B, 1, L, D//2, 2, 2), fp32."""
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1).float()
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
@pytest.mark.parametrize("shape", [(2, 33, 4, 128), (1, 64, 1, 128), (2, 17, 3, 64)])
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
    qkv = flat[..., : 3 * H * D].reshape(B, L, 3, H, D)  # non-contiguous parent slice
    q_scale = torch.ones(D, device="cuda")
    k_scale = torch.ones(D, device="cuda")
    pe = make_pe(B, L, D, "cuda")
    out = fused_qknorm_rope(qkv, q_scale, k_scale, pe)
    ref = eager_reference(qkv, q_scale, k_scale, pe)
    torch.testing.assert_close(out, ref, **TOLS[torch.float32])
