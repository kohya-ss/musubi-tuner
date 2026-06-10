"""Benchmark fused QKNorm+RoPE (packed) vs eager path at FLUX.2 shapes.

Run with:
  uv run --extra cu132 python benchmarking/bench_fused_qknorm_rope.py
"""

import torch
from einops import rearrange

from musubi_tuner.flux_2.flux2_models import apply_rope, rope
from musubi_tuner.modules.fused_qknorm_rope import fused_qknorm_rope

DEVICE = "cuda"
DTYPE = torch.bfloat16


def eager_rms(x, scale, eps=1e-6):
    x_dtype = x.dtype
    x = x.float()
    rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps)
    return (x * rrms).to(dtype=x_dtype) * scale


def eager_path(qkv, q_scale, k_scale, pe):
    q, k, v = rearrange(qkv, "B L K H D -> K B H L D")
    q = eager_rms(q, q_scale).to(v.dtype)
    k = eager_rms(k, k_scale).to(v.dtype)
    q, k = apply_rope(q, k, pe)
    return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)


def bench(fn, n_warmup=10, n_iter=50):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def peak_mem(fn):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def main():
    for B, L, H, D in [(1, 4096, 48, 128), (1, 8192, 48, 128), (4, 4096, 24, 128)]:
        pos = torch.arange(L, device=DEVICE).unsqueeze(0).float()
        pe = rope(pos, D, 2000).unsqueeze(1)
        qkv = torch.randn(B, L, 3, H, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        q_scale = torch.ones(D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        k_scale = torch.ones(D, device=DEVICE, dtype=DTYPE, requires_grad=True)
        go = torch.randn(B, L, 3, H, D, device=DEVICE, dtype=DTYPE)

        def fwd_fused():
            return fused_qknorm_rope(qkv, q_scale, k_scale, pe)

        def fwd_eager():
            return eager_path(qkv, q_scale, k_scale, pe)

        def fwdbwd_fused():
            out = fused_qknorm_rope(qkv, q_scale, k_scale, pe)
            out.backward(go)
            qkv.grad = q_scale.grad = k_scale.grad = None

        def fwdbwd_eager():
            q, k, v = eager_path(qkv, q_scale, k_scale, pe)
            torch.autograd.backward([q, k, v], [go[:, :, 0], go[:, :, 1], go[:, :, 2]])
            qkv.grad = q_scale.grad = k_scale.grad = None

        print(f"\nB={B} L={L} H={H} D={D} ({DTYPE})")
        print(f"  fwd      eager {bench(fwd_eager):8.3f} ms   fused {bench(fwd_fused):8.3f} ms")
        print(f"  fwd+bwd  eager {bench(fwdbwd_eager):8.3f} ms   fused {bench(fwdbwd_fused):8.3f} ms")
        print(f"  peak mem fwd  eager {peak_mem(fwd_eager):8.1f} MB  fused {peak_mem(fwd_fused):8.1f} MB")


if __name__ == "__main__":
    main()
