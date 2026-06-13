"""torch.profiler: fused vs eager QKNorm + RoPE (forward and forward+backward).

Follows the repo's profiling conventions (see profile_fused_norm_rope.py):
  - schedule(wait=1, warmup=1, active=3) to skip cold-start noise
  - record_function annotations for readable traces
  - CPU + CUDA activities, memory profiling enabled
  - Table (txt) + Chrome trace (json) exported per variant

Usage:
  uv run --no-sync python benchmarking/profile_fused_qknorm_rope.py
  uv run --no-sync python benchmarking/profile_fused_qknorm_rope.py --shape 1 8192 48 128
  uv run --no-sync python benchmarking/profile_fused_qknorm_rope.py --dtype fp16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.profiler
from einops import rearrange

from musubi_tuner.flux_2.flux2_models import apply_rope, rope
from musubi_tuner.modules.fused_qknorm_rope import HAS_TRITON, fused_qknorm_rope

SCHEDULE = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
WARMUP_ITERS = 30  # runs outside the profiler to eliminate cold-start


# ---------------------------------------------------------------------------
# Eager path (mirrors flux2_models: rearrange -> RMSNorm -> apply_rope)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Profiling helpers
# ---------------------------------------------------------------------------


def profile_variant(label: str, fn, trace_dir: Path, *, warmup: int = WARMUP_ITERS) -> None:
    """Run torch.profiler on *fn*, save table + chrome trace, print table."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    def step():
        with torch.profiler.record_function(label):
            fn()

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=SCHEDULE,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(5):  # wait=1 + warmup=1 + active=3
            step()
            prof.step()

    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=25)

    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace("+", "_")
    table_path = trace_dir / f"{safe_label}.txt"
    trace_path = trace_dir / f"{safe_label}.json"
    table_path.write_text(table)
    prof.export_chrome_trace(str(trace_path))

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(table)
    print(f"  Table  -> {table_path}")
    print(f"  Trace  -> {trace_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Profile fused QKNorm+RoPE with torch.profiler")
    parser.add_argument(
        "--shape",
        nargs=4,
        type=int,
        metavar=("B", "L", "H", "D"),
        default=[1, 4096, 48, 128],
        help="Input shape B L H D (default: 1 4096 48 128)",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERS, help="Pre-profiler warmup iterations")
    parser.add_argument("--out", default="traces/profile_fused_qknorm_rope", help="Output directory for traces")
    parser.add_argument(
        "--frozen-scales",
        action="store_true",
        help="freeze the RMSNorm scales (the LoRA training case; skips dscale atomics via NEEDS_DSCALE)",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available() and HAS_TRITON, "CUDA + Triton required"

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    B, L, H, D = args.shape
    device = "cuda"

    tag = f"B{B}_L{L}_H{H}_D{D}_{args.dtype}" + ("_frozen" if args.frozen_scales else "")
    trace_dir = Path(args.out) / tag
    trace_dir.mkdir(parents=True, exist_ok=True)

    print(f"dtype  : {dtype}")
    print(f"Shape  : B={B} L={L} H={H} D={D}")
    print(f"Scales : {'frozen (LoRA case)' if args.frozen_scales else 'trainable'}")
    print(f"Traces : {trace_dir.resolve()}")

    torch.manual_seed(42)
    pos = torch.arange(L, device=device).unsqueeze(0).float()
    pe = rope(pos, D, 2000).unsqueeze(1)
    qkv = torch.randn(B, L, 3, H, D, device=device, dtype=dtype, requires_grad=True)
    train_scales = not args.frozen_scales
    q_scale = torch.ones(D, device=device, dtype=dtype, requires_grad=train_scales)
    k_scale = torch.ones(D, device=device, dtype=dtype, requires_grad=train_scales)
    go = torch.randn(B, L, 3, H, D, device=device, dtype=dtype)

    def clear_grads():
        qkv.grad = q_scale.grad = k_scale.grad = None

    def fwd_eager():
        with torch.no_grad():
            eager_path(qkv, q_scale, k_scale, pe)

    def fwd_fused():
        with torch.no_grad():
            fused_qknorm_rope(qkv, q_scale, k_scale, pe)

    def fwdbwd_eager():
        q, k, v = eager_path(qkv, q_scale, k_scale, pe)
        torch.autograd.backward([q, k, v], [go[:, :, 0], go[:, :, 1], go[:, :, 2]])
        clear_grads()

    def fwdbwd_fused():
        out = fused_qknorm_rope(qkv, q_scale, k_scale, pe)
        out.backward(go)
        clear_grads()

    profile_variant("eager fwd", fwd_eager, trace_dir, warmup=args.warmup)
    profile_variant("fused fwd", fwd_fused, trace_dir, warmup=args.warmup)
    profile_variant("eager fwd+bwd", fwdbwd_eager, trace_dir, warmup=args.warmup)
    profile_variant("fused fwd+bwd", fwdbwd_fused, trace_dir, warmup=args.warmup)

    print(f"\nAll traces written to: {trace_dir.resolve()}")
    print("Open .json files in https://ui.perfetto.dev/ to view traces.")


if __name__ == "__main__":
    main()
