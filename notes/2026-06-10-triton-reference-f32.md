# Triton fused norm+rope: reference path diverged from original (2026-06-10)

Branch: `packed`. Files: `src/musubi_tuner/kernels/fused_norm_rope.py`,
`tests/test_packed_vs_original.py`, `tests/test_fused_norm_rope.py`,
`tests/test_fused_norm_rope_gradients.py`.

## Context

The `packed` branch adds a fused QK-norm + RoPE Triton kernel for FLUX.2
(`SingleStreamBlock` / `DoubleStreamBlock` in `flux2_models.py`). The kernel has
a pure-PyTorch "reference" that doubles as:

1. the **production fallback** (no Triton, `MUSUBI_DISABLE_TRITON=1`, or CPU tensors), and
2. the **test oracle** for the Triton kernel.

## The bug

Commit `a329d5e` ("Simplify RMSNorm: remove intermediate dtype casts") changed
the reference to compute RMSNorm entirely in float32:

```python
x = x * rrms * weight.float()   # all fp32
```

But `main`'s original RMSNorm rounds back to the input dtype **before** the
scale multiply, and QKNorm then casts again:

```python
# RMSNorm.forward
return (x.float() * rrms).to(x_dtype) * self.scale
# QKNorm.forward
return q.to(v), k.to(v)
```

So the fallback no longer matched `main`'s numerics in bf16 (an extra bf16
rounding of the normalized tensor before/at the scale multiply was dropped). The
RoPE math itself was fine and equivalent.

Two extra problems made this hard to see:

- `reference_norm_rope`'s docstring still **claimed** it matched the cast path.
- The test helper `_original_path._rmsnorm` in `test_packed_vs_original.py` had
  *also* been changed to the simplified fp32 math, so the "reference vs original"
  tests were actually "simplified vs simplified" — they documented a bf16 "gap"
  that mostly wasn't `main`'s.

## Decision

- The **fused Triton kernel stays fp32** (unchanged) — this is the desired,
  more-accurate path matching the TensorRT-LLM formulation.
- The **reference / fallback must be byte-identical to `main`** so that turning
  Triton off reproduces original behavior exactly.
- Consequence accepted: with the fallback now bf16-faithful and the kernel fp32,
  the kernel no longer equals its old reference. So the kernel needs a separate
  fp32 oracle for tests.

## Changes

`kernels/fused_norm_rope.py`
- `reference_norm_rope` / `reference_packed_qk_norm_rope` → reproduce `main`
  exactly: `(x.float()*rrms).to(orig_dtype) * weight`, then `.to(orig_dtype)`
  (QKNorm `.to(v)`), then RoPE in float32, then cast back. Docstring corrected.
- Added `reference_norm_rope_fp32` / `reference_packed_qk_norm_rope_fp32` —
  float32-throughout oracle that matches the fused kernel. Use these to verify
  the kernel; use the non-`_fp32` ones as the production fallback.

Tests
- `test_packed_vs_original.py`: fixed `_original_path._rmsnorm` to `main`'s real
  `(x*rrms).to(dtype)*scale`; added `_reference_fp32_path`; tightened the bf16
  forward/backward "gap" tests to assert exact equality (they now match);
  pointed the fused-vs-reference CUDA tests at the fp32 oracle.
- `test_fused_norm_rope.py` and `test_fused_norm_rope_gradients.py`: kernel
  comparison tests now use the `_fp32` oracle instead of the bf16-faithful
  reference.

## Verification

`MUSUBI_DISABLE_TRITON=1 uv run --extra cu132 pytest tests/test_packed_vs_original.py tests/test_fused_norm_rope.py tests/test_fused_norm_rope_gradients.py`

- 21 CPU tests pass; 31 Triton/CUDA tests skip (no GPU on this machine).
- Forward gap reference-vs-original = **0.0 exactly** in both fp32 and bf16.
- Backward gap: 0.0 for qkv; ~1e-5 for q/k scale (float accumulation order from
  the `[B,H,L,D]` vs `[B,L,H,D]` layout in `apply_rope`'s reduction — not a
  precision-path difference).

## Follow-ups

- The Triton/CUDA tests were rewired to the fp32 oracle but not executed here —
  run them on a CUDA box to confirm.
- Be aware: enabling vs. disabling fusion now gives slightly different numerics
  (fp32 kernel vs bf16-faithful fallback). This is intentional per the decision
  above.
