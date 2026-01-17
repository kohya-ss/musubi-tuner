# CuTE (CUDA Templates) Integration Plan for blissful-tuner

**Created:** 2026-01-16
**Target:** NVIDIA B300 (SM 10.3) with flash-attention 2.8.3+varlen.sm103
**Status:** Implemented (see `docs/cute_attention.md`)

---

## Executive Summary

This document outlines the plan to integrate Flash-Attention CuTE (CUDA Templates) backend into blissful-tuner for optimized attention on Blackwell GPUs (B300/SM103). CuTE provides 30-70% faster attention compared to FA2 on Blackwell hardware.

**Key Finding:** Integration is straightforward because:
1. Tensor layout already matches (`[B, L, H, D]`)
2. All models use head_dim=128 (CuTE supported)
3. Backend selection infrastructure exists (`attn_mode` parameter)

---

## 1. Prerequisites

### Flash-Attention Build (Already Complete)

```bash
# Installed version
flash-attention: 2.8.3+varlen.sm103
Location: /root/flash-attention

# Key features enabled
- 72 native sm_103 kernels
- Varlen backward working on SM103
- CUTLASS DSL 4.3.5

# CuTE dependencies
pip install 'quack-kernels==0.2.4'
```

### Verification Commands

```bash
# Check flash-attention
python3 -c "import flash_attn; print(flash_attn.__version__)"
# Expected: 2.8.3+varlen.sm103

# Check CuTE availability
python3 -c "from flash_attn.cute import flash_attn_func; print('CuTE OK')"

# Check GPU
python3 -c "import torch; print(torch.cuda.get_device_capability())"
# Expected: (10, 3) for B300
```

---

## 2. Model Compatibility

| Model | Head Dim | Num Heads | Dim | CuTE Compatible |
|-------|----------|-----------|-----|-----------------|
| WAN 2.1/2.2 14B | 128 | 40 | 5120 | ✅ |
| WAN 1.3B | 128 | 12 | 1536 | ✅ |
| Qwen Image | 128 | 16+ | varies | ✅ |
| HunyuanVideo | 128 | varies | varies | ✅ |

**CuTE supports head_dim: 64, 96, 128** - All models compatible!

---

## 3. Current Architecture

### Attention Backend Selection Flow

```
Training Script CLI flags
    │
    ├── --sdpa          → attn_mode = "torch"
    ├── --flash_attn    → attn_mode = "flash"
    ├── --flash3        → attn_mode = "flash3"
    ├── --xformers      → attn_mode = "xformers"
    ├── --sage_attn     → attn_mode = "sageattn"
    └── --cute [NEW]    → attn_mode = "cute"
    │
    ▼
Model Constructor (attn_mode passed)
    │
    ▼
Attention Module (dispatches based on attn_mode)
```

### Key Attention Files

| File | Purpose | Models Using It |
|------|---------|-----------------|
| `src/musubi_tuner/wan/modules/attention.py` | WAN-specific attention | WAN 2.1, WAN 2.2 |
| `src/musubi_tuner/hunyuan_model/attention.py` | Hunyuan attention | Qwen Image, HunyuanVideo |
| `src/musubi_tuner/modules/attention.py` | Generic attention | Various |

### Tensor Layout

**Current:** `[B, L, H, D]` (Batch, SeqLen, Heads, HeadDim)
**CuTE expects:** `[B, L, H, D]`

**No transpose needed!** Direct compatibility.

---

## 4. Implementation Plan

### Phase 1: WAN Attention Module

**File:** `/root/blissful-tuner/src/musubi_tuner/wan/modules/attention.py`

#### Step 1.1: Add CuTE Import (after line 31)

```python
try:
    from flash_attn.cute import flash_attn_func as cute_flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as cute_flash_attn_varlen_func
    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
```

#### Step 1.2: Add CuTE Case in flash_attention() (after line ~141, after flash2 block)

```python
    # CuTE (Blackwell/Hopper optimized) - returns (out, lse) tuple
    if attn_mode == "cute":
        if not CUTE_AVAILABLE:
            raise ImportError("CuTE not available. Install with: pip install 'quack-kernels==0.2.4'")
        if q_scale is not None:
            q = q * q_scale
        q = half(q)
        k = half(k)
        v = half(v)

        if not split_attn:
            out, _ = cute_flash_attn_func(
                q, k, v,
                dropout_p=0.0,  # CuTE doesn't support dropout
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic
            )
            x = out
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                out, _ = cute_flash_attn_func(
                    q[i : i + 1], k[i : i + 1], v[i : i + 1],
                    dropout_p=0.0,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic
                )
                x[i : i + 1] = out
        del q, k, v
        return x.type(out_dtype)
```

#### Step 1.3: Add CuTE Varlen Case (after sageattn block, ~line 259)

```python
    elif attn_mode == "cute_varlen":
        if not CUTE_AVAILABLE:
            raise ImportError("CuTE not available")
        out, _ = cute_flash_attn_varlen_func(
            q=q, k=k, v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )
        x = out.unflatten(0, (b, lq))
```

---

### Phase 2: Hunyuan Attention Module

**File:** `/root/blissful-tuner/src/musubi_tuner/hunyuan_model/attention.py`

#### Step 2.1: Add CuTE Import (after line 22)

```python
try:
    from flash_attn.cute import flash_attn_func as cute_flash_attn_func
    from flash_attn.cute import flash_attn_varlen_func as cute_flash_attn_varlen_func
    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
```

#### Step 2.2: Add to MEMORY_LAYOUT dict (after line 70)

```python
    "cute": (
        lambda x: x,  # No transpose needed - already [B, L, H, D]
        lambda x: x,
    ),
    "cute_varlen": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
```

#### Step 2.3: Add CuTE case in attention() function (after flash_fixlen block, ~line 211)

```python
    elif mode == "cute":
        if split_attn:
            x = []
            for i in range(len(q)):
                out, _ = cute_flash_attn_func(q[i], k[i], v[i], dropout_p=0.0, causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(out)
            del q, k, v
        else:
            out, _ = cute_flash_attn_func(q, k, v, dropout_p=0.0, causal=causal)
            x = out
            del q, k, v

    elif mode == "cute_varlen":
        out, _ = cute_flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        del q, k, v
        x = out.view(batch_size, max_seqlen_q, out.shape[-2], out.shape[-1])
```

---

### Phase 3: Training Script CLI Flags

#### Files to Modify:
- `src/musubi_tuner/hv_train_network.py`
- `src/musubi_tuner/wan_train_network.py`
- `src/musubi_tuner/qwen_image_train_network.py`
- `src/musubi_tuner/qwen_image_train.py`
- `src/musubi_tuner/hv_train.py`

#### Step 3.1: Add argparse flag (in each training script's argument parser)

```python
parser.add_argument(
    "--cute",
    action="store_true",
    help="Use CuTE attention backend (optimized for Blackwell/Hopper GPUs, requires flash_attn.cute)"
)
```

#### Step 3.2: Add attn_mode selection (in the attn_mode selection block)

```python
elif args.cute:
    attn_mode = "cute"
```

#### Step 3.3: Update error message

```python
raise ValueError(
    "either --sdpa, --flash-attn, --flash3, --cute, --sage-attn or --xformers must be specified"
)
```

---

### Phase 4: Auto-Detection (Optional Enhancement)

**File:** `src/blissful_tuner/blissful_core.py` or new `src/musubi_tuner/utils/attention_utils.py`

```python
def get_optimal_attn_mode() -> str:
    """Auto-select best attention backend for current GPU."""
    import torch

    if not torch.cuda.is_available():
        return "torch"

    capability = torch.cuda.get_device_capability()[0]

    # Check CuTE availability
    try:
        from flash_attn.cute import flash_attn_func
        cute_available = True
    except ImportError:
        cute_available = False

    # Check FA3 availability
    try:
        import flash_attn_interface
        fa3_available = True
    except ImportError:
        fa3_available = False

    # Check FA2 availability
    try:
        import flash_attn
        fa2_available = True
    except ImportError:
        fa2_available = False

    # Selection logic
    if capability >= 10 and cute_available:  # Blackwell (SM100+)
        return "cute"
    elif capability >= 9 and fa3_available:  # Hopper (SM90)
        return "flash3"
    elif fa2_available:
        return "flash"
    else:
        return "torch"


def check_cute_compatibility(head_dim: int, dtype: torch.dtype) -> bool:
    """Check if CuTE is compatible with given parameters."""
    supported_head_dims = [64, 96, 128]
    supported_dtypes = [torch.float16, torch.bfloat16]
    return head_dim in supported_head_dims and dtype in supported_dtypes
```

---

## 5. Expected Performance

### Benchmark Comparison (B300)

| Sequence Length | FA2 | CuTE | Improvement |
|-----------------|-----|------|-------------|
| 1024 | 0.37ms | 0.25ms | 32% faster |
| 2048 | 1.16ms | 0.45ms | 61% faster |
| 4096 | 3.98ms | 1.10ms | 72% faster |

### Training Speedup Estimate

| Model | Current | With CuTE | Notes |
|-------|---------|-----------|-------|
| WAN 2.2 14B | Baseline | ~15-25% faster | Attention-bound |
| Qwen Image | Baseline | ~15-25% faster | Attention-bound |

---

## 6. Feature Comparison

| Feature | FA2 | CuTE | Notes |
|---------|-----|------|-------|
| BF16 | ✅ | ✅ | |
| FP16 | ✅ | ✅ | |
| Causal | ✅ | ✅ | |
| Non-causal | ✅ | ✅ | |
| Sliding window | ✅ | ✅ | |
| Varlen forward | ✅ | ✅ | |
| **Varlen backward** | ✅ | ✅ | **Works on SM103!** |
| Deterministic | ✅ | ✅ | SM100+ only |
| GQA/MQA | ✅ | ✅ | |
| Dropout | ✅ | ❌ | CuTE limitation |
| Block sparsity bwd | ✅ | SM90 only | |

**Note:** Attention dropout is rarely used in diffusion model training, so this limitation is acceptable.

---

## 7. Testing Plan

### Unit Tests

```bash
# Test WAN with CuTE
cd /root/blissful-tuner
python -c "
from src.musubi_tuner.wan.modules.attention import flash_attention, CUTE_AVAILABLE
import torch
print(f'CuTE available: {CUTE_AVAILABLE}')

if CUTE_AVAILABLE:
    q = torch.randn(2, 1024, 40, 128, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(2, 1024, 40, 128, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(2, 1024, 40, 128, device='cuda', dtype=torch.bfloat16)
    out = flash_attention([q, k, v], attn_mode='cute')
    print(f'Output shape: {out.shape}')
    print('CuTE attention test PASSED')
"
```

### Integration Tests

```bash
# WAN 2.2 LoRA training with CuTE
python wan_train_network.py \
    --task t2v-A14B \
    --cute \
    --mixed_precision bf16 \
    --network_module musubi_tuner.networks.lora_wan \
    --network_dim 64 \
    --max_train_steps 100 \
    ...

# Qwen Image training with CuTE
python qwen_image_train_network.py \
    --cute \
    --mixed_precision bf16 \
    --max_train_steps 100 \
    ...
```

### Backward Pass Verification

```python
# Critical: Verify gradients flow correctly
import torch
from flash_attn.cute import flash_attn_func

q = torch.randn(2, 1024, 16, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)

out, _ = flash_attn_func(q, k, v)
loss = out.sum()
loss.backward()

assert q.grad is not None, "Q gradient missing"
assert k.grad is not None, "K gradient missing"
assert v.grad is not None, "V gradient missing"
print("Backward pass verification PASSED")
```

---

## 8. Rollback Plan

If issues occur, users can simply use existing backends:

```bash
# Instead of --cute, use:
--flash_attn    # FA2
--flash3        # FA3
--sdpa          # PyTorch SDPA
--xformers      # xformers
```

No code changes needed for rollback - just different CLI flag.

---

## 9. Documentation Updates

After implementation, update:

1. `/root/blissful-tuner/README.md` - Add CuTE to attention backend options
2. `/root/blissful-tuner/docs/wan.md` - Add CuTE usage example
3. Training config examples - Add `--cute` examples

---

## 10. File Change Summary

| File | Changes | Lines |
|------|---------|-------|
| `wan/modules/attention.py` | Add CuTE import + 2 cases | ~50 |
| `hunyuan_model/attention.py` | Add CuTE import + 2 cases | ~40 |
| `modules/attention.py` | Add CuTE import + 2 cases | ~40 |
| `hv_train_network.py` | Add --cute flag | ~5 |
| `wan_train_network.py` | Add --cute flag | ~5 |
| `qwen_image_train_network.py` | Add --cute flag | ~5 |
| `qwen_image_train.py` | Add --cute flag | ~5 |
| `hv_train.py` | Add --cute flag | ~5 |
| **Total** | | **~155 lines** |

---

## 11. Dependencies

### Required (Already Installed)

```
flash-attention 2.8.3+varlen.sm103  # /root/flash-attention
quack-kernels==0.2.4                # CuTE runtime
nvidia-cutlass-dsl>=4.3.4           # CUTLASS DSL
```

### GPU Requirements

- Hopper (SM 9.0+) or Blackwell (SM 10.0+)
- B300 = SM 10.3 ✅

---

## 12. References

- Flash-Attention CuTE: `/root/flash-attention/flash_attn/cute/`
- Flash-Attention Docs: `/root/FLASH-ATTENTION-DOCS/`
- Build Guide: `/root/FLASH-ATTENTION-DOCS/FLASH_ATTENTION_B300_GUIDE.md`
- Technical Reference: `/root/FLASH-ATTENTION-DOCS/FLASH_ATTENTION_B300_REFERENCE.md`

---

## Appendix A: Quick Implementation Checklist

- [x] Add CuTE import to `wan/modules/attention.py` ✅ (2026-01-16)
- [x] Add `attn_mode="cute"` case to `wan/modules/attention.py` ✅ (2026-01-16)
- [x] Add `attn_mode="cute_varlen"` case to `wan/modules/attention.py` ✅ (2026-01-16)
- [x] Add CuTE import to `hunyuan_model/attention.py` ✅ (2026-01-16)
- [x] Add `mode="cute"` case to `hunyuan_model/attention.py` ✅ (2026-01-16)
- [x] Add MEMORY_LAYOUT entry for cute ✅ (2026-01-16)
- [x] Add `--cute` flag to `hv_train_network.py` ✅ (2026-01-16)
- [x] Add `--cute` flag to `wan_train_network.py` (inherits from hv_train_network) ✅
- [x] Add `--cute` flag to `qwen_image_train_network.py` (inherits from hv_train_network) ✅
- [x] Add `--cute` flag to `qwen_image_train.py` ✅ (2026-01-16)
- [x] Add `--cute` flag to `hv_train.py` ✅ (2026-01-16)
- [ ] Test WAN 2.2 training with CuTE (manual testing needed)
- [ ] Test Qwen Image training with CuTE (manual testing needed)
- [x] Verify backward pass gradients ✅ (2026-01-16)
- [x] Update documentation ✅ (2026-01-16)

---

## Appendix B: Troubleshooting

### "No module named 'quack'"

```bash
pip install 'quack-kernels==0.2.4'
```

### "Unsupported compute capability"

CuTE requires SM 9.0+ (Hopper/Blackwell). Use `--flash_attn` for older GPUs.

### "CuTE returns tuple"

CuTE returns `(out, lse)`. Always unpack: `out, _ = cute_flash_attn_func(...)`

Note: `lse` (log-sum-exp) is typically `None` in normal operation.

### Performance worse than FA2

- Check sequence length (CuTE better at SeqLen >= 1024)
- Check head_dim is 64, 96, or 128
- Verify GPU is Blackwell (SM 10.x)

---

## Appendix C: Varlen Routing Behavior

**Important:** When using `--cute` with HunyuanVideo or other models that use `cu_seqlens_*` for padding masks:

- The `hunyuan_model/attention.py` module automatically routes `cute` → `cute_varlen` when `cu_seqlens_q` is provided
- This ensures correct attention masking for variable-length text sequences
- When `--split_attn` is used OR `cu_seqlens_q` is None, it uses the fixed-length `cute` implementation

This mirrors the behavior of `flash` and `sageattn` modes.

### Qwen-Image cu_seqlens Fix (2026-01-16)

`src/musubi_tuner/qwen_image/qwen_image_model.py:827` was updated to build `cu_seqlens_*` from `txt_seq_lens` and pass them to `hunyuan_attention()`. Previously, the `vl_mask` padding mask was ignored by flash/cute/sageattn backends when `split_attn=false`. Now `--cute` correctly routes to `cute_varlen` and padded tokens don't contaminate attention.

---

## Appendix D: CuTE JIT Cache Configuration

CuTE compiles kernels at runtime using CUTLASS DSL. To avoid recompilation across runs, set:

```bash
export CUTE_DSL_CACHE_DIR=/root/.cache/cute_dsl
```

Add this to your training environment scripts (e.g., `env_*.sh`) for persistent kernel caching.

---

## Appendix E: CuTE API Differences from FA2

**Important:** CuTE has a different function signature than FA2. Key differences:

| Parameter | FA2 | CuTE |
|-----------|-----|------|
| `dropout_p` | Supported | **Not supported** |
| `window_size` | `(-1, -1)` = no window | `(None, None)` = no window |
| Return value | `out` tensor | `(out, lse)` tuple |

**CuTE flash_attn_func signature:**
```python
cute_flash_attn_func(
    q, k, v,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),  # NOT (-1, -1)
    deterministic=False
)
```

**CuTE flash_attn_varlen_func signature:**
```python
cute_flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=...,
    cu_seqlens_k=...,  # Note: cu_seqlens_k, not cu_seqlens_kv
    max_seqlen_q=...,
    max_seqlen_k=...,  # Note: max_seqlen_k, not max_seqlen_kv
    softmax_scale=None,
    causal=False,
    deterministic=False
)
