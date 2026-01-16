# CuTE Attention Training Guide

## Overview

CuTE (“CUDA Templates”) is an attention backend exposed via `flash_attn.cute`. On Hopper/Blackwell GPUs it can be noticeably faster than FA2/SDPA, especially for long sequence lengths.

In blissful-tuner, CuTE is enabled with `--cute` (or `cute = true` in a TOML config), and is supported in:
- WAN 2.1/2.2 (`src/musubi_tuner/wan/modules/attention.py`)
- HunyuanVideo + Qwen-Image (`src/musubi_tuner/hunyuan_model/attention.py`)

## Requirements

- GPU: Hopper (SM 9.0+) or Blackwell (SM 10.0+). B300 = SM 10.3.
- `flash-attention` with CuTE enabled (example in this repo: `2.8.3+varlen.sm103`)
- CuTE runtime deps:
  - `quack-kernels==0.2.4` (includes CUTLASS DSL runtime)

Quick checks:
```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
python -c "import flash_attn; print(flash_attn.__version__)"
python -c "from flash_attn.cute import flash_attn_func; print('CuTE OK')"
```

## Recommended Environment

For Qwen-Image-2512 training on B300, source:
```bash
source configs/DLAY/QWEN-IMAGE/env_qwen2512.sh
```

This configures:
- `TORCHINDUCTOR_CACHE_DIR` (torch.compile cache)
- `CUTE_DSL_CACHE_DIR` (CuTE JIT cache)
- allocator settings that avoid known CUDAGraph + `cudaMallocAsync` issues on the B300 torch nightly

## How To Enable CuTE

### Option A: Use the repo launcher (DLAY/Qwen-Image example)

```bash
./configs/DLAY/QWEN-IMAGE/train_dlay_qwen2512.sh cute
```

### Option B: CLI flags (any supported training entrypoint)

Use exactly one attention backend flag:
```bash
--sdpa | --flash-attn | --flash3 | --sage-attn | --xformers | --cute
```

## Masking & Variable-Length Text (Important)

Some attention backends (FA2/CuTE/Sage) do **not** consume an explicit padding mask tensor. For correctness with padded / variable-length text:

- `src/musubi_tuner/hunyuan_model/attention.py` auto-routes:
  - `cute` → `cute_varlen` when `cu_seqlens_q` is provided (and `--split_attn` is not used)
- `src/musubi_tuner/qwen_image/qwen_image_model.py` builds and passes `cu_seqlens_*` from `txt_seq_lens` when using CuTE/FA2/Sage, ensuring padding tokens don’t contaminate attention.

If you see identity “drift” or inconsistent training when `batch_size > 1`, verify you’re using a varlen-capable backend (`--cute` / `--flash-attn` / `--sage-attn`) and that `txt_seq_lens` is present in the batch (it is for Qwen-Image caching).

## torch.compile Notes

- CuTE kernels JIT-compile independently from `torch.compile`.
- First run may be slower due to:
  - Inductor autotuning/compiles
  - CuTE kernel JIT compilation

Recommended starting point for training:
- `compile = true`
- `compile_mode = "max-autotune-no-cudagraphs"` (stable)

If using `compile_mode="max-autotune"` (CUDAGraphs enabled), keep the allocator on the native backend (the provided env script does this).

## Mask-Weighted Loss + Caching Reminder

Mask-weighted loss requires masks baked into the latent cache. If you change `mask_directory`, you must cache into a **fresh** `cache_directory` (or remove `--skip_existing`).

See: `docs/MASKED_LOSS_TRAINING_GUIDE.md`

## Troubleshooting

- `ImportError: CuTE not available`:
  - Install CuTE deps: `pip install 'quack-kernels==0.2.4'`
  - Confirm `from flash_attn.cute import flash_attn_func` works
- Performance worse than FA2:
  - CuTE tends to win at longer sequence lengths (often >= 1024)
  - Ensure dtype is bf16/fp16 and head_dim is supported (commonly 128 in these models)
- Want to rollback:
  - Switch to `--flash-attn` (FA2) or `--sdpa` (PyTorch)

