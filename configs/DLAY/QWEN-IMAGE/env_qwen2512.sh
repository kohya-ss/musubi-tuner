#!/bin/bash
# =============================================================================
# Runtime environment for Qwen-Image-2512 training (B300 optimized)
# =============================================================================
# Source this file before running training:
#   source configs/DLAY/QWEN-IMAGE/env_qwen2512.sh
# =============================================================================

export PYTHONUNBUFFERED=1

# TorchInductor / Triton (for torch.compile)
export TORCHINDUCTOR_CACHE_DIR="/root/.cache/torch_inductor"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=0

# CuTE JIT cache (flash_attn.cute)
export CUTE_DSL_CACHE_DIR="${CUTE_DSL_CACHE_DIR:-/root/.cache/cute_dsl}"

# Ensure cache directories exist (no-op if already present)
mkdir -p "${CUTE_DSL_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true

# CUDA allocator (PyTorch)
# NOTE:
# - `torch.compile` mode="max-autotune" enables CUDAGraphs by default.
# - PyTorch (as of the B300 CUDA 13.1 nightly used here) can error when combining
#   CUDAGraph trees + `backend:cudaMallocAsync` with:
#     "cudaMallocAsync does not yet support checkPoolLiveAllocations"
# To avoid that, we default to the native allocator when CUDAGraphs are enabled.
if [ "${TORCHINDUCTOR_DISABLE_CUDAGRAPHS}" = "0" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="backend:native,expandable_segments:True,garbage_collection_threshold:0.8"
else
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-backend:cudaMallocAsync,expandable_segments:True,garbage_collection_threshold:0.8}"
fi
# Back-compat (some older scripts reference this name)
export PYTORCH_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"

# CPU thread caps (tune as needed)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-16}"

# Misc
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export BNB_CUDA_VERSION="${BNB_CUDA_VERSION:-}"

echo "Environment configured for Qwen-Image-2512 training on B300"
