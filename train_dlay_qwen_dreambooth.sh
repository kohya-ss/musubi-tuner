#!/bin/bash
# =============================================================================
# DLAY Qwen-Image DreamBooth Training Launcher (Masked)
# =============================================================================
# Automated workflow for Qwen-Image persona fine-tuning with mask-weighted loss
#
# Steps:
#   1. Cache VAE latents (with masks baked in)
#   2. Cache text encoder outputs
#   3. Run DreamBooth training
#
# Usage:
#   ./train_dlay_qwen_dreambooth.sh                    # Use default config
#   ./train_dlay_qwen_dreambooth.sh /path/to/config.toml  # Use custom config
#
# Default config: DLAY_DreamBooth_CosineMinLR_5e5_B300.toml
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default training config (masked DreamBooth)
DEFAULT_CONFIG="/root/blissful-tuner/configs/DLAY/DLAY_DreamBooth_CosineMinLR_5e5_B300.toml"

# Model paths
VAE_MODEL="/root/models/qwen-image/qwen_train_vae.safetensors"
TEXT_ENCODER="/root/models/qwen-image/qwen_2.5_vl_7b_bf16.safetensors"

# Caching settings
CACHE_BATCH_SIZE=8
CACHE_DEVICE="cuda"

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

if [ -n "$1" ] && [ -f "$1" ]; then
    CONFIG_FILE="$1"
else
    CONFIG_FILE="${DEFAULT_CONFIG}"
fi

# =============================================================================
# EXTRACT DATASET CONFIG FROM TRAINING CONFIG
# =============================================================================

echo "============================================="
echo "DLAY Qwen-Image DreamBooth Training (Masked)"
echo "============================================="
echo ""

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "Training config: ${CONFIG_FILE}"
echo ""

# Parse dataset_config path from training TOML
DATASET_CONFIG=$(python3 -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('dataset_config', ''))
")

if [ -z "${DATASET_CONFIG}" ]; then
    echo "ERROR: Could not parse 'dataset_config' from training config"
    echo "Make sure your training TOML has: dataset_config = \"/path/to/dataset.toml\""
    exit 1
fi

if [ ! -f "${DATASET_CONFIG}" ]; then
    echo "ERROR: Dataset config not found: ${DATASET_CONFIG}"
    exit 1
fi

# Parse output_dir for info
OUTPUT_DIR=$(python3 -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('output_dir', './output'))
")

# Check if mask loss is enabled
USE_MASK_LOSS=$(python3 -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print('true' if config.get('use_mask_loss', False) else 'false')
")

echo "Parsed from config:"
echo "  Dataset config: ${DATASET_CONFIG}"
echo "  Output dir:     ${OUTPUT_DIR}"
echo "  VAE:            ${VAE_MODEL}"
echo "  Text encoder:   ${TEXT_ENCODER}"
echo "  Mask loss:      ${USE_MASK_LOSS}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# VERIFY MASK SETUP (if mask loss enabled)
# =============================================================================

if [ "${USE_MASK_LOSS}" = "true" ]; then
    echo "============================================="
    echo "Verifying mask configuration..."
    echo "============================================="

    # Check if mask_directory is set in dataset config
    MASK_DIR=$(python3 -c "
import tomllib
with open('${DATASET_CONFIG}', 'rb') as f:
    config = tomllib.load(f)
for ds in config.get('datasets', []):
    if 'mask_directory' in ds:
        print(ds['mask_directory'])
        break
")

    if [ -z "${MASK_DIR}" ]; then
        echo "WARNING: use_mask_loss=true but no mask_directory found in dataset config!"
        echo "Training will fail at step 0. Add mask_directory under [[datasets]]."
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "  Mask directory: ${MASK_DIR}"
        MASK_COUNT=$(find "${MASK_DIR}" -name "*.png" -o -name "*.jpg" 2>/dev/null | wc -l)
        echo "  Mask files found: ${MASK_COUNT}"
        echo ""
    fi
fi

# =============================================================================
# STEP 1: Cache VAE Latents (with masks)
# =============================================================================

echo "============================================="
echo "Step 1/3: Caching VAE latents..."
echo "============================================="
echo ""

if [ "${USE_MASK_LOSS}" = "true" ]; then
    echo "NOTE: Masks will be baked into latent cache."
    echo "      If you change masks, delete cache and re-run."
    echo ""
fi

python "${SCRIPT_DIR}/src/musubi_tuner/qwen_image_cache_latents.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE_MODEL}" \
    --device "${CACHE_DEVICE}" \
    --batch_size "${CACHE_BATCH_SIZE}" \
    --skip_existing

echo ""
echo "VAE latent caching complete!"

# =============================================================================
# STEP 2: Cache Text Encoder Outputs
# =============================================================================

echo ""
echo "============================================="
echo "Step 2/3: Caching text encoder outputs..."
echo "============================================="
echo ""

python "${SCRIPT_DIR}/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --text_encoder "${TEXT_ENCODER}" \
    --device "${CACHE_DEVICE}" \
    --batch_size "${CACHE_BATCH_SIZE}" \
    --skip_existing

echo ""
echo "Text encoder caching complete!"

# =============================================================================
# VERIFY MASKS IN CACHE (if mask loss enabled)
# =============================================================================

if [ "${USE_MASK_LOSS}" = "true" ]; then
    echo ""
    echo "============================================="
    echo "Verifying masks in cache..."
    echo "============================================="

    # Get cache directory from dataset config
    CACHE_DIR=$(python3 -c "
import tomllib
with open('${DATASET_CONFIG}', 'rb') as f:
    config = tomllib.load(f)
for ds in config.get('datasets', []):
    if 'cache_directory' in ds:
        print(ds['cache_directory'])
        break
")

    if [ -n "${CACHE_DIR}" ] && [ -d "${CACHE_DIR}" ]; then
        # Check first cache file for mask_weights key
        CACHE_CHECK=$(python3 -c "
from safetensors.torch import load_file
import glob
import os
cache_files = glob.glob('${CACHE_DIR}/*_qi.safetensors')
if cache_files:
    keys = load_file(cache_files[0]).keys()
    mask_keys = [k for k in keys if k.startswith('mask_weights')]
    if mask_keys:
        print(f'OK: Found {mask_keys[0]} in cache')
    else:
        print('ERROR: No mask_weights found in cache!')
else:
    print('ERROR: No cache files found')
" 2>&1)
        echo "  ${CACHE_CHECK}"
    fi
    echo ""
fi

# =============================================================================
# STEP 3: Run DreamBooth Training
# =============================================================================

echo "============================================="
echo "Step 3/3: Starting DreamBooth training..."
echo "============================================="
echo ""

python "${SCRIPT_DIR}/src/musubi_tuner/qwen_image_train.py" \
    --config_file "${CONFIG_FILE}"

# =============================================================================
# DONE
# =============================================================================

echo ""
echo "============================================="
echo "Training complete!"
echo "============================================="
echo ""
echo "Output saved to: ${OUTPUT_DIR}"
echo ""
echo "View training logs with TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs"
echo ""
echo "Sample images saved to: ${OUTPUT_DIR}/samples/"
echo ""
