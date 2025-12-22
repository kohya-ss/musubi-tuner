#!/bin/bash
# =============================================================================
# DLAY Training Launcher (Config-Based)
# =============================================================================
# This script uses TOML config files instead of command-line arguments
# Edit the config files in ./configs/ to change settings
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load runtime env if present
if [ -f "${SCRIPT_DIR}/env.sh" ]; then
    # shellcheck source=/dev/null
    source "${SCRIPT_DIR}/env.sh"
fi

# =============================================================================
# SELECT TRAINING CONFIGURATION
# =============================================================================
# Available configs:
#   lokr_prodigy  - LyCORIS LoKr + ProdigyPlusScheduleFree (recommended)
#   lokr          - LyCORIS LoKr + AdamW8bit + LoRA+
#   lora          - Standard LoRA + AdamW
#   lora_prodigy  - Standard LoRA + Prodigy (Schedule-Free disabled)

ARG1="${1:-lokr_prodigy}"

# Allow passing an explicit config path as the first arg:
#   ./train_dlay_config.sh ./configs/my_config.toml
if [ -f "${ARG1}" ]; then
    TRAINING_TYPE="custom"
    CONFIG_FILE="${ARG1}"
else
    TRAINING_TYPE="${ARG1}"

    case "${TRAINING_TYPE}" in
        "lokr_prodigy")
            CONFIG_FILE="${SCRIPT_DIR}/configs/DLAY/dlay_lokr_prodigy_training_full.toml"
            ;;
        "lokr")
            CONFIG_FILE="${SCRIPT_DIR}/configs/DLAY/dlay_lokr_training.toml"
            ;;
        "lora")
            CONFIG_FILE="${SCRIPT_DIR}/configs/DLAY/dlay_lora_training.toml"
            ;;
        "lora_prodigy")
            CONFIG_FILE="${SCRIPT_DIR}/configs/DLAY/dlay_lora_prodigy_training.toml"
            ;;
        *)
            echo "Usage: $0 [lokr_prodigy|lokr|lora|lora_prodigy|/path/to/config.toml]"
            echo ""
            echo "Available training configurations:"
            echo "  lokr_prodigy  - LyCORIS LoKr + ProdigyPlusScheduleFree (default)"
            echo "  lokr          - LyCORIS LoKr + AdamW8bit + LoRA+"
            echo "  lora          - Standard LoRA + AdamW"
            echo "  lora_prodigy  - Standard LoRA + Prodigy (Schedule-Free disabled)"
            exit 1
            ;;
    esac
fi

# =============================================================================
# CHECK CONFIG EXISTS
# =============================================================================

if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "============================================="
echo "DLAY Training (Config-Based)"
echo "============================================="
echo ""
echo "Training type: ${TRAINING_TYPE}"
echo "Config file:   ${CONFIG_FILE}"
echo ""

# =============================================================================
# CHECK DEPENDENCIES (SKIPPED)
# =============================================================================
# NOTE: Dependency checks disabled due to bus errors with custom PyTorch build.
# Packages are already installed - if you get import errors during training,
# manually install: pip install lycoris-lora
# For prodigyplus: cd /root/prodigy-plus-schedule-free && pip install -e .
echo "Skipping dependency checks (packages assumed installed)"
echo ""

# =============================================================================
# EXTRACT PATHS FROM CONFIG FOR CACHING
# =============================================================================

# Parse paths from training config
DATASET_CONFIG=$("${SCRIPT_DIR}/venv/bin/python" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('dataset', {}).get('dataset_config', ''))
")

VAE_MODEL=$("${SCRIPT_DIR}/venv/bin/python" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('model', {}).get('vae', ''))
")

T5_MODEL=$("${SCRIPT_DIR}/venv/bin/python" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('model', {}).get('t5', ''))
")

OUTPUT_DIR=$("${SCRIPT_DIR}/venv/bin/python" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('output', {}).get('output_dir', './output'))
")

if [ -z "${DATASET_CONFIG}" ] || [ -z "${VAE_MODEL}" ] || [ -z "${T5_MODEL}" ]; then
    echo "ERROR: Could not parse required paths from config file"
    exit 1
fi

echo "Parsed from config:"
echo "  Dataset: ${DATASET_CONFIG}"
echo "  VAE:     ${VAE_MODEL}"
echo "  T5:      ${T5_MODEL}"
echo "  Output:  ${OUTPUT_DIR}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# STEP 1: Cache VAE latents
# =============================================================================

echo "============================================="
echo "Step 1: Caching VAE latents..."
echo "============================================="

"${SCRIPT_DIR}/venv/bin/python" "${SCRIPT_DIR}/src/musubi_tuner/wan_cache_latents.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE_MODEL}" \
    --batch_size 4 \
    --skip_existing

echo "VAE latent caching complete!"

# =============================================================================
# STEP 2: Cache text encoder outputs
# =============================================================================

echo ""
echo "============================================="
echo "Step 2: Caching text encoder outputs..."
echo "============================================="

"${SCRIPT_DIR}/venv/bin/python" "${SCRIPT_DIR}/src/musubi_tuner/wan_cache_text_encoder_outputs.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --t5 "${T5_MODEL}" \
    --batch_size 8 \
    --skip_existing

echo "Text encoder caching complete!"

# =============================================================================
# STEP 3: Run training with config file
# =============================================================================

echo ""
echo "============================================="
echo "Step 3: Starting training..."
echo "============================================="

"${SCRIPT_DIR}/venv/bin/accelerate" launch --num_cpu_threads_per_process 1 --mixed_precision fp16 \
    "${SCRIPT_DIR}/src/musubi_tuner/wan_train_network.py" \
    --config_file "${CONFIG_FILE}"

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
echo "Merge checkpoints with EMA (optional):"
echo "  ./merge_CRLN_ema.sh"
