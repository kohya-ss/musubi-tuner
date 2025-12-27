#!/bin/bash
# =============================================================================
# DLAY Post-Hoc EMA Checkpoint Merger
# Merges multiple training checkpoints into a more stable final model
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Which training output to merge
# Options: "lora", "lokr", "lokr_prodigy"
# Can also be passed as first argument: ./merge_dlay_ema.sh lokr
TRAINING_TYPE="${1:-lokr_prodigy}"

# EMA Settings
# sigma_rel: Power Function EMA (recommended)
#   0.15 = emphasize later epochs (avoid overfitting)
#   0.20 = balanced (general use)
#   0.25 = emphasize earlier epochs (early convergence)
SIGMA_REL="0.2"

# Alternative: use fixed beta instead of sigma_rel
# Uncomment to use beta mode instead
# USE_BETA="True"
# BETA="0.9"

# =============================================================================
# DERIVED PATHS
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${TRAINING_TYPE}" in
    "lora")
        OUTPUT_DIR="${SCRIPT_DIR}/output/dlay_lora"
        OUTPUT_NAME="dlay_persona_lora"
        ;;
    "lokr")
        OUTPUT_DIR="${SCRIPT_DIR}/output/dlay_lokr"
        OUTPUT_NAME="dlay_persona_lokr"
        ;;
    "lokr_prodigy")
        OUTPUT_DIR="${SCRIPT_DIR}/output/dlay_lokr_prodigy"
        OUTPUT_NAME="dlay_persona_lokr_prodigy"
        ;;
    *)
        echo "ERROR: Unknown training type: ${TRAINING_TYPE}"
        echo "Valid options: lora, lokr, lokr_prodigy"
        exit 1
        ;;
esac

EMA_OUTPUT="${OUTPUT_DIR}/${OUTPUT_NAME}_ema.safetensors"

# =============================================================================
# FIND CHECKPOINTS
# =============================================================================

echo "============================================="
echo "DLAY Post-Hoc EMA Checkpoint Merger"
echo "============================================="
echo ""
echo "Training type: ${TRAINING_TYPE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Find all epoch checkpoints (excluding the final one without epoch suffix and any existing EMA)
CHECKPOINTS=$(find "${OUTPUT_DIR}" -name "${OUTPUT_NAME}*.safetensors" \
    ! -name "*_ema.safetensors" \
    ! -name "${OUTPUT_NAME}.safetensors" \
    -type f | sort)

# Count checkpoints
CHECKPOINT_COUNT=$(echo "${CHECKPOINTS}" | grep -c . || true)

if [ "${CHECKPOINT_COUNT}" -lt 2 ]; then
    echo "ERROR: Need at least 2 checkpoints to merge."
    echo "Found ${CHECKPOINT_COUNT} checkpoint(s) in ${OUTPUT_DIR}"
    echo ""
    echo "Make sure training has completed with --save_every_n_epochs set."
    exit 1
fi

echo "Found ${CHECKPOINT_COUNT} checkpoints to merge:"
echo "${CHECKPOINTS}" | while read -r ckpt; do
    echo "  - $(basename "${ckpt}")"
done
echo ""

# =============================================================================
# MERGE CHECKPOINTS
# =============================================================================

echo "============================================="
echo "Merging with Post-Hoc EMA..."
echo "============================================="

if [ "${USE_BETA}" = "True" ]; then
    echo "Mode: Fixed beta = ${BETA}"
    echo ""

    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/src/musubi_tuner/lora_post_hoc_ema.py" \
        ${CHECKPOINTS} \
        --output_file "${EMA_OUTPUT}" \
        --beta "${BETA}"
else
    echo "Mode: Power Function EMA (sigma_rel = ${SIGMA_REL})"
    echo ""

    # shellcheck disable=SC2086
    python "${SCRIPT_DIR}/src/musubi_tuner/lora_post_hoc_ema.py" \
        ${CHECKPOINTS} \
        --output_file "${EMA_OUTPUT}" \
        --sigma_rel "${SIGMA_REL}"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================="
echo "EMA Merge Complete!"
echo "============================================="
echo ""
echo "Original checkpoints: ${CHECKPOINT_COUNT}"
echo "Merged output: ${EMA_OUTPUT}"
echo ""

# Show file sizes for comparison
echo "File sizes:"
echo "${CHECKPOINTS}" | tail -1 | xargs -I {} sh -c 'echo "  Last checkpoint: $(du -h "{}" | cut -f1)"'
echo "  EMA merged:       $(du -h "${EMA_OUTPUT}" | cut -f1)"
echo ""

echo "Benefits of EMA-merged model:"
echo "  - More stable than any single checkpoint"
echo "  - Reduced overfitting on small datasets"
echo "  - Smoother interpolation between training stages"
echo ""
echo "Recommended: Compare generations from:"
echo "  1. Final epoch checkpoint"
echo "  2. EMA merged checkpoint"
echo "  to see which produces better results for your use case."
