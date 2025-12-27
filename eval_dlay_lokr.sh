#!/bin/bash
# =============================================================================
# DLAY LoKr Evaluation Script
# Generate sample images to evaluate training quality
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model paths
MODEL_DIR="/root/models/wan22"
DIT_LOW_NOISE="${MODEL_DIR}/dit/wan2.2_t2v_low_noise_14B_fp16.safetensors"
DIT_HIGH_NOISE="${MODEL_DIR}/dit/wan2.2_t2v_high_noise_14B_fp16.safetensors"
T5_MODEL="${MODEL_DIR}/text_encoder/models_t5_umt5-xxl-enc-bf16.pth"
VAE_MODEL="${MODEL_DIR}/vae/wan_2.1_vae.safetensors"

# LoKr checkpoint to evaluate (change this to test different checkpoints)
LOKR_CHECKPOINT="${1:-/root/output/dlay_lokr_prodigy/dlay_persona_lokr_prodigy-step00000250.safetensors}"

# Output directory
OUTPUT_DIR="/root/output/dlay_lokr_prodigy/eval_samples"

# Generation settings
VIDEO_SIZE="544 960"      # Height Width (standard Wan2.2 size)
VIDEO_LENGTH=1            # 1 frame = image, use 25/45/81 for video
STEPS=30                  # Inference steps
CFG_SCALE=5.0             # Guidance scale
FLOW_SHIFT=5.0            # Flow shift for Wan2.2

# =============================================================================
# PROMPTS
# =============================================================================

# Test prompts - mix of training-style and novel scenarios
PROMPTS=(
    "DLAY man professional headshot, studio lighting, high quality, 4k"
    "DLAY man smiling warmly, natural outdoor lighting, casual pose"
    "DLAY man in a business suit, confident stance, modern office background"
    "DLAY man close-up portrait, cinematic lighting, shallow depth of field"
    "DLAY man walking in a park, sunny day, full body shot"
)

# =============================================================================
# GENERATE SAMPLES
# =============================================================================

echo "============================================="
echo "DLAY LoKr Evaluation"
echo "============================================="
echo ""
echo "Checkpoint: ${LOKR_CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Check if checkpoint exists
if [ ! -f "${LOKR_CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${LOKR_CHECKPOINT}"
    echo ""
    echo "Available checkpoints:"
    ls -la /root/output/dlay_lokr_prodigy/*.safetensors 2>/dev/null || echo "  No checkpoints found"
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Get checkpoint name for output naming
CKPT_NAME=$(basename "${LOKR_CHECKPOINT}" .safetensors)

echo "Generating ${#PROMPTS[@]} sample images..."
echo ""

for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    OUTPUT_FILE="${OUTPUT_DIR}/${CKPT_NAME}_sample_$((i+1)).png"

    echo "[$((i+1))/${#PROMPTS[@]}] Generating: ${PROMPT:0:60}..."

    python "${SCRIPT_DIR}/src/musubi_tuner/wan_generate_video.py" \
        --task t2v-A14B \
        --dit "${DIT_LOW_NOISE}" \
        --dit_high_noise "${DIT_HIGH_NOISE}" \
        --vae "${VAE_MODEL}" \
        --t5 "${T5_MODEL}" \
        --prompt "${PROMPT}" \
        --video_size ${VIDEO_SIZE} \
        --video_length ${VIDEO_LENGTH} \
        --infer_steps ${STEPS} \
        --cfg_scale ${CFG_SCALE} \
        --flow_shift ${FLOW_SHIFT} \
        --lora_weight "${LOKR_CHECKPOINT}" \
        --lora_multiplier 1.0 \
        --flash_attn \
        --save_path "${OUTPUT_FILE}" \
        --seed $((42 + i))

    echo "  Saved: ${OUTPUT_FILE}"
    echo ""
done

echo "============================================="
echo "Evaluation complete!"
echo "============================================="
echo ""
echo "Generated ${#PROMPTS[@]} samples in: ${OUTPUT_DIR}"
echo ""
echo "To view samples:"
echo "  ls -la ${OUTPUT_DIR}"
echo ""
echo "To test a different checkpoint:"
echo "  $0 /path/to/checkpoint.safetensors"
echo ""

# Also generate one WITHOUT the LoKr for comparison
echo "Generating baseline (no LoKr) for comparison..."
BASELINE_FILE="${OUTPUT_DIR}/baseline_no_lokr.png"

python "${SCRIPT_DIR}/src/musubi_tuner/wan_generate_video.py" \
    --task t2v-A14B \
    --dit "${DIT_LOW_NOISE}" \
    --dit_high_noise "${DIT_HIGH_NOISE}" \
    --vae "${VAE_MODEL}" \
    --t5 "${T5_MODEL}" \
    --prompt "DLAY man professional headshot, studio lighting, high quality, 4k" \
    --video_size ${VIDEO_SIZE} \
    --video_length ${VIDEO_LENGTH} \
    --infer_steps ${STEPS} \
    --cfg_scale ${CFG_SCALE} \
    --flow_shift ${FLOW_SHIFT} \
    --flash_attn \
    --save_path "${BASELINE_FILE}" \
    --seed 42

echo "Baseline saved: ${BASELINE_FILE}"
echo ""
echo "Compare the LoKr samples to the baseline to see if DLAY features are being learned."
