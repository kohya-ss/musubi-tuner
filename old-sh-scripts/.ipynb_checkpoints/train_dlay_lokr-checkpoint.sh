#!/bin/bash
# =============================================================================
# DLAY Persona LyCORIS LoKr Training Script for Wan2.2-T2V-A14B
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Edit these paths to match your setup
# =============================================================================

# Model paths (run download_wan22_models.sh first to download these)
MODEL_DIR="/root/models/wan22"
DIT_LOW_NOISE="${MODEL_DIR}/dit/wan2.2_t2v_14B_low_noise_fp16.safetensors"
DIT_HIGH_NOISE="${MODEL_DIR}/dit/wan2.2_t2v_14B_high_noise_fp16.safetensors"
T5_MODEL="${MODEL_DIR}/text_encoder/models_t5_umt5-xxl-enc-bf16.pth"
VAE_MODEL="${MODEL_DIR}/vae/wan_2.1_vae.safetensors"

# Dataset paths
IMAGE_DIR="/root/DATASETS/DLAY/wan"
MASK_DIR="/root/DATASETS/DLAY/mask_weighted"
CACHE_DIR="/root/DATASETS/DLAY/wan_cache_lokr"

# Output settings
OUTPUT_DIR="./output/dlay_lokr"
OUTPUT_NAME="dlay_persona_lokr"

# LoKr Network hyperparameters
# LoKr uses lower dims than LoRA due to Kronecker product efficiency
NETWORK_DIM=8          # LoKr rank (lower than LoRA, 4-16 typical)
NETWORK_ALPHA=4        # Usually half of dim
LOKR_FACTOR=-1         # -1 = auto, or specify integer factor
USE_TUCKER="True"      # Enable Tucker decomposition for better quality
DECOMPOSE_BOTH="False" # Decompose both matrices (more params but better)
PRESET="attn-mlp"      # Target WanAttentionBlock

# Training hyperparameters
LEARNING_RATE="2e-4"   # LoKr can often use slightly higher LR
BATCH_SIZE=2
GRADIENT_ACCUMULATION=2
MAX_EPOCHS=24
SAVE_EVERY_N_EPOCHS=4

# Mask-weighted loss settings
MASK_LOSS_SCALE=1.5
MASK_MIN_WEIGHT=0.1

# Misc
SEED=42
RESOLUTION_W=1024
RESOLUTION_H=1024

# =============================================================================
# ADVANCED OPTIMIZATIONS (Blackwell B300 optimized)
# =============================================================================

# torch.compile settings (10-25% speedup on Blackwell)
USE_TORCH_COMPILE="True"
COMPILE_MODE="default"           # default, reduce-overhead, max-autotune-no-cudagraphs
COMPILE_CACHE_SIZE=32

# Hardware optimizations
USE_TF32="True"                  # TF32 for Ampere+ (safe for training)
USE_CUDNN_BENCHMARK="True"       # cuDNN autotuning

# Timestep bucketing for small datasets (uniform sampling)
NUM_TIMESTEP_BUCKETS=5           # 4-10 recommended for small datasets

# LoRA+ (faster convergence - increases LR for LoRA-B/UP side)
LORAPLUS_LR_RATIO=4              # Original paper recommends 16, 4 is a good start

# =============================================================================
# DERIVED PATHS - Don't edit unless you know what you're doing
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_CONFIG="${SCRIPT_DIR}/dataset_config_dlay_lokr.toml"
SAMPLE_PROMPTS="${SCRIPT_DIR}/sample_prompts_dlay.txt"

# =============================================================================
# STEP 0: Check dependencies and create config files
# =============================================================================

echo "============================================="
echo "Step 0: Checking dependencies and creating configs..."
echo "============================================="

# Check if lycoris-lora is installed
if ! python3 -c "import lycoris" 2>/dev/null; then
    echo "Installing lycoris-lora..."
    pip install lycoris-lora
fi

# Create dataset config TOML
cat > "${DATASET_CONFIG}" << EOF
[general]
resolution = [${RESOLUTION_W}, ${RESOLUTION_H}]
caption_extension = ".txt"
batch_size = ${BATCH_SIZE}
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "${IMAGE_DIR}"
cache_directory = "${CACHE_DIR}"
mask_directory = "${MASK_DIR}"
num_repeats = 1
EOF

echo "Created dataset config: ${DATASET_CONFIG}"

# Create sample prompts file (reuse if exists)
if [ ! -f "${SAMPLE_PROMPTS}" ]; then
cat > "${SAMPLE_PROMPTS}" << 'EOF'
DLAY man professional headshot, studio lighting, high quality
DLAY man walking outdoors, natural lighting, full body shot
DLAY man close-up portrait, cinematic lighting
DLAY man casual pose, indoor setting, soft lighting
DLAY man standing confidently, urban background
EOF
echo "Created sample prompts: ${SAMPLE_PROMPTS}"
else
    echo "Using existing sample prompts: ${SAMPLE_PROMPTS}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

# =============================================================================
# STEP 1: Cache VAE latents
# =============================================================================

echo ""
echo "============================================="
echo "Step 1: Caching VAE latents..."
echo "============================================="

python "${SCRIPT_DIR}/src/musubi_tuner/wan_cache_latents.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE_MODEL}" \
    --task t2v-A14B \
    --batch_size 4

echo "VAE latent caching complete!"

# =============================================================================
# STEP 2: Cache text encoder outputs
# =============================================================================

echo ""
echo "============================================="
echo "Step 2: Caching text encoder outputs..."
echo "============================================="

python "${SCRIPT_DIR}/src/musubi_tuner/wan_cache_text_encoder_outputs.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --t5 "${T5_MODEL}" \
    --batch_size 8

echo "Text encoder caching complete!"

# =============================================================================
# STEP 3: Run LoKr training
# =============================================================================

echo ""
echo "============================================="
echo "Step 3: Starting LyCORIS LoKr training..."
echo "============================================="
echo "Network: LyCORIS LoKr"
echo "Network dim: ${NETWORK_DIM}, alpha: ${NETWORK_ALPHA}"
echo "LoKr settings: factor=${LOKR_FACTOR}, tucker=${USE_TUCKER}, decompose_both=${DECOMPOSE_BOTH}"
echo "Preset: ${PRESET}"
echo "Learning rate: ${LEARNING_RATE}"
echo "LoRA+ ratio: ${LORAPLUS_LR_RATIO}x (faster convergence)"
echo ""
echo "Advanced Optimizations:"
echo "  torch.compile=${USE_TORCH_COMPILE} (mode=${COMPILE_MODE})"
echo "  TF32=${USE_TF32}, cuDNN benchmark=${USE_CUDNN_BENCHMARK}"
echo "  Timestep buckets=${NUM_TIMESTEP_BUCKETS}"
echo ""
echo "Batch size: ${BATCH_SIZE} x ${GRADIENT_ACCUMULATION} accumulation"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"
echo "============================================="

# Build compile arguments conditionally
COMPILE_ARGS=""
if [ "${USE_TORCH_COMPILE}" = "True" ]; then
    COMPILE_ARGS="--compile --compile_mode ${COMPILE_MODE} --compile_cache_size_limit ${COMPILE_CACHE_SIZE}"
fi

HARDWARE_ARGS=""
if [ "${USE_TF32}" = "True" ]; then
    HARDWARE_ARGS="${HARDWARE_ARGS} --cuda_allow_tf32"
fi
if [ "${USE_CUDNN_BENCHMARK}" = "True" ]; then
    HARDWARE_ARGS="${HARDWARE_ARGS} --cuda_cudnn_benchmark"
fi

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    "${SCRIPT_DIR}/src/musubi_tuner/wan_train_network.py" \
    --task t2v-A14B \
    --dit "${DIT_LOW_NOISE}" \
    --dit_high_noise "${DIT_HIGH_NOISE}" \
    --t5 "${T5_MODEL}" \
    --vae "${VAE_MODEL}" \
    --dataset_config "${DATASET_CONFIG}" \
    --sdpa \
    --mixed_precision bf16 \
    --timestep_boundary 875 \
    --network_module networks.lycoris \
    --network_dim ${NETWORK_DIM} \
    --network_alpha ${NETWORK_ALPHA} \
    --network_args "algo=lokr" "preset=${PRESET}" "factor=${LOKR_FACTOR}" "use_tucker=${USE_TUCKER}" "decompose_both=${DECOMPOSE_BOTH}" "loraplus_lr_ratio=${LORAPLUS_LR_RATIO}" \
    --optimizer_type adamw8bit \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler cosine \
    --lr_warmup_steps 100 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION} \
    --max_train_epochs ${MAX_EPOCHS} \
    --save_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --timestep_sampling shift \
    --discrete_flow_shift 3.0 \
    --num_timestep_buckets ${NUM_TIMESTEP_BUCKETS} \
    --use_mask_loss \
    --mask_loss_scale ${MASK_LOSS_SCALE} \
    --mask_min_weight ${MASK_MIN_WEIGHT} \
    --rope_func comfy \
    --seed ${SEED} \
    --sample_prompts "${SAMPLE_PROMPTS}" \
    --sample_every_n_epochs ${SAVE_EVERY_N_EPOCHS} \
    --sample_at_first \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --log_prefix "${OUTPUT_NAME}_" \
    --output_dir "${OUTPUT_DIR}" \
    --output_name "${OUTPUT_NAME}" \
    ${COMPILE_ARGS} \
    ${HARDWARE_ARGS}

echo ""
echo "============================================="
echo "LoKr Training complete!"
echo "Output saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
echo "============================================="
echo ""
echo "LoKr advantages over LoRA:"
echo "  - Smaller file size (~50-70% fewer parameters)"
echo "  - Often comparable quality"
echo "  - More efficient Kronecker product decomposition"
echo ""
echo "View training logs with TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs"
