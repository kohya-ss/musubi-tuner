#!/bin/bash
# =============================================================================
# DLAY Persona LyCORIS LoKr Training Script with ProdigyPlusScheduleFree
# Wan2.2-T2V-A14B
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
CACHE_DIR="/root/DATASETS/DLAY/wan_cache_lokr_prodigy"

# Output settings
OUTPUT_DIR="./output/dlay_lokr_prodigy"
OUTPUT_NAME="dlay_persona_lokr_prodigy"

# LoKr Network hyperparameters
NETWORK_DIM=8          # LoKr rank (4-16 typical)
NETWORK_ALPHA=4        # Usually half of dim
LOKR_FACTOR=-1         # -1 = auto
USE_TUCKER="True"      # Enable Tucker decomposition
DECOMPOSE_BOTH="False" # Decompose both matrices
PRESET="attn-mlp"      # Target WanAttentionBlock

# ProdigyPlusScheduleFree optimizer settings
# NOTE: lr=1.0 is recommended - Prodigy handles adaptive LR automatically
LEARNING_RATE="1.0"
D_COEF="2.0"           # Main tuning knob (0.5-2.0, higher = faster adaptation)
WEIGHT_DECAY="0.01"    # Decoupled weight decay
USE_ORTHOGRAD="True"   # Prevents overfitting on small datasets (269 images)
USE_BIAS_CORRECTION="False"  # False = faster LR adaptation (10x!)
SCHEDULEFREE_C="0"     # 0=original SF, 50-200 for long runs

# Training hyperparameters
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

# =============================================================================
# DERIVED PATHS - Don't edit unless you know what you're doing
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_CONFIG="${SCRIPT_DIR}/dataset_config_dlay_lokr_prodigy.toml"
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

# Check if prodigyplus is installed
if ! python3 -c "from prodigyplus import ProdigyPlusScheduleFree" 2>/dev/null; then
    echo "ERROR: prodigy-plus-schedule-free is not installed!"
    echo "Please install it from /root/prodigy-plus-schedule-free:"
    echo "  cd /root/prodigy-plus-schedule-free && pip install -e ."
    exit 1
fi

echo "All dependencies installed!"

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
# STEP 3: Run LoKr training with ProdigyPlusScheduleFree
# =============================================================================

echo ""
echo "============================================="
echo "Step 3: Starting LyCORIS LoKr training..."
echo "============================================="
echo "Network: LyCORIS LoKr"
echo "Network dim: ${NETWORK_DIM}, alpha: ${NETWORK_ALPHA}"
echo "LoKr settings: factor=${LOKR_FACTOR}, tucker=${USE_TUCKER}"
echo ""
echo "Optimizer: ProdigyPlusScheduleFree"
echo "  d_coef=${D_COEF} (main tuning knob)"
echo "  weight_decay=${WEIGHT_DECAY}"
echo "  use_orthograd=${USE_ORTHOGRAD} (anti-overfitting)"
echo "  use_bias_correction=${USE_BIAS_CORRECTION}"
echo "  lr=${LEARNING_RATE} (Prodigy adapts automatically)"
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
    --network_args "algo=lokr" "preset=${PRESET}" "factor=${LOKR_FACTOR}" "use_tucker=${USE_TUCKER}" "decompose_both=${DECOMPOSE_BOTH}" \
    --optimizer_type prodigyplus.ProdigyPlusScheduleFree \
    --optimizer_args "d_coef=${D_COEF}" "weight_decay=${WEIGHT_DECAY}" "use_orthograd=${USE_ORTHOGRAD}" "use_bias_correction=${USE_BIAS_CORRECTION}" "schedulefree_c=${SCHEDULEFREE_C}" \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler constant \
    --max_grad_norm 0 \
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
echo "LoKr + Prodigy Training complete!"
echo "Output saved to: ${OUTPUT_DIR}/${OUTPUT_NAME}.safetensors"
echo "============================================="
echo ""
echo "ProdigyPlusScheduleFree advantages:"
echo "  - Automatic learning rate adaptation (no LR tuning needed!)"
echo "  - Schedule-free (no warmup/decay scheduling)"
echo "  - OrthoGrad prevents overfitting on small datasets"
echo "  - Tracks d*lr in logs for monitoring adaptation"
echo ""
echo "View training logs with TensorBoard:"
echo "  tensorboard --logdir ${OUTPUT_DIR}/logs"
