#!/bin/bash
# ============================================================================
# Flux2 dev LoRA training on Intel Arc (XPU) with H2D-only block swap
# ----------------------------------------------------------------------------
# Validated end-to-end on an Arc Pro B70 (32GB): 540 steps at ~36 s/it (1024px,
# bucketed), output LoRA confirmed working in ComfyUI. The classic swap path
# (drop --block_swap_h2d_only) runs the same recipe at ~47 s/it.
#
# Recipe: dim 64 / alpha 64 (scale 1.0), LR 5e-5 constant, AdamW8bit,
# timestep_sampling flux2_shift, discrete_flow_shift 1.0, weighting none,
# mixed bf16, gradient checkpointing + cpu offload, blocks_to_swap 40.
#
# In-training sampling is intentionally OFF (broken on XPU — produces noise);
# evaluate checkpoints in ComfyUI instead.
#
# Usage: ./train_lora_flux2_xpu_h2d.sh /path/to/dataset_config.toml [--training-only]
#   (see flux2_dataset_config.example.toml; --training-only skips re-caching)
# ============================================================================

set -e

DATASET_CONFIG="${1:?usage: $0 /path/to/dataset_config.toml [--training-only]}"
TRAINING_ONLY=false
[ "$2" = "--training-only" ] && TRAINING_ONLY=true

# Adjust these to your setup
MUSUBI_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="/path/to/FLUX.2-dev"
VAE="${MODEL_DIR}/ae.safetensors"
DIT="${MODEL_DIR}/flux2-dev.safetensors"
TEXT_ENCODER="${MODEL_DIR}/text_encoder/model-00001-of-00010.safetensors"
OUTPUT_DIR="${OUTPUT_DIR:-./output/flux2-lora-xpu}"
OUTPUT_NAME="${OUTPUT_NAME:-flux2-lora-xpu}"
MODEL_VERSION="dev"

[ ! -f "$DATASET_CONFIG" ] && echo "Error: $DATASET_CONFIG not found" && exit 1
[ ! -f "$VAE" ] && echo "Error: VAE not found at $VAE" && exit 1
[ ! -f "$DIT" ] && echo "Error: DIT not found at $DIT" && exit 1

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/logs"
source "$MUSUBI_DIR/venv/bin/activate"
cd "$MUSUBI_DIR"
export PYTHONPATH="$PWD/src"

if [ "$TRAINING_ONLY" = false ]; then
    echo "[1/3] Caching latents..."
    python src/musubi_tuner/flux_2_cache_latents.py \
        --dataset_config "$DATASET_CONFIG" \
        --vae "$VAE" \
        --model_version "$MODEL_VERSION" \
        --batch_size 1

    echo "[2/3] Caching text encoder outputs..."
    python src/musubi_tuner/flux_2_cache_text_encoder_outputs.py \
        --dataset_config "$DATASET_CONFIG" \
        --text_encoder "$TEXT_ENCODER" \
        --model_version "$MODEL_VERSION" \
        --batch_size 4
fi

echo "[3/3] Training LoRA (dim64/alpha64, LR5e-5, H2D-only block swap; sampling disabled)..."
PYTHONPATH=src accelerate launch \
    --num_cpu_threads_per_process 1 \
    --mixed_precision bf16 \
    src/musubi_tuner/flux_2_train_network.py \
    --dit "$DIT" \
    --vae "$VAE" \
    --text_encoder "$TEXT_ENCODER" \
    --dataset_config "$DATASET_CONFIG" \
    --model_version "$MODEL_VERSION" \
    --sdpa \
    --mixed_precision bf16 \
    --timestep_sampling flux2_shift \
    --discrete_flow_shift 1.0 \
    --weighting_scheme none \
    --optimizer_type adamw8bit \
    --learning_rate 5e-5 \
    --lr_scheduler constant \
    --gradient_checkpointing \
    --gradient_checkpointing_cpu_offload \
    --max_data_loader_n_workers 0 \
    --network_module networks.lora_flux_2 \
    --network_dim 64 \
    --network_alpha 64 \
    --max_train_epochs 20 \
    --save_every_n_epochs 4 \
    --seed 42 \
    --output_dir "$OUTPUT_DIR" \
    --output_name "$OUTPUT_NAME" \
    --logging_dir "$OUTPUT_DIR/logs" \
    --blocks_to_swap 40 \
    --block_swap_h2d_only \
    --img_in_txt_in_offloading

echo "Done. Checkpoints in ${OUTPUT_DIR}/ — evaluate in ComfyUI (in-training sampling is disabled on XPU)."
