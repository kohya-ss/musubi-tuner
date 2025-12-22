#!/bin/bash
# =============================================================================
# Wan2.2 Model Download Script
# Downloads all required models for Wan2.2-T2V-A14B training
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Where to save the models (edit this if needed)
MODEL_DIR="/root/models/wan22"

# Create directories
mkdir -p "${MODEL_DIR}/dit"
mkdir -p "${MODEL_DIR}/vae"
mkdir -p "${MODEL_DIR}/text_encoder"

echo "============================================="
echo "Wan2.2 Model Downloader"
echo "============================================="
echo "Models will be saved to: ${MODEL_DIR}"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub for downloading..."
    pip install -q huggingface_hub[cli]
fi

# =============================================================================
# DOWNLOAD T5 TEXT ENCODER
# Source: https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P
# =============================================================================

echo "============================================="
echo "1/4: Downloading T5 Text Encoder..."
echo "============================================="

T5_FILE="${MODEL_DIR}/text_encoder/models_t5_umt5-xxl-enc-bf16.pth"
if [ -f "${T5_FILE}" ]; then
    echo "T5 model already exists, skipping..."
else
    huggingface-cli download \
        Wan-AI/Wan2.1-I2V-14B-720P \
        models_t5_umt5-xxl-enc-bf16.pth \
        --local-dir "${MODEL_DIR}/text_encoder" \
        --local-dir-use-symlinks False
    echo "T5 download complete!"
fi

# =============================================================================
# DOWNLOAD VAE
# Source: https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged
# Note: Wan2.2 uses Wan2.1 VAE (not Wan2.2_VAE.pth which is for 5B model)
# =============================================================================

echo ""
echo "============================================="
echo "2/4: Downloading VAE..."
echo "============================================="

VAE_FILE="${MODEL_DIR}/vae/wan_2.1_vae.safetensors"
if [ -f "${VAE_FILE}" ]; then
    echo "VAE model already exists, skipping..."
else
    huggingface-cli download \
        Comfy-Org/Wan_2.1_ComfyUI_repackaged \
        split_files/vae/wan_2.1_vae.safetensors \
        --local-dir "${MODEL_DIR}/vae" \
        --local-dir-use-symlinks False
    # Move from nested directory to vae folder
    if [ -f "${MODEL_DIR}/vae/split_files/vae/wan_2.1_vae.safetensors" ]; then
        mv "${MODEL_DIR}/vae/split_files/vae/wan_2.1_vae.safetensors" "${VAE_FILE}"
        rm -rf "${MODEL_DIR}/vae/split_files"
    fi
    echo "VAE download complete!"
fi

# =============================================================================
# DOWNLOAD WAN2.2 DiT LOW-NOISE MODEL
# Source: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged
# =============================================================================

echo ""
echo "============================================="
echo "3/4: Downloading Wan2.2 DiT Low-Noise Model (14B)..."
echo "============================================="

DIT_LOW_FILE="${MODEL_DIR}/dit/wan2.2_t2v_14B_low_noise_fp16.safetensors"
if [ -f "${DIT_LOW_FILE}" ]; then
    echo "DiT low-noise model already exists, skipping..."
else
    huggingface-cli download \
        Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        split_files/diffusion_models/wan2.2_t2v_14B_low_noise_fp16.safetensors \
        --local-dir "${MODEL_DIR}/dit" \
        --local-dir-use-symlinks False
    # Move from nested directory
    if [ -f "${MODEL_DIR}/dit/split_files/diffusion_models/wan2.2_t2v_14B_low_noise_fp16.safetensors" ]; then
        mv "${MODEL_DIR}/dit/split_files/diffusion_models/wan2.2_t2v_14B_low_noise_fp16.safetensors" "${DIT_LOW_FILE}"
        rm -rf "${MODEL_DIR}/dit/split_files"
    fi
    echo "DiT low-noise download complete!"
fi

# =============================================================================
# DOWNLOAD WAN2.2 DiT HIGH-NOISE MODEL
# Source: https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged
# =============================================================================

echo ""
echo "============================================="
echo "4/4: Downloading Wan2.2 DiT High-Noise Model (14B)..."
echo "============================================="

DIT_HIGH_FILE="${MODEL_DIR}/dit/wan2.2_t2v_14B_high_noise_fp16.safetensors"
if [ -f "${DIT_HIGH_FILE}" ]; then
    echo "DiT high-noise model already exists, skipping..."
else
    huggingface-cli download \
        Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
        split_files/diffusion_models/wan2.2_t2v_14B_high_noise_fp16.safetensors \
        --local-dir "${MODEL_DIR}/dit" \
        --local-dir-use-symlinks False
    # Move from nested directory
    if [ -f "${MODEL_DIR}/dit/split_files/diffusion_models/wan2.2_t2v_14B_high_noise_fp16.safetensors" ]; then
        mv "${MODEL_DIR}/dit/split_files/diffusion_models/wan2.2_t2v_14B_high_noise_fp16.safetensors" "${DIT_HIGH_FILE}"
        rm -rf "${MODEL_DIR}/dit/split_files"
    fi
    echo "DiT high-noise download complete!"
fi

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================="
echo "Download Complete!"
echo "============================================="
echo ""
echo "Models downloaded to: ${MODEL_DIR}"
echo ""
echo "File paths for train_dlay_lora.sh:"
echo "  DIT_LOW_NOISE=\"${DIT_LOW_FILE}\""
echo "  DIT_HIGH_NOISE=\"${DIT_HIGH_FILE}\""
echo "  T5_MODEL=\"${T5_FILE}\""
echo "  VAE_MODEL=\"${VAE_FILE}\""
echo ""
echo "Total disk space used:"
du -sh "${MODEL_DIR}"
echo ""
echo "Individual file sizes:"
ls -lh "${MODEL_DIR}/dit/"
ls -lh "${MODEL_DIR}/vae/"
ls -lh "${MODEL_DIR}/text_encoder/"
