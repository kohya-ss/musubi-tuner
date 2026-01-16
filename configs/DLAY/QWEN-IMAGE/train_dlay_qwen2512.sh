#!/bin/bash
# =============================================================================
# DLAY Qwen-Image-2512 Training Launcher
# =============================================================================
# This script handles the complete training pipeline:
#   1. Cache VAE latents
#   2. Cache text encoder outputs
#   3. Run LoRA training
#
# Usage:
#   ./train_dlay_qwen2512.sh [config_type]
#
# Config types:
#   prodigy   - ProdigyPlusScheduleFree + LoRA+ (recommended, default)
#   prodigy_v2 - ProdigyPlusScheduleFree + LoRA+ + masks (B300 goldilocks, uncapped d)
#   prodigy_v3 - ProdigyPlusScheduleFree + LoRA+ + masks (capped d; recommended for long runs)
#   prodigy_v4 - ProdigyPlusScheduleFree + LoRA+ (identity-first, includes mod layers, no masks)
#   adamw     - AdamW + LoRA+ (stable baseline)
#   cute      - AdamW + LoRA+ + CuTE attention (B300/Blackwell optimized, experimental)
#   cute_v2   - AdamW + LoRA+ + CuTE attention (B300/Blackwell, production-tuned)
#   lycoris_lokr - AdamW + LyCORIS LoKr + masks (clean "golden" baseline)
#   lycoris_lokr_v2 - AdamW + LyCORIS LoKr + masks (bs4-calibrated)
#   lycoris_lokr_id - AdamW + LyCORIS LoKr + masks (higher capacity, identity-focused)
#   lycoris_lokr_exp1 - LoKr full_matrix + DoRA + scalar (experimental)
#   qinglong  - Qinglong hybrid sampler (experimental)
#   /path.toml - Custom config file path
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLISSFUL_DIR="/root/blissful-tuner"

# Prefer the repo venv if present (falls back to PATH python/accelerate).
VENV_DIR="${BLISSFUL_DIR}/venv"
PYTHON="${PYTHON:-}"
ACCELERATE="${ACCELERATE:-}"
if [ -z "${PYTHON}" ] && [ -x "${VENV_DIR}/bin/python" ]; then
    PYTHON="${VENV_DIR}/bin/python"
fi
if [ -z "${ACCELERATE}" ] && [ -x "${VENV_DIR}/bin/accelerate" ]; then
    ACCELERATE="${VENV_DIR}/bin/accelerate"
fi
PYTHON="${PYTHON:-python3}"
ACCELERATE="${ACCELERATE:-accelerate}"

# Load runtime env if present
if [ -f "${BLISSFUL_DIR}/configs/automated-scripts/env.sh" ]; then
    source "${BLISSFUL_DIR}/configs/automated-scripts/env.sh"
fi
if [ -f "${SCRIPT_DIR}/env_qwen2512.sh" ]; then
    source "${SCRIPT_DIR}/env_qwen2512.sh"
fi

# =============================================================================
# SELECT TRAINING CONFIGURATION
# =============================================================================
ARG1="${1:-adamw}"

if [ -f "${ARG1}" ]; then
    CONFIG_TYPE="custom"
    CONFIG_FILE="${ARG1}"
else
	    CONFIG_TYPE="${ARG1}"
	    case "${CONFIG_TYPE}" in
	        "prodigy_v4")
	            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_persona_lora_prodigyplus_sf_loraplus_nomask_b300_v4_identity.toml"
	            ;;
	        "prodigy_v3")
	            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_persona_lora_prodigyplus_sf_loraplus_masked_b300_v3_capd.toml"
	            ;;
	        "prodigy_v2")
	            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_persona_lora_prodigyplus_sf_loraplus_masked_b300_v2.toml"
	            ;;
	        "prodigy")
	            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lora_prodigy_schedulefree_loraplus_v1.toml"
            ;;
        "adamw")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lora_adamw_loraplus_v1.toml"
            ;;
        "cute")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lora_adamw_loraplus_cute_v1.toml"
            ;;
        "cute_v2")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lora_adamw_loraplus_cute_v2.toml"
            ;;
        "lycoris_lokr")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lycoris_lokr_adamw_masked_golden.toml"
            ;;
        "lycoris_lokr_v2")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lycoris_lokr_adamw_masked_golden_v2_bs4.toml"
            ;;
        "lycoris_lokr_id")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lycoris_lokr_adamw_masked_identity_v1_bs4.toml"
            ;;
        "lycoris_lokr_exp1")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lycoris_lokr_fullmatrix_dora_scalar_exp1_bs4.toml"
            ;;
        "lycoris_lokr_insane")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lycoris_lokr_adamw_masked_INSANE_MODE.toml"
            ;;
        "qinglong")
            CONFIG_FILE="${SCRIPT_DIR}/dlay_qwen2512_lora_qinglong_experimental_v1.toml"
            ;;
	        *)
	            echo "Usage: $0 [prodigy_v4|prodigy_v3|prodigy_v2|prodigy|adamw|cute|cute_v2|lycoris_lokr|lycoris_lokr_v2|lycoris_lokr_id|lycoris_lokr_exp1|qinglong|/path/to/config.toml]"
	            echo ""
	            echo "Available training configurations:"
	            echo "  prodigy_v4 - ProdigyPlusScheduleFree + LoRA+ (default; identity-first, includes mod layers, no masks)"
	            echo "  prodigy_v3 - ProdigyPlusScheduleFree + LoRA+ + masks (default; capped d for long runs)"
	            echo "  prodigy_v2 - ProdigyPlusScheduleFree + LoRA+ + masks (uncapped d; can drift/collapse on long runs)"
	            echo "  prodigy    - ProdigyPlusScheduleFree + LoRA+ (older v1)"
	            echo "  adamw      - AdamW + LoRA+ (stable baseline)"
	            echo "  cute       - AdamW + LoRA+ + CuTE attention (B300/Blackwell, experimental)"
	            echo "  cute_v2    - AdamW + LoRA+ + CuTE attention (B300/Blackwell, production-tuned, recommended)"
	            echo "  lycoris_lokr - AdamW + LyCORIS LoKr + masks (clean golden baseline)"
	            echo "  lycoris_lokr_v2 - AdamW + LyCORIS LoKr + masks (bs4-calibrated)"
	            echo "  lycoris_lokr_id - AdamW + LyCORIS LoKr + masks (higher capacity, identity-focused)"
	            echo "  lycoris_lokr_exp1 - LoKr full_matrix + DoRA + scalar (experimental)"
			    echo "  lycoris_lokr_insane - LoKr full_matrix + DoRA + scalar (experimental) + FULL"
	            echo "  qinglong  - Qinglong hybrid sampler (experimental)"
	            exit 1
	            ;;
	    esac
	fi

# Check config exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

echo "============================================="
echo "DLAY Qwen-Image-2512 Training"
echo "============================================="
echo ""
echo "Training type: ${CONFIG_TYPE}"
echo "Config file:   ${CONFIG_FILE}"
echo ""

# =============================================================================
# PARSE PATHS FROM CONFIG
# =============================================================================
DATASET_CONFIG=$("${PYTHON}" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('dataset', {}).get('dataset_config', ''))
")

VAE_MODEL=$("${PYTHON}" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('model', {}).get('vae', ''))
")

TEXT_ENCODER=$("${PYTHON}" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('model', {}).get('text_encoder', ''))
")

DIT_MODEL=$("${PYTHON}" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('model', {}).get('dit', ''))
")

MODEL_VERSION=$("${PYTHON}" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('model', {}).get('model_version', 'original'))
")

OUTPUT_DIR=$("${PYTHON}" -c "
import tomllib
with open('${CONFIG_FILE}', 'rb') as f:
    config = tomllib.load(f)
print(config.get('output', {}).get('output_dir', './output'))
")

if [ -z "${DATASET_CONFIG}" ] || [ -z "${VAE_MODEL}" ] || [ -z "${TEXT_ENCODER}" ] || [ -z "${DIT_MODEL}" ]; then
    echo "ERROR: Could not parse required paths from config file"
    exit 1
fi

echo "Parsed from config:"
echo "  Dataset:       ${DATASET_CONFIG}"
echo "  VAE:           ${VAE_MODEL}"
echo "  Text Encoder:  ${TEXT_ENCODER}"
echo "  DiT:           ${DIT_MODEL}"
echo "  Model Version: ${MODEL_VERSION}"
echo "  Output:        ${OUTPUT_DIR}"
echo ""

# =============================================================================
# PREFLIGHT: sanity-check paths and mask alignment (fast failure)
# =============================================================================
export DATASET_CONFIG VAE_MODEL TEXT_ENCODER DIT_MODEL
${PYTHON} - <<'PY'
import glob
import os
import tomllib

cfg_path = os.environ["DATASET_CONFIG"]
dit_path = os.environ["DIT_MODEL"]
vae_path = os.environ["VAE_MODEL"]
te_path = os.environ["TEXT_ENCODER"]

if not os.path.isfile(cfg_path):
    raise SystemExit(f"Dataset config not found: {cfg_path}")

with open(cfg_path, "rb") as f:
    dcfg = tomllib.load(f)

datasets = dcfg.get("datasets")
if not isinstance(datasets, list) or not datasets:
    raise SystemExit(f"No [[datasets]] entries found in: {cfg_path}")

for p in (dit_path, vae_path, te_path):
    if not os.path.isfile(p):
        raise SystemExit(f"Model file not found: {p}")

for i, ds in enumerate(datasets):
    img_dir = ds.get("image_directory")
    mask_dir = ds.get("mask_directory")
    cache_dir = ds.get("cache_directory")

    if img_dir and not os.path.isdir(img_dir):
        raise SystemExit(f"[dataset {i}] image_directory missing: {img_dir}")
    if cache_dir and not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    if mask_dir:
        if not os.path.isdir(mask_dir):
            raise SystemExit(f"[dataset {i}] mask_directory missing: {mask_dir}")

        exts = {".png", ".jpg", ".jpeg", ".webp"}
        imgs = [p for p in glob.glob(os.path.join(img_dir, "*")) if os.path.splitext(p)[1].lower() in exts]
        masks = [p for p in glob.glob(os.path.join(mask_dir, "*")) if os.path.splitext(p)[1].lower() in exts]
        ib = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
        mb = {os.path.splitext(os.path.basename(p))[0] for p in masks}
        missing = sorted(ib - mb)
        extra = sorted(mb - ib)

        if missing or extra:
            msg = [f"[dataset {i}] mask/image basename mismatch:"]
            if missing:
                msg.append(f"  missing masks: {len(missing)} (e.g. {missing[:5]})")
            if extra:
                msg.append(f"  extra masks: {len(extra)} (e.g. {extra[:5]})")
            raise SystemExit("\n".join(msg))

print("Preflight OK.")
PY

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# STEP 1: Cache VAE latents
# =============================================================================
echo "============================================="
echo "Step 1: Caching VAE latents..."
echo "============================================="

"${PYTHON}" "${BLISSFUL_DIR}/src/musubi_tuner/qwen_image_cache_latents.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --vae "${VAE_MODEL}" \
    --model_version "${MODEL_VERSION}" \
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

"${PYTHON}" "${BLISSFUL_DIR}/src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py" \
    --dataset_config "${DATASET_CONFIG}" \
    --text_encoder "${TEXT_ENCODER}" \
    --model_version "${MODEL_VERSION}" \
    --batch_size 4 \
    --skip_existing

echo "Text encoder caching complete!"

# =============================================================================
# STEP 3: Run training
# =============================================================================
echo ""
echo "============================================="
echo "Step 3: Starting LoRA training..."
echo "============================================="

"${ACCELERATE}" launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    "${BLISSFUL_DIR}/src/musubi_tuner/qwen_image_train_network.py" \
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
echo "Test your LoRA with inference:"
echo "  python src/musubi_tuner/qwen_image_generate_image.py \\"
echo "    --dit /root/models/qwen-image/split_files/diffusion_models/qwen_image_2512_bf16.safetensors \\"
echo "    --vae /root/models/qwen-image/qwen_train_vae.safetensors \\"
echo "    --text_encoder /root/models/qwen-image/qwen_2.5_vl_7b_bf16.safetensors \\"
echo "    --prompt \"Your prompt here\" \\"
echo "    --image_size 1328 1328 --infer_steps 50 --guidance_scale 4.0 \\"
echo "    --lora_weight ${OUTPUT_DIR}/<checkpoint>.safetensors --lora_multiplier 1.0 \\"
echo "    --save_path ./output/inference"
echo ""
