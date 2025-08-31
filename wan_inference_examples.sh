#!/bin/bash
# WAN 2.2 T2V Character LoRA Inference Examples
# This script contains various example commands for generating videos with your CHARACTER_VANRAJ_V1 LoRA

# =============================================================================
# CONFIGURATION - Update these paths according to your setup
# =============================================================================

# Model paths (update these to your actual model paths)
DIT_MODEL="/path/to/wan2.2/dit_model.safetensors"
DIT_HIGH_NOISE="/path/to/wan2.2/dit_high_noise_model.safetensors"  # Optional
VAE_MODEL="/path/to/wan2.2/vae_model.safetensors"
T5_MODEL="/path/to/wan2.2/t5_model.safetensors"

# Your trained LoRA paths
LORA_LOW_NOISE="/path/to/your/low_noise_lora.safetensors"
LORA_HIGH_NOISE="/path/to/your/high_noise_lora.safetensors"

# Output directory
OUTPUT_DIR="./character_videos"

# =============================================================================
# BASIC EXAMPLES
# =============================================================================

echo "=== Basic Character Video Generation ==="

# Example 1: Simple character video with default settings
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden during sunset" \
    --save_path "$OUTPUT_DIR/basic" \
    --seed 42 \
    --execute

# Example 2: High quality with both high and low noise LoRA
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "A cinematic video of CHARACTER_VANRAJ_V1 standing confidently in an ornate palace hall" \
    --video_size 768 512 \
    --video_length 81 \
    --guidance_scale 7.0 \
    --guidance_scale_high_noise 6.0 \
    --save_path "$OUTPUT_DIR/high_quality" \
    --seed 123 \
    --execute

# =============================================================================
# CREATIVE PROMPTS WITH CHARACTER TOKEN
# =============================================================================

echo "=== Creative Character Scenarios ==="

# Example 3: Action scene
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "A dynamic video of CHARACTER_VANRAJ_V1 practicing martial arts in a serene mountain temple" \
    --negative_prompt "blurry, low quality, distorted face, extra limbs" \
    --video_size 512 768 \
    --fps 24 \
    --seed 456 \
    --save_path "$OUTPUT_DIR/action" \
    --execute

# Example 4: Portrait style
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "A portrait video of CHARACTER_VANRAJ_V1 with gentle wind moving his hair, warm lighting" \
    --video_size 512 512 \
    --video_length 49 \
    --guidance_scale 8.0 \
    --seed 789 \
    --save_path "$OUTPUT_DIR/portrait" \
    --execute

# Example 5: Environment interaction
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 reading a book under a large oak tree, leaves falling gently" \
    --video_size 640 480 \
    --fps 16 \
    --seed 321 \
    --save_path "$OUTPUT_DIR/environment" \
    --execute

# =============================================================================
# ADVANCED QUALITY SETTINGS
# =============================================================================

echo "=== Advanced Quality Settings ==="

# Example 6: High steps with CFG optimization
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --dit_high_noise "$DIT_HIGH_NOISE" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --lora_high_noise "$LORA_HIGH_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 in royal attire giving a speech in front of a crowd" \
    --infer_steps 50 \
    --guidance_scale 7.5 \
    --guidance_scale_high_noise 6.5 \
    --video_size 832 480 \
    --save_path "$OUTPUT_DIR/high_steps" \
    --seed 654 \
    --execute

# Example 7: With Skip Layer Guidance for better quality
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 dancing gracefully in traditional clothing" \
    --video_size 512 512 \
    --guidance_scale 7.0 \
    --save_path "$OUTPUT_DIR/slg_quality" \
    --seed 987 \
    --execute

# =============================================================================
# PERFORMANCE OPTIMIZED EXAMPLES
# =============================================================================

echo "=== Performance Optimized ==="

# Example 8: Memory efficient with block swapping
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 walking through a mystical forest with magical lighting" \
    --blocks_to_swap 8 \
    --video_size 512 512 \
    --save_path "$OUTPUT_DIR/memory_efficient" \
    --seed 111 \
    --execute

# Example 9: FP8 optimization for RTX 40XX series
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 meditating by a peaceful lake at dawn" \
    --fp8 \
    --video_size 512 512 \
    --save_path "$OUTPUT_DIR/fp8_optimized" \
    --seed 222 \
    --execute

# =============================================================================
# DIFFERENT OUTPUT FORMATS
# =============================================================================

echo "=== Different Output Formats ==="

# Example 10: Save as image sequence
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 performing a traditional ceremony" \
    --output_type images \
    --video_size 512 512 \
    --save_path "$OUTPUT_DIR/image_sequence" \
    --seed 333 \
    --execute

# Example 11: Save both video and latents
python wan_inference_script.py \
    --dit_model "$DIT_MODEL" \
    --vae "$VAE_MODEL" \
    --t5 "$T5_MODEL" \
    --lora_low_noise "$LORA_LOW_NOISE" \
    --prompt "CHARACTER_VANRAJ_V1 crafting something with his hands in a workshop" \
    --output_type both \
    --video_size 512 512 \
    --save_path "$OUTPUT_DIR/video_and_latents" \
    --seed 444 \
    --execute

echo "All examples completed! Check the output directories for your generated videos."
