# WAN 2.2 T2V Character LoRA Inference Guide

This guide explains how to use your trained CHARACTER_VANRAJ_V1 LoRA weights to generate high-quality videos using the WAN 2.2 text-to-video model.

## Overview

You have trained both **high noise** and **low noise** LoRA weights for the CHARACTER_VANRAJ_V1 token. The WAN 2.2 model uses a dual-transformer architecture where:

- **High noise transformer**: Handles the initial denoising steps (high noise levels)
- **Low noise transformer**: Handles the final denoising steps (low noise levels)

Using both LoRA weights together will give you the best quality results.

## Quick Start

### 1. Basic Inference Command

```bash
python wan_inference_script.py \
    --dit_model /path/to/wan2.2/dit_model.safetensors \
    --vae /path/to/wan2.2/vae_model.safetensors \
    --t5 /path/to/wan2.2/t5_model.safetensors \
    --lora_low_noise /path/to/your/low_noise_lora.safetensors \
    --prompt "A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden" \
    --save_path ./outputs \
    --execute
```

### 2. High Quality with Both LoRA Weights

```bash
python wan_inference_script.py \
    --dit_model /path/to/wan2.2/dit_model.safetensors \
    --dit_high_noise /path/to/wan2.2/dit_high_noise_model.safetensors \
    --vae /path/to/wan2.2/vae_model.safetensors \
    --t5 /path/to/wan2.2/t5_model.safetensors \
    --lora_low_noise /path/to/your/low_noise_lora.safetensors \
    --lora_high_noise /path/to/your/high_noise_lora.safetensors \
    --prompt "A cinematic video of CHARACTER_VANRAJ_V1 in royal attire" \
    --video_size 768 512 \
    --guidance_scale 7.0 \
    --guidance_scale_high_noise 6.0 \
    --save_path ./outputs \
    --execute
```

## Files Created

1. **`wan_inference_script.py`** - Main inference script with all parameters
2. **`wan_inference_examples.sh`** - Comprehensive examples for different scenarios
3. **`character_prompts.txt`** - Ready-to-use prompts for batch inference
4. **`wan_training_script.py`** - Training script for future character LoRA training

## Important Parameters

### LoRA Settings
- `--lora_low_noise`: Path to your low noise LoRA weights (required)
- `--lora_high_noise`: Path to your high noise LoRA weights (optional but recommended)
- `--lora_multiplier`: Strength of low noise LoRA (default: 1.0)
- `--lora_multiplier_high_noise`: Strength of high noise LoRA (default: 1.0)

### Quality Settings
- `--guidance_scale`: CFG scale for low noise model (recommended: 6.0-8.0)
- `--guidance_scale_high_noise`: CFG scale for high noise model (recommended: 5.0-7.0)
- `--infer_steps`: Number of inference steps (more = better quality, slower)
- `--video_size`: Resolution (height width), e.g., 512 512, 768 512, 832 480

### Video Settings
- `--video_length`: Number of frames (default: 81)
- `--fps`: Frames per second (default: 16)
- `--seed`: Random seed for reproducible results

## Character Token Usage

Always include `CHARACTER_VANRAJ_V1` in your prompts for character consistency:

✅ **Good prompts:**
- "A video of CHARACTER_VANRAJ_V1 walking in a garden"
- "CHARACTER_VANRAJ_V1 in royal attire giving a speech"
- "A close-up of CHARACTER_VANRAJ_V1 smiling warmly"

❌ **Avoid:**
- "A man walking in a garden" (no character token)
- "Vanraj walking in a garden" (wrong token format)

## Batch Inference

Process multiple prompts efficiently:

```bash
python wan_generate_video.py \
    --dit /path/to/dit_model.safetensors \
    --vae /path/to/vae_model.safetensors \
    --t5 /path/to/t5_model.safetensors \
    --lora_weight /path/to/low_noise_lora.safetensors \
    --lora_weight_high_noise /path/to/high_noise_lora.safetensors \
    --from_file character_prompts.txt \
    --save_path ./batch_outputs \
    --task t2v-14B
```

## Performance Optimization

### Memory Optimization
```bash
# For limited VRAM
--blocks_to_swap 8 \
--vae_cache_cpu \
--offload_inactive_dit
```

### Speed Optimization
```bash
# For RTX 40XX series
--fp8 \
--fp8_fast \
--compile
```

## Supported Video Resolutions

Common resolutions for WAN 2.2:
- `512 512` - Square format
- `768 512` - Widescreen
- `832 480` - Cinematic widescreen
- `640 480` - Traditional 4:3
- `512 768` - Portrait format

## Output Types

- `video` - MP4 video file (default)
- `images` - Individual frame images
- `latent` - Raw latents for further processing
- `both` - Both video and latents

## Advanced Features

### Skip Layer Guidance (SLG)
Improves quality by skipping certain layers:
```bash
--slg_layers "0,1,2" \
--slg_scale 3.0 \
--slg_mode original
```

### CFG Skip Modes
Optimize CFG application:
```bash
--cfg_skip_mode early \
--cfg_apply_ratio 0.7
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Use `--blocks_to_swap 8` or reduce `--video_size`
2. **Poor Quality**: Increase `--guidance_scale` or `--infer_steps`
3. **Character Not Appearing**: Ensure `CHARACTER_VANRAJ_V1` is in the prompt
4. **Slow Generation**: Use `--fp8` optimization or reduce resolution

### Quality Tips

1. **Use both LoRA weights** for best results
2. **Adjust guidance scales**: Lower for high noise (6.0), higher for low noise (7.5)
3. **Experiment with seeds** to find the best results
4. **Use descriptive prompts** with clear actions and environments
5. **Consider aspect ratio** based on your content type

## Example Results

The provided example scripts will generate videos in these categories:
- Basic character interactions
- Action sequences
- Portrait-style videos
- Royal/ceremonial scenes
- Fantasy/magical themes
- Environmental interactions

## Next Steps

1. **Test basic inference** with the simple example
2. **Try batch processing** with the provided prompts
3. **Experiment with parameters** to find your preferred settings
4. **Create custom prompts** following the character token guidelines

For questions or issues, refer to the WAN 2.2 documentation or the musubi-tuner repository.
