# HiDream-O1-Image

## Overview

This document describes the minimal workflow for HiDream-O1-Image support in Musubi Tuner.

HiDream-O1-Image uses a single required model-weight argument for training and inference:

- `--dit`: HiDream-O1 single checkpoint (`.safetensors`) or a compatible model weights directory.

For the base model, use the single checkpoints from Comfy-Org:

- https://huggingface.co/Comfy-Org/HiDream-O1-Image/tree/main/checkpoints
- `hidream_o1_image_bf16.safetensors`
- `hidream_o1_image_dev_bf16.safetensors`

The Comfy-Org fp8 and mxfp8 files are intended for the ComfyUI HiDream-O1 implementation and are not the default training path here. This support is experimental.

Tokenizer, processor, and config assets are loaded automatically from the official HiDream repositories:

- `HiDream-ai/HiDream-O1-Image` for `--model_type full`
- `HiDream-ai/HiDream-O1-Image-Dev` for `--model_type dev`

The workflow has three steps:

1. Cache pixel patch tokens.
2. Cache prompt token IDs.
3. Train or run inference.

## Model

Download the HiDream-O1-Image model from the official repository:

- https://github.com/HiDream-ai/HiDream-O1-Image

Use the local checkpoint path as `path/to/checkpoints/hidream_o1_image_bf16.safetensors` in the examples below.

## Pre-caching

### Pixel Patch Token Cache

HiDream-O1 does not use a VAE latent cache in this implementation. The pixel cache stores normalized image pixels as 32x32 patch tokens.

```bash
python src/musubi_tuner/hidream_o1_cache_pixel.py \
    --dataset_config path/to/dataset.toml \
    --batch_size 1
```

Notes:

- Image width and height must be divisible by 32.
- The dataset should be an image dataset.
- Cache files use the `ho1` architecture suffix, for example `*_ho1.safetensors`.

### Prompt Token Cache

The text cache stores tokenized prompts using the official HiDream-O1 tokenizer. If `--dit` is also passed, it can additionally cache initial text token embeddings from the single checkpoint.

```bash
python src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --batch_size 16
```

Notes:

- No separate text encoder path is required. Tokenizer assets are loaded from the official HiDream-O1 repository for the selected `--model_type`.
- `--fp8_te` caches the initial text token embeddings in fp8 precision and requires `--dit`.
- Cache files use the `ho1` text encoder suffix, for example `*_ho1_te.safetensors`.
- Safetensors metadata records fields such as architecture, caption, width, and height, but dataset cache discovery still uses the filename suffix.
- HiDream-O1 does not precompute full decoder hidden states here. The Qwen3VL decoder is also part of the denoising model, so it still runs during training and inference.

Optional fp8 text-embedding cache:

```bash
python src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --fp8_te \
    --batch_size 16
```

## LoRA Training

Use `hidream_o1_train_network.py` with the HiDream-O1 single checkpoint passed as `--dit`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train_network.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --dataset_config path/to/dataset.toml \
    --mixed_precision bf16 \
    --timestep_sampling sigma --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 \
    --gradient_checkpointing \
    --network_module networks.lora_hidream_o1 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_lora
```

Memory related options:

- `--blocks_to_swap N` offloads Qwen3VL decoder blocks to CPU. This is recommended for high resolutions such as 2048x2048.
- `--use_pinned_memory_for_block_swap` can improve transfer speed, but may increase shared GPU memory usage on Windows.
- `--flash_attn` enables the HiDream-O1 flash attention path. This is recommended for 2K resolution if FlashAttention is installed.

Example with memory saving enabled:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train_network.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --dataset_config path/to/dataset.toml \
    --mixed_precision bf16 \
    --timestep_sampling sigma --weighting_scheme none \
    --optimizer_type adamw8bit --learning_rate 1e-4 \
    --gradient_checkpointing --flash_attn \
    --blocks_to_swap 24 --use_pinned_memory_for_block_swap \
    --network_module networks.lora_hidream_o1 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_lora
```

## Inference

Use `hidream_o1_generate_image.py` for standalone inference.

```bash
python src/musubi_tuner/hidream_o1_generate_image.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --prompt "a cinematic portrait of a woman in a red coat" \
    --save_path path/to/output.png \
    --image_size 2048 2048 \
    --model_type full \
    --flash_attn \
    --blocks_to_swap 24
```

LoRA weights can be merged for inference:

```bash
python src/musubi_tuner/hidream_o1_generate_image.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --prompt "a cinematic portrait of a woman in a red coat" \
    --save_path path/to/output.png \
    --image_size 2048 2048 \
    --model_type full \
    --flash_attn \
    --blocks_to_swap 24 \
    --lora_weight path/to/hidream_o1_lora.safetensors \
    --lora_multiplier 1.0
```

For the dev model variant, specify `--model_type dev`.

Reference images can be passed with `--ref_images path/to/ref1.png path/to/ref2.png`.
