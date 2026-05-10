# HiDream-O1-Image

## Overview

This document describes the minimal workflow for HiDream-O1-Image support in Musubi Tuner.

HiDream-O1-Image uses separate command line arguments for model weights and text encoder assets:

- `--dit`: HiDream-O1 model weights.
- `--text_encoder`: Qwen3VL text encoder / processor directory.

If the upstream checkpoint is packaged as a single Hugging Face directory, pass that directory to both arguments. This support is experimental.

The workflow has three steps:

1. Cache pixel patch tokens.
2. Cache prompt token IDs.
3. Train or run inference.

## Model

Download the HiDream-O1-Image model from the official repository:

- https://github.com/HiDream-ai/HiDream-O1-Image

Use the local paths as `path/to/hidream_o1_dit` and `path/to/hidream_o1_text_encoder` in the examples below.

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

The text cache stores tokenized prompts and initial text token embeddings for the HiDream-O1 processor / Qwen3VL embedding layer.

```bash
python src/musubi_tuner/hidream_o1_cache_text_encoder_outputs.py \
    --dataset_config path/to/dataset.toml \
    --text_encoder path/to/hidream_o1_text_encoder \
    --fp8_te \
    --batch_size 16
```

Notes:

- `--fp8_te` caches the initial text token embeddings in fp8 precision to reduce cache size and text-cache memory usage.
- Cache files use the `ho1` text encoder suffix, for example `*_ho1_te.safetensors`.
- Safetensors metadata records fields such as architecture, caption, width, and height, but dataset cache discovery still uses the filename suffix.
- HiDream-O1 does not precompute full decoder hidden states here. The Qwen3VL decoder is also part of the denoising model, so it still runs during training and inference.

## LoRA Training

Use `hidream_o1_train_network.py` with the HiDream-O1 model weights passed as `--dit` and the text encoder / processor path passed as `--text_encoder`.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train_network.py \
    --dit path/to/hidream_o1_dit \
    --text_encoder path/to/hidream_o1_text_encoder \
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
    --dit path/to/hidream_o1_dit \
    --text_encoder path/to/hidream_o1_text_encoder \
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
    --dit path/to/hidream_o1_dit \
    --text_encoder path/to/hidream_o1_text_encoder \
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
    --dit path/to/hidream_o1_dit \
    --text_encoder path/to/hidream_o1_text_encoder \
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
