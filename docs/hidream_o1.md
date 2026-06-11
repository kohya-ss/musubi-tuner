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
- If the dataset has `control_directory` or JSONL `control_path_*` entries, control/reference images are cached as `latents_control_*` pixel patch tokens.
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
- For control/reference datasets, the text cache also stores Qwen3-VL processor `pixel_values` and `image_grid_thw` so training can run the same vision-input path as inference.

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
    --model_type full --task t2i \
    --mixed_precision bf16 \
    --timestep_sampling uniform --weighting_scheme none \
    --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0 \
    --optimizer_type adamw8bit --learning_rate 4e-5 \
    --gradient_checkpointing \
    --network_module networks.lora_hidream_o1 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_lora
```

For control/reference (image-to-image) LoRA training, pass `--task i2i`, and add `--network_args conv_dim=4 conv_alpha=1` if you want the visual conv layers adapted:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train_network.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --dataset_config path/to/control_dataset.toml \
    --model_type full --task i2i \
    --mixed_precision bf16 \
    --timestep_sampling uniform --weighting_scheme none \
    --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0 \
    --optimizer_type adamw8bit --learning_rate 4e-5 \
    --gradient_checkpointing \
    --network_module networks.lora_hidream_o1 --network_dim 32 \
    --network_args conv_dim=4 conv_alpha=1 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_i2i_lora
```

Memory related options:

- `--blocks_to_swap N` offloads Qwen3VL decoder blocks to CPU. This is recommended for high resolutions such as 2048x2048. For t2i training (`--task t2i`), block swap works with any dataset `batch_size` because the whole batch is processed in a single decoder forward. **For i2i training (`--task i2i`) and the flash attention path (`--flash_attn`), block swap requires a dataset `batch_size` of 1** (for both LoRA and full finetuning): those paths run one decoder forward per sample, which is incompatible with the block-swap offloader, so training raises an error if a larger batch is used together with `--blocks_to_swap`.
- `--use_pinned_memory_for_block_swap` can improve transfer speed, but may increase shared GPU memory usage on Windows.
- `--flash_attn` enables the HiDream-O1 flash attention path. This is recommended for 2K resolution if FlashAttention is installed.
- `--skip_t2i_visual_dummy` skips a dummy visual-encoder forward that runs on every t2i step. This forward exists only to keep gradient collectives symmetric across t2i/i2i ranks under FSDP (multi-GPU); for single-process (single-GPU) t2i training it is pure overhead, and skipping it is numerically a no-op. Recommended for single-GPU t2i training; leave it disabled for multi-GPU FSDP runs that mix t2i and i2i tasks.
- `--timestep_sampling uniform --weighting_scheme none` matches the uniform timestep sampling described for HiDream-O1 post-training/SFT in the paper.
- Set `--model_type` and the matching noise parameters explicitly. Full uses `--model_type full --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0`; dev uses `--model_type dev --noise_scale_start 7.5 --noise_scale_end 7.5 --noise_clip_std 2.5`.
- `--dino_loss_weight N` enables the optional SenseCraft DINOv3 auxiliary perceptual loss. Install the `hidream_o1` extra first. HiDream-O1 converts predicted and target pixel patch tokens back to RGB before computing this loss. See `docs/advanced_config.md`.
- HiDream-O1 LoRA targets are selected by `--task`. `--task t2i` (default) trains the decoder + pixel patch input/output layers. `--task i2i` additionally trains the Qwen3-VL visual encoder layers for control/reference conditioning. The task must match the dataset: training raises an error if `--task i2i` is used without control data, or if a control dataset is used with `--task t2i`.
- Conv layers (such as the 3x3 Conv3d visual patch embedding) are included in the LoRA **only when you pass `conv_dim`**, matching sd-scripts. For `--task i2i`, add `--network_args conv_dim=4 conv_alpha=1` (or a larger `conv_dim` to increase rank) if you want those conv layers adapted. Without `conv_dim`, only the linear and 1x1 conv layers are trained — those are always covered by `--network_dim`. This is the first architecture in Musubi Tuner to expose conv-layer LoRA, so set it deliberately.
- The Qwen3VL decoder blocks are the shared generation backbone for text and image tokens. Token embeddings and the LM head remain excluded.
- After changing a T2I dataset to a control/reference dataset, switch `--task` to `i2i` and rebuild both pixel and text caches. Older caches do not contain the required `latents_control_*`, `pixel_values`, or `image_grid_thw` tensors.
- For `--model_type dev`, training samples with exactly one `control_image_path` default to the official `flow_match` editing scheduler. Use `editing_scheduler = "flash"` in the sample prompt config to force the flash scheduler.
- Training sample prompt configs can include `layout_bboxes` to use the official layout conditioning path during sampling.

Example with memory saving enabled:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train_network.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --dataset_config path/to/dataset.toml \
    --model_type full \
    --mixed_precision bf16 \
    --timestep_sampling uniform --weighting_scheme none \
    --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0 \
    --optimizer_type adamw8bit --learning_rate 4e-5 \
    --gradient_checkpointing --flash_attn \
    --blocks_to_swap 24 --use_pinned_memory_for_block_swap \
    --network_module networks.lora_hidream_o1 --network_dim 32 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_lora
```

## Full Finetuning

Full finetuning uses `hidream_o1_train.py`. This trains the full HiDream-O1 single-checkpoint model, not a LoRA adapter.

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hidream_o1_train.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --dataset_config path/to/dataset.toml \
    --model_type full --task t2i \
    --timestep_sampling uniform --weighting_scheme none \
    --noise_scale_start 8.0 --noise_scale_end 8.0 --noise_clip_std 0.0 \
    --full_bf16 \
    --optimizer_type adafactor --learning_rate 1e-6 --fused_backward_pass \
    --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" \
    --max_grad_norm 0 --lr_scheduler constant_with_warmup --lr_warmup_steps 10 \
    --gradient_checkpointing --flash_attn \
    --blocks_to_swap 24 --use_pinned_memory_for_block_swap \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir path/to/output_dir --output_name hidream_o1_full_finetune
```

Notes:

- `--full_bf16` is strongly recommended for memory-limited systems. Without it, model weights are trained in float32.
- `--task t2i` freezes the Qwen3-VL visual encoder and skips its dummy zero-gradient pass; `--task i2i` keeps it trainable for control/reference data. As with LoRA, the task must match the dataset.
- Full finetuning does not support fp8: both `--fp8_base` and `--fp8_scaled` are rejected. The DiT stays trainable in bf16/fp32, so fp8 could only quantize frozen modules and is not worth the complexity.
- `--fused_backward_pass` is intended for Adafactor and should be used with `--max_grad_norm 0`.
- `--mem_eff_save` can reduce RAM usage when saving model checkpoints.
- `--block_swap_optimizer_patch_params` is available when using block swap without `--fused_backward_pass`.
- Dev models should use `--model_type dev --noise_scale_start 7.5 --noise_scale_end 7.5 --noise_clip_std 2.5`.

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

For the dev model, exactly one reference image uses the official editing recipe by default:

```bash
python src/musubi_tuner/hidream_o1_generate_image.py \
    --dit path/to/checkpoints/hidream_o1_image_dev_bf16.safetensors \
    --model_type dev \
    --prompt "remove the earphones" \
    --ref_images path/to/reference.png \
    --save_path path/to/output.png
```

Pass `--editing_scheduler flash` to force the flash scheduler for dev editing. Layout conditioning can be used with multiple references:

```bash
python src/musubi_tuner/hidream_o1_generate_image.py \
    --dit path/to/checkpoints/hidream_o1_image_bf16.safetensors \
    --prompt "City council members pose on a sunlit terrace" \
    --ref_images path/to/ref1.png path/to/ref2.png \
    --layout_bboxes "[[0.205, 0.488, 0.439, 0.742], [0.576, 0.801, 0.088, 0.342]]" \
    --save_path path/to/output.png
```
