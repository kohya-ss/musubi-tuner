# overview.md

This file provides guidance to AI agents and developers working with this repository.

## Project Overview

Musubi Tuner is a training and inference framework for LoRA (and LoHa/LoKr) fine-tuning of video/image generation models. It supports multiple architectures: HunyuanVideo, HunyuanVideo 1.5, Wan2.1/2.2, FramePack, FLUX.1 Kontext, FLUX.2, Kandinsky 5, Z-Image, and Qwen-Image series. Some architectures also support full fine-tuning.

The project prioritizes memory-efficient training through fp8 precision, block swapping (CPU offloading), gradient checkpointing, and VAE tiling/chunking.

## Installation

See `pyproject.toml` for dependencies. Requires Python 3.10+ and PyTorch 2.5.1+ with CUDA. Install via `pip install -e .` or `uv run --extra cu124` (experimental).

## Workflow

Training follows a three-step pipeline, each with per-architecture scripts:

1. **Cache latents** — `cache_latents.py`, `wan_cache_latents.py`, etc.
2. **Cache text encoder outputs** — `cache_text_encoder_outputs.py`, `wan_cache_text_encoder_outputs.py`, etc.
3. **Train** — `hv_train_network.py`, `wan_train_network.py`, etc. (launched via `accelerate launch`)

Inference scripts follow the same naming pattern: `hv_generate_video.py`, `wan_generate_video.py`, etc.

Utility scripts: `merge_lora.py`, `convert_lora.py`, `lora_post_hoc_ema.py`.

## Code Architecture

```
src/musubi_tuner/
├── training/          # Shared training infrastructure (trainer base class, accelerator setup, timesteps, etc.)
├── dataset/           # Dataset configuration, loading, bucketing, caching I/O
├── networks/          # LoRA/LoHa/LoKr network implementations per architecture
├── modules/           # Shared modules (attention, LR schedulers, fp8 optimization, offloading)
├── utils/             # Common utilities (model loading, device management, safetensors, etc.)
├── hunyuan_model/     # HunyuanVideo model implementation
├── hunyuan_video_1_5/ # HunyuanVideo 1.5 model implementation
├── wan/               # Wan2.1/2.2 model implementation
├── frame_pack/        # FramePack model implementation
├── flux/              # FLUX.1 Kontext model implementation
├── flux_2/            # FLUX.2 model implementation
├── kandinsky5/        # Kandinsky 5 model implementation
├── qwen_image/        # Qwen-Image model implementation
├── zimage/            # Z-Image model implementation
└── *_train_network.py, *_cache_*.py, *_generate_*.py  # Per-architecture entry points
```

Each architecture follows the same pattern: an architecture-specific subdirectory for model code, plus top-level scripts for caching/training/inference that share the common `training/` and `dataset/` infrastructure.

## Development Notes

- No formal test suite — manual testing via training and inference scripts
- Uses `accelerate` for distributed training
- Dataset configuration uses TOML format (see `docs/dataset_config.md` for schema)
- User-facing documentation is in `docs/`
