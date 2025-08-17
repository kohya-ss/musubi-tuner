# overview.md

This file provides guidance to developers when working with code in this repository.

## Project Overview

Musubi Tuner is a Python-based training framework for LoRA (Low-Rank Adaptation) models with multiple video generation architectures including HunyuanVideo, Wan2.1, FramePack, and FLUX.1 Kontext. The project focuses on memory-efficient training and inference for video generation models.

## Installation and Environment

The project uses `pyproject.toml` for dependency management with both pip and uv (experimental) installation methods:

- **pip installation**: `pip install -e .` after installing PyTorch with CUDA support
- **uv installation**: `uv run --extra cu124` or `uv run --extra cu128` (experimental)
- **Python requirement**: 3.10 or later (verified with 3.10)
- **PyTorch requirement**: 2.5.1 or later

Optional dependencies include `ascii-magic`, `matplotlib`, `tensorboard`, and `prompt-toolkit`.

## Common Development Commands

### Dataset Preparation
```bash
# Cache latents (required before training)
python src/musubi_tuner/cache_latents.py --dataset_config path/to/toml --vae path/to/vae --vae_chunk_size 32 --vae_tiling

# Cache text encoder outputs (required before training)
python src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/toml --text_encoder1 path/to/te1 --text_encoder2 path/to/te2 --batch_size 16
```

### Training Commands
```bash
# HunyuanVideo training
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py --dit path/to/dit --dataset_config path/to/toml --network_module networks.lora --network_dim 32

# Wan2.1 training
python src/musubi_tuner/wan_train_network.py [similar args]

# FramePack training
python src/musubi_tuner/fpack_train_network.py [similar args]

# FLUX.1 Kontext training
python src/musubi_tuner/flux_kontext_train_network.py [similar args]
```

### Inference Commands
```bash
# HunyuanVideo inference
python src/musubi_tuner/hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --prompt "text" --dit path/to/dit --vae path/to/vae

# Wan2.1 inference
python src/musubi_tuner/wan_generate_video.py [similar args]

# FramePack inference
python src/musubi_tuner/fpack_generate_video.py [similar args]
```

### Utility Commands
```bash
# Merge LoRA weights
python src/musubi_tuner/merge_lora.py --dit path/to/dit --lora_weight path/to/lora.safetensors --save_merged_model path/to/output

# Convert LoRA formats
python src/musubi_tuner/convert_lora.py --input path/to/lora.safetensors --output path/to/converted.safetensors --target other

# Post-hoc EMA for LoRA
python src/musubi_tuner/lora_post_hoc_ema.py [args]
```

### Testing and Development
No formal test suite is present in this repository. The project relies on manual testing through the training and inference scripts.

## Code Architecture

### Core Structure
- `src/musubi_tuner/`: Main package containing all training and inference scripts
- `src/musubi_tuner/dataset/`: Dataset configuration and loading utilities
- `src/musubi_tuner/networks/`: LoRA network implementations for different architectures
- `src/musubi_tuner/utils/`: Common utilities for model handling, device management, etc.

### Architecture-Specific Modules
- `hunyuan_model/`: HunyuanVideo model implementation and utilities
- `wan/`: Wan2.1 model configurations and modules
- `frame_pack/`: FramePack model implementation and utilities
- `flux/`: FLUX model utilities

### Key Components
- **Dataset Configuration**: Uses TOML files for complex dataset setups supporting images, videos, control images, and metadata JSONL files
- **Memory Optimization**: Supports fp8 precision, block swapping, and various attention mechanisms (SDPA, FlashAttention, SageAttention, xformers)
- **Multi-Architecture Support**: Each architecture has its own training/inference scripts with shared utilities
- **LoRA Networks**: Modular LoRA implementations with support for different target modules and configurations

### Configuration System
- Dataset configuration uses TOML format with support for multiple datasets, bucketing, and architecture-specific settings
- Training configuration via command line arguments and accelerate config
- Support for advanced features like timestep sampling, discrete flow shift, and memory-saving options

### Memory Management
- Aggressive memory optimization with options like `--blocks_to_swap`, `--fp8_base`, `--fp8_llm`
- VAE tiling support for handling large resolutions
- Gradient checkpointing and mixed precision training

## Development Notes
- The project is under active development with experimental features
- No formal CI/CD or automated testing
- Uses accelerate for distributed training setup
- Supports both interactive and batch inference modes
- Comprehensive documentation in `docs/` directory for advanced configurations and architecture-specific guides