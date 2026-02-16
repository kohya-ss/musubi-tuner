# Blissful-Tuner LoRA Training Configuration Reference

A comprehensive guide for configuring LoRA training in blissful-tuner, covering all features, network arguments, and optimization techniques.

**Last Updated:** 2026-02-16

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration File Structure](#configuration-file-structure)
3. [Network Configuration](#network-configuration)
   - [Basic LoRA Settings](#basic-lora-settings)
   - [RS-LoRA (Rank-Stabilized LoRA)](#rs-lora-rank-stabilized-lora)
   - [DoRA (Weight-Decomposed Low-Rank Adaptation)](#dora-weight-decomposed-low-rank-adaptation)
   - [LoRA+](#lora-1)
   - [Module Selection Patterns](#module-selection-patterns)
4. [Optimizer Configuration](#optimizer-configuration)
   - [AdamW](#adamw)
   - [Schedule-Free Optimizers](#schedule-free-optimizers)
   - [Learning Rate Schedulers](#learning-rate-schedulers)
5. [Training Configuration](#training-configuration)
   - [Basic Training Parameters](#basic-training-parameters)
   - [Timestep Sampling](#timestep-sampling)
   - [Gradient Management](#gradient-management)
6. [Dataset Configuration](#dataset-configuration)
   - [Image Datasets](#image-datasets)
   - [Weighted Mask Loss](#weighted-mask-loss)
7. [Performance Optimization](#performance-optimization)
   - [torch.compile](#torchcompile)
   - [CuTE Attention](#cute-attention)
   - [FP8 Optimization](#fp8-optimization)
8. [Sampling During Training](#sampling-during-training)
9. [Complete Configuration Examples](#complete-configuration-examples)
10. [Quick Reference Tables](#quick-reference-tables)

---

## Overview

This document serves as a reference for configuring LoRA training in blissful-tuner. It covers:

- **Network arguments** (`network_args`): Control LoRA behavior including RS-LoRA, DoRA, LoRA+
- **Optimizer settings**: Learning rates, weight decay, schedulers
- **Training parameters**: Steps, accumulation, timestep sampling
- **Dataset configuration**: Images, masks, bucketing
- **Performance optimizations**: torch.compile, CuTE attention, FP8

### Supported Architectures

| Architecture | Training Script | Network Module |
|--------------|----------------|----------------|
| WAN 2.1/2.2 | `wan_train_network.py` | `networks.lora_wan` |
| HunyuanVideo | `hv_train_network.py` | `networks.lora` |
| HunyuanVideo 1.5 | `hv_1_5_train_network.py` | `networks.lora_hv_1_5` |
| FramePack | `fpack_train_network.py` | `networks.lora_framepack` |
| FLUX.1 Kontext | `flux_kontext_train_network.py` | `networks.lora_flux` |
| FLUX.2 (Dev, Klein 4B, Klein 9B) | `flux_2_train_network.py` | `networks.lora_flux_2` |
| Qwen-Image | `qwen_image_train_network.py` | `networks.lora_qwen_image` |
| Z-Image | `zimage_train_network.py` | `networks.lora_zimage` |
| Kandinsky5 | `kandinsky5_train_network.py` | `networks.lora_kandinsky` |

You can also use LoHa / LoKr by setting `network_module` to `networks.loha` or `networks.lokr`. See `docs/loha_lokr.md`.

---

## Configuration File Structure

Training uses TOML configuration files organized into logical sections:

```toml
# =============================================================================
# MODEL PATHS
# =============================================================================
[model]
dit = "/path/to/model.safetensors"
text_encoder = "/path/to/text_encoder.safetensors"
vae = "/path/to/vae.safetensors"

# =============================================================================
# DATASET
# =============================================================================
[dataset]
dataset_config = "/path/to/dataset.toml"

# =============================================================================
# NETWORK (LoRA Configuration)
# =============================================================================
[network]
network_module = "networks.lora_qwen_image"
network_dim = 64
network_alpha = 32
network_args = [
  "use_rslora=True",
  "use_dora=True",
  "loraplus_lr_ratio=8",
]

# =============================================================================
# OPTIMIZER
# =============================================================================
[optimizer]
optimizer_type = "adamw"
learning_rate = 5e-5
optimizer_args = ["betas=(0.9, 0.99)", "weight_decay=0.01"]
lr_scheduler = "cosine_with_min_lr"
lr_warmup_steps = 100

# =============================================================================
# TRAINING
# =============================================================================
[training]
mixed_precision = "bf16"
max_train_steps = 4000
save_every_n_steps = 250
gradient_accumulation_steps = 1

# =============================================================================
# OUTPUT
# =============================================================================
[output]
output_dir = "/path/to/output"
output_name = "my_lora"
logging_dir = "/path/to/logs"
```

---

## Network Configuration

### Basic LoRA Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network_module` | string | required | LoRA module (e.g., `networks.lora_qwen_image`) |
| `network_dim` | int | 32 | Rank (r) of LoRA matrices. Higher = more capacity |
| `network_alpha` | float | 16 | Scaling factor. Effective scale = `alpha / dim` |
| `network_weights` | string | None | Path to pretrained LoRA weights to resume from |

**Relationship between dim and alpha:**
- Scale factor = `alpha / dim` (standard LoRA)
- Scale factor = `alpha / sqrt(dim)` (RS-LoRA)
- `alpha = 0` or `alpha = None` means "no scaling" (scale = 1.0)

```toml
[network]
network_module = "networks.lora_qwen_image"
network_dim = 64      # Higher rank = more parameters, more capacity
network_alpha = 32    # scale = 32/64 = 0.5 for standard LoRA
```

---

### LoHa (Low-rank Hadamard Product)

LoHa is an alternative PEFT method that uses Hadamard (element-wise) products instead of standard matrix multiplication. The weight delta is computed as:

```
ΔW = (W1_a @ W1_b) ⊙ (W2_a @ W2_b)
```

where `⊙` is element-wise multiplication. This provides different expressivity than standard LoRA.

**Supported Architectures:**
- HunyuanVideo (`hv`)
- HunyuanVideo 1.5 (`hv15`)
- WAN (`wan`)
- FramePack (`fp`)
- FLUX Kontext (`fk`)
- FLUX.2 (Dev, Klein 4B, Klein 9B) (`f2d`, `f2k4`, `f2k9`)
- Qwen-Image / Qwen-Image-Edit / Qwen-Image-Layered (`qi`, `qie`, `qil`)
- Kandinsky5 (`k5`)
- Z-Image (`zi`)

**When to use:**
- Experimenting with alternative PEFT methods
- When standard LoRA doesn't capture desired adaptations
- Research into different low-rank parameterizations

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `network_module` | string | `networks.loha` | Use LoHa module |
| `network_dim` | int | 8 | Rank of LoHa matrices |
| `network_alpha` | float | 1 | Scaling factor |

**Example:**
```toml
[network]
network_module = "networks.loha"
network_dim = 8
network_alpha = 1
```

**Inference:** LoHa weights are natively detected and merged by all generation scripts — use them exactly like LoRA weights with `--lora_weight`. See `docs/loha_lokr.md` for details.

---

### RS-LoRA (Rank-Stabilized LoRA)

RS-LoRA changes the scaling formula from `alpha/r` to `alpha/sqrt(r)` for more stable gradients across different ranks.

**When to use:**
- Training with higher ranks (dim > 32)
- When gradient magnitudes vary too much across layers
- When you want more consistent training dynamics

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_rslora` | bool | False | Enable RS-LoRA scaling |

**Example:**
```toml
[network]
network_dim = 128
network_alpha = 64
network_args = [
  "use_rslora=True",
]
```

**Scaling comparison (dim=128, alpha=64):**
- Standard LoRA: scale = 64/128 = 0.5
- RS-LoRA: scale = 64/sqrt(128) = 64/11.31 ≈ 5.66

**Technical notes:**
- Weights saved with RS-LoRA include a `use_rslora_flag` buffer for auto-detection
- Loading RS-LoRA weights with `use_rslora=False` (or vice versa) raises a hard error
- If `alpha=0` with RS-LoRA, alpha is set to `sqrt(dim)` so scale = 1.0

---

### DoRA (Weight-Decomposed Low-Rank Adaptation)

DoRA decomposes weight updates into **magnitude** and **direction** components, adding a trainable magnitude vector per layer. This can improve training quality and convergence.

**When to use:**
- When standard LoRA produces inconsistent quality
- For fine-grained control over weight magnitudes
- Training style or identity LoRAs where magnitude matters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_dora` | bool | False | Enable DoRA magnitude decomposition |

**Example:**
```toml
[network]
network_dim = 64
network_alpha = 32
network_args = [
  "use_dora=True",
]
```

**Limitations:**
| Constraint | Behavior |
|------------|----------|
| **Linear layers only** | DoRA is disabled for all Conv layers |
| **No split_dims** | DoRA is disabled when split_dims is used |
| **No dropout/rank_dropout** | DoRA is disabled when these are set (module_dropout is OK) |

**When DoRA is disabled for a layer, it falls back to standard LoRA silently.** A summary is logged at network creation showing how many modules have DoRA enabled vs disabled.

**Technical notes:**
- DoRA adds a `dora_layer.weight` parameter per module (shape: `[out_features]`)
- Uses memory-efficient norm computation without materializing B@A
- Loading weights with DoRA enabled but missing magnitude vectors raises a hard error
- Loading weights with DoRA disabled but containing DoRA magnitudes logs a warning (and does not allow the `use_dora_flag` buffer to flip to `True`)
- `lora_multiplier=0` is treated as a true no-op (DoRA included): base output/weights are unchanged

**Combining RS-LoRA and DoRA:**
```toml
[network]
network_args = [
  "use_rslora=True",
  "use_dora=True",
]
```

---

### LoRA+

LoRA+ increases the learning rate of the LoRA-B (up) matrix relative to LoRA-A (down) matrix for faster convergence.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loraplus_lr_ratio` | float | 1.0 | Multiplier for LoRA-B learning rate |

**Effective learning rates:**
- LoRA-A (down): `learning_rate`
- LoRA-B (up): `learning_rate * loraplus_lr_ratio`

**Recommendations:**
- Start with ratio of 4-8 for most use cases
- Original paper recommends 16, but lower values are often more stable
- Higher ratios = faster convergence but potentially less stable

**Example:**
```toml
[network]
network_args = [
  "loraplus_lr_ratio=8",  # LoRA-B LR = base_lr * 8
]

[optimizer]
learning_rate = 5e-5      # Base LR
# Effective: LoRA-A = 5e-5, LoRA-B = 4e-4
```

---

### Module Selection Patterns

Control which layers receive LoRA adapters using regex patterns.

| Parameter | Type | Description |
|-----------|------|-------------|
| `exclude_patterns` | list | Modules matching these patterns are excluded |
| `include_patterns` | list | Only modules matching these patterns are included |
| `exclude_mod` | bool | For Qwen-Image: exclude img_mod/txt_mod layers (default: True) |

**Default exclusions:**
By default, modulation layers (`img_mod`, `txt_mod`, `modulation`) are excluded from LoRA.

**Examples:**

```toml
# Include modulation layers for Qwen-Image (useful for identity/persona training)
network_args = ["exclude_mod=False"]

# Only train double blocks
network_args = ["exclude_patterns=[r'.*single_blocks.*']"]

# Only train single blocks from index 10 onwards
network_args = [
  "exclude_patterns=[r'.*']",
  "include_patterns=[r'.*single_blocks\\.\\d{2}\\.linear.*']"
]
```

---

## Optimizer Configuration

### AdamW

The standard optimizer for LoRA training.

```toml
[optimizer]
optimizer_type = "adamw"
learning_rate = 5e-5
optimizer_args = [
  "betas=(0.9, 0.99)",
  "weight_decay=0.01",
  "eps=1e-8",
]
```

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `learning_rate` | 1e-5 to 1e-4 | Base learning rate |
| `betas` | (0.9, 0.99) | Adam momentum parameters |
| `weight_decay` | 0.01 | L2 regularization strength |
| `eps` | 1e-8 | Numerical stability epsilon |

### Schedule-Free Optimizers

Alternative optimizers that don't require LR scheduling:

```toml
[optimizer]
optimizer_type = "schedulefree.AdamWScheduleFree"
learning_rate = 1e-4
optimizer_args = ["weight_decay=0.01", "betas=(0.9,0.95)"]
```

### Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| `constant` | Fixed learning rate |
| `cosine` | Cosine annealing to 0 |
| `cosine_with_min_lr` | Cosine annealing with minimum LR floor |
| `rex` | REX scheduler (gradual decrease) |
| `polynomial` | Polynomial decay |

**Cosine with minimum LR (recommended):**
```toml
[optimizer]
lr_scheduler = "cosine_with_min_lr"
lr_warmup_steps = 100
lr_scheduler_min_lr_ratio = 0.1  # Final LR = 10% of initial
```

**REX scheduler:**
```toml
[optimizer]
lr_scheduler = "rex"
lr_scheduler_args = ["rex_alpha=0.1", "rex_beta=0.9"]
lr_warmup_steps = 0
lr_scheduler_min_lr_ratio = 0.01
```

---

## Training Configuration

### Basic Training Parameters

```toml
[training]
mixed_precision = "bf16"          # bf16 recommended for modern GPUs
gradient_accumulation_steps = 1   # Effective batch = batch_size * accumulation
seed = 42                         # Reproducibility

max_train_steps = 4000            # Total training steps
save_every_n_steps = 250          # Checkpoint frequency
```

### Timestep Sampling

Controls which noise levels are emphasized during training.

| Method | Description | Use Case |
|--------|-------------|----------|
| `qwen_shift` | Qwen-specific shifted sampling | Qwen-Image models |
| `flux_shift` | FLUX-specific shifted sampling | FLUX models |
| `sigma` | Standard sigma sampling | General use |
| `logsnr` | Style-friendly high-noise focus | Style training |
| `qinglong_qwen` | Hybrid (80% qwen_shift + 20% style) | Balanced style |

```toml
[training]
timestep_sampling = "qwen_shift"
sigmoid_scale = 1.0
discrete_flow_shift = 2.2         # Qwen default
```

**For style-focused training:**
```toml
[training]
timestep_sampling = "logsnr"
logit_mean = -6.0
logit_std = 2.0
```

### Gradient Management

```toml
[training]
gradient_checkpointing = false    # Trade compute for memory
scale_weight_norms = 2.0          # Prevents LoRA weight drift
```

---

## Dataset Configuration

### Image Datasets

Create a separate TOML file for dataset configuration:

```toml
# dataset.toml
[general]
caption_extension = ".txt"
batch_size = 4
enable_bucket = false             # Disable for single resolution
bucket_no_upscale = false

[[datasets]]
image_directory = "/path/to/images/subject"
cache_directory = "/path/to/cache"
mask_directory = "/path/to/weighted_masks"  # Optional
resolution = [1328, 1328]
num_repeats = 1
```

**Resolution recommendations by architecture:**

| Architecture | Resolutions |
|--------------|-------------|
| Qwen-Image | 1328×1328 (1:1), 1664×928 (16:9), 928×1664 (9:16) |
| WAN 2.1/2.2 | 960×544, 544×960, 720×720 |
| FLUX | 1024×1024, 1280×768 |

### Weighted Mask Loss

For selective learning on different image regions (face vs body vs background).

**Mask values:**
| Region | Grayscale Value | Normalized Weight |
|--------|-----------------|-------------------|
| Face | 255 | 1.000 |
| Body | 128 | 0.502 |
| Hair | 80 | 0.314 |
| Background | 0 | 0.000 |

**Training configuration:**
```toml
[training]
use_mask_loss = true
mask_gamma = 0.7              # <1 = softer, >1 = sharper
mask_min_weight = 0.05        # Minimum weight for all regions
```

**Dataset configuration:**
```toml
[[datasets]]
image_directory = "/path/to/subject"
mask_directory = "/path/to/weighted_masks"
cache_directory = "/path/to/cache"
```

**Recommended settings by training goal:**

| Goal | mask_gamma | mask_min_weight |
|------|------------|-----------------|
| Person/character LoRA | 0.7 | 0.05 |
| Face-focused | 1.5 | 0.0 |
| Style/clothing | 0.5 | 0.1 |

---

## Performance Optimization

### torch.compile

JIT compilation for faster training (10-25% speedup on modern GPUs).

```toml
[advanced]
compile = true
compile_mode = "max-autotune-no-cudagraphs"
compile_cache_size_limit = 128
cuda_allow_tf32 = true
cuda_cudnn_benchmark = true
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `compile` | false | Enable torch.compile |
| `compile_mode` | "default" | Compilation mode |
| `compile_cache_size_limit` | 32 | Graph cache size |
| `cuda_allow_tf32` | false | TF32 on Ampere+ GPUs |
| `cuda_cudnn_benchmark` | false | cuDNN autotuning |

**Compile modes:**
- `default`: Balanced (recommended for training)
- `max-autotune-no-cudagraphs`: Best optimization without CUDA graphs
- `reduce-overhead`: Minimize Python overhead

### CuTE Attention

CUTLASS-based attention for Blackwell/Hopper GPUs (30-70% faster than FA2).

```toml
[training]
cute = true
```

**Requirements:**
- NVIDIA Blackwell (SM 10.3) or Hopper (SM 9.0+)
- flash-attention 2.8.3+ with CuTE support
- quack-kernels >= 0.2.4

**Environment setup:**
```bash
export CUTE_DSL_CACHE_DIR="/root/.cache/cute_dsl"
mkdir -p "${CUTE_DSL_CACHE_DIR}"
```

### FP8 Optimization

Reduce VRAM usage with FP8 quantized base weights.

```toml
[advanced]
fp8_base = true
fp8_scaled = true
```

**Note:** Input checkpoints must be FP16/BF16. Pre-quantized FP8 weights cannot be used.

---

## Sampling During Training

Generate sample images during training to monitor progress.

```toml
[sampling]
sample_prompts = "/path/to/sample_prompts.txt"
sample_every_n_steps = 250
sample_at_first = true
```

**Sample prompts file format:**
```text
# Comments start with #
# Minimal test
DLAY, a man, portrait photo. --w 1328 --h 1328 --s 50 --g 4.0 --fs 2.2 --d 10000

# Detailed scenario
DLAY, a man wearing a navy blazer, at a formal event. --w 1328 --h 1328 --s 50 --g 4.0 --fs 2.2 --d 10001
```

**Parameters:**
- `--w/--h`: Width/height
- `--s`: Inference steps
- `--g`: Guidance scale (CFG)
- `--fs`: Discrete flow shift
- `--d`: Seed

---

## Complete Configuration Examples

### Example 1: Person/Identity LoRA (Qwen-Image)

High-quality persona training with RS-LoRA, LoRA+, and masked loss.

**Main config (`train_config.toml`):**
```toml
# =============================================================================
# MODEL
# =============================================================================
[model]
dit = "/root/models/Qwen_Image_2512_BF16.safetensors"
text_encoder = "/root/models/qwen_2.5_vl_7b_bf16.safetensors"
vae = "/root/models/qwen_train_vae.safetensors"
model_version = "original"

# =============================================================================
# DATASET
# =============================================================================
[dataset]
dataset_config = "/path/to/dataset.toml"

# =============================================================================
# NETWORK - RS-LoRA + LoRA+ + Include Modulation
# =============================================================================
[network]
network_module = "networks.lora_qwen_image"
network_dim = 64
network_alpha = 32
network_args = [
  "use_rslora=True",
  "loraplus_lr_ratio=8",
  "exclude_mod=False",
]

# =============================================================================
# OPTIMIZER - Conservative AdamW with LoRA+
# =============================================================================
[optimizer]
optimizer_type = "adamw"
learning_rate = 5e-5
optimizer_args = ["betas=(0.9, 0.99)", "weight_decay=0.01", "eps=1e-8"]
lr_scheduler = "cosine_with_min_lr"
lr_warmup_steps = 100
lr_scheduler_min_lr_ratio = 0.1

# =============================================================================
# TRAINING
# =============================================================================
[training]
mixed_precision = "bf16"
gradient_accumulation_steps = 1
seed = 42
max_train_steps = 4000
save_every_n_steps = 250

timestep_sampling = "qwen_shift"
sigmoid_scale = 1.0
discrete_flow_shift = 2.2

scale_weight_norms = 2.0
use_mask_loss = true
mask_gamma = 0.7
mask_min_weight = 0.05

# =============================================================================
# SAMPLING
# =============================================================================
[sampling]
sample_prompts = "/path/to/sample_prompts.txt"
sample_every_n_steps = 250
sample_at_first = true

# =============================================================================
# OUTPUT
# =============================================================================
[output]
output_dir = "/root/output/my_persona_lora"
output_name = "persona_lora"
logging_dir = "/root/output/my_persona_lora/logs"

# =============================================================================
# ADVANCED - torch.compile + CuTE
# =============================================================================
[advanced]
compile = true
compile_mode = "max-autotune-no-cudagraphs"
compile_cache_size_limit = 128
cuda_allow_tf32 = true
cuda_cudnn_benchmark = true
cute = true
```

**Dataset config (`dataset.toml`):**
```toml
[general]
caption_extension = ".txt"
batch_size = 4
enable_bucket = false

[[datasets]]
image_directory = "/root/DATASETS/subject"
cache_directory = "/root/DATASETS/cache"
mask_directory = "/root/DATASETS/weighted_masks"
resolution = [1328, 1328]
num_repeats = 1
```

### Example 2: Style LoRA with DoRA

Style training focusing on artistic characteristics.

```toml
[network]
network_module = "networks.lora_qwen_image"
network_dim = 32
network_alpha = 16
network_args = [
  "use_dora=True",
  "loraplus_lr_ratio=4",
]

[training]
timestep_sampling = "logsnr"
logit_mean = -6.0
logit_std = 2.0
max_train_steps = 2000
```

### Example 3: High-Rank RS-LoRA + DoRA Combined

Maximum expressiveness with stability.

```toml
[network]
network_module = "networks.lora_wan"
network_dim = 128
network_alpha = 64
network_args = [
  "use_rslora=True",
  "use_dora=True",
  "loraplus_lr_ratio=8",
]

[optimizer]
learning_rate = 2e-5          # Lower LR for high rank
lr_scheduler = "rex"
lr_scheduler_args = ["rex_alpha=0.1", "rex_beta=0.9"]
```

---

## Quick Reference Tables

### network_args Summary

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_rslora` | bool | False | RS-LoRA scaling (alpha/sqrt(r)) |
| `use_dora` | bool | False | DoRA magnitude decomposition (Linear only) |
| `loraplus_lr_ratio` | float | 1.0 | LoRA-B learning rate multiplier |
| `exclude_mod` | bool | True | Exclude modulation layers (Qwen-Image) |
| `exclude_patterns` | list | [] | Regex patterns to exclude modules |
| `include_patterns` | list | [] | Regex patterns to include modules |
| `verbose` | bool | False | Print detailed LoRA info |

### Training Parameter Defaults

| Parameter | Default | Recommended Range |
|-----------|---------|-------------------|
| `network_dim` | 32 | 16-128 |
| `network_alpha` | 16 | dim/2 to dim |
| `learning_rate` | 1e-4 | 1e-5 to 1e-4 |
| `max_train_steps` | - | 1000-10000 |
| `batch_size` | 1 | 1-8 |
| `gradient_accumulation_steps` | 1 | 1-8 |

### Feature Compatibility Matrix

| Feature | Linear | Conv1x1 | Conv3x3 | With dropout | With split_dims |
|---------|--------|---------|---------|--------------|-----------------|
| Standard LoRA | ✅ | ✅ | ✅ | ✅ | ✅ |
| RS-LoRA | ✅ | ✅ | ✅ | ✅ | ✅ |
| DoRA | ✅ | ❌ | ❌ | ❌ | ❌ |
| LoRA+ | ✅ | ✅ | ✅ | ✅ | ✅ |

### Troubleshooting Quick Reference

| Issue | Possible Cause | Solution |
|-------|----------------|----------|
| RS-LoRA flag mismatch error | Loading RS-LoRA weights without flag | Add `use_rslora=True` to network_args |
| DoRA flag mismatch error | Loading DoRA weights without flag | Add `use_dora=True` to network_args |
| DoRA silently disabled | Conv layers or dropout enabled | Check logs for DoRA summary |
| Unstable training | Learning rate too high | Lower LR or use RS-LoRA |
| Slow convergence | Learning rate too low | Increase LoRA+ ratio or LR |
| Out of memory | Batch size or rank too high | Reduce batch_size or network_dim |

---

## Changelog

### 2026-01-16
- Added RS-LoRA (Rank-Stabilized LoRA) documentation
- Added DoRA (Weight-Decomposed Low-Rank Adaptation) documentation
- Initial document creation with comprehensive configuration reference

---

*Document created: 2026-01-16*
*For blissful-tuner LoRA training*
