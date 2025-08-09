# LORAID System: Deterministic Parameter Space Allocation for LoRA

> **⚠️ Model Compatibility**: This LORAID implementation has been tested exclusively on **Wan 2.2 (14B parameter DiT model)**. It may require modifications for other models like Hunyuan, as block counts and architecture patterns could differ.

Note: Code was made written by Claude Code with a whip. Still testing some stuff, but training, inferencing and merging works. The basic idea was for this, to be able to train a big dataset in slices. If I would merge my loras together or back to the DiT they would overwrite their knowledge. I wanted a way to prevent that. You can think of this as a nice experimental project :) With love!



## 🎯 **Why LORAID?**

### **Problem: Catastrophic Forgetting with Traditional LoRA Merging**
When training 200+ LoRAs and merging them sequentially into a DiT model, traditional approaches suffer from:
- **Catastrophic Forgetting**: Later LoRAs overwrite knowledge from earlier LoRAs
- **Parameter Conflicts**: Multiple LoRAs compete for the same parameter space
- **Knowledge Loss**: Previously learned character identities get degraded over time
- **Unpredictable Results**: No guarantee which LoRA's knowledge will survive the merging process

### **Solution: Deterministic Parameter Space Allocation**
LORAID (LoRA ID) system provides:
- ✅ **Zero Parameter Conflicts**: Each LoRA targets non-overlapping model parameters
- ✅ **Preserved Knowledge**: All LoRAs retain their learned features indefinitely
- ✅ **Scalable Training**: Support for 240+ independent LoRAs
- ✅ **Predictable Behavior**: Deterministic parameter allocation based on LORAID
- ✅ **Composable Effects**: Multiple LoRAs can work together without interference
- ✅ **Hardware Efficiency**: Train large datasets in manageable slices

---

## 🏗️ **Architecture Overview**

### **Parameter Space Allocation**
The DiT model has **240 transformer blocks** (blocks 0-239). LORAID allocates these blocks deterministically:

```
Model: 240 blocks total (blocks.0 through blocks.239)
├── LORAID 1:  blocks 0-11    (12 blocks)
├── LORAID 2:  blocks 12-23   (12 blocks)  
├── LORAID 3:  blocks 24-35   (12 blocks)
├── ...
└── LORAID 20: blocks 228-239 (12 blocks)
```

**Scalability Options:**
- **12 blocks per LORAID**: 20 LORAIDs maximum (current default)
- **3 blocks per LORAID**: 80 LORAIDs maximum  
- **2 blocks per LORAID**: 120 LORAIDs maximum
- **1 block per LORAID**: 240 LORAIDs maximum

Each block contains ~10 weight matrices (self_attn, cross_attn, ffn) with millions of parameters per LORAID.

### **How It Works**
1. **Training**: Each LORAID targets only its designated blocks using include/exclude patterns
2. **Inference**: Multiple LORAIDs can be loaded simultaneously without conflicts
3. **Merging**: LoRAs merge into their designated parameter spaces only
4. **Composition**: Neural network processes through multiple modified block ranges

---

## 📁 **Main Modified Files**

### **Core Training Files**
- `src/musubi_tuner/wan_train_network.py` - LORAID argument parsing and parameter allocation
- `src/musubi_tuner/networks/lora_wan.py` - LoRA network creation with LORAID filtering
- `src/musubi_tuner/networks/lora.py` - Include/exclude pattern logic and metadata storage

### **Inference Files**  
- `src/musubi_tuner/wan_generate_video.py` - Multi-LORAID inference and merging
- `src/musubi_tuner/utils/lora_utils.py` - LORAID compatibility validation and loading

### **Post-Processing Files**
- `src/musubi_tuner/lora_post_hoc_ema.py` - LORAID-aware EMA and democratic blending - EXPERIMENTAL -

---

## 🛠️ **Implementation Details**

### **1. Parameter Pattern Generation**
```python
def get_loraid_parameter_patterns(loraid):
    """Generate include patterns for LORAID parameter allocation"""
    total_blocks = 240
    blocks_per_loraid = 12  # Configurable
    start_block = (loraid - 1) * blocks_per_loraid  
    end_block = min(start_block + blocks_per_loraid - 1, total_blocks - 1)
    
    include_patterns = []
    for block_id in range(start_block, end_block + 1):
        patterns = [
            f"blocks\\.{block_id}\\.self_attn\\.",
            f"blocks\\.{block_id}\\.cross_attn\\.",  
            f"blocks\\.{block_id}\\.ffn\\."
        ]
        include_patterns.extend(patterns)
    
    return include_patterns
```

### **2. Whitelist/Blacklist Logic** 
```python
# networks/lora.py - Fixed include/exclude precedence
if include_re_patterns:
    # 1. Inclusion check: Only modules in whitelist pass
    included = any(pattern.match(original_name) for pattern in include_re_patterns)
    if not included:
        continue

# 2. Exclusion check: Blacklist overrides whitelist  
excluded = any(pattern.match(original_name) for pattern in exclude_re_patterns)
if excluded:
    continue
```

### **3. LORAID Metadata Storage**
```python
# networks/lora.py - save_weights()
if hasattr(self, '_loraid_value') and self._loraid_value is not None:
    metadata["ss_loraid"] = str(self._loraid_value)
    logger.info(f"Added LORAID {self._loraid_value} to saved model metadata")
```

### **4. Multi-LORAID Argument Parsing**
```python
# wan_generate_video.py
parser.add_argument(
    "--LORAID", type=int, nargs="*", action="append", default=None,
    help="Supports '--LORAID 1 2 3' or '--LORAID 1 --LORAID 2 --LORAID 3'"
)

def parse_loraid_args(args):
    target_loraids = []
    for loraid_list in args.LORAID:
        if isinstance(loraid_list, list):
            target_loraids.extend(loraid_list)
        else:
            target_loraids.append(loraid_list)
    return list(dict.fromkeys(target_loraids))  # Remove duplicates
```

---

## 📚 **Usage Examples**

### **Training LORAIDs**
```bash
# Train LORAID 1 (targets blocks 0-11)
accelerate launch wan_train_network.py \
    --task t2v-A14B \
    --dit F:/models/wan22_t2v_14B.safetensors \
    --dataset_config F:/data/testset1_dataset.toml \
    --network_module networks.lora_wan \
    --network_dim 64 --network_alpha 64 \
    --output_name Sarah_512 \
    --LORAID 1

# Train LORAID 2 (targets blocks 12-23)  
accelerate launch wan_train_network.py \
    --task t2v-A14B \
    --dit F:/models/wan22_t2v_14B.safetensors \
    --dataset_config F:/data/testset2_dataset.toml \
    --network_module networks.lora_wan \
    --network_dim 64 --network_alpha 64 \
    --output_name Jessica_512 \
    --LORAID 2
```

### **Single LORAID Inference**
```bash
# Inference with only LORAID 1
python wan_generate_video.py \
    --task t2v-14B \
    --prompt "Sarah on a beach, standing, perfect" \
    --dit F:/models/wan22_t2v_14B.safetensors \
    --vae F:/models/wan_2.1_vae.safetensors \
    --t5 F:/models/t5-xxl.pth \
    --lora_weight F:/loras/Sarah_512-000010.safetensors \
    --lora_multiplier 1.0 \
    --LORAID 1 \
    --save_path F:/output/
```

### **Multi-LORAID Inference**
```bash
# Inference with multiple LORAIDs
python wan_generate_video.py \
    --task t2v-14B \
    --prompt "Sarah and jessica on a beach together" \
    --dit F:/models/wan22_t2v_14B.safetensors \
    --vae F:/models/wan_2.1_vae.safetensors \
    --t5 F:/models/t5-xxl.pth \
    --lora_weight F:/loras/Jessica_512-000010.safetensors,F:/loras/Sarah_512-000005.safetensors \
    --lora_multiplier 1.0 0.7 \
    --LORAID 1 2 \
    --save_path F:/output/
```

### **LoRA Multiplier Control**
```bash
# Fine-grained control over LoRA strength
# Only LORAID 1 active (Sarah only)
--LORAID 1 2 --lora_multiplier 1.0 0.0

# Only LORAID 2 active (Jessica only)  
--LORAID 1 2 --lora_multiplier 0.0 1.0

# Both at half strength
--LORAID 1 2 --lora_multiplier 0.5 0.5

# Mixed strengths
--LORAID 1 2 3 --lora_multiplier 1.0 0.7 0.3

# Lazy usage (defaults to 1.0 for all)
--LORAID 1 2 3 --lora_multiplier 1.0  # [1.0, 1.0, 1.0]
```

### **Model Merging**
```bash
# Merge multiple LORAIDs permanently into DiT
python wan_generate_video.py \
    --task t2v-14B \
    --dit F:/models/wan22_t2v_14B.safetensors \
    --lora_weight F:/loras/Sarah.safetensors,F:/loras/Jessica.safetensors \
    --lora_multiplier 1.0 1.0 \
    --LORAID 1 2 \
    --save_merged_model F:/models/merged.safetensors
```

### **Post-Hoc EMA with LORAID Preservation**
```bash
# EMA merge while preserving LORAID metadata, script automatically detects sibling and subfolder epochs
python lora_post_hoc_ema.py \
    --sigma_rel 0.2 \
    --lora1 F:/loras/Sarah_epoch_005.safetensors \
    --lora2 F:/loras/Sarah_epoch_010.safetensors \
    --output_file F:/loras/Sarah_ema.safetensors \
    --average 1.0 # Regular lora blendy merge

# Democratic knowledge sharing between LORAIDs, EXPERIMENTAL
python lora_post_hoc_ema.py \
    --sigma_rel 0.2 \
    --lora1 F:/loras/Sarah.safetensors \
    --lora2 F:/loras/jessica.safetensors \
    --output_file_lora1 F:/loras/Sarah_shared.safetensors \
    --output_file_lora2 F:/loras/jessica_shared.safetensors \
    --average 1.0 --democratic
```

---

## ⚙️ **Configuration Options**

### **Block Allocation Strategies**
Modify `blocks_per_loraid` in `wan_train_network.py`:

```python
# Conservative (20 LORAIDs max, robust learning, TESTED)
blocks_per_loraid = 12  

# Balanced (80 LORAIDs max, good learning, TESTED)  
blocks_per_loraid = 3

# Aggressive (120 LORAIDs max, questionable learning capacity)
blocks_per_loraid = 2

# Maximum (240 LORAIDs max, risky for character learning)
blocks_per_loraid = 1
```

### **Supported Argument Formats**
All LORAID arguments support dual formats:
```bash
# Space-separated format
--LORAID 1 2 3 4
--lora_multiplier 1.0 0.5 0.7 1.0

# Repeated flag format  
--LORAID 1 --LORAID 2 --LORAID 3 --LORAID 4
--lora_multiplier 1.0 --lora_multiplier 0.5
```

---

## 🔧 **Technical Benefits**

### **1. Zero Parameter Conflicts**
```bash
# Validation output confirms no overlaps
INFO: LoRA 0 contains parameters for blocks: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
INFO: LoRA 1 contains parameters for blocks: [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23] 
INFO: No block overlaps detected - LoRAs should be independent
```

### **2. Preserved Alpha Values**
Each LoRA maintains its own alpha/scale independently:
```python
# Each LoRA uses its own alpha - no interference
alpha_lora1 = lora1_sd.get("alpha", dim)  # e.g., 64
scale_lora1 = alpha_lora1 / dim           # e.g., 1.0

alpha_lora2 = lora2_sd.get("alpha", dim)  # e.g., 64
scale_lora2 = alpha_lora2 / dim           # e.g., 1.0

# Final weight = base_weight + delta_lora1 + delta_lora2
```

### **3. Deterministic Behavior**
- **LORAID 1** always targets blocks 0-11
- **LORAID 20** always targets blocks 228-239  
- **Reproducible** parameter allocation across training sessions
- **Predictable** inference behavior

### **4. Scalable Architecture**
- Train **240 independent character LoRAs** 
- **Hardware-limited training**: Process large datasets in slices
- **Memory efficiency**: Load only needed LoRAs during inference
- **Deployment flexibility**: Distribute individual LoRAs or merged models

---

## 🚨 **Important Notes**

### **Neural Network Composition Effects**
When multiple LORAIDs are active simultaneously:
- **Expected behavior**: Signal flows through multiple modified block ranges
- **Compositional output**: Natural blending of learned features through network processing
- **Not a bug**: This is the intended behavior of independent parameter modifications

### **Training Considerations**  
- **Undertrained LoRAs**: May not override base model biases effectively
- **Training duration**: Longer training produces stronger, more distinct features
- **Base model bias**: Strong base model preferences can leak through weak LoRA adaptations

### **Compatibility**
- **Backward compatible**: Works with existing LoRA workflows
- **Metadata preserved**: LORAID information survives EMA and democratic blending
- **Standard tools**: Compatible with existing LoRA utilities and formats

---

## 📊 **Performance Characteristics**

### **Memory Usage**
- **Training**: Same as standard LoRA (only target parameters loaded)
- **Inference**: Scales linearly with number of active LORAIDs
- **Merging**: No additional overhead

### **Computational Overhead**
- **Training**: Negligible (parameter filtering happens once)
- **Inference**: Minimal (standard LoRA application per active LORAID)
- **Merging**: Same as standard LoRA merging

### **Storage Requirements**
- **Per LoRA**: ~600-1700MB depending on rank and target blocks
- **Metadata**: <1KB additional per LoRA for LORAID information
- **Merged models**: Same size as standard merged models

---

## 🎯 **Use Cases**

### **Character Training Pipeline**
1. **Slice large datasets** by character identity
2. **Train individual LORAIDs** for each character (LORAID 1, 2, 3...)
3. **Validate independence** using single-LORAID inference
4. **Combine as needed** for multi-character scenes
5. **Merge permanently** for production deployment

### **Style Transfer Applications**  
- **Art styles**: Different LORAIDs for different artistic styles
- **Photography modes**: Portrait, landscape, macro each with dedicated LORAIDs
- **Content types**: Architecture, nature, people with separate parameter spaces

### **Fine-Grained Control**
- **Strength adjustment**: Individual multipliers per LORAID
- **Selective activation**: Enable/disable specific LORAIDs per generation
- **A/B testing**: Compare different LORAID combinations systematically

---

## 🔬 **Validation & Testing**

The LORAID system has been validated with:
- ✅ **Parameter space isolation**: No block overlaps detected  
- ✅ **Independent learning**: Single LORAID produces clean results
- ✅ **Multiplier control**: 0.0 multiplier completely disables LORAID
- ✅ **Metadata preservation**: LORAID survives all processing pipelines
- ✅ **Merging functionality**: Permanent weight merging works correctly
- ✅ **Scalability testing**: Supports 240+ theoretical LORAIDs

---

## 🚀 **Getting Started**

1. **Train your first LORAID**:
   ```bash
   --LORAID 1 --output_name character1
   ```

2. **Test single-character inference**:
   ```bash
   --LORAID 1 --lora_multiplier 1.0
   ```

3. **Train additional LORAIDs**:
   ```bash
   --LORAID 2 --output_name character2  
   ```

4. **Test multi-character composition**:
   ```bash
   --LORAID 1 2 --lora_multiplier 1.0 1.0
   ```

5. **Merge for production**:
   ```bash
   --LORAID 1 2 --save_merged_model merged.safetensors
   ```

The LORAID system transforms LoRA training from a conflicted parameter sharing approach to a clean, deterministic parameter space allocation system, enabling scalable character training without catastrophic forgetting.


# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

## Table of Contents

<details>
<summary>Click to expand</summary>

- [Musubi Tuner](#musubi-tuner)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Support the Project](#support-the-project)
    - [Recent Updates](#recent-updates)
    - [Releases](#releases)
  - [Overview](#overview)
    - [Hardware Requirements](#hardware-requirements)
    - [Features](#features)
  - [Installation](#installation)
    - [pip based installation](#pip-based-installation)
    - [uv based installation](#uv-based-installation)
    - [Linux/MacOS](#linuxmacos)
    - [Windows](#windows)
  - [Model Download](#model-download)
    - [Use the Official HunyuanVideo Model](#use-the-official-hunyuanvideo-model)
    - [Using ComfyUI Models for Text Encoder](#using-comfyui-models-for-text-encoder)
  - [Usage](#usage)
    - [Dataset Configuration](#dataset-configuration)
    - [Latent Pre-caching](#latent-pre-caching)
    - [Text Encoder Output Pre-caching](#text-encoder-output-pre-caching)
    - [Configuration of Accelerate](#configuration-of-accelerate)
    - [Training](#training)
    - [Merging LoRA Weights](#merging-lora-weights)
    - [Inference](#inference)
    - [Inference with SkyReels V1](#inference-with-skyreels-v1)
    - [Convert LoRA to another format](#convert-lora-to-another-format)
  - [Miscellaneous](#miscellaneous)
    - [SageAttention Installation](#sageattention-installation)
    - [PyTorch version](#pytorch-version)
  - [Disclaimer](#disclaimer)
  - [Contributing](#contributing)
  - [License](#license)

</details>

## Introduction

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo, Wan2.1/2.2, FramePack and FLUX.1 Kontext architectures. 

This repository is unofficial and not affiliated with the official HunyanVideo/Wan2.1/2.2/FramePack/FLUX.1 Kontext repositories. 

For Wan2.1/2.2, please also refer to [Wan2.1/2.2 documentation](./docs/wan.md). For FramePack, please also refer to [FramePack documentation](./docs/framepack.md). For FLUX.1 Kontext, please refer to [FLUX.1 Kontext documentation](./docs/flux_kontext.md).

*This repository is under development.*

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

### Recent Updates

- GitHub Discussions Enabled: We've enabled GitHub Discussions for community Q&A, knowledge sharing, and technical information exchange. Please use Issues for bug reports and feature requests, and Discussions for questions and sharing experiences. [Join the conversation →](https://github.com/kohya-ss/musubi-tuner/discussions)

- August 8, 2025:
    - Added support for Wan2.2.  [PR #399](https://github.com/kohya-ss/musubi-tuner/pull/399). See [Wan2.1/2.2 documentation](./docs/wan.md). 

        Wan2.2 consists of two models: high noise and low noise. During LoRA training, you can choose either one or both. Please refer to the documentation for details on specifying timesteps.

- August 7, 2025:
    - Added new sampling methods for timesteps: `logsnr` and `qinglong`. Thank you to sdbds for proposing this in [PR #407](https://github.com/kohya-ss/musubi-tuner/pull/407). `logsnr` is designed for style learning, while `qinglong` is a hybrid sampling method that considers style learning, model stability, and detail reproduction. For details, see the [Style-friendly SNR Sampler documentation](./docs/advanced_config.md#style-friendly-snr-sampler).

- August 2, 2025:
    - Reduced peak memory usage during model loading for FramePack and Wan2.1 when using `--fp8_scaled`. This reduces VRAM usage during model loading before training and inference.

- August 1, 2025:
    - Fixed the issue where block swapping did not work in FLUX. Kontext LoRA training. Thanks to sdbds for [PR #402](https://github.com/kohya-ss/musubi-tuner/pull/402). [PR #403](https://github.com/kohya-ss/musubi-tuner/pull/403).

- July 31, 2025:
    - Added [a section for developers using AI coding agents](#for-developers-using-ai-coding-agents). If you are using AI agents, please read this section.

- July 29, 2025:
    - Added `sentencepiece` to `pyproject.toml` to fix the issue where FLUX.1 Kontext LoRA training was not possible due to missing dependencies.

- July 28, 2025:
    - Added LoRA training for FLUX.1 Kontext \[dev\]. For details, see the [FLUX.1 Kontext LoRA training documentation](./docs/flux_kontext.md).

### Releases

We are grateful to everyone who has been contributing to the Musubi Tuner ecosystem through documentation and third-party tools. To support these valuable contributions, we recommend working with our [releases](https://github.com/kohya-ss/musubi-tuner/releases) as stable reference points, as this project is under active development and breaking changes may occur.

You can find the latest release and version history in our [releases page](https://github.com/kohya-ss/musubi-tuner/releases).

### For Developers Using AI Coding Agents

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt (currently they are the almost same):

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so it won't be committed to the repository.

## Overview

### Hardware Requirements

- VRAM: 12GB or more recommended for image training, 24GB or more for video training
    - *Actual requirements depend on resolution and training settings.* For 12GB, use a resolution of 960x544 or lower and use memory-saving options such as `--blocks_to_swap`, `--fp8_llm`, etc.
- Main Memory: 64GB or more recommended, 32GB + swap may work

### Features

- Memory-efficient implementation
- Windows compatibility confirmed (Linux compatibility confirmed by community)
- Multi-GPU support not implemented

## Installation

### pip based installation

Python 3.10 or later is required (verified with 3.10).

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. 

PyTorch 2.5.1 or later is required (see [note](#PyTorch-version)).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command.

```bash
pip install -e .
```

Optionally, you can use FlashAttention and SageAttention (**for inference only**; see [SageAttention Installation](#sageattention-installation) for installation instructions).

Optional dependencies for additional features:
- `ascii-magic`: Used for dataset verification
- `matplotlib`: Used for timestep visualization
- `tensorboard`: Used for logging training progress
- `prompt-toolkit`: Used for interactive prompt editing in Wan2.1 and FramePack inference scripts. If installed, it will be automatically used in interactive mode. Especially useful in Linux environments for easier prompt editing.

```bash
pip install ascii-magic matplotlib tensorboard prompt-toolkit
```

### uv based installation (experimenal)

You can also install using uv, but installation with uv is experimental. Feedback is welcome.

1. Install uv (if not already present on your OS).

#### Linux/MacOS

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Follow the instructions to add the uv path manually until you restart your session...

#### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Follow the instructions to add the uv path manually until you reboot your system... or just reboot your system at this point.

## Model Download

There are two ways to download the model.

### Use the Official HunyuanVideo Model

Download the model following the [official README](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md) and place it in your chosen directory with the following structure:

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

### Using ComfyUI Models for Text Encoder

This method is easier.

For DiT and VAE, use the HunyuanVideo models.

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/transformers, download [mp_rank_00_model_states.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt) and place it in your chosen directory.

(Note: The fp8 model on the same page is unverified.)

If you are training with `--fp8_base`, you can use `mp_rank_00_model_states_fp8.safetensors` from [here](https://huggingface.co/kohya-ss/HunyuanVideo-fp8_e4m3fn-unofficial) instead of `mp_rank_00_model_states.pt`. (This file is unofficial and simply converts the weights to float8_e4m3fn.)

From https://huggingface.co/tencent/HunyuanVideo/tree/main/hunyuan-video-t2v-720p/vae, download [pytorch_model.pt](https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt) and place it in your chosen directory.

For the Text Encoder, use the models provided by ComfyUI. Refer to [ComfyUI's page](https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_video/), from https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged/tree/main/split_files/text_encoders, download `llava_llama3_fp16.safetensors` (Text Encoder 1, LLM) and `clip_l.safetensors` (Text Encoder 2, CLIP)  and place them in your chosen directory.

(Note: The fp8 LLM model on the same page is unverified.)

## Usage

### Dataset Configuration

Please refer to [dataset configuration guide](./src/musubi_tuner/dataset/dataset_config.md).

### Latent Pre-caching

Latent pre-caching is required. Create the cache using the following command:

If you have installed using pip:

```bash
python src/musubi_tuner/cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

If you have installed with `uv`, you can use `uv run --extra cu124` to run the script. If CUDA 12.8 is supported, `uv run --extra cu128` is also available. Other scripts can be run in the same way. (Note that the installation with `uv` is experimental. Feedback is welcome. If you encounter any issues, please use the pip-based installation.)

```bash
uv run --extra cu124 src/musubi_tuner/cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

For additional options, use `python src/musubi_tuner/cache_latents.py --help`.

If you're running low on VRAM, reduce `--vae_spatial_tile_sample_min_size` to around 128 and lower the `--batch_size`.

Use `--debug_mode image` to display dataset images and captions in a new window, or `--debug_mode console` to display them in the console (requires `ascii-magic`). 

With `--debug_mode video`, images or videos will be saved in the cache directory (please delete them after checking). The bitrate of the saved video is set to 1Mbps for preview purposes. The images decoded from the original video (not degraded) are used for the cache (for training).

When `--debug_mode` is specified, the actual caching process is not performed.

By default, cache files not included in the dataset are automatically deleted. You can still keep cache files as before by specifying `--keep_cache`.

### Text Encoder Output Pre-caching

Text Encoder output pre-caching is required. Create the cache using the following command:

```bash
python src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

For additional options, use `python src/musubi_tuner/cache_text_encoder_outputs.py --help`.

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_llm` to run the LLM in fp8 mode.

By default, cache files not included in the dataset are automatically deleted. You can still keep cache files as before by specifying `--keep_cache`.

### Configuration of Accelerate

Run `accelerate config` to configure Accelerate. Choose appropriate values for each question based on your environment (either input values directly or use arrow keys and enter to select; uppercase is default, so if the default value is fine, just press enter without inputting anything). For training with a single GPU, answer the questions as follows:


```txt
- In which compute environment are you running?: This machine
- Which type of machine are you using?: No distributed training
- Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)?[yes/NO]: NO
- Do you wish to optimize your script with torch dynamo?[yes/NO]: NO
- Do you want to use DeepSpeed? [yes/NO]: NO
- What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]: all
- Would you like to enable numa efficiency? (Currently only supported on NVIDIA hardware). [yes/NO]: NO
- Do you wish to use mixed precision?: bf16
```

*Note*: In some cases, you may encounter the error `ValueError: fp16 mixed precision requires a GPU`. If this happens, answer "0" to the sixth question (`What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`). This means that only the first GPU (id `0`) will be used.

### Training

Start training using the following command (input as a single line):

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

or for uv:

```bash
uv run --extra cu124 accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing 
    --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module networks.lora --network_dim 32 
    --timestep_sampling shift --discrete_flow_shift 7.0 
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

__Update__: Changed the sample training settings to a learning rate of 2e-4, `--timestep_sampling` to `shift`, and `--discrete_flow_shift` to 7.0. Faster training is expected. If the details of the image are not learned well, try lowering the discete flow shift to around 3.0.

However, the training settings are still experimental. Appropriate learning rates, training steps, timestep distribution, loss weighting, etc. are not yet known. Feedback is welcome.

For additional options, use `python src/musubi_tuner/hv_train_network.py --help` (note that many options are unverified).

Specifying `--fp8_base` runs DiT in fp8 mode. Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality. If `--fp8_base` is not specified, 24GB or more VRAM is recommended. Use `--blocks_to_swap` as needed.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 36.

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

Use `--sdpa` for PyTorch's scaled dot product attention. Use `--flash_attn` for [FlashAttention](https://github.com/Dao-AILab/flash-attention). Use `--xformers` for xformers, but specify `--split_attn` when using xformers. `--sage_attn` for SageAttention, but SageAttention is not yet supported for training, so it raises a ValueError.

`--split_attn` processes attention in chunks. Speed may be slightly reduced, but VRAM usage is slightly reduced.

The format of LoRA trained is the same as `sd-scripts`.

You can also specify the range of timesteps 
with `--min_timestep` and `--max_timestep`. See [advanced configuration](./docs/advanced_config.md#specify-time-step-range-for-training--学習時のタイムステップ範囲の指定) for details.

`--show_timesteps` can be set to `image` (requires `matplotlib`) or `console` to display timestep distribution and loss weighting during training.

You can record logs during training. Refer to [Save and view logs in TensorBoard format](./docs/advanced_config.md#save-and-view-logs-in-tensorboard-format--tensorboard形式のログの保存と参照).

For PyTorch Dynamo optimization, refer to [this document](./docs/advanced_config.md#pytorch-dynamo-optimization-for-model-training--モデルの学習におけるpytorch-dynamoの最適化).

For sample image generation during training, refer to [this document](./docs/sampling_during_training.md). For advanced configuration, refer to [this document](./docs/advanced_config.md).

### Merging LoRA Weights

Note: Wan2.1 is not supported for merging LoRA weights.

```bash
python src/musubi_tuner/merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/merge_lora.py \
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --lora_weight path/to/lora.safetensors \
    --save_merged_model path/to/merged_model.safetensors \
    --device cpu \
    --lora_multiplier 1.0
```

Specify the device to perform the calculation (`cpu` or `cuda`, etc.) with `--device`. Calculation will be faster if `cuda` is specified.

Specify the LoRA weights to merge with `--lora_weight` and the multiplier for the LoRA weights with `--lora_multiplier`. Multiple values can be specified, and the number of values must match.

### Inference

Generate videos using the following command:

```bash
python src/musubi_tuner/hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa --split_attn
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

For additional options, use `python src/musubi_tuner/hv_generate_video.py --help`.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

`--fp8_fast` option is also available for faster inference on RTX 40x0 GPUs. This option requires `--fp8` option. 

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 38.

For `--attn_mode`, specify either `flash`, `torch`, `sageattn`, `xformers`, or `sdpa` (same as `torch`). These correspond to FlashAttention, scaled dot product attention, SageAttention, and xformers, respectively. Default is `torch`. SageAttention is effective for VRAM reduction.

Specifing `--split_attn` will process attention in chunks. Inference with SageAttention is expected to be about 10% faster.

For `--output_type`, specify either `both`, `latent`, `video` or `images`. `both` outputs both latents and video. Recommended to use `both` in case of Out of Memory errors during VAE processing. You can specify saved latents with `--latent_path` and use `--output_type video` (or `images`) to only perform VAE decoding.

`--seed` is optional. A random seed will be used if not specified.

`--video_length` should be specified as "a multiple of 4 plus 1".

`--flow_shift` can be specified to shift the timestep (discrete flow shift). The default value when omitted is 7.0, which is the recommended value for 50 inference steps. In the HunyuanVideo paper, 7.0 is recommended for 50 steps, and 17.0 is recommended for less than 20 steps (e.g. 10).

By specifying `--video_path`, video2video inference is possible. Specify a video file or a directory containing multiple image files (the image files are sorted by file name and used as frames). An error will occur if the video is shorter than `--video_length`. You can specify the strength with `--strength`. It can be specified from 0 to 1.0, and the larger the value, the greater the change from the original video.

Note that video2video inference is experimental.

`--compile` option enables PyTorch's compile feature (experimental). Requires triton. On Windows, also requires Visual C++ build tools installed and PyTorch>=2.6.0 (Visual C++ build tools is also required). You can pass arguments to the compiler with `--compile_args`.

The `--compile` option takes a long time to run the first time, but speeds up on subsequent runs.

You can save the DiT model after LoRA merge with the `--save_merged_model` option. Specify `--save_merged_model path/to/merged_model.safetensors`. Note that inference will not be performed when this option is specified.

### Inference with SkyReels V1

SkyReels V1 T2V and I2V models are supported (inference only). 

The model can be downloaded from [here](https://huggingface.co/Kijai/SkyReels-V1-Hunyuan_comfy). Many thanks to Kijai for providing the model. `skyreels_hunyuan_i2v_bf16.safetensors` is the I2V model, and `skyreels_hunyuan_t2v_bf16.safetensors` is the T2V model. The models other than bf16 are not tested (`fp8_e4m3fn` may work).

For T2V inference, add the following options to the inference command:

```bash
--guidance_scale 6.0 --embedded_cfg_scale 1.0 --negative_prompt "Aerial view, aerial view, overexposed, low quality, deformation, a poor composition, bad hands, bad teeth, bad eyes, bad limbs, distortion" --split_uncond
```

SkyReels V1 seems to require a classfier free guidance (negative prompt).`--guidance_scale` is a guidance scale for the negative prompt. The recommended value is 6.0 from the official repository. The default is 1.0, it means no classifier free guidance.

`--embedded_cfg_scale` is a scale of the embedded guidance. The recommended value is 1.0 from the official repository (it may mean no embedded guidance).

`--negative_prompt` is a negative prompt for the classifier free guidance. The above sample is from the official repository. If you don't specify this, and specify `--guidance_scale` other than 1.0, an empty string will be used as the negative prompt.

`--split_uncond` is a flag to split the model call into unconditional and conditional parts. This reduces VRAM usage but may slow down inference. If `--split_attn` is specified, `--split_uncond` is automatically set.

You can also perform image2video inference with SkyReels V1 I2V model. Specify the image file path with `--image_path`. The image will be resized to the given `--video_size`.

```bash
--image_path path/to/image.jpg
``` 

### Convert LoRA to another format

You can convert LoRA to a format compatible with ComfyUI (presumed to be Diffusion-pipe) using the following command:

```bash
python src/musubi_tuner/convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

or for uv:

```bash
uv run --extra cu124 src/musubi_tuner/convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

Specify the input and output file paths with `--input` and `--output`, respectively.

Specify `other` for `--target`. Use `default` to convert from another format to the format of this repository.

Wan2.1 is also supported. 

## Miscellaneous

### SageAttention Installation

sdbsd has provided a Windows-compatible SageAttention implementation and pre-built wheels here:  https://github.com/sdbds/SageAttention-for-windows. After installing triton, if your Python, PyTorch, and CUDA versions match, you can download and install the pre-built wheel from the [Releases](https://github.com/sdbds/SageAttention-for-windows/releases) page. Thanks to sdbsd for this contribution.

For reference, the build and installation instructions are as follows. You may need to update Microsoft Visual C++ Redistributable to the latest version.

1. Download and install triton 3.1.0 wheel matching your Python version from [here](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5).

2. Install Microsoft Visual Studio 2022 or Build Tools for Visual Studio 2022, configured for C++ builds.

3. Clone the SageAttention repository in your preferred directory:
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. Open `x64 Native Tools Command Prompt for VS 2022` from the Start menu under Visual Studio 2022.

5. Activate your venv, navigate to the SageAttention folder, and run the following command. If you get a DISTUTILS not configured error, set `set DISTUTILS_USE_SDK=1` and try again:
    ```shell
    python setup.py install
    ```

This completes the SageAttention installation.

### PyTorch version

If you specify `torch` for `--attn_mode`, use PyTorch 2.5.1 or later (earlier versions may result in black videos).

If you use an earlier version, use xformers or SageAttention.

## Disclaimer

This repository is unofficial and not affiliated with the official HunyuanVideo repository. 

This repository is experimental and under active development. While we welcome community usage and feedback, please note:

- This is not intended for production use
- Features and APIs may change without notice
- Some functionalities are still experimental and may not work as expected
- Video training features are still under development

If you encounter any issues or bugs, please create an Issue in this repository with:
- A detailed description of the problem
- Steps to reproduce
- Your environment details (OS, GPU, VRAM, Python version, etc.)
- Any relevant error messages or logs

## Contributing

We welcome contributions! However, please note:

- Due to limited maintainer resources, PR reviews and merges may take some time
- Before starting work on major changes, please open an Issue for discussion
- For PRs:
  - Keep changes focused and reasonably sized
  - Include clear descriptions
  - Follow the existing code style
  - Ensure documentation is updated

## License

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.

Code under the `wan` directory is modified from [Wan2.1](https://github.com/Wan-Video/Wan2.1). The license is under the Apache License 2.0.

Code under the `frame_pack` directory is modified from [FramePack](https://github.com/lllyasviel/FramePack). The license is under the Apache License 2.0.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.
