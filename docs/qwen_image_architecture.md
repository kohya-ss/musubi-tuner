# Qwen-Image-2512 Architecture Reference

This document provides a comprehensive reference for the Qwen-Image-2512 model architecture, extracted directly from the official HuggingFace model configuration files. This information is intended to aid debugging, optimization, and development work on the Qwen-Image training pipeline.

**Source**: [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512)
**Diffusers Version**: 0.36.0.dev0
**Pipeline Class**: `QwenImagePipeline`

---

## Component Summary

| Component | Class | Parameters | Key Characteristics |
|-----------|-------|------------|---------------------|
| Text Encoder | `Qwen2_5_VLForConditionalGeneration` | ~8.3B | VL model with 28-layer LLM + 32-layer ViT |
| Transformer (DiT) | `QwenImageTransformer2DModel` | ~20.4B | 60-layer MMDiT with joint attention |
| VAE | `AutoencoderKLQwenImage` | ~83M est. | 16-channel latent, 8x spatial compression |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | - | Flow matching with dynamic shift |
| Tokenizer | `Qwen2Tokenizer` | 152k vocab | BPE tokenizer |

**Total Model Size**: ~38GB (bf16), ~40.8GB raw weights

---

## Data Flow Overview

```
Input Image (H × W × 3)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  VAE Encoder                                                     │
│  - Spatial: 8× downsampling                                      │
│  - Output: (H/8) × (W/8) × 16 latent channels                   │
│  - Normalized with per-channel mean/std                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Latent: (H/8) × (W/8) × 16
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
┌─────────────────────┐              ┌─────────────────────────────┐
│  Patchify           │              │  Text Encoder (Qwen2.5-VL)  │
│  patch_size=2       │              │  - Tokenize prompt          │
│  in_channels=64     │              │  - 28-layer transformer     │
│                     │              │  - Output: (seq_len × 3584) │
└─────────────────────┘              └─────────────────────────────┘
         │                                      │
         │      ┌───────────────────────────────┘
         │      │
         ▼      ▼
┌─────────────────────────────────────────────────────────────────┐
│  MMDiT Transformer (60 layers)                                   │
│  - Joint attention between image and text tokens                 │
│  - MSRoPE positional encoding: axes [16, 56, 56]                │
│  - Timestep embedding via sinusoidal + MLP                       │
│  - Separate img_mlp and txt_mlp per block                        │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Predicted Noise/Velocity
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  VAE Decoder                                                     │
│  - Spatial: 8× upsampling                                        │
│  - Output: (H × W × 3) RGB image                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Text Encoder: Qwen2.5-VL

The text encoder is a full Vision-Language model based on Qwen2.5-VL architecture, capable of processing both text and images as input context.

### Language Model Configuration

```json
{
  "architectures": ["Qwen2_5_VLForConditionalGeneration"],
  "model_type": "qwen2_5_vl",
  "hidden_size": 3584,
  "intermediate_size": 18944,
  "num_hidden_layers": 28,
  "num_attention_heads": 28,
  "num_key_value_heads": 4,
  "hidden_act": "silu",
  "max_position_embeddings": 128000,
  "rms_norm_eps": 1e-06,
  "vocab_size": 152064,
  "rope_theta": 1000000.0,
  "sliding_window": 32768,
  "max_window_layers": 28,
  "tie_word_embeddings": false
}
```

### Key Architectural Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Hidden Dimension | 3584 | Matches MMDiT joint_attention_dim |
| Attention Heads | 28 | Full attention |
| KV Heads | 4 | Grouped Query Attention (GQA) 7:1 ratio |
| FFN Intermediate | 18944 | ~5.3× expansion ratio |
| Layers | 28 | All use full attention (no sliding window) |
| Max Context | 128k tokens | Extremely long context support |
| Activation | SiLU | Smooth activation function |
| Normalization | RMSNorm (eps=1e-6) | Pre-norm architecture |

### MROPE Configuration (Text Encoder)

The text encoder uses Multi-dimensional RoPE with three sections:

```json
{
  "rope_scaling": {
    "mrope_section": [16, 24, 24],
    "rope_type": "default"
  }
}
```

- Total RoPE dimensions: 16 + 24 + 24 = 64
- Per-head dimension: 3584 / 28 = 128
- RoPE covers half of head dim (64/128)

### Vision Encoder (Embedded in Text Encoder)

The text encoder includes an integrated vision transformer for processing image inputs:

```json
{
  "vision_config": {
    "depth": 32,
    "hidden_size": 1280,
    "out_hidden_size": 3584,
    "num_heads": 16,
    "patch_size": 14,
    "spatial_patch_size": 14,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "window_size": 112,
    "intermediate_size": 3420,
    "fullatt_block_indexes": [7, 15, 23, 31],
    "hidden_act": "silu"
  }
}
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| Vision Layers | 32 | Deep vision encoder |
| Vision Hidden | 1280 | Internal ViT dimension |
| Output Projection | 3584 | Projects to LLM hidden size |
| Patch Size | 14×14 | Spatial tokenization |
| Spatial Merge | 2×2 | Pools 4 patches → 1 token |
| Full Attention Layers | [7, 15, 23, 31] | Every 8th layer uses global attention |

### Special Token IDs

```json
{
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "image_token_id": 151655,
  "video_token_id": 151656,
  "vision_start_token_id": 151652,
  "vision_end_token_id": 151653,
  "vision_token_id": 151654
}
```

### Weight Tensor Patterns (Text Encoder)

```
model.embed_tokens.weight                    # Embedding layer
model.layers.{0-27}.input_layernorm.weight   # Pre-attention norm
model.layers.{0-27}.self_attn.q_proj.*       # Query projection (+ bias)
model.layers.{0-27}.self_attn.k_proj.*       # Key projection (+ bias)
model.layers.{0-27}.self_attn.v_proj.*       # Value projection (+ bias)
model.layers.{0-27}.self_attn.o_proj.weight  # Output projection
model.layers.{0-27}.post_attention_layernorm.weight
model.layers.{0-27}.mlp.gate_proj.weight     # FFN gating
model.layers.{0-27}.mlp.up_proj.weight       # FFN up projection
model.layers.{0-27}.mlp.down_proj.weight     # FFN down projection
model.norm.weight                            # Final layer norm
lm_head.weight                               # Output projection

# Vision encoder weights (visual.*)
visual.patch_embed.proj.weight
visual.blocks.{0-31}.attn.qkv.*              # Fused QKV
visual.blocks.{0-31}.attn.proj.*             # Attention output
visual.blocks.{0-31}.mlp.gate_proj.*
visual.blocks.{0-31}.mlp.up_proj.*
visual.blocks.{0-31}.mlp.down_proj.*
visual.blocks.{0-31}.norm1.weight
visual.blocks.{0-31}.norm2.weight
visual.merger.ln_q.weight                    # Merger layer norm
visual.merger.mlp.{0,2}.*                    # Merger MLP
```

---

## Transformer (MMDiT): QwenImageTransformer2DModel

The core diffusion transformer uses a Multi-Modal DiT architecture with joint image-text attention.

### Configuration

```json
{
  "_class_name": "QwenImageTransformer2DModel",
  "num_layers": 60,
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "joint_attention_dim": 3584,
  "in_channels": 64,
  "out_channels": 16,
  "patch_size": 2,
  "axes_dims_rope": [16, 56, 56],
  "guidance_embeds": false
}
```

### Key Architectural Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Layers | 60 | Deep transformer |
| Attention Heads | 24 | For image processing |
| Head Dimension | 128 | Per-head dimension |
| Hidden Dimension | 24 × 128 = 3072 | Image stream hidden |
| Joint Attention Dim | 3584 | Text conditioning dimension |
| Input Channels | 64 | After patchify (16 latent × 2×2 patch) |
| Output Channels | 16 | Matches VAE latent channels |
| Patch Size | 2×2 | Patchifies latent space |
| Guidance Embeds | false | No CFG embedding in model |

### MSRoPE Configuration (Transformer)

The transformer uses Multi-Scale RoPE with axes optimized for image generation:

```python
axes_dims_rope = [16, 56, 56]  # [temporal/batch, height, width]
```

- **Axis 0** (16 dims): Temporal/batch position (for video extension)
- **Axis 1** (56 dims): Height position encoding
- **Axis 2** (56 dims): Width position encoding
- **Total**: 128 dims = attention_head_dim

This design allows the model to encode 2D spatial positions separately, critical for maintaining spatial coherence in generated images.

### Transformer Block Structure

Each of the 60 transformer blocks contains:

```
TransformerBlock
├── img_mod          # Image modulation (timestep-conditioned)
├── txt_mod          # Text modulation (timestep-conditioned)
├── attn             # Joint attention module
│   ├── to_q, to_k, to_v           # Image projections
│   ├── add_q_proj, add_k_proj, add_v_proj  # Text projections
│   ├── norm_q, norm_k              # Image QK normalization
│   ├── norm_added_q, norm_added_k  # Text QK normalization
│   ├── to_out                      # Image output projection
│   └── to_add_out                  # Text output projection
├── img_mlp          # Image FFN
│   ├── net.0.proj   # Gated linear unit
│   └── net.2        # Output projection
└── txt_mlp          # Text FFN
    ├── net.0.proj
    └── net.2
```

### Weight Tensor Patterns (Transformer)

```
img_in.{weight,bias}                         # Latent to hidden projection
time_text_embed.timestep_embedder.linear_{1,2}.*  # Timestep embedding MLP

transformer_blocks.{0-59}.img_mod.1.*        # Image modulation
transformer_blocks.{0-59}.txt_mod.1.*        # Text modulation

# Joint attention weights
transformer_blocks.{0-59}.attn.to_q.*        # Image query
transformer_blocks.{0-59}.attn.to_k.*        # Image key
transformer_blocks.{0-59}.attn.to_v.*        # Image value
transformer_blocks.{0-59}.attn.to_out.0.*    # Image output
transformer_blocks.{0-59}.attn.add_q_proj.*  # Text query
transformer_blocks.{0-59}.attn.add_k_proj.*  # Text key
transformer_blocks.{0-59}.attn.add_v_proj.*  # Text value
transformer_blocks.{0-59}.attn.to_add_out.*  # Text output
transformer_blocks.{0-59}.attn.norm_q.*      # Image Q normalization
transformer_blocks.{0-59}.attn.norm_k.*      # Image K normalization
transformer_blocks.{0-59}.attn.norm_added_q.* # Text Q normalization
transformer_blocks.{0-59}.attn.norm_added_k.* # Text K normalization

# FFN weights
transformer_blocks.{0-59}.img_mlp.net.0.proj.*  # Image FFN gate+up
transformer_blocks.{0-59}.img_mlp.net.2.*       # Image FFN down
transformer_blocks.{0-59}.txt_mlp.net.0.proj.*  # Text FFN gate+up
transformer_blocks.{0-59}.txt_mlp.net.2.*       # Text FFN down

norm_out.linear.*                            # Final normalization
proj_out.*                                   # Output projection to latent
```

### Memory Optimization Notes

With 60 layers, the transformer is a prime target for memory optimization:

- **Block Swapping**: Up to 59 blocks can be swapped to CPU
- **Gradient Checkpointing**: Reduces activation memory ~3-4×
- **Attention Precision**: bf16 recommended, fp8 possible with quality tradeoff

---

## VAE: AutoencoderKLQwenImage

The VAE handles encoding images to latent space and decoding back to pixel space.

### Configuration

```json
{
  "_class_name": "AutoencoderKLQwenImage",
  "z_dim": 16,
  "base_dim": 96,
  "dim_mult": [1, 2, 4, 4],
  "num_res_blocks": 2,
  "dropout": 0.0,
  "temperal_downsample": [false, true, true],
  "attn_scales": []
}
```

### Architecture Details

| Parameter | Value | Notes |
|-----------|-------|-------|
| Latent Channels | 16 | High-capacity latent space |
| Base Dimension | 96 | Initial conv channels |
| Dimension Multipliers | [1, 2, 4, 4] | Channel scaling per stage |
| Stages | 4 | Encoder/decoder depth |
| Spatial Compression | 8× | 2× per stage (2^3) |
| Res Blocks per Stage | 2 | Standard depth |
| Temporal Downsampling | [F, T, T] | For video (stages 1, 2 only) |

### Channel Progression

```
Stage 0: 96 × 1  = 96  channels
Stage 1: 96 × 2  = 192 channels
Stage 2: 96 × 4  = 384 channels
Stage 3: 96 × 4  = 384 channels
```

### Latent Normalization Statistics

The VAE uses per-channel normalization for stable training:

```python
latents_mean = [
    -0.7571, -0.7089, -0.9113,  0.1075,
    -0.1745,  0.9653, -0.1517,  1.5508,
     0.4134, -0.0715,  0.5517, -0.3632,
    -0.1922, -0.9497,  0.2503, -0.2921
]

latents_std = [
    2.8184, 1.4541, 2.3275, 2.6558,
    1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160
]
```

**Usage**: Normalize latents with `(latent - mean) / std` before training, denormalize after inference.

### Latent Space Properties

For an input image of size H × W:

- **Latent Size**: (H/8) × (W/8) × 16
- **Example** (1664×928 image): 208 × 116 × 16 = 386,048 latent values

---

## Scheduler: FlowMatchEulerDiscreteScheduler

The scheduler implements flow matching with dynamic timestep shifting.

### Configuration

```json
{
  "_class_name": "FlowMatchEulerDiscreteScheduler",
  "num_train_timesteps": 1000,
  "base_shift": 0.5,
  "max_shift": 0.9,
  "shift_terminal": 0.02,
  "time_shift_type": "exponential",
  "use_dynamic_shifting": true,
  "base_image_seq_len": 256,
  "max_image_seq_len": 8192,
  "stochastic_sampling": false,
  "invert_sigmas": false,
  "use_karras_sigmas": false,
  "use_exponential_sigmas": false,
  "use_beta_sigmas": false
}
```

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Training Timesteps | 1000 | Discrete timestep range |
| Base Shift | 0.5 | Shift for base sequence length |
| Max Shift | 0.9 | Shift for max sequence length |
| Shift Terminal | 0.02 | Terminal shift value |
| Time Shift Type | exponential | Exponential shift curve |
| Dynamic Shifting | true | Adapts to image resolution |
| Base Seq Length | 256 | Reference sequence length |
| Max Seq Length | 8192 | Maximum supported sequence |

### Dynamic Shift Formula

The scheduler adjusts the noise schedule based on image resolution:

```python
# Simplified dynamic shift calculation
image_seq_len = (height // 8 // patch_size) * (width // 8 // patch_size)
shift = base_shift + (max_shift - base_shift) * (image_seq_len / max_image_seq_len)
```

This ensures consistent training dynamics across different resolutions.

---

## Supported Resolutions

The model is trained on specific aspect ratios for optimal quality:

| Aspect Ratio | Resolution | Latent Size | Sequence Length |
|--------------|------------|-------------|-----------------|
| 1:1 | 1328 × 1328 | 166 × 166 | 6889 |
| 16:9 | 1664 × 928 | 208 × 116 | 6032 |
| 9:16 | 928 × 1664 | 116 × 208 | 6032 |
| 4:3 | 1472 × 1104 | 184 × 138 | 6348 |
| 3:4 | 1104 × 1472 | 138 × 184 | 6348 |
| 3:2 | 1584 × 1056 | 198 × 132 | 6534 |
| 2:3 | 1056 × 1584 | 132 × 198 | 6534 |

**Note**: Sequence length = (latent_h / patch_size) × (latent_w / patch_size) where patch_size=2

---

## LoRA Target Modules

For fine-tuning with LoRA, the following module patterns are typically targeted:

### Transformer Targets (High Impact)

```python
# Joint attention projections
"transformer_blocks.*.attn.to_q"
"transformer_blocks.*.attn.to_k"
"transformer_blocks.*.attn.to_v"
"transformer_blocks.*.attn.to_out.0"
"transformer_blocks.*.attn.add_q_proj"
"transformer_blocks.*.attn.add_k_proj"
"transformer_blocks.*.attn.add_v_proj"
"transformer_blocks.*.attn.to_add_out"

# FFN projections
"transformer_blocks.*.img_mlp.net.0.proj"
"transformer_blocks.*.img_mlp.net.2"
"transformer_blocks.*.txt_mlp.net.0.proj"
"transformer_blocks.*.txt_mlp.net.2"
```

### Text Encoder Targets (Optional)

```python
# Self-attention
"model.layers.*.self_attn.q_proj"
"model.layers.*.self_attn.k_proj"
"model.layers.*.self_attn.v_proj"
"model.layers.*.self_attn.o_proj"

# FFN
"model.layers.*.mlp.gate_proj"
"model.layers.*.mlp.up_proj"
"model.layers.*.mlp.down_proj"
```

---

## Prompt Enhancement System

Qwen-Image uses a sophisticated prompt rewriting system to improve generation quality:

### Language Detection

```python
# Detects CJK characters to determine language
if any('\u4e00' <= char <= '\u9fff' for char in prompt):
    return 'zh'  # Chinese
return 'en'  # English
```

### Rewriting Categories

The prompt enhancer classifies prompts into three categories:

1. **Portrait Images**: Detailed human descriptions (ethnicity, age, clothing, pose)
2. **Text-Containing Images**: Images with visible text (signs, labels, UI)
3. **General Images**: Landscapes, objects, abstract compositions

### Recommended Prompting

- **For Chinese prompts**: Add "超清，4K，电影级构图"
- **For English prompts**: Add "Ultra HD, 4K, cinematic composition"
- **Text in images**: Wrap displayed text in quotation marks ("text here")
- **Negative prompt**: "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感"

---

## Training Recommendations

### Memory Requirements

| Configuration | VRAM (bf16) | Notes |
|--------------|-------------|-------|
| Full model | ~80GB | A100 80GB minimum |
| Gradient checkpointing | ~50GB | Recommended baseline |
| + Block swap (30) | ~35GB | Good for 40GB GPUs |
| + FP8 quantization | ~25GB | Quality tradeoff |

### Timestep Sampling

For Qwen-Image training, use flow matching with shift:

```bash
--timestep_sampling shift
--discrete_flow_shift 3.0  # Typical value, adjust as needed
```

### Batch Accumulation

With the deep 60-layer transformer, gradient accumulation is often necessary:

```bash
--gradient_accumulation_steps 4
--gradient_checkpointing
```

---

## Technical Report Insights

The following insights are extracted from the official Qwen-Image Technical Report and provide deeper understanding of the architectural decisions and training methodology.

### MSRoPE Design Rationale

The Multi-Scale Rotary Position Embedding design serves a critical purpose beyond standard 2D position encoding:

**Text Token Positioning**: Text tokens from the T5-based text encoder are positioned along the diagonal of the image grid rather than prepended or appended. This design:

1. Preserves the gradual positional encoding from the T5 encoder
2. Allows smooth interpolation between text and image positions
3. Enables better handling of long text sequences for text rendering tasks
4. The axes `[16, 56, 56]` allocate 16 dims for temporal/text position and 56 dims each for spatial height/width

### VAE Architecture & Performance

The Qwen-Image VAE uses a **single-encoder, dual-decoder** architecture designed for both image and video:

| Metric | Qwen-Image-VAE | FLUX-VAE | SD-3.5-VAE | Wan2.1-VAE |
|--------|----------------|----------|------------|------------|
| **Image PSNR↑** | **33.42** | 29.41 | 31.22 | 31.04 |
| **Image SSIM↑** | **0.9159** | 0.8596 | 0.8839 | 0.8916 |
| **Text-Rich PSNR↑** | **36.63** | 28.02 | 29.93 | 31.17 |
| **Text-Rich SSIM↑** | **0.9839** | 0.9374 | 0.9658 | 0.9702 |

**Effective Parameters** (image processing mode):
- Encoder: 19M (vs. 34M for SD-3.5-VAE)
- Decoder: 25M (vs. 50M for SD-3.5-VAE)

The decoder was specifically fine-tuned on text-rich images to improve reconstruction of small text, which is critical for the model's text rendering capabilities.

### Training Strategy

Qwen-Image employs a sophisticated **progressive curriculum learning** approach:

#### Resolution Progression
Training proceeds from low to high resolution, allowing the model to learn coarse structure before fine details:
```
Phase 1: 256×256 → Phase 2: 512×512 → Phase 3: 1024×1024 → Phase 4: 2512×2512
```

#### Text Complexity Progression
The model learns to generate images before learning text rendering:
```
Phase 1: Non-text images → Phase 2: Simple text → Phase 3: Complex text layouts
```

#### Data Pipeline (7 Stages)
1. Aesthetic filtering
2. Resolution filtering
3. Deduplication
4. NSFW filtering
5. Quality scoring
6. Caption quality assessment
7. Text rendering quality assessment (for text-rich data)

### Multi-Task Training

The model supports unified training across multiple tasks:

| Task | Conditioning | Description |
|------|-------------|-------------|
| **T2I** | Text only | Standard text-to-image generation |
| **TI2I** | Text + Image | Image editing with text instructions |
| **I2I** | Image only | Image reconstruction/manipulation |

For image editing (TI2I), the model uses a **frame dimension extension** mechanism where the input image is treated as an additional frame, enabling the model to learn editing operations through paired data.

### Post-Training Methods

After base training, Qwen-Image undergoes additional fine-tuning stages:

#### 1. SFT (Supervised Fine-Tuning)
Fine-tuning on high-quality curated datasets for specific capabilities like text rendering and image editing.

#### 2. DPO (Direct Preference Optimization)
```
L_DPO = -log σ(β(log π(y_w|x) - log π(y_l|x)))
```
Where `y_w` and `y_l` are preferred and rejected generations respectively.

#### 3. GRPO (Group Relative Policy Optimization)
A variant of PPO adapted for diffusion models, using group-based reward normalization:
```
L_GRPO = -E[r(y) · log π(y|x)] + β · KL(π || π_ref)
```

**Impact**: RL fine-tuning improves GenEval benchmark score from **0.87 → 0.91**, making Qwen-Image the first foundation model to exceed 0.9 on this benchmark.

### Benchmark Performance Summary

#### Text-to-Image Generation
| Benchmark | Qwen-Image Score | Ranking |
|-----------|-----------------|---------|
| DPG | 88.32 | #1 |
| GenEval | 0.91 (RL) | #1 |
| OneIG-EN | 0.539 | #1 |
| OneIG-ZH | 0.548 | #1 |

#### Chinese Text Rendering (ChineseWord Benchmark)
| Difficulty | Qwen-Image | GPT Image 1 | Seedream 3.0 |
|-----------|------------|-------------|--------------|
| Level-1 (3500 chars) | **97.29%** | 68.37% | 53.48% |
| Level-2 (3000 chars) | **40.53%** | 15.97% | 26.23% |
| Level-3 (1605 chars) | **6.48%** | 3.55% | 1.25% |

#### Image Editing (TI2I)
| Benchmark | Qwen-Image | Ranking |
|-----------|-----------|---------|
| GEdit-Bench-EN | 7.56 | #1 |
| GEdit-Bench-CN | 7.52 | #1 |
| ImgEdit | 4.27 | #1 |

### Extended Capabilities

The multi-task training enables Qwen-Image to perform tasks beyond standard T2I:

1. **Novel View Synthesis**: Competitive with specialized 3D models (PSNR 15.11 on GSO dataset)
2. **Depth Estimation**: Performs on par with state-of-the-art depth models (δ1=0.951 on KITTI)
3. **Chained Editing**: Sequential edits while maintaining consistency
4. **Pose Manipulation**: Modify subject poses while preserving identity
5. **Material Editing**: Change object materials/textures realistically

### Producer-Consumer Training Framework

Qwen-Image uses a distributed training architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  Producer Workers (Data Pipeline)                           │
│  - Image loading and preprocessing                          │
│  - Caption processing and tokenization                      │
│  - VAE encoding (cached or on-the-fly)                     │
│  - Data augmentation                                        │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Consumer Workers (Model Training)                          │
│  - DiT forward/backward passes                             │
│  - Gradient synchronization (FSDP/DeepSpeed)               │
│  - Optimizer steps                                          │
└─────────────────────────────────────────────────────────────┘
```

This separation allows efficient utilization of heterogeneous hardware and prevents data loading from becoming a bottleneck.

---

## Official Prompting Guidelines

Based on Alibaba's official T2I prompt guide, follow this structure for optimal results:

### Basic Formula
```
Subject + Scene + Style
```

### Advanced Formula
```
Subject + Scene + Style + Camera Language + Atmosphere + Detail
```

### Prompt Components

| Component | Description | Examples |
|-----------|-------------|----------|
| **Subject** | Main focus of image | "a young woman", "a mountain landscape" |
| **Scene** | Environment/setting | "in a coffee shop", "at sunset" |
| **Style** | Visual aesthetic | "oil painting", "photorealistic", "anime" |
| **Camera** | Shot type & angle | "close-up", "wide angle", "bird's eye view" |
| **Atmosphere** | Mood/lighting | "warm golden hour", "moody and dramatic" |
| **Detail** | Specific attributes | "wearing a red dress", "with intricate patterns" |

### Recommended Shot Types
- Extreme close-up (ECU)
- Close-up (CU)
- Medium shot (MS)
- Full shot (FS)
- Wide shot (WS)
- Extreme wide shot (EWS)

### Recommended Styles
- Photorealistic / Photography
- Oil painting / Watercolor
- Digital art / Concept art
- Anime / Manga
- 3D render / CGI
- Ink wash / Chinese painting

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| Qwen-Image | 2025-08-04 | Initial release |
| Qwen-Image-2512 | 2025-12-31 | Improved realism, textures, text rendering |

---

## References

- [Qwen-Image HuggingFace](https://huggingface.co/Qwen/Qwen-Image-2512)
- [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- [Qwen-Image Blog](https://qwenlm.github.io/blog/qwen-image/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
