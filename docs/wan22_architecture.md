# WAN 2.2 Architecture Reference

A comprehensive architecture reference for the WAN 2.2 video generation model family, optimized for coding agents, training pipeline development, and debugging.

---

## Quick Reference

| Component | Value | Notes |
|-----------|-------|-------|
| **Model Family** | WAN 2.2 | Alibaba Wan Team |
| **Architecture** | DiT (Diffusion Transformer) | Flow Matching + Cross-Attention |
| **A14B Checkpoints** | 2× ~14B (high-noise + low-noise) | Switches between two *separate* DiT checkpoints via `boundary` (not in-model MoE routing) |
| **DiT Dimension** | 5120 | Hidden size |
| **DiT Layers** | 40 | Transformer blocks |
| **Attention Heads** | 40 | 128 dim per head |
| **FFN Dimension** | 13824 | ~2.7x hidden dim |
| **Text Encoder** | `google/umt5-xxl` | ~5.3B parameters (umT5-XXL) |
| **Text Length** | 512 tokens | Max sequence |
| **VAE Compression** | 4×8×8 | Temporal × Height × Width |
| **Latent Channels** | 16 | z_dim |
| **Patch Size** | (1, 2, 2) | Additional 2x spatial |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WAN 2.2 Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Input      │     │   Text Prompt    │     │   Timestep t     │    │
│  │   Video      │     │                  │     │                  │    │
│  │  [F,H,W,3]   │     │                  │     │                  │    │
│  └──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘    │
│         │                      │                        │              │
│         ▼                      ▼                        ▼              │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │   Wan-VAE    │     │   umT5-XXL       │     │   Sinusoidal     │    │
│  │   Encoder    │     │   Encoder        │     │   + MLP          │    │
│  │  (4×8×8)     │     │   (5.3B)         │     │                  │    │
│  └──────┬───────┘     └────────┬─────────┘     └────────┬─────────┘    │
│         │                      │                        │              │
│         ▼                      │                        │              │
│  ┌──────────────┐              │                        │              │
│  │ Patch Embed  │              │                        │              │
│  │ Conv3d(1,2,2)│              │                        │              │
│  └──────┬───────┘              │                        │              │
│         │                      │                        │              │
│         ▼                      ▼                        ▼              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │     High/Low-Noise Checkpoint Selection (boundary-based)        │   │
│  │  ┌───────────────────┐    ┌───────────────────┐                 │   │
│  │  │ High-noise model   │    │ Low-noise model    │                 │   │
│  │  │ (t >= boundary)    │    │ (t < boundary)     │                 │   │
│  │  │ Layout/Structure   │    │ Details/Refine     │                 │   │
│  │  └───────────────────┘    └───────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                              │
│         ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              WanAttentionBlock × 40                             │   │
│  │  ┌─────────────────────────────────────────────────────────┐   │   │
│  │  │ Self-Attention (Full Spatio-Temporal + 3D RoPE)         │   │   │
│  │  │ ├── LayerNorm + Modulation (scale/shift)               │   │   │
│  │  │ ├── Q, K, V projection (dim → dim)                     │   │   │
│  │  │ ├── RMSNorm on Q, K (QK-Norm)                         │   │   │
│  │  │ ├── 3D RoPE: rope_apply(q/k, grid_sizes, freqs)       │   │   │
│  │  │ └── FlashAttention                                     │   │   │
│  │  ├─────────────────────────────────────────────────────────┤   │   │
│  │  │ Cross-Attention (Text Conditioning)                    │   │   │
│  │  │ ├── Q from visual, K/V from text embeddings           │   │   │
│  │  │ └── No positional encoding on text                     │   │   │
│  │  ├─────────────────────────────────────────────────────────┤   │   │
│  │  │ Feed-Forward Network                                   │   │   │
│  │  │ ├── LayerNorm + Modulation                            │   │   │
│  │  │ ├── Linear(dim → ffn_dim) + GELU                      │   │   │
│  │  │ └── Linear(ffn_dim → dim)                             │   │   │
│  │  └─────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────┐                                                      │
│  │    Head      │  LayerNorm + Modulation + Linear                     │
│  │  (Unpatchify)│  Output: [C_out, F, H/8, W/8]                       │
│  └──────┬───────┘                                                      │
│         │                                                              │
│         ▼                                                              │
│  ┌──────────────┐                                                      │
│  │   Wan-VAE    │                                                      │
│  │   Decoder    │                                                      │
│  └──────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Model Variants

### WAN 2.2 A14B (Dual-checkpoint “MoE-like” setup)

WAN 2.2 A14B is commonly described as a 2-expert “MoE”, but in this repo it’s implemented as **two separate DiT checkpoints**
(`low_noise_model` and `high_noise_model`) and the pipeline **switches which checkpoint is used** based on a fixed `boundary`.

| Checkpoint | Activation | Focus | Parameters |
|--------|------------|-------|------------|
| **High-noise model** | t >= boundary | Overall layout, composition | ~14B |
| **Low-noise model** | t < boundary | Fine details, textures | ~14B |
| **Total** | - | - | ~27B (only one “active” at a given timestep; memory may include both if loaded) |

#### Boundary (`boundary` / `timestep_boundary`)

Official materials often motivate the switch using SNR, but the code path here uses a **fixed threshold** from config.
Internally, timesteps are typically integers in `[0..1000]` and the code compares `t / 1000.0` against `boundary`.

| Task | Boundary | Interpretation (this repo) |
|------|----------|----------------|
| **T2V** | 0.875 | Use **high-noise** checkpoint for `t >= 0.875`, otherwise low-noise |
| **I2V** | 0.900 | Use **high-noise** checkpoint for `t >= 0.900`, otherwise low-noise |

```python
# Checkpoint switching logic used in this repo
# (t is typically in [0..1000] during training/inference codepaths)
if (t / 1000.0) >= boundary:
    model = high_noise_model  # high-noise region (early steps if timesteps decrease during sampling)
else:
    model = low_noise_model   # lower-noise region (later refinement)
```

### Model Configurations

#### T2V-A14B (Text-to-Video)

```python
# From wan/configs/wan_t2v_A14B.py
t2v_A14B = {
    # Text Encoder
    "t5_checkpoint": "models_t5_umt5-xxl-enc-bf16.pth",
    "t5_tokenizer": "google/umt5-xxl",
    "t5_dtype": torch.bfloat16,
    "text_len": 512,

    # VAE
    "vae_checkpoint": "Wan2.1_VAE.pth",
    "vae_stride": (4, 8, 8),  # Temporal, Height, Width compression

    # Transformer (DiT)
    "patch_size": (1, 2, 2),
    "dim": 5120,
    "ffn_dim": 13824,
    "freq_dim": 256,
    "num_heads": 40,
    "num_layers": 40,
    "window_size": (-1, -1),  # Global attention
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,

    # MoE Checkpoints
    "low_noise_checkpoint": "low_noise_model",
    "high_noise_checkpoint": "high_noise_model",

    # Inference
    "sample_shift": 12.0,
    "sample_steps": 40,
    "boundary": 0.875,
    "sample_guide_scale": (3.0, 4.0),  # (low_noise, high_noise)

    # Generation
    "num_train_timesteps": 1000,
    "sample_fps": 16,
    "frame_num": 81,
    "param_dtype": torch.bfloat16,
}
```

#### I2V-A14B (Image-to-Video)

```python
# From wan/configs/wan_i2v_A14B.py
i2v_A14B = {
    # Same as T2V except:
    "in_dim": 36,             # IMPORTANT: I2V changes patch_embedding input channels (16 -> 36)
    "sample_shift": 5.0,      # Lower shift for I2V
    "boundary": 0.900,        # Later switch to low-noise expert
    "sample_guide_scale": (3.5, 3.5),  # Equal guidance for both
}
```

> **I2V conditioning note (WAN 2.2 in this repo):** The image conditioning is done via **extra latent channels** (passed as `y` and concatenated
> to the noisy latents), *not* via CLIP tokens in the cross-attention context (that CLIP path exists for WAN 2.1 I2V).

#### TI2V-5B (Dense Model with High-Compression VAE)

| Parameter | Value |
|-----------|-------|
| **Parameters** | 5B (dense, no MoE) |
| **VAE Compression** | 4×16×16 (vs 4×8×8) |
| **Total Compression** | 4×32×32 with patchification |
| **FPS** | 24 (vs 16) |
| **Consumer GPU** | Yes (single 4090) |

> **Integration note:** Upstream WAN 2.2 includes TI2V-5B configs/sizes, but blissful-tuner currently does not expose a `--task ti2v-5B`
> entry in `WAN_CONFIGS` (even though some TI2V sizes appear in `SUPPORTED_SIZES`). Treat this section as upstream/paper reference unless you
> add the missing config wiring.

---

## Component Details

### 1. Wan-VAE (Spatio-Temporal VAE)

#### Architecture

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model Size** | 127M | Compact design |
| **Compression Ratio** | 4×8×8 | T×H×W |
| **Latent Channels** | 16 | z_dim |
| **Input Format** | T×H×W×3 | First frame processed separately, remaining frames in chunks of 4 |
| **Output Format** | (1 + (T-1)/4)×H/8×W/8×16 | Training typically uses `T = 4k + 1` (e.g. 81 → 21 latent frames) |
| **Normalization** | RMSNorm | Replaces GroupNorm for causality |

#### Causal 3D Convolution

```python
class CausalConv3d(nn.Conv3d):
    """Causal 3D convolution - future frames don't influence past"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Asymmetric temporal padding: (2*pad, 0) instead of (pad, pad)
        self._padding = (
            self.padding[2], self.padding[2],  # Width: symmetric
            self.padding[1], self.padding[1],  # Height: symmetric
            2 * self.padding[0], 0             # Temporal: causal (left only)
        )
        self.padding = (0, 0, 0)
```

#### Feature Cache Mechanism (Infinite Video Support)

```python
# Chunk-wise encoding/decoding for arbitrarily long videos
# - Process video in (1 + T/4) chunks
# - Each chunk handles 4 frames maximum
# - Cache features from preceding chunks for temporal continuity
# - Zero padding for initial chunk
CACHE_T = 2  # Cache last 2 frames for temporal convolutions
```

#### VAE Performance Benchmarks

| Metric | Wan-VAE | HunyuanVideo | CogVideoX | Open Sora Plan |
|--------|---------|--------------|-----------|----------------|
| **PSNR (720p)** | ~32.5 | ~31.8 | ~30.2 | ~29.5 |
| **Speed** | 2.5x faster | 1x baseline | 0.8x | 1.2x |
| **Parameters** | 127M | ~200M | ~150M | ~180M |

### 2. Text Encoder (umT5-XXL)

| Parameter | Value |
|-----------|-------|
| **Model** | umT5-XXL (Google) |
| **Parameters** | 5.3B |
| **Attention** | Bidirectional |
| **Languages** | Multilingual (Chinese, English, etc.) |
| **Max Length** | 512 tokens |
| **Output Dim** | 4096 → 5120 (projected) |
| **Precision** | bfloat16 |

#### Why umT5 over LLMs?

From ablation studies in the technical report:
1. **Bidirectional attention** better suited for diffusion models than causal LLMs
2. **Superior convergence** at same parameter scale
3. **Strong multilingual** support for Chinese and English visual text generation
4. **Better compositional understanding** compared to Qwen/GLM

### 3. Diffusion Transformer (DiT)

#### WanAttentionBlock Structure

```python
class WanAttentionBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, ...):
        # Normalization layers
        self.norm1 = WanLayerNorm(dim, eps)      # Pre-self-attn
        self.norm2 = WanLayerNorm(dim, eps)      # Pre-FFN
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True)  # Pre-cross-attn

        # Attention layers
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim)
        )

        # Modulation (shared AdaLN - reduces parameters by ~25%)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
```

#### Modulation (AdaLN-single)

```python
# 6 modulation slots per block:
# [shift1, scale1, gate1, shift2, scale2, gate2]
# Applied to: self-attn (1-3), FFN (4-6)

def forward(self, x, e, ...):
    # e: time embedding projected to [B, L, 6, dim]
    e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)

    # Self-attention with modulation
    y = self.self_attn(
        self.norm1(x) * (1 + e[1]) + e[0],  # scale + shift
        seq_lens, grid_sizes, freqs
    )
    x = x + y * e[2]  # gate

    # Cross-attention (no modulation)
    x = x + self.cross_attn(self.norm3(x), context, context_lens)

    # FFN with modulation
    y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
    x = x + y * e[5]
```

#### 3D RoPE (Rotary Position Embedding)

```python
# RoPE frequency dimensions split for 3D:
# Real-valued per-head dimension is `d = dim // num_heads` (A14B: 5120/40 = 128).
# The model constructs freqs using the real-dim split:
# - Temporal: d - 4*(d//6) dims
# - Height:   2*(d//6) dims
# - Width:    2*(d//6) dims
#
# Note: RoPE freqs are stored as complex pairs, so the freqs tensor’s last dim is `c = d//2`,
# and split sizes use the complex-pair dimensions.

def rope_params(max_seq_len, dim, theta=10000):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2) / dim)
    )
    return torch.polar(torch.ones_like(freqs), freqs)

# Application to Q, K (matches `rope_apply_inplace_cached` / `rope_apply` in this repo)
def rope_apply(x, grid_sizes, freqs):
    # x: [B, L, num_heads, head_dim]
    # freqs: [max_seq_len, head_dim//2] complex
    d = x.size(3)        # head_dim
    c = d // 2           # complex-pair dim

    # freqs split (complex-pair dims): [temporal, height, width]
    # Equivalent real-dim split: [d - 4*(d//6), 2*(d//6), 2*(d//6)]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # For each sample, expand freqs to match grid
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1)
        x_i = x_i * freqs_i  # Complex multiplication
```

---

## Weight Tensor Naming Patterns

### DiT Weights (per block)

```
blocks.{N}.self_attn.q.weight        # [5120, 5120]
blocks.{N}.self_attn.q.bias          # [5120]
blocks.{N}.self_attn.k.weight        # [5120, 5120]
blocks.{N}.self_attn.k.bias          # [5120]
blocks.{N}.self_attn.v.weight        # [5120, 5120]
blocks.{N}.self_attn.v.bias          # [5120]
blocks.{N}.self_attn.o.weight        # [5120, 5120]
blocks.{N}.self_attn.o.bias          # [5120]
blocks.{N}.self_attn.norm_q.weight   # [5120] (RMSNorm)
blocks.{N}.self_attn.norm_k.weight   # [5120] (RMSNorm)

blocks.{N}.cross_attn.q.weight       # [5120, 5120]
blocks.{N}.cross_attn.q.bias         # [5120]
blocks.{N}.cross_attn.k.weight       # [5120, 5120]
blocks.{N}.cross_attn.k.bias         # [5120]
blocks.{N}.cross_attn.v.weight       # [5120, 5120]
blocks.{N}.cross_attn.v.bias         # [5120]
blocks.{N}.cross_attn.o.weight       # [5120, 5120]
blocks.{N}.cross_attn.o.bias         # [5120]
blocks.{N}.cross_attn.norm_q.weight  # [5120]
blocks.{N}.cross_attn.norm_k.weight  # [5120]

blocks.{N}.ffn.0.weight              # [13824, 5120] (Linear + GELU)
blocks.{N}.ffn.0.bias                # [13824]
blocks.{N}.ffn.2.weight              # [5120, 13824]
blocks.{N}.ffn.2.bias                # [5120]

blocks.{N}.norm3.weight              # [5120] (cross_attn_norm)
blocks.{N}.norm3.bias                # [5120]

blocks.{N}.modulation                # [1, 6, 5120]
```

### Embeddings

```
patch_embedding.weight               # [5120, 16, 1, 2, 2] (Conv3d)
patch_embedding.bias                 # [5120]

text_embedding.0.weight              # [5120, 4096]
text_embedding.0.bias                # [5120]
text_embedding.2.weight              # [5120, 5120]
text_embedding.2.bias                # [5120]

time_embedding.0.weight              # [5120, 256]
time_embedding.0.bias                # [5120]
time_embedding.2.weight              # [5120, 5120]
time_embedding.2.bias                # [5120]

time_projection.1.weight             # [30720, 5120] (6 * dim)
time_projection.1.bias               # [30720]
```

> **Variant caveat:** for **I2V-A14B**, `in_dim=36`, so `patch_embedding.weight` is `[5120, 36, 1, 2, 2]`.

> **Checkpoint key caveats:** Some WAN checkpoints (notably 1.3B) use a `model.diffusion_model.` prefix (the loader strips it).
> If you train with `--compile` and swap high/low weights, keys inside `blocks.*` may be nested under `blocks.{N}._orig_mod.*`.

### Head

```
head.norm.weight                     # (none - no affine)
head.head.weight                     # [64, 5120] (out_dim * prod(patch_size))
head.head.bias                       # [64]
head.modulation                      # [1, 2, 5120]
```

### LoRA Target Modules (Blissful Tuner)

```python
# src/musubi_tuner/networks/lora_wan.py
# LoRA targeting is class-based, not a hard-coded module list.
WAN_TARGET_REPLACE_MODULES = ["WanAttentionBlock"]

# Default exclude patterns (anything matching these paths is skipped):
exclude_patterns.append(r".*(patch_embedding|text_embedding|time_embedding|time_projection|norm|head).*")
```

---

## Training Pipeline

### Flow Matching Objective

```python
# Rectified Flow formulation
x_t = t * x_1 + (1 - t) * x_0  # Linear interpolation
v_t = x_1 - x_0                 # Target velocity

# Loss
loss = MSE(model(x_t, t, context), v_t)
```

### Timestep Sampling

| Parameter | T2V Value | I2V Value |
|-----------|-----------|-----------|
| **Sampling** | `shift` | `shift` |
| **Discrete Flow Shift** | 12.0 | 5.0 |
| **Train Timesteps** | 1000 | 1000 |
| **Sample Steps** | 40 | 40 |

### Training Stages (from Technical Report)

| Stage | Resolution | Content | Duration |
|-------|------------|---------|----------|
| **1. Image Pre-training** | 256px | Text-to-Image | - |
| **2. Joint Training 1** | 256px images + 192px video | 5s clips @ 16fps | - |
| **3. Joint Training 2** | 480px | Images + 5s videos | - |
| **4. Joint Training 3** | 720px | Images + 5s videos | - |
| **5. Post-training** | 480px + 720px | Curated high-quality data | - |

### Negative Prompt (Default)

```
色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，
最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，
画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，
杂乱的背景，三条腿，背景人很多，倒着走
```

Translation: Vivid colors, overexposure, static, blurry details, subtitles, style, artwork, painting, still image, overall gray, worst quality, low quality, JPEG artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, still frame, cluttered background, three legs, many people in background, walking backwards.

---

## Memory & Performance

### GPU Requirements

| Config | VRAM | Notes |
|--------|------|-------|
| **14B Single GPU** | 80GB+ | With offloading |
| **14B Multi-GPU** | 8× 80GB | FSDP + Ulysses |
| **14B Consumer** | 24GB | Heavy offloading, slower |
| **5B (TI2V)** | 24GB | Single 4090 supported |
| **1.3B** | 8.19GB | Consumer-grade |

### Optimization Flags

```bash
# Memory optimization
--offload_model True           # Offload model to CPU
--convert_model_dtype          # Convert to param_dtype (bf16)
--t5_cpu                       # Keep T5 on CPU

# Multi-GPU (FSDP + Ulysses)
--dit_fsdp                     # Enable FSDP for DiT
--t5_fsdp                      # Enable FSDP for T5
--ulysses_size 8               # Sequence parallel size
```

### Inference Acceleration

| Technique | Speedup | Notes |
|-----------|---------|-------|
| **Diffusion Cache** | 1.62× | Attention + CFG caching |
| **FP8 GEMM** | 1.13× | DiT linear layers |
| **8-bit FlashAttention** | 1.27× | INT8 QK, FP8 PV |
| **Multi-GPU Scaling** | ~Linear | Up to 8 GPUs |

---

## Supported Resolutions

| Resolution | Aspect Ratio | Frame Count |
|------------|--------------|-------------|
| **480P** | 848×480, 480×848, 624×624 | 81 |
| **720P** | 1280×720, 720×1280, 960×960 | 81 |

### Frame Calculation

```python
# VAE temporal compression = 4x
# First-frame special: T frames -> (1 + (T-1)/4) latent frames (for T = 4k+1)
# Example: 81 frames -> 1 + 80/4 = 21 latent frames

# Sequence length calculation (matches training code)
latent_frames = 1 + (frames - 1) // 4  # e.g., 81 -> 21
lat_h = height // 8                   # VAE latent grid
lat_w = width // 8
seq_len = latent_frames * lat_h * lat_w // 4  # patch_size=(1,2,2) reduces spatial tokens by 2x2
```

---

## Official Prompting Guidelines

### Basic Formula (Beginners)

```
Prompt = Subject + Scene + Motion
```

**Example**: "A cat playing with a ball in a garden"

### Standard Formula

```
Prompt = Subject Description + Scene Description + Motion Description
         + Aesthetic Control + Stylization
```

**Aesthetic Control** includes:
- Light source and lighting environment
- Shot size (framing): close-up, medium shot, wide shot
- Camera angle: eye-level, high angle, low angle
- Lens type: wide-angle, telephoto, macro
- Camera movement: dolly, pan, tilt, tracking

**Example**: "A young woman with long flowing hair walks through a neon-lit cyberpunk city street at night. She turns to look at the camera with a mysterious smile. Medium shot, tracking camera following from behind, warm orange neon lighting contrasting with cool blue shadows, cinematic film grain, Blade Runner aesthetic"

### Image-to-Video Formula

```
Prompt = Motion Description + Camera Movement
```

Since the source image establishes subject, scene, and style, focus on:
- What should move and how
- Camera movement (or "static shot" / "fixed shot" for stationary)

**Example**: "The woman turns her head slowly to the right while her hair flows in the wind. Subtle dolly in."

### Multi-Shot Formula

```
Prompt = Overall Description + [Shot 1: Timestamp + Subject Behavior]
         + [Shot 2: Timestamp + Subject Behavior] + ...
```

**Example**:
```
A dramatic scene of a warrior facing a dragon.
[0:00-0:03] Wide shot establishing the battlefield, warrior stands ready
[0:03-0:06] Close-up on warrior's determined face
[0:06-0:10] The dragon breathes fire, warrior raises shield
```

### Camera Movement Terms

| Term | Description |
|------|-------------|
| **Dolly in/out** | Camera moves toward/away from subject |
| **Pan left/right** | Camera rotates horizontally |
| **Tilt up/down** | Camera rotates vertically |
| **Tracking shot** | Camera follows moving subject |
| **Aerial shot** | Camera from above, often moving |
| **Static shot** | Camera remains stationary |
| **Handheld** | Slight camera shake for realism |

### Shot Types

| Term | Description |
|------|-------------|
| **Extreme close-up (ECU)** | Very tight on face/detail |
| **Close-up (CU)** | Face fills frame |
| **Medium close-up (MCU)** | Head and shoulders |
| **Medium shot (MS)** | Waist up |
| **Full shot (FS)** | Entire body |
| **Wide shot (WS)** | Subject + environment |
| **Extreme wide shot (EWS)** | Vast landscape, subject small |

---

## Benchmark Performance

### Wan-Bench Results

| Metric | Wan 14B | Wan 1.3B | Sora | HunyuanVideo |
|--------|---------|----------|------|--------------|
| **Large Motion** | 0.415 | 0.468 | 0.482 | 0.413 |
| **Physical Plausibility** | 0.939 | 0.912 | 0.933 | 0.898 |
| **Smoothness** | 0.910 | 0.790 | 0.930 | 0.890 |
| **Image Quality** | 0.640 | 0.596 | 0.665 | 0.605 |
| **ID Consistency** | 0.946 | 0.938 | 0.925 | 0.935 |
| **Weighted Score** | **0.724** | 0.689 | 0.700 | 0.673 |

### VBench Results

| Model | Quality Score | Semantic Score | Total |
|-------|--------------|----------------|-------|
| **Wan 14B** | **86.67%** | **84.44%** | **86.22%** |
| Wan 1.3B | 84.92% | 80.10% | 83.96% |
| Sora | 85.51% | 79.35% | 84.28% |
| HunyuanVideo | 85.09% | 75.82% | 83.24% |

---

## Blissful Tuner Integration

### Training Command

```bash
# Cache latents
python wan_cache_latents.py --dataset_config config.toml \
    --vae /path/to/Wan2.1_VAE.pth --vae_chunk_size 32 --vae_tiling

# Cache text encoder outputs
python wan_cache_text_encoder_outputs.py --dataset_config config.toml \
    --t5 /path/to/models_t5_umt5-xxl-enc-bf16.pth --batch_size 16

# Train LoRA (WAN 2.2)
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
    wan_train_network.py \
    --task t2v-A14B \
    --dit /path/to/low_noise_model.safetensors \
    --dit_high_noise /path/to/high_noise_model.safetensors \
    --dataset_config config.toml \
    --network_module networks.lora_wan \
    --network_dim 32 \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0
```

### Key Training Parameters

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| **network_dim** | 32-64 | LoRA rank |
| **timestep_sampling** | shift | Flow matching |
| **discrete_flow_shift** | 12.0 (T2V), 5.0 (I2V) | Task-specific |
| **blocks_to_swap** | 0-39 | Memory optimization |
| **gradient_checkpointing** | True | Recommended |

---

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| **Wan 2.1** | Early 2025 | Initial release, T2V + I2V |
| **Wan 2.2** | Jul 28, 2025 | MoE architecture, TI2V-5B, Wan2.2-VAE |

---

## References

- [Wan 2.2 GitHub](https://github.com/Wan-Video/Wan2.2)
- [Wan 2.2 HuggingFace](https://huggingface.co/Wan-AI/)
- [Technical Report: arXiv:2503.20314](https://arxiv.org/abs/2503.20314)
- [Wan-Bench Evaluation](https://wan.video)
