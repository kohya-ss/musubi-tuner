# LoHa & LoKr Training Guide

LoHa (Low-Rank Hadamard Product) and LoKr (Low-Rank Kronecker Product) are alternative parameter-efficient fine-tuning methods alongside standard LoRA. Both produce smaller weight deltas with different mathematical decompositions.

## Overview

| Method | Decomposition | Module | Conv2d Support |
|--------|--------------|--------|----------------|
| LoRA | `lora_up @ lora_down` | `networks.lora` / `networks.lora_wan` | Yes |
| LoHa | `(w1a @ w1b) * (w2a @ w2b)` (Hadamard) | `networks.loha` | Yes |
| LoKr | `kron(w1, w2) * scale` (Kronecker) | `networks.lokr` | No (v1, Linear-only) |

## Supported Architectures

LoHa and LoKr share the same architecture registry. All architectures supported by LoRA training are also supported:

- WAN 2.1/2.2
- HunyuanVideo
- HunyuanVideo 1.5
- FramePack
- FLUX.1 Kontext
- FLUX.2 (Dev, Klein 4B, Klein 9B)
- Qwen-Image (all variants)
- Z-Image-Turbo
- Kandinsky 5

## Training with LoHa

```bash
accelerate launch --mixed_precision bf16 wan_train_network.py \
    --task t2v-A14B \
    --dit /path/to/dit.safetensors \
    --dataset_config config.toml \
    --network_module networks.loha \
    --network_dim 32 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0
```

LoHa supports Conv2d layers in addition to Linear layers. Use `--conv_dim` and `--conv_alpha` to configure Conv2d rank separately:

```bash
    --network_dim 32 --network_alpha 16 \
    --conv_dim 16 --conv_alpha 8
```

## Training with LoKr

```bash
accelerate launch --mixed_precision bf16 wan_train_network.py \
    --task t2v-A14B \
    --dit /path/to/dit.safetensors \
    --dataset_config config.toml \
    --network_module networks.lokr \
    --network_dim 4 \
    --network_alpha 1 \
    --timestep_sampling shift \
    --discrete_flow_shift 12.0
```

### LoKr-Specific Options

Pass LoKr options via `--network_args`:

```bash
    --network_args "factor=-1"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `factor` | `-1` | Kronecker factorization factor. `-1` = auto-balance (recommended). Positive integer forces that factor for dimension splitting. |

### How Factor Works

LoKr decomposes each weight matrix `W (out_dim x in_dim)` into a Kronecker product:

```
W = kron(w1, w2) * scale
```

Where `w1` is shape `(out_l, in_m)` and `w2` is shape `(out_k, in_n)`, with `out_l * out_k = out_dim` and `in_m * in_n = in_dim`.

- `factor=-1` (default): Finds the most balanced factorization automatically. For a 512-dim layer, this might choose `(16, 32)`.
- `factor=N` (positive): Forces `N` as one of the factors. For example, `factor=4` on a 512-dim layer gives `(4, 128)`.

The `w2` component can be either:
- **Low-rank** (`w2_a @ w2_b`): When `network_dim < max(out_k, in_n) / 2`
- **Full matrix**: When rank is large relative to the factored dimensions

### LoKr v1 Limitations

- **Linear-only**: Conv2d layers are automatically skipped with a warning. `--conv_dim` / `--conv_alpha` are ignored if provided.
- Full Conv2d support is planned for a future update.

## Factor Persistence

LoKr saves the factor value in the checkpoint for reproducible loading:

1. **State dict buffer**: `lokr_factor` tensor key (works with both `.safetensors` and `.pt`)
2. **Safetensors metadata**: `ss_lokr_factor` string (human-readable, for tooling)

When loading a LoKr checkpoint:
- Explicit `factor=` kwarg takes highest precedence
- Persisted buffer value used if no explicit override
- Default `-1` if nothing persisted (with warning)
- Mismatch between explicit and persisted triggers a warning

## Inference

### Native Inference (Recommended)

LoHa and LoKr weights are merged natively during model loading. Use them exactly like LoRA weights:

```bash
python wan_generate_video.py \
    --task t2v-A14B \
    --dit /path/to/dit.safetensors \
    --vae /path/to/vae.pth \
    --t5 /path/to/t5.pth \
    --prompt "your prompt" \
    --lora_weight /path/to/lokr_lora.safetensors
```

The backend automatically detects the network type (LoRA, LoHa, or LoKr) from the state dict keys and routes to the appropriate merge function.

### LyCORIS Backend (Force Override)

To force all weight merging through the LyCORIS library instead of native merge, use `--prefer_lycoris`. Requires LyCORIS to be installed:

```bash
python wan_generate_video.py \
    ... \
    --lora_weight /path/to/lora.safetensors \
    --prefer_lycoris
```

Note: `--lycoris` is a deprecated alias for `--prefer_lycoris` and will be removed in a future version.

## Conversion

LoHa and LoKr weights are supported by the format converters:

```bash
# Convert from sd-scripts to Diffusers format
python convert_lora.py --input lokr_lora.safetensors --output converted.safetensors --target other

# Convert from Diffusers back to sd-scripts format
python convert_lora.py --input converted.safetensors --output reconverted.safetensors --target default
```

For Z-Image LoKr QKV conversion (ComfyUI compatibility), use the dedicated Z-Image converter:

```bash
python src/musubi_tuner/networks/convert_z_image_lora_to_comfy.py lokr_lora.safetensors comfy_lora.safetensors --lokr_rank 64
```

Converters preserve:
- All LoHa keys (`hada_w1_a`, `hada_w1_b`, `hada_w2_a`, `hada_w2_b`)
- All LoKr keys (`lokr_w1`, `lokr_w2` / `lokr_w2_a` + `lokr_w2_b`)
- The `lokr_factor` buffer and `ss_lokr_factor` metadata
- QKV-split LoRA keys (for hybrid LoKr+LoRA dicts)

## Hybrid State Dicts

After QKV conversion, a state dict may contain mixed key types (e.g., `lokr_*` for non-QKV layers + `lora_*` for QKV layers). The merge dispatch handles this by processing each key family independently in order: LoHa, LoKr, then LoRA.
