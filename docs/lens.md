# Lens

Lens support currently covers the `lens_bf16` text-to-image MVP:

- image latent caching
- GPT-OSS selected-layer text caching
- standalone image generation
- LoRA training against the Lens DiT

Lens LoRA training can optionally use `--fp8_base --fp8_scaled` for a scaled-fp8 frozen DiT base. The first implementation does not support Lens-Turbo, bare fp8 base training, mxfp8 training, scaled-mm, reasoner API calls, GUI setup, image edit/control data, video data, or full-model finetuning.

## Model Download

Use the download helper to fetch the supported MVP files:

```bash
python lens_download.py --output_dir /path/to/models/lens
```

To inspect the exact file list without downloading:

```bash
python lens_download.py --output_dir /path/to/models/lens --dry_run
```

The helper downloads Lens DiT/VAE and GPT-OSS text encoder weights from `Comfy-Org/Lens`, plus the GPT-OSS config/tokenizer metadata from `microsoft/Lens`:

```text
/path/to/models/lens/
  diffusion_models/lens_bf16.safetensors
  text_encoders/gpt_oss_20b_nvfp4.safetensors
  vae/flux2-vae.safetensors
  text_encoder/config.json
  text_encoder/generation_config.json
  tokenizer/chat_template.jinja
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
```

Training and inference scripts never download model weights implicitly. GPT-OSS config and tokenizer metadata are resolved automatically from sibling local files when present, otherwise from `microsoft/Lens`; these are not user-facing CLI parameters. The default text encoder weight is the Comfy `text_encoders/gpt_oss_20b_nvfp4.safetensors` file, which the Lens loader converts from Comfy NVFP4 expert tensors into the local GPT-OSS module state dict before caching or inference.

## Cache Latents

```bash
python lens_cache_latents.py \
  --dataset_config /path/to/dataset.toml \
  --vae /path/to/models/lens/vae/flux2-vae.safetensors \
  --vae_dtype float32
```

Lens uses the FLUX.2 VAE loader and saves latent tensors with keys like `latents_{H}x{W}_{dtype}` under the `lens` architecture.

## Cache Text Encoder Outputs

```bash
python lens_cache_text_encoder_outputs.py \
  --dataset_config /path/to/dataset.toml \
  --text_encoder /path/to/models/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors \
  --text_encoder_dtype bfloat16
```

The cache stores four variable-length GPT-OSS feature tensors:

- `varlen_lens_ctx_0_{dtype}`
- `varlen_lens_ctx_1_{dtype}`
- `varlen_lens_ctx_2_{dtype}`
- `varlen_lens_ctx_3_{dtype}`

## Generate

```bash
python lens_generate_image.py \
  --dit /path/to/models/lens/diffusion_models/lens_bf16.safetensors \
  --vae /path/to/models/lens/vae/flux2-vae.safetensors \
  --text_encoder /path/to/models/lens/text_encoders/gpt_oss_20b_nvfp4.safetensors \
  --prompt "a compact ceramic tea set on a walnut table" \
  --save_path output/lens.png \
  --infer_steps 20 \
  --guidance_scale 5.0 \
  --seed 1
```

Use `--base_resolution 1024 --aspect_ratio 1:1` for Lens buckets, or `--image_size H W` for an explicit size divisible by 16.

## Train LoRA

```bash
accelerate launch lens_train_network.py \
  --dataset_config /path/to/dataset.toml \
  --dit /path/to/models/lens/diffusion_models/lens_bf16.safetensors \
  --vae /path/to/models/lens/vae/flux2-vae.safetensors \
  --sdpa \
  --mixed_precision bf16 \
  --network_module networks.lora_lens \
  --network_dim 16 \
  --network_alpha 16 \
  --learning_rate 1e-4 \
  --output_dir /path/to/output \
  --output_name lens_lora
```

`networks.lora_lens` only trains Lens transformer attention and MLP linear layers. It does not train the GPT-OSS text encoder.

`--blocks_to_swap N` is supported for low-VRAM LoRA training and uses Musubi's existing `ModelOffloader` on Lens transformer blocks.

For lower VRAM, add both `--fp8_base --fp8_scaled`. Lens does not support `--fp8_base` by itself; scaled fp8 keeps the frozen DiT base quantized and dequantizes Linear weights through Musubi's existing fp8 path while LoRA weights train normally.

## Dataset Limits

Lens MVP accepts image datasets only. Control images and video datasets are rejected during dataset preparation.
