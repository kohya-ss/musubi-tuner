# Ideogram 4 FP8

This adapter supports the Comfy-Org single-file component layout for Ideogram 4 FP8.
The model license is non-commercial; read and accept the relevant terms before downloading or using the weights.

## Download

Download the component files yourself from https://huggingface.co/Comfy-Org/Ideogram-4 and pass local paths to the scripts.
The scripts do not accept `--repo_id` and do not download Comfy-Org weights for you.

Required files:

- `diffusion_models/ideogram4_fp8_scaled.safetensors`
- `diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors`
- `text_encoders/qwen3vl_8b_fp8_scaled.safetensors`
- `vae/flux2-vae.safetensors`

Tokenizer and text-encoder config are downloaded automatically from `Qwen/Qwen3-VL-8B-Instruct`; there are no tokenizer/config CLI arguments.

## Cache

Latent cache:

```powershell
python src/musubi_tuner/ideogram4_cache_latents.py `
  --dataset_config path\to\dataset.toml `
  --vae path\to\flux2-vae.safetensors `
  --vae_dtype bfloat16
```

Text cache:

```powershell
python src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py `
  --dataset_config path\to\dataset.toml `
  --text_encoder path\to\qwen3vl_8b_fp8_scaled.safetensors `
  --text_cache_dtype bf16
```

Plain text captions are accepted by default. Add `--validate_caption_structure` to check official structured JSON
captions before caching; combine it with `--warn_on_caption_issues` to warn instead of failing.

Text cache size is large because each token stores 53,248 channels. Approximate BF16 cost:

- 128 tokens: 13.6 MB/image, 13.6 GB/1k images
- 256 tokens: 27.3 MB/image, 27.3 GB/1k images
- 512 tokens: 54.5 MB/image, 54.5 GB/1k images
- 2048 tokens: 218.1 MB/image, 218.1 GB/1k images

`--text_cache_dtype fp8_e4m3fn` halves the disk cost but is an explicit quality/precision tradeoff.

## Generate

```powershell
python src/musubi_tuner/ideogram4_generate_image.py `
  --dit path\to\ideogram4_fp8_scaled.safetensors `
  --unconditional_dit path\to\ideogram4_unconditional_fp8_scaled.safetensors `
  --text_encoder path\to\qwen3vl_8b_fp8_scaled.safetensors `
  --vae path\to\flux2-vae.safetensors `
  --prompt "A structured JSON or plain text prompt" `
  --image_size 1024 1024 `
  --sampler_preset V4_DEFAULT_20 `
  --initial_sigma 1.004 `
  --save_path outputs\ideogram4.png
```

Sampler presets:

- `V4_QUALITY_48`: 45 main steps at guidance 7, then 3 polish steps at guidance 3
- `V4_DEFAULT_20`: 18 main steps at guidance 7, then 2 polish steps at guidance 3
- `V4_TURBO_12`: 11 main steps at guidance 7, then 1 polish step at guidance 3

Ideogram 4 v1 uses the official asymmetric CFG path. `negative_prompt` is ignored.
`--initial_sigma` overrides the first denoising sigma and defaults to `1.004`.

## Train LoRA

```powershell
python src/musubi_tuner/ideogram4_train_network.py `
  --dataset_config path\to\dataset.toml `
  --dit path\to\ideogram4_fp8_scaled.safetensors `
  --network_module networks.lora_ideogram4 `
  --output_dir outputs\i4_lora `
  --output_name my_i4_lora `
  --mixed_precision bf16 `
  --sdpa
```

Sampling during training requires the remaining local components:

```powershell
python src/musubi_tuner/ideogram4_train_network.py `
  --dataset_config path\to\dataset.toml `
  --dit path\to\ideogram4_fp8_scaled.safetensors `
  --unconditional_dit path\to\ideogram4_unconditional_fp8_scaled.safetensors `
  --text_encoder path\to\qwen3vl_8b_fp8_scaled.safetensors `
  --vae path\to\flux2-vae.safetensors `
  --network_module networks.lora_ideogram4 `
  --sample_prompts path\to\prompts.txt `
  --sample_every_n_steps 500 `
  --sampler_preset V4_DEFAULT_20 `
  --initial_sigma 1.004 `
  --mixed_precision bf16 `
  --sdpa
```

LoRA v1 trains only the conditional transformer. It targets `attention.qkv`, `attention.o`, and `feed_forward.w1/w2/w3` inside `Ideogram4TransformerBlock`, including official `Fp8Linear` layers. Merging LoRA back into FP8 base weights is not supported.
