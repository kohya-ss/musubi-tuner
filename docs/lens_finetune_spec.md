# Lens Full Finetune Spec

## Scope

Lens full finetuning trains only `LensTransformer2DModel` from `lens_bf16` weights. It reuses the existing Lens latent cache and four-layer GPT-OSS text cache. GPT-OSS text encoder weights and the FLUX.2 VAE stay frozen and are loaded only for sample generation when requested.

## CLI

- Root shim: `lens_train.py`
- Module entrypoint: `src/musubi_tuner/lens_train.py`
- Required training inputs: `--dataset_config`, `--dit`, `--sdpa`
- Optional sampling inputs: `--sample_prompts` requires `--vae` and `--text_encoder`
- Recommended timestep setting: `--timestep_sampling flux2_shift`

## Supported

- cached image datasets under Lens architecture
- fp32 DiT finetune by default
- `--mixed_precision bf16 --full_bf16`
- gradient checkpointing
- block swap
- Adafactor fused backward pass
- saving local Lens DiT state dicts with non-LoRA SAI Lens metadata

## Rejected

- LoRA/network arguments
- `lens_turbo_bf16`
- `--fp8_base`
- `--fp8_scaled`
- non-SDPA attention modes
- text encoder finetuning
- VAE finetuning
- control/edit/video data

## Save Format

Checkpoints are plain Lens DiT `state_dict` safetensors using local `LensTransformer2DModel` keys. Metadata records `modelspec.architecture=lens`, `modelspec.implementation=https://github.com/microsoft/Lens`, `ss_training_type=full-finetune`, and `ss_full_finetune=True`.
