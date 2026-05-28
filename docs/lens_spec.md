# Lens support spec

Status: reviewed and implementation scope locked.
Branch base: `upstream/main` at `78acafb`.

## 直接执行

### Review conclusion

The Lens support direction is accepted with two mandatory corrections:

- `Comfy-Org/Lens` is used only as the packaged weight source.
- GPT-OSS `config.json`, `generation_config.json`, and `tokenizer/*` are taken from the official `microsoft/Lens` repository so the text encoder can be reconstructed reproducibly.

The MVP supports only `lens_bf16` text-to-image inference, latent caching, GPT-OSS text-cache generation, and LoRA training. The first patch explicitly excludes `lens_turbo_bf16`, mxfp8/fp8 training, the reasoner API, GUI integration, image edit/control workflows, video data, and full-model finetuning.

Lens LoRA training supports Musubi block swap on local `LensTransformerBlock` modules through `ModelOffloader`.

### Source contract

Reference implementation: <https://github.com/microsoft/Lens>.

Weight source: <https://huggingface.co/Comfy-Org/Lens>.

MVP download list:

- `Comfy-Org/Lens/diffusion_models/lens_bf16.safetensors`
- `Comfy-Org/Lens/text_encoders/gpt_oss_20b_nvfp4.safetensors`
- `Comfy-Org/Lens/vae/flux2-vae.safetensors`
- `microsoft/Lens/text_encoder/config.json`
- `microsoft/Lens/text_encoder/generation_config.json`
- `microsoft/Lens/tokenizer/chat_template.jinja`
- `microsoft/Lens/tokenizer/tokenizer.json`
- `microsoft/Lens/tokenizer/tokenizer_config.json`

Training and inference scripts must not implicitly download weights. All runtime entry points require local paths.

### Architecture contract

Lens is a separate Musubi architecture, not a FLUX.2 variant:

- short cache name: `lens`
- full metadata name: `lens`
- SAI model-spec architecture: `lens`
- SAI implementation URL: `https://github.com/microsoft/Lens`
- dataset resolution step: `16`
- image datasets only
- no control images
- no video datasets

### Implemented file map

New files:

- `src/musubi_tuner/lens/__init__.py`
- `src/musubi_tuner/lens/lens_model.py`
- `src/musubi_tuner/lens/lens_text_encoder.py`
- `src/musubi_tuner/lens/lens_utils.py`
- `src/musubi_tuner/lens/resolution.py`
- `src/musubi_tuner/lens_cache_latents.py`
- `src/musubi_tuner/lens_cache_text_encoder_outputs.py`
- `src/musubi_tuner/lens_download.py`
- `src/musubi_tuner/lens_generate_image.py`
- `src/musubi_tuner/lens_train_network.py`
- `src/musubi_tuner/networks/lora_lens.py`
- root launcher shims: `lens_*.py`
- user documentation: `docs/lens.md`

Modified shared files:

- `src/musubi_tuner/dataset/architectures.py`
- `src/musubi_tuner/dataset/bucket.py`
- `src/musubi_tuner/dataset/cache_io.py`
- `src/musubi_tuner/dataset/image_video_dataset.py`
- `src/musubi_tuner/utils/sai_model_spec.py`
- `README.md`

### Reuse decisions

Reuse from Musubi:

- `flux_2.flux2_utils.load_ae` and FLUX.2 VAE implementation for `vae/flux2-vae.safetensors`
- `training.trainer_base.NetworkTrainer`
- existing image dataset, bucket, and cache path behavior
- existing LoRA core in `networks/lora.py`

Adapt from `microsoft/Lens`:

- `LensTransformer2DModel` and `LensTransformerBlock`
- GPT-OSS selected-layer extraction for `(5, 11, 17, 23)`
- Lens GPT-OSS chat-template prompt handling
- Lens resolution buckets
- dynamic FlowMatch shift

The implementation vendors only the required code shape as normal `torch.nn.Module` classes. It does not add `microsoft/Lens` as a runtime dependency, does not use `DiffusionPipeline`, and does not upgrade global `diffusers`.

### Cache format

Latent cache:

- Lens uses the existing FLUX.2 VAE loader.
- Saved latent key: `latents_{H}x{W}_{dtype}`.
- Tensor shape: `[128, H, W]`, where `H` and `W` are latent-grid dimensions.
- Metadata architecture: `lens`.

Text cache:

- GPT-OSS selected hidden layers `(5, 11, 17, 23)` are saved separately.
- The fixed Lens text offset `97` is removed before caching.
- Padding is trimmed before writing.
- Batch loading keeps varlen tensors as lists; `lens_train_network.py` pads them and rebuilds a bool mask.

Keys:

- `varlen_lens_ctx_0_{dtype}`
- `varlen_lens_ctx_1_{dtype}`
- `varlen_lens_ctx_2_{dtype}`
- `varlen_lens_ctx_3_{dtype}`

### LoRA scope

`networks/lora_lens.py` targets only `Linear` modules inside `LensTransformerBlock`:

- `attn.img_qkv`
- `attn.txt_qkv`
- `attn.to_out.0`
- `attn.to_add_out`
- `img_mlp.w1/w2/w3`
- `txt_mlp.w1/w2/w3`

It excludes normalization, RoPE/position embedding, timestep embedding, modulation layers, and final projection by default. GPT-OSS text encoder LoRA is out of scope.

`LensTransformer2DModel` is a local `torch.nn.Module` implementation, not a runtime import from `microsoft/Lens`. It implements the Musubi block-swap lifecycle methods expected by the shared trainer:

- `enable_block_swap`
- `move_to_device_except_swap_blocks`
- `prepare_block_swap_before_forward`
- `switch_block_swap_for_inference`
- `switch_block_swap_for_training`

### CLI contract

New root shims and package entry scripts:

- `lens_download.py`
- `lens_cache_latents.py`
- `lens_cache_text_encoder_outputs.py`
- `lens_generate_image.py`
- `lens_train_network.py`

`lens_download.py --dry_run` must print the exact MVP download list and perform no network writes.

### Acceptance checks

Minimum local checks:

1. `lens_download.py --dry_run` prints the exact file list above.
2. All Lens CLI `--help` commands exit successfully.
3. A mocked batch with two prompt lengths roundtrips through latent cache and four-layer text cache loading.
4. Existing `flux_2_generate_image.py --help`, `qwen_image_train_network.py --help`, and `zimage_train_network.py --help` still pass in an environment with the same optional training dependencies.
5. With real weights, one prompt generates a non-empty PNG.
6. With real weights and a tiny dataset, one LoRA optimizer step completes without NaN and produces safetensors metadata for `lens/lora`.

## 深度交互

### Challenge 1: avoid a hidden FLUX.2 fork

Lens reuses a FLUX.2 VAE latent format, but the denoiser is a Lens-specific joint image/text transformer. Treating it as a FLUX.2 variant would blur cache names, metadata, and target modules. A separate `lens` architecture is the lower-risk boundary.

### Challenge 2: Comfy weights are not enough

Comfy's repository is convenient for safetensors, but it does not replace the official GPT-OSS tokenizer/config contract. If text encoding is reconstructed from only safetensors, prompt embeddings are not reproducible. The implementation therefore downloads Comfy weights plus official metadata.

### Challenge 3: narrow the first training target

`lens_turbo` and mxfp8 are inference and memory-pressure questions, not prerequisites for proving LoRA training. Supporting them before the bf16 path is stable would multiply loader and optimizer failure modes. The MVP intentionally trains only `lens_bf16`.

### Challenge 4: keep dependency blast radius small

Upgrading global `diffusers` to match the official reference would affect unrelated Musubi architectures. The lower-cost path is to vendor the Lens transformer/text-encoder logic and reuse Musubi's existing VAE and training infrastructure.
