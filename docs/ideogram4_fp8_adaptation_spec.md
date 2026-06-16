# Ideogram 4 FP8 Adaptation Spec

Branch: `codex/ideogram4-fp8-adaptation`

Base: `upstream/main` at `de648d36b719d21b24e424da0a2774cc9de29e84`

## Goal

Add first-class Musubi Tuner support for the ComfyUI single-file component layout in
`Comfy-Org/Ideogram-4`, using the official `ideogram-oss/ideogram4` implementation as the
behavioral reference.

The useful end state is not "the model imports". The useful end state is:

- deterministic text-to-image inference from the Comfy-Org FP8 safetensors,
- latent and text-encoder-output pre-caching compatible with this repo's dataset pipeline,
- LoRA training through the existing `NetworkTrainer` flow,
- a documented CLI path that clearly states the non-commercial model-license boundary.

## Hard Facts From The Reference

Official sources:

- Target weights: https://huggingface.co/Comfy-Org/Ideogram-4
- Original weights: https://huggingface.co/ideogram-ai/ideogram-4-fp8
- Code: https://github.com/ideogram-oss/ideogram4
- Architecture docs: https://github.com/ideogram-oss/ideogram4/blob/main/docs/model_architecture.md
- Pipeline docs: https://github.com/ideogram-oss/ideogram4/blob/main/docs/pipeline.md
- Prompting docs: https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md
- Model license: https://github.com/ideogram-oss/ideogram4/blob/main/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL

Facts to preserve:

- The original FP8 model is listed as 9.3B params, FP8 quantized, non-commercial, and without
  Diffusers support in the official model zoo. The target Comfy-Org repository is a repackaged
  ComfyUI layout of the original FP8 weights, not a separate architecture.
- The backbone is a 34-layer, fully single-stream DiT. Text tokens and image latent tokens are
  concatenated into one sequence and processed by the same transformer blocks.
- Transformer config:
  - `emb_dim = 4608`
  - `num_layers = 34`
  - `num_heads = 18`
  - `intermediate_size = 12288`
  - `adanln_dim = 512`
  - `in_channels = 128`
  - `rope_theta = 5_000_000`
  - `mrope_section = (24, 20, 20)`
  - max text tokens = `2048`
- Text encoder is frozen `Qwen3-VL-8B-Instruct` in text-only mode. Hidden states from layers
  `(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)` are concatenated, producing
  `4096 * 13 = 53248` feature channels.
- The reference pipeline has two DiTs: `conditional_transformer` and `unconditional_transformer`.
  Asymmetric CFG uses the conditional text+image sequence for the positive branch and an
  image-only sequence with zero text features for the negative branch.
- The Comfy-Org checkpoint file tree confirms those are two separately published DiT files:
  - `diffusion_models/ideogram4_fp8_scaled.safetensors`
    - LFS sha256 `49a946f1b0f8bcf5eab7d3b1ecc7b453c104e034cb1b592032745692724bd306`
    - size `9,280,741,285` bytes
    - safetensors metadata `model_type=ideogram4_cond`
  - `diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors`
    - LFS sha256 `9b359007dae162cca7591d00868feea733eb7c56e56e3a214a4d5a9a2a07cd60`
    - size `9,280,741,293` bytes
    - safetensors metadata `model_type=ideogram4_uncond`
  - Budget as two DiTs, not one reused DiT.
- Additional Comfy-Org component files:
  - `text_encoders/qwen3vl_8b_fp8_scaled.safetensors`
    - LFS sha256 `4ba424cf62e51392e4d1a39933e803706f4e823c1065f36aaf149c6453f66bcd`
    - size `10,588,637,512` bytes
  - `vae/flux2-vae.safetensors`
    - LFS sha256 `868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3`
    - size `336,211,292` bytes
- Official FP8 is weight-only E4M3 FP8 with per-row scale buffers and BF16 activations. It is not
  the same thing as this repo's `--fp8_scaled` optimization path.
- Despite the Comfy filenames ending in `fp8_scaled`, safetensors metadata confirms the DiT and
  text encoder use official `*.weight_scale` keys, not Musubi's `*.scale_weight` format.
- VAE is a KL autoencoder with 32 latent channels and 8x spatial compression. The DiT consumes
  2x2 latent patches, so image token grids are `height / 16` by `width / 16`, and each token has
  `32 * 2 * 2 = 128` channels.
- Official inference supports resolutions that are multiples of 16, 256 to 2048 per side, with
  aspect ratio up to 6:1.
- Prompts are plain strings at the pipeline boundary, but the model was trained on structured JSON
  captions. Plain text works only as a degraded path unless expanded by a magic-prompt layer.

## Local Fit

Use the existing architecture pattern instead of inventing a new training stack:

- `src/musubi_tuner/ideogram4/`
  - `constants.py`
  - `ideogram4_model.py`
  - `ideogram4_autoencoder.py`
  - `ideogram4_scheduler.py`
  - `ideogram4_utils.py`
  - `ideogram4_quantized_loading.py`
  - `caption_verifier.py`
  - `latent_norm.py`
- `src/musubi_tuner/ideogram4_cache_latents.py`
- `src/musubi_tuner/ideogram4_cache_text_encoder_outputs.py`
- `src/musubi_tuner/ideogram4_train_network.py`
- `src/musubi_tuner/ideogram4_generate_image.py`
- `src/musubi_tuner/networks/lora_ideogram4.py`
- `docs/ideogram4.md`

Architecture registration:

- Add `ARCHITECTURE_IDEOGRAM4 = "i4"` and `ARCHITECTURE_IDEOGRAM4_FULL = "ideogram4"` in
  `src/musubi_tuner/dataset/architectures.py`.
- Add Ideogram-specific cache save helpers in `src/musubi_tuner/dataset/cache_io.py`.
- Let `ImageDataset` use the normal image-dataset path. Ideogram 4 is text-to-image only for this
  initial adaptation; no control images or video path.

Dependency stance:

- Current repo pins `transformers==4.56.1`, which is newer than the official minimum
  `transformers>=4.49.0`.
- Official code asks for `torch>=2.11`; this repo currently allows lower torch versions depending
  on CUDA extra. Do not bump all torch extras blindly. First test which minimum version can load
  Qwen3-VL and float8 buffers correctly.
- Do not depend on Diffusers for FP8 until the official "Diffusers Support: No" contradiction is
  resolved by a local smoke test.
- The target Comfy-Org repository does not include tokenizer/config subfolders. Follow this repo's
  existing pattern for local/single-file weights: load weights from the local/Comfy file, but
  automatically download tokenizer/config from the canonical model repo. For Ideogram 4, default
  tokenizer/config source is `ideogram-ai/ideogram-4-fp8`.

## Weight Loading

Support explicit local component paths only. The implementation should not download Comfy-Org
weights by `repo_id`, should not infer a remote repository layout at runtime, and should not expose
a manual `--repo_id` argument. The docs should tell users to download the component files they need
from https://huggingface.co/Comfy-Org/Ideogram-4, then pass the local paths, matching this repo's
other single-file/local-weight model workflows.

Required Comfy component files:

- conditional DiT: `diffusion_models/ideogram4_fp8_scaled.safetensors`
- unconditional DiT: `diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors`
- text encoder: `text_encoders/qwen3vl_8b_fp8_scaled.safetensors`
- VAE: `vae/flux2-vae.safetensors`
- tokenizer/config: not present in the Comfy repo; auto-download by default from
  `ideogram-ai/ideogram-4-fp8`.

Implementation rules:

- Add constants:
  - `IDEOGRAM4_OFFICIAL_REPO_ID = "ideogram-ai/ideogram-4-fp8"`
  - `IDEOGRAM4_TOKENIZER_SUBFOLDER = "tokenizer"`
  - `IDEOGRAM4_TEXT_ENCODER_CONFIG_SUBFOLDER = "text_encoder"`
- CLI required/local-path arguments:
  - `--dit path/to/ideogram4_fp8_scaled.safetensors`
  - `--unconditional_dit path/to/ideogram4_unconditional_fp8_scaled.safetensors`
  - `--text_encoder path/to/qwen3vl_8b_fp8_scaled.safetensors`
  - `--vae path/to/flux2-vae.safetensors`
- Do not add tokenizer/config CLI arguments. Tokenizer/config are internal automatic downloads,
  like other Musubi single-file/local-weight loaders.
- Do not add `--repo_id` or any equivalent remote Comfy-Org download argument. Model component
  weight paths are supplied by the user as local files.
- Only tokenizer/config should come from `IDEOGRAM4_OFFICIAL_REPO_ID`. Do not load transformer,
  text encoder, or VAE weights from the official repo; those weights come from explicit local
  component paths.
- Port the official `Fp8Linear` path instead of routing it through existing `scale_weight` FP8
  monkey patches. Official checkpoints use `weight_scale`; current Musubi optimized FP8 uses
  `scale_weight`.
- For the text encoder, preserve the official special case where FP8 text-encoder config sets
  `ideogram_fp8_weight_only` and requires rebuilding via `AutoModel.from_config`.
- Use current `load_split_weights` / `MemoryEfficientSafeOpen` style for local component paths
  where possible. Do not use `huggingface_hub.hf_hub_download` for Comfy component weights in the
  runtime scripts.
- Use `AutoTokenizer.from_pretrained(IDEOGRAM4_OFFICIAL_REPO_ID, subfolder=IDEOGRAM4_TOKENIZER_SUBFOLDER)` and
  `AutoConfig.from_pretrained(IDEOGRAM4_OFFICIAL_REPO_ID, subfolder=IDEOGRAM4_TEXT_ENCODER_CONFIG_SUBFOLDER)`
  for the text side. This mirrors current project patterns such as Qwen-Image loading tokenizer
  assets from `Qwen/Qwen-Image` while accepting separately supplied weights.
- Keep conditional and unconditional transformers as separate modules. Training can initially
  optimize only the conditional transformer for LoRA, but sample inference must still load/use the
  unconditional transformer for CFG.
- Do not ship an implementation that assumes the pipeline-doc pseudocode's single `dit(...)` call
  means shared weights. The official code path and Comfy-Org file tree both prove separate
  conditional/unconditional weights for the FP8 release.
- Do not call this an all-in-one checkpoint. It is a single-file-per-component layout.

## Latent Caching

Add `ideogram4_cache_latents.py`.

Input:

- standard image dataset items,
- RGB only,
- bucket sizes must be divisible by 16.

Processing:

1. Normalize image pixels to the VAE input convention used by the official autoencoder.
2. Encode with the VAE encoder and use deterministic mode/mean latents shaped `(B, 32, H/8, W/8)`.
3. Patchify the VAE mean into token-grid shape `(B, 128, H/16, W/16)` with the single official
   channel order `(patch_i, patch_j, latent_channel)`.
4. Save the raw patchified VAE mean as `latents_1xHxW_<dtype>` under `ARCHITECTURE_IDEOGRAM4_FULL`.

The cache stores raw patchified latents, not normalized model latents. Training applies
`(token_grid - latent_shift) / latent_scale` once before adding noise. Sampling applies
`token_grid * latent_scale + latent_shift` before unpatchifying. The 128-dimension latent
normalization constants are ordered for `(patch_i, patch_j, latent_channel)`, so caches created with
the older `(latent_channel, patch_i, patch_j)` order must be rebuilt.

## Text Encoder Caching

Add `ideogram4_cache_text_encoder_outputs.py`.

Input:

- captions as raw strings,
- recommended captions are structured JSON serialized with `ensure_ascii=False`,
- optional `--warn_on_caption_issues` for non-blocking verification.

Processing:

1. Tokenize with Qwen3-VL chat template exactly as official `_tokenize` does.
2. Reject prompts above 2048 text tokens.
3. Run Qwen3-VL in text-only mode.
4. Capture hidden states from the 13 official activation layers.
5. Concatenate to `(text_len, 53248)`.
6. Save variable-length features as `varlen_i4_llm_features_<dtype>`.

Do not cache only a pooled vector. The transformer consumes per-token multi-layer features.

Text cache size is a first-order constraint:

- Feature width is `53248`.
- BF16 cache cost is `106,496` bytes per token, before safetensors overhead.
- FP8 activation cache cost is `53,248` bytes per token, before overhead, but must be treated as
  a quality/precision option rather than silently forced.
- Approximate BF16 disk cost:
  - 128 tokens: `13.6 MB/image`, `13.6 GB/1k images`
  - 256 tokens: `27.3 MB/image`, `27.3 GB/1k images`
  - 512 tokens: `54.5 MB/image`, `54.5 GB/1k images`
  - 2048 tokens: `218.1 MB/image`, `218.1 GB/1k images`
- Approximate FP8 disk cost is half of BF16:
  - 256 tokens: `13.6 MB/image`, `13.6 GB/1k images`
  - 2048 tokens: `109.1 MB/image`, `109.1 GB/1k images`

Implementation requirement:

- Add `--text_cache_dtype {bf16,fp8_e4m3fn,float32}` to `ideogram4_cache_text_encoder_outputs.py`.
- Default to `bf16` for first correctness parity unless local tests show FP8 cached activations do
  not degrade LoRA training. Document the disk cost and let users opt into FP8 cache explicitly.
- Store the selected cache dtype in safetensors metadata, and cast to `network_dtype` in `call_dit`.

Batch-time reconstruction:

- Pad variable-length `i4_llm_features` to batch max text length.
- Build text position ids as `[pos, pos, pos]`.
- Build image position ids from `(0, h, w) + IMAGE_POSITION_OFFSET`.
- Build `indicator` with `LLM_TOKEN_INDICATOR` and `OUTPUT_IMAGE_INDICATOR`.
- Build `segment_ids` using the existing packed-batch semantics.

## Training Forward

Add `Ideogram4NetworkTrainer(NetworkTrainer)`.

`handle_model_specific_args`:

- default DiT compute dtype: BF16 unless mixed precision explicitly requires FP32,
- `_i2v_training = False`,
- `_control_training = False`,
- `default_guidance_scale = 7.0` for sampling,
- use a new sampler preset argument rather than overloading `--discrete_flow_shift`,
- set the default training timestep sampler to Ideogram 4 logit-normal, not the repo-wide
  `sigma` default.

Training timestep sampling:

- Add an Ideogram-specific sampler path whose model-facing timestep matches official
  `LogitNormalSchedule`. `NetworkTrainer` treats `t` as the noise coefficient
  (`t=0` clean, `t=1` noise), while Ideogram 4's DiT receives the inverse convention
  (`model_t=0` noise, `model_t=1` clean). Therefore sample the training noise
  coefficient as:
  `t_noise = clamp(sigmoid(mu_adjusted + std * normal()), 1 - t_max, 1 - t_min)`.
  `call_dit` then passes `model_t = 1 - t_noise`, matching the official inference
  schedule's `1 - sigmoid(mu_adjusted + std * normal())`.
- Use the same resolution adjustment as inference:
  `mu_adjusted = train_mu + 0.5 * log(num_pixels / (512 * 512))`.
- Start defaults:
  - `--ideogram4_train_mu 0.0`
  - `--ideogram4_train_std 1.5`
  - `--weighting_scheme none`
- Do not approximate this with the existing `timestep_sampling=logsnr` without a test. The current
  base implementation computes `t = sigmoid(-logsnr / 2)`, which is not exactly the official
  schedule.
- Add a histogram/debug test comparing sampled model-facing `model_t` against the official
  scheduler for at least 512x512 and 2048x2048 buckets.

`scale_shift_latents`:

- no-op if latent cache already saves model-normalized latents,
- otherwise apply `(latents - latent_shift) / latent_scale`.

`call_dit`:

1. Receive `noisy_model_input = (1 - t) * latents + t * noise` from `NetworkTrainer`.
2. Patchify `noisy_model_input` to `(B, image_tokens, 128)`.
3. Pad cached `i4_llm_features` to batch max text length.
4. Build text+image sequence metadata.
5. Concatenate zero latent padding for text slots with image latent tokens.
6. Call the conditional transformer only.
7. Slice image-token output and unpatchify to latent shape.
8. Use target `latents - noise`.

The `latents - noise` target follows the inference update direction because sampling integrates
from high noise toward clean latents with `z = z + v * (s - t)`, where `s - t` is positive along
the Ideogram 4 model-time path. Add a one-step unit test before trusting training loss.

## Sampling

Add `ideogram4_generate_image.py` and sampling support inside `ideogram4_train_network.py`.

Presets:

- `V4_QUALITY_48`: 48 steps, final 3 polish steps at guidance 3, all earlier steps at guidance 7,
  `mu=0.0`, `std=1.5`.
- `V4_DEFAULT_20`: 20 steps, final 2 polish steps at guidance 3, all earlier steps at guidance 7,
  `mu=0.0`, `std=1.75`.
- `V4_TURBO_12`: 12 steps, final 1 polish step at guidance 3, all earlier steps at guidance 7,
  `mu=0.5`, `std=1.75`.

Sampling math:

- use official logit-normal schedule with resolution adjustment,
- initialize noise token grid as `(B, grid_h * grid_w, 128)`,
- loop from high noise to low noise,
- positive branch: conditional transformer with padded text+image sequence,
- negative branch: unconditional transformer with image-only sequence,
- combine as `v = gw * v_cond + (1 - gw) * v_uncond`,
- Euler update `z = z + v * (s - t)`,
- decode by undoing patchification and applying `z * latent_scale + latent_shift`.

## LoRA Targets

Add `src/musubi_tuner/networks/lora_ideogram4.py`.

Initial target module:

- `Ideogram4TransformerBlock`

Default exclude patterns:

- modulation layers: `.*adaln_modulation.*`
- final normalization/projection can stay excluded initially until inference parity is proven.

Candidate trainable linears inside target blocks:

- `attention.qkv`
- `attention.o`
- `feed_forward.w1`
- `feed_forward.w2`
- `feed_forward.w3`

Keep text encoder frozen. Do not train VAE.

FP8 base-layer integration is mandatory:

- Official FP8 swaps `nn.Linear` layers into custom `Fp8Linear` modules.
- Current `src/musubi_tuner/networks/lora.py` only discovers linears with
  `child_module.__class__.__name__ == "Linear"`. It will not find `Fp8Linear`.
- Extend the generic LoRA discovery API with an optional linear class-name allowlist, for example
  `linear_module_class_names=("Linear",)`, and have `lora_ideogram4.py` pass
  `("Linear", "Fp8Linear")`.
- `LoRAModule` can use `Fp8Linear.in_features`, `Fp8Linear.out_features`, and wrap its forward in
  the existing "base forward + LoRA delta" style. The base forward dequantizes to compute dtype,
  then the LoRA delta is added in the same activation dtype.
- Do not support "save merged model" into FP8 in v1. Merging would require dequantizing the FP8
  base, adding the LoRA delta, and either saving BF16 or re-quantizing back to official
  `weight_scale` format. Treat that as a separate feature.
- A one-step LoRA smoke test must assert that at least one `Fp8Linear` target receives a LoRA
  module and that changing the LoRA multiplier changes the transformer output.

Open point: whether LoRA should be applied to `llm_cond_proj` and `input_proj`. Start without them,
then enable only if low-rank capacity is clearly insufficient.

## Tests And Verification

Minimum tests:

- `caption_verifier` accepts a valid structured JSON caption and warns/rejects malformed ordering.
- tokenizer/build-inputs test:
  - text length under 2048,
  - image token count equals `(height / 16) * (width / 16)`,
  - indicator and segment ids isolate padding/text/image slots correctly.
- FP8 loader test with a tiny synthetic `nn.Linear` state dict:
  - keys ending in `.weight_scale` swap to `Fp8Linear`,
  - output dtype follows compute dtype,
  - no trainable parameters are created for FP8 weight buffers.
- LoRA-on-FP8 test:
  - `lora_ideogram4.py` discovers `Fp8Linear` modules,
  - the LoRA module wraps forward without dequantizing weights into parameters,
  - the output changes when the LoRA multiplier changes.
- text-cache budget test:
  - verify saved dtype matches `--text_cache_dtype`,
  - log estimated MB/image and GB/1k images from actual token counts.
- patchify/unpatchify round trip for `(B, 32, H/8, W/8)` latents.
- autoencoder encode/decode round trip with all 32 VAE channels, proving cached latents use the
  same `(patch_i, patch_j, latent_channel)` order as sampling decode.
- flow target one-step test:
  - if model predicts `latents - noise`, Euler from `x_t` moves toward latents along the
    Ideogram 4 model-time path.
- timestep sampler test:
  - sampled Ideogram 4 training timesteps match official logit-normal distribution with resolution-adjusted `mu`.

Smoke tests:

- local Comfy-Org component paths exist and safetensors metadata matches the expected component
  types,
- generate one 512x512 image with `V4_TURBO_12`,
- cache latents and text features for a two-image dataset,
- run a one-step LoRA training job with `network_module networks.lora_ideogram4`,
- sample during training without loading the text encoder twice unnecessarily.

## Risks

- License: weights and derivatives are non-commercial. Docs must say this plainly.
- HF access: docs must tell users to accept the relevant model terms before manually downloading
  the component files.
- Dependency floor: official code requests newer torch than this repo currently declares.
- Memory: Qwen3-VL-8B plus two 9.3B DiTs is heavy even with FP8 weights. Text caching must be a
  first-class workflow, not an optional optimization.
- Cache size: `53248` feature channels per token dominates disk usage. The implementation must
  display/ document concrete MB-per-image and GB-per-1k-images estimates.
- Shape drift: official FP8 uses `weight_scale`; existing Musubi FP8 helpers use `scale_weight`.
  Mixing them silently will break loading.
- LoRA drift: if `Fp8Linear` is not explicitly discoverable by the LoRA injector, training may run
  with zero attached target modules.
- Timestep drift: Ideogram 4 should not inherit the repo-wide `sigma` training default. Use a
  schedule matching official logit-normal math unless direct experiments prove another distribution.
- Diffusers ambiguity: HF card snippets may not match the official model-zoo support statement.

## Implementation Order

0. Confirm the target checkpoint tree before coding against a new Ideogram 4 release. For
   `Comfy-Org/Ideogram-4`, this has been verified: two distinct DiT LFS objects exist under
   `diffusion_models/`, plus separate text encoder and VAE files.
1. Port minimal official model, scheduler, autoencoder, latent norm, caption verifier, and FP8
   loader under `src/musubi_tuner/ideogram4/`.
2. Add standalone `ideogram4_generate_image.py` and verify image generation before training code.
3. Extend LoRA discovery to support `Fp8Linear`, then add `networks/lora_ideogram4.py`.
4. Add architecture constants and cache save helpers.
5. Add latent and text encoder cache scripts with explicit text-cache dtype and disk-budget logs.
6. Add `ideogram4_train_network.py` with conditional-transformer LoRA training only and
   Ideogram 4 logit-normal timestep sampling.
7. Add `docs/ideogram4.md` with download, cache, train, inference, and license notes.
8. Only after those pass, consider full fine-tune support, merged FP8 LoRA export, or extra LoRA
   targets.
