# Ideogram 4 FP8 Adaptation Spec

Branch: `codex/ideogram4-fp8-adaptation`

Base: `upstream/main` at `de648d36b719d21b24e424da0a2774cc9de29e84`

## Goal

Add first-class Musubi Tuner support for `ideogram-ai/ideogram-4-fp8`, using the official
`ideogram-oss/ideogram4` implementation as the behavioral reference.

The useful end state is not "the model imports". The useful end state is:

- deterministic text-to-image inference from the gated FP8 HF weights,
- latent and text-encoder-output pre-caching compatible with this repo's dataset pipeline,
- LoRA training through the existing `NetworkTrainer` flow,
- a documented CLI path that clearly states the non-commercial model-license boundary.

## Hard Facts From The Reference

Official sources:

- Model: https://huggingface.co/ideogram-ai/ideogram-4-fp8
- Code: https://github.com/ideogram-oss/ideogram4
- Architecture docs: https://github.com/ideogram-oss/ideogram4/blob/main/docs/model_architecture.md
- Pipeline docs: https://github.com/ideogram-oss/ideogram4/blob/main/docs/pipeline.md
- Prompting docs: https://github.com/ideogram-oss/ideogram4/blob/main/docs/prompting.md
- Model license: https://github.com/ideogram-oss/ideogram4/blob/main/model_licenses/LICENSE-IDEOGRAM-4-NON-COMMERCIAL

Facts to preserve:

- The FP8 model is listed as 9.3B params, FP8 quantized, non-commercial, and without Diffusers
  support in the official model zoo. The HF card also shows a generic Diffusers snippet, so do
  not treat Diffusers as the authoritative path until it is locally verified.
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
- Official FP8 is weight-only E4M3 FP8 with per-row scale buffers and BF16 activations. It is not
  the same thing as this repo's `--fp8_scaled` optimization path.
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

## Weight Loading

Support two inputs:

1. `--repo_id ideogram-ai/ideogram-4-fp8`
2. local component paths for users who already downloaded gated files

Required HF/default component layout:

- conditional DiT: `transformer/diffusion_pytorch_model.safetensors[.index.json]`
- unconditional DiT: `unconditional_transformer/diffusion_pytorch_model.safetensors[.index.json]`
- VAE: `vae/diffusion_pytorch_model.safetensors`
- text encoder: `text_encoder/`
- tokenizer: `tokenizer/`

Implementation rules:

- Port the official `Fp8Linear` path instead of routing it through existing `scale_weight` FP8
  monkey patches. Official checkpoints use `weight_scale`; current Musubi optimized FP8 uses
  `scale_weight`.
- For the text encoder, preserve the official special case where FP8 text-encoder config sets
  `ideogram_fp8_weight_only` and requires rebuilding via `AutoModel.from_config`.
- Use `huggingface_hub` downloads for repo mode and current `load_split_weights` /
  `MemoryEfficientSafeOpen` style for local mode where possible.
- Keep conditional and unconditional transformers as separate modules. Training can initially
  optimize only the conditional transformer for LoRA, but sample inference must still load/use the
  unconditional transformer for CFG.

## Latent Caching

Add `ideogram4_cache_latents.py`.

Input:

- standard image dataset items,
- RGB only,
- bucket sizes must be divisible by 16.

Processing:

1. Normalize image pixels to the VAE input convention used by the official autoencoder.
2. Encode with the VAE encoder and use deterministic mode/mean latents.
3. Apply Ideogram latent normalization for training:
   - official decode does `vae_latent = model_latent * latent_scale + latent_shift`;
   - therefore cached model latents should be `(vae_latent - latent_shift) / latent_scale`.
4. Save as `latents_1xHxW_<dtype>` under `ARCHITECTURE_IDEOGRAM4_FULL`.

The cache stores unpatchified latents shaped like other image architectures. `call_dit` will patch
to token shape `(B, grid_h * grid_w, 128)`.

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
- use a new sampler preset argument rather than overloading `--discrete_flow_shift`.

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
8. Use target `noise - latents`.

The `noise - latents` target follows the invariant for
`x_t = (1 - t) * x_0 + t * eps`: `dx_t/dt = eps - x_0`. Add a one-step unit test
before trusting training loss.

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
- patchify/unpatchify round trip for `(B, 32, H/8, W/8)` latents.
- flow target one-step test:
  - if model predicts `noise - latents`, Euler from `x_t` moves toward noise as `t` increases and toward latents as `t` decreases.

Smoke tests:

- gated HF repo can be resolved after license acceptance and token setup,
- generate one 512x512 image with `V4_TURBO_12`,
- cache latents and text features for a two-image dataset,
- run a one-step LoRA training job with `network_module networks.lora_ideogram4`,
- sample during training without loading the text encoder twice unnecessarily.

## Risks

- License: weights and derivatives are non-commercial. Docs must say this plainly.
- HF access: users must accept model terms before downloads work.
- Dependency floor: official code requests newer torch than this repo currently declares.
- Memory: Qwen3-VL-8B plus two 9.3B DiTs is heavy even with FP8 weights. Text caching must be a
  first-class workflow, not an optional optimization.
- Cache size: `53248` feature channels per token is large. Use BF16 cache by default and trim to
  actual token length.
- Shape drift: official FP8 uses `weight_scale`; existing Musubi FP8 helpers use `scale_weight`.
  Mixing them silently will break loading.
- Diffusers ambiguity: HF card snippets may not match the official model-zoo support statement.

## Implementation Order

1. Port minimal official model, scheduler, autoencoder, latent norm, caption verifier, and FP8
   loader under `src/musubi_tuner/ideogram4/`.
2. Add standalone `ideogram4_generate_image.py` and verify image generation before training code.
3. Add architecture constants and cache save helpers.
4. Add latent and text encoder cache scripts.
5. Add `ideogram4_train_network.py` with conditional-transformer LoRA training only.
6. Add `networks/lora_ideogram4.py`.
7. Add `docs/ideogram4.md` with download, cache, train, inference, and license notes.
8. Only after those pass, consider full fine-tune support or extra LoRA targets.

