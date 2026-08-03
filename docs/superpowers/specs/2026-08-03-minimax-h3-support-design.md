# MiniMax-H3 R1 BF16 Support Design

Date: 2026-08-03

Status: Revised R1 proposal for upstream confirmation

Branch: `codex/minimax-h3-support`

Base: `kohya-ss/musubi-tuner@8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`

## 1. Summary

R1 adds native BF16 MiniMax-H3 LoRA training and joint video/audio inference to Musubi Tuner for:

- `t2va`: text-to-video-with-audio.
- `fl2va`: first/last-frame-to-video-with-audio.
- `ref2va`: ordered JSONL image, video, and audio references-to-video-with-audio.

The implementation must fit Musubi's existing cache filename, tensor-key, collator, trainer, LoRA, compilation, and block-offload contracts. In particular, the target video cache must load as `batch["latents"]`, variable-length text tensors must use the `varlen_` prefix, and the H3 trainer must explicitly create target-audio noise inside its `process_batch` override.

R1 does not force dataset `batch_size` to 1, but it adds no new batching system. It does not group by an H3 layout signature, pad heterogeneous media layouts, or add a token-budget sampler. Incompatible batches fail clearly through the existing stack or H3 shape validation.

ConvRot, prequantized INT8 loading, runtime LoRA over prequantized weights, and pruned AdaLN are deferred to R2. R1 does not depend on unmerged PR 1008.

## 2. Source Anchors

Model artifacts and numerical behavior:

- <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- <https://huggingface.co/Comfy-Org/MiniMax-H3>
- <https://github.com/Comfy-Org/ComfyUI/pull/15224>
- <https://github.com/huggingface/diffusers/pull/14355>

Repository contracts that take precedence over a model-specific abstraction:

- `src/musubi_tuner/dataset/architectures.py`
- `src/musubi_tuner/dataset/cache_io.py`
- `src/musubi_tuner/dataset/bucket.py`
- `src/musubi_tuner/dataset/image_video_dataset.py`
- `src/musubi_tuner/training/trainer_base.py`
- `src/musubi_tuner/modules/custom_offloading_utils.py`

PR 1008 is an R2 dependency only:

- <https://github.com/kohya-ss/musubi-tuner/pull/1008>
- Previously inspected head: `fe4818daf4e41bc6d98959a35f55627f07f70d90`
- Previously inspected parent: `8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`

The upstream author is handling its integration. This R1 spec neither merges that commit nor defines ConvRot correctness or acceptance criteria ahead of the final upstream API.

## 3. R1 Goals

- Register MiniMax-H3 as a first-class Musubi dataset architecture.
- Cache synchronized target video and required target audio latents.
- Cache FL2VA first/last visual conditions.
- Cache Ref2VA ordered visual/audio reference latents from JSONL.
- Cache Qwen3-VL-32B layer-50 conditioning in the repository's variable-length format.
- Train BF16-base LoRA adapters with a dual-modality flow objective.
- Generate video and audio jointly and mux the result.
- Load official sharded BF16 and Comfy-Org single-file BF16 artifacts.
- Support block swap for the 50 H3 main blocks in training and inference.
- Preserve the absence of an H3-specific `batch_size == 1` check.
- Reject unsupported training knobs rather than silently applying the wrong timestep or loss convention.

## 4. R1 Non-Goals

- Full transformer fine-tuning.
- Training either VAE or Qwen3-VL.
- Dynamic ConvRot quantization.
- Prequantized INT8 ConvRot checkpoint loading.
- Pruned `adaln_t_table` checkpoints.
- Runtime floating-point LoRA branches over prequantized weights.
- NVFP4/AWQ text encoder loading.
- Classifier-free guidance.
- A numbered reference-directory parser.
- More than one reference input representation.
- Layout-signature cache buckets.
- Ragged reference-media batching.
- Text padding or attention masks added solely to enable heterogeneous training batches.
- A token-budget sampler.
- A dedicated `batch_size=1/2/3` test matrix.
- Running all three tasks at `batch_size=2` as an acceptance gate.
- Loading FL2VA and Ref2VA transformer weights in one process.
- CI with the real 33B transformer or 32B text encoder.

## 5. Released Configuration Contract

The exact source fields are in `MiniMaxAI/MiniMax-H3/transformer/config.json` and `transformer_ref/config.json`:

| Config field | Value | Meaning |
| --- | ---: | --- |
| `num_layers` | 50 | Main transformer blocks |
| `num_refiner_layers` | 2 | Refiner blocks |
| `hidden_size` | 5376 | Residual-stream width |
| `num_attention_heads` | 56 | Attention heads |
| `attention_head_dim` | 128 | Width per attention head |
| `ffn_dim` | 14336 | MLP width |
| `in_channels` | 24 | Video latent channels |
| `audio_in_channels` | 32 | Audio latent width per stereo channel |
| `patch_size` | `[1, 2, 2]` | Video latent patch |
| `text_dim` | 5120 | Qwen3-VL feature width |
| `freq_dim` | 256 | Frequency embedding width |
| `time_embed_hidden_dim` | 5376 | Time MLP hidden width |
| `time_embed_dim` | 2688 | Standard BF16 AdaLN input width |
| `rope_freq_dim` | 16 | RoPE frequency width |

`hidden_size` is not the attention projection width. The released attention projection width is:

```text
num_attention_heads * attention_head_dim = 56 * 128 = 7168
```

The native model therefore projects from a 5376-wide residual stream into a 7168-wide head space and back. No implementation may infer `hidden_size = heads * head_dim`.

Other fixed constants:

| Property | Value |
| --- | ---: |
| Video frame rate | 24 fps |
| Audio sample rate | 32000 Hz |
| Audio VAE hop | 800 samples |
| Audio latent rate | 40 Hz |
| Audio channels | 2 |
| Video modality tag | 0 |
| Text modality tag | 1 |
| Audio modality tag | 2 |

## 6. Artifact Matrix

### 6.1 Transformer

R1 accepts BF16 only:

- `MiniMaxAI/MiniMax-H3/transformer`: FL2VA weights, also used for T2VA.
- `MiniMaxAI/MiniMax-H3/transformer_ref`: Ref2VA weights.
- `MiniMaxAI/MiniMax-H3/FL2VA/transformer`.
- `MiniMaxAI/MiniMax-H3/Ref2VA/transformer`.
- `Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_fl2va_bf16.safetensors`.
- `Comfy-Org/MiniMax-H3/diffusion_models/minimax_h3_ref2va_bf16.safetensors`.

`t2va` and `fl2va` require FL2VA weights. `ref2va` requires Ref2VA weights. Because both variants share tensor shapes, `--task` is authoritative and path-derived mismatch detection is a warning, not a proof.

### 6.2 Text encoder

R1 accepts:

- An official sharded `text_encoder` directory from the root, FL2VA, or Ref2VA layout.
- `qwen3vl_32b_minimax_h3_bf16.safetensors`.

INT8 ConvRot and NVFP4/AWQ text encoders are rejected before allocation with an R2-scope message.

### 6.3 Autoencoders

R1 accepts official directories and:

- `minimax_h3_video_vae_fp16.safetensors`.
- `minimax_h3_audio_vae_fp32.safetensors`.

The two VAEs are shared by all tasks.

## 7. Code Organization

Add a native package:

```text
src/musubi_tuner/minimax_h3/
  __init__.py
  model.py
  packing.py
  video_vae.py
  audio_vae.py
  text_encoder.py
  checkpoint.py
  sampling.py
```

Responsibilities:

- `model.py`: BF16 transformer, refiner, attention, AdaLN, output heads, gradient checkpointing, and block-swap lifecycle.
- `packing.py`: modality rows, exact row counts, task layouts, position grids, timestep rows, and unpacking.
- `video_vae.py`: released video VAE modules, normalization, encode, and decode.
- `audio_vae.py`: released stereo audio VAE modules, normalization, encode, and decode.
- `text_encoder.py`: BF16 Qwen3-VL loading, presentation, multimodal preprocessing, layer-50 extraction, and token-limit validation.
- `checkpoint.py`: BF16 artifact discovery, sharded CPU streaming, state-dict normalization, and strict validation.
- `sampling.py`: paired schedules, denoising, decoding, synchronization, and mux helpers.

Add source entry points and matching root wrappers:

```text
minimax_h3_cache_latents.py
minimax_h3_cache_text_encoder_outputs.py
minimax_h3_train_network.py
minimax_h3_generate_video.py
```

## 8. Required Repository Wiring

### 8.1 Architecture registration

Add to `dataset/architectures.py`:

```python
ARCHITECTURE_MINIMAX_H3 = "mmh3"
ARCHITECTURE_MINIMAX_H3_FULL = "minimax_h3"
```

The short name has no underscore because cache filenames are parsed by underscore-separated suffixes.

Import `ARCHITECTURE_MINIMAX_H3` in `dataset/bucket.py` and add:

```python
RESOLUTION_STEPS_MINIMAX_H3 = 32
ARCHITECTURE_STEPS_MAP[ARCHITECTURE_MINIMAX_H3] = RESOLUTION_STEPS_MINIMAX_H3
```

The 32-pixel step enforces R1's target-axis divisibility before VAE encoding.

### 8.2 Cache filenames

Reuse the existing filename contract without adding tokens:

```text
{item_key}_{frame_pos}-{frame_count}_{width}x{height}_mmh3.safetensors
{item_key}_mmh3_te.safetensors
```

`VideoDataset.prepare_for_training` continues to recover `item_key`, frame range, resolution, and architecture from these names. R1 does not encode task or reference layout in the filename and does not read safetensors headers during bucket construction.

Task, VAE fingerprints, media fingerprints, temporal alignment, and ordered reference kinds remain safetensors metadata for cache-command reuse checks and diagnostics. Training compatibility is determined by the standard filename plus required tensor roles; R1 does not add header reads to bucket construction.

### 8.3 Cache I/O functions

Add architecture-specific writers to `dataset/cache_io.py`:

```python
save_latent_cache_minimax_h3(...)
save_text_encoder_output_cache_minimax_h3(...)
```

Both call the existing common writers with `ARCHITECTURE_MINIMAX_H3_FULL`. Shared cache parsing and `BucketBatchManager.__getitem__` are not changed for H3.

## 9. Dataset Contract

### 9.1 Target media

Every sample has a target video, caption, and required target audio. Resolve target audio in this order:

1. JSONL `audio_path`.
2. One exact same-stem sidecar next to the target video.
3. The target video's embedded audio stream.

Multiple matching sidecars are an error. An explicit path that fails decode does not fall back. Missing target audio is an error; R1 never substitutes silence or zero latents.

### 9.2 JSONL

Standard fields remain `video_path` and `caption`. H3 adds `audio_path` and an ordered `references` list:

```json
{
  "video_path": "targets/clip_001.mp4",
  "caption": "A concise description of the target scene and sound.",
  "audio_path": "targets/clip_001.wav",
  "references": [
    {"type": "image", "path": "refs/character.png"},
    {"type": "video", "path": "refs/action.mp4", "audio_path": "refs/action.wav"},
    {"type": "audio", "path": "refs/voice.wav"}
  ]
}
```

For a video reference, explicit `audio_path` overrides embedded audio. A video without audio remains a visual-only reference.

H3 cache commands load the JSONL records into an H3-only canonical-path map while still using the existing `VideoJsonlDatasource` for target video/caption iteration. This avoids changing the shared datasource tuple contract. `ref2va` requires `video_jsonl_file`; directory datasets are supported only for `t2va` and `fl2va`.

### 9.3 Ref2VA limits

Validate before either VAE or Qwen3-VL runs:

- At most 9 image references.
- At most 3 video references.
- At most 3 audio-bearing references.
- At most 12 reference items total.
- At least one visual reference.
- Reference videos between 2 and 15 seconds before target-duration truncation.

A video plus explicit soundtrack is one reference item and one audio-bearing video. Reference list order is semantic and drives text presentation and packed rotary time.

### 9.4 FL2VA

Training derives first and last conditions from the selected target crop. Inference requires external first and last images. R1 does not add dataset condition-path fields for FL2VA.

## 10. Geometry and Temporal Alignment

### 10.1 Video

- Normalize target video to 24 fps.
- Use the existing dataset bucket crop/resize.
- Require both pixel axes divisible by 32.
- Use the released 768-pixel short-edge canvas with a soft area cap of `768 * 1344`.
- Crop training clips downward to `F = 17 * n + 5` frames.
- Fewer than 5 usable frames is an error.
- Normal released duration is 5 through 15 seconds.
- `--allow_experimental_duration` permits out-of-range training duration while preserving structural checks and logging the deviation.

For `F = 17 * n + 5`, target video latent frames are:

```text
Fv = 5 * n + 2
```

### 10.2 Audio

Decode with PyAV and resample to stereo 32000 Hz. Align to the selected video crop at:

```text
audio_start_seconds = crop_start_frame / 24
```

Do not use floating-point `round` as a cache identity calculation. The exact audio latent count is:

```text
A = (10 * F + 3) // 6
```

This is the integer form of nearest-grid rounding for the valid H3 frame sequence. Required reference cases are:

| `F` | `A` |
| ---: | ---: |
| 5 | 8 |
| 22 | 37 |
| 39 | 65 |
| 56 | 93 |

The exact waveform window is:

```text
samples_per_channel = A * 800
```

Longer audio is truncated. Padding is allowed only for a short terminal decoder window within timestamp tolerance; a materially short or discontinuous stream is an error.

### 10.3 References

- Prepare images independently of the target canvas using the released reference transform.
- Resample reference video to 24 fps and truncate it to target duration.
- Sample reference video for Qwen3-VL at 2 fps with timestamps.
- Resample reference audio to stereo 32000 Hz and truncate to target duration.
- Keep a video reference's visual/audio streams on one decoded timeline.

## 11. Latent Cache Contract

### 11.1 Tensor keys

Cache tensors use names that `BucketBatchManager.__getitem__` already understands. After dtype and geometry stripping, training receives the keys shown in the last column.

| Meaning | Safetensors key | Loaded batch key |
| --- | --- | --- |
| Target video | `latents_{Fv}x{Hv}x{Wv}_{dtype}` | `latents` |
| Target audio | `latents_audio_2x32x{A}_{dtype}` | `latents_audio` |
| FL first condition | `latents_first_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_first` |
| FL last condition | `latents_last_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_last` |
| Ref image 000 | `latents_ref_000_image_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_ref_000_image` |
| Ref video 000 | `latents_ref_000_video_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_ref_000_video` |
| Ref audio 000 | `latents_ref_000_audio_2x32x{Ac}_{dtype}` | `latents_ref_000_audio` |

Numbered reference tensor keys follow the JSONL order. The geometry suffix after the last underscore is opaque to the collator; its purpose is to preserve the existing role-key conversion.

Target video shape is `[24, Fv, Hv, Wv]`. Target audio shape is `[2, 32, A]`. Visual condition/reference shapes are `[24, Fc, Hc, Wc]`; audio reference shapes are `[2, 32, Ac]`.

### 11.2 Posterior policy

- Target video and target audio use reproducible posterior samples derived from cache seed plus canonical item key.
- FL2VA and Ref2VA visual conditions sample with fixed seed 42.
- Visual condition samples round through FP16 before normalization to match released condition behavior.
- Reference audio uses posterior mode.

The cache metadata records the posterior policy, source fingerprints, crop timestamps, target geometry, ordered reference kinds, normalization constants, and VAE fingerprints.

### 11.3 Collation behavior

All `latents_` tensors use the existing `torch.stack` path. R1 introduces no custom H3 collator and no new bucket dimension.

Different target audio lengths cannot occur within an existing `(width, height, frame_count)` bucket because `A` is a deterministic function of target `F`. Heterogeneous reference counts or shapes are not repaired. A shape mismatch fails in `torch.stack`; missing per-sample roles are caught by H3 `process_batch` when their leading dimension differs from `batch["latents"].shape[0]`.

## 12. Text Cache Contract

MiniMax-H3 uses Qwen3-VL-32B `hidden_states[50]` without final normalization. Feature width is 5120.

Exact keys are:

```text
varlen_mmh3_hidden_states_{dtype}
varlen_mmh3_token_tags_{integer_dtype}
```

`BucketBatchManager` removes `varlen_` and the dtype suffix and returns:

```text
batch["mmh3_hidden_states"]  # list[Tensor[L, 5120]]
batch["mmh3_token_tags"]     # list[Tensor[L]]
```

Presentations are non-chat:

- `t2va`: raw caption.
- `fl2va`: released first/last `Picture` presentation plus caption.
- `ref2va`: ordered `Picture`, `Video`, and `Audio` blocks plus caption; video is sampled at 2 fps with timestamps.

Exact labels, separators, and timestamps are locked with golden fixtures from the reference behavior.

### 12.1 Size bound

R1 enforces `L <= 32768` after multimodal processor expansion and before the Qwen3-VL forward. The limit is not silently truncated. The cache command reports the sample, total tokens, counts by modality, and an estimated hidden-state payload.

For BF16 hidden states:

```text
payload_bytes = L * 5120 * 2
```

At the R1 limit this is 335,544,320 bytes, or 320 MiB, before the small token-tag tensor and safetensors header. Users must reduce reference count or duration when a sample exceeds the limit. The R1 limit is fixed; a larger operational envelope requires a separately reviewed cache/storage design.

### 12.2 Training collation

The shared collator returns variable-length tensors as lists and does not pad them. H3 `call_dit` checks that all samples in a batch have equal `L`, then stacks them. If lengths differ, it raises a clear incompatibility error recommending `batch_size=1` or homogeneous data. R1 does not add padding or an attention mask to rescue such a batch.

## 13. Packed Row Contract

Every forward is one joint self-attention sequence:

```text
t2va:   [text | target audio | target video]
fl2va:  [text | first/last conditions | target audio | target video]
ref2va: [text | ordered reference blocks | target audio | target video]
```

### 13.1 Target rows

For target video latent `[24, Fv, Hv, Wv]` and patch `[1, 2, 2]`:

```text
video_patch_width = 24 * 1 * 2 * 2 = 96
target_video_rows = Fv * (Hv // 2) * (Wv // 2)
```

For target audio latent `[2, 32, A]`:

```text
target_audio_rows = 2 * A
```

Audio is converted to rows with channel-major order equivalent to:

```python
audio_latents.permute(0, 2, 1).reshape(2 * A, 32)
```

The 32-wide rows are projected to the 5376-wide residual stream.

### 13.2 Condition rows

Each visual condition/reference contributes:

```text
condition_video_rows = Fc * (Hc // 2) * (Wc // 2)
```

Each audio reference contributes:

```text
condition_audio_rows = 2 * Ac
```

For a sample with text length `L`:

```text
packed_rows = L
            + sum(condition_video_rows)
            + sum(condition_audio_rows)
            + 2 * A
            + Fv * (Hv // 2) * (Wv // 2)
```

The cache and training logs can compute this value without model weights.

### 13.3 Tags and timesteps

- Video rows use tag `0`.
- Text rows use tag `1`.
- Audio rows use tag `2`.
- FL2VA and Ref2VA visual conditions are noised and held at model timestep `0.999`.
- Reference audio stays clean at model timestep `1.0`.
- Generated target rows receive modality-specific model timesteps.

The packer returns explicit row indices for target video/audio and never infers row roles from tensor-key sorting.

## 14. Trainer Integration and Dual-Modality Loss

### 14.1 Fixed base-loop contract

`NetworkTrainer` reads `batch["latents"]`, applies `scale_shift_latents`, and creates:

```python
noise = torch.randn_like(latents)
```

before it calls `process_batch`. H3 therefore uses `batch["latents"]` for target video and treats the incoming `noise` as video noise only.

The H3 cache stores already normalized target-video latents, so the H3 trainer's `scale_shift_latents` implementation is the identity. Target-audio latents are also cached normalized and are consumed directly inside `process_batch`.

The H3 trainer overrides both:

- `process_batch`: construct dual-modality noise/noisy inputs, pack, call the DiT, and assemble an H3 output object.
- `compute_loss`: compute unweighted video/audio mean MSE and return decomposed metrics.

It does not call the base `get_noisy_model_input_and_timesteps` or base `compute_loss`.

### 14.2 Audio noise

Inside `process_batch`:

```python
audio_latents = batch["latents_audio"]
audio_noise = torch.randn_like(audio_latents)
```

Visual/audio reference tensors are conditions, not supervised targets, and do not receive this target-noise draw.

### 14.3 Supported timestep arguments

R1 accepts only:

```text
--timestep_sampling uniform
--weighting_scheme none
--discrete_flow_shift 1.0
```

Any other value is rejected during argument validation. This prevents the base SD3 weighting or a second generic flow shift from being applied silently.

Because the shared parser defaults `--timestep_sampling` to `sigma`, `minimax_h3_train_network.py` explicitly calls `parser.set_defaults(timestep_sampling="uniform", weighting_scheme="none", discrete_flow_shift=1.0)` before parsing. A normal H3 command therefore gets the supported values without extra flags.

`--num_timestep_buckets` remains supported. `BucketBatchManager` supplies each batch's pre-generated values through `batch["timesteps"]`; H3 interprets those values as the common unshifted base `u`. Without a pool, H3 samples `u = torch.rand(B)`.

`--min_timestep` and `--max_timestep` first restrict `u` on the common `[0, 1]` base interval. H3 then applies its two fixed shifts. The two modalities never draw separate base values.

### 14.4 Noising and model time

For each sample:

```text
shift(u, s) = s * u / (1 + (s - 1) * u)
sigma_video = shift(u, 12)
sigma_audio = shift(u, 3)

model_t_video = 1 - sigma_video
model_t_audio = 1 - sigma_audio
```

Noisy inputs use sigma as the noise fraction:

```text
x_video = (1 - sigma_video) * x0_video + sigma_video * noise_video
x_audio = (1 - sigma_audio) * x0_audio + sigma_audio * noise_audio
```

Velocity targets are:

```text
v_video = x0_video - noise_video
v_audio = x0_audio - noise_audio
```

The model receives `model_t_*`, whose clean endpoint is `1.0`. The base loss weighting function is never called with this reversed convention.

### 14.5 Loss object and reduction

Define an H3-specific output structure containing:

```text
video_pred
video_target
audio_pred
audio_target
```

The overridden `compute_loss` calculates:

```text
video_loss = mean((video_pred - video_target) ** 2)
audio_loss = mean((audio_pred - audio_target) ** 2)
loss = video_loss_weight * video_loss + audio_loss_weight * audio_loss
```

Defaults are `1.0` and `1.0`. It logs `loss/video`, `loss/audio`, and total loss. Condition rows do not enter either mean.

## 15. Batch Semantics

R1 adds no validation that requires dataset `batch_size == 1`.

It also adds no special batching feature:

- Existing `(width, height, frame_count)` buckets remain unchanged.
- Latent roles use the existing stack behavior.
- Variable-length text stays a list until H3 `call_dit`.
- H3 stacks text only when lengths are already equal.
- H3 verifies every conditioning role's leading dimension equals target-video batch size.
- Incompatible shapes, missing roles, or unequal text lengths fail with the conflicting role and shapes.

This permits naturally compatible batches without promising that arbitrary Ref2VA samples can share one forward. No layout signature, media padding, text padding, attention-mask machinery, per-sample forward loop, or batch-size matrix is part of R1.

## 16. LoRA Contract

R1 trains LoRA-family network weights on a frozen BF16 transformer.

Default targets in each of the 50 main blocks are:

- `attn.qkv_proj`
- `attn.out_proj`
- `mlp.fc1`
- `mlp.fc2`

Default targeting excludes AdaLN, time conditioning, input/output projections, refiner-only differences, VAEs, and Qwen3-VL.

Saved metadata includes:

- architecture `minimax_h3`
- task
- BF16 base artifact fingerprint
- FL2VA/Ref2VA base family
- target module policy
- latent/text cache format versions

Inference uses the existing BF16 streamed/static LoRA merge path. Prequantized runtime branches are R2 scope.

## 17. BF16 Block Swap Contract

Block swap is required in R1 for LoRA training and inference.

### 17.1 Configuration

- `--blocks_to_swap 0` or omission disables swapping.
- Valid enabled values are 1 through 48 for 50 main blocks.
- Reuse `BlockSwapConfig.from_args` and `create_offloader`.
- Backward-capable training uses `ModelOffloader`.
- Frozen-base LoRA may use H2D-only mode and the shared ring-size/pinned-memory controls.
- H2D-only training requires gradient checkpointing.
- Inference uses the shared forward-only exchange mode.

### 17.2 Lifecycle

`model.py` exposes:

- `enable_block_swap(blocks_to_swap, config)`
- `move_to_device_except_swap_blocks(device)`
- `prepare_block_swap_before_forward()`
- `switch_block_swap_for_inference()`
- `switch_block_swap_for_training()`

Load the transformer on CPU when swap is enabled. Move non-block components without temporarily moving all 50 main blocks. After `accelerator.prepare` with transformer device placement disabled, call `prepare_block_swap_before_forward`.

The main block loop is:

1. `offloader.wait_for_block(index)`.
2. In debug mode, assert the block's Linear weights are on the activation device.
3. Execute directly or through non-reentrant gradient checkpointing.
4. `offloader.submit_move_blocks_forward(blocks, index)`.

R1 does not invent an H3 offloader adapter. `ModelOffloader.prepare_block_devices_before_forward` already moves the block to the accelerator, which places buffers there, and then `weighs_to_device` relocates Linear `.weight` tensors for exchange. H3 only supplies the standard model lifecycle and the post-wait device assertion.

Training-time sample generation switches to forward-only mode and prepares placement, then switches back to training mode and prepares placement again.

The compile helper receives `[transformer.blocks]` and disables Linear compilation when block swap is active, matching existing architectures.

## 18. Inference Flow

1. Validate task, BF16 artifacts, JSONL references, geometry, duration, and output path.
2. Run Qwen3-VL conditioning and release it before loading the 33B transformer unless cached features are supplied.
3. Encode first/last frames or ordered references.
4. Draw target video noise followed by target audio noise from the request generator.
5. Build the packed row layout and log its exact row count.
6. Run paired video shift-12 and audio shift-3 schedules in one transformer forward per step.
7. Unpack and advance both modalities.
8. Decode video and audio.
9. Trim to the planned common duration and mux with PyAV.

No unconditional sequence or CFG pass is created.

## 19. Errors and Diagnostics

Fail before expensive allocation where possible for:

- Unsupported R2 artifact formats.
- Missing target audio.
- Audio/video decode or timestamp failures.
- Materially short target audio.
- Fewer than 5 usable video frames.
- Invalid `17 * n + 5` geometry.
- Released-duration violations without override.
- Invalid Ref2VA count, order, or duration.
- Ref2VA without `video_jsonl_file`.
- FL2VA/Ref2VA cache used under the wrong task.
- Missing H3 tensor roles, invalid dtypes/shapes, or cache architecture/format mismatch.
- Qwen3-VL expanded length over 32768.
- Unsupported timestep sampling or loss weighting.
- Block swap outside 1 through 48.
- H2D-only training without gradient checkpointing.
- Unequal H3 text lengths in a multi-sample batch.
- Conditioning tensor batch dimension different from target video batch size.

OOM-oriented logs include target video/audio shapes, exact `A`, text length, reference counts and shapes, packed row count, dtype, and block-swap configuration. R1 does not rewrite batch size automatically.

## 20. Test Strategy

Tests use tiny synthetic model configurations unless marked manual.

### 20.1 Cache and dataset contract

- Save a synthetic H3 latent cache through `save_latent_cache_minimax_h3` and load it through `BucketBatchManager`; assert keys are exactly `latents`, `latents_audio`, and task-specific `latents_*` roles.
- Save H3 text tensors and assert the collator returns lists under `mmh3_hidden_states` and `mmh3_token_tags`.
- Assert the standard `mmh3` latent filename and `_mmh3_te.safetensors` filename round-trip through `VideoDataset.prepare_for_training` without header reads.
- Assert architecture `mmh3` selects a 32-pixel bucket step.
- Assert JSONL reference order and limits.
- Assert Ref2VA rejects directory datasets.

### 20.2 Geometry and packing

- Assert `F -> A` cases `5->8`, `22->37`, `39->65`, and `56->93` using only integer arithmetic.
- Assert waveform samples equal `A * 800`.
- Assert `F = 17n + 5` and `Fv = 5n + 2` conversions.
- Assert target audio produces `2 * A` rows in channel-major order.
- Assert target video produces `Fv * (Hv // 2) * (Wv // 2)` rows of width 96 before projection.
- Assert packed row formula, tags, row indices, and condition ordering for all three tasks.

### 20.3 Trainer hooks

- Assert `process_batch` uses incoming base-loop noise only for video and creates independent audio noise with the audio shape.
- Assert a timestep bucket value is interpreted as common `u`, then produces shift-12/shift-3 sigma and `1-sigma` model time.
- Assert unsupported `timestep_sampling`, `weighting_scheme`, and generic flow shift are rejected.
- Assert H3 `compute_loss` never calls SD3 weighting and reports separate video/audio means.
- Assert unequal text lengths and mismatched conditioning batch dimensions fail clearly.

### 20.4 Model, LoRA, and block swap

- Tiny BF16 forward for T2VA, FL2VA, and Ref2VA packed layouts.
- Default LoRA target discovery and metadata.
- Tiny H3 offloader wait/prefetch order.
- One LoRA forward/backward with gradient checkpointing and block swap.
- One forward-only multi-step inference with block swap.
- Post-wait block-weight device assertion.
- Root entrypoint existence/import tests.

No dedicated multi-batch-size matrix or per-task `batch_size=2` run is added.

### 20.5 Manual R1 acceptance

- Official FL2VA sharded BF16 load.
- Official Ref2VA sharded BF16 load.
- Comfy FL2VA BF16 load and generation.
- Comfy Ref2VA BF16 load and generation.
- One BF16 Qwen3-VL text cache near the documented token limit.
- One real 33B BF16 LoRA forward/backward with block swap.
- One real 33B BF16 forward-only generation with block swap and muxed audio/video.

Record commands, hardware, peak VRAM/RAM, cache sizes, packed rows, and output media properties.

## 21. R1 Acceptance Criteria

R1 is complete when:

- `mmh3` architecture registration and 32-pixel bucket steps work.
- Standard cache filenames are discovered without H3-specific parsing.
- Target video loads as `batch["latents"]`.
- Target audio and condition/reference roles load under `latents_*` batch keys.
- Qwen3-VL caches load as `varlen_` lists and enforce the 32768-token limit.
- Target audio is required, aligned, independently noised, and supervised.
- The exact integer `F -> A` formula is shared by cache and inference.
- Packed audio/video row counts match the documented formulas.
- T2VA, FL2VA, and JSONL-only Ref2VA execute through native BF16 packing.
- H3 rejects base loss weighting and unsupported timestep sampling rather than silently reversing curves.
- BF16 LoRA training and inference work.
- BF16 block swap works in LoRA training and inference.
- No H3-specific `batch_size == 1` assertion exists.
- Incompatible multi-sample layouts fail clearly rather than being padded or silently misbatched.
- Automated tests pass and real-model R1 evidence is recorded.
- User documentation states BF16-only R1 scope, JSONL Ref2VA, cache limits, batching limitations, and block-swap commands.

## 22. Deferred R2

R2 begins only after PR 1008 is merged upstream and its final public interfaces are available. It requires a separate design review for:

- Dynamic ConvRot policy and hard assertions against silently skipped layers.
- Prequantized Comfy ConvRot loading.
- Cross-implementation rotation basis and dequantization correctness.
- Runtime floating-point LoRA over a prequantized base.
- Normal versus pruned AdaLN time conditioning.
- INT8 block-swap weight/scale placement and device assertions.
- R2 artifact-level numerical and quality acceptance, not merely "loads and executes."

No R2 behavior is an R1 dependency or acceptance criterion.
