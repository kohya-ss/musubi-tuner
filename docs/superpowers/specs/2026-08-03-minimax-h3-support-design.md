# MiniMax-H3 R1 BF16 Support Design

Date: 2026-08-03

Status: Revised R1 proposal after merged ComfyUI and end-to-end dataset audits

Branch: `codex/minimax-h3-support`

Base: `kohya-ss/musubi-tuner@8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`

## 1. Summary

R1 adds native BF16 MiniMax-H3 LoRA training and joint video/audio inference to Musubi Tuner for:

- `t2va`: text-to-video-with-audio.
- `fl2va`: first/last-frame-to-video-with-audio.
- `ref2va`: ordered JSONL image, video, and audio references-to-video-with-audio.

The implementation must fit Musubi's existing cache filename, tensor-key, collator, trainer, LoRA, compilation, and block-offload contracts. In particular, the target video cache must load as `batch["latents"]`, variable-length text tensors must use the `varlen_` prefix, and the H3 trainer must explicitly create target-audio noise inside its `process_batch` override.

R1 does not force dataset `batch_size` to 1, but it adds no new batching system. A multi-sample forward is a pure replication axis: every sample shares one packed layout and one base timestep while carrying different latent/text values. R1 does not group by an H3 layout signature, pad heterogeneous media layouts, or add a token-budget sampler. An H3-only post-build preflight rejects incompatible buckets before accelerator or model allocation.

ConvRot, prequantized INT8 loading, runtime LoRA over prequantized weights, and pruned AdaLN are deferred to R2. R1 does not depend on unmerged PR 1008.

## 2. Source Anchors

Model artifacts and released configuration:

- <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- <https://huggingface.co/Comfy-Org/MiniMax-H3>

Merged ComfyUI integration audited for packing and numerical conventions:

- PR: <https://github.com/Comfy-Org/ComfyUI/pull/15224>
- Merge commit: `57500fc5bc92566a63f2046824f522cd55c335ca`
- Transformer/packing: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy/ldm/minimax/model.py>
- Condition payload: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy/model_base.py>
- Text presentation/tags: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy/text_encoders/minimax.py>
- Public task/shift nodes: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy_extras/nodes_minimax_h3.py>

The open Diffusers implementation is a secondary cross-check, pinned to inspected head `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`:

- <https://github.com/huggingface/diffusers/pull/14355>

Released config files and tensors are authoritative for architecture and state-dict shape. The pinned merged ComfyUI implementation is authoritative for its own packed-row, token-tag, and adapter behavior. This spec explicitly defines Musubi's native training and dual-scheduler inference semantics; ComfyUI-specific sign/slope adapters are not model semantics and must not be copied. The open Diffusers PR is not normative when it differs.

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
- Per-sample timesteps inside one replicated packed layout.
- Loading FL2VA and Ref2VA transformer weights in one process.
- Training-time sample generation through the shared `--sample_prompts` hook; R1 uses the standalone joint-AV generator.
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

Other architecture constants and released defaults:

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
| Video temporal span cycle | `(5/3) * (1, 4, 4, 4, 4)` |
| Default video flow shift | 12.0 |
| Default audio flow shift | 3.0 |
| Default visual condition clean coefficient | 0.999 |
| Default audio condition clean coefficient | 1.0 |

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

### 8.2 VideoDataset construction and frame-grid normalization

Architecture registration alone is insufficient because `VideoDataset.__init__` has an architecture whitelist for target FPS. Import `ARCHITECTURE_MINIMAX_H3` in `dataset/image_video_dataset.py`, add:

```python
TARGET_FPS_MINIMAX_H3 = 24.0
```

and route `ARCHITECTURE_MINIMAX_H3` to that value before the unsupported-architecture branch.

H3 frame counts cannot use the repository's legacy `4 * n + 1` expression. Add one shared architecture-aware helper in `dataset/architectures.py` and call it from both dataset and trainer code:

```python
def round_down_frame_count(frame_count, architecture, vae_frame_stride):
    if architecture == ARCHITECTURE_MINIMAX_H3:
        if frame_count < 5:
            raise ValueError("MiniMax-H3 requires at least 5 frames")
        return 5 + ((frame_count - 5) // 17) * 17
    return 1 + ((frame_count - 1) // vae_frame_stride) * vae_frame_stride
```

Replace all three direct rounding sites:

1. `VideoDataset.__init__`, where configured `target_frames` are normalized.
2. `VideoDataset.retrieve_latent_cache_batches`, where `frame_extraction="full"` chooses the cropped length.
3. `NetworkTrainer.sample_image_inference`, where training-time sample generation normalizes `frame_count`.

For H3, `5`, `22`, `39`, and `56` must remain unchanged. Setting `vae_frame_stride = 17` is explicitly incorrect because the legacy `1 + n * stride` expression would produce `18`, not `22`. The stride argument has no default, so every call site must supply its architecture's value; this preserves stride-1 behavior for Krea2 and Qwen Image instead of silently falling back to 4.

### 8.3 Post-build H3 batch preflight

`minimax_h3_train_network.py` overrides `_build_dataset`, calls the base implementation, and validates the returned dataset group before accelerator creation or model loading. The preflight inspects each H3 `BucketBatchManager` and, for every bucket capable of producing an effective batch larger than one, verifies that all items share:

- text length and the exact token-tag vector;
- task and ordered condition-role keys;
- every target/condition tensor shape;
- packed row count and rotary-layout inputs.

The validator reads safetensors headers/slices plus the small token-tag tensor; it does not load latent payloads or `[L, 5120]` hidden states. Because buckets are reshuffled every epoch, validating only the current batch partition is insufficient: every item in a multi-item bucket must have the same structural fingerprint. This fingerprint is preflight-only and is not added to the cache filename or bucket key.

The same preflight rejects a per-sample timestep pool when any effective H3 batch can exceed one. Runtime `call_dit` checks remain defense in depth, but structural incompatibility must be reported at step zero with dataset index, bucket, cache paths, and the conflicting fields.

### 8.4 Cache filenames

Reuse the existing filename contract without adding tokens:

```text
{item_key}_{frame_pos}-{frame_count}_{width}x{height}_mmh3.safetensors
{item_key}_{frame_pos}-{frame_count}_mmh3_te.safetensors
```

`VideoDataset.prepare_for_training` continues to recover `item_key`, frame range, resolution, and architecture from these names. Its H3 branch pairs each latent cache with the text cache carrying the same frame-range token. This is required for FL2VA because the selected crop's first and last frames are part of the Qwen presentation; a source-level text cache would silently alias different `chunk` or `slide` crops. T2VA and Ref2VA use the same crop-specific naming contract for one unambiguous lookup rule. R1 does not encode task or reference layout in the filename, and the shared bucket-construction path does not read safetensors headers. The H3-only post-build preflight in Section 8.3 runs after those buckets exist.

Task, VAE fingerprints, media fingerprints, temporal alignment, and ordered reference kinds remain safetensors metadata for cache-command reuse checks and diagnostics. Training compatibility is determined by the standard filename plus required tensor roles; R1 does not add header reads to bucket construction.

### 8.5 Cache I/O functions

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
| Target audio | `latents_audio_32x2x{A}_{dtype}` | `latents_audio` |
| FL first condition | `latents_first_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_first` |
| FL last condition | `latents_last_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_last` |
| Ref image 000 | `latents_ref_000_image_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_ref_000_image` |
| Ref video 000 | `latents_ref_000_video_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_ref_000_video` |
| Ref audio 000 | `latents_ref_000_audio_32x2x{Ac}_{dtype}` | `latents_ref_000_audio` |

Numbered reference tensor keys follow the JSONL order. The geometry suffix after the last underscore is opaque to the collator; its purpose is to preserve the existing role-key conversion.

Target video shape is `[24, Fv, Hv, Wv]`. Target audio shape is `[32, 2, A]`. Visual condition/reference shapes are `[24, Fc, Hc, Wc]`; audio reference shapes are `[32, 2, Ac]`.

The audio cache deliberately preserves the released audio VAE layout: feature width, stereo channel, then time. The encoder boundary stores `[32, 2, A]` directly from released `[B, 32, 2, A]` output; it must not transpose to `[2, 32, A]` while retaining a misleading geometry key.

### 11.2 Posterior policy

- Target video uses a reproducible posterior sample derived from cache seed plus canonical item key.
- Target audio and all reference audio use the audio posterior mean/mode; the released H3 audio path does not sample `logs_proj`.
- FL2VA and Ref2VA visual conditions sample with fixed seed 42.
- Visual condition samples round through FP16 before normalization to match released condition behavior.

The cache metadata records the posterior policy, source fingerprints, crop timestamps, target geometry, ordered reference kinds, normalization constants, and VAE fingerprints.

### 11.3 Collation behavior

All `latents_` tensors use the existing `torch.stack` path. R1 introduces no custom H3 collator and no new bucket dimension.

Different target audio lengths cannot occur within an existing `(width, height, frame_count)` bucket because `A` is a deterministic function of target `F`. Heterogeneous reference counts or shapes are not repaired. Section 8.3 inspects cache keys/shapes and rejects such a multi-item bucket before training; `torch.stack` and H3 `process_batch` retain runtime assertions only as defense in depth.

## 12. Text Cache Contract

MiniMax-H3 uses Qwen3-VL-32B `hidden_states[50]` without final normalization. Hugging Face indexes `hidden_states[0]` as the embedding output, so `hidden_states[50]` means the state after exactly 50 decoder layers, not after layer index 50 in zero-based module numbering. Feature width is 5120.

For a full 64-layer Qwen3-VL checkpoint, request hidden states and select index 50. For a released/converted stack truncated to exactly 50 decoder layers, take its last decoder state before the final norm; do not use a top-level `last_hidden_state` path that applies the final normalization. Both artifact paths must produce the same pre-norm layer-50 convention, and the cache metadata records it explicitly.

Exact keys are:

```text
varlen_mmh3_hidden_states_{dtype}
varlen_mmh3_token_tags_int64
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

The cached token tags are not one constant tag for the whole Qwen output. Build them from the expanded multimodal presentation exactly as follows:

- Initialize all `L` positions to text tag `1`.
- For every expanded vision embedding span, set the vision rows and both flanking vision-start/vision-end token rows to video tag `0`.
- Keep prompt text, `Picture`/`Video`/timestamp labels, and `Audio` labels at text tag `1`.
- Qwen does not receive reference-audio latents, so tag `2` never appears in the text cache. The packer assigns tag `2` only to packed audio-latent rows.

The cache writer validates `token_tags.shape == [L]`, dtype `int64`, and values in `{0, 1}`. Text-cache metadata fingerprints the tokenizer, multimodal processor, layer index, presentation format, and token-tag algorithm. A fingerprint mismatch invalidates reuse instead of silently retaining stale tags.

### 12.1 Size bound

R1 enforces `L <= 32768` after multimodal processor expansion and before the Qwen3-VL forward. The limit is not silently truncated. The cache command reports the sample, total tokens, counts by modality, and an estimated hidden-state payload.

For BF16 hidden states:

```text
payload_bytes = L * 5120 * 2
```

At the R1 limit this is 335,544,320 bytes, or 320 MiB, before the small token-tag tensor and safetensors header. Users must reduce reference count or duration when a sample exceeds the limit. The R1 limit is fixed; a larger operational envelope requires a separately reviewed cache/storage design.

### 12.2 Training collation

The shared collator returns variable-length tensors as lists and does not pad them. The post-build preflight in Section 8.3 verifies equal `L` and identical token-tag vectors across every multi-item bucket before training starts; equal length alone is insufficient because the one-dimensional AdaLN tag/index plan is shared across the replicated batch axis. H3 `call_dit` repeats the checks as defense in depth, then stacks the hidden states. R1 does not add padding or an attention mask to rescue an incompatible batch.

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

For target audio latent `[32, 2, A]`:

```text
target_audio_rows = 2 * A
```

Audio is converted to rows with channel-major order equivalent to:

```python
audio_latents.permute(1, 2, 0).reshape(2 * A, 32)
```

For batched cache input `[B, 32, 2, A]`, the equivalent operation is `permute(0, 2, 3, 1).reshape(B, 2 * A, 32)`. The result remains channel-major, and the 32-wide rows are projected to the 5376-wide residual stream.

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

- Packed video rows use tag `0`; packed audio rows use tag `2`.
- The text span preserves the cached per-token tags. Ordinary text and labels use `1`, while expanded vision blocks and their flanking vision tokens use `0`.
- Text rows and generated target-video rows use `model_t_video`.
- Generated target-audio rows use `model_t_audio`.
- With visual clean coefficient `a_v`, FL2VA/Ref2VA visual condition rows use `max(model_t_video, a_v)`. The default is `a_v = 0.999`; this is not a constant row timestep when `model_t_video > 0.999`.
- With audio clean coefficient `a_a`, reference-audio rows use `max(model_t_audio, a_a)`. The default `a_a = 1.0` keeps the default row timestep at `1.0`.

For each packed sequence, sort the distinct model-time values and build one `row_timestep_indices[S]` vector. Main transformer blocks have three modality slots per distinct time, so each row selects block AdaLN modulation with:

```text
block_adaln_index[row] = 3 * row_timestep_indices[row] + token_tag[row]
```

The text span must therefore be split at token-tag runs or indexed row-by-row; treating it as a uniform tag-1 segment is incorrect. `token_tags[S]`, `row_timestep_indices[S]`, and the position grid are structural one-dimensional tensors shared by every item on the replicated batch axis.

The FinalLayer is different: its AdaLN projection has one slot per distinct time, not three modality slots. Target video selects `video_timestep_index` directly and target audio selects `audio_timestep_index` directly. FinalLayer must never receive `3 * index + tag`; modality separation there comes from the two output heads, not a tagged AdaLN table. Text and condition rows do not enter either final output head.

The packer returns explicit row indices for target video/audio and never infers row roles from tensor-key sorting.

### 13.4 Exact FP64 rotary clock

The rotary grid is a checkpoint contract, not an arbitrary monotonic position. Construct `position_ids[S, 3]` in FP64 and preserve this exact clock before the model converts it for frequency multiplication.

For latent video frame `k`:

```text
frame_span(k) = (5 / 3) * (1, 4, 4, 4, 4)[k mod 5]
video_time(k, origin) = origin + sum(frame_span(j), j=0..k-1)
```

For a latent frame of height `H`, width `W`, and spatial patch `2x2`, let `q = sqrt(H * W)`. For axis dimension `d` and index `i = 0..d/2-1`:

```text
axis(d, i) = 32 * ((1 - d / q) / 2 + i * (d / q) / (d / 2))
```

The frame grid is the row-major meshgrid of `axis(H, i)` and `axis(W, i)`. Row placement is:

- Text row `i`: `(i, 0, 0)` for `i = 0..L-1`.
- Target video: `video_time(k, cursor)` plus the target frame grid.
- Target audio: channel-major stereo rows; both channels use `cursor + a` at audio latent index `a`, `h = 0`, and `w` fixed to the first/last target-width grid coordinate for channels 0/1.
- FL2VA first condition: time `L` on the target frame grid.
- FL2VA last condition: time `L + sum(frame_span(k), k=0..Fv-1) - 5/3` on the target frame grid. FL conditions do not advance the target cursor; target audio/video still start at `L`.

Ref2VA starts `cursor = L` and advances references in semantic order:

- Image reference: place its frame grid at `cursor`, then add `1`.
- Standalone audio reference of length `Ac`: use its `Ac` audio times and the target-width endpoints, then add `Ac`.
- Video reference: place video at the current cursor. If it has soundtrack, place channel-major audio at the same cursor using that reference video's width endpoints. Then add `max(Ac, sum(frame_span(k), k=0..Fc-1))`.

After all references, target audio and target video share the final cursor. Golden tests compare the full FP64 grid, not only shape or monotonicity, including first/last FL anchors and mixed image/video/audio reference cursor advances.

## 14. Trainer Integration and Dual-Modality Loss

### 14.1 Fixed base-loop contract

`NetworkTrainer` reads `batch["latents"]`, applies `scale_shift_latents`, and creates:

```python
noise = torch.randn_like(latents)
```

before it calls `process_batch`. H3 therefore uses `batch["latents"]` for target video and treats the incoming `noise` as video noise only.

The H3 cache stores already normalized target-video latents, so the H3 trainer's `scale_shift_latents` implementation is the identity. Target-audio latents are also cached normalized and are consumed directly inside `process_batch`.

The H3 trainer overrides both:

- `process_batch`: construct dual-modality noise/noisy inputs, pack, call the DiT, and return the standard `DiTOutput` with audio tensors in `extra`.
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

R1 accepts only the generic training convention below and adds four H3-specific values:

```text
--timestep_sampling uniform
--weighting_scheme none
--discrete_flow_shift 1.0
--h3_shift_video 12.0
--h3_shift_audio 3.0
--h3_visual_cond_clean 0.999
--h3_audio_cond_clean 1.0
```

Any other generic sampling/weighting value is rejected during argument validation. This prevents the base SD3 weighting or a second generic flow shift from being applied silently. H3 shift values must be in `[0.01, 100.0]`; condition clean coefficients must be in `[0.0, 1.0]`. Training metadata and sample logs record all four H3 values.

Because the shared parser defaults `--timestep_sampling` to `sigma`, `minimax_h3_train_network.py` explicitly sets all supported defaults before parsing. A normal H3 command therefore gets the released convention without extra flags.

One packed forward has one scalar unshifted base value `u`, shared by every replicated batch item. Without a timestep pool, H3 samples one scalar, not `B` independent values. `--num_timestep_buckets > 1` remains usable when the effective batch contains one sample. The existing bucket manager emits one unrelated value per sample, so the post-build preflight rejects the option when any H3 bucket can produce an effective batch larger than one; values must never be silently discarded or averaged.

`--min_timestep` and `--max_timestep` first restrict `u` on the common `[0, 1]` base interval after the existing `/1000` conversion. H3 then applies its two configurable shifts. The two modalities never draw separate base values.

### 14.4 Coordinate conversion and noising

There are two opposite coordinates. Keep the conversion at the H3 `process_batch` boundary:

```text
Musubi base domain:
  u == t_m in [0, 1] is noise amount
  unshifted_x = (1 - u) * x0 + u * noise

H3 model domain:
  model_t in [0, 1] is cleanliness
  model_t = 1 - sigma

shift(u, s) = s * u / (1 + (s - 1) * u)
sigma_video = shift(u, h3_shift_video)
sigma_audio = shift(u, h3_shift_audio)

model_t_video = 1 - sigma_video
model_t_audio = 1 - sigma_audio
```

Broadcast the two scalar sigmas over the batch. Noisy inputs use their modality sigma as the noise fraction:

```text
x_video = (1 - sigma_video) * x0_video + sigma_video * noise_video
x_audio = (1 - sigma_audio) * x0_audio + sigma_audio * noise_audio
```

The model receives `model_t_*`, whose clean endpoint is `1.0`. `batch["timesteps"]` and generic Musubi loss weighting remain in the noise-amount coordinate and are never passed directly into H3 AdaLN.

### 14.5 Native output and target sign

The released H3 output heads predict data-ward velocity:

```text
target_video = x0_video - noise_video
target_audio = x0_audio - noise_audio
```

This is the opposite of the `noise - latents` target used by most Musubi architectures and matches the exceptional sign used by Ideogram4. H3 must not reuse a Wan/Hunyuan target template.

The native `minimax_h3/model.py` forward returns both raw head predictions unchanged. It must not copy either adapter from ComfyUI's return statement:

- ComfyUI negates both outputs to convert `x0 - noise` into its stock sampler's `noise - x0` convention.
- ComfyUI additionally multiplies audio by `d(sigma_audio) / d(sigma_video)` so one sampler on the video-sigma grid can integrate both streams.

For default shifts `12 -> 3`, that slope ranges from `0.25` near sigma zero to `4.0` near sigma one. It is a ComfyUI single-sampler chain-rule adapter, not a training target or model property. Neither the negative sign nor the audio slope is allowed in native training or Musubi's dual-scheduler inference.

### 14.6 Condition augmentation and RNG

Condition augmentation uses the configured clean coefficients themselves, while AdaLN uses the `max` row timesteps from Section 13.3. These are deliberately distinct when the current target is cleaner than its condition augmentation:

```text
a_v = h3_visual_cond_clean
visual_condition_input = a_v * visual_condition + (1 - a_v) * condition_noise_video
visual_condition_model_t = max(model_t_video, a_v)

a_a = h3_audio_cond_clean
audio_condition_input = a_a * audio_condition + (1 - a_a) * condition_noise_audio
audio_condition_model_t = max(model_t_audio, a_a)
```

At the defaults, visual conditions stay 99.9% clean but their model time follows `model_t_video` above `0.999`; reference audio is fully clean at model time `1.0`.

Condition noise is not VAE posterior sampling and does not reuse the cache's fixed seed 42. The policy is:

- Training draws a fresh condition seed per sample on every `process_batch` call.
- Within one sample, every visual condition restarts a CPU generator at that same seed, matching ComfyUI's intentional shared noise stream. Equal shapes receive identical noise; unequal shapes share the same prefix.
- Audio conditions use a separate stream at `condition_seed + 1` and likewise restart it for each audio condition.
- Inference uses the request seed for visual conditions and request seed plus one for audio conditions. These dedicated generators do not advance the target video/audio noise generator.
- When a clean coefficient is `1.0`, do not draw unused condition noise.

The training seed is re-sampled per step rather than frozen so LoRA training does not overfit one condition-noise realization. Checkpointed training RNG state must reproduce the sequence after resume.

### 14.7 Loss object and reduction

Reuse the repository's existing `training.trainer_base.DiTOutput` extension seam rather than defining a parallel result type:

```python
DiTOutput(
    pred=video_pred,
    target=video_target,
    extra={"audio_pred": audio_pred, "audio_target": audio_target},
)
```

The overridden `compute_loss` calculates:

```text
video_loss = mean((output.pred - output.target) ** 2)
audio_loss = mean((output.extra["audio_pred"] - output.extra["audio_target"]) ** 2)
loss = video_loss_weight * video_loss + audio_loss_weight * audio_loss
```

Defaults are `1.0` and `1.0`. It logs `loss/video`, `loss/audio`, and total loss. Condition rows do not enter either mean. The overridden path never calls `compute_loss_weighting_for_sd3`.

## 15. Batch Semantics

R1 adds no validation that requires dataset `batch_size == 1`.

The supported multi-sample case is a pure replication axis. The transformer receives batched values `[B, S, D]`, but one shared structural plan: `position_ids[S, 3]`, `token_tags[S]`, `row_timestep_indices[S]`, and explicit modality row indices. One scalar `u` and therefore one set of distinct AdaLN times applies to the whole forward.

R1 adds no special batching feature:

- Existing `(width, height, frame_count)` buckets remain unchanged.
- Latent roles use the existing stack behavior.
- Variable-length text stays a list until H3 `call_dit`.
- H3 stacks text only when lengths and token-tag vectors are already identical.
- Every sample must have the same task, ordered condition kinds, condition shapes, target shape, and packed row count.
- H3 verifies every conditioning role's leading dimension equals target-video batch size.
- H3 uses one base timestep for the whole batch; it does not flatten a per-sample timestep vector and take element zero.
- Section 8.3 validates the entire multi-item bucket before model allocation because epoch shuffling can pair any two items; runtime checks repeat the same invariants only as defense in depth.

This permits naturally compatible batches without promising that arbitrary Ref2VA samples can share one forward. Incompatible shapes, missing roles, tag plans, or text lengths fail at step zero with the conflicting cache paths and fields. The existing per-sample timestep-bucket pool is rejected when an effective batch can exceed one because it violates the shared-time contract. No layout-signature bucket, media padding, text padding, attention-mask machinery, per-sample forward loop, or batch-size matrix is part of R1.

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

The standalone generator uses forward-only block swap. R1 rejects the shared training-time `--sample_prompts` hook before model allocation because that hook owns only one VAE and one video output, while H3 requires two independently placed VAEs and a muxed AV result.

The compile helper receives `[transformer.blocks]` and disables Linear compilation when block swap is active, matching existing architectures.

## 18. Inference Flow

`minimax_h3_generate_video.py` exposes `--h3_shift_video` and `--h3_shift_audio` with defaults `12.0` and `3.0`, plus `--h3_visual_cond_clean` and `--h3_audio_cond_clean`. Generic `--flow_shift` is not silently reused for one modality.

1. Validate task, BF16 artifacts, JSONL references, geometry, duration, H3 shifts/condition augmentation, and output path.
2. Run Qwen3-VL conditioning and release it before loading the 33B transformer unless cached features are supplied.
3. Encode first/last frames or ordered references and apply dedicated request-seeded condition augmentation.
4. Draw target video noise followed by target audio noise from the request generator.
5. Build the packed row layout and log its exact row count.
6. Build a common descending base grid `u_i`, then derive `sigma_video_i = shift(u_i, h3_shift_video)` and `sigma_audio_i = shift(u_i, h3_shift_audio)`.
7. Run one transformer forward per common base interval with raw data-ward predictions and `model_t_* = 1 - sigma_*`.
8. Advance each modality on its own finite sigma interval:

```text
x_video_next = x_video + (sigma_video_i - sigma_video_next) * pred_video
x_audio_next = x_audio + (sigma_audio_i - sigma_audio_next) * pred_audio
```

9. Decode video and audio.
10. Trim to the planned common duration and mux with PyAV.

The native dual-scheduler path does not negate the predictions and does not apply `d(sigma_audio) / d(sigma_video)`. ComfyUI instead uses one video-sigma sampler plus a pointwise audio slope. Those updates agree only to first order; at finite step size their trajectories are not bit-exact. R1 acceptance compares packing, timesteps, raw-head parity, scheduler invariants, and output quality, not final ComfyUI tensors or media hashes.

No unconditional sequence or CFG pass is created.

## 19. Errors and Diagnostics

Fail before expensive allocation where possible for:

- Unsupported R2 artifact formats.
- Missing target audio.
- Audio/video decode or timestamp failures.
- Materially short target audio.
- Fewer than 5 frames or invalid `17 * n + 5` geometry after architecture-aware normalization.
- Released-duration violations without override.
- Invalid Ref2VA count, order, or duration.
- Ref2VA without `video_jsonl_file`.
- FL2VA/Ref2VA cache used under the wrong task.
- Missing H3 tensor roles, invalid dtypes/shapes, or cache architecture/format mismatch.
- Invalid/stale text token tags or presentation fingerprints.
- Qwen3-VL expanded length over 32768.
- Unsupported timestep sampling, loss weighting, H3 shift, or condition-clean value.
- Per-sample timestep bucketing when an effective H3 batch can exceed one.
- Block swap outside 1 through 48.
- H2D-only training without gradient checkpointing.
- Unequal H3 text lengths, token-tag plans, or packed layouts found by the post-build bucket preflight.
- Conditioning tensor batch dimension different from target video batch size.

OOM-oriented logs include target video/audio shapes, exact `A`, text length, reference counts and shapes, packed row count, dtype, and block-swap configuration. R1 does not rewrite batch size automatically.

## 20. Test Strategy

Tests use tiny synthetic model configurations unless marked manual.

### 20.1 Cache and dataset contract

- Save a synthetic H3 latent cache through `save_latent_cache_minimax_h3` and load it through `BucketBatchManager`; assert keys are exactly `latents`, `latents_audio`, and task-specific `latents_*` roles.
- Construct `VideoDataset(architecture="mmh3")` and assert it selects 24 fps instead of reaching the unsupported-architecture branch.
- Save H3 text tensors and assert the collator returns lists under `mmh3_hidden_states` and `mmh3_token_tags`.
- Assert full-Qwen `hidden_states[50]` and truncated-50 pre-norm last-state paths use the same after-layer-50/no-final-norm convention.
- Golden-test mixed text tags: ordinary/label tokens are `1`, each expanded vision span plus both flanking tokens is `0`, and no text-cache row is `2`.
- Reject text-cache reuse when the presentation/tag fingerprint changes.
- Assert the standard `mmh3` latent filename and `_mmh3_te.safetensors` filename round-trip through `VideoDataset.prepare_for_training` without header reads.
- Assert architecture `mmh3` selects a 32-pixel bucket step.
- Assert `architectures.py` exports both `mmh3` and `minimax_h3` constants and all dataset/bucket imports resolve.
- Assert JSONL reference order and limits.
- Assert Ref2VA rejects directory datasets.

### 20.2 Geometry and packing

- Assert `F -> A` cases `5->8`, `22->37`, `39->65`, and `56->93` using only integer arithmetic.
- Assert waveform samples equal `A * 800`.
- Assert `F = 17n + 5` and `Fv = 5n + 2` conversions.
- Assert configured target frames, `frame_extraction="full"`, and training sample generation all preserve `5`, `22`, `39`, and `56`; values below 5 fail and other values round down with `5 + 17 * floor((F-5)/17)`. Assert the helper requires an explicit stride and preserves the existing stride-1 behavior used by Krea2 and Qwen Image.
- Assert the audio VAE posterior mode `[B, 32, 2, A]` is cached directly as `[32, 2, A]` under a `32x2xA` key, round-trips through the collator, and produces `2 * A` channel-major rows without evaluating/sampling `logs_proj`.
- Assert target video produces `Fv * (Hv // 2) * (Wv // 2)` rows of width 96 before projection.
- Assert packed row formula, mixed tags, row indices, and condition ordering for all three tasks.
- Golden-test the full FP64 rotary grid: `(5/3) * (1,4,4,4,4)` video spans, normalized spatial axes, FL first/last anchors, stereo audio endpoints, and Ref2VA cursor advances.
- Assert visual condition row time is `max(model_t_video, a_v)`, not constant `a_v`, and text row time follows video.
- Assert main blocks use `3 * timestep_index + tag`, while FinalLayer selects video/audio timestep indices directly with no tag offset.

### 20.3 Trainer hooks

- Assert `process_batch` uses incoming base-loop noise only for video and creates independent audio noise with the audio shape.
- Assert one scalar base `u` is shared across a replicated batch, then produces configurable video/audio sigmas and `1-sigma` model times.
- Assert post-build preflight accepts one compatible replicated bucket and rejects a heterogeneous bucket before accelerator/model creation, including mismatched equal-length tag vectors.
- Assert the existing per-sample timestep pool is rejected when an effective H3 batch exceeds one instead of taking its first value.
- Assert unsupported `timestep_sampling`, `weighting_scheme`, generic flow shift, H3 shifts, and condition coefficients are rejected.
- Assert condition inputs use `a*x0 + (1-a)*noise`; training seeds change per step, visual conditions restart one shared stream, and audio uses the `seed+1` stream.
- Mock raw output heads and assert `process_batch` targets are `latents - noise` with no prediction negation or audio slope scaling.
- Assert H3 returns the standard `DiTOutput`, stores audio tensors in `extra`, and `compute_loss` never calls SD3 weighting while reporting separate video/audio means.
- Assert unequal text lengths, unequal token-tag plans, and mismatched conditioning batch dimensions fail clearly.

### 20.4 Model, LoRA, and block swap

- Tiny BF16 forward for T2VA, FL2VA, and Ref2VA packed layouts. The T2VA fixture uses one structurally identical `B=2` replicated forward to catch accidental `[0]` slicing; no batch-size matrix or backward matrix is added.
- Default LoRA target discovery and metadata.
- Tiny H3 offloader wait/prefetch order.
- One LoRA forward/backward with gradient checkpointing and block swap.
- One forward-only dual-scheduler multi-step inference with block swap, configurable shifts, native velocity sign, and no audio slope.
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

Record commands, hardware, peak VRAM/RAM, cache sizes, packed rows, shifts, condition augmentation, and output media properties. Comfy generation is a qualitative/artifact compatibility check; final tensors and media hashes are not required to match its finite-step single-scheduler trajectory.

## 21. R1 Acceptance Criteria

R1 is complete when:

- `architectures.py` registers `mmh3`/`minimax_h3`; `VideoDataset` constructs at 24 fps; and 32-pixel bucket steps work.
- All three legacy `4 * n + 1` call sites use architecture-aware frame normalization, preserving valid H3 counts such as `22`, `39`, and `56`.
- Standard cache filenames are discovered without H3-specific parsing.
- Target video loads as `batch["latents"]`.
- Target audio and condition/reference roles load under `latents_*` batch keys, with released `[32, 2, A]` audio axis order preserved.
- Qwen3-VL caches load as `varlen_` lists, use the exact after-layer-50 pre-norm state, preserve mixed text/vision token tags, and enforce the 32768-token limit.
- Target audio is required, aligned, independently noised, and supervised.
- The exact integer `F -> A` formula is shared by cache and inference.
- Packed audio/video row counts match the documented formulas.
- FP64 rotary clocks, FL anchors, and reference cursor advances match the pinned merged ComfyUI implementation.
- Main-block and FinalLayer AdaLN indices follow their distinct three-slot and one-slot rules.
- Condition augmentation, row timesteps, and per-step/shared-stream RNG follow the documented separate contracts.
- T2VA, FL2VA, and JSONL-only Ref2VA execute through native BF16 packing.
- H3 rejects base loss weighting and unsupported timestep sampling rather than silently reversing curves.
- Training uses raw `latents - noise` targets; native model/inference outputs contain neither ComfyUI's negative sign nor its audio slope adapter.
- Video/audio shifts are configurable and native inference advances two finite sigma schedules from one common base grid.
- BF16 LoRA training and inference work.
- BF16 block swap works in LoRA training and inference.
- No H3-specific `batch_size == 1` assertion exists.
- One structurally compatible replicated batch executes with a shared layout and base timestep.
- Incompatible multi-sample layouts fail in the post-build step-zero preflight rather than after cache work and partial training.
- H3 loss transport reuses `DiTOutput.extra` rather than introducing a parallel output type.
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
