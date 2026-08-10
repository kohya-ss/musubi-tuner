# MiniMax-H3 R1 BF16 Support Design

Date: 2026-08-03

Status: Approved R1 after upstream scope, provenance, and video-only review

Branch: `codex/minimax-h3-support`

Base: `kohya-ss/musubi-tuner@8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`

## 1. Summary

R1 adds native BF16 MiniMax-H3 LoRA training and joint video/audio inference to Musubi Tuner for:

- `t2va`: text-to-video-with-audio.
- `fl2va`: first/last-frame-to-video-with-audio.
- `ref2va`: ordered JSONL image, video, and audio references-to-video-with-audio.

The implementation must fit Musubi's existing cache filename, tensor-key, collator, trainer, LoRA, compilation, and block-offload contracts. In particular, the target video cache must load as `batch["latents"]`, variable-length text tensors must use the `varlen_` prefix, and the H3 trainer must explicitly create target-audio noise inside its `process_batch` override.

R1 requires every H3 dataset to use `batch_size = 1`; gradient accumulation is the effective-batch mechanism. Real packed batching needs text padding, attention masks, and per-sample structural plans, so it belongs in a later PR rather than a replication-only R1 path. Structural tensors still keep an explicit leading batch axis to avoid baking an unbatched model interface into the first release.

R1 has two cache-time target-audio policies without changing the released packed layout. The default uses and supervises real target audio when available; when audio is absent, it stores an Audio-VAE-encoded silence placeholder but marks audio supervision unavailable for that sample. Cache-time `--h3_video_only` ignores real target audio and marks every target-audio placeholder unsupervised. Ref2VA reference audio remains conditioning under either policy.

ConvRot, prequantized INT8 loading, runtime LoRA over prequantized weights, and pruned AdaLN are deferred to R2. R1 does not depend on unmerged PR 1008.

## 2. Source Anchors

Model artifacts and released configuration:

- <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- <https://huggingface.co/Comfy-Org/MiniMax-H3>

Apache-2.0 implementation source, pinned to the reviewed Diffusers PR head `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`:

- PR: <https://github.com/huggingface/diffusers/pull/14355>
- Transformer: `src/diffusers/models/transformers/transformer_minimax_h3.py`
- Video VAE: `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3.py`
- Audio VAE: `src/diffusers/models/autoencoders/autoencoder_kl_minimax_h3_audio.py`
- Packing: `src/diffusers/modular_pipelines/minimax_h3/packing.py` and `packing_ref2va.py`
- Text presentation: `src/diffusers/modular_pipelines/minimax_h3/encoders.py`
- Dual scheduler: `src/diffusers/schedulers/scheduling_minimax_h3.py` and `modular_pipelines/minimax_h3/denoise.py`

GPL-3.0 ComfyUI integration used only as an independent numerical and artifact-compatibility reference:

- PR: <https://github.com/Comfy-Org/ComfyUI/pull/15224>
- Merge commit: `57500fc5bc92566a63f2046824f522cd55c335ca`
- Transformer/packing: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy/ldm/minimax/model.py>
- Condition payload: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy/model_base.py>
- Text presentation/tags: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy/text_encoders/minimax.py>
- Public task/shift nodes: <https://github.com/Comfy-Org/ComfyUI/blob/57500fc5bc92566a63f2046824f522cd55c335ca/comfy_extras/nodes_minimax_h3.py>

Released config files and tensors are authoritative for architecture and state-dict shape. The pinned Diffusers code is the implementation lineage for the transformer, VAEs, packing, presentation, and scheduler modules. ComfyUI is not a code source: it is used only to verify observable packing and adapter behavior against released artifacts. ComfyUI-specific sign/slope adapters are not model semantics and are not copied.

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
- Cache synchronized target video and structurally required target-audio latents, using real audio when known and duration-matched silence only as an unsupervised placeholder when missing or intentionally ignored.
- Cache FL2VA first/last visual conditions.
- Cache Ref2VA ordered visual/audio reference latents from JSONL.
- Cache Qwen3-VL-32B layer-50 conditioning in the repository's variable-length format.
- Train BF16-base LoRA adapters with per-sample audio supervision: known audio adds the equal-modality audio mean, while unknown or intentionally ignored audio uses video loss only.
- Generate video and audio jointly and mux the result.
- Generate scheduled joint video/audio samples from the live training transformer and LoRA.
- Load official sharded BF16 and Comfy-Org single-file BF16 artifacts.
- Support block swap for the 50 H3 main blocks in training and inference.
- Require dataset `batch_size = 1`; use gradient accumulation for larger effective batches.
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
- Real multi-sample packed training; it is deferred to a separate PR with padding, masking, and per-sample plans.
- A dedicated `batch_size=1/2/3` test matrix.
- Running all three tasks at `batch_size=2` as an acceptance gate.
- Per-sample timesteps inside one packed forward.
- Loading FL2VA and Ref2VA transformer weights in one process.
- CI with the real 33B transformer or 32B text encoder.
- I2V training or conversion from Ref2VA/FL2VA records.
- Removing target-audio rows from the released packed sequence.
- A standalone `AudioDataset` or audio-only training.
- Deduplicating silence latents across cache files.

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

### 8.3 R1 batch-size gate

`minimax_h3_train_network.py` overrides `_build_dataset`, calls the base implementation, and checks each returned H3 `BucketBatchManager` before accelerator creation or model loading. Every configured dataset must have `batch_size = 1`; any other value raises with the dataset index and directs the user to gradient accumulation.

The gate reads no cache files, constructs no layout fingerprints, and does not inspect bucket contents. Task and tensor-role correctness remain ordinary cache/runtime validation concerns. The model and runtime repeat the `B == 1` assertion as defense in depth so direct API calls cannot accidentally enter an unsupported replicated path.

### 8.4 Cache filenames

Reuse the existing filename contract without adding tokens:

```text
{item_key}_{frame_pos}-{frame_count}_{width}x{height}_mmh3.safetensors
{item_key}_{frame_pos}-{frame_count}_mmh3_te.safetensors
```

`VideoDataset.prepare_for_training` continues to recover `item_key`, frame range, resolution, and architecture from these names. Its H3 branch pairs each latent cache with the text cache carrying the same frame-range token. This is required for FL2VA because the selected crop's first and last frames are part of the Qwen presentation; a source-level text cache would silently alias different `chunk` or `slide` crops. T2VA and Ref2VA use the same crop-specific naming contract for one unambiguous lookup rule. R1 does not encode task or reference layout in the filename, and the shared bucket-construction path does not read safetensors headers.

Task, VAE fingerprints, media fingerprints, temporal alignment, and ordered reference kinds remain safetensors metadata for cache-command reuse checks and diagnostics. Training compatibility is determined by the standard filename plus required tensor roles; R1 does not add header reads to bucket construction.

### 8.5 Cache I/O functions

Add architecture-specific writers to `dataset/cache_io.py`:

```python
save_latent_cache_minimax_h3(...)
save_text_encoder_output_cache_minimax_h3(...)
```

Both call the existing common writers with `ARCHITECTURE_MINIMAX_H3_FULL`. Shared cache parsing and `BucketBatchManager.__getitem__` are not changed for H3.

The latent writer also requires exactly one scalar `mmh3_audio_loss_weight_float32` tensor. It validates shape `[]`, dtype `float32`, and an exact value of `0.0` or `1.0`; missing, incorrectly named, non-finite, or non-binary policy data is an error. The shared collator already removes the dtype suffix and stacks this scalar as `batch["mmh3_audio_loss_weight"]`.

## 9. Dataset Contract

### 9.1 Target media

Every sample has a target video and caption. Target audio is synchronized media owned by that video record, not a separate dataset. `H3Record.target_audio` is optional because a structurally valid target-audio latent can be encoded from silence.

The media-layer change is confined to `minimax_h3/media.py`: `_resolve_target_audio(..., video_only=...)` returns `Optional[H3AudioSource]`; `make_h3_directory_record` and `load_h3_jsonl_records` pass the cache policy through their existing resolution calls. In video-only mode, `_resolve_target_audio` returns `None` before inspecting target-audio fields or media. Reference parsing keeps its existing independent audio rules.

In default joint AV mode, resolve target audio in this order:

1. JSONL `audio_path`.
2. One exact same-stem sidecar next to the target video.
3. The target video's embedded audio stream.
4. No source, represented by `target_audio = None`.

Case 4 is not an error. Cache construction encodes duration-matched stereo silence only as the structurally required placeholder and sets that sample's audio loss weight to `0.0`. Missing audio means unknown, not an observed silent target. In default mode, log per-item warnings for at most the first 10 missing-audio records, then suppress further per-item warnings and emit one completion summary with real-audio, missing-audio, video-only, and supervised-fraction counts. Explicit video-only mode emits the summary but no per-item missing-audio warnings. A source that the user actually supplied remains strict: a missing or undecodable explicit path, a selected sidecar without audio, multiple matching sidecars, an undecodable embedded stream, or materially short selected audio raises rather than silently falling back.

`--h3_video_only` is a cache-command flag only. During video-only caching, target-audio discovery is bypassed entirely: do not validate JSONL `audio_path`, scan sidecars, probe embedded target audio for this purpose, decode target audio, or fingerprint a target-audio file. Always store `target_audio = None`, encode the silence placeholder, and set audio loss weight `0.0`. This rule affects target audio only. Ref2VA reference audio remains resolved, validated, fingerprinted, encoded, and presented as conditioning.

Media identity always includes the target video and reference media. It includes a target-audio path only when `target_audio` is not `None`. Consequently ignored audio changes do not invalidate video-only caches, while selected real audio remains part of joint AV cache identity.

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

When `target_audio is None`, do not call the media decoder. Allocate a stereo FP32 waveform of shape `[2, A * 800]` filled with zeros and pass it through the released Audio VAE posterior-mode path. The resulting latent must still be `[32, 2, A]`; silence must not be represented by an invented all-zero latent. This applies to missing audio in joint AV mode and to every sample in video-only mode. `--audio_vae` therefore remains required in both modes.

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
| Audio loss policy | `mmh3_audio_loss_weight_float32` | `mmh3_audio_loss_weight` |
| FL first condition | `latents_first_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_first` |
| FL last condition | `latents_last_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_last` |
| Ref image 000 | `latents_ref_000_image_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_ref_000_image` |
| Ref video 000 | `latents_ref_000_video_{Fc}x{Hc}x{Wc}_{dtype}` | `latents_ref_000_video` |
| Ref audio 000 | `latents_ref_000_audio_32x2x{Ac}_{dtype}` | `latents_ref_000_audio` |

Numbered reference tensor keys follow the JSONL order. The geometry suffix after the last underscore is opaque to the collator; its purpose is to preserve the existing role-key conversion.

Target video shape is `[24, Fv, Hv, Wv]`. Target audio shape is `[32, 2, A]`. Visual condition/reference shapes are `[24, Fc, Hc, Wc]`; audio reference shapes are `[32, 2, Ac]`.

The audio loss policy is a scalar float32 with an exact value of `1.0` when a real target-audio source was decoded and `0.0` when target audio was missing or intentionally ignored by `--h3_video_only`. It is behavioral per-sample data, not a checksum for a training CLI flag. A normal dataset may therefore contain both values.

Each cache stores its own silence latent. For the common `F=124`, `A=207` case, a `[32,2,207]` FP32 audio latent is about 52 KB, versus about 7.16 MB for a `[24,37,84,48]` BF16 video latent. The audio placeholder is below one percent of that payload, so R1 does not add shared-blob references, canonical-silence lookup, or any other deduplication mechanism.

The audio cache deliberately preserves the released audio VAE layout: feature width, stereo channel, then time. The encoder boundary stores `[32, 2, A]` directly from released `[B, 32, 2, A]` output; it must not transpose to `[2, 32, A]` while retaining a misleading geometry key.

### 11.2 Posterior policy

- Target video uses a reproducible posterior sample derived from cache seed plus canonical item key.
- Target audio and all reference audio use the audio posterior mean/mode; the released H3 audio path does not sample `logs_proj`.
- Silence placeholders are encoded through that same posterior-mode path and are not assumed to produce zero latents.
- FL2VA and Ref2VA visual conditions sample with fixed seed 42.
- Visual condition samples round through FP16 before normalization to match released condition behavior.
- The video VAE runs target and condition encoding in FP32. The explicit FP16 round-trip applies only to the sampled condition latent, not to VAE weights or encoder compute. Video decoding uses the published FP16 artifact in FP16.

The cache metadata records the posterior policy, source fingerprints, crop timestamps, target geometry, ordered reference kinds, normalization constants, and VAE fingerprints. It additionally records diagnostic provenance as exactly one of `target_audio_policy=real-supervised|missing-unsupervised|video-only-unsupervised`.

For `--skip_existing`, the H3 comparator maps the three-valued policy to the derived content class `supervised = (target_audio_policy == "real-supervised")` and compares that class rather than the raw provenance string. This still rebuilds whenever real audio and silence differ, including embedded audio whose path is already the target-video path, while default missing-audio and explicit video-only caches can reuse one another because both contain the same silence latent and scalar `0.0`. The raw policy remains creation provenance for cache inspection and may therefore describe the policy that originally produced a tensor-equivalent reused cache. Do not store a second boolean metadata field for this derived fact.

The scalar is a backward-compatible extension of H3 latent cache version 1. New writers always emit it. Legacy H3 caches without audio-policy metadata or the scalar are known to contain real audio because the previous cache command rejected missing target audio, so the runtime interprets complete absence as supervised weight `1.0`. For cache reuse, an absent legacy policy maps only to the supervised class. Existing latent and text caches therefore do not require a blanket rebuild.

### 11.3 Collation behavior

All `latents_` tensors and the scalar loss-policy tensor use the existing `torch.stack` path. At runtime the scalar has shape `[1]` under R1's batch-size gate. R1 introduces no custom H3 collator and no new bucket dimension.

Different target audio lengths cannot occur within an existing `(width, height, frame_count)` bucket because `A` is a deterministic function of target `F`. R1 requires `batch_size = 1`, so heterogeneous references never enter one collated batch. The collator and H3 runtime retain shape assertions as defense in depth for direct API calls.

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

The shared collator returns variable-length tensors as one-element lists and does not pad them. H3 stacks that one item into `[1, L, D]` hidden states and `[1, L]` token tags. R1 does not add padding or an attention mask; those are prerequisites for a later real-batching PR.

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

For each packed sequence, sort the distinct model-time values and build `row_timestep_indices[B, S]`. Main transformer blocks have three modality slots per distinct time, so each row selects block AdaLN modulation with:

```text
block_adaln_index[row] = 3 * row_timestep_indices[row] + token_tag[row]
```

The text span must therefore be split at token-tag runs or indexed row-by-row; treating it as a uniform tag-1 segment is incorrect. R1 produces `token_tags[1, S]`, `row_timestep_indices[1, S]`, and `row_timesteps[1, S]`. The leading batch axis is explicit even though its only supported size is one.

The derived modulation run plan is flat for that one supported packed sequence: `tuple[tuple[start, stop, adaln_row], ...]`. It does not add a second per-batch nesting level. A future real-batching PR needs per-sample text padding and masking and will define its own plan representation; R1 does not build that unused mechanism early.

The FinalLayer is different: its AdaLN projection has one slot per distinct time, not three modality slots. Target video selects `video_timestep_index` directly and target audio selects `audio_timestep_index` directly. FinalLayer must never receive `3 * index + tag`; modality separation there comes from the two output heads, not a tagged AdaLN table. Text and condition rows do not enter either final output head.

The packer returns explicit row indices for target video/audio and never infers row roles from tensor-key sorting.

### 13.4 Exact FP64 rotary clock

The rotary grid is a checkpoint contract, not an arbitrary monotonic position. Construct `position_ids[1, S, 3]` in FP64 and preserve this exact clock before the model converts it for frequency multiplication.

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

### 13.5 Derived-layout reuse

The checkpoint owns `rope.inv_freq`; the model registers an empty persistent FP32 buffer and strict loading must supply it. No analytic fallback frequency is synthesized.

`position_ids` and the resulting rotation table depend only on `(layout, execution device, activation dtype)` after the checkpoint is loaded. The model keeps a bounded two-entry LRU of the completed rotation table, so a cache hit rebuilds neither tensor. Device/dtype moves and state-dict loads clear this cache. The bound prevents a variable-resolution dataset from retaining one large GPU table per bucket.

The dataset does not guarantee one layout for the entire run: epoch shuffling can alternate several valid buckets. Complete timestep rows are also not invariant because `model_t_video` and `model_t_audio` change every training and sampling step. Caching those values would produce almost no hits or grow without bound. Instead, the dynamic AdaLN run boundaries are detected with a tensor comparison and `nonzero`; Python work scales with the number of runs, not `S` scalar tensor reads.

## 14. Trainer Integration and Mode-Dependent Loss

### 14.1 Fixed base-loop contract

`NetworkTrainer` reads `batch["latents"]`, applies `scale_shift_latents`, and creates:

```python
noise = torch.randn_like(latents)
```

before it calls `process_batch`. H3 therefore uses `batch["latents"]` for target video and treats the incoming `noise` as video noise only.

The H3 cache stores already normalized target-video latents, so the H3 trainer's `scale_shift_latents` implementation is the identity. Target-audio latents are also cached normalized and are consumed directly inside `process_batch`.

The H3 trainer overrides both:

- `process_batch`: construct dual-modality noise/noisy inputs, pack, call the DiT, and return the standard `DiTOutput` with audio tensors in `extra`.
- `compute_loss`: apply the validated per-sample audio loss weight and return decomposed metrics.

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

One packed forward has one scalar unshifted base value `u` for its single item. Without a timestep pool, H3 samples one scalar. When the bucket manager supplies a timestep value, the runtime requires exactly one value; values must never be silently discarded or averaged.

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

Musubi applies condition noise to latent-shaped tensors before the pack permutation. ComfyUI applies statistically equivalent noise after packing. The distribution is the same, but random numbers land on different packed coordinates, so equal seeds do not imply bitwise-equal conditioned rows.

### 14.7 Loss object and reduction

Reuse the repository's existing `training.trainer_base.DiTOutput` extension seam rather than defining a parallel result type:

```python
DiTOutput(
    pred=video_pred,
    target=video_target,
    extra={
        "audio_pred": audio_pred,
        "audio_target": audio_target,
        "audio_loss_weight": audio_loss_weight,
    },
)
```

For samples whose cached audio loss weight is `1.0`, the overridden `compute_loss` calculates:

```text
video_loss = mean((output.pred - output.target) ** 2)
audio_loss = mean((output.extra["audio_pred"] - output.extra["audio_target"]) ** 2)
loss = video_loss + audio_loss
```

This is an intentional equal-modality policy for samples with known audio, not a row-weighted global mean: each head contributes one scalar mean even though video contains many more elements. Consequently an individual audio element has greater influence than an individual video element. R1 chooses this explicitly so the much smaller audio head is not diluted by video row count. The trainer logs `loss/video`, `loss/audio`, and total loss. Condition rows do not enter either mean, and the overridden path never calls `compute_loss_weighting_for_sd3`.

For samples whose cached audio loss weight is `0.0`, whether from missing audio or explicit video-only caching, `compute_loss` calculates only `video_loss`. It must not evaluate audio MSE and multiply the result by zero; it skips the audio loss expression and reports `loss/audio = 0`. This removes the direct supervised gradient from the audio output objective. The shared transformer can still receive video-loss gradients through silence target-audio input rows because H3 uses joint self-attention; unsupervised audio does not mean those rows are computationally disconnected.

The artifact policy is recorded as `ss_minimax_h3_loss_policy=video_mean_plus_optional_audio_mean`: the binary cache scalar determines whether the optional audio mean is present for each sample.

With `batch_size = 1` and gradient accumulation over `N` micro-steps, video loss appears in all `N` steps while audio loss appears only in supervised steps. Accelerate divides the accumulated gradient by `N`, so the expected aggregate audio coefficient is the supervised-audio sample fraction `p`; equal head weighting applies only inside a sample whose audio weight is `1.0`. R1 deliberately does not divide audio loss by `p`, because that would amplify a small supervised subset.

The coefficient `p` is a long-run expectation, not a uniform multiplier on every optimizer step. With accumulation length `K`, supervised micro-steps arrive intermittently; under shuffled independent sampling the probability of no audio gradient in one optimizer step is approximately `(1-p)^K` (about 81% for `p=0.05`, `K=4`). The remaining steps receive full-strength per-sample audio terms. Optimizer momentum may smooth this sequence but does not make it equivalent to multiplying every step by `p`.

After the batch-size gate, the H3 trainer computes `supervised_audio_fraction` from the `ItemInfo` objects already present in the constructed batch managers; it must not run a second glob. Build one `Counter` keyed by resolved `latent_cache_path`, so `num_repeats` contributes to the denominator while each unique cache file is opened exactly once. Header/scalar reads are therefore `O(unique cache files)`, independent of repeats. Read metadata, tensor keys, and only the 4-byte policy scalar; never load video or audio latent payloads. The trainer logs the fraction once at startup and saves the same `[0,1]` decimal as `ss_minimax_h3_supervised_audio_fraction`.

The startup scan handles cache states exhaustively:

- Policy metadata and scalar both absent: legacy real-audio cache, weight `1.0`.
- Scalar present while policy metadata is absent: invalid incomplete cache.
- Policy metadata present while the scalar is absent: invalid incomplete cache.
- Both present: validate the policy enum and scalar dtype/shape/value, and require `real-supervised <-> 1.0` or either unsupervised policy `<-> 0.0`.

Any invalid or contradictory state fails before model allocation. Runtime retains the missing-key fallback only for direct legacy `B=1` calls that bypass dataset construction.

### 14.8 Audio supervision and packed semantics

`minimax_h3_cache_latents.py` exposes `--h3_video_only`, defaulting to false. The training command has no corresponding mode flag: supervision follows each cache record rather than duplicating that fact in CLI state.

Before layout construction or transformer execution, `_runtime_batch_plan` validates a present `batch["mmh3_audio_loss_weight"]` as a float32 tensor of shape `[1]` with exact value `0.0` or `1.0`. If the key is absent, it uses the legacy value `1.0`. The validated scalar is carried in `DiTOutput.extra` and directly selects the loss branch; it is not compared with a run-wide switch.

Every sample still loads and noises the cached target-audio latent, builds both timesteps, includes target-audio rows in the packed transformer sequence, and produces both output heads. Removing those rows would change position ids, AdaLN routing, and full self-attention behavior from the released model. The scalar changes supervision, not model topology.

## 15. Batch Semantics

R1 supports one real sample per packed forward. Every H3 dataset must configure `batch_size = 1`; gradient accumulation supplies a larger effective batch without requiring unrelated captions or reference layouts to share one attention sequence.

The tensor interfaces retain their batch axes: values use `[1, S, D]`, positions use `[1, S, 3]`, and tags/timestep indices use `[1, S]`. Dataset, runtime, sampler, and model entry points all reject `B != 1`. No structural fingerprint scan, layout-signature bucket, replicated-forward path, media padding, text padding, attention mask, or token-budget sampler is part of R1. Real T2VA/FL2VA batching is deferred to a separate PR; heterogeneous Ref2VA layouts may remain single-item even then.

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
- latent/text cache format versions, retaining backward-compatible H3 latent cache version `1`
- `ss_minimax_h3_audio_supervision=per_sample_binary_cache_weight`
- loss policy `video_mean_plus_optional_audio_mean`
- `ss_minimax_h3_supervised_audio_fraction`, the effective fraction of training examples with audio weight `1.0`

`ss_minimax_h3_task` records the requested task. `ss_minimax_h3_base_family` records the released checkpoint family, so T2VA correctly records task `t2va` with base family `fl2va`; there is no separate released T2VA transformer.

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

The standalone generator uses forward-only block swap. Training-time sampling reuses the shared cadence, distributed prompt assignment, RNG restoration, and block-swap mode transitions, but overrides the shared single-VAE, video-output-only preparation and per-prompt inference hooks. H3 prepares Qwen3-VL states and condition latents before loading the transformer, retains the video and audio VAEs on CPU, moves them to the accelerator one at a time after joint denoising, and muxes the decoded result. Sampling uses the live transformer so the currently trained LoRA remains active.

Training-time sample preparation parses each prompt's generation record once. Text presentation and Video-VAE encoding deliberately decode visual references in separate phases rather than retaining potentially hundreds of MB of raw pixels per prompt across text-encoder teardown; the second phase reuses the stored canonical record and documents this memory-for-decode tradeoff. Audio-condition encoding receives only the recorded reference-video frame-count map and never depends on a dead raw-visual argument.

Training-time and standalone sampling remain joint video/audio generation. Zero audio loss does not preserve base-model audio behavior: H3 is a single-stream model, and the default LoRA targets modify the same 50-block attention and MLP weights used by video and audio tokens. Video-only or low-`supervised_audio_fraction` training can therefore make generated audio worse than the base model even though the audio head has no direct loss. The risk generally grows with adapter capacity/strength and training exposure, but degradation is not guaranteed to be monotonic. Treat audio from a fully video-only LoRA as unconstrained output, not merely an ignorable metric.

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

The native dual-scheduler path does not negate the predictions and does not apply `d(sigma_audio) / d(sigma_video)`. ComfyUI instead uses one video-sigma sampler plus a pointwise audio slope. Those updates agree only to first order; at finite step size their trajectories are not bit-exact. Together with the condition-noise placement difference in Section 14.6, this means a shared seed is not a promise of bitwise-identical tensors or media. R1 acceptance compares packing, timesteps, raw-head parity, scheduler invariants, and output quality, not final ComfyUI tensors or media hashes.

No unconditional sequence or CFG pass is created.

## 19. Errors and Diagnostics

Fail before expensive allocation where possible for:

- Unsupported R2 artifact formats.
- Broken explicitly selected target audio; genuinely absent target audio logs a warning and uses an unsupervised silence placeholder.
- Audio/video decode or timestamp failures for selected target or reference media.
- Materially short target audio.
- Fewer than 5 frames or invalid `17 * n + 5` geometry after architecture-aware normalization.
- Released-duration violations without override.
- Invalid Ref2VA count, order, or duration.
- Ref2VA without `video_jsonl_file`.
- FL2VA/Ref2VA cache used under the wrong task.
- Missing H3 tensor roles, invalid dtypes/shapes, or cache architecture/format mismatch.
- Invalid present audio-loss policy tensors; absence is accepted only as the legacy real-audio value `1.0`.
- Invalid/stale text token tags or presentation fingerprints.
- Qwen3-VL expanded length over 32768.
- Unsupported timestep sampling, loss weighting, H3 shift, or condition-clean value.
- Dataset or runtime `batch_size` other than one.
- Block swap outside 1 through 48.
- H2D-only training without gradient checkpointing.
- Conditioning tensor batch dimension different from target video batch size.

Startup logs include `supervised_audio_fraction`. OOM-oriented logs include target video/audio shapes, exact `A`, target-audio policy, text length, reference counts and shapes, packed row count, dtype, and block-swap configuration. R1 does not rewrite batch size automatically.

## 20. Test Strategy

Tests use tiny synthetic model configurations unless marked manual.

### 20.1 Cache and dataset contract

- Save a synthetic H3 latent cache through `save_latent_cache_minimax_h3` and load it through `BucketBatchManager`; assert keys are exactly `latents`, `latents_audio`, `mmh3_audio_loss_weight`, and task-specific `latents_*` roles.
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
- Assert joint AV resolves explicit, sidecar, and embedded target audio in order, but returns `target_audio = None` when none exists; broken selected sources and ambiguous sidecars still fail.
- Assert both directory and JSONL record construction propagate the cache-only video policy into `_resolve_target_audio`, and `H3Record.target_audio` is optional without changing reference-audio parsing.
- Assert video-only bypasses target-audio probing, decoding, sidecar discovery, and fingerprinting while retaining Ref2VA reference-audio validation and identity.
- Assert default missing audio encodes a `[2, A * 800]` FP32 zero waveform and stores loss weight `0.0`; real audio stores `1.0`; video-only ignores real audio and stores the same silence shape with `0.0`. More than 10 missing records produce only 10 per-item warnings plus one final policy-count/fraction summary.
- Assert `--skip_existing` compares the supervision class derived from policy rather than raw provenance: embedded/external real audio versus silence rebuilds, while default-missing versus video-only silence reuses the tensor-equivalent cache.
- Exhaustively test legacy complete absence, scalar-without-policy, policy-without-scalar, invalid enums/dtypes/shapes/values, and every policy/scalar contradiction. Only complete legacy absence defaults to `1.0`.
- Assert silence latents remain ordinary per-sample cache tensors and no shared/deduplicated silence artifact or cache reference is created.

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
- Assert the flat single-sequence modulation run plan and compare FP64 in-place scale/shift/gate outputs and all input, AdaLN, and RMSNorm-weight gradients against an out-of-place segmented reference.

### 20.3 Trainer hooks

- Assert `process_batch` uses incoming base-loop noise only for video and creates independent audio noise with the audio shape.
- Assert dataset construction rejects `batch_size != 1` before accelerator/model creation and recommends gradient accumulation without reading cache files.
- Assert runtime and model entry points reject `B != 1`, and a timestep pool must contain exactly one value.
- Assert `position_ids`, packed token tags, row timesteps, and row timestep indices preserve an explicit leading batch axis.
- Assert unsupported `timestep_sampling`, `weighting_scheme`, generic flow shift, H3 shifts, and condition coefficients are rejected.
- Assert condition inputs use `a*x0 + (1-a)*noise`; training seeds change per step, visual conditions restart one shared stream, and audio uses the `seed+1` stream.
- Mock raw output heads and assert `process_batch` targets are `latents - noise` with no prediction negation or audio slope scaling.
- Assert H3 returns the standard `DiTOutput`, stores audio tensors in `extra`, and `compute_loss` never calls SD3 weighting while reporting separate video/audio means.
- Assert new `0.0`/`1.0` cache weights and the implicit legacy `1.0` select their loss branch before transformer execution without a training-side video-only flag.
- Assert both missing-audio and explicit video-only samples use video MSE, report zero audio loss, and have no direct gradient from `audio_pred`; assert real-audio samples remain `video_mean + audio_mean` with both gradients.
- For a synthetic mixed dataset, assert `supervised_audio_fraction` follows effective repeated micro-step counts, is logged and saved in metadata, and does not renormalize the audio loss or gradients by the reciprocal fraction. Assert paths come from constructed batch managers with no second glob and each unique cache is opened once regardless of repeats.
- Assert video-only forwards retain silence target-audio rows and Ref2VA reference-audio conditions.
- Assert mismatched conditioning batch dimensions fail clearly.
- Assert training sample preparation loads Qwen3-VL once, builds task-specific layouts, retains both VAEs on CPU, and returns no shared single VAE.
- Assert each training sample record is loaded once, visual references are intentionally re-decoded for the Video-VAE phase, and audio conditions consume the stored reference-video frame counts without raw pixels.
- Assert scheduled sampling runs the live transformer/LoRA, decodes the two latent outputs sequentially, restores transformer mode, and muxes a joint AV file.

### 20.4 Model, LoRA, and block swap

- Tiny BF16 forward for T2VA, FL2VA, and Ref2VA packed layouts at `B=1`.
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
- The default policy supervises aligned real target audio when available; missing audio warns, retains an Audio-VAE-encoded silence placeholder, and contributes no audio MSE. A broken selected source remains an error.
- Cache-time video-only ignores real target audio, retains silence target-audio rows, and contributes no audio MSE; Ref2VA reference audio remains conditioning.
- New latent caches carry consistent policy provenance and a per-sample scalar; complete legacy real-audio absence remains valid with implicit weight `1.0`. Cache reuse compares only the supervision class derived from provenance, while contradictory or partial new states fail before model allocation.
- Per-sample equal weighting is documented separately from aggregate weighting; the logged and saved `supervised_audio_fraction` describes the dataset-dependent expected audio coefficient, and training does not renormalize by that fraction.
- Documentation states that `supervised_audio_fraction` is an expectation and low values produce intermittent full-strength audio gradients rather than a uniform small gradient on every optimizer step.
- Every cache keeps its own small silence latent; R1 adds no silence-deduplication storage or lookup mechanism.
- The exact integer `F -> A` formula is shared by cache and inference.
- Packed audio/video row counts match the documented formulas.
- FP64 rotary clocks, FL anchors, and reference cursor advances follow the pinned Apache-2.0 Diffusers implementation and are independently cross-checked against ComfyUI behavior.
- Main-block and FinalLayer AdaLN indices follow their distinct three-slot and one-slot rules.
- Condition augmentation, row timesteps, and per-step/shared-stream RNG follow the documented separate contracts.
- T2VA, FL2VA, and JSONL-only Ref2VA execute through native BF16 packing.
- H3 rejects base loss weighting and unsupported timestep sampling rather than silently reversing curves.
- Training uses raw `latents - noise` targets; native model/inference outputs contain neither ComfyUI's negative sign nor its audio slope adapter.
- Video/audio shifts are configurable and native inference advances two finite sigma schedules from one common base grid.
- BF16 LoRA training and inference work.
- Scheduled `--sample_prompts` generation writes muxed video/audio from the live training LoRA for T2VA, FL2VA, and Ref2VA inputs.
- BF16 block swap works in LoRA training and inference.
- Dataset construction, runtime, sampling, and model calls enforce `batch_size == 1`; documentation points larger effective batches to gradient accumulation.
- Packed positions, tags, and row timestep indices keep explicit batch axes despite the R1 limit.
- Ported modules carry Apache-2.0 provenance headers pinned to Diffusers PR #14355; ComfyUI remains validation-only and contributes no implementation code.
- H3 loss transport reuses `DiTOutput.extra` rather than introducing a parallel output type.
- Automated tests pass and real-model R1 evidence is recorded.
- User documentation states BF16-only R1 scope, JSONL Ref2VA, unsupervised missing-audio placeholders, why video-only still requires the Audio VAE, per-sample versus expected dataset-level loss weighting, intermittent low-fraction gradients, `supervised_audio_fraction`, shared-weight audio degradation risk, the cache-only video-only flag, capped missing-audio warnings and summary, legacy-cache compatibility, per-sample silence storage, sampling semantics, batching limitations, and block-swap commands.

## 22. Deferred R2

R2 begins only after PR 1008 is merged upstream and its final public interfaces are available. It requires a separate design review for:

- Dynamic ConvRot policy and hard assertions against silently skipped layers.
- Prequantized Comfy ConvRot loading.
- Cross-implementation rotation basis and dequantization correctness.
- Runtime floating-point LoRA over a prequantized base.
- Normal versus pruned AdaLN time conditioning.
- INT8 block-swap weight/scale placement and device assertions.
- R2 artifact-level numerical and quality acceptance, not merely "loads and executes."
- Before any real `batch_size > 1` support, inject the implicit legacy audio weight at each item-read boundary. Do not leave missing-key fallback after collation: mixing old caches without the key and new caches with it would otherwise shorten the stacked weight tensor and silently misalign weights with samples.

No R2 behavior is an R1 dependency or acceptance criterion.
