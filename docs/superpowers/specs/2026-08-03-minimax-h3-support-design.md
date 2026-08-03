# MiniMax-H3 Support Design

Date: 2026-08-03

Status: Proposed for implementation after user review

Branch: `codex/minimax-h3-support`

Base: `kohya-ss/musubi-tuner@8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`

## 1. Summary

Add native MiniMax-H3 LoRA training and joint video/audio inference to Musubi Tuner for these task modes:

- `t2va`: text-to-video-with-audio.
- `fl2va`: first/last-frame-to-video-with-audio.
- `ref2va`: ordered image, video, and audio references-to-video-with-audio.

The implementation will use Musubi's existing dataset, cache, trainer, LoRA, compilation, and block-offload infrastructure rather than embedding a Diffusers pipeline. It will support the original sharded MiniMax checkpoints and the BF16 and INT8 ConvRot single-file checkpoints published by Comfy-Org.

The first release is intentionally LoRA-only. It must support block swap in both training and inference. It will not force dataset `batch_size` to 1, but it will not introduce a ragged or token-budget batching system.

## 2. Source Anchors and Provenance

The implementation has four source-of-truth layers, in this order:

1. The released model artifacts and configs:
   - <https://huggingface.co/MiniMaxAI/MiniMax-H3>
   - <https://huggingface.co/Comfy-Org/MiniMax-H3>
2. The current ComfyUI integration, including the pruned AdaLN curve behavior:
   - <https://github.com/Comfy-Org/ComfyUI/pull/15224>
   - The earlier reference supplied for this work, <https://github.com/Comfy-Org/ComfyUI/pull/15210>, is superseded by PR 15224 where behavior differs.
3. The Diffusers implementation as a numerical and presentation contract, not as a runtime dependency:
   - <https://github.com/huggingface/diffusers/pull/14355>
4. Musubi's own architecture and lifecycle conventions.

ConvRot support will start by integrating the upstream Musubi PR supplied for this work:

- PR: <https://github.com/kohya-ss/musubi-tuner/pull/1008>
- Pinned head: `fe4818daf4e41bc6d98959a35f55627f07f70d90`
- Verified parent: `8934cfbbb4b9bcfa8071ce209129f0c5eb5df2e6`, the same commit used as this branch's base.

The implementation must fetch the PR ref, verify that its head still equals the pinned SHA, and merge that exact commit. It must not reimplement the PR's ConvRot kernels, quantizer protocol, or LoRA integration. MiniMax-H3 adds only the prequantized Comfy checkpoint adapter and H3-specific wiring that PR 1008 deliberately leaves for its next phase.

## 3. Goals

- Cache MiniMax-H3 target video and target audio latents with exact temporal alignment.
- Cache Qwen3-VL-32B conditioning outputs for all three task presentations.
- Train LoRA adapters against the FL2VA or Ref2VA transformer.
- Generate video and audio jointly and mux them into the requested output.
- Load official sharded checkpoints and all supported Comfy-Org BF16/INT8 variants.
- Support normal and pruned INT8 ConvRot DiT checkpoints.
- Support BF16 and ConvRot INT8 text encoders.
- Preserve a real batch dimension and avoid any validation that requires `batch_size == 1`.
- Support Musubi block swap for the 50 H3 main blocks in LoRA training and inference.
- Produce clear failures for invalid data, incompatible task/checkpoint choices, and unsupported quantization formats.

## 4. Non-Goals

- Full transformer fine-tuning.
- Selective head, AdaLN-only, or other partial-base fine-tuning modes.
- Training the video VAE, audio VAE, or Qwen3-VL text encoder.
- NVFP4/AWQ text encoder loading in the first release.
- Physical LoRA fusion into a prequantized INT8 checkpoint.
- Loading FL2VA and Ref2VA transformer weights in the same process.
- Classifier-free guidance. MiniMax-H3 uses one conditional pass.
- Arbitrary heterogeneous or ragged reference layouts inside one batch.
- A token-budget sampler or a new dynamic batching subsystem.
- A dedicated `batch_size=1/2/3` forward/backward matrix.
- Running every task once at `batch_size=2` as an acceptance requirement.
- CI execution with the real 33B transformer or 32B text encoder.
- Dependence on an unreleased Diffusers build.

## 5. Supported Artifact Matrix

### 5.1 Transformer

The loader accepts these official sharded layouts:

- `MiniMaxAI/MiniMax-H3/transformer`: FL2VA weights, also used for T2VA.
- `MiniMaxAI/MiniMax-H3/transformer_ref`: Ref2VA weights.
- `MiniMaxAI/MiniMax-H3/FL2VA/transformer`.
- `MiniMaxAI/MiniMax-H3/Ref2VA/transformer`.

The loader also accepts these Comfy-Org files:

- `minimax_h3_fl2va_bf16.safetensors`
- `minimax_h3_ref2va_bf16.safetensors`
- `minimax_h3_fl2va_int8_convrot.safetensors`
- `minimax_h3_ref2va_int8_convrot.safetensors`
- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`

`t2va` and `fl2va` require the FL2VA transformer. `ref2va` requires the Ref2VA transformer. One run selects exactly one `--task` and one transformer.

### 5.2 Text Encoder

The loader accepts:

- An official `text_encoder` sharded directory from the root, FL2VA, or Ref2VA layout.
- `qwen3vl_32b_minimax_h3_bf16.safetensors`.
- `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`.

`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` is rejected with an explicit message stating that NVFP4/AWQ is outside first-release scope.

### 5.3 Autoencoders

The loader accepts the official directories and these Comfy-Org files:

- `minimax_h3_video_vae_fp16.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`

VAE weights are shared across tasks.

## 6. Fixed Model Contract

The native transformer implementation uses the released configuration:

| Property | Value |
| --- | ---: |
| Main blocks | 50 |
| Refiner blocks | 2 |
| Hidden size | 5376 |
| Attention heads | 56 |
| Attention head dimension | 128 |
| FFN dimension | 14336 |
| Video latent channels | 24 |
| Audio latent channels | 32 |
| Audio channels | 2 |
| Text feature dimension | 5120 |
| Video patch size | `(1, 2, 2)` |
| Time embedding input dimension | 2688 |
| Video frame rate | 24 fps |
| Audio sample rate | 32000 Hz |
| Audio VAE hop | 800 samples |
| Audio latent rate | 40 Hz |

The apparent difference between `hidden_size` and `num_attention_heads * attention_head_dim` is part of the released architecture and must not be normalized into a conventional transformer shape.

Packed modality tags are fixed:

- Video: `0`
- Text: `1`
- Audio: `2`

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

- `model.py`: transformer, refiner, attention, modulation, output heads, gradient checkpointing, and block swap lifecycle.
- `packing.py`: task layouts, modality tags, position grids, masks, patchify/unpatchify, and timestep rows.
- `video_vae.py`: native released video VAE modules, normalization, encode, and decode.
- `audio_vae.py`: native released stereo audio VAE modules, normalization, encode, and decode.
- `text_encoder.py`: Qwen3-VL loading, prompt presentation, multimodal preprocessing, layer-50 extraction, and text checkpoint adaptation.
- `checkpoint.py`: artifact discovery, sharded streaming, state-dict normalization, variant detection, strict validation, and ConvRot adapters.
- `sampling.py`: paired schedules, denoising loop, latent initialization, decoding, and audio/video synchronization helpers.

Add top-level source entry points and matching root wrappers:

```text
minimax_h3_cache_latents.py
minimax_h3_cache_text_encoder_outputs.py
minimax_h3_train_network.py
minimax_h3_generate_video.py
```

The wrappers follow the repository's existing top-level entrypoint convention and contain no model logic.

## 8. Public CLI Contract

### 8.1 Common task selection

Every H3 command that depends on task semantics accepts:

```text
--task {t2va,fl2va,ref2va}
```

The task is required. It controls presentation, cached condition fields, packed layout, and transformer checkpoint compatibility.

### 8.2 Cache commands

The latent cache command accepts the standard dataset configuration plus:

- `--task`
- `--video_vae`
- `--audio_vae`
- `--allow_experimental_duration`

The text cache command accepts:

- `--task`
- `--text_encoder`
- the tokenizer/processor directory when it cannot be discovered beside the text encoder
- the same dataset configuration used for latent caching

### 8.3 Training

The trainer follows the existing `NetworkTrainer` CLI and adds:

- `--task`
- `--dit`
- H3 cache validation controls
- `--video_loss_weight`, default `1.0`
- `--audio_loss_weight`, default `1.0`
- ConvRot options contributed by PR 1008

The existing block swap flags remain authoritative:

- `--blocks_to_swap`
- `--use_pinned_memory_for_block_swap`
- `--block_swap_h2d_only`
- `--block_swap_ring_size`

### 8.4 Inference

The generator accepts the model paths, `--task`, prompt, output geometry, frame count, seed, step count, LoRA weights, and block swap controls.

Task conditions are:

- `t2va`: prompt only.
- `fl2va`: `--first_frame` and `--last_frame`.
- `ref2va`: ordered repeated reference arguments or a JSON request containing the same reference schema used by training.

Inference emits a video stream and a 32 kHz stereo audio stream, then muxes them. An explicit audio-only or video-only output can be a debug option, but joint muxed output is the normal result.

## 9. Dataset Contract

### 9.1 Target media

MiniMax-H3 training uses a target video and a required target audio track. The target audio source is resolved in this strict priority order:

1. JSONL `audio_path`.
2. A same-stem sidecar next to the target video.
3. The target video's embedded audio stream.

Supported sidecars are discovered by exact target stem and a documented audio extension list. More than one matching sidecar is an error rather than an arbitrary choice. An explicit path that cannot be decoded is an error and does not fall back to another source.

Missing target audio is an error. The cache command must never substitute zero latents, silence, or a generated waveform.

### 9.2 JSONL records

The standard `video_path` and `caption` fields remain required. H3 adds optional `audio_path` and, for Ref2VA, an ordered `references` list:

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

For a video reference, `audio_path` overrides embedded audio. A video reference without an audio stream remains a valid visual-only video reference.

The H3 cache layer keeps raw records indexed by canonical target path while reusing Musubi's existing video bucketing and `ItemInfo` flow. H3-only fields must not change the tuple contract used by unrelated architectures.

### 9.3 Numbered reference directories

Ref2VA also supports a deterministic directory convention. A dataset-level `reference_directory` contains one subdirectory per target stem:

```text
references/
  clip_001/
    000_image.png
    001_video.mp4
    001_audio.wav
    002_audio.wav
```

Rules:

- The numeric prefix defines reference order and starts at `000`.
- Indices are contiguous. A gap is an error.
- One index may contain one visual medium.
- A video and audio with the same index form one video reference with an explicit soundtrack.
- An audio file without a same-index video forms an audio reference.
- An image and audio with the same index is rejected as ambiguous.
- Duplicate media of the same kind at one index are rejected.
- Unknown filenames in a reference directory are rejected rather than ignored.
- If a JSONL record contains `references` and a matching numbered directory also exists, caching fails and asks the user to choose one source.

Reference ordering is semantic: it controls both Qwen3-VL presentation and the packed rotary clock.

### 9.4 Ref2VA limits

Validate before VAE or text encoding:

- At most 9 image references.
- At most 3 video references.
- At most 3 audio-bearing references.
- At most 12 reference items total.
- At least one visual reference is required.
- Reference videos must be between 2 and 15 seconds before target-duration truncation.

A paired video plus explicit soundtrack counts as one reference item and one video/audio-bearing reference.

### 9.5 FL2VA training conditions

FL2VA training derives its first and last conditioning frames from the selected target crop. It does not require separate condition files in the dataset. This keeps the condition exactly aligned with the supervised target. Inference accepts external first and last images.

## 10. Geometry and Temporal Alignment

### 10.1 Target video

- Decode and normalize target video to a fixed 24 fps timeline.
- Use the existing dataset bucket to determine spatial crop/resize.
- Both output axes are multiples of 32.
- The normal released canvas uses a 768-pixel short edge with a soft area cap of `768 * 1344`.
- The target frame count must have the form `17 * n + 5`.
- Training crops downward to the largest valid frame count available in the selected clip.
- Fewer than 5 usable frames is an error.
- Normal supported duration is 5 through 15 seconds.
- `--allow_experimental_duration` permits an out-of-range training crop while preserving all structural checks and logging that the run is outside released conditions.

For `F = 17 * n + 5` target frames, the video VAE produces `5 * n + 2` latent frames.

### 10.2 Target audio

Audio is decoded with PyAV and resampled to stereo 32000 Hz. The video crop start is mapped to audio time as:

```text
audio_start_seconds = crop_start_frame / 24
```

The target audio latent length is:

```text
A = round(F / 24 * 40)
```

The waveform window is exactly `A * 800` samples per channel. The decoder pads a short terminal window only when the source stream reaches its legitimate end within timestamp tolerance; a materially short or discontinuous stream is an error. Longer audio is truncated to the exact window.

The cache records source timestamps, crop start, target frame count, waveform sample count, and audio latent count so alignment failures can be diagnosed without reopening model weights.

### 10.3 Reference media

- Reference images are prepared independently of the target canvas using the released reference transform.
- Reference videos are resampled to 24 fps, prepared on the released 768-short-edge canvas for their own aspect ratio, and truncated to target duration.
- Qwen3-VL presentation samples reference videos at 2 fps with timestamps.
- Reference audio is resampled to stereo 32000 Hz and truncated to target duration.
- A video reference's visual and audio streams share one decoded timeline.

## 11. Latent Cache Contract

The H3 latent cache remains a safetensors artifact and includes a versioned metadata schema. Tensor keys are stable and numbered where order matters.

Core tensors:

- `video_latents`: normalized target video latents, shape `[24, Fv, Hv, Wv]`.
- `audio_latents`: normalized target audio latents, shape `[2, 32, A]`.
- `fl2va_first_latents` and `fl2va_last_latents` for FL2VA.
- `ref_000_video_latents`, `ref_000_audio_latents`, and subsequent ordered reference tensors as applicable.

Metadata includes:

- cache schema version
- task
- source paths and stable file fingerprints
- crop and resampling parameters
- target and latent geometry
- ordered reference kinds
- VAE checkpoint fingerprints
- normalization constants
- layout signature

The `layout_signature` covers task, target latent geometry, ordered reference kinds, and every reference-media latent dimension that must agree for a normal stacked batch. Raw text length is excluded because the collator pads text rows and carries a valid-row attention mask. The signature becomes part of H3's cache bucket key; this permits ordinary compatible batches without inventing ragged reference-media packing.

Posterior policy is part of cache identity:

- Target video and target audio use posterior samples. Caching supplies each item a deterministic generator derived from the cache seed and canonical item key, and stores the seed policy in metadata so rebuilding the same cache is reproducible.
- FL2VA and Ref2VA visual conditions sample the video VAE posterior with a fixed seed of 42.
- Visual condition samples are rounded through FP16 before normalization to reproduce released inference behavior.
- Reference audio uses posterior mode.

The cache reader rejects a task mismatch, missing fields, stale source fingerprints, incompatible layout version, or VAE identity mismatch.

## 12. Text Encoder Cache Contract

MiniMax-H3 uses Qwen3-VL-32B hidden state index 50 without applying the model's final normalization. The cached feature dimension is 5120.

Presentations are non-chat presentations:

- `t2va`: the raw caption.
- `fl2va`: ordered `Picture` blocks for the first and last frames followed by the caption presentation required by the released processor.
- `ref2va`: ordered `Picture`, `Video`, and `Audio` blocks matching the reference list, followed by the caption. Video blocks use 2 fps samples with timestamps.

Exact separators, labels, timestamp formatting, and processor inputs are copied from the current reference implementation and locked by golden tests. They are not approximated by a generic chat template.

The text cache stores:

- `text_hidden_states`, shape `[L, 5120]` before collation.
- `token_tags`, shape `[L]`, using text tag `1` for text-encoder rows.
- the task and ordered presentation manifest.
- text encoder, tokenizer, and processor fingerprints.
- reference media fingerprints and video-sampling metadata.

Text caching is a separate process from latent caching and training. The 32B text encoder is released before the 33B denoiser is loaded.

## 13. Packed Sequence and Forward Contract

Every forward is one joint self-attention sequence. The logical layouts are:

```text
t2va:   [text | target audio | target video]
fl2va:  [text | keyframe conditions | target audio | target video]
ref2va: [text | ordered reference blocks | target audio | target video]
```

The packing layer produces, per sample:

- packed hidden rows
- FP64 rotary position grid before frequency projection
- modality tags
- target video indices
- target audio indices
- condition row counts
- timestep values per row
- an unpacking description for the video and audio heads

Text, condition, audio, and video order must never be inferred from tensor key sorting. It comes from the explicit task layout.

Visual condition rows are noised and held at timestep `0.999`. Reference audio condition rows remain clean at timestep `1.0`. Generated audio and video rows receive their modality-specific timesteps.

The model returns predictions for the generated target rows. The packing layer unpacks them to:

- video prediction shaped like target video latents
- audio prediction shaped like target audio latents

No classifier-free unconditional sequence is built.

## 14. Training Objective

Draw one base flow value `u` per sample using the selected Musubi timestep distribution. Use the released rational shift:

```text
shift(u, s) = s * u / (1 + (s - 1) * u)
sigma_video = shift(u, 12)
sigma_audio = shift(u, 3)
t_video = 1 - sigma_video
t_audio = 1 - sigma_audio
```

Noise target rows independently for the two modalities using their shifted sigma while retaining the common underlying `u`. The velocity targets are:

```text
v_video = x0_video - noise_video
v_audio = x0_audio - noise_audio
```

Compute mean MSE separately after unpacking:

```text
loss = video_loss_weight * mean_mse_video
     + audio_loss_weight * mean_mse_audio
```

Defaults are `1.0` and `1.0`. Log total, video, and audio losses independently. Conditioning rows never contribute to the supervised loss.

## 15. Batch Semantics

There is no H3 check that forces `batch_size` to 1.

The implementation preserves the leading batch dimension through text collation, noise creation, timestep generation, packing, attention, output unpacking, and loss reduction. It uses the existing dataset bucket/collator model:

- Samples may share a batch when target geometry and the H3 `layout_signature` are compatible.
- Text rows are padded to the longest prompt in the batch, and the resulting valid-row mask is applied by attention so padded rows cannot affect any modality.
- Incompatible condition/reference shapes are placed in different cache buckets.
- A malformed batch fails with the conflicting sample keys and shapes.

This is deliberately narrower than arbitrary ragged batching. No token-budget scheduler, per-sample forward loop, or special batch-size test matrix is added. A lightweight validation test only verifies that values greater than one are not rejected by H3-specific argument or dataset validation.

## 16. Checkpoint Normalization

`checkpoint.py` discovers files, streams shards on CPU, maps all supported external names into one native schema, and performs strict missing/unexpected-key validation after known conversions.

Variant detection uses tensor structure and safetensors metadata:

- Standard BF16 AdaLN has the released 2688-dimensional time embedding path.
- Pruned ConvRot has `adaln_t_table` of shape `[1025, 8]` and block AdaLN projection input width 8.
- Normal block 0 AdaLN projection has shape `[96768, 2688]`.
- Pruned block 0 AdaLN projection has shape `[96768, 8]`.

The pruned model evaluates its 1025-row curve table at a continuous training or inference timestep by linear interpolation between neighboring rows. It must not round random training timesteps to an integer table row.

FL2VA and Ref2VA transformers share a tensor schema, so semantic variant detection cannot be infallible. The loader treats `--task` as authoritative, warns when the path or filename strongly suggests the opposite variant, and records both the requested task and artifact fingerprint. It must not claim certainty from identical tensor shapes.

## 17. ConvRot Integration

### 17.1 Upstream foundation

Merge pinned PR 1008 before H3 implementation. Reuse its:

- ConvRot INT8 kernels.
- Quantization utilities.
- generic `quantizer=` LoRA loading seam.
- module representation that keeps quantized layers compatible with LoRA and block swap.

### 17.2 Dynamic ConvRot

Supported BF16 transformer and text-encoder checkpoints may be quantized during loading through PR 1008's quantizer protocol. H3 supplies only architecture-specific include/exclude policy and CLI wiring.

### 17.3 Prequantized Comfy ConvRot

Comfy files declare:

```json
{"format":"int8_tensorwise","convrot":true,"convrot_groupsize":256}
```

Their linear tensors use `.weight` plus `.weight_scale`; PR 1008's internal module uses `.weight` plus `.scale_weight`. The adapter requires FP32 scales shaped `[out_features, 1]`, validates metadata, shape, dtype, group size, and paired-key presence, then renames the scale key during streamed loading.

Unknown quantization formats, missing metadata, partial weight/scale pairs, or unexpected scale shapes are hard errors. The adapter must not guess that an arbitrary INT8 tensor is ConvRot.

### 17.4 Pruned ConvRot

Pruned checkpoints use the same ConvRot linear path plus the continuous AdaLN table described above. Normal and pruned checkpoints share the public H3 model class but use separate, explicit time-conditioning implementations selected at load.

### 17.5 Unsupported NVFP4

The published text NVFP4 artifact declares `format=nvfp4` and `full_precision_matrix_mult=true` and contains `pre_quant_scale`. It is rejected before model allocation with a message naming the supported BF16 and INT8 ConvRot alternatives.

## 18. LoRA Contract

The first release trains only LoRA-family network weights supported by the existing Musubi network trainer. The frozen H3 base can be BF16, dynamically ConvRot-quantized, or a supported prequantized ConvRot checkpoint.

Default target modules are the invariant projections in each of the 50 main blocks:

- `attn.qkv_proj`
- `attn.out_proj`
- `mlp.fc1`
- `mlp.fc2`

Default targeting excludes AdaLN, time conditioning, input/output projections, VAE modules, text encoder modules, and refiner-only differences. This makes the default adapter structurally loadable on standard and pruned bases. Structural portability does not promise numerical equivalence across those bases.

Saved network metadata includes:

- architecture `minimax_h3`
- training task
- base artifact fingerprint
- standard/pruned base variant
- BF16/dynamic-ConvRot/prequantized-ConvRot base representation
- target module policy
- cache schema versions

At load, a cross-task or cross-base warning is emitted when metadata differs.

Inference behavior:

- A BF16 base may use the existing streamed static merge path.
- A prequantized INT8 base applies LoRA as a floating-point runtime branch over the quantized base result.
- The prequantized path never dequantizes, merges, and requantizes the base, because that can erase small adapter deltas.
- The public CLI continues to use the existing `--lora_weight` convention.

## 19. Block Swap Contract

Block swap is a required first-release feature for both H3 LoRA training and inference.

### 19.1 Configuration

- `--blocks_to_swap 0` or an omitted value disables swapping.
- Valid enabled values are 1 through 48 for the 50 main blocks.
- The implementation reuses `BlockSwapConfig.from_args` and `create_offloader` from `custom_offloading_utils.py`.
- Normal training uses the existing backward-capable `ModelOffloader` path.
- Frozen-base LoRA training may use `--block_swap_h2d_only` and the configured ring size.
- H2D-only training requires gradient checkpointing, as enforced by the shared block-swap configuration.
- Pinned-memory behavior remains controlled by the shared flag.
- Inference uses the established forward-only exchange offloader.

### 19.2 Model lifecycle

The native model exposes the same lifecycle used by current Musubi architectures:

- `enable_block_swap(blocks_to_swap, config)`
- `move_to_device_except_swap_blocks(device)`
- `prepare_block_swap_before_forward()`
- `switch_block_swap_for_inference()`
- `switch_block_swap_for_training()`

With swap enabled, checkpoint construction and large-weight loading begin on CPU. Only non-block components move directly to the accelerator. The model prepares block placement after `accelerator.prepare` using disabled automatic device placement for the swapped transformer.

Every main-block iteration follows this order:

1. `offloader.wait_for_block(index)`.
2. Execute the block directly or through non-reentrant gradient checkpointing.
3. `offloader.submit_move_blocks_forward(blocks, index)`.

Training-time sample generation switches to forward-only mode, prepares block placement, generates, switches back to training mode, and prepares again before the next training forward.

### 19.3 ConvRot interaction

BF16, dynamically quantized ConvRot, normal prequantized ConvRot, and pruned prequantized ConvRot use the same block lifecycle.

For prequantized linears, the large INT8 `.weight` participates in block streaming. The H3 block-offload adapter explicitly keeps scale and other small quantization buffers resident on the accelerator. Device assertions test that the kernel receives weight, scale, activation, and rotation state on valid devices after each wait.

The compile helper receives the H3 main block list and disables Linear compilation when block swap is active, following existing Musubi behavior. Block swap must not silently fall back to keeping all 50 blocks on the accelerator.

## 20. Inference Flow

1. Validate task, checkpoint variants, references, geometry, duration, and output path.
2. Load and run Qwen3-VL conditioning, then release it unless precomputed text features were supplied.
3. Encode first/last frames or ordered references with the appropriate VAEs.
4. Initialize target video noise and target stereo audio noise from the request generator.
5. Build the packed layout once.
6. Run paired schedules for the requested step count: video shift 12, audio shift 3.
7. At each step, build row timesteps, run one transformer forward, unpack both predictions, and advance both schedulers.
8. Decode video and audio.
9. Trim decoded media to the common planned duration and mux with PyAV.

No CFG duplication or unconditional pass is performed. Seeds cover condition posterior/noise ordering exactly; the fixed visual-condition posterior seed 42 remains independent from the request seed.

## 21. Error Handling and Diagnostics

Fail early with sample identity and corrective detail for:

- Missing target audio.
- Audio/video decode failures or unusable timestamps.
- Materially short audio after alignment.
- Fewer than 5 video frames.
- A target that cannot form `17 * n + 5` frames.
- Released-duration violations without the experimental override.
- Invalid Ref2VA counts, ordering, pairing, or duration.
- Simultaneous JSONL and directory reference definitions.
- FL2VA or Ref2VA cache used under the wrong task.
- Unsupported checkpoint keys or tensor shapes.
- Unsupported or inconsistent quantization metadata.
- NVFP4/AWQ text encoder selection.
- Block swap outside 1 through 48.
- H2D-only training without gradient checkpointing.
- Incompatible batch layout signatures.

OOM-oriented logs include target latent shape, audio latent length, text length, reference layout signature, packed row count, dtype/quantization mode, and block-swap configuration. They do not automatically rewrite batch sizes or sampling policy.

## 22. Test Strategy

Tests use tiny synthetic configurations unless a test is explicitly marked manual.

### 22.1 Unit tests

- `17 * n + 5` frame alignment and video latent geometry.
- 24 fps crop-to-audio timestamp mapping and 40 Hz latent rounding.
- Target audio source precedence and missing-audio failures.
- JSONL and numbered-directory reference parsing, ordering, pairing, and limits.
- T2VA, FL2VA, and Ref2VA presentation golden fixtures.
- Packed row order, modality tags, rotary positions, timestep rows, and unpacking.
- Shift-12/shift-3 noising and separate loss reductions.
- Official and Comfy state-dict mapping with strict key checks.
- Normal/pruned AdaLN detection and continuous table interpolation.
- ConvRot metadata validation and `.weight_scale` to `.scale_weight` adaptation.
- NVFP4 rejection before allocation.
- LoRA default target discovery and saved metadata.
- H3 validation permits dataset batch sizes greater than one.

### 22.2 Focused integration tests

- Tiny BF16 H3 forward for each packed task layout.
- Tiny normal and pruned ConvRot forward parity against explicit dequantization within defined tolerance.
- Tiny H3 offloader wait/prefetch order.
- One LoRA training forward/backward with gradient checkpointing and block swap.
- One forward-only, multi-step inference with block swap.
- One ConvRot block-swap device/buffer placement check.
- Root entrypoint import/existence tests.

These are not expanded into a three-task `batch_size=2` run or a `batch_size=1/2/3` forward/backward matrix.

### 22.3 Manual acceptance

Run outside CI with published artifacts:

- One official sharded load for each transformer task family.
- One Comfy BF16 generation.
- One normal INT8 ConvRot generation with runtime LoRA.
- One pruned INT8 ConvRot generation with runtime LoRA.
- One text cache using BF16 Qwen3-VL and one using INT8 ConvRot Qwen3-VL.
- One real 33B LoRA forward/backward smoke with block swap.
- One real 33B forward-only generation with block swap and muxed audio/video.

Record commands, hardware, peak VRAM/RAM, packed row count, and output media properties. These manual checks are release evidence, not normal CI gates.

## 23. Acceptance Criteria

The feature is complete when:

- All four H3 commands are usable from root wrappers.
- All three tasks cache their required inputs and execute through native packing.
- Target audio is always supervised and never silently synthesized as silence.
- Official sharded and listed Comfy BF16 artifacts load strictly.
- Normal and pruned Comfy INT8 ConvRot artifacts load and execute.
- The PR 1008 ConvRot foundation is present at the pinned provenance.
- LoRA training works on a frozen H3 base and LoRA inference works on BF16 and prequantized INT8 bases.
- H3 does not force dataset batch size to 1.
- Block swap works in LoRA training and inference, including ConvRot variants.
- Unsupported NVFP4/AWQ fails clearly before expensive allocation.
- Unit and focused integration tests pass.
- The manual real-model smoke evidence is recorded.
- User-facing MiniMax-H3 documentation describes task data, checkpoint choices, ConvRot, block swap, LoRA limitations, and command examples.

## 24. Resolved Tradeoffs

- Native Musubi modules are preferred over importing the draft Diffusers implementation because training, caching, LoRA, compilation, and offload lifecycles are the actual product surface here.
- PR 1008 is merged rather than copied because duplicate ConvRot implementations would immediately diverge in kernel, LoRA, and offloader behavior.
- Exact PR and external-reference SHAs are pinned because moving PR heads are not reproducible dependencies.
- Compatible batching is preserved, but arbitrary ragged batching is excluded because it is a separate scheduling and attention-mask project rather than a requirement for removing a forced batch-size check.
- Runtime LoRA branches are preferred on INT8 bases because physical requantization would trade away adapter signal for a cosmetic merged artifact.
- Real 33B tests stay manual because CI-scale mocks can validate control flow but cannot prove memory feasibility or media quality.
