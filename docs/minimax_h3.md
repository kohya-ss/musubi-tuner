# MiniMax-H3

## Overview

Musubi Tuner supports MiniMax-H3 text-to-video-with-audio (T2VA), first/last-frame-to-video-with-audio (FL2VA), and reference-to-video-with-audio (Ref2VA) LoRA training and standalone generation.

The implementation follows the released MiniMax-H3 packing, Qwen3-VL conditioning, dual video/audio flow schedules, and two VAE layouts. It supports the published BF16 transformers, the full and pruned ConvRot INT8 transformers, and the ConvRot INT8 and NVFP4+AWQ Qwen3-VL text encoders.

Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or using the weights.

## Model Files

Download the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | Supported file |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| FL2VA and T2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` |
| FL2VA and T2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Ref2VA transformer | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` |
| Ref2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` |
| Ref2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Qwen3-VL-32B ConvRot INT8 text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Qwen3-VL-32B NVFP4+AWQ text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |

T2VA uses an FL2VA transformer without first/last conditions. Pre-quantized files (ConvRot INT8 full or pruned, transformer or text encoder; NVFP4+AWQ text encoder) are detected automatically from their tensor structure — no extra flag is needed. FP8, NVFP4 transformers, and malformed or partial quantized files are rejected rather than silently interpreted as BF16. See [ConvRot INT8 Quantized Base Weights](#convrot-int8-quantized-base-weights) and [NVFP4 Text Encoder](#nvfp4-text-encoder) for details.

The Qwen3-VL processor and config are downloaded by Transformers from the official [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) repository (`processor` and `text_encoder` subfolders, a few config and tokenizer files only, no weights). The upstream `Qwen/Qwen3-VL-32B-Instruct` files are not interchangeable: the H3 tokenizer adds `<d>`, `</d>`, `<|cutoff|>`, `<|lyrics_start|>`, `<|lyrics_end|>`, `<|caption_start|>`, and `<|caption_end|>` as special tokens, and the released prompt format writes dialogue and lyrics as `<d>[Language] ...</d>`.

## Implementation Provenance

The transformer, video VAE, packed-sequence logic, text presentation, and dual scheduler are adapted from Apache-2.0 [Diffusers PR #14355](https://github.com/huggingface/diffusers/pull/14355), pinned at commit `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc`. Source files retain their upstream copyright and license headers. ComfyUI is used only as an independent numerical and artifact-compatibility reference; its GPL-3.0 implementation is not a source for Musubi code. Model weights remain governed by the MiniMax-H3 Community License linked above.

## Geometry And Media Contract

- Target video is 24 fps.
- Width and height must be positive multiples of 32.
- Frame count must be `17*n+5`.
- The released duration range is 5 to 15 seconds. At 24 fps, the valid released frame counts run from 124 through 345 in steps of 17.
- Real target audio is optional. When present, it is decoded as stereo 32000 Hz audio from the target video, JSONL `audio_path`, or one same-stem sidecar. When absent, the cache stores an unsupervised Audio-VAE silence placeholder so the released packed layout remains intact.
- Ref2VA uses ordered JSONL references only. Numbered reference directories are not supported.
- Expanded Qwen conditioning is limited to 32768 rows. A BF16 cache at the limit is approximately 320 MiB for one sample.

`--allow_experimental_duration` bypasses only the released 5-to-15-second check. It does not bypass frame geometry, reference limits, or validation of an explicitly selected audio source.

## Dataset Configuration

T2VA and FL2VA accept ordinary video directories. FL2VA derives its first and last conditions from each selected target crop.

```toml
[general]
resolution = [768, 1344]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "/data/h3/videos"
cache_directory = "/data/h3/cache"
caption_extension = ".txt"
target_frames = [124]
frame_extraction = "head"
```

H3 always normalizes source videos to 24 fps using frame timestamps, so `source_fps` is not needed and is ignored if set.

For a directory item such as `clip.mp4`, put the caption in `clip.txt`. Target audio is resolved in this order: the JSONL `audio_path` when JSONL is used, exactly one same-stem audio sidecar such as `clip.wav`, then the video's embedded audio stream, then an unsupervised silence placeholder. Audio sources are resolved and validated when the dataset is constructed, so a broken explicit `audio_path` fails before any caching work starts.

### FL2VA Image Training Modes

MiniMax-H3 can also cache still-image training samples on the FL2VA base. These modes are intended for image-edit-like LoRA training, following the same H3-compatible frame grid used by short MiniMax-H3 single-frame workflows.

Use `image_directory` for the output/target image and `control_directory` for the input condition image(s). Captions live next to the target images. Audio is always encoded as an unsupervised silence placeholder, so these LoRAs are video/image-supervised only.

```toml
[general]
resolution = [768, 1344]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/data/h3-image/targets"
control_directory = "/data/h3-image/inputs"
cache_directory = "/data/h3-image/cache"
caption_extension = ".txt"
h3_image_frame_count = 5
```

File names are not fixed. For a target such as `pose front.png`, put the caption at `pose front.txt` and the controls at `pose front_0.png`, `pose front_1.png`. The same rule works for any basename; only the shared basename and numeric control suffix matter.

Input image sizes may be mixed. With `enable_bucket = true`, target and control images are resized into a matching H3 bucket automatically, and H3 buckets are aligned to 32-pixel steps. `resolution` is the maximum bucket area, not a required exact source size.

Text encoder caching also caps large H3 image-mode visuals before they are passed to Qwen3-VL. The default `--h3_text_visual_max_pixels 1048576` keeps the aspect ratio, rounds width/height down to 32-pixel multiples, and affects only the text encoder's temporary visual input; original files and latent-cache images are not rewritten. Use a smaller value such as `786432` for faster text caching, or `0` to disable this cap. Changing this value changes the text-cache metadata, so stale caches are rebuilt even with `--skip_existing`.

The two supported still-image conditioning modes are:

| Mode | Control files per sample | Target files per sample | Meaning |
| --- | ---: | ---: | --- |
| `first` | 1 | 1 or more | One input image is used as both FL2VA first and last conditions. |
| `first_last` | 2 | 1 or more | First and end condition images are supplied separately. |

For one-input/one-output training, place one matching control image for each target and cache with `--h3_image_mode first`. The same input latent is written as both FL2VA first and last conditions:

```bash
python minimax_h3_cache_latents.py \
  --dataset_config /data/h3-image/dataset.toml \
  --task fl2va \
  --h3_image_mode first \
  --h3_image_frame_count 5 \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --skip_existing
```

For two-input/one-output training, place two matching control images for each target and cache with `--h3_image_mode first_last`. They become the FL2VA first and last conditions:

```bash
python minimax_h3_cache_latents.py \
  --dataset_config /data/h3-image/dataset.toml \
  --task fl2va \
  --h3_image_mode first_last \
  --h3_image_frame_count 5 \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --skip_existing
```

Then cache text encoder outputs with the same image mode and frame count. The text encoder may be the BF16 file or a supported quantized Comfy-Org text encoder:

```bash
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config /data/h3-image/dataset.toml \
  --task fl2va \
  --h3_image_mode first \
  --h3_image_frame_count 5 \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --h3_text_visual_max_pixels 1048576 \
  --text_cache_dtype bf16 \
  --skip_existing
```

`h3_image_frame_count` may be set in each image dataset block, so different image datasets can use different H3-compatible temporal lengths. The value must satisfy `frame_count = 17 * k + 5` with integer `k >= 0`; the minimum is `5`, and the next valid value is `22`. For still-image LoRA training, keep `5` unless you explicitly want the target image repeated across a longer H3-compatible frame grid. The CLI option `--h3_image_frame_count` overrides the TOML value for all image datasets. If source, condition images, or frame count change, rebuild caches without `--skip_existing`.

To train against multiple output frames from the same input condition(s), enable the existing image-dataset `multiple_target` option. Two filename layouts are supported:

- Unsuffixed base frame: `pose.png`, `pose_1.png`, `pose_2.png`, ...
- Zero-indexed base frame: `pose_0.png`, `pose_1.png`, `pose_2.png`, ...

The zero-indexed layout is useful when every output frame should have an explicit frame index. In that layout, keep captions and one-image controls unsuffixed: `pose.txt` and `control/pose.png` are matched to `image/pose_0.png`.

```toml
[[datasets]]
image_directory = "/data/h3-image/targets"
control_directory = "/data/h3-image/inputs"
cache_directory = "/data/h3-image/cache"
caption_extension = ".txt"
h3_image_frame_count = 5
multiple_target = true
```

Example for a 5-frame target sequence with an unsuffixed base frame:

```text
targets/pose.png
targets/pose_1.png
targets/pose_2.png
targets/pose_3.png
targets/pose_4.png
targets/pose.txt
```

Example for a 5-frame target sequence with a zero-indexed base frame:

```text
targets/pose_0.png
targets/pose_1.png
targets/pose_2.png
targets/pose_3.png
targets/pose_4.png
targets/pose.txt
```

For `first`, use one control image:

```text
inputs/pose.png
```

For `first_last`, use two control images:

```text
inputs/pose_0.png
inputs/pose_1.png
```

For `h3_image_frame_count = 22`, provide up to 22 target frames using the same suffix order. If the number of target frames differs from the requested H3 frame count, the provided frames are resampled across the requested timeline with nearest-neighbor frame selection. For example, five keyed target images can be expanded to 22 H3 frames while preserving the first and last frames. Changing `multiple_target`, target frame files, or `h3_image_frame_count` changes the latent content, so rebuild both latent and text caches without `--skip_existing`.

Windows example for one-input image training:

```bat
python minimax_h3_cache_latents.py ^
  --dataset_config "E:\AI\dataset\h3_image.toml" ^
  --task fl2va ^
  --h3_image_mode first ^
  --video_vae "C:\ComfyUI\models\vae\minimax_h3_video_vae_fp16.safetensors" ^
  --audio_vae "C:\ComfyUI\models\vae\minimax_h3_audio_vae_fp32.safetensors"

python minimax_h3_cache_text_encoder_outputs.py ^
  --dataset_config "E:\AI\dataset\h3_image.toml" ^
  --task fl2va ^
  --h3_image_mode first ^
  --text_encoder "C:\ComfyUI\models\text_encoders\qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors" ^
  --h3_text_visual_max_pixels 1048576 ^
  --text_cache_dtype bf16
```

Ref2VA requires `video_jsonl_file`:

```toml
[general]
resolution = [768, 1344]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_jsonl_file = "/data/h3/ref2va.jsonl"
cache_directory = "/data/h3/cache-ref2va"
target_frames = [124]
frame_extraction = "head"
```

Each JSONL line contains the target plus its ordered references. Relative paths resolve from the JSONL directory.

```json
{"video_path":"targets/clip.mp4","audio_path":"targets/clip.wav","caption":"A singer performs under stage lights.","references":[{"type":"image","path":"refs/style.png"},{"type":"video","path":"refs/motion.mp4","audio_path":"refs/motion.wav"},{"type":"audio","path":"refs/voice.wav"}]}
```

Audio for a `video` reference resolves in this order: an explicit `audio_path` file, then the video's embedded audio track. Writing `"audio_path": null` disables audio for that reference: the video conditions visuals only (for example a motion or composition reference) even when the file contains an audio track, and it does not count as audio-bearing. A reference video without any audio track is likewise a visual-only reference; the official prompt guide treats reference-video audio as an explicitly enabled track, so silent reference videos are a normal input. `audio_path` is valid only on `video` references.

Limits per Ref2VA record:

- At most 12 references total.
- At most 9 image references.
- At most 3 video references.
- At most 3 audio-bearing references, counting standalone audio and video with audio together.
- At least one image or video reference.
- Reference videos must be 2 to 15 seconds.

## Cache Latents

Use the same authoritative `--task` for caching and training.

```bash
python minimax_h3_cache_latents.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --cache_seed 42 \
  --skip_existing
```

The video VAE is upcast to FP32 for target and condition encoding so cached training targets do not inherit FP16 encoder outliers. It uses a reproducible posterior sample for each target. Visual conditions use the released fixed sampling policy, including the required FP16 round-trip of the sampled condition latent before normalization. Video decode keeps the released FP16 artifact in FP16. Target and reference audio use the audio posterior mode directly in `[32,2,A]` layout.

Caching always uses real target audio when available. If a video has no target audio, caching encodes duration-matched silence as the structurally required audio latent and records `audio_present=0`; missing audio is never treated as a silent supervision target, and such samples are automatically excluded from audio supervision during training. To avoid flooding large silent datasets, dataset construction warns with paths for only the first 10 missing-audio records, and the cache command prints one completion summary with the supervised fraction.

The cache stores only this fact about the data. Whether and how strongly audio is supervised is decided at training time with `--video_only` and `--audio_loss_weight` (see LoRA Training below); there is no cache-time video-only mode.

`--audio_vae` is always required. H3 always includes target-audio rows, and the released Audio VAE encoding of a zero waveform is not guaranteed to be an all-zero latent. Each cache stores its own small silence latent: at `F=124`, it is about 52 KB versus about 7.16 MB for the BF16 video latent, so no shared-silence or deduplication mechanism is used.

`--skip_existing` compares the stored cache metadata (task, cache seed, crop start, cache format version, and fingerprints of the media files and VAE checkpoints) and rebuilds any cache that no longer matches. Fingerprints are lightweight file identities (size + mtime), not content hashes: re-copying or re-downloading a file changes its identity and triggers a one-time re-cache.

Latent caches created before the `audio_present` contract (releases with `target_audio_policy` metadata) are not compatible; re-run latent caching. Caches written with the earlier metadata format remain trainable but are treated as stale by `--skip_existing` and rebuilt once.

## Cache Text Encoder Outputs

```bash
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --text_cache_dtype bf16 \
  --skip_existing
```

The same command accepts the ConvRot INT8 text encoder (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`) and the NVFP4+AWQ text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`); the formats are detected automatically. On VRAM-limited GPUs add `--text_encoder_blocks_to_swap 50` to stream the encoder layers from CPU, and `--text_encoder_attn_mode flash_attention_2` for long Ref2VA presentations (see Text Encoder Layer Streaming below). The cache stores the state after the first 50 Qwen layers, before a final language-model norm. `hidden_states[0]` is the embedding output, so this is `hidden_states[50]`. The cache also stores per-row modality tags and presentation fingerprints; stale or structurally incompatible caches are rejected.

## LoRA Training

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 minimax_h3_train_network.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --network_module networks.lora_minimax_h3 \
  --network_dim 16 \
  --network_alpha 16 \
  --sdpa \
  --mixed_precision bf16 \
  --gradient_checkpointing \
  --blocks_to_swap 48 \
  --optimizer_type adamw8bit \
  --learning_rate 1e-4 \
  --max_train_epochs 16 \
  --save_every_n_epochs 1 \
  --output_dir /data/h3/output \
  --output_name h3-lora
```

On Windows with the Comfy-Org quantized FL2VA transformer, the command is the same except for paths and the attention backend:

```bat
accelerate launch ^
--num_cpu_threads_per_process 1 ^
--mixed_precision bf16 ^
minimax_h3_train_network.py ^
--dataset_config "E:\AI\dataset\h3_image.toml" ^
--task fl2va ^
--dit "C:\ComfyUI\models\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" ^
--network_module networks.lora_minimax_h3 ^
--network_dim 16 ^
--network_alpha 16 ^
--mixed_precision bf16 ^
--gradient_checkpointing ^
--blocks_to_swap 48 ^
--xformers ^
--optimizer_type adamw8bit ^
--learning_rate 1e-4 ^
--max_train_epochs 20 ^
--save_every_n_epochs 1 ^
--output_dir "E:\AI\dataset\output" ^
--output_name h3-lora
```

One of `--sdpa`, `--xformers`, `--flash-attn`, `--flash3`, or `--sage-attn` must be selected. For training, `--sdpa` is the safest baseline. `--xformers` can reduce memory or improve speed when a working xFormers build is available. On Blackwell/RTX 50-series Windows systems, xFormers may require a local build and a backend that accepts compute capability 12.0; if xFormers reports a dtype mismatch, update to a version of this branch that aligns Q/K/V dtypes before dispatch.

The default LoRA targets only `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2` in the 50 main DiT blocks. Every sample contributes `mean(video_mse)`. A sample cached with real target audio additionally contributes `audio_loss_weight * mean(audio_mse)`; a sample cached from missing audio (`audio_present=0`) never contributes audio MSE. With `batch_size=1` and gradient accumulation, the expected run-level audio coefficient is therefore `audio_loss_weight` times the supervised-audio sample fraction. This fraction is not a uniform per-step scale: at low values, most optimizer steps receive no audio gradient and occasional steps receive the full audio term. Training does not renormalize by the fraction, because doing so would amplify a small supervised subset.

Two training arguments control audio supervision:

- `--video_only` disables audio supervision entirely (audio loss weight 0 for all samples). The model still attends to the real audio latents as context, which matches the inference-time distribution where audio tokens are always generated audio, never silence.
- `--audio_loss_weight` (default 1.0) scales the audio loss term for supervised samples, e.g. to rebalance a small audio loss against the video loss.

The latent caching script logs the supervised fraction as `supervised_audio_fraction` in its end-of-run summary, and warns when no cached item has real audio. The trainer records the fraction it actually observed during training as `ss_minimax_h3_supervised_audio_fraction` (exact once a full epoch has run), along with `ss_minimax_h3_audio_loss_weight` and `ss_minimax_h3_video_only`, and warns at the end of the first epoch if audio supervision is enabled but no sample with real audio was seen. It also records `ss_minimax_h3_loss_policy=video_mean_plus_weighted_audio_mean` and `ss_minimax_h3_audio_supervision=presence_gated_training_weight`. H3 enforces uniform base-time sampling, no generic SD3 loss weighting, and independent video/audio shifts of 12 and 3. This mirrors the released inference schedule (and the ai-toolkit trainer): one base time is drawn uniformly and both per-stream sigmas are derived from it, so video and audio always sit on the same `(sigma_video, sigma_audio)` curve the sampler visits.

`--min_timestep` and `--max_timestep` clip the shared base variable, in base units where 1000 is pure noise, before the two per-stream shifts are applied. Clipping in base space keeps the video and audio streams consistent; the bounds are not sigma values of either stream. For example `--max_timestep 900` removes the highest-noise 10% of the base range, which corresponds to `sigma_video > 0.9908` (shift 12) and `sigma_audio > 0.9643` (shift 3).

Zero audio loss does not preserve the base model's audio behavior. H3 is single-stream, and these LoRA targets modify the same attention and MLP weights used by video and audio tokens. A `--video_only` or low-`supervised_audio_fraction` LoRA can therefore produce audio worse than the base model; the risk generally increases with adapter capacity/strength and training exposure, although degradation is not guaranteed to be monotonic. Treat audio from a fully video-only LoRA as unconstrained output.

Block swap supports up to 48 of the 50 main blocks. `--block_swap_h2d_only` is also supported for frozen-base LoRA training and requires `--gradient_checkpointing`.

MiniMax-H3 requires `batch_size = 1` in every H3 dataset. Use Accelerate gradient accumulation for a larger effective batch. The latent caching script warns when a dataset config sets any other value, and the trainer rejects the first batch whose size is not 1. Real packed batching needs text padding, an attention mask, and per-sample structural tensors, so it is deferred to a separate PR.

Saved `ss_minimax_h3_base_family` names the released transformer family, not the task. T2VA therefore records `ss_minimax_h3_task=t2va` and `ss_minimax_h3_base_family=fl2va`, because T2VA uses the released FL2VA base.

### Training-time joint AV samples

H3 overrides the shared `prepare_sampling` hook (whose default covers single-VAE architectures) and returns both VAEs as its sampling resources. It samples with the live transformer and current LoRA, decodes the video and audio latents with their own VAEs in sequence, and writes a muxed MP4 under `OUTPUT_DIR/sample`.

Add the sampling assets and normal sampling schedule flags to the training command:

```text
--sample_prompts /data/h3/sample_prompts.json \
--sample_every_n_epochs 1 \
--video_vae /models/minimax_h3_video_vae_fp16.safetensors \
--audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
--text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors
```

The text presentations and condition latents are prepared once before the transformer is loaded. The two decode VAEs then remain on CPU and are moved to the accelerator one at a time for each scheduled sample. The shared trainer still owns sampling cadence, distributed prompt assignment, RNG restoration, and the block-swap inference/training transition.

Training-time samples load the selected Qwen3-VL text encoder on the training accelerator before the transformer. The BF16 artifact is approximately 48 GB, so `--sample_prompts` requires roughly 50 GB of available accelerator memory there; the ConvRot INT8 artifact lowers the persistent text-encoder weights to ~25 GB and the NVFP4+AWQ artifact to ~15 GB, selected simply by passing their paths. `--text_encoder_blocks_to_swap` (see Text Encoder Layer Streaming below) removes most of the remaining weight footprint by streaming the encoder layers from CPU during this phase.

All entries in one run use the training `--task`. T2VA JSON entries use the common prompt fields:

```json
[
  {
    "prompt": "A singer performs under stage lights.",
    "width": 768,
    "height": 1344,
    "frame_count": 124,
    "sample_steps": 30,
    "seed": 42
  }
]
```

FL2VA entries additionally use `first_frame` and `last_frame`; the common `image_path` and `end_image_path` names are accepted as aliases. Ref2VA entries use `reference_jsonl`, optional `reference_index`, and an optional `prompt` override. Ref2VA keeps the same ordered JSONL schema as caching and standalone generation.

Sample geometry must be 32-pixel aligned. Frame counts of at least 5 are rounded down to the nearest `17*n+5` value, matching the shared training-sample convention. Released durations are 5-15 seconds; `--h3_allow_experimental_sample_duration` permits shorter smoke samples. H3 sampling does not accept negative prompts, CFG, or a per-prompt generic flow shift.

## ConvRot INT8 Quantized Base Weights

MiniMax-H3 supports ConvRot INT8 ([arXiv:2512.03673](https://arxiv.org/abs/2512.03673)) frozen base weights for both LoRA training and generation, the same scheme as Krea 2 (see `docs/krea2.md` for the mechanism and backward modes). Two base artifacts are accepted:

- **ComfyUI pre-quantized ConvRot INT8 checkpoints** (`weight` int8 + `weight_scale` + `comfy_quant` tensors) are detected automatically from their tensor structure — pass them as `--dit` and no extra flag is needed. The tensors are converted to the Musubi layout during the streaming load.
- **BF16 checkpoints** are quantized on the fly at load time when `--convrot_int8` is passed.

Both routes produce bit-identical models: Musubi's dynamic quantization reproduces the published ComfyUI INT8 ConvRot distribution exactly, layer by layer.

The published quantization scope is the five Linears in each of the 50 main DiT blocks (`attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, `mlp.fc2`, and `adaln_proj.linear`). `adaln_proj` uses ConvRot group size 64 (its input width 2688 is not a multiple of 256); the rest use 256. The token refiner, final layer, embedders, and heads stay BF16/FP32. The base checkpoint shrinks from ~66 GB (BF16) to ~34 GB of weights. For pre-quantized files the checkpoint itself dictates the quantized set: the per-layer `comfy_quant` specs are validated strictly (malformed or partial triples are rejected), while artifacts that quantize a different layer set than the published scope load as declared.

**Pruned transformers.** The released pruned ConvRot INT8 artifacts additionally replace the sinusoidal time embedder with a published FP32 `[1025, 8]` AdaLN curve table (`adaln_t_table`) and 8-wide AdaLN projections. They are recognized structurally and interpolated in FP32 over the model time `t = 1 - sigma` in `[0, 1]`. Pruned BF16 files are not published and not supported.

**Text encoder.** `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` is likewise detected automatically wherever `--text_encoder` is accepted (TE caching, training-time sampling, generation), lowering the text-encoder weight footprint from ~48 GB to ~25 GB.

**Training.** Flags match Krea 2: `--convrot_int8` for BF16 sources (pre-quantized files need no flag) plus optional `--convrot_int8_bwd {bf16,int8}` (default `bf16`; `int8` requires triton and CUDA). The LoRA trains in BF16 on top of the int8 base as usual, and block swap (including `--block_swap_h2d_only`) combines with quantization — quantization runs on the accelerator while the weights load to CPU, and the FP32 scale buffers stay resident on the execution device. Because block-swap training is transfer-bound, halving the weight bytes roughly halves the step time (measured: classic swap 48 27 -> 12.7 s/it, `--block_swap_h2d_only` + 32 8 -> ~4 s/it on the same GPU). `--fp8_base`/`--fp8_scaled` and `--base_weights` remain unsupported for an INT8 base. Triton (`triton-windows` on Windows) is required for the fused int8 kernels; without it the forward falls back to a slower transient dequantization (the memory saving remains). `torch.compile` excludes the patched Linears automatically.

**Generation.** Pre-quantized checkpoints work as-is; add `--convrot_int8` only to quantize a BF16 checkpoint at load time. With `--lora_weight` the route depends on the base: a BF16 base with `--convrot_int8` merges the LoRA into the BF16 weights during the streaming load and quantizes the merged result (fastest inference); a pre-quantized base attaches each LoRA as a runtime additive branch with its own multiplier for the sampling lifetime — the INT8 base tensors are never modified or requantized, so LoRA generation no longer requires downloading the BF16 checkpoint.

## NVFP4 Text Encoder

The published NVFP4+AWQ Qwen3-VL text encoder (`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, ~14.6 GB) is accepted wherever `--text_encoder` is accepted (TE caching, training-time sampling, generation) and is detected automatically from its tensor structure, like the ConvRot INT8 artifacts.

The artifact quantizes the 350 language-model Linears to NVFP4 (4-bit E2M1 values with FP8 per-16-block scales and an FP32 per-tensor scale) and the token embedding to per-row INT8; norms, biases, and the vision tower stay BF16. The quantization is AWQ-calibrated: two projections per layer carry an explicit `pre_quant_scale` that Musubi multiplies into their inputs at runtime, and the remaining scales are already folded into the checkpoint's norm weights. Because AWQ requires calibration data, there is deliberately no on-the-fly NVFP4 quantization of BF16 weights — for dynamic quantization use ConvRot INT8, which reproduces the published INT8 artifact bit-exactly.

By default the patched layers run weight-only: the NVFP4 weight is transiently dequantized each forward and multiplied in BF16, which works on any GPU and matches the artifact's own `full_precision_matrix_mult` declaration. `--nvfp4_scaled_mm` opts into W4A4 matmuls via `torch.nn.functional.scaled_mm`, which also quantizes the activations to NVFP4; it requires PyTorch 2.10+ and a Blackwell-generation GPU, and trades some quality for speed (measured end-to-end against a BF16 text encoder with an identical DiT/seed: mean 5.9/255 decoded deviation in the default mode, 7.2/255 with `--nvfp4_scaled_mm`; the ConvRot INT8 text encoder scores 3.5/255 and a typical LoRA effect ~79/255 on the same pipeline).

The text encoder is frozen in every Musubi flow, so the NVFP4 path is inference-only; it does not affect LoRA training of the transformer. Text-encoder LoRAs cannot be merged into or attached to the NVFP4 artifact.

## Text Encoder Layer Streaming

`--text_encoder_blocks_to_swap N` streams `N` of the 50 Qwen3-VL decoder layers from CPU memory instead of keeping them resident on the GPU, wherever `--text_encoder` is accepted (TE caching, training-time sampling, generation). `N=50` minimizes the device footprint; smaller values keep `50-N` layers resident and transfer proportionally less per record (useful when the encoder almost fits). Requires CUDA.

The text encoder is frozen and forward-only in every Musubi flow, so this uses the H2D-only streaming machinery from `docs/block_swap.md`: the layer weights stay in pageable CPU masters (no large pinned allocation) and are prefetched layer by layer into a small ring of two reused GPU buffers while earlier layers compute; nothing is ever copied back. Quantized layers stream their weights together with their scale tensors, so the mechanism combines with every accepted artifact. The computed values are identical to a fully resident run — the same weights are read from the same bytes, only their location changes.

With `--text_encoder_blocks_to_swap 50` the resident weights reduce to the embedding, the vision tower, and the norms, plus two ring buffers of one layer each (per-layer stream size: ~0.9 GB BF16, ~0.5 GB ConvRot INT8, ~0.3 GB NVFP4) and activations. This brings TE caching and training-time sampling with the quantized artifacts into reach of consumer GPUs; the trade-off is the CPU-side resident copy (the artifact size) and per-record transfer time of the streamed layers.

`--text_encoder_attn_mode {sdpa,flash_attention_2,eager}` separately selects the transformers attention implementation for the text encoder (default: transformers' own default, sdpa). At long context, transformers' sdpa can fall back to the O(L^2) FP32 math kernel — around 12k rows the attention workspace alone exceeds 30 GB, defeating the streaming savings. Pass `flash_attention_2` (requires flash-attn) for long Ref2VA presentations of more than a few thousand rows.

## Generation

T2VA generation with the FL2VA base:

```bash
python minimax_h3_generate_video.py \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --prompt "A singer performs under stage lights." \
  --width 768 \
  --height 1344 \
  --frame_count 124 \
  --steps 30 \
  --seed 42 \
  --blocks_to_swap 48 \
  --output output.mp4
```

Add a trained LoRA with:

```text
--lora_weight /data/h3/output/h3-lora.safetensors --lora_multiplier 1.0
```

The same command accepts the full or pruned ConvRot INT8 transformer and the ConvRot INT8 or NVFP4+AWQ text encoder; formats are detected automatically. With a BF16 transformer, LoRAs are merged destructively once after loading (fastest inference); with a ConvRot INT8 base, each `--lora_weight` stays a separate runtime additive branch with its corresponding multiplier (see [ConvRot INT8 Quantized Base Weights](#convrot-int8-quantized-base-weights)).

For FL2VA, keep the FL2VA base and replace the task inputs:

```text
--task fl2va --prompt "..." --first_frame first.png --last_frame last.png
```

For still-image inference with an image-trained FL2VA LoRA, use the same image mode used for caching/training. `first` takes one input image and internally uses it as both FL2VA conditions; `first_last` takes separate first and last condition images. PNG/JPEG/WebP outputs decode only video and save one selected frame, so the generated audio latent is discarded.

```bash
python minimax_h3_generate_video.py \
  --task fl2va \
  --h3_image_mode first \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --lora_weight /data/h3-image/output/h3-image-lora.safetensors \
  --prompt "..." \
  --first_frame input.png \
  --width 768 \
  --height 1344 \
  --steps 30 \
  --seed 42 \
  --blocks_to_swap 48 \
  --output output.png
```

For two-condition still-image inference:

```text
--h3_image_mode first_last --first_frame first.png --last_frame end.png --output output.png
```

`--frame_count` defaults to `5` in image mode. Increase it only if the LoRA was trained with a longer H3-compatible image frame count. `--h3_select_frame` selects which decoded frame is saved for image outputs and defaults to `0`.

For Ref2VA, use a Ref2VA base (BF16 or ConvRot INT8) and an ordered JSONL record:

```text
--task ref2va --dit /models/minimax_h3_ref2va_bf16.safetensors --reference_jsonl /data/h3/ref2va.jsonl --reference_index 0
```

The Ref2VA generation JSONL intentionally uses the same validated schema as training, including target `video_path`, optional target audio, caption, and references. The target media identifies the record but is not used as a generation target. `--prompt` may override the record caption when encoding fresh text conditioning.

T2VA and Ref2VA generation may use `--text_cache` instead of `--text_encoder`. The cache must match the requested task, cache format version, and exact presentation fingerprint (which covers the prompt, frame count, and size+mtime identities of the reference media, so the cache must be used on the machine holding the original files). T2VA still requires `--prompt` so that identity can be verified; Ref2VA uses the selected record caption unless `--prompt` overrides it. FL2VA generation does not accept a dataset text cache because external first/last images cannot be proven identical to the crop presentation that produced that cache.

`--steps N` means N model evaluations, so the schedule uses N+1 grid points. The released implementations (SGLang serving and the diffusers scheduler) instead count grid points: their `num_inference_steps = N` performs N-1 evaluations. Musubi `--steps N` is therefore grid-identical to official `num_inference_steps = N+1`; to reproduce the official 50-step serving default exactly, pass `--steps 49`.

The native sampler builds one common base grid, derives independent shifted video and audio sigma grids, and advances each modality with its own finite sigma interval. It does not apply CFG, negate the model heads, or apply ComfyUI's single-sampler audio slope adapter. Musubi also adds condition noise before packing, while ComfyUI adds it after packing; the distributions agree but RNG placement does not. These two intentional differences mean the same seed is not bitwise reproducible against ComfyUI. Video and audio are decoded sequentially, trimmed to a common duration, and muxed with PyAV as H.264 plus AAC.

## Limitations

- Released BF16 full and ConvRot INT8 full/pruned FL2VA/Ref2VA transformer bases only; pruned BF16 files are not published and not supported.
- BF16, ConvRot INT8, or NVFP4+AWQ Qwen3-VL text encoder only.
- No FP8 artifact loading, and no NVFP4 transformer loading.
- No CFG or negative prompt.
- No numbered reference-directory convention.
- Dataset `batch_size` is fixed to 1; use gradient accumulation for larger effective batches.
- No padded multi-sample packed layouts.
- Image training uses dummy/unsupervised target audio. It can affect audio behavior because MiniMax-H3 is a joint video/audio transformer; image-only LoRAs should be treated as unconstrained for audio output.
