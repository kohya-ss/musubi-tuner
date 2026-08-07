# MiniMax-H3

## Overview

Musubi Tuner supports MiniMax-H3 text-to-video-with-audio (T2VA), first/last-frame-to-video-with-audio (FL2VA), and reference-to-video-with-audio (Ref2VA) LoRA training and standalone generation.

The MiniMax-H3 path follows the released packing, Qwen3-VL conditioning, dual video/audio flow schedules, and two VAE layouts. It supports the published BF16 FL2VA and Ref2VA transformers, plus experimental Comfy-Org quantized artifacts for local LoRA training.

Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or using the weights.

## Model Files

Download the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | File |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| Ref2VA transformer | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |

T2VA uses the FL2VA transformer without first/last conditions.

Experimental quantized checkpoints are also accepted when they use the Comfy-style quant metadata supported by Musubi:

| Component | Experimental file |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |

The quantized transformer is dequantized/materialized to ordinary PyTorch modules during model loading so LoRA training does not dequantize every layer on every forward pass. The quantized text encoder is supported for text-cache creation. Keep these artifacts isolated from upstream Musubi licensing assumptions; the quant loader code is separated in `src/musubi_tuner/minimax_h3/comfy_quant_loader.py` with its own license header.

The Qwen processor/config defaults to `Qwen/Qwen3-VL-32B-Instruct` and is downloaded by Transformers. Pass `--processor` when using a local copy. The text encoder cache format is the same for BF16 and supported quantized text encoders, so training consumes the generated cache normally.

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
source_fps = 24.0
```

For a directory item such as `clip.mp4`, put the caption in `clip.txt`. Target audio is resolved in this order: the JSONL `audio_path` when JSONL is used, exactly one same-stem audio sidecar such as `clip.wav`, then the video's embedded audio stream, then an unsupervised silence placeholder.

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

For `h3_image_frame_count = 22`, provide up to 22 target frames using the same suffix order. If fewer target frames are present, the last target frame is repeated to fill the requested H3 frame count; if more are present, extras are ignored. Changing `multiple_target` or target frame files changes the latent content, so rebuild both latent and text caches without `--skip_existing`.

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
source_fps = 24.0
```

Each JSONL line contains the target plus its ordered references. Relative paths resolve from the JSONL directory.

```json
{"video_path":"targets/clip.mp4","audio_path":"targets/clip.wav","caption":"A singer performs under stage lights.","references":[{"type":"image","path":"refs/style.png"},{"type":"video","path":"refs/motion.mp4","audio_path":"refs/motion.wav"},{"type":"audio","path":"refs/voice.wav"}]}
```

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

By default, caching uses real target audio when available. If a video has no target audio, caching encodes duration-matched silence as the structurally required audio latent and marks that sample's audio loss disabled; missing audio is not treated as a silent supervision target. To avoid flooding large silent datasets, the cache command warns with paths for only the first 10 missing-audio records and then prints one completion summary with policy counts and the supervised fraction. To ignore all target audio intentionally, add `--h3_video_only` to the latent-cache command only. Ref2VA reference audio remains active conditioning.

`--audio_vae` is still required with `--h3_video_only`. H3 always includes target-audio rows, and the released Audio VAE encoding of a zero waveform is not guaranteed to be an all-zero latent. Each cache stores its own small silence latent: at `F=124`, it is about 52 KB versus about 7.16 MB for the BF16 video latent, so no shared-silence or deduplication mechanism is used. Existing real-audio H3 latent caches remain compatible and do not require a blanket rebuild.

## Cache Text Encoder Outputs

```bash
python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config /data/h3/dataset.toml \
  --task t2va \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --text_cache_dtype bf16 \
  --skip_existing
```

The cache stores the state after the first 50 Qwen layers, before a final language-model norm. `hidden_states[0]` is the embedding output, so this is `hidden_states[50]`. The cache also stores per-row modality tags and presentation fingerprints; stale or structurally incompatible caches are rejected.

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

The default LoRA targets only `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`, and `mlp.fc2` in the 50 main DiT blocks. Every sample contributes `mean(video_mse)`. A sample with real target audio additionally contributes `mean(audio_mse)` at equal per-sample coefficient; a missing-audio or video-only sample contributes no audio MSE. With `batch_size=1` and gradient accumulation, the expected run-level audio coefficient is therefore the supervised-audio sample fraction, not always one. This fraction is not a uniform per-step scale: at low values, most optimizer steps receive no audio gradient and occasional steps receive the full audio term. Training does not renormalize by the fraction, because doing so would amplify a small supervised subset.

The trainer logs this value as `supervised_audio_fraction` and saves it as `ss_minimax_h3_supervised_audio_fraction`. It also records `ss_minimax_h3_loss_policy=video_mean_plus_optional_audio_mean` and `ss_minimax_h3_audio_supervision=per_sample_binary_cache_weight`. H3 enforces uniform base-time sampling, no generic SD3 loss weighting, and independent video/audio shifts of 12 and 3.

Zero audio loss does not preserve the base model's audio behavior. H3 is single-stream, and these LoRA targets modify the same attention and MLP weights used by video and audio tokens. A video-only or low-`supervised_audio_fraction` LoRA can therefore produce audio worse than the base model; the risk generally increases with adapter capacity/strength and training exposure, although degradation is not guaranteed to be monotonic. Treat audio from a fully video-only LoRA as unconstrained output.

Block swap supports up to 48 of the 50 main blocks. `--block_swap_h2d_only` is also supported for frozen-base LoRA training and requires `--gradient_checkpointing`.

R1 requires `batch_size = 1` in every H3 dataset. Use Accelerate gradient accumulation for a larger effective batch. The batch-size gate runs immediately after dataset construction and reads no cache files. Fraction validation then uses the cache paths already stored in the constructed batch managers, counts repeats from those entries, and opens each unique cache once; it does not run a second glob or load video/audio latent payloads. The runtime/model repeat the batch-size check for direct API calls. Real packed batching needs text padding, an attention mask, and per-sample structural tensors, so it is deferred to a separate PR.

Saved `ss_minimax_h3_base_family` names the released transformer family, not the task. T2VA therefore records `ss_minimax_h3_task=t2va` and `ss_minimax_h3_base_family=fl2va`, because T2VA uses the released FL2VA base.

### Training-time joint AV samples

H3 overrides the shared single-VAE/video-only sampling hook. It samples with the live transformer and current LoRA, decodes the video and audio latents with their own VAEs in sequence, and writes a muxed MP4 under `OUTPUT_DIR/sample`.

Add the sampling assets and normal sampling schedule flags to the training command:

```text
--sample_prompts /data/h3/sample_prompts.json \
--sample_every_n_epochs 1 \
--video_vae /models/minimax_h3_video_vae_fp16.safetensors \
--audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
--text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors
```

The text presentations and condition latents are prepared once before the transformer is loaded. The two decode VAEs then remain on CPU and are moved to the accelerator one at a time for each scheduled sample. The shared trainer still owns sampling cadence, distributed prompt assignment, RNG restoration, and the block-swap inference/training transition.

R1 prepares sample prompts by loading the released Qwen3-VL text encoder on the training accelerator. The BF16 artifact is approximately 48 GB, so `--sample_prompts` currently requires roughly 50 GB of available accelerator memory before the transformer is loaded. Omit scheduled sampling on smaller accelerators; a text-encoder device override or cached sample conditioning is follow-up scope.

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

## Generation

T2VA generation with the FL2VA BF16 base:

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

For Ref2VA, use the Ref2VA BF16 base and an ordered JSONL record:

```text
--task ref2va --dit /models/minimax_h3_ref2va_bf16.safetensors --reference_jsonl /data/h3/ref2va.jsonl --reference_index 0
```

The Ref2VA generation JSONL intentionally uses the same validated schema as training, including target `video_path`, optional target audio, caption, and references. The target media identifies the record but is not used as a generation target. `--prompt` may override the record caption when encoding fresh text conditioning.

T2VA and Ref2VA generation may use `--text_cache` instead of `--text_encoder`. The cache must match the requested task, frame count, and exact presentation fingerprint. T2VA still requires `--prompt` so that identity can be verified; Ref2VA uses the selected record caption unless `--prompt` overrides it. FL2VA generation does not accept a dataset text cache because external first/last images cannot be proven identical to the crop presentation that produced that cache.

The native sampler builds one common base grid, derives independent shifted video and audio sigma grids, and advances each modality with its own finite sigma interval. It does not apply CFG, negate the model heads, or apply ComfyUI's single-sampler audio slope adapter. Musubi also adds condition noise before packing, while ComfyUI adds it after packing; the distributions agree but RNG placement does not. These two intentional differences mean the same seed is not bitwise reproducible against ComfyUI. Video and audio are decoded sequentially, trimmed to a common duration, and muxed with PyAV as H.264 plus AAC.

## Limitations

- BF16 FL2VA/Ref2VA transformer bases are the reference path. The Comfy-Org FL2VA INT8 ConvRot/pruned transformer is supported experimentally for LoRA training.
- BF16 Qwen3-VL text encoder is the reference path. Comfy-Org NVFP4/AWQ and INT8 ConvRot text encoders are supported experimentally for text-cache creation.
- Other FP8/quantized formats outside the supported Comfy metadata are not supported.
- No CFG or negative prompt.
- No numbered reference-directory convention.
- Dataset `batch_size` is fixed to 1; use gradient accumulation for larger effective batches.
- No padded multi-sample packed layouts.
- Image training uses dummy/unsupervised target audio. It can affect audio behavior because MiniMax-H3 is a joint video/audio transformer; image-only LoRAs should be treated as unconstrained for audio output.
