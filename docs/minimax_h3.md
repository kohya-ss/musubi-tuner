# MiniMax-H3

## Overview

Musubi Tuner supports MiniMax-H3 text-to-video-with-audio (T2VA), first/last-frame-to-video-with-audio (FL2VA), and reference-to-video-with-audio (Ref2VA) LoRA training and standalone generation.

R1 follows the released MiniMax-H3 packing, Qwen3-VL conditioning, dual video/audio flow schedules, and two VAE layouts. It supports the published BF16 FL2VA and Ref2VA transformers. Quantized ConvRot, pruned AdaLN, and quantized text-encoder artifacts are deferred to R2.

Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or using the weights.

## Model Files

Download the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | R1 file |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| Ref2VA transformer | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |

T2VA uses the FL2VA transformer without first/last conditions. R1 rejects the INT8 ConvRot, pruned, FP8, and NVFP4/AWQ files rather than silently interpreting them as BF16.

The Qwen processor/config defaults to `Qwen/Qwen3-VL-32B-Instruct` and is downloaded by Transformers. Pass `--processor` when using a local copy.

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

For Ref2VA, use the Ref2VA BF16 base and an ordered JSONL record:

```text
--task ref2va --dit /models/minimax_h3_ref2va_bf16.safetensors --reference_jsonl /data/h3/ref2va.jsonl --reference_index 0
```

The Ref2VA generation JSONL intentionally uses the same validated schema as training, including target `video_path`, optional target audio, caption, and references. The target media identifies the record but is not used as a generation target. `--prompt` may override the record caption when encoding fresh text conditioning.

T2VA and Ref2VA generation may use `--text_cache` instead of `--text_encoder`. The cache must match the requested task, frame count, and exact presentation fingerprint. T2VA still requires `--prompt` so that identity can be verified; Ref2VA uses the selected record caption unless `--prompt` overrides it. FL2VA generation does not accept a dataset text cache because external first/last images cannot be proven identical to the crop presentation that produced that cache.

The native sampler builds one common base grid, derives independent shifted video and audio sigma grids, and advances each modality with its own finite sigma interval. It does not apply CFG, negate the model heads, or apply ComfyUI's single-sampler audio slope adapter. Musubi also adds condition noise before packing, while ComfyUI adds it after packing; the distributions agree but RNG placement does not. These two intentional differences mean the same seed is not bitwise reproducible against ComfyUI. Video and audio are decoded sequentially, trimmed to a common duration, and muxed with PyAV as H.264 plus AAC.

## R1 Limitations

- BF16 FL2VA/Ref2VA transformer bases only.
- BF16 Qwen3-VL text encoder only.
- No ConvRot INT8, pruned AdaLN, FP8, or NVFP4/AWQ artifact loading.
- No CFG or negative prompt.
- No numbered reference-directory convention.
- Dataset `batch_size` is fixed to 1; use gradient accumulation for larger effective batches.
- No padded multi-sample packed layouts.
