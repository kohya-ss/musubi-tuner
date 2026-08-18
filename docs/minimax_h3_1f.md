# MiniMax-H3 One-Frame (Image) Generation

> [!WARNING]
> This mode is **experimental**. The released MiniMax-H3 checkpoints were trained on 5-15 second videos; one-frame generation drives them with a single-token target (`T_lat=1`), which is outside the release distribution but works well in practice: plain one-frame T2VA produces high-quality photographic and illustrated images with the FL2VA base, and Ref2VA with a single image reference generates novel views of the referenced subject. See `docs/minimax_h3.md` for the shared setup (models, quantization, block swap, text-encoder streaming).

## Overview

`--frame_count 1` switches `minimax_h3_generate_video.py` into one-frame mode:

- The target is one video latent token plus the two audio latent frames the joint layout requires. The audio is a byproduct and is never decoded; the output is a PNG (`--output` must use `.png`).
- The single-token VAE decode duplicates the latent to a pseudo two-token clip and keeps pixel frame 0 (a solo token decode breaks down; the duplication decodes within ~1-2 dB of a true two-token decode). This happens inside the VAE automatically.
- All tasks are available: `t2va` (plain image), `fl2va` with one or two condition images (editing/inbetween-style probes), and `ref2va` (reference-driven images, including single-image novel-view generation).
- `--trajectory_dir` writes per-step PNGs instead of per-step videos.
- Standalone `audio` references are rejected in one-frame mode (their window is defined by the target duration, which a single frame does not have); video references keep their embedded audio.
- The released 5-15 s duration gate does not apply; `--allow_experimental_duration` is not needed.

Training on one-frame targets is available for plain image LoRA (T2VA) and for editing/inbetween LoRA with 1-2 control images (FL2VA); see [One-frame training](#one-frame-training-t2va-image-lora) and [One-frame editing training](#one-frame-editing-training-fl2va-control-images) below. Ref2VA one-frame training (reference-driven image LoRA) is not implemented yet.

## Time semantics: `--one_frame`

```text
--one_frame "target_index=N,control_index=A;B"
```

Positions on H3's rotary time axis are expressed as **0-based 24 fps pixel-frame indices** on a nominal timeline (one pixel frame = 5/3 rotary units = 1/24 s). All times are relative to the target-block cursor, which itself moves with the text length — only relative placement carries meaning.

- `target_index` (default 0) places the generated frame.
- `control_index` places the FL2VA condition images, in `--first_frame`/`--last_frame` order, `;`-separated. It is required when condition images are present and rejected otherwise.
- There is no separate duration parameter: "frame 24 of a 10-second video" is `control_index=0;240` with `target_index=24`.

The base model reads these times as a real signal: an FL2VA anchor at the target's exact time is reproduced almost verbatim (anchor snapping), and intermediate positions interpolate when the caption follows the official alignment-line prompt format. For plain T2VA the index is nearly inert for the base model but remains a trainable input.

## Plain image generation (T2VA)

```bash
python minimax_h3_generate_video.py \
  --task t2va \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --prompt "A watercolor lighthouse at dusk." \
  --width 1024 --height 1024 \
  --frame_count 1 \
  --steps 30 --seed 42 \
  --blocks_to_swap 48 \
  --output output.png
```

## Conditioned images (FL2VA, one or two pictures)

One-frame FL2VA accepts `--first_frame` and/or `--last_frame` — a single picture is officially in-distribution for the FL2VA checkpoint (its released API takes zero, one, or two pictures). The text presentation numbers `<Picture i>` over the pictures that are present, so a lone last frame is still `<Picture 1>`; the first/last distinction is carried by the rotary times alone.

```bash
# generate "frame 24" of a nominal clip anchored by one condition image at frame 0
... --task fl2va --frame_count 1 \
  --first_frame anchor.png \
  --one_frame "target_index=24,control_index=0" \
  --prompt "..." --output frame24.png
```

For best results the caption should follow the official alignment-line formats from the prompt-writing guide (I2VA/L2VA/FL2VA opening lines); the base model reads condition times far more continuously with official-format captions than with plain ones.

## Reference-driven images (Ref2VA)

Ref2VA one-frame combines with inline `--ref` references (see `docs/minimax_h3.md`):

```bash
... --task ref2va --dit /models/minimax_h3_ref2va_bf16.safetensors \
  --frame_count 1 \
  --ref character.png \
  --prompt "..." --output view.png
```

With a full-reference-style caption, a single image reference yields novel views of the referenced subject (front/side/back selectable by text) with the environment plausibly extended — useful for synthesizing character-LoRA training data. Note that for dense 2D illustrations the reference is re-drawn rather than preserved pixel-exactly, and unseen-angle environments are plausible inventions, not geometry.

Audio-bearing video references are accepted and keep their own duration; combining them with a one-frame target is untested territory.

## One-frame training (T2VA image LoRA)

> [!WARNING]
> Experimental, like the rest of this mode. The single-token target is outside the released training distribution; quality expectations come from the one-frame generation results above, and image-trained LoRAs applied to video generation are unvalidated territory.

`--one_frame` on the two cache scripts and the trainer enables plain image LoRA training: each image becomes a single-token video target with a silence audio placeholder. The FL2VA base checkpoint with `--task t2va` is the normal choice, mirroring plain one-frame generation.

### Dataset configuration

Image datasets use the standard image keys. `fp_1f_target_index` (optional, default 0) places the target on the rotary time axis, in the same 0-based 24 fps pixel-frame indices as generation's `--one_frame target_index=N`; for plain image LoRA the default is fine. Control images and `fp_1f_clean_indices` belong to the FL2VA editing mode (next section); `multiple_target` is not supported.

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/data/h3/images"
cache_directory = "/data/h3/cache-images"
caption_extension = ".txt"
```

`image_jsonl_file` works as usual (`image_path` + `caption` per line). Buckets snap to the 32-pixel H3 grid. Image and video datasets may share one TOML but must not share a `cache_directory`.

Captions should follow the official T2VA caption format where possible. Because every one-frame item carries silent audio rows (excluded from supervision), it is recommended to state the absence of sound explicitly in the caption (for example a `sound:`-style field describing it as a silent still) so the text stays consistent with what the model sees — this likely also helps the LoRA transfer to normal video generation, where audio is live.

### Caching

```bash
python minimax_h3_cache_latents.py \
  --dataset_config /data/h3/images.toml \
  --task t2va --one_frame \
  --video_vae /models/minimax_h3_video_vae_fp16.safetensors \
  --audio_vae /models/minimax_h3_audio_vae_fp32.safetensors \
  --cache_seed 42 --skip_existing

python minimax_h3_cache_text_encoder_outputs.py \
  --dataset_config /data/h3/images.toml \
  --task t2va --one_frame \
  --text_encoder /models/qwen3vl_32b_minimax_h3_bf16.safetensors \
  --skip_existing
```

Each latent cache holds the single-token target (`[24,1,H/16,W/16]`, seeded posterior like video targets), the constant 2-frame silence audio latent (`audio_present=0`, encoded once per run), and the target index as a tensor entry. Text caches are plain T2VA presentations of the caption — time indices never enter the text, so changing `fp_1f_target_index` re-caches latents (cheap) but not text. The duration gate does not apply; `--allow_experimental_duration` is not needed.

### Training

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 minimax_h3_train_network.py \
  --dataset_config /data/h3/images.toml \
  --task t2va --one_frame \
  --dit /models/minimax_h3_fl2va_bf16.safetensors \
  --network_module networks.lora_minimax_h3 --network_dim 16 \
  --video_only \
  ... # remaining flags as in docs/minimax_h3.md
```

- **The guidance loss is effectively mandatory for one-frame training.** Without it, de-distillation drift surfaces within ~50 steps as structural degradation — wobbly lines and broken proportions, like low-CFG output of an undistilled model — rather than the washout seen in video training (image steps average over far fewer target rows and repeat a small dataset quickly). `--h3_guidance_loss_scale 4.0 --h3_guidance_loss_sigma_min 0.15` with an uncond cache (see `docs/minimax_h3.md`) restored clean structure in testing; a short LR warmup (e.g. 50 steps) also helps the early phase.
- `--video_only` is recommended for image-only runs: the silence placeholders are excluded from audio supervision by presence gating either way, so the audio loss would always be 0.
- Steps are much cheaper than video steps (a 1 MP image is a few hundred target rows); with block swap active, per-step time is dominated by weight streaming rather than compute.
- Mixed image+video training in one run is expected to work (`--one_frame` only adds acceptance of one-frame batches; video batches are unaffected) but is untested — treat it as experimental.
- `--h3_teacher_matching` is not supported with `--one_frame` yet.

Training-time samples support one-frame outputs: `--f 1` in a sample prompt line switches that sample to a PNG (audio is never decoded), and `--of target_index=N` optionally places it on the time axis:

```text
A watercolor lighthouse at dusk. --w 1024 --h 1024 --f 1 --s 30 --d 42
```

The LoRA metadata records `ss_minimax_h3_one_frame` for provenance. The resulting LoRA loads into generation as usual (one-frame or video).

## One-frame editing training (FL2VA, control images)

> [!WARNING]
> Experimental. This trains the base model's timed-anchor pathway directly; read the index guidance below before building a dataset.

With `--task fl2va`, an image dataset pairs each target image with 1-2 **time-annotated control images**: the controls become FL2VA condition latents (and `<Picture i>` visuals in the text presentation), and their positions on the rotary time axis come from the dataset config. This trains editing LoRAs (control = source image, target = edited image) and inbetween/中割り LoRAs (controls = endpoint frames, target = an intermediate frame).

### Dataset configuration

```toml
[[datasets]]
image_directory = "/data/h3/edit/targets"
control_directory = "/data/h3/edit/sources"
cache_directory = "/data/h3/cache-edit"
caption_extension = ".txt"
fp_1f_clean_indices = [0]     # control image positions (24 fps pixel-frame indices)
fp_1f_target_index = 24       # target position — REQUIRED when controls are present
```

- `control_directory` matches controls to targets by filename (`image.png` ↔ `image.png` / `image_0.png`), or use `image_jsonl_file` with `control_path` (or `control_path_0`/`control_path_1`) per line.
- `fp_1f_clean_indices` gives one index per control image, in packed (first, last) order: control 0 is the "first" slot, control 1 the "last" slot. With one control only the "first" slot is used; the slot name carries no time meaning of its own — only the indices do.
- Both `fp_1f_clean_indices` and an explicit `fp_1f_target_index` are required when controls are present; there are no defaults. Controls are resized to the target's bucket resolution.
- The alpha channel of RGBA control images is ignored (dropped before both VAE and text-encoder processing) — unlike FramePack one-frame training, it does not act as a mask.
- Time-order is unconstrained: an anchor **after** the target (`fp_1f_clean_indices = [120]`, `fp_1f_target_index = 24`) trains an L2VA-style LoRA (generate the image that precedes an end state). Note the official pipeline stretches a lone last picture to the canvas at inference while training resizes to the bucket — a minor known divergence.

### Choosing indices

The base model's strongest prior is **verbatim anchor copying at coinciding timestamps**: a control whose index equals the target index is reproduced almost exactly, so such a dataset trains head-on against copying — only do this when copy-at-the-anchor is the desired behavior. The recommended starting recipe for editing is `fp_1f_clean_indices = [0]`, `fp_1f_target_index = 24` (a one-second separation); inference must then use the same relative placement (`--one_frame "target_index=24,control_index=0"`). For inbetween triplets extracted from real videos, use the real frame distances: (first@0, last@N, target@αN) → `fp_1f_clean_indices = [0, N]`, `fp_1f_target_index = round(αN)`. Since the indices live in the dataset config, one α per dataset block; several blocks can share a TOML.

**Captions must follow the official alignment-line formats** (I2VA/L2VA/FL2VA opening lines from the prompt-writing guide): plain captions actively suppress the base model's continuous reading of condition times, which is exactly the pathway this training relies on.

### Caching and training

Same commands as plain image training with `--task fl2va` instead of `--task t2va` on both cache scripts and the trainer. The latent cache additionally holds the condition latents (`latents_first`, plus `latents_last` for two controls) and the control indices as a tensor entry; the text cache embeds the bucket-resized control images in the FL2VA presentation. Changing `fp_1f_target_index` or `fp_1f_clean_indices` re-caches latents only (`--skip_existing` detects it); changing control image files re-caches both.

The guidance-loss recommendation from plain image training applies unchanged. Training-time samples mirror the generation CLI: provide the condition image(s) and the placement per prompt line:

```text
Official-format caption... --w 1024 --h 1024 --f 1 --s 30 --i source.png --of target_index=24,control_index=0
```

(`--i` is the first/only condition, `--ei` the last; `control_index` takes one `;`-separated entry per provided image, and is required.)
