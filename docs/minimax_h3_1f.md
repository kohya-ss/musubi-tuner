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

Training on one-frame targets (image LoRA) is not implemented yet; this mode currently serves inference and dataset-synthesis experiments.

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
