# MiniMax-H3

## Overview

Musubi Tuner supports MiniMax-H3 text-to-video-with-audio (T2VA), first/last-frame-to-video-with-audio (FL2VA), and reference-to-video-with-audio (Ref2VA) LoRA training and standalone generation.

The implementation follows the released MiniMax-H3 packing, Qwen3-VL conditioning, dual video/audio flow schedules, and two VAE layouts. It supports the published full and pruned BF16 transformers, the full and pruned ConvRot INT8 transformers, and the ConvRot INT8 and NVFP4+AWQ Qwen3-VL text encoders.

Read and accept the [MiniMax-H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) before downloading or using the weights.

## Model Files

Download the following files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3):

| Component | Supported file |
| --- | --- |
| FL2VA and T2VA transformer | `diffusion_models/minimax_h3_fl2va_bf16.safetensors` |
| FL2VA and T2VA pruned transformer | `diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors` |
| FL2VA and T2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors` |
| FL2VA and T2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` |
| Ref2VA transformer | `diffusion_models/minimax_h3_ref2va_bf16.safetensors` |
| Ref2VA pruned transformer | `diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors` |
| Ref2VA ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors` |
| Ref2VA pruned ConvRot INT8 transformer | `diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors` |
| Qwen3-VL-32B text encoder | `text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors` |
| Qwen3-VL-32B ConvRot INT8 text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Qwen3-VL-32B NVFP4+AWQ text encoder | `text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` |

T2VA uses an FL2VA transformer without first/last conditions. Pre-quantized files (ConvRot INT8 full or pruned, transformer or text encoder; NVFP4+AWQ text encoder) and pruned BF16 transformers are detected automatically from their tensor structure — no extra flag is needed. FP8, NVFP4 transformers, and malformed or partial quantized files are rejected rather than silently interpreted as BF16. See [ConvRot INT8 Quantized Base Weights](#convrot-int8-quantized-base-weights) and [NVFP4 Text Encoder](#nvfp4-text-encoder) for details.

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

Add `--uncond_output /data/h3/uncond_space.safetensors` to also write the tiny uncond probe embedding used by the training guidance loss (see Guidance-distillation countermeasure below); `--uncond_text` overrides the probe text (default: a single space).

Add `--teacher_conditions first,last` (with `--task t2va`) to also store the FL2VA teacher presentation of each item alongside the plain caption rows, for teacher-matching training (see Teacher-matching training below). The caption is shared between the two presentations; the teacher rows only add the `<Picture 1>`/`<Picture 2>` prefix with the first/last frames of the crop window. The latent caches for that mode must be created with `--task fl2va`.

`--teacher_conditions ref` (also `--task t2va` only) instead stores the Ref2VA teacher presentation: the training crop itself as the copy-source reference (its 2 fps sampled frames plus an `<Audio 1>` declaration), with the copy-declaration boilerplate wrapped around the shared caption automatically (see Reference teacher below). The two teacher kinds use distinct cache keys, so the trainer hard-fails when the cache and `--h3_teacher_conditions` disagree. Latent caches for the ref mode can be `--task fl2va` or plain `--task t2va`: the reference condition latents at training time are the cached target latents themselves.

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

### Guidance-distillation countermeasure (guidance loss)

The released H3 checkpoints are CFG-distilled: their prediction lives in the amplified space `g(c) = u + s*(c(c) - u)`, where `u` is the unconditional velocity baked in during distillation. Training a LoRA on the plain flow-matching target pulls the model out of that space (de-distillation drift: washed-out, low-adherence outputs as training progresses). The guidance loss re-anchors the target in the amplified space instead:

```
target = uncond + scale * (velocity - uncond)
```

where `uncond` is the model's own no-grad prediction for the same noised input under an uncond probe embedding, with the LoRA active (matching how the adapted model runs at inference). This is the same mechanism as the ai-toolkit guidance loss; community reports suggest `scale` 3-4, with 4 more reliable for longer runs.

```text
--h3_guidance_loss_scale 4.0 \
--h3_guidance_loss_uncond_cache /data/h3/uncond_space.safetensors
```

The uncond cache is written by `minimax_h3_cache_text_encoder_outputs.py --uncond_output` (about 10 KB; one text-encoder forward). The default probe, a single space, was selected by screening candidate uncond conditions against the released checkpoint: probes carrying content (e.g. quality-style negative prompts) are interpreted as conditions and amplified, while near-empty probes (single space, single EOS token, an all-zero row) form one tight equivalent cluster whose stand-alone generations show no CFG burn — the signature of the true distillation uncond.

Optional refinements:

- `--h3_guidance_loss_scale_audio` sets a separate scale for the audio target (default: same as video). The audio guidance signal survives to lower noise levels than video on its own sigma axis.
- `--h3_guidance_loss_sigma_min` skips the extra uncond forward when the drawn base sigma (pre-shift, 1 = pure noise) is below the threshold. Measured on the released checkpoint, the relative text-guidance magnitude `|g(cond) - g(uncond)| / |g|` for video collapses from ~46% at `sigma_video 0.98` to ~3% at `0.71`, so low-sigma corrections mostly amplify per-sample noise while still paying a full forward. Training logs confirm this: below base sigma ~0.15 the logged gap magnitudes are dominated by the irreducible per-sample residual (the part of the noise draw no prediction can know — especially visible in the audio gap, which *rises* toward low sigma), i.e. amplified label noise rather than guidance signal. **Recommended: `0.15`** — it skips the noisiest ~15% of steps while keeping nearly all of the video and audio guidance signal. `0` (default) always applies the loss, matching ai-toolkit.

Each step logs `guidance/applied` and the sigma-dependent gap magnitudes `guidance/video_gap_rms` / `guidance/audio_gap_rms`; the metadata records `ss_minimax_h3_guidance_loss_scale`, `..._scale_audio`, and `..._sigma_min`. The cost is one extra no-grad forward per applied step (roughly +50% step time without gating; less with `--h3_guidance_loss_sigma_min`).

### Teacher-matching training (privileged-condition teacher for a T2VA student)

`--h3_teacher_matching` trains a T2VA LoRA against the frozen base model's FL2VA predictions instead of the flow-matching target. Each step runs one extra no-grad forward of the same transformer with the LoRA disabled, conditioned on the real first and last frames of the training clip and the Picture-prefixed FL2VA text presentation — privileged information the text-only student never sees:

```
loss = || student(x_t, text) - teacher(x_t, text, first, last) ||^2
```

The teacher prediction lives in the distilled guided space, so the target needs no guidance scale and no uncond probe, and the de-distillation drift of plain flow targets is structurally avoided. `--h3_teacher_matching` is therefore mutually exclusive with `--h3_guidance_loss_scale`, and requires `--task t2va`.

Data preparation differs from plain T2VA in two places:

- Latent caching runs with `--task fl2va`, so the caches include the first/last condition latents (they feed only the teacher forward).
- Text caching runs with `--task t2va --teacher_conditions first,last`, so each cache stores both presentations: the plain caption rows for the student and the FL2VA presentation for the teacher. The caption itself is shared.

Training then runs with `--task t2va --h3_teacher_matching`. What to expect:

- **The loss does not converge to zero.** The teacher sees the real endpoints, so its prediction contains content the text alone cannot determine; this information gap is an irreducible floor, largest at high sigma. Read the per-step logs `teacher/video_flow_gap_rms` / `teacher/audio_flow_gap_rms` binned by `teacher/base_sigma` — they measure how far the teacher deviates from the raw velocity target (guidance amplification plus endpoint information) — rather than expecting `loss/video` to vanish.
- **Audio degenerates to a base-preservation anchor.** The visual endpoints carry almost no audio information, so the teacher's audio prediction stays close to the base model's text-conditioned prediction, and matching it preserves the base audio behavior instead of learning the audio content of the training data. Because H3 is single-stream, this anchor also protects the video path from drift entering through the shared weights. Training voice or audio content needs the guidance loss or the reference teacher (below) instead. `--video_only` and `--audio_loss_weight` gate the audio term as usual.
- The appearance signal is the strongest part of the teacher target (endpoints plus `x_t` leakage); intermediate motion is weaker at high sigma, following the sigma profile measured for the guidance loss.
- The cost is one extra no-grad forward per step, roughly +50% step time (same as the ungated guidance loss).

Each teacher-matching step also logs a direction/magnitude decomposition of the student-teacher residual: `teacher/video_cos` / `teacher/audio_cos` (cosine similarity between the student prediction and the teacher target) and `teacher/video_norm_ratio` / `teacher/audio_norm_ratio` (student norm over teacher norm, 1 = matched). MSE mixes both components, but content errors are direction-flavored while burn/wash-out drift is magnitude-flavored; a norm ratio drifting above 1 is an early warning for amplification-style degradation. Bin them by `teacher/base_sigma` like the flow gaps. At the conditional-mean optimum of the MSE, the per-bin averages of cos and norm_ratio converge to a common value (the square root of the band's predictable energy share), so within one sigma bin the cos/norm_ratio gap reads as remaining training distance and their common limit as the band's irreducible endpoint-information share.

The residual is further split into a per-channel mean (`teacher/video_residual_dc_rms` / `teacher/audio_residual_dc_rms`) and the zero-mean remainder (`teacher/video_residual_ac_rms` / `teacher/audio_residual_ac_rms`), with `rms(residual)^2 = dc_rms^2 + ac_rms^2`. The DC component is a global color/tone cast — the style axis — while the AC component carries spatially structured content, so a gap that shrinks mostly in DC means the student is learning the dataset's palette rather than its subjects. Bin by `teacher/base_sigma` as above.

**High-sigma protection (`--h3_teacher_condition_sigma_max`, default 0.75).** The teacher target is a noiseless regression label: unlike the flow-matching target, whose high-sigma content is mostly per-sample noise that self-cancels across steps, the teacher deviation is a deterministic function of the noised input, so every step pushes consistently in the same direction and fitting is very fast. Near pure noise the endpoint content is unpredictable from the text, so unrestricted teaching there rapidly overwrites the base model's composition prior with the dataset mean. Above this threshold the teacher drops the endpoint conditions and runs on the student's own text — the target becomes a pure base-preservation anchor that pins the high-sigma behavior to the base and also counters collateral drift from the LoRA's shared weights. The identity-decision band was measured at base sigma 0.6-0.75 on diverse character data (color/style decisions start around 0.73, so style and identity overlap — the band gate alone cannot separate them); the default keeps that band in the teaching regime. Lower toward 0.4-0.5 for low-diversity or generically-captioned data, where the high band carries mostly dataset-mean composition; `1.0` disables the protection. Each step logs `teacher/conditioned` so the two regimes can be separated when reading the sigma-binned logs; note that `loss/video` means different things in the two regimes (teaching residual vs preservation residual). `--h3_teacher_preservation_weight` (default 1.0) scales the loss of the anchor steps, on top of an automatic correction that keeps the anchor's expected gradient share invariant under timestep focus; raise it if the anchor-band drift (`teacher/*_residual_dc_rms` on unconditioned steps) keeps growing instead of reaching an equilibrium.

**Loss shape.** The teacher-matching loss is the exact magnitude/direction split of the MSE, `(||p||-||t||)^2 + 2*||p||*||t||*(1-cos)`, with the `||p||` factor of the direction term detached. Plain MSE couples the two components — hedging an unpredictable direction pays off by shrinking the norm, so the student converges to the conditional mean's reduced magnitude, which appears at inference as delayed content commitment and washed-out contrast. With the coupling removed, the direction gradient is purely rotational and the magnitude optimum becomes the per-sample teacher norm `E[||t||]` (full commitment) instead of `||E[t]||` (measured: the wall-band norm ratio holds at ~1.0 instead of decaying to 0.97, and the wash-out disappears at inference). At the default weights the loss value equals the plain MSE, so loss curves stay comparable. Two knobs shape it:

- `--h3_teacher_loss_mag_weight` (default 1.0) weights the magnitude term relative to the direction term (fixed at 1.0). Lower it to prioritize direction matching (0 = pure direction); a per-sample norm ratio drifting above ~1.05 in the sigma-binned `teacher/*_norm_ratio` logs is the signal to raise it back.
- `--h3_teacher_loss_dc_weight` (default 1.0) scales the video residual's per-channel DC component on conditioned teaching steps. The DC axis is a global color/tone cast: a dataset with a consistent palette teaches it as a small (measured ~7% of the teaching-band residual energy) but fully coherent signal that accumulates into a visible style shift. Lowering the weight (e.g. 0.0-0.3) removes that source without touching the spatially structured content signal. Preservation steps and the audio anchor always keep their full DC penalty — there it is what pulls palette drift back to the base. Task-dependent: keep 1.0 when the dataset's palette is part of what should be learned (style LoRA).

**Timestep focus (`--h3_timestep_focus_prob`, any H3 training).** The base sigma is drawn uniformly, which spends most steps at very high effective noise (video shift 12 maps base 0.2 to sigma 0.75). With `--h3_timestep_focus_prob P` the draw lands uniformly inside `[--h3_timestep_focus_min, --h3_timestep_focus_max)` (default 0.4-0.8, the band where content is decided) with probability P and stays uniform over [0,1) otherwise, so the band density becomes `P + (1-P)*(max-min)` while the rest of the range keeps `(1-P)` of the samples (measured at P=0.5: the wall band converges about twice as fast). The preservation anchor's loss is automatically re-weighted for the thinned anchor band, so focus does not silently weaken the drift protection. Does not compose with `--min_timestep`/`--max_timestep`. The remap is a deterministic function of the uniform draw, so pre-drawn dataset timesteps keep working.

A validated starting recipe for identity training (character data, appearance kept out of the captions or bound to a trigger word):

```text
--h3_teacher_matching --h3_teacher_loss_dc_weight 0.3 --h3_timestep_focus_prob 0.5
```

`--h3_teacher_conditions` selects the teacher's privileged conditions: `first,last` (default; training videos always provide both endpoints, and the FL2VA base is most in-distribution with both anchors) or `ref` (the reference teacher below). The seam reserves the interface for further variants (single-sided, anchored, segmented teachers). The metadata records `ss_minimax_h3_teacher_matching`, `ss_minimax_h3_teacher_conditions`, `ss_minimax_h3_teacher_condition_sigma_max`, the loss-shape settings (`ss_minimax_h3_teacher_loss*`, `ss_minimax_h3_teacher_preservation_weight`), and the timestep-focus settings when enabled.

#### Reference teacher (`--h3_teacher_conditions ref`)

`--h3_teacher_conditions ref` switches the teacher from the two endpoints to the training clip itself: the teacher runs on the Ref2VA layout with the cached target video and audio latents as its reference condition (same per-step clean augmentation as regular condition latents), and the teacher text rows carry the clip's 2 fps sampled frames plus the official editing-style copy declaration (`fully_preserved` / `fully_copy`), which the cache script wraps around the shared caption automatically. The teacher therefore sees complete information at every sigma instead of only what the endpoints pin down.

Measured on the released FL2VA weights — which handle a self-reference far more literally than the Ref2VA weights (their condition semantics is "exact frames of this video", so the weight-shared FL2VA base is also the better copy machine), while an unrelated reference is simply ignored (content-specific use, no style bleed from the mechanism itself):

- The teaching-band video gap collapses to a flat model-error floor (base sigma 0.15-0.75: rms ~0.10-0.14, cos 0.995+), 3-5x below the endpoint teacher in the identity-decision band. The irreducible endpoint-information floor is gone, and with it most of the conditional-mean hedging.
- The reference audio is copied too, so **audio becomes a real teaching target** in this mode; the `fully_copy` declaration is what opens the full teaching band for audio (without it, audio education stops around base sigma 0.55). The audio loss stays presence-gated as usual; for items without real audio the reference degenerates to encoded silence and the audio term stays off.
- Above base sigma ~0.85 the FL2VA weights fail to align the reference against the noise-dominated `x_t` (structural — prompting does not fix it), so keep `--h3_teacher_condition_sigma_max` at its default 0.75: the gate boundary coincides with the edge of the healthy range.

Data preparation: re-run text caching with `--teacher_conditions ref` (the teacher rows use their own cache keys, so a cache/flag mode mismatch is a hard error rather than a silent layout desync). Latent caches need no changes: existing `--task fl2va` caches work as-is (the first/last latents are simply unused), and plain `--task t2va` caches suffice for new datasets.

Caveats:

- The teacher forward carries the full reference video and audio tokens on top of the target tokens, so the teacher step is slower and more memory-hungry than with the endpoint teacher.
- **Audio education is only as good as the dataset's audio.** The teacher copies each clip's actual audio track, and with the default `--audio_loss_weight 1.0` the audio term carries a large share of the education loss (measured: over half). A consistent voice across clips is learnable signal; when the training audio is not consistent (e.g. synthesized clips generated without an audio reference), only its energy/DC statistics are learnable and the rest of the audio residual never shrinks. A paired A/B (weights 1.0 vs 0.25 on identical noise/timestep sequences) showed that this unlearnable remainder is essentially neutral for video: the video-side education was unchanged, and the post-plateau anchor drift was marginally *larger* at the lower weight (the audio gradient noise inflates Adam's second moment and acts as mild damping). Choose the weight by audio policy — lower it or pass `--video_only` when the dataset's audio is not worth learning — and manage video quality with the step budget and the preservation weight instead.
- The teaching-band residual can plateau well before a long run ends (measured around 300 steps on a small character set). Training past the plateau mostly optimizes irreducible floors while the anchor-band drift keeps growing, and sample quality oscillates with the drift-and-recover cycle — in 500-step runs the strongest checkpoints sat at or just after the plateau. Save and validate intermediate checkpoints rather than judging only the final one; for longer runs, raise `--h3_teacher_preservation_weight` and/or decay the learning rate past the plateau.
- A complete-information teacher leaves almost no guided-space wedge inside the teaching band, so the de-distillation pressure there returns to roughly flow-matching levels; the protection moves to the anchor band, the decomposed loss, and monitoring. Starting recipe relative to the endpoint teacher: keep `--h3_teacher_condition_sigma_max 0.75`, lower `--h3_teacher_loss_mag_weight` (the remaining distillation wedge inside the band is mostly a magnitude effect), consider raising `--h3_teacher_preservation_weight`, and watch the sigma-binned `teacher/*_norm_ratio` — a student norm ratio shrinking toward 1.0 in the upper teaching band (base 0.65-0.75, where the text-only base still over-commits) is the early de-amplification signature.
- Because the teacher is complete-information at every sigma, the learning signal is no longer tied to the high-sigma attribution window; shifting the focus band lower (e.g. `--h3_timestep_focus_min 0.3 --h3_timestep_focus_max 0.6`) becomes a meaningful A/B, at the cost of thinning the composition-commitment band 0.6-0.75.

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

**Pruned transformers.** The released pruned artifacts (BF16 and ConvRot INT8) replace the sinusoidal time embedder with a published FP32 `[1025, 8]` AdaLN curve table (`adaln_t_table`) and 8-wide F16 AdaLN projections (loaded as BF16). They are recognized structurally and interpolated in FP32 over the model time `t = 1 - sigma` in `[0, 1]`. Pruning removes the ~26 GB AdaLN projection weights (BF16: ~66 GB to ~40 GB; INT8: ~34 GB to ~21 GB) and shrinks each swappable block by ~40%, cutting block-swap transfer time by the same fraction. `--convrot_int8` also works on a pruned BF16 file: the 8-wide AdaLN projections fall outside every ConvRot group size and stay BF16, reproducing the published pruned INT8 scope.

**Self-pruning (`--prune_adaln`).** For checkpoints only published as full BF16, `--prune_adaln` (training and generation) prunes at load time instead: a mean-centered rank-8 SVD basis of the SiLU'd time-embedding curve is computed on the fly (seconds), and each AdaLN projection is rewritten to a 9-wide BF16 Linear (8 basis coefficients plus a constant channel carrying the curve mean) during the streaming load, so peak memory stays near the pruned model size. Unlike the published artifacts, the sinusoidal time embedder is retained: timesteps stay exact and continuous with no curve-table interpolation. Measured against the full FP32 modulation on the released FL2VA weights, the reconstruction error is at parity with (marginally below) the published pruned artifacts. The flag combines with `--convrot_int8` to reproduce the published pruned INT8 scope from a full BF16 file, is a no-op on already-pruned checkpoints, and is rejected for pre-quantized ConvRot INT8 checkpoints (use the published pruned artifact instead).

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

The same command accepts the full or pruned BF16 or ConvRot INT8 transformer and the ConvRot INT8 or NVFP4+AWQ text encoder; formats are detected automatically. With a BF16 transformer, LoRAs are merged destructively once after loading (fastest inference); with a ConvRot INT8 base, each `--lora_weight` stays a separate runtime additive branch with its corresponding multiplier (see [ConvRot INT8 Quantized Base Weights](#convrot-int8-quantized-base-weights)).

`--lora_runtime_attach` forces the runtime-branch route on any base. The merge rounds the fused weights back to the base storage grid, and a BF16 mantissa step is about 0.4% of each weight's magnitude — per-element LoRA deltas below that are silently erased. Adapters trained toward small equilibria (teacher matching in particular) can lose most or all of their effect this way while behaving normally during training, whose forward keeps the LoRA as a separate full-precision branch. Runtime attachment reproduces the training-time forward exactly, at a small speed cost.

For FL2VA, keep the FL2VA base and replace the task inputs:

```text
--task fl2va --prompt "..." --first_frame first.png --last_frame last.png
```

For Ref2VA, use a Ref2VA base (BF16 or ConvRot INT8) and an ordered JSONL record:

```text
--task ref2va --dit /models/minimax_h3_ref2va_bf16.safetensors --reference_jsonl /data/h3/ref2va.jsonl --reference_index 0
```

The Ref2VA generation JSONL intentionally uses the same validated schema as training, including target `video_path`, optional target audio, caption, and references. The target media identifies the record but is not used as a generation target. `--prompt` may override the record caption when encoding fresh text conditioning.

T2VA and Ref2VA generation may use `--text_cache` instead of `--text_encoder`. The cache must match the requested task, cache format version, and exact presentation fingerprint (which covers the prompt, frame count, and size+mtime identities of the reference media, so the cache must be used on the machine holding the original files). T2VA still requires `--prompt` so that identity can be verified; Ref2VA uses the selected record caption unless `--prompt` overrides it. FL2VA generation does not accept a dataset text cache because external first/last images cannot be proven identical to the crop presentation that produced that cache.

`--steps N` means N model evaluations, so the schedule uses N+1 grid points. The released implementations (SGLang serving and the diffusers scheduler) instead count grid points: their `num_inference_steps = N` performs N-1 evaluations. Musubi `--steps N` is therefore grid-identical to official `num_inference_steps = N+1`; to reproduce the official 50-step serving default exactly, pass `--steps 49`.

`--trajectory_dir DIR` is a diagnostic: it logs each step's base/video/audio sigma (also written to `DIR/sigma_schedule.csv`) and decodes each step's clean estimate (`x0_hat = x_t + sigma * v`, the model's current best guess of the final video) to a silent per-step MP4 in `DIR`, named with the step index and its base and video sigmas. Scrubbing through the files shows at which step composition, palette, and identity settle. The per-step latents are held on the CPU and decoded after the normal output, so peak VRAM is unchanged, but decode time grows with the step count; `--trajectory_stride N` decodes every N-th step (the last step is always included).

The native sampler builds one common base grid, derives independent shifted video and audio sigma grids, and advances each modality with its own finite sigma interval. It does not apply CFG, negate the model heads, or apply ComfyUI's single-sampler audio slope adapter. Musubi also adds condition noise before packing, while ComfyUI adds it after packing; the distributions agree but RNG placement does not. These two intentional differences mean the same seed is not bitwise reproducible against ComfyUI. Video and audio are decoded sequentially, trimmed to a common duration, and muxed with PyAV as H.264 plus AAC.

## Limitations

- Released BF16 and ConvRot INT8 (each full or pruned) FL2VA/Ref2VA transformer bases only.
- BF16, ConvRot INT8, or NVFP4+AWQ Qwen3-VL text encoder only.
- No FP8 artifact loading, and no NVFP4 transformer loading.
- No CFG or negative prompt.
- No numbered reference-directory convention.
- Dataset `batch_size` is fixed to 1; use gradient accumulation for larger effective batches.
- No padded multi-sample packed layouts.
