# MiniMax-H3 Modality Best-of-K Design

**Status:** Approved design; implementation deferred
**Date:** 2026-08-16
**Branch:** `codex/issue-1019-explorative-modeling`
**Base:** `upstream/dev` at `ca221b10`, merged as `ce0a325`
**Issue:** [kohya-ss/musubi-tuner#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019)

## 1. Scope

This document supersedes the MiniMax-H3 best-of-K sections of
`2026-08-09-explorative-modeling-design.md` and the earlier audio-only draft.
It defines three H3-owned controls:

```text
--h3_video_best_of_k
--h3_audio_best_of_k
--h3_image_best_of_k
```

Video and audio modes apply only to multi-frame batches. Image mode applies
only to upstream's experimental one-frame T2VA batches. This batch-disjoint
policy lets a mixed image/video run configure image K alone or together with at
most one multi-frame policy.

Only the old design document was carried from `codex/h3-audio-best-of-k`.
No `qinglong` production commit or unrelated LongCat, Lens, or Mage Flow change
is part of this revision. Any required best-of-K correctness repair must be
ported and tested directly against the current upstream-based issue branch.

## 2. Source Findings

The design is grounded in these primary sources:

- The [open feature request](https://github.com/kohya-ss/musubi-tuner/issues/1019)
  asks Musubi Tuner to support Explorative Modeling but does not define
  modality semantics for a joint video/audio transformer.
- The [Forward XM paper](https://arxiv.org/abs/2607.27372v1) holds the training
  example and timestep fixed, samples K noise matches, scores each match, and
  trains only the winner. The
  [official project page](https://explorative-modeling.github.io/) presents the
  same minimal loop.
- The [official `alexiglad/XM` implementation at commit
  `9d06ced6`](https://github.com/alexiglad/XM/tree/9d06ced61e2d2775a34782eb5830584ae4ef6094)
  uses per-sample winner selection, a `best_of_k == 1` short circuit, no-grad
  candidate search, and one gradient-enabled winner recomputation. All other
  stochastic conditions are replayed across search and recomputation.
- Musubi H3 has one joint transformer and two flow streams. Video latents are
  `[B,24,F,H,W]`, audio latents are `[B,32,2,A]`, one base time produces two
  shifted sigmas, and the ordinary objective is
  `video_mse + effective_audio_weight * audio_mse`.
- Upstream commit `ca221b10` adds experimental one-frame T2VA training. An
  image batch has target video latents `[B,24,1,H,W]`, a constant silence audio
  placeholder `[B,32,2,2]`, `audio_present=0`, and an H3 time override from
  `one_frame_target_index`. The existing `--one_frame` flag permits image and
  video datasets to mix; it does not turn every batch into an image batch.
- Upstream classifies a batch from the actual target latent shape (`F == 1`)
  and validates the one-frame cache marker. Best-of-K dispatch must use that
  runtime fact, not the global `--one_frame` flag.

The paper and official XM repository do not define modality-only selection for
a joint video/audio model. Video and audio modes therefore remain H3-specific
heuristics. Image mode is closer to the Forward XM objective because the
silence audio rows have zero effective loss and raw image/video MSE is the full
effective batch objective. It still uses an H3-specific option and metric
namespace: neither the paper nor official implementation validates H3's
experimental joint-stream one-frame regime.

## 3. Goals

- Add `--h3_audio_best_of_k K` and `--h3_image_best_of_k K`, both defaulting to
  `1`, beside the existing video option.
- For a multi-frame video batch, activate at most one policy:
  - video mode varies target-video noise and ranks raw per-sample video MSE;
  - audio mode varies target-audio noise and ranks raw per-sample audio MSE.
- For a one-frame image batch, image mode varies its target-video-stream noise
  and ranks raw per-sample image/video MSE.
- Permit image K with either video K or audio K because they apply to disjoint
  batch types. Keep video K and audio K mutually exclusive.
- Hold the other stream, base time, both shifted times, cached data, time
  overrides, augmented conditions, and effective weights fixed across a search.
- Recompute the selected candidate once with gradients and optimize the
  unchanged ordinary objective for that batch.
- Use one runtime-mode-driven H3 candidate loop rather than three copied loops.
- Preserve ordinary RNG, loss, gradients, call count, and metrics for every
  batch whose resolved K is `1`, including inactive batch types inside a mixed
  run where another mode has K greater than one.
- Reject `--video_only` with active audio K before dataset, accelerator, or
  model allocation.
- Reject `--h3_teacher_matching` with any active H3 best-of-K mode until a
  teacher-target candidate-selection contract is designed.
- Fall back to one ordinary prepared forward when an audio-search batch has
  zero effective audio weight.

## 4. Non-Goals

- Jointly varying video and audio noise in one candidate.
- Selecting multi-frame candidates by the composite video-plus-audio loss.
- Enabling common `--xm_best_of_k` for H3.
- Applying video K to one-frame batches or image K to multi-frame batches.
- Adding one-frame FL2VA or Ref2VA training, control images, or multiple image
  targets; upstream supports plain one-frame T2VA training only.
- Treating image best-of-K as a replacement for the guidance loss recommended
  by upstream one-frame training documentation.
- Lifting H3's current `batch_size = 1` restriction.
- Candidate batching, chunk controls, or retaining K activation graphs.
- Dataset, cache, VAE, scheduler, network, optimizer, or checkpoint-format
  changes.
- Quality claims for H3 LoRA training.
- A new PyTorch 2.5.1 or CUDA 12.4 compatibility matrix. Verification covers
  the current environment and repository test suite only.

## 5. User Contract

### 5.1 CLI and validation

H3 exposes:

```text
--h3_video_best_of_k INT
--h3_audio_best_of_k INT
--h3_image_best_of_k INT
```

All default to `1`. Runtime validation accepts only an exact Python `int >= 1`;
booleans, floats, strings, zero, and negative values fail even when TOML loading
bypasses `argparse` conversion.

Configuration is resolved as follows:

| Video K | Audio K | Image K | Multi-frame batches | One-frame batches | Result |
|---:|---:|---:|---|---|---|
| 1 | 1 | 1 | ordinary | ordinary | ordinary H3 training |
| >1 | 1 | 1 | video search | ordinary | valid |
| 1 | >1 | 1 | audio search or zero-weight fallback | ordinary | valid |
| 1 | 1 | >1 | ordinary | image search | valid with `--one_frame` |
| >1 | 1 | >1 | video search | image search | valid with `--one_frame` |
| 1 | >1 | >1 | audio search or fallback | image search | valid with `--one_frame` |
| >1 | >1 | any | none | none | startup error naming video and audio options |

Additional validation:

- `--xm_best_of_k` must be exactly `1` for H3.
- `--h3_image_best_of_k > 1` requires `--one_frame`.
- `--video_only` with `--h3_audio_best_of_k > 1` is a startup error.
- Image K and video K remain valid with `--video_only`.
- `--h3_teacher_matching` is incompatible with any H3 K greater than one.
  Image K is already transitively incompatible because upstream rejects
  teacher matching with `--one_frame`; video/audio modes keep the explicit
  rejection because their teacher-conditioned candidate objective is undefined.
- `--audio_loss_weight 0` with audio K is allowed, but every multi-frame batch
  uses the zero-effective-audio fallback. Startup logs warn that audio search
  and audio best-of-K metrics will not occur.

Validation remains before resource allocation. Failed fresh or repeated
validation restores the base routing state to `(count=1, enabled=False)`.

### 5.2 Naming and logging

Video and audio modes are named "MiniMax-H3 video-focused/audio-focused
best-of-K heuristic (not Forward XM)." Image mode is named "MiniMax-H3 image
best-of-K (Forward-XM-style selection; experimental H3 one-frame mode)." No H3
mode logs under `xm/`.

Startup logging reports every configured K greater than one, its eligible batch
type, selection score, final objective, sequential execution, and operation
count estimate. It must not imply that `--one_frame` makes multi-frame batches
use image K.

## 6. Configuration and Batch Resolution

Use two pure resolution steps:

```text
resolve_h3_best_of_k_config(args)
  -> {video_count, audio_count, image_count, routing_count}

resolve_h3_batch_best_of_k(config, runtime_batch)
  -> (mode: "video" | "audio" | "image" | None, count: int)
```

The configuration resolver owns exact-type/range checks, common-XM rejection,
video/audio mutual exclusion, the `video_only` conflict, teacher-matching
rejection, and the image/one-frame requirement. `routing_count` is the maximum
of the three H3 counts and exists only to activate the base trainer's best-of-K
dispatch when any batch type may need search.

The batch resolver uses the validated runtime layout:

- one-frame target (`F == 1` and validated one-frame layout): image K;
- multi-frame target: video K when active, otherwise audio K when active;
- any resolved count of `1`: ordinary prepared execution.

The base `_best_of_k_count` is therefore a routing maximum, not the loop count
for every batch. The H3 candidate loop must use the per-batch resolved count.
No mutable "current mode" is stored on the trainer, which prevents stale state
across alternating image/video batches and failed revalidation.

`get_best_of_k_count(args)` and `on_best_of_k_enabled(args)` delegate to the
configuration resolver. `get_best_of_k_option_name(args)` returns a stable H3
configuration label for the base dispatch error path rather than pretending
that the routing maximum belongs to one CLI option. Option-specific validation
errors originate in the resolver and name the exact conflicting option(s).

## 7. Prepared-State Structure

Refactor the H3 preparation boundary so fixed step state is distinct from the
two noisy modality inputs.

The fixed state contains:

- validated runtime packed-batch plan, including one-frame classification and
  any `H3TimeOverrides`;
- clean video and audio latents;
- one base time;
- `sigma_video` and `sigma_audio`;
- model-visible video and audio times;
- augmented visual and audio conditions;
- effective per-sample audio loss weight.

The noisy-input structure contains:

- video noise and noisy video input;
- audio noise and noisy audio input.

The ordinary path prepares both structures once and executes one shared "run
prepared step" helper. Candidate search replaces only the selected stream's
noise/input pair. Image and video modes both replace the video-stream pair, but
the runtime batch resolver determines which named policy and K applies. This
makes fixed-versus-variable state mechanically visible and avoids hidden audio
overrides inside `_call_training_dit`.

`sigma_audio` becomes an explicit fixed-state field. It is required to rebuild
candidate audio inputs as:

```text
noisy_audio = (1 - sigma_audio) * audio_latents
              + sigma_audio * candidate_audio_noise
```

The 4D audio layout is not synthetic image coverage: it is H3's production
`[B,32,2,A]` latent contract and must be exercised directly.

## 8. One Runtime-Mode Candidate Loop

`process_batch_best_of_k` performs these steps:

1. Prepare the fixed H3 state and candidate-zero noisy inputs exactly once.
2. Resolve the batch mode and loop count from the validated runtime layout.
3. If the resolved count is `1`, take the ordinary prepared fallback in Section
   9 before creating a candidate generator.
4. If audio mode has zero effective audio weight, take the audio fallback in
   Section 9 before creating a candidate generator.
5. Choose the searched noise tensor:
   - video or image mode: the training loop's existing video-stream noise;
   - audio mode: the ordinary audio noise drawn during H3 preparation.
6. Create one private, device-local candidate generator from that tensor.
   Candidate zero reuses the ordinary noise; candidates `1..K-1` use the
   private generator and do not advance the default RNG stream.
7. Enter the reentrant block-swap forward-only context once for the candidate
   phase.
8. For each candidate, enter `torch.random.fork_rng` for CPU and the active CUDA
   device, then call the joint H3 DiT under `torch.no_grad()`.
9. Compute both canonical per-sample component losses, but select the score by
   mode:
   - video mode: raw multi-frame video MSE;
   - audio mode: raw audio MSE;
   - image mode: raw one-frame video-stream MSE.
10. Stream per-sample winners through `update_winners`. Do not retain a K-sized
   noise or activation tensor.
11. Rebuild the winning searched input with its fixed per-stream sigma. Keep the
    other modality's candidate-zero noise and input unchanged.
12. Leave forward-only mode and run one gradient-enabled joint H3 forward.
13. Call the ordinary `compute_loss` and return its component metrics plus only
    the active mode's exploration metrics.

Image and video policies share tensor mechanics, not eligibility or metrics.
The loop must not infer image mode merely from `--one_frame`; it uses the
validated runtime layout so alternating image and video batches select their
own configured counts.

Every candidate target stays paired with its candidate noise. Audio mode
changes both noisy audio and the audio target. Video/image mode changes both
noisy video and the video target. The other stream remains at candidate zero.
For image batches, the silence audio placeholder, `audio_present=0`, target
index, one-frame layout, and time override remain identical across candidates.

## 9. Inactive-Batch and Zero-Audio Fallbacks

When global routing is enabled because some configured K is greater than one,
a batch can still resolve to K=1. Examples include a video batch when only
image K is active and an image batch when only audio K is active. That batch:

- does not create a candidate generator;
- does not enter a no-grad candidate loop;
- executes exactly one gradient-enabled joint forward from its already
  prepared candidate-zero state;
- emits ordinary H3 loss metrics only.

Audio best-of-K cannot rank a batch whose effective audio loss weight is zero.
This occurs when the batch has `audio_present=0` or when the configured audio
loss weight is zero.

After ordinary H3 preparation reveals a zero effective weight, the path:

- does not create a candidate generator;
- does not enter a no-grad candidate loop;
- executes exactly one gradient-enabled joint forward from the already
  prepared candidate-zero state;
- computes the ordinary video-only effective loss;
- emits ordinary H3 loss metrics only;
- emits no `h3_audio_best_of_k/` metrics.

It must not call `process_batch` after preparation, because doing so would draw
audio noise, time, and condition augmentation twice. Both ordinary dispatch and
both fallback cases share the same prepared-step helper. For the same initial
RNG state, an inactive batch is numerically and stochastically identical to the
ordinary path.

One-frame image batches always have zero effective audio weight, but image mode
does not take the audio fallback: it ranks the supervised image/video component.

H3 currently enforces `B = 1`, so the effective weight is either active or zero
for the whole batch. Mixed supervised/unsupervised sub-batch search is outside
this revision.

## 10. Loss Semantics

Candidate selection and final optimization are deliberately different:

```text
multi-frame video score = mean(video prediction MSE per sample)
multi-frame audio score = mean(audio prediction MSE per sample)
one-frame image score   = mean(video-stream prediction MSE per sample)

multi-frame final loss  = mean(video MSE
                               + effective_audio_weight * audio MSE)
one-frame final loss    = mean(image/video MSE + 0 * audio MSE)
```

Here "raw" means the canonical unweighted component MSE after ordinary H3 has
constructed the target, including guidance-loss target rewriting when enabled.
It does not mean bypassing the target logic or comparing against an unguided
flow target.

Audio selection uses the raw audio component, not the weighted composite.
For today's `B = 1` and positive fixed weight, multiplying the score by the
weight would not change the winner; keeping it raw makes the contract stable
and the metric interpretable.

Because H3 is joint, changing audio noise can change the predicted video. Audio
mode intentionally ignores that video difference during ranking. Likewise,
the winner under raw audio MSE need not minimize the final composite loss. This
is why the feature remains a modality-focused heuristic rather than Forward XM.

Video mode has the symmetric limitation when audio supervision is active. Image
mode does not: upstream presence gating makes the raw image/video component the
entire effective one-frame objective. The fixed silence audio stream is context,
not a supervised candidate component.

The final forward always preserves ordinary behavior:

- video loss remains active;
- audio loss is multiplied by the effective audio weight;
- both video-path and audio-path trainable parameters receive gradients when
  the model and effective weight make those paths differentiable;
- guidance-distilled video/audio targets, when enabled, are the same targets
  used by ordinary H3 training.

For image mode, the final loss has no audio term. Shared transformer weights can
still be affected through the joint forward, but the silence audio prediction
is not a supervised target.

Candidate non-finite component losses fail fast with mode, candidate index, and
sample indices. They are never treated as losing candidates.

## 11. RNG and Guidance Contract

When all three H3 counts are `1`, the base dispatcher calls ordinary
`process_batch`, creating no candidate seed, fork scope, or exploration metric.
When global routing is active for another batch type, a batch-resolved K of `1`
uses the prepared fallback from Section 9 with the same externally observable
RNG and numeric behavior.

For `K > 1`:

- candidate zero is the ordinary modality noise;
- private generator creation consumes exactly one draw from the appropriate
  default PyTorch stream;
- private candidate draws do not perturb the default stream;
- every candidate and the final recomputation see the same stochastic
  condition state through `fork_rng` replay;
- Python and NumPy RNG state are not manipulated;
- no production call to `torch.clear_autocast_cache()` is introduced.

Guidance-loss probes remain part of `_call_training_dit`. In the candidate
phase, both unconditional and conditional work stay in the reentrant
forward-only block-swap state and under no-grad. For the winner, the temporary
unconditional probe may enter forward-only mode, but the conditional winner
forward runs after training weights are restored and builds the sole backward
graph.

The selected stream's guidance target is recomputed for each correctly paired
input/noise candidate. One-frame candidates retain the same unconditioned
one-frame layout and time override. Fixed conditions and replayed RNG prevent
candidates from competing on dropout or augmentation luck.

Upstream reports that one-frame training effectively requires guidance loss to
avoid rapid de-distillation drift. Image best-of-K does not replace that
recommendation; tests must exercise their composition.

## 12. Metrics and Metadata

Video mode retains:

```text
h3_video_best_of_k/candidate_loss_mean
h3_video_best_of_k/selection_gain
```

Audio mode adds:

```text
h3_audio_best_of_k/candidate_loss_mean
h3_audio_best_of_k/selection_gain
```

Image mode adds:

```text
h3_image_best_of_k/candidate_loss_mean
h3_image_best_of_k/selection_gain
```

`candidate_loss_mean` is the mean raw component selection score over K and B.
`selection_gain` is candidate-zero mean minus selected-best mean. Ordinary
`loss/video` and `loss/audio` continue to describe the final winner forward;
`loss/audio` remains the unweighted component diagnostic while the effective
weight is applied in `loss/current`.

No exploration keys are emitted for a batch-resolved K of `1` or an audio
zero-weight fallback. No H3 mode emits `xm/` keys.

For checkpoint reproducibility, H3 metadata records:

```text
ss_minimax_h3_video_best_of_k = configured video K
ss_minimax_h3_audio_best_of_k = configured audio K
ss_minimax_h3_image_best_of_k = configured image K
```

A single mode/count pair is insufficient because image K can coexist with one
multi-frame policy. The explicit fields also close the existing video K
metadata gap. Existing `ss_minimax_h3_one_frame` remains the provenance flag for
accepting one-frame batches.

## 13. Code Changes

Production changes are limited to:

- `src/musubi_tuner/minimax_h3_train_network.py`
  - audio/image parser options and pure configuration resolver;
  - runtime-layout batch resolver;
  - fixed/noisy state split;
  - one runtime-mode-driven candidate loop;
  - inactive-batch and zero-effective-audio fallbacks;
  - mode-specific logging, metrics, and metadata.
- `src/musubi_tuner/training/trainer_base.py` only for issue-wide correctness
  prerequisites still absent on the current branch, such as reentrant
  forward-only block-swap candidate execution and legacy DiT output
  normalization. These changes must be ported narrowly with current-base tests;
  do not merge the old `qinglong` branch.
- Existing shared best-of-K utilities only when a genuinely mode-neutral helper
  is reused. No second H3-specific winner implementation is added.

Tests and docs:

- `tests/test_minimax_h3_training.py`
- focused shared explorative tests when trainer-base behavior changes;
- `docs/minimax_h3.md`
- `docs/minimax_h3_1f.md`
- `docs/explorative_modeling.md` in both English and Japanese sections
- this design document and the subsequent implementation plan

Upstream already provides one-frame cache and runtime contracts. No dataset,
cache-format, VAE, network, optimizer, or scheduler change is required.

## 14. Test Plan

Floating-point comparisons use explicit tolerances rather than exact equality
when reduction order or autocast can change the last bits:

| dtype | rtol | atol |
|---|---:|---:|
| float32 | `1e-5` | `1e-8` |
| float16 | `1e-3` | `1e-3` |
| bfloat16 | `1e-2` | `1e-2` |

Exact equality is reserved for integer indices, flags, shapes, call counts, and
state that is required to be bitwise unchanged.

### 14.1 Parser and validation

- CLI and TOML expose video, audio, and image K, all defaulting to 1.
- Exact-int/range validation covers zero, float, bool, and string values for
  common, video, audio, and image K.
- Video K and audio K above one fail together before allocation.
- Image K composes successfully with video K and separately with audio K.
- Image K above one without `--one_frame` fails before allocation.
- `--video_only` and audio K above one fail before allocation; image/video K
  remain valid with `--video_only`.
- Common XM above one remains rejected.
- Teacher matching with any H3 K above one remains rejected before allocation;
  cover video and audio directly and image through the upstream one-frame
  incompatibility.
- Failed fresh and repeated validation restores base dispatch state.
- Audio K with global audio weight zero logs the documented no-op warning.
- Startup logs and metadata preserve all configured counts without collapsing a
  valid image-plus-multi-frame configuration into one mode.

### 14.2 Per-batch dispatch and fallback paths

- All three K values at one take the existing ordinary method and preserve RNG,
  loss, gradients, call count, and metric keys.
- In a mixed run with video K=2 and image K=3, alternate real one-frame and
  multi-frame batches and assert the loop counts are 3 and 2 respectively.
- Repeat mixed dispatch with audio K=2 and image K=3.
- When only image K is active, a multi-frame batch performs one gradient forward
  with no candidate generator or image metrics. When only video/audio K is
  active, a one-frame batch does the symmetric prepared fallback.
- Inactive-batch fallback matches ordinary execution exactly for the same RNG
  state; it does not prepare the batch twice.
- For audio K above one, parameterize `audio_present=0` and
  `audio_loss_weight=0`:
  - one gradient-enabled joint forward;
  - zero candidate forwards;
  - no candidate generator/global seed draw;
  - no audio best-of-K metrics;
  - exact agreement with ordinary prepared-step execution.

### 14.3 Audio candidate correctness

- Run CPU and available CUDA coverage through the real
  `MiniMaxH3NetworkTrainer` production path.
- Assert exactly K no-grad candidate conditional forwards and one grad-enabled
  final conditional forward, accounting separately for optional guidance
  probes.
- Assert video noise/input, base time, both model times, visual/audio
  conditions, effective weight, and cached latents are identical across
  candidates.
- Assert 4D audio candidate noise/input differ, have the correct dtype/device,
  and follow the shifted flow formula.
- Assert every audio target is paired with its candidate noise.
- Construct a case where the audio-best candidate is not the video-best or
  composite-best candidate, and assert audio-only selection wins.
- Assert final loss includes ordinary video plus weighted audio components and
  produces finite, nonzero gradients through both modality paths.
- Assert candidate-zero and selection-gain metrics with the table's tolerance
  for the exercised dtype.
- Assert NaN/Inf audio candidate loss fails with candidate/sample diagnostics.

### 14.4 Image and video candidate correctness

- Run image K through the exact `MiniMaxH3NetworkTrainer` production path on CPU
  and available CUDA with target latents `[1,24,1,H,W]`.
- Assert exactly K no-grad candidate forwards and one gradient-enabled winner
  forward, accounting separately for guidance probes.
- Assert video-stream candidate noise/input changes and follows the shifted flow
  formula; every target remains paired with its candidate noise.
- Assert the 4D two-frame silence audio latent/noise/input, `audio_present=0`,
  zero effective weight, base time, both shifted model times, text, one-frame
  layout, `one_frame_target_index`, and `H3TimeOverrides` are fixed across all
  candidates and the winner recomputation.
- Assert image selection ranks raw per-sample video-stream MSE, that this equals
  the effective final one-frame objective within
  `torch.allclose(rtol=1e-5, atol=1e-8)`, and that only
  `h3_image_best_of_k/` exploration keys are emitted.
- Assert finite nonzero gradients through trainable shared/video-path parameters;
  do not require an audio-loss gradient when its effective weight is zero.
- Assert `--h3_video_best_of_k` does not search a one-frame batch and
  `--h3_image_best_of_k` does not search a multi-frame batch.
- Retain multi-frame video K tests proving audio noise/input and weight are fixed,
  selection remains video-only, and the final objective still includes weighted
  audio.
- Assert NaN/Inf image or video candidate losses fail with mode, candidate, and
  sample diagnostics.

### 14.5 Integration regressions

- Reuse the real classic block-swap/ModelOffloader fixture with multi-frame and
  one-frame K=2 cases on available CUDA.
- Exercise nested H3 guidance probes for image mode. Verify every unconditioned
  probe carries the one-frame layout/time override, the forward-only state
  sequence is valid, and the final conditional forward builds the sole graph.
- Verify alternating image/video batches leave no stale runtime mode or count.
- Verify mutual exclusion through the real startup validation order before any
  dataset/session/model sentinel is touched.
- Run focused H3/explorative tests, the full suite with `PYTHONPATH=src`, Ruff
  check, changed-file format check, and `git diff --check`.

The validation record names only the current Python, PyTorch, CUDA, and GPU
environment. It makes no CUDA 12.4 compatibility claim.

## 15. Acceptance Criteria

The revision is complete when:

- required issue-wide best-of-K correctness repairs are present without merging
  unrelated `qinglong` history;
- H3 exposes audio and image K with strict startup validation;
- video and audio best-of-K cannot be active together;
- image K can coexist with at most one active multi-frame policy;
- runtime layout, not the global `--one_frame` flag, selects the per-batch mode;
- video/audio K never searches one-frame batches and image K never searches
  multi-frame batches;
- `video_only` plus active audio best-of-K fails before allocation;
- teacher matching plus any active H3 best-of-K mode fails before allocation;
- audio search varies only 4D audio noise/input and ranks raw per-sample audio
  loss;
- image search varies only one-frame video-stream noise/input, ranks the full
  effective image objective, and keeps the silence audio stream and time
  overrides fixed;
- the fixed video/timestep/condition state is identical across candidates;
- zero-effective-audio batches take the ordinary one-forward fallback without
  search metrics or an extra candidate seed draw;
- one gradient-enabled winner forward optimizes the unchanged joint objective;
- all-K=1 and batch-resolved-K=1 paths preserve ordinary RNG and numerics;
- guidance, autocast, and classic block swap preserve a valid final graph;
- image best-of-K composes with upstream one-frame guidance and time overrides;
- user docs distinguish H3 video/audio heuristics from the closer but still
  experimental Forward-XM-style image selection;
- focused and full tests pass in the current environment.

## 16. Deferred Work

- Joint video-and-audio candidate exploration with a composite selection loss.
- Mixed supervised/unsupervised search after H3 gains real `B > 1` packed
  batching.
- One-frame FL2VA/Ref2VA or multi-target image best-of-K after upstream training
  supports those layouts.
- Batched/chunked K execution and performance characterization against the
  sequential implementation.
- A broader versioned PyTorch/CUDA compatibility matrix.
- Empirical LoRA quality studies comparing video-focused, audio-focused,
  image-focused, and composite selection, including whether image K changes the
  guidance-loss scale or early-drift behavior.
