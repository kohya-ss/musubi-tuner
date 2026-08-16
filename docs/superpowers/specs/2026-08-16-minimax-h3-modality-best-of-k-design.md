# MiniMax-H3 Unified Best-of-K Design

**Status:** Approved design; implementation deferred
**Date:** 2026-08-16
**Branch:** `codex/issue-1019-explorative-modeling`
**Base:** `upstream/dev` at `ca221b10`, merged as `ce0a325`
**Issue:** [kohya-ss/musubi-tuner#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019)

## 1. Decision

MiniMax-H3 uses one best-of-K count and one multi-frame stream selector:

```text
--h3_best_of_k INT
--h3_best_of_k_stream {video,audio}
```

The resolved count defaults to `1`; the stream defaults to `video`.

The stream selector applies only to multi-frame batches. A validated one-frame
batch always searches the video latent stream because its audio stream is a
silence placeholder with zero effective supervision. Image best-of-K is
therefore the video mechanism applied to an `F == 1` target, not a third search
mode or a third count.

The same K applies to every eligible batch in a mixed image/video run. Separate
image and multi-frame K values are deliberately deferred until measurements
show that independent tuning is worth the extra state and compatibility cost.

This document supersedes the MiniMax-H3 best-of-K sections of
`2026-08-09-explorative-modeling-design.md` and the earlier audio-only draft.
The common Forward XM contract remains defined in the 2026-08-09 document.

## 2. Source and Repository Findings

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
  candidate search, and one gradient-enabled winner recomputation. Stochastic
  conditions are replayed across search and recomputation.
- Musubi H3 has one joint transformer and two flow streams. Video latents are
  `[B,24,F,H,W]`; audio latents are `[B,32,2,A]`. One base time produces two
  shifted sigmas, and the ordinary objective is
  `video_mse + effective_audio_weight * audio_mse`.
- Upstream commit `ca221b10` adds experimental one-frame T2VA training. An
  image batch has video latents `[B,24,1,H,W]`, a constant clean silence audio
  latent `[B,32,2,2]`, `audio_present=0`, and an H3 time override from
  `one_frame_target_index`.
- `--one_frame` permits image and video datasets to mix; it does not turn every
  batch into an image batch. Upstream validates the actual runtime layout and
  classifies one-frame targets from `F == 1`.

Current branch behavior also matters for compatibility. After the upstream
one-frame merge, `--one_frame --h3_video_best_of_k K` already runs video-noise
search on one-frame batches. Replacing this with an image-only option would
silently reduce those batches from K candidate forwards to one. The unified
design preserves the existing behavior through a deprecated alias instead.

The paper and official XM code do not define modality-only selection for a
joint video/audio model. H3 stream selection is objective-equivalent to Forward
XM only when the selected raw component is the complete effective objective:

- always for one-frame image batches, because effective audio weight is zero;
- for multi-frame video-stream selection when effective audio weight is zero;
- not for multi-frame video-stream selection with positive audio weight;
- not for multi-frame audio-stream selection, because video loss remains in
  the final objective.

## 3. Goals

- Replace the proposed three-count interface with one K and one multi-frame
  stream selector.
- Preserve existing `--h3_video_best_of_k K` training behavior as a deprecated
  alias, including one-frame batches.
- For multi-frame batches, vary and rank the selected stream only.
- For one-frame batches, always vary video noise and rank raw per-sample
  image/video MSE.
- Hold the other stream, base time, both shifted times, cached data, layout,
  time overrides, augmented conditions, and effective weights fixed.
- Recompute the selected candidate once with gradients and optimize the
  unchanged ordinary H3 objective.
- Keep `_best_of_k_count` as the actual loop count everywhere; do not repurpose
  it as a routing maximum.
- Preserve ordinary control flow, RNG, loss, gradients, metrics, and counters
  at K=1.
- Make zero-effective-audio fallback numerically and stochastically identical
  to one ordinary prepared step.
- Land the existing classic block-swap correctness repair independently before
  extending H3 best-of-K.

## 4. Non-Goals

- Separate K values for image and multi-frame batches.
- A separate image search mechanism or `--h3_image_best_of_k` option.
- Simultaneously exploring video and audio noise in one candidate.
- Selecting multi-frame candidates by composite video-plus-audio loss.
- Enabling common `--xm_best_of_k` for H3.
- One-frame FL2VA or Ref2VA training, control images, or multiple image targets.
- Treating image best-of-K as a replacement for the guidance loss recommended
  by upstream one-frame documentation.
- Lifting H3's current `batch_size = 1` restriction.
- Candidate batching, chunk controls, or retaining K activation graphs.
- Dataset, cache, VAE, scheduler, network, optimizer, or checkpoint-format
  changes.
- Quality claims for H3 LoRA training.
- A PyTorch 2.5.1 or CUDA 12.4 compatibility matrix. Verification covers only
  the current environment and repository tests.

## 5. User Contract

### 5.1 Canonical options

H3 exposes:

```text
--h3_best_of_k INT
--h3_best_of_k_stream {video,audio}
```

`--h3_best_of_k` resolves to `1` when omitted. Runtime validation accepts only
an exact Python `int >= 1`; booleans, floats, strings, zero, and negative values
fail even when TOML loading bypasses `argparse` conversion.

`--h3_best_of_k_stream` defaults to `video`. It chooses the searched stream for
multi-frame batches only. At K=1 it is inert.

Runtime behavior is:

| K | Runtime batch | Configured stream | Search behavior |
|---:|---|---|---|
| 1 | any valid H3 batch | either | ordinary H3 step |
| >1 | multi-frame | video | vary video noise; rank raw video MSE |
| >1 | multi-frame | audio | vary audio noise; rank raw audio MSE |
| >1 | one-frame | video | vary video noise; rank raw image/video MSE |
| >1 | one-frame | audio | vary video noise; rank raw image/video MSE |

The final row is intentional. Searching noise for the unsupervised silence
audio placeholder has no useful objective, so one-frame layout overrides the
multi-frame stream selector.

### 5.2 Compatibility alias

The existing option remains accepted:

```text
--h3_video_best_of_k INT
```

When greater than one, it maps to the same count with stream `video`, emits one
startup deprecation warning, and applies to both multi-frame and one-frame
batches exactly as the current branch does.

The alias is normalized at the input boundary rather than carried into runtime
state:

```text
legacy K > 1 -> count=K, multi_frame_stream=video, warn once
```

It shares the canonical exact-integer contract. An active legacy alias with an
explicit `audio` stream is rejected as contradictory. Legacy K equal to one is
disabled behavior and emits no warning. After normalization, the source
spelling is discarded, so runtime behavior and metrics cannot depend on which
option name the user typed.

`--h3_audio_best_of_k` and `--h3_image_best_of_k` were design-only proposals and
are not introduced as aliases.

Alias removal is deferred and requires a separate compatibility decision.

### 5.3 Other validation

- `--xm_best_of_k` must be exactly `1` for H3.
- `--video_only` with K greater than one and stream `audio` is a startup error.
- `--video_only` with video-stream search remains valid.
- `--h3_teacher_matching` with K greater than one is rejected because its
  teacher-conditioned candidate objective is undefined. Upstream separately
  rejects teacher matching with `--one_frame`.
- `--audio_loss_weight 0` with K greater than one and stream `audio` is allowed.
  Multi-frame batches take the zero-effective-audio fallback. One-frame batches
  still search video noise when `--one_frame` is enabled. Startup logging warns
  that multi-frame audio exploration will not occur.
- With K greater than one, stream `audio`, and `--one_frame` enabled, startup
  logging states that actual one-frame batches override the configured stream
  and search video noise. Dataset composition is not known yet, so this is a
  warning rather than an error.

The different static-zero policies are intentional. `--video_only` is an
explicit declaration that audio supervision and its bookkeeping are disabled,
so requesting audio search contradicts the selected training mode.
`--audio_loss_weight 0` is a sweepable numeric endpoint; keeping that
configuration valid lets one config family cross zero while the exact fallback
avoids meaningless candidate work.

Validation runs before dataset, accelerator, or model allocation. Failed fresh
or repeated validation restores base state to `(count=1, enabled=False)`.

## 6. Configuration and Runtime Resolution

Use one immutable configuration object:

```text
H3BestOfKConfig {
  count: int
  multi_frame_stream: "video" | "audio"
}
```

After alias normalization, it is produced by one pure resolver:

```text
resolve_h3_best_of_k_config(args) -> H3BestOfKConfig
```

The input boundary owns alias mapping and its one warning. The resolver owns
exact-type/range validation, common-XM rejection, the `video_only` conflict,
and teacher-matching rejection. Hooks such as `get_best_of_k_count`,
`get_best_of_k_option_name`, and `on_best_of_k_enabled` consume this resolver
rather than copying conditions.

The base `_best_of_k_count` always equals `config.count`. It is never a routing
maximum or boolean proxy, so existing base consumers retain one meaning.

After ordinary H3 runtime layout validation, a second pure resolver selects the
batch behavior:

```text
resolve_h3_best_of_k_batch(config, runtime_layout)
  -> "image" | "video" | "audio"
```

- validated `F == 1`: `image`;
- multi-frame with configured video stream: `video`;
- multi-frame with configured audio stream: `audio`.

The searched stream is derived where used: only `audio` kind searches audio;
`image` and `video` kinds search video. Returning both kind and stream would
encode the same fact twice and create an avoidable consistency invariant.

No mutable current-mode field is stored on the trainer. Alternating image and
video batches cannot leak a stale kind or stream into the next step.

## 7. Prerequisite: Repository-Wide Block-Swap Bug

The base common-XM loop and the current H3 loop both perform consecutive
no-grad forwards while classic `ModelOffloader` remains in training mode. That
offloader expects a backward pass before the next training forward. With block
swap active, the second candidate can therefore find the first block on CPU and
fail before image or audio support is involved.

This is not H3-local. Every trainer dispatched through the base candidate loop
has the same risk when its transformer enables block swap. The repository-wide
public protocol is implemented by Flux, Flux 2, FramePack, HiDream-O1,
HunyuanVideo, HunyuanVideo 1.5, Ideogram 4, Kandinsky 5, Krea 2, MiniMax-H3,
Qwen-Image, Wan, and Z-Image transformers:

```text
switch_block_swap_for_inference()
switch_block_swap_for_training()
```

The base sampling path already hand-codes this pair, but restoration is after
the sampling body rather than in `finally`. An exception from
`sample_image_inference` can leave the transformer in forward-only mode.

Before unified H3 best-of-K implementation, land a separate correctness commit
that introduces one shared reentrant trainer context and uses it for both
candidate search and sampling:

```text
with trainer block-swap-forward-only context:
    run no-grad candidates, nested guidance, or sampling
restore training block-swap state in finally
run the gradient-enabled winner only after candidate restoration
```

The context contract is:

- unwrap the accelerator model only at the outermost nesting depth;
- invoke the two public transformer methods when both are callable;
- act as a no-op when the model exposes neither method, and fail clearly if it
  exposes only half of the protocol;
- treat method internals as opaque rather than locating or mutating an
  offloader directly;
- switch once at outermost entry, restore once in `finally`, and tolerate nested
  H3 guidance scopes without premature restoration.

Calling only the public protocol matters for delegated models. HiDream-O1's
top-level methods forward to `language_model`; the context must not assume an
offloader is attached directly to the unwrapped top-level module.

Standard XM, H3 candidate search, and `sample_images` reuse this one context.
The separate bugfix receives CPU protocol/no-method/nesting tests, real CUDA
`ModelOffloader` coverage, at least one non-H3 best-of-K block-swap regression,
and a sampling-exception restoration regression. Port only this narrow behavior
onto the current upstream-based branch; do not merge unrelated `qinglong`
history or bundle it into the unified H3 feature commit.

## 8. Prepared-State Boundary

Refactor H3 preparation so fixed step state is separate from the two noisy
stream inputs.

The fixed state contains:

- validated runtime packed layout and one-frame classification;
- clean video and audio latents;
- one base time;
- `sigma_video` and `sigma_audio`;
- model-visible video and audio times;
- augmented visual and audio conditions;
- any `H3TimeOverrides`;
- effective per-sample audio loss weight.

The noisy-input state contains:

- video noise and noisy video input;
- audio noise and noisy audio input.

Preparation happens exactly once per training batch. In addition to drawing
audio noise, time, and condition augmentation, it increments
`_audio_items_seen` and `_audio_supervised_seen` exactly once. Candidate search,
fallback, and winner recomputation must reuse the prepared state and must not
repeat either RNG draws or accounting side effects.

`sigma_audio` becomes explicit so an audio candidate is rebuilt as:

```text
noisy_audio = (1 - sigma_audio) * audio_latents
              + sigma_audio * candidate_audio_noise
```

The 4D audio shape `[B,32,2,A]` is H3's production audio contract, not synthetic
image coverage.

## 9. Candidate Algorithm

For K greater than one, `process_batch_best_of_k` performs these steps:

1. Prepare fixed state and candidate-zero noisy inputs once.
2. Resolve runtime kind from the validated layout; derive audio search only for
   `audio` kind and video search otherwise.
3. If a multi-frame audio selection has zero effective audio weight, take the
   fallback in Section 10 before creating a candidate generator.
4. Choose the searched noise:
   - image or video kind: the training loop's existing video noise;
   - audio kind: the audio noise drawn during H3 preparation.
5. Create one private device-local generator. Candidate zero reuses ordinary
   noise; candidates `1..K-1` use the private generator without advancing the
   default stream.
6. Enter the reentrant forward-only block-swap context once.
7. For each candidate, enter `torch.random.fork_rng` for CPU and the active CUDA
   device, then run the joint H3 DiT under `torch.no_grad()`.
8. Compute the canonical raw per-sample MSE for the selected stream after any
   ordinary guidance-target rewriting.
9. Pass only that selection score to `update_winners` and stream per-sample
   winner noise/indices without retaining K activation tensors.
10. Rebuild the winning selected-stream input with its fixed sigma; keep the
    other stream's candidate-zero noise and input unchanged.
11. Restore training block-swap state and run one gradient-enabled joint H3
    winner forward.
12. Call ordinary `compute_loss` and return its loss/metrics plus the active
    runtime kind's exploration metrics.

Candidate input and target remain paired. Audio search changes both noisy audio
and the audio target. Image/video search changes both noisy video and the video
target. The other stream remains at candidate zero.

For one-frame batches, clean silence audio, its candidate-zero noise/input,
`audio_present=0`, text, target index, packed layout, and time override remain
fixed across all candidates and winner recomputation.

Only the selected score has the candidate fail-fast policy. A non-finite score
raises with runtime kind, candidate index, and sample indices. A non-selected
component is not separately inspected during search; the ordinary final loss
path retains its existing non-finite behavior.

## 10. Zero-Effective-Audio Fallback

Audio-stream selection cannot rank a multi-frame batch whose effective audio
weight is zero. This occurs for `audio_present=0` or global audio loss weight
zero.

After one preparation reveals zero effective weight, the path:

- creates no candidate generator and consumes no candidate seed draw;
- performs zero candidate forwards;
- executes one gradient-enabled joint forward from candidate-zero state;
- computes the ordinary video-only effective objective;
- emits ordinary H3 loss metrics only;
- emits no best-of-K exploration metrics;
- increments `_audio_items_seen` and `_audio_supervised_seen` exactly once;
- matches ordinary prepared-step RNG and numerics for the same initial state.

It must not call `process_batch` after preparation. That would draw audio noise,
time, and augmentations twice and would double-count audio supervision.

One-frame batches do not take this fallback: runtime resolution selects the
supervised video stream before the audio-weight check.

## 11. Selection and Final Loss

```text
multi-frame video score = mean(video prediction MSE per sample)
multi-frame audio score = mean(audio prediction MSE per sample)
one-frame image score   = mean(video prediction MSE per sample)

multi-frame final loss  = mean(video MSE
                               + effective_audio_weight * audio MSE)
one-frame final loss    = mean(image/video MSE + 0 * audio MSE)
```

"Raw" means the canonical unweighted component MSE after ordinary H3 target
construction, including guidance-loss target rewriting when enabled. It does
not bypass target logic.

Audio selection intentionally ignores candidate-induced video differences.
Video selection intentionally ignores candidate-induced audio differences when
audio supervision is active. These cases are modality-focused heuristics.

Image selection and video selection with zero effective audio weight rank the
complete effective objective and are Forward-XM-style objective-equivalent
cases. User-facing text calls the feature "MiniMax-H3 best-of-K" and explains
this condition; it does not unconditionally label video search "not Forward
XM" or emit H3 metrics under `xm/`.

The winner forward always preserves ordinary H3 behavior. Video loss remains
active; audio loss uses its effective weight; guidance-distilled targets remain
ordinary H3 targets; shared trainable parameters receive whatever gradients
the unchanged joint objective provides.

## 12. RNG, Guidance, Metrics, and Metadata

### 12.1 RNG and guidance

At K=1, the base dispatcher calls ordinary `process_batch`. There is no private
generator, fork scope, candidate forward, deprecation warning, or exploration
metric, and RNG/counters match the current path.

At K greater than one:

- candidate zero is ordinary selected-stream noise;
- generator creation consumes exactly one draw from the corresponding default
  PyTorch stream;
- private candidate draws do not perturb the default stream;
- every candidate and final recomputation sees replayed stochastic condition
  state through `fork_rng`;
- Python and NumPy RNG states are not manipulated;
- production code does not call `torch.clear_autocast_cache()`.

Guidance probes stay inside `_call_training_dit`. Candidate unconditional and
conditional work runs no-grad inside the reentrant forward-only block-swap
context. For the winner, any temporary unconditional probe may use a nested
forward-only context, but the conditional winner runs after training weights
are restored and builds the sole backward graph.

One-frame guidance retains the same layout and `H3TimeOverrides`. Upstream
reports guidance loss as effectively necessary for one-frame stability;
best-of-K does not replace it.

### 12.2 Metrics

Metrics identify the runtime selection kind. The already published video
prefix remains canonical for every `video` kind run, including runs configured
with the new option:

```text
h3_video_best_of_k/candidate_loss_mean
h3_video_best_of_k/selection_gain

h3_best_of_k/audio/candidate_loss_mean
h3_best_of_k/audio/selection_gain

h3_best_of_k/image/candidate_loss_mean
h3_best_of_k/image/selection_gain
```

`candidate_loss_mean` is the mean selected raw component score over K and B.
`selection_gain` is candidate-zero mean minus selected-best mean. Ordinary
`loss/video`, `loss/audio`, and `loss/current` continue to describe the final
winner forward.

No exploration keys are emitted at K=1 or on zero-effective-audio fallback.
Canonical H3 configurations emit no `xm/` keys. The metric schema depends only
on runtime kind, never on whether K came from the canonical name or deprecated
alias. Canonical and legacy spellings of the same video run therefore emit the
same two-key set rather than dual-writing metrics.

This deliberately keeps the asymmetric published video prefix while using the
unified prefix for new audio and image kinds. Existing one-frame runs currently
report under the video prefix; after runtime image classification they move to
`h3_best_of_k/image/*`. User documentation includes this one-frame rename note.

### 12.3 Metadata

Checkpoint metadata records:

```text
ss_minimax_h3_best_of_k = resolved K
ss_minimax_h3_best_of_k_stream = configured multi-frame stream
```

Existing `ss_minimax_h3_one_frame` remains the provenance flag for accepting
one-frame batches. Old checkpoints have neither new key and remain loadable.

## 13. Code and Documentation Scope

The independent prerequisite bugfix may modify:

- `src/musubi_tuner/training/trainer_base.py` for the reentrant forward-only
  context, issue-wide candidate-loop integration, and replacement of the
  hand-written sampling switches;
- `src/musubi_tuner/minimax_h3_train_network.py` for H3 candidate integration;
- focused shared, H3, sampling, and non-H3 architecture tests.

The unified H3 feature modifies:

- `src/musubi_tuner/minimax_h3_train_network.py` for parser compatibility,
  immutable config resolution, prepared-state split, runtime-kind resolution,
  candidate search, fallback, metrics, and metadata;
- shared utilities only when a genuinely mode-neutral helper is reused;
- `tests/test_minimax_h3_training.py` and focused shared regressions;
- `docs/minimax_h3.md`, `docs/minimax_h3_1f.md`, and English/Japanese sections
  of `docs/explorative_modeling.md`.

No dataset, cache-format, VAE, scheduler, network, optimizer, or checkpoint-file
format change is required.

## 14. Test Plan

Floating-point comparisons use explicit tolerances when reduction order or
autocast can change the last bits:

| dtype | rtol | atol |
|---|---:|---:|
| float32 | `1e-5` | `1e-8` |
| float16 | `1e-3` | `1e-3` |
| bfloat16 | `1e-2` | `1e-2` |

Exact equality is reserved for indices, flags, shapes, call counts, counters,
RNG states, and tensors required to be bitwise unchanged.

### 14.1 Parser, compatibility, and validation

- Canonical CLI/TOML resolves omitted K to 1 and stream to video.
- Exact-int validation covers zero, float, bool, string, and negative canonical
  and legacy K values before allocation.
- Canonical K selects video or audio stream without mutual-exclusion branches.
- Legacy video K maps to canonical video behavior, warns once only when active,
  conflicts with an explicit audio stream, and leaves no source-spelling field
  in the resolved configuration.
- Common XM greater than one remains rejected.
- Active audio-stream search with `video_only` is rejected before allocation.
- Teacher matching with K greater than one remains rejected.
- Active audio stream with global audio weight zero emits the documented
  multi-frame fallback warning.
- Active audio stream with one-frame support enabled warns that actual image
  batches search video instead.
- Failed fresh and repeated validation restores `(count=1, enabled=False)`.
- `_best_of_k_count` equals the actual global K in every accepted case.

### 14.2 Dispatch, compatibility, and fallback

- K=1 calls the existing ordinary method once and preserves RNG, loss,
  gradients, call count, counters, and metric keys.
- Deprecated video K greater than one searches K candidates for both existing
  multi-frame and one-frame batches, preserving current training behavior.
- Canonical and legacy spellings of the same multi-frame video run emit exactly
  the same published `h3_video_best_of_k/*` exploration-key set.
- Canonical audio stream searches audio for multi-frame batches but video for a
  validated one-frame batch.
- One-frame search emits only `h3_best_of_k/image/*`, independent of CLI
  spelling; no runtime kind dual-writes exploration metrics.
- Alternating one-frame/multi-frame batches use the correct runtime kind without
  mutable state leakage.
- For multi-frame audio selection, parameterize `audio_present=0` and global
  audio weight zero. Assert one preparation, one grad forward, zero candidate
  forwards, no candidate seed draw, no exploration metrics, exact ordinary
  prepared-step agreement, and exactly one increment of both audio counters.

### 14.3 Candidate correctness

- Exercise the exact `MiniMaxH3NetworkTrainer` production path on CPU and
  available CUDA.
- Assert K no-grad conditional candidate forwards and one grad-enabled winner
  forward, accounting separately for guidance probes.
- Audio selection changes only 4D audio noise/input, follows the shifted flow
  formula, pairs each target with candidate noise, holds video state fixed, and
  chooses raw audio-MSE winners even when composite/video winners differ.
- Video selection changes only video noise/input, holds audio state fixed, and
  retains weighted audio in the final objective.
- One-frame selection changes video noise/input while clean/noisy silence audio,
  zero effective weight, text, target index, layout, and time override remain
  fixed. For the same candidate output, `torch.equal` proves its raw video
  selection score is the zero-audio effective per-sample objective.
- Winner assembly uses per-sample selected noise and the common fixed sigma.
- NaN/Inf in the selected score fails with kind/candidate/sample diagnostics.
  A non-selected candidate component is not given an extra fail-fast rule.
- Final trainable-parameter gradients are finite and nonzero where the ordinary
  effective objective makes the tested path differentiable.

### 14.4 Integration regressions

- Before feature work, reproduce the current classic offloader failure and
  verify the separate fix with two candidate forwards plus a final backward on
  real available CUDA. Confirm forward-only mode is restored in `finally`.
- Cover the optional public protocol: neither method is a no-op, half a method
  is an explicit error, and nested contexts switch and restore exactly once.
- Exercise at least one non-H3 base-XM architecture with block swap and verify
  consecutive candidates plus final backward.
- Force `sample_image_inference` to raise and verify sampling still restores
  training block-swap mode.
- Exercise nested H3 guidance probes and assert a valid forward-only state
  sequence plus one final graph under available CUDA autocast.
- Verify one-frame guidance carries the original layout and time override.
- Run focused H3/explorative tests, the full suite with `PYTHONPATH=src`, Ruff
  check, changed-file format check, and `git diff --check`.

The validation record names only the current Python, PyTorch, CUDA, and GPU
environment. It makes no CUDA 12.4 claim.

## 15. Acceptance Criteria

The revision is complete when:

- the classic block-swap candidate bug is fixed and tested in a separate commit;
- H3 exposes one canonical K and one multi-frame stream selector;
- the deprecated video option preserves current multi-frame and one-frame
  search behavior and emits one compatibility warning;
- exploration metrics are determined only by runtime kind: video retains the
  published prefix, while new audio and image kinds use the unified prefix;
- `_best_of_k_count` retains one actual-count meaning;
- actual validated layout, not global `--one_frame`, selects image behavior;
- one-frame batches always search video noise at K greater than one;
- multi-frame batches search exactly the configured stream;
- `video_only` plus active audio-stream search fails before allocation;
- teacher matching plus active best-of-K fails before allocation;
- zero-effective multi-frame audio batches take one ordinary prepared forward
  without a seed draw, candidate metrics, duplicate RNG, or duplicate counters;
- candidate selection validates only the selected raw component score;
- winner recomputation optimizes the unchanged ordinary H3 objective;
- K=1 preserves existing control flow, RNG, numerics, counters, and metrics;
- guidance, autocast, sampling exceptions, and classic block swap preserve a
  valid final graph and restore training state;
- user docs explain when H3 selection is objective-equivalent to Forward XM and
  when it is a modality-focused heuristic;
- focused and full tests pass in the current environment.

## 16. Deferred Work

- Independent image and multi-frame K values, after profiling demonstrates a
  material benefit in mixed training.
- Joint video/audio exploration with a composite selection loss.
- Mixed supervised/unsupervised search after H3 supports real `B > 1` packed
  batches.
- One-frame FL2VA/Ref2VA or multi-target image best-of-K after upstream support.
- Batched/chunked candidate execution and performance characterization.
- A broader versioned PyTorch/CUDA compatibility matrix.
- Empirical LoRA quality studies comparing video-stream, audio-stream, and
  composite selection, including one-frame guidance interactions.
