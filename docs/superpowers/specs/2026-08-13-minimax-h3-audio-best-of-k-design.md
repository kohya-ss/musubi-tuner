# MiniMax-H3 Audio Best-of-K Design

**Status:** Draft for user review  
**Date:** 2026-08-13  
**Branch:** `codex/h3-audio-best-of-k`  
**Base:** `qinglong` at `57e2307`  
**Issue:** [kohya-ss/musubi-tuner#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019)

## 1. Scope

This is a focused extension to
`2026-08-09-explorative-modeling-design.md`. It adds an H3-owned
`--h3_audio_best_of_k` mode beside the existing
`--h3_video_best_of_k` mode. It supersedes only the H3 best-of-K parts of the
earlier design; the common Forward XM contract remains unchanged.

The implementation first preserves the review fixes already landed on this
branch:

- `4d91d85`: reentrant block-swap forward-only candidate execution, legacy DiT
  output normalization, and explicit rejection in custom full-finetune loops.
- `2c4782c`: LongCat runtime entry-point repairs found by the independent
  review.

Audio best-of-K is added on top of those contracts rather than weakening or
duplicating them.

## 2. Source Findings

The design is grounded in these primary sources:

- The [open feature request](https://github.com/kohya-ss/musubi-tuner/issues/1019)
  is broad: support the Explorative Modeling method in Musubi Tuner. It does
  not define H3 modality semantics.
- The [Forward XM paper](https://arxiv.org/abs/2607.27372v1) defines exploration
  as sampling K candidate matches, scoring each candidate against the same
  data, and training only the best match. Its diffusion/flow example holds the
  timestep fixed and varies noise. The
  [official project page](https://explorative-modeling.github.io/) presents the
  same minimal loop.
- The [official `alexiglad/XM` implementation at commit
  `9d06ced6`](https://github.com/alexiglad/XM/tree/9d06ced61e2d2775a34782eb5830584ae4ef6094)
  implements per-sample winner selection in `xm_chunked_best_of_k`, a
  `best_of_k == 1` short circuit, no-grad memory-saving search, and a
  gradient-enabled winner recomputation. It also documents that all stochastic
  conditions must be deterministic across search and recomputation.
- Musubi's H3 implementation has one joint transformer but two flow streams.
  Video latents are 5D `[B,24,F,H,W]`; audio latents are genuinely 4D
  `[B,32,2,A]`. Both sigmas derive from one fixed base time with separate
  shifts, and the ordinary objective is
  `video_mse + effective_audio_weight * audio_mse`.

The paper and official XM repository do not define a video-only or audio-only
selection objective for a joint video/audio model. Therefore both H3 modes are
architecture-specific best-of-K heuristics, not claims of paper-equivalent
Forward XM.

## 3. Goals

- Add `--h3_audio_best_of_k K`, default `1`, to the H3 CLI and TOML parser.
- For `K > 1`, vary only target-audio noise and rank candidates only by raw
  per-sample audio MSE.
- Hold target-video noise/input, base time, both shifted times, cached data,
  augmented conditions, and effective loss weight fixed across candidates.
- Recompute the winning audio candidate once with gradients and optimize the
  unchanged joint H3 objective, including video and weighted audio loss.
- Keep video and audio best-of-K mutually exclusive.
- Reject `--video_only` together with active audio best-of-K before dataset,
  accelerator, or model allocation.
- Fall back to the ordinary single-forward H3 step for a batch whose effective
  audio loss weight is zero, without emitting audio best-of-K metrics.
- Keep `K = 1` on the existing ordinary dispatch with unchanged RNG behavior.
- Use one mode-driven H3 candidate loop rather than parallel video and audio
  implementations.

## 4. Non-Goals

- Jointly varying video and audio noise in one candidate.
- Selecting by the composite video-plus-audio objective.
- Enabling common `--xm_best_of_k` for H3.
- Supporting simultaneous video and audio best-of-K.
- Lifting H3's existing `batch_size = 1` restriction.
- Candidate batching, chunk-size controls, or retaining K activation graphs.
- Making quality claims for H3 LoRA training.
- Adding a PyTorch 2.5.1 or CUDA 12.4 CI/runtime matrix. Verification is limited
  to the current environment and the repository's existing test suite.

## 5. User Contract

### 5.1 CLI

H3 exposes:

```text
--h3_video_best_of_k INT
--h3_audio_best_of_k INT
```

Both default to `1`. Only an exact Python `int >= 1` is accepted after TOML
loading; booleans, floats, strings, zero, and negative values fail validation.
This explicit runtime validation is required because TOML values can bypass
`argparse`'s `type=int` conversion.

The active configuration is resolved as follows:

| Video K | Audio K | Result |
|---:|---:|---|
| 1 | 1 | Ordinary H3 training |
| >1 | 1 | Video-focused best-of-K |
| 1 | >1 | Audio-focused best-of-K |
| >1 | >1 | Startup error naming both options |

Additional validation:

- `--xm_best_of_k` must be exactly `1` for H3.
- `--video_only` with `--h3_audio_best_of_k > 1` is a startup error.
- `--audio_loss_weight 0` with audio best-of-K is allowed, but it means every
  batch takes the zero-effective-weight fallback. Startup logs warn that no
  candidate search or audio best-of-K metrics will occur.
- Video best-of-K remains valid with `--video_only` because its selection
  objective is video loss.

Validation remains in `_validate_args_and_init` before resource allocation.
Any failed or repeated validation leaves the base dispatch state at
`(count=1, enabled=False)`.

### 5.2 Naming

User-visible text must call the new mode "MiniMax-H3 audio-focused best-of-K
heuristic" and state "not Forward XM". It must not be logged under `xm/`.

The startup log states:

- active modality and K;
- selection objective: raw audio loss only;
- final objective: video plus weighted audio;
- sequential memory-saving execution and its operation-count estimate.

## 6. Mode Resolution

Add one pure H3 resolver conceptually equivalent to:

```text
resolve_h3_best_of_k(args) -> (mode: "video" | "audio" | None, count: int)
```

The resolver owns exact-type/range checks, common-XM rejection, mutual
exclusion, and the `video_only` conflict. It has no side effects. The existing
base `_best_of_k_count` and `_best_of_k_enabled` remain the only dispatch state;
the H3 mode is resolved from the already validated arguments when needed.
Avoiding a second mutable mode flag prevents stale H3 state after failed
revalidation.

`get_best_of_k_count(args)`, `get_best_of_k_option_name(args)`, and
`on_best_of_k_enabled(args)` delegate to this single resolver or a non-raising
option-name projection. They must not reproduce independent validation chains.

## 7. Prepared-State Structure

Refactor the H3 preparation boundary so fixed step state is distinct from the
two noisy modality inputs.

The fixed state contains:

- runtime packed-batch plan and clean audio latents;
- one base time;
- `sigma_video` and `sigma_audio`;
- model-visible video and audio times;
- augmented visual and audio conditions;
- effective per-sample audio loss weight.

The noisy-input structure contains:

- video noise and noisy video input;
- audio noise and noisy audio input.

The ordinary path prepares both structures once and executes one shared
"run prepared step" helper. Candidate search replaces only the selected
modality's pair in the noisy-input structure. This makes fixed-versus-variable
state mechanically visible and avoids hidden audio overrides inside
`_call_training_dit`.

`sigma_audio` becomes an explicit fixed-state field. It is required to rebuild
candidate audio inputs as:

```text
noisy_audio = (1 - sigma_audio) * audio_latents
              + sigma_audio * candidate_audio_noise
```

The 4D audio layout is not synthetic image coverage: it is H3's production
`[B,32,2,A]` latent contract and must be exercised directly.

## 8. One Mode-Driven Candidate Loop

For either active H3 mode, `process_batch_best_of_k` performs these steps:

1. Resolve the validated mode and assert that it agrees with the active base K.
2. Prepare the fixed H3 state and candidate-zero noisy inputs exactly once.
3. If audio mode has effective audio weight zero, take the fallback in Section
   9 before creating a candidate generator.
4. Choose the searched noise tensor:
   - video mode: the training loop's existing video noise;
   - audio mode: the ordinary audio noise drawn during H3 preparation.
5. Create one private, device-local candidate generator from that tensor.
   Candidate zero reuses the ordinary noise; candidates `1..K-1` use the
   private generator and do not advance the default RNG stream.
6. Enter the reentrant block-swap forward-only context once for the candidate
   phase.
7. For each candidate, enter `torch.random.fork_rng` for CPU and the active CUDA
   device, then call the joint H3 DiT under `torch.no_grad()`.
8. Compute both canonical per-sample component losses, but select the score by
   mode:
   - video mode: raw video MSE;
   - audio mode: raw audio MSE.
9. Stream per-sample winners through `update_winners`. Do not retain a K-sized
   noise or activation tensor.
10. Rebuild the winning searched input with its fixed per-stream sigma. Keep the
    other modality's candidate-zero noise and input unchanged.
11. Leave forward-only mode and run one gradient-enabled joint H3 forward.
12. Call the ordinary `compute_loss` and return its component metrics plus only
    the active mode's exploration metrics.

The candidate target must stay paired with its candidate noise. In particular,
audio mode changes both the noisy audio input and the flow/audio target passed
through `call_dit`; it never changes only the input.

## 9. Zero-Effective-Audio Fallback

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
fallback instead share the same prepared-step helper, which makes their RNG and
numerics identical for the same initial RNG state.

H3 currently enforces `B = 1`, so the effective weight is either active or zero
for the whole batch. Mixed supervised/unsupervised sub-batch search is outside
this revision.

## 10. Loss Semantics

Candidate selection and final optimization are deliberately different:

```text
video selection score = mean(video prediction MSE per sample)
audio selection score = mean(audio prediction MSE per sample)
final loss            = mean(video MSE
                             + effective_audio_weight * audio MSE)
```

Audio selection uses the raw audio component, not the weighted composite.
For today's `B = 1` and positive fixed weight, multiplying the score by the
weight would not change the winner; keeping it raw makes the contract stable
and the metric interpretable.

Because H3 is joint, changing audio noise can change the predicted video. Audio
mode intentionally ignores that video difference during ranking. Likewise,
the winner under raw audio MSE need not minimize the final composite loss. This
is why the feature remains a modality-focused heuristic rather than Forward XM.

The final forward always preserves ordinary behavior:

- video loss remains active;
- audio loss is multiplied by the effective audio weight;
- both video-path and audio-path trainable parameters receive gradients when
  the model and effective weight make those paths differentiable;
- guidance-distilled video/audio targets, when enabled, are the same targets
  used by ordinary H3 training.

Candidate non-finite component losses fail fast with mode, candidate index, and
sample indices. They are never treated as losing candidates.

## 11. RNG and Guidance Contract

`K = 1` never enters this implementation. The base dispatcher calls ordinary
`process_batch`, creating no candidate seed, fork scope, or exploration metric.

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

The audio guidance target is recomputed for each candidate from that
candidate's correctly paired audio input/noise. Fixed conditions and replayed
RNG prevent candidates from competing on dropout or augmentation luck.

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

`candidate_loss_mean` is the mean raw component selection score over K and B.
`selection_gain` is candidate-zero mean minus selected-best mean. Ordinary
`loss/video` and `loss/audio` continue to describe the final winner forward;
`loss/audio` remains the unweighted component diagnostic while the effective
weight is applied in `loss/current`.

No exploration keys are emitted at K=1 or on zero-effective-audio fallback.
No H3 mode emits `xm/` keys.

For checkpoint reproducibility, H3 metadata records:

```text
ss_minimax_h3_best_of_k_mode = none | video | audio
ss_minimax_h3_best_of_k_count = active K
```

This also closes the existing metadata gap for video best-of-K.

## 13. Code Changes

Production changes are limited to:

- `src/musubi_tuner/minimax_h3_train_network.py`
  - parser option and pure mode resolver;
  - fixed/noisy state split;
  - one mode-driven candidate loop;
  - zero-effective-audio fallback;
  - mode-specific logging, metrics, and metadata.
- Existing shared best-of-K utilities only if a mode-neutral validation helper
  is genuinely reused. No second H3-specific winner implementation is added.

Tests and docs:

- `tests/test_minimax_h3_training.py`
- `docs/minimax_h3.md`
- `docs/explorative_modeling.md` in both English and Japanese sections
- this design document and the subsequent implementation plan

No dataset, cache format, VAE, network, optimizer, sampler, or checkpoint file
format changes are required.

## 14. Test Plan

### 14.1 Parser and validation

- CLI and TOML expose audio K, defaulting to 1.
- Exact-int/range validation covers zero, float, bool, and string values for
  common, video, and audio K.
- Video K and audio K above one fail together before allocation.
- `video_only` and audio K above one fail before allocation.
- Common XM above one remains rejected.
- Failed fresh and repeated validation restores base dispatch state.
- Audio K with global audio weight zero logs the documented no-op warning.

### 14.2 Disabled and fallback paths

- Both H3 K values at one take the existing ordinary method and preserve RNG,
  loss, gradients, call count, and metric keys.
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
- Assert candidate-zero and selection-gain metrics with
  `torch.allclose(rtol=1e-5, atol=1e-8)` or dtype-specific tolerances where
  lower precision is intentionally exercised.
- Assert NaN/Inf audio candidate loss fails with candidate/sample diagnostics.

### 14.4 Integration regressions

- Reuse the real classic block-swap/ModelOffloader fixture with audio K=2 on
  available CUDA.
- Exercise nested H3 guidance probes and verify the forward-only state sequence
  and final gradients.
- Verify mutual exclusion through the real startup validation order before any
  dataset/session/model sentinel is touched.
- Run focused H3/explorative tests, the full suite with `PYTHONPATH=src`, Ruff
  check, changed-file format check, and `git diff --check`.

The validation record names only the current Python, PyTorch, CUDA, and GPU
environment. It makes no CUDA 12.4 compatibility claim.

## 15. Acceptance Criteria

The revision is complete when:

- all review repairs remain present and their regressions pass;
- H3 exposes `--h3_audio_best_of_k` with strict startup validation;
- video and audio best-of-K cannot be active together;
- `video_only` plus active audio best-of-K fails before allocation;
- audio search varies only 4D audio noise/input and ranks raw per-sample audio
  loss;
- the fixed video/timestep/condition state is identical across candidates;
- zero-effective-audio batches take the ordinary one-forward fallback without
  search metrics or an extra candidate seed draw;
- one gradient-enabled winner forward optimizes the unchanged joint objective;
- K=1 stays on the existing ordinary dispatch and preserves RNG behavior;
- guidance, autocast, and classic block swap preserve a valid final graph;
- user docs clearly distinguish both H3 heuristics from Forward XM;
- focused and full tests pass in the current environment.

## 16. Deferred Work

- Joint video-and-audio candidate exploration with a composite selection loss.
- Mixed supervised/unsupervised search after H3 gains real `B > 1` packed
  batching.
- Batched/chunked K execution and performance characterization against the
  sequential implementation.
- A broader versioned PyTorch/CUDA compatibility matrix.
- Empirical LoRA quality studies comparing video-focused, audio-focused, and
  composite selection.
