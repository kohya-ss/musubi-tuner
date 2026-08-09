# Forward XM and MiniMax-H3 Video Best-of-K Support Design

Date: 2026-08-09

Status: Revised draft after current-environment scope clarification

Branch: `codex/issue-1019-explorative-modeling`

Base: `kohya-ss/musubi-tuner@ff8a5c2db832f3d1b458c898e040cfb5d19d0a3d`

## 1. Summary

R1 adds opt-in Forward Explorative Modeling (Forward XM) to compatible
`NetworkTrainer`-based LoRA training entry points. MiniMax-H3 uses the same
best-of-K execution machinery for a separately named video-focused heuristic;
that H3 mode is not Forward XM as defined by the paper because its selection
score differs from its final training objective.

Users enable paper-defined Forward XM with:

```text
--xm_best_of_k K
```

MiniMax-H3 uses a separate, honestly named option:

```text
--h3_video_best_of_k K
```

The default is `K = 1`, which takes the existing training path without extra
random draws, forwards, validation restrictions, or logging, and keeps the same
mathematical loss objective. For `K > 1`, each sample keeps its clean
data/latent, timestep, text/image
conditioning, and condition-drop decision fixed while exploring K independent
noise candidates. Standard trainers measure candidate quality with the
architecture's actual per-sample training loss. MiniMax-H3 instead varies video
noise only and ranks candidates by video loss while retaining weighted audio
loss in the final update; Sections 3 and 11 define that distinct contract. Only
the selected candidate for each sample is recomputed with gradients and used
for the optimizer update.

R1 deliberately uses a sequential memory-saving implementation as the
correctness reference. It performs K no-grad candidate forwards and one
gradient-tracked winner forward, while retaining only the current noise and
per-sample winners. This preserves the ordinary microbatch shape and
approximately the ordinary activation-memory footprint. Official-style
candidate batching is deferred because several current trainers consume
variable-length lists, per-sample loops, or packed index structures that cannot
be expanded by a generic `repeat_interleave` rule. Section 5 names the concrete
contracts rather than treating this as an unspecified compatibility risk.

R1 does not add the official XDiT or XJumpy architectures, Reverse XM, or
end-to-end one-step generation. Sampling and inference are unchanged.

## 2. Source Anchors

Request:

- Issue 1019: <https://github.com/kohya-ss/musubi-tuner/issues/1019>

Method sources:

- Paper v1, submitted 2026-07-29: <https://arxiv.org/abs/2607.27372v1>
- Project page: <https://explorative-modeling.github.io/>
- Official Apache-2.0 code: <https://github.com/alexiglad/XM>
- Reviewed official code commit:
  `9d06ced61e2d2775a34782eb5830584ae4ef6094`
- Official core helper: `model/model_utils.py::xm_chunked_best_of_k`
- Official image integration: `model/img/dit_cc.py`
- Official video integration: `model/vid/dit_wm.py`
- Official flow objective: `model/flow/flow_matching.py`

Current validation runtime, recorded on 2026-08-09:

- Python 3.10.11.
- PyTorch `2.13.0+cu130` with `torch.version.cuda == "13.0"`.
- NVIDIA GeForce RTX 4090, compute capability 8.9.

R1 validates this environment and the repository's existing tests. It does not
create or claim compatibility for a separate runtime.

Musubi repository contracts:

- `src/musubi_tuner/training/trainer_base.py`
- `src/musubi_tuner/training/parser_common.py`
- `src/musubi_tuner/training/timesteps.py`
- Architecture-specific `src/musubi_tuner/*_train_network.py` trainers

The implementation will be repository-native rather than a copy of the
official helper. The official code is the behavioral reference for candidate
selection, fixed conditioning, memory-saving recomputation, and the `K = 1`
meaning.

## 3. Research Findings

### 3.1 Method semantics

For data sample `x`, candidate generations `y_i`, and reconstruction loss `J`,
Forward XM uses the hard best-of-K objective:

```text
L(theta) = min_i J(y_i, x), i in {1, ..., K}
```

For diffusion and flow hybrids in the paper, candidates differ only in input
noise. The data sample, timestep, semantic condition, and guidance condition
drop are shared. The minimum is selected per data sample, not once for the
whole microbatch. Only the selected candidate receives gradients.

Forward XM is coverage-oriented. The paper's clean maximum-likelihood argument
applies to a smooth mixture form; the implemented hard minimum differs from
that form and becomes a support-coverage objective as K grows. R1 therefore
describes the feature as Forward XM best-of-K training, not as an exact
likelihood implementation.

MiniMax-H3 does not satisfy that Forward XM objective under R1. It explores
video-noise candidates while keeping audio noise fixed, ranks them only by
video reconstruction loss, then trains the winner with the full weighted
video-plus-audio loss. Because H3's joint transformer can change its audio
prediction when video input changes, the video-best candidate need not minimize
the loss used for the update. The H3 mode is therefore named the
**video-focused best-of-K heuristic**, not Forward XM or a projection of XM. It
may trade composite-objective alignment for stronger video-loss selection; R1
makes no claim that this improves video or multimodal quality.

Reverse XM searches over data targets instead of generated candidates. It is
mode-seeking and needs an entropy or coverage mechanism to avoid collapse. It
requires a condition-aware data search contract that Musubi does not have, so
it is outside R1.

### 3.2 Official implementation behavior

The official helper:

- Tiles K candidates over the batch dimension, optionally in chunks.
- Reduces the reconstruction loss to one scalar per original sample and
  candidate before selecting the minimum.
- Holds CFG drop decisions fixed across candidates.
- In memory-saving mode, evaluates candidates without gradients, stores the
  winning random inputs, then reruns only the winners with gradients.
- Explicitly clears the PyTorch autocast cache before the gradient-tracked
  recomputation.
- In FLOP-efficient mode, retains all candidate graphs and backpropagates the
  selected losses.
- Treats `xm_best_of_k = 1` as baseline training.

The official job scripts use memory-saving mode for released image and video
runs. The helper also supports candidate chunking, but its generic condition
tiling only covers the data structures used by that repository. The explicit
autocast-cache clear is an implementation detail, not a portable XM invariant:
PyTorch itself clears the cache when the outermost autocast context exits, and
Musubi's architecture `call_dit` methods enter and exit autocast per forward.
R1 therefore does not copy that call without a failing reproducer.

### 3.3 Compute and memory

For a transformer, the paper approximates a forward as one third of a standard
forward-plus-backward training step. For `K > 1`:

- FLOP-efficient XM costs approximately `(K + 2) / 3` standard steps and stores
  K candidates' activations.
- Memory-saving XM adds a winner recomputation and costs approximately
  `(K + 3) / 3` standard steps while keeping ordinary activation memory.

R1 chooses the second trade-off. Sequential execution has the same theoretical
FLOP count as batched memory-saving execution, but lower accelerator occupancy
and may therefore have lower wall-clock throughput. The paper's forward-cost
approximation gives these concrete idealized ratios relative to ordinary
training:

| K | Memory-saving XM compute ratio |
| --- | --- |
| 2 | `5 / 3`, approximately 1.67x |
| 4 | `7 / 3`, approximately 2.33x |
| 8 | `11 / 3`, approximately 3.67x |

Those are operation-count estimates, not wall-clock measurements. Kernel
utilization, model shape, block swap, and hardware determine the additional
sequential-execution penalty. R1 documentation must report measured step time
when hardware evidence is available and otherwise say that wall-clock overhead
is unknown; it must not invent a universal 2x or 3x slowdown.

### 3.4 Evidence limits

The paper reports pretraining experiments on class-conditional image models,
goal-conditioned video models, masked language models, and control tasks. It
does not establish that Forward XM improves:

- LoRA fine-tuning of a pretrained text-to-image or text-to-video model.
- Small personal datasets.
- Low-rank adapters with limited trainable capacity.
- Musubi's timestep samplers, loss weighting, block swap, or gradient
  accumulation combinations.

The paper also states that losses are not comparable across K and reports that
existing guidance methods transfer unevenly. R1 must not promise quality,
convergence, or efficiency gains. Its acceptance criteria cover semantic and
runtime correctness; downstream quality remains an experiment.

## 4. First-Principles Requirements

The feature is correct only if all of these invariants hold:

1. A candidate search changes only the latent noise assignment.
2. Every candidate for one sample uses the same clean data/latent, timestep,
   condition, condition-drop decision, and other stochastic training choices.
   A flow velocity or noise-prediction target may change as the deterministic
   counterpart of the candidate noise; it must remain paired with that noise.
3. Paper-defined Forward XM selects with the same per-sample weighted
   reconstruction objective that the final update uses.
4. MiniMax-H3's separately named video-focused heuristic selects with video
   loss while its final update retains the full video-plus-audio objective. The
   mismatch must remain explicit in code, metric names, startup logs, and docs.
5. Selection happens independently for each sample in the microbatch.
6. Only selected candidates contribute gradients.
7. `K = 1` preserves the existing control flow and random-number consumption.
8. Gradient accumulation and DDP keep their existing optimizer and reduction
   semantics.
9. Trainer modes whose complete candidate state or per-sample objective is
   unavailable fail before dataset or model allocation rather than silently
   applying a partial XM objective.

These requirements take precedence over matching the structure of the official
helper.

## 5. Approaches Considered

### 5.1 Sequential memory-saving search (selected)

Run one ordinary-shaped candidate at a time under `torch.no_grad()`, retain the
best noise per sample, and rerun the assembled winners once with gradients.

Advantages:

- Constant activation memory and constant candidate-noise storage.
- No generic replication of heterogeneous batch values.
- Reuses every architecture's existing `call_dit` implementation.
- Compatible by construction with block swap, gradient checkpointing, and
  variable-length condition lists.
- Smallest behavioral change to `trainer_base.py`.

Costs:

- Lower throughput than folding candidates into one larger batch.
- One extra winner forward compared with the all-grad mode.

### 5.2 Official-style chunked candidate batching (deferred)

Repeat latents and all conditions by a configurable candidate chunk size and
run larger forwards.

This can improve device utilization, but the current entry points do not share
one batch-expansion contract:

- Wan passes `batch["t5"]` as a list of per-sample, potentially variable-length
  tensors. Repeating tensor dimension zero does not repeat that list in the
  required candidate-major order.
- Kandinsky 5 explicitly loops over `b`, indexes text, pooled, mask, and visual
  condition values per sample, then creates a stochastic frame mask inside that
  loop. Candidate batching must expand all of those values and preserve one
  replayed mask per original sample.
- FramePack passes `latent_indices`, `clean_latent_indices`, optional 2x/4x
  clean-latent index tensors, packed Llama states, clean latents, and image
  embeddings together. These index-bearing structures must be expanded as one
  architecture-owned unit; blindly repeating tensors independently does not
  prove that their packed references remain aligned.
- MiniMax-H3 has a joint video/audio candidate state with different shifted
  sigmas and condition augmentation. It cannot be represented by repeating only
  the base video latent and noise.

A future batched implementation therefore needs an architecture-owned
`expand_candidate_batch` contract plus peak-memory tuning; it is not a generic
tensor tiling optimization. Sequential execution is the R1 semantic reference,
not a claim that batching is unimportant. The optional hardware characterization
in Section 19 records the actual cost and can motivate that follow-up.

### 5.3 Per-architecture copies versus one H3-owned loop

Copying the entire candidate loop into every trainer is rejected. It would
duplicate timestep freezing, RNG replay, winner selection, validation, and
metrics across more than ten trainers, then make `K = 1` compatibility harder
to prove.

A generic callback-driven search framework is also rejected in R1. There are
only two materially different consumers, and callbacks would hide the important
difference between standard full-objective Forward XM and H3's video-only
selection score. The base trainer owns the standard loop; MiniMax-H3 owns one
explicit local loop and shares only narrow pure mechanics such as candidate
generator setup and `update_winners`. This small duplication keeps each data
flow readable without creating an abstraction for hypothetical consumers.

## 6. R1 Goals

- Add Forward XM to compatible `NetworkTrainer` LoRA entry points and the
  separately identified MiniMax-H3 video-focused best-of-K heuristic.
- Match the paper's fixed-data, fixed-timestep, fixed-condition semantics while
  keeping each derived prediction target paired with its candidate noise.
- Select Forward XM winners per sample with the real weighted training loss.
- Let MiniMax-H3 rank video-noise candidates by video loss without presenting
  that mismatched selection objective as paper-defined XM.
- Keep activation memory near the baseline through no-grad search and winner
  recomputation.
- Preserve baseline control flow, RNG consumption, and mathematical objective
  when disabled; reduction-order comparisons use the declared tolerance.
- Work with microbatch sizes greater than one, gradient accumulation, DDP,
  gradient checkpointing, compilation, and block swap through ordinary-shaped
  forwards.
- Provide metrics that prove candidate search is active without claiming a
  cross-K loss comparison.
- Reject trainer modes whose candidate state or objective cannot satisfy the R1
  contract before expensive allocation.
- Document the compute cost and the lack of LoRA quality evidence.

## 7. R1 Non-Goals

- Reverse XM.
- Official XDiT, XJumpy, XMDLM, policy, or world-model architecture loading.
- End-to-end one-step generation.
- Full-transformer fine-tuning scripts with independent training loops.
- Candidate batching or `xm_chunk_bs_mult`.
- An all-grad/FLOP-efficient mode.
- Full joint video/audio MiniMax-H3 candidate exploration; its R1 heuristic
  varies video noise and ranks video loss only.
- Candidate caches, vector databases, gradient-based latent search, or soft-min
  objectives.
- Changing inference, samplers, guidance, checkpoint formats, cache formats, or
  LoRA network structure.
- Recommending a universal K.
- Claiming FID, FVD, convergence, data-efficiency, or generalization gains for
  LoRA fine-tuning.

## 8. Supported Trainer Matrix

R1 supports trainers that satisfy the standard one-noise/base-flow Forward XM
contract. It also provides an explicit `process_batch_best_of_k` adapter for H3's
separately named video-focused heuristic. The relevant selection objective must
be available per sample. A trainer is not rejected merely because it overrides
`compute_loss`.

| Entry point | R1 status | Contract or concrete blocker |
| --- | --- | --- |
| `flux_2_train_network.py` | Supported | Base weighted flow MSE |
| `flux_kontext_train_network.py` | Supported | Base weighted flow MSE |
| `fpack_train_network.py` | Supported | Base weighted flow MSE |
| `hv_train_network.py` | Supported | Base weighted flow MSE |
| `hv_1_5_train_network.py` | Supported | Base weighted flow MSE |
| `ideogram4_train_network.py` | Supported | Explicit unweighted per-sample MSE hook |
| `kandinsky5_train_network.py` | Supported | Base weighted flow MSE; replay visual-condition RNG |
| `krea2_train_network.py` | Supported | Base weighted flow MSE |
| `qwen_image_train_network.py` | Supported | Base weighted flow MSE |
| `wan_train_network.py` | Supported | Base weighted flow MSE; freeze high/low timestep choice |
| `zimage_train_network.py` | Supported | Base weighted flow MSE |
| `flux_2_train_network_self_flow.py` | Supported when `--self_flow` is off; rejected when on | This is a working vanilla Flux 2 mode: the first branch of `process_batch` immediately returns `super().process_batch(...)`. Only the enabled Self-Flow branch reaches its `NotImplementedError`; that unfinished branch requires two timesteps, teacher/student forwards, a per-token mask, EMA weights, and `L_gen + gamma * L_rep`. |
| `hidream_o1_train_network.py` | Rejected for `K > 1` | It always applies sigma-dependent noise scaling outside the shared affine formula; dev mode may also derive clipping from one standard deviation over the candidate tensor, so assembled winners can change the transform. The optional DINO backend returns a batch-reduced scalar rather than `[B]`. |
| `minimax_h3_train_network.py` | Supports `--h3_video_best_of_k`; rejects Forward XM | Vary and rank video noise only. Sample audio noise, the shared base time, separate video/audio sigmas, and condition augmentation once. Final recomputation keeps the selected video noise, fixed audio state, and ordinary `video_loss + weight * audio_loss` update. This is a video-focused heuristic, not Forward XM. |

`K = 1` remains valid for every trainer and mode, including the rejected rows,
because it does not enter best-of-K code.

The remaining exclusions are semantic, not permanent architecture bans.
HiDream can be added after defining a stored candidate-noising state and a
per-sample DINO objective. Self-Flow first needs a complete enabled training
step, then an explicit definition of which teacher/student random variables XM
explores.

The standalone full-finetune entry points such as `hv_train.py`,
`qwen_image_train.py`, `zimage_train.py`, and `hidream_o1_train.py` do not use
the shared `NetworkTrainer` loop and are not modified in R1.

## 9. CLI Contract

Add one common training argument in `training/parser_common.py`:

```text
--xm_best_of_k INT
```

Contract:

- Default: `1`.
- Valid range: integer `>= 1`.
- `1`: XM disabled, existing non-XM path and mathematically equivalent loss.
- `> 1`: sequential memory-saving Forward XM.
- Available through command line and TOML config like other common arguments.

MiniMax-H3 adds its own model-specific argument:

```text
--h3_video_best_of_k INT
```

Its default and valid range are also `1` and integer `>= 1`. A value greater
than one enables the video-focused heuristic described in Section 11.3. H3
rejects `--xm_best_of_k > 1` rather than silently assigning the Forward XM name
to a different objective. Non-H3 trainers do not expose the H3-specific option.

R1 does not expose `--xm_save_mem_mode`, `--xm_chunk_bs_mult`, or
`--xm_debug_mode`. There is only one execution strategy, so additional controls
would imply unsupported behavior.

Validation runs once from `_validate_args_and_init` after model-specific argument
handling and before dataset, accelerator, or model construction. A neutral
`get_best_of_k_count(args)` hook returns `args.xm_best_of_k` in the base trainer
and `args.h3_video_best_of_k` in H3 after rejecting XM there. Validation checks
`K >= 1`, stores `self._best_of_k_count`, and sets
`self._best_of_k_enabled = K > 1`. Only when enabled does it ask the trainer for
an actionable incompatibility reason. The base capability check rejects a
custom `process_batch` unless that trainer explicitly confirms that the shared
Forward XM data flow is equivalent. It does not reject a trainer solely for
overriding `compute_loss`; the canonical per-sample primitive in Section 14
prevents that loss-specific special case.

## 10. Code Organization

Add:

```text
src/musubi_tuner/training/explorative.py
tests/test_explorative_modeling.py
docs/explorative_modeling.md
```

`training/explorative.py` owns pure, model-independent mechanics:

- Candidate noise generator creation.
- Per-sample winner-state initialization and updates.

The pure update operation has this contract:

```text
update_winners(
    best_losses,
    winner_noise,
    winner_indices,
    candidate_losses,
    candidate_noise,
    candidate_index,
) -> (best_losses, winner_noise, winner_indices)
```

It validates loss finiteness/shape and noise shape/dtype/device before returning
updated tensors; it owns no trainer or callback state.

The module does not own a callback-based search loop. The standard and H3 paths
invoke `torch.random.fork_rng` directly so their different noising and scoring
contracts remain visible at the call site. Each path also accumulates its two
summary metrics directly; two scalar reductions do not justify a framework.

Modify `training/timesteps.py` to add one pure
`get_noise_coefficients_from_timesteps` helper. It returns the already sampled,
broadcast noising coefficient; it does not sample a timestep or own a trainer
virtual method.

Modify `training/trainer_base.py` to own trainer-aware integration:

- Best-of-K option and compatibility validation.
- The initialized neutral `_best_of_k_count` and `_best_of_k_enabled` dispatch
  state.
- `compute_per_sample_loss`, the canonical per-sample objective from which the
  ordinary scalar loss is derived.
- A base `process_batch_best_of_k` implementation for standard Forward XM.
- Standard-flow candidate scoring and the final winner forward.
- One standard-versus-best-of-K dispatch branch in the training loop.

Modify `ideogram4_train_network.py` to move its unweighted MSE into
`compute_per_sample_loss`; its scalar wrapper calls that primitive and only adds
the existing diagnostics.

Modify the Self-Flow and HiDream-O1 trainers only to return precise mode-specific
incompatibility reasons.

Modify `minimax_h3_train_network.py` to:

- Add the model-specific `--h3_video_best_of_k` option, reject
  `--xm_best_of_k > 1`, and override `get_best_of_k_count`.
- Override `process_batch_best_of_k` with an explicit H3-owned candidate loop.
- Prepare one fixed audio/timestep/condition state for all video candidates.
- Factor video and audio per-sample losses through one private component helper,
  using video loss for selection and a separate shared combiner for the final
  weighted total.

Extend `tests/test_minimax_h3_training.py` for H3 component and final-gradient
coverage in addition to the model-independent cases in
`tests/test_explorative_modeling.py`.

No new minimum-version CI workflow or environment-bootstrap script is added.
The focused tests run in the current validation runtime recorded in Section 2
and print its Python, torch, CUDA, and device versions in captured test evidence.

No model package, root entry point, dataset, cache, network, optimizer, or
checkpoint module changes.

## 11. Training Data Flow

### 11.1 Disabled path

At initialization, the active trainer's best-of-K value of 1 sets
`self._best_of_k_enabled` to false. The current loop then takes the standard
dispatch arm:

1. Scale/shift latents.
2. Draw one `torch.randn_like(latents)` noise tensor.
3. Call `process_batch` once.
4. Backpropagate the returned scalar.

There are no nested or helper-level `K == 1` checks. Compatibility checks are
skipped at initialization when `_best_of_k_enabled` is false. The standard arm
creates no candidate seed, RNG scope, winner state, or exploration metrics. This
is required for random-stream compatibility. The per-sample loss refactor may
change the final reduction's last bits; Section 19 defines the accepted numeric
tolerance rather than claiming bitwise identity.

### 11.2 Standard Forward XM path

For `K > 1`:

1. Reuse the training loop's first noise tensor as candidate zero.
2. Call the existing `get_noisy_model_input_and_timesteps` once with candidate
   zero. This samples the microbatch timesteps and preserves architecture-owned
   choices such as Wan high/low model selection.
3. Derive one broadcast sigma tensor from those sampled timesteps.
4. Seed a dedicated per-step generator for candidates 1 through `K - 1` from
   the seeded training RNG. It does not advance the default RNG while drawing
   candidate tensors.
5. Treat the default PyTorch RNG state after timestep and candidate-generator
   setup as the stochastic-condition state.
6. For each candidate:
   - Use candidate zero's already constructed input, or build a later
     candidate's input directly from the fixed sigma and new noise.
   - Enter `torch.random.fork_rng` for the active accelerator device. Because
     the context restores CPU and device states on exit, every candidate starts
     from the same condition state.
   - Call `call_dit` under `torch.no_grad()`.
   - Compute a finite selection-score vector of shape `[batch_size]` with
     `compute_per_sample_loss`.
   - Update the winning loss, index, and noise independently per sample.
7. Assemble one winner-noise tensor by replacing only the improved samples as
   candidates stream through memory, then apply the fixed per-sample sigma:

   ```text
   improved_mask = improved.view(batch_size, 1, ..., 1)
   winner_noise = where(improved_mask, candidate_noise, winner_noise)
   winner_input = (1 - sigma) * latents + sigma * winner_noise
   ```

   This is memory-equivalent to assigning
   `winner_noise[b] = candidate_noise[winner_indices[b]][b]` without retaining a
   K-sized candidate tensor. The implementation includes a short comment that
   candidates may be mixed across samples safely because sigma is per sample
   and was held identical across candidates.
8. Call `call_dit` with gradients outside the forked candidate scopes. The
   default RNG is still at the shared condition state, so this forward advances
   it exactly once. Do not call `torch.clear_autocast_cache()`.
9. Use the trainer's ordinary scalar
   `compute_loss` result and metrics.
10. Continue through the existing `accelerator.backward`, DDP reduction,
   clipping, optimizer, scheduler, and zero-grad path.

The final microbatch may combine sample 0's candidate 3, sample 1's candidate 0,
and sample 2's candidate 2. Selecting one candidate index for the entire batch
is incorrect.

### 11.3 MiniMax-H3 video-focused best-of-K path

MiniMax-H3 overrides `process_batch_best_of_k` because its ordinary
`process_batch` owns joint video/audio preparation. For
`h3_video_best_of_k > 1` it:

1. Builds `_runtime_batch_plan` once and preserves the existing batch-size-one
   rule.
2. Samples one `audio_noise` and one base time, then derives the existing
   separately shifted `sigma_video` and `sigma_audio` once.
3. Builds `noisy_audio`, condition seeds, visual conditions, audio conditions,
   and effective audio-loss weights once outside the candidate loop.
4. Runs its own sequential candidate loop. Each iteration draws only video
   noise, enters `torch.random.fork_rng` directly, builds `noisy_video`, calls
   the joint DiT under `torch.no_grad()`, and uses only the `[B]` video-loss
   component as the selection score.
5. Calls the shared pure `update_winners` helper to validate and stream the
   winning video noise without retaining a K-sized candidate tensor. It
   accumulates its candidate-score metrics locally.
6. Recomputes once with the selected video noise and all fixed audio/condition
   state, then applies the ordinary total loss
   `video_loss + audio_loss_weight * audio_loss` and ordinary component metrics.

The fixed audio input and target do not imply a fixed audio loss: H3 is a joint
model, so its audio prediction may respond to a changed video input. The audio
component is intentionally ignored for candidate ranking but remains in the
gradient-tracked final objective. No candidate audio noise is generated, and
audio supervision is never disabled by the heuristic. Because its selection
score differs from that final objective, this path must not use XM terminology
in user-visible logs, metrics, or documentation.

## 12. Timestep and Noising Contract

The existing `get_noisy_model_input_and_timesteps` method currently samples a
timestep and constructs the corresponding noisy input together. Calling it K
times would let candidates compete on timestep difficulty and violate the
paper.

Do not refactor that baseline method or add a trainer virtual method merely to
apply the flow formula. Call it once for candidate zero, retain its sampled
timesteps, and add this pure helper in `training/timesteps.py`:

```text
get_noise_coefficients_from_timesteps(
    timestep_sampling, noise_scheduler, timesteps, device, n_dim, dtype
) -> Tensor[B, 1, ...]
```

The helper returns coefficients only. Candidate inputs are constructed inline:

```text
x_t = (1 - sigma) * x + sigma * noise
```

For sampling modes whose model-visible timestep is `1000 * sigma + 1`, recover
the noising coefficient as:

```text
sigma = clamp((timestep - 1) / 1000, 0, 1)
x_t = (1 - sigma) * x + sigma * noise
```

For scheduler-indexed modes, reuse `get_sigmas` with the fixed timestep and the
existing scheduler. Define this exact shared constant in `training/timesteps.py`
using strings because the current parser has no timestep enum:

```text
BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS = frozenset({
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "qwen_shift",
    "krea2_shift",
    "ideogram4_shift",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
    "flux2_shift",
})
```

Both the baseline `get_noisy_model_input_and_timesteps` branch and
`get_noise_coefficients_from_timesteps` must test membership in this one
constant. The helper must not copy the current eleven-way `if` chain. Existing
architecture-local timestep-convention subsets in HiDream and Ideogram are not
silently widened as part of this refactor.

This is intentionally smaller than the original proposed
`get_noisy_model_input_from_timesteps` seam: the baseline method keeps its
sampling, arithmetic, and return contract. Its inline explicit-sampler list may
be replaced mechanically by the shared constant used by the coefficient helper,
while the reusable value XM actually needs is made explicit.
HiDream's candidate-local clipping/scaling transform does not fit this pure
coefficient contract and is handled as an explicit R1 incompatibility.

Tests must prove that reconstructing candidate zero from the helper is close to
the input returned by the baseline method for explicit-coefficient and
scheduler-indexed samplers, using the explicit per-dtype tolerances in Section
19.5 rather than bitwise equality or framework defaults. They must also prove
that all candidates receive identical timesteps, including Wan high/low
training and distribution-preserving timestep sampling.

## 13. RNG and Condition Contract

Neither Forward XM nor the H3 heuristic may select a candidate because it
received more conditioning, a different dropout mask, or a different stochastic
augmentation.

The candidate search uses `torch.random.fork_rng` with only the active
accelerator device listed. PyTorch always forks the CPU RNG as part of that
context, so this covers the two RNG streams that supported training forwards can
currently consume without hand-written state containers. Do not capture Python
`random` or NumPy state: an audit of supported `call_dit` training paths found no
such use, and speculative capture would add a contract for code that does not
exist.

No-grad candidate forwards run inside separate forked scopes and leave the
default CPU/device streams at the shared condition state. The final
gradient-tracked winner forward runs once outside those scopes, starts from that
state, and advances it once, matching the semantics of one ordinary stochastic
model forward after candidate-generator setup.

This fixes Kandinsky 5 visual-condition frame selection, which currently uses
`torch.rand` on `cond_lat.device`, and covers model dropout. If a future
supported trainer introduces Python or NumPy randomness inside its forward, its
best-of-K compatibility contract and tests must explicitly add the required state;
R1 does not prepay that complexity. It also does not try to make
nondeterministic accelerator kernels bitwise deterministic. Candidate selection
is based on the no-grad scores; the final recomputation trains the selected
latent even if low-level numerical nondeterminism changes its loss slightly.

## 14. Loss Contract

### 14.1 Canonical per-sample objective

Make `compute_per_sample_loss(args, output, timesteps, noise_scheduler,
dit_dtype, network_dtype, global_step)` the canonical objective primitive. Its
output is exactly shape `[batch_size]`. Do not add an XM-specific duplicate loss
method.

The base implementation:

1. Computes elementwise MSE between `DiTOutput.pred` and `DiTOutput.target`.
2. Applies the existing SD3 loss weighting before reduction.
3. Averages every non-batch dimension.

The base `compute_loss` becomes a stable wrapper that returns
`compute_per_sample_loss(...).mean()` plus its metrics dict. The XM search calls
the primitive before the final mean. This leaves one weighted-MSE formula, so a
future weighting change cannot silently diverge between ordinary and XM paths.
Averaging the whole batch before `min` is still incorrect.

Ideogram 4 overrides only the objective primitive for its unweighted MSE. Its
scalar `compute_loss` wrapper obtains the mean from that primitive and adds the
existing prediction/target diagnostics; it does not restate the loss formula.
Architecture-specific auxiliary objectives are XM-compatible only when the
complete objective can be returned per sample. HiDream's optional DINO path
currently produces a reduced scalar, which is a concrete incompatibility rather
than a reason to maintain two loss implementations.

Scalar-versus-per-sample tests compare the two reduction paths with
`torch.allclose(rtol=1e-5, atol=1e-8)` on float32 fixtures. They must not require
bitwise equality: a direct full-tensor mean and a mean of per-sample means may
accumulate floating-point sums in different orders.

### 14.2 MiniMax-H3 loss components

MiniMax-H3 factors its existing objective into one private
`_compute_per_sample_component_losses` helper returning two vectors:

```text
(video_per_sample, audio_per_sample)  # each shape [B]
```

One private `_combine_per_sample_losses` helper applies
`video_per_sample + audio_loss_weight * audio_per_sample`.
`compute_per_sample_loss` calls both helpers and returns the total vector. The
ordinary `compute_loss` obtains the two components once, calls the same combiner,
returns the total mean, and logs the component means. The H3 best-of-K loop calls
the component helper and selects with `video_per_sample` only. Video/audio MSE
and weighted-total formulas therefore each have one definition; adding another
modality or cross-modal term must update the canonical component/combiner pair,
not a public three-value return contract.

The audio-loss weight keeps its existing validation and zero-weight behavior.
Although H3 currently enforces `B = 1`, its actual inputs and model outputs
already have a leading batch dimension, and loss reduction must preserve that
dimension while averaging the others. Returning `[1]` therefore requires no
synthetic wrap. Converting to a scalar and wrapping it again at the shared
winner-update boundary would add a shape-destroying special case with no user
benefit. Lifting H3's batch-size-one restriction is a separate model/runtime
change and is outside R1.

For `K > 1`, every candidate loss must be finite. A NaN or infinity raises with
the trainer name, candidate index, and affected sample indices before backward.
This does not affect `K = 1`. The current shared training loop has no generic
"skip non-finite loss and continue" policy, and `argmin` over a vector containing
NaN has no defined best-of-K selection meaning. Treating non-finite values as
losing candidates would silently mask instability, so fail-fast is the explicit
policy for both selection modes.

## 15. Logging

For standard Forward XM with `K > 1`, merge these detached values into
`loss_metrics`:

- `xm/candidate_loss_mean`: mean no-grad loss across all samples/candidates.
- `xm/selection_gain`: candidate-zero mean loss minus selected mean loss.

`xm/selection_gain` is nonnegative up to floating-point comparison behavior and
shows whether exploration is selecting alternatives. No metric claims that
losses from different K values are comparable. R1 does not emit one histogram
key per candidate or a mean winner index: candidate labels are exchangeable, so
their average index has no useful training interpretation, and tracker schemas
should stay bounded.

MiniMax-H3 emits the same two quantities under distinct names:

- `h3_video_best_of_k/candidate_loss_mean`
- `h3_video_best_of_k/selection_gain`

Those H3 metrics summarize video loss only. The existing final `loss/video` and
`loss/audio` metrics continue to describe the gradient-tracked composite update.
Using an H3-specific prefix prevents dashboards from presenting the heuristic
as paper-defined XM.

## 16. Distributed and Runtime Compatibility

### 16.1 Gradient accumulation

Each microbatch performs its own per-sample search and one backward pass. The
existing Accelerate accumulation boundary is unchanged. K does not multiply the
configured effective batch size.

### 16.2 DDP

Each rank searches candidates for its local samples. Only the final winner
forward builds a graph, after which the existing gradient reduction runs. No
candidate loss or winner index needs a cross-rank collective.

### 16.3 Gradient checkpointing and block swap

Every forward keeps the original microbatch shape and enters the existing
architecture `call_dit`. Block movement and checkpointing therefore retain
their current lifecycle. The no-grad search still executes block swap but does
not retain activations.

### 16.4 Autocast

R1 does not surround the whole candidate loop with one new autocast context.
Each architecture continues to enter and leave its existing
`accelerator.autocast()` scope inside `call_dit`. PyTorch's autocast context
clears its weight cache when the outermost nesting level exits, so the no-grad
candidate casts do not survive into the later winner context. R1 must not call
the global `torch.clear_autocast_cache()` without a supported-version reproducer
that demonstrates missing gradients or a detached cached cast.

A focused mixed-precision regression runs a no-grad candidate forward followed
by the gradient-enabled winner forward and asserts finite, nonzero
trainable-parameter gradients. It runs with the current CUDA environment from
Section 2 and records Python, torch, CUDA, and device versions in the captured
output. Inspection of the installed PyTorch 2.13.0 implementation confirms that
the outermost autocast exit clears its cache and that `fork_rng` always includes
the CPU RNG state.

A pre-implementation smoke test in that environment already completed one
no-grad CUDA autocast forward followed by a gradient-enabled forward without a
manual cache clear and produced finite, nonzero parameter gradients. The final
integration test must repeat the assertion through the actual best-of-K path;
the smoke test alone is not feature acceptance.

R1 does not create a second environment, add a minimum-version CI job, or claim
coverage for PyTorch/CUDA combinations that were not run. If the current-runtime
test exposes a failure, amend the design with its minimal reproducer before
adding a workaround.

### 16.5 Compilation

R1 introduces no K-dependent tensor shape into the transformer, avoiding a new
compile graph for each K or candidate chunk size.

### 16.6 Checkpoints and resume

The selection modes add no persistent model or optimizer state. The existing
config/logging path records the applicable `xm_best_of_k` or
`h3_video_best_of_k` option. Candidate generators are recreated per step from
the restored seeded RNG, so checkpoint resume at an optimizer-step boundary
remains reproducible under the repository's existing guarantees.

## 17. Errors and Diagnostics

Fail before dataset/model allocation for:

- `xm_best_of_k < 1`.
- `h3_video_best_of_k < 1` in MiniMax-H3.
- MiniMax-H3 with `xm_best_of_k > 1`; the error points to
  `--h3_video_best_of_k` and explains the objective mismatch.
- `xm_best_of_k > 1` when
  `get_best_of_k_incompatibility_reason(args) -> str | None` returns a reason.
- A custom `process_batch` without an explicit confirmation that the shared
  Forward XM data flow is equivalent or an architecture-owned best-of-K
  integration.

Fail during the affected step for:

- Per-sample loss with the wrong shape.
- Non-finite candidate loss.
- Winner noise shape or dtype mismatch.
- A candidate-zero input that violates the tested fixed-timestep coefficient
  contract in a newly added architecture.

Startup logs state the active option, K, sequential memory-saving mode,
supported trainer name, and approximate `(K + 3) / 3` operation-count multiplier
for `K > 1`. They do not translate it into an unmeasured wall-clock claim. They
also warn that published XM gains are pretraining results and are not validated
for LoRA. MiniMax-H3 instead logs `method: video-focused best-of-K heuristic
(not Forward XM)`, `selection objective: video only`, and `final objective:
video + weighted audio` so a run cannot be mistaken for joint multimodal XM or
video-only final training.

## 18. Documentation

Add `docs/explorative_modeling.md` and link it from the configuration lists in
both `README.md` and `README.ja.md`. The document covers:

- A minimal TOML and CLI example.
- Exact fixed-data/fixed-timestep semantics and candidate-derived target pairing.
- Supported and rejected entry points.
- Why MiniMax-H3 uses the separate `--h3_video_best_of_k` option, is not Forward
  XM, and keeps fixed audio noise/input while using a video-only candidate score
  and full video-plus-audio final update.
- Why `K = 1` is disabled behavior.
- Sequential memory-saving compute cost, with wall-clock cost explicitly marked
  hardware/model dependent unless measured.
- Loss non-comparability across K.
- The lack of established LoRA quality gains.
- Guidance and inference remaining unchanged.
- The non-finite policy: standard mixed-precision training with GradScaler may
  skip an update after detecting non-finite gradients during backward, depending
  on precision and loss-scaling configuration. Forward XM and H3 best-of-K
  instead raise before backward if any candidate selection loss is NaN or
  infinite. Best-of-K training is deliberately stricter and fails faster; users
  must not assume the modes recover identically from numerical instability.
- A conservative experiment recipe: compare `K = 1` and `K = 2` with the same
  seed, data, optimizer-step budget, and downstream validation metric; do not
  compare raw training loss as the decision metric.

The docs do not recommend a universal K or repeat the paper's efficiency claims
as expectations for fine-tuning.

## 19. Test Strategy

Tests use a tiny synthetic trainer and tensors; no production model weights are
required.

### 19.1 Parser and validation

- Default `xm_best_of_k` is 1; H3 also defaults `h3_video_best_of_k` to 1.
- TOML/CLI parsing accepts integers `>= 1` for the applicable option.
- Values below 1 fail before dataset construction.
- `K > 1` accepts the Self-Flow entry point when `--self_flow` is off and rejects
  the mode when it is on; it rejects Forward XM for MiniMax-H3 while accepting
  H3's separate video-focused option, and rejects HiDream-O1 with the concrete
  reason from Section 8.
- `K = 1` remains accepted for those trainers.

### 19.2 Baseline compatibility

- With `K = 1`, the original `process_batch` is called once.
- The explorative module is not called and no extra random number is drawn.
- Float32 loss and trainable gradients match the pre-feature path under
  `rtol=1e-5, atol=1e-8`; metrics retain their existing keys/meaning, and the
  post-step RNG state is exactly equal.

### 19.3 Candidate semantics

- Construct a batch where different samples have different best candidate
  indices, then assert each winner-noise sample equals the corresponding sample
  from its expected candidate. The test does not prescribe `where`, indexing,
  scatter, or another assembly implementation.
- Assert exactly K no-grad forwards and one grad-enabled winner forward.
- Assert only winner noise contributes to trainable-parameter gradients.
- Under an available autocast dtype, assert no-grad exploration followed by the
  winner recomputation yields finite, nonzero trainable-parameter gradients
  without an explicit cache-clear call.
- Run that regression under the current PyTorch `2.13.0+cu130` CUDA environment
  and capture the Python, torch, CUDA, and device versions with the result.
- Assert clean latents, timesteps, and conditions are identical across
  candidates, while each derived flow/noise target matches its candidate noise.
- Assert candidate noise differs and uses the expected dtype/device/shape.
- Assert weighted per-sample selection can choose a different winner than an
  incorrectly unweighted/global reduction.

### 19.4 RNG replay

- A synthetic condition-drop/random-mask forward sees the same mask for every
  candidate and the final recomputation.
- `fork_rng` scopes restore both CPU and the explicitly listed active-device
  PyTorch RNG streams; no Python/NumPy state helper is invoked.
- The final global RNG state equals a control that performed one ordinary
  stochastic forward after candidate-noise setup.
- Candidate search remains reproducible after restoring a saved RNG state.

### 19.5 Timesteps and noising

- Assert the shared explicit-coefficient sampler constant contains exactly the
  eleven names enumerated in Section 12, and both baseline and helper branch on
  that constant rather than duplicated lists.
- Candidate-zero reconstruction uses `torch.testing.assert_close` with explicit
  tolerances for both explicit-coefficient and scheduler-indexed samplers:
  float32 `rtol=1e-5, atol=1e-6`, float16 `rtol=1e-3, atol=1e-3`, and bfloat16
  `rtol=1e-2, atol=1e-2`. No test relies on framework defaults or bitwise
  equality across devices.
- The tolerances reflect representation precision: float16's ten stored
  fraction bits give a relative scale near `2^-10 ~= 1e-3`, while bfloat16's
  seven give `2^-7 ~= 1e-2`. Float32's `1e-5` relative tolerance is deliberately
  above its single-operation epsilon to cover accumulated multi-operation and
  backend reduction-order differences; its absolute tolerance covers values
  near zero.
- Distribution-preserving sampling draws timesteps once per microbatch.
- Wan high/low mode selection happens once and every candidate stays on the
  selected side of the boundary.
- 4D image and 5D video latents broadcast coefficients correctly.

### 19.6 Losses and diagnostics

- Base weighted MSE returns shape `[B]`, and float32 scalar-versus-vector
  reduction uses `torch.allclose(rtol=1e-5, atol=1e-8)`.
- Ideogram 4 uses one unweighted per-sample MSE primitive; its scalar loss equals
  the primitive mean with the same tolerance while retaining diagnostic metrics.
- MiniMax-H3's private component helper returns `[B]` video and audio losses;
  the separate canonical combiner returns a weighted-total `[B]` vector whose
  mean matches the ordinary scalar loss with the same reduction tolerance.
- MiniMax-H3 candidate selection ignores audio loss, keeps audio noise/time and
  augmented conditions identical across candidates, and the final winner loss
  still includes audio loss and produces audio-path gradients when its weight is
  nonzero.
- Construct an H3 fixture where the video-best candidate is not the
  full-objective-best candidate and assert that video-only selection wins. This
  prevents a later refactor from silently reverting to composite selection.
- NaN/Inf candidate losses fail with candidate/sample diagnostics.
- Exploration metrics are absent at K=1 and present with bounded keys at K>1;
  standard trainers use the `xm/` prefix and H3 uses
  `h3_video_best_of_k/`.

### 19.7 Runtime integration

- Gradient accumulation performs one backward per microbatch, not K.
- A fake DDP/Accelerate path sees a graph only for the final forward.
- Gradient checkpoint and block-swap callbacks run for every forward without
  retaining candidate graphs.
- Compiled synthetic call paths see a stable original microbatch shape.

### 19.8 Optional hardware characterization

Automated tests are the merge gate. When artifacts and suitable hardware are
available, separately record `K = 1` and `K = 2` step time and peak VRAM for one
supported image trainer and one supported video trainer, plus successful
checkpoint save and selection-gain metrics. This is non-blocking performance
characterization, not correctness proof or a quality benchmark. Its purpose is
to quantify the sequential penalty and inform the deferred batching decision,
not to replace tests with "ran once" evidence.

If an architecture-local batched `K = 2` prototype exists, run it on the same
model, batch, precision, block-swap configuration, warmup, and measurement
window. Report raw median sequential and batched step times, their ratio, both
peak VRAM values, the VRAM delta, device model, and software versions. Without
such a prototype, report only measured sequential numbers and do not label a
hypothetical batched time. R1 defines no universal go/no-go threshold: an R2
decision must interpret the measured trade-off against target hardware and
user workloads rather than encode arbitrary percentages in this spec.

## 20. Acceptance Criteria

R1 is complete when:

- The branch remains based on the recorded `upstream/dev` commit unless an
  intentional rebase is documented.
- The common parser exposes `--xm_best_of_k`, defaulting to 1; MiniMax-H3
  exposes a separate `--h3_video_best_of_k`, also defaulting to 1, and rejects
  XM rather than conflating the two methods.
- `K = 1` follows the non-XM baseline path with exactly matching RNG behavior
  and loss/gradient agreement under the declared float32 tolerance.
- `K > 1` holds clean data/latent, timestep, condition, and stochastic condition
  choices fixed while varying only noise and its deterministically derived
  prediction target.
- Forward XM winners are selected independently per sample with the actual
  weighted loss. MiniMax-H3's separately named heuristic selects video noise by
  video loss and is never described as Forward XM.
- Candidate forwards build no graph and exactly one winner forward builds the
  graph used by backward.
- No best-of-K code mutates the global autocast cache, and mixed-precision winner
  recomputation produces valid trainable-parameter gradients.
- The autocast regression passes in the recorded current CUDA environment;
  untested PyTorch/CUDA combinations are outside the compatibility claim.
- Sequential search stores no K-sized latent or activation tensor.
- Base and Ideogram 4 scalar losses are derived from their canonical per-sample
  objective primitives.
- MiniMax-H3 varies only video noise, selects only by video loss, keeps the audio
  candidate state fixed, and retains weighted audio loss/gradients in the final
  update.
- The supported trainer matrix runs through the neutral best-of-K dispatch;
  unsupported candidate-state/objective modes fail before allocation with
  concrete reasons.
- Gradient accumulation and DDP retain one backward/update contract per
  microbatch/accumulation boundary.
- Block swap, checkpointing, compilation, checkpoint saving, and resume require
  no best-of-K-specific persistent state.
- Automated tests pass.
- User documentation states cost, scope, supported trainers, and evidence
  limits without promising LoRA quality gains.

## 21. Deferred Work

- Candidate batching and a measured chunk-size control.
- FLOP-efficient all-grad execution.
- HiDream-O1 support with a recomposable candidate-noising state and an explicit
  decision about whether selection uses flow MSE, DINO loss, or their composite.
- Full MiniMax-H3 joint video/audio candidate exploration and composite
  candidate selection beyond R1's video-focused heuristic.
- Self-Flow candidate semantics.
- Full-finetune loop support.
- Reverse XM with a condition-aware data-search and anti-collapse contract.
- Soft-min, cached candidates, learned latent codes, or gradient-based search.
- Quality studies for LoRA rank, dataset size, K, guidance, and
  optimizer-step-versus-FLOP budgets.
