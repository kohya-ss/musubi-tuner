# Explorative Modeling (Forward XM) Support Design

Date: 2026-08-09

Status: Draft for user review

Branch: `codex/issue-1019-explorative-modeling`

Base: `kohya-ss/musubi-tuner@ff8a5c2db832f3d1b458c898e040cfb5d19d0a3d`

## 1. Summary

R1 adds opt-in Forward Explorative Modeling (Forward XM) to compatible
`NetworkTrainer`-based LoRA training entry points.

Users enable it with:

```text
--xm_best_of_k K
```

The default is `K = 1`, which takes the existing training path without extra
random draws, forwards, validation restrictions, logging, or changes to the
loss. For `K > 1`, each sample keeps its data target, timestep, text/image
conditioning, and condition-drop decision fixed while exploring K independent
noise candidates. Candidate quality is measured with the architecture's actual
per-sample training loss, and only the lowest-loss candidate for each sample is
recomputed with gradients and used for the optimizer update.

R1 deliberately uses a sequential memory-saving implementation. It performs K
no-grad candidate forwards and one gradient-tracked winner forward, while
retaining only the current noise and per-sample winners. This preserves the
ordinary microbatch shape and approximately the ordinary activation-memory
footprint. Official-style candidate batching is deferred because generic batch
replication is unsafe across Musubi's tensor, list, packed, and multimodal batch
contracts and would make the first release harder to validate.

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
- Clears the PyTorch autocast cache before the gradient-tracked recomputation so
  parameter casts created by no-grad exploration are not reused as detached
  graph inputs.
- In FLOP-efficient mode, retains all candidate graphs and backpropagates the
  selected losses.
- Treats `xm_best_of_k = 1` as baseline training.

The official job scripts use memory-saving mode for released image and video
runs. The helper also supports candidate chunking, but its generic condition
tiling only covers the data structures used by that repository.

### 3.3 Compute and memory

For a transformer, the paper approximates a forward as one third of a standard
forward-plus-backward training step. For `K > 1`:

- FLOP-efficient XM costs approximately `(K + 2) / 3` standard steps and stores
  K candidates' activations.
- Memory-saving XM adds a winner recomputation and costs approximately
  `(K + 3) / 3` standard steps while keeping ordinary activation memory.

R1 chooses the second trade-off. Sequential execution has the same theoretical
FLOP count as batched memory-saving execution, but lower accelerator occupancy
and therefore lower wall-clock throughput.

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
2. Every candidate for one sample uses the same target, timestep, condition,
   condition-drop decision, and other stochastic training choices.
3. Selection uses the same weighted reconstruction objective that the final
   update uses.
4. Selection happens independently for each sample in the microbatch.
5. Only selected candidates contribute gradients.
6. `K = 1` preserves the existing control flow and random-number consumption.
7. Gradient accumulation and DDP keep their existing optimizer and reduction
   semantics.
8. Unsupported composite losses fail before dataset or model allocation rather
   than silently applying a partial XM objective.

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

This can improve device utilization, but a correct generic implementation must
replicate tensors, lists of variable-length tensors, packed layouts, metadata,
and architecture-owned side inputs without changing per-sample ordering. It
also needs candidate-aware condition-drop masks and OOM-oriented tuning. This is
a performance follow-up after R1 semantics are stable.

### 5.3 Per-architecture XM implementations (rejected)

Each trainer could copy the official pattern and own its own candidate loop.
This gives maximum local control but duplicates timestep freezing, RNG replay,
minimum selection, metrics, and validation across more than ten trainers. The
copies would drift and make `K = 1` compatibility harder to prove.

## 6. R1 Goals

- Add Forward XM to compatible `NetworkTrainer` LoRA entry points.
- Match the paper's fixed-target, fixed-timestep, fixed-condition semantics.
- Select winners per sample with the real weighted training loss.
- Keep activation memory near the baseline through no-grad search and winner
  recomputation.
- Preserve baseline behavior exactly when disabled.
- Work with microbatch sizes greater than one, gradient accumulation, DDP,
  gradient checkpointing, compilation, and block swap through ordinary-shaped
  forwards.
- Provide metrics that prove candidate search is active without claiming a
  cross-K loss comparison.
- Reject unsupported trainer objectives before expensive allocation.
- Document the compute cost and the lack of LoRA quality evidence.

## 7. R1 Non-Goals

- Reverse XM.
- Official XDiT, XJumpy, XMDLM, policy, or world-model architecture loading.
- End-to-end one-step generation.
- Full-transformer fine-tuning scripts with independent training loops.
- Candidate batching or `xm_chunk_bs_mult`.
- An all-grad/FLOP-efficient mode.
- Candidate caches, vector databases, gradient-based latent search, or soft-min
  objectives.
- Changing inference, samplers, guidance, checkpoint formats, cache formats, or
  LoRA network structure.
- Recommending a universal K.
- Claiming FID, FVD, convergence, data-efficiency, or generalization gains for
  LoRA fine-tuning.

## 8. Supported Trainer Matrix

R1 supports trainers that use the base `process_batch` flow and either the base
weighted-MSE loss or an explicit architecture-owned per-sample XM loss hook.

| Entry point | R1 status | Loss contract |
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
| `flux_2_train_network_self_flow.py` | Rejected for `K > 1` | Composite Self-Flow objective and custom batch flow |
| `hidream_o1_train_network.py` | Rejected for `K > 1` | Custom noise transform and optional DINO loss |
| `minimax_h3_train_network.py` | Rejected for `K > 1` | Joint video/audio noises and composite multimodal loss |

`K = 1` remains valid for every trainer, including the rejected rows, because it
does not enter XM code.

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
- `1`: disabled, exact existing path.
- `> 1`: sequential memory-saving Forward XM.
- Available through command line and TOML config like other common arguments.

R1 does not expose `--xm_save_mem_mode`, `--xm_chunk_bs_mult`, or
`--xm_debug_mode`. There is only one execution strategy, so additional controls
would imply unsupported behavior.

Validation runs from `_validate_args_and_init` after model-specific argument
handling and before dataset, accelerator, or model construction. For `K > 1`,
validation rejects a trainer that overrides `process_batch` or `compute_loss`
without implementing the corresponding explicit XM hook.

## 10. Code Organization

Add:

```text
src/musubi_tuner/training/explorative.py
tests/test_explorative_modeling.py
docs/explorative_modeling.md
```

`training/explorative.py` owns pure, model-independent mechanics:

- Candidate noise generator creation.
- CPU, accelerator-device, Python, and NumPy RNG state capture/replay.
- Per-sample winner-state initialization and updates.
- Candidate-search summary metrics.

Modify `training/trainer_base.py` to own trainer-aware integration:

- XM compatibility validation.
- `get_noisy_model_input_from_timesteps`, which reconstructs noisy inputs from
  already sampled timesteps.
- `compute_xm_per_sample_loss`, the default per-sample weighted-MSE scoring
  hook.
- The sequential candidate search and final winner forward.
- The `K = 1` fast-path branch in the training loop.

Modify `ideogram4_train_network.py` to provide unweighted per-sample MSE for
candidate selection, matching its existing scalar `compute_loss` behavior.

No model package, root entry point, dataset, cache, network, optimizer, or
checkpoint module changes.

## 11. Training Data Flow

### 11.1 Disabled path

When `xm_best_of_k == 1`, the current loop remains structurally unchanged:

1. Scale/shift latents.
2. Draw one `torch.randn_like(latents)` noise tensor.
3. Call `process_batch` once.
4. Backpropagate the returned scalar.

The condition is checked before XM validation, seed creation, RNG capture,
metric allocation, or helper calls. This is required for random-stream and
numerical compatibility.

### 11.2 Enabled path

For `K > 1`:

1. Reuse the training loop's first noise tensor as candidate zero.
2. Create a dedicated per-step generator for the remaining candidate noises.
   Its seed is derived from the seeded training RNG, so resume-at-step remains
   reproducible. Drawing candidates does not advance the condition RNG stream.
3. Call the existing `get_noisy_model_input_and_timesteps` once with candidate
   zero. This samples the microbatch timesteps and preserves architecture-owned
   choices such as Wan high/low model selection.
4. Capture the stochastic condition state after timestep sampling.
5. For each candidate:
   - Build its noisy input from the fixed sampled timesteps.
   - Replay the same stochastic condition state.
   - Call `call_dit` under `torch.no_grad()`.
   - Compute a finite loss vector of shape `[batch_size]`.
   - Update the winning loss, index, and noise independently per sample.
6. Clear the PyTorch autocast cache, then restore the same stochastic condition
   state once for the real update.
7. Build one noisy input from the assembled per-sample winning noise tensor.
8. Call `call_dit` with gradients and use the trainer's ordinary scalar
   `compute_loss` result and metrics.
9. Continue through the existing `accelerator.backward`, DDP reduction,
   clipping, optimizer, scheduler, and zero-grad path.

The final microbatch may combine sample 0's candidate 3, sample 1's candidate 0,
and sample 2's candidate 2. Selecting one candidate index for the entire batch
is incorrect.

## 12. Timestep and Noising Contract

The existing `get_noisy_model_input_and_timesteps` method currently samples a
timestep and constructs the corresponding noisy input together. Calling it K
times would let candidates compete on timestep difficulty and violate the
paper.

Refactor the base trainer to expose
`get_noisy_model_input_from_timesteps(args, noise, latents, timesteps,
noise_scheduler, device, dtype)`. It constructs a noisy input from an already
sampled, model-visible timestep. The existing method and the XM path both call
it, so there is one coefficient implementation.

For sampling modes whose model-visible timestep is `1000 * sigma + 1`, recover
the noising coefficient as:

```text
sigma = clamp((timestep - 1) / 1000, 0, 1)
x_t = (1 - sigma) * x + sigma * noise
```

For scheduler-indexed modes, reuse `get_sigmas` with the fixed timestep and the
existing scheduler. HiDream's custom noise transform is not generalized in R1
and is one reason that trainer is rejected.

Tests must prove that the refactored baseline first-candidate input equals the
pre-refactor formula for explicit coefficient samplers and scheduler-indexed
samplers. They must also prove that all candidates receive identical timesteps,
including Wan high/low training and distribution-preserving timestep sampling.

## 13. RNG and Condition Contract

Forward XM must not select a candidate because it received more conditioning,
a different dropout mask, or a different stochastic augmentation.

The candidate search captures and replays:

- PyTorch CPU RNG state.
- PyTorch RNG state for the accelerator device.
- Python `random` state.
- NumPy RNG state.

No-grad candidate forwards are isolated and leave the main random stream at the
captured state. The final gradient-tracked winner forward starts from that state
and advances it once, matching the semantics of one ordinary model forward.

This fixes Kandinsky 5 visual-condition frame selection across candidates and
also covers model dropout or future stochastic condition logic. It does not try
to make nondeterministic accelerator kernels bitwise deterministic. Candidate
selection is based on the no-grad scores; the final recomputation trains the
selected latent even if low-level numerical nondeterminism changes its loss by
a small amount.

## 14. Loss Contract

Add `compute_xm_per_sample_loss(args, output, timesteps, noise_scheduler,
dit_dtype, network_dtype, global_step)`. Its output is exactly shape
`[batch_size]`.

The base implementation:

1. Computes elementwise MSE between `DiTOutput.pred` and `DiTOutput.target`.
2. Applies the existing SD3 loss weighting before reduction.
3. Averages every non-batch dimension.

This is the objective used to compare candidates. Averaging the whole batch
before `min` or choosing by unweighted MSE while training with weighted MSE is
incorrect.

The final winner forward still calls the existing scalar `compute_loss` method,
so baseline logging and any explicitly approved architecture diagnostics remain
intact. If a subclass overrides `compute_loss`, validation requires that the
same subclass also override `compute_xm_per_sample_loss`; merely inheriting the
base weighted-MSE hook is not an opt-in. Tests must prove that the mean of the
custom per-sample scores equals the custom scalar training loss for the same
output. A mismatch is a design error, not a tolerated approximation.

Every candidate loss must be finite. A NaN or infinity raises with the trainer
name, candidate index, and affected sample indices before backward; treating it
as a losing candidate would hide numerical instability and change baseline
failure behavior.

## 15. Logging

For `K > 1`, merge these detached values into `loss_metrics`:

- `xm/candidate_loss_mean`: mean no-grad loss across all samples/candidates.
- `xm/selection_gain`: candidate-zero mean loss minus selected mean loss.
- `xm/best_candidate_mean`: mean winning candidate index, diagnostic only.

`xm/selection_gain` is nonnegative up to floating-point comparison behavior and
shows whether exploration is selecting alternatives. No metric claims that
losses from different K values are comparable. R1 does not emit one histogram
key per candidate because K is user-controlled and tracker schemas should stay
bounded.

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

### 16.4 Compilation

R1 introduces no K-dependent tensor shape into the transformer, avoiding a new
compile graph for each K or candidate chunk size.

### 16.5 Checkpoints and resume

XM adds no persistent model or optimizer state. The common config/logging path
records `xm_best_of_k`. Candidate generators are recreated per step from the
restored seeded RNG, so checkpoint resume at an optimizer-step boundary remains
reproducible under the repository's existing guarantees.

## 17. Errors and Diagnostics

Fail before dataset/model allocation for:

- `xm_best_of_k < 1`.
- `xm_best_of_k > 1` on an unsupported trainer.
- A custom `process_batch` without an explicit XM integration.
- A custom `compute_loss` without a matching per-sample XM scoring hook.

Fail during the affected step for:

- Per-sample loss with the wrong shape.
- Non-finite candidate loss.
- Winner noise shape or dtype mismatch.
- A fixed-timestep noising path unsupported by an architecture that bypassed
  validation.

Startup logs state K, sequential memory-saving mode, supported trainer name,
and the approximate `(K + 3) / 3` compute multiplier for `K > 1`. They also warn
that published gains are pretraining results and are not validated for LoRA.

## 18. Documentation

Add `docs/explorative_modeling.md` and link it from the configuration lists in
both `README.md` and `README.ja.md`. The document covers:

- A minimal TOML and CLI example.
- Exact fixed-target/fixed-timestep semantics.
- Supported and rejected entry points.
- Why `K = 1` is disabled behavior.
- Sequential memory-saving compute cost and expected wall-clock slowdown.
- Loss non-comparability across K.
- The lack of established LoRA quality gains.
- Guidance and inference remaining unchanged.
- A conservative experiment recipe: compare `K = 1` and `K = 2` with the same
  seed, data, optimizer-step budget, and downstream validation metric; do not
  compare raw training loss as the decision metric.

The docs do not recommend a universal K or repeat the paper's efficiency claims
as expectations for fine-tuning.

## 19. Test Strategy

Tests use a tiny synthetic trainer and tensors; no production model weights are
required.

### 19.1 Parser and validation

- Default `xm_best_of_k` is 1.
- TOML/CLI parsing accepts integers `>= 1`.
- Values below 1 fail before dataset construction.
- `K > 1` rejects Self-Flow, HiDream-O1, and MiniMax-H3 with actionable errors.
- `K = 1` remains accepted for those trainers.

### 19.2 Baseline compatibility

- With `K = 1`, the original `process_batch` is called once.
- The XM module is not called and no extra random number is drawn.
- Loss, gradient, metrics, and post-step RNG state match the pre-feature path.

### 19.3 Candidate semantics

- Construct a batch where different samples have different best candidate
  indices and assert per-sample winner assembly.
- Assert exactly K no-grad forwards and one grad-enabled winner forward.
- Assert the autocast cache is cleared between selection and the winner
  recomputation.
- Assert only winner noise contributes to trainable-parameter gradients.
- Assert targets, timesteps, and conditions are identical across candidates.
- Assert candidate noise differs and uses the expected dtype/device/shape.
- Assert weighted per-sample selection can choose a different winner than an
  incorrectly unweighted/global reduction.

### 19.4 RNG replay

- A synthetic condition-drop/random-mask forward sees the same mask for every
  candidate and the final recomputation.
- The final global RNG state equals a control that performed one ordinary
  stochastic forward after candidate-noise setup.
- Candidate search remains reproducible after restoring a saved RNG state.

### 19.5 Timesteps and noising

- Explicit coefficient samplers reconstruct candidate-zero noisy input exactly.
- Scheduler-indexed samplers reconstruct it within dtype tolerance.
- Distribution-preserving sampling draws timesteps once per microbatch.
- Wan high/low mode selection happens once and every candidate stays on the
  selected side of the boundary.
- 4D image and 5D video latents broadcast coefficients correctly.

### 19.6 Losses and diagnostics

- Base weighted MSE returns shape `[B]` before final mean.
- Ideogram 4 uses unweighted per-sample MSE.
- NaN/Inf candidate losses fail with candidate/sample diagnostics.
- XM metrics are absent at K=1 and present with bounded keys at K>1.

### 19.7 Runtime integration

- Gradient accumulation performs one backward per microbatch, not K.
- A fake DDP/Accelerate path sees a graph only for the final forward.
- Gradient checkpoint and block-swap callbacks run for every forward without
  retaining candidate graphs.
- Compiled synthetic call paths see a stable original microbatch shape.

### 19.8 Manual smoke evidence

When artifacts and suitable hardware are available, record one `K = 2` LoRA
optimizer step for one supported image trainer and one supported video trainer,
including command, hardware, peak VRAM, step time, selected-gain metrics, and
successful checkpoint save. This is runtime evidence, not a quality benchmark
and not an automated merge gate.

## 20. Acceptance Criteria

R1 is complete when:

- The branch remains based on the recorded `upstream/dev` commit unless an
  intentional rebase is documented.
- The common parser exposes only `--xm_best_of_k`, defaulting to 1.
- `K = 1` follows the unmodified baseline path with matching RNG behavior.
- `K > 1` holds target, timestep, condition, and stochastic condition choices
  fixed while varying only noise.
- Winners are selected independently per sample with the actual weighted loss.
- Candidate forwards build no graph and exactly one winner forward builds the
  graph used by backward.
- The autocast cache is cleared before that winner forward.
- Sequential search stores no K-sized latent or activation tensor.
- Base and Ideogram 4 loss semantics match their ordinary scalar losses.
- The supported trainer matrix runs through shared integration, and unsupported
  composite trainers fail before allocation.
- Gradient accumulation and DDP retain one backward/update contract per
  microbatch/accumulation boundary.
- Block swap, checkpointing, compilation, checkpoint saving, and resume require
  no XM-specific persistent state.
- Automated tests pass.
- User documentation states cost, scope, supported trainers, and evidence
  limits without promising LoRA quality gains.

## 21. Deferred Work

- Candidate batching and a measured chunk-size control.
- FLOP-efficient all-grad execution.
- HiDream-O1 support with an explicit decision about whether candidate selection
  uses flow MSE, DINO loss, or their composite.
- MiniMax-H3 joint video/audio candidate semantics.
- Self-Flow candidate semantics.
- Full-finetune loop support.
- Reverse XM with a condition-aware data-search and anti-collapse contract.
- Soft-min, cached candidates, learned latent codes, or gradient-based search.
- Quality studies for LoRA rank, dataset size, K, guidance, and
  optimizer-step-versus-FLOP budgets.
