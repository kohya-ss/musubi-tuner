# Explorative Modeling (Forward XM) Support Design

Date: 2026-08-09

Status: Revised draft after external review

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
loss. For `K > 1`, each sample keeps its clean data/latent, timestep, text/image
conditioning, and condition-drop decision fixed while exploring K independent
noise candidates. Candidate quality is measured with the architecture's actual
per-sample training loss, and only the lowest-loss candidate for each sample is
recomputed with gradients and used for the optimizer update.

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
- Minimum-supported PyTorch 2.5.1 autocast context implementation:
  <https://github.com/pytorch/pytorch/blob/v2.5.1/torch/amp/autocast_mode.py#L329-L337>
- Minimum-supported PyTorch 2.5.1 `fork_rng` implementation:
  <https://github.com/pytorch/pytorch/blob/v2.5.1/torch/random.py>

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
3. Selection uses the same weighted reconstruction objective that the final
   update uses.
4. Selection happens independently for each sample in the microbatch.
5. Only selected candidates contribute gradients.
6. `K = 1` preserves the existing control flow and random-number consumption.
7. Gradient accumulation and DDP keep their existing optimizer and reduction
   semantics.
8. Trainer modes whose complete candidate state or per-sample objective is
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

### 5.3 Per-architecture XM implementations (rejected)

Each trainer could copy the official pattern and own its own candidate loop.
This gives maximum local control but duplicates timestep freezing, RNG replay,
minimum selection, metrics, and validation across more than ten trainers. The
copies would drift and make `K = 1` compatibility harder to prove.

## 6. R1 Goals

- Add Forward XM to compatible `NetworkTrainer` LoRA entry points.
- Match the paper's fixed-data, fixed-timestep, fixed-condition semantics while
  keeping each derived prediction target paired with its candidate noise.
- Select winners per sample with the real weighted training loss.
- Keep activation memory near the baseline through no-grad search and winner
  recomputation.
- Preserve baseline behavior exactly when disabled.
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
- Candidate caches, vector databases, gradient-based latent search, or soft-min
  objectives.
- Changing inference, samplers, guidance, checkpoint formats, cache formats, or
  LoRA network structure.
- Recommending a universal K.
- Claiming FID, FVD, convergence, data-efficiency, or generalization gains for
  LoRA fine-tuning.

## 8. Supported Trainer Matrix

R1 supports trainers whose candidate is fully described by one latent-noise
tensor, whose forward can reuse the base `process_batch` data flow, and whose
complete selection objective is available per sample. A trainer is not rejected
merely because it overrides `compute_loss`.

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
| `flux_2_train_network_self_flow.py` | Supported when `--self_flow` is off; rejected when on | The vanilla branch delegates to the Flux 2 base flow. Self-Flow requires two timesteps, teacher/student forwards, a per-token mask, EMA weights, and `L_gen + gamma * L_rep`; its current `process_batch` is also still `NotImplementedError`. |
| `hidream_o1_train_network.py` | Rejected for `K > 1` | It always applies sigma-dependent noise scaling outside the shared affine formula; dev mode may also derive clipping from one standard deviation over the candidate tensor, so assembled winners can change the transform. The optional DINO backend returns a batch-reduced scalar rather than `[B]`. |
| `minimax_h3_train_network.py` | Rejected for `K > 1` | One candidate is a joint video/audio noise pair with separately shifted sigmas and condition augmentation. A single base `noise` tensor is not the candidate state, even though its batch-size-one composite loss could be made per-sample. |

`K = 1` remains valid for every trainer and mode, including the rejected rows,
because it does not enter XM code.

The exclusions are semantic, not permanent architecture bans. HiDream can be
added after defining a stored candidate-noising state and a per-sample DINO
objective. MiniMax-H3 needs a joint candidate-state protocol. Self-Flow first
needs a complete ordinary training step, then an explicit definition of which
of its teacher/student random variables XM explores.

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

Validation runs once from `_validate_args_and_init` after model-specific argument
handling and before dataset, accelerator, or model construction. It validates
`K >= 1`, sets `self._xm_enabled = args.xm_best_of_k > 1`, and only when enabled
asks the trainer for an actionable incompatibility reason. The base capability
check rejects a custom `process_batch` unless that trainer explicitly confirms
that the shared XM data flow is equivalent. It does not reject a trainer solely
for overriding `compute_loss`; the canonical per-sample primitive in Section 14
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
- PyTorch CPU and active-accelerator RNG isolation via `torch.random.fork_rng`.
- Per-sample winner-state initialization and updates.
- Candidate-search summary metrics.

Modify `training/timesteps.py` to add one pure
`get_noise_coefficients_from_timesteps` helper. It returns the already sampled,
broadcast noising coefficient; it does not sample a timestep or own a trainer
virtual method.

Modify `training/trainer_base.py` to own trainer-aware integration:

- XM compatibility validation.
- The initialized `_xm_enabled` dispatch flag.
- `compute_per_sample_loss`, the canonical per-sample objective from which the
  ordinary scalar loss is derived.
- The sequential candidate search and final winner forward.
- One standard-versus-XM dispatch branch in the training loop.

Modify `ideogram4_train_network.py` to move its unweighted MSE into
`compute_per_sample_loss`; its scalar wrapper calls that primitive and only adds
the existing diagnostics.

Modify the Self-Flow, HiDream-O1, and MiniMax-H3 trainers only to return precise
mode-specific incompatibility reasons. They do not receive partial XM
implementations in R1.

No model package, root entry point, dataset, cache, network, optimizer, or
checkpoint module changes.

## 11. Training Data Flow

### 11.1 Disabled path

At initialization, `xm_best_of_k == 1` sets `self._xm_enabled` to false. The
current loop then takes the standard dispatch arm:

1. Scale/shift latents.
2. Draw one `torch.randn_like(latents)` noise tensor.
3. Call `process_batch` once.
4. Backpropagate the returned scalar.

There are no nested or helper-level `K == 1` checks. Compatibility checks are
skipped at initialization when `_xm_enabled` is false, and the standard arm
creates no candidate seed, RNG scope, winner state, or XM metrics. This is
required for random-stream and numerical compatibility.

### 11.2 Enabled path

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
   - Compute a finite `compute_per_sample_loss` vector of shape `[batch_size]`.
   - Update the winning loss, index, and noise independently per sample.
7. Build one noisy input from the assembled per-sample winning noise tensor.
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
existing scheduler. The helper must use the same explicit-coefficient sampling
set as the base method; that set is centralized rather than copied into a second
XM-only list.

This is intentionally smaller than the original proposed
`get_noisy_model_input_from_timesteps` seam: the baseline method keeps its
sampling, arithmetic, and return contract. Its inline explicit-sampler list may
be replaced mechanically by the shared constant used by the coefficient helper,
while the reusable value XM actually needs is made explicit.
HiDream's candidate-local clipping/scaling transform does not fit this pure
coefficient contract and is handled as an explicit R1 incompatibility.

Tests must prove that reconstructing candidate zero from the helper equals the
input returned by the baseline method for explicit-coefficient and
scheduler-indexed samplers, using exact equality where dtype arithmetic permits
and a declared dtype tolerance otherwise. They must also prove that all
candidates receive identical timesteps, including Wan high/low training and
distribution-preserving timestep sampling.

## 13. RNG and Condition Contract

Forward XM must not select a candidate because it received more conditioning,
a different dropout mask, or a different stochastic augmentation.

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
XM compatibility contract and tests must explicitly add the required state;
R1 does not prepay that complexity. It also does not try to make
nondeterministic accelerator kernels bitwise deterministic. Candidate selection
is based on the no-grad scores; the final recomputation trains the selected
latent even if low-level numerical nondeterminism changes its loss slightly.

## 14. Loss Contract

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

For `K > 1`, every candidate loss must be finite. A NaN or infinity raises with
the trainer name, candidate index, and affected sample indices before backward.
This does not affect `K = 1`. The current shared training loop has no generic
"skip non-finite loss and continue" policy, and `argmin` over a vector containing
NaN has no defined XM selection meaning. Treating non-finite values as losing
candidates would silently mask instability, so fail-fast is the explicit XM
policy.

## 15. Logging

For `K > 1`, merge these detached values into `loss_metrics`:

- `xm/candidate_loss_mean`: mean no-grad loss across all samples/candidates.
- `xm/selection_gain`: candidate-zero mean loss minus selected mean loss.

`xm/selection_gain` is nonnegative up to floating-point comparison behavior and
shows whether exploration is selecting alternatives. No metric claims that
losses from different K values are comparable. R1 does not emit one histogram
key per candidate or a mean winner index: candidate labels are exchangeable, so
their average index has no useful training interpretation, and tracker schemas
should stay bounded.

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

An automated mixed-precision regression runs a no-grad candidate forward
followed by the gradient-enabled winner forward and asserts finite, nonzero
trainable-parameter gradients. If that test exposes a version-specific PyTorch
bug, the design must be amended with the failing versions and minimal
reproducer before adding a workaround.

### 16.5 Compilation

R1 introduces no K-dependent tensor shape into the transformer, avoiding a new
compile graph for each K or candidate chunk size.

### 16.6 Checkpoints and resume

XM adds no persistent model or optimizer state. The common config/logging path
records `xm_best_of_k`. Candidate generators are recreated per step from the
restored seeded RNG, so checkpoint resume at an optimizer-step boundary remains
reproducible under the repository's existing guarantees.

## 17. Errors and Diagnostics

Fail before dataset/model allocation for:

- `xm_best_of_k < 1`.
- `xm_best_of_k > 1` when
  `get_xm_incompatibility_reason(args) -> str | None` returns a reason.
- A custom `process_batch` without an explicit confirmation that the shared XM
  data flow is equivalent or an architecture-owned XM integration.

Fail during the affected step for:

- Per-sample loss with the wrong shape.
- Non-finite candidate loss.
- Winner noise shape or dtype mismatch.
- A candidate-zero input that violates the tested fixed-timestep coefficient
  contract in a newly added architecture.

Startup logs state K, sequential memory-saving mode, supported trainer name, and
the approximate `(K + 3) / 3` operation-count multiplier for `K > 1`. They do
not translate it into an unmeasured wall-clock claim. They also warn that
published gains are pretraining results and are not validated for LoRA.

## 18. Documentation

Add `docs/explorative_modeling.md` and link it from the configuration lists in
both `README.md` and `README.ja.md`. The document covers:

- A minimal TOML and CLI example.
- Exact fixed-data/fixed-timestep semantics and candidate-derived target pairing.
- Supported and rejected entry points.
- Why `K = 1` is disabled behavior.
- Sequential memory-saving compute cost, with wall-clock cost explicitly marked
  hardware/model dependent unless measured.
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
- `K > 1` accepts the Self-Flow entry point when `--self_flow` is off and rejects
  the mode when it is on; it rejects HiDream-O1 and MiniMax-H3 with the concrete
  candidate-state/objective reasons from Section 8.
- `K = 1` remains accepted for those trainers.

### 19.2 Baseline compatibility

- With `K = 1`, the original `process_batch` is called once.
- The XM module is not called and no extra random number is drawn.
- Loss, gradient, metrics, and post-step RNG state match the pre-feature path.

### 19.3 Candidate semantics

- Construct a batch where different samples have different best candidate
  indices and assert per-sample winner assembly.
- Assert exactly K no-grad forwards and one grad-enabled winner forward.
- Assert only winner noise contributes to trainable-parameter gradients.
- Under an available autocast dtype, assert no-grad exploration followed by the
  winner recomputation yields finite, nonzero trainable-parameter gradients
  without an explicit cache-clear call.
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

- Explicit coefficient samplers reconstruct candidate-zero noisy input exactly.
- Scheduler-indexed samplers reconstruct it within dtype tolerance.
- Distribution-preserving sampling draws timesteps once per microbatch.
- Wan high/low mode selection happens once and every candidate stays on the
  selected side of the boundary.
- 4D image and 5D video latents broadcast coefficients correctly.

### 19.6 Losses and diagnostics

- Base weighted MSE returns shape `[B]`, and `compute_loss` equals its mean.
- Ideogram 4 uses one unweighted per-sample MSE primitive; its scalar loss equals
  the primitive mean while retaining diagnostic metrics.
- NaN/Inf candidate losses fail with candidate/sample diagnostics.
- XM metrics are absent at K=1 and present with bounded keys at K>1.

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

## 20. Acceptance Criteria

R1 is complete when:

- The branch remains based on the recorded `upstream/dev` commit unless an
  intentional rebase is documented.
- The common parser exposes only `--xm_best_of_k`, defaulting to 1.
- `K = 1` follows the unmodified baseline path with matching RNG behavior.
- `K > 1` holds clean data/latent, timestep, condition, and stochastic condition
  choices fixed while varying only noise and its deterministically derived
  prediction target.
- Winners are selected independently per sample with the actual weighted loss.
- Candidate forwards build no graph and exactly one winner forward builds the
  graph used by backward.
- No XM code mutates the global autocast cache, and mixed-precision winner
  recomputation produces valid trainable-parameter gradients.
- Sequential search stores no K-sized latent or activation tensor.
- Base and Ideogram 4 scalar losses are derived from their canonical per-sample
  objective primitives.
- The supported trainer matrix runs through shared integration, and unsupported
  candidate-state/objective modes fail before allocation with concrete reasons.
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
- HiDream-O1 support with a recomposable candidate-noising state and an explicit
  decision about whether selection uses flow MSE, DINO loss, or their composite.
- MiniMax-H3 joint video/audio candidate semantics.
- Self-Flow candidate semantics.
- Full-finetune loop support.
- Reverse XM with a condition-aware data-search and anti-collapse contract.
- Soft-min, cached candidates, learned latent codes, or gradient-based search.
- Quality studies for LoRA rank, dataset size, K, guidance, and
  optimizer-step-versus-FLOP budgets.
