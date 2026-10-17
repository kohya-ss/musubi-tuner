# DLSS-NR Training Runtime Extensions

**Status:** Approved for implementation on 2026-10-01.
**Date:** 2026-10-01
**Implementation checkout:** `DLSSNR` at `cf09716`; preserve its training-admission fix.

## 1. Intent and Agreed Boundary

Add useful memory and throughput options to the existing NR full/LoRA trainer:
mixed precision, activation checkpointing, distributed data parallel training,
FP8 frozen bases, and optional attention backends. Reuse existing project
facilities where their contracts fit NR. Do not replace the NR runner with a
diffusion trainer or reintroduce training settings in dataset TOML.

The user explicitly accepted keeping the current default baseline unchanged
and exposing numerically different acceleration as a separate experimental
mode. This is not permission to describe alternative attention as native NR,
silently ignore learned priors, or claim training support for a forward-only
kernel.

Success means correct gradients, state recovery and measured resource use,
not just accepting more command-line flags.

## 2. Findings

- `dlssnr/numerics.py` implements half/E4 publications, custom exponential
  weights, reduction trees, window priors and ViT denominator correction.
  Ordinary softmax attention is a different operator.
- `qwen_image/qwen_image_model.py` uses non-reentrant activation checkpointing.
  Its block-level pattern is reusable without adopting its model architecture.
- `training/accelerator_setup.py` already configures Accelerate, mixed
  precision and DDP handlers. NR needs the same runtime conventions, but not
  the diffusion logging, timestep or latent-data requirements.
- `modules/fp8_optimization_utils.py` provides weight quantization. Its forward
  patch targets `nn.Linear`; NR uses `ChannelLinear`, including NCHW inputs.
  Applying that patch unchanged would miss NR projections.
- `modules/attention.py` exposes the requested backend APIs, but its default
  scaling and mask contract cannot be copied blindly into NR.
- Current NR state files save optimizer and RNG state, but not an AMP scaler,
  per-rank state, runtime quantization recipe or backend identity.

Measured on the current verification environment:

- Windows, one RTX 4090 (SM 8.9), PyTorch `2.13.0+cu130`; BF16 is available.
- Gloo is available; NCCL is not. There is no second GPU for GPU-DDP validation.
- `flash_attn`, `sageattention`, `xformers` and `triton` are not installed in
  the isolated verification environment.
- PyTorch reports its Flash SDPA kernel unavailable, but efficient and cuDNN
  attention usable for a small FP16 test shape. This is a capability check,
  not a guarantee for every shape, mask or gradient configuration.
- NR has 145,755,115 parameters, approximately 556.01 MiB in FP32. Compressing
  every projection matrix from FP32 to FP8 has a theoretical storage difference
  of 411.51 MiB before scales, exclusions and materialized weights. FP8 is not
  assumed to solve activation-memory pressure.

## 3. Runtime Contract

Keep `train_surrogate` as the default numerical profile. Add
`train_experimental` for mixed precision, quantized bases and alternative
attention. Checkpointing and DDP may also be used with the baseline profile
because they do not intentionally replace its operators.

| Setting | Initial contract |
| --- | --- |
| `--gradient_checkpointing` | Off by default; full/LoRA, single-frame/temporal |
| `--mixed_precision no/fp16/bf16` | `no` by default; low precision initially CUDA-only and experimental |
| Multi-process launch | Accelerate or torchrun; DDP only, not FSDP/DeepSpeed |
| `--fp8_base` | Experimental frozen projection storage for LoRA only |
| `--fp8_scaled` | Requires `--fp8_base`; reuse the project's scaled quantization primitive |
| `--sdpa`, `--flash_attn`, `--xformers`, `--sage_attn` | Mutually exclusive; no flag means existing NR attention |
| `--attention_scope all/global` | `all` by default; a restricted backend requires explicit `global` |
| `--max_overflow_retries` | Bounded FP16 overflow recovery, default 16 |

Alternative attention requires `train_experimental`. Flash/Sage also require
a supported FP16/BF16 compute dtype. Requested but unavailable capabilities
fail with a specific error before training; they never silently select another
named backend. PyTorch may dispatch SDPA internally, but SDPA must not be
reported as FlashAttention unless that kernel was actually verified.

The experimental profile is an explicit opt-in and targets only the float
runtime. Preserve `cf09716`: `--development_smoke` is only required for random
initialization or incomplete conversion provenance, not to enable acceleration. It cannot consume a baseline validation report as evidence of
native equivalence or claim native-roundtrip deployment compatibility.

## 4. Activation Checkpointing

Checkpoint block-sized FFN/attention regions using `use_reentrant=False` and
RNG preservation. Keep stage transitions, skip ownership and half/E4 operator
ordering unchanged. Bind the actual block when creating a recomputation
function; do not capture a changing loop index.

Include the special stem and final block as well as encoder, ViT and decoder
blocks. Recompute only in gradient-enabled training. Burn-in and evaluation
remain non-checkpointed. A frozen input must not prevent LoRA parameters in a
checkpointed region from receiving gradients.

Do not add CPU activation offload or block swapping in the initial slice.

## 5. Mixed Precision and Update Transactions

Keep trainable parameters and optimizer master state in FP32. Limit reduced
precision to the intended projection/matmul execution. Explicit half/E4
publications, cosine/reduction arithmetic, losses, history and metric
accumulation retain deliberate FP32 boundaries. An outer autocast alone is not
an implementation of this policy.

Use Accelerate's scaler integration for FP16. Save and restore its state.
Inspect unscaled gradients before applying clipping or treating a gradient as
invalid. BF16 does not acquire a fake scaler merely to match FP16 metadata.

A finite-forward FP16 overflow retries the same effective batch with the
updated scaler, up to `--max_overflow_retries`. Restore the attempt's per-rank
RNG state so dropout is replayed; do not advance sample cursors or successful
optimizer-update counters. Clear gradients and release the previous graph
before retrying. A non-finite forward, unrecoverable overflow or non-finite
parameter aborts without producing a successful checkpoint.

DDP ranks must agree on whether an update succeeded. A retry cannot proceed on
only a subset of ranks. Log overflows separately from successful updates.

## 6. Distributed Data Parallel

Prepare exactly one owning `NRTrainModule`, including its LoRA network. Do not
prepare the same trainable parameters twice. Reuse project backend conventions:
NCCL on supported Linux CUDA builds and Gloo where appropriate, including the
Windows initialization workaround. Document CPU/Gloo validation separately
from GPU/NCCL validation.

Shard the deterministic bucket plan by global microbatch index:

```text
global_microbatch = (successful_update * accumulation + micro) * world_size + rank
```

Every rank performs the same number of forward/backward calls. Different ranks
may receive different spatial shapes and partial-batch sizes; a local batch
still contains one bucket. Do not duplicate or drop samples to equalize tails.
The existing cyclic plan supplies the next epoch where needed.

Sum valid-element denominators across ranks for the entire accumulated update.
Compensate for DDP's gradient averaging and Accelerate's accumulation division.
Do not average rank-local mean losses when their valid pixel counts differ.
Keep the existing seed derivation tied to each sample's global epoch/index.
Initialize identical base/adapter parameters before rank-specific runtime RNG
streams are established.

Only the main process creates the run directory, saves weights, writes shared
logs or evaluates the unwrapped model. Use barriers and coordinated error
handling around those operations. Store RNG and relevant runtime state for
each rank, capturing the rank's active CUDA device rather than initializing
every GPU merely to collect RNG state.

Exact resume requires the same world size and batch plan. Resharding an old
state, multi-node operation and elastic world-size changes are not included.

## 7. FP8 Frozen Bases

FP8 storage is initially a LoRA-only option. Full fine-tuning continues to own
FP32 trainable weights; merely casting them to FP8 is not a valid optimizer
implementation.

Add an NR weight-materialization path for `ChannelLinear`, reusing shared
quantization primitives without globally patching unrelated models. Quantize
eligible frozen matrices only. Preserve input/RGB/logit heads, scalar/vector
parameters, priors and the lane-15 invariant in their existing representation.
Check unscaled FP8 casts for overflow; do not sanitize invalid weights into
apparently valid models.

Unscaled storage uses E4M3FN casts. Scaled storage uses the shared block-wise
quantizer with block size 64 and its existing per-channel fallback for matrix
widths not divisible by 64. Quantization is deterministic from canonical FP32
weights; persist the resolved recipe rather than relying on future defaults.

LoRA's effective weight remains materialized base plus its FP32 delta. Do not
switch to split projection GEMMs: the existing real-weight tests show that this
changes NR output materially. Use checkpointing to avoid retaining every
materialized effective weight for the entire backward pass.

Record both original canonical base identity and effective quantized-base
identity, plus the exact quantization recipe and scales. An FP8-trained adapter
must not silently merge against a different effective base. Reconstruct and
verify the recorded base before merge, then export ordinary canonical FP32
weights and the required runtime metadata. Legacy FP32 adapters retain their
existing base interpretation.

A merged artifact is explicitly marked as an already materialized FP32
effective base plus delta. Loading it must not quantize the merged matrices a
second time. Its attention and compute-precision policies remain relevant to
reproducing the training-time evaluation.

No claim of FP8 Tensor Core execution is made by an FP8 storage flag. Scaled-MM
kernels are a separate optimization and are not required for this slice.

## 8. Alternative Attention

Provide a small NR-specific backend adapter rather than pretending the common
diffusion attention call has identical mathematics. Keep Q/K normalization,
temperature, layout, residual wiring and output-publication boundaries
explicit. The experimental attention core uses ordinary softmax on the
prepared Q/K/V with `scale=1.0`, not an accidental extra `1/sqrt(head_dim)`.

Window attention must retain every learned prior entry and the existing
out-of-field zero-token participation. Global attention must not include its
artificial padding as real keys in ordinary softmax. Test both details with
nonzero priors and non-multiple-of-64 token counts.

| Backend | Initial intended coverage |
| --- | --- |
| Native NR | Existing baseline operators, every layer |
| SDPA | Experimental window/global; explicit window prior |
| xFormers | Experimental window/global only when dtype, bias and bias-gradient support pass capability tests |
| FlashAttention | Experimental global scope; do not drop arbitrary window priors to fit the API |
| SageAttention | Experimental global inference initially; training remains rejected without a verified genuine backward implementation |

An explicit `global` scope leaves window attention on native NR operations and
records that mixed policy. Reject unsupported `all` scope rather than silently
falling back. Do not implement a Sage forward plus secretly substituted SDPA
backward and call that Sage training support.

Optional backends are lazily loaded. Native and SDPA operation must not fail
because an unrelated optional CUDA extension is absent or ABI-incompatible.
Backend selection and numerical profile must carry into evaluation, standalone
inference, saved artifacts and resume checks, not only the training CLI.

Standalone inference inherits a saved artifact's runtime policy when the user
does not explicitly supply one. Legacy artifacts without that policy keep the
current baseline default. Explicit inference overrides are recorded in output
metadata and are not described as equivalent to the saved training runtime.
Training resume rejects a policy mismatch rather than treating it as an
inference-style override.

## 9. State and Validation

Introduce a new training-state schema for scaler/per-rank/runtime state. Bind
precision policy, attention backend and scope, quantization identity, world
size, batch-plan identity and implementation version. Old state formats must
fail with migration guidance, not lose scaler or per-rank information silently.
Publish a complete checkpoint manifest only after every required state file is
present. Preserve canonical weights and source opaque records.

Required checks for each delivered slice:

- Defaults: existing FP32 forward, gradient, LoRA merge and data tests pass.
- Checkpointing: forward/gradient/update comparisons, dropout RNG replay,
  frozen-input LoRA, temporal burn-in, and measured saved activation memory.
- AMP: real-weight CUDA FP16/BF16 updates, finite gradients, loss arithmetic,
  clipping after unscale, deliberate overflow/retry, and scaler resume.
- DDP: two-process Gloo tests against a one-process global-batch reference with
  uneven valid-pixel counts, bucket tails, per-rank RNG and coordinated saves;
  real multi-GPU smoke remains a separately reported hardware requirement.
- FP8: actual resident dtype/storage checks, finite materialization, frozen
  base and lane invariants, LoRA gradients, quantization-aware merge and resume.
- Attention: explicit math-oracle comparisons, prior sensitivity and gradients,
  padding tests, missing-dependency errors, and actual optional-kernel tests
  wherever those dependencies/hardware are available.
- Benchmark: report allocated-memory peak and step time on the same weights,
  batch, bucket, seed and warm-up procedure. Distinguish weight storage from
  peak training memory. Do not equate an import check with kernel validation.
- Run the repository regression suite and Ruff before integration. Missing
  hardware/backend coverage must remain visible in the report.

## 10. Delivery Order and Non-Goals

Deliver in this order: checkpointing; experimental AMP and runtime identity;
DDP/state transactions; FP8 LoRA storage/merge; alternative attention adapters.
Each slice remains independently usable and has explicit acceptance results.
The order targets memory pressure before adding optional binary dependencies.

Native DLL parity, learned-output quality, TF32 enablement, full FP8 optimizer
training, FSDP/ZeRO/DeepSpeed, multi-node execution, activation CPU offload,
custom native-math fused CUDA/Triton kernels and arbitrary cross-mode exact
resume are not promised by this work.

The user approved implementation in the current DLSSNR checkout, with inline
implementation and one independent review at the end. This document does not
mark the features above as implemented or validated.

## Primary References

- [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint):
  recomputation, non-reentrant behavior and RNG preservation.
- [PyTorch AMP recipe](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html):
  autocast, scaling, unscale-before-clipping and scaler state.
- [Accelerate variable-size accumulation](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation#gradient-accumulation-on-training-samples-of-variable-size):
  global denominators and distributed scaling.
- [FlashAttention API](https://github.com/Dao-AILab/flash-attention#how-to-use-flashattention):
  softmax contract, backend limitations and supported arguments.
- [SageAttention](https://github.com/thu-ml/SageAttention) and its
  [core implementation](https://github.com/thu-ml/SageAttention/blob/main/sageattention/core.py):
  inference-oriented quantized attention APIs; these are not evidence of NR
  training gradients.
- [xFormers FMHA interface](https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py):
  current API exports, including the upstream move into `mslk`.
