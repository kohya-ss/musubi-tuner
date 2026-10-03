# DLSS-NR Training Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce measured NR training memory and add explicit, reproducible acceleration without changing default FP32 behavior.

**Architecture:** Keep one owning NRTrainModule and the existing paired-data plan. Add model-local compute/attention/weight policies, non-reentrant block checkpointing, and distributed optimizer-boundary transactions. Reuse Accelerate, the shared optimizer factory and FP8 quantizer.

**Tech Stack:** Python, PyTorch, Accelerate, safetensors; optional FlashAttention/xFormers/SageAttention loaded lazily.

**Spec:** `docs/superpowers/specs/2026-10-01-dlssnr-training-runtime-design.md`

## Global Constraints

- Keep `train_surrogate` as the default numerical profile.
- Checkpointing and DDP may also be used with the baseline profile.
- `train_experimental` targets only the float runtime; it never certifies native parity.
- Preserve `cf09716`: development smoke controls source provenance, not acceleration permission.
- Trainable parameters and optimizer master state remain FP32.
- Dataset TOML stays dataset-only. Training/runtime options remain CLI flags.
- Never silently replace a requested optional backend or drop a learned prior.
- No FSDP/ZeRO/DeepSpeed, multi-node, full FP8 optimizer, block swapping or CPU activation offload.
- Work in the user's current DLSSNR checkout. Do not push or merge as part of this request.

## Review Focus

- Frozen inputs must still produce checkpointed LoRA gradients, including dropout replay (Task 1).
- FP16 retry must replay the same effective batch without moving counters or saving a failed update (Task 2).
- Uneven rank-local masks/bucket tails must produce the global pixel-weighted gradient (Task 3).
- FP8 adapters must merge against the quantized effective base, never requantize merged deltas (Task 4).
- Inference must inherit saved runtime policy; a missing optional backend must not silently change output (Tasks 5-6).

## Verification Environment

Use `references/dlssnr_pr_verification/.venv/Scripts/python.exe`, `PYTHONPATH=src`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. Real-weight CUDA tests use `DLSSNR_SOURCE_DIR=D:/UGit/OpenDLSS-NR/models/nr` and `DLSSNR_RUN_CUDA_SMOKE=1`. Keep weights and generated measurements under ignored `references/`; do not ship them. The machine has one CUDA GPU, so two-process CPU/Gloo tests cannot certify multi-GPU kernels.

## Task 1: Block Activation Checkpointing

**Files:** model.py, training/dlssnr_parser.py, dlssnr/config.py, training/dlssnr_trainer.py; create tests/test_dlssnr_checkpointing.py (source paths under src/musubi_tuner).

**Interfaces:** `NRModel.enable_gradient_checkpointing()` sets a model-local switch, default off. `_run_window_stage(..., checkpointing=False)` and `_decode_stage(..., checkpointing=False)` preserve their default contracts. No saved parameter-name changes.

- [x] Write tests comparing forward, gradients and update with/without checkpointing, including multiple different blocks, frozen input plus LoRA dropout, eval/no-grad bypass, stem/final block coverage, and reduced saved tensor bytes.
- [x] Run `pytest -q tests/test_dlssnr_checkpointing.py`; expect missing-method/flag failures.
- [x] Wrap bound FFN/attention regions with `torch.utils.checkpoint.checkpoint(..., use_reentrant=False, preserve_rng_state=True)` only during grad-enabled training. Wire the CLI flag through config and trainer.
- [x] Run the checkpoint tests and existing NR forward/LoRA/training tests; require equal default outputs and finite matching gradients.

## Task 2: Explicit Precision and AMP Update Transactions

**Files:** create dlssnr/runtime.py and tests/test_dlssnr_runtime.py; edit model.py, numerics.py, networks/lora_dlssnr.py, training/dlssnr_{parser,services,trainer,state}.py, dlssnr/config.py.

**Interfaces:** `runtime_policy(config) -> dict`, `configure_model_runtime(model, policy, *, training=False) -> None`; policy contains schema, numerical profile, mixed precision, checkpointing, backend/scope and FP8 recipe. `ChannelLinear.project(weight, x)` isolates autocast products and returns FP32; explicit arithmetic/loss/history remain FP32. `save_state(..., accelerator=None)` / `restore_state(..., accelerator=None)` persist the scaler with new v3 state identity.

- [x] Test experimental-profile admission, unchanged defaults, unsupported device/dtype rejection, actual projection dtype and FP32 masters/losses, finite-overflow retry and bounded failure, clipping after unscale, saved/restored scaler and exact resume.
- [x] Run runtime tests and observe failures against the current FP32-only guard.
- [x] Add CUDA fp16/bf16 options, explicit product contexts, Accelerator scaler integration, same-batch/RNG overflow retry (default maximum 16), and v3 state validation. Do not use an outer autocast as the entire implementation.
- [x] Run CPU contracts plus real-weight CUDA full/LoRA updates in FP16 and BF16 with checkpointing. Compare FP32 default regressions separately from experimental quality.

## Task 3: Deterministic DDP and Coordinated State

**Files:** training/dlssnr_services.py, dlssnr_trainer.py, dlssnr_state.py; create tests/test_dlssnr_distributed.py and its subprocess worker if needed.

**Interfaces:** prepare exactly one NRTrainModule. Global microbatch index is `(update * accumulation + micro) * world_size + rank`. Global denominators are SUM-reduced; backward multiplies by `accumulation * world_size`. Per-rank runtime state binds world size, rank and active-device RNG.

- [x] Test two CPU/Gloo ranks against one-process effective-batch SGD, uneven masks and bucket tails, unique sample consumption, per-rank RNG resume, rank-zero shared I/O and coordinated rank-zero failures.
- [x] Run the subprocess tests; expect the existing multi-GPU guard to reject them.
- [x] Reuse platform DDP setup, disable unsupported distributed modes, shard indices, reduce denominators/metrics, coordinate overflow decisions and main-process operations. Publish manifest only after all rank states are complete.
- [x] Require subprocess success with finite timeouts, identical global updates/resume and no hangs. Mark real two-GPU validation separately unavailable on this host.

## Task 4: FP8 Frozen Bases and Correct Merge

**Files:** create dlssnr/fp8.py and tests/test_dlssnr_fp8.py; edit model.py, runtime.py, config.py, lora_dlssnr.py and trainer.

**Interfaces:** `quantize_frozen_base(model, *, scaled: bool) -> dict` returns deterministic recipe and original/effective identities; `materialize_state_dict(model) -> dict` exports FP32 tensors. ChannelLinear materializes its frozen weight before adding the FP32 LoRA delta. Adapter metadata binds its effective-base recipe.

- [x] Test real resident float8 storage, exclusions/lane invariants, finite overflow rejection, block64/per-channel scaled reconstruction, LoRA gradients and unchanged frozen identity, quantization-aware merged forward, and materialized-artifact no-double-quantization.
- [x] Run tests; expect missing FP8 support.
- [x] Reuse the shared quantizer, exclude heads/input/scalars/priors, reject full-training FP8, and retain effective-weight projection semantics. Save/verify original and effective identity in adapters and merge reports.
- [x] Run focused tests and one CUDA FP8 LoRA update/merge smoke. Do not claim FP8 GEMM acceleration.

## Task 5: Experimental Attention Backends

**Files:** create dlssnr/attention.py and tests/test_dlssnr_attention.py; edit runtime.py, model.py, numerics.py, parser and config.

**Interfaces:** `attention(query, key, value, *, backend, prior=None) -> Tensor` takes prepared Q/K/V, uses explicit scale=1.0 and returns FP32. Native remains the default. Scope `global` leaves windows native. Reject Sage training and all-scope Flash/Sage.

- [x] Test SDPA against explicit softmax oracle with nonzero learned priors and bias gradients, out-of-field tokens, global non64 lengths, backend exclusivity, missing dependencies, scope restrictions and Sage training rejection.
- [x] Run tests and observe unsupported policy/backend failures.
- [x] Implement small lazy adapters using public APIs. Flatten window batch dimensions without losing broadcast-bias gradients; do not include artificial global padding in ordinary softmax. Capability-check requested optional kernels using real forward/backward when available, not imports alone.
- [x] Run CPU/SDPA and available CUDA-kernel checks; explicit skips must name absent optional packages/hardware.

## Task 6: Saved Policy and Standalone Inference

**Files:** dlssnr/infer.py, dlssnr_generate_image.py, dlssnr_generate_video.py, artifacts.py, runtime.py and LoRA merge metadata; extend runtime/host/artifact tests.

**Interfaces:** `load_model(..., runtime_overrides=None)` inherits artifact metadata; missing policy means legacy baseline. Explicit CLI overrides are tracked in generated output metadata. Resume never permits a policy override.

- [x] Test legacy default loading, experimental policy inheritance, metadata tamper/invalid policy rejection, explicit override audit, merged policy inheritance and no second FP8 cast.
- [x] Run the tests before implementing these cases.
- [x] Add shared inference runtime arguments with None defaults, artifact policy resolution, and output provenance. Preserve canonical opaque records and existing output overwrite protections.
- [x] Run host, artifact, still/sequence and merge regressions.

## Task 7: Reproducible Memory Evidence and Usage

**Files:** docs/dlssnr.md; create tools/benchmark_dlssnr_runtime.py (if no suitable existing benchmark command), tests for its argument/policy validation; raw results ignored under references.

**Interfaces:** benchmark same real canonical weights, input shape, batch, seed, optimizer, warmup and measured steps in a fresh process per policy. Report peak allocated/reserved CUDA memory, step time and finite loss; no quality or native-parity claim.

- [x] Add argument validation tests before benchmark behavior. Select a fitting bucket conservatively; isolate OOM trials in subprocesses.
- [x] Run default, checkpoint-only, checkpoint+BF16, checkpoint+FP16 and applicable LoRA/FP8 measurements with identical workloads.
- [x] Document actual CLI recipes, experimental boundaries, state migration and hardware/backend validation gaps; use measured numbers only.

## Task 8: Whole-Change Verification and Review

- [x] Run the full repository `pytest -q --tb=short` suite with real-weight CUDA enabled; read every skip/failure and compare to the current branch baseline.
- [x] Run `ruff check .`, `ruff format --check .`, and `git diff --check`.
- [x] Request one independent whole-change code review, including the review-focus cases and documented deviations. Fix important findings with failing tests first and rerun affected/full checks.
- [x] Leave the changes on the requested branch without pushing. Report measured memory, actual support vs missing hardware validation, and any remaining work accurately.

## Completion Evidence

Final complete regression on 2026-10-01: **1133 passed, 9 skipped, 46 warnings** in 405.32 seconds, including real canonical-weight CUDA tests. Six skips require optional Triton and three require absent FlashAttention/xFormers/Sage kernels. The warnings are existing SWIG and Torch pinned-memory deprecations. Ruff check and format check (379 files), and `git diff --check`, passed.

One independent whole-change review found a Windows single-worker launcher bug. A failing real-Accelerate initialization boundary test reproduced the incorrect NCCL selection; the fix recognizes `LOCAL_RANK` even at world size one. All eight platform/device/world-size cases passed after the fix, as did an actual single-worker CUDA/Gloo backward and SGD update. No review findings were deferred. The final full suite above was run after that fix.

At implementation handoff, the changes were local on `DLSSNR`, without a commit, push or merge. The user subsequently requested committing and pushing the changes, then merging them into `qinglong`. Resource measurements and usage are in `docs/dlssnr.md`; real two-GPU operation, absent optional kernels, training quality and native DLL equivalence remain explicitly unverified.
