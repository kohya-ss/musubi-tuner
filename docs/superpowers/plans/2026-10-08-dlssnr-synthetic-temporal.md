# DLSS-NR Synthetic Temporal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Lazily turn paired still images into reproducible subpixel clips for the existing temporal trainer and labeled evaluation.

**Architecture:** A dataset wrapper generates each frame from the original bucket-transformed pair and provides exact translation motion plus conservative validity masks. Epoch-aware collection loading gives repeated samples distinct deterministic trajectories. Reuse shared masked temporal residuals from the control-randomization plan; enabling synthetic clips does not require enabling control randomization.

**Tech Stack:** Python, PyTorch bilinear sampling, existing dataset/bucket/Accelerate/Gloo code, pytest and Ruff. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-08-dlssnr-control-randomization-synthetic-temporal-design.md`, especially sections 4-8.

**Execution:** Native/inline in this chat, without subagents or delegated reviews. Implement after `2026-10-08-dlssnr-control-randomization.md`; only its seed and temporal-mask helpers are runtime prerequisites. Approved and implemented inline on 2026-10-08. See the final verification record below.

## Global Constraints

- Preserve unrelated work; no GUI, outer-launcher, learning-rate-default or DLL-format changes.
- Synthesis is opt-in per dataset, requires temporal training and preserves existing explicit clip-length/burn-in/TBPTT validation.
- Default `synthetic_max_shift_px=0.5`, finite range `[0,1]`; frame zero uses `(0,0)` and later frames sample independently within the configured range.
- Every generated frame samples the original transformed pair. Sampling is bilinear with `align_corners=False`; motion is current-to-previous `d_t - d_(t-1)`.
- Border extension provides model context only. Labels use normalized masked resampling; history requires the full previous bilinear footprint to be supported.
- Data remain lazy; no expanded RGB/flow dataset is written. Original source sequence IDs and file fingerprints remain authoritative.
- Marked `validation_manifest` entries produce fixed epoch-zero synthetic clips. `sequence_manifest` always remains a real explicit-motion sequence.
- Synthetic temporal metrics use joint label support and are labeled; real-sequence metric behavior remains unchanged.
- Real scene motion, occlusion, long-run stability and DLL history sampling are not validated by synthetic translation tests.

## Review Focus

1. A previous-frame center can be in bounds while its bilinear footprint touches invalid border extension: history must still be rejected (Task 1).
2. Zero maximum shift and a mask supported only on an edge must not be rejected by an accidental `0:-0` crop (Task 1).
3. Repeat IDs must be applied before seeding; otherwise repeated dataset entries silently reuse the same trajectory (Task 2).
4. Mixed real/synthetic sources and inherited settings must retain their own support/configuration rules instead of applying synthesis to all clips (Task 2).
5. Neutral target fill, reset boundaries or the native comparison pass must not contaminate synthetic temporal MAE or leak its policy into real cases (Task 3).

## Files and Shared Interfaces

Create `src/musubi_tuner/dlssnr/synthetic_temporal.py`,
`configs/dlssnr_dataset_synthetic_temporal.toml`, and tests
`tests/test_dlssnr_synthetic_temporal.py`,
`tests/test_dlssnr_synthetic_temporal_training.py`,
`tests/test_dlssnr_synthetic_temporal_evaluation.py`,
`tests/test_dlssnr_synthetic_temporal_distributed.py`.

Consume these interfaces from the preceding plan:

- `temporal.augmentation_seed(seed, epoch, sample_id, crop_id, *, domain) -> int`.
- `losses.joint_temporal_support(current_mask, previous_mask, motion, temporal_valid, reset) -> (valid, warped_previous_mask)`.
- `losses.masked_temporal_residual(current, previous, target, previous_target, current_mask, previous_mask, motion, temporal_valid, reset) -> (error, valid)`.
- Batch `temporal_support='joint_loss_mask'`, understood by both loss numerators and denominators.

Use the same per-process Python/PYTHONPATH/thread settings as the preceding plan.
Run focused synthetic CPU tests after each task, then the complete NR suite once
at the end. Preserve and report optional dependency/environment limitations.

## Task 1: Translation, Label Coverage and History Validity

**Files:** Create `src/musubi_tuner/dlssnr/synthetic_temporal.py` and `tests/test_dlssnr_synthetic_temporal.py`.

**Interfaces:**
- `sample_offsets(length: int, max_shift_px: float, *, seed: int, epoch: int, sample_id: str, crop_id: int) -> Tensor`: CPU FP32 `[T,2]`, first row zero, private domain `synthetic_jitter`.
- `validate_synthetic_support(sample: dict, max_shift_px: float) -> None`: require a positive loss-mask pixel in the interior with margin `ceil(max_shift_px)`; handle margin zero without slicing away the image.
- `synthesize_clip(sample: dict, offsets: Tensor) -> dict`: return existing clip metadata plus `frames` and `temporal_support='joint_loss_mask'`. Frames expose the existing tensor/reset/frame-index fields. Reject nonzero initial offsets. Source and target shapes remain the bucket size.

- [x] **Step 1: Write failing translation/oracle tests.** Use 48x64 ramp and impulse images and a source frame index of 10. Assert frame indices `[10,11,12]`, first-frame exact values, reset/history rules, motion signs, and that each frame matches independently sampling the original image rather than the previously warped frame.
- [x] **Step 2: Add the border-footprint and zero-shift regressions.** For offsets `[(0,0),(0.5,0),(0,0)]`, the last frame's right-edge previous center is in bounds but its history mask must be false. For zero shifts and an edge-only mask:
  ```python
  validate_synthetic_support(sample, 0)
  clip = synthesize_clip(sample, torch.zeros(3, 2))
  for frame in clip['frames']:
      torch.testing.assert_close(frame['source'], sample['source'], rtol=0, atol=0)
      torch.testing.assert_close(frame['loss_mask'], sample['loss_mask'], rtol=0, atol=0)
  ```
  Name these `test_history_rejects_invalid_previous_footprint` and `test_zero_shift_keeps_edge_mask_support`.
- [x] **Step 3: Add soft-mask/label-hole tests and run RED.** Assert `warp(target*mask)/warp(mask)` against an independent oracle, excluded-label invariance, finite neutral fill, constant fixed controls, translated spatial control maps, and rejection of absent safe support. Run `python -m pytest -q tests/test_dlssnr_synthetic_temporal.py`.
- [x] **Step 4: Implement the generator.** Use PyTorch bilinear sampling with explicit pixel-center conventions. Preserve exact zero-offset values. Build geometric support and test the contributing previous-frame pixel footprint, not merely its center. Generate binary history/temporal geometry masks; continuous label masks are combined by the shared loss helper.
- [x] **Step 5: Run GREEN and temporal regressions.** `python -m pytest -q tests/test_dlssnr_synthetic_temporal.py tests/test_dlssnr_temporal.py tests/test_dlssnr_frequency_loss.py`.
- [x] **Step 6: Review and commit.** Commit message: `feat(dlssnr): generate masked subpixel clips from paired stills`.

## Task 2: Epoch-Aware Datasets and Configuration

**Files:** Modify `src/musubi_tuner/dlssnr/synthetic_temporal.py`, `src/musubi_tuner/dlssnr/dataset.py`, `src/musubi_tuner/dlssnr/config.py`, `src/musubi_tuner/training/dlssnr_trainer.py`, and `tests/test_dlssnr_synthetic_temporal.py`; create `tests/test_dlssnr_synthetic_temporal_training.py` and `configs/dlssnr_dataset_synthetic_temporal.toml`.

**Interfaces:**
- Add `get_sample(index: int, *, epoch: int = 0, sample_id: str | None = None) -> dict` to `NRDataset` and `NRDatasetCollection`. `__getitem__` remains its deterministic epoch-zero entry point. Ordinary datasets ignore epoch; collections pass the final repeat-prefixed ID to the child before augmentation.
- Add `NRSyntheticTemporalDataset(base: NRDataset, sequence_length: int, *, seed: int, max_shift_px: float)`. Implement the same access method plus `iter_frames`, `validate` and `fingerprint`; expose `rows`, `bucket_sizes`, `sequence_ids`, and `synthetic_protocol`.
- Virtual rows report generated frame indices/reset flags and actual clip length. `synthetic_protocol` uses schema `dlssnr_synthetic_temporal_v1` and records seed policy/seed, length, shift, interpolation, footprint validity and temporal metric policy.
- Resolve enabled TOML fields to `data_entry['synthetic_temporal'] = {'schema': 'dlssnr_synthetic_temporal_v1', 'max_shift_px': value}`; omit the entry when disabled. Wrapper construction supplies length/seed from the existing training config.

- [x] **Step 1: Write config RED tests.** Cover general/dataset inheritance, bool/numeric validation, explicit shift with disabled synthesis, non-temporal rejection, directory acceptance only when marked, single-frame manifest enforcement and unchanged real-clip validation.
  Inheritance clarification: an explicitly disabled dataset may discard an inherited shift default, but declaring a local shift while disabled is an error. A general shift default with no enabled consumer is also an error. Pin this in `test_synthetic_inheritance_respects_dataset_disable`.
- [x] **Step 2: Write epoch/collection tests.** Two repeats of the same source get different logical seeds and trajectories at positive shift. Fetching the same logical sample/epoch after other accesses returns exactly the same clip; changing epoch changes its trajectory without consuming global RNG. `dataset[index]` must match `get_sample(index, epoch=0)`.
- [x] **Step 3: Add batch-plan/collation tests and run RED.** Assert generated `frames` counts, consumed-sample counts and fingerprints with multiple entries, repeats, bucket tails and mixed real/synthetic sources. `collate_clips` preserves homogeneous support policy and rejects a manually mixed-policy microbatch. Run `python -m pytest -q tests/test_dlssnr_synthetic_temporal.py tests/test_dlssnr_synthetic_temporal_training.py`.
- [x] **Step 4: Implement loaders and the wrapper.** Move existing ordinary sample loading behind `get_sample` without changing its tensors. Delegate source validation/fingerprints, perform safe-support checks, and generate clips only on access. In `_microbatch`, calculate epoch before loading and use `get_sample`; the final sample metadata remains available to the control sampler.
- [x] **Step 5: Implement the data/config factory changes.** Wrap marked still sources in temporal mode; preserve existing real-clip factories. Keep dataset-specific grouping, IDs and overlap checks. Resolve fixed controls as before. Bind the new transform module/protocol to metadata and exact resume; do not allocate an NR reference merely because synthetic clips are enabled.
- [x] **Step 6: Add a two-update synthetic-only trainer test and the example TOML.** Reuse `small_math`; keep `control_randomization` off. Assert temporal head/blend gradients are finite/nonzero, no frozen reference is constructed, and the saved batch plan reports the configured clip length. Document `4/2/2` and explicit `--loss_temporal 0.1` as an example, not defaults.
- [x] **Step 7: Run GREEN and loader/runner regressions.** `python -m pytest -q tests/test_dlssnr_synthetic_temporal.py tests/test_dlssnr_synthetic_temporal_training.py tests/test_dlssnr_dataset.py tests/test_dlssnr_directory_dataset.py tests/test_dlssnr_buckets.py tests/test_dlssnr_shuffle.py tests/test_dlssnr_native_crop.py tests/test_dlssnr_training.py`.
- [x] **Step 8: Review and commit.** Commit message: `feat(dlssnr): integrate reproducible synthetic temporal datasets`.

## Task 3: Labeled Synthetic Validation

**Files:** Modify `src/musubi_tuner/training/dlssnr_trainer.py`, `src/musubi_tuner/dlssnr/evaluation.py`, `docs/dlssnr.md`; create `tests/test_dlssnr_synthetic_temporal_evaluation.py`.

**Interfaces:**
- Marked `validation_manifest` entries load paired stills and wrap them with the same sequence length/shift/seed, always at epoch zero. Never wrap `sequence_manifest`.
- `evaluate` records `synthetic_temporal=dataset.synthetic_protocol` for synthetic configured/native cases, tracks previous label masks and resets them with histories/targets.
- Extend `_accumulate(..., *, previous_mask=None, temporal_support=None)` without changing default behavior. The joint policy uses `masked_temporal_residual` for both absolute sums and valid-RGB counts; ordinary cases keep the original path.

- [x] **Step 1: Write fixed-validation tests.** Two evaluations after arbitrary training-epoch accesses return equal metrics and trajectories. Assert synthetic protocol tags exist for configured/native outputs and real sequence manifests retain their original frames and masks.
- [x] **Step 2: Write `test_synthetic_temporal_metric_ignores_neutral_fill_and_resets`.** Perturb target values outside masks and compare temporal MAE; assert equality. Insert a reset and verify no previous target/mask survives it. Compare the metric with a direct joint-support oracle. Add a real-case regression asserting the report is unchanged.
- [x] **Step 3: Add `test_native_comparison_uses_previous_frame_mask_for_both_modes`.** Compare native temporal MAE in a combined report with a standalone native-policy evaluation using varying masks. Update the remembered mask only after both runtime modes finish the frame.
- [x] **Step 4: Run RED.** `python -m pytest -q tests/test_dlssnr_synthetic_temporal_evaluation.py`.
- [x] **Step 5: Implement factory/evaluator changes and document the separate evidence types.** Maintain independent histories for configured/native and alternate-noise probes. Reuse existing content/detail metrics on valid pixels. Do not introduce randomized validation targets or controls.
- [x] **Step 6: Run GREEN and evaluation regressions.** `python -m pytest -q tests/test_dlssnr_synthetic_temporal_evaluation.py tests/test_dlssnr_content_evaluation.py tests/test_dlssnr_detail_evaluation.py tests/test_dlssnr_deployment_evaluation.py`.
- [x] **Step 7: Review and commit.** Commit message: `feat(dlssnr): evaluate synthetic clips with valid temporal support`.

## Task 4: Combined Training and Final Acceptance

**Files:** Extend `tests/test_dlssnr_synthetic_temporal_training.py`; create `tests/test_dlssnr_synthetic_temporal_distributed.py`; extend only needed markers in `tests/test_dlssnr_distributed.py`; finalize `docs/dlssnr.md`.

**Interfaces:** Reuse the existing two-process harness and the preceding plan's control marker. A `synthetic_temporal` marker selects still inputs with temporal mode `3/1/2`, while the loader reports actual generated clips. Failure markers must be specific to this path.

- [x] **Step 1: Add combined integration tests.** Cover synthesis alone and synthesis plus randomized controls, full/LoRA, QAT, FP8 LoRA, EMA, anchoring, both loss profiles and tiny DINO. Assert each sampled control point stays constant over all frames, targets use the correctly ordered reference snapshots, and exports contain no teachers.
- [x] **Step 2: Add exact-resume and retry tests.** Compare raw/EMA/optimizer state, logical sample IDs, sampled controls, trajectories and RNG for uninterrupted/resumed runs across an epoch boundary and partial bucket tails. Changed shift, seed, sampler, clip length or source files must invalidate resume. Overflow retries must not resample controls or motion.
- [x] **Step 3: Add DDP equivalence and error tests.** Compare equivalent global accumulation with fractional masks, mixed source groups and short tails. Assert all control metrics are globally weighted once. Inject a rank-local synthetic sample failure and a reference failure, and verify both workers stop without complete checkpoint output.
- [x] **Step 4: Run the new synthetic and control acceptance files.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_control_randomization_training.py tests/test_dlssnr_control_randomization_distributed.py tests/test_dlssnr_synthetic_temporal.py tests/test_dlssnr_synthetic_temporal_training.py tests/test_dlssnr_synthetic_temporal_evaluation.py tests/test_dlssnr_synthetic_temporal_distributed.py`. Expected: all new tests pass, all processes exit, and legacy opt-out paths remain unchanged. Fix failures before broad verification.
- [x] **Step 5: Run the full DLSS-NR test selection once, then Ruff/format/diff checks on all touched files.** Enumerate `test_dlssnr*.py` using `rg --files`; do not select the whole repository with `-k` and accidentally collect unrelated missing-dependency modules. Record passes, skips and any reproduced pre-existing environment failures separately. No pretrained download or real CUDA/DLL workload is enabled implicitly.
- [x] **Step 6: Review both features against the approved spec and commit the acceptance tests/docs.** Commit message: `test(dlssnr): verify synthetic and randomized temporal training`.

The final handoff must state which code paths were verified and which real-data
experiments remain. Do not equate deterministic synthetic motion or target
construction with measured quality, monotonic controls, or native DLL acceptance.

## Final Verification

- New control and synthetic-temporal tests: 157 passed, no skips or failures.
- Full DLSS-NR selection: 897 passed, 25 skipped, 1 pre-existing optional FlashAttention extension failure (`flash_attn_2_cuda` DLL import).
- Bare repository pytest: collection blocked by missing `easydict` in four existing MiniMax H3 modules (generation modes, generation request, sampling and temporal stretch).
- Ruff, formatting and whitespace checks passed on touched files. No pretrained download or real-weight quality experiment was run.
- Review is by the implementer, not an independent agent, as requested. Commits remain local; no push or merge.
- Two narrow implementation corrections were verified: explicit CPU/FP32 private sampling, and exact fixed-control re-encoding after bucket interpolation on the enabled synthetic path.
