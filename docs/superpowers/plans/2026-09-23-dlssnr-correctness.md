# DLSS-NR Correctness Repair Plan

> Execution: inline, without subagents, as requested by the user. Use test-driven development and a final self-review. Do not commit the user's previously untracked work automatically.

**Goal:** Repair the six reviewed defects without changing existing diffusion training paths.

**Architecture:** Keep thin root CLI wrappers, NR algorithms in `dlssnr/`, adapters in `networks/`, and the shared supervised runner in `training/`. Full and LoRA training use one train module and update loop. Unsupported backends remain explicit errors.

**Tech Stack:** Python 3.10+, PyTorch, Accelerate, NumPy, safetensors, TOML, pytest.

**Spec:** `docs/dlssnr_310_8_0_musubi_tuner_spec_v0.2.md`.

## Constraints

- FP32 train_surrogate, single process; do not claim native parity.
- No changes to old architecture entrypoints or diffusion mathematics.
- Pretrained canonical input is mandatory except for explicit development smoke runs.
- Save at optimizer update boundaries; full and LoRA accumulation/save/resume must agree.
- Preserve source metadata and opaque records; identify configuration and data by content.

## Tasks

- [x] 1. Add independent coordinate-window and attention-adapter regression tests, observe failures, fix window order and projection dispatch; test non-default LoRA ranks.
- [x] 2. Add strict config tests for unknown fields, invalid values, model/profile requirements, and unsupported features. Move schema validation into `dlssnr/config.py`.
- [x] 3. Add lazy manifest and identity tests. Validate ordering, encodings, masks, finite arrays, and train/validation separation before updates.
- [x] 4. Add shared training-step and runner tests for CPU/CUDA placement, full/LoRA accumulation, temporal gradients, finite updates, and deterministic seeds. Keep runtime services in `training/dlssnr_services.py`.
- [x] 5. Add canonical artifact, identity mismatch, and resume tests. Preserve metadata/opaque data and all update-boundary state for full and LoRA outputs.
- [x] 6. Execute validation manifests with independent history and preserved RNG/mode; compare the frozen initial baseline. Update commands and implementation status; run focused and repository-wide tests plus lint.

## Review Focus

- Nonuniform pixels and shifted edge windows must never exchange spatial identities.
- Every declared LoRA target must affect the real forward and match merged inference.
- Invalid or ignored configuration must fail before expensive model allocation.
- Edited source data or base weights must invalidate exact resume, while TOML comments must not.
- CUDA, accumulation, dropout, and evaluation must not change sample/noise order after resume.

## Verification

Run with `PYTHONPATH=src`, `OMP_NUM_THREADS=4`, and `MKL_NUM_THREADS=4`.

```powershell
python -m pytest -q tests/test_dlssnr_forward.py tests/test_dlssnr_lora.py
python -m pytest -q tests/test_dlssnr_config.py tests/test_dlssnr_dataset.py tests/test_dlssnr_training.py
python -m pytest -q
```

## Decisions And Evidence

- The existing `DLSSNR` checkout contains the user's untracked implementation. Work in place; do not create a worktree that would omit it.
- The deployment target remains float_runtime; native export and numerical-reference implementation are not part of these repairs.
- Cache construction remains unsupported and rejected. Evaluation and checkpoint options must execute rather than be silently ignored.
- Window and projection regression tests: 11 failed before repairs, then 11 passed. Strict config: 20 failed before repairs, then 20 passed. Lazy dataset: 8 passed after fixes.
- Shared runner tests covered full/LoRA single/temporal accumulation and exact resume, changed data, evaluation RNG, split overlap and non-finite gradients: 9 passed before adding final-state/TF32 cases.
- A combined run had 71 passes and one resume identity rejection because source files changed during the test. Repeating the real-model CUDA resume test without edits passed (65 seconds). Do not edit implementation files while identity-sensitive tests run.
- Repository-wide verification before the final review fix: 156 passed, 2 opt-in CUDA cases skipped (290 seconds). Focused lint passed.
- Final review: self-review, honoring the user's prohibition on subagents. Found that the lane-15 constraint rewrote signed zero in a frozen base. Added a failing full-base-identity test and restricted the constraint to trainable base parameters. Also corrected the converter's stale surrogate-status metadata; conversion alone never grants layout/float/native validation labels.
- Real-weight CUDA testing exposed merged/unmerged drift despite the small projection tests passing. Diagnostics: blocks 0 and 30 were identical, block 31 diverged by 1.46; individual projection rounding differed by about 1e-5. Ruling: compute LoRA with canonical W + delta, with a rank-dropout correction when enabled, rather than relaxing parity tolerance. This preserves the mathematical update but costs temporary dense weights and more backward work; record the forward mode in adapters.
- Final verification: `DLSSNR_RUN_CUDA_SMOKE=1 python -m pytest -q --tb=short` completed with 159 passed, no skips, in 319.73 seconds. Targeted Ruff checks passed. The original 310.8.0 CUDA full/LoRA update tests and merged raw-head comparison passed without changing their tolerance.
- No subagents, branch changes, commits, dependency installations, or edits to existing diffusion architecture files. Native/reference parity, real-dataset quality, and long real-model video acceptance remain outside this repair's completed claims.
