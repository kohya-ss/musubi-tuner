# DLSS-NR Control Randomization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Train control-dependent enhancements against detached, base-relative targets without changing disabled behavior.

**Architecture:** Sample one effective control point per logical sample and epoch. Reuse one frozen reference provider for reference-point and sampled-point predictions, construct loss-specific targets, and retain the existing optimizer/DDP runner. Shared temporal-mask helpers will also support the separately planned synthetic clips.

**Tech Stack:** Python, PyTorch, existing Accelerate/Gloo, pytest and Ruff. No new dependency.

**Spec:** `docs/superpowers/specs/2026-10-08-dlssnr-control-randomization-synthetic-temporal-design.md`, especially sections 2, 3 and 5-8.

**Execution:** Native/inline in this chat, without subagents or delegated reviews. This plan precedes `2026-10-08-dlssnr-synthetic-temporal.md`; control randomization is usable independently when this plan is complete. Approved and implemented inline on 2026-10-08. See the final verification record below.

## Global Constraints

- Start from the approved design on `DLSSNR`; preserve unrelated work and do not change GUI, outer launchers or learning-rate defaults.
- Both additions are opt-in. Existing behavior remains unchanged when disabled.
- Every randomized dataset resolves to fixed controls with positive tone/structure after FP16 encoding. Style/auto-mask stay fixed; skin retains the existing fixed/follow-structure rule.
- Sampling defaults: 25% reference, 25% uniform four corners, 50% independent uniform ratios. Use actual encoded ratios and one point for the whole clip.
- Residual sigma defaults to 6 and must satisfy `0 < sigma <= 32`. Probabilities are finite in `[0,1]` with sum at most one.
- Targets are detached FP32; preclamp targets are not RGB-clamped. Teacher state is initial, frozen, mode/RNG-preserving and never exported or optimized.
- Use current valid RGB mass and joint temporal label support consistently across accumulation and DDP. No extra training loss is enabled implicitly.
- Real-weight quality, GPU performance and DLL/game acceptance remain separate experiments.

## Review Focus

1. A positive requested reference can round to zero, or a random point can alias a reference endpoint in FP16: reject the former and use exact effective endpoints for the latter (Tasks 1 and 4).
2. Saturated preclamp values and the input-edge objective can fight a zero-enhancement target: all loss roles must match the base at that endpoint (Tasks 2 and 3).
3. Randomization may need a teacher while `base_anchor_weight=0`: do not accidentally enable the anchor denominator, loss or logs (Task 4).
4. Fractional masks and short rank-local batches must not bias control means or endpoint/clipping fractions (Tasks 4 and 5).
5. A failing reference or checkpoint replay must not leave adapters disabled or advance the training RNG (Tasks 2 and 5).

## Files and Verification

Create `src/musubi_tuner/dlssnr/control_randomization.py` and three tests:
`tests/test_dlssnr_control_randomization.py`,
`tests/test_dlssnr_control_randomization_training.py`, and
`tests/test_dlssnr_control_randomization_distributed.py`.
Extend the existing `temporal.py`, `base_anchor.py`, `losses.py`, `training_step.py`,
`dataset.py`, parser/config/trainer and the shared distributed test harness.

Run verification from the checkout root with `PYTHONPATH=<checkout>/src`,
`OMP_NUM_THREADS=2` and `MKL_NUM_THREADS=2` applied to each test process. Use the
project Python (`E:/Python310/python.exe` in the current environment). Commands
below use `python` for that executable. Optional dependency skips and the known
local FlashAttention DLL failure must be reported, not disguised as a green
suite. Focused tests use synthetic CPU fixtures; do not download pretrained weights.

## Task 1: Effective Control Sampling

**Files:** Modify `src/musubi_tuner/dlssnr/temporal.py`; create `src/musubi_tuner/dlssnr/control_randomization.py` and `tests/test_dlssnr_control_randomization.py`.

**Interfaces:**
- Add `augmentation_seed(seed: int, epoch: int, sample_id: str, crop_id: int, *, domain: str) -> int` and `AUGMENTATION_SEED_POLICY = 'dlssnr_augmentation_seed_v1'` in `temporal.py`. Hash `[policy, domain, seed, epoch, sample_id, crop_id]` with existing `json_sha256`; interpret the first 16 hex digits as the private CPU generator seed. Keep `stable_frame_seed` unchanged.
- Add `encode_control_point(reference: dict, ratios: tuple[float, float]) -> dict[str, Tensor]`. Return CPU FP32 `reference_lanes[5]`, `sampled_lanes[5]`, effective `ratios[2]`, and actual tone/structure `values[2]`.
- Add `sample_control_point(reference: dict, settings: dict, *, seed: int, epoch: int, sample_id: str, crop_id: int) -> dict[str, Tensor]`. Use domain `controls`, then the encoder above. `settings` contains `anchor_probability`, `corner_probability` and `residual_sigma`.

- [x] **Step 1: Write failing encoding and sampling tests.** Include these assertions, both auto-mask modes, fixed/following skin, epoch/repeat changes, RNG preservation and the three distribution branches:
  ```python
  point = encode_control_point({'nr_tone': 1, 'nr_structure': 1}, (0.9999, 0.9999))
  assert point['ratios'].tolist() == [1.0, 1.0]
  with pytest.raises(ValueError, match='reference'):
      encode_control_point({'nr_tone': 1e-12, 'nr_structure': 1}, (1, 1))
  assert augmentation_seed(4, 2, 'd0-r0-a', 0, domain='controls') != augmentation_seed(
      4, 2, 'd0-r0-a', 0, domain='synthetic_jitter')
  ```
  Name the endpoint test `test_effective_control_alias_uses_exact_endpoint` and the rejection test `test_reference_tone_that_rounds_to_zero_is_rejected`.
- [x] **Step 2: Run RED.** `python -m pytest -q tests/test_dlssnr_control_randomization.py`; expect missing APIs, not a dependency error.
- [x] **Step 3: Implement the interfaces.** Use `resolve_fixed_controls` and `fixed_control_tensor`, not hand-built feature lanes. Validate ratios/probabilities; never consume process-global RNG. Choose the reference/corner/continuous branch with the specified probabilities.
- [x] **Step 4: Run GREEN and the existing control/shuffle tests.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_control_scan.py tests/test_dlssnr_shuffle.py`.
- [x] **Step 5: Review the diff and commit the sampler and tests.** Commit message: `feat(dlssnr): sample reproducible effective control points`.

## Task 2: Reference Snapshots and Target Algebra

**Files:** Modify `src/musubi_tuner/dlssnr/base_anchor.py`, `src/musubi_tuner/dlssnr/control_randomization.py`, `tests/test_dlssnr_base_anchor.py`, and `tests/test_dlssnr_control_randomization.py`.

**Interfaces:**
- Add `NRBaseAnchor.predict(model, network, batch, seeds, burn_in) -> dict[str, Tensor]`, returning `neural_preclamp` and `rendered_proxy` in existing time-major supervised order `[N,3,H,W]`. Its current `forward` remains a rendered-only wrapper. Expose `reference_identity` containing base kind/hash, runtime, conditioning and rollout policy without mislabeling the provider as an enabled loss.
- Add `control_targets(source, target, mask, reference, sampled, ratios, *, sigma: float, input_edge: bool) -> dict[str, Tensor]`. All image arguments and snapshot fields are `[N,3,H,W]`, mask is `[N,1,H,W]`, and ratios are `[N,2]`. Return detached `target`, `preclamp_target`, `edge_target`, `rgb_clipped` and `edge_clipped`; the last two are per-RGB boolean indicators before clamping.

- [x] **Step 1: Add failing tests for both snapshot fields, time-major order and existing anchor equivalence.** Reuse `SmallNR`, `small_inject`, `batch_and_seeds`, and the existing RNG/mode failure fixture. Assert the rendered wrapper is exactly equal to the snapshot's rendered field, and no returned value has a gradient graph.
- [x] **Step 2: Add target tests with an independent 2-D Gaussian oracle.** In `test_zero_point_preserves_preclamp_and_frequency_edge`, use a non-identity base with preclamp 1.3 and rendered 0.8, input 0.2 and target photo 0.6:
  ```python
  result = control_targets(source, target, mask, reference, sampled,
                           torch.zeros(1, 2), sigma=6, input_edge=True)
  torch.testing.assert_close(result['preclamp_target'], sampled['neural_preclamp'], rtol=0, atol=0)
  torch.testing.assert_close(result['target'], sampled['rendered_proxy'], rtol=0, atol=0)
  torch.testing.assert_close(result['edge_target'], sampled['rendered_proxy'], rtol=0, atol=0)
  ```
  Also assert exact reference targets on positive support, bounded RGB/edge targets, unclamped preclamp targets, correct clipping indicators, detached outputs, no input mutation, soft/empty masks, excluded-label invariance and finite-value errors.
- [x] **Step 3: Run RED.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_base_anchor.py`.
- [x] **Step 4: Implement the provider extension and flat target kernel.** Retain the existing exception-safe adapter/mode/RNG contexts. Apply the spec's formula separately to each loss role with `masked_gaussian_lowpass`; select effective endpoints exactly and fill excluded target positions with the appropriate teacher field.
- [x] **Step 5: Run GREEN, including checkpoint/reference regression tests.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_base_anchor.py tests/test_dlssnr_base_anchor_training.py`.
- [x] **Step 6: Review and commit.** Commit message: `feat(dlssnr): construct detached control-dependent targets`.

## Task 3: Loss-Role Overrides and Shared Temporal Support

**Files:** Modify `src/musubi_tuner/dlssnr/losses.py`, `src/musubi_tuner/dlssnr/training_step.py`, and `tests/test_dlssnr_control_randomization.py`.

**Interfaces:**
- Add `joint_temporal_support(current_mask, previous_mask, motion, temporal_valid, reset) -> tuple[Tensor, Tensor]` in `losses.py`: return the joint validity weight and warped previous mask, both `[B,1,H,W]`.
- Add `masked_temporal_residual(current, previous, target, previous_target, current_mask, previous_mask, motion, temporal_valid, reset) -> tuple[Tensor, Tensor]`: return the normalized masked residual error and the same validity weight. Do not apply a Gaussian inside this helper.
- Add `supervised_values(value: Tensor, burn_in: int) -> Tensor` in `training_step.py`: BCHW passes through; BTCHW becomes supervised time-major BCHW.
- Extend the batch contract with optional `preclamp_target` and `edge_target`, each exactly the shape of `target`, plus `temporal_support='joint_loss_mask'`. Unknown policies and broadcastable-but-wrong target shapes are errors. Existing public loss signatures remain usable.

- [x] **Step 1: Write RED tests for target-role routing and joint support.** `test_zero_point_has_zero_loss_in_both_profiles` must compare a student reproducing both teacher fields with the generated targets and assert zero loss/gradients. Parameterize pixel and frequency-split profiles.
- [x] **Step 2: Add `test_joint_temporal_denominator_matches_residual_with_soft_masks` and `test_joint_policy_does_not_square_frequency_support`.** Use fractional motion, mask holes, fractional masks, reset/outside cases and an empty local mask. Assert microbatch gradients equal concatenated global-mask gradients; perturbing excluded targets must not change loss. Assert legacy unaugmented pixel results remain unchanged.
- [x] **Step 3: Run RED.** `python -m pytest -q tests/test_dlssnr_control_randomization.py`.
- [x] **Step 4: Implement the shared helpers and routing.** Reuse the current frequency-profile normalized warp. Joint support activates when the batch requests it or frequency-split already needs it; apply the mask product once. Smooth only the frequency profile. Use target overrides for preclamp/edge losses and the rendered target for DINO/temporal terms. Name an overridden edge metric `loss/control_edge` rather than claiming it always compares with the input.
- [x] **Step 5: Run GREEN and loss regressions.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_frequency_loss.py tests/test_dlssnr_dino_training.py tests/test_dlssnr_training.py`.
- [x] **Step 6: Review and commit.** Commit message: `feat(dlssnr): support controlled targets and joint temporal masks`.

## Task 4: Configuration and Training Integration

**Files:** Modify `src/musubi_tuner/training/dlssnr_parser.py`, `src/musubi_tuner/dlssnr/config.py`, `src/musubi_tuner/dlssnr/dataset.py`, `src/musubi_tuner/training/dlssnr_trainer.py`, `src/musubi_tuner/dlssnr/control_randomization.py`, and `docs/dlssnr.md`; create `tests/test_dlssnr_control_randomization_training.py`.

**Interfaces:**
- Parse the four control options from spec section 5. Store enabled settings only at `config['control_randomization']`, with schema `dlssnr_control_randomization_v1`, the three resolved numeric settings and the seed policy. Validate all fixed references before model/reference allocation.
- Fixed-control datasets expose a copied `fixed_controls` dictionary in sample metadata; collections retain it while applying their logical sample-ID prefixes.
- Add `attach_control_batch(batch: dict, draws: list[dict]) -> dict`. Broadcast sampled lanes over images/frames, store `control_reference_lanes[B,5]`, `control_ratios[B,2]`, `control_values[B,2]`, and request joint temporal support without mutating the original batch.
- Add `apply_control_targets(batch, reference, sampled, *, burn_in: int, settings: dict, loss_profile: dict | None) -> tuple[dict, dict[str, float]]`. Use `supervised_values`, repeat per-clip ratios in time-major order, invoke Task 2's kernel, and restore the full target shapes with untouched burn-in labels. Return raw statistics under `control/_mass`, `control/_tone`, `control/_structure`, `control/_reference`, `control/_zero`, `control/_rgb_clipped`, `control/_edge_clipped`.
- Statistics use float64 sums: mass is `3*sum(mask)`, knob and endpoint numerators use that per-image mass, and clipping numerators sum per-RGB clipping indicators times the mask. Include every key even when its numerator is zero.
- Add `finalize_control_metrics(metrics: dict[str, float]) -> dict[str, float]`, replacing summed raw counters with `control/tone_mean`, `control/structure_mean`, `control/reference_fraction`, `control/zero_fraction`, `control/rgb_clipped_fraction`, `control/edge_clipped_fraction` after cross-rank reduction.

- [x] **Step 1: Add the test helper `fixed_args(tmp_path, *, lora=False, mode='single_frame', evaluate=False)`.** Wrap existing `make_args` and change each dataset entry to fixed controls with tone/structure 1. Keep the real parser, data loader, optimizer and persistence paths.
- [x] **Step 2: Write failing configuration tests.** Assert disabled settings add no config key, enabled defaults are 6/0.25/0.25, invalid/ignored options fail, files-mode or FP16-zero references fail before teacher construction, and CLI/config hashes bind every setting.
- [x] **Step 3: Write `test_randomization_teacher_does_not_enable_base_anchor`.** Run two CPU updates with `base_anchor_weight=0`; assert a single frozen provider exists, no anchor term/denominator/log is added, and exported parameter names equal the ordinary model/adapter map. Add `test_control_metrics_use_global_rgb_mass` using two unequal fractional masks and hand-computed sums.
- [x] **Step 4: Run RED.** `python -m pytest -q tests/test_dlssnr_control_randomization_training.py`.
- [x] **Step 5: Implement the CLI/config and batch attachment.** `_microbatch` samples from final logical IDs and epoch before global denominators are computed. Do not replace the frame-noise seed policy. `_initialize_model` behavior stays intact.
- [x] **Step 6: Integrate reference/target construction in `NRTrainModule.forward`.** Allocate a provider when either feature needs one, but gate anchor terms/denominators by the actual anchor weight. Predict the sampled controls, predict reference controls or reuse a wholly reference-point batch, construct targets, then build the student graph. Reuse the sampled prediction for optional base anchoring. Merge raw control statistics with loss metrics and finalize them only after global reduction.
- [x] **Step 7: Bind reference/sampler implementation identities and document usage/cost.** Disabled runs do not load a reference unless anchoring independently requests it. Reference objects never enter raw/EMA exports. Do not change inference or DLL publication formats.
- [x] **Step 8: Run GREEN and nearby regressions.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_control_randomization_training.py tests/test_dlssnr_base_anchor_training.py tests/test_dlssnr_config.py tests/test_dlssnr_training.py tests/test_dlssnr_ema.py`.
- [x] **Step 9: Review and commit.** Commit message: `feat(dlssnr): integrate randomized control supervision`.

## Task 5: Resume, Runtime and Distributed Acceptance

**Files:** Extend the two new test modules; create `tests/test_dlssnr_control_randomization_distributed.py`; extend only the necessary marker/recording hooks in `tests/test_dlssnr_distributed.py`.

**Interfaces:** Reuse `SmallNR`, `small_math`, `TinyFP8NR`, `tiny_fp8_inject`, `tiny_dino_loss`, `_data`, `_args` and `_launch`. Add a `control_randomization` worker marker that sets the CLI options and fixed dataset controls, plus narrowly scoped failure markers. Do not duplicate the worker launcher.

- [x] **Step 1: Add endpoint and recovery tests.** A reference-only run with full masks must preserve ordinary update weights. Test full/LoRA, single/real temporal clips, pixel/frequency profiles, QAT, FP8 LoRA, EMA, optional anchor and DINO. Assert uninterrupted/resumed raw weights, EMA, optimizer and per-rank RNG match; reject changed sampling settings or reference identity.
- [x] **Step 2: Add failure/replay tests.** `test_control_reference_failure_restores_adapters_and_rng` and `test_checkpoint_replay_after_control_references_keeps_adapters_enabled` must check nested modes, all RNG families and actual recomputation. Simulated overflow retries must reuse the sampled points and leave EMA/update counters correct.
- [x] **Step 3: Add two-process tests.** Compare no-dropout DDP with equivalent single-process accumulation using uneven buckets and fractional masks. Assert globally weighted control metrics agree. Test dropout resume separately, and make a rank-specific teacher failure abort both ranks without writing a complete checkpoint.
- [x] **Step 4: Run the new acceptance files and fix evidenced failures.** `python -m pytest -q tests/test_dlssnr_control_randomization.py tests/test_dlssnr_control_randomization_training.py tests/test_dlssnr_control_randomization_distributed.py`. Expected: no new failures. All worker processes must finish.
- [x] **Step 5: Run Ruff, formatting and `git diff --check` on touched files; review against the spec.** Commit message: `test(dlssnr): verify randomized controls across runtimes and resume`.

Continue with the synthetic-temporal plan only after this independently usable
feature passes its acceptance tests. Do not push or claim actual model quality
from these synthetic CPU results.

## Final Verification

All five tasks were implemented and tested before starting synthetic clips. The combined run subsequently passed all 157 new-feature tests. Full-suite results and environment limitations are recorded in the synthetic-temporal plan. No GUI, launcher, learning-rate default or export-format change was made.
