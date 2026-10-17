# DLSS-NR Random Controls and Synthetic Temporal Training

**Status:** Approved on 2026-10-08. Approach A, the detailed design and both
implementation plans were confirmed. Implementation and validation are recorded
in the corresponding plans.
**Baseline:** `DLSSNR` at `cf50676`.

## 1. Scope

Teach the new enhancement to vary with tone and structure, and train the existing
temporal model on subpixel sequences generated from paired still images. Reuse
the current full/LoRA trainer, QAT publication, data bucketing, EMA, evaluation
and optimizer-boundary resume.

Both additions are opt-in. Control randomization works with stills, real clips
and synthetic clips. Synthetic clips can also be used without random controls.
The existing default learning rates and objectives remain unchanged when the
features are disabled. No GUI or outer-launcher work is included in this stage.

The first version excludes random style selection, spatially varying randomized
controls, optical-flow estimation, rotation/zoom, generated occluders, unpaired
training data, new perceptual backends and changes to the exported model format.
Paired images must already be suitable supervision; moving both images together
does not correct a misregistered or redrawn target.

## 2. Control Semantics

The dataset's fixed controls are the reference point `c_ref` at which its target
photo represents the requested enhancement. Tone and structure are sampled
between zero and that reference point. Style and auto-mask stay fixed. Explicit
skin strength stays fixed; `nr_skin=-1` continues to follow structure through the
existing five-lane encoder.

Every participating training dataset must resolve to `nr_controls_mode=fixed`.
Manifest control maps are not reverse-engineered into knob values or silently
ignored. Both reference tone and structure must remain strictly positive after
the existing FP16 lane encoding. Different dataset entries may have different
reference points. The inference interface continues to accept actual native
control values, not normalized training strengths.

Let `a_t` and `a_s` be sampled tone/reference-tone and
structure/reference-structure ratios, computed from the *effective encoded*
values. This avoids giving different targets to values that collapse to the same
FP16 controls. One pair is sampled per logical image/clip and held constant over
all its frames, including burn-in.

### Sampling

Use a private CPU generator with a versioned seed derived from the training
seed, epoch, logical sample ID, crop ID and a control-specific domain tag.
Dataset/repeat prefixes are part of the logical sample ID. Rank, batch packing,
accumulation boundaries and optimizer-overflow attempts are not seed inputs.

Initial sampling probabilities are:

- 25%: the reference point `(1, 1)`.
- 25%: a uniform choice among `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`.
- 50%: independent continuous uniform ratios on `[0, 1]`.

The reference point therefore has total probability 31.25%, before FP16 aliases.
These are starting settings for experiments, not calibrated optimal proportions.
The reference and corner probabilities are configurable; their sum cannot
exceed one. A reference-only configuration is permitted for endpoint tests.

## 3. Control-Dependent Targets

Use the initial frozen NR model `F0` under the run's numerical policy. For LoRA,
this is the effective frozen base with adapters bypassed, including any selected
FP8 storage. It is never the current student or an EMA of the student.

For each clip, obtain reference outputs at `c_ref` and at sampled controls `c`.
Both passes use the same transformed inputs, frame-noise seeds, motion, reset
flags and burn-in as the student. Each pass rolls out its own history. Teacher
passes finish before constructing the student graph; checkpoint replay must
always see the adapters enabled.

Define `G_M` as the existing mask-normalized Gaussian low-pass operator. For a
teacher output field `B` and that loss's original reference target `A`, construct:

```text
R       = A - B(c_ref)
R_low   = G_M(R)
target  = B(c) + a_s * R + (a_t - a_s) * R_low
```

This is equivalent to scaling the low-frequency residual by tone and the
remaining residual by structure. Gaussian bands are not orthogonal, so it does
not assert an exact physical separation between the two controls. The rule
defines synthetic supervision for the new enhancement; it does not assert that
the original NR model responds linearly to its controls.

Use separate targets for the existing loss roles:

| Loss role | Teacher field `B` | Reference target `A` | Output handling |
| --- | --- | --- | --- |
| Preclamp | `neural_preclamp` | Paired target photo | Finite FP32, no RGB clamp |
| Rendered RGB, DINO and temporal residual | `rendered_proxy` | Paired target photo | Clamp to `[0,1]` |
| Edge, pixel profile | `rendered_proxy` | Paired target photo | Reuse rendered target |
| Edge, frequency-split profile | `rendered_proxy` | Current input RGB | Clamp to `[0,1]` |

Separate preclamp targets prevent a saturated but unchanged teacher from being
penalized merely because its preclamp values differ from its clamped render.
The control-dependent edge guide also prevents the input-edge loss from forcing
a change at the zero-enhancement endpoint.

At the effective reference point, return the original target values exactly on
supervised pixels rather than relying on floating-point cancellation. At
effective `(0,0)`, return the corresponding teacher fields exactly. Zero requests
the base response at zero controls; it does not request an identity image
transform. Intermediate targets follow the formula above. No gradient enters a
teacher or a target.

Excluded target pixels must not enter Gaussian filtering or temporal target
warps. Construct residuals with the supervised mask and fill excluded target
locations with the corresponding teacher values. Any non-finite reference or
constructed target aborts the update instead of being hidden by clipping.

The selected loss profile and weights still determine which discrepancies are
penalized. In particular, `frequency_split` does not become full-band pixel
regression when this feature is enabled. It may need the existing DINO term or
other weight choices to learn the desired high-frequency response. No additional
loss or feature model is enabled implicitly.

### Temporal support

For randomized targets and synthetic clips, temporal supervision uses joint
current/previous label support. Reuse the existing normalized warp of the masked
output-minus-target residual, even when the selected loss profile is `pixel`.
Apply a Gaussian afterward only for `frequency_split`.

The validity weight is the product of temporal validity, geometric inside/reset
validity, the current loss mask and the warped previous loss mask. Numerators
and global denominators use exactly that same weight. This stricter support is
an opt-in change for augmented pixel-profile training. Unaugmented pixel-profile
training retains its current temporal formula and support.

## 4. Synthetic Temporal Data

Generate clips lazily from paired stills after the existing shared bucket resize
or native-size crop. Do not write an expanded image/flow dataset to disk. Every
frame samples the original transformed pair, avoiding repeated resampling blur.

The first frame has sampling offset `d_0=(0,0)` and preserves the original input
and target values exactly. Each later frame samples independent x/y offsets
uniformly in `[-max_shift, max_shift]`, with default `max_shift=0.5` transformed
image pixels. Version one accepts finite values in `[0,1]`; zero provides a
static-sequence control experiment. This models subpixel jitter of a static
scene, not object motion.

For pixel center `p`, the frame samples the source at `p + d_t`. The corresponding
current-to-previous motion is therefore:

```text
motion_t = d_t - d_(t-1)
```

Use bilinear sampling with `align_corners=False` and the same pixel-center
convention as `temporal.warp_bilinear`. Source RGB and spatial control maps use
border extension only to provide finite model context. The validity masks must
exclude that extension. Fixed controls remain spatially constant.

The source image is sampled independently of its supervision mask. For labels,
sample `target * loss_mask` and the mask together, then divide by positive mask
coverage. This prevents an excluded target pixel from leaking into a retained
label. The new loss mask is the sampled coverage intersected with geometric
support. Zero-coverage target values use a neutral finite fill and are never
supervision. Frame zero avoids resampling and preserves its original masks.

Each generated clip has:

- Increasing frame indices starting at the source still's frame index.
- A reset, zero motion and invalid history/temporal masks on frame zero.
- Dense current-to-previous translation fields on later frames.
- Binary history validity requiring current source support and a fully supported
  bilinear footprint in the previous generated frame. Checking only the warped
  center or declaring the whole image valid is insufficient.
- Geometric temporal validity combined with the joint label support in section 3.

Validate that the source mask retains positive support away from the maximum
shift margin before training. Reject a sample with no such support instead of
randomly dropping it, changing its transform, or altering the batch plan.
Fractional resampling followed by a second warp is not generally identical to a
single resample; tests must not assume this for arbitrary textured images.

Generate jitter with a separate versioned random domain from control sampling
and model noise. The epoch-aware loader must use the final logical sample ID,
including collection/repeat prefixes. Direct dataset indexing remains a
deterministic epoch-zero view. No mutable global epoch or process-global RNG is
used by the dataset.

### Data sources and validation

Enable synthesis per dataset entry. A marked entry accepts directory pairs or a
single-frame paired training manifest. A real multi-frame training manifest
under that flag is rejected. In temporal mode, marked and ordinary real-clip
datasets may coexist with their existing dataset-specific batch sizes/repeats.

For a marked entry, `validation_manifest`, when present, is also a single-frame
paired manifest and is evaluated as synthetic clips at fixed epoch zero.
`sequence_manifest` always retains its existing meaning as a real temporal
sequence with explicit motion and masks. Reports identify synthetic cases and
their transform protocol. Control randomization does not change validation
targets or sample random validation controls.

Synthetic-case temporal MAE uses the same joint label support and normalized
masked residual warp as synthetic training. It must not measure neutral target
fill or border extension as real observations. Record this metric policy with
the synthetic case; leave existing real-sequence metric semantics unchanged.

Keep source sequence IDs when generating frames, so existing train/validation
overlap checks remain useful. Generating more frames does not create an
independent validation scene. The batch plan must report the actual generated
clip length, and fingerprints bind the original files and transform settings.

## 5. Configuration

Training options belong on the Python training CLI:

| Option | Default | Validation |
| --- | --- | --- |
| `--control_randomization` | Off | Requires fixed reference controls in every training entry |
| `--control_residual_sigma` | `6` when enabled | Finite, `0 < sigma <= 32` |
| `--control_anchor_probability` | `0.25` when enabled | Finite in `[0,1]` |
| `--control_corner_probability` | `0.25` when enabled | Finite in `[0,1]`; sum with anchor probability at most one |

Explicit control-specific options without `--control_randomization` are rejected
rather than ignored. Residual sigma defines the synthetic target bands; it is
independent of `--loss_lowpass_sigma`, which defines a training loss filter.

Dataset TOML gains these augmentation settings, inheritable from `[general]`:

| Field | Default | Validation |
| --- | --- | --- |
| `synthetic_temporal` | `false` | Boolean; requires temporal training |
| `synthetic_max_shift_px` | `0.5` when enabled | Finite in `[0,1]`; reject an explicit value when synthesis is disabled |

An entry that explicitly disables synthesis may discard an inherited shift
default, but must not declare its own shift value. A general shift setting with
no enabled dataset is unused and rejected.

The existing explicit `sequence_length`, `burn_in` and `tbptt_length` CLI settings
remain required, with `sequence_length = burn_in + tbptt_length`, burn-in at
least one, and at least two supervised frames when temporal loss is positive.
Synthesis does not silently enable `--loss_temporal` or change its weight.

Example dataset entry for using both features:

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "../data/targets"
control_directory = "../data/inputs"
nr_controls_mode = "fixed"
nr_tone = 1.0
nr_structure = 1.0
synthetic_temporal = true
synthetic_max_shift_px = 0.5
```

Options to append to a full or LoRA training command with the usual model,
dataset and output arguments:

```text
--training_mode temporal --sequence_length 4 --burn_in 2 --tbptt_length 2
--loss_temporal 0.1 --control_randomization --native_weight_qat
```

QAT still requires zero LoRA dropout. The example is a starting experiment;
neither its loss weight nor its sequence length is a quality recommendation.

## 6. Integration and Ownership

Keep augmentation responsibilities separate:

- `control_randomization.py`: deterministic sampling and detached target algebra.
- `synthetic_temporal.py`: epoch-aware still-to-clip sampling and validity rules.
- `base_anchor.py`: extend the existing frozen-reference provider to expose both
  rendered and preclamp outputs while retaining its current rendered-only anchor
  interface. Only one frozen base is owned by the prepared training module.
- `dataset.py`: normal and collection loaders pass epoch and final logical sample
  identity to augmentation; batch grouping and source fingerprints remain here.
  Collation retains the per-batch synthetic/temporal-support policy instead of
  dropping it when stacking tensors.
- `training_step.py`: explicit preclamp/edge target overrides and the augmented
  temporal-support policy, with identical numerator/denominator handling.
- `evaluation.py`: labeled synthetic cases and mask-consistent synthetic temporal
  metrics, without altering the existing real-sequence path.
- Parser/config/trainer: validation, data-source construction, reference ownership,
  deterministic batch preparation, metadata and weighted logging.

Prepare geometric augmentation and sample controls before computing the global
loss denominators. Construct image targets in the prepared training module under
disabled autocast and `no_grad`, then run the student normally. The student,
both references and target construction all receive the same finalized masks.

Full training needs one frozen initial model when random controls are enabled.
LoRA reuses its frozen effective base with adapters temporarily disabled. Reuse
the same provider if base anchoring is enabled, and reuse the sampled-control
reference output for that loss. Distinct control points normally add two
reference rollouts; identical reference points can share one rollout. Do not
duplicate the base model for the two losses.

Reference allocation does not enable the base-anchor loss. Its term, denominator
and logs remain conditional on a positive `--base_anchor_weight`, including when
random controls alone require a frozen reference.

Frozen references are initialized before restoring student state and remain
outside the optimizer, EMA and exported weights. Preserve module modes and all
training RNG streams on successful calls and exceptions. Dataset/teacher failures
must propagate through the existing coordinated DDP error handling.

## 7. Reproducibility and Evidence

Record sampler versions, probabilities, residual sigma, effective reference
controls, synthetic geometry and interpolation/mask rules in the resolved run
metadata. Bind them, frozen-reference weight/runtime identity and the new
implementation files to exact resume. Configuration or implementation changes
require a new run rather than silently changing a resumed experiment.

Sampling must be reproducible for a logical sample and epoch across different
rank assignments. Retries reuse the same already-prepared microbatches. Retain
the existing valid-pixel normalization over all accumulated microbatches/ranks,
including fractional masks and short bucket tails.

Log effective tone/structure means, reference/zero endpoint fractions and target
clipping fractions. Use valid RGB mass for these summaries, publish the same
metric keys on every rank, and aggregate sums/counts rather than local means.
Keep synthetic-target adherence separate from real paired-photo quality claims.

The existing control scanner remains the inspection tool for output changes
across controls. Ordinary held-out photo metrics use the reference controls;
synthetic temporal reports are labeled, and real-sequence reports stay separate.
Raw and EMA candidates are evaluated under their recorded policy and optional
native-quantized proxy, as they are now.

## 8. Acceptance Tests

1. Deterministic sampling: domain-separated RNG, effective FP16 controls, fixed
   style/skin rules, per-clip constancy, epoch variation, repeat identities and
   unchanged Python/NumPy/Torch training RNG.
2. Target algebra: independent low/high residual oracle, exact reference and
   zero endpoints with a non-identity base, unclamped preclamp endpoint,
   frequency-profile edge guide, soft masks, holes, clipping and detached targets.
3. Geometry: known integer/fractional translations, motion sign, pixel-center
   convention, zero-shift identity, independent resampling of each frame, strict
   previous-frame footprint validity and no excluded-label leakage.
4. Dataset contracts: folders and single-frame manifests, mixed real/synthetic
   entries, repeats, variable buckets, batch-plan frame counts, fixed synthetic
   validation, fingerprints and clear rejection of unsupported inputs.
5. Gradients and reductions: temporal head/blend receive finite gradients in full
   training, supported LoRA parameters receive gradients, teachers never do, and
   numerator/denominator agreement holds with fractional or empty local support.
6. Integration: full/LoRA, QAT, frozen FP8 LoRA, EMA, base anchoring, both loss
   profiles, DINO and activation-checkpoint replay; no teacher tensors in exports.
7. Exact recovery: uninterrupted versus resumed training, overflow retries,
   accumulation and two-process Gloo tests, including one-rank failures without
   hangs and rejection of changed augmentation/reference identities.
8. Disabled behavior: no reference allocation for disabled randomization unless
   the existing base-anchor loss requests one; no new data resampling, changes to
   legacy losses or consumption of the training RNG.

These tests establish implementation behavior. Real-weight control response,
clipping rates, detail/content retention and real temporal stability still need
held-out experiments on the training machine. Synthetic subpixel consistency
does not establish behavior under occlusion, moving objects or actual DLL history
sampling. Changes to `mix`, `strength` or LoRA multiplier remain outside the
QAT training/export alignment configuration.
