# DLSS-NR HT / CC Training Support

## Overview

This is an experimental, standalone supervised-training path for DLSS-NR HT (310.8.0/310.9.16) and CC (310.9.16),
not a diffusion trainer. Native DLL equivalence remains unverified. It provides
canonical weight conversion, checked DLL weight import/export, default FP32
full/LoRA training, still/closed-loop inference, adapter merging and checked
optimizer-state resume. The two dataset examples enable the
project's shared resolution buckets with a 1024 x 1024 area budget and `bucket_no_upscale` enabled.
Set `enable_bucket = false` for the original strict fixed-resolution behavior.
Each input and its target, controls, motion and masks must share the original pixel grid.

Users must provide their own weights, paired RGB data, fixed condition settings or encoded control lanes and,
for non-reset temporal frames, motion and validity masks. No weights, DLLs or game
assets are included. See [source attribution](../src/musubi_tuner/dlssnr/NOTICE.md).

Normal full/LoRA training requires pretrained canonical weights, complete
conversion provenance and a successful source round-trip, but no forward-validation
report. Optional evidence is checked only when explicitly supplied; training
eligibility does not certify native/DLL compatibility. `--development_smoke`
remains a developer-only option for incomplete provenance or random initialization.
The surrogate uses half/E4M3 forward
publications with surrogate gradients and FP32 master weights; it is not ordinary
FP32 SiLU/softmax and is not bit-identical to the DLL. Non-reentrant gradient
checkpointing and single-node DDP retain the default numerical profile.
CUDA FP16/BF16, LoRA-only FP8 frozen storage and alternative attention require
explicit `--numerics_profile train_experimental`. SDPA supports windows and
global attention; optional FlashAttention is global-only, xFormers is capability
checked, and SageAttention is inference-only. Missing or incompatible optional
extensions fail explicitly, without fallback to another named backend.
Outputs precede the DLL's Natural/Cinematic color grading.

Training writes all artifacts under `output_dir/output_name/`. A new run refuses
an existing run directory; use `--resume` for a matching checkpoint. Sample IDs
must be portable filenames, unique without regard to case. The commands and
manifest contract below apply to both full and LoRA training.

Training defaults to `--profile auto`, reading the architecture from
`--model_dir/model_config.json`; inference and LoRA merging do the same.
The explicit profiles are `dlss_nr_310_8_0` (HT) and `dlss_nr_310_9_16_cc` (CC).
An explicit training profile must match the checkpoint. Without a model directory,
development smoke mode still defaults to HT. HT and CC adapters are not interchangeable.

Use `--dataset_config` for a **dataset-only TOML** with `[general]` and one or more
`[[datasets]]` entries. Model, optimizer, LoRA, loss, evaluation and output settings
are CLI arguments, using the project's standard names such as `--optimizer_type`,
`--optimizer_args`, `--learning_rate`, `--network_dim` and `--network_alpha`.
The old all-in-one `--config_file` training interface is no longer accepted.
Dataset-relative paths resolve against the TOML directory; CLI paths resolve
against the working directory. The generated `run_config.json` is an effective
configuration snapshot for provenance/resume, not another user configuration file.

Paired supervision uses FP32 master weights and matrix accumulation by default.
Intermediate activations, attention and reductions include explicit half/E4M3
fake quantization and surrogate gradients. Experimental modes can change matrix
precision and attention kernels, but require separate native-equivalence checks.
Output remains in proxy color space, before the DLL's Natural/Cinematic grading.

## Configuration

As with other architectures in this project, datasets use `--dataset_config`
and training hyperparameters use CLI arguments. Single-frame full and LoRA
training share the [single-frame dataset example](../configs/dlssnr_dataset_single.toml).
Temporal training uses the [temporal dataset example](../configs/dlssnr_dataset_temporal.toml).

```toml
[general]
resolution = [1024, 1024]
batch_size = 1
enable_bucket = true
bucket_no_upscale = true

[[datasets]]
image_directory = "../data/target"
control_directory = "../data/input"
num_repeats = 1
nr_controls_mode = "fixed"
nr_style = 0
nr_tone = 1.0
nr_structure = 1.0
nr_skin = -1.0
nr_auto_mask = true
# validation_manifest = "../data/validation_single.jsonl"
# sequence_manifest = "../data/validation_sequences.jsonl"
```

Each `[[datasets]]` entry can override `resolution`, `batch_size`, `enable_bucket`,
`bucket_no_upscale` and `num_repeats` from `[general]`. Datasets retain their own
batch sizes, repeat counts, dimensions and conditions. Directory mode reuses the
project's image-editing source-matching rules, but each target must match exactly
one input control image. Missing or ambiguous matches and mismatched pixel grids
raise errors; the loader never picks the first match or resizes inputs independently.

The default `resolution` is `[1024, 1024]`; existing explicit dimensions are
unchanged. With bucketing enabled, this is an area budget, not a requirement to
stretch every image into a square.

The five condition channels are generated at load time from `nr_style`, `nr_tone`,
`nr_structure`, `nr_skin` and `nr_auto_mask`, without additional JSONL/NPY files.
Tone and structure range from `0` to `1`; skin accepts `[0,1]` or `-1` to follow
structure. Auto mask is a model condition: it does not generate supervision masks
or add a segmentation model. Style changes the network condition only; output
still excludes the DLL's postprocessing color grading.

Existing data can still use `train_manifest` instead of the two directory fields.
Its default `nr_controls_mode = "files"` reads the existing `controls_path`.
Selecting `fixed` explicitly overrides `controls_path` in that dataset and its
validation manifests with the TOML conditions; overridden NPY files are not read.
Real temporal data requires a manifest declaring frame order, motion and validity
masks. Directory pairs or single-frame manifests can instead opt into
`synthetic_temporal = true`, as described under Synthetic Temporal Clips below;
ordinary still images are not automatically treated as clips.
`caption_extension` and `cache_directory` may remain in shared dataset settings,
but NR reads neither captions nor diffusion caches.

Legacy TOML training sections such as `[model]`, `[training]`, `[optimizer]`,
`[lora]`, `[loss]` and `[output]` raise errors rather than providing a second source
of hyperparameters. The generated `run_config.json` stores fully resolved values
for provenance and resume validation only.

## Bucketing and Paired Transforms

Bucketing uses the shared `BucketSelector`, and same-bucket batches use the public
sample-selection interface of `BucketBatchManager`. Neither reads diffusion latent
or text-encoder caches. NR has its own architecture ID, `nr`, with a bucket step of
16 and a minimum of 48 pixels on each axis. Internal padded-field geometry and
forward computation are unchanged.

- With `enable_bucket = true`, the product of the `resolution` dimensions is the area budget. Bucket selection follows the original aspect ratio. For example, a `128 x 128` budget maps a `320 x 192` input to a `160 x 96` bucket; portrait inputs use the corresponding portrait bucket.
- `bucket_no_upscale = true` follows the shared bucket selection. Images within the area budget are rounded down to multiples of 16 and **center-cropped without resampling**. If either resulting axis is below 48, the sample is rejected rather than enlarged or padded. Images over budget still select aspect-ratio candidates and use cover resizing/cropping; extreme aspect ratios can enlarge the shorter axis, so this flag does not prohibit every possible upscale.
- Omitting `enable_bucket` keeps it disabled and retains strict fixed-resolution behavior. All paired tensors must already match `resolution`; mismatches raise errors. The example TOMLs explicitly enable bucketing and no-upscale.
- Paired tensors must align on the original pixel grid. Every frame in a clip must share the original input dimensions, bucket and center-crop position, with the same cover resize when one is needed. There are no per-frame random crops or independent target/control resizes to hide bad pairing.
- When resizing, RGB, encoded controls and continuous loss masks share FP32 bilinear/antialias sampling. Binary history/temporal masks use pixel-center-aligned nearest-exact sampling to remain binary. Motion uses bilinear sampling; x/y displacement is scaled by the actual rounded width/height ratios. The native-size crop skips all interpolation and leaves retained motion vectors unchanged. Identical current/previous-frame crop offsets cancel out, and the reprojection inside mask still excludes history outside the crop.
- A microbatch contains only samples with matching dimensions and frame counts. Different buckets can participate in one gradient-accumulation window; loss remains normalized by the total valid pixel count. Each bucket's short final batch is retained without dropping or duplicate padding. `consumed_samples` records the actual count.

Indexing reads image headers or NPY mmap shapes to select buckets without keeping
pixels resident. Before training, each sample is decoded to validate pairing,
values and the remaining supervision region after transformation. Training and
validation share transform rules. The `bucket_plan` in `run_config.json` records
the actual buckets, samples per bucket, batches per epoch and a fingerprint of
the deterministic batch order. NR training shuffles samples and batches each
epoch by default. Center crops and paired spatial transforms remain deterministic
with either ordering policy.

For example, an `834 x 1257` pair fits a `1024 x 1024` area budget. With no-upscale,
it becomes `832 x 1248` by cropping at `left=1, top=4`, without the previous
`832/834` cover rescale. The retained RGB, controls and masks keep their original
values. Odd crop margins put the extra removed pixel on the right/bottom. The
budget check uses original dimensions: specifying a smaller `832 x 1248` budget
still takes the over-budget resize path. No new parameter is needed for this fix.
Dataset implementation hashes protect exact resume; pre-fix training states
cannot resume exactly under the changed transform.

Bucketing does not establish native-output equivalence: enhancing an image before
resizing generally differs from enhancing it afterward, and cropping changes the
available context. To preserve native paired pixel spacing, use no-upscale with
source dimensions within the area budget; alignment then removes only border
pixels. These transforms do not establish DLL forward parity.

### Reproducible Epoch Shuffling

Both Python training entry points enable epoch shuffling automatically; no
shuffle argument is needed. The existing `--seed` defaults to `42`.

Samples are shuffled within each dataset/bucket/frame-count group, followed by
the global batch order. Dataset-specific batch sizes and repeat counts are
preserved. Every row instance, including configured repeats, occurs exactly once
per epoch; short tails are neither dropped nor padded with duplicates. Frames
inside a temporal clip are never shuffled. Evaluation order is unchanged.

The permutation is derived from the shuffle-policy version, `--seed` and epoch
using a private CPU PyTorch generator. It does not consume the Python, NumPy,
Torch training or dropout RNG streams. Per-sample noise seeds still depend on
sample identity and epoch, not the sample's shuffled position. Plans can jump
directly to a resumed epoch without replaying earlier epochs; only one epoch's
permutation and prefix counts are cached.

DDP ranks reconstruct the same global plan and take their existing interleaved
microbatch positions. An accumulation window can cross an epoch boundary.
Because a short tail can occupy a different position in each epoch,
`consumed_samples` and resume checks use that epoch's actual prefix counts.
`run_config.json` records `bucket_plan.policy=dlssnr_epoch_shuffle_v1`,
`shuffle_seed`, `shuffle_rng` and `order_sha256_scope=epoch0`; the order hash
describes the first epoch, not a fixed ordering reused forever.

Shuffling is enabled by default. `--no-shuffle_dataset` optionally selects the
legacy fixed order. Changing the shuffle flag or seed during `--resume`
is rejected, as are the existing data/runtime/world-size/implementation identity
changes. To enable shuffling for an older run, start a new run from saved canonical
weights (merge an existing LoRA first) rather than reusing incompatible optimizer
state. Use the same PyTorch version for exact continuation.

### Parameter EMA

EMA is disabled by default in both training entry points. Supply `--ema_decay 0.99`
(or another finite value strictly between zero and one) to enable it. Omitting the
option or passing `--no-ema` allocates no EMA weights and publishes no EMA products.
`--ema_decay` and `--no-ema` are mutually exclusive.

The shadow starts from the initial trainable FP32 parameters and follows
`shadow = decay * shadow + (1 - decay) * parameter` after each successful optimizer
update. There is no decay warmup or bias correction. Accumulation microsteps and
FP16 overflow retries do not advance it. QAT averages the unrounded FP32 masters,
not native quantization codes. Frozen parameters and buffers, including frozen
FP8 bases and single-frame temporal heads, are not averaged. Each DDP rank tracks
the same parameter EMA after synchronized updates.

For LoRA, this averages the trainable adapter factors. In general,
`EMA(B) @ EMA(A)` is not `EMA(B @ A)`: this is **parameter EMA**, not an exact
average of merged full-model weights or predictions. Averaging can also move a
QAT-trained weight back across a native quantization boundary. Neither EMA nor
the example decay is a claim of better DLL output.

The raw optimizer weights stay in `final/` and each periodic checkpoint directory.
An additional `ema/` subdirectory contains a normal canonical model or LoRA
adapter with the same runtime and base-quantization metadata. Training metadata
records `weight_variant`, EMA scope, decay and successful-update count. The EMA
model uses the normal inference/DLL packing path; merge an EMA LoRA before DLL
packing, just as for a raw adapter. Resume uses the **parent** state directory,
never its `ema/` product.

When validation is configured, reports retain the raw `candidate` and add a
separate `ema_candidate`. Each evaluation starts its own histories; parameters,
training modes and training RNG are restored afterward. `--eval_native` evaluates
both candidates through the native-quantized proxy. With QAT or native evaluation,
raw and EMA products each receive their own `native_quantization.json`.

`--save_state` stores the FP32 shadow and update count inside the checksummed
`trainer_state.pt`. Missing/inconsistent EMA state, a changed decay, or enabling
or disabling EMA during exact resume is rejected. EMA adds one FP32 copy of
trainable parameters per device and another temporary copy during EMA evaluation
or export; it also adds evaluation work and output storage.

### Optional Frequency-Split Loss

The existing `pixel` objective remains the default. Both training entry points
accept `--loss_profile frequency_split` to replace it with low-frequency target
matching and input-gradient anchoring. The existing loss weights retain their
defaults, but select these terms in the new profile:

| Weight | Frequency-split objective | Metric |
| --- | --- | --- |
| `--loss_pre` | Charbonnier of the low-pass preclamp-minus-target residual | `loss/lowpass_pre` |
| `--loss_out` | Charbonnier of the low-pass rendered-minus-target residual | `loss/lowpass_out` |
| `--loss_edge` | Horizontal/vertical RGB gradient consistency with the **input**, not the target | `loss/input_edge` |
| `--loss_temporal` | Low-pass, motion-compensated change in output-minus-target residuals | `loss/lowpass_temporal` |

There is no additional full-band target pixel/edge term in this profile. The
gradient anchor is a weighted constraint, not an exact edge lock or a depth/normal
estimator. Excessive edge weight can restrict desired appearance changes.

`--loss_lowpass_sigma` defaults to `6` for this profile and accepts finite values
in `(0, 32]`. Sigma is measured in image pixels **after** bucket transforms, not
in source-image pixels. The bound limits filter support. The Gaussian is separable,
truncated at radius `ceil(3 * sigma)`, and uses replicated image boundaries. The
value is a starting setting, not a validated optimum. Supplying sigma with the
default pixel profile is rejected instead of silently ignoring it.

Spatial filtering uses `G(mask * residual) / G(mask)`, with zero-support outputs
set to zero, then weights the Charbonnier term by the original loss mask. Excluded
target values cannot bleed into adjacent supervision, and holes do not turn
constant residuals into artificial edges. Soft masks retain their weight in the
global valid-pixel denominator; input edges require both neighboring pixels.

For temporal loss, the previous residual is masked **before** bilinear warping
and normalized by its warped mask. This prevents invalid bilinear neighbors from
entering otherwise valid pixels. The common support combines current and warped
previous loss masks, temporal validity, the inside test and reset status. The
motion-compensated residual is low-pass filtered on that support. Burn-in remains
excluded. Loss and denominator calculation use the same support, including across
accumulation windows, variable bucket tails and DDP ranks.

The selected profile and sigma are saved in the resolved run configuration and
resume identity. Changing either requires a new run, not exact resume. QAT, EMA
and deployment-proxy evaluation remain independent options. Validation MAE/PSNR
are unchanged; training loss values from different profiles are not directly
comparable. This is a conservative objective for imperfect pairs, not evidence
that new photorealistic texture or noise sensitivity has been learned.

### Optional Control Randomization

`--control_randomization` teaches the new enhancement to vary with tone and
structure. Every training dataset must use `nr_controls_mode = "fixed"`, with
positive `nr_tone` and `nr_structure` after FP16 control encoding. These fixed
values define the reference point where the paired photo is the full enhancement
target. File-based control maps are rejected rather than silently replaced.

```text
--control_randomization --control_residual_sigma 6
```

One point is sampled per logical sample and epoch, constant over every clip frame
including burn-in. Tone and structure range from zero to their dataset reference
values; inference still uses actual native knob values. Style and auto-mask stay
fixed. Explicit skin stays fixed, while `nr_skin = -1` follows structure. Ratios
use effective FP16-encoded values, so controls that encode identically receive
the same endpoint target.

| Option | Enabled default | Constraint |
| --- | --- | --- |
| `--control_residual_sigma` | `6` | Finite, `0 < sigma <= 32`, in transformed pixels |
| `--control_anchor_probability` | `0.25` | Reference-point probability in `[0,1]` |
| `--control_corner_probability` | `0.25` | Uniform four-corner probability in `[0,1]` |

The probabilities must sum to at most one. The remainder samples independent
uniform tone/structure ratios. Because the four corners include the reference
point, its default total probability is 31.25%, before FP16 aliases. All three
numeric options require the enable flag. These defaults are experiment settings,
not measured optimal proportions.

For each loss role, let `B` be the initial frozen NR output, `A` its ordinary
target, and `G_M` a mask-normalized Gaussian filter. The generated target is:

```text
R = A - B(reference_controls)
target = B(sampled_controls) + structure_ratio * R
         + (tone_ratio - structure_ratio) * G_M(R)
```

Rendered RGB, DINO and temporal terms use the paired target and rendered teacher
field, clamped to `[0,1]`. The preclamp term uses the teacher's preclamp field and
is not RGB-clamped. The frequency-split edge term uses the input as `A`; its
generated guide is logged as `loss/control_edge`. Pixel-profile edges use the
generated rendered target. At zero, all roles reproduce their base fields; at
the reference point they reproduce their original targets exactly on supervised
pixels. Zero enhancement is the base response at zero controls, not an identity
image transform. Excluded labels never enter the residual Gaussian.

The residual sigma defines target construction independently of
`--loss_lowpass_sigma`. Existing loss weights and the selected profile still
apply; no full-band or perceptual term is enabled implicitly. For augmented
clips, both loss profiles use the normalized masked temporal residual with joint
current/previous label support. Only frequency-split additionally low-pass filters
that residual. Ordinary unaugmented pixel training is unchanged.

Distinct points normally add two no-gradient reference rollouts before the
student forward, each with independent history and matched frame-noise seeds.
Full training keeps one frozen initial model; LoRA bypasses adapters on its
frozen effective base. A wholly reference-point microbatch needs one rollout.
Optional base anchoring shares this provider and its sampled-point prediction;
provider allocation alone never enables the anchor loss. References stay outside
the optimizer, EMA and exports, and are initialized before restoring student state.

Sampling uses private, domain-separated RNG keyed by seed, epoch, final logical
sample/repeat ID and crop ID. Overflow retries reuse prepared points. Metadata
binds settings, encoded references, frozen weights, numerical policy and
implementation hashes to exact resume. Logs report effective tone/structure
means, reference/zero fractions and RGB/edge clipping fractions, weighted by
global valid RGB mass rather than rank-local means. Validation retains fixed
reference controls and ordinary photo targets. Synthetic supervision does not
prove real-weight control monotonicity or improved image quality.

### Optional DINOv3 Loss

Both NR trainers accept `--dino_loss_weight`; its default `0` leaves the objective
unchanged and does not import SenseCraft or load a feature model. This reuses the
SenseCraft ViT backend used by the HiDream-O1 implementation associated with
[PR #947](https://github.com/kohya-ss/musubi-tuner/pull/947). HiDream-O1 is unchanged.
Install the tested optional dependency in the training environment:

```shell
uv pip install ".[dinov3]"
```

This extra pins SenseCraft 0.3.11. It uses the project's Transformers version;
other SenseCraft feature APIs have not been validated. The existing repository
`uv.lock` predates the current project dependency pins; this installation command
resolves from `pyproject.toml`, rather than relying on that stale lockfile.
Enabling the loss loads pretrained weights through Hugging Face's normal cache
and authentication. Prepare access/cache on the training machine first. Loading
errors abort training; there is no random-feature fallback.

Add these arguments to an existing full/LoRA training command to combine it with
the frequency-split objective:

```text
--loss_profile frequency_split --loss_lowpass_sigma 6 --dino_loss_weight 0.1
```

The weight is an example for an ablation, not a tuned recommendation. DINO is an
auxiliary term; at least one base NR loss must remain enabled.

| Option | Enabled Default | Meaning |
| --- | --- | --- |
| `--dino_loss_model_type` | `small` | DINOv3 ViT `small`, `small_plus`, `base` or `large` |
| `--dino_loss_layer` | `-4` | Zero-based layer or Python-style negative index into the original model |
| `--dino_loss_resize` | `224` | Longest-side limit, multiple of 16 in `[16, 1024]` |
| `--dino_loss_use_gram` | enabled | L1 distance between channel Gram statistics of patch features |
| `--dino_loss_use_norm` | enabled | L2-normalize each patch feature before comparison |

Use `--no-dino_loss_use_gram` for position-wise patch-feature MSE only on aligned
pairs, or `--no-dino_loss_use_norm` to compare unnormalized features. Secondary
options with zero weight are rejected. The first NR integration supports ViT
patches only, excluding CLS/register tokens. It resolves the layer before
truncating the network, avoiding repeated negative-index resolution in SenseCraft
0.3.11. The feature model stays frozen, in FP32 eval mode with eager attention;
target extraction has no gradient, while prediction gradients reach the NR model.
NR's own attention/QAT settings remain independent.

Rendered RGB is clamped to `[0, 1]` and normalized by the backend. Images are
downscaled with antialiased bilinear sampling, preserving aspect ratio and never
upscaled, then padded to the patch grid. Masked values are removed **before**
resampling; RGB is divided by the resampled mask. Empty support and padding use
neutral RGB `0.5`. Patch weights are average-pooled mask coverage. Channel Gram
statistics divide by total patch coverage, excluding empty/padded tokens. The
per-image loss is weighted by the original valid RGB mass and normalized over the
whole accumulation window and all DDP ranks. Empty support gives zero loss and
gradient; the dataset loader still rejects wholly unsupervised manifest frames.
Temporal burn-in frames are excluded.

Gram comparison does not require corresponding patch positions. DINO features
still depend on position and global context, including neutral mask fills, so
this is not exact image-translation invariance or complete isolation of masked
regions. Resizing can remove fine texture. Keep the input structure anchor and
inspect detail/noise sensitivity separately; a smaller DINO loss alone does not
establish better texture or structure.

Logs add `loss/dino` and `loss/dino_weighted`. Normalized Gram values can be small;
do not choose weights by raw scale alone. Validation MAE/PSNR remain unchanged.
QAT, frozen FP8 LoRA bases and EMA are supported. The feature network adds training
memory/compute but is excluded from the optimizer, EMA and exported NR/LoRA
weights, with no new DLL dependency. Settings, frozen tensor checksum, provider
versions and feature implementation hashes are recorded in metadata and exact
resume identity. Resume rebuilds the teacher and rejects mismatches. Model loading
preserves training RNG; each DDP rank must load the same frozen teacher.

### Optional Base-Model Anchoring

`--base_anchor_weight` adds an output-retention loss to both full and LoRA
training. The default `0` disables this loss; a reference is only allocated if
control randomization independently needs it. For an ablation, add this to an
existing command:

```text
--base_anchor_weight 0.1
```

The example weight is not tuned. At least one ordinary NR loss must remain
enabled. The reference sees the same transformed source frames, control maps
and per-frame noise seeds as the student. Anchoring alone adds no randomized
controls or extra retention dataset. Temporal reference clips build their own history
from the first frame, using the same motion, resets and validity masks. They do
not reuse the student's changing history. Burn-in runs on both models but is
excluded from the loss.

The term is full-band Charbonnier distance between student and reference
`rendered_proxy` RGB, weighted by the supervised loss mask. Its denominator is
the valid RGB mass across all accumulated microbatches and DDP ranks. It stays
full-band with `frequency_split`: both predictions share the same source
geometry, unlike potentially misaligned source/target pairs. Logs add
`loss/base_anchor` and `loss/base_anchor_weighted`.

Full training keeps a frozen, eval-mode copy of the run's initial model. LoRA
reuses its frozen base with adapters temporarily disabled, avoiding another
base-weight copy. Both modes add a no-gradient reference forward for each
microbatch; full training also needs memory for the reference weights. The pass
preserves RNG and restores adapter/mode state before the student forward and
checkpoint replay. The teacher is excluded from the optimizer, EMA and exported
NR/LoRA tensors, and adds no inference or DLL dependency.

QAT uses the same publication policy for reference and student weights. For
frozen FP8 LoRA, the anchor is the **effective quantized base**, not an unquantized
or stock-DLL reference. The runtime policy, reference checksum and loss protocol
are recorded in metadata and exact-resume identity. Resume reconstructs the
original reference before restoring the trained student; it never makes the
checkpoint itself the new anchor. Changed weights, settings or implementation
hashes reject exact resume.

This term can reduce unwanted drift, but can also suppress intended changes.
It constrains only the sampled training conditions, so it does not establish
retention on unseen styles, scenes or control settings. Compare held-out raw,
EMA and native-proxy outputs, along with detail/noise diagnostics, before choosing
a weight. Image quality, GPU cost and actual DLL/game behavior still need testing
on the deployment machine.

### Optional Detail Diagnostics

`--eval_detail_diagnostics` adds high-frequency energy and paired-seed noise
sensitivity to validation. It is **off by default**, requires a held-out
`validation_manifest` or `sequence_manifest`, and needs no feature model or new
dependency. Add it to an existing full/LoRA training command, for example:

```text
--eval_detail_diagnostics --eval_native --sample_every_n_steps 100
```

`--eval_native` is independent: omit it to measure only the configured runtime.
With interval `0`, the final update is still evaluated. The reports in
`evaluation/stepNNNNNN.json` retain their existing metrics and add
`detail_diagnostics` to each case in `baseline`, `candidate` and, when enabled,
`ema_candidate`. Each case's `native` section gets the same diagnostics under
native-quantized weights. The baseline is this run's initial model, not
necessarily the stock DLL. `--no-compare_baseline` leaves only the candidate
comparisons. The top-level diagnostic protocol and each case's `protocol` record
the definitions and seed policy; the flag, protocol and implementation are bound
to exact-resume identity.

`high_frequency.sigma_1px` and `sigma_4px` use Gaussian high-pass residuals at
sigma 1 and 4 **transformed image pixels**, respectively. For image `x`, mask `m`
and Gaussian filter `G`, the residual is `x - G(m*x)/G(m)`. Filters are separable,
truncated at `ceil(3*sigma)`, and use replicated boundaries. Masked values are
removed before filtering, so holes do not create black-mask edges or import
excluded pixels. The measurements run in FP32 without autocast; TF32 follows the
external environment and backend settings. Sums use
FP64. No further resizing is performed for these diagnostics.

Each scale reports `input_energy`, `target_energy` and `output_energy`: the sum
of squared RGB residuals weighted by `loss_mask`, divided by
`valid_rgb_values = 3 * sum(loss_mask)`. Sequence frames are combined by valid
pixel mass, not by averaging frame scores. `output_to_input`, `target_to_input`
and `output_to_target` are ratios of those aggregate energies. Ratios with
denominator energy at or below `1e-12` are JSON `null`, not infinity or a claimed
zero change. For a non-null ratio `r`, `100*(r-1)` is the percentage energy change.
These two high-pass scales overlap; they are not disjoint FFT bands and should
not be added together or compared directly with differently defined measurements.

Noise sensitivity uses one deterministic seed pair per frame. The primary seed
is the existing evaluation seed; the alternate is `primary_seed XOR 0x9E3779B9`.
Only the current frame's noise lanes change. Input, controls, incoming history,
motion, validity and reset flags are identical within the pair. The alternate
history is discarded. Each model/runtime still maintains its own primary rollout.

`noise_sensitivity.rgb_mae` measures the masked mean absolute difference between
the two rendered outputs. `preclamp_mae` measures their neural outputs before
clipping and temporal blending, helping distinguish network response from
postprocessing attenuation. `high_frequency_rms` reports the RMS of the rendered
pair's high-pass difference at both scales. These are paired perturbation
measurements, not a Jacobian, an estimate from many noise trials, or a second
closed-loop temporal rollout.

Enabling diagnostics adds one forward per frame per evaluated runtime, plus
filtering. Baseline and EMA evaluations incur that cost too. Training loss and
weights are unchanged; primary histories, RNG, model modes and numerical flags
are preserved. Non-finite alternate outputs fail evaluation rather than producing
misleading JSON. Higher energy or sensitivity can reflect artifacts or noise,
so neither metric is a quality score or an automatic checkpoint-selection rule.
Compare against the same baseline/data/protocol, inspect images and temporal
behavior, and retain separate DLL/game acceptance.

### Optional Content Preservation Evaluation

`--eval_content_preservation` adds a frozen DINOv3 measurement of **output versus
input**, independently of the training target. It is off by default, adds no
training loss and does not enable `--dino_loss_weight`. Both NR trainers support
it when a held-out validation or sequence manifest is configured:

```text
--eval_content_preservation --eval_native --sample_every_n_steps 100
```

The existing optional `.[dinov3]` dependency and pretrained-weight cache/access
requirements apply. No DINOv2 or LPIPS dependency is introduced. Disabled runs
do not load a content feature model. A requested but unavailable backend aborts
with an error rather than omitting the measurement or substituting random weights.

The first protocol is fixed: DINOv3 ViT Small, layer `-4`, longest-side limit
224 without upscaling, L2-normalized **spatial patch MSE**. It excludes CLS/register
tokens and compares corresponding patch positions, not Gram statistics. This
can detect a rearrangement that preserves texture statistics. The training
`--dino_loss_*` options do not change this evaluation protocol. When the training
loss already uses the same model and feature layer, evaluation shares its frozen
backbone through a separate wrapper without changing its Gram/normalization settings.

Mask handling reuses the NR DINO preprocessing: remove excluded values before
antialiased aspect-preserving resize, normalize by resized mask coverage and pad
with neutral RGB. Patch contributions use average-pooled mask coverage. Frame
scores are channel-mean squared differences of normalized patch features,
averaged over valid patch coverage; sequence scores weight frames by their
original valid RGB mass. The feature network remains in FP32 eval mode without
gradients; TF32 follows the external environment and backend settings. Resizing
and global attention still limit detection of fine or localized changes; a mask
does not isolate the network's receptive field.

Each evaluated case adds `content_preservation` with `dinov3_patch_mse`,
`valid_rgb_values` and its protocol. Baseline, raw, EMA and native-proxy results
use the same feature model. Existing metrics and primary histories are unchanged;
alternate-noise diagnostic outputs are not scored by this metric. With sampling
interval zero, only the final update is evaluated. Baseline comparison remains
controlled by `--compare_baseline`.

Under DDP only the evaluating main rank creates the content wrapper; compatible
training feature weights can be reused there. Its protocol, frozen weights,
provider identity and implementation hashes are bound to metadata and exact
resume on all ranks. The evaluator does not enter the optimizer, EMA or exported
NR/LoRA tensors. Loading and evaluation preserve training RNG. Enabling it adds
feature-forward cost and, when no compatible training backbone exists, extra
frozen-model memory. Existing states from a different implementation cannot be
resumed exactly.

The control scanner accepts the same flag for source-only inputs. It adds
`dinov3_patch_mse_vs_input` to each primary-output JSON/CSV row and records the
feature protocol once in `scan_report.json`. It does not add NR forward passes,
but does extract input and output DINO features for each measurement.

This is a content-drift indicator, not a quality score or a new constraint on
training. Copying the input gives zero distance but performs no enhancement.
Compare it with target fidelity, detail/noise diagnostics and actual images.
Its values are not interchangeable with published DINOv2 or LPIPS scores, and
must not be compared across different feature identities or preprocessing rules.

## Current Scope

Root scripts remain thin entry points. Model code, numerics, data, training losses
and evaluation live in `src/musubi_tuner/dlssnr/`; LoRA lives in
`networks/lora_dlssnr.py`, and the training lifecycle in `training/dlssnr_trainer.py`.
Full and LoRA training share `NRTrainModule`, Accelerate wrapping and the same
update loop. They do not call the diffusion trainer.

- The default is single-process FP32, automatically using CUDA when available and CPU otherwise. `--device` accepts `auto`, `cpu` or `cuda`; explicitly requesting unavailable CUDA raises an error. Mixed precision requires CUDA. DDP supports multiple processes on one machine.
- Single-frame training and one finite temporal segment support full/LoRA gradient accumulation, periodic saves and resume. Accumulation is normalized by the total valid element count; steps count optimizer updates.
- Dataset configuration and CLI arguments are strictly validated. Manifest paths resolve relative to the TOML directory; CLI paths resolve relative to the working directory. Resolved configuration and source fingerprints are saved in `run_config.json`.
- Dataset `validation_manifest` and `sequence_manifest` entries are evaluated with independent student history. Evaluation compares the initial base by default and writes `evaluation/stepNNNNNN.json`, without changing training RNG or dropout state.
- Manifests are decoded on demand rather than keeping the entire training set resident. Inference and evaluation replay frames in order.

The native reference forward pass and compatibility/quality acceptance on real
clips remain incomplete. Layout byte round-trips, floating-point training tests
and native compatibility establish different things.

The first comparison on 2026-09-24 used Steam screenshots and a local GPU-adapted
310.8.0 DLL. It found that **the FP32 forward pass did not match**. Four screenshots
and 19 image/parameter combinations covered style, intensity, tone, structure,
skin and auto mask. Default-parameter MAE against the DLL ranged from 0.0294 to
0.0420. Identical packed weights do not prove forward correctness or compatibility
of trained artifacts with the native DLL. See the
[screenshot forward experiment](dlssnr_forward_experiment_2026-09-24.md).

Later that day, the main default-parameter factors were ruled out and the cubic
activation and key numerical boundaries were corrected. Default-parameter MAE
across the four images fell to 0.00262-0.00520. Nine of 19 combinations met the
original three display-metric thresholds, including one zero-intensity passthrough
case. **Full forward acceptance remained incomplete.** The full suite passed 172
tests, including real-weight CUDA full/LoRA updates and merged-output comparisons.
See the [trainable forward alignment report](dlssnr_forward_alignment_2026-09-24.md).
The numerical implementation version changed; older training states cannot resume
exactly across that boundary.

## Training Inputs and Validation Evidence

Normal full and LoRA training require `--model_dir`, canonical weights, a matching
schema/profile, complete source files and a conversion report with a passing
source round-trip. Actual data and tensors are still validated. **A forward
validation report is no longer a training prerequisite**, so development mode is
unnecessary for that reason. Missing model or provenance files stop the run;
there is no implicit random initialization or provenance bypass.

`--development_smoke` is reserved for explicit developer smoke tests. It allows
incomplete provenance, or random weights when `--model_dir` is omitted. It is not
a normal-training option and cannot bypass an invalid report supplied explicitly.

To bind validation evidence, pass `--forward_validation_report` explicitly. The
report must match `--model_dir`; missing paths, identity mismatches and failed checks
raise errors even in development mode. Normal training does not automatically read
`forward_validation_report.json` from the model directory, so an unrequested stale
report cannot change training or resume behavior. Tool callers can require evidence
with `inspect_canonical(..., require_forward_validation=True)`; only that call uses
the canonical directory's default report when no `validation_report` path is given.

The report contract remains `dlssnr_forward_validation_v1`, containing `profile`,
`numerics_profile`, the current `model_sha256`, `implementation_sha256`, a fixed
`reference_identity`, `float_validated` and `checks`. The checks are `raw_head`,
`neural_preclamp` and `rendered_proxy`, all backed by actual reference validation.
The implementation fingerprint is `json_sha256(implementation_identity())`.
Ordinary training evaluation cannot replace this report. Do not fill in passing
flags manually.

Run metadata records whether a valid base report was checked in
`source_forward_validated`. Without a report, it is `false` and
`experimental_surrogate=true` remains set, but `development_smoke` is not enabled
automatically. Developer smoke artifacts also retain the experimental flag.
Regardless of base evidence, newly trained artifacts set `float_validated`,
`temporal_validated` and `native_export_validated` to `false`: evidence for the base
does not prove compatibility of updated weights.

## Memory and Runtime Modes

### Export-Aligned Weight QAT

Native weight publication is disabled by default in full and LoRA training.
`--native_weight_qat` explicitly enables it; `--no-native_weight_qat` disables it.
FP32 master weights and optimizer state are retained. Each projection uses the
storage type declared by the canonical profile: direct FP32 to E4M3FN or FP16,
round-to-nearest/ties-to-even, without a learned scale or silent clipping. LoRA
publishes the **combined effective base plus delta**, not separately quantized
adapters. Existing scalar publications remain in place; temporal blend is also
published as FP16. Temperatures retain their FP32 storage and existing runtime
half publication. Activation rounding is unchanged: its half-then-E4 rule is not
the weight export rule.

QAT requires `--network_dropout 0` for LoRA. The existing nonzero-dropout path is a
separate projection branch, not a single exportable weight. Frozen FP8 base storage
can still be used with its existing experimental-mode restrictions; QAT runs after
that effective base is materialized and the adapter is added.

Full-model training batches finite/range checks before each optimizer attempt,
including every overflow retry. That validation remains active through backward
recomputation and ends before optimizer or EMA updates. Quantized weights are
still rebuilt on every publication; no quantized-weight cache crosses attempts.
LoRA keeps per-publication checks on its fused effective weights. The batched
path does not support untracked `.data` mutations during forward/backward.

To isolate deployment mismatch first, add these arguments to an existing training
command (the dataset must include a held-out `validation_manifest` or
`sequence_manifest` for `--eval_native`):

```text
--native_weight_qat --eval_native --attention_backend native --mixed_precision no --gradient_checkpointing --sample_every_n_steps 100
```

This does not select a learning rate or change the loss. Gradient accumulation,
warmup/cosine scheduling and `--max_grad_norm` remain independent options. Start
with a short run at the actual training resolution before committing to a long
run; QAT adds rounding and range-check work and is not an FP8 GEMM acceleration.

`--eval_native` is independent of QAT and can diagnose ordinary FP32 or experimental
training too. Each validation case retains its existing metrics and adds:

- `native`: target-error, saturation and temporal metrics under native-quantized
  effective weights, FP32 products and native attention.
- `native_gap`: masked RGB/preclamp MAE and RGB maximum absolute difference between
  the configured training runtime and that deployment proxy, using the same seed.

The two paths roll out **independent histories**; neither receives the other path's
previous output. Evaluation restores numerical policies, model/adapter modes and
training RNG, and never rewrites master weights or optimizer state. A zero gap
when training already uses QAT and FP32 native attention is expected; it is not
evidence of equivalence to DLL accumulation kernels.

With QAT or native evaluation enabled, every saved checkpoint and `final` directory
also contains `native_quantization.json`. It uses the DLL exporter's quantizer
and per-tensor statistics, with totals and `by_storage` summaries of code flips,
updates lost to rounding, maximum/mean error and RMSE. `flip_fraction` is a fraction
in `[0,1]`, not a percentage. The reference is the run's **initial effective weights
after native quantization**, including any frozen-base quantization recipe. This
snapshot stays compressed on CPU; it does not retain a second GPU model. For a run
starting from an off-grid fine-tuned checkpoint or scaled FP8 base, this reference
is not necessarily the original DLL. Use the DLL export report for changes relative
to that original template.

These probes assume `mix=1`, `strength=1` and `lora_multiplier=1`. Changing export
controls requires a new comparison. They remain surrogate measurements and record
`native_equivalent=false`; game/DLL output acceptance is separate. High flip rates
or lower target MAE alone do not establish better appearance or temporal stability.

Runtime metadata uses one format, `dlssnr_runtime_v3`, with explicit QAT and
checkpoint-threshold fields. Older runtime formats are rejected. The policy is
stored in canonical weight headers and LoRA files and restored on inference/merge.
`--native_weight_qat` or `--no-native_weight_qat` can
explicitly override publication in the image/video generators. Ordinary saved
canonical tensors default to native mixed precision. Use `--save_precision fp32`
to retain unrounded masters; `--save_state` always retains them for exact resume.
Resume requires the same QAT setting, checkpoint threshold, EMA decay, parameter-group multipliers and
implementation identity as the original run. Restore explicit settings from that
run rather than relying on current defaults. Changes to numerical implementation,
including the former forced-TF32-off scope, require a new run from saved weights
rather than reusing incompatible optimizer state.

E4M3 activation fake quantization was already enabled and remains active even with
weight QAT disabled. These training defaults do not enable `--eval_native` or
retroactively change the runtime policy stored in existing models and adapters.

On 2026-10-07, the real canonical weights completed a synthetic 832 x 1248,
batch-size-1 CUDA smoke on an RTX 4090 with PyTorch 2.13.0+cu130. Both runs used
FP32 products, native attention, QAT and gradient checkpointing, with one warm-up
update and **one measured update**:

| Mode | Peak Allocated GiB | Measured Update, Seconds |
| --- | ---: | ---: |
| Full, AdamW at 1e-5 | 11.49 | 5.63 |
| ViT LoRA rank 16, scaled FP8 base, AdamW at 1e-4 | 8.72 | 3.45 |

These are feasibility measurements, not throughput guarantees or quality results.
The benchmark helper also accepts `--native_weight_qat`; it never exports its
synthetic training updates. Different data, batch sizes, hardware or temporal
training require their own measurements.

### Compute and Storage

When GPU memory is limited, try `--gradient_checkpointing` first. It recomputes
FFN/attention during gradient-enabled training segments, covering boundary blocks,
the encoder, ViT and decoder. Burn-in and evaluation are not recomputed. Full and
LoRA training both support it; frozen inputs do not cut off LoRA gradients, and
dropout RNG state is restored during recomputation. It is disabled by default and
does not require a different numerical profile.

To trade more activation memory for less recomputation, add a feature-map area
threshold:

```text
--gradient_checkpointing --gradient_checkpointing_min_pixels 20000
```

Only blocks whose input feature map has `H * W > 20000` are checkpointed. Smaller
blocks retain their activations. This measures each block's padded feature map,
not the source image or batch size. The default threshold is `0`, which retains
all-block checkpointing when `--gradient_checkpointing` is enabled. A positive
threshold requires that flag. Higher thresholds can exhaust VRAM; measure a short
run at the intended resolution and batch size. Changing this setting requires a
new run rather than exact resume. Inference does not checkpoint blocks.

Dynamic Gaussian noise uses broadcast coordinates and bounded row chunks. It
preserves the original published FP32 values for the same size and seed and does
not consume the Python, NumPy or Torch RNG streams. Noise is still generated for
each requested seed; this does not introduce a dataset or pixel cache.

An isolated RTX 4090 check on 2026-10-17 used the full CC model, one 1024 x 1024
image pair, batch size 1, FP16, global FlashAttention, native-weight QAT,
AdamW_adv and EMA 0.99. Each timing run warmed up for 3 updates and measured 10;
data preparation and optimizer/EMA work were included, exports and evaluation
were not. No measured update retried an overflow.

| Runtime | Median Seconds/Update | Peak Reserved GiB |
| --- | ---: | ---: |
| Original noise and individual QAT checks, baseline / repeat | 1.865 / 1.894 | 12.72 |
| Tiled noise and batched QAT checks | 1.724 | 12.73 |
| Above plus checkpoint threshold 20000 | 1.473 | 17.73 |

The combined setting reduced update time by about 21-22% in this workload at a
cost of about 5 GiB reserved memory. These are not RTX 5090 speedup estimates.
In a separate deterministic-kernel check, all three updates matched the original
path bit for bit in outputs, gradients, parameters, optimizer, EMA, scheduler,
scaler and RNG state. Normal timing runs, including the repeated baseline, were
not bitwise reproducible; their speed measurements do not establish model quality.

For other options, select experimental mode explicitly:

```text
--numerics_profile train_experimental --mixed_precision bf16 --gradient_checkpointing
```

FP16/BF16 apply only to selected projections and matrix multiplications. Training
parameters, LoRA deltas, explicit half/E4 publications, reductions, loss and history
retain their FP32 boundaries. FP16 uses GradScaler, unscaling before checking or
clipping gradients. On overflow it reduces the scale and replays the same effective
batch and RNG state without advancing update/sample counts. `--max_overflow_retries`
defaults to 16. Nonfinite forward values or exhausted retries stop the run without
publishing a successful checkpoint. BF16 does not use a scaler. Accelerate's
mixed-precision environment settings must match the explicit CLI arguments.

FP8 currently compresses only the frozen projection base for LoRA:

```text
--numerics_profile train_experimental --gradient_checkpointing --fp8_base --fp8_scaled
```

`--fp8_scaled` uses the shared block-64 quantizer, with per-output-channel scaling
when the input width is not divisible by 64. Without it, weights use direct E4M3FN
storage and values outside `[-448,448]` are rejected. The input adapter, RGB/logit
heads, priors and scalars retain their original representation. Computation
materializes an FP32 base before adding the LoRA delta; **this does not execute
FP8 GEMM**. Checkpointing reduces the need to retain temporary materialized matrices
through backward. Artifacts bind the original base, effective quantized base and
scale identities. A quantized-base adapter cannot be added to an original base
that has not undergone the same quantization.

Attention options are mutually exclusive. The default remains the custom NR operator:

| Option | Scope and Restrictions |
| --- | --- |
| `--sdpa` | Experimental window/global attention, retaining window priors. PyTorch chooses its internal kernel; a Flash kernel is not guaranteed. |
| `--xformers` | Probes the required dtype, bias and backward capabilities. Unsupported combinations raise errors; scope can be limited explicitly with `--attention_scope global`. |
| `--flash_attn` | Requires FP16/BF16 and explicit `--attention_scope global`. Windows retain the NR operator and their learned priors. |
| `--sage_attn` | Currently supports only FP16/BF16 global inference. Training is rejected; no substitute backward is supplied. |
| `--attention_backend native` | Explicitly selects the NR operator and can override a saved experimental backend during inference. |

Experimental attention uses standard softmax with `scale=1.0` on preprocessed
Q/K/V, without another `1/sqrt(32)` factor. Zero tokens outside windows still
participate in normalization with priors; artificial alignment padding in global
layers is excluded as real keys. This differs from NR's half/E4 exponentials and
reductions and requires separate output-quality checks. Optional extensions load
on demand. Missing installations or ABI mismatches do not silently select another backend.

### Multi-GPU

For example, launch two processes on one machine as follows. Other training
arguments are the same as for a single GPU:

```bash
torchrun --standalone --nproc_per_node=2 dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml --model_dir models/canonical_dlssnr \
  --gradient_checkpointing --gradient_accumulation_steps 2 \
  --output_dir output/dlssnr_ddp --output_name full_ddp --save_state
```

A correctly configured `accelerate launch` is also supported. Linux CUDA uses
NCCL; Windows/CPU use Gloo initialization. Each GPU still holds a complete model.
DDP does not shard the model or make a larger model fit on an individual GPU.

Batch positions use `(update * accumulation + micro) * world_size + rank`.
Short final batches are not padded, and the deterministic batch plan continues
into the next epoch. By default, each epoch has its own seeded
permutation, shared by all ranks rather than independently shuffled on each rank.
The loss denominator is the total valid pixel count across
the accumulated global batch, rather than an average of per-rank means. The main
process handles shared weights, logs and evaluation; each rank's RNG/scaler is
saved. Resume requires the same world size, data plan and runtime policy. Automatic
resharding across world sizes, FSDP, ZeRO, DeepSpeed and multi-node execution are unsupported.

### Local Measurements

Measurements on 2026-10-01 used an RTX 4090, PyTorch `2.13.0+cu130`, the same real
canonical weights, one `512 x 512` effective frame (internal field `576 x 512`),
batch size 1, seed 42 and AdamW. LoRA used ViT rank 16. Each mode ran in its own
process with one warm-up update followed by two measured updates. Memory values
are PyTorch **peak allocated** GPU memory, excluding other processes. The small
timing sample is only a local comparison.

| Mode | Peak GiB | Median Update Time, Seconds |
| --- | ---: | ---: |
| Full FP32 | 15.92 | 1.96 |
| Full FP32 + checkpoint | 4.52 | 2.90 |
| Full BF16 + checkpoint | 4.46 | 3.17 |
| Full FP16 + checkpoint | 4.46 | 4.65 |
| Full BF16 + checkpoint + SDPA | 3.95 | 3.12 |
| LoRA FP32 | 8.17 | 1.39 |
| LoRA FP32 + checkpoint | 2.96 | 1.94 |
| LoRA BF16 + checkpoint | 2.91 | 2.14 |
| LoRA BF16 + checkpoint + scaled FP8 | 2.52 | 2.35 |

Checkpointing reduced memory by about 72% for full training and 64% for LoRA.
Both FP32 on/off comparisons had identical losses at the measured steps. FP16 had
one overflow retry within the measured interval, included in its timing.
Checkpointing provided the largest memory saving here; mixed precision added a
smaller saving and did not improve speed on this machine. Experimental-mode losses
differed. These synthetic-input tests do not establish learning quality or native parity.

To repeat the measurements, run each mode in a separate process with a distinct output file:

```bash
python tools/benchmark_dlssnr_runtime.py --model_dir models/canonical_dlssnr \
  --output output/benchmark-checkpoint.json --width 512 --height 512 \
  --gradient_checkpointing --warmup 1 --steps 2
```

Local checks covered real-weight CUDA AMP/SDPA updates and FP8 LoRA updates/merges.
Two-process CPU/Gloo tests covered unequal valid pixel counts, short final batches
and resume. This machine has one GPU, so real two-GPU execution was not verified.
The isolated test environment lacked FlashAttention, SageAttention and xFormers;
their actual CUDA kernels remain unverified. Installing an extension does not
establish capability or quality acceptance.

## DLL Unpacking and Repacking

The existing unpack/pack scripts support HT/CC canonical training weights and CC
lossless packed records. By default, unpacking selects every supported resource
present in the DLL:

```bash
python dlssnr_unpack_model.py --source_dll /path/to/original/nvngx_dlssnr.dll --output_dir models/dlssnr
```

For the audited 310.9.16 DLL containing both resources, this produces:

```text
models/dlssnr/
  models.json
  HT/
    model.safetensors
    model_config.json
    conversion_report.json
    ...
  CC/
    manifest.json
    records/
      block0.layer0.layer.bin
      ...
      block30.layer0.blend_scale.bin
```

`HT/` retains the canonical layout and training provenance files, with native
mixed-precision storage by default.
`CC/` contains 77 raw records across 31 blocks, totaling 113,887,490 payload bytes.
This raw CC output is **not a trainable canonical checkpoint**. Its serialized dtype and dimensions
describe the packed container, not homogeneous FP8/FP16 logical tensors. Do not
reinterpret or scale entire record files as one numeric dtype.

For CC training or inference, request canonical conversion explicitly:

```bash
python dlssnr_unpack_model.py --source_dll /path/to/original/nvngx_dlssnr.dll --weights CC --cc_format canonical --output_dir models/canonical_cc
python dlssnr_train.py --model_dir models/canonical_cc --dataset_config configs/dlssnr_dataset_single.toml --output_dir output/full --output_name cc
python dlssnr_train_network.py --model_dir models/canonical_cc --dataset_config configs/dlssnr_dataset_single.toml --output_dir output/lora --output_name cc
```

`--cc_format canonical` decodes the mixed E4M3, FP16 and FP32 regions into
`model.safetensors` and writes the same provenance files as HT. It also works
with `--weights all`, producing canonical HT and CC children. The legacy default
`--cc_format records` keeps the raw-record workflow unchanged.

Canonical conversion and ordinary full/EMA/merged model saves default to
`--save_precision native`: E4M3FN matrices, the original FP16 fields, and FP32
temperatures. Loading expands numeric tensors into FP32 model parameters; this
does not enable experimental `--fp8_base` execution. Legacy FP32 checkpoints remain
readable. `--save_precision fp32` keeps all numeric weights unrounded. Storage
policy is bound to both `model_config.json` and the safetensors header.

CC contains 113,342,367 logical parameters: two window blocks per multiscale
encoder/decoder stage, ViT blocks 11-18, the bridge at 19 and the output head at 30.
The channel widths and per-block packing layouts match the corresponding HT
types. Layout decoding/re-encoding is byte-verified against the audited resource;
the forward graph reuses the existing surrogate operators and per-resolution
window-phase progression. This is not native forward, native loader or image-quality
certification. CC saves, LoRA targets, temporal freezing and native-weight QAT use
the CC block numbering, while existing HT artifacts keep their original profile.

If the DLL contains only HT, unpacking keeps the original single-directory behavior:
`model.safetensors` is written directly under `--output_dir`. Explicit selection
also writes directly to the requested output directory:

```bash
python dlssnr_unpack_model.py --source_dll /path/to/original/nvngx_dlssnr.dll --weights HT --output_dir models/canonical_dlssnr
python dlssnr_unpack_model.py --source_dll /path/to/original/nvngx_dlssnr.dll --weights CC --output_dir models/cc_records
```

The supported resource identities are:

- HT: `WEIGHTS_HT`, SHA256 `836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4`. Identical in the audited 310.8.0 and 310.9.16 DLLs.
- CC: `WEIGHTS_CC_SMALL_FP8`, SHA256 `f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962`. Audited in 310.9.16.

Selection uses PE resource names and content hashes, not filenames or fixed DLL
offsets. `--input` / `--output` remain aliases. The source DLL is neither executed
nor modified, and the output directory must be new. A failed multi-model unpack
does not publish a partial output directory. Existing HT directories converted
by `dlssnr_convert_model.py` remain supported.

Repacking infers HT or CC from the selected directory. To select a child of a
multi-model output, pass `--weights`; omitting it at the bundle root is an error:

```bash
python dlssnr_pack_model.py --template_dll /path/to/original/nvngx_dlssnr.dll --model_dir models/dlssnr --weights CC --output_dll output/cc/nvngx_dlssnr.dll
python dlssnr_pack_model.py --template_dll /path/to/original/nvngx_dlssnr.dll --model_dir models/dlssnr/HT --output_dll output/ht/nvngx_dlssnr.dll
```

For raw CC record directories, edit only the files in `records/`, keeping each file's size unchanged.
Keep `manifest.json` unchanged: it records the original payload hashes, names,
sizes and serialized metadata. Repacking replaces only CC payload bytes and
preserves HT, resource headers, PE data and overlay bytes. The report lists changed
records and byte counts. An unedited extraction rebuilds the whole DLL identically.
Same-size edits are not validated for numerical meaning, image quality or native
execution. Raw CC record directories reject nondefault mix/strength settings and LoRA options;
`--record_dir` is an alias for `--model_dir`.

Canonical CC directories support the same `--mix`, `--strength`, and
`--merge_lora` options as HT. The packer selects the resource from the canonical
profile and preserves the other weight resource. The original audited DLL is
still required as the template.

After full training, or after merging LoRA separately, repack the complete model directory:

```bash
python dlssnr_pack_model.py --template_dll /path/to/original/nvngx_dlssnr.dll --model_dir output/full/run/final --output_dll output/export/nvngx_dlssnr.dll --mix 1.0 --strength 1.0
```

LoRA can also be merged during repacking, without manually creating an intermediate
model directory. In this case, `--model_dir` must identify that adapter's matching
base, not a different fine-tuned model:

```bash
python dlssnr_pack_model.py --template_dll /path/to/original/nvngx_dlssnr.dll --model_dir models/canonical_dlssnr --merge_lora output/lora/run/final/adapter.safetensors --lora_multiplier 0.8 --mix 1.0 --strength 1.0 --output_dll output/export-lora/nvngx_dlssnr.dll
```

`--merge_lora` takes an adapter file path; `--lora_weight` is an alias. Only this
project's DLSS-NR LoRA format is accepted, one adapter per invocation. Existing base
identity and FP8 quantized-base checks remain in place; arbitrary safetensors files
are not treated as compatible adapters. The temporary merge directory is removed
automatically, and neither the original base nor the adapter is modified.

For HT, the three controls are independent. They apply in this order: LoRA merge, mix,
strength, then native quantization.

| Option | Default | Meaning |
| --- | --- | --- |
| `--lora_multiplier` | `1.0` | Scales only the LoRA delta. A nondefault value requires `--merge_lora`. |
| `--mix` | `1.0` | Interpolates between the original DLL weights and the full trained/merged result, within `[0,1]`. |
| `--strength` | `1.0` | Multiplies all decoded numeric weights after mixing. Alias: `--multiplier`. |

```text
W_effective = W_trained                              # full checkpoint
W_effective = W_lora_base + lora_multiplier * delta   # adapter merge
W_export = quantize_native(strength * (W_original + mix * (W_effective - W_original)))
```

For FP8 LoRA, `W_lora_base` is the effective quantized base bound to the adapter.
`--mix 0 --strength 1` restores the template DLL's original weights.
`--strength 0` zeros decoded numeric weights; it does not restore the original
model. Strength scales matrices, priors, scales and other decoded floating-point
values, rather than controlling a visual effect in an inference UI. All three
controls must be finite. Strength and the LoRA multiplier may be negative or exceed
1, but final values must fit the finite range of their native storage types.
Opaque bytes and padding are not scaled.

Repacking rounds FP32 directly to the E4M3FN, FP16 or FP32 storage required by each
original resource region, using nearest/ties-to-even. It adds no scale and never
silently clips overflow. NaN/Inf, out-of-range values, missing/extra tensors,
shape/dtype mismatches, nonzero input lane 15, modified opaque data and source
mismatches are rejected. Quantization never overwrites the input training artifact.
Use FP32 model saves when experimenting with several `mix`/`strength` settings:
native saves have already rounded off-grid updates. Inline `--merge_lora` keeps
its temporary merge in FP32, so mix/strength precede the single export rounding.

Both the output DLL and `<output_dll>.report.json` must be new paths. Only same-sized
weight payloads are replaced; PE structure, resource metadata and all other content
are preserved. The tool verifies decoded quantized values per tensor, then extracts
the final DLL again to check its payloads. The report records input/output hashes,
all three controls, and per-tensor counts of trained changes, changes after
mix/strength, changes actually written, updates lost to rounding and quantization
errors. The CLI warns if every requested update rounds back to the original weights.

These checks do not certify native deployment. The tool does not repair the PE
checksum, re-sign the DLL or bypass load checks; a loader may reject a modified
DLL. Reports always retain `native_export_validated=false`. In-game loading and
image quality require separate acceptance checks. Keep the original DLL, and
continue HT training from canonical model directories. The selected template
resource must still match its audited original; previously modified *other*
resources are preserved. An already modified CC resource cannot serve as a new
CC template, and an already modified HT resource cannot serve as a new HT template.

The public unpack command deliberately rejects modified selected resources. Export
verification re-extracts their payloads with the record decoder instead of relaxing
the audited-template check. The opt-in real-DLL tests cover original unpacking,
QAT master export, decoded-weight bit equality and surrogate forward equality after
re-extraction. In PowerShell, using this repository's Python environment:

```powershell
$env:PYTHONPATH = "src"
$env:DLSSNR_DLL_PATH = "C:\path\to\original\nvngx_dlssnr.dll"
python -m pytest -q tests/test_dlssnr_dll_io.py
```

The combined HT/CC CLI roundtrip and isolated CC-payload edit test is also opt-in:

```powershell
$env:DLSSNR_CC_DLL_PATH = "C:\path\to\310.9.16\nvngx_dlssnr.dll"
python -m pytest -q tests/test_dlssnr_cc_dll.py
```

The tests write only temporary model directories and DLL copies, never the original.
They do not load the modified DLL into a game or certify native kernel parity.

The standalone merge script supports the same LoRA multiplier:

```bash
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/lora/run/final/adapter.safetensors --lora_multiplier 0.8 --output_dir output/merged
```

## Commands

Convert a user-supplied OpenDLSS-NR `models/nr` directory into canonical weights:

```bash
python dlssnr_convert_model.py --source_dir ../OpenDLSS-NR/models/nr --output_dir models/canonical_dlssnr --verify_roundtrip
```

Single-frame full training. Multiline commands below use Bash continuation;
in PowerShell, use one line or replace trailing backslashes with backticks:

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-5 \
  --optimizer_args weight_decay=0.0 \
  --output_dir output/dlssnr_full_single --output_name dlssnr_310_8_0_full \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

Temporal full training:

```bash
python dlssnr_train.py \
  --dataset_config configs/dlssnr_dataset_temporal.toml \
  --model_dir models/canonical_dlssnr \
  --training_mode temporal --sequence_length 4 --burn_in 2 --tbptt_length 2 \
  --gradient_checkpointing \
  --loss_temporal 0.10 --optimizer_type AdamW --learning_rate 1e-5 \
  --output_dir output/dlssnr_full_temporal --output_name dlssnr_310_8_0_temporal \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

ViT LoRA, selecting rank, alpha and optimizer through CLI arguments and using the
same single-frame dataset TOML:

```bash
python dlssnr_train_network.py \
  --dataset_config configs/dlssnr_dataset_single.toml \
  --model_dir models/canonical_dlssnr \
  --network_dim 16 --network_alpha 16 --network_dropout 0.0 \
  --gradient_checkpointing \
  --optimizer_type AdamW --learning_rate 1e-4 \
  --optimizer_args weight_decay=0.0 betas=0.9,0.999 \
  --output_dir output/dlssnr_lora --output_name dlssnr_310_8_0_lora_vit \
  --max_train_steps 1000 --save_every_n_steps 100 --save_state
```

Common arguments:

| Option | Purpose |
| --- | --- |
| `--optimizer_type` / `--optimizer_args` | Uses the shared optimizer factory, for example `AdamW`, `SGD`, `AdamW8bit` or a fully qualified class path. Install any extra dependencies separately. |
| `--learning_rate` | Defaults to `1e-5` for full training and `1e-4` for LoRA. AdamW weight decay defaults to `0.0`. |
| `--network_dim` / `--network_alpha` | ViT LoRA rank defaults to `16`; omitted alpha follows rank. |
| `--network_dropout` / `--network_args` | Dropout defaults to `0`. Use `--network_args profile=multiscale` for multiscale LoRA, optionally with `rank_by_width` / `alpha_by_width` dictionaries. Do not combine this with a single dim/alpha. |
| `--max_train_steps` / `--gradient_accumulation_steps` / `--seed` | Update count, gradient accumulation and random seed. |
| `--shuffle_dataset` / `--no-shuffle_dataset` | Seeded epoch shuffling of bucket samples and batch order is enabled by default; no argument is required. The negative flag optionally selects legacy fixed order. |
| `--max_grad_norm` | Clips gradients at the accumulated update boundary. Default `0` disables clipping. |
| `--ema_decay` / `--no-ema` | FP32 trainable-parameter EMA is disabled by default; supply a decay such as `0.99` to enable it. Raw and EMA products/evaluation remain separate. |
| `--save_precision native\|fp32` | Full/EMA/merged checkpoints default to native FP8/FP16/FP32 storage. Exact resume snapshots and LoRA deltas remain FP32. |
| `--native_weight_qat` / `--no-native_weight_qat` | Export-aligned weight QAT is disabled by default. These flags control weight QAT, not the existing E4M3 activation publications. |
| `--gradient_checkpointing` | Recomputes activations to reduce memory, preserving the default numerical profile. Disabled by default. |
| `--gradient_checkpointing_min_pixels` | Checkpoint only blocks with input feature-map `H * W` above this nonnegative threshold. Default `0` keeps all-block checkpointing; positive values require `--gradient_checkpointing`. |
| `--numerics_profile train_experimental` / `--mixed_precision` | Explicit experimental mode. CUDA precision accepts `no`, `fp16` or `bf16`; master weights remain FP32. |
| `--fp8_base` / `--fp8_scaled` | LoRA-only frozen-base storage quantization. Scaled requires base, and both require experimental mode. |
| `--sdpa` / `--xformers` / `--flash_attn` / `--attention_scope` | Explicit experimental attention and its scope; restrictions are listed above. |
| `--loss_pre` / `--loss_out` / `--loss_edge` / `--loss_temporal` | Loss weights, defaulting to `1`, `1`, `0.05` and `0`, respectively. |
| `--loss_profile` / `--loss_lowpass_sigma` | `pixel` is the unchanged default. Optional `frequency_split` uses low-frequency target matching and input edges; sigma defaults to `6` in transformed pixels. |
| `--control_randomization` | Opt-in tone/structure sampling with detached base-relative targets; requires positive fixed reference controls in every training dataset. |
| `--control_residual_sigma` / `--control_anchor_probability` / `--control_corner_probability` | Target-band sigma and sampling probabilities; enabled defaults are `6`, `0.25`, `0.25`. These options require control randomization. |
| `--prior_lr_multiplier` / `--scale_lr_multiplier` / `--temporal_blend_lr_multiplier` | Full-training parameter-group multipliers default to `1`, `1` and `0.1`, respectively. The LoRA entry point rejects these arguments. |
| `--sample_every_n_steps` / `--min_sequence_frames` | Validation interval and minimum temporal evaluation frame count. A nonzero interval requires validation manifests in the dataset TOML. |
| `--no-compare_baseline` | Disables the initial-base comparison, which is enabled by default. |
| `--save_state` / `--save_every_n_steps` / `--resume` | Training-state saves, save interval and resume checkpoint. Without `--save_state`, only weights are saved. |
| `--forward_validation_report` | Optional base-validation evidence, checked only when explicitly supplied. It is not a training prerequisite. |
| `--development_smoke` | Developer tests only: explicitly permits incomplete provenance or random initialization. Normal training does not need it. |

Multiscale rank limits follow the narrowest target in each width group: `32` for
widths 32/64/128/256, `64` for width 512, and `1024` for the ViT width 1024.
Oversized entries in `rank_by_width` are capped automatically, with alpha reduced
by the same factor to preserve `alpha / rank`. For example, requesting `128/64`
for every group produces `32/16`, `32/16`, `32/16`, `32/16`, `64/32`, `128/64`.
Omitted alpha still follows rank, and already-valid values are unchanged.
Adjustments are logged; resolved run configuration and adapter metadata store
the effective rank and alpha. This does not enable a single `--network_dim` for
multiscale or alter strict `vit_only` and saved-adapter loading checks.

For example, `--optimizer_type SGD --optimizer_args momentum=0.9 weight_decay=0.0`
constructs an actual SGD optimizer rather than changing only a configuration label.
Adafactor requires explicit `--optimizer_args relative_step=False warmup_init=False`.
Learning-rate schedules use the shared factory, supporting constant,
constant_with_warmup, linear, cosine, cosine_with_restarts, cosine_with_min_lr,
polynomial, inverse_sqrt and warmup_stable_decay. Warmup/decay, cycles, power,
timescale and min_lr_ratio follow the shared CLI. The scheduler advances once per
successful global optimizer update, not per accumulation microstep, DDP rank or
overflow retry. Scheduler state is saved and restored with training state;
changing its configuration prevents exact resume. Schedule-free weight switching
and optimizers requiring a closure remain unsupported.

To resume at an optimizer-update boundary, append the resume argument to the
**original complete training command**, retaining its effective parameters and
dataset configuration. For example:

```text
--resume output/dlssnr_full_single/dlssnr_310_8_0_full/state-step000100
```

The corresponding LoRA resume directory is
`output/dlssnr_lora/dlssnr_310_8_0_lora_vit/state-stepNNNNNN`.
Loading weights without optimizer state starts a new run rather than resuming one.

Still-image and closed-loop clip inference. Inference manifests may omit targets;
non-reset clip frames must provide motion and a history mask:

```bash
python dlssnr_generate_image.py --model_dir models/canonical_dlssnr --sample_manifest data/inference_single.jsonl --bucket_width 512 --bucket_height 512 --output_dir output/preview
python dlssnr_generate_video.py --model_dir models/canonical_dlssnr --sequence_manifest data/inference_sequence.jsonl --bucket_width 512 --bucket_height 512 --output_dir output/video
python dlssnr_merge_lora.py --base_model_dir models/canonical_dlssnr --adapter output/dlssnr_lora/dlssnr_310_8_0_lora_vit/final/adapter.safetensors --output_dir output/dlssnr_merged
```

Inference inherits the artifact's runtime policy unless overridden. Legacy models
without a policy use FP32/native defaults. Explicit overrides are recorded in
`inference_metadata.json` in the output directory, or the clip subdirectory for
videos. Returning to baseline requires compatible options together, for example
`--numerics_profile train_surrogate --mixed_precision no --attention_backend native`.
Disable FP8 with `--no-fp8_base` / `--no-fp8_scaled`. Sage global inference requires
explicit `--numerics_profile train_experimental --mixed_precision bf16 --sage_attn --attention_scope global`
and an available local extension.

### Control Response Scan

Use the standalone scanner to measure how tone and structure affect a frozen
model, without changing training losses or synthesizing new targets:

```shell
python tools/scan_dlssnr_controls.py --model_dir models/canonical_dlssnr --sample_manifest data/scan.jsonl --bucket_width 832 --bucket_height 1248 --output_dir output/control_scan --device cuda --eval_native
```

The JSONL manifest has exactly one reset frame per sample. Targets and control
files are not required; a minimal record is:

```json
{"schema":"dlssnr_pairs_v1","sample_id":"frame000","sequence_id":"scene_a","source_encoding":"srgb_proxy","frames":[{"frame_index":0,"input_path":"frame.png","reset":true}]}
```

Paths are relative to the manifest. Input dimensions must match `bucket_width`
and `bucket_height`; this tool does not crop or resize. RGB formats and optional
`loss_mask_path` follow the existing data contract. A mask must have nonempty
support. The mask affects measurements, not the network input. Existing
`controls_path` entries are explicitly overridden with fixed maps. Optional
target paths left in a reused manifest are still validated by the shared loader,
but targets are not scored. Multi-frame rows are rejected rather than silently
flattened into independent stills.

The default Cartesian grid is `--tone_values 0 0.5 1` and
`--structure_values 0 0.5 1`, producing nine control points per input. Custom
values must be in `[0,1]`. Points that become identical after FP16 control-lane
encoding are rejected. The JSON report records requested values and the actual
five encoded lanes for every point.

Other controls stay fixed: `--nr_style 0`, `--nr_skin -1` and auto mask enabled.
`--nr_skin -1` follows structure, so with auto mask enabled a structure sweep
also changes the skin lane. Set an explicit skin value to hold that lane fixed.
Use `--no-nr_auto_mask` to select the existing non-auto encoding. Auto mask is a
model condition, not a generated validity mask.

Each sample uses the same `--seed`-derived primary noise and XOR-paired alternate
noise at every control point and runtime. All forwards are independent stills
with no history. The default grid takes 18 forwards per image; `--eval_native`
doubles this to 36. The configured runtime inherits the model artifact's policy
unless explicitly overridden using the ordinary inference flags. The additional
native scan uses native-quantized weights, FP32 products and native attention.
It is a deployment proxy, **not actual DLL execution**. Scan canonical full or
EMA directories directly; merge LoRA into a canonical directory first.

The new output directory contains:

- `scan_report.json`: protocol, effective controls, runtime policies, source and
  parameter identities, data fingerprints, implementation hashes and per-image
  measurements. This file is written last and marks successful completion.
- `scan_metrics.csv`: one row per image, control point and runtime, with the
  corresponding PNG path. Formula-like text identifiers are escaped for
  spreadsheet readers; JSON preserves the original identifiers.
- `images/<sample_id>/input.png` and `configured/` / `native/` subdirectories:
  primary-seed previews. Metrics use FP32 proxy values before 8-bit PNG rounding.

`rgb_mae_vs_input` and `preclamp_mae_vs_input` measure change from the input,
not target accuracy. `lowpass_delta_rms` and `highpass_delta_rms` split
output-minus-input with the mask-normalized Gaussian at `--lowpass_sigma`
(default 6 pixels, positive and at most 32). These bands overlap; their RMS
values are not additive energy percentages. High-frequency energy and paired-seed
noise metrics at sigma 1 and 4 reuse the detail-diagnostics definitions. Ratios
with negligible input energy are JSON `null` / empty CSV cells. Measurements
are per image, with no implicit dataset average or quality ranking.

The scanner preserves model modes, runtime settings and RNG state. Existing
output directories are refused, and input changes during a scan prevent a final
report. Failed runs may leave partial PNGs; use a new directory for a retry.
Run it as a single process, not with a distributed launcher. Default scanning
needs no feature-model dependency or pretrained feature download. The optional
`--eval_content_preservation` flag adds the DINOv3 measurement described above,
using the existing `.[dinov3]` extra and recording its feature identity in the report.

Zero tone/structure is measured like any other point, not assumed to be a
pass-through. A stronger response can also be unwanted distortion. Repeat with
held-out scenes and multiple seeds on the training machine before deciding how
to construct randomized-control supervision; the tool does not calibrate that
mapping automatically.

## Unsupported Features

Full FP8 optimizer training, SageAttention training, CPU activation offload, block
swapping, FSDP/ZeRO/DeepSpeed and multi-node execution are unsupported. Unknown
CLI arguments still raise errors. Experimental modes target `float_runtime` only;
`native_roundtrip` cannot be selected as that runtime mode.

History reprojection uses bilinear sampling rather than the original five-tap
Catmull-Rom. History is stored as FP32 rather than truncated toward zero to float16.

Pixel caching and multi-worker prefetching are not implemented, and dataset TOML
does not accept placeholder cache fields. NR entry points do not set or restore
TF32 backend flags: TF32 follows the external environment and PyTorch backend
settings. Numerical metadata records `tf32=environment`, not a forced-off claim.
Keep those external settings consistent for comparisons and continuation; allowing
TF32 does not establish native/DLL numerical equivalence.

## Data Contract

Each JSONL record requires a unique, filename-safe `sample_id`, a `sequence_id`,
`schema = dlssnr_pairs_v1`, `source_encoding = srgb_proxy` and
`controls_encoding = dlssnr_lanes_10_14_v1`. Training and paired evaluation also
require a target and `target_encoding = srgb_proxy`. The first frame must reset;
`frame_index` must be nonnegative and strictly increasing. Training and validation
cannot share sequence IDs.

`sample_id` uniqueness is checked case-insensitively on every platform: `Frame`
and `frame` cannot occur in the same manifest. Names cannot contain path separators,
Windows-invalid characters, control characters, trailing periods/spaces or Windows
reserved names such as `NUL` and `CON`. Errors are raised during manifest indexing,
before partial inference outputs are written. Original IDs are never rewritten automatically.

- RGB accepts 8-bit images or numeric `[3,H,W]` NPY arrays, with proxy values in `[0,1]`. High-bit-depth images are not silently reduced to 8-bit.
- Controls are finite-valued `[5,H,W]` NPY arrays. Each record explicitly declares motion layout with `motion_layout = chw|hwc`; values are current-to-previous pixel displacements.
- Masks accept single-channel PNGs or `[1,H,W]` NPY arrays. History/temporal masks must be binary; optional `loss_mask_path` supports weights in `[0,1]`.
- With nonzero temporal loss, non-reset frames require a temporal mask and `tbptt_length >= 2`. Loss uses adjacent frames within the differentiable segment only, never across the burn-in boundary.
- `--loss_temporal 0` allows temporal masks to be omitted, but non-reset frames still require motion and history masks.

### Synthetic Temporal Clips

Set `synthetic_temporal = true` on a dataset entry to turn paired stills into
clips online, after the shared bucket resize or native crop. The entry accepts
directory pairs or a **single-frame** training manifest. Unmarked temporal
entries still require explicit real frames, motion and validity. Both kinds can
coexist; each retains its own batch-size, repeat and bucket grouping.

See [the synthetic dataset example](../configs/dlssnr_dataset_synthetic_temporal.toml).
Append explicit temporal settings to an existing training command, for example:

```text
--training_mode temporal --sequence_length 4 --burn_in 2 --tbptt_length 2
--loss_temporal 0.1
```

These are example settings, not changed defaults. Synthesis does not enable
temporal loss, control randomization, a reference model or any perceptual loss.
The usual clip-length and burn-in checks still apply.

`synthetic_max_shift_px` defaults to `0.5` and accepts finite values in `[0,1]`.
Both fields can be inherited from `[general]`. An explicitly disabled entry may
discard an inherited shift, but cannot declare a local shift; a general shift
with no enabled consumer is rejected. The source loss mask must retain positive
support inside the maximum-shift margin. Zero shift permits edge-only support
and provides a static-sequence control experiment.

Frame zero preserves the original pair. Later frames independently sample the
original transformed images at `p + d_t`, avoiding repeated-resampling blur.
Sampling is bilinear with `align_corners=False`; dense current-to-previous motion
is `d_t - d_(t-1)`. RGB and spatial controls use border extension as model context.
Fixed controls stay constant. Labels use `warp(target * mask) / warp(mask)`;
zero coverage has neutral fill, and label masks exclude geometric extension.
History validity also requires every contributing bilinear neighbor in the
previous frame to be supported. Joint current/previous label masks are applied
once in both temporal loss and its global denominator.

Trajectories use private CPU RNG keyed by seed, epoch, final logical sample/repeat
ID and crop ID, in a separate domain from controls and frame noise. No expanded
image or flow dataset is written. Batch reports count generated frames; resume
identity binds original files, seed, transform settings and implementation.

For a marked entry, `validation_manifest` must also contain paired stills and
uses fixed epoch-zero synthetic clips. `sequence_manifest` **always** remains a
real explicit-motion sequence. Original sequence IDs are retained, so generated
frames cannot bypass train/validation overlap checks. Both configured and native
case reports include `synthetic_temporal` with the generation protocol. Synthetic
`temporal_mae` measures the normalized masked residual over joint label support,
excluding neutral fill, invalid history and reset transitions; ordinary real-case
metric semantics are unchanged. Random control targets are not used in validation.

These clips test subpixel consistency for static scenes. They do not establish
behavior under moving objects, occlusion, long histories or actual DLL sampling,
and cannot repair misaligned source/target pairs. Keep synthetic validation,
held-out photo quality and real temporal evidence separate.

## Saving and Resuming

Each run uses its own `output_dir/output_name/` directory for `final/`, periodic
weights, training state, logs and evaluation reports. `--output_name` must follow
the portable-name rules above and cannot contain a path. A new run refuses an
existing directory with the same name. Continue with explicit `--resume`, or
choose another name/output directory to start over.

Existing artifacts are not moved automatically. Resolved configuration still uses
schema version 2. Training state now uses `dlssnr_train_state_v3`, including scaler,
per-rank state and runtime identity. Old v1/v2 states and implementation fingerprints
cannot resume exactly across versions. Older full weights can serve as a new run's
base; older LoRA can be merged into a canonical base first, but this does not restore
the original optimizer or adapter parameterization. Do not edit checkpoint identity
markers manually.

Full-training `final/` contains native mixed-precision weights by default, canonical configuration, the source
manifest, original opaque records, numerics/preprocessing information and training
metadata. Runtime policy is bound to both required model config and safetensors
headers. Missing or inconsistent bindings are rejected; reproducing the policy does
not depend on an optional training sidecar. LoRA v2 stores the adapter,
rank/alpha/targets, complete base identity and required runtime/quantization metadata.
Conforming v1 FP32 adapters remain readable. FP8 adapter merging reconstructs and
verifies the quantized base, then materializes base-plus-delta weights before saving.
The default native save rounds these effective weights; scaled FP8 bases and
off-grid training updates are not losslessly preserved. `storage_quantization` in
the training/conversion report records rounded values, maximum absolute error and
stored bytes. Choose `--save_precision fp32` for unrounded merged weights and exact
adapter/merge comparisons. Loading does not apply frozen-base quantization again.
Merging preserves canonical provenance and opaque data; LoRA delta files stay FP32.

LoRA forward computation uses `W + alpha/r * B@A` on the weight side, following
the same projection computation as merged weights. Real-weight tests found that
FP32 error from two separate GEMMs followed by addition is amplified in ViT; small
linear-layer tests alone cannot establish whole-network merge equivalence. Dropout
is added as a rank-space correction and disabled for evaluation. This choice favors
consistency at the cost of temporary matrices and backward work. Artifacts record
`canonical_weight_plus_delta_v1`.

`--save_state` creates `state-stepNNNNNN/` at periodic and final update boundaries.
Full-model weights and EMA products in those state directories stay FP32 regardless
of `--save_precision`; optimizer state alone cannot restore rounded master weights.
State includes the optimizer, actual global consumed-sample count, each rank's
CPU/current-CUDA-device/Python/NumPy RNG, FP16 scaler, resolved configuration and
data/batch-plan/base/implementation fingerprints, plus parameter EMA state when
enabled. The checksum manifest is published
only after all state is complete. Successful update count, gradient accumulation
and world size locate the global resume batch, including short final batches and
epoch transitions. Without this option, `stepNNNNNN/` weights are still saved, but
those directories cannot provide exact resume.

Changing only dataset TOML comments or CLI argument order does not prevent resume.
Changing learning rate, optimizer, rank, bucketing, precision, attention, FP8 policy,
world size, referenced data, base or training implementation rejects exact resume.
Explicit report paths and contents are also part of resume identity; unrequested
reports in the model directory are not. Nondeterministic CUDA backward operators
still require tolerance-based acceptance. Bitwise equivalence across all hardware
is not promised.

Legacy adapters missing complete base identity, rank/alpha or forward-mode metadata
are also rejected. Keep the old files and regenerate validated artifacts.

## Validation History and Regression Tests

On 2026-09-23, the local full suite passed 159 tests with no skips, and Ruff passed.
Tests covered original 310.8.0 weights on an RTX 4090: single FP32 full/LoRA updates,
gradients for every ViT LoRA target, and raw-head comparisons after merging. The CUDA
smoke tests used synthetic paired `48 x 48` inputs; they do not establish production-
resolution performance, real-data training quality or native parity.

If the project package is not installed in a development environment, set
`PYTHONPATH` at the repository root before testing. Normal use should still run in
the project's dependency environment. Real-weight CUDA tests require local source
weights, defaulting to the sibling `../OpenDLSS-NR/models/nr` directory, or a path
set through `DLSSNR_SOURCE_DIR`. These external-weight tests skip when weights are
unavailable; a skip is not real-weight acceptance.

```powershell
$env:PYTHONPATH = "src"
$env:OMP_NUM_THREADS = "4"
$env:MKL_NUM_THREADS = "4"
$env:DLSSNR_RUN_CUDA_SMOKE = "1"
python -m pytest -q --tb=short
```

DLL import/export integration tests are separately opt-in. Set `DLSSNR_DLL_PATH`
to a user-supplied original DLL containing the audited resource:

```powershell
$env:DLSSNR_DLL_PATH = "D:/path/to/original/nvngx_dlssnr.dll"
python -m pytest tests/test_dlssnr_dll_io.py -q
```

These tests cover original-byte reconstruction, changed-weight re-extraction,
resetting to the original weights with `--mix 0`, and LoRA merging with its own
multiplier. They write only temporary artifacts and never execute the DLL.
