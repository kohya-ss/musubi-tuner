# Blissful-Tuner Weighted Mask Loss Training Guide

A comprehensive reference for understanding and using weighted mask loss training in blissful-tuner for WAN2.2 and Qwen Image LoRA training.

**Last Updated:** 2026-01-29

---

## Table of Contents

1. [Overview](#overview)
2. [How Weighted Masks Work](#how-weighted-masks-work)
3. [⚠️ Critical: Understanding What Masked Training Actually Does](#️-critical-understanding-what-masked-training-actually-does)
4. [Weighted Mask Pipeline](#weighted-mask-pipeline)
5. [Training Arguments](#training-arguments)
6. [Dataset Configuration](#dataset-configuration)
7. [Mask Value Transformations](#mask-value-transformations)
8. [Recommended Settings](#recommended-settings)
9. [Technical Implementation Details](#technical-implementation-details)
10. [Troubleshooting](#troubleshooting)
11. [Changelog](#changelog)

---

## Overview

Weighted mask loss training allows you to control **how much the model learns from different regions** of your training images. This is particularly useful for person/character LoRA training where you want:

- **Strong learning on faces** (highest priority)
- **Moderate learning on body/clothing**
- **Reduced learning on hair** (often styled differently)
- **Minimal/no learning on backgrounds** (to avoid baking in specific environments)

### Supported Architectures

| Architecture | Dataset Type | Mask Support | Notes |
|-------------|--------------|--------------|-------|
| WAN 2.1/2.2 | ImageDataset | ✅ Full | One mask per image |
| WAN 2.1/2.2 | VideoDataset | ✅ Full | One mask per video (by basename) |
| WAN 2.1/2.2 | `--one_frame` | ✅ Full | Mask on target frame only |
| Qwen Image | ImageDataset | ✅ Full | One mask per image |
| HunyuanVideo | Any | ⚠️ **Not Yet** | Cache format doesn't store masks |

> **Note:** HunyuanVideo latent caching (`cache_latents.py`) does not currently save mask weights. This is a known limitation.

#### VideoDataset Mask Matching

For video training, masks are matched by **basename** (filename without extension):

```
videos/
├── clip001.mp4
├── clip002.mp4

weighted_masks/
├── clip001.png  # Matches clip001.mp4
├── clip002.png  # Matches clip002.mp4
```

One static mask image is used for all frames of each video. Per-frame mask videos are not supported.

#### One-Frame Mode (`--one_frame`)

When using `--one_frame` for image-to-video training:
- **Control frames** (extracted from source image) receive weight = **1.0** (full learning)
- **Target frame** (the frame being predicted) receives the **mask weights**

This ensures the model learns the control frames fully while applying selective attention to the generated content.

---

## How Weighted Masks Work

### The Core Concept

Instead of treating all pixels equally during training, weighted masks assign different **loss weights** to different regions:

```
Loss = MSE(predicted, target) * mask_weight
```

Where `mask_weight` ranges from 0.0 (ignore this region) to 1.0 (full training weight).

### Our Weighted Mask Values

We use grayscale masks with specific values for different body parts:

| Region | Grayscale Value | Normalized Weight | Purpose |
|--------|-----------------|-------------------|---------|
| Face | 255 | 1.000 | Maximum learning - facial features are critical |
| Body | 128 | 0.502 | Moderate learning - clothing/body proportions |
| Hair | 80 | 0.314 | Reduced learning - hair varies by style |
| Background | 0 | 0.000 | No learning - avoid environment overfitting |

---

## ⚠️ Critical: Understanding What Masked Training Actually Does

> **This section explains a common and important misconception about masked training. Please read carefully before using this feature.**

### The Misconception vs Reality

| What Users Often Think | What Actually Happens |
|------------------------|----------------------|
| "Mask = crop; only train on the masked region" | Mask = loss weight; model still predicts the entire image |
| "Background weight 0 = model ignores the background" | Background weight 0 = model can predict **anything** there without penalty |
| "Masking isolates my subject from context" | Masking **de-emphasizes** context, but doesn't remove it |

### Why This Matters: The "Phantom Limb" Problem

When you set `mask_weight = 0` for a region, you're telling the model:

> *"I don't care what you predict in this region—don't penalize wrong predictions here."*

This is **very different** from saying "this region doesn't exist." The model still makes predictions for the entire latent tensor. With zero loss weight outside the mask:

- The model has **no incentive** to predict correctly outside the mask
- If it learned from other images that people/objects might be there, it may hallucinate them
- You may see "phantom limbs," random body parts, or duplicated subjects bleeding outside the mask
- The model might settle on generating cluttered backgrounds because there's no penalty for doing so

**Real-world example:** Training a person LoRA with group photos where only your subject is masked. Even though bystanders have `weight = 0`, the model learned "people exist in images" from your dataset. With no loss penalty outside the mask, it may generate extra people, floating limbs, or duplicate faces in the unmasked regions.

### Why `--mask_min_weight` Is Critical

The `--mask_min_weight` parameter exists specifically to address this problem:

```bash
--mask_min_weight 0.05  # Background gets 5% training signal
```

With a small non-zero weight:
- The model still **prioritizes** the masked subject (95% vs 5% effective weight)
- But it receives **some gradient signal** for background regions
- This prevents unconstrained hallucination outside the mask
- Edge artifacts and phantom limbs are significantly reduced

**This is why the recommended settings use `--mask_min_weight 0.05`, not `0.0` (unless you enable** [Masked Prior Preservation](#masked-prior-preservation-optional)**).**

### When to Use Cropping Instead

If you truly want to train **only** on a subject with no context whatsoever, **cropping is the only solution**:

| Approach | Effect |
|----------|--------|
| Masking with `weight=0` | Model can hallucinate freely outside mask |
| Masking with `min_weight=0.05` | Model is lightly constrained outside mask (recommended) |
| Cropping to subject | Context is **physically removed**—model never sees it |

**Use cropping when:**
- You want zero influence from backgrounds/other people
- Your subject fills most of the frame anyway
- You're willing to lose spatial context (pose relative to environment)

**Use masking when:**
- You want to **emphasize** certain regions while **de-emphasizing** others
- You need to preserve spatial relationships and context
- You're training on group shots but want to focus on one person

### Practical Recommendations

1. **Avoid `--mask_min_weight 0.0`** unless you have a specific reason, or you are using **Masked Prior Preservation** (`--prior_preservation_weight > 0`).

2. **For character LoRA training**, use:
   ```bash
   --mask_min_weight 0.05  # Prevents phantom limbs
   --mask_gamma 0.7        # Balanced emphasis
   ```

3. **If you see hallucinations outside the mask**, increase `--mask_min_weight` to 0.1 or higher.

4. **If you want to keep `--mask_min_weight 0.0` but still avoid hallucinations**, enable **Masked Prior Preservation** and consider `--prior_mask_threshold 0.1`.

5. **For group photos**, consider combining masking with cropping:
   - Crop to include only your subject + reasonable padding
   - Then apply masks within the cropped region for face/body emphasis

6. **Alpha channel masks don't provide "true transparency"**—they're still converted to loss weights. The same trade-offs apply.

---

## Weighted Mask Pipeline

### Step 1: Create Weighted Masks

Weighted masks are created using semantic segmentation (Meta Sapiens model) that identifies:
- Face regions
- Hair regions
- Body/clothing regions
- Background

Each region is assigned a specific grayscale value.

#### Group Photos / Multiple People in One Image

If an image contains multiple people, **semantic segmentation alone** (e.g., Sapiens) will usually label *everyone* as “person parts”.
That’s fine for general segmentation, but for **LoRA likeness training** you typically want the loss to focus on **only your subject**.

The recommended workflow is:

1. Generate normal weighted masks for the full image (Sapiens, etc.)
2. Generate a **binary instance mask** for *only the target person* (FaceID + instance segmentation)
3. Multiply the weighted mask by the instance mask so **all non-subject people become weight 0**

This avoids using Photoshop / generative fill and prevents “bystander” faces from leaking into training.

Tools included in this repo:

```bash
# 1) Create binary instance masks for the subject in group photos
python tools/create_instance_masks.py \
  --input /path/to/images \
  --output /path/to/instance_masks \
  --reference /path/to/reference_face.jpg \
  --backend yolo

# 2) Apply instance masks to your existing weighted masks
python tools/apply_instance_mask_to_weighted_mask.py \
  --weighted-masks /path/to/weighted_masks \
  --instance-masks /path/to/instance_masks \
  --output /path/to/weighted_masks_subject_only
```

Notes:
- Set `require_mask = true` in your dataset config to fail fast if any mask is missing.
- Use `--backend sam3` in `create_instance_masks.py` only if you have a working SAM 3 environment + checkpoints (see SAM 3 README).

### Step 2: Dataset Configuration

Add `mask_directory` to your dataset TOML config:

```toml
[[datasets]]
image_directory = "/path/to/images/subject"
mask_directory = "/path/to/images/weighted_masks"  # Weighted masks here!
cache_directory = "/path/to/cache"
caption_extension = ".txt"
resolution = [1024, 1024]
```

### Step 3: Cache Latents with Masks

When you run the cache latents script, masks are:

1. **Loaded** from `mask_directory` (matched by filename)
2. **Normalized** from 0-255 to 0.0-1.0
3. **Downsampled** to latent space dimensions (8x smaller)
4. **Saved** alongside latents in the cache file

```bash
# For WAN 2.2
python wan_cache_latents.py --dataset_config config.toml --vae /path/to/vae

# For Qwen Image
python qwen_image_cache_latents.py --dataset_config config.toml --vae /path/to/vae
```

### Step 4: Train with Mask Loss

Enable mask-weighted loss during training:

```bash
python wan_train_network.py \
  --dataset_config config.toml \
  --use_mask_loss \
  --mask_gamma 0.7 \
  --mask_min_weight 0.05 \
  # ... other training args
```

Optional: enable **Masked Prior Preservation** to reduce phantom limbs / background drift (slower, but often higher quality):

```bash
python wan_train_network.py \
  --dataset_config config.toml \
  --use_mask_loss \
  --mask_gamma 0.7 \
  --mask_min_weight 0.0 \
  --prior_preservation_weight 1.0 \
  --prior_mask_threshold 0.1 \
  --normalize_per_sample \
  # ... other training args
```

---

## Training Arguments

### `--use_mask_loss`

**Type:** Flag (boolean)
**Default:** Disabled

Enables mask-weighted loss training. **Required** to activate mask functionality.

```bash
--use_mask_loss
```

### `--mask_gamma`

**Type:** Float
**Default:** 1.0
**Range:** > 0

Controls the **contrast** of the mask weights through gamma correction:

```python
mask_weight = mask_weight ** mask_gamma
```

| Value | Effect | Use Case |
|-------|--------|----------|
| < 1.0 | Softer mask, more midtones | More training on hair/body |
| = 1.0 | Linear (no change) | Default behavior |
| > 1.0 | Sharper mask, more binary | Stronger focus on face only |

**Examples:**
- `--mask_gamma 0.5`: Softer - Face=1.0, Body=0.71, Hair=0.56, Background=0.0
- `--mask_gamma 1.0`: Linear - Face=1.0, Body=0.50, Hair=0.31, Background=0.0
- `--mask_gamma 2.0`: Sharper - Face=1.0, Body=0.25, Hair=0.10, Background=0.0

### `--mask_min_weight`

**Type:** Float
**Default:** 0.0
**Range:** 0.0 - 1.0

Sets a **minimum floor** for all mask weights, ensuring even masked-out regions receive some training signal:

```python
mask_weight = mask_weight * (1.0 - mask_min_weight) + mask_min_weight
```

| Value | Effect |
|-------|--------|
| 0.0 | Background completely ignored |
| 0.05-0.1 | Background gets 5-10% training weight |
| 0.2 | Background gets 20% training weight |

**Example with `--mask_min_weight 0.1`:**
- Face: 1.0 → 0.9 + 0.1 = 1.0
- Body: 0.50 → 0.45 + 0.1 = 0.55
- Hair: 0.31 → 0.28 + 0.1 = 0.38
- Background: 0.0 → 0.0 + 0.1 = 0.1

---

## Masked Prior Preservation (Optional)

Masked Prior Preservation adds a second objective to masked training:

- **Inside mask:** train toward the normal target (current behavior)
- **Outside mask:** train toward the **base model** prediction (teacher), which reduces phantom limbs / background drift

This requires an additional forward pass with LoRA temporarily disabled (under `torch.no_grad()`), so it is slower than normal masked training.

> **Note:** Masked Prior Preservation requires **LoRA training** (adapters must be temporarily disabled to compute the teacher prediction).  
> It is not available for **full fine-tuning** runs without adapters (e.g. `qwen_image_train.py` full fine-tune mode).

> **Important:** When using prior preservation, `--mask_min_weight 0.0` is typically recommended.  
> Non-zero `--mask_min_weight` reduces the prior preservation effect.

### `--prior_preservation_weight`

**Type:** Float  
**Default:** 0.0 (disabled)  
**Range:** >= 0

Enables prior preservation when set to a value > 0:

```bash
--prior_preservation_weight 1.0
```

**Recommended starting points:** `0.5` or `1.0`

### `--prior_mask_threshold`

**Type:** Float (optional)  
**Default:** None (continuous mode)  
**Range:** (0, 1)

Controls how the prior mask is constructed:

- **Continuous mode (default):** prior preservation scales smoothly with inverse mask weights.
- **Threshold mode:** prior preservation applies only where the **raw** mask is below a threshold.

```bash
# Preserve only true background (recommended starting point for weighted masks)
--prior_mask_threshold 0.1
```

**Notes:**
- Thresholding is applied to the **latent-resolution** mask (after caching-time resizing/interpolation), not the original pixel mask.
- In threshold mode, target/prior overlap is prevented by zeroing the target mask wherever the prior mask applies.

**Recommended values:** `0.05–0.15` (start at `0.1`)

### `--normalize_per_sample`

**Type:** Flag (boolean)  
**Default:** Disabled

Changes masked loss reduction from a **global weighted mean** (default) to **per-sample weighted mean then batch average**.

This is recommended when using prior preservation so `--prior_preservation_weight` behaves predictably even when mask coverage varies across samples.

```bash
--normalize_per_sample
```

### Argument Validation

All mask-related arguments are validated early in training. Invalid values will raise clear errors (some settings may also emit warnings):

| Argument | Valid Range | Error if Invalid |
|----------|-------------|------------------|
| `--mask_gamma` | > 0 | `--mask_gamma must be > 0` |
| `--mask_min_weight` | [0, 1) | `--mask_min_weight must be in range [0, 1)` |
| `--prior_preservation_weight` | >= 0 | `--prior_preservation_weight must be >= 0` |
| `--prior_mask_threshold` | (0, 1) | `--prior_mask_threshold must be in range (0, 1)` |

---

## Dataset Configuration

### Method 1: Separate Mask Files

The traditional approach using a separate `mask_directory`:

```toml
# Dataset configuration for weighted mask training
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
# Image source
image_directory = "/path/to/images"

# Weighted masks directory - MUST match image filenames!
mask_directory = "/path/to/weighted_masks"

# Cache directory - masks are baked into cache
cache_directory = "/path/to/cache"

# Standard settings
caption_extension = ".txt"
resolution = [1024, 1024]
enable_bucket = true
min_bucket_reso = 512
max_bucket_reso = 1536
bucket_reso_steps = 64
batch_size = 1
```

**File Naming Convention:**

Mask files **must match** the image filename (extension can differ):

```
subject/
├── kyla001.png
├── kyla002.png
└── kyla003.png

weighted_masks/
├── kyla001.png  # Matches kyla001.png
├── kyla002.png  # Matches kyla002.png
└── kyla003.png  # Matches kyla003.png
```

### Method 2: Alpha Channel Masks (Recommended)

Embed masks directly in RGBA PNG images to **eliminate filename mismatch bugs**:

```toml
[[datasets]]
image_directory = "/path/to/rgba_images"  # RGBA PNGs with embedded masks
cache_directory = "/path/to/cache_alpha"  # Always use fresh cache!
alpha_mask = true
mask_directory = "/path/to/masks"         # Optional fallback for non-RGBA images
require_mask = false                      # Set true to error if any image lacks mask
resolution = [1024, 1024]
caption_extension = ".txt"
```

**Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `alpha_mask` | bool | false | Extract mask from image alpha channel |
| `require_mask` | bool | false | Error during caching if any item has no mask |
| `mask_directory` | string | none | Optional fallback for non-RGBA images |

**Fallback Chain:**
1. `alpha_mask=true` + RGBA image → use alpha channel
2. Else if `mask_directory` has matching file → use mask file
3. Else → full-weight mask (or error if `require_mask=true`)

### Converting Existing Datasets to RGBA

Use the conversion utility to merge existing image+mask pairs into RGBA PNGs:

```bash
python convert_masks_to_alpha.py \
    --image_directory /path/to/images \
    --mask_directory /path/to/masks \
    --output_directory /path/to/rgba_output \
    --caption_directory /path/to/captions  # Optional: copy captions
```

**Options:**
- `--dry_run`: Preview without writing files
- `--skip_existing`: Skip files already in output directory
- `--caption_extension`: Default `.txt`

**Important Notes:**
- The conversion preserves exact RGB values (never composites against black)
- Aspect ratio mismatches > 1% will error (prevents distortion)
- Always use a fresh `cache_directory` after conversion!

---

## Mask Value Transformations

### Complete Transformation Pipeline

```
Raw Mask Value (0-255)
        ↓
    Normalize (÷ 255.0)
        ↓
    0.0 - 1.0
        ↓
    Gamma Correction (** gamma)
        ↓
    Min Weight Floor (* (1-min) + min)
        ↓
    Final Loss Weight
```

### Transformation Examples

**Input: Face=255, Body=128, Hair=80, Background=0**

| Setting | Face | Body | Hair | Background |
|---------|------|------|------|------------|
| Default (gamma=1.0, min=0.0) | 1.000 | 0.502 | 0.314 | 0.000 |
| Softer (gamma=0.5) | 1.000 | 0.709 | 0.560 | 0.000 |
| Sharper (gamma=2.0) | 1.000 | 0.252 | 0.099 | 0.000 |
| With floor (min=0.1) | 1.000 | 0.552 | 0.383 | 0.100 |
| Recommended (gamma=0.7, min=0.05) | 1.000 | 0.622 | 0.473 | 0.050 |

---

## Recommended Settings

> **Important:** If you are **not** using prior preservation, use `--mask_min_weight > 0` to prevent the ["phantom limb" problem](#why-this-matters-the-phantom-limb-problem).  
> If you **are** using prior preservation (`--prior_preservation_weight > 0`), `--mask_min_weight 0.0` is typically recommended.

### For Person/Character LoRA Training with Prior Preservation (Recommended)

```bash
--use_mask_loss \
--prior_preservation_weight 1.0 \
--prior_mask_threshold 0.1 \
--normalize_per_sample \
--mask_gamma 0.7 \
--mask_min_weight 0.0
```

**Rationale:**
- `prior_preservation_weight=1.0`: Strongly preserves base model behavior outside the mask
- `prior_mask_threshold=0.1`: Preserves true background while letting body/hair learn normally for weighted masks
- `normalize_per_sample`: Makes `--prior_preservation_weight` predictable across mixed mask coverage
- `mask_min_weight=0.0`: Maximizes the "preserve outside mask" effect
  - If you set `mask_min_weight > 0`, the prior mask is reduced (by design)

### For Person/Character LoRA Training (Recommended)

```bash
--use_mask_loss \
--mask_gamma 0.7 \
--mask_min_weight 0.05
```

**Rationale:**
- `gamma=0.7`: Slightly softer masks to ensure good body/clothing learning
- `min_weight=0.05`: **Critical**—prevents phantom limbs and hallucinations outside the mask while still heavily prioritizing the subject

### For Face-Focused Training

```bash
--use_mask_loss \
--mask_gamma 1.5 \
--mask_min_weight 0.02
```

**Rationale:**
- `gamma=1.5`: Sharper masks to heavily prioritize face learning
- `min_weight=0.02`: Minimal but non-zero—prevents unconstrained hallucination in background

> ⚠️ **Warning:** Using `--mask_min_weight 0.0` is **not recommended** unless you are also using prior preservation (`--prior_preservation_weight > 0`).  
> Without prior preservation, a zero floor allows the model to hallucinate freely outside the mask, potentially causing phantom limbs, duplicate subjects, or cluttered backgrounds.

### For Style/Clothing Training

```bash
--use_mask_loss \
--mask_gamma 0.5 \
--mask_min_weight 0.1
```

**Rationale:**
- `gamma=0.5`: Very soft masks to include body/clothing equally
- `min_weight=0.1`: Higher floor since style/clothing benefits from environmental context

---

## Technical Implementation Details

### Loss Calculation

The mask-weighted loss reducer lives in `apply_masked_loss_with_prior()` in `src/musubi_tuner/modules/mask_loss.py` and is called from the training loops.

Example (video/non-layered loss layout):

```python
from musubi_tuner.modules.mask_loss import apply_masked_loss_with_prior

loss = apply_masked_loss_with_prior(loss, batch.get("mask_weights"), prior_loss_unreduced=None, args=args, layout="video")
```

Example (Qwen Image layered loss layout):

```python
loss = apply_masked_loss_with_prior(
    loss,
    batch.get("mask_weights"),
    prior_loss_unreduced=None,
    args=args,
    layout="layered",
    drop_base_frame=args.remove_first_image_from_target,
)
```

High-level steps performed:
1. Normalize mask shape to `(B, 1, F, H, W)` (accepts `(B, F, H, W)` too).
2. If `layout="layered"`, optionally drop base-frame weights, then permute to match `(B, L, C, H, W)`.
3. Apply gamma correction: `mask = mask ** mask_gamma`.
4. Apply min-weight floor: `mask = mask * (1 - mask_min_weight) + mask_min_weight`.
5. Compute region-normalized target loss: `sum(loss * mask) / sum(mask)`.
6. If `--prior_preservation_weight > 0` and `prior_loss_unreduced` is provided:
   - Build `prior_mask = 1 - mask` (or threshold on the raw mask)
   - Compute region-normalized prior loss in the prior region
   - Combine: `L_target + w_prior * L_prior`

### Key Implementation Notes

1. **Weighted Mean Normalization**: The loss is normalized by `sum(weights)` not `count`, preventing bias toward larger masked areas.

2. **Latent Space Downsampling**: Masks are downsampled to latent dimensions using **area interpolation** (`F.interpolate(..., mode="area")`), which properly averages pixel values.

3. **Batch Alignment**: If a batch contains mixed items (some with masks, some without), items without masks are assigned `torch.ones_like()` (full weight everywhere).

4. **Float32 Precision**: Mask weights are saved in float32 in cache files for numerical precision.

5. **Mask Resizing Interpolation**: When resizing masks to bucket resolution:
   - **Downscaling** uses `INTER_AREA` (preserves average weight correctly)
   - **Upscaling** uses `NEAREST` (preserves discrete weight tiers, avoids ringing/halos)

   This ensures weighted mask values (255/128/80/0) are not corrupted by interpolation artifacts.

### Verifying Mask Loss is Active

When mask loss is enabled, you'll see a prominent banner in the training logs:

```
============================================================
MASKED PRIOR PRESERVATION TRAINING ENABLED
============================================================
  mask_min_weight: 0.0
  mask_gamma: 0.7
  prior_preservation_weight: 1.0
  prior_mask_threshold: 0.1 (threshold mode)
  normalize_per_sample: True
------------------------------------------------------------
PRIOR PRESERVATION: Unmasked regions will match base model.
                    Expect ~1.3-1.7x training time.
------------------------------------------------------------
IMPORTANT: Masks must be baked into latent cache!
If you see 'batch has no mask_weights' errors, re-run
the cache script with alpha_mask and/or mask_directory enabled.
============================================================
```

If prior preservation is disabled, the banner will instead say `MASK-WEIGHTED LOSS TRAINING ENABLED`.

If you don't see this banner, `--use_mask_loss` is not enabled.

### File Locations

| File | Purpose |
|------|---------|
| `src/musubi_tuner/modules/mask_loss.py` | Shared mask loss reducer + validation helpers |
| `src/musubi_tuner/hv_train_network.py` | Base NetworkTrainer (uses shared mask loss reducer) |
| `src/musubi_tuner/wan_train_network.py` | WAN trainer (inherits NetworkTrainer; uses shared mask loss reducer) |
| `src/musubi_tuner/qwen_image_train.py` | Qwen Image trainer (uses shared reducer; handles layered layout) |
| `src/musubi_tuner/wan_cache_latents.py` | WAN latent caching with mask processing |
| `src/musubi_tuner/qwen_image_cache_latents.py` | Qwen Image latent caching with mask processing |
| `src/musubi_tuner/dataset/image_video_dataset.py` | Dataset handling and mask storage |
| `src/musubi_tuner/dataset/config_utils.py` | Dataset config with `mask_directory` parameter |

---

## Troubleshooting

### Error: "FATAL: --use_mask_loss is enabled but batch has no mask_weights!"

**Cause:** Masks were not baked into the latent cache.

**Solution:**
1. Add `alpha_mask = true` and/or `mask_directory` to your dataset TOML config
2. Use a **fresh** `cache_directory` (or delete existing cache)
3. Re-run the cache latents script

### Warning: "X images have no matching mask file"

**Cause:** Some images in `image_directory` don't have corresponding masks in `mask_directory`.

**Solution:**
- Ensure mask filenames match image filenames exactly (minus extension)
- Generate missing masks
- Or use `alpha_mask = true` with RGBA images instead

### Warning: "X items had no mask (no alpha + no file). Filled with full-weight (255)."

**Cause:** Masking is enabled but some items have neither alpha channel nor matching mask file.

**Solution:**
- Convert images to RGBA using `convert_masks_to_alpha.py`
- Or ensure all images have matching files in `mask_directory`
- Or set `require_mask = true` to catch these during caching

### Error: "require_mask=true but no mask found for ..."

**Cause:** `require_mask = true` is set and an item has no mask source.

**Solution:**
- Ensure the image has an alpha channel, or
- Ensure a matching mask file exists in `mask_directory`

### Masks not being applied

**Cause:** `--use_mask_loss` flag not set.

**Solution:** Add `--use_mask_loss` to your training command.

### Very low loss values

**Cause:** Large portions of images are masked out (weight=0).

**Solution:**
- Increase `--mask_min_weight` to 0.05-0.1
- Or use `--mask_gamma` < 1.0 for softer masks

---

## Quick Reference

### Minimum Required Setup (Method 1: Separate Files)

```toml
# In dataset config TOML
mask_directory = "/path/to/weighted_masks"
```

```bash
# Cache with masks
python wan_cache_latents.py --dataset_config config.toml --vae /path/to/vae

# Train with masks
python wan_train_network.py --dataset_config config.toml --use_mask_loss
```

### Minimum Required Setup (Method 2: Alpha Channel)

```toml
# In dataset config TOML
alpha_mask = true
cache_directory = "/path/to/fresh_cache"  # Use fresh cache!
```

```bash
# Convert existing images + masks to RGBA
python convert_masks_to_alpha.py --image_directory ... --mask_directory ... --output_directory ...

# Cache with alpha masks
python wan_cache_latents.py --dataset_config config.toml --vae /path/to/vae

# Train with masks
python wan_train_network.py --dataset_config config.toml --use_mask_loss
```

### All Dataset Config Options

| Option | Type | Default | Applies To | Description |
|--------|------|---------|------------|-------------|
| `mask_directory` | string | none | Image & Video | Directory with grayscale mask PNGs |
| `alpha_mask` | bool | false | **Image only** | Extract mask from RGBA alpha channel |
| `require_mask` | bool | false | **Image only** | Error if any item lacks mask |

> **Note:** `alpha_mask` and `require_mask` are only available for image datasets (`[[datasets]]` with `image_directory` / `image_jsonl_file`). VideoDataset (`video_directory` / `video_jsonl_file`) config validation will **reject** these options—use `mask_directory` only for video datasets.

### All Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--use_mask_loss` | flag | disabled | Enable mask-weighted loss |
| `--mask_gamma` | float | 1.0 | Gamma correction (< 1 softer, > 1 sharper) |
| `--mask_min_weight` | float | 0.0 | Minimum weight for all regions |
| `--prior_preservation_weight` | float | 0.0 | Prior preservation weight (0.0 disables) |
| `--prior_mask_threshold` | float | none | Optional threshold mode for prior mask (e.g. 0.05–0.15) |
| `--normalize_per_sample` | flag | disabled | Normalize masked loss per-sample (recommended with prior) |

### Our Weighted Mask Values

| Region | Value | Normalized | Purpose |
|--------|-------|------------|---------|
| Face | 255 | 1.000 | Maximum priority |
| Body | 128 | 0.502 | Moderate priority |
| Hair | 80 | 0.314 | Low priority |
| Background | 0 | 0.000 | Ignored |

---

## Changelog

### 2026-01-29
- **NEW:** Masked Prior Preservation (`--prior_preservation_weight`, `--prior_mask_threshold`, `--normalize_per_sample`)
- **CHANGED:** Training loop uses `apply_masked_loss_with_prior()` (handles both normal masked loss and optional prior preservation)
- Updated banner to reflect prior preservation when enabled

### 2026-01-21
- **NEW:** Added "Critical: Understanding What Masked Training Actually Does" section
  - Explains the common misconception (masks as crops vs loss weights)
  - Documents the "phantom limb" problem when using `mask_min_weight=0.0`
  - Provides guidance on when to use cropping vs masking
- **CHANGED:** Updated recommended settings to use `--mask_min_weight > 0` when prior preservation is disabled
- Refactor: mask loss moved to `src/musubi_tuner/modules/mask_loss.py`
- Removed `--mask_loss_scale` (it had no effect with weighted-mean normalization)

### 2026-01-17
- **NEW:** Alpha channel mask support (`alpha_mask = true`)
- **NEW:** `require_mask` option for strict mask enforcement at caching time
- **NEW:** `convert_masks_to_alpha.py` utility for converting existing datasets
- **NEW:** Fallback chain: alpha → mask_directory → full-weight
- **NEW:** Thread-safe mask source counting with summary logging
- **NEW:** O(1) mask path matching (performance improvement for large datasets)
- Updated error messages to include alpha_mask option

### 2026-01-14
- Added mask resizing fix: NEAREST interpolation for upscaling (preserves discrete values)
- Added argument validation with clear error messages
- Added VideoDataset mask matching documentation
- Added one-frame mode mask behavior documentation
- Added "Verifying Mask Loss is Active" section with logging banner

### 2026-01-13
- Initial document creation
- VideoDataset and one-frame mask support enabled
- Core mask loss implementation documented

---

*Document created: 2026-01-13*
*Last updated: 2026-01-29*
*Based on blissful-tuner codebase analysis*
