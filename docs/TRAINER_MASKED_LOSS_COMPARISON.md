# Comprehensive Masked Loss Training Comparison Report

**Analysis Date:** 2025-01-13
**Codebases Analyzed:**
- ai-toolkit (https://github.com/ostris/ai-toolkit)
- SimpleTuner (https://github.com/bghira/SimpleTuner)
- sd-scripts/kohya (https://github.com/kohya-ss/sd-scripts)
- blissful-tuner (this repository) - Reference

---

## Executive Summary

All four training frameworks support masked loss training, but with significantly different implementations and feature sets:

| Trainer | Masked Loss | Unique Strength |
|---------|-------------|-----------------|
| **ai-toolkit** | ✅ Full | Inverted mask regularization, Gaussian blur smoothing |
| **SimpleTuner** | ✅ Full | Probabilistic application, TREAD integration, auto-generation scripts |
| **sd-scripts** | ✅ Full | Alpha channel support, mature documentation |
| **blissful-tuner** | ✅ Full | Gamma correction, min weight floor, multi-architecture |

**Key Finding:** Each trainer has unique features worth "stealing" for blissful-tuner enhancement.

---

## Feature Comparison Matrix

### Core Masked Loss Features

| Feature | ai-toolkit | SimpleTuner | sd-scripts | blissful-tuner |
|---------|------------|-------------|------------|----------------|
| Mask file directory | ✅ `mask_path` | ✅ `conditioning_data` | ✅ `conditioning_data_dir` | ✅ `mask_directory` |
| Alpha channel masks | ✅ `alpha_mask` | ❌ | ✅ `alpha_mask` | ❌ |
| Grayscale weighting | ✅ 0-255 → 0-1 | ✅ (mask type) | ✅ 0-255 → 0-1 | ✅ 0-255 → 0-1 |
| Binary segmentation | ❌ | ✅ (segmentation type) | ❌ | ❌ |
| Gamma correction | ❌ | ❌ | ❌ | ✅ `--mask_gamma` |
| Min weight floor | ✅ `mask_min_value` | ❌ | ❌ | ✅ `--mask_min_weight` |
| Mask inversion | ✅ `invert_mask` | ❌ | ❌ | ❌ |
| Probabilistic application | ❌ | ✅ `masked_loss_probability` | ❌ | ❌ |
| Video support | ✅ 5D tensors | ❌ | ❌ | ✅ (WAN) |

### Advanced Features

| Feature | ai-toolkit | SimpleTuner | sd-scripts | blissful-tuner |
|---------|------------|-------------|------------|----------------|
| Inverted mask prior | ✅ Regularizes unmasked | ❌ | ❌ | ❌ |
| Attention masking | ✅ Flux only | ✅ Flux only | ❌ | ❌ |
| TREAD integration | ❌ | ✅ Force-keep tokens | ❌ | ❌ |
| Mask blur smoothing | ✅ Gaussian blur | ❌ | ❌ | ❌ |
| Mask normalization | ✅ Mean=1.0 | ❌ | ❌ | ✅ Weighted mean |
| Auto mask generation | ✅ Random blobs | ✅ Florence-2 + SAM2 | ❌ | ❌ |
| Latent caching | ✅ Lazy load | ✅ Yes | ✅ With alpha | ✅ Yes |

### Interpolation Methods

| Trainer | Mask Resize Method | Notes |
|---------|-------------------|-------|
| ai-toolkit | `bicubic` | Smooth gradients |
| SimpleTuner | `area` | Proper downsampling |
| sd-scripts | `area` | Proper downsampling |
| blissful-tuner | `LANCZOS` → `area` | Image resize uses LANCZOS (BUG-003) |

---

## Detailed Analysis by Trainer

### 1. ai-toolkit Analysis

**Repository:** `/Users/dustin/ai-toolkit`
**Status:** ✅ FULL MASKED LOSS SUPPORT

#### Configuration Options
```yaml
datasets:
  - folder_path: "/path/to/images"
    mask_path: "/path/to/masks"        # Mask directory
    mask_min_value: 0.0                 # Floor value (0-1)
    invert_mask: false                  # Invert mask values
    alpha_mask: false                   # Use alpha channel
    loss_multiplier: 1.0                # Per-dataset weight
```

#### Key Implementation Files
| File | Purpose | Key Lines |
|------|---------|-----------|
| `toolkit/config_modules.py` | Config parsing | 1137-1151 |
| `toolkit/dataloader_mixins.py` | Mask loading | 1296-1402 |
| `extensions_built_in/sd_trainer/SDTrainer.py` | Loss application | 480-860, 1309-1329 |
| `toolkit/util/mask.py` | Random mask generation | Full file |

#### Unique Features to Steal

**1. Inverted Mask Prior (Regularization)**
```python
# Regularize unmasked regions using base model predictions
if self.train_config.inverted_mask_prior and prior_pred is not None:
    prior_mask_multiplier = 1.0 - mask  # Inverted mask
    prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
    prior_loss = mse_loss(pred, prior_pred, reduction="none")
    prior_loss = prior_loss * prior_mask_multiplier * inverted_mask_prior_multiplier
    total_loss = loss + prior_loss
```
**Benefit:** Prevents model from "forgetting" how to generate unmasked regions.

**2. Gaussian Blur Smoothing**
```python
# Apply random Gaussian blur to mask edges
blur_amount = random.uniform(0, 0.005) * min(mask.shape)  # 0-0.5% of image size
if blur_amount > 0:
    mask = gaussian_blur(mask, kernel_size=int(blur_amount * 2 + 1))
```
**Benefit:** Prevents sharp mask boundaries from creating training artifacts.

**3. Mask Normalization to Mean=1.0**
```python
# Normalize so average mask value is 1.0
mask_multiplier = mask_multiplier / mask_multiplier.mean()
```
**Benefit:** Maintains consistent loss magnitude regardless of mask coverage.

**4. Random Mask Generation Utility**
```python
masks = generate_random_mask(
    batch_size=20, height=256, width=256,
    min_coverage=0.2, max_coverage=0.8,
    num_blobs_range=(1, 3)
)
```
**Benefit:** Useful for data augmentation and testing.

---

### 2. SimpleTuner Analysis

**Repository:** `/Users/dustin/SimpleTuner`
**Status:** ✅ FULL MASKED LOSS SUPPORT

#### Configuration Options
```json
{
    "id": "training-data",
    "type": "local",
    "dataset_type": "image",
    "conditioning_data": "mask-data",
    "instance_data_dir": "/path/to/images"
},
{
    "id": "mask-data",
    "type": "local",
    "dataset_type": "conditioning",
    "conditioning_type": "mask",  // or "segmentation"
    "conditioning_data_paired_from": "training-data",
    "conditioning_instance_data_dir": "/path/to/masks"
}
```

**CLI Option:**
```bash
--masked_loss_probability 1.0  # 0.0-1.0, probability of applying mask per batch
```

#### Key Implementation Files
| File | Purpose | Key Lines |
|------|---------|-----------|
| `simpletuner/helpers/models/common.py` | Loss application | 4270-4287 |
| `simpletuner/helpers/models/flux/model.py` | Attention masking | 639-668 |
| `simpletuner/helpers/training/collate.py` | Data loading | 676-797 |
| `scripts/datasets/masked_loss/` | Auto-generation | Full directory |

#### Unique Features to Steal

**1. Probabilistic Mask Application**
```python
# Configuration field
masked_loss_probability = 1.0  # Default: always apply

# In loss calculation
if conditioning_type == "segmentation":
    if random.random() < self.config.masked_loss_probability:
        loss = loss * mask_image
```
**Benefit:** Useful for datasets with sparse masks or for regularization.

**2. Dual Conditioning Types**
```python
if conditioning_type == "mask":
    # Continuous weighting: use first channel directly
    mask_image = batch["conditioning_pixel_values"][:, 0].unsqueeze(1)
    mask_image = mask_image / 2 + 0.5  # Normalize to [0, 1]
    loss = loss * mask_image

elif conditioning_type == "segmentation":
    # Binary weighting: average RGB, binarize
    mask_image = torch.sum(batch["conditioning_pixel_values"], dim=1, keepdim=True) / 3
    mask_image = mask_image / 2 + 0.5
    mask_image = (mask_image > 0.5).to(dtype=loss.dtype)  # Binarize
    loss = loss * mask_image
```
**Benefit:** Flexibility between soft (continuous) and hard (binary) masking.

**3. TREAD Integration (Token Routing)**
```python
# Prevent token dropping in masked regions
if conditioning_type in ("mask", "segmentation"):
    mask_lat = F.interpolate(mask_img, size=(h_tokens, w_tokens), mode="area")
    force_keep = mask_lat.flatten(2).squeeze(1) > 0.5  # (B, S_img)
    flux_transformer_kwargs["force_keep_mask"] = force_keep
```
**Benefit:** Ensures important regions are not dropped during token routing optimization.

**4. Automated Mask Generation Scripts**
```bash
# Using Florence-2 + SAM2
python scripts/datasets/masked_loss/generate_dataset_masks.py \
    --input_dir /images/input \
    --output_dir /images/output \
    --text_input "person"
```
**Benefit:** Automated pipeline for mask generation from text prompts.

---

### 3. sd-scripts (kohya) Analysis

**Repository:** `/Users/dustin/sd-scripts`
**Status:** ✅ FULL MASKED LOSS SUPPORT

#### Configuration Options

**TOML Config:**
```toml
[[datasets.subsets]]
image_dir = "/path/to/images"
caption_extension = ".txt"

# Method A: Separate mask files
conditioning_data_dir = "/path/to/masks"
num_repeats = 8

# Method B: Alpha channel
alpha_mask = true
num_repeats = 8
```

**CLI:**
```bash
--masked_loss
--conditioning_data_dir /path/to/masks
```

#### Key Implementation Files
| File | Purpose | Key Lines |
|------|---------|-----------|
| `library/custom_train_functions.py` | `apply_masked_loss()` | 485-501 |
| `library/train_util.py` | Data loading, caching | 1242-1346, 2555-2610 |
| `library/config_util.py` | Config parsing | 89-100 |
| `docs/masked_loss_README.md` | Documentation | Full file |

#### Unique Features to Steal

**1. Alpha Channel Support**
```python
# Extract alpha channel from RGBA image
if subset.alpha_mask:
    if img.shape[2] == 4:  # RGBA
        alpha_mask = img[:, :, 3]  # Extract alpha channel
        alpha_mask = alpha_mask.astype(np.float32) / 255.0
        alpha_mask = torch.FloatTensor(alpha_mask)
    else:
        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
```
**Benefit:** Users can use transparent PNGs directly without separate mask files.

**2. Automatic Cache Regeneration**
```python
# When alpha_mask setting changes, cache is automatically invalidated
# Stored in cache metadata, triggers regeneration on mismatch
```
**Benefit:** Prevents stale cache issues when toggling mask settings.

**3. Dual Validation Path**
```python
# Check both global flag AND per-batch presence
if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
    loss = apply_masked_loss(loss, batch)
```
**Benefit:** Allows mixed datasets with some items having masks, others not.

**4. Comprehensive Documentation**
- Full README explaining both methods
- Visual examples with screenshots
- Notes about latent space resolution limitations
- Guidance on mask dilation for fine details

---

## Recommendations for blissful-tuner

### High Priority (Low Effort, High Value)

#### 1. Add Alpha Channel Support
**Source:** sd-scripts, ai-toolkit
**Implementation:**
```python
# In wan_cache_latents.py, add alpha channel extraction
if args.alpha_mask and item.image_content.shape[2] == 4:
    alpha = item.image_content[:, :, 3]
    mask_weights = torch.from_numpy(alpha).float() / 255.0
elif item.mask_content is not None:
    mask_weights = torch.from_numpy(item.mask_content).float() / 255.0
```
**Effort:** ~2 hours
**Value:** High - users can use transparent PNGs directly

#### 2. Add Mask Inversion Option
**Source:** ai-toolkit
**Implementation:**
```python
# Add argument
parser.add_argument("--invert_mask", action="store_true")

# In loss calculation
if args.invert_mask:
    mask_weights = 1.0 - mask_weights
```
**Effort:** ~30 minutes
**Value:** Medium - useful for "train everything except X" scenarios

#### 3. Fix Mask Interpolation (BUG-003)
**Source:** SimpleTuner, sd-scripts both use `area` mode
**Implementation:**
```python
# Change from LANCZOS to area interpolation
mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")  # Already correct in blissful-tuner!
# But image resize uses LANCZOS - change to NEAREST for masks
```
**Effort:** ~1 hour
**Value:** High - preserves discrete mask values

### Medium Priority (Medium Effort, High Value)

#### 4. Add Probabilistic Mask Application
**Source:** SimpleTuner
**Implementation:**
```python
# Add argument
parser.add_argument("--mask_probability", type=float, default=1.0)

# In loss calculation
if args.use_mask_loss and random.random() < args.mask_probability:
    loss = loss * mask_weights
```
**Effort:** ~1 hour
**Value:** Medium - useful for sparse mask datasets

#### 5. Add Inverted Mask Prior (Regularization)
**Source:** ai-toolkit
**Implementation:**
```python
# Add arguments
parser.add_argument("--inverted_mask_prior", action="store_true")
parser.add_argument("--inverted_mask_prior_weight", type=float, default=0.5)

# In loss calculation
if args.inverted_mask_prior:
    with torch.no_grad():
        prior_pred = model(noisy_latents, timesteps, encoder_hidden_states)
    inverted_mask = 1.0 - mask_weights
    inverted_mask = inverted_mask / inverted_mask.mean()
    prior_loss = F.mse_loss(noise_pred, prior_pred, reduction="none")
    prior_loss = prior_loss * inverted_mask * args.inverted_mask_prior_weight
    loss = loss + prior_loss.mean()
```
**Effort:** ~4 hours
**Value:** High - prevents forgetting unmasked regions

#### 6. Add Gaussian Blur Smoothing
**Source:** ai-toolkit
**Implementation:**
```python
# Add argument
parser.add_argument("--mask_blur", type=float, default=0.0)

# In mask processing (during caching)
if args.mask_blur > 0:
    blur_size = int(args.mask_blur * min(mask.shape) * 2 + 1)
    if blur_size > 1:
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
```
**Effort:** ~2 hours
**Value:** Medium - reduces edge artifacts

### Low Priority (Higher Effort)

#### 7. Add Automated Mask Generation Pipeline
**Source:** SimpleTuner
**Implementation:** Create `scripts/generate_masks.py` using Florence-2 + SAM2
**Effort:** ~1 day
**Value:** High but complex - automates mask creation

#### 8. Add Attention Masking for Transformer Models
**Source:** ai-toolkit, SimpleTuner
**Implementation:** Model-specific attention mask injection
**Effort:** ~1 week
**Value:** Medium - requires per-architecture implementation

---

## Implementation Priority Summary

| Priority | Feature | Source | Effort | Value |
|----------|---------|--------|--------|-------|
| 1 | Alpha channel support | sd-scripts | 2h | High |
| 2 | Mask inversion | ai-toolkit | 30m | Medium |
| 3 | Fix NEAREST interpolation for mask resize | All | 1h | High |
| 4 | Probabilistic application | SimpleTuner | 1h | Medium |
| 5 | Inverted mask prior | ai-toolkit | 4h | High |
| 6 | Gaussian blur smoothing | ai-toolkit | 2h | Medium |
| 7 | Auto mask generation | SimpleTuner | 1d | High |
| 8 | Attention masking | ai-toolkit/SimpleTuner | 1w | Medium |

---

## Code Snippets Ready to Adapt

### Alpha Mask Support (from sd-scripts)
```python
def load_image_with_alpha(image_path, use_alpha_mask=False):
    """Load image and optionally extract alpha channel as mask."""
    img = Image.open(image_path)

    if use_alpha_mask and img.mode == 'RGBA':
        # Extract alpha channel
        alpha = np.array(img.split()[-1])
        mask = alpha.astype(np.float32) / 255.0
        # Convert to RGB for training
        img = img.convert('RGB')
        return np.array(img), mask
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img), None
```

### Mask Normalization (from ai-toolkit)
```python
def normalize_mask(mask_weights):
    """Normalize mask to mean=1.0 to maintain loss magnitude."""
    mask_mean = mask_weights.mean()
    if mask_mean > 0:
        mask_weights = mask_weights / mask_mean
    return mask_weights
```

### Inverted Mask Prior (from ai-toolkit)
```python
def compute_inverted_mask_prior_loss(
    noise_pred, prior_pred, mask_weights, weight=0.5
):
    """Regularize unmasked regions using base model predictions."""
    inverted_mask = 1.0 - mask_weights
    inverted_mask = inverted_mask / inverted_mask.mean()
    prior_loss = F.mse_loss(noise_pred, prior_pred, reduction="none")
    prior_loss = prior_loss * inverted_mask * weight
    return prior_loss.mean()
```

### Probabilistic Application (from SimpleTuner)
```python
def should_apply_mask(probability=1.0):
    """Determine if mask should be applied this batch."""
    return random.random() < probability
```

---

## Conclusion

All three external trainers (ai-toolkit, SimpleTuner, sd-scripts) have mature masked loss implementations with unique features. The most valuable features to port to blissful-tuner are:

1. **Alpha channel support** - Easy win, high user value
2. **Inverted mask prior** - Unique regularization technique
3. **Probabilistic application** - Flexibility for sparse masks
4. **Gaussian blur smoothing** - Reduces artifacts

Blissful-tuner already has unique features (gamma correction, min weight floor) not found in other trainers, making it competitive. Implementing the recommended enhancements would make it the most feature-complete masked loss training solution available.

---

*Report compiled from source code analysis of all four codebases*
*Analysis Date: 2025-01-13*
