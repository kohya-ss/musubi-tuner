# Masked Loss Training Enhancement Implementation Plan

**Created:** January 2026
**Based On:** Comprehensive analysis of sd-scripts, SimpleTuner, ai-toolkit, and blissful-tuner codebases
**Purpose:** Detailed roadmap for implementing recommended masked loss improvements

---

## Executive Summary

This document outlines the implementation plan for enhancing blissful-tuner's masked loss training capabilities. The recommendations are derived from:
- Three existing documentation files (`MASKED_LOSS_TRAINING_GUIDE.md`, `TRAINER_MASKED_LOSS_COMPARISON.md`, `MASKED_LOSS_RESEARCH_FINDINGS.md`)
- Deep code analysis of sd-scripts, SimpleTuner, and ai-toolkit implementations
- Academic research on attention-based masking methods

### Current State

Blissful-tuner already has **unique features** not found in other trainers:
- `--mask_gamma`: Gamma correction for contrast control
- `--mask_min_weight`: Minimum weight floor for background regions
- Weighted mean normalization (`loss.sum() / weights.sum()`)
- Multi-architecture support (WAN 2.1/2.2, Qwen-Image)

### Proposed Enhancements

| Priority | Feature | Source | Effort | Value |
|----------|---------|--------|--------|-------|
| **P0** | HunyuanVideo mask caching | Internal bug | 4h | Critical |
| **P1** | Alpha channel support | sd-scripts | 3h | High |
| **P1** | Mask inversion option | ai-toolkit | 30m | Medium |
| **P2** | Inverted mask prior (regularization) | ai-toolkit | 6h | High |
| **P2** | Gaussian blur smoothing | ai-toolkit | 2h | Medium |
| **P2** | Probabilistic mask application | SimpleTuner | 1h | Medium |
| **P3** | Mean normalization mode | ai-toolkit | 2h | Medium |
| **P3** | Automated mask generation script | SimpleTuner | 1d | High |
| **P4** | Attention masking (transformer models) | ai-toolkit/SimpleTuner | 1w | Medium |

---

## Phase 0: Critical Bug Fixes

### 0.1 HunyuanVideo Mask Caching Support

**Status:** NOT IMPLEMENTED
**Impact:** HunyuanVideo training cannot use mask-weighted loss

**Problem:**
The HunyuanVideo latent caching flow (`cache_latents.py`) does not currently write `mask_weights_*` keys into cache files, even though `hv_train_network.py` supports applying mask weights during training.

**Files to Modify:**
- `src/musubi_tuner/cache_latents.py` (add mask processing)
- `src/musubi_tuner/dataset/image_video_dataset.py` (ensure mask loading for HV)

**Implementation:**

```python
# In cache_latents.py, add mask handling similar to wan_cache_latents.py

def encode_and_save_batch(
    vae, items, cache_dir, image_size, vae_dtype, device, vae_chunk_size, vae_tiling
):
    # ... existing latent encoding ...

    # Add mask processing
    for item in items:
        if item.mask_content is not None:
            # Normalize mask from 0-255 to 0.0-1.0
            mask = torch.from_numpy(item.mask_content).float() / 255.0

            # Downsample to latent resolution (8x smaller)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            lat_h, lat_w = latents.shape[-2], latents.shape[-1]
            mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")
            mask = mask.squeeze()  # (lat_H, lat_W)

            # Save alongside latents
            tensors[f"mask_weights_{lat_h}x{lat_w}_float32"] = mask
```

**Testing:**
1. Cache latents with `mask_directory` specified
2. Verify `mask_weights_*` key exists in cache files
3. Run training with `--use_mask_loss` and verify no errors

---

## Phase 1: High Priority Enhancements (Low Effort)

### 1.1 Alpha Channel Support

**Source:** sd-scripts (`library/train_util.py:1328-1336, 2564-2572`)
`**Effort:** 3 hours
`**Files to Modify:**
- `src/musubi_tuner/wan_cache_latents.py`
- `src/musubi_tuner/qwen_image_cache_latents.py`
- `src/musubi_tuner/hv_train_network.py` (CLI args)

**Feature Description:**
Allow users to use RGBA images directly, where the alpha channel serves as the training mask. This eliminates the need for separate mask files.

**CLI Argument:**
```bash
--alpha_mask  # Enable alpha channel as mask source
```

**Implementation:**

```python
# Add to argument parser in setup_parser_common()
parser.add_argument(
    "--alpha_mask",
    action="store_true",
    help="Use image alpha channel as mask (RGBA images required)"
)

# In cache_latents.py, modify image loading:
def load_image_with_mask(image_path, use_alpha_mask=False):
    """Load image and optionally extract alpha channel as mask."""
    img = Image.open(image_path)

    if use_alpha_mask:
        if img.mode == 'RGBA':
            # Extract alpha channel as mask
            alpha = np.array(img.split()[-1])
            mask = alpha.astype(np.float32) / 255.0
            # Convert image to RGB for training
            img = img.convert('RGB')
            return np.array(img), mask
        else:
            logger.warning(f"Alpha mask requested but image is not RGBA: {image_path}")
            return np.array(img.convert('RGB')), None
    else:
        return np.array(img.convert('RGB')), None
```

**Validation Logic:**
```python
# In dataset validation
if args.alpha_mask:
    # Check at least some images have alpha channel
    rgba_count = sum(1 for item in items if item.has_alpha)
    if rgba_count == 0:
        logger.warning("--alpha_mask enabled but no RGBA images found")
```

**Priority over mask_directory:**
```python
# Alpha mask takes precedence if both specified
if args.alpha_mask and image.mode == 'RGBA':
    mask = extract_alpha_channel(image)
elif item.mask_content is not None:
    mask = item.mask_content
else:
    mask = None
```

---

### 1.2 Mask Inversion Option

**Source:** ai-toolkit (`toolkit/dataloader_mixins.py:1339-1340`)
**Effort:** 30 minutes
**Files to Modify:**
- `src/musubi_tuner/hv_train_network.py` (CLI + loss calculation)

**Feature Description:**
Invert mask values so black becomes high weight and white becomes low weight. Useful for "train everything except X" scenarios.

**CLI Argument:**
```bash
--invert_mask  # Invert mask values (black=high weight, white=low weight)
```

**Implementation:**

```python
# Add to argument parser
parser.add_argument(
    "--invert_mask",
    action="store_true",
    help="Invert mask values (black=train, white=ignore)"
)

# In loss calculation (hv_train_network.py, after loading mask_weights)
if args.invert_mask:
    mask_weights = 1.0 - mask_weights
```

**Alternative: Apply during caching:**
```python
# In wan_cache_latents.py
if args.invert_mask:
    mask = 1.0 - mask  # Invert before saving
```

**Recommendation:** Apply during training (not caching) for flexibility. Users can experiment with inversion without re-caching.

---

## Phase 2: Medium Priority Enhancements

### 2.1 Inverted Mask Prior (Regularization)

**Source:** ai-toolkit (`extensions_built_in/sd_trainer/SDTrainer.py:536-559`)
**Effort:** 6 hours
**Files to Modify:**
- `src/musubi_tuner/hv_train_network.py`
- `src/musubi_tuner/wan_train_network.py`

**Feature Description:**
Regularize unmasked regions by using base model predictions as targets. This prevents the model from "forgetting" how to generate backgrounds/unmasked areas.

**CLI Arguments:**
```bash
--inverted_mask_prior           # Enable prior regularization
--inverted_mask_prior_weight 0.5  # Weight of prior loss (default: 0.5)
```

**Implementation:**

```python
# Add to argument parser
parser.add_argument("--inverted_mask_prior", action="store_true",
    help="Regularize unmasked regions using base model predictions")
parser.add_argument("--inverted_mask_prior_weight", type=float, default=0.5,
    help="Weight for inverted mask prior loss (default: 0.5)")

# In training loop, after computing model_pred:
if args.inverted_mask_prior and mask_weights is not None:
    with torch.no_grad():
        # Get prior prediction from frozen/base model
        # (requires storing original model or using EMA)
        prior_pred = frozen_model(noisy_model_input, timesteps, encoder_hidden_states)

    # Create inverted mask multiplier
    inverted_mask = 1.0 - mask_weights
    inverted_mask = inverted_mask / (inverted_mask.mean() + 1e-8)  # Normalize to mean=1.0

    # Compute prior loss
    prior_loss = F.mse_loss(model_pred, prior_pred, reduction="none")
    prior_loss = prior_loss * inverted_mask
    prior_loss = prior_loss.mean() * args.inverted_mask_prior_weight

    # Add to total loss
    loss = loss + prior_loss
```

**Important Considerations:**
1. **Frozen Model Access:** Need to either:
   - Keep a frozen copy of the base model (memory intensive)
   - Use EMA model if available
   - Compute prior once per batch with `torch.no_grad()`

2. **Performance Impact:** Additional forward pass through model (~50% slower)

3. **Compatibility:** May conflict with block swapping/offloading strategies

**Alternative: Gradient-Free Version:**
```python
# Simpler version: Use current model prediction but detach
if args.inverted_mask_prior:
    with torch.no_grad():
        # Predict from same model but don't backprop
        prior_pred = model(noisy_model_input, timesteps, encoder_hidden_states).detach()

    # Rest same as above
```

---

### 2.2 Gaussian Blur Smoothing

**Source:** ai-toolkit (`toolkit/dataloader_mixins.py:1368-1371`)
**Effort:** 2 hours
**Files to Modify:**
- `src/musubi_tuner/wan_cache_latents.py`
- `src/musubi_tuner/qwen_image_cache_latents.py`

**Feature Description:**
Apply Gaussian blur to mask edges during caching to prevent sharp boundary artifacts.

**CLI Argument:**
```bash
--mask_blur 0.005  # Blur radius as fraction of min dimension (default: 0, max: 0.01)
```

**Implementation:**

```python
import cv2

# Add to argument parser
parser.add_argument("--mask_blur", type=float, default=0.0,
    help="Gaussian blur radius as fraction of min dimension (0-0.01)")

# In mask processing during caching
if args.mask_blur > 0:
    min_dim = min(mask.shape[:2])
    blur_radius = int(min_dim * args.mask_blur)
    if blur_radius > 0:
        # Kernel size must be odd
        kernel_size = blur_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
```

**Random Blur (ai-toolkit style):**
```python
# Optional: Random blur per image for augmentation
if args.mask_blur > 0:
    min_dim = min(mask.shape[:2])
    max_blur = int(min_dim * args.mask_blur)
    blur_radius = random.randint(0, max_blur)
    if blur_radius > 0:
        kernel_size = blur_radius * 2 + 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
```

**Recommendation:** Use deterministic blur (not random) for reproducibility during caching.

---

### 2.3 Probabilistic Mask Application

**Source:** SimpleTuner (`helpers/models/common.py:4280-4287`)
**Effort:** 1 hour
**Files to Modify:**
- `src/musubi_tuner/hv_train_network.py`

**Feature Description:**
Apply masked loss with a probability less than 1.0, allowing some batches to train without masking. Useful for datasets with sparse/imperfect masks.

**CLI Argument:**
```bash
--mask_probability 1.0  # Probability of applying mask (default: 1.0 = always)
```

**Implementation:**

```python
import random

# Add to argument parser
parser.add_argument("--mask_probability", type=float, default=1.0,
    help="Probability of applying mask per batch (0.0-1.0)")

# In loss calculation
if mask_weights is not None and args.use_mask_loss:
    if random.random() < args.mask_probability:
        # Apply masked loss (existing code)
        loss = loss * mask_weights
        loss = loss.sum() / (mask_weights.sum() + 1e-8)
    else:
        # Standard loss without mask
        loss = loss.mean()
else:
    loss = loss.mean()
```

**Logging Enhancement:**
```python
# Track mask application rate
if hasattr(args, '_mask_applied_count'):
    args._mask_applied_count += 1 if applied_mask else 0
    args._total_batch_count += 1
```

---

## Phase 3: Lower Priority Enhancements

### 3.1 Mean Normalization Mode

**Source:** ai-toolkit (`SDTrainer.py:1329`)
**Effort:** 2 hours

**Feature Description:**
Normalize mask weights to have mean=1.0, ensuring consistent loss magnitude regardless of mask coverage.

**Current blissful-tuner approach:** Weighted mean (`loss.sum() / weights.sum()`)
**ai-toolkit approach:** Mean normalization (`weights / weights.mean()`)

**CLI Argument:**
```bash
--mask_normalization weighted_mean  # Options: weighted_mean (default), mean_one
```

**Implementation:**

```python
parser.add_argument("--mask_normalization", type=str, default="weighted_mean",
    choices=["weighted_mean", "mean_one"],
    help="Mask normalization method")

# In loss calculation
if args.mask_normalization == "mean_one":
    mask_weights = mask_weights / (mask_weights.mean() + 1e-8)
    loss = (loss * mask_weights).mean()
else:  # weighted_mean (current default)
    loss = loss * mask_weights
    loss = loss.sum() / (mask_weights.sum() + 1e-8)
```

**Trade-offs:**
- `weighted_mean`: Better gradient scaling, handles sparse masks well
- `mean_one`: Consistent loss magnitude, ai-toolkit default

---

### 3.2 Automated Mask Generation Script

**Source:** SimpleTuner (`scripts/datasets/masked_loss/generate_dataset_masks.py`)
**Effort:** 1 day

**Feature Description:**
Generate training masks automatically from text prompts using Florence-2 + SAM2.

**Script Location:** `scripts/generate_masks.py`

**Dependencies:**
```
transformers
sam2 (or segment-anything-2)
supervision
opencv-python
```

**Usage:**
```bash
python scripts/generate_masks.py \
    --input_dir /path/to/images \
    --output_dir /path/to/masks \
    --text_prompt "person" \
    --mask_padding 5 \
    --mask_blur 2
```

**Implementation Outline:**

```python
#!/usr/bin/env python3
"""Generate training masks using Florence-2 + SAM2."""

import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

def load_florence_model(device):
    """Load Florence-2 for object detection."""
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large",
        trust_remote_code=True
    )
    return model, processor

def load_sam2_model(device):
    """Load SAM2 for segmentation."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2("sam2_hiera_large.yaml", "sam2_hiera_large.pt")
    predictor = SAM2ImagePredictor(model)
    return predictor

def generate_mask(image_path, text_prompt, florence_model, sam_predictor, args):
    """Generate mask for a single image."""
    image = Image.open(image_path).convert("RGB")

    # Florence-2: Detect objects from text prompt
    inputs = florence_processor(
        text=f"<CAPTION_TO_PHRASE_GROUNDING>{text_prompt}",
        images=image,
        return_tensors="pt"
    ).to(device)

    outputs = florence_model.generate(**inputs, max_new_tokens=1024)
    response = florence_processor.post_process_generation(outputs, task="<CAPTION_TO_PHRASE_GROUNDING>")

    # Extract bounding boxes
    bboxes = response.get("bboxes", [])
    if not bboxes:
        return None

    # SAM2: Segment detected regions
    image_np = np.array(image)
    sam_predictor.set_image(image_np)
    masks, _, _ = sam_predictor.predict(box=np.array(bboxes), multimask_output=False)

    # Combine masks
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    # Post-processing
    if args.mask_padding > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.dilate(combined_mask, kernel, iterations=args.mask_padding)

    if args.mask_blur > 0:
        ksize = args.mask_blur * 2 + 1
        combined_mask = cv2.GaussianBlur(combined_mask, (ksize, ksize), 0)

    return combined_mask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--text_prompt", default="person")
    parser.add_argument("--mask_padding", type=int, default=0)
    parser.add_argument("--mask_blur", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load models
    florence_model, florence_processor = load_florence_model(args.device)
    sam_predictor = load_sam2_model(args.device)

    # Process images
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for image_file in input_path.glob("*.png"):
        mask = generate_mask(image_file, args.text_prompt, florence_model, sam_predictor, args)
        if mask is not None:
            mask_path = output_path / image_file.name
            cv2.imwrite(str(mask_path), mask)
            print(f"Saved mask: {mask_path}")

if __name__ == "__main__":
    main()
```

---

## Phase 4: Advanced Research (Future)

### 4.1 Attention Masking for Transformer Models

**Source:** ai-toolkit, SimpleTuner
**Effort:** 1 week
**Impact:** Model-specific, requires per-architecture implementation

**Concept:** Apply masking at the attention level rather than pixel/loss level.

**SimpleTuner TREAD Integration:**
```python
# Prevent token dropping in masked regions
if conditioning_type in ("mask", "segmentation"):
    mask_lat = F.interpolate(mask_img, size=(h_tokens, w_tokens), mode="area")
    force_keep = mask_lat.flatten(2).squeeze(1) > 0.5  # (B, S_img)
    transformer_kwargs["force_keep_mask"] = force_keep
```

**Deferred:** Requires significant architectural changes and per-model implementation.

---

## Implementation Checklist

### Phase 0: Critical (Week 1)
- [ ] Add mask caching to HunyuanVideo `cache_latents.py`
- [ ] Add `mask_weights_*` key handling in HV training loop
- [ ] Test HunyuanVideo masked loss end-to-end

### Phase 1: High Priority (Week 2)
- [ ] Implement `--alpha_mask` CLI argument
- [ ] Add alpha channel extraction in cache scripts
- [ ] Implement `--invert_mask` CLI argument
- [ ] Add mask inversion in loss calculation
- [ ] Update documentation with new options

### Phase 2: Medium Priority (Week 3-4)
- [ ] Implement `--inverted_mask_prior` with frozen model
- [ ] Add `--inverted_mask_prior_weight` parameter
- [ ] Implement `--mask_blur` in caching scripts
- [ ] Implement `--mask_probability` in training loop
- [ ] Performance testing with new features

### Phase 3: Lower Priority (Week 5+)
- [ ] Add `--mask_normalization` mode selection
- [ ] Create `scripts/generate_masks.py` utility
- [ ] Add Florence-2 + SAM2 dependencies (optional)

---

## Testing Strategy

### Unit Tests
```python
def test_alpha_mask_extraction():
    """Verify alpha channel is correctly extracted as mask."""
    rgba_image = create_test_rgba_image()
    img, mask = load_image_with_mask(rgba_image, use_alpha_mask=True)
    assert mask is not None
    assert mask.shape == img.shape[:2]
    assert mask.max() <= 1.0
    assert mask.min() >= 0.0

def test_mask_inversion():
    """Verify mask inversion works correctly."""
    mask = torch.tensor([0.0, 0.5, 1.0])
    inverted = 1.0 - mask
    assert torch.allclose(inverted, torch.tensor([1.0, 0.5, 0.0]))

def test_inverted_mask_prior():
    """Verify inverted mask prior loss computation."""
    # ... test implementation
```

### Integration Tests
1. **End-to-End Alpha Mask:**
   - Create RGBA test images with varying alpha
   - Cache with `--alpha_mask`
   - Train for 10 steps
   - Verify loss values are reasonable

2. **Inverted Mask Prior:**
   - Train with and without `--inverted_mask_prior`
   - Compare background quality in generated samples
   - Measure training stability

3. **Probabilistic Application:**
   - Set `--mask_probability 0.5`
   - Verify ~50% of batches apply mask (via logging)

---

## Risk Assessment

| Feature | Risk | Mitigation |
|---------|------|------------|
| Alpha mask | Low | Clear fallback to RGB if no alpha |
| Mask inversion | Low | Simple arithmetic operation |
| Inverted mask prior | Medium | Performance overhead, memory usage |
| Gaussian blur | Low | Well-tested OpenCV function |
| Probabilistic application | Low | Simple random gate |
| Mean normalization | Low | Mathematical equivalence testing |
| Auto mask generation | Medium | External dependencies, model downloads |

---

## Appendix: Code Snippets from External Codebases

### A. sd-scripts: apply_masked_loss()
```python
def apply_masked_loss(loss, batch):
    if "conditioning_images" in batch:
        mask_image = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        mask_image = mask_image / 2 + 0.5
    elif "alpha_masks" in batch and batch["alpha_masks"] is not None:
        mask_image = batch["alpha_masks"].to(dtype=loss.dtype).unsqueeze(1)
    else:
        return loss

    mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
    loss = loss * mask_image
    return loss
```

### B. ai-toolkit: Inverted Mask Prior
```python
if self.train_config.inverted_mask_prior and prior_pred is not None and has_mask:
    with torch.no_grad():
        prior_mask = batch.mask_tensor.to(self.device_torch, dtype=dtype)
        prior_mask = torch.nn.functional.interpolate(prior_mask, size=(lat_height, lat_width), mode='bicubic')
        prior_mask = torch.cat([prior_mask[:1]] * noise_pred.shape[1], dim=1)
        prior_mask_multiplier = 1.0 - prior_mask
        prior_mask_multiplier = prior_mask_multiplier / prior_mask_multiplier.mean()
```

### C. SimpleTuner: Probabilistic Application
```python
if conditioning_type == "segmentation" and apply_conditioning_mask:
    if random.random() < self.config.masked_loss_probability:
        mask_image = prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)
        mask_image = torch.sum(mask_image, dim=1, keepdim=True) / 3
        mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
        mask_image = mask_image / 2 + 0.5
        mask_image = (mask_image > 0).to(dtype=loss.dtype, device=loss.device)
        loss = loss * mask_image
```

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01 | 1.0 | Initial implementation plan |

---

*Plan compiled from analysis of sd-scripts, SimpleTuner, ai-toolkit, and blissful-tuner codebases*
