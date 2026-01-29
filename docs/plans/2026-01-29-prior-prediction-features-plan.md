# Prior Prediction Features for Blissful-Tuner

**Date:** 2026-01-29
**Status:** Planning

---

## Executive Summary

This document outlines three related features inspired by OneTrainer's masked training capabilities that address the "phantom limb" problem in masked LoRA training:

1. **Masked Prior Preservation** (HIGH VALUE) - Train masked regions toward target, unmasked regions toward base model behavior
2. **Unmasked Probability** (MEDIUM VALUE) - Randomly skip masks for regularization
3. **Prior Prediction Concepts** (MEDIUM VALUE) - Dataset-level regularization via distillation

---

## Background: The Problem We're Solving

### Current Blissful-Tuner Masked Loss Behavior

```python
# Current: single-target weighted loss
loss = mse(model_pred, target) * mask_weights
return weighted_loss.sum() / mask_weights.sum()
```

**Outside-mask behavior today:**
- With `--mask_min_weight > 0`: Weakly pulled toward each training image's inconsistent backgrounds → gradient conflicts
- With `--mask_min_weight == 0`: Completely unconstrained → phantom limbs, hallucinations, clutter

### OneTrainer's Solution: Teacher-Student Output Preservation

OneTrainer adds a **second loss term** that constrains outside-mask regions to match **base model predictions**:

```python
# OneTrainer: dual-objective masked loss
loss_target = mse(model_pred, target) * mask              # Learn inside mask
loss_prior = mse(model_pred, base_pred) * (1 - mask) * w  # Preserve outside mask
total_loss = loss_target + loss_prior
```

**Why this works:** Instead of trying to match inconsistent backgrounds across training images, the model learns to match ONE consistent teacher (the base model under identical noise/timestep/conditioning).

---

## Feature 1: Masked Prior Preservation

### Overview

The core "killer feature" - adds explicit base-model-matching constraint in non-learn regions.

### How It Works (Technical)

1. **Disable LoRA temporarily** via `network.set_enabled(False)`
2. **Run forward pass** with base model → get `prior_pred` (under `torch.no_grad()`)
3. **Re-enable LoRA** via `network.set_enabled(True)`
4. **Run forward pass** with LoRA model → get `model_pred`
5. **Compute dual loss:**
   - Target loss (masked regions): `mse(model_pred, target) * clamped_mask`
   - Prior loss (unmasked regions): `mse(model_pred, prior_pred) * (1 - clamped_mask) * prior_weight`

### Key Implementation Details

**Mask clamping order matters:** OneTrainer clamps THEN inverts:
```python
clamped_mask = clamp(mask, unmasked_weight, 1)  # Floor at unmasked_weight
prior_mask = (1 - clamped_mask)                  # Invert for prior loss
```

This means: if `unmasked_weight=0.1`, then `prior_mask` maxes out at `0.9` (not `1.0`). Higher `unmasked_weight` automatically weakens prior preservation.

**Weighted masks consideration:** Blissful-tuner's masks are multi-level (face=1.0, body=0.5, hair=0.3, background=0.0). This creates:
- Face: full target loss, zero prior loss
- Body: 50% target, 50% prior (preserves body structure while learning)
- Hair: 30% target, 70% prior (mostly preserves base hair behavior)
- Background: ~0% target, ~100% prior (full preservation)

**Option:** Add thresholding for users who want binary prior preservation behavior.

### Performance Characteristics

- **Speed:** ~1.3-1.7x slower (one extra forward pass, but under `no_grad` so no activation storage)
- **VRAM:** Minimal overhead (only stores `prior_pred` tensor, ~same size as `model_pred`)
- **Requirement:** LoRA training only (need to disable/enable network)

### Proposed CLI Arguments

```bash
--prior_preservation_weight FLOAT  # Weight for prior loss term (default: 0.0 = disabled)
                                   # Recommended: 0.5-1.0 when using masked training

--prior_mask_threshold FLOAT       # Optional: binarize mask for prior computation
                                   # Values below threshold → full prior preservation
                                   # (default: None = use continuous weighted masks)
```

### Recommended Settings

| Use Case | `--mask_min_weight` | `--prior_preservation_weight` |
|----------|---------------------|-------------------------------|
| Current behavior (no prior) | 0.05 | 0.0 |
| Light prior preservation | 0.0 | 0.5 |
| Strong prior preservation | 0.0 | 1.0 |
| OneTrainer-equivalent | 0.0 | 1.0 |

---

## Feature 2: Unmasked Probability

### Overview

"Mask dropout" - randomly skip the mask for some training steps so the model occasionally learns globally.

### How It Works

With probability `p`, replace `mask_weights` with `ones_like(mask_weights)`:
```python
if random.random() < unmasked_probability:
    mask_weights = torch.ones_like(mask_weights)
```

### Why It Helps

- Prevents over-reliance on masked training
- Regularization: model doesn't "forget" how to generate full images
- Synergizes with prior preservation (global steps teach full-image coherence)

### Implementation Notes

- Implement at **training time**, not caching (masks are cached)
- Apply per-step, before mask processing
- When unmasked, prior preservation is disabled for that step (mask=1 everywhere → prior_mask=0 everywhere)

### Proposed CLI Arguments

```bash
--unmasked_probability FLOAT  # Probability to skip mask (default: 0.0)
                              # Recommended: 0.05-0.1
```

---

## Feature 3: Prior Prediction Concepts (Future)

### Overview

Dataset-level feature: mark certain dataset entries as "prior prediction" type, meaning the training target for those samples is the base model's output (not the original image).

### Conceptual Difference from Masked Prior Preservation

| Feature | Masked Prior Preservation | Prior Prediction Concepts |
|---------|--------------------------|---------------------------|
| Granularity | Per-pixel (within an image) | Per-sample (entire image) |
| Use case | Focus on subject, preserve background | Regularize with class-representative images |
| Requires masks | Yes | No |
| Similar to | Inpainting | Traditional regularization images |

### How It Would Work

1. Mark dataset entries with `concept_type = "prior_prediction"`
2. For those samples: `target = prior_pred` (base model output)
3. Effect: LoRA learns to match base model for those samples (regularization)

### Why It's Useful

- Alternative to classic "reg images" approach
- Regularize by distillation rather than curated class images
- Can use any images (don't need to match subject class exactly)

### Implementation Complexity

- **Medium-High:** Requires dataset/config changes, concept type enum
- **Recommendation:** Defer to Phase 2, focus on masked prior preservation first

---

## Implementation Phases

### Phase 1: Masked Prior Preservation (Highest Value)

**Scope:**
- Add `--prior_preservation_weight` and `--prior_mask_threshold` arguments
- Implement `prior_model_context()` context manager for disabling LoRA
- Modify loss computation to add prior loss term
- Update `apply_masked_loss()` to support dual-objective loss
- Add logging/banner when prior preservation is enabled

**Supported Architectures:**
- WAN 2.1/2.2 (primary target)
- FLUX.2
- Qwen Image
- Other LoRA-training architectures

**Not Supported:**
- Full fine-tuning (no LoRA to disable)
- Architectures without `network.set_enabled()` support

### Phase 2: Unmasked Probability (Quick Win)

**Scope:**
- Add `--unmasked_probability` argument
- Implement mask dropout in training loop
- Document interaction with prior preservation

### Phase 3: Prior Prediction Concepts (Future)

**Scope:**
- Add `concept_type` to dataset configuration
- Implement concept type handling in data loader
- Modify training loop to detect and handle prior prediction samples

---

## Comparison: OneTrainer vs Blissful-Tuner (After Implementation)

| Feature | OneTrainer | Blissful-Tuner (Planned) |
|---------|-----------|-------------------------|
| Masked prior preservation | ✅ `masked_prior_preservation_weight` | ✅ `--prior_preservation_weight` |
| Unmasked probability | ✅ `unmasked_probability` | ✅ `--unmasked_probability` |
| Prior prediction concepts | ✅ `ConceptType.PRIOR_PREDICTION` | ⏳ Phase 3 |
| Weighted/continuous masks | ❌ Binary only | ✅ Multi-level (face/body/hair/bg) |
| Optional mask thresholding | ❌ N/A | ✅ `--prior_mask_threshold` |
| Per-sample normalization | ✅ Default | ⏳ Consider adding option |

---

## Technical Notes

### Why Cached Prior Prediction Doesn't Work

The prior prediction depends on **per-step random values**:
- Timestep (sampled each step)
- Noise (sampled each step)
- Conditioning (same, but combined with timestep-dependent model state)

You cannot cache "what the base model would predict" because it changes every step. To make caching work, you'd need deterministic fixed noise/timesteps per sample, which fundamentally changes training dynamics.

**Conclusion:** Real-time dual forward pass is the only correct approach.

### Normalization Considerations

**Current blissful-tuner:** Global weighted mean across batch
```python
return weighted_loss.sum() / mask_weights.sum()
```

**OneTrainer:** Per-sample weighted mean, then average over batch
```python
per_sample_loss = (loss * mask).sum(dim=spatial) / mask.sum(dim=spatial)
return per_sample_loss.mean()
```

**Difference:** Global reduction weights samples with larger mask sums more heavily. Per-sample treats all samples equally regardless of mask coverage.

**Recommendation:** Consider adding `--normalize_per_sample` option for Phase 1, or make it default when prior preservation is enabled.

### Weighted Mask Behavior with Prior Preservation

Blissful-tuner's weighted masks create smooth blending:

| Region | Mask Value | Target Loss Weight | Prior Loss Weight |
|--------|------------|-------------------|-------------------|
| Face | 1.0 | 1.0 | 0.0 |
| Body | 0.5 | 0.5 | 0.5 |
| Hair | 0.3 | 0.3 | 0.7 |
| Background | 0.0 | 0.0 | 1.0 |

This means **mid-weight regions (body, hair) get partial prior preservation**, which may or may not be desired.

**Options:**
1. **Continuous mode (default):** Smooth blending, hair/body partially preserved
2. **Threshold mode (`--prior_mask_threshold 0.5`):** Everything below 0.5 gets full prior preservation

---

## References

- OneTrainer PR #505: [Original Prior Prediction Implementation](https://github.com/Nerogar/OneTrainer/pull/505)
- OneTrainer Wiki: [Prior Prediction Documentation](https://github.com/Nerogar/OneTrainer/wiki/Prior-Prediction)
- Blissful-Tuner: [Masked Loss Training Guide](../MASKED_LOSS_TRAINING_GUIDE.md)

---

## Next Steps

1. ✅ Complete overall plan document (this document)
2. ⏳ Create detailed design for Phase 1 (Masked Prior Preservation)
3. ⏳ Implement Phase 1
4. ⏳ Test and validate
5. ⏳ Update documentation
6. ⏳ Implement Phase 2 (Unmasked Probability)
7. ⏳ Consider Phase 3 (Prior Prediction Concepts)
