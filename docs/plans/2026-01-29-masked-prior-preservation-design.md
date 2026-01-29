# Masked Prior Preservation: Detailed Design

**Date:** 2026-01-29
**Status:** Draft - Awaiting Review
**Parent Document:** [Prior Prediction Features Plan](./2026-01-29-prior-prediction-features-plan.md)

---

## 1. Feature Overview

Masked Prior Preservation adds a teacher-student distillation objective to masked training:
- **Inside mask:** Learn from training target (current behavior)
- **Outside mask:** Match base model predictions (new constraint)

This directly solves the "phantom limb" problem by giving the model a consistent, coherent target in non-learn regions.

---

## 2. Mathematical Formulation

### Current Blissful-Tuner Loss

```
L_current = weighted_mean(mse(pred, target) * mask)
          = sum(mse(pred, target) * mask) / sum(mask)
```

### Proposed Loss with Prior Preservation

```
# Let:
#   pred       = LoRA model prediction
#   target     = training target (noise - latents for flow matching)
#   prior      = base model prediction (LoRA disabled)
#   mask_raw   = original mask weights in [0, 1]
#   w_prior    = prior_preservation_weight (user arg)
#   w_min      = mask_min_weight (user arg, existing)
#   gamma      = mask_gamma (user arg, existing)
#   threshold  = prior_mask_threshold (user arg, optional)

# Step 1: Keep raw mask for thresholding
mask_raw = mask.clamp(0, 1)

# Step 2: Process mask for target loss (existing behavior)
mask_processed = mask_raw ** gamma
if w_min > 0:
    mask_processed = mask_processed * (1 - w_min) + w_min

# Step 3: Compute prior mask (complement of processed mask)
prior_mask = (1 - mask_processed)

# Step 4: Optional thresholding (on RAW mask, before gamma/min)
if threshold is not None:
    prior_mask = (mask_raw < threshold).float()  # Full prior where raw mask < threshold
    mask_processed = mask_processed * (1 - prior_mask)  # Remove overlap: no target loss where prior applies

# Step 5: Target loss - weighted mean, region-normalized
L_target = weighted_mean(mse(pred, target), mask_processed)
         = sum(mse(pred, target) * mask_processed) / sum(mask_processed)

# Step 6: Prior loss - weighted mean, region-normalized, THEN scaled
L_prior = weighted_mean(mse(pred, prior), prior_mask) * w_prior
        = (sum(mse(pred, prior) * prior_mask) / sum(prior_mask)) * w_prior

# Step 7: Combine (additive, w_prior acts as true knob)
loss = L_target + L_prior
```

### Why This Normalization Matters

**CRITICAL:** The formulation above uses **region-normalized means + explicit weighting**.

This ensures `w_prior` behaves as a true independent knob:
- `w_prior=0.5` means "prior loss contributes half as much as target loss"
- `w_prior=1.0` means "prior loss contributes equally to target loss"

**Rejected alternative (combined normalization):**
```python
# DON'T DO THIS - w_prior interacts unpredictably with mask area
loss = sum(L_target + L_prior) / sum(mask_processed + prior_mask * w_prior)
```

This would dilute the target loss when masks are small and make `w_prior` dependent on mask coverage.

### Per-Sample vs Global Normalization

**Global (current blissful-tuner):** `sum_all / sum_all_weights`
- Samples with larger mask areas contribute more to loss
- Can be unpredictable when mask coverage varies across batch

**Per-sample:** `mean([sum_per_sample / sum_weights_per_sample])`
- Each sample contributes equally regardless of mask size
- More predictable behavior

**Decision:** Add `--normalize_per_sample` option, default to global for backward compatibility, but recommend per-sample when prior preservation is enabled.

---

## 3. Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Training Loop                               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 1. Sample batch (latents, noise, timesteps, mask_weights) │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 2. If prior_preservation_weight > 0:                      │  │
│  │    a. Disable LoRA: network.set_enabled(False)            │  │
│  │    b. Forward pass (no_grad): prior_pred = call_dit(...)  │  │
│  │    c. Enable LoRA: network.set_enabled(True)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 3. Forward pass (grad): model_pred = call_dit(...)        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 4. Compute loss:                                          │  │
│  │    loss = apply_masked_loss_with_prior(                   │  │
│  │        loss_unreduced, mask_weights, prior_loss, args     │  │
│  │    )                                                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ 5. Backward pass + optimizer step                         │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### File Changes

| File | Changes |
|------|---------|
| `src/musubi_tuner/modules/mask_loss.py` | Add `apply_masked_loss_with_prior()`, add new args |
| `src/musubi_tuner/hv_train_network.py` | Add `prior_model_context()`, modify training loop |
| `src/musubi_tuner/wan_train_network.py` | Inherit from updated base, may need overrides |
| `src/musubi_tuner/flux_2_train_network.py` | Similar updates for FLUX.2 |
| `src/musubi_tuner/qwen_image_train.py` | Similar updates for Qwen Image |
| `docs/MASKED_LOSS_TRAINING_GUIDE.md` | Document new features |

---

## 4. Detailed Implementation

### 4.1 New Arguments (`mask_loss.py`)

```python
def add_mask_loss_args(parser: argparse.ArgumentParser) -> None:
    # ... existing args ...

    parser.add_argument(
        "--prior_preservation_weight",
        type=float,
        default=0.0,
        help="Weight for prior preservation loss in unmasked regions (default: 0.0 = disabled). "
        "When enabled, unmasked regions are trained to match base model predictions, preventing "
        "phantom limbs and background hallucinations. Recommended: 0.5-1.0. "
        "NOTE: When enabled, recommend --mask_min_weight 0.0. Requires LoRA training. "
        "/ マスク外領域での事前保存損失の重み（デフォルト：0.0=無効）。",
    )

    parser.add_argument(
        "--prior_mask_threshold",
        type=float,
        default=None,
        help="Optional: Apply prior preservation only where RAW mask < threshold (before gamma/min_weight). "
        "Default: None (continuous mode - prior preservation scales with inverse mask). "
        "Set to 0.05-0.1 to preserve only true background while body/hair still train to target. "
        "/ 事前保存を適用するマスクしきい値（オプション）。",
    )

    parser.add_argument(
        "--normalize_per_sample",
        action="store_true",
        help="Normalize loss per-sample before averaging across batch (default: global normalization). "
        "Recommended when prior preservation is enabled for more predictable behavior. "
        "/ サンプルごとに損失を正規化してからバッチ全体で平均する。",
    )
```

### 4.2 Validation (`mask_loss.py`)

```python
def validate_mask_loss_args(args: argparse.Namespace) -> None:
    # ... existing validation ...

    prior_preservation_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    if prior_preservation_weight < 0:
        raise ValueError("--prior_preservation_weight must be >= 0")

    if prior_preservation_weight > 0:
        # Check for LoRA training (will be validated more precisely in trainer)
        if not getattr(args, "use_mask_loss", False):
            logger.warning(
                "--prior_preservation_weight > 0 but --use_mask_loss is not enabled. "
                "Prior preservation requires masked training."
            )

    prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
    if prior_mask_threshold is not None:
        if prior_mask_threshold <= 0 or prior_mask_threshold >= 1:
            raise ValueError("--prior_mask_threshold must be in range (0, 1)")
        # Warn about mask_min_weight interaction
        mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))
        if mask_min_weight > 0:
            logger.warning(
                f"--prior_mask_threshold={prior_mask_threshold} with --mask_min_weight={mask_min_weight}: "
                "threshold mode removes overlap but min_weight still applies to remaining target regions."
            )
```

### 4.3 Prior Model Context Manager (`hv_train_network.py`)

```python
from contextlib import contextmanager

class NetworkTrainer:
    # ... existing code ...

    @contextmanager
    def prior_model_context(self, network):
        """
        Context manager to temporarily disable LoRA for computing prior predictions.

        Design decision: Keep model in train() mode during teacher forward.
        - Matches OneTrainer behavior
        - DiT models (WAN, FLUX, etc.) typically have no dropout, so train/eval makes no difference
        - If a future model has dropout, train mode gives "realistic" teacher predictions

        Usage:
            with self.prior_model_context(network):
                prior_pred = call_dit(...)  # Uses base model only
        """
        if network is None:
            yield
            return

        try:
            network.set_enabled(False)
            yield
        finally:
            network.set_enabled(True)
```

### 4.4 Training Loop Modification (`hv_train_network.py`)

The key modification is in the training step. Here's the pseudocode for the change:

```python
def training_step(self, batch, network, transformer, args, accelerator, ...):
    # ... existing setup (noise, timesteps, noisy_model_input) ...

    prior_pred = None
    prior_loss_unreduced = None
    prior_preservation_weight = getattr(args, "prior_preservation_weight", 0.0)
    mask_weights = batch.get("mask_weights")

    # Compute prior prediction if prior preservation is enabled
    need_prior = (
        prior_preservation_weight > 0
        and getattr(args, "use_mask_loss", False)
        and mask_weights is not None
    )

    if need_prior:
        # Optimization: skip teacher forward if mask is all ones (prior_mask will be all zeros)
        # This happens during "unmasked probability" steps or full-coverage masks
        mask_min = mask_weights.min().item()
        if mask_min >= (1.0 - 1e-6):
            need_prior = False  # No prior loss contribution possible

    if need_prior:
        with torch.no_grad():
            with self.prior_model_context(network):
                # IMPORTANT: Use exact same inputs (noisy_model_input, timesteps, etc.)
                prior_pred_raw, _ = self.call_dit(
                    args, accelerator, transformer, latents, batch,
                    noise, noisy_model_input, timesteps, network_dtype
                )
        prior_pred = prior_pred_raw.detach()

    # Compute model prediction (with LoRA enabled)
    model_pred, target = self.call_dit(
        args, accelerator, transformer, latents, batch,
        noise, noisy_model_input, timesteps, network_dtype
    )

    # Compute losses
    loss_unreduced = F.mse_loss(model_pred.float(), target.float(), reduction="none")

    if prior_pred is not None:
        prior_loss_unreduced = F.mse_loss(model_pred.float(), prior_pred.float(), reduction="none")

    # Apply masked loss with prior preservation
    loss = apply_masked_loss_with_prior(
        loss_unreduced,
        mask_weights,
        prior_loss_unreduced=prior_loss_unreduced,
        args=args,
        layout="video",  # or "layered" for Qwen Image
    )

    return loss
```

### 4.5 Loss Function (`mask_loss.py`)

```python
def apply_masked_loss_with_prior(
    loss: torch.Tensor,
    mask_weights: torch.Tensor | None,
    *,
    prior_loss_unreduced: torch.Tensor | None = None,
    args: argparse.Namespace,
    layout: MaskLossLayout = "video",
    drop_base_frame: bool = False,
) -> torch.Tensor:
    """
    Apply masked loss with optional prior preservation.

    Uses region-normalized means + explicit weighting:
        L_target = weighted_mean(mse, mask_processed)
        L_prior  = weighted_mean(mse, prior_mask) * w_prior
        loss = L_target + L_prior

    This ensures w_prior acts as a true independent knob.

    Args:
        loss: Unreduced loss tensor (B, C, F, H, W) or (B, C, H, W)
        mask_weights: Mask weights tensor, or None to use uniform weights
        prior_loss_unreduced: Unreduced prior loss tensor (same shape as loss), or None
        args: Namespace with mask_gamma, mask_min_weight, prior_preservation_weight,
              prior_mask_threshold, normalize_per_sample
        layout: "video" or "layered"
        drop_base_frame: Whether to drop base frame for layered layout

    Returns:
        Scalar loss tensor
    """
    prior_preservation_weight = float(getattr(args, "prior_preservation_weight", 0.0))
    normalize_per_sample = getattr(args, "normalize_per_sample", False)

    # If no mask or mask loss disabled, fall back to simple mean
    if mask_weights is None or not getattr(args, "use_mask_loss", False):
        return loss.float().mean()

    # Handle tensor shapes
    loss, mask_weights = _prepare_tensors(loss, mask_weights, layout, drop_base_frame)

    # Keep raw mask for thresholding (before gamma/min_weight)
    mask_raw = mask_weights.clamp(0.0, 1.0)

    # Apply gamma and min_weight to get processed mask for target loss
    mask_gamma = float(getattr(args, "mask_gamma", 1.0))
    mask_min_weight = float(getattr(args, "mask_min_weight", 0.0))

    mask_processed = mask_raw.clone()
    if mask_gamma != 1.0:
        mask_processed = mask_processed ** mask_gamma
    if mask_min_weight > 0:
        mask_processed = mask_processed * (1.0 - mask_min_weight) + mask_min_weight

    # Compute prior mask (complement of processed mask)
    prior_mask = (1 - mask_processed)

    # Optional: threshold on RAW mask (before gamma/min_weight)
    prior_mask_threshold = getattr(args, "prior_mask_threshold", None)
    if prior_mask_threshold is not None:
        # Binarize: full prior preservation where raw mask < threshold
        prior_mask = (mask_raw < prior_mask_threshold).float()

    # === Threshold mode: prevent target/prior overlap ===
    # In threshold mode, zero out target weight where prior applies to avoid
    # simultaneously pushing the same region toward both target and prior
    if prior_mask_threshold is not None:
        mask_processed = mask_processed * (1 - prior_mask)

    # === Target Loss (inside mask) ===
    target_loss_weighted = loss * mask_processed
    # All sums in float32 for numerical stability (1e-8 meaningless in fp16)
    target_weight_sum = mask_processed.sum(dtype=torch.float32)

    if normalize_per_sample:
        # Per-sample weighted mean, then average over batch
        # Reduce over C, F, H, W dimensions (keep batch)
        reduce_dims = tuple(range(1, loss.ndim))
        target_sum = target_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)
        target_weight = mask_processed.sum(dim=reduce_dims, dtype=torch.float32)
        # Handle samples with zero target weight: treat as 0 contribution
        valid_target = target_weight > 1e-8
        per_sample_target = torch.where(valid_target, target_sum / target_weight.clamp_min(1e-8), torch.zeros_like(target_sum))
        L_target = per_sample_target.mean()
    else:
        # Global weighted mean
        if target_weight_sum < 1e-8:
            L_target = loss.new_zeros(())
        else:
            L_target = target_loss_weighted.sum(dtype=torch.float32) / target_weight_sum

    # === Prior Loss (outside mask) ===
    if prior_preservation_weight > 0 and prior_loss_unreduced is not None:
        # Check if prior mask is effectively all zeros (skip computation)
        # Use float32 for the sum so clamp_min(1e-8) is meaningful under fp16/bf16
        prior_mask_sum = prior_mask.sum(dtype=torch.float32)
        if prior_mask_sum < 1e-8:
            # No prior loss contribution (e.g., unmasked step or mask=1 everywhere)
            L_prior = loss.new_zeros(())
        else:
            # Prepare prior loss tensor
            prior_loss_unreduced = prior_loss_unreduced.to(loss.device, dtype=loss.dtype)
            if prior_loss_unreduced.ndim == 4:
                prior_loss_unreduced = prior_loss_unreduced.unsqueeze(2)

            prior_loss_weighted = prior_loss_unreduced * prior_mask

            if normalize_per_sample:
                reduce_dims = tuple(range(1, loss.ndim))
                prior_sum = prior_loss_weighted.sum(dim=reduce_dims, dtype=torch.float32)
                prior_weight = prior_mask.sum(dim=reduce_dims, dtype=torch.float32)
                # Handle samples with zero prior weight: treat as 0 contribution
                valid_prior = prior_weight > 1e-8
                per_sample_prior = torch.where(valid_prior, prior_sum / prior_weight.clamp_min(1e-8), torch.zeros_like(prior_sum))
                L_prior = per_sample_prior.mean() * prior_preservation_weight
            else:
                L_prior = (prior_loss_weighted.sum(dtype=torch.float32) / prior_mask_sum) * prior_preservation_weight
    else:
        L_prior = loss.new_zeros(())

    # === Combine: region-normalized means + explicit weighting ===
    total_loss = L_target + L_prior

    return total_loss


def _prepare_tensors(
    loss: torch.Tensor,
    mask_weights: torch.Tensor,
    layout: MaskLossLayout,
    drop_base_frame: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare loss and mask tensors for computation.
    Extracted from apply_masked_loss for reuse.
    """
    # Handle 4D vs 5D loss
    if loss.ndim == 4:
        if layout != "video":
            raise ValueError("4D loss is only supported for layout='video'")
        loss = loss.unsqueeze(2)
    elif loss.ndim != 5:
        raise ValueError(f"Expected loss to be 4D or 5D, got {loss.ndim}D")

    # Handle mask dimensions
    if mask_weights.ndim == 4:
        mask_weights = mask_weights.unsqueeze(1)
    elif mask_weights.ndim != 5:
        raise ValueError(f"Unexpected mask_weights shape: {tuple(mask_weights.shape)}")

    mask_weights = mask_weights.to(loss.device, dtype=loss.dtype)

    # Layout-specific handling
    if layout == "video":
        if mask_weights.shape[0] != loss.shape[0] or mask_weights.shape[2:] != loss.shape[2:]:
            raise ValueError(f"Shape mismatch: mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)}")
        mask_weights = mask_weights.expand_as(loss)
    elif layout == "layered":
        if drop_base_frame:
            mask_weights = mask_weights[:, :, 1:, :, :]
        if (
            mask_weights.shape[0] != loss.shape[0]
            or mask_weights.shape[2] != loss.shape[1]
            or mask_weights.shape[3:] != loss.shape[3:]
        ):
            raise ValueError(f"Shape mismatch for layered: mask={tuple(mask_weights.shape)} loss={tuple(loss.shape)}")
        mask_weights = mask_weights.permute(0, 2, 1, 3, 4)
        mask_weights = mask_weights.expand_as(loss)

    return loss, mask_weights
```

### 4.6 Updated Banner (`mask_loss.py`)

```python
def log_mask_loss_banner(logger: Any, args: argparse.Namespace, cache_hint: str | None = None) -> None:
    if not getattr(args, "use_mask_loss", False):
        return

    prior_weight = getattr(args, "prior_preservation_weight", 0.0)
    prior_threshold = getattr(args, "prior_mask_threshold", None)
    mask_min_weight = getattr(args, "mask_min_weight", 0.0)
    normalize_per_sample = getattr(args, "normalize_per_sample", False)

    logger.info("=" * 60)
    if prior_weight > 0:
        logger.info("MASKED PRIOR PRESERVATION TRAINING ENABLED")
    else:
        logger.info("MASK-WEIGHTED LOSS TRAINING ENABLED")
    logger.info("=" * 60)
    logger.info(f"  mask_min_weight: {mask_min_weight}")
    logger.info(f"  mask_gamma: {getattr(args, 'mask_gamma', None)}")
    if prior_weight > 0:
        logger.info(f"  prior_preservation_weight: {prior_weight}")
        if prior_threshold is not None:
            logger.info(f"  prior_mask_threshold: {prior_threshold}")
        logger.info(f"  normalize_per_sample: {normalize_per_sample}")
        logger.info("-" * 60)
        logger.info("PRIOR PRESERVATION: Unmasked regions will match base model.")
        logger.info("                    Expect ~1.3-1.7x training time.")
        if mask_min_weight > 0:
            logger.warning(f"  NOTE: mask_min_weight={mask_min_weight} reduces prior effect.")
            logger.warning("        Recommend --mask_min_weight 0.0 with prior preservation.")
    logger.info("-" * 60)
    logger.info("IMPORTANT: Masks must be baked into latent cache!")
    if cache_hint:
        logger.info(cache_hint)
    logger.info("=" * 60)
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

| Test | Description |
|------|-------------|
| `test_prior_model_context_disables_network` | Verify `set_enabled(False)` is called |
| `test_apply_masked_loss_with_prior_shapes` | Verify tensor shape handling |
| `test_prior_loss_only_in_unmasked_regions` | Verify prior loss is zero where mask=1 |
| `test_target_loss_only_in_masked_regions` | Verify target loss is zero where mask=0 |
| `test_threshold_binarizes_prior_mask` | Verify threshold behavior |
| `test_backward_compat_without_prior` | Verify `prior_preservation_weight=0` matches existing behavior |

### 5.2 Integration Tests

| Test | Description |
|------|-------------|
| `test_wan_training_with_prior_preservation` | Full training loop with WAN |
| `test_prior_prediction_matches_base_model` | Verify prior_pred is base model output |
| `test_lora_required_for_prior_preservation` | Verify error for non-LoRA training |

### 5.3 Manual Validation

1. Train character LoRA with and without prior preservation
2. Generate images with prompt "woman playing chess at the park" (from OneTrainer example)
3. Compare background coherence and phantom limb artifacts

---

## 6. Implementation Order

### Step 1: Add Arguments and Validation
- Add new CLI arguments to `mask_loss.py`
- Add validation logic
- Update banner

### Step 2: Add Context Manager
- Implement `prior_model_context()` in `hv_train_network.py`
- Test that LoRA is properly disabled/enabled

### Step 3: Modify Loss Function
- Implement `apply_masked_loss_with_prior()`
- Extract shared tensor preparation logic
- Keep backward compatibility with existing `apply_masked_loss()`

### Step 4: Modify Training Loops
- Update `hv_train_network.py` training step
- Update `wan_train_network.py` (may inherit from base)
- Update `flux_2_train_network.py`
- Update `qwen_image_train.py`

### Step 5: Testing
- Write and run unit tests
- Run integration tests
- Manual validation with character LoRA

### Step 6: Documentation
- Update `MASKED_LOSS_TRAINING_GUIDE.md`
- Add examples and recommended settings
- Document limitations (LoRA only)

---

## 7. Design Decisions (Resolved)

### Q1: Should we refactor `apply_masked_loss` or create new function?

**Decision: Option A - Create new function**

- Create `apply_masked_loss_with_prior()` as new function
- Keep existing `apply_masked_loss()` untouched for backward compatibility
- Extract shared logic into `_prepare_tensors()` helper to minimize duplication

### Q2: How to handle `mask_min_weight` interaction with prior preservation?

**Decision: Match OneTrainer coupling in continuous mode**

- Compute `mask_processed` (clamp + gamma + min floor) once
- Use `prior_mask = 1 - mask_processed` (continuous complement)
- Document strongly: **when prior preservation is enabled, recommend `--mask_min_weight 0.0`**

**Rationale:** Keeps semantics consistent ("mask = how much you learn from target; complement = how much you preserve base") and gives users one mental model.

### Q3: Per-sample vs global normalization?

**Decision: Add `--normalize_per_sample` option**

- Default: global normalization (backward compatible)
- When prior preservation enabled: recommend `--normalize_per_sample` for predictable behavior
- Even with batch_size=1, this is worth doing for correctness and future-proofing

### Q4: Should `prior_mask_threshold` operate on original or processed mask?

**Decision: Option A - Threshold on RAW mask (before gamma/min_weight)**

- Keep `mask_raw = mask.clamp(0, 1)` before applying gamma/min
- Threshold check: `prior_mask = (mask_raw < threshold).float()`
- This avoids weird cases where a large `mask_min_weight` accidentally disables prior due to thresholding
- More intuitive: "apply prior where original mask is below X"

---

## 8. Performance Optimizations

### Skip Teacher Forward When Not Needed

If the effective prior mask is all zeros, skip computing `prior_pred`:

```python
# In training loop, before computing prior
if prior_preservation_weight > 0 and mask_weights is not None:
    # Quick check: if mask is all ones, prior_mask will be all zeros
    if mask_weights.min() >= (1.0 - 1e-6):
        # Skip teacher forward - no prior loss contribution
        prior_pred = None
    else:
        with torch.no_grad():
            with self.prior_model_context(network):
                prior_pred = ...
```

This provides significant speedup for "unmasked probability" steps and images with full-coverage masks.

### Logging for Debugging

Periodically log loss breakdown to help users diagnose issues:

```python
if global_step % log_interval == 0:
    logger.info(f"  mask_mean: {mask_processed.mean():.4f}")
    logger.info(f"  prior_mask_mean: {prior_mask.mean():.4f}")
    logger.info(f"  L_target: {L_target.item():.6f}")
    logger.info(f"  L_prior: {L_prior.item():.6f}")
```

This lets users see if prior is accidentally dominating (or not contributing at all).

### torch.compile / CUDA Graph Compatibility

Toggling `network.set_enabled(False/True)` creates two forward modes, which can affect torch.compile graph reuse.

**If users report slowdown with `--compile`:**

Consider using `set_multiplier(0)` instead of `set_enabled(False)`:

```python
@contextmanager
def prior_model_context(self, network, use_multiplier=False):
    """
    use_multiplier=True may be faster with torch.compile
    (single graph with multiplier=0 vs two distinct graphs)
    """
    if network is None:
        yield
        return

    if use_multiplier:
        old_multiplier = network.multiplier
        try:
            network.set_multiplier(0)
            yield
        finally:
            network.set_multiplier(old_multiplier)
    else:
        try:
            network.set_enabled(False)
            yield
        finally:
            network.set_enabled(True)
```

For v1: use `set_enabled(False)` (cleaner). Add `--prior_use_multiplier` flag if compile issues arise.

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression for non-prior users | High | Guard prior computation with `if prior_weight > 0` |
| Breaking existing masked training | High | Extensive backward compatibility testing |
| Memory issues with large batches | Medium | Prior forward is `no_grad`, minimal overhead |
| Inconsistent behavior across architectures | Medium | Share implementation via base class |
| User confusion about settings | Medium | Clear documentation, recommended presets |
| torch.compile slowdown | Low | Document; add multiplier-based context if needed |

---

## 10. Recommended Settings

### For Weighted Mask Users (face/body/hair/background tiers)

**Continuous mode (default):** Prior preservation scales smoothly with inverse mask
```bash
--use_mask_loss \
--prior_preservation_weight 1.0 \
--mask_min_weight 0.0 \
--mask_gamma 0.7 \
--normalize_per_sample
```

**Threshold mode:** Prior preservation only on true background (body/hair still train fully)
```bash
--use_mask_loss \
--prior_preservation_weight 1.0 \
--mask_min_weight 0.0 \
--prior_mask_threshold 0.1 \
--normalize_per_sample
```

### For Binary Mask Users (subject vs background)

```bash
--use_mask_loss \
--prior_preservation_weight 1.0 \
--mask_min_weight 0.0 \
--normalize_per_sample
```

---

## Understanding Weighted Mask Behavior with Prior Preservation

### How Tiers Affect Learning

With typical weighted masks (face=255, body=128, hair=80, background=0):

| Region | Mask (raw) | mask_processed (γ=0.7) | prior_mask | Effect |
|--------|------------|------------------------|------------|--------|
| Face | 1.0 | 1.0 | 0.0 | **Full learning**, no preservation |
| Body | 0.50 | 0.62 | 0.38 | ~62% target, ~38% prior |
| Hair | 0.31 | 0.47 | 0.53 | ~47% target, **~53% prior** |
| Background | 0.0 | 0.0 | 1.0 | No learning, **full preservation** |

**Key implications:**
- **Hair will learn slowly** with prior preservation in continuous mode (hair often has low mask values like the example, meaning strong prior pull)
- **Body/clothing will also be partially preserved**, which can improve coherence but slow learning
- If hair style is important to your subject, consider **threshold mode** with `--prior_mask_threshold 0.1` so only true background (mask < 0.1) gets prior preservation

### Threshold Semantics

The `--prior_mask_threshold` operates on the **latent-resolution mask** (after any resizing/area interpolation during caching). Because downsampling uses area interpolation, edge pixels often have intermediate values.

**Recommended threshold values:**
- `0.05`: Very aggressive - only near-pure-black regions get prior preservation
- `0.10`: Recommended for weighted masks - preserves background, body/hair train normally
- `0.15`: More conservative - includes soft mask edges in prior preservation

### Threshold Mode Behavior

When `--prior_mask_threshold` is set, the implementation **mutually excludes** target and prior regions to prevent overlap:

```python
# In threshold mode, zero out target where prior applies
prior_mask = (mask_raw < threshold).float()  # Binary: 1 where raw < threshold
mask_processed = mask_processed * (1 - prior_mask)  # Remove prior regions from target
```

This ensures each pixel is either learning from target OR being preserved, never both simultaneously.

### Mode Summary

| Mode | `--prior_mask_threshold` | Behavior |
|------|--------------------------|----------|
| **Continuous** | Not set (default) | **Soft mixing** - each pixel has weighted contributions from both target and prior based on mask value |
| **Threshold** | Set to 0.05-0.15 | **Hard background preservation** - binary split: pixels below threshold get full prior, above get full target |

**When to use each:**
- **Continuous**: When you want smooth transitions and partial preservation of body/clothing regions
- **Threshold**: When you want sharp subject vs background distinction with full learning on subject

---

## 11. Success Criteria

1. **Functional:** Prior preservation reduces phantom limb artifacts in character LoRA training
2. **Performance:** <1.5x slowdown when prior preservation enabled (with skip optimization)
3. **Compatibility:** Existing configs work unchanged (prior_preservation_weight=0 is default)
4. **Quality:** Generated images show cleaner backgrounds when using prior preservation
5. **Documentation:** Users can enable feature with clear argument additions

---

## 12. Appendix: OneTrainer Reference Code

### masked_loss.py (simplified)

```python
def masked_losses_with_prior(
        losses: Tensor,
        prior_losses: Tensor | None,
        mask: Tensor,
        unmasked_weight: float,
        normalize_masked_area_loss: bool,
        masked_prior_preservation_weight: float,
) -> Tensor:
    clamped_mask = torch.clamp(mask, unmasked_weight, 1)
    losses *= clamped_mask

    if normalize_masked_area_loss:
        losses = losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    if masked_prior_preservation_weight == 0 or prior_losses is None:
        return losses

    clamped_mask = (1 - clamped_mask)
    prior_losses *= clamped_mask * masked_prior_preservation_weight

    if normalize_masked_area_loss:
        prior_losses = prior_losses / clamped_mask.mean(dim=(1, 2, 3), keepdim=True)

    return losses + prior_losses
```

### GenericTrainer.py (simplified)

```python
if len(prior_pred_indices) > 0 \
        or (self.config.masked_training
            and self.config.masked_prior_preservation_weight > 0
            and self.config.training_method == TrainingMethod.LORA):
    with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
        prior_model_output_data = self.model_setup.predict(...)
    model_output_data = self.model_setup.predict(...)
    prior_model_prediction = prior_model_output_data['predicted'].to(dtype=model_output_data['target'].dtype)
    model_output_data['prior_target'] = prior_model_prediction
```
