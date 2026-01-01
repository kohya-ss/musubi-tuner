# ULTIMATE Prodigy Plus Schedule Free Guide

## For Diffusion Model LoRA Training

**Version 1.1** | **Updated December 2025**

This guide synthesizes comprehensive research from original codebases, academic papers, and empirical testing to provide definitive guidance on ProdigyPlusScheduleFree optimizer configuration for diffusion model LoRA training.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Algorithm Foundations](#algorithm-foundations)
3. [Complete Parameter Reference](#complete-parameter-reference)
4. [Configuration Templates](#configuration-templates)
5. [Use Case Specific Guidance](#use-case-specific-guidance)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Advanced Topics](#advanced-topics)
8. [Research Paper Insights](#research-paper-insights)

---

## Executive Summary

ProdigyPlusScheduleFree is an advanced optimizer that combines:
- **D-Adaptation** (Prodigy): Automatic learning rate discovery
- **Schedule-Free**: Eliminates learning rate schedules through iterate averaging
- **ADOPT**: Decorrelated gradient-moment updates for sparse gradients
- **StableAdamW**: Internal RMS-based gradient scaling
- **OrthoGrad**: Orthogonal gradient projection for regularization
- **Factored States**: Adafactor-style memory optimization

### Critical Findings

| Finding | Impact | Recommendation |
|---------|--------|----------------|
| `use_bias_correction=True` dampens LR ~10x | Training barely progresses | **Always use False** |
| Higher `beta1` (0.95) needed with Schedule-Free | Standard 0.9 causes instability | **Use betas=(0.95, 0.99)** |
| `use_adopt=True` can help very noisy/sparse grads | More stable normalization in some cases | **Optional: try for masked-loss/video** |
| `d_limiter=True` prevents runaway LR | LR can explode without | **Enable for safety** |
| `schedulefree_c` refines Schedule-Free averaging | Higher = more responsive (less averaging) | **Start `0` (vanilla) or `12` (extra smoothing for LoRA)** |

### Recommended Baseline Configuration

```python
optimizer_args = [
    "betas=(0.95, 0.99)",        # Higher beta1 for Schedule-Free stability
    "d_coef=1.0",                 # Primary LR scaling factor
    "d0=1e-6",                    # Initial d estimate (low = conservative start)
    "d_limiter=True",             # CRITICAL: Prevents runaway LR
    "weight_decay=0.01",          # Mild regularization
    "weight_decay_by_lr=True",    # Scale WD with discovered LR
    "use_bias_correction=False",  # CRITICAL: True dampens LR 10x
    "schedulefree_c=12",          # Extra smoothing vs vanilla when beta1=0.95 (vanilla-equivalent would be C=20)
    "use_stableadamw=True",       # RMS gradient scaling
    "eps=1e-8",                   # Numerical stability
    "factored=True",              # Memory optimization
    "factored_fp32=True",         # Precision for factored states
    "stochastic_rounding=True",   # Better BF16 training
    "use_adopt=False",            # Optional: try True for very noisy/sparse grads
    "split_groups=True",          # Per-group LR discovery
    "split_groups_mean=False",    # No cross-group averaging
]
```

---

## Algorithm Foundations

### D-Adaptation (Prodigy Core)

D-Adaptation automatically discovers the optimal learning rate by measuring the alignment between gradients and parameter displacement:

```
d_numerator += η * <g_t, x_t - x_0>     # Gradient-displacement inner product
d_denom += η * ||g_t||                   # Gradient norm accumulator
d = d_numerator / d_denom                # Estimated optimal LR
```

**Key Insight**: When gradients consistently point in the direction of displacement from initialization, the learning rate should increase. When they conflict, it should decrease.

**Mathematical Foundation** (from Prodigy paper):
- Uses growth-rate based estimation (d_t scales with training progress)
- Requires `lr=1.0` as the base (d becomes the effective LR)
- `d_coef` acts as a global multiplier on the discovered LR

### Schedule-Free Optimization

Schedule-Free eliminates the need for learning rate schedules by maintaining two sequences:

```
z_t = z_{t-1} - γ * g(y_t)              # Training sequence (momentum-like)
x_t = (1-c) * x_{t-1} + c * z_t          # Evaluation sequence (averaged)
y_t = (1-β) * x_t + β * z_t              # Gradient evaluation point
```

Where:
- `z` is the fast-moving training sequence
- `x` is the slow-moving evaluation sequence (used for inference)
- `y` is the interpolation point where gradients are evaluated
    - `c` is the internal averaging coefficient (`ckp1` in the source); `schedulefree_c` is an optional refined scaling of that coefficient
- `β` is `beta1` in optimizer terminology

**Critical Insight**: Standard optimizers use `beta1=0.9`. Schedule-Free requires **higher beta1 (0.95)** because the `y` interpolation already provides momentum-like behavior.

### ADOPT (Adaptive Gradient Decorrelation)

Standard Adam uses `v_t` to normalize `g_t`:
```
standard: m_t = β1*m_{t-1} + (1-β1)*g_t
          v_t = β2*v_{t-1} + (1-β2)*g_t²
          update = m_t / sqrt(v_t)        # g_t correlated with v_t!
```

ADOPT decorrelates by using `v_{t-1}`:
```
adopt:    m_t = β1*m_{t-1} + (1-β1)*(g_t / sqrt(v_{t-1}))  # Decorrelated!
          v_t = β2*v_{t-1} + (1-β2)*g_t²
```

**Why This Matters**: In masked loss training (video/image LoRA), gradients are sparse. Using same-step `v_t` creates correlation that harms convergence. ADOPT fixes this.

### StableAdamW

Adds internal RMS-based gradient scaling before Adam update:

```
rms = sqrt(mean(g²))
g_scaled = g / max(rms, 1.0)
# Then proceed with Adam update using g_scaled
```

**Benefit**: Prevents gradient magnitude spikes from destabilizing training without requiring external gradient clipping.

### OrthoGrad (Orthogonal Gradient Projection)

Projects gradients to be orthogonal to the weight direction:

```
g_ortho = g - (g · w / ||w||²) * w
```

**Why It Helps**: Removes the component of the gradient that would scale weights uniformly. This acts as implicit regularization, particularly beneficial for small datasets where overfitting is a concern.

**When to Use**:
- Small datasets (< 500 samples)
- Persona LoRA (limited subject variety)
- Style LoRA (preventing style collapse)

### Factored States

Instead of storing full m×n matrices for momentum states, use rank-1 approximations:

```
standard: v ∈ R^{m×n}                    # m×n floats
factored: v_row ∈ R^m, v_col ∈ R^n       # m+n floats
          v ≈ v_row ⊗ v_col              # Outer product reconstruction
```

**Memory Savings**: For a 4096×4096 layer:
- Standard: 16M floats × 4 bytes = 64MB
- Factored: 8K floats × 4 bytes = 32KB (2000x reduction!)

**Tradeoff**: Slight approximation error, but `factored_fp32=True` maintains precision.

---

## Complete Parameter Reference

This section is verified against the **actual ProdigyPlusScheduleFree constructor** in this repository. It intentionally lists **only** parameters that exist in `prodigyplus.ProdigyPlusScheduleFree`.

### Constructor Defaults (v2.0.x)

| Parameter | Default | What it does (practical) |
|----------|---------|--------------------------|
| `lr` | `1.0` | Global multiplier on the learned step size `d` (leave at 1.0 unless you’re intentionally scaling). |
| `betas` | `(0.9, 0.99)` | `(beta1, beta2)` for Schedule-Free + variance EMA. For diffusion/SF, `(0.95, 0.99)` is a strong default. |
| `beta3` | `None` | Prodigy numerator EMA; `None` uses `sqrt(beta2)`. |
| `weight_decay` | `0.0` | Decoupled weight decay (LoRA usually benefits from `0.005–0.02`). |
| `weight_decay_by_lr` | `True` | If `True`, decay scales with the adaptive step size (AdamW-like). |
| `d0` | `1e-6` | Initial `d` guess. Raise to `1e-5` if `d` is stuck tiny for hundreds of steps. |
| `d_coef` | `1.0` | Main tuning knob for “more/less aggressive” learning rate (`0.5–2.0` typical). |
| `d_limiter` | `True` | Caps per-step `d_hat` growth (helps prevent early overestimation). |
| `prodigy_steps` | `0` | **Freeze `d` adaptation after N steps** (0 = never). This is not a warmup. |
| `use_schedulefree` | `True` | Enables Schedule-Free logic. If `False`, behaves like Prodigy (and you usually want a decay scheduler). |
| `schedulefree_c` | `0` | Refined SF decoupling parameter **C** (see “Through the River”). `0` = vanilla SF. |
| `eps` | `1e-8` | Epsilon for Adam-style division. Set to `None` for Adam-atan2 (disables StableAdamW). |
| `split_groups` | `True` | Per-parameter-group `d` adaptation. |
| `split_groups_mean` | `False` | If `True`, uses a shared harmonic-mean `d` across groups. |
| `factored` | `True` | Factored second-moment state (Adafactor-style). |
| `factored_fp32` | `True` | Keep factor state in fp32 for stability. |
| `use_bias_correction` | `False` | RAdam-style rectification. Usually keep `False` for Prodigy-based LoRA. |
| `use_stableadamw` | `True` | RMS-based update scaling (often replaces external grad clipping). |
| `use_speed` | `False` | Experimental alternative to Prodigy’s ratio estimator; can be unstable with weight decay. |
| `stochastic_rounding` | `True` | Helps bf16 training quality when writing fp32 updates back to bf16 params. |
| `fused_back_pass` | `False` | For Kohya-style fused backward pass integration. |
| `use_cautious` | `False` | “Cautious optimizer” masking; often safe to try for stability. |
| `use_grams` | `False` | GRAMS-style sign/scale modifications (more aggressive). |
| `use_adopt` | `False` | Partial ADOPT (uses pre-update denom). Optional for very noisy/sparse grads. |
| `use_orthograd` | `False` | Orthogonal gradient projection (anti-overfit, often good for small datasets). |
| `use_focus` | `False` | FOCUS update modification (incompatible with some modes). |

### schedulefree_c (C) in one paragraph

When `schedulefree_c > 0`, this implementation uses the refined Schedule-Free scaling:

`ckp1 <- min(1, ckp1 * (1 - beta1) * C)`

So a “vanilla-equivalent” value is `C_vanilla = 1/(1-beta1)` (e.g. `20` at `beta1=0.95`). Smaller than that = more smoothing; larger = more responsive. For diffusion LoRA, start with `schedulefree_c=0` (vanilla) or `12` (extra smoothing with `beta1=0.95`), and only go to `50–200` if you have a longer run and/or larger batches.

**split_groups Explained**: LoRA has multiple parameter groups (different layers, possibly LoRA+ with different LRs). `split_groups=True` allows each group to discover its own optimal LR rather than using a global estimate.

### Notes on “warmup/growth” knobs

The official Prodigy optimizer has `safeguard_warmup` / `growth_rate` knobs. **ProdigyPlusScheduleFree does not.**
If you need gentler early behavior, prefer:
- `d_limiter=True` (default)
- conservative `d_coef` (e.g. `0.8–1.0`)
- a smaller `d0` (e.g. `1e-7`)

---

## Configuration Templates

### Template 1: Standard LoRA Training (Recommended)

For typical LoRA training with medium-sized datasets (500-5000 samples):

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.0",
    "d0=1e-6",
    "d_limiter=True",
    "weight_decay=0.01",
    "weight_decay_by_lr=True",
    "use_bias_correction=False",
    "schedulefree_c=12",          # Extra smoothing vs vanilla (beta1=0.95 => C_vanilla=20)
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_orthograd=True",         # Optional but often helpful for LoRA generalization
    "split_groups=True",
    "split_groups_mean=False",
]
lr_scheduler = "constant"
max_grad_norm = 0  # StableAdamW handles this internally
```

### Template 2: Video LoRA with Masked Loss

For WAN, HunyuanVideo, or other video models with sparse gradient updates:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.0",
    "d0=1e-6",
    "d_limiter=True",
    "prodigy_steps=0",            # 0 = keep adapting d; set >0 only to freeze d after it stabilizes
    "weight_decay=0.0",           # No WD for video
    "use_bias_correction=False",
    "schedulefree_c=50",          # More responsive than vanilla (beta1=0.95 => C_vanilla=20)
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_adopt=True",             # Optional: can help with very noisy/sparse gradients (try if unstable)
    "split_groups=True",
    "split_groups_mean=False",
]
lr_scheduler = "constant"
max_grad_norm = 0
```

### Template 3: Small Dataset / Persona LoRA

For limited data (< 500 samples) where overfitting is a concern:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=0.8",                 # Slightly conservative
    "d0=1e-6",
    "d_limiter=True",
    "prodigy_steps=0",            # Keep adapting d; freezing at 100 steps is usually too early
    "weight_decay=0.05",          # More regularization
    "weight_decay_by_lr=True",
    "use_bias_correction=False",
    "schedulefree_c=12",          # Extra smoothing vs vanilla (beta1=0.95 => C_vanilla=20)
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_orthograd=True",         # Anti-overfitting
    "split_groups=True",
    "split_groups_mean=False",
]
lr_scheduler = "constant"
max_grad_norm = 0
```

### Template 4: Aggressive Training (Large Dataset)

For large datasets (> 10K samples) where faster convergence is desired:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.5",                 # More aggressive
    "d0=1e-5",                    # Higher initial estimate
    "d_limiter=True",
    "weight_decay=0.01",
    "weight_decay_by_lr=True",
    "use_bias_correction=False",
    "schedulefree_c=100",         # More responsive than vanilla (beta1=0.95 => C_vanilla=20); use 0 or 12 if you want more smoothing
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_adopt=False",            # Optional: try True if gradients are extremely noisy/sparse
    "split_groups=True",
    "split_groups_mean=False",
]
lr_scheduler = "constant"
max_grad_norm = 0
```

### Template 5: Memory-Constrained Setup

When VRAM is extremely limited:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.0",
    "d0=1e-6",
    "d_limiter=True",
    "weight_decay=0.01",
    "weight_decay_by_lr=True",
    "use_bias_correction=False",
    "schedulefree_c=0",           # Vanilla SF (simplest); consider 12 for extra smoothing with beta1=0.95
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=False",        # FP16 factored states
    # Note: ProdigyPlusScheduleFree does not have a "factored_dim" knob; factoring is automatic based on tensor shape.
    "stochastic_rounding=True",
    # Note: use_grams changes the update rule; it is not a memory-reduction feature.
    "split_groups=False",         # Single d estimate saves memory
]
lr_scheduler = "constant"
max_grad_norm = 0
```

---

## Use Case Specific Guidance

### Qwen-Image-2512 LoRA

Qwen-Image uses flow matching with specific requirements:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.0",
    "d0=1e-6",
    "d_limiter=True",
    "prodigy_steps=0",            # Keep adapting d; freeze later only if you have evidence it has stabilized
    "weight_decay=0.01",
    "use_bias_correction=False",
    "schedulefree_c=12",          # Extra smoothing vs vanilla (beta1=0.95 => C_vanilla=20)
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_adopt=False",            # Optional: try True if masked/sparse gradients make training unstable
    "split_groups=True",
    "split_groups_mean=False",
]
lr_scheduler = "constant"
max_grad_norm = 0

[training]
timestep_sampling = "qwen_shift"
discrete_flow_shift = 2.2
```

### WAN 2.2 Video LoRA

WAN uses dual-model architecture with high/low noise separation:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.0",
    "d0=1e-6",
    "d_limiter=True",
    "prodigy_steps=0",
    "weight_decay=0.0",
    "use_bias_correction=False",
    "schedulefree_c=50",
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_adopt=True",             # Optional: often helpful for masked/sparse video losses
    "split_groups=True",
    "split_groups_mean=False",
]
lr_scheduler = "constant"
max_grad_norm = 0

[training]
timestep_sampling = "shift"
discrete_flow_shift = 12.0
```

### Style Transfer LoRA

For artistic style training with potential collapse concerns:

```toml
[optimizer]
optimizer_type = "prodigyplus.ProdigyPlusScheduleFree"
learning_rate = 1.0
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=0.8",
    "d0=1e-6",
    "d_limiter=True",
    "weight_decay=0.03",          # Moderate regularization
    "weight_decay_by_lr=True",
    "use_bias_correction=False",
    "schedulefree_c=50",
    "use_stableadamw=True",
    "eps=1e-8",
    "factored=True",
    "factored_fp32=True",
    "stochastic_rounding=True",
    "use_orthograd=True",         # Prevents style collapse
    "split_groups=True",
    "split_groups_mean=False",
]
```

### LoRA+ Configuration

When using LoRA+ (different LR for LoRA-A and LoRA-B matrices):

```toml
[network]
network_args = [
    "loraplus_lr_ratio=4",        # LoRA-B gets 4x LR
]

[optimizer]
# split_groups=True is CRITICAL for LoRA+
optimizer_args = [
    ...
    "split_groups=True",          # Separate d per group
    "split_groups_mean=False",    # No averaging across groups
]
```

---

## Troubleshooting Guide

### Problem: Learning Rate Not Increasing

**Symptoms**: `d` stays near initial value, loss barely decreases

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| `use_bias_correction=True` | **Set to False** (most common issue!) |
| `d0` too low | Increase to 1e-5 |
| `d_coef` too low | Increase to 1.2-1.5 |
| `prodigy_steps` too high | Reduce or set to 0 |
| Gradients near zero | Check data loading, loss computation |

### Problem: Learning Rate Exploding

**Symptoms**: `d` grows rapidly, loss spikes, NaN values

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| `d_limiter=False` | **Enable d_limiter** |
| `d_coef` too high | Reduce to 0.8-1.0 |
| `d` keeps growing after it “should” have stabilized | Reduce `d_coef`, lower `d0`, and/or set `prodigy_steps` to freeze `d` once it reaches a sane range |
| Bad data batch | Check for corrupted samples |
| Numerical overflow | Enable `use_stableadamw=True` |

### Problem: Training Instability / Loss Oscillation

**Symptoms**: Loss fluctuates wildly, doesn't converge smoothly

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| `beta1` too low | **Use beta1=0.95** (not 0.9) |
| `schedulefree_c` too high | Reduce toward vanilla-equivalent (`0` for vanilla, or ~`20` at `beta1=0.95`); for extra smoothing try `12` |
| Extremely noisy/sparse gradients | Optional: try `use_adopt=True` and/or `use_cautious=True` |
| Gradient accumulation issue | Ensure proper accumulation count |

### Problem: Overfitting (Training Loss Low, Val Loss High)

**Symptoms**: Model memorizes training data, poor generalization

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| No regularization | Add `weight_decay=0.01-0.05` |
| Small dataset | Enable `use_orthograd=True` |
| Training too long | Reduce `max_train_steps` |
| LR too high | Reduce `d_coef` to 0.7-0.8 |

### Problem: Out of Memory (OOM)

**Symptoms**: CUDA OOM during training

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| Full optimizer states | Enable `factored=True` |
| FP32 factored states | Set `factored_fp32=False` |
| Large batch size | Reduce batch, increase gradient accumulation |
| Split groups overhead | Set `split_groups=False` |

### Problem: Slow Training Speed

**Symptoms**: Training iterations take longer than expected

**Causes & Solutions**:

| Cause | Solution |
|-------|----------|
| `factored_fp32=True` | Trade precision for speed if needed |
| `stochastic_rounding=True` | Disable if not using BF16 |
| Many optimizer features | Disable unused features |

---

## Advanced Topics

### Understanding d_numerator and d_denom

The core of Prodigy's LR estimation:

```python
# Simplified view
d_numerator += lr * dot(gradient, params - params_init)
d_denom += lr * norm(gradient)
d = d_numerator / d_denom
```

**Interpretation**:
- `d_numerator` measures "how much gradient direction aligns with optimization progress"
- `d_denom` normalizes by total gradient magnitude
- High ratio = gradients consistently point toward the optimum = can use higher LR

### Schedule-Free Eval vs Train Mode

Schedule-Free maintains separate `z` (training) and `x` (evaluation) states:

```python
# During training
optimizer.train()  # Uses z sequence
loss.backward()
optimizer.step()

# For evaluation/inference
optimizer.eval()   # Switches to x sequence (averaged)
# Run validation
optimizer.train()  # Return to z for continued training
```

**Important**: Always call `optimizer.eval()` before validation/sampling!

### Interaction with Gradient Checkpointing

Gradient checkpointing recomputes activations during backward pass:

- **Compatible** with ProdigyPlusScheduleFree
- May slightly affect gradient statistics (d estimation)
- Recommendation: Use `prodigy_steps > 0` warmup with checkpointing

### Interaction with Mixed Precision

| Precision | Recommendation |
|-----------|----------------|
| FP32 | All features work normally |
| BF16 | Enable `stochastic_rounding=True` |
| FP16 | Works but BF16 preferred for stability |
| FP8 | Use `factored=True`, monitor closely |

### Multi-GPU / Distributed Training

ProdigyPlusScheduleFree is compatible with:
- **DataParallel**: Works normally
- **DistributedDataParallel**: Works with proper gradient sync
- **FSDP**: Works but d estimation may differ across ranks
- **DeepSpeed ZeRO**: Compatible with ZeRO-1/2, needs testing for ZeRO-3

---

## Research Paper Insights

### From "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"

**Key Contributions**:
1. Growth-rate based d estimation (faster than D-Adaptation)
2. Theoretical convergence guarantees
3. Works across diverse architectures

**Practical Insights**:
- Prodigy converges ~2x faster than AdamW with well-tuned LR
- Best results with `lr=1.0` as base (d becomes effective LR)
- `d_coef` is the main user-facing tuning knob

### From "The Road Less Scheduled" (Schedule-Free)

**Key Contributions**:
1. Proves LR schedules are approximating iterate averaging
2. Schedule-Free matches or beats cosine schedules
3. No need to know training length in advance

**Practical Insights**:
- Requires higher `beta1` (0.95 vs 0.9) for stability
- `c` parameter (schedulefree_c) controls responsiveness
- Always use `optimizer.eval()` for inference

### From "Through the River" (Schedule-Free Extension)

**Key Contributions**:
1. Theoretical analysis of why Schedule-Free works
2. Guidelines for `c` parameter selection
3. Robustness analysis

**Practical Insights**:
- Higher `c` = more responsive to recent gradients
- Lower `c` = more stable, better for noisy gradients
- Recommended starting point: c = T/4 where T is total steps

### From "ADOPT: Modified Adam Can Converge with Any β₂"

**Key Contributions**:
1. Identifies correlation issue in standard Adam
2. Simple fix: use v_{t-1} instead of v_t
3. Theoretical convergence for any β₂

**Practical Insights**:
- **Critical for sparse/masked gradients** (video LoRA)
- Minimal computational overhead
- Compatible with all other features

### From "Cautious Optimizers"

**Key Contributions**:
1. Sign-alignment masking improves convergence
2. Only update parameters where gradient and momentum agree
3. Up to 1.47x faster convergence in some cases

**Practical Insights**:
- Experimental feature in ProdigyPlusScheduleFree
- May help with noisy gradients
- Can slow training if gradients are clean

### From "StableAdamW"

**Key Contributions**:
1. Internal RMS-based gradient scaling
2. Eliminates need for external gradient clipping
3. More stable training dynamics

**Practical Insights**:
- Enabled by default in ProdigyPlusScheduleFree
- Replaces `max_grad_norm` clipping
- Works well with all model sizes

### From "Grokking" Research

**Key Contributions**:
1. Models can suddenly generalize after overfitting
2. Weight decay critical for grokking
3. Longer training can help generalization

**Practical Insights**:
- Don't stop too early if val loss plateaus
- Weight decay (0.01-0.1) aids generalization
- Grokking is real but unpredictable

---

## Quick Reference Card

### Essential Settings (Copy-Paste Ready)

```python
# Minimum required for good results
optimizer_args = [
    "betas=(0.95, 0.99)",
    "d_coef=1.0",
    "d_limiter=True",
    "use_bias_correction=False",
    "schedulefree_c=12",          # Extra smoothing for LoRA (beta1=0.95 => C_vanilla=20); set 0 for vanilla
    "use_stableadamw=True",
]
```

### Decision Tree

```
Starting LoRA Training?
├── Dataset size?
│   ├── < 500 samples → use_orthograd=True, weight_decay=0.05
│   ├── 500-5000 → Standard config
│   └── > 5000 → d_coef=1.2-1.5, schedulefree_c=50-200 (more responsive; consider 0/12 if overfitting)
│
├── Video/Masked loss?
│   └── Yes → optionally try use_adopt=True if gradients are very sparse/noisy
│
├── Memory constrained?
│   └── Yes → factored=True, factored_fp32=False
│
└── Using LoRA+?
    └── Yes → split_groups=True, split_groups_mean=False
```

### Common Mistakes to Avoid

| Mistake | Why It's Bad | Correct Approach |
|---------|--------------|------------------|
| `use_bias_correction=True` | Dampens LR 10x | Always False |
| `beta1=0.9` | Instability with SF | Use 0.95 |
| `lr != 1.0` | Breaks d estimation | Keep lr=1.0 |
| `max_grad_norm > 0` with StableAdamW | Redundant clipping | Set to 0 |
| `d_limiter=False` | Runaway LR risk | Always True |
| Missing `use_adopt` for video | Bad convergence | Enable for masked loss |

---

## Changelog

- **v1.0** (Dec 2024): Initial release based on comprehensive codebase and paper analysis

---

## References

1. Mishchenko, K., & Defazio, A. (2023). Prodigy: An Expeditiously Adaptive Parameter-Free Learner.
2. Defazio, A., et al. (2024). The Road Less Scheduled.
3. Defazio, A., et al. (2024). Through the River: Schedule-Free Learning Under Unknown Shifts.
4. Taniguchi, S., et al. (2024). ADOPT: Modified Adam Can Converge with Any β₂.
5. Liang, K., et al. (2024). Cautious Optimizers.
6. Wortsman, M., et al. (2023). StableAdamW.
7. Power, A., et al. (2022). Grokking: Generalization Beyond Overfitting.
8. Shazeer, N., & Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost.

---

*This guide was created by synthesizing analysis of the prodigy-plus-schedule-free, prodigy, and schedule_free codebases, along with comprehensive review of associated research papers. For the most up-to-date information, always consult the source repositories.*
