# Network Configuration: `network_args` Reference

**For musubi-tuner / blissful-tuner LoRA training**

This document covers all `network_args` options available when configuring LoRA networks.

---

## How `network_args` Works

This repo passes `--network_args` as key/value strings (e.g., `use_dora=True`). In training scripts, each entry is split on `=` and forwarded into the network module's `create_arch_network(..., **net_kwargs)` / `create_network(..., **kwargs)`.

**Example:**
```toml
[network]
network_module = "networks.lora_wan"
network_dim = 128
network_alpha = 64
network_args = ["use_rslora=True", "use_dora=True", "loraplus_lr_ratio=8"]
```

**Command-line equivalent:**
```bash
--network_args "use_rslora=True" "use_dora=True" "loraplus_lr_ratio=8"
```

---

## Table of Contents

1. [RS-LoRA](#rs-lora-use_rslora)
2. [DoRA](#dora-use_dora)
3. [LoRA+](#lora-loraplus_lr_ratio)
4. [Dropout Options](#dropout-options)
5. [Conv-Specific Settings](#conv-specific-settings)
6. [Module Selection Patterns](#module-selection-patterns)
7. [Architecture-Specific Options](#architecture-specific-options)
8. [Utility Options](#utility-options)
9. [Quick Reference Table](#quick-reference-table)
10. [Known-Good Combinations](#known-good-combinations)

---

## RS-LoRA (`use_rslora`)

**Option:** `use_rslora=True|False` (default: `False`)

### What it does

Changes LoRA scaling from standard to rank-stabilized:

| Mode | Scaling Formula | alpha=0 Behavior |
|------|-----------------|------------------|
| Standard LoRA | `scale = alpha / r` | scale = 1.0 |
| RS-LoRA | `scale = alpha / sqrt(r)` | scale = 1.0 |

### When to use

- Training with **higher ranks** (dim > 32) where gradient magnitudes vary significantly
- When comparing runs with **different `network_dim`** values (RS-LoRA normalizes the effect)
- When you want **more stable training dynamics** across layers

### Technical notes / gotchas

| Topic | Detail |
|-------|--------|
| **Weight compatibility** | RS-LoRA vs standard LoRA is a semantic difference. Loading weights with the wrong setting produces incorrect scaling. The implementation enforces this with a **hard error** on mismatch. |
| **alpha=0 persistence** | When `alpha=0` under RS-LoRA, the saved alpha becomes `sqrt(r)` so that `alpha/sqrt(r)=1`. External tools that ignore `use_rslora_flag` and assume `alpha/r` will misinterpret these weights. |
| **Regularization** | `scale_weight_norms` (max-norm regularization) uses RS-LoRA scaling when enabled. |
| **Flag storage** | Weights include a network-level `use_rslora_flag` buffer for unambiguous detection on load. |
| **Suspicious alpha hint** | If loading fails due to flag mismatch and >50% of alphas equal `sqrt(dim)`, the error message hints that RS-LoRA may have been used. |

### Example

```toml
network_args = ["use_rslora=True"]
```

**Scaling comparison (dim=128, alpha=64):**
- Standard: `64/128 = 0.5`
- RS-LoRA: `64/sqrt(128) = 64/11.31 ≈ 5.66`

---

## DoRA (`use_dora`)

**Option:** `use_dora=True|False` (default: `False`)

### What it does

Enables **DoRA (Weight-Decomposed Low-Rank Adaptation)**:

- Adds a **per-layer magnitude vector** (`dora_layer.weight`)
- Applies magnitude normalization using `||W + ΔW||` (row-wise L2 norm, detached)
- Uses **PEFT-style bias handling** (bias not scaled by the magnitude term)

**DoRA formula:**
```
weight_norm = ||W + scaling * (B @ A)||_row  (detached)
mag_norm_scale = magnitude / weight_norm
delta = (mag_norm_scale - 1) * base_wo_bias + mag_norm_scale * lora_out * scaling
```

### Supported layers / limitations

| Constraint | Behavior |
|------------|----------|
| **Linear layers only** | DoRA enabled only for `torch.nn.Linear` |
| **Conv layers** | DoRA disabled (all Conv types, including Conv1x1) |
| **split_dims** | DoRA disabled when qkv-style ModuleLists are used |
| **dropout > 0** | DoRA disabled for that module |
| **rank_dropout > 0** | DoRA disabled for that module |
| **module_dropout** | Allowed with DoRA |

When DoRA is disabled for a module, it **silently falls back to standard LoRA**. A summary is logged at network creation:
```
DoRA enabled on 45 modules, disabled on: 12 non-Linear, 3 dropout
```

### When to use

- When standard LoRA produces **inconsistent quality** across layers
- For **fine-grained control** over weight magnitudes
- Training **style or identity LoRAs** where magnitude decomposition helps

### Technical notes / gotchas

| Topic | Detail |
|-------|--------|
| **Saved weights** | Includes network-level `use_dora_flag` and per-module `dora_layer.weight` tensors. |
| **Fallback detection** | For older/external weights without the flag, DoRA is detected by scanning for `dora_layer.weight` keys. |
| **Merging** | Magnitude is read from the weights file. If missing for a DoRA-enabled module, this is treated as an error (prevents silent “all-ones” magnitudes). |
| **Mismatch: network expects DoRA, weights lack it** | **Hard error** (to avoid uninitialized magnitudes). |
| **Mismatch: network doesn't expect DoRA, weights have it** | **Warning** (treats as standard LoRA; DoRA magnitudes ignored; `use_dora_flag` is not allowed to flip to `True`). |
| **Memory-efficient norm** | Uses expanded-norm formula without materializing B@A during forward pass. |
| **`lora_multiplier=0` semantics** | Treated as a true no-op (DoRA included): base output/weights are unchanged. |

### Example

```toml
network_args = ["use_dora=True"]
```

---

## LoRA+ (`loraplus_lr_ratio`)

**Option:** `loraplus_lr_ratio=<float>` (default: unset/disabled)

### What it does

Applies a **learning-rate multiplier** to LoRA-B (up) parameters during optimizer param group creation:

| Parameter Group | Learning Rate |
|-----------------|---------------|
| LoRA-A (down) | `learning_rate` |
| LoRA-B (up) | `learning_rate * loraplus_lr_ratio` |

### When to use

- When you want **faster convergence** without changing other hyperparameters
- Typical ratios: **4-16** (original paper recommends 16, but 4-8 is often more stable)

### Technical notes

- Implemented via `LoRANetwork.set_loraplus_lr_ratio()` and applied in `prepare_optimizer_params()`
- Only affects LoRA parameters, not text encoder or other trainable params

### Example

```toml
[network]
network_args = ["loraplus_lr_ratio=8"]

[optimizer]
learning_rate = 5e-5
# Effective: LoRA-A = 5e-5, LoRA-B = 4e-4
```

---

## Dropout Options

### `rank_dropout`

**Option:** `rank_dropout=<float>` (default: None/disabled)

Applies dropout to the **rank dimension** during training. Randomly zeros out entire rank slices.

**DoRA interaction:** If `rank_dropout > 0`, DoRA is **disabled** for that module.

```toml
network_args = ["rank_dropout=0.1"]  # 10% rank dropout
```

### `module_dropout`

**Option:** `module_dropout=<float>` (default: None/disabled)

Applies dropout to the **entire LoRA module output** during training. With probability `module_dropout`, the LoRA contribution is zeroed (only base model output used).

**DoRA interaction:** `module_dropout` is **allowed** with DoRA.

```toml
network_args = ["module_dropout=0.1"]  # 10% module dropout
```

### Dropout + DoRA compatibility

| Dropout Type | DoRA Compatible |
|--------------|-----------------|
| `dropout` (neuron) | ❌ DoRA disabled |
| `rank_dropout` | ❌ DoRA disabled |
| `module_dropout` | ✅ DoRA allowed |

**Note:** `dropout=0.0` and `rank_dropout=0.0` are treated as disabled (DoRA remains enabled).

---

## Conv-Specific Settings

### `conv_dim`

**Option:** `conv_dim=<int>` (default: same as `network_dim`)

Sets a **separate rank** for Conv2d layers. Useful when you want different capacity for convolutions vs linear layers.

```toml
network_args = ["conv_dim=32"]  # Conv layers use rank 32
```

### `conv_alpha`

**Option:** `conv_alpha=<float>` (default: same as `network_alpha`)

Sets a **separate alpha** for Conv2d layers.

```toml
network_args = ["conv_dim=32", "conv_alpha=16"]
```

---

## Module Selection Patterns

Control which submodules receive LoRA adapters using regex patterns.

### `exclude_patterns`

**Option:** `exclude_patterns=<python-literal-list-of-regex>`

A list of regex patterns matched against the module's original dotted name (before `.` → `_` conversion). If a module matches an exclude pattern, it is **skipped** unless it also matches an include pattern.

**Example:**
```toml
network_args = ["exclude_patterns=['.*(img_mod|txt_mod|modulation).*']"]
```

### `include_patterns`

**Option:** `include_patterns=<python-literal-list-of-regex>`

A list of regex patterns that **force inclusion** even if excluded (acts as an override).

**Example - Only train attention Q/K/V:**
```toml
network_args = [
  "exclude_patterns=['.*attn.*']",
  "include_patterns=['.*attn\\.to_q.*', '.*attn\\.to_k.*', '.*attn\\.to_v.*']",
]
```

### Pattern application order

1. Check `exclude_patterns` → if matches, mark for exclusion
2. Check `include_patterns` → if matches, override exclusion
3. Apply default exclusions (architecture-specific)

### Default exclusions

By default, **modulation layers** are excluded:
```
.*(img_mod|txt_mod|modulation).*
```

---

## Architecture-Specific Options

### Qwen-Image: `exclude_mod`

**Option:** `exclude_mod=True|False` (default: `True`)

**Module:** `networks.lora_qwen_image`

Controls whether modulation layers (`img_mod`, `txt_mod`) are excluded from LoRA.

| Value | Behavior |
|-------|----------|
| `True` (default) | Modulation layers excluded (standard for style/concept LoRAs) |
| `False` | Modulation layers included (recommended for **persona/identity** LoRAs) |

**Example - Include modulation for identity training:**
```toml
[network]
network_module = "networks.lora_qwen_image"
network_args = ["exclude_mod=False"]
```

---

## Utility Options

### `verbose`

**Option:** `verbose=True|False` (default: `False`)

Prints detailed information about the LoRA network during creation, including:
- All modules that receive LoRA adapters
- Modules that were excluded and why
- Parameter counts per module

```toml
network_args = ["verbose=True"]
```

---

## Quick Reference Table

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `use_rslora` | bool | `False` | RS-LoRA scaling (`alpha/sqrt(r)`) |
| `use_dora` | bool | `False` | DoRA magnitude decomposition (Linear only) |
| `loraplus_lr_ratio` | float | None | LoRA-B learning rate multiplier |
| `rank_dropout` | float | None | Dropout on rank dimension (disables DoRA) |
| `module_dropout` | float | None | Dropout on entire module (DoRA OK) |
| `conv_dim` | int | `network_dim` | Separate rank for Conv2d layers |
| `conv_alpha` | float | `network_alpha` | Separate alpha for Conv2d layers |
| `exclude_patterns` | list | `[]` | Regex patterns to exclude modules |
| `include_patterns` | list | `[]` | Regex patterns to force-include modules |
| `exclude_mod` | bool | `True` | Exclude modulation layers (Qwen-Image only) |
| `verbose` | bool | `False` | Print detailed network info |

---

## Known-Good Combinations

### RS-LoRA only
```toml
network_args = ["use_rslora=True"]
```
Best for: Higher-rank training (dim > 32), cross-rank comparisons.

### DoRA only
```toml
network_args = ["use_dora=True"]
```
Best for: Linear-layer focused training without dropout.

### RS-LoRA + DoRA
```toml
network_args = ["use_rslora=True", "use_dora=True"]
```
Best for: Maximum expressiveness with stable scaling.

### LoRA+ + RS-LoRA
```toml
network_args = ["use_rslora=True", "loraplus_lr_ratio=8"]
```
Best for: Faster convergence with stable high-rank training.

### Full combo (RS-LoRA + DoRA + LoRA+)
```toml
network_args = ["use_rslora=True", "use_dora=True", "loraplus_lr_ratio=8"]
```
Best for: High-quality persona/identity LoRAs with faster training.

### Persona/Identity (Qwen-Image)
```toml
network_args = ["use_rslora=True", "loraplus_lr_ratio=8", "exclude_mod=False"]
```
Best for: Character/person LoRAs where modulation layers help capture identity.

### Style training with module dropout
```toml
network_args = ["use_dora=True", "module_dropout=0.1"]
```
Best for: Style LoRAs with regularization (module_dropout is DoRA-compatible).

### Selective layer training
```toml
network_args = [
  "exclude_patterns=['.*single_blocks.*']",
  "use_rslora=True",
]
```
Best for: Training only double blocks (or vice versa).

---

## Troubleshooting

| Error/Warning | Cause | Solution |
|---------------|-------|----------|
| `RS-LoRA flag mismatch` (hard error) | Loading RS-LoRA weights without `use_rslora=True` or vice versa | Match `use_rslora` to how weights were trained |
| `DoRA flag mismatch: network expects DoRA but weights lack it` (hard error) | Weights don't contain DoRA magnitudes | Remove `use_dora=True` or use DoRA-trained weights |
| `DoRA flag mismatch: weights contain DoRA but network doesn't expect it` (warning) | Ignoring DoRA magnitudes | Add `use_dora=True` if you want DoRA behavior |
| `DoRA magnitude appears uninitialized` (warning) | Called `get_weight()` before loading weights | Ensure `load_state_dict()` is called first |
| `DoRA disabled for X modules` (info) | Conv layers, dropout, or split_dims | Expected behavior; those modules use standard LoRA |

---

## Changelog

### 2026-01-16
- Added RS-LoRA (`use_rslora`) with flag mismatch handling and suspicious alpha hints
- Added DoRA (`use_dora`) with Linear-only constraints and dropout interaction
- Added dropout options with DoRA compatibility notes
- Added Conv-specific settings (`conv_dim`, `conv_alpha`)
- Added architecture-specific options (`exclude_mod` for Qwen-Image)
- Added troubleshooting section

---

*Document created: 2026-01-16*
*For blissful-tuner LoRA training*
