# Upstream Musubi-Tuner Integration Design (v3 - Final)

**Date**: 2026-01-21
**Version**: 3.1 (Final with all corrections)
**Status**: Ready for Implementation
**Author**: Claude (with Dustin)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-21 | Initial design |
| 2.0 | 2026-01-21 | Major corrections: fixed Phase 1 scope, updated estimates, added missing components |
| 3.0 | 2026-01-21 | Final corrections: wildcards in batch, argparse accuracy, LoHa limitations, logging strategy, docs, torch compat |
| 3.1 | 2026-01-21 | Item count consistency, wildcard syntax fix, flux2_shift inline approach, test fixes, control_count_per_image, fp8_m3 guard, float8 import safety |

---

## Executive Summary

This document outlines the integration plan for upstream musubi-tuner changes into blissful-tuner:

1. **Phase 1**: Qwen-Image Bug Fixes (4 items from commit 045eed5 + 1 wildcard propagation fix = **5 items**)
2. **Phase 2**: LoHa Network Module (with documented inference limitations)
3. **Phase 3**: FLUX.2 Architecture Support (~3,900 LOC)

**Total Scope**: ~20 files modified/added, **~4,200 lines of code**

---

## Current State Analysis

### Already Integrated

| Commit | Description | Verified Location |
|--------|-------------|-------------------|
| `b3edf9e` | varlen attention fix | `qwen_image_model.py` |
| `fc0e691` | img_shapes + remove_first_image | `qwen_image_train_network.py` |
| `d50909f` | z-image key mapping | `convert_lora.py` |
| `ab32f89` | Mu calculation fix | `qwen_image_generate_image.py:795` |
| `6bcf3e5` | Layered control load | `qwen_image_generate_image.py:1223` |

### Pending

| Commit | Description | Priority |
|--------|-------------|----------|
| `045eed5` | Batch generation fixes | Critical |
| `92ef4ee` | LoHa network module | Medium |
| `737f5a8` + `d5f1ca7` | FLUX.2 support | High |

---

## Phase 1: Qwen-Image Bug Fixes

### Scope: 5 Items (4 from 045eed5 + 1 wildcard fix)

| # | Issue | Source | Location | Status |
|---|-------|--------|----------|--------|
| 1.1 | `--only` prompt lines | 045eed5 | `parse_prompt_line()` :217 | Missing |
| 1.2 | **Wildcard propagation to batch** | Blissful-specific | `preprocess_prompts_for_batch()` :1141 | Missing |
| 1.3 | VL processor in `load_shared_models()` | 045eed5 | :1166 | Missing |
| 1.4 | VL processor in `process_batch_prompts()` | 045eed5 | :1205 | Missing |
| 1.5 | `latent[0]` bug in save_output | 045eed5 | :1325 | Missing |

**Note**: Interactive mode (:1373) already handles wildcards via the main generation path; no change needed there.

### 1.1 Prompt Lines Starting with `--`

**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Location**: Line 217

```python
# CURRENT
def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()
    if prompt_wildcards is not None:
        prompt = process_wildcards(prompt, prompt_wildcards)
    overrides = {"prompt": prompt}
    ...
    for part in parts[1:]:

# REQUIRED
def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    if line.strip().startswith("--"):  # No prompt, only options
        parts = (" " + line.strip()).split(" --")
        prompt = None
    else:
        parts = line.split(" --")
        prompt = parts[0].strip()
        if prompt_wildcards is not None:
            prompt = process_wildcards(prompt, prompt_wildcards)
        parts = parts[1:]  # Remove prompt from iteration

    overrides = {} if prompt is None else {"prompt": prompt}
    ...
    for part in parts:  # Now iterates options only
```

### 1.2 Wildcard Propagation to Batch Mode

**Problem**: `preprocess_prompts_for_batch()` calls `parse_prompt_line(line)` WITHOUT passing `prompt_wildcards`, so wildcards never work in `--from_file` mode.

**File**: `src/musubi_tuner/qwen_image_generate_image.py`
**Location**: Line 1141

```python
# CURRENT (line 1141)
prompt_data = parse_prompt_line(line)

# REQUIRED (note: function parameter is base_args, not args)
prompt_data = parse_prompt_line(line, prompt_wildcards=base_args.prompt_wildcards)
```

**Known limitation - Interactive mode**: Line 1373 also calls `parse_prompt_line(line)` without `prompt_wildcards`, so wildcards typed in interactive mode won't expand either. Options:
1. Document as limitation (simpler)
2. Add as optional follow-up fix: pass `args.prompt_wildcards` at :1373

**Consider**: Also apply wildcards to `negative_prompt` overrides if that feature is desired.

### 1.3 VL Processor in `load_shared_models()`

**Location**: Line 1166

```python
# CURRENT
if args.is_edit:
    vl_processor = qwen_image_utils.load_vl_processor()

# REQUIRED
if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None):
    vl_processor = qwen_image_utils.load_vl_processor()
```

### 1.4 VL Processor in `process_batch_prompts()`

**Location**: Line 1205

```python
# CURRENT
vl_processor_batch = qwen_image_utils.load_vl_processor() if args.is_edit else None

# REQUIRED
vl_processor_batch = (
    qwen_image_utils.load_vl_processor()
    if args.is_edit or (args.is_layered and args.automatic_prompt_lang_for_layered is not None)
    else None
)
```

### 1.5 Latent Shape Bug

**Location**: Line 1325

```python
# CURRENT
save_output(current_args, vae_for_batch, latent[0], device)

# REQUIRED
save_output(current_args, vae_for_batch, latent, device)
```

---

## Phase 2: LoHa Network Module

### 2.1 New File

**File**: `src/musubi_tuner/networks/loha.py` (~767 lines)

Copy from upstream. For logging, use one of these strategies:

#### Option A: Minimal Diff (Recommended)

Keep upstream logging as-is. Configure at entry points:

```python
# In script entry points (train scripts, not library modules)
import logging
from rich.logging import RichHandler

# Configure root logger to use Rich
logging.basicConfig(
    level=logging.INFO,
    handlers=[RichHandler(show_time=False, rich_tracebacks=True)]
)
```

**Note**: Upstream modules call `logging.basicConfig()` inside library code. When integrating:
- Either strip those calls from library modules, or
- Accept that first basicConfig wins (call yours first in entry points)

#### Option B: BlissfulLogger Replacement

Replace in loha.py only:
```python
# Replace these lines
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# With
from blissful_tuner.blissful_logger import BlissfulLogger
logger = BlissfulLogger(__name__, "green")
```

### 2.2 Training Script Changes

**File**: `src/musubi_tuner/hv_train_network.py`

Add `architecture=self.architecture` at 3 locations:

| Location | Line | Context |
|----------|------|---------|
| 1 | ~1793 | `create_arch_network_from_weights` in LoRA merging |
| 2 | ~1809 | `create_arch_network_from_weights` for dim_from_weights |
| 3 | ~1823 | `create_arch_network` for new network creation |

### 2.3 LoHa Inference/Merge Limitation (IMPORTANT)

**Problem**: Scripts that merge/apply LoRA weights **hardcode the lora module**:

| File | Import | Issue |
|------|--------|-------|
| `merge_lora.py:5` | `from musubi_tuner.networks import lora` | Hardcoded |
| `hv_generate_video.py:26` | `from musubi_tuner.networks import lora` | Hardcoded |
| `wan_generate_video.py:25` | `from musubi_tuner.networks import lora_wan` | Architecture-specific |

**The `architecture` parameter alone won't make LoHa work in these scripts** because they don't load the loha module at all.

#### Decision Required

**Option A: Document Limitation (Simpler)**
- LoHa only supported during training
- Inference requires merging LoHa into base model first
- Document this clearly in user-facing docs

**Option B: Add LoHa Detection (More Work)**

Add to `merge_lora.py` and generation scripts:

```python
def detect_network_type(weights_sd: dict) -> str:
    """Detect if weights are LoRA or LoHa based on key names."""
    for key in weights_sd.keys():
        if "hada_w1_a" in key or "hada_w2_a" in key:
            return "loha"
        if "lora_up" in key or "lora_down" in key:
            return "lora"
    return "lora"  # default

# Then load appropriate module
network_type = detect_network_type(weights_sd)
if network_type == "loha":
    from musubi_tuner.networks import loha as network_module
else:
    from musubi_tuner.networks import lora as network_module
```

**Recommendation**: Start with Option A (document limitation), add Option B later if users need it.

---

## Phase 3: FLUX.2 Integration

### 3.1 Scope

| Metric | Value |
|--------|-------|
| Lines of Code | ~3,911 |
| New Files | ~12 |
| Modified Files | ~6 |

### 3.2 New Files

#### Core Model Files
| File | Description |
|------|-------------|
| `src/musubi_tuner/flux_2/__init__.py` | Package init |
| `src/musubi_tuner/flux_2/flux2_models.py` | Transformer, VAE, blocks (~1,500 lines) |
| `src/musubi_tuner/flux_2/flux2_utils.py` | Loading, encoding, scheduling (~800 lines) |

#### Scripts
| File | Description |
|------|-------------|
| `src/musubi_tuner/flux_2_train_network.py` | Training loop |
| `src/musubi_tuner/flux_2_cache_latents.py` | Latent caching |
| `src/musubi_tuner/flux_2_cache_text_encoder_outputs.py` | Text encoder caching |
| `src/musubi_tuner/flux_2_generate_image.py` | Generation |

#### Network & Wrappers
| File | Description |
|------|-------------|
| `src/musubi_tuner/networks/lora_flux_2.py` | LoRA module |
| `flux_2_*.py` (root) | 4 thin wrapper scripts |

#### Documentation (NEW)
| File | Description |
|------|-------------|
| `docs/flux_2.md` | **Import from upstream** - user-facing documentation |

**Action**: Copy `~/musubi-tuner/docs/flux_2.md` and update README to mention FLUX.2 support.

### 3.3 Dataset Module Changes

**File**: `src/musubi_tuner/dataset/image_video_dataset.py`

```python
# Constants (use upstream values exactly)
ARCHITECTURE_FLUX_2 = "f2"
ARCHITECTURE_FLUX_2_FULL = "flux_2"
RESOLUTION_STEPS_FLUX_2 = 16  # NOT 32!

# Cache functions use ctx_vec key naming (not m3_vec)
def save_text_encoder_output_cache_flux_2(item_info: ItemInfo, ctx_vec: torch.Tensor):
    # Key: ctx_vec_{dtype}
    ...
```

### 3.4 Model Spec Metadata

**File**: `src/musubi_tuner/utils/sai_model_spec.py`

```python
# Add import
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FLUX_2

# Add constants (~line 76, 94)
ARCH_FLUX_2 = "Flux.2-dev"
IMPL_FLUX_2 = "https://github.com/black-forest-labs/flux2"

# Add mapping (~line 171)
elif architecture == ARCHITECTURE_FLUX_2:
    arch = ARCH_FLUX_2
    impl = IMPL_FLUX_2
```

### 3.5 Timestep Sampling: flux2_shift

**File**: `src/musubi_tuner/hv_train_network.py`

#### Argparse (Line 2861)

**Current choices**:
```python
choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "qwen_shift", "logsnr", "qinglong_flux", "qinglong_qwen"],
default="sigma",
```

**Required change**: Add `"flux2_shift"` to the choices list:
```python
choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "flux2_shift", "qwen_shift", "logsnr", "qinglong_flux", "qinglong_qwen"],
```

#### Sampling Logic

**Location 1** (~line 854-861): Add to the condition block
```python
or args.timestep_sampling == "flux2_shift"
```

**Location 2** (~line 861): Add case in the existing shift block

The upstream implementation uses `train_utils.get_lin_function()` with `h * w` (not `(h // 2) * (w // 2)` like flux_shift):

```python
# Context: inside the timestep sampling loop where h, w = latents.shape[-2:]
# Existing code (~line 859-864):
if args.timestep_sampling == "flux_shift":
    mu = train_utils.get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
elif args.timestep_sampling == "flux2_shift":  # ADD THIS CASE
    mu = train_utils.get_lin_function(y1=0.5, y2=1.15)(h * w)  # Note: h*w not (h//2)*(w//2)
elif args.timestep_sampling == "qwen_shift":
    mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=8192, y2=0.9)((h // 2) * (w // 2))
```

**Important**:
- Uses existing `train_utils.get_lin_function()` - no new imports needed
- FLUX.2 uses full `h * w` (not halved like flux_shift) because of different latent dimensions
- Do NOT use sqrt-based calculation - that was incorrect in earlier doc versions

### 3.6 Dependencies

**File**: `pyproject.toml`

```toml
# REQUIRED CHANGE
transformers = ">=4.56.1"  # Was >=4.46.0

# NOT REQUIRED for Flux.2 (uses own VAE, not diffusers)
# diffusers already at 0.32.1, no change needed
```

### 3.7 Dataset: control_count_per_image for FLUX.2 (NEW)

**File**: `src/musubi_tuner/dataset/image_video_dataset.py`
**Location**: ImageDataset `__init__` (~line 1817)

FLUX.2 allows multiple reference images per training sample. Add the architecture case:

```python
# Existing code (~line 1809-1820)
control_count_per_image: Optional[int] = 1
if self.architecture == ARCHITECTURE_FRAMEPACK or self.architecture == ARCHITECTURE_WAN:
    ...
elif self.architecture == ARCHITECTURE_FLUX_KONTEXT:
    control_count_per_image = 1
elif self.architecture == ARCHITECTURE_FLUX_2:  # ADD THIS
    control_count_per_image = None  # Can be multiple control/ref images
elif self.architecture == ARCHITECTURE_QWEN_IMAGE_EDIT:
    control_count_per_image = None
```

**Why**: Setting `None` allows arbitrary numbers of reference images, which is required for FLUX.2's flexible ref image support.

### 3.8 fp8_m3 Flag: NotImplementedError Guard (NEW)

**File**: `src/musubi_tuner/flux_2/flux2_utils.py`
**Location**: Line 627-628

**Problem**: The `--fp8_m3` flag exists but raises `NotImplementedError`:

```python
if is_fp8(dtype):
    logger.info(f"prepare Mistral 3 for fp8: set to {dtype}")
    raise NotImplementedError(f"Mistral 3 {dtype}")  # TODO
```

**Options**:
1. **Document limitation**: Note in `docs/flux_2.md` that `--fp8_m3` is not implemented
2. **Guard the flag**: Remove `--fp8_m3` from argparse or add early validation
3. **Keep as-is**: Let it crash with clear error message (current upstream behavior)

**Recommendation**: Option 1 - document that fp8 text encoder is not yet supported for FLUX.2.

### 3.9 torch.float8 Compatibility

**Risk**: Upstream uses `torch.float8e4m3fn` (no underscore) in multiple places:
- `flux_2_cache_text_encoder_outputs.py:70`
- `flux_2_generate_image.py:312,412,819,873`
- `flux_2_train_network.py:65`

Also uses `torch.float8_e4m3fn` (with underscore) in:
- `flux_2_generate_image.py:1076`
- `flux_2/flux2_utils.py:91`

**Potential Issue**: Some PyTorch builds may not expose the no-underscore alias.

**Mitigation Strategy**:

1. **Avoid import-time references**: Don't put float8 dtypes in module-level lists or dicts
2. **Runtime check**: Wrap usage in try/except or hasattr check
3. **Normalize**: During integration, replace all `float8e4m3fn` with `float8_e4m3fn`

```python
# Safe pattern - check at runtime, not import time
def get_float8_dtype():
    if hasattr(torch, 'float8_e4m3fn'):
        return torch.float8_e4m3fn
    raise RuntimeError("PyTorch build does not support float8 dtypes")
```

**Recommendation**: Normalize all to underscore version (`torch.float8_e4m3fn`) during integration.

### 3.10 Batch Size Limitation

**File**: `flux_2_train_network.py:287`

```python
assert bsize == 1, "Flux 2 can't be trained with higher batch size since ref images may different size and number"
```

**Impact**: When using reference images, batch_size MUST be 1.

**Action**: Document prominently in `docs/flux_2.md`.

### 3.11 Logging Strategy for Upstream Files

**Recommendation**: Keep `flux2_models.py` and `flux2_utils.py` as close to upstream as possible.

**Upstream issue to note**: These files call `logging.basicConfig()` inside library modules, which is hostile to embedders. Options:
1. Strip those calls during integration
2. Accept and document the behavior
3. Contribute fix upstream

For Blissful-specific scripts (`flux_2_train_network.py`, etc.), apply full BlissfulLogger treatment.

---

## Testing Strategy

### Phase 1 Tests

```bash
# Test 1.1 + 1.2: Batch with wildcards and --only lines
# Note: Blissful wildcards use __name__ syntax (not {name})
#
# IMPORTANT: --only lines (starting with --) inherit the CLI's --prompt value.
# They do NOT inherit from previous lines in the file.
# If no --prompt is provided on CLI, --only lines will have prompt=None and fail.

# Option A: Provide base prompt on CLI, override options per-line
echo "A __color__ portrait --w 1024 --h 1024 --d 42" > test.txt
echo "--w 512 --h 512 --d 123" >> test.txt  # Uses CLI's --prompt

python qwen_image_generate_image.py \
    --from_file test.txt \
    --prompt "Default prompt for option-only lines" \
    --prompt_wildcards /path/to/wildcards \
    ...

# Option B: Every line has its own prompt (no --only lines)
echo "A __color__ portrait --w 1024 --h 1024 --d 42" > test.txt
echo "Another __animal__ image --w 512 --h 512 --d 123" >> test.txt

python qwen_image_generate_image.py \
    --from_file test.txt \
    --prompt_wildcards /path/to/wildcards \
    ...

# Verify: All lines processed, __wildcards__ replaced with values from .txt files

# Test 1.3-1.4: Layered with auto-prompt
python qwen_image_generate_image.py \
    --is_layered \
    --automatic_prompt_lang_for_layered en \
    ...

# Verify: No VL processor error

# Test 1.5: Batch output shapes
# Verify: No IndexError, images saved correctly
```

### Phase 2 Tests

```bash
# Test training
accelerate launch hv_train_network.py \
    --network_module networks.loha \
    --network_dim 8 \
    --max_train_steps 10 \
    ...

# Verify: "create LoHa network" in logs

# Test merge (expect limitation if Option A)
python merge_lora.py --lora_weight loha_weights.safetensors ...
# Expected: May fail or use wrong network type (document this)
```

### Phase 3 Tests

```bash
# Test argparse - use actual CLI, not internal function
python flux_2_train_network.py --help | grep flux2_shift
# Verify: flux2_shift appears in --timestep_sampling choices

# Alternative: Quick parse test
python -c "
from musubi_tuner.hv_train_network import setup_parser_common, hv_setup_parser
p = setup_parser_common()
p = hv_setup_parser(p)
args = p.parse_args(['--timestep_sampling', 'flux2_shift', '--dit', 'dummy'])
print('flux2_shift accepted')
"

# Test float8 compat (runtime, not import)
python -c "import torch; print('float8_e4m3fn:', hasattr(torch, 'float8_e4m3fn'))"
# Verify: True (underscore version is the canonical name)

# Test caching
python flux_2_cache_text_encoder_outputs.py ...
# Verify: Cache contains ctx_vec_* keys

# Test training (batch=1 with refs)
# Verify: Assertion triggers if batch_size > 1
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Wildcards still broken in batch | Medium | Medium | Explicitly test --from_file with wildcards |
| LoHa inference unsupported | High | Medium | Document limitation clearly |
| flux2_shift argparse mismatch | N/A (fixed) | High | Use exact current choices list |
| torch.float8 alias missing | Low | High | Normalize to underscore version |
| FLUX.2 batch=1 surprises users | High | Medium | Document in flux_2.md prominently |
| logging.basicConfig conflicts | Medium | Low | Configure at entry points first |

---

## Implementation Checklist

### Phase 1: Qwen-Image (5 items + testing)

- [ ] **1.1** Fix `parse_prompt_line()` for `--only` lines (:217)
- [ ] **1.2** Pass `prompt_wildcards` in `preprocess_prompts_for_batch()` (:1141)
- [ ] **1.3** Fix VL processor in `load_shared_models()` (:1166)
- [ ] **1.4** Fix VL processor in `process_batch_prompts()` (:1205)
- [ ] **1.5** Fix `latent[0]` → `latent` in save_output (:1325)
- [ ] **1.6** Test: batch with `__wildcard__` syntax and `--only` lines
- [ ] **1.7** Test: layered with `automatic_prompt_lang_for_layered`

### Phase 2: LoHa (6 items)

- [ ] **2.1** Copy `networks/loha.py` from upstream
- [ ] **2.2** Apply logging strategy (Option A or B)
- [ ] **2.3** Add `architecture` param to `hv_train_network.py` (3 locations)
- [ ] **2.4** Document inference/merge limitation
- [ ] **2.5** (Optional) Add LoHa detection to merge_lora.py
- [ ] **2.6** Test LoHa training

### Phase 3: FLUX.2 (29 items)

#### Core Files
- [ ] **3.1** Create `flux_2/` directory
- [ ] **3.2** Copy `flux2_models.py` (strip/handle basicConfig)
- [ ] **3.3** Copy `flux2_utils.py` (normalize float8 aliases)
- [ ] **3.4** Copy `lora_flux_2.py`

#### Scripts
- [ ] **3.5** Copy `flux_2_train_network.py` with Blissful enhancements
- [ ] **3.6** Copy `flux_2_cache_latents.py`
- [ ] **3.7** Copy `flux_2_cache_text_encoder_outputs.py`
- [ ] **3.8** Copy `flux_2_generate_image.py`
- [ ] **3.9** Create 4 root wrapper scripts

#### Dataset Module
- [ ] **3.10** Add `ARCHITECTURE_FLUX_2` constants
- [ ] **3.11** Add `RESOLUTION_STEPS_FLUX_2 = 16`
- [ ] **3.12** Add `control_count_per_image = None` case for FLUX.2 (:1817)
- [ ] **3.13** Add cache functions (use `ctx_vec` key)

#### Supporting
- [ ] **3.14** Update `sai_model_spec.py` with FLUX.2
- [ ] **3.15** Add `flux2_shift` to argparse choices (:2862)
- [ ] **3.16** Add `flux2_shift` case: `mu = train_utils.get_lin_function(y1=0.5, y2=1.15)(h * w)` (:861)
- [ ] **3.17** Update `pyproject.toml`: `transformers>=4.56.1`
- [ ] **3.18** Normalize `float8e4m3fn` → `float8_e4m3fn` (avoid import-time refs)

#### Documentation
- [ ] **3.19** Copy `docs/flux_2.md` from upstream
- [ ] **3.20** Document batch=1 limitation in flux_2.md
- [ ] **3.21** Document `--fp8_m3` NotImplementedError in flux_2.md
- [ ] **3.22** Update README with FLUX.2 mention
- [ ] **3.23** Update CLAUDE.md

#### Testing
- [ ] **3.24** Test argparse with flux2_shift (use CLI, not internal function)
- [ ] **3.25** Test float8 compatibility (runtime check)
- [ ] **3.26** Test latent caching
- [ ] **3.27** Test text encoder caching (verify ctx_vec keys)
- [ ] **3.28** Test training
- [ ] **3.29** Test generation

### Post-Integration (4 items)

- [ ] **4.1** Smoke test existing architectures
- [ ] **4.2** Verify wildcards work in batch mode
- [ ] **4.3** Document LoHa limitation in user docs
- [ ] **4.4** Consider regression test for batch latent shapes

---

## Appendix: Key File References

| Purpose | Blissful Path | Upstream Path |
|---------|---------------|---------------|
| Qwen generate | `src/musubi_tuner/qwen_image_generate_image.py` | Same |
| Training base | `src/musubi_tuner/hv_train_network.py` | Same |
| LoHa module | (to create) | `src/musubi_tuner/networks/loha.py` |
| FLUX.2 models | (to create) | `src/musubi_tuner/flux_2/flux2_models.py` |
| FLUX.2 docs | (to create) | `docs/flux_2.md` |
| Dataset | `src/musubi_tuner/dataset/image_video_dataset.py` | Same |
| Model spec | `src/musubi_tuner/utils/sai_model_spec.py` | Same |

---

*v3 Final - All review feedback incorporated. Ready for implementation.*
