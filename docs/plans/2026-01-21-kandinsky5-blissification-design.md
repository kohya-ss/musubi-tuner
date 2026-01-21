# Kandinsky 5 Blissification Integration Design

**Date:** 2026-01-21
**Status:** Ready for Implementation

## Overview

Integrate the "Blissified" Kandinsky 5 features from Sarania/blissful-tuner into our codebase, achieving full feature parity while preserving our local customizations.

## Source

- **Upstream repo:** Sarania/blissful-tuner (cloned at `~/musubi-tuner-forks/blissful-tuner-original/`)
- **Branch:** main (add-features merged)

## Features Being Integrated

### Generation Script Enhancements
- CFG scheduling (`--cfg_schedule`)
- CFGZero* guidance (`--cfgzerostar_scaling`, `--cfgzerostar_init_steps`)
- Latent preview during generation (`--preview_latent_every`, `--preview_vae`)
- DPM++ scheduler option (`--scheduler`)
- Attention backend override (`--force_traditional_attn`, `--force_nabla_attn`)
- NF4 quantization for Qwen text encoder (`--quantized_qwen`)
- CPU text encoder support (`--text_encoder_cpu`)
- Metadata embedding in outputs
- torch.compile support (`--compile`)
- Better dtype handling (not hardcoded bf16)
- Output type options (`--output_type`: video/latent/both)
- Video codec/container options (`--codec`, `--container`)

### Training Enhancements
- CPU text encoder support
- Quantized Qwen option
- Improved compile during training
- Sample video fix (only save video if >1 frame)

## Integration Strategy

**Hybrid Approach:**
- **Replace wholesale:** K5-specific files (not customized locally)
- **Manual merge:** Shared infrastructure files (preserve local changes)

## File Classification

### Files to Replace (7 files)
| File | Current Lines | New Lines |
|------|---------------|-----------|
| `src/musubi_tuner/kandinsky5_generate_video.py` | 302 | 473 |
| `src/musubi_tuner/kandinsky5/generation_utils.py` | 818 | 913 |
| `src/musubi_tuner/kandinsky5/models/attention.py` | TBD | TBD |
| `src/musubi_tuner/kandinsky5/models/dit.py` | TBD | TBD |
| `src/musubi_tuner/kandinsky5/models/nn.py` | TBD | TBD |
| `src/musubi_tuner/kandinsky5/models/vae.py` | TBD | TBD |
| `src/musubi_tuner/kandinsky5_train_network.py` | 982 | 979 |

### Files to Merge (1 file)
| File | Changes |
|------|---------|
| `src/blissful_tuner/blissful_core.py` | Add K5 detection + `add_blissful_k5_args()` function |

### Files Unchanged
- `src/musubi_tuner/kandinsky5/configs.py`
- `src/musubi_tuner/kandinsky5_cache_latents.py`
- `src/musubi_tuner/kandinsky5_cache_text_encoder_outputs.py`
- Root wrapper scripts

## Implementation Phases

### Phase 1: Replace K5-Specific Files
1. Copy `kandinsky5_generate_video.py` from Sarania
2. Copy `kandinsky5/generation_utils.py` from Sarania
3. Copy `kandinsky5/models/attention.py` from Sarania
4. Copy `kandinsky5/models/dit.py` from Sarania
5. Copy `kandinsky5/models/nn.py` from Sarania
6. Copy `kandinsky5/models/vae.py` from Sarania
7. Copy `kandinsky5_train_network.py` from Sarania

### Phase 2: Merge blissful_core.py
1. Add K5 detection in ROOT_SCRIPT block (~line 57):
   ```python
   elif "kandinsky_" in ROOT_SCRIPT:
       DIFFUSION_MODEL = "k5"
   ```
2. Add `add_blissful_k5_args()` function after existing args functions (~line 447)

### Phase 3: Verification
1. Run `ruff check` on all modified files
2. Run `ruff format` if needed
3. Run Python syntax validation

### Phase 4: Documentation & Commit
1. Commit all changes with descriptive message

## Potential Issues and Mitigations

| Issue | Mitigation |
|-------|------------|
| Import dependencies | Verify blissful_tuner modules exist (BlissfulLogger, LatentPreviewer, etc.) |
| DPM++ scheduler import | Verify `FlowDPMSolverMultistepScheduler` exists in `wan.utils.fm_solvers` |
| Ruff lint differences | Run `ruff format` after copying |

## Version Strategy

- **Keep current version:** `0.12.67` (independent version line)
- Do not adopt Sarania's `0.13.66`

## Verification Level

- Syntax and lint checks only (no runtime testing)
- Full inference testing deferred until K5 model weights available
