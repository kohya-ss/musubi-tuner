# Agent Notes — musubi-tuner XPU fork

This is a fork of kohya-ss/musubi-tuner ported to **Intel Arc Pro B70 (XPU)**. There is no
NVIDIA GPU on this machine. The port is complete and validated — see `b70_support_plan.md`
for the current state, design decisions, and verification steps.

## Rules

1. **Stay on the active branch** (`port/upstream-v0.3.4`, or `main` once fast-forwarded).
   Local training scripts outside this repo run against whatever is checked out here.
2. **Don't upgrade `torch`, `transformers`, or `bitsandbytes`** without an explicit decision —
   the XPU stack is version-sensitive (transformers pinned at 4.56.1; Ideogram4/Krea2 need newer, we don't use them).
3. **Keep `torch.cuda.*` primitives out of shared code paths.** Device-specific transfer logic
   belongs behind the copier seam in `modules/custom_offloading_utils.py` (`_InOrderCopier` is the XPU shim).
4. **No mid-autograd `torch.xpu.synchronize()`** and no XPU sync from worker threads — both
   crash the driver (`UR_RESULT_ERROR_DEVICE_LOST` / deadlock). See the design table in `b70_support_plan.md`.
5. **In-training sampling stays disabled on XPU** (produces noise). Evaluate in ComfyUI.
6. Before any long training run: pytest (ideogram4 tests excluded) + a 3-step smoke run.
   Both commands are in `b70_support_plan.md`.

## Quick facts

- venv: `./venv` (Python 3.13, torch 2.11.0+xpu); invoke with `PYTHONPATH=src`
- Preferred LoRA training path: `--blocks_to_swap N --block_swap_h2d_only` (~22% faster than classic swap)
- Reference recipe: `examples/train_lora_flux2_xpu_h2d.sh` + `examples/flux2_dataset_config.example.toml`
