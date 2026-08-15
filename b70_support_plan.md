# Intel Arc B70 XPU Port — Current State

Status doc for the XPU port of musubi-tuner (fork of kohya-ss/musubi-tuner).
Not a plan — the port is done and validated. Last updated: 2026-07-08.

## Environment

- **GPU:** Intel Arc Pro B70 (Xe2/Battlemage, 32GB VRAM, PCI ID `e223`), requires compute-runtime v26.14+
- **PyTorch:** 2.11.0+xpu (wheels from `https://download.pytorch.org/whl/xpu`), Python 3.13 venv at `./venv`
- **Branch:** `port/upstream-v0.3.4` = upstream v0.3.4 + the XPU commits (pre-rebase history: `main-pre-rebase`)

## What works (validated)

- **Flux2 dev LoRA training end-to-end**: 540-step dim64/alpha64 bf16 run with
  `--blocks_to_swap 40 --block_swap_h2d_only`, output LoRA validated in ComfyUI against a
  known-good reference (2026-07-07). Recipe: `examples/train_lora_flux2_xpu_h2d.sh`.
- **Caching** (latents + text encoder) on XPU; `--device_map_auto` splits the Mistral-3 TE across GPU+CPU.
- **Wan 2.1/2.2 LoRA training** (validated earlier: 2×1000-step I2V run).
- **adamw8bit** (bitsandbytes 0.49.2 XPU kernels).

## XPU design decisions (why the code looks the way it does)

| Area | Decision | Reason |
|---|---|---|
| Classic block swap | Synchronous on the calling thread (`_SyncFuture`, no ThreadPool) | `torch.xpu.synchronize()` is not thread-safe; SYCL context is bound to the main thread |
| Swap copies | `non_blocking=True` + one trailing sync (`swap_weight_devices_no_cuda`) | Pipelines to ~10.6 GB/s vs ~3.3 blocking; per-tensor sync mid-autograd triggers `UR_RESULT_ERROR_DEVICE_LOST` |
| **H2D-only swap** (`--block_swap_h2d_only`) | `LoRAStreamOffloader` + `_InOrderCopier`: flat H2D copies on the default in-order queue, no events | Frozen-base LoRA never needs the D2H half; in-order queue provides the ordering CUDA needs events for. **36 s/it vs 47 s/it classic** on the Flux2 dim64 recipe → use this for LoRA training |
| Trainer loop | `torch.xpu.synchronize()` every 50 steps | Prevents driver command-queue buildup / hangs on long runs |
| Pinned memory | Not used on XPU (forced off in `LoRAStreamOffloader`) | Measured slower than pageable `non_blocking` on this stack |
| Flux2 `Modulation`/`LastLayer` | fp32 upcasts removed | Matches the numerics of the validated LoRAs; restoring them is an untested A/B |
| Wan time embedding | bf16 autocast on XPU (fp32 path misbehaves) | Plus bf16 `e` accepted in `WanAttentionBlock`/`Head` |
| Device detection | `xpu` preferred over `cuda` in caching scripts and defaults | Single-GPU XPU box |

## Known limitations

- **In-training sampling on XPU produces noise** — keep it disabled; evaluate checkpoints in ComfyUI.
- **`transformers` pinned at 4.56.1** — upstream's Ideogram4/Krea2 code (and their tests) need newer.
  Do not upgrade casually; the XPU stack is sensitive. Excluded tests: `tests/test_ideogram4_*`.
- No copy/compute overlap in `_InOrderCopier` (single in-order queue). A `torch.xpu.Stream`
  copier is the known next optimization if more speed is needed.

## Verifying changes

```bash
source venv/bin/activate
PYTHONPATH=src python -m pytest tests/ -q --ignore=tests/test_ideogram4_synthetic.py \
  --ignore=tests/test_ideogram4_autoencoder.py --ignore=tests/test_ideogram4_fp8_loading.py \
  --ignore=tests/test_ideogram4_lora_sampling.py --ignore=tests/test_ideogram4_te_fp8_loading.py \
  --ignore=tests/test_ideogram4_timesteps.py
```

Then a 3-step smoke of the Flux2 recipe (`--max_train_steps 3`, see
`examples/train_lora_flux2_xpu_h2d.sh`) before any long run: expect the block swap line
`double blocks: 6, single blocks: 46` at `--blocks_to_swap 40`, finite loss, a saved
checkpoint, and ~36 s/it (h2d) / ~47 s/it (classic) at 1024px on the B70.
