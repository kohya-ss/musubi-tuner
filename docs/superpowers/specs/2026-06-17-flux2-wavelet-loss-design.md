# Design: Official FLUX.2 Wavelet Loss Trainer

**Date:** 2026-06-17
**Status:** Approved
**Source:** Ports the working prototype on branch `example-wavelet-loss`
(`src/musubi_tuner/flux_2_train_network_wavelet_loss.py`) into an official,
cleaned-up form.

## Overview

Provide an official FLUX.2 training entry point that augments the standard
flow-matching MSE loss with a frequency-domain auxiliary loss computed via the
external `wavelet_loss` package.

The wavelet math stays in the external `wavelet_loss` /
`wavelet_transform` packages (currently at `/home/rockerboo/code/wavelet-loss`).
It remains an **optional dependency**, guarded by `try/except ImportError`; the
dependency-tree wiring (pyproject entry, vendoring decision) is deliberately
deferred and handled separately.

No core files change — no edits to `trainer_base.py` or any shared module. The
feature lives entirely in one new entry-point script plus one new test file.

## Decisions (settled during brainstorming)

1. **Wavelet code source:** stays external, optional dependency. Import guarded;
   dependency wiring handled later.
2. **Trainer structure:** a dedicated subclass entry-point script (not folded
   into `flux_2_train_network.py`), cleaned of example/experimental framing.
3. **Architecture scope:** FLUX.2 only. Other architectures may follow later.
4. **`trainer_base.py`:** untouched — the example branch's unrelated docstring
   trim is excluded.
5. **Tests:** add integration tests for the musubi-side wiring, skip-guarded
   when the optional dependency is absent.
6. **Dead args:** drop all unwired/unimplemented args; rewrite the docstring to
   describe actual behavior.

## Component: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`

`Flux2WaveletLossNetworkTrainer(Flux2NetworkTrainer)` wires the wavelet loss
through the existing extension seams:

- **`handle_model_specific_args(args)`** — calls super, then raises a clear
  `ImportError` if `args.wavelet_loss` is set but the `wavelet_loss` package is
  not installed.
- **`call_dit(...)`** — delegates to parent, then stashes `noisy_model_input`
  into `DiTOutput.extra` so `compute_loss` can recover x0 without changing the
  shared `compute_loss` signature. (`output.pred` and `output.target` are
  already on the `DiTOutput`; `noise` is not needed for the corrected target.)
- **`on_train_start(...)`** — when `--wavelet_loss` is set, constructs
  `WaveletLoss(...)` on `accelerator.device` (filters live as registered
  buffers), moves it to device, and logs the active configuration.
- **`compute_loss(...)`** — weighted flow-matching MSE plus
  `alpha * wavelet_loss`:
  - Base term: `F.mse_loss(pred, target, reduction="none")` optionally scaled by
    `compute_loss_weighting_for_sd3(...)`.
  - Wavelet term operates on **estimated clean latents (x0)** derived from the
    velocity prediction and the velocity target. In FLUX.2,
    `output.target = noise - latents` (velocity) and
    `noisy = (1-sigma)*latents + sigma*noise`, so subtracting `sigma * velocity`
    from the noisy input recovers the clean latents:
    - `sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=output.pred.ndim, ...)`
    - `x0_pred   = noisy_model_input - sigmas * output.pred`   (≈ `latents`)
    - `x0_target = noisy_model_input - sigmas * output.target` (= `latents` exactly)
    - **Note:** this corrects a bug in the `example-wavelet-loss` prototype,
      which used `x0_target = noisy - sigma*noise = (1-sigma)*latents`. That
      biased target left a residual of `sigma*latents` even at a perfect
      prediction; the corrected target makes the wavelet loss reach zero at the
      optimum. `output.target` is already stashed in `DiTOutput`, so no extra
      tensor needs to be carried through `extra` — only `noisy_model_input` is
      stashed (and `noise` is no longer needed for the loss).
  - Per-step `set_loss_fn(...)` selects l1 / huber(smooth_l1) / mse from
    `--wavelet_loss_type`, falling back to `--loss_type`.
  - `WaveletLoss.forward(x0_pred.float(), x0_target.float(), timesteps)` returns
    `(wav_loss, wav_metrics)`; the package applies its own internal sigmoid
    timestep weighting.
  - Returns `(mse_loss.mean() + alpha * wav_loss, {"wavelet_loss/<k>": v ...})`.
  - When `--wavelet_loss` is off (or module is None): returns
    `(mse_loss.mean(), {})` — identical to base FLUX.2 behavior.
- **`extra_metadata(args)`** — when enabled, embeds `ss_wavelet_loss*` keys
  (alpha, type, transform, wavelet, level, band weights as JSON, ll threshold)
  into the saved safetensors metadata; `{}` when disabled.
- **`main()`** — `setup_parser_common()` → `flux2_setup_parser()` →
  `wavelet_loss_setup_parser()`, then `read_config_from_file`, set
  `args.dit_dtype = None`, default `vae_dtype` to `float32`, and run the trainer.

### Arg surface

`wavelet_loss_setup_parser(parser)` adds:

| Arg | Type | Default | Notes |
|-----|------|---------|-------|
| `--wavelet_loss` | flag | off | Enable the auxiliary loss. |
| `--wavelet_loss_alpha` | float | 0.1 | Weight on the wavelet term. |
| `--wavelet_loss_type` | str | None | l1/l2/huber/smooth_l1; falls back to `--loss_type`. |
| `--wavelet_loss_transform` | choice | swt | dwt / swt / qwt. |
| `--wavelet_loss_wavelet` | str | sym7 | Wavelet family (e.g. sym7, db4). |
| `--wavelet_loss_level` | int | 1 | Decomposition levels. |
| `--wavelet_loss_band_weights` | parsed | None | `ll=0.1,lh=0.01,...` or JSON dict. |
| `--wavelet_loss_band_level_weights` | parsed | None | `ll1=0.1,hh2=0.05,...`; overrides band weights. |
| `--wavelet_loss_quaternion_component_weights` | parsed | None | `r=1.0,i=0.7,...`; qwt only. |
| `--wavelet_loss_ll_level_threshold` | int | None | Level at which LL band is included. |
| `--wavelet_loss_normalize_bands` | flag | None | Normalize each band before loss. |
| `--wavelet_loss_metrics` | flag | off | Log detailed per-band metrics (overhead). |

Plus helper `_parse_band_weights(s)`: returns `None` for `None`; parses a JSON /
literal dict when the string starts with `{`; otherwise parses
comma-separated `key=value` pairs into `dict[str, float]`.

### Dropped from the prototype

- `--wavelet_loss_primary` — described in the prototype docstring but never
  implemented (`compute_loss` only ever did `mse + alpha*wav`).
- `--wavelet_loss_timestep_intensity` — no-op; the package's
  `smooth_timestep_weight` hardcodes its transition.
- `--wavelet_loss_use_snr_aware_huber`, `--wavelet_loss_snr_huber_cmin`,
  `--wavelet_loss_snr_huber_cmax`, `--wavelet_loss_snr_huber_gamma`,
  `--wavelet_loss_snr_huber_alpha` — only logged, never wired into the loss.
- `--wavelet_loss_min_snr_beta` — defined, never referenced.

The module docstring is rewritten to describe actual behavior (mse + alpha *
wavelet on x0 estimates, internal sigmoid timestep weighting from the package),
with no API-stability or fork-breakage caveats.

## Component: `tests/test_wavelet_loss_trainer.py`

Module-level `pytest.importorskip("wavelet_loss")` so the suite skips cleanly
when the optional dependency is absent. Tests cover the musubi-side wiring only,
not the package internals:

- **`_parse_band_weights`**: `key=value` form → dict of floats; JSON-dict form;
  `None` passthrough.
- **x0-recovery identity**: given `noisy = (1-sigma)*latents + sigma*noise` and
  `target = noise - latents`, assert `x0_target = noisy - sigma*target ==
  latents` within tolerance; and with a perfect `pred == target`,
  `x0_pred == latents`. This locks in the corrected (unbiased) recovery math.
- **`compute_loss`**: with a small fabricated `DiTOutput` and a real
  `WaveletLoss`, returns a finite scalar loss and a non-empty `wavelet_loss/*`
  metrics dict; with `--wavelet_loss` off, returns empty metrics and a value
  equal to the plain weighted MSE.
- **`extra_metadata`**: emits the expected `ss_wavelet_loss*` keys when enabled;
  returns `{}` when disabled.

Run with `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py`.

## Out of scope

- Other architecture trainers (Wan, Qwen-Image, etc.).
- Vendoring the wavelet packages or adding pyproject dependencies.
- The `trainer_base.py` docstring change from the example branch.
- Implementing the dropped features (SNR-aware Huber, min-SNR weighting,
  primary mode) — these can return later as real implementations.
