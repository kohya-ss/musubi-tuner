# FLUX.2 Wavelet Loss Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an official FLUX.2 training entry point that augments flow-matching MSE with a wavelet-domain auxiliary loss, ported and cleaned up from the `example-wavelet-loss` prototype.

**Architecture:** A single new entry-point script `src/musubi_tuner/flux_2_train_network_wavelet_loss.py` defining `Flux2WaveletLossNetworkTrainer(Flux2NetworkTrainer)`. It wires the loss through existing extension seams (`handle_model_specific_args`, `call_dit`, `on_train_start`, `compute_loss`, `extra_metadata`). The wavelet math comes from the external, optional `wavelet_loss` package, imported behind a `try/except ImportError` guard. No core files are modified.

**Tech Stack:** Python 3.10+, PyTorch 2.12, `wavelet_loss` (optional external package, already installed in this venv), pytest. Run everything with `uv run --no-sync`.

## Global Constraints

- Environment: ALWAYS run Python/pytest via `uv run --no-sync` (the venv is hand-built; `uv sync`/plain `uv run` would clobber the manual torch+flash_attn install).
- Run tests with `uv run --no-sync pytest` (never `python -m pytest`).
- Do NOT modify `src/musubi_tuner/training/trainer_base.py` or any other core/shared file. The feature is additive: one new script + one new test file.
- The `wavelet_loss` package is an OPTIONAL dependency. The module must import successfully even when it is absent (guard with `try/except ImportError`); only raise if `--wavelet_loss` is actually set.
- Do NOT add the dropped args: `--wavelet_loss_primary`, `--wavelet_loss_timestep_intensity`, `--wavelet_loss_use_snr_aware_huber`, `--wavelet_loss_snr_huber_cmin/cmax/gamma/alpha`, `--wavelet_loss_min_snr_beta`.
- x0 recovery MUST use the velocity target: `x0_target = noisy_model_input - sigmas * output.target` (NOT `... * noise`).
- Git hygiene: stage only the specific files named in each commit step. Never `git add -A`.
- End every commit message with a trailing `AI-assisted` line.
- Branch: work happens on the currently checked-out `wavelet-loss` branch.

## File Structure

- Create: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py` — the entire feature (parser helper, parser, trainer subclass, `main`).
- Create: `tests/test_wavelet_loss_trainer.py` — integration tests for the musubi-side wiring.

## Reference: facts verified in the codebase (do not re-derive)

- `DiTOutput` (`src/musubi_tuner/training/trainer_base.py:86`): dataclass with fields `pred: torch.Tensor`, `target: torch.Tensor`, `extra: dict = field(default_factory=dict)`.
- `Flux2NetworkTrainer.call_dit(...)` returns `DiTOutput(pred=model_pred, target=noise - latents)` — `pred` and `target` are both **velocity** space (`flux_2_train_network.py:335`).
- Noisy input formula: `noisy_model_input = (1 - sigma) * latents + sigma * noise`.
- Base `compute_loss` signature (`trainer_base.py:1146`): `compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step) -> tuple[torch.Tensor, dict[str, float]]`.
- Base `call_dit` signature (`trainer_base.py:1083`): `call_dit(self, args, accelerator, transformer_arg, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs) -> DiTOutput`.
- Base `on_train_start` signature (`trainer_base.py:1191`): `on_train_start(self, args, accelerator, network, transformer, optimizer) -> None`.
- Base `extra_metadata` signature (`trainer_base.py:1283`): `extra_metadata(self, args) -> dict`.
- `Flux2NetworkTrainer.handle_model_specific_args(self, args)` (`flux_2_train_network.py:42`).
- Helpers in `musubi_tuner.training.timesteps`: `get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32)` (reads `noise_scheduler.sigmas` and `noise_scheduler.timesteps`) and `compute_loss_weighting_for_sd3(weighting_scheme, noise_scheduler, timesteps, device, dtype)` (returns `None` unless scheme is `sigma_sqrt`/`cosmap`).
- `WaveletLoss(transform_type, wavelet, level, band_weights, band_level_weights, quaternion_component_weights, ll_level_threshold, metrics, normalize_bands, device)`; `.to(device)`; `.set_loss_fn(fn)`; `forward(pred_4d, target_4d, timestep) -> (loss_tensor, metrics_dict)`. Expects 4D `[B, C, H, W]` float tensors.
- Entry points: `setup_parser_common()`, `read_config_from_file(args, parser)` from `musubi_tuner.hv_train_network`; `flux2_setup_parser(parser)` from `musubi_tuner.flux_2_train_network`.

---

### Task 1: Module scaffold + `_parse_band_weights` helper

**Files:**
- Create: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`
- Test: `tests/test_wavelet_loss_trainer.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: module `musubi_tuner.flux_2_train_network_wavelet_loss` importable even without the `wavelet_loss` package; function `_parse_band_weights(weights_str: Optional[str]) -> Optional[dict[str, float]]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_wavelet_loss_trainer.py`:

```python
"""Integration tests for the FLUX.2 wavelet-loss trainer wiring.

These exercise the musubi-side glue (arg parsing, x0 recovery, compute_loss
combination, metadata), not the wavelet_loss package internals.
"""

from musubi_tuner.flux_2_train_network_wavelet_loss import _parse_band_weights


def test_parse_band_weights_key_value():
    result = _parse_band_weights("ll=0.1,lh=0.01,hl=0.02,hh=0.05")
    assert result == {"ll": 0.1, "lh": 0.01, "hl": 0.02, "hh": 0.05}


def test_parse_band_weights_json():
    result = _parse_band_weights('{"ll": 0.1, "hh": 0.05}')
    assert result == {"ll": 0.1, "hh": 0.05}


def test_parse_band_weights_none():
    assert _parse_band_weights(None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -v`
Expected: FAIL — `ModuleNotFoundError` / `ImportError` (module does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`:

```python
"""Wavelet-loss training entry point for FLUX.2.

Augments the standard FLUX.2 flow-matching loss with a frequency-domain
auxiliary term computed by the optional ``wavelet_loss`` package.

The wavelet loss operates on *estimated clean latents* (x0) recovered from the
model's velocity prediction and the velocity target via the flow-matching
identity (``noisy = (1-sigma)*latents + sigma*noise``,
``target = noise - latents``):

    x0_pred   = noisy_model_input - sigma * pred     (approximately latents)
    x0_target = noisy_model_input - sigma * target   (exactly latents)

This gives the wavelet transform meaningful frequency structure to penalise and
makes the auxiliary loss reach zero at a perfect prediction.

Usage (extends a normal FLUX.2 training command):

    accelerate launch flux_2_train_network_wavelet_loss.py \\
        --wavelet_loss \\
        --wavelet_loss_alpha 0.1 \\
        --wavelet_loss_transform swt \\
        --wavelet_loss_level 2 \\
        <...normal FLUX.2 training args...>

The wavelet term combines additively with the base MSE:
``loss = mse.mean() + alpha * wavelet_loss``.
"""

import argparse
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import (
    DiTOutput,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3, get_sigmas

try:
    from wavelet_loss import WaveletLoss
except ImportError:
    WaveletLoss = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_band_weights(weights_str: Optional[str]) -> Optional[dict[str, float]]:
    """Parse ``ll=0.1,lh=0.01,hl=0.01,hh=0.05`` or a JSON/literal dict string."""
    if weights_str is None:
        return None
    import ast
    import json as _json

    if weights_str.strip().startswith("{"):
        try:
            return ast.literal_eval(weights_str)
        except (ValueError, SyntaxError):
            return _json.loads(weights_str.replace("'", '"'))

    result = {}
    for pair in weights_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = float(v.strip())
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_wavelet_loss.py tests/test_wavelet_loss_trainer.py
git commit -m "feat: scaffold FLUX.2 wavelet-loss module + band-weight parser

AI-assisted"
```

---

### Task 2: `wavelet_loss_setup_parser`

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`
- Test: `tests/test_wavelet_loss_trainer.py`

**Interfaces:**
- Consumes: `_parse_band_weights` (Task 1).
- Produces: `wavelet_loss_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser` adding all `--wavelet_loss*` args (defaults: `alpha=0.1`, `transform="swt"`, `wavelet="sym7"`, `level=1`, all others `None`/off).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wavelet_loss_trainer.py`:

```python
import argparse

from musubi_tuner.flux_2_train_network_wavelet_loss import wavelet_loss_setup_parser


def _wavelet_only_parser() -> argparse.ArgumentParser:
    # Standalone parser with only the wavelet args, to test them in isolation.
    return wavelet_loss_setup_parser(argparse.ArgumentParser())


def test_parser_defaults():
    parser = _wavelet_only_parser()
    args = parser.parse_args([])
    assert args.wavelet_loss is False
    assert args.wavelet_loss_alpha == 0.1
    assert args.wavelet_loss_transform == "swt"
    assert args.wavelet_loss_wavelet == "sym7"
    assert args.wavelet_loss_level == 1
    assert args.wavelet_loss_type is None
    assert args.wavelet_loss_band_weights is None


def test_parser_band_weights_parsed():
    parser = _wavelet_only_parser()
    args = parser.parse_args(["--wavelet_loss", "--wavelet_loss_band_weights", "ll=0.1,hh=0.05"])
    assert args.wavelet_loss is True
    assert args.wavelet_loss_band_weights == {"ll": 0.1, "hh": 0.05}


def test_parser_does_not_define_dropped_args():
    parser = _wavelet_only_parser()
    args = parser.parse_args([])
    for dropped in (
        "wavelet_loss_primary",
        "wavelet_loss_timestep_intensity",
        "wavelet_loss_use_snr_aware_huber",
        "wavelet_loss_min_snr_beta",
    ):
        assert not hasattr(args, dropped), f"dropped arg leaked: {dropped}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k parser -v`
Expected: FAIL — `ImportError: cannot import name 'wavelet_loss_setup_parser'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`:

```python
def wavelet_loss_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Wavelet-loss-specific CLI arguments."""
    parser.add_argument("--wavelet_loss", action="store_true", help="Enable wavelet auxiliary loss. Default: False")
    parser.add_argument("--wavelet_loss_alpha", type=float, default=0.1, help="Wavelet loss weight. Default: 0.1")
    parser.add_argument(
        "--wavelet_loss_type",
        default=None,
        help="Loss function for wavelet bands: l1, l2, huber, smooth_l1. Defaults to --loss_type.",
    )
    parser.add_argument(
        "--wavelet_loss_transform",
        default="swt",
        choices=["dwt", "swt", "qwt"],
        help="Wavelet transform: dwt (discrete), swt (stationary), qwt (quaternion). Default: swt",
    )
    parser.add_argument("--wavelet_loss_wavelet", default="sym7", help="Wavelet family (e.g. sym7, db4). Default: sym7")
    parser.add_argument(
        "--wavelet_loss_level",
        type=int,
        default=1,
        help="Decomposition levels. Level 1 captures coarse structure; higher levels add detail. Default: 1",
    )
    parser.add_argument(
        "--wavelet_loss_band_weights",
        type=_parse_band_weights,
        default=None,
        help="Per-band weights as ll=0.1,lh=0.01,hl=0.01,hh=0.05 or JSON dict. Default: library defaults.",
    )
    parser.add_argument(
        "--wavelet_loss_band_level_weights",
        type=_parse_band_weights,
        default=None,
        help="Per-band-per-level weights as ll1=0.1,lh1=0.01,hh2=0.05 etc. Overrides --wavelet_loss_band_weights.",
    )
    parser.add_argument(
        "--wavelet_loss_quaternion_component_weights",
        type=_parse_band_weights,
        default=None,
        help="QWT component weights as r=1.0,i=0.7,j=0.7,k=0.5. Only used with --wavelet_loss_transform qwt.",
    )
    parser.add_argument(
        "--wavelet_loss_ll_level_threshold",
        type=int,
        default=None,
        help="Level at which to include LL (low-frequency) band. -1 = last level only. Default: None (use all).",
    )
    parser.add_argument(
        "--wavelet_loss_normalize_bands",
        action="store_true",
        default=None,
        help="Normalise each wavelet band before computing the loss.",
    )
    parser.add_argument(
        "--wavelet_loss_metrics",
        action="store_true",
        help="Log detailed per-band wavelet metrics each step (adds overhead). Default: False",
    )
    return parser
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k parser -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_wavelet_loss.py tests/test_wavelet_loss_trainer.py
git commit -m "feat: add wavelet_loss_setup_parser with cleaned arg surface

AI-assisted"
```

---

### Task 3: Trainer class — `handle_model_specific_args`, `call_dit`, `on_train_start`

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`
- Test: `tests/test_wavelet_loss_trainer.py`

**Interfaces:**
- Consumes: `wavelet_loss_setup_parser` (Task 2), `WaveletLoss` (optional), `DiTOutput`.
- Produces: class `Flux2WaveletLossNetworkTrainer(Flux2NetworkTrainer)` with attribute `self.wavelet_loss: Optional[WaveletLoss]`, and methods `handle_model_specific_args`, `call_dit` (stashes `noisy_model_input` in `output.extra`), `on_train_start` (builds the module).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wavelet_loss_trainer.py`:

```python
import pytest

from musubi_tuner.flux_2_train_network_wavelet_loss import Flux2WaveletLossNetworkTrainer


def test_handle_model_specific_args_requires_package(monkeypatch):
    import musubi_tuner.flux_2_train_network_wavelet_loss as mod

    monkeypatch.setattr(mod, "WaveletLoss", None)
    trainer = Flux2WaveletLossNetworkTrainer()
    args = argparse.Namespace(
        wavelet_loss=True,
        model_version="flux2-dev",  # any valid key; only reached if import guard passes
    )
    # The guard must fire before any model-version logic.
    with pytest.raises(ImportError):
        trainer.handle_model_specific_args(args)
```

> Note: `handle_model_specific_args` calls `super().handle_model_specific_args(args)` which reads `args.model_version`. To keep this test independent of FLUX.2 model tables, the implementation in Step 3 checks the import guard BEFORE calling super, so the `ImportError` is raised first. The test relies on that ordering.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k handle_model_specific -v`
Expected: FAIL — `ImportError: cannot import name 'Flux2WaveletLossNetworkTrainer'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/musubi_tuner/flux_2_train_network_wavelet_loss.py` (place the class definition ABOVE `_parse_band_weights`/`wavelet_loss_setup_parser` is not required — Python resolves names at call time — but for readability add it after the imports block and before `_parse_band_weights`). Insert:

```python
class Flux2WaveletLossNetworkTrainer(Flux2NetworkTrainer):
    """FLUX.2 + wavelet-domain auxiliary loss.

    Owned state:
    - ``self.wavelet_loss``: ``WaveletLoss`` module, constructed in
      ``on_train_start`` when ``--wavelet_loss`` is set. Holds the wavelet
      filters as registered buffers so they move to the correct device with
      ``.to(device)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelet_loss: Optional["WaveletLoss"] = None  # type: ignore[type-arg]

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        # Check the optional-dependency guard FIRST so a clear error is raised
        # before any FLUX.2-specific model-version logic runs.
        if args.wavelet_loss and WaveletLoss is None:
            raise ImportError(
                "wavelet-loss package is not installed. Install it with: pip install -e /path/to/wavelet-loss"
            )
        super().handle_model_specific_args(args)

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        """Delegate to parent and stash ``noisy_model_input`` for compute_loss.

        ``output.pred`` and ``output.target`` (both velocity space) are already
        on the returned ``DiTOutput``; only ``noisy_model_input`` needs stashing
        to recover the clean latents in ``compute_loss``.
        """
        output = super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )
        output.extra["noisy_model_input"] = noisy_model_input
        return output

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        """Construct and move the WaveletLoss module to the training device."""
        if not args.wavelet_loss:
            return

        assert WaveletLoss is not None, "wavelet-loss package not installed"
        device = accelerator.device
        self.wavelet_loss = WaveletLoss(
            transform_type=args.wavelet_loss_transform,
            wavelet=args.wavelet_loss_wavelet,
            level=args.wavelet_loss_level,
            band_weights=args.wavelet_loss_band_weights,
            band_level_weights=args.wavelet_loss_band_level_weights,
            quaternion_component_weights=args.wavelet_loss_quaternion_component_weights,
            ll_level_threshold=args.wavelet_loss_ll_level_threshold,
            metrics=args.wavelet_loss_metrics,
            normalize_bands=args.wavelet_loss_normalize_bands,
            device=device,
        )
        self.wavelet_loss.to(device)

        logger.info("Wavelet loss enabled:")
        logger.info(f"\tTransform: {args.wavelet_loss_transform}")
        logger.info(f"\tWavelet:   {args.wavelet_loss_wavelet}")
        logger.info(f"\tLevel:     {args.wavelet_loss_level}")
        logger.info(f"\tAlpha:     {args.wavelet_loss_alpha}")
        if args.wavelet_loss_band_weights:
            logger.info(f"\tBand weights: {args.wavelet_loss_band_weights}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k handle_model_specific -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_wavelet_loss.py tests/test_wavelet_loss_trainer.py
git commit -m "feat: add wavelet trainer class with import guard, call_dit stash, on_train_start

AI-assisted"
```

---

### Task 4: `compute_loss` — weighted MSE + alpha * wavelet on corrected x0

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`
- Test: `tests/test_wavelet_loss_trainer.py`

**Interfaces:**
- Consumes: `Flux2WaveletLossNetworkTrainer` (Task 3), `get_sigmas`, `compute_loss_weighting_for_sd3`, `WaveletLoss`.
- Produces: `Flux2WaveletLossNetworkTrainer.compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step) -> tuple[torch.Tensor, dict[str, float]]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wavelet_loss_trainer.py`:

```python
pytest.importorskip("wavelet_loss")


class _FakeScheduler:
    """Minimal stand-in for get_sigmas: needs .sigmas and .timesteps."""

    def __init__(self, timesteps: torch.Tensor, sigmas: torch.Tensor):
        self.timesteps = timesteps
        self.sigmas = sigmas


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        wavelet_loss=True,
        wavelet_loss_alpha=0.1,
        wavelet_loss_type=None,
        wavelet_loss_transform="swt",
        wavelet_loss_wavelet="sym7",
        wavelet_loss_level=1,
        wavelet_loss_band_weights=None,
        wavelet_loss_band_level_weights=None,
        wavelet_loss_quaternion_component_weights=None,
        wavelet_loss_ll_level_threshold=None,
        wavelet_loss_normalize_bands=None,
        wavelet_loss_metrics=False,
        weighting_scheme="none",
        loss_type="l2",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _setup_trainer_and_batch():
    from musubi_tuner.flux_2_train_network_wavelet_loss import WaveletLoss as _WL

    torch.manual_seed(0)
    b, c, h, w = 1, 4, 16, 16
    latents = torch.randn(b, c, h, w)
    noise = torch.randn(b, c, h, w)

    # schedule with a single timestep present so get_sigmas resolves cleanly
    timesteps = torch.tensor([2])
    schedule_ts = torch.tensor([0, 1, 2, 3])
    schedule_sigmas = torch.tensor([0.0, 0.25, 0.5, 0.75])
    scheduler = _FakeScheduler(schedule_ts, schedule_sigmas)
    sigma = 0.5  # sigmas[index of timestep 2]

    noisy = (1.0 - sigma) * latents + sigma * noise
    target = noise - latents  # velocity target (matches Flux2 call_dit)

    trainer = Flux2WaveletLossNetworkTrainer()
    args = _make_args()
    trainer.wavelet_loss = _WL(
        transform_type="swt", wavelet="sym7", level=1, ll_level_threshold=None, device=torch.device("cpu")
    )
    return trainer, args, latents, noise, noisy, target, timesteps, scheduler, sigma


def test_x0_target_recovers_latents():
    _, _, latents, noise, noisy, target, _, _, sigma = _setup_trainer_and_batch()
    x0_target = noisy - sigma * target
    assert torch.allclose(x0_target, latents, atol=1e-5)
    # perfect prediction (pred == target) recovers latents too
    x0_pred = noisy - sigma * target
    assert torch.allclose(x0_pred, latents, atol=1e-5)


def test_compute_loss_with_wavelet_returns_metrics():
    trainer, args, latents, noise, noisy, target, timesteps, scheduler, _ = _setup_trainer_and_batch()
    pred = target + 0.1 * torch.randn_like(target)  # imperfect prediction
    output = DiTOutput(pred=pred, target=target, extra={"noisy_model_input": noisy})

    loss, metrics = trainer.compute_loss(
        args, output, timesteps, scheduler, torch.float32, torch.float32, global_step=0
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert len(metrics) > 0
    assert all(k.startswith("wavelet_loss/") for k in metrics)


def test_compute_loss_disabled_equals_weighted_mse():
    trainer, args, latents, noise, noisy, target, timesteps, scheduler, _ = _setup_trainer_and_batch()
    args.wavelet_loss = False
    trainer.wavelet_loss = None
    pred = target + 0.1 * torch.randn_like(target)
    output = DiTOutput(pred=pred, target=target, extra={"noisy_model_input": noisy})

    loss, metrics = trainer.compute_loss(
        args, output, timesteps, scheduler, torch.float32, torch.float32, global_step=0
    )
    expected = F.mse_loss(pred, target).detach()
    assert metrics == {}
    assert torch.allclose(loss, expected, atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k compute_loss -v`
Expected: FAIL — `AttributeError`/`NotImplementedError` (subclass `compute_loss` not defined; falls back to base which lacks the wavelet path / returns `{}` but the enabled test asserts non-empty metrics).

- [ ] **Step 3: Write minimal implementation**

Add the `compute_loss` method inside `Flux2WaveletLossNetworkTrainer` (after `on_train_start`):

```python
    def compute_loss(
        self,
        args: argparse.Namespace,
        output: DiTOutput,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Weighted flow-matching MSE + optional wavelet auxiliary loss.

        The wavelet term runs on estimated clean latents (x0) recovered from the
        velocity prediction and the velocity target:

            x0_pred   = noisy_model_input - sigma * output.pred
            x0_target = noisy_model_input - sigma * output.target  (= latents)

        Combination is additive: ``mse.mean() + alpha * wavelet_loss``.
        """
        weighting = compute_loss_weighting_for_sd3(
            args.weighting_scheme, noise_scheduler, timesteps, timesteps.device, dit_dtype
        )
        mse_loss = F.mse_loss(output.pred.to(network_dtype), output.target, reduction="none")
        if weighting is not None:
            mse_loss = mse_loss * weighting

        if not args.wavelet_loss or self.wavelet_loss is None:
            return mse_loss.mean(), {}

        # --- wavelet path ---
        noisy_model_input = output.extra["noisy_model_input"]

        sigmas = get_sigmas(
            noise_scheduler, timesteps, noisy_model_input.device, n_dim=output.pred.ndim, dtype=output.pred.dtype
        )
        x0_pred = noisy_model_input - sigmas * output.pred.to(noisy_model_input.dtype)
        x0_target = noisy_model_input - sigmas * output.target.to(noisy_model_input.dtype)

        loss_type = args.wavelet_loss_type if args.wavelet_loss_type is not None else args.loss_type

        def _loss_fn(input: torch.Tensor, target: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
            if loss_type in ("l1", "mae"):
                return F.l1_loss(input, target, reduction=reduction)
            if loss_type in ("huber", "smooth_l1"):
                return F.smooth_l1_loss(input, target, reduction=reduction)
            return F.mse_loss(input, target, reduction=reduction)

        self.wavelet_loss.set_loss_fn(_loss_fn)

        wav_loss, wav_metrics = self.wavelet_loss(x0_pred.float(), x0_target.float(), timesteps)
        loss_metrics = {f"wavelet_loss/{k}": v for k, v in wav_metrics.items()}

        return mse_loss.mean() + args.wavelet_loss_alpha * wav_loss, loss_metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k compute_loss -v`
Expected: PASS (2 passed) and the x0 test passes too: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k "compute_loss or x0" -v`.

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_wavelet_loss.py tests/test_wavelet_loss_trainer.py
git commit -m "feat: add wavelet compute_loss with corrected x0 recovery

AI-assisted"
```

---

### Task 5: `extra_metadata`

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`
- Test: `tests/test_wavelet_loss_trainer.py`

**Interfaces:**
- Consumes: `Flux2WaveletLossNetworkTrainer` (Task 3).
- Produces: `Flux2WaveletLossNetworkTrainer.extra_metadata(self, args) -> dict` returning `ss_wavelet_loss*` keys when enabled, `{}` when disabled.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wavelet_loss_trainer.py`:

```python
def test_extra_metadata_enabled():
    trainer = Flux2WaveletLossNetworkTrainer()
    args = _make_args(wavelet_loss_band_weights={"ll": 0.1, "hh": 0.05})
    meta = trainer.extra_metadata(args)
    assert meta["ss_wavelet_loss"] is True
    assert meta["ss_wavelet_loss_alpha"] == 0.1
    assert meta["ss_wavelet_loss_transform"] == "swt"
    assert meta["ss_wavelet_loss_wavelet"] == "sym7"
    assert meta["ss_wavelet_loss_level"] == 1
    # dict-valued args are JSON-encoded strings
    import json
    assert json.loads(meta["ss_wavelet_loss_band_weights"]) == {"ll": 0.1, "hh": 0.05}


def test_extra_metadata_disabled():
    trainer = Flux2WaveletLossNetworkTrainer()
    args = _make_args(wavelet_loss=False)
    assert trainer.extra_metadata(args) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k extra_metadata -v`
Expected: FAIL — base `extra_metadata` returns `{}` for both, so `test_extra_metadata_enabled` fails on `meta["ss_wavelet_loss"]` `KeyError`.

- [ ] **Step 3: Write minimal implementation**

Add the `extra_metadata` method inside `Flux2WaveletLossNetworkTrainer` (after `compute_loss`):

```python
    def extra_metadata(self, args: argparse.Namespace) -> dict:
        """Embed wavelet-loss configuration into the saved safetensors metadata."""
        if not args.wavelet_loss:
            return {}
        import json

        return {
            "ss_wavelet_loss": True,
            "ss_wavelet_loss_alpha": args.wavelet_loss_alpha,
            "ss_wavelet_loss_type": args.wavelet_loss_type,
            "ss_wavelet_loss_transform": args.wavelet_loss_transform,
            "ss_wavelet_loss_wavelet": args.wavelet_loss_wavelet,
            "ss_wavelet_loss_level": args.wavelet_loss_level,
            "ss_wavelet_loss_band_weights": json.dumps(args.wavelet_loss_band_weights)
            if args.wavelet_loss_band_weights
            else None,
            "ss_wavelet_loss_band_level_weights": json.dumps(args.wavelet_loss_band_level_weights)
            if args.wavelet_loss_band_level_weights
            else None,
            "ss_wavelet_loss_quaternion_component_weights": json.dumps(args.wavelet_loss_quaternion_component_weights)
            if args.wavelet_loss_quaternion_component_weights
            else None,
            "ss_wavelet_loss_ll_level_threshold": args.wavelet_loss_ll_level_threshold,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k extra_metadata -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_wavelet_loss.py tests/test_wavelet_loss_trainer.py
git commit -m "feat: embed wavelet-loss config in saved metadata

AI-assisted"
```

---

### Task 6: `main()` entry point + full-suite verification

**Files:**
- Modify: `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`
- Test: `tests/test_wavelet_loss_trainer.py`

**Interfaces:**
- Consumes: `setup_parser_common`, `flux2_setup_parser`, `wavelet_loss_setup_parser`, `read_config_from_file`, `Flux2WaveletLossNetworkTrainer`.
- Produces: `main()` and an importable, runnable module.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_wavelet_loss_trainer.py`:

```python
def test_combined_parser_has_common_and_wavelet_args():
    from musubi_tuner.hv_train_network import setup_parser_common
    from musubi_tuner.flux_2_train_network import flux2_setup_parser
    from musubi_tuner.flux_2_train_network_wavelet_loss import wavelet_loss_setup_parser

    parser = wavelet_loss_setup_parser(flux2_setup_parser(setup_parser_common()))
    # wavelet args coexist with the common/flux2 args without conflict
    dests = {a.dest for a in parser._actions}
    assert "wavelet_loss" in dests
    assert "wavelet_loss_alpha" in dests


def test_main_is_callable():
    from musubi_tuner.flux_2_train_network_wavelet_loss import main

    assert callable(main)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -k "combined_parser or main_is_callable" -v`
Expected: FAIL — `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/musubi_tuner/flux_2_train_network_wavelet_loss.py`:

```python
def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)
    parser = wavelet_loss_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "float32"  # make float32 the default for VAE

    trainer = Flux2WaveletLossNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `uv run --no-sync pytest tests/test_wavelet_loss_trainer.py -v`
Expected: PASS (all tests in the file pass; nothing skipped since `wavelet_loss` is installed in this venv).

- [ ] **Step 5: Smoke-check the CLI help renders (no crash on import/parser build)**

Run: `uv run --no-sync python src/musubi_tuner/flux_2_train_network_wavelet_loss.py --help`
Expected: argparse help text prints, including the `--wavelet_loss*` options; exit code 0. (No training starts.)

- [ ] **Step 6: Commit**

```bash
git add src/musubi_tuner/flux_2_train_network_wavelet_loss.py tests/test_wavelet_loss_trainer.py
git commit -m "feat: add main() entry point for FLUX.2 wavelet-loss trainer

AI-assisted"
```

---

## Self-Review (completed by plan author)

- **Spec coverage:** entry-point script (Tasks 1-6), import guard (Task 3), `call_dit` stash (Task 3), `on_train_start` build (Task 3), corrected `compute_loss` x0 recovery (Task 4), `extra_metadata` (Task 5), arg surface with dropped args (Task 2), helper (Task 1), tests for parser/x0/compute_loss/metadata (Tasks 1,2,4,5), `main` (Task 6). All spec sections map to a task.
- **Placeholder scan:** no TBD/TODO; every code step contains full code; every command has expected output.
- **Type consistency:** `Flux2WaveletLossNetworkTrainer`, `_parse_band_weights`, `wavelet_loss_setup_parser`, `main`, and the `output.extra["noisy_model_input"]` key are used consistently across tasks. `compute_loss`/`call_dit`/`on_train_start`/`extra_metadata`/`handle_model_specific_args` signatures match the verified base-class signatures.
