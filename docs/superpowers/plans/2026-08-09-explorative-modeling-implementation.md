# Forward XM and MiniMax-H3 Video Best-of-K Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add sequential memory-saving Forward XM to compatible `NetworkTrainer` LoRA trainers and a separately named MiniMax-H3 video-focused best-of-K heuristic.

**Architecture:** The shared trainer keeps its existing `K = 1` path and dispatches `K > 1` batches through a standard Forward XM implementation. Pure candidate generation and per-sample winner updates live in `training/explorative.py`; timestep reconstruction and canonical per-sample losses remain in their existing ownership modules. MiniMax-H3 owns a local video-noise search loop and reuses only the small pure winner utilities.

**Tech Stack:** Python 3.10.11, PyTorch 2.13.0+cu130, CUDA 13.0, Accelerate, pytest, Ruff.

**Decomposition:** Standard Forward XM and the H3 heuristic remain in one plan because H3 consumes the same candidate generator, winner update, early-validation state, and training-loop dispatch. Tasks 5 and 6 are separate reviewer gates, so the distinct selection objectives cannot be merged accidentally.

## Global Constraints

- The approved design is `docs/superpowers/specs/2026-08-09-explorative-modeling-design.md`; any implementation deviation from it requires a failing test and an explicit design update.
- Standard trainers use `--xm_best_of_k`; MiniMax-H3 uses `--h3_video_best_of_k` and rejects `--xm_best_of_k > 1`.
- `K = 1` must use the pre-feature batch path, draw no additional random values, emit no exploration metrics, and preserve post-step RNG state.
- `K > 1` uses K sequential no-grad candidate forwards plus one gradient-enabled winner forward; no K-sized latent or activation tensor may be retained.
- Standard Forward XM selects independently per sample with the canonical weighted training loss.
- MiniMax-H3 varies only video noise, selects by video loss, fixes audio noise/time/conditions, and keeps weighted audio loss and gradients in the final update. It must never be presented as Forward XM.
- Every candidate shares clean latents, timestep, conditioning, condition-drop decisions, and other stochastic forward choices. Candidate-derived prediction targets remain paired with candidate noise.
- Candidate loss NaN or infinity is a fail-fast error for `K > 1`; baseline `K = 1` behavior is unchanged.
- Do not call `torch.clear_autocast_cache()` explicitly.
- Validate on `E:\Python310\python.exe`, PyTorch `2.13.0+cu130`, CUDA 13.0, and RTX 4090. Do not add a minimum-version CI job or a separate CUDA environment.
- Use `torch.allclose(rtol=1e-5, atol=1e-8)` for float32 scalar-loss reduction comparisons. Use the noising tolerances from the approved spec.
- Every production behavior follows a failing-test, minimal-implementation, passing-test cycle.

---

## File Map

- Create `src/musubi_tuner/training/explorative.py`: candidate generator, candidate draw, and pure per-sample winner update.
- Modify `src/musubi_tuner/training/timesteps.py`: shared explicit-coefficient sampler set and fixed-timestep coefficient reconstruction.
- Modify `src/musubi_tuner/training/parser_common.py`: common `--xm_best_of_k` argument.
- Modify `src/musubi_tuner/training/trainer_base.py`: validation, dispatch, canonical per-sample loss, and standard Forward XM batch processing.
- Modify `src/musubi_tuner/ideogram4_train_network.py`: canonical unweighted per-sample MSE and scalar diagnostics wrapper.
- Modify `src/musubi_tuner/flux_2_train_network_self_flow.py`: mode-specific compatibility response.
- Modify `src/musubi_tuner/hidream_o1_train_network.py`: explicit Forward XM incompatibility response.
- Modify `src/musubi_tuner/minimax_h3_train_network.py`: H3-specific CLI, component losses, fixed state preparation, and local video best-of-K loop.
- Create `tests/test_explorative_modeling.py`: pure helpers, noising, parser, validation, standard Forward XM, RNG, loss, and CUDA autocast coverage.
- Modify `tests/test_ideogram4_synthetic.py`: Ideogram 4 per-sample loss agreement.
- Modify `tests/test_minimax_h3_training.py`: H3 parser, component, selection, fixed-state, and gradient coverage.
- Create `docs/explorative_modeling.md`: user contract, examples, costs, limitations, and non-finite policy in English and Japanese.
- Modify `docs/minimax_h3.md`, `README.md`, and `README.ja.md`: discoverability and H3-specific method warning.

### Task 1: Pure Candidate And Winner Mechanics

**Files:**
- Create: `src/musubi_tuner/training/explorative.py`
- Create: `tests/test_explorative_modeling.py`

**Interfaces:**
- Produces: `create_candidate_generator(reference: torch.Tensor) -> torch.Generator`.
- Produces: `draw_candidate_noise(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor`.
- Produces: `update_winners(best_losses, winner_noise, winner_indices, candidate_losses, candidate_noise, candidate_index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`.
- Guarantees: candidate draws do not advance the global RNG after generator creation; winner storage stays `[B, ...]`, not `[K, B, ...]`.

- [ ] **Step 1: Write failing candidate-stream and mixed-winner tests**

Create `tests/test_explorative_modeling.py` with the repository `src` path setup and these tests:

```python
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.training.explorative import (
    create_candidate_generator,
    draw_candidate_noise,
    update_winners,
)


def test_update_winners_selects_each_sample_independently():
    best_losses = torch.full((3,), torch.inf)
    winner_noise = torch.empty(3, 1)
    winner_indices = torch.full((3,), -1, dtype=torch.long)

    best_losses, winner_noise, winner_indices = update_winners(
        best_losses,
        winner_noise,
        winner_indices,
        torch.tensor([3.0, 1.0, 4.0]),
        torch.tensor([[30.0], [10.0], [40.0]]),
        0,
    )
    best_losses, winner_noise, winner_indices = update_winners(
        best_losses,
        winner_noise,
        winner_indices,
        torch.tensor([2.0, 5.0, 0.5]),
        torch.tensor([[20.0], [50.0], [5.0]]),
        1,
    )

    torch.testing.assert_close(best_losses, torch.tensor([2.0, 1.0, 0.5]))
    torch.testing.assert_close(winner_noise[:, 0], torch.tensor([20.0, 10.0, 5.0]))
    assert winner_indices.tolist() == [1, 0, 1]


def test_candidate_generator_owns_draws_after_one_global_seed_draw():
    torch.manual_seed(1234)
    reference = torch.empty(2, 3)
    generator = create_candidate_generator(reference)
    global_state_after_creation = torch.random.get_rng_state().clone()

    first = draw_candidate_noise(reference, generator)
    second = draw_candidate_noise(reference, generator)

    assert torch.equal(torch.random.get_rng_state(), global_state_after_creation)
    assert first.shape == reference.shape
    assert first.dtype == reference.dtype
    assert first.device == reference.device
    assert not torch.equal(first, second)

    torch.manual_seed(1234)
    replay = create_candidate_generator(reference)
    torch.testing.assert_close(draw_candidate_noise(reference, replay), first)
    torch.testing.assert_close(draw_candidate_noise(reference, replay), second)
```

- [ ] **Step 2: Run the helper tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py -v
```

Expected: collection fails with `ModuleNotFoundError: musubi_tuner.training.explorative`.

- [ ] **Step 3: Implement the private candidate stream and pure winner update**

Create `src/musubi_tuner/training/explorative.py`:

```python
"""Pure mechanics shared by sequential best-of-K training paths."""

import torch


def create_candidate_generator(reference: torch.Tensor) -> torch.Generator:
    seed = torch.randint(
        0,
        torch.iinfo(torch.int64).max,
        (),
        device=reference.device,
        dtype=torch.int64,
    ).item()
    return torch.Generator(device=reference.device).manual_seed(seed)


def draw_candidate_noise(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(
        reference.shape,
        dtype=reference.dtype,
        device=reference.device,
        generator=generator,
    )


def update_winners(
    best_losses: torch.Tensor,
    winner_noise: torch.Tensor,
    winner_indices: torch.Tensor,
    candidate_losses: torch.Tensor,
    candidate_noise: torch.Tensor,
    candidate_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if best_losses.ndim != 1:
        raise ValueError("best-of-K best loss must have shape [B]")
    batch_size = best_losses.shape[0]
    if candidate_losses.shape != (batch_size,):
        raise ValueError(
            f"best-of-K candidate loss must have shape [{batch_size}], got {tuple(candidate_losses.shape)}"
        )
    if candidate_losses.device != best_losses.device or candidate_losses.dtype != best_losses.dtype:
        raise ValueError("best-of-K candidate and best losses must share dtype and device")
    if (
        winner_indices.shape != (batch_size,)
        or winner_indices.dtype != torch.long
        or winner_indices.device != best_losses.device
    ):
        raise ValueError("best-of-K winner indices must be int64 on the loss device with shape [B]")
    if candidate_noise.shape != winner_noise.shape or candidate_noise.shape[0] != batch_size:
        raise ValueError("best-of-K candidate and winner noise shapes must match [B, ...]")
    if candidate_noise.dtype != winner_noise.dtype or candidate_noise.device != winner_noise.device:
        raise ValueError("best-of-K candidate and winner noise must share dtype and device")
    nonfinite = (~torch.isfinite(candidate_losses)).nonzero(as_tuple=False).flatten()
    if nonfinite.numel():
        raise ValueError(
            f"candidate {candidate_index} has non-finite loss for sample indices {nonfinite.tolist()}"
        )

    improved = candidate_losses < best_losses
    noise_mask = improved.reshape(batch_size, *([1] * (candidate_noise.ndim - 1)))
    return (
        torch.where(improved, candidate_losses, best_losses),
        torch.where(noise_mask, candidate_noise, winner_noise),
        torch.where(improved, torch.full_like(winner_indices, candidate_index), winner_indices),
    )
```

- [ ] **Step 4: Add failing validation tests for malformed and non-finite candidates**

Append:

```python
@pytest.mark.parametrize("bad_losses", [torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("inf")])])
def test_update_winners_rejects_nonfinite_candidate_scores(bad_losses):
    with pytest.raises(ValueError, match=r"candidate 2.*sample indices \[1\]"):
        update_winners(
            torch.full((2,), torch.inf),
            torch.empty(2, 1),
            torch.full((2,), -1, dtype=torch.long),
            bad_losses,
            torch.zeros(2, 1),
            2,
        )


def test_update_winners_rejects_batch_reduced_candidate_score():
    with pytest.raises(ValueError, match=r"shape \[2\]"):
        update_winners(
            torch.full((2,), torch.inf),
            torch.empty(2, 1),
            torch.full((2,), -1, dtype=torch.long),
            torch.tensor(1.0),
            torch.zeros(2, 1),
            0,
        )


def test_update_winners_keeps_lower_index_on_equal_loss():
    best, noise, indices = update_winners(
        torch.tensor([1.0]),
        torch.tensor([[10.0]]),
        torch.tensor([0], dtype=torch.long),
        torch.tensor([1.0]),
        torch.tensor([[20.0]]),
        1,
    )
    torch.testing.assert_close(best, torch.tensor([1.0]))
    torch.testing.assert_close(noise, torch.tensor([[10.0]]))
    assert indices.tolist() == [0]


def test_update_winners_rejects_noise_shape_or_dtype_mismatch():
    common = (
        torch.full((2,), torch.inf),
        torch.empty(2, 1, dtype=torch.float32),
        torch.full((2,), -1, dtype=torch.long),
        torch.tensor([1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="shapes must match"):
        update_winners(*common, torch.zeros(2, 2), 0)
    with pytest.raises(ValueError, match="share dtype and device"):
        update_winners(*common, torch.zeros(2, 1, dtype=torch.float64), 0)
```

- [ ] **Step 5: Run the helper tests and verify GREEN**

Run the Step 2 command. Expected: all helper tests pass.

- [ ] **Step 6: Commit the pure mechanics**

```powershell
git add src/musubi_tuner/training/explorative.py tests/test_explorative_modeling.py
git commit -m "feat: add best-of-k winner mechanics"
```

### Task 2: Fixed-Timestep Noise Coefficients

**Files:**
- Modify: `src/musubi_tuner/training/timesteps.py`
- Modify: `src/musubi_tuner/training/trainer_base.py`
- Modify: `tests/test_explorative_modeling.py`

**Interfaces:**
- Produces: `BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS: frozenset[str]` as the single code-owned explicit-coefficient membership list.
- Produces: `get_noise_coefficients_from_timesteps(timestep_sampling, noise_scheduler, timesteps, device, n_dim, dtype) -> torch.Tensor`.
- Guarantees: explicit samplers invert model-visible `1000 * sigma + 1`; scheduler-indexed samplers delegate to `get_sigmas`.

- [ ] **Step 1: Write failing fixed-timestep reconstruction tests**

Append these imports and helpers to `tests/test_explorative_modeling.py`:

```python
from types import SimpleNamespace

from musubi_tuner.training.timesteps import (
    BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS,
    get_noise_coefficients_from_timesteps,
)
from musubi_tuner.training.trainer_base import NetworkTrainer


def _timestep_args(timestep_sampling: str) -> SimpleNamespace:
    return SimpleNamespace(
        timestep_sampling=timestep_sampling,
        discrete_flow_shift=3.0,
        sigmoid_scale=1.0,
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
    )


@pytest.mark.parametrize("timestep_sampling", sorted(BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS))
def test_explicit_sampler_candidate_zero_reconstructs_from_returned_timestep(timestep_sampling):
    torch.manual_seed(9)
    trainer = NetworkTrainer()
    latents = torch.randn(2, 3, 2, 2, dtype=torch.float32)
    noise = torch.randn_like(latents)
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
        _timestep_args(timestep_sampling),
        noise,
        latents,
        [0.25, 0.75],
        None,
        torch.device("cpu"),
        torch.float32,
    )
    sigma = get_noise_coefficients_from_timesteps(
        timestep_sampling,
        None,
        timesteps,
        torch.device("cpu"),
        latents.ndim,
        latents.dtype,
    )

    reconstructed = (1.0 - sigma) * latents + sigma * noise
    torch.testing.assert_close(reconstructed, noisy, rtol=1e-5, atol=1e-6)


def test_scheduler_indexed_coefficients_reuse_fixed_scheduler_timestep():
    scheduler = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 500.0, 1.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    sigma = get_noise_coefficients_from_timesteps(
        "sigma",
        scheduler,
        torch.tensor([500.0, 1.0]),
        torch.device("cpu"),
        5,
        torch.float32,
    )

    assert sigma.shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(sigma[:, 0, 0, 0, 0], torch.tensor([0.5, 0.0]))
```

Also prove reconstruction against the baseline scheduler branch rather than only checking the helper in isolation:

```python
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_scheduler_candidate_zero_reconstructs_with_declared_dtype_tolerance(monkeypatch, dtype, rtol, atol):
    import musubi_tuner.training.trainer_base as trainer_base_module

    scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=3),
        timesteps=torch.tensor([1000.0, 500.0, 1.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    monkeypatch.setattr(
        trainer_base_module,
        "compute_density_for_timestep_sampling",
        lambda **kwargs: torch.tensor([0.34, 0.67]),
    )
    trainer = NetworkTrainer()
    latents = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32).to(dtype).reshape(2, 1, 2, 2)
    noise = torch.flip(latents, dims=(-1,))
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
        _timestep_args("sigma"), noise, latents, None, scheduler, torch.device("cpu"), dtype
    )
    sigma = get_noise_coefficients_from_timesteps(
        "sigma", scheduler, timesteps, torch.device("cpu"), latents.ndim, dtype
    )

    torch.testing.assert_close(
        (1.0 - sigma) * latents + sigma * noise,
        noisy,
        rtol=rtol,
        atol=atol,
    )
```

- [ ] **Step 2: Run the noising tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py -k 'coefficient or reconstructs' -v
```

Expected: import failure because the shared constant and coefficient helper do not exist.

- [ ] **Step 3: Add the shared sampler set and coefficient helper**

Add to `training/timesteps.py` next to `get_sigmas`:

```python
BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS = frozenset(
    {
        "uniform",
        "sigmoid",
        "shift",
        "flux_shift",
        "flux2_shift",
        "ideogram4_shift",
        "qwen_shift",
        "krea2_shift",
        "logsnr",
        "qinglong_flux",
        "qinglong_qwen",
    }
)


def get_noise_coefficients_from_timesteps(
    timestep_sampling: str,
    noise_scheduler,
    timesteps: torch.Tensor,
    device: torch.device,
    n_dim: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if timestep_sampling in BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS:
        sigma = ((timesteps.to(device=device, dtype=dtype) - 1.0) / 1000.0).clamp_(0.0, 1.0)
        while sigma.ndim < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    return get_sigmas(noise_scheduler, timesteps, device, n_dim=n_dim, dtype=dtype)
```

Import the constant into `trainer_base.py` and replace only the outer eleven-term condition at the start of `get_noisy_model_input_and_timesteps` with:

```python
if args.timestep_sampling in BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS:
```

Leave Ideogram 4 and HiDream's architecture-local sampler subsets unchanged.

- [ ] **Step 4: Add dtype and broadcast tolerance tests**

Append:

```python
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
@pytest.mark.parametrize("shape", [(2, 3, 4, 4), (2, 3, 2, 4, 4)])
def test_explicit_coefficients_broadcast_with_declared_dtype_tolerance(dtype, rtol, atol, shape):
    latents = (
        torch.linspace(
            -1.0,
            1.0,
            int(torch.tensor(shape).prod().item()),
            dtype=torch.float32,
        )
        .to(dtype)
        .reshape(shape)
    )
    noise = torch.flip(latents, dims=(-1,))
    timesteps = torch.tensor([251.0, 751.0], dtype=torch.float32)
    sigma = get_noise_coefficients_from_timesteps(
        "uniform", None, timesteps, torch.device("cpu"), len(shape), dtype
    )
    expected_sigma = torch.tensor([0.25, 0.75], dtype=dtype).reshape(2, *([1] * (len(shape) - 1)))

    torch.testing.assert_close(sigma, expected_sigma, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        (1.0 - sigma) * latents + sigma * noise,
        (1.0 - expected_sigma) * latents + expected_sigma * noise,
        rtol=rtol,
        atol=atol,
    )
```

- [ ] **Step 5: Run noising and existing timestep tests**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py tests/test_ideogram4_timesteps.py tests/test_krea2_timesteps.py -v
```

Expected: all tests pass, and existing sampler behavior remains unchanged.

- [ ] **Step 6: Commit fixed-timestep reconstruction**

```powershell
git add src/musubi_tuner/training/timesteps.py src/musubi_tuner/training/trainer_base.py tests/test_explorative_modeling.py
git commit -m "feat: reconstruct fixed timestep noise coefficients"
```

### Task 3: Common CLI, Early Validation, And Training Dispatch

**Files:**
- Modify: `src/musubi_tuner/training/parser_common.py`
- Modify: `src/musubi_tuner/training/trainer_base.py`
- Modify: `src/musubi_tuner/flux_2_train_network_self_flow.py`
- Modify: `src/musubi_tuner/hidream_o1_train_network.py`
- Modify: `tests/test_explorative_modeling.py`

**Interfaces:**
- Produces: common `--xm_best_of_k INT`, default `1`.
- Produces: `get_best_of_k_count`, `get_best_of_k_option_name`, `get_best_of_k_incompatibility_reason`, `on_best_of_k_enabled`, `_validate_and_init_best_of_k`, and `_process_batch_for_training` trainer methods.
- Guarantees: validation occurs after architecture arguments are known but before dataset/model allocation; `K = 1` calls the original `process_batch` directly; a custom `process_batch` is rejected unless its trainer explicitly handles compatibility.

- [ ] **Step 1: Write failing parser and configuration tests**

Add imports for `read_config_from_file` and `setup_parser_common`, then append:

```python
def test_common_parser_defaults_xm_best_of_k_to_one():
    args = setup_parser_common().parse_args([])
    assert args.xm_best_of_k == 1


def test_common_parser_accepts_xm_best_of_k_from_cli():
    args = setup_parser_common().parse_args(["--xm_best_of_k", "3"])
    assert args.xm_best_of_k == 3


def test_common_parser_accepts_xm_best_of_k_from_toml(tmp_path, monkeypatch):
    config = tmp_path / "xm.toml"
    config.write_text("xm_best_of_k = 4\n", encoding="utf-8")
    parser = setup_parser_common()
    monkeypatch.setattr(sys, "argv", ["trainer", "--config_file", str(config)])
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    assert args.xm_best_of_k == 4
```

- [ ] **Step 2: Write failing validation and dispatch tests**

Define small trainer classes in the test module:

```python
class _CompatibleTrainer(NetworkTrainer):
    @property
    def architecture(self):
        return "synthetic"

    @property
    def architecture_full_name(self):
        return "Synthetic"


class _CustomBatchTrainer(_CompatibleTrainer):
    def process_batch(self, *args, **kwargs):
        return torch.tensor(0.0), {}


class _ExplicitlyCompatibleCustomBatchTrainer(_CustomBatchTrainer):
    def get_best_of_k_incompatibility_reason(self, args):
        return None


def test_best_of_k_validation_rejects_values_below_one():
    trainer = _CompatibleTrainer()
    with pytest.raises(ValueError, match=r"--xm_best_of_k must be at least 1"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=0))


def test_best_of_k_validation_rejects_unconfirmed_custom_process_batch():
    trainer = _CustomBatchTrainer()
    with pytest.raises(ValueError, match=r"overrides process_batch"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))


def test_best_of_k_validation_accepts_explicit_custom_process_compatibility():
    trainer = _ExplicitlyCompatibleCustomBatchTrainer()
    trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))
    assert trainer._best_of_k_enabled is True


def test_training_dispatch_preserves_original_k_one_path(monkeypatch):
    trainer = _CompatibleTrainer()
    trainer._best_of_k_enabled = False
    calls = []
    monkeypatch.setattr(trainer, "process_batch", lambda *a, **k: calls.append("ordinary") or (torch.tensor(1.0), {}))
    monkeypatch.setattr(
        trainer,
        "process_batch_best_of_k",
        lambda *a, **k: calls.append("best-of-k") or (torch.tensor(2.0), {}),
    )
    state = torch.random.get_rng_state().clone()

    loss, metrics = trainer._process_batch_for_training(None)

    assert calls == ["ordinary"]
    assert loss.item() == 1.0
    assert metrics == {}
    assert torch.equal(torch.random.get_rng_state(), state)


def test_training_dispatch_uses_best_of_k_only_when_enabled(monkeypatch):
    trainer = _CompatibleTrainer()
    trainer._best_of_k_enabled = True
    calls = []
    monkeypatch.setattr(trainer, "process_batch", lambda *a, **k: calls.append("ordinary"))
    monkeypatch.setattr(
        trainer,
        "process_batch_best_of_k",
        lambda *a, **k: calls.append("best-of-k") or (torch.tensor(2.0), {"xm/selection_gain": 1.0}),
    )

    loss, metrics = trainer._process_batch_for_training(None)

    assert calls == ["best-of-k"]
    assert loss.item() == 2.0
    assert metrics == {"xm/selection_gain": 1.0}
```

Call `_process_batch_for_training(None)` deliberately: the dispatch method must forward `*args, **kwargs` unchanged and perform no work before choosing an arm.

- [ ] **Step 3: Run parser, validation, and dispatch tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py -k 'parser or validation or dispatch' -v
```

Expected: parser and trainer attributes are missing.

- [ ] **Step 4: Add the common option and trainer hooks**

Add this argument in `_add_timestep_args` immediately after the timestep selection controls:

```python
parser.add_argument(
    "--xm_best_of_k",
    type=int,
    default=1,
    help="Number of sequential Forward XM noise candidates; 1 disables XM. / "
    "逐次Forward XMノイズ候補数。1でXMを無効化します。",
)
```

Initialize the dispatch state in `NetworkTrainer.__init__`:

```python
self._best_of_k_count = 1
self._best_of_k_enabled = False
```

Add these methods at the start of the trainer extension section:

```python
def get_best_of_k_count(self, args: argparse.Namespace) -> int:
    return args.xm_best_of_k

def get_best_of_k_option_name(self, args: argparse.Namespace) -> str:
    return "--xm_best_of_k"

def get_best_of_k_incompatibility_reason(self, args: argparse.Namespace) -> Optional[str]:
    del args
    if type(self).process_batch is not NetworkTrainer.process_batch:
        return (
            f"{self.architecture_full_name} overrides process_batch and has not confirmed "
            "the standard Forward XM data-flow contract"
        )
    return None

def on_best_of_k_enabled(self, args: argparse.Namespace) -> None:
    del args
    multiplier = (self._best_of_k_count + 3) / 3
    logger.info(
        "Forward XM enabled for %s: K=%d, sequential memory-saving mode, "
        "approximate operation-count multiplier %.2fx",
        self.architecture_full_name,
        self._best_of_k_count,
        multiplier,
    )
    logger.warning(
        "Published Forward XM gains are pretraining results and have not been validated for LoRA fine-tuning."
    )

def _validate_and_init_best_of_k(self, args: argparse.Namespace) -> None:
    option_name = self.get_best_of_k_option_name(args)
    count = self.get_best_of_k_count(args)
    if count < 1:
        raise ValueError(f"{option_name} must be at least 1, got {count}")
    self._best_of_k_count = count
    self._best_of_k_enabled = count > 1
    if not self._best_of_k_enabled:
        return
    reason = self.get_best_of_k_incompatibility_reason(args)
    if reason is not None:
        raise ValueError(f"{option_name}={count} is not supported: {reason}")
    self.on_best_of_k_enabled(args)

def _process_batch_for_training(self, *args, **kwargs):
    if self._best_of_k_enabled:
        return self.process_batch_best_of_k(*args, **kwargs)
    return self.process_batch(*args, **kwargs)
```

Call `self._validate_and_init_best_of_k(args)` immediately after `self.handle_model_specific_args(args)` in `_validate_args_and_init`. Replace the training loop's `self.process_batch(...)` call with `self._process_batch_for_training(...)` without moving the existing `torch.randn_like(latents)` call.

Declare the temporary method below; Task 5 replaces its body before any supported `K > 1` run is complete:

```python
def process_batch_best_of_k(
    self,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
    network,
    batch: dict[str, torch.Tensor],
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    sample_resources,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    raise NotImplementedError("best-of-K processing is implemented in Task 5")
```

- [ ] **Step 5: Add precise Self-Flow and HiDream compatibility overrides**

In `Flux2SelfFlowNetworkTrainer`, add:

```python
def get_best_of_k_incompatibility_reason(self, args: argparse.Namespace) -> Optional[str]:
    if args.self_flow:
        return "--self_flow requires teacher/student candidate state and is not supported by Forward XM"
    return None
```

Replace the module's blanket not-runnable warning with:

```text
The vanilla Flux 2 branch is runnable when ``--self_flow`` is off. The enabled
Self-Flow algorithm remains an incomplete skeleton and must not be used until
its teacher/student step and lifecycle hooks are implemented.
```

In `HiDreamO1NetworkTrainer`, add:

```python
def get_best_of_k_incompatibility_reason(self, args: argparse.Namespace) -> Optional[str]:
    del args
    return (
        "HiDream-O1 uses candidate-local noise scaling/clipping and may add a batch-reduced DINO loss; "
        "its candidate state cannot be reconstructed by the standard Forward XM path"
    )
```

Add focused tests that instantiate each trainer with `__new__`, call `NetworkTrainer.__init__`, and assert:

```python
from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer
from musubi_tuner.hidream_o1_train_network import HiDreamO1NetworkTrainer


def test_architecture_specific_standard_xm_compatibility_reasons(monkeypatch):
    self_flow = Flux2SelfFlowNetworkTrainer.__new__(Flux2SelfFlowNetworkTrainer)
    NetworkTrainer.__init__(self_flow)
    hidream = HiDreamO1NetworkTrainer.__new__(HiDreamO1NetworkTrainer)
    NetworkTrainer.__init__(hidream)

    assert (
        self_flow.get_best_of_k_incompatibility_reason(
            SimpleNamespace(self_flow=False)
        )
        is None
    )
    assert "--self_flow" in self_flow.get_best_of_k_incompatibility_reason(
        SimpleNamespace(self_flow=True)
    )
    assert "noise scaling/clipping" in hidream.get_best_of_k_incompatibility_reason(
        SimpleNamespace()
    )

    for trainer in (self_flow, hidream):
        monkeypatch.setattr(
            trainer,
            "get_best_of_k_incompatibility_reason",
            lambda args: (_ for _ in ()).throw(AssertionError("hook called at K=1")),
        )
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=1))
        assert trainer._best_of_k_enabled is False
```

- [ ] **Step 6: Prove validation precedes allocation**

Add:

```python
def test_invalid_best_of_k_fails_before_session_or_dataset_allocation():
    class _EarlyValidationTrainer(_CompatibleTrainer):
        def __init__(self):
            super().__init__()
            self.events = []

        def handle_model_specific_args(self, args):
            self.events.append("handle_model_specific_args")

        def _init_session(self, args):
            self.events.append("allocation_started")
            raise AssertionError("session initialization must not start")

        def _build_dataset(self, args):
            raise AssertionError("dataset construction must not start")

    trainer = _EarlyValidationTrainer()
    args = SimpleNamespace(
        cuda_allow_tf32=False,
        cuda_cudnn_benchmark=False,
        dataset_config="unused.toml",
        dit="unused.safetensors",
        fp8_scaled=False,
        fp8_base=False,
        sage_attn=False,
        disable_numpy_memmap=False,
        show_timesteps=None,
        xm_best_of_k=0,
    )
    with pytest.raises(ValueError, match="--xm_best_of_k"):
        trainer.train(args)
    assert trainer.events == ["handle_model_specific_args"]


def test_standard_xm_startup_log_is_explicit_and_k_one_is_silent(caplog):
    caplog.set_level("INFO")
    trainer = _CompatibleTrainer()
    trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=1))
    assert "Forward XM enabled" not in caplog.text

    trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))
    assert "Forward XM enabled for Synthetic" in caplog.text
    assert "K=2" in caplog.text
    assert "sequential memory-saving mode" in caplog.text
    assert "1.67x" in caplog.text
    assert "pretraining results" in caplog.text
    assert "LoRA fine-tuning" in caplog.text
```

- [ ] **Step 7: Run tests and commit common integration**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py -k 'parser or validation or dispatch or self_flow or hidream' -v
```

Expected: all selected tests pass.

```powershell
git add src/musubi_tuner/training/parser_common.py src/musubi_tuner/training/trainer_base.py src/musubi_tuner/flux_2_train_network_self_flow.py src/musubi_tuner/hidream_o1_train_network.py tests/test_explorative_modeling.py
git commit -m "feat: validate and dispatch explorative training"
```

### Task 4: Canonical Per-Sample Training Losses

**Files:**
- Modify: `src/musubi_tuner/training/trainer_base.py`
- Modify: `src/musubi_tuner/ideogram4_train_network.py`
- Modify: `tests/test_explorative_modeling.py`
- Modify: `tests/test_ideogram4_synthetic.py`

**Interfaces:**
- Produces: `NetworkTrainer.compute_per_sample_loss(...) -> torch.Tensor` with exact shape `[B]`.
- Produces: `Ideogram4NetworkTrainer.compute_per_sample_loss(...) -> torch.Tensor` with unweighted Ideogram 4 MSE.
- Guarantees: each scalar `compute_loss` is the mean of its canonical vector; no XM-only loss copy exists.

- [ ] **Step 1: Write failing base loss tests**

Append:

```python
from musubi_tuner.training.trainer_base import DiTOutput
import musubi_tuner.training.trainer_base as trainer_base_module


def test_base_per_sample_loss_applies_weight_before_nonbatch_reduction(monkeypatch):
    trainer = NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([[[[1.0, 3.0]]], [[[2.0, 4.0]]]]),
        target=torch.zeros(2, 1, 1, 2),
    )
    weighting = torch.tensor([2.0, 0.5]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(trainer_base_module, "compute_loss_weighting_for_sd3", lambda *a, **k: weighting)
    args = SimpleNamespace(weighting_scheme="cosmap")

    per_sample = trainer.compute_per_sample_loss(
        args, output, torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0
    )
    scalar, metrics = trainer.compute_loss(
        args, output, torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0
    )

    torch.testing.assert_close(per_sample, torch.tensor([10.0, 5.0]))
    assert torch.allclose(scalar, per_sample.mean(), rtol=1e-5, atol=1e-8)
    assert metrics == {}


def test_base_per_sample_loss_rejects_missing_batch_axis():
    trainer = NetworkTrainer()
    output = DiTOutput(pred=torch.tensor(1.0), target=torch.tensor(0.0))
    with pytest.raises(ValueError, match=r"per-sample loss requires a leading batch axis"):
        trainer.compute_per_sample_loss(
            SimpleNamespace(weighting_scheme="none"), output, torch.tensor([1.0]), None, torch.float32, torch.float32, 0
        )
```

Add the standard-trainer K=1 mathematical compatibility proof; do not use exact equality:

```python
def test_base_scalar_loss_and_gradient_match_direct_baseline_reduction(monkeypatch):
    trainer = NetworkTrainer()
    weighting = torch.tensor([0.75, 1.25]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(
        trainer_base_module,
        "compute_loss_weighting_for_sd3",
        lambda *args, **kwargs: weighting,
    )
    new_parameter = torch.nn.Parameter(torch.tensor(0.3))
    old_parameter = torch.nn.Parameter(new_parameter.detach().clone())
    inputs = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1)
    target = torch.tensor([-1.0, 0.5]).reshape(2, 1, 1, 1)
    timesteps = torch.tensor([100.0, 900.0])
    args = SimpleNamespace(weighting_scheme="cosmap")

    new_loss, metrics = trainer.compute_loss(
        args,
        DiTOutput(pred=new_parameter * inputs, target=target),
        timesteps,
        None,
        torch.float32,
        torch.float32,
        0,
    )
    old_elementwise = torch.nn.functional.mse_loss(
        old_parameter * inputs, target, reduction="none"
    )
    old_loss = (old_elementwise * weighting).mean()
    new_grad = torch.autograd.grad(new_loss, new_parameter)[0]
    old_grad = torch.autograd.grad(old_loss, old_parameter)[0]

    assert torch.allclose(new_loss, old_loss, rtol=1e-5, atol=1e-8)
    assert torch.allclose(new_grad, old_grad, rtol=1e-5, atol=1e-8)
    assert metrics == {}
```

- [ ] **Step 2: Extend the existing Ideogram synthetic loss test**

In the existing test that loads `Ideogram4NetworkTrainer`, calculate both forms from the same `DiTOutput`:

```python
per_sample = trainer.compute_per_sample_loss(
    args, output, timesteps, None, torch.float32, torch.float32, 0
)
loss, metrics = trainer.compute_loss(
    args, output, timesteps, None, torch.float32, torch.float32, 0
)

assert per_sample.shape == (output.pred.shape[0],)
assert torch.allclose(loss, per_sample.mean(), rtol=1e-5, atol=1e-8)
assert set(metrics) == {
    "loss/zero_pred",
    "loss/flipped_pred",
    "loss/pred_rms",
    "loss/target_rms",
    "loss/pred_target_cosine",
    "timestep/mean",
    "timestep/min",
    "timestep/max",
}
```

- [ ] **Step 3: Run the loss tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py -k 'per_sample or compute_loss' -v
```

Expected: `compute_per_sample_loss` is missing.

- [ ] **Step 4: Implement the base primitive and scalar wrapper**

Add beside `compute_loss`:

```python
def compute_per_sample_loss(
    self,
    args: argparse.Namespace,
    output: DiTOutput,
    timesteps: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    global_step: int,
) -> torch.Tensor:
    del global_step
    if output.pred.ndim < 1 or output.target.ndim < 1:
        raise ValueError("per-sample loss requires a leading batch axis")
    batch_size = output.pred.shape[0]
    if output.target.shape[0] != batch_size:
        raise ValueError("prediction and target batch sizes differ")
    weighting = compute_loss_weighting_for_sd3(
        args.weighting_scheme, noise_scheduler, timesteps, timesteps.device, dit_dtype
    )
    elementwise = torch.nn.functional.mse_loss(
        output.pred.to(network_dtype), output.target, reduction="none"
    )
    if weighting is not None:
        elementwise = elementwise * weighting
    per_sample = elementwise.reshape(batch_size, -1).mean(dim=1)
    if per_sample.shape != (batch_size,):
        raise ValueError(f"per-sample loss must have shape [{batch_size}], got {tuple(per_sample.shape)}")
    return per_sample
```

Replace the base `compute_loss` formula with:

```python
per_sample = self.compute_per_sample_loss(
    args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step
)
return per_sample.mean(), {}
```

- [ ] **Step 5: Implement Ideogram 4's single loss primitive**

Move only the unweighted elementwise MSE and non-batch reduction to:

```python
def compute_per_sample_loss(
    self,
    args,
    output,
    timesteps,
    noise_scheduler,
    dit_dtype,
    network_dtype,
    global_step,
):
    del args, timesteps, noise_scheduler, dit_dtype, global_step
    pred = output.pred.to(network_dtype)
    target = output.target.to(network_dtype)
    if pred.ndim < 1 or target.ndim < 1 or pred.shape[0] != target.shape[0]:
        raise ValueError("Ideogram 4 per-sample loss requires matching leading batch axes")
    return torch.nn.functional.mse_loss(pred, target, reduction="none").reshape(pred.shape[0], -1).mean(dim=1)
```

Keep all existing diagnostic calculations in `compute_loss`, but obtain `loss` only through:

```python
loss = self.compute_per_sample_loss(
    args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step
).mean()
```

- [ ] **Step 6: Run loss and regression tests, then commit**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py -v
```

Expected: scalar/vector comparisons pass with the declared tolerance and Ideogram metrics remain unchanged.

```powershell
git add src/musubi_tuner/training/trainer_base.py src/musubi_tuner/ideogram4_train_network.py tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py
git commit -m "refactor: expose canonical per-sample training losses"
```

### Task 5: Sequential Standard Forward XM

**Files:**
- Modify: `src/musubi_tuner/training/trainer_base.py`
- Modify: `tests/test_explorative_modeling.py`

**Interfaces:**
- Consumes: Task 1's `create_candidate_generator`, `draw_candidate_noise`, and `update_winners`; Task 2's coefficient helper; Task 3's dispatch state; Task 4's `compute_per_sample_loss`.
- Implements: `NetworkTrainer.process_batch_best_of_k` with K no-grad candidate forwards and one gradient-enabled winner forward.
- Guarantees: fixed timestep/conditioning, candidate-derived target pairing, per-sample mixed winners, direct RNG replay, constant candidate storage, and `xm/` metrics.

- [ ] **Step 1: Add a deterministic synthetic trainer fixture**

Add these imports and fixture code:

```python
import gc
import weakref
from contextlib import nullcontext


class _ToyAccelerator:
    def __init__(self, device="cpu", autocast_dtype=None):
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype

    def autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)


class _ToyTransformer:
    def __init__(self):
        self.forward_shapes = []
        self.block_swap_calls = 0
        self.checkpoint_calls = 0

    def __call__(self, value):
        self.forward_shapes.append(tuple(value.shape))
        self.block_swap_calls += 1
        self.checkpoint_calls += 1
        return value


class _ToyXMTrainer(_CompatibleTrainer):
    def __init__(self, device="cpu"):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.records = []
        self.output_refs = []
        self.noising_calls = 0

    def get_noisy_model_input_and_timesteps(self, *args, **kwargs):
        self.noising_calls += 1
        return super().get_noisy_model_input_and_timesteps(*args, **kwargs)

    def call_dit(
        self,
        args,
        accelerator,
        transformer,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
        **kwargs,
    ):
        del args, kwargs
        cpu_mask = torch.rand((), device="cpu")
        device_mask = (
            torch.rand((), device=accelerator.device)
            if accelerator.device.type == "cuda"
            else torch.rand((), device="cpu")
        )
        with accelerator.autocast():
            features = transformer(noisy_model_input)
            prediction = self.scale.to(network_dtype) * features.to(network_dtype)
            target = (latents - noise).to(network_dtype)
        output = DiTOutput(pred=prediction, target=target)
        self.output_refs.append(weakref.ref(output.pred))
        self.records.append(
            {
                "noise": noise.detach().clone(),
                "noisy_model_input": noisy_model_input.detach().clone(),
                "timesteps": timesteps.detach().clone(),
                "condition": batch["condition"].detach().clone(),
                "target": target.detach().clone(),
                "grad_enabled": torch.is_grad_enabled(),
                "cpu_mask": cpu_mask.detach().clone(),
                "device_mask": device_mask.detach().cpu().clone(),
            }
        )
        return output


def _xm_args(best_of_k=2):
    args = _timestep_args("uniform")
    args.xm_best_of_k = best_of_k
    args.gradient_checkpointing = True
    return args


def _run_toy_xm(
    monkeypatch,
    device="cpu",
    autocast_dtype=None,
    transformer_factory=_ToyTransformer,
    trainer_factory=_ToyXMTrainer,
    args=None,
):
    accelerator = _ToyAccelerator(device, autocast_dtype)
    trainer = trainer_factory(device)
    trainer._best_of_k_count = 2
    trainer._best_of_k_enabled = True
    transformer = transformer_factory()
    latents = torch.zeros(2, 1, 1, 1, device=device)
    candidate_zero = torch.tensor([1.0, 4.0], device=device).reshape(2, 1, 1, 1)
    candidate_one = torch.tensor([5.0, 2.0], device=device).reshape(2, 1, 1, 1)
    monkeypatch.setattr(
        trainer_base_module,
        "draw_candidate_noise",
        lambda reference, generator: candidate_one.to(dtype=reference.dtype),
    )
    batch = {
        "timesteps": [0.5, 0.5],
        "condition": torch.tensor([[7.0], [11.0]], device=device),
    }
    loss, metrics = trainer.process_batch_best_of_k(
        args or _xm_args(),
        accelerator,
        transformer,
        None,
        batch,
        latents,
        candidate_zero,
        None,
        autocast_dtype or torch.float32,
        autocast_dtype or torch.float32,
        None,
        0,
    )
    return trainer, transformer, loss, metrics
```

Use two samples with zero latents, fixed raw timesteps `[0.5, 0.5]`, candidate-zero noise values `[1.0, 4.0]`, and a monkeypatched candidate-one draw `[5.0, 2.0]`, each expanded to shape `[2, 1, 1, 1]`. Set `scale=0`; the expected winners are candidate indices `[0, 1]`, candidate score mean is `11.5`, selection gain is `6.0`, and the final scalar loss is `2.5`.

- [ ] **Step 2: Write failing mixed-winner and gradient tests**

Add:

```python
def test_standard_xm_selects_mixed_winners_and_builds_one_gradient_graph(monkeypatch):
    trainer, transformer, loss, metrics = _run_toy_xm(monkeypatch)

    assert [record["grad_enabled"] for record in trainer.records] == [False, False, True]
    torch.testing.assert_close(
        trainer.records[-1]["noise"][:, 0, 0, 0], torch.tensor([1.0, 2.0])
    )
    torch.testing.assert_close(
        trainer.records[-1]["target"], -trainer.records[-1]["noise"]
    )
    assert all(
        torch.equal(record["timesteps"], trainer.records[0]["timesteps"])
        for record in trainer.records
    )
    assert all(
        torch.equal(record["condition"], trainer.records[0]["condition"])
        for record in trainer.records
    )
    assert torch.allclose(loss, torch.tensor(2.5), rtol=1e-5, atol=1e-8)
    assert metrics == {
        "xm/candidate_loss_mean": 11.5,
        "xm/selection_gain": 6.0,
    }
    assert transformer.forward_shapes == [(2, 1, 1, 1)] * 3
    assert transformer.block_swap_calls == 3
    assert transformer.checkpoint_calls == 3
    assert trainer.noising_calls == 1

    gc.collect()
    assert trainer.output_refs[0]() is None
    assert trainer.output_refs[1]() is None
    assert loss.grad_fn is not None

    class _BackwardRecorder:
        def __init__(self):
            self.calls = 0

        def backward(self, value):
            self.calls += 1
            value.backward()

    backward = _BackwardRecorder()
    backward.backward(loss)
    assert backward.calls == 1
    assert trainer.scale.grad is not None
    assert torch.isfinite(trainer.scale.grad)
    assert trainer.scale.grad.item() == pytest.approx(2.5)
```

Add:

```python
@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_standard_xm_compiled_path_keeps_original_microbatch_shape(monkeypatch):
    class _CompiledToyTransformer(_ToyTransformer):
        def __init__(self):
            super().__init__()
            self.compiled_identity = torch.compile(lambda value: value, backend="eager")

        def __call__(self, value):
            self.forward_shapes.append(tuple(value.shape))
            self.block_swap_calls += 1
            self.checkpoint_calls += 1
            return self.compiled_identity(value)

    _, transformer, _, _ = _run_toy_xm(
        monkeypatch, transformer_factory=_CompiledToyTransformer
    )
    assert transformer.forward_shapes == [(2, 1, 1, 1)] * 3
```

- [ ] **Step 3: Write failing RNG replay and reproducibility tests**

Add this CPU/CUDA test. It intentionally touches only PyTorch RNG APIs:

```python
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_standard_xm_replays_forward_rng_and_advances_final_forward_once(monkeypatch, device):
    torch.manual_seed(1234)
    if device == "cuda":
        torch.cuda.manual_seed_all(1234)
    initial_cpu = torch.random.get_rng_state().clone()
    initial_device = (
        torch.cuda.get_rng_state(torch.device(device)).clone()
        if device == "cuda"
        else None
    )
    captured = {}
    real_create = trainer_base_module.create_candidate_generator

    def capture_after_generator(reference):
        generator = real_create(reference)
        captured["cpu"] = torch.random.get_rng_state().clone()
        if reference.device.type == "cuda":
            captured["device"] = torch.cuda.get_rng_state(reference.device).clone()
        return generator

    monkeypatch.setattr(
        trainer_base_module, "create_candidate_generator", capture_after_generator
    )
    trainer, _, _, metrics = _run_toy_xm(monkeypatch, device=device)
    first_winner = trainer.records[-1]["noise"].clone()
    first_post_cpu = torch.random.get_rng_state().clone()
    first_post_device = (
        torch.cuda.get_rng_state(torch.device(device)).clone()
        if device == "cuda"
        else None
    )

    for record in trainer.records[1:]:
        torch.testing.assert_close(record["cpu_mask"], trainer.records[0]["cpu_mask"])
        torch.testing.assert_close(record["device_mask"], trainer.records[0]["device_mask"])

    torch.random.set_rng_state(captured["cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state(captured["device"], torch.device(device))
    torch.rand((), device="cpu")
    torch.rand((), device=torch.device(device) if device == "cuda" else "cpu")
    assert torch.equal(torch.random.get_rng_state(), first_post_cpu)
    if device == "cuda":
        assert torch.equal(
            torch.cuda.get_rng_state(torch.device(device)), first_post_device
        )

    torch.random.set_rng_state(initial_cpu)
    if device == "cuda":
        torch.cuda.set_rng_state(initial_device, torch.device(device))
    replay_trainer, _, _, replay_metrics = _run_toy_xm(monkeypatch, device=device)
    torch.testing.assert_close(replay_trainer.records[-1]["noise"], first_winner)
    assert replay_metrics == metrics
    assert torch.equal(torch.random.get_rng_state(), first_post_cpu)
    if device == "cuda":
        assert torch.equal(
            torch.cuda.get_rng_state(torch.device(device)), first_post_device
        )
```

The code does not import or snapshot Python `random` or NumPy state.

Add this focused Wan fixture and test without loading weights:

```python
from musubi_tuner.wan_train_network import WanNetworkTrainer


class _ToyWanTrainer(WanNetworkTrainer):
    def __init__(self):
        NetworkTrainer.__init__(self)
        self.high_low_training = True
        self.timestep_boundary = 0.5
        self.num_timestep_buckets = 1
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.noising_calls = 0
        self.routes = []
        self.forward_timesteps = []

    def get_bucketed_timestep(self):
        return 0.75

    def get_noisy_model_input_and_timesteps(self, *args, **kwargs):
        self.noising_calls += 1
        return super().get_noisy_model_input_and_timesteps(*args, **kwargs)

    def call_dit(
        self,
        args,
        accelerator,
        transformer,
        latents,
        batch,
        noise,
        noisy_model_input,
        timesteps,
        network_dtype,
        **kwargs,
    ):
        del args, accelerator, transformer, batch, kwargs
        self.routes.append(bool(self.next_model_is_high_noise))
        self.forward_timesteps.append(timesteps.detach().clone())
        return DiTOutput(
            pred=self.scale.to(network_dtype) * noisy_model_input.to(network_dtype),
            target=(latents - noise).to(network_dtype),
        )


def test_standard_xm_freezes_wan_high_low_route_and_timesteps(monkeypatch):
    trainer = _ToyWanTrainer()
    trainer._best_of_k_count = 2
    trainer._best_of_k_enabled = True
    latents = torch.zeros(2, 1, 1, 1)
    noise = torch.tensor([1.0, 4.0]).reshape(2, 1, 1, 1)
    later = torch.tensor([5.0, 2.0]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(
        trainer_base_module,
        "draw_candidate_noise",
        lambda reference, generator: later,
    )
    trainer.process_batch_best_of_k(
        _xm_args(),
        _ToyAccelerator(),
        None,
        None,
        {"timesteps": [0.75, 0.75], "condition": torch.ones(2, 1)},
        latents,
        noise,
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert trainer.noising_calls == 1
    assert trainer.routes == [True, True, True]
    assert all(
        torch.equal(value, trainer.forward_timesteps[0])
        for value in trainer.forward_timesteps[1:]
    )
```

Add the direct distribution-preserving regression:

```python
def test_standard_xm_samples_distribution_preserving_timestep_once(monkeypatch):
    args = _xm_args()
    args.preserve_distribution_shape = True
    trainer, _, _, _ = _run_toy_xm(monkeypatch, args=args)
    assert trainer.noising_calls == 1
```

- [ ] **Step 4: Write failing score validation and weighted/global-selection tests**

Add:

```python
def test_standard_xm_prefixes_nonfinite_candidate_diagnostics(monkeypatch):
    class _NaNTrainer(_ToyXMTrainer):
        def __init__(self, device="cpu"):
            super().__init__(device)
            self.score_calls = 0

        def compute_per_sample_loss(self, *args, **kwargs):
            losses = super().compute_per_sample_loss(*args, **kwargs)
            if self.score_calls == 1:
                losses = losses.clone()
                losses[1] = torch.nan
            self.score_calls += 1
            return losses

    with pytest.raises(
        ValueError,
        match=r"Synthetic.*candidate 1.*sample indices \[1\]",
    ):
        _run_toy_xm(monkeypatch, trainer_factory=_NaNTrainer)


def test_weighted_per_sample_selection_is_not_a_whole_candidate_reduction(monkeypatch):
    trainer = NetworkTrainer()
    weighting = torch.tensor([10.0, 1.0]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(
        trainer_base_module,
        "compute_loss_weighting_for_sd3",
        lambda *args, **kwargs: weighting,
    )
    args = SimpleNamespace(weighting_scheme="cosmap")
    timesteps = torch.tensor([100.0, 900.0])

    def score(unweighted):
        output = DiTOutput(
            pred=torch.sqrt(torch.tensor(unweighted)).reshape(2, 1, 1, 1),
            target=torch.zeros(2, 1, 1, 1),
        )
        return trainer.compute_per_sample_loss(
            args, output, timesteps, None, torch.float32, torch.float32, 0
        )

    candidate_zero = score([1.0, 4.0])
    candidate_one = score([2.0, 1.0])
    best = torch.full((2,), torch.inf)
    winner_noise = torch.empty(2, 1)
    indices = torch.full((2,), -1, dtype=torch.long)
    best, winner_noise, indices = update_winners(
        best, winner_noise, indices, candidate_zero, torch.tensor([[0.0], [0.0]]), 0
    )
    best, winner_noise, indices = update_winners(
        best, winner_noise, indices, candidate_one, torch.tensor([[1.0], [1.0]]), 1
    )

    assert indices.tolist() == [0, 1]
    assert candidate_zero.mean() < candidate_one.mean()
```

The last assertion shows that an incorrect whole-candidate reduction would choose candidate zero. The test does not claim that a positive fixed per-sample weight changes one sample's candidate ordering.

- [ ] **Step 5: Run the standard XM tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py -k 'xm or candidate or rng or weighted' -v
```

Expected: `process_batch_best_of_k` is not implemented.

- [ ] **Step 6: Implement the sequential candidate loop**

Import the three pure helpers and `get_noise_coefficients_from_timesteps`. Replace the temporary `NotImplementedError` with:

```python
def process_batch_best_of_k(
    self,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
    network,
    batch: dict[str, torch.Tensor],
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    sample_resources,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    del network, sample_resources
    noisy_candidate_zero, timesteps = self.get_noisy_model_input_and_timesteps(
        args,
        noise,
        latents,
        batch["timesteps"],
        noise_scheduler,
        accelerator.device,
        dit_dtype,
    )
    sigma = get_noise_coefficients_from_timesteps(
        args.timestep_sampling,
        noise_scheduler,
        timesteps,
        accelerator.device,
        latents.ndim,
        dit_dtype,
    )
    generator = create_candidate_generator(noise)
    batch_size = latents.shape[0]
    best_losses = torch.full((batch_size,), torch.inf, device=latents.device, dtype=torch.float32)
    winner_noise = torch.empty_like(noise)
    winner_indices = torch.full((batch_size,), -1, device=latents.device, dtype=torch.long)
    candidate_loss_sum = torch.zeros((), device=latents.device, dtype=torch.float32)
    candidate_zero_mean = None
    device = torch.device(accelerator.device)
    fork_devices = [device] if device.type == "cuda" else []

    for candidate_index in range(self._best_of_k_count):
        if candidate_index == 0:
            candidate_noise = noise
            candidate_input = noisy_candidate_zero
        else:
            candidate_noise = draw_candidate_noise(noise, generator)
            candidate_input = (1.0 - sigma) * latents + sigma * candidate_noise

        with torch.random.fork_rng(devices=fork_devices):
            with torch.no_grad():
                output = self.call_dit(
                    args,
                    accelerator,
                    transformer,
                    latents,
                    batch,
                    candidate_noise,
                    candidate_input,
                    timesteps,
                    network_dtype,
                )
                candidate_losses = self.compute_per_sample_loss(
                    args,
                    output,
                    timesteps,
                    noise_scheduler,
                    dit_dtype,
                    network_dtype,
                    global_step,
                )
        candidate_losses_f32 = candidate_losses.detach().float()
        candidate_loss_sum = candidate_loss_sum + candidate_losses_f32.sum()
        if candidate_index == 0:
            candidate_zero_mean = candidate_losses_f32.mean()
        try:
            best_losses, winner_noise, winner_indices = update_winners(
                best_losses,
                winner_noise,
                winner_indices,
                candidate_losses_f32,
                candidate_noise,
                candidate_index,
            )
        except ValueError as error:
            raise ValueError(f"{self.architecture_full_name}: {error}") from error

    # Samples may choose different candidates because sigma is per sample and fixed across candidates.
    winner_input = (1.0 - sigma) * latents + sigma * winner_noise
    output = self.call_dit(
        args,
        accelerator,
        transformer,
        latents,
        batch,
        winner_noise,
        winner_input,
        timesteps,
        network_dtype,
    )
    loss, metrics = self.compute_loss(
        args,
        output,
        timesteps,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        global_step,
    )
    assert candidate_zero_mean is not None
    return loss, {
        **metrics,
        "xm/candidate_loss_mean": (
            candidate_loss_sum / (self._best_of_k_count * batch_size)
        ).item(),
        "xm/selection_gain": (
            candidate_zero_mean - best_losses.detach().float().mean()
        ).item(),
    }
```

Review the implementation against this order:

1. Call `get_noisy_model_input_and_timesteps` once for the incoming candidate-zero `noise`.
2. Reconstruct one broadcast `sigma` from the returned timesteps.
3. Create the private generator, then initialize `best_losses=[inf]`, `winner_noise=empty_like(noise)`, and `winner_indices=[-1]`.
4. For each candidate, use the incoming tensors for index zero or draw one same-shaped noise and reconstruct its input with `(1 - sigma) * latents + sigma * candidate_noise`.
5. Enter `torch.random.fork_rng(devices=[device] if device.type == "cuda" else [])`, then `torch.no_grad()`, call `call_dit`, and call `compute_per_sample_loss`.
6. Prefix `update_winners` errors with `self.architecture_full_name` without swallowing the candidate/sample diagnostics.
7. Accumulate only detached float32 score sums, the candidate-zero mean, and the three `[B]`/`[B,...]` winner tensors.
8. Build the final noisy input and add this exact orienting comment:

```python
# Samples may choose different candidates because sigma is per sample and fixed across candidates.
winner_input = (1.0 - sigma) * latents + sigma * winner_noise
```

9. Call `call_dit` once outside every fork/no-grad scope with `winner_noise` and `winner_input`, then call the ordinary `compute_loss`.
10. Merge, without replacing architecture metrics:

```python
metrics = {
    **metrics,
    "xm/candidate_loss_mean": (candidate_loss_sum / (self._best_of_k_count * batch_size)).item(),
    "xm/selection_gain": (candidate_zero_mean - best_losses.detach().float().mean()).item(),
}
```

Do not store candidate outputs, candidate noise lists, or candidate losses after each iteration. Do not add a K check inside this method; dispatch owns that condition. Do not call `torch.clear_autocast_cache()`.

- [ ] **Step 7: Add the current-runtime CUDA autocast regression**

Add the test, with no cache manipulation before the final forward:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_standard_xm_cuda_autocast_recomputes_finite_winner_gradients(monkeypatch):
    trainer, _, loss, _ = _run_toy_xm(
        monkeypatch,
        device="cuda",
        autocast_dtype=torch.float16,
    )
    loss.backward()

    assert trainer.scale.grad is not None
    assert torch.isfinite(trainer.scale.grad).all()
    assert trainer.scale.grad.abs().item() > 0
```

Run it and print runtime evidence from the invocation, not production code:

```powershell
& 'E:\Python310\python.exe' -c "import platform, torch; print(platform.python_version()); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py -k 'autocast' -v -s
```

- [ ] **Step 8: Run focused and existing trainer tests, then commit**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py -v
```

Expected: K=1 dispatch compatibility, mixed winners, RNG replay, finite errors, metrics, gradient, and autocast tests pass.

```powershell
git add src/musubi_tuner/training/trainer_base.py tests/test_explorative_modeling.py
git commit -m "feat: add sequential Forward XM training"
```

### Task 6: MiniMax-H3 Video-Focused Best-of-K Heuristic

**Files:**
- Modify: `src/musubi_tuner/minimax_h3_train_network.py`
- Modify: `tests/test_minimax_h3_training.py`

**Interfaces:**
- Consumes: Task 1's pure candidate generator/draw/winner helpers and Task 3's best-of-K validation/dispatch contract.
- Produces: H3-only `--h3_video_best_of_k INT`, default `1`.
- Produces: `_H3TrainingState`, `_prepare_training_state`, `_call_training_dit`, `_compute_per_sample_component_losses`, and `_combine_per_sample_losses`.
- Implements: H3-owned `process_batch_best_of_k` that varies and scores video noise only.
- Guarantees: common Forward XM is rejected for H3; audio noise/time/conditions are fixed; final audio supervision and gradients remain active.

- [ ] **Step 1: Write failing parser and H3 validation tests**

Extend the existing H3 parser tests:

```python
def test_h3_parser_exposes_separate_video_best_of_k_option(tmp_path, monkeypatch):
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args(["--task", "t2va"]).h3_video_best_of_k == 1
    assert (
        parser.parse_args(
            ["--task", "t2va", "--h3_video_best_of_k", "3"]
        ).h3_video_best_of_k
        == 3
    )

    config = tmp_path / "h3_best_of_k.toml"
    config.write_text(
        'task = "t2va"\nh3_video_best_of_k = 4\n', encoding="utf-8"
    )
    common_parser = minimax_h3_setup_parser(setup_parser_common())
    monkeypatch.setattr(
        sys, "argv", ["minimax_h3_train_network", "--config_file", str(config)]
    )
    args = common_parser.parse_args()
    assert read_config_from_file(args, common_parser).h3_video_best_of_k == 4
```

Extend `_trainer_args` so `xm_best_of_k=1` and `h3_video_best_of_k=1` are always present, and import `read_config_from_file` plus `setup_parser_common` for the test above.

Instantiate `MiniMaxH3NetworkTrainer`, then assert:

```python
with pytest.raises(ValueError, match=r"--h3_video_best_of_k must be at least 1"):
    trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=0, xm_best_of_k=1))

with pytest.raises(ValueError, match=r"not Forward XM.*--h3_video_best_of_k"):
    trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=1, xm_best_of_k=2))

trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=2, xm_best_of_k=1))
assert trainer._best_of_k_count == 2
assert trainer._best_of_k_enabled is True
```

- [ ] **Step 2: Write failing component-loss tests**

Construct an H3 `DiTOutput` with batch size one, different video/audio errors, and `audio_loss_weight=torch.tensor([0.25], dtype=torch.float32)`. Assert:

```python
def test_h3_component_vectors_and_combiner_define_scalar_loss():
    trainer = MiniMaxH3NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([[1.0, 5.0]]),
        target=torch.tensor([[3.0, 1.0]]),
        extra={
            "audio_pred": torch.tensor([[0.0, 2.0]]),
            "audio_target": torch.tensor([[2.0, 2.0]]),
            "audio_loss_weight": torch.tensor([0.25], dtype=torch.float32),
        },
    )
    video, audio = trainer._compute_per_sample_component_losses(
        output, torch.float32
    )
    total = trainer._combine_per_sample_losses(
        video, audio, output.extra["audio_loss_weight"]
    )
    loss, metrics = trainer.compute_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        None,
        torch.float32,
        torch.float32,
        0,
    )

    assert video.shape == audio.shape == total.shape == (1,)
    torch.testing.assert_close(video, torch.tensor([10.0]))
    torch.testing.assert_close(audio, torch.tensor([2.0]))
    torch.testing.assert_close(total, torch.tensor([10.5]))
    assert torch.allclose(loss, total.mean(), rtol=1e-5, atol=1e-8)
    assert set(metrics) == {"loss/video", "loss/audio"}
```

Keep the existing exact validation for a finite nonnegative float32 weight of shape `[1]`. Retain the zero-weight behavior that does not evaluate missing audio prediction tensors.

Update the pre-existing scalar-loss fixtures so their real H3 batch axis is explicit: video/audio prediction and target tensors change from shape `[2]` to `[1, 2]`. Make the same change in the zero-weight gradient test; its expected scalar values and gradient-presence assertions remain unchanged.

- [ ] **Step 3: Write a failing video-only selection test**

Import the H3 module as `h3_module`, then add:

```python
class _ToyH3BestOfKTrainer(MiniMaxH3NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.video_parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.audio_parameter = torch.nn.Parameter(torch.tensor(0.0))
        self.best_of_k_records = []

    def _call_training_dit(
        self,
        args,
        accelerator,
        transformer,
        batch,
        latents,
        video_noise,
        noisy_video,
        state,
        network_dtype,
    ):
        del args, transformer, batch
        is_candidate_zero = torch.count_nonzero(video_noise).item() == 0
        video_error = 2.0 if is_candidate_zero else 1.0
        audio_error = 0.0 if is_candidate_zero else 10.0
        cpu_mask = torch.rand((), device="cpu")
        device_mask = (
            torch.rand((), device=accelerator.device)
            if accelerator.device.type == "cuda"
            else torch.rand((), device="cpu")
        )
        self.best_of_k_records.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "video_noise": video_noise.detach().clone(),
                "noisy_video": noisy_video.detach().clone(),
                "audio_latents": state.audio_latents.detach().clone(),
                "audio_noise": state.audio_noise.detach().clone(),
                "noisy_audio": state.noisy_audio.detach().clone(),
                "base_time": state.base_time.detach().clone(),
                "model_t_video": state.model_t_video.detach().clone(),
                "model_t_audio": state.model_t_audio.detach().clone(),
                "visual_conditions": tuple(
                    value.detach().clone() for value in state.visual_conditions
                ),
                "audio_conditions": tuple(
                    value.detach().clone() for value in state.audio_conditions
                ),
                "audio_loss_weight": state.audio_loss_weight.detach().clone(),
                "cpu_mask": cpu_mask.detach().clone(),
                "device_mask": device_mask.detach().cpu().clone(),
            }
        )
        return DiTOutput(
            pred=torch.ones_like(latents, dtype=network_dtype) * video_error
            + self.video_parameter.to(network_dtype),
            target=torch.zeros_like(latents, dtype=network_dtype),
            extra={
                "audio_pred": torch.ones_like(
                    state.audio_latents, dtype=network_dtype
                )
                * audio_error
                + self.audio_parameter.to(network_dtype),
                "audio_target": torch.zeros_like(
                    state.audio_latents, dtype=network_dtype
                ),
                "audio_loss_weight": state.audio_loss_weight,
            },
        )


def _run_h3_best_of_k(monkeypatch, trainer=None):
    trainer = trainer or _ToyH3BestOfKTrainer()
    trainer._best_of_k_count = 2
    trainer._best_of_k_enabled = True
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_video_best_of_k=2,
    )
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    latents = torch.zeros(1, 24, 2, 4, 4)
    candidate_zero = torch.zeros_like(latents)
    candidate_one = torch.ones_like(latents)
    monkeypatch.setattr(
        h3_module,
        "draw_candidate_noise",
        lambda reference, generator: candidate_one,
    )
    loss, metrics = trainer.process_batch_best_of_k(
        args,
        _Accelerator(),
        None,
        None,
        batch,
        latents,
        candidate_zero,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    return trainer, loss, metrics


def test_h3_best_of_k_selects_video_only_and_keeps_audio_gradient(monkeypatch):
    real_randn_like = torch.randn_like
    audio_noise_draws = 0
    captured = {}
    real_create = h3_module.create_candidate_generator

    def count_audio_noise(reference, *args, **kwargs):
        nonlocal audio_noise_draws
        if tuple(reference.shape) == (1, 32, 2, 8):
            audio_noise_draws += 1
        return real_randn_like(reference, *args, **kwargs)

    def capture_after_generator(reference):
        generator = real_create(reference)
        captured["cpu"] = torch.random.get_rng_state().clone()
        return generator

    monkeypatch.setattr(torch, "randn_like", count_audio_noise)
    monkeypatch.setattr(
        h3_module, "create_candidate_generator", capture_after_generator
    )
    torch.manual_seed(321)
    trainer, loss, metrics = _run_h3_best_of_k(monkeypatch)
    post_state = torch.random.get_rng_state().clone()
    records = trainer.best_of_k_records

    assert [record["grad_enabled"] for record in records] == [False, False, True]
    assert torch.count_nonzero(records[-1]["video_noise"]).item() > 0
    for key in (
        "audio_latents",
        "audio_noise",
        "noisy_audio",
        "base_time",
        "model_t_video",
        "model_t_audio",
        "audio_loss_weight",
    ):
        assert all(torch.equal(record[key], records[0][key]) for record in records[1:])
    for key in ("visual_conditions", "audio_conditions"):
        assert all(
            all(torch.equal(left, right) for left, right in zip(record[key], records[0][key]))
            for record in records[1:]
        )
    assert not torch.equal(records[0]["noisy_video"], records[1]["noisy_video"])
    assert all(torch.equal(record["cpu_mask"], records[0]["cpu_mask"]) for record in records[1:])
    assert all(torch.equal(record["device_mask"], records[0]["device_mask"]) for record in records[1:])
    assert audio_noise_draws == 1
    assert loss.item() == pytest.approx(101.0)
    assert metrics == {
        "loss/video": pytest.approx(1.0),
        "loss/audio": pytest.approx(100.0),
        "h3_video_best_of_k/candidate_loss_mean": pytest.approx(2.5),
        "h3_video_best_of_k/selection_gain": pytest.approx(3.0),
    }
    assert not any(key.startswith("xm/") for key in metrics)

    torch.random.set_rng_state(captured["cpu"])
    torch.rand((), device="cpu")
    torch.rand((), device="cpu")
    assert torch.equal(torch.random.get_rng_state(), post_state)

    loss.backward()
    assert trainer.video_parameter.grad is not None
    assert trainer.audio_parameter.grad is not None
    assert trainer.video_parameter.grad.item() == pytest.approx(2.0)
    assert trainer.audio_parameter.grad.item() == pytest.approx(20.0)
```

Candidate zero has component losses `(4, 0)` and candidate one `(1, 100)`, so the asserted winner is deliberately not the full-objective winner.

- [ ] **Step 4: Write failing H3 non-finite and K=1 compatibility tests**

Add the non-finite selection test:

```python
def test_h3_best_of_k_rejects_nonfinite_video_candidate(monkeypatch):
    class _NaNH3Trainer(_ToyH3BestOfKTrainer):
        def __init__(self):
            super().__init__()
            self.component_calls = 0

        def _compute_per_sample_component_losses(self, output, network_dtype):
            video, audio = super()._compute_per_sample_component_losses(
                output, network_dtype
            )
            if self.component_calls == 1:
                video = torch.full_like(video, torch.nan)
            self.component_calls += 1
            return video, audio

    with pytest.raises(
        ValueError,
        match=r"MiniMax-H3.*candidate 1.*sample indices \[0\]",
    ):
        _run_h3_best_of_k(monkeypatch, _NaNH3Trainer())
```

Before changing production code, turn the existing `test_process_batch_uses_one_shared_base_time_and_independent_audio_noise` into the K=1 characterization. Keep its direct `expected_video_loss` and `expected_audio_loss` formula, but initialize dispatch and call through it:

```python
trainer._validate_and_init_best_of_k(args)
rng_before = torch.random.get_rng_state().clone()
loss, metrics = trainer._process_batch_for_training(
    args,
    _Accelerator(),
    transformer,
    None,
    batch,
    video_latents,
    video_noise,
    None,
    torch.bfloat16,
    torch.float32,
    None,
    0,
)
rng_after = torch.random.get_rng_state().clone()

assert torch.allclose(
    loss,
    expected_video_loss + expected_audio_loss,
    rtol=1e-5,
    atol=1e-8,
)
assert set(metrics) == {"loss/video", "loss/audio"}
assert torch.equal(rng_after, rng_before)
assert "h3_video_best_of_k/candidate_loss_mean" not in metrics
```

The test already monkeypatches the ordinary audio/time draws to deterministic non-RNG functions, so equality proves dispatch added no seed, fork, or candidate draw. Do not call `process_batch` a second time as its own reference.

- [ ] **Step 5: Run H3 tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_minimax_h3_training.py -k 'best_of_k or component or video_only' -v
```

Expected: H3 option, component helpers, and local best-of-K path are missing.

- [ ] **Step 6: Add H3 option, hooks, and fixed training state**

Add the parser argument in `minimax_h3_setup_parser`:

```python
parser.add_argument(
    "--h3_video_best_of_k",
    type=int,
    default=1,
    help="Number of sequential video-noise candidates for the MiniMax-H3 video-focused heuristic; 1 disables it",
)
```

Add H3 overrides:

```python
def get_best_of_k_count(self, args: argparse.Namespace) -> int:
    if args.xm_best_of_k > 1:
        raise ValueError(
            "MiniMax-H3 does not support --xm_best_of_k because its video-only selection is not Forward XM; "
            "use --h3_video_best_of_k"
        )
    return args.h3_video_best_of_k

def get_best_of_k_option_name(self, args: argparse.Namespace) -> str:
    del args
    return "--h3_video_best_of_k"

def get_best_of_k_incompatibility_reason(self, args: argparse.Namespace) -> str | None:
    del args
    return None

def on_best_of_k_enabled(self, args: argparse.Namespace) -> None:
    del args
    logger.info(
        "MiniMax-H3 video-focused best-of-K heuristic enabled (not Forward XM): K=%d, "
        "selection objective: video only, final objective: video + weighted audio, "
        "sequential memory-saving mode, approximate operation-count multiplier %.2fx",
        self._best_of_k_count,
        (self._best_of_k_count + 3) / 3,
    )
```

Add:

```python
def test_h3_best_of_k_startup_log_names_the_distinct_objective(caplog):
    caplog.set_level("INFO")
    trainer = MiniMaxH3NetworkTrainer()
    trainer._validate_and_init_best_of_k(
        _trainer_args(h3_video_best_of_k=2, xm_best_of_k=1)
    )

    assert "video-focused best-of-K heuristic" in caplog.text
    assert "not Forward XM" in caplog.text
    assert "selection objective: video only" in caplog.text
    assert "final objective: video + weighted audio" in caplog.text
    assert "K=2" in caplog.text
    assert "1.67x" in caplog.text
    assert "Forward XM enabled for MiniMax-H3" not in caplog.text
```

Create frozen `_H3TrainingState` fields for `runtime`, `audio_latents`, `audio_noise`, `base_time`, `sigma_video`, `model_t_video`, `model_t_audio`, `noisy_audio`, `visual_conditions`, `audio_conditions`, and `audio_loss_weight`. Move the current `process_batch` preparation statements into `_prepare_training_state` in their existing RNG order. It must update audio accounting once, validate task once, and call `effective_audio_loss_weights` once.

Add the state beside `_H3RuntimeBatch`:

```python
@dataclass(frozen=True)
class _H3TrainingState:
    runtime: _H3RuntimeBatch
    audio_latents: torch.Tensor
    audio_noise: torch.Tensor
    base_time: torch.Tensor
    sigma_video: torch.Tensor
    model_t_video: torch.Tensor
    model_t_audio: torch.Tensor
    noisy_audio: torch.Tensor
    visual_conditions: tuple[torch.Tensor, ...]
    audio_conditions: tuple[torch.Tensor, ...]
    audio_loss_weight: torch.Tensor
```

Add these trainer methods:

```python
def _prepare_training_state(
    self,
    args: argparse.Namespace,
    batch: dict[str, Any],
    latents: torch.Tensor,
) -> _H3TrainingState:
    runtime = _runtime_batch_plan(batch, latents)
    self._audio_items_seen += int(runtime.audio_present.numel())
    self._audio_supervised_seen += int(runtime.audio_present.sum().item())
    if runtime.layout.task != args.task:
        raise ValueError(
            f"MiniMax-H3 --task {args.task} cannot train a {runtime.layout.task.upper()} cache batch"
        )
    device = latents.device
    audio_latents = batch["latents_audio"].to(device=device)
    audio_noise = torch.randn_like(audio_latents)
    base_time = _sample_base_time(args, batch, device)
    sigma_video = _shift_noise_amount(base_time, args.h3_shift_video)
    sigma_audio = _shift_noise_amount(base_time, args.h3_shift_audio)
    model_t_video = 1.0 - sigma_video
    model_t_audio = 1.0 - sigma_audio
    noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise
    needs_condition_noise = (
        bool(runtime.visual_conditions) and args.h3_visual_cond_clean != 1.0
    ) or (bool(runtime.audio_conditions) and args.h3_audio_cond_clean != 1.0)
    condition_seeds = (
        torch.randint(
            0,
            2**63 - 2,
            (latents.shape[0],),
            dtype=torch.int64,
            device="cpu",
        )
        if needs_condition_noise
        else torch.empty(0, dtype=torch.int64)
    )
    visual_conditions = _augment_conditions(
        runtime.visual_conditions,
        args.h3_visual_cond_clean,
        condition_seeds,
        seed_offset=0,
        device=device,
    )
    audio_conditions = _augment_conditions(
        runtime.audio_conditions,
        args.h3_audio_cond_clean,
        condition_seeds,
        seed_offset=1,
        device=device,
    )
    return _H3TrainingState(
        runtime=runtime,
        audio_latents=audio_latents,
        audio_noise=audio_noise,
        base_time=base_time,
        sigma_video=sigma_video,
        model_t_video=model_t_video,
        model_t_audio=model_t_audio,
        noisy_audio=noisy_audio,
        visual_conditions=visual_conditions,
        audio_conditions=audio_conditions,
        audio_loss_weight=effective_audio_loss_weights(runtime.audio_present, args),
    )

def _call_training_dit(
    self,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
    batch: dict[str, Any],
    latents: torch.Tensor,
    video_noise: torch.Tensor,
    noisy_video: torch.Tensor,
    state: _H3TrainingState,
    network_dtype: torch.dtype,
) -> DiTOutput:
    return self.call_dit(
        args,
        accelerator,
        transformer,
        latents,
        batch,
        video_noise,
        noisy_video,
        state.base_time,
        network_dtype,
        audio_latents=state.audio_latents,
        audio_noise=state.audio_noise,
        noisy_audio_input=state.noisy_audio,
        runtime=state.runtime,
        model_t_video=state.model_t_video,
        model_t_audio=state.model_t_audio,
        visual_conditions=state.visual_conditions,
        audio_conditions=state.audio_conditions,
        audio_loss_weight=state.audio_loss_weight,
    )
```

Refactor ordinary `process_batch` to prepare state once without inserting a K branch or changing draw order:

```python
del network, sample_resources
state = self._prepare_training_state(args, batch, latents)
noisy_video = (
    (1.0 - state.sigma_video) * latents + state.sigma_video * noise
)
output = self._call_training_dit(
    args,
    accelerator,
    transformer,
    batch,
    latents,
    noise,
    noisy_video,
    state,
    network_dtype,
)
return self.compute_loss(
    args,
    output,
    state.base_time,
    noise_scheduler,
    dit_dtype,
    network_dtype,
    global_step,
)
```

- [ ] **Step 7: Factor H3 component and total losses**

Implement `_compute_per_sample_component_losses(output, network_dtype)` by reducing every axis except batch. Validate the existing audio weight before touching audio tensors; when its value is zero, return a detached zero vector matching video shape. Implement `_combine_per_sample_losses(video, audio, weight)` as:

```python
def _compute_per_sample_component_losses(
    self,
    output: DiTOutput,
    network_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = output.pred.to(network_dtype)
    target = output.target.to(network_dtype)
    if pred.ndim < 1 or target.ndim < 1 or pred.shape[0] != target.shape[0]:
        raise ValueError("MiniMax-H3 component losses require matching leading batch axes")
    batch_size = pred.shape[0]
    video = torch.nn.functional.mse_loss(pred, target, reduction="none").reshape(batch_size, -1).mean(dim=1)
    audio_loss_weight = output.extra.get("audio_loss_weight")
    if (
        not isinstance(audio_loss_weight, torch.Tensor)
        or audio_loss_weight.shape != (1,)
        or audio_loss_weight.dtype != torch.float32
        or not torch.isfinite(audio_loss_weight).all().item()
        or audio_loss_weight.item() < 0.0
    ):
        raise ValueError(
            "MiniMax-H3 audio loss weight must be a finite nonnegative float32 tensor with shape [1]"
        )
    if audio_loss_weight.item() == 0.0:
        return video, video.detach().new_zeros(video.shape)
    audio_pred = output.extra["audio_pred"].to(network_dtype)
    audio_target = output.extra["audio_target"].to(network_dtype)
    if audio_pred.ndim < 1 or audio_target.ndim < 1 or audio_pred.shape[0] != batch_size or audio_target.shape[0] != batch_size:
        raise ValueError("MiniMax-H3 audio loss requires the same leading batch axis as video")
    audio = (
        torch.nn.functional.mse_loss(audio_pred, audio_target, reduction="none")
        .reshape(batch_size, -1)
        .mean(dim=1)
    )
    return video, audio

def _combine_per_sample_losses(
    self,
    video: torch.Tensor,
    audio: torch.Tensor,
    audio_loss_weight: torch.Tensor,
) -> torch.Tensor:
    if video.shape != audio.shape:
        raise ValueError("MiniMax-H3 video and audio per-sample losses must have matching shapes")
    return video + audio_loss_weight.to(device=video.device, dtype=video.dtype) * audio
```

Add the canonical H3 wrapper:

```python
def compute_per_sample_loss(
    self,
    args,
    output,
    timesteps,
    noise_scheduler,
    dit_dtype,
    network_dtype,
    global_step,
):
    del args, timesteps, noise_scheduler, dit_dtype, global_step
    video, audio = self._compute_per_sample_component_losses(output, network_dtype)
    return self._combine_per_sample_losses(video, audio, output.extra["audio_loss_weight"])
```

Replace H3 `compute_loss` with:

```python
del args, timesteps, noise_scheduler, dit_dtype, global_step
video, audio = self._compute_per_sample_component_losses(output, network_dtype)
total = self._combine_per_sample_losses(video, audio, output.extra["audio_loss_weight"])
return total.mean(), {
    "loss/video": video.detach().mean(),
    "loss/audio": audio.detach().mean(),
}
```

H3 remains B=1, but no helper may squeeze away its leading dimension.

- [ ] **Step 8: Implement the H3-owned candidate loop**

Use the incoming training-loop video noise as candidate zero. Prepare `_H3TrainingState` once before creating the private candidate generator. For each candidate:

- draw only later video noise tensors from the private generator;
- construct `noisy_video=(1-sigma_video)*latents+sigma_video*candidate_video_noise`;
- fork CPU plus the active CUDA RNG, call the joint `call_dit` under `torch.no_grad()`, and use only the video vector from `_compute_per_sample_component_losses`;
- stream winner video noise through `update_winners` and prefix validation failures with `MiniMax-H3`;
- retain no candidate output or audio candidate list.

Reconstruct final noisy video from fixed `sigma_video` and mixed winner video noise, call the joint DiT once with gradients and every stored audio/condition field, then call ordinary `compute_loss`. Merge the two H3-prefixed video-score metrics. Never generate another audio noise tensor and never emit an `xm/` metric.

Use this complete H3 override:

```python
def process_batch_best_of_k(
    self,
    args: argparse.Namespace,
    accelerator: Accelerator,
    transformer,
    network,
    batch: dict[str, torch.Tensor],
    latents: torch.Tensor,
    noise: torch.Tensor,
    noise_scheduler,
    dit_dtype: torch.dtype,
    network_dtype: torch.dtype,
    sample_resources,
    global_step: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    del network, sample_resources
    state = self._prepare_training_state(args, batch, latents)
    generator = create_candidate_generator(noise)
    batch_size = latents.shape[0]
    best_losses = torch.full((batch_size,), torch.inf, device=latents.device, dtype=torch.float32)
    winner_noise = torch.empty_like(noise)
    winner_indices = torch.full((batch_size,), -1, device=latents.device, dtype=torch.long)
    candidate_loss_sum = torch.zeros((), device=latents.device, dtype=torch.float32)
    candidate_zero_mean = None
    device = torch.device(accelerator.device)
    fork_devices = [device] if device.type == "cuda" else []

    for candidate_index in range(self._best_of_k_count):
        candidate_noise = noise if candidate_index == 0 else draw_candidate_noise(noise, generator)
        noisy_video = (
            (1.0 - state.sigma_video) * latents
            + state.sigma_video * candidate_noise
        )
        with torch.random.fork_rng(devices=fork_devices):
            with torch.no_grad():
                output = self._call_training_dit(
                    args,
                    accelerator,
                    transformer,
                    batch,
                    latents,
                    candidate_noise,
                    noisy_video,
                    state,
                    network_dtype,
                )
                video_losses, _ = self._compute_per_sample_component_losses(
                    output, network_dtype
                )
        video_losses_f32 = video_losses.detach().float()
        candidate_loss_sum = candidate_loss_sum + video_losses_f32.sum()
        if candidate_index == 0:
            candidate_zero_mean = video_losses_f32.mean()
        try:
            best_losses, winner_noise, winner_indices = update_winners(
                best_losses,
                winner_noise,
                winner_indices,
                video_losses_f32,
                candidate_noise,
                candidate_index,
            )
        except ValueError as error:
            raise ValueError(f"MiniMax-H3: {error}") from error

    winner_input = (
        (1.0 - state.sigma_video) * latents
        + state.sigma_video * winner_noise
    )
    output = self._call_training_dit(
        args,
        accelerator,
        transformer,
        batch,
        latents,
        winner_noise,
        winner_input,
        state,
        network_dtype,
    )
    loss, metrics = self.compute_loss(
        args,
        output,
        state.base_time,
        noise_scheduler,
        dit_dtype,
        network_dtype,
        global_step,
    )
    assert candidate_zero_mean is not None
    return loss, {
        **metrics,
        "h3_video_best_of_k/candidate_loss_mean": (
            candidate_loss_sum / (self._best_of_k_count * batch_size)
        ).item(),
        "h3_video_best_of_k/selection_gain": (
            candidate_zero_mean - best_losses.detach().float().mean()
        ).item(),
    }
```

- [ ] **Step 9: Run all H3 tests and commit**

Run:

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_minimax_h3_training.py -v
```

Expected: all pre-existing H3 tests plus parser, component, selection, fixed-state, non-finite, K=1 RNG, and final audio-gradient tests pass.

```powershell
git add src/musubi_tuner/minimax_h3_train_network.py tests/test_minimax_h3_training.py
git commit -m "feat: add H3 video best-of-k training"
```

### Task 7: User Documentation And Discoverability

**Files:**
- Create: `docs/explorative_modeling.md`
- Modify: `docs/minimax_h3.md`
- Modify: `README.md`
- Modify: `README.ja.md`

**Interfaces:**
- Consumes: the finalized CLI names, compatibility matrix, metrics, runtime evidence, and failure policy from Tasks 3, 5, and 6.
- Produces: one bilingual user guide and discoverability links; no executable API.

- [ ] **Step 1: Write the English and Japanese user contract**

Create `docs/explorative_modeling.md` with parallel `English` and `日本語` sections. Each language must contain:

- CLI and TOML examples for `xm_best_of_k = 2` and H3's separate `h3_video_best_of_k = 2`;
- links to issue 1019, arXiv paper `2607.27372v1`, the project page, and official Apache-2.0 implementation commit `9d06ced61e2d2775a34782eb5830584ae4ef6094`;
- an explicit statement that `1` is disabled behavior and preserves the ordinary path;
- fixed clean data, timestep, conditions, and stochastic condition decisions, with candidate-specific noise and paired targets;
- standard selection by the real per-sample weighted training loss;
- H3 selection by video loss only, fixed audio noise/input/time/conditions, and final video plus weighted audio loss;
- the label `video-focused best-of-K heuristic (not Forward XM)` for H3;
- the exact supported standard entry points: `flux_2_train_network.py`, `flux_kontext_train_network.py`, `fpack_train_network.py`, `hv_train_network.py`, `hv_1_5_train_network.py`, `ideogram4_train_network.py`, `kandinsky5_train_network.py`, `krea2_train_network.py`, `qwen_image_train_network.py`, `wan_train_network.py`, and `zimage_train_network.py`;
- Self-Flow-off support, and explicit rejection of Self-Flow-on, HiDream-O1, H3 common XM, and non-`NetworkTrainer` full-training entry points;
- K sequential no-grad forwards plus one winner forward, approximate `(K + 3) / 3` operation-count multiplier, and hardware/model-dependent wall-clock language;
- loss non-comparability across K, unchanged inference/guidance, and no established LoRA quality claim;
- the fail-fast policy: any non-finite candidate selection loss raises before backward, while baseline mixed-precision training may instead skip an update depending on GradScaler/loss-scaling configuration;
- the conservative K=1 versus K=2 recipe with identical seed, data, optimizer-step budget, and downstream validation metric;
- current validation only on Python 3.10.11, PyTorch 2.13.0+cu130, CUDA 13.0, RTX 4090; stable APIs may work elsewhere, but no other runtime is in this test matrix.

Do not recommend a universal K and do not quote a paper throughput number as a fine-tuning expectation.

Use this exact document structure and wording as the starting content; line-wrap it with the repository formatter but do not weaken its claims:

````markdown
# Explorative Modeling And Forward XM

## English

### Sources

This integration addresses [musubi-tuner issue
#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019). Its behavioral
references are the [Forward Explorative Modeling paper
v1](https://arxiv.org/abs/2607.27372v1), the [project
page](https://explorative-modeling.github.io/), and the Apache-2.0 [official
implementation](https://github.com/alexiglad/XM) at commit
`9d06ced61e2d2775a34782eb5830584ae4ef6094`.

### Enabling The Modes

Standard `NetworkTrainer` LoRA entry points use:

```toml
xm_best_of_k = 2
```

or:

```powershell
accelerate launch src/musubi_tuner/wan_train_network.py --config_file training.toml --xm_best_of_k 2
```

MiniMax-H3 uses its separately named mode:

```toml
h3_video_best_of_k = 2
```

```powershell
accelerate launch src/musubi_tuner/minimax_h3_train_network.py --config_file h3_training.toml --h3_video_best_of_k 2
```

For both options, `K = 1` disables exploration and uses the ordinary training
path. It performs no extra candidate draw, forward, metric, or RNG operation.

### Standard Forward XM Contract

For `xm_best_of_k > 1`, the trainer evaluates K noise candidates sequentially
without gradients and recomputes the per-sample winners once with gradients.
Clean latents/data, timestep, conditioning, condition-drop decisions, and
other stochastic forward choices stay fixed. Each prediction target remains
paired with the noise that produced that candidate. Winners are selected
independently per sample with the architecture's actual weighted per-sample
training loss.

The sequential implementation retains one candidate and one `[B, ...]` winner
tensor rather than K activation graphs. It performs K no-grad forwards plus
one ordinary gradient-enabled forward. Relative to an approximate
forward-plus-backward cost of three forward operations, its operation-count
multiplier is `(K + 3) / 3`. Wall-clock cost depends on the model, hardware,
offloading, checkpointing, and compiler settings.

### MiniMax-H3 Video-Focused Heuristic

`h3_video_best_of_k > 1` is a **video-focused best-of-K heuristic (not Forward
XM)**. It varies video noise and ranks candidates by video loss only. Audio
noise, base time, shifted audio time, noisy audio input, and audio/visual
conditions stay fixed. The selected candidate is recomputed with the ordinary
`video_loss + audio_loss_weight * audio_loss` objective, so audio supervision
and audio-path gradients remain active.

Because MiniMax-H3 is a joint video/audio model, its audio prediction can
change when video noise changes. The video-best candidate therefore need not
minimize the final composite loss. MiniMax-H3 rejects `--xm_best_of_k > 1`;
use `--h3_video_best_of_k` only when that video-focused tradeoff is intended.

### Compatibility

Standard Forward XM supports these LoRA entry points:

- `flux_2_train_network.py`
- `flux_kontext_train_network.py`
- `fpack_train_network.py`
- `hv_train_network.py`
- `hv_1_5_train_network.py`
- `ideogram4_train_network.py`
- `kandinsky5_train_network.py`
- `krea2_train_network.py`
- `qwen_image_train_network.py`
- `wan_train_network.py`
- `zimage_train_network.py`

`flux_2_train_network_self_flow.py` supports standard XM only when
`--self_flow` is off. Enabled Self-Flow is rejected because its
teacher/student candidate state and objective are not implemented. HiDream-O1
is rejected because its candidate-local noising transform cannot be rebuilt by
the standard affine path and its optional DINO term is batch-reduced.
MiniMax-H3 rejects standard XM and exposes only its H3-specific heuristic.

Standalone full-finetune entry points such as `hv_train.py`,
`qwen_image_train.py`, `zimage_train.py`, and `hidream_o1_train.py` do not use
the shared `NetworkTrainer` loop and are unchanged.

### Numerical Failure Policy

Forward XM and H3 video best-of-K raise before backward when any candidate
selection loss is NaN or infinite. They do not silently discard that candidate.
This is intentionally stricter than ordinary mixed-precision training, where
GradScaler or another loss-scaling configuration may detect non-finite
gradients during backward and skip an update. Do not assume the modes recover
from numerical instability in the same way.

### Comparing K Values

Training loss is not directly comparable across K because candidate selection
changes its distribution. Published Forward XM improvements are pretraining
results; this integration establishes no LoRA quality, convergence, or
data-efficiency gain. Guidance and inference are unchanged.

Start with a controlled `K = 1` versus `K = 2` experiment using the same seed,
data, optimizer-step budget, and downstream validation metric. Do not choose a
run by comparing raw training loss across K. There is no universal recommended
K.

### Validated Runtime

R1 is tested on Python 3.10.11, PyTorch `2.13.0+cu130`, CUDA 13.0, and an NVIDIA
GeForce RTX 4090 with compute capability 8.9. The implementation uses existing
public PyTorch APIs and adds no version gate, but other runtimes are outside
this test matrix and compatibility claim.

## 日本語

### 参照資料

この実装は [musubi-tuner issue
#1019](https://github.com/kohya-ss/musubi-tuner/issues/1019) に対応します。動作上の
参照元は [Forward Explorative Modeling 論文
v1](https://arxiv.org/abs/2607.27372v1)、[プロジェクトページ](https://explorative-modeling.github.io/)、
および commit `9d06ced61e2d2775a34782eb5830584ae4ef6094` の Apache-2.0
[公式実装](https://github.com/alexiglad/XM) です。

### 有効化

標準の `NetworkTrainer` LoRA エントリポイントでは次を指定します。

```toml
xm_best_of_k = 2
```

```powershell
accelerate launch src/musubi_tuner/wan_train_network.py --config_file training.toml --xm_best_of_k 2
```

MiniMax-H3 では別名の専用オプションを使います。

```toml
h3_video_best_of_k = 2
```

```powershell
accelerate launch src/musubi_tuner/minimax_h3_train_network.py --config_file h3_training.toml --h3_video_best_of_k 2
```

どちらも `K = 1` では探索を無効化し、従来の学習経路をそのまま使います。追加の
候補生成、forward、メトリクス、RNG 操作は発生しません。

### 標準 Forward XM の契約

`xm_best_of_k > 1` では K 個のノイズ候補を勾配なしで逐次評価し、サンプルごとの
勝者だけを勾配ありで一度再計算します。clean latent/data、timestep、conditioning、
condition drop、その他の確率的 forward 条件は候補間で固定されます。prediction
target は必ず、その候補を生成したノイズと組にしたまま扱います。勝者はモデルの
実際の重み付き per-sample 学習 loss で、サンプルごとに独立して選択されます。

実装は K 個の activation graph を保持せず、現在の候補と `[B, ...]` の勝者だけを
保持します。計算は K 回の no-grad forward と 1 回の通常の勾配あり forward です。
forward + backward をおよそ 3 forward 相当とみなした演算回数倍率は
`(K + 3) / 3` です。実時間はモデル、GPU、offload、checkpoint、compile 設定に
依存します。

### MiniMax-H3 の動画重視ヒューリスティック

`h3_video_best_of_k > 1` は **動画重視の best-of-K ヒューリスティックであり、
Forward XM ではありません**。動画ノイズだけを変え、動画 loss だけで候補を
順位付けします。音声ノイズ、base time、shift 後の音声 time、noisy audio input、
動画・音声 condition は固定されます。選択後の候補は通常どおり
`video_loss + audio_loss_weight * audio_loss` で再計算するため、音声 supervision と
音声経路の勾配は残ります。

MiniMax-H3 は動画・音声の joint model なので、動画ノイズを変えると音声予測も
変わり得ます。そのため動画 loss が最小の候補が最終 composite loss の最小候補とは
限りません。MiniMax-H3 は `--xm_best_of_k > 1` を拒否します。この動画重視の
tradeoff を意図する場合だけ `--h3_video_best_of_k` を使ってください。

### 対応範囲

標準 Forward XM は次の LoRA エントリポイントに対応します。

- `flux_2_train_network.py`
- `flux_kontext_train_network.py`
- `fpack_train_network.py`
- `hv_train_network.py`
- `hv_1_5_train_network.py`
- `ideogram4_train_network.py`
- `kandinsky5_train_network.py`
- `krea2_train_network.py`
- `qwen_image_train_network.py`
- `wan_train_network.py`
- `zimage_train_network.py`

`flux_2_train_network_self_flow.py` は `--self_flow` が off の通常 Flux 2 経路だけを
サポートします。Self-Flow 有効時は teacher/student の候補状態と目的関数が未実装の
ため拒否します。HiDream-O1 は候補ごとの noising 変換を標準 affine 経路から再構成
できず、任意の DINO 項も batch reduction なので拒否します。MiniMax-H3 は標準 XM
を拒否し、H3 専用ヒューリスティックだけを公開します。

`hv_train.py`、`qwen_image_train.py`、`zimage_train.py`、
`hidream_o1_train.py` などの full-finetune エントリポイントは共有
`NetworkTrainer` loop を使わないため変更しません。

### 非有限 loss の方針

Forward XM と H3 video best-of-K は、候補選択 loss に NaN または infinity が一つでも
あれば backward 前に例外を送出します。その候補を黙って捨てることはしません。
これは通常の mixed-precision 学習より厳しい方針です。通常経路では GradScaler や
loss scaling 設定によって backward 中に非有限 gradient を検出し、更新を skip する
場合があります。両経路が数値不安定から同じように回復するとは考えないでください。

### K の比較

候補選択によって分布が変わるため、K が異なる run の training loss は直接比較
できません。論文の改善結果は pretraining の結果であり、この実装は LoRA の品質、
収束、data efficiency の改善を保証しません。guidance と inference は変わりません。

同一の seed、data、optimizer step 数、downstream validation metric を使い、まず
`K = 1` と `K = 2` を比較してください。異なる K の raw training loss で run を
選ばないでください。すべての環境に通用する推奨 K はありません。

### 検証環境

R1 は Python 3.10.11、PyTorch `2.13.0+cu130`、CUDA 13.0、compute capability 8.9 の
NVIDIA GeForce RTX 4090 で検証します。既存の public PyTorch API だけを使い version
gate は追加しませんが、その他の runtime はこの test matrix と互換性主張の対象外です。
````

- [ ] **Step 2: Link the new guide and make H3 terminology consistent**

Add these exact common-configuration list entries:

```markdown
- [Explorative Modeling and Forward XM](./docs/explorative_modeling.md)
```

```markdown
- [探索的モデリングとForward XM](./docs/explorative_modeling.md)
```

The Japanese architecture list currently lacks H3, so add this parity entry after FLUX.2:

```markdown
- [MiniMax-H3](./docs/minimax_h3.md)
```

After the H3 loss-policy paragraph in `docs/minimax_h3.md`, add:

```markdown
`--h3_video_best_of_k K` enables a video-focused best-of-K heuristic (not
Forward XM) when `K > 1`. It varies and ranks video noise by video loss while
keeping the audio candidate state fixed; the selected update still optimizes
video loss plus weighted audio loss. MiniMax-H3 rejects the common
`--xm_best_of_k` option for `K > 1`. See [Explorative Modeling and Forward
XM](./explorative_modeling.md) for semantics, cost, compatibility, and the
strict non-finite-loss policy.
```

Do not rename existing H3 audio loss arguments or imply audio supervision is disabled.

- [ ] **Step 3: Check documentation terms and links**

Run:

```powershell
rg -n "xm_best_of_k|h3_video_best_of_k|not Forward XM|非有限|non-finite|2.13.0\+cu130" docs/explorative_modeling.md docs/minimax_h3.md README.md README.ja.md
rg -n "h3.*Forward XM|MiniMax-H3.*XM" docs/explorative_modeling.md docs/minimax_h3.md README.md README.ja.md
```

Expected: both flags, both language sections, the strict non-finite policy, and runtime scope are present; any H3/XM match explicitly says H3 is not Forward XM.

- [ ] **Step 4: Commit documentation**

```powershell
git add docs/explorative_modeling.md docs/minimax_h3.md README.md README.ja.md
git commit -m "docs: explain explorative training modes"
```

### Task 8: Full Verification And Review Readiness

**Files:**
- Verify only; modify production/tests/docs only for failures traced through the TDD loop.

**Interfaces:**
- Consumes: every executable and documentation deliverable from Tasks 1-7.
- Produces: fresh focused/full-suite/lint/diff evidence for review; no new runtime interface.

- [ ] **Step 1: Record the actual validation runtime**

Run:

```powershell
& 'E:\Python310\python.exe' -c "import platform, torch; print('python', platform.python_version()); print('torch', torch.__version__); print('cuda-runtime', torch.version.cuda); print('cuda-available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'none')"
```

Expected on the approved environment: Python 3.10.11, torch 2.13.0+cu130, CUDA 13.0, RTX 4090, capability `(8, 9)`.

- [ ] **Step 2: Run focused feature tests**

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py tests/test_ideogram4_timesteps.py tests/test_krea2_timesteps.py tests/test_minimax_h3_training.py -v
```

- [ ] **Step 3: Run the repository test suite**

```powershell
$env:PYTHONPATH='src'; & 'E:\Python310\python.exe' -m pytest tests -v
```

If the complete suite contains an environment-dependent failure unrelated to this branch, rerun the failing test in isolation, record its exact output, and do not weaken or skip it without a demonstrated feature interaction.

- [ ] **Step 4: Run formatting, lint, and structural checks**

```powershell
& 'E:\Python310\python.exe' -m ruff check src/musubi_tuner/training/explorative.py src/musubi_tuner/training/timesteps.py src/musubi_tuner/training/parser_common.py src/musubi_tuner/training/trainer_base.py src/musubi_tuner/ideogram4_train_network.py src/musubi_tuner/flux_2_train_network_self_flow.py src/musubi_tuner/hidream_o1_train_network.py src/musubi_tuner/minimax_h3_train_network.py tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py tests/test_minimax_h3_training.py
& 'E:\Python310\python.exe' -m ruff format --check src/musubi_tuner/training/explorative.py src/musubi_tuner/training/timesteps.py src/musubi_tuner/training/parser_common.py src/musubi_tuner/training/trainer_base.py src/musubi_tuner/ideogram4_train_network.py src/musubi_tuner/flux_2_train_network_self_flow.py src/musubi_tuner/hidream_o1_train_network.py src/musubi_tuner/minimax_h3_train_network.py tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py tests/test_minimax_h3_training.py
git diff --check upstream/dev...HEAD
rg -n "torch\.clear_autocast_cache|\[K,|candidate_noises|xm/" src/musubi_tuner/training src/musubi_tuner/minimax_h3_train_network.py
```

Expected: no explicit autocast-cache clear, no retained K-shaped candidate collection, H3 contains no `xm/` metric, and all changed Python files satisfy repository Ruff rules. If `ruff format --check` fails, run `ruff format` on only the listed changed Python files, inspect the diff, and rerun every focused test.

- [ ] **Step 5: Inspect the final branch diff against the approved design**

```powershell
git status --short
git diff --stat upstream/dev...HEAD
git diff upstream/dev...HEAD -- src/musubi_tuner/training/explorative.py src/musubi_tuner/training/timesteps.py src/musubi_tuner/training/trainer_base.py src/musubi_tuner/minimax_h3_train_network.py
git log --oneline upstream/dev..HEAD
```

Confirm explicitly:

- no unrelated file or generated artifact is present;
- K=1 dispatch has no generator/fork/metric work;
- the shared sampler classification is referenced by both baseline noising and coefficient reconstruction;
- standard candidate selection calls the canonical per-sample full loss;
- H3 candidate selection calls only its video component and final recomputation calls the weighted total;
- every compatibility error is raised during `_validate_args_and_init` before allocation;
- docs describe the stricter NaN policy before experiment advice.

- [ ] **Step 6: Create the final verification commit only if checks changed files**

When verification-driven fixes were necessary:

```powershell
git add src/musubi_tuner/training/explorative.py src/musubi_tuner/training/timesteps.py src/musubi_tuner/training/parser_common.py src/musubi_tuner/training/trainer_base.py src/musubi_tuner/ideogram4_train_network.py src/musubi_tuner/flux_2_train_network_self_flow.py src/musubi_tuner/hidream_o1_train_network.py src/musubi_tuner/minimax_h3_train_network.py tests/test_explorative_modeling.py tests/test_ideogram4_synthetic.py tests/test_minimax_h3_training.py docs/explorative_modeling.md docs/minimax_h3.md README.md README.ja.md
git commit -m "test: harden explorative training regressions"
```

When verification requires no fixes, do not create an empty commit.

---

## Spec Coverage Matrix

| Approved contract | Implemented and proved in |
| --- | --- |
| Separate standard and H3 flags | Tasks 3 and 6 |
| K=1 original path and RNG | Tasks 3, 5, and 6 |
| Sequential constant-memory K+1 forwards | Tasks 5 and 6 |
| Fixed timestep/data/conditions and paired targets | Tasks 2, 5, and 6 |
| Per-sample weighted standard selection | Tasks 4 and 5 |
| H3 video-only selection with final audio loss/gradients | Task 6 |
| Shared explicit-coefficient sampler classification | Task 2 |
| Self-Flow, HiDream, and H3 compatibility errors | Tasks 3 and 6 |
| Fail-fast candidate NaN/Inf diagnostics | Tasks 1, 5, 6, and 7 |
| RNG replay without Python/NumPy helpers | Task 5 |
| No explicit autocast cache clearing | Tasks 5 and 8 |
| Current PyTorch/CUDA environment evidence only | Tasks 5, 7, and 8 |
| Bounded, distinct metrics | Tasks 5 and 6 |
| English/Japanese user contract | Task 7 |
| Full regression and lint evidence | Task 8 |
