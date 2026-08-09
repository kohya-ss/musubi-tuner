from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _explorative_helpers():
    try:
        module = import_module("musubi_tuner.training.explorative")
    except ModuleNotFoundError:
        pytest.fail("explorative best-of-K mechanics are not implemented")

    return (
        getattr(module, "create_candidate_generator"),
        getattr(module, "draw_candidate_noise"),
        getattr(module, "update_winners"),
    )


def test_update_winners_selects_each_sample_independently():
    _, _, update_winners = _explorative_helpers()
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
    create_candidate_generator, draw_candidate_noise, _ = _explorative_helpers()
    torch.manual_seed(1234)
    reference = torch.empty(2, 3)
    torch.randint(
        0,
        torch.iinfo(torch.int64).max,
        (),
        device=reference.device,
        dtype=torch.int64,
    )
    control_state_after_one_seed_draw = torch.random.get_rng_state().clone()

    torch.manual_seed(1234)
    generator = create_candidate_generator(reference)
    global_state_after_creation = torch.random.get_rng_state().clone()
    assert torch.equal(global_state_after_creation, control_state_after_one_seed_draw)

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA generator coverage")
def test_cuda_candidate_generator_owns_draws_after_one_global_seed_draw():
    create_candidate_generator, draw_candidate_noise, _ = _explorative_helpers()
    device = torch.device("cuda:0")
    previous_device = torch.cuda.current_device()
    previous_rng_state = torch.cuda.get_rng_state(device)
    try:
        torch.cuda.set_device(device)
        reference = torch.empty(2, 3, device=device, dtype=torch.float16)
        torch.cuda.manual_seed(4321)
        torch.randint(
            0,
            torch.iinfo(torch.int64).max,
            (),
            device=device,
            dtype=torch.int64,
        )
        control_state_after_one_seed_draw = torch.cuda.get_rng_state(device).clone()

        torch.cuda.manual_seed(4321)
        generator = create_candidate_generator(reference)
        global_state_after_creation = torch.cuda.get_rng_state(device).clone()
        assert torch.equal(global_state_after_creation, control_state_after_one_seed_draw)

        first = draw_candidate_noise(reference, generator)
        second = draw_candidate_noise(reference, generator)

        assert torch.equal(torch.cuda.get_rng_state(device), global_state_after_creation)
        assert first.shape == reference.shape
        assert first.dtype == reference.dtype
        assert first.device == reference.device
        assert not torch.equal(first, second)
    finally:
        torch.cuda.set_rng_state(previous_rng_state, device)
        torch.cuda.set_device(previous_device)


@pytest.mark.parametrize("bad_losses", [torch.tensor([1.0, float("nan")]), torch.tensor([1.0, float("inf")])])
def test_update_winners_rejects_nonfinite_candidate_scores(bad_losses):
    _, _, update_winners = _explorative_helpers()
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
    _, _, update_winners = _explorative_helpers()
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
    _, _, update_winners = _explorative_helpers()
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
    _, _, update_winners = _explorative_helpers()
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


def test_update_winners_rejects_scalar_noise_with_a_value_error():
    _, _, update_winners = _explorative_helpers()
    with pytest.raises(ValueError, match="shapes must match"):
        update_winners(
            torch.full((1,), torch.inf),
            torch.tensor(0.0),
            torch.full((1,), -1, dtype=torch.long),
            torch.tensor([1.0]),
            torch.tensor(0.0),
            0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for cross-device validation coverage")
def test_update_winners_rejects_noise_on_a_different_device_from_losses():
    _, _, update_winners = _explorative_helpers()
    with pytest.raises(ValueError, match="must share the loss device"):
        update_winners(
            torch.full((2,), torch.inf),
            torch.empty(2, 1, device="cuda"),
            torch.full((2,), -1, dtype=torch.long),
            torch.tensor([1.0, 2.0]),
            torch.zeros(2, 1, device="cuda"),
            0,
        )


def _noise_coefficient_helpers():
    module = import_module("musubi_tuner.training.timesteps")
    trainer_module = import_module("musubi_tuner.training.trainer_base")
    return (
        getattr(module, "BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS"),
        getattr(module, "get_noise_coefficients_from_timesteps"),
        getattr(trainer_module, "NetworkTrainer"),
    )


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


def test_explicit_sampler_candidate_zero_reconstructs_from_returned_timestep():
    sampler_names, get_noise_coefficients, trainer_type = _noise_coefficient_helpers()
    torch.manual_seed(9)
    trainer = trainer_type()
    latents = torch.randn(2, 3, 2, 2, dtype=torch.float32)
    noise = torch.randn_like(latents)

    for timestep_sampling in sorted(sampler_names):
        noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
            _timestep_args(timestep_sampling),
            noise,
            latents,
            [0.25, 0.75],
            None,
            torch.device("cpu"),
            torch.float32,
        )
        sigma = get_noise_coefficients(
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
    _, get_noise_coefficients, _ = _noise_coefficient_helpers()
    scheduler = SimpleNamespace(
        timesteps=torch.tensor([1000.0, 500.0, 1.0]),
        sigmas=torch.tensor([1.0, 0.5, 0.0]),
    )
    sigma = get_noise_coefficients(
        "sigma",
        scheduler,
        torch.tensor([500.0, 1.0]),
        torch.device("cpu"),
        5,
        torch.float32,
    )

    assert sigma.shape == (2, 1, 1, 1, 1)
    torch.testing.assert_close(sigma[:, 0, 0, 0, 0], torch.tensor([0.5, 0.0]))


@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_scheduler_candidate_zero_reconstructs_with_declared_dtype_tolerance(monkeypatch, dtype, rtol, atol):
    _, get_noise_coefficients, trainer_type = _noise_coefficient_helpers()
    trainer_base_module = import_module("musubi_tuner.training.trainer_base")
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
    trainer = trainer_type()
    latents = torch.linspace(-1.0, 1.0, 8, dtype=torch.float32).to(dtype).reshape(2, 1, 2, 2)
    noise = torch.flip(latents, dims=(-1,))
    args = _timestep_args("sigma")
    args.max_timestep = scheduler.config.num_train_timesteps
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
        args, noise, latents, None, scheduler, torch.device("cpu"), dtype
    )
    sigma = get_noise_coefficients("sigma", scheduler, timesteps, torch.device("cpu"), latents.ndim, dtype)

    torch.testing.assert_close(
        (1.0 - sigma) * latents + sigma * noise,
        noisy,
        rtol=rtol,
        atol=atol,
    )


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
    _, get_noise_coefficients, _ = _noise_coefficient_helpers()
    latents = torch.linspace(-1.0, 1.0, int(torch.tensor(shape).prod().item()), dtype=torch.float32).to(dtype).reshape(shape)
    noise = torch.flip(latents, dims=(-1,))
    timesteps = torch.tensor([251.0, 751.0], dtype=torch.float32)
    sigma = get_noise_coefficients("uniform", None, timesteps, torch.device("cpu"), len(shape), dtype)
    expected_sigma = torch.tensor([0.25, 0.75], dtype=dtype).reshape(2, *([1] * (len(shape) - 1)))

    torch.testing.assert_close(sigma, expected_sigma, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        (1.0 - sigma) * latents + sigma * noise,
        (1.0 - expected_sigma) * latents + expected_sigma * noise,
        rtol=rtol,
        atol=atol,
    )
