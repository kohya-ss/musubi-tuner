import inspect
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer
from musubi_tuner.hidream_o1_train_network import HiDreamO1NetworkTrainer
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
import musubi_tuner.training.trainer_base as trainer_base_module


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


@pytest.mark.parametrize(
    "timestep_sampling",
    sorted(getattr(import_module("musubi_tuner.training.timesteps"), "BASE_NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS")),
)
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [
        (torch.float32, 1e-5, 1e-6),
        (torch.float16, 1e-3, 1e-3),
        (torch.bfloat16, 1e-2, 1e-2),
    ],
)
def test_explicit_sampler_candidate_zero_reconstructs_from_returned_timestep(timestep_sampling, dtype, rtol, atol):
    _, get_noise_coefficients, trainer_type = _noise_coefficient_helpers()
    torch.manual_seed(9)
    trainer = trainer_type()
    latents = torch.randn(2, 3, 2, 2, dtype=dtype)
    noise = torch.randn_like(latents)
    noisy, timesteps = trainer.get_noisy_model_input_and_timesteps(
        _timestep_args(timestep_sampling),
        noise,
        latents,
        [0.25, 0.75],
        None,
        torch.device("cpu"),
        dtype,
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
    torch.testing.assert_close(reconstructed, noisy, rtol=rtol, atol=atol, check_dtype=False)


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


class _InheritedCustomBatchTrainer(_CustomBatchTrainer):
    pass


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


def _early_validation_args(xm_best_of_k):
    return SimpleNamespace(
        cuda_allow_tf32=False,
        cuda_cudnn_benchmark=False,
        dataset_config="unused.toml",
        dit="unused.safetensors",
        fp8_scaled=False,
        fp8_base=False,
        sage_attn=False,
        disable_numpy_memmap=False,
        show_timesteps=None,
        xm_best_of_k=xm_best_of_k,
    )


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


def test_best_of_k_validation_rejects_values_below_one():
    trainer = _CompatibleTrainer()
    with pytest.raises(ValueError, match=r"--xm_best_of_k.*integer.*at least 1"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=0))


@pytest.mark.parametrize("toml_value", ["1.5", "true", '"2"'])
def test_best_of_k_validation_rejects_non_integer_toml_before_allocation(tmp_path, monkeypatch, toml_value):
    config = tmp_path / "invalid-xm.toml"
    config.write_text(
        f'dataset_config = "unused.toml"\ndit = "unused.safetensors"\nxm_best_of_k = {toml_value}\n',
        encoding="utf-8",
    )
    parser = setup_parser_common()
    monkeypatch.setattr(sys, "argv", ["trainer", "--config_file", str(config)])
    args = read_config_from_file(parser.parse_args(), parser)
    trainer = _EarlyValidationTrainer()

    with pytest.raises(ValueError, match=r"--xm_best_of_k.*integer.*at least 1"):
        trainer.train(args)

    assert trainer.events == ["handle_model_specific_args"]
    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)


def test_best_of_k_validation_rejects_unconfirmed_custom_process_batch():
    trainer = _CustomBatchTrainer()
    with pytest.raises(ValueError, match=r"overrides process_batch"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))


def test_best_of_k_validation_rejects_inherited_custom_process_batch():
    trainer = _InheritedCustomBatchTrainer()
    with pytest.raises(ValueError, match=r"overrides process_batch"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))


def test_best_of_k_validation_rejects_instance_process_batch_replacement():
    trainer = _CompatibleTrainer()
    trainer.process_batch = lambda *args, **kwargs: (torch.tensor(0.0), {})
    with pytest.raises(ValueError, match=r"overrides process_batch"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))


def test_best_of_k_validation_accepts_explicit_custom_process_compatibility():
    trainer = _ExplicitlyCompatibleCustomBatchTrainer()
    trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))
    assert trainer._best_of_k_enabled is True


def test_best_of_k_validation_leaves_fresh_incompatible_trainer_disabled():
    trainer = _CustomBatchTrainer()

    with pytest.raises(ValueError, match=r"overrides process_batch"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))

    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)


@pytest.mark.parametrize("invalid_count", [0, 1.5, True, "2"])
def test_best_of_k_validation_resets_successful_configuration_after_invalid_revalidation(invalid_count):
    trainer = _CompatibleTrainer()
    trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=2))

    with pytest.raises(ValueError, match=r"--xm_best_of_k.*(?:integer|at least 1)"):
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=invalid_count))

    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)


def test_training_dispatch_preserves_original_k_one_path(monkeypatch):
    trainer = _CompatibleTrainer()
    trainer._best_of_k_enabled = False
    positional_sentinels = (object(), object(), object())
    keyword_sentinels = {"keyword_one": object(), "keyword_two": object()}
    loss = object()
    metrics = {"ordinary": object()}
    calls = []

    def ordinary(*args, **kwargs):
        calls.append("ordinary")
        assert args == positional_sentinels
        assert kwargs == keyword_sentinels
        return loss, metrics

    monkeypatch.setattr(trainer, "process_batch", ordinary)
    monkeypatch.setattr(
        trainer,
        "process_batch_best_of_k",
        lambda *args, **kwargs: pytest.fail("best-of-k arm must not be called at K=1"),
    )
    state = torch.random.get_rng_state().clone()

    returned_loss, returned_metrics = trainer._process_batch_for_training(*positional_sentinels, **keyword_sentinels)

    assert calls == ["ordinary"]
    assert returned_loss is loss
    assert returned_metrics is metrics
    assert torch.equal(torch.random.get_rng_state(), state)


def test_training_dispatch_uses_best_of_k_only_when_enabled(monkeypatch):
    trainer = _CompatibleTrainer()
    trainer._best_of_k_enabled = True
    positional_sentinels = (object(), object(), object())
    keyword_sentinels = {"keyword_one": object(), "keyword_two": object()}
    loss = object()
    metrics = {"xm/selection_gain": object()}
    calls = []

    def best_of_k(*args, **kwargs):
        calls.append("best-of-k")
        assert args == positional_sentinels
        assert kwargs == keyword_sentinels
        return loss, metrics

    monkeypatch.setattr(trainer, "process_batch", lambda *args, **kwargs: pytest.fail("ordinary arm must not be called at K>1"))
    monkeypatch.setattr(
        trainer,
        "process_batch_best_of_k",
        best_of_k,
    )

    returned_loss, returned_metrics = trainer._process_batch_for_training(*positional_sentinels, **keyword_sentinels)

    assert calls == ["best-of-k"]
    assert returned_loss is loss
    assert returned_metrics is metrics


def test_best_of_k_placeholder_signature_matches_process_batch():
    ordinary_parameters = tuple(inspect.signature(NetworkTrainer.process_batch).parameters)
    best_of_k_parameters = tuple(inspect.signature(NetworkTrainer.process_batch_best_of_k).parameters)
    assert best_of_k_parameters == ordinary_parameters


def test_architecture_specific_standard_xm_compatibility_reasons(monkeypatch):
    self_flow = Flux2SelfFlowNetworkTrainer.__new__(Flux2SelfFlowNetworkTrainer)
    NetworkTrainer.__init__(self_flow)
    hidream = HiDreamO1NetworkTrainer.__new__(HiDreamO1NetworkTrainer)
    NetworkTrainer.__init__(hidream)

    assert self_flow.get_best_of_k_incompatibility_reason(SimpleNamespace(self_flow=False)) is None
    assert "--self_flow" in self_flow.get_best_of_k_incompatibility_reason(SimpleNamespace(self_flow=True))
    assert "noise scaling/clipping" in hidream.get_best_of_k_incompatibility_reason(SimpleNamespace())

    for trainer in (self_flow, hidream):
        monkeypatch.setattr(
            trainer,
            "get_best_of_k_incompatibility_reason",
            lambda args: (_ for _ in ()).throw(AssertionError("hook called at K=1")),
        )
        trainer._validate_and_init_best_of_k(SimpleNamespace(xm_best_of_k=1))
        assert trainer._best_of_k_enabled is False


def test_invalid_best_of_k_fails_before_session_or_dataset_allocation():
    trainer = _EarlyValidationTrainer()
    args = _early_validation_args(0)
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


def test_base_per_sample_loss_applies_weight_before_nonbatch_reduction(monkeypatch):
    trainer = NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([[[[1.0, 3.0]]], [[[2.0, 4.0]]]]),
        target=torch.zeros(2, 1, 1, 2),
    )
    weighting = torch.tensor([2.0, 0.5]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(trainer_base_module, "compute_loss_weighting_for_sd3", lambda *a, **k: weighting)
    args = SimpleNamespace(weighting_scheme="cosmap")

    per_sample = trainer.compute_per_sample_loss(args, output, torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0)
    scalar, metrics = trainer.compute_loss(args, output, torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0)

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


def test_base_per_sample_loss_rejects_mismatched_batch_axis():
    trainer = NetworkTrainer()
    output = DiTOutput(pred=torch.zeros(2, 1), target=torch.zeros(3, 1))

    with pytest.raises(ValueError, match=r"prediction and target batch sizes differ"):
        trainer.compute_per_sample_loss(
            SimpleNamespace(weighting_scheme="none"), output, torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0
        )


def test_base_scalar_loss_and_gradient_match_direct_baseline_reduction(monkeypatch):
    trainer = NetworkTrainer()
    weighting = torch.tensor([0.75, 1.25]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(trainer_base_module, "compute_loss_weighting_for_sd3", lambda *args, **kwargs: weighting)
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
    old_elementwise = torch.nn.functional.mse_loss(old_parameter * inputs, target, reduction="none")
    old_loss = (old_elementwise * weighting).mean()
    new_grad = torch.autograd.grad(new_loss, new_parameter)[0]
    old_grad = torch.autograd.grad(old_loss, old_parameter)[0]

    assert torch.allclose(new_loss, old_loss, rtol=1e-5, atol=1e-8)
    assert torch.allclose(new_grad, old_grad, rtol=1e-5, atol=1e-8)
    assert metrics == {}
