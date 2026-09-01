import gc
import inspect
import weakref
from contextlib import nullcontext
from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer
from musubi_tuner.hidream_o1_train_network import HiDreamO1NetworkTrainer
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
import musubi_tuner.training.trainer_base as trainer_base_module


class _EasyDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error

    __setattr__ = dict.__setitem__


easydict_stub = ModuleType("easydict")
easydict_stub.EasyDict = _EasyDict
with patch.dict(sys.modules, {"easydict": easydict_stub, "flash_attn": None}):
    from musubi_tuner.wan_train_network import WanNetworkTrainer


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


def test_update_winners_rejects_nonvector_best_loss():
    _, _, update_winners = _explorative_helpers()
    with pytest.raises(ValueError, match=r"best loss must have shape \[B\]"):
        update_winners(
            torch.full((2, 1), torch.inf),
            torch.empty(2, 1),
            torch.full((2,), -1, dtype=torch.long),
            torch.tensor([1.0, 2.0]),
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
    with pytest.raises(ValueError, match=r"share dtype$"):
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


@pytest.mark.parametrize("shape", [(2, 1, 1, 2), (2, 1, 1, 1, 2)])
def test_base_per_sample_loss_applies_weight_before_nonbatch_reduction(monkeypatch, shape):
    trainer = NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([1.0, 3.0, 2.0, 4.0]).reshape(shape),
        target=torch.zeros(shape),
    )
    # This is the production helper's rank-5 contract, including for 4D images.
    weighting = torch.tensor([2.0, 0.5]).reshape(2, 1, 1, 1, 1)
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


@pytest.mark.parametrize(
    ("pred", "target", "message"),
    [
        (torch.zeros(2, 3, 1), torch.zeros(2, 1, 4), "prediction and target shapes must match"),
        (torch.zeros(0, 3), torch.zeros(0, 3), "per-sample loss requires a non-empty batch"),
        (torch.zeros(2, 0), torch.zeros(2, 0), "per-sample loss requires at least one element per batch sample"),
    ],
)
def test_base_per_sample_loss_rejects_malformed_objectives(pred, target, message):
    trainer = NetworkTrainer()

    with pytest.raises(ValueError, match=message):
        trainer.compute_per_sample_loss(
            SimpleNamespace(weighting_scheme="none"),
            DiTOutput(pred=pred, target=target),
            torch.ones(pred.shape[0]),
            None,
            torch.float32,
            torch.float32,
            0,
        )


def test_base_scalar_loss_and_gradient_match_direct_baseline_reduction(monkeypatch):
    trainer = NetworkTrainer()
    weighting = torch.tensor([0.75, 1.25]).reshape(2, 1, 1, 1, 1)
    monkeypatch.setattr(trainer_base_module, "compute_loss_weighting_for_sd3", lambda *args, **kwargs: weighting)
    new_parameter = torch.nn.Parameter(torch.tensor(0.3))
    old_parameter = torch.nn.Parameter(new_parameter.detach().clone())
    inputs = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1)
    target = torch.tensor([-1.0, 0.5]).reshape(2, 1, 1, 1, 1)
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


class _ToyAccelerator:
    def __init__(self, device="cpu", autocast_dtype=None):
        self.device = torch.device(device)
        self.autocast_dtype = autocast_dtype
        self.unwrap_calls = 0

    def autocast(self):
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(self.device.type, dtype=self.autocast_dtype)

    def unwrap_model(self, transformer):
        self.unwrap_calls += 1
        return transformer


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


class _BlockSwapProtocolTransformer(_ToyTransformer):
    def __init__(self):
        super().__init__()
        self.mode = "training"
        self.events = []

    def switch_block_swap_for_inference(self):
        assert self.mode == "training"
        self.mode = "forward-only"
        self.events.append("inference")

    def switch_block_swap_for_training(self):
        assert self.mode == "forward-only"
        self.mode = "training"
        self.events.append("training")

    def __call__(self, value):
        expected_mode = "training" if torch.is_grad_enabled() else "forward-only"
        assert self.mode == expected_mode
        self.events.append(f"forward:{expected_mode}")
        return super().__call__(value)


class _RealCudaBlockSwapTransformer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1, bias=False) for _ in range(3)])
        self.blocks.to(device)
        self.offloader = create_offloader(
            "test-xm",
            self.blocks,
            num_blocks=len(self.blocks),
            blocks_to_swap=1,
            config=BlockSwapConfig(device=torch.device(device), supports_backward=True),
        )
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self):
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_training(self):
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward(self, value):
        for index, block in enumerate(self.blocks):
            self.offloader.wait_for_block(index)
            value = block(value)
            self.offloader.submit_move_blocks_forward(self.blocks, index)
        return value


def test_block_swap_forward_only_context_is_reentrant_and_restores_after_error():
    trainer = NetworkTrainer()
    transformer = _BlockSwapProtocolTransformer()
    accelerator = _ToyAccelerator()

    with pytest.raises(RuntimeError, match="candidate failed"):
        with trainer.block_swap_forward_only(accelerator, transformer):
            with trainer.block_swap_forward_only(accelerator, transformer):
                raise RuntimeError("candidate failed")

    assert transformer.events == ["inference", "training"]
    assert transformer.mode == "training"
    assert accelerator.unwrap_calls == 2


def test_block_swap_forward_only_context_rejects_nested_different_transformer_and_restores_state():
    trainer = NetworkTrainer()
    transformer_a = _BlockSwapProtocolTransformer()
    transformer_b = _BlockSwapProtocolTransformer()
    accelerator = _ToyAccelerator()

    with pytest.raises(RuntimeError, match="same transformer instance"):
        with trainer.block_swap_forward_only(accelerator, transformer_a):
            with trainer.block_swap_forward_only(accelerator, transformer_b):
                pass

    assert transformer_a.events == ["inference", "training"]
    assert transformer_a.mode == "training"
    assert transformer_b.events == []
    assert transformer_b.mode == "training"
    assert trainer._block_swap_forward_only_depth == 0
    assert trainer._block_swap_forward_only_transformer is None


def test_block_swap_forward_only_context_allows_wrapped_and_unwrapped_same_transformer():
    class _Wrapper:
        def __init__(self, module):
            self.module = module

    class _UnwrappingAccelerator(_ToyAccelerator):
        def unwrap_model(self, transformer):
            self.unwrap_calls += 1
            return getattr(transformer, "module", transformer)

    trainer = NetworkTrainer()
    transformer = _BlockSwapProtocolTransformer()
    accelerator = _UnwrappingAccelerator()

    with trainer.block_swap_forward_only(accelerator, _Wrapper(transformer)) as outer:
        with trainer.block_swap_forward_only(accelerator, transformer) as inner:
            assert outer is transformer
            assert inner is transformer

    assert transformer.events == ["inference", "training"]
    assert transformer.mode == "training"
    assert accelerator.unwrap_calls == 2


def test_block_swap_forward_only_context_accepts_absent_protocol_and_rejects_half_protocol():
    trainer = NetworkTrainer()

    with trainer.block_swap_forward_only(_ToyAccelerator(), object()):
        pass

    class _HalfProtocol:
        def switch_block_swap_for_inference(self):
            return None

    with pytest.raises(RuntimeError, match="must provide both"):
        with trainer.block_swap_forward_only(_ToyAccelerator(), _HalfProtocol()):
            pass


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_sample_images_restores_runtime_state_when_sampling_raises(monkeypatch, tmp_path, device):
    class _SamplingTrainer(NetworkTrainer):
        def sample_image_inference(self, accelerator, *args, **kwargs):
            torch.rand((), device="cpu")
            if accelerator.device.type == "cuda":
                torch.rand((), device=accelerator.device)
            raise RuntimeError("sampling failed")

    monkeypatch.setattr(trainer_base_module, "should_sample_images", lambda *args: True)
    monkeypatch.setattr(trainer_base_module, "PartialState", lambda: SimpleNamespace(num_processes=1))
    cleaned_devices = []
    monkeypatch.setattr(trainer_base_module, "clean_memory_on_device", cleaned_devices.append)
    trainer = _SamplingTrainer()
    transformer = _BlockSwapProtocolTransformer()
    accelerator = _ToyAccelerator(device)
    args = SimpleNamespace(sample_prompts="prompts.toml", output_dir=str(tmp_path))
    torch.manual_seed(2026)
    cpu_rng_state = torch.get_rng_state().clone()
    cuda_rng_state = None
    if accelerator.device.type == "cuda":
        torch.cuda.manual_seed_all(2026)
        cuda_rng_state = torch.cuda.get_rng_state(accelerator.device).clone()

    with pytest.raises(RuntimeError, match="sampling failed"):
        trainer.sample_images(
            accelerator,
            args,
            epoch=1,
            steps=1,
            sample_resources=None,
            transformer=transformer,
            sample_parameters=[{}],
            dit_dtype=torch.float32,
        )

    assert transformer.events == ["inference", "training"]
    assert transformer.mode == "training"
    assert torch.equal(torch.get_rng_state(), cpu_rng_state)
    if cuda_rng_state is not None:
        assert torch.equal(torch.cuda.get_rng_state(accelerator.device), cuda_rng_state)
    assert cleaned_devices == [accelerator.device]


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
            torch.rand((), device=accelerator.device) if accelerator.device.type == "cuda" else torch.rand((), device="cpu")
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


def test_standard_xm_rejects_zero_iteration_internal_state_before_final_forward():
    trainer = _ToyXMTrainer()
    trainer._best_of_k_count = 0
    trainer._best_of_k_enabled = True
    transformer = _ToyTransformer()
    latents = torch.zeros(2, 1, 1, 1)
    noise = torch.ones_like(latents)

    with pytest.raises(RuntimeError, match="candidate loop ran zero iterations"):
        trainer.process_batch_best_of_k(
            _xm_args(),
            _ToyAccelerator(),
            transformer,
            None,
            {"timesteps": [0.5, 0.5], "condition": torch.ones(2, 1)},
            latents,
            noise,
            None,
            torch.float32,
            torch.float32,
            None,
            0,
        )

    assert transformer.forward_shapes == []


def test_standard_xm_selects_mixed_winners_and_builds_one_gradient_graph(monkeypatch):
    trainer, transformer, loss, metrics = _run_toy_xm(monkeypatch)

    assert [record["grad_enabled"] for record in trainer.records] == [False, False, True]
    torch.testing.assert_close(trainer.records[-1]["noise"][:, 0, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(trainer.records[-1]["target"], -trainer.records[-1]["noise"])
    assert all(torch.equal(record["timesteps"], trainer.records[0]["timesteps"]) for record in trainer.records)
    assert all(torch.equal(record["condition"], trainer.records[0]["condition"]) for record in trainer.records)
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


def test_standard_xm_uses_forward_only_for_candidates_and_training_for_winner(monkeypatch):
    _, transformer, loss, _ = _run_toy_xm(monkeypatch, transformer_factory=_BlockSwapProtocolTransformer)

    assert transformer.events == [
        "inference",
        "forward:forward-only",
        "forward:forward-only",
        "training",
        "forward:training",
    ]
    loss.backward()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real ModelOffloader coverage")
def test_standard_xm_real_cuda_model_offloader_runs_candidates_and_final_backward(monkeypatch):
    device = torch.device("cuda:0")
    trainer, transformer, loss, _ = _run_toy_xm(
        monkeypatch,
        device=device,
        transformer_factory=lambda: _RealCudaBlockSwapTransformer(device),
    )

    loss.backward()
    transformer.offloader.set_forward_only(False)

    assert transformer.offloader.forward_only is False
    assert trainer.scale.grad is not None
    assert torch.isfinite(trainer.scale.grad)
    assert trainer.scale.grad.abs().item() > 0


def test_standard_xm_preserves_final_architecture_metrics(monkeypatch):
    class _MetricToyTrainer(_ToyXMTrainer):
        def compute_loss(self, *args, **kwargs):
            loss, metrics = super().compute_loss(*args, **kwargs)
            return loss, {**metrics, "loss/architecture": 7.0}

    _, _, _, metrics = _run_toy_xm(monkeypatch, trainer_factory=_MetricToyTrainer)

    assert metrics == {
        "loss/architecture": 7.0,
        "xm/candidate_loss_mean": 11.5,
        "xm/selection_gain": 6.0,
    }


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

    _, transformer, _, _ = _run_toy_xm(monkeypatch, transformer_factory=_CompiledToyTransformer)
    assert transformer.forward_shapes == [(2, 1, 1, 1)] * 3


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
    initial_device = torch.cuda.get_rng_state(torch.device(device)).clone() if device == "cuda" else None
    captured = {}
    real_create = trainer_base_module.create_candidate_generator

    def capture_after_generator(reference):
        generator = real_create(reference)
        captured["cpu"] = torch.random.get_rng_state().clone()
        if reference.device.type == "cuda":
            captured["device"] = torch.cuda.get_rng_state(reference.device).clone()
        return generator

    monkeypatch.setattr(trainer_base_module, "create_candidate_generator", capture_after_generator)
    trainer, _, _, metrics = _run_toy_xm(monkeypatch, device=device)
    first_winner = trainer.records[-1]["noise"].clone()
    first_post_cpu = torch.random.get_rng_state().clone()
    first_post_device = torch.cuda.get_rng_state(torch.device(device)).clone() if device == "cuda" else None

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
        assert torch.equal(torch.cuda.get_rng_state(torch.device(device)), first_post_device)

    torch.random.set_rng_state(initial_cpu)
    if device == "cuda":
        torch.cuda.set_rng_state(initial_device, torch.device(device))
    replay_trainer, _, _, replay_metrics = _run_toy_xm(monkeypatch, device=device)
    torch.testing.assert_close(replay_trainer.records[-1]["noise"], first_winner)
    assert replay_metrics == metrics
    assert torch.equal(torch.random.get_rng_state(), first_post_cpu)
    if device == "cuda":
        assert torch.equal(torch.cuda.get_rng_state(torch.device(device)), first_post_device)


class _ToyWanTrainer(WanNetworkTrainer):
    def __init__(self):
        NetworkTrainer.__init__(self)
        self.high_low_training = True
        self.timestep_boundary = 0.5
        self.num_timestep_buckets = 1
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.noising_calls = 0
        self.resident_model_is_high_noise = False

    def get_bucketed_timestep(self):
        return 0.75

    def get_noisy_model_input_and_timesteps(self, *args, **kwargs):
        self.noising_calls += 1
        return super().get_noisy_model_input_and_timesteps(*args, **kwargs)


def test_standard_xm_freezes_wan_route_and_uses_one_resident_transition(monkeypatch):
    trainer = _ToyWanTrainer()
    assert trainer.call_dit.__func__ is WanNetworkTrainer.call_dit
    trainer._best_of_k_count = 2
    trainer._best_of_k_enabled = True
    swap_requests = []
    transitions = []
    forward_records = []
    block_swap_transformer = _BlockSwapProtocolTransformer()

    def fake_swap_high_low_weights(args, accelerator, model):
        del args, accelerator, model
        requested = bool(trainer.next_model_is_high_noise)
        swap_requests.append(requested)
        if trainer.resident_model_is_high_noise != requested:
            transitions.append((trainer.resident_model_is_high_noise, requested))
            trainer.resident_model_is_high_noise = requested

    def fake_call_dit(
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
        del args, accelerator, batch, kwargs
        features = transformer(noisy_model_input)
        forward_records.append(
            {
                "resident_high_noise": trainer.resident_model_is_high_noise,
                "timestep": timesteps.detach().clone(),
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return DiTOutput(
            pred=trainer.scale.to(network_dtype) * features.to(network_dtype),
            target=(latents - noise).to(network_dtype),
        )

    monkeypatch.setattr(trainer, "swap_high_low_weights", fake_swap_high_low_weights)
    monkeypatch.setattr(trainer, "_call_dit", fake_call_dit)
    latents = torch.zeros(2, 1, 1, 1)
    noise = torch.tensor([1.0, 4.0]).reshape(2, 1, 1, 1)
    later = torch.tensor([5.0, 2.0]).reshape(2, 1, 1, 1)
    monkeypatch.setattr(
        trainer_base_module,
        "draw_candidate_noise",
        lambda reference, generator: later,
    )
    loss, _ = trainer.process_batch_best_of_k(
        _xm_args(),
        _ToyAccelerator(),
        block_swap_transformer,
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
    assert swap_requests == [True, True, True]
    assert transitions == [(False, True)]
    assert trainer.resident_model_is_high_noise is True
    assert [record["resident_high_noise"] for record in forward_records] == [True, True, True]
    assert [record["grad_enabled"] for record in forward_records] == [False, False, True]
    assert all(torch.equal(record["timestep"], forward_records[0]["timestep"]) for record in forward_records[1:])
    assert block_swap_transformer.events == [
        "inference",
        "forward:forward-only",
        "forward:forward-only",
        "training",
        "forward:training",
    ]

    loss.backward()
    assert trainer.scale.grad is not None
    assert torch.isfinite(trainer.scale.grad)
    assert trainer.scale.grad.abs().item() > 0


def test_standard_xm_samples_distribution_preserving_timestep_once(monkeypatch):
    args = _xm_args()
    args.preserve_distribution_shape = True
    trainer, _, _, _ = _run_toy_xm(monkeypatch, args=args)
    assert trainer.noising_calls == 1


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

    with pytest.raises(ValueError, match=r"Synthetic.*candidate 1.*sample indices \[1\]"):
        _run_toy_xm(monkeypatch, trainer_factory=_NaNTrainer)


def test_weighted_per_sample_selection_is_not_a_whole_candidate_reduction(monkeypatch):
    _, _, update_winners = _explorative_helpers()
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
        return trainer.compute_per_sample_loss(args, output, timesteps, None, torch.float32, torch.float32, 0)

    candidate_zero = score([1.0, 4.0])
    candidate_one = score([2.0, 1.0])
    best = torch.full((2,), torch.inf)
    winner_noise = torch.empty(2, 1)
    indices = torch.full((2,), -1, dtype=torch.long)
    best, winner_noise, indices = update_winners(best, winner_noise, indices, candidate_zero, torch.tensor([[0.0], [0.0]]), 0)
    best, winner_noise, indices = update_winners(best, winner_noise, indices, candidate_one, torch.tensor([[1.0], [1.0]]), 1)

    assert indices.tolist() == [0, 1]
    assert candidate_zero.mean() < candidate_one.mean()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_standard_xm_cuda_autocast_recomputes_cached_weight_with_gradients(monkeypatch):
    class _AutocastLinearTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1, bias=False, device="cuda", dtype=torch.float32)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)

        def forward(self, value):
            return self.linear(value)

    class _AutocastToyTrainer(_ToyXMTrainer):
        def __init__(self, device="cpu"):
            super().__init__(device)
            with torch.no_grad():
                self.scale.fill_(1.0)

    real_clear_autocast_cache = torch.clear_autocast_cache
    clear_callers = []

    def record_clear_autocast_cache():
        caller = inspect.currentframe().f_back
        clear_callers.append(Path(caller.f_code.co_filename).name)
        return real_clear_autocast_cache()

    monkeypatch.setattr(torch, "clear_autocast_cache", record_clear_autocast_cache)
    trainer, transformer, loss, _ = _run_toy_xm(
        monkeypatch,
        device="cuda",
        autocast_dtype=torch.float16,
        transformer_factory=_AutocastLinearTransformer,
        trainer_factory=_AutocastToyTrainer,
    )
    loss.backward()

    weight_grad = transformer.linear.weight.grad
    assert weight_grad is not None
    assert torch.isfinite(weight_grad).all()
    assert weight_grad.abs().item() > 0
    assert clear_callers == ["autocast_mode.py"] * 3
    assert "clear_autocast_cache" not in inspect.getsource(NetworkTrainer.process_batch_best_of_k)
