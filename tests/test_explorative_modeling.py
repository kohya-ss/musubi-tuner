from importlib import import_module
from pathlib import Path
import sys

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
