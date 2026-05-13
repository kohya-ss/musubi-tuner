"""Tests for self-flow EMA weight updates."""

import torch
import pytest
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    """Create a NetworkTrainer instance to access helper methods."""
    return NetworkTrainer()


def test_ema_update_formula(trainer):
    """Verify EMA update formula: ema = decay * ema + (1 - decay) * current."""
    decay = 0.9

    ema_state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    current_state = {"weight": torch.tensor([10.0, 20.0, 30.0])}

    trainer._update_ema_weights(ema_state, current_state, decay)

    # Expected: 0.9 * [1,2,3] + 0.1 * [10,20,30] = [1.9, 3.8, 5.7]
    expected = torch.tensor([1.9, 3.8, 5.7])
    assert torch.allclose(ema_state["weight"], expected, atol=1e-5)


def test_ema_converges_to_current(trainer):
    """With decay=0, EMA should immediately equal current weights."""
    decay = 0.0

    ema_state = {"weight": torch.tensor([1.0, 2.0, 3.0])}
    current_state = {"weight": torch.tensor([10.0, 20.0, 30.0])}

    trainer._update_ema_weights(ema_state, current_state, decay)

    assert torch.equal(ema_state["weight"], current_state["weight"])


def test_ema_frozen_with_decay_one(trainer):
    """With decay=1, EMA should not change."""
    decay = 1.0

    original = torch.tensor([1.0, 2.0, 3.0])
    ema_state = {"weight": original.clone()}
    current_state = {"weight": torch.tensor([10.0, 20.0, 30.0])}

    trainer._update_ema_weights(ema_state, current_state, decay)

    assert torch.equal(ema_state["weight"], original)


def test_ema_multiple_updates_converge(trainer):
    """Multiple EMA updates should slowly converge toward current value."""
    decay = 0.9999

    ema_state = {"weight": torch.tensor([0.0])}
    current_state = {"weight": torch.tensor([1.0])}

    # Apply many updates
    for _ in range(10000):
        trainer._update_ema_weights(ema_state, current_state, decay)

    # With decay=0.9999, after 10k steps: 1 - exp(-1) ≈ 0.632
    assert ema_state["weight"].item() > 0.6


def test_ema_handles_non_floating_point(trainer):
    """Non-floating point tensors (int, bool) should be copied not lerped."""
    decay = 0.9

    ema_state = {
        "int_buffer": torch.tensor([1, 2, 3], dtype=torch.int64),
        "bool_buffer": torch.tensor([True, False], dtype=torch.bool),
    }
    current_state = {
        "int_buffer": torch.tensor([10, 20, 30], dtype=torch.int64),
        "bool_buffer": torch.tensor([False, True], dtype=torch.bool),
    }

    trainer._update_ema_weights(ema_state, current_state, decay)

    # Should be exact copy, not interpolated
    assert torch.equal(ema_state["int_buffer"], current_state["int_buffer"])
    assert torch.equal(ema_state["bool_buffer"], current_state["bool_buffer"])


def test_ema_modifies_in_place(trainer):
    """Verify EMA state is modified in-place."""
    decay = 0.9

    ema_state = {"weight": torch.tensor([1.0, 2.0])}
    original_tensor = ema_state["weight"]
    current_state = {"weight": torch.tensor([10.0, 20.0])}

    trainer._update_ema_weights(ema_state, current_state, decay)

    # Should be same tensor object (in-place modification)
    assert ema_state["weight"] is original_tensor
    assert not torch.equal(original_tensor, torch.tensor([1.0, 2.0]))


def test_ema_realistic_decay_rate(trainer):
    """Test with realistic decay rate (0.9999) used in paper."""
    decay = 0.9999

    ema_state = {"weight": torch.tensor([5.0])}
    current_state = {"weight": torch.tensor([6.0])}

    trainer._update_ema_weights(ema_state, current_state, decay)

    # Expected: 0.9999 * 5.0 + 0.0001 * 6.0 = 4.9995 + 0.0006 = 5.0001
    expected = 0.9999 * 5.0 + 0.0001 * 6.0
    assert torch.allclose(ema_state["weight"], torch.tensor([expected]), atol=1e-6)
