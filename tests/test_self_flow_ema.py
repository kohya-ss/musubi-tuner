import torch

from musubi_tuner.flux_2_train_network_self_flow import (
    compute_ema_weight_drift,
    update_ema_weights,
)


def test_ema_lerp():
    ema = {"w": torch.zeros(4)}
    cur = {"w": torch.ones(4)}
    update_ema_weights(ema, cur, decay=0.9)
    assert torch.allclose(ema["w"], torch.full((4,), 0.1))


def test_ema_decay_one_freezes_teacher():
    ema = {"w": torch.zeros(4)}
    cur = {"w": torch.ones(4)}
    update_ema_weights(ema, cur, decay=1.0)
    assert torch.equal(ema["w"], torch.zeros(4))


def test_ema_non_float_copied():
    ema = {"step": torch.tensor(0)}
    cur = {"step": torch.tensor(5)}
    update_ema_weights(ema, cur, decay=0.9)
    assert ema["step"].item() == 5


def test_ema_updates_in_place():
    w = torch.zeros(4)
    ema = {"w": w}
    update_ema_weights(ema, {"w": torch.ones(4)}, decay=0.5)
    assert torch.allclose(w, torch.full((4,), 0.5))  # same tensor object mutated


def test_drift_zero_when_identical():
    state = {"w": torch.ones(4)}
    assert compute_ema_weight_drift(state, {"w": torch.ones(4)}).item() == 0.0


def test_drift_positive_when_different():
    ema = {"w": torch.zeros(4), "step": torch.tensor(3)}
    cur = {"w": torch.ones(4), "step": torch.tensor(9)}
    drift = compute_ema_weight_drift(ema, cur)
    assert drift.item() > 0.0
