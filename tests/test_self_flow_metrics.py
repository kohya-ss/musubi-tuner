"""Tests for self-flow training metrics: EMA weight drift and feature cosine similarity."""

import torch
import pytest
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    return NetworkTrainer()


# ---------------------------------------------------------------------------
# RED: _compute_ema_weight_drift
# ---------------------------------------------------------------------------

def test_ema_drift_zero_for_identical_states(trainer):
    """EMA drift is 0 when EMA and current weights are identical."""
    state = {
        "layer.weight": torch.tensor([1.0, 2.0, 3.0]),
        "layer.bias": torch.tensor([0.1, 0.2]),
    }
    drift = trainer._compute_ema_weight_drift(state, state)
    assert drift.item() == pytest.approx(0.0, abs=1e-6)


def test_ema_drift_positive_for_different_states(trainer):
    """EMA drift is > 0 when weights have diverged."""
    ema_state = {"w": torch.tensor([1.0, 0.0])}
    cur_state = {"w": torch.tensor([0.0, 1.0])}
    drift = trainer._compute_ema_weight_drift(ema_state, cur_state)
    assert drift.item() > 0.0


def test_ema_drift_larger_when_weights_differ_more(trainer):
    """Larger weight differences produce larger drift."""
    base = {"w": torch.zeros(4)}
    small_diff = {"w": torch.ones(4) * 0.1}
    large_diff = {"w": torch.ones(4) * 10.0}

    drift_small = trainer._compute_ema_weight_drift(base, small_diff)
    drift_large = trainer._compute_ema_weight_drift(base, large_diff)

    assert drift_large.item() > drift_small.item()


def test_ema_drift_skips_non_floating_point_params(trainer):
    """Integer buffers (e.g. step counters) should not affect drift."""
    ema_state = {
        "weight": torch.tensor([1.0, 2.0]),
        "step": torch.tensor(100, dtype=torch.int64),  # integer buffer
    }
    cur_state = {
        "weight": torch.tensor([1.0, 2.0]),
        "step": torch.tensor(999, dtype=torch.int64),  # different int
    }
    drift = trainer._compute_ema_weight_drift(ema_state, cur_state)
    assert drift.item() == pytest.approx(0.0, abs=1e-6)


def test_ema_drift_returns_scalar_tensor(trainer):
    """Return value should be a scalar tensor (0-dim)."""
    state = {"w": torch.randn(8)}
    drift = trainer._compute_ema_weight_drift(state, state)
    assert drift.shape == torch.Size([])


# ---------------------------------------------------------------------------
# RED: self_flow_logs contains feat_cosine_sim and ema_weight_drift
# ---------------------------------------------------------------------------

def test_self_flow_logs_contain_feat_cosine_sim_key(trainer):
    """
    When L_rep is computed, logs should include raw cosine similarity
    (positive value: how aligned student/teacher reps are).
    """
    # Simulate the metrics dict being built with L_rep available
    # feat_cosine_sim = -L_rep (since L_rep = -cos_sim.mean())
    L_gen = torch.tensor(0.5)
    L_rep = torch.tensor(-0.6)  # means cosine_sim = 0.6

    trainer._self_flow_logs = {}
    trainer._update_self_flow_logs(
        L_gen=L_gen,
        L_rep=L_rep,
        timesteps_student=torch.tensor([400, 600]),
        timesteps_teacher=torch.tensor([100, 200]),
        mask_flat=torch.zeros(2, 16, dtype=torch.bool),
        per_token_timesteps_student=torch.full((2, 16), 400),
        ema_weight_drift=torch.tensor(0.03),
    )

    assert "self_flow/feat_cosine_sim" in trainer._self_flow_logs
    assert trainer._self_flow_logs["self_flow/feat_cosine_sim"] == pytest.approx(0.6, abs=1e-5)


def test_self_flow_logs_contain_ema_weight_drift_key(trainer):
    """Logs must include EMA weight drift metric."""
    trainer._self_flow_logs = {}
    trainer._update_self_flow_logs(
        L_gen=torch.tensor(0.5),
        L_rep=None,
        timesteps_student=torch.tensor([400]),
        timesteps_teacher=torch.tensor([100]),
        mask_flat=torch.zeros(1, 16, dtype=torch.bool),
        per_token_timesteps_student=torch.full((1, 16), 400),
        ema_weight_drift=torch.tensor(0.07),
    )

    assert "self_flow/ema_weight_drift" in trainer._self_flow_logs
    assert trainer._self_flow_logs["self_flow/ema_weight_drift"] == pytest.approx(0.07, abs=1e-5)


def test_self_flow_logs_feat_cosine_sim_none_when_no_rep(trainer):
    """When L_rep is None (no feature layers), feat_cosine_sim should be 0.0."""
    trainer._self_flow_logs = {}
    trainer._update_self_flow_logs(
        L_gen=torch.tensor(0.5),
        L_rep=None,
        timesteps_student=torch.tensor([400]),
        timesteps_teacher=torch.tensor([100]),
        mask_flat=torch.zeros(1, 16, dtype=torch.bool),
        per_token_timesteps_student=torch.full((1, 16), 400),
        ema_weight_drift=torch.tensor(0.0),
    )

    assert trainer._self_flow_logs["self_flow/feat_cosine_sim"] == pytest.approx(0.0, abs=1e-5)
