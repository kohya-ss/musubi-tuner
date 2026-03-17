"""
Tests for per-token timestep conditioning in self-flow training.

Root cause: _apply_per_token_mask creates inputs at two noise levels but the model
receives a single global timestep. This causes contradictory gradients on masked tokens.

Fix requires:
1. _apply_per_token_mask returns the mask so callers can build per-token timestep maps
2. _build_per_token_timestep_map creates (B, N) per-token timestep tensor
3. Flux2.forward handles (B, N) timesteps for per-token conditioning
"""

import torch
import pytest
from musubi_tuner.hv_train_network import NetworkTrainer


@pytest.fixture
def trainer():
    return NetworkTrainer()


# ---------------------------------------------------------------------------
# Document the bug: mixed-noise input with a single global timestep
# ---------------------------------------------------------------------------

def test_masked_input_has_two_noise_levels_but_single_timestep(trainer):
    """
    Demonstrates the root cause of distortions.

    After masking, the student input contains tokens at two different noise levels
    (some at t_teacher, some at t_student). Without per-token timesteps, the model
    receives a single t_student for all tokens, creating wrong gradients on masked tokens.
    """
    torch.manual_seed(0)
    latents = torch.ones(1, 4, 8, 8)
    noise = torch.zeros(1, 4, 8, 8)

    t_teacher = torch.tensor([100])   # cleaner: x = 0.9*latents + 0.1*noise
    t_student = torch.tensor([800])   # noisier: x = 0.2*latents + 0.8*noise

    t_teacher_f = (t_teacher.float() - 1.0) / 1000.0
    t_student_f = (t_student.float() - 1.0) / 1000.0

    noisy_teacher = (1 - t_teacher_f.view(-1, 1, 1, 1)) * latents + t_teacher_f.view(-1, 1, 1, 1) * noise
    noisy_student = (1 - t_student_f.view(-1, 1, 1, 1)) * latents + t_student_f.view(-1, 1, 1, 1) * noise

    # Teacher tokens are cleaner (closer to latents=1)
    assert noisy_teacher.mean() > noisy_student.mean(), "Teacher should be cleaner"

    # After masking: student input has mixed noise levels
    masked, mask = trainer._apply_per_token_mask(noisy_student, noisy_teacher, 0.3, torch.device("cpu"))

    # Masked positions have teacher noise level (cleaner)
    mask_spatial = mask  # (B, N)
    mask_4d = mask_spatial.view(1, 1, 8, 8).expand_as(masked)

    teacher_positions = masked[mask_4d].mean()
    student_positions = masked[~mask_4d].mean()

    # Teacher positions (masked) should be closer to latents value (1.0)
    assert teacher_positions > student_positions, (
        "Masked positions should have teacher (cleaner) noise level"
    )
    # But the model will be told t_student for ALL tokens — this is the bug.


# ---------------------------------------------------------------------------
# RED: _apply_per_token_mask must return (masked_input, mask)
# ---------------------------------------------------------------------------

def test_apply_per_token_mask_returns_tuple_with_mask_4d(trainer):
    """_apply_per_token_mask should return (masked_input, mask) where mask is (B, N)."""
    torch.manual_seed(42)
    student = torch.ones(2, 4, 8, 8)
    teacher = torch.zeros(2, 4, 8, 8)

    result = trainer._apply_per_token_mask(student, teacher, 0.3, torch.device("cpu"))

    assert isinstance(result, tuple), "Should return a tuple (masked_input, mask)"
    assert len(result) == 2, "Should return exactly 2 values"

    masked_input, mask = result
    assert masked_input.shape == student.shape, "First element should be masked input"
    assert mask.shape == (2, 8 * 8), "Second element should be flat spatial mask (B, N)"


def test_apply_per_token_mask_returns_tuple_with_mask_5d(trainer):
    """_apply_per_token_mask returns (masked_input, mask) for 5D video tensors."""
    torch.manual_seed(42)
    student = torch.ones(2, 4, 4, 8, 8)
    teacher = torch.zeros(2, 4, 4, 8, 8)

    result = trainer._apply_per_token_mask(student, teacher, 0.3, torch.device("cpu"))

    assert isinstance(result, tuple)
    masked_input, mask = result
    assert masked_input.shape == student.shape
    assert mask.shape == (2, 4 * 8 * 8), "Mask should be (B, T*H*W) for 5D input"


def test_apply_per_token_mask_mask_true_means_teacher_token(trainer):
    """mask=True at a position means that position uses teacher input."""
    torch.manual_seed(42)
    student = torch.ones(1, 4, 4, 4)
    teacher = torch.zeros(1, 4, 4, 4)

    masked_input, mask = trainer._apply_per_token_mask(student, teacher, 0.5, torch.device("cpu"))

    # mask is (B, N), check per-token: True → teacher (0), False → student (1)
    B, N = mask.shape
    masked_reshaped = masked_input.reshape(B, 4, N)  # (B, C, N)
    for i in range(N):
        if mask[0, i]:
            assert (masked_reshaped[0, :, i] == 0).all(), f"Token {i}: mask=True should use teacher (0)"
        else:
            assert (masked_reshaped[0, :, i] == 1).all(), f"Token {i}: mask=False should use student (1)"


# ---------------------------------------------------------------------------
# RED: _build_per_token_timestep_map — new function that does not exist yet
# ---------------------------------------------------------------------------

def test_build_per_token_timestep_map_assigns_teacher_to_masked(trainer):
    """
    _build_per_token_timestep_map(t_teacher, t_student, mask) → (B, N) tensor.
    Masked positions (mask=True) get teacher timestep.
    Unmasked positions (mask=False) get student timestep.
    """
    t_teacher = torch.tensor([100, 200])   # (B,)
    t_student = torch.tensor([500, 600])   # (B,)

    # mask: (B, N) - True = masked = use teacher
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[0, :4] = True   # first 4 tokens masked in batch 0
    mask[1, 8:] = True   # last 8 tokens masked in batch 1

    timestep_map = trainer._build_per_token_timestep_map(t_teacher, t_student, mask)

    assert timestep_map.shape == (2, 16)

    # Batch 0: teacher=100, student=500
    assert (timestep_map[0, :4] == 100).all(),  "Masked tokens should get t_teacher"
    assert (timestep_map[0, 4:] == 500).all(),  "Unmasked tokens should get t_student"

    # Batch 1: teacher=200, student=600
    assert (timestep_map[1, :8] == 600).all(),  "Unmasked tokens should get t_student"
    assert (timestep_map[1, 8:] == 200).all(),  "Masked tokens should get t_teacher"


def test_build_per_token_timestep_map_no_mask_all_student(trainer):
    """With empty mask (all False), all tokens get student timestep."""
    t_teacher = torch.tensor([100])
    t_student = torch.tensor([500])
    mask = torch.zeros(1, 32, dtype=torch.bool)

    timestep_map = trainer._build_per_token_timestep_map(t_teacher, t_student, mask)

    assert (timestep_map == 500).all()


def test_build_per_token_timestep_map_all_masked_all_teacher(trainer):
    """With fully masked input, all tokens get teacher timestep."""
    t_teacher = torch.tensor([100])
    t_student = torch.tensor([500])
    mask = torch.ones(1, 32, dtype=torch.bool)

    timestep_map = trainer._build_per_token_timestep_map(t_teacher, t_student, mask)

    assert (timestep_map == 100).all()


def test_build_per_token_timestep_map_dtype_preserved(trainer):
    """Output dtype matches input timestep dtype."""
    t_teacher = torch.tensor([100], dtype=torch.float32)
    t_student = torch.tensor([500], dtype=torch.float32)
    mask = torch.zeros(1, 16, dtype=torch.bool)

    timestep_map = trainer._build_per_token_timestep_map(t_teacher, t_student, mask)

    assert timestep_map.dtype == torch.float32


# ---------------------------------------------------------------------------
# RED: Flux2 model forward accepts per-token (B, N) timesteps
# ---------------------------------------------------------------------------

def test_flux2_modulation_handles_per_token_vec():
    """
    Flux2 Modulation already handles (B, N, D) vec without modification
    because the ndim==2 check is skipped for 3D input.
    This test verifies the existing behavior is correct for per-token.
    """
    from musubi_tuner.flux_2.flux2_models import Modulation

    hidden = 64
    mod = Modulation(dim=hidden, double=True)
    mod.eval()

    B, N = 2, 16

    # Global conditioning: (B, D) → Modulation adds (B, 1, D*mult) → broadcasts
    vec_global = torch.randn(B, hidden)
    out_global, _ = mod(vec_global)
    shift_global, scale_global, gate_global = out_global
    assert shift_global.shape == (B, 1, hidden), f"Expected (B, 1, D), got {shift_global.shape}"

    # Per-token conditioning: (B, N, D) → Modulation gives (B, N, D*mult)
    vec_per_token = torch.randn(B, N, hidden)
    out_per_token, _ = mod(vec_per_token)
    shift_per_token, scale_per_token, gate_per_token = out_per_token
    assert shift_per_token.shape == (B, N, hidden), (
        f"Per-token vec should give (B, N, D) modulation, got {shift_per_token.shape}"
    )


def test_flux2_last_layer_handles_per_token_vec():
    """
    Flux2 LastLayer already handles (B, N, D) vec without modification
    because the ndim==2 check is skipped for 3D input.
    """
    from musubi_tuner.flux_2.flux2_models import LastLayer

    hidden = 64
    out_channels = 128
    layer = LastLayer(hidden_size=hidden, out_channels=out_channels)
    layer.eval()

    B, N = 2, 16
    x = torch.randn(B, N, hidden)

    # Global: (B, D) vec → shifts broadcast to all tokens
    vec_global = torch.randn(B, hidden)
    out_global = layer(x, vec_global)
    assert out_global.shape == (B, N, out_channels)

    # Per-token: (B, N, D) vec → each token gets its own shift/scale
    vec_per_token = torch.randn(B, N, hidden)
    out_per_token = layer(x, vec_per_token)
    assert out_per_token.shape == (B, N, out_channels)

    # Per-token output should differ from global (different conditioning per token)
    assert not torch.allclose(out_global, out_per_token), (
        "Per-token and global conditioning should produce different outputs"
    )
