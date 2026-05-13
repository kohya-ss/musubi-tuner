"""
Tests for per-token timestep conditioning in Flux2 model.

The original (B,) global timestep path must work unchanged.
The new (B, N) per-token path is only used during self-flow training.
"""

import torch
import pytest
from musubi_tuner.flux_2.flux2_models import Flux2, Flux2Params


@pytest.fixture
def tiny_params():
    """Minimal Flux2 config for fast tests — not real weights."""
    # hidden_size=16, num_heads=2 → pe_dim=8, axes_dim must sum to 8
    return Flux2Params(
        in_channels=4,
        context_in_dim=8,
        hidden_size=16,
        num_heads=2,
        depth=1,
        depth_single_blocks=1,
        axes_dim=[2, 2, 2, 2],
        theta=2000,
        mlp_ratio=2.0,
        use_guidance_embed=False,
    )


@pytest.fixture
def tiny_model(tiny_params):
    model = Flux2(tiny_params, attn_mode="torch")
    model.eval()
    return model


def make_inputs(B=2, N_img=4, N_txt=3, in_channels=4, hidden_ctx=8):
    """Build minimal inputs for Flux2.forward."""
    torch.manual_seed(0)
    x = torch.randn(B, N_img, in_channels)
    x_ids = torch.zeros(B, N_img, 4, dtype=torch.long)
    ctx = torch.randn(B, N_txt, hidden_ctx)
    ctx_ids = torch.zeros(B, N_txt, 4, dtype=torch.long)
    guidance = None
    return x, x_ids, ctx, ctx_ids, guidance


# ---------------------------------------------------------------------------
# Regression: global (B,) timestep path must work exactly as before
# ---------------------------------------------------------------------------


def test_flux2_global_timesteps_still_work(tiny_model):
    """Original (B,) timestep path must produce correct-shaped output."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)
    timesteps = torch.tensor([0.3, 0.7])  # (B,) global

    out = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert out.shape == (B, N_img, tiny_model.out_channels)


def test_flux2_global_timesteps_deterministic(tiny_model):
    """Same global timestep input → same output (sanity check)."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)
    timesteps = torch.tensor([0.3, 0.7])

    out1 = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
    out2 = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# RED: per-token (B, N) timestep path — new behavior
# ---------------------------------------------------------------------------


def test_flux2_per_token_timesteps_correct_output_shape(tiny_model):
    """Per-token (B, N_img) timesteps should produce same-shaped output as global."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)
    # Per-token timesteps: each token has its own noise level
    timesteps = torch.rand(B, N_img)  # (B, N_img) in [0, 1]

    out = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert out.shape == (B, N_img, tiny_model.out_channels), f"Expected ({B}, {N_img}, {tiny_model.out_channels}), got {out.shape}"


def test_flux2_uniform_per_token_matches_global(tiny_model):
    """When all tokens have the same per-token timestep, output should match global."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)

    t_val = torch.tensor([0.3, 0.7])  # (B,)

    # Global: one scalar per batch item
    out_global = tiny_model(x=x, x_ids=x_ids, timesteps=t_val, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    # Per-token: every token in each batch item gets the same value
    t_per_token = t_val.unsqueeze(1).expand(B, N_img)  # (B, N_img) all same
    out_per_token = tiny_model(x=x, x_ids=x_ids, timesteps=t_per_token, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert torch.allclose(out_global, out_per_token, atol=1e-5), "Uniform per-token timesteps should match global timestep output"


def test_flux2_varied_per_token_differs_from_global(tiny_model):
    """When per-token timesteps vary, output should differ from global."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)

    t_global = torch.tensor([0.3, 0.7])
    # Mix of teacher (0.1) and student (0.8) timesteps per token
    t_per_token = torch.tensor([[0.1, 0.8, 0.1, 0.8], [0.1, 0.8, 0.1, 0.8]])  # (B, N_img)

    out_global = tiny_model(x=x, x_ids=x_ids, timesteps=t_global, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
    out_per_token = tiny_model(x=x, x_ids=x_ids, timesteps=t_per_token, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert not torch.allclose(out_global, out_per_token, atol=1e-5), (
        "Varied per-token timesteps should produce different output than global"
    )


def test_flux2_per_token_timesteps_raises_when_ref_tokens_not_covered(tiny_model):
    """Per-token timesteps that don't cover ref tokens must raise RuntimeError.

    This documents the bug: call_dit passes (B, N_noisy) timesteps but x
    has N_noisy + N_ref tokens, causing a shape mismatch in modulation.
    """
    B, N_noisy, N_ref = 2, 4, 3
    x_noisy, _, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_noisy)
    x_ref = torch.randn(B, N_ref, x_noisy.shape[-1])
    x = torch.cat([x_noisy, x_ref], dim=1)
    x_ids = torch.zeros(B, N_noisy + N_ref, 4, dtype=torch.long)

    timesteps = torch.rand(B, N_noisy)  # only N_noisy — missing N_ref coverage

    with pytest.raises(RuntimeError):
        tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)


def test_flux2_per_token_timesteps_padded_for_ref_tokens(tiny_model):
    """Per-token timesteps padded with t=0 for ref tokens must work.

    This is the correct call pattern after the call_dit fix: extend
    per_token_timesteps to cover ref/control tokens (at t=0, clean).
    """
    B, N_noisy, N_ref = 2, 4, 3
    x_noisy, _, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_noisy)
    x_ref = torch.randn(B, N_ref, x_noisy.shape[-1])
    x = torch.cat([x_noisy, x_ref], dim=1)
    x_ids = torch.zeros(B, N_noisy + N_ref, 4, dtype=torch.long)

    t_noisy = torch.rand(B, N_noisy)
    t_ref = torch.zeros(B, N_ref)  # ref tokens are clean (t=0)
    timesteps = torch.cat([t_noisy, t_ref], dim=1)  # (B, N_noisy + N_ref)

    out = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert out.shape == (B, N_noisy + N_ref, tiny_model.out_channels)


def test_flux2_per_token_hidden_features_work(tiny_model):
    """Per-token timesteps should work with hidden_features=True."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)
    timesteps = torch.rand(B, N_img)

    out, features = tiny_model(
        x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance, hidden_features=True
    )

    assert out.shape == (B, N_img, tiny_model.out_channels)
    assert features is not None
    assert features.shape[0] == B
