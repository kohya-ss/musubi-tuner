"""Tests for PerTokenModulationController forward hooks (Task 7).

Three proof tests:
1. hooks installed but unstaged → bitwise identical to vanilla forward
2. uniform per-token map → allclose with the global 1D path
3. heterogeneous map → allclose with reference_per_token_forward (verbatim port
   of self-flow-network branch's modified Flux2.forward per-token path)
"""

import torch

from musubi_tuner.flux_2.flux2_models import AttentionParams, timestep_embedding
from musubi_tuner.flux_2_train_network_self_flow import PerTokenModulationController

from .conftest import make_inputs


def reference_per_token_forward(model, x, x_ids, tau, ctx, ctx_ids, guidance):
    """Ground truth: verbatim port of the self-flow-network branch's modified
    Flux2.forward per-token path (git show self-flow-network:src/musubi_tuner/
    flux_2/flux2_models.py, forward()). Kept in the test so the production
    model needs no modification."""
    num_txt_tokens = ctx.shape[1]
    B, N_img = tau.shape

    emb_flat = timestep_embedding(tau.reshape(-1), 256)
    vec_img = model.time_in(emb_flat).reshape(B, N_img, -1)
    if model.use_guidance_embed:
        guidance_emb = timestep_embedding(guidance, 256)
        vec_img = vec_img + model.guidance_in(guidance_emb).unsqueeze(1)

    vec_global = vec_img.mean(dim=1)

    double_block_mod_img = model.double_stream_modulation_img(vec_img)
    double_block_mod_txt = model.double_stream_modulation_txt(vec_global)
    vec_txt_expanded = vec_global.unsqueeze(1).expand(-1, num_txt_tokens, -1)
    vec_combined = torch.cat([vec_txt_expanded, vec_img], dim=1)
    single_block_mod, _ = model.single_stream_modulation(vec_combined)

    img = model.img_in(x)
    txt = model.txt_in(ctx)
    pe_x = model.pe_embedder(x_ids)
    pe_ctx = model.pe_embedder(ctx_ids)
    attn_params = AttentionParams.create_attention_params(model.attn_mode, model.split_attn)

    for block in model.double_blocks:
        img, txt = block(img, txt, pe_x, pe_ctx, double_block_mod_img, double_block_mod_txt, attn_params)

    img = torch.cat((txt, img), dim=1)
    pe = torch.cat((pe_ctx, pe_x), dim=2)
    for block in model.single_blocks:
        img = block(img, pe, single_block_mod, attn_params)

    img = img[:, num_txt_tokens:, ...]
    return model.final_layer(img, vec_img)


def test_unstaged_hooks_are_bitwise_noop(tiny_model):
    x, x_ids, ctx, ctx_ids, guidance = make_inputs()
    timesteps = torch.tensor([0.3, 0.7])

    baseline = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    controller = PerTokenModulationController()
    controller.install(tiny_model)
    hooked = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert torch.equal(baseline, hooked)  # bitwise — vanilla path untouched


def test_uniform_per_token_map_matches_global_path(tiny_model):
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)
    timesteps = torch.tensor([0.3, 0.7])

    baseline = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    controller = PerTokenModulationController()
    controller.install(tiny_model)
    tau = timesteps.unsqueeze(1).expand(B, N_img)  # every token at its sample's global t
    controller.stage(tau, num_txt_tokens=ctx.shape[1])
    try:
        per_token = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
    finally:
        controller.clear()

    assert torch.allclose(baseline, per_token, atol=1e-5)


def test_heterogeneous_map_matches_reference(tiny_model):
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(B=B, N_img=N_img)
    torch.manual_seed(7)
    tau = torch.rand(B, N_img)  # genuinely heterogeneous
    decoy = tau.max(dim=1).values  # what the trainer passes as 1D timesteps

    expected = reference_per_token_forward(tiny_model, x, x_ids, tau, ctx, ctx_ids, guidance)

    controller = PerTokenModulationController()
    controller.install(tiny_model)
    controller.stage(tau, num_txt_tokens=ctx.shape[1])
    try:
        actual = tiny_model(x=x, x_ids=x_ids, timesteps=decoy, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
    finally:
        controller.clear()

    assert torch.allclose(expected, actual, atol=1e-5)


def test_stage_clear_round_trip(tiny_model):
    x, x_ids, ctx, ctx_ids, guidance = make_inputs()
    timesteps = torch.tensor([0.3, 0.7])
    baseline = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    controller = PerTokenModulationController()
    controller.install(tiny_model)
    controller.stage(torch.rand(2, 4), num_txt_tokens=3)
    controller.clear()
    after = tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
    assert torch.equal(baseline, after)


def test_install_raises_on_missing_module(tiny_model):
    import pytest

    del tiny_model.time_in  # simulate upstream rename
    controller = PerTokenModulationController()
    with pytest.raises(AttributeError, match="time_in"):
        controller.install(tiny_model)


def test_unstaged_hooks_are_bitwise_noop_with_guidance(tiny_model_guidance):
    """Install hooks (including guidance_in hook) but don't stage — must be bitwise noop."""
    B = 2
    x, x_ids, ctx, ctx_ids, _ = make_inputs(B=B)
    timesteps = torch.tensor([0.3, 0.7])
    guidance = torch.full((B,), 1.0)

    baseline = tiny_model_guidance(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    controller = PerTokenModulationController()
    controller.install(tiny_model_guidance)
    hooked = tiny_model_guidance(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    assert torch.equal(baseline, hooked)  # bitwise — unstaged guidance path untouched


def test_heterogeneous_map_matches_reference_with_guidance(tiny_model_guidance):
    """Heterogeneous tau with guidance embed must match reference_per_token_forward."""
    B, N_img = 2, 4
    x, x_ids, ctx, ctx_ids, _ = make_inputs(B=B, N_img=N_img)
    guidance = torch.full((B,), 1.0)
    torch.manual_seed(7)
    tau = torch.rand(B, N_img)  # genuinely heterogeneous
    decoy = tau.max(dim=1).values  # what the trainer passes as 1D timesteps

    expected = reference_per_token_forward(tiny_model_guidance, x, x_ids, tau, ctx, ctx_ids, guidance)

    controller = PerTokenModulationController()
    controller.install(tiny_model_guidance)
    controller.stage(tau, num_txt_tokens=ctx.shape[1])
    try:
        actual = tiny_model_guidance(x=x, x_ids=x_ids, timesteps=decoy, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)
    finally:
        controller.clear()

    assert torch.allclose(expected, actual, atol=1e-5)
