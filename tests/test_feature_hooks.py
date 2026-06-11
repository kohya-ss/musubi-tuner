import pytest
import torch

from musubi_tuner.flux_2_train_network_self_flow import BlockFeatureExtractor

from .conftest import make_inputs


def _run(model, **kw):
    x, x_ids, ctx, ctx_ids, guidance = make_inputs(**kw)
    timesteps = torch.tensor([0.3, 0.7])
    return model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)


def test_double_block_capture_shape(tiny_model):
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])
    ex.arm(0, num_txt_tokens=3)
    _run(tiny_model)
    feats = ex.drain()
    assert feats.shape == (2, 4, 16)  # (B, N_img, hidden) — img stream only


def test_single_block_capture_slices_txt(tiny_model):
    # tiny model: 2 double + 2 single; global index 2 = single_blocks[0]
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [2])
    ex.arm(2, num_txt_tokens=3)
    _run(tiny_model)
    feats = ex.drain()
    assert feats.shape == (2, 4, 16)  # txt tokens sliced off


def test_disarmed_captures_nothing(tiny_model):
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])
    _run(tiny_model)
    assert ex.drain() is None


def test_drain_resets(tiny_model):
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])
    ex.arm(0, num_txt_tokens=3)
    _run(tiny_model)
    assert ex.drain() is not None
    assert ex.drain() is None


def test_captured_features_carry_grad(tiny_model):
    tiny_model.train()
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])
    ex.arm(0, num_txt_tokens=3)
    _run(tiny_model)
    feats = ex.drain()
    assert feats.requires_grad


def test_out_of_range_layer_raises(tiny_model):
    ex = BlockFeatureExtractor()
    with pytest.raises(ValueError, match="out of range"):
        ex.install(tiny_model, [99])


# --- Fix 2: arm() validation ---


def test_arm_uninstalled_layer_raises(tiny_model):
    """arm() on a layer that was never installed should raise ValueError."""
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])  # only layer 0 installed
    with pytest.raises(ValueError, match="not installed"):
        ex.arm(1, num_txt_tokens=3)  # layer 1 not installed


def test_arm_installed_layer_ok(tiny_model):
    """arm() on a properly installed layer should not raise."""
    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])
    ex.arm(0, num_txt_tokens=3)  # should not raise


# --- Fix 3: grad flow under gradient checkpointing ---


def test_grad_flows_through_features_with_gradient_checkpointing(tiny_model):
    """Features captured under gradient checkpointing must carry grad; backward must flow to params.

    Teeth-check: this test would catch a regression where drain() is called
    outside the checkpointed region, returning a detached tensor (requires_grad
    False) — so feats.sum().backward() would raise RuntimeError or block
    parameter grads would stay None.
    """
    tiny_model.enable_gradient_checkpointing()
    tiny_model.train()

    ex = BlockFeatureExtractor()
    ex.install(tiny_model, [0])
    ex.arm(0, num_txt_tokens=3)

    # At least one input must have requires_grad for reentrant=False checkpoint
    # to propagate gradients through the hooked block.
    from .conftest import make_inputs

    x, x_ids, ctx, ctx_ids, guidance = make_inputs()
    x = x.detach().requires_grad_(True)
    timesteps = torch.tensor([0.3, 0.7])
    tiny_model(x=x, x_ids=x_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance)

    feats = ex.drain()
    assert feats is not None
    assert feats.requires_grad, "features must carry grad (needed for L_rep backward)"

    feats.sum().backward()
    first_param = next(tiny_model.double_blocks[0].parameters())
    assert first_param.grad is not None, "backward must flow into double_blocks[0] parameters"
