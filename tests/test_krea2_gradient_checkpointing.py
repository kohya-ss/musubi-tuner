"""Tests for SingleStreamDiT's gradient-checkpointing activation CPU offload.

Krea 2 accepted --gradient_checkpointing_cpu_offload for interface parity with every other
architecture in this codebase but silently dropped it, causing linear per-step GPU memory growth
during long gradient-enabled rollouts. These tests cover the interface toggle, that plain
checkpointing still works once the flag exists, and that the CPU-offload round-trip doesn't
strand output on the wrong device or corrupt gradients.
"""

import pytest
import torch

from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2 import krea2_sampling


def _tiny_model() -> SingleStreamDiT:
    cfg = SingleMMDiTConfig(
        features=32,
        tdim=32,
        txtdim=32,
        heads=2,
        multiplier=1,
        layers=2,
        patch=2,
        channels=4,
        bias=False,
        theta=1e3,
        kvheads=None,
        txtlayers=1,
        txtheads=2,
        txtkvheads=2,
    )
    return SingleStreamDiT(cfg, attn_mode="torch")


def _forward_inputs(model: SingleStreamDiT, device: torch.device, batch: int = 1):
    """Builds a valid (img, context, t, pos, mask) call for model.forward on the given device,
    using krea2_sampling.prepare the same way krea2_train_network.py's own sampling path does."""
    torch.manual_seed(0)
    patch = model.config.patch
    lat_h, lat_w = 8, 8  # multiple of patch=2, small for a fast test
    noise = torch.randn(batch, model.config.channels, lat_h, lat_w, device=device)
    txt_len = 3
    # Context is 4D: (batch, txt_len, num_txt_layers, txtdim). The text encoder output stacks
    # hidden states from num_txt_layers, which the text fusion transformer then projects to 1.
    num_txt_layers = 1  # matches model.config.txtlayers
    context = torch.randn(batch, txt_len, num_txt_layers, model.config.txtdim, device=device)
    txtmask = torch.ones(batch, txt_len, device=device, dtype=torch.bool)
    img_tokens, pos, mask = krea2_sampling.prepare(noise, txt_len, patch, txtmask)
    t = torch.rand(batch, device=device)
    return img_tokens, context, t, pos, mask


def test_gradient_checkpointing_interface_toggles_both_flags():
    model = _tiny_model()

    model.enable_gradient_checkpointing(cpu_offload=True)
    assert model.gradient_checkpointing is True
    assert model.activation_cpu_offloading is True

    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False
    assert model.activation_cpu_offloading is False


def test_checkpointed_forward_and_backward_still_work_without_offload():
    """Regression guard: plain checkpointing (cpu_offload left at its False default) must be
    unaffected by adding the activation_cpu_offloading attribute."""
    model = _tiny_model()
    model.train()
    model.enable_gradient_checkpointing()  # cpu_offload defaults to False
    assert model.activation_cpu_offloading is False

    img, context, t, pos, mask = _forward_inputs(model, device=torch.device("cpu"))
    output = model(img=img, context=context, t=t, pos=pos, mask=mask)
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    grad = model.blocks[0].attn.wq.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="activation CPU offloading requires CUDA")
def test_activation_cpu_offloading_round_trip_preserves_device_and_gradients():
    device = torch.device("cuda")
    model = _tiny_model().to(device)
    model.train()
    model.enable_gradient_checkpointing(cpu_offload=True)

    img, context, t, pos, mask = _forward_inputs(model, device=device)
    output = model(img=img, context=context, t=t, pos=pos, mask=mask)

    assert output.device.type == "cuda"
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    grad = model.blocks[0].attn.wq.weight.grad
    assert grad is not None
    assert grad.device.type == "cuda"
    assert torch.isfinite(grad).all()
