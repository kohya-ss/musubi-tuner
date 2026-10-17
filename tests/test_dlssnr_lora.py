from pathlib import Path

import torch
import pytest

from musubi_tuner.dlssnr.model import ChannelLinear, GlobalAttn, NRModel, WindowAttn
from musubi_tuner.dlssnr.numerics import apply_linear
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA, VIT_ONLY_ELEMENTS, inject, load_adapter, merge_adapter, save_adapter
from musubi_tuner.training.dlssnr_parser import setup_parser
from musubi_tuner.training.dlssnr_trainer import single_frame_update


@pytest.mark.parametrize("kind", ["window", "global"])
def test_attention_adapters_receive_gradients_and_match_merged_forward(kind):
    torch.manual_seed(11)
    attention = WindowAttn(32, 1) if kind == "window" else GlobalAttn(32)
    for name, parameter in attention.named_parameters():
        if parameter.ndim == 1:
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.05)
        parameter.requires_grad_(False)
    network = DLSSNRLoRA()
    network.add("qkv.weight", attention.qkv, 4, 4, 0.0)
    network.add("proj.weight", attention.proj, 4, 4, 0.0)
    source = torch.randn(1, 32, 8, 16, requires_grad=True)
    plain = attention(source).detach()
    with torch.no_grad():
        for adapter in network.adapters:
            adapter.lora_up.normal_(std=0.1)
    hooked = attention(source)
    assert not torch.equal(hooked.detach(), plain)
    hooked.square().mean().backward()
    for adapter in network.adapters:
        for parameter in adapter.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    fresh = WindowAttn(32, 1) if kind == "window" else GlobalAttn(32)
    fresh.load_state_dict(merge_adapter(attention.state_dict(), network))
    torch.testing.assert_close(fresh(source), hooked, rtol=2e-5, atol=2e-5)


def test_vit_only_supports_nondefault_rank():
    network = inject(NRModel(), {"profile": "vit_only", "rank": 8, "alpha": 4, "dropout": 0.0})
    assert len(network.target_names) == 32
    assert network.elements == 1_048_576


def test_lora_math_merges_and_updates_b_before_a(tmp_path: Path):
    lone = ChannelLinear(8, 8)
    torch.nn.init.normal_(lone.weight)
    direct = DLSSNRLoRA()
    direct.profile = "manual"
    direct.add("linear.weight", lone, 4, 4, 0.0)
    adapter = direct.adapters[0]
    assert torch.count_nonzero(adapter.lora_up) == 0
    assert not lone.weight.requires_grad
    source = torch.randn(2, 8)
    image = torch.randn(2, 8, 3, 3)
    optimizer = torch.optim.AdamW(direct.parameters(), lr=1e-1, weight_decay=0.0)
    before_base = lone.weight.detach().clone()
    before_a = adapter.lora_down.detach().clone()
    lone(source).sum().backward()
    assert adapter.lora_down.grad is not None and torch.count_nonzero(adapter.lora_down.grad) == 0
    assert torch.count_nonzero(adapter.lora_up.grad) > 0
    optimizer.step()
    assert torch.equal(lone.weight, before_base)
    assert torch.equal(adapter.lora_down, before_a)
    assert not torch.equal(adapter.lora_up, torch.zeros_like(adapter.lora_up))
    optimizer.zero_grad(set_to_none=True)
    lone(source).sum().backward()
    optimizer.step()
    assert not torch.equal(adapter.lora_down, before_a)
    with torch.no_grad():
        adapter.lora_up.normal_()
    merged = merge_adapter({"linear.weight": lone.weight.detach().clone()}, direct)["linear.weight"]
    tokens = torch.randn(4, 5, 8)
    hooked = lone(tokens)
    # The hook still adds LoRA, so compare a fresh linear using the merged weight.
    reference = apply_linear(merged, tokens)
    plain = apply_linear(lone.weight.detach(), tokens) + adapter(tokens)
    assert torch.allclose(plain, reference, atol=1e-5)
    assert torch.allclose(hooked, reference, atol=1e-5)
    assert torch.allclose(lone(image), apply_linear(merged, image), atol=1e-5)
    save_adapter(direct, tmp_path / "adapter.safetensors", "abc")
    other = type(direct)()
    other.profile = "manual"
    fresh = ChannelLinear(8, 8)
    other.add("linear.weight", fresh, 4, 4, 0.0)
    load_adapter(other, tmp_path / "adapter.safetensors")
    assert torch.equal(other.adapters[0].lora_up, adapter.lora_up)
    bad = tmp_path / "bad.safetensors"
    from safetensors.torch import save_file

    save_file({"adapters.0.lora_down": torch.zeros(4, 8)}, bad, metadata={"schema": "other"})
    try:
        load_adapter(other, bad)
    except ValueError as exc:
        assert "schema" in str(exc)
    else:
        raise AssertionError("unknown LoRA schema was accepted")


def test_vit_only_budget_and_real_step_keeps_the_base_fixed():
    torch.manual_seed(0)
    model = NRModel()
    # The default tiny random transitions underflow through six E4 levels.
    # Give this synthetic fixture a nonzero bottleneck; real weights have a
    # separate end-to-end CUDA test requiring gradients on all 32 adapters.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and any(token in name for token in ("input_adapter", ".down.", ".to_vit.")):
                parameter.normal_(std=parameter.shape[1] ** -0.5)
    network = inject(model, {"profile": "vit_only", "rank": 16, "alpha": 16, "dropout": 0.0, "qkv_mode": "fused_head_major"})
    assert network.elements == VIT_ONLY_ELEMENTS
    assert len(network.target_names) == 32
    assert all(name.startswith("blocks.3") and int(name.split(".")[1]) <= 38 for name in network.target_names)
    assert not any("input_adapter" in name or ".head." in name or name.startswith("blocks.39") for name in network.target_names)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    named = dict(model.named_parameters())
    before = {name: named[name].detach().clone() for name in network.target_names}
    before_down = [adapter.lora_down.detach().clone() for adapter in network.adapters]
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-2, weight_decay=0.0)
    source = torch.full((1, 3, 48, 48), 0.4)
    target = torch.full((1, 3, 48, 48), 0.2)
    controls = torch.zeros(1, 5, 48, 48)
    metrics = single_frame_update(model, optimizer, source, target, controls, 9, {"pre": 1, "out": 1, "edge": 0.05})
    assert metrics["loss"] == metrics["loss"]
    named = dict(model.named_parameters())
    assert all(torch.equal(named[name], before[name]) for name in network.target_names)
    assert all(torch.equal(adapter.lora_down, previous) for adapter, previous in zip(network.adapters, before_down))
    assert any(torch.count_nonzero(adapter.lora_up) > 0 for adapter in network.adapters)


def test_multiscale_ranks_follow_width_and_skip_transitions():
    model = NRModel()
    network = inject(model, {"profile": "multiscale", "dropout": 0.0})
    by_name = {adapter.target: adapter for adapter in network.adapters}
    assert by_name["blocks.1.attn.qkv.weight"].rank == 2
    assert by_name["blocks.31.ffn.fc1.weight"].rank == 16
    assert by_name["blocks.23.ffn.contract.weight"].rank == 16
    assert "blocks.0.input_adapter.weight" not in by_name
    assert "blocks.4.down.weight" not in by_name
    assert "blocks.39.proj.weight" not in by_name
    assert "blocks.70.head.rgb.weight" not in by_name
    try:
        inject(model, {"profile": "multiscale", "rank": 4, "alpha": 4})
    except ValueError as exc:
        assert "single rank" in str(exc)
    else:
        raise AssertionError("multiscale accepted a single rank")
    small = ChannelLinear(8, 8)
    try:
        broken = DLSSNRLoRA()
        broken.add("too.big.weight", small, 16, 16, 0.0)
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("oversized rank was accepted")


def test_lora_arguments_are_separate_from_full_training(tmp_path: Path):
    common = [
        "--dataset_config",
        str(tmp_path / "dataset.toml"),
        "--output_dir",
        str(tmp_path / "out"),
        "--output_name",
        "run",
    ]
    with pytest.raises(SystemExit):
        setup_parser().parse_args([*common, "--network_dim", "16"])
    with pytest.raises(SystemExit):
        setup_parser(lora=True).parse_args([*common, "--prior_lr_multiplier", "0.1"])
