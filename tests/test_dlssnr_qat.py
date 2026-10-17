"""Native weight publication must match the DLL exporter, not activation rounding."""

import importlib

import numpy as np
import pytest
import torch

from musubi_tuner.dlssnr.model import ChannelLinear
from musubi_tuner.dlssnr.native import quantize_tensor
from musubi_tuner.dlssnr.packing import decode_e4m3
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA


def test_projection_publishes_weights_without_half_double_rounding():
    linear = ChannelLinear(1, 1)
    linear.native_weight_kind = "e4"
    with torch.no_grad():
        linear.weight.fill_(1.0626)
    output = linear(torch.ones(1, 1))
    torch.testing.assert_close(output, torch.tensor([[1.125]]), rtol=0, atol=0)
    output.sum().backward()
    assert linear.weight.item() == pytest.approx(1.0626)
    assert linear.weight.grad.item() == 1


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
)
def test_weight_ste_matches_exporter_at_every_e4_midpoint(device):
    quantization = importlib.import_module("musubi_tuner.dlssnr.weight_quantization")
    grid = decode_e4m3(np.arange(127, dtype=np.uint8))
    midpoints = (grid[:-1] + grid[1:]) / np.float32(2)
    positive = np.concatenate(
        (grid, midpoints, np.nextafter(midpoints, np.float32(0)), np.nextafter(midpoints, np.float32(np.inf)))
    )
    values = np.concatenate((positive, -positive))
    weights = torch.tensor(values, device=device, requires_grad=True)
    actual = quantization.native_weight_ste(weights, "e4")
    np.testing.assert_array_equal(actual.detach().cpu().numpy().view(np.uint32), quantize_tensor(values, "e4").view(np.uint32))
    actual.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["f16", "f16frag", "prior", "f32"])
def test_weight_ste_matches_half_and_float_storage(kind):
    quantization = importlib.import_module("musubi_tuner.dlssnr.weight_quantization")
    values = np.array([-0.0, 0.0, 1.00048828125, 1.00146484375, 65504, 2**-25], np.float32)
    weights = torch.tensor(values, requires_grad=True)
    actual = quantization.native_weight_ste(weights, kind)
    np.testing.assert_array_equal(actual.detach().numpy().view(np.uint32), quantize_tensor(values, kind).view(np.uint32))
    actual.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights), rtol=0, atol=0)


@pytest.mark.parametrize("kind,value", [("e4", 448.1), ("e4", float("nan")), ("f16", 65505), ("f32", float("inf"))])
def test_weight_ste_does_not_hide_unexportable_weights(kind, value):
    quantization = importlib.import_module("musubi_tuner.dlssnr.weight_quantization")
    with pytest.raises(ValueError, match="finite|range"):
        quantization.native_weight_ste(torch.tensor([value], dtype=torch.float32), kind)


def qat_policy(**changes):
    from musubi_tuner.dlssnr.runtime import default_runtime_policy

    return {**default_runtime_policy(), "native_weight_qat": True, **changes}


def test_runtime_assigns_native_storage_without_changing_masters():
    from musubi_tuner.dlssnr.runtime import configure_model_runtime
    from test_dlssnr_fp8 import TinyFP8NR

    model = TinyFP8NR()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    configure_model_runtime(model, qat_policy(), training=True)
    assert model.blocks["31"].ffn.fc1.native_weight_kind == "e4"
    assert model.blocks["0"].input_adapter.native_weight_kind == "f16frag"
    assert model.blocks["70"].head.rgb.native_weight_kind == "f16frag"
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)


def test_qat_cli_is_opt_in_and_requires_zero_lora_dropout(tmp_path):
    from musubi_tuner.dlssnr.config import build_train_config
    from musubi_tuner.dlssnr.runtime import runtime_policy
    from test_dlssnr_config import make_args

    normal = runtime_policy(build_train_config(make_args(tmp_path)))
    assert normal["schema"] == "dlssnr_runtime_v3"
    assert normal["native_weight_qat"] is False
    args = make_args(tmp_path, ["--native_weight_qat"])
    assert runtime_policy(build_train_config(args)) == qat_policy()
    args = make_args(tmp_path, ["--native_weight_qat", "--network_dropout", "0.2"], lora=True)
    with pytest.raises(ValueError, match="dropout"):
        build_train_config(args, lora=True)
    args = make_args(tmp_path, ["--no-native_weight_qat", "--network_dropout", "0.2"], lora=True)
    assert build_train_config(args, lora=True)["lora"]["dropout"] == 0.2


def test_qat_artifact_preserves_master_weights_and_restores_publication(tmp_path, monkeypatch):
    from musubi_tuner.dlssnr import infer
    from musubi_tuner.dlssnr.artifacts import save_canonical
    from musubi_tuner.dlssnr.runtime import configure_model_runtime
    from test_dlssnr_fp8 import TinyFP8NR

    model = TinyFP8NR()
    configure_model_runtime(model, qat_policy(), training=True)
    with torch.no_grad():
        model.blocks["31"].ffn.fc1.weight.fill_(1.0626)
    folder = tmp_path / "qat"
    save_canonical(model, folder, save_precision="fp32")
    monkeypatch.setattr(infer, "NRModel", TinyFP8NR)
    loaded = infer.load_model(folder, "cpu")
    torch.testing.assert_close(loaded.state_dict(), model.state_dict(), rtol=0, atol=0)
    linear = loaded.blocks["31"].ffn.fc1
    assert linear.published_weight(linear.weight)[0, 0].item() == 1.125
    floating = infer.load_model(folder, "cpu", runtime_overrides={"native_weight_qat": False})
    assert floating.blocks["31"].ffn.fc1.native_weight_kind is None


@pytest.mark.parametrize("field,value", [("native_weight_qat", "false"), ("extra", True)])
def test_qat_runtime_rejects_malformed_policy(field, value):
    from musubi_tuner.dlssnr.runtime import validate_runtime_policy

    with pytest.raises(ValueError, match="policy|boolean"):
        validate_runtime_policy(qat_policy(**{field: value}))


@pytest.mark.parametrize("scaled", [None, False, True])
def test_qat_lora_saved_merge_matches_attached_forward(tmp_path, monkeypatch, scaled):
    from musubi_tuner.dlssnr import infer
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base
    from musubi_tuner.dlssnr.runtime import configure_model_runtime
    from musubi_tuner.networks import lora_dlssnr
    from test_dlssnr_artifacts import make_canonical
    from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

    model = TinyFP8NR()
    base = make_canonical(tmp_path / "base", model)
    identity = lora_dlssnr.base_target_sha256(model, [])
    policy = qat_policy()
    if scaled is not None:
        policy.update(numerics_profile="train_experimental", fp8_base=True, fp8_scaled=scaled)
    configure_model_runtime(model, policy, training=True)
    network = tiny_fp8_inject(model, {})
    network.runtime_policy = policy
    if scaled is not None:
        network.base_quantization = quantize_frozen_base(model, scaled=scaled)
    with torch.no_grad():
        network.adapters[0].lora_up.fill_(0.015)
    adapter = tmp_path / "adapter.safetensors"
    lora_dlssnr.save_adapter(network, adapter, identity)
    monkeypatch.setattr(lora_dlssnr, "NRModel", TinyFP8NR)
    monkeypatch.setattr(infer, "NRModel", TinyFP8NR)
    lora_dlssnr.merge_to_directory(base, adapter, tmp_path / "merged")
    merged = infer.load_model(tmp_path / "merged", "cpu")
    source = torch.rand(1, 16, 5, 7)
    torch.testing.assert_close(merged(source, None), model(source, None), rtol=0, atol=0)


def test_native_evaluation_cli_requires_validation_data(tmp_path):
    from musubi_tuner.dlssnr.config import build_train_config
    from test_dlssnr_config import make_args

    with pytest.raises(ValueError, match="validation_manifest"):
        build_train_config(make_args(tmp_path, ["--eval_native"]))


def test_lora_quantizes_the_fused_weight_and_keeps_adapter_gradients():
    linear = ChannelLinear(1, 1)
    linear.native_weight_kind = "e4"
    with torch.no_grad():
        linear.weight.fill_(1.0)
    network = DLSSNRLoRA()
    network.add("blocks.31.ffn.fc1.weight", linear, 1, 1, 0)
    adapter = network.adapters[0]
    with torch.no_grad():
        adapter.lora_down.fill_(1)
        adapter.lora_up.fill_(0.07)
    actual = linear(torch.ones(1, 1))
    torch.testing.assert_close(actual, torch.tensor([[1.125]]), rtol=0, atol=0)
    actual.sum().backward()
    assert adapter.lora_up.grad.item() == 1
    assert adapter.lora_down.grad.item() == pytest.approx(0.07)
    assert linear.weight.grad is None


def test_qat_publishes_temporal_blend_as_half():
    from musubi_tuner.dlssnr.pipeline import forward_frame
    from test_dlssnr_training import SmallNR

    model = SmallNR()
    model.native_weight_qat = True
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.blocks["70"].blend_scale.fill_(0.5003)
    source = torch.full((1, 3, 48, 48), 0.5)
    output = forward_frame(model, source, torch.ones(1, 5, 48, 48), 42, history=source, motion=torch.zeros(1, 2, 48, 48))
    torch.testing.assert_close(output["blend_weight"], torch.full((1, 1, 48, 48), 0.250244140625), rtol=0, atol=0)
