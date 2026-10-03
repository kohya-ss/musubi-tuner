"""FP8 storage must retain an identifiable effective base and merge semantics."""

import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.model import ChannelLinear
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA, base_target_sha256, merge_adapter, save_adapter
from test_dlssnr_config import make_args
from test_dlssnr_training import SmallNR


class TinyFP8NR(SmallNR):
    def __init__(self):
        super().__init__()
        block = torch.nn.Module()
        block.ffn = torch.nn.Module()
        block.ffn.fc1 = ChannelLinear(64, 32)
        block.ffn.fc2 = ChannelLinear(32, 64)
        torch.nn.init.uniform_(block.ffn.fc1.weight, -0.133, 0.147)
        torch.nn.init.uniform_(block.ffn.fc2.weight, -0.063, 0.081)
        self.blocks["31"] = block

    def forward(self, features, geometry):
        hidden = self.blocks["0"].input_adapter(features)
        hidden = self.blocks["31"].ffn.fc2(torch.nn.functional.silu(self.blocks["31"].ffn.fc1(hidden)))
        return torch.cat((self.blocks["70"].head.rgb(hidden), self.blocks["70"].head.logit(hidden)), dim=1)


def tiny_fp8_inject(model, table):
    network = DLSSNRLoRA()
    target = "blocks.31.ffn.fc1.weight"
    network.add(target, model.blocks["31"].ffn.fc1, 2, 2, table.get("dropout", 0.0))
    model.requires_grad_(False)
    network.profile = "manual"
    network.report = {"profile": "manual", "targets": [{"name": target, "rank": 2, "alpha": 2, "out": 64, "in": 32}]}
    return network


@pytest.mark.parametrize("scaled", [False, True])
def test_fp8_policy_is_lora_only_and_explicit(tmp_path, scaled):
    options = ["--numerics_profile", "train_experimental", "--fp8_base"] + (["--fp8_scaled"] if scaled else [])
    with pytest.raises(ValueError, match="LoRA"):
        build_train_config(make_args(tmp_path, options))
    config = build_train_config(make_args(tmp_path, options, lora=True), lora=True)
    assert config["precision"]["fp8_base"] is True
    assert config["precision"]["fp8_scaled"] is scaled


def test_scaled_fp8_requires_base_flag(tmp_path):
    with pytest.raises(ValueError, match="fp8_base"):
        build_train_config(make_args(tmp_path, ["--numerics_profile", "train_experimental", "--fp8_scaled"], lora=True), lora=True)


@pytest.mark.parametrize("scaled", [False, True])
def test_fp8_reduces_resident_weights_and_preserves_exclusions_and_frozen_identity(scaled):
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

    model = TinyFP8NR().requires_grad_(False)
    original = base_target_sha256(model, [])
    protected = {
        name: parameter.clone() for name, parameter in model.named_parameters() if ".head." in name or "input_adapter" in name
    }
    report = quantize_frozen_base(model, scaled=scaled)
    assert report["source_base_sha256"] == original
    assert report["effective_base_sha256"] == base_target_sha256(model, [])
    assert report["effective_base_sha256"] != original
    for module in (model.blocks["31"].ffn.fc1, model.blocks["31"].ffn.fc2):
        assert module.weight.dtype == torch.float8_e4m3fn
        assert module.weight.element_size() == 1
        assert not module.weight.requires_grad
        assert module.materialized_weight().dtype == torch.float32
        assert torch.isfinite(module.materialized_weight()).all()
    if scaled:
        assert tuple(model.blocks["31"].ffn.fc1.scale_weight.shape) == (64, 1)
        assert tuple(model.blocks["31"].ffn.fc2.scale_weight.shape) == (32, 1, 1)
    for name, value in protected.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], value, rtol=0, atol=0)
    assert torch.count_nonzero(model.blocks["0"].input_adapter.weight[:, 15]) == 0


@pytest.mark.parametrize("scaled", [False, True])
def test_fp8_effective_weight_lora_backward_and_merge_match(scaled):
    from musubi_tuner.dlssnr.fp8 import materialize_state_dict, quantize_frozen_base

    torch.manual_seed(81)
    model = TinyFP8NR()
    network = tiny_fp8_inject(model, {})
    network.base_quantization = quantize_frozen_base(model, scaled=scaled)
    base = materialize_state_dict(model)
    source = torch.rand(1, 16, 5, 7)
    before = base_target_sha256(model, [])
    optimizer = torch.optim.SGD(network.parameters(), lr=0.03)
    model(source, None).square().sum().backward()
    assert torch.count_nonzero(network.adapters[0].lora_up.grad)
    optimizer.step()
    assert base_target_sha256(model, []) == before
    merged = TinyFP8NR()
    merged.load_state_dict(merge_adapter(base, network), strict=True)
    torch.testing.assert_close(merged(source, None), model(source, None), rtol=0, atol=0)


def test_fp8_merge_rejects_the_original_unquantized_base():
    from musubi_tuner.dlssnr.fp8 import materialize_state_dict, quantize_frozen_base

    model = TinyFP8NR()
    network = tiny_fp8_inject(model, {})
    original = materialize_state_dict(model)
    network.base_quantization = quantize_frozen_base(model, scaled=True)
    with pytest.raises(ValueError, match="effective|quantiz"):
        merge_adapter(original, network)


@pytest.mark.parametrize("value,scaled", [(float("nan"), True), (float("inf"), True), (1e6, False)])
def test_invalid_fp8_weights_fail_before_mutating_the_model(value, scaled):
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

    model = TinyFP8NR().requires_grad_(False)
    model.blocks["31"].ffn.fc2.weight.fill_(value)
    with pytest.raises(ValueError, match="finite|overflow"):
        quantize_frozen_base(model, scaled=scaled)
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert model.blocks["31"].ffn.fc1.weight.dtype == model.blocks["31"].ffn.fc2.weight.dtype == torch.float32


def test_fp8_rejects_trainable_base_matrices():
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

    with pytest.raises(ValueError, match="frozen|trainable"):
        quantize_frozen_base(TinyFP8NR(), scaled=True)


@pytest.mark.parametrize("scaled", [False, True])
def test_saved_fp8_adapter_merge_materializes_once_and_records_runtime(tmp_path, monkeypatch, scaled):
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base
    from musubi_tuner.networks import lora_dlssnr
    from test_dlssnr_artifacts import make_canonical

    torch.manual_seed(31)
    model = TinyFP8NR()
    base_dir = make_canonical(tmp_path / "base", model)
    identity = base_target_sha256(model, [])
    network = tiny_fp8_inject(model, {})
    network.base_quantization = quantize_frozen_base(model, scaled=scaled)
    with torch.no_grad():
        network.adapters[0].lora_up.fill_(0.015)
    network.runtime_policy = {
        "schema": "dlssnr_runtime_v1",
        "numerics_profile": "train_experimental",
        "mixed_precision": "no",
        "gradient_checkpointing": True,
        "attention_backend": "native",
        "attention_scope": "all",
        "fp8_base": True,
        "fp8_scaled": scaled,
    }
    adapter = tmp_path / "adapter.safetensors"
    save_adapter(network, adapter, identity)
    monkeypatch.setattr(lora_dlssnr, "NRModel", TinyFP8NR)
    output = tmp_path / "merged"
    lora_dlssnr.merge_to_directory(base_dir, adapter, output)
    merged = TinyFP8NR()
    merged.load_state_dict(load_file(output / "model.safetensors"), strict=True)
    source = torch.rand(1, 16, 6, 9)
    torch.testing.assert_close(merged(source, None), model(source, None), rtol=0, atol=0)
    metadata = json.loads((output / "training_metadata.json").read_text())
    assert metadata["runtime_policy"]["fp8_base"] is False
    assert metadata["base_quantization"]["effective_base_sha256"] == network.base_quantization["effective_base_sha256"]
    assert metadata["fp8_materialized"] is True


@pytest.mark.parametrize("scaled", [False, True])
def test_fp8_lora_runner_updates_and_exactly_resumes(tmp_path, monkeypatch, scaled):
    from accelerate.state import AcceleratorState
    from musubi_tuner.training import dlssnr_trainer as trainer
    from musubi_tuner.networks import lora_dlssnr
    from test_dlssnr_training import make_args as train_args

    monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
    monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        args = train_args(tmp_path, lora=True)
        args.numerics_profile, args.fp8_base, args.fp8_scaled = "train_experimental", True, scaled
        trainer.train_lora_from_args(args)
        folder = args.output_dir / args.output_name
        expected = {name: value.clone() for name, value in load_file(folder / "final/adapter.safetensors").items()}
        assert any(torch.count_nonzero(value) for name, value in expected.items() if name.endswith("lora_up"))
        metadata = json.loads((folder / "run_config.json").read_text())
        assert metadata["base_quantization"]["scaled"] is scaled
        assert metadata["identity"]["base_quantization"] == metadata["base_quantization"]
        args.resume = folder / "state-step000001"
        trainer.train_lora_from_args(args)
        torch.testing.assert_close(load_file(folder / "final/adapter.safetensors"), expected, rtol=0, atol=0)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
