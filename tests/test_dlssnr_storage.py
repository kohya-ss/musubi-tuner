import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.dlssnr.artifacts import inspect_canonical, save_canonical
from musubi_tuner.dlssnr.model import ChannelLinear, NRModel
from musubi_tuner.dlssnr.profiles import CC_PROFILE_ID, PROFILE_ID


def test_storage_decoder_is_bound_to_implementation_identity():
    from musubi_tuner.dlssnr.identity import implementation_identity

    assert "storage.py" in implementation_identity()


class StorageModel(torch.nn.Module):
    """Small real projections with native names, covering all three storage dtypes."""

    def __init__(self, profile=PROFILE_ID):
        super().__init__()
        self.profile = profile
        first, window, last = torch.nn.Module(), torch.nn.Module(), torch.nn.Module()
        first.input_adapter = ChannelLinear(32, 16)
        window.attn = torch.nn.Module()
        window.attn.proj = ChannelLinear(32, 32)
        window.attn.temperature = torch.nn.Parameter(torch.tensor([1.0001]))
        window.attn.prior = torch.nn.Parameter(torch.full((1, 64, 64), -0.07))
        last.head = torch.nn.Module()
        last.head.rgb = ChannelLinear(3, 32)
        last.head.logit = ChannelLinear(1, 32)
        last.blend_scale = torch.nn.Parameter(torch.tensor([0.7671]))
        self.head_block = "30" if profile == CC_PROFILE_ID else "70"
        self.blocks = torch.nn.ModuleDict({"0": first, "1": window, self.head_block: last})
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("weight"):
                    parameter.fill_(0.07)
            first.input_adapter.weight[:, 15] = -0.0

    load_canonical = NRModel.load_canonical


@pytest.mark.parametrize("profile", [PROFILE_ID, CC_PROFILE_ID])
def test_default_model_save_uses_native_mixed_storage_without_changing_masters(tmp_path, profile):
    model = StorageModel(profile)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    save_canonical(model, tmp_path)
    stored = load_file(tmp_path / "model.safetensors")
    assert stored["blocks.1.attn.proj.weight"].dtype == torch.float8_e4m3fn
    assert stored["blocks.0.input_adapter.weight"].dtype == torch.float16
    assert stored["blocks.1.attn.prior"].dtype == torch.float16
    assert stored["blocks.1.attn.temperature"].dtype == torch.float32
    assert stored[f"blocks.{model.head_block}.head.rgb.weight"].dtype == torch.float16
    config = json.loads((tmp_path / "model_config.json").read_text())
    assert config["weight_storage"] == {"schema": "dlssnr_weight_storage_v1", "profile": profile, "precision": "native"}
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt") as handle:
        assert json.loads(handle.metadata()["dlssnr_weight_storage"]) == config["weight_storage"]
    report = json.loads((tmp_path / "training_metadata.json").read_text())
    assert report["storage_quantization"]["rounded_values"] > 0
    assert report["storage_quantization"]["stored_bytes"] < sum(value.numel() * 4 for value in before.values())
    for name, value in model.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    loaded = StorageModel(profile)
    loaded.load_canonical(str(tmp_path / "model.safetensors"))
    for name, value in loaded.state_dict().items():
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, stored[name].float(), rtol=0, atol=0)
    assert torch.signbit(loaded.blocks["0"].input_adapter.weight[:, 15]).all()


def test_explicit_fp32_save_and_legacy_fp32_files_remain_lossless(tmp_path):
    model = StorageModel()
    save_canonical(model, tmp_path / "fp32", save_precision="fp32")
    stored = load_file(tmp_path / "fp32/model.safetensors")
    assert all(value.dtype == torch.float32 for value in stored.values())
    torch.testing.assert_close(stored, model.state_dict(), rtol=0, atol=0)
    legacy = tmp_path / "legacy.safetensors"
    save_file(model.state_dict(), str(legacy))
    restored = StorageModel()
    restored.load_canonical(str(legacy))
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("field,value", [("matrix", 449.0), ("head", 65505.0), ("temperature", float("nan"))])
def test_native_save_rejects_out_of_range_and_nonfinite_values_before_publication(tmp_path, field, value):
    model = StorageModel()
    parameters = {
        "matrix": model.blocks["1"].attn.proj.weight,
        "head": model.blocks["70"].head.rgb.weight,
        "temperature": model.blocks["1"].attn.temperature,
    }
    with torch.no_grad():
        parameters[field].flatten()[0] = value
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite|range"):
        save_canonical(model, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["config_missing", "config_changed", "header_missing", "wrong_dtype"])
def test_storage_policy_and_tensor_dtypes_cannot_silently_disagree(tmp_path, change):
    model = StorageModel()
    save_canonical(model, tmp_path)
    path = tmp_path / "model_config.json"
    config = json.loads(path.read_text())
    if change.startswith("config"):
        if change == "config_missing":
            config.pop("weight_storage")
        else:
            config["weight_storage"]["precision"] = "fp32"
        path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match="storage"):
            inspect_canonical(tmp_path, development_smoke=True)
    else:
        path = tmp_path / "model.safetensors"
        tensors = load_file(path)
        with safe_open(str(path), framework="pt") as handle:
            metadata = handle.metadata()
        if change == "header_missing":
            metadata.pop("dlssnr_weight_storage")
        else:
            tensors["blocks.1.attn.temperature"] = tensors["blocks.1.attn.temperature"].half()
        save_file(tensors, str(path), metadata=metadata)
        with pytest.raises(ValueError, match="storage|dtype|FP32"):
            model.load_canonical(str(path))


def test_training_parser_defaults_to_native_storage(tmp_path):
    from test_dlssnr_config import make_args
    from musubi_tuner.dlssnr.config import build_train_config

    args = make_args(tmp_path)
    assert args.save_precision == "native"
    assert build_train_config(args, lora=False)["output"]["save_precision"] == "native"


def test_native_training_products_do_not_quantize_exact_resume_snapshots(tmp_path, monkeypatch):
    from accelerate.state import AcceleratorState
    from test_dlssnr_training import SmallNR, make_args
    from musubi_tuner.training import dlssnr_trainer as trainer

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setattr(trainer, "NRModel", SmallNR)
    args = make_args(tmp_path)
    args.save_precision = "native"
    args.ema_decay = 0.99
    try:
        trainer.train_from_args(args)
        folder = tmp_path / "output/dlssnr"
        snapshots = {}
        for variant in ("", "ema"):
            snapshot = load_file(folder / "state-step000002" / variant / "model.safetensors")
            snapshots[variant] = {name: value.clone() for name, value in snapshot.items()}
            final = load_file(folder / "final" / variant / "model.safetensors")
            assert all(value.dtype == torch.float32 for value in snapshot.values())
            assert all(value.dtype == torch.float16 for value in final.values())
            for name, value in final.items():
                torch.testing.assert_close(value, snapshot[name].half(), rtol=0, atol=0)
        args.resume = folder / "state-step000001"
        trainer.train_from_args(args)
        for variant, expected in snapshots.items():
            resumed = load_file(folder / "state-step000002" / variant / "model.safetensors")
            torch.testing.assert_close(resumed, expected, rtol=0, atol=0)
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
