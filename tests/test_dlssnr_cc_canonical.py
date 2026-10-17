import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.dlssnr import artifacts


CC_PROFILE = "dlss_nr_310_9_16_cc"


def cc_provenance(folder):
    from test_dlssnr_artifacts import make_canonical

    make_canonical(folder)
    for name in ("model_config.json", "conversion_report.json"):
        path = folder / name
        document = json.loads(path.read_text())
        document["profile"] = CC_PROFILE
        path.write_text(json.dumps(document))
    return folder


def test_cc_canonical_provenance_accepts_cc_but_rejects_ht_conversion(tmp_path):
    folder = cc_provenance(tmp_path / "cc")
    assert artifacts.inspect_canonical(folder)["model_sha256"]
    report = folder / "conversion_report.json"
    document = json.loads(report.read_text())
    document["profile"] = "dlss_nr_310_8_0"
    report.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="round-trip"):
        artifacts.inspect_canonical(folder)


def test_cc_canonical_export_rejects_selecting_ht_before_publishing(tmp_path):
    from musubi_tuner.dlssnr.dll_models import pack_selected_model

    folder = cc_provenance(tmp_path / "cc")
    output = tmp_path / "wrong.dll"
    with pytest.raises(ValueError, match="resource HT.*CC"):
        pack_selected_model(tmp_path / "not-read.dll", folder, output, weights="HT")
    assert not output.exists()


def test_cc_save_rejects_ht_source_provenance(tmp_path):
    from test_dlssnr_artifacts import make_canonical

    source = make_canonical(tmp_path / "ht")
    model = torch.nn.Linear(2, 2, bias=False)
    model.profile = CC_PROFILE
    with pytest.raises(ValueError, match="profile"):
        artifacts.save_canonical(model, tmp_path / "cc", source_dir=source)
    assert not (tmp_path / "cc").exists()


def test_cc_save_preserves_profile_and_opaque_vit_bytes(tmp_path):
    folder = cc_provenance(tmp_path / "cc")
    model = torch.nn.Linear(2, 2, bias=False)
    model.profile = CC_PROFILE
    save_file(
        {"weight": model.weight.detach(), "blocks.11.opaque.layer3": torch.tensor([3, 128], dtype=torch.uint8)},
        str(folder / "model.safetensors"),
    )
    output = tmp_path / "saved"
    artifacts.save_canonical(model, output, source_dir=folder, save_precision="fp32")
    assert json.loads((output / "model_config.json").read_text())["profile"] == CC_PROFILE
    assert json.loads((output / "training_metadata.json").read_text())["profile"] == CC_PROFILE
    with safe_open(str(output / "model.safetensors"), framework="pt") as handle:
        assert handle.get_tensor("blocks.11.opaque.layer3").tolist() == [3, 128]
    artifacts.save_canonical(model, tmp_path / "scratch", save_precision="fp32")
    assert json.loads((tmp_path / "scratch/model_config.json").read_text())["profile"] == CC_PROFILE


@pytest.mark.parametrize("lora", [False, True])
def test_training_config_detects_cc_and_rejects_an_explicit_ht_profile(tmp_path, lora):
    from test_dlssnr_config import make_args
    from musubi_tuner.dlssnr.config import build_train_config

    folder = cc_provenance(tmp_path / "cc")
    args = make_args(tmp_path, ["--model_dir", str(folder)], lora=lora)
    assert build_train_config(args, lora=lora)["model"]["profile"] == CC_PROFILE
    args.profile = "dlss_nr_310_8_0"
    with pytest.raises(ValueError, match="profile"):
        build_train_config(args, lora=lora)
    args.profile = CC_PROFILE
    assert build_train_config(args, lora=lora)["model"]["profile"] == CC_PROFILE


def test_cc_adapter_metadata_is_profile_bound(tmp_path):
    from musubi_tuner.dlssnr.model import ChannelLinear
    from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA, load_adapter, save_adapter

    network = DLSSNRLoRA()
    network.profile = "manual"
    network.base_profile = CC_PROFILE
    network.add("blocks.11.ffn.fc1.weight", ChannelLinear(8, 8), 2, 2, 0.0)
    path = tmp_path / "adapter.safetensors"
    save_adapter(network, path, "base")
    with safe_open(str(path), framework="pt") as handle:
        assert handle.metadata()["base_profile"] == CC_PROFILE
    load_adapter(network, path)
    network.base_profile = "dlss_nr_310_8_0"
    with pytest.raises(ValueError, match="base profile"):
        load_adapter(network, path)


@pytest.mark.parametrize("lora", [False, True])
def test_training_initialization_constructs_the_requested_cc_model(tmp_path, lora):
    from test_dlssnr_config import make_args
    from musubi_tuner.dlssnr.config import build_train_config
    from musubi_tuner.dlssnr.runtime import runtime_policy
    from musubi_tuner.training.dlssnr_trainer import NRSupervisedTrainer

    config = build_train_config(make_args(tmp_path, ["--profile", CC_PROFILE], lora=lora), lora=lora)
    model, network, optimizer, identity = NRSupervisedTrainer(config, lora=lora)._initialize_model(runtime_policy(config))
    assert model.profile == CC_PROFILE
    assert len(model.blocks) == 31
    assert identity
    if lora:
        assert network.base_profile == CC_PROFILE
        assert len(network.target_names) == 32
    else:
        assert network is None
        assert not model.blocks["30"].blend_scale.requires_grad
    assert optimizer.param_groups


def test_real_cc_canonical_unpack_load_and_dll_roundtrip(tmp_path):
    source = os.environ.get("DLSSNR_CC_DLL_PATH")
    if not source:
        pytest.skip("set DLSSNR_CC_DLL_PATH to the audited 310.9.16 DLL")
    from musubi_tuner.dlssnr.dll_models import unpack_models, pack_selected_model
    from musubi_tuner.dlssnr.infer import load_model

    folder = tmp_path / "cc"
    report = unpack_models(source, folder, weights="CC", cc_format="canonical")
    assert report["profile"] == CC_PROFILE
    assert report["roundtrip"] == "byte_identical"
    assert report["resource_name"] == "WEIGHTS_CC_SMALL_FP8"
    assert report["storage_quantization"]["rounded_values"] == 0
    assert (folder / "model.safetensors").stat().st_size < 115_000_000
    with safe_open(str(folder / "model.safetensors"), framework="pt") as handle:
        assert handle.get_tensor("blocks.11.ffn.fc1.weight").dtype == torch.float8_e4m3fn
        assert handle.get_tensor("blocks.30.head.rgb.weight").dtype == torch.float16
        assert handle.get_tensor("blocks.11.attn.temperature").dtype == torch.float32
    assert artifacts.inspect_canonical(folder)["model_sha256"] == report["canonical_sha256"]
    model = load_model(folder, device="cpu")
    assert model.profile == CC_PROFILE
    assert len(model.blocks) == 31
    assert model.blocks["30"].blend_scale.item() == 0.76708984375
    output = tmp_path / "unchanged.dll"
    exported = pack_selected_model(source, folder, output)
    assert exported["resource_name"] == "WEIGHTS_CC_SMALL_FP8"
    assert exported["profile"] == CC_PROFILE
    assert exported["whole_dll_byte_identical"]
    assert output.read_bytes() == Path(source).read_bytes()
    assert exported["native_export_validated"] is False
