"""A trained artifact must not silently forget or replace its runtime policy."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from musubi_tuner.dlssnr import infer
from musubi_tuner.dlssnr.artifacts import save_canonical
from musubi_tuner.networks import lora_dlssnr
from test_dlssnr_artifacts import make_canonical
from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject
from test_dlssnr_training import SmallNR, make_args


def _policy(**changes):
    return {
        "schema": "dlssnr_runtime_v1",
        "numerics_profile": "train_surrogate",
        "mixed_precision": "no",
        "gradient_checkpointing": False,
        "attention_backend": "native",
        "attention_scope": "all",
        "fp8_base": False,
        "fp8_scaled": False,
        **changes,
    }


def test_legacy_model_loading_retains_baseline_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "NRModel", SmallNR)
    source = make_canonical(tmp_path / "base", SmallNR())
    model = infer.load_model(source, "cpu")
    assert model.runtime_policy == _policy()
    assert not model.runtime_provenance["overrides"]


def test_saved_runtime_is_inherited_even_without_optional_training_sidecar(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "NRModel", SmallNR)
    policy = _policy(numerics_profile="train_experimental", attention_backend="sdpa")
    folder = tmp_path / "artifact"
    save_canonical(SmallNR(), folder, metadata={"runtime_policy": policy})
    config = json.loads((folder / "model_config.json").read_text())
    assert config["runtime_policy"] == policy
    with safe_open(folder / "model.safetensors", framework="pt") as handle:
        assert json.loads(handle.metadata()["dlssnr_runtime_policy"]) == policy
    (folder / "training_metadata.json").unlink()
    model = infer.load_model(folder, "cpu")
    assert model.runtime_policy == policy


@pytest.mark.parametrize("change", ["missing", "changed", "extra_field", "non_boolean"])
def test_new_artifact_cannot_silently_drop_or_corrupt_its_runtime(tmp_path, monkeypatch, change):
    monkeypatch.setattr(infer, "NRModel", SmallNR)
    folder = tmp_path / "artifact"
    policy = _policy(numerics_profile="train_experimental", attention_backend="sdpa")
    save_canonical(SmallNR(), folder, metadata={"runtime_policy": policy})
    config = json.loads((folder / "model_config.json").read_text())
    if change == "missing":
        config.pop("runtime_policy", None)
    else:
        config["runtime_policy"] = dict(policy)
        if change == "changed":
            config["runtime_policy"]["attention_backend"] = "native"
        elif change == "extra_field":
            config["runtime_policy"]["silently_ignored"] = True
        else:
            config["runtime_policy"]["fp8_base"] = "false"
    (folder / "model_config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime|policy"):
        infer.load_model(folder, "cpu")


def test_explicit_inference_override_is_recorded_in_generated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(infer, "NRModel", SmallNR)
    args = make_args(tmp_path / "data")
    policy = _policy(numerics_profile="train_experimental", attention_backend="sdpa")
    folder = tmp_path / "artifact"
    save_canonical(SmallNR(), folder, metadata={"runtime_policy": policy})
    overrides = {"numerics_profile": "train_surrogate", "attention_backend": "native"}
    model = infer.load_model(folder, "cpu", runtime_overrides=overrides)
    assert model.runtime_policy == _policy()
    output = tmp_path / "images"
    infer.generate_stills(model, args.dataset_config.parent / "data.jsonl", 48, 48, output, 4)
    metadata = json.loads((output / "inference_metadata.json").read_text())
    assert metadata["runtime_policy"] == _policy()
    assert metadata["runtime_provenance"]["overrides"] == overrides
    assert metadata["runtime_provenance"]["saved_policy"] == policy
    assert metadata["source_identity"]["model_sha256"]


def test_canonical_export_materializes_fp8_once_and_inference_does_not_requantize(tmp_path, monkeypatch):
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

    monkeypatch.setattr(infer, "NRModel", TinyFP8NR)
    model = TinyFP8NR().requires_grad_(False)
    model.runtime_policy = _policy(numerics_profile="train_experimental", fp8_base=True, fp8_scaled=True)
    quantize_frozen_base(model, scaled=True)
    folder = tmp_path / "artifact"
    save_canonical(model, folder)
    loaded = infer.load_model(folder, "cpu")
    assert loaded.runtime_policy["fp8_base"] is False
    assert loaded.blocks["31"].ffn.fc1.weight.dtype == torch.float32
    source = torch.rand(1, 16, 5, 7)
    torch.testing.assert_close(loaded(source, None), model(source, None), rtol=0, atol=0)


@pytest.mark.parametrize("missing", ["runtime_policy", "base_quantization"])
def test_new_adapter_merge_rejects_missing_required_runtime_metadata(tmp_path, monkeypatch, missing):
    from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

    monkeypatch.setattr(lora_dlssnr, "NRModel", TinyFP8NR)
    model = TinyFP8NR()
    base = make_canonical(tmp_path / "base", model)
    identity = lora_dlssnr.base_target_sha256(model, [])
    network = tiny_fp8_inject(model, {})
    network.runtime_policy = _policy(numerics_profile="train_experimental", fp8_base=True, fp8_scaled=True)
    network.base_quantization = quantize_frozen_base(model, scaled=True)
    path = tmp_path / "adapter.safetensors"
    lora_dlssnr.save_adapter(network, path, identity)
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    metadata.pop(missing)
    save_file({name: value.clone() for name, value in load_file(path).items()}, str(path), metadata=metadata)
    with pytest.raises(ValueError, match="runtime|quantization"):
        lora_dlssnr.merge_to_directory(base, path, tmp_path / "merged")


def test_legacy_v1_adapter_still_merges_with_fp32_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_dlssnr, "NRModel", TinyFP8NR)
    monkeypatch.setattr(infer, "NRModel", TinyFP8NR)
    model = TinyFP8NR()
    base = make_canonical(tmp_path / "base", model)
    identity = lora_dlssnr.base_target_sha256(model, [])
    network = tiny_fp8_inject(model, {})
    with torch.no_grad():
        network.adapters[0].lora_up.fill_(0.015)
    path = tmp_path / "adapter.safetensors"
    lora_dlssnr.save_adapter(network, path, identity)
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    metadata["schema"] = "dlssnr_lora_v1"
    for name in ("runtime_policy", "base_quantization"):
        metadata.pop(name, None)
    save_file({name: value.clone() for name, value in load_file(path).items()}, str(path), metadata=metadata)
    lora_dlssnr.merge_to_directory(base, path, tmp_path / "merged")
    merged = infer.load_model(tmp_path / "merged", "cpu")
    assert merged.runtime_policy == _policy()
    source = torch.rand(1, 16, 5, 7)
    torch.testing.assert_close(merged(source, None), model(source, None), rtol=0, atol=0)


@pytest.mark.parametrize("changed", ["runtime.py", "attention.py", "fp8.py"])
def test_forward_evidence_identity_changes_with_runtime_implementation(tmp_path, monkeypatch, changed):
    from musubi_tuner.dlssnr import identity

    files = set(identity.implementation_identity()) | {"runtime.py", "attention.py", "fp8.py"}
    for name in files:
        (tmp_path / name).write_text("original implementation", encoding="utf-8")
    monkeypatch.setattr(identity, "__file__", str(tmp_path / "identity.py"))
    before = identity.json_sha256(identity.implementation_identity())
    (tmp_path / changed).write_text("different runtime implementation", encoding="utf-8")
    assert identity.json_sha256(identity.implementation_identity()) != before
