import importlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.dlssnr.model import ChannelLinear
from musubi_tuner.networks.lora_dlssnr import base_target_sha256


def make_canonical(folder, model=None):
    folder.mkdir(parents=True, exist_ok=True)
    tensors = model.state_dict() if model is not None else {"weight": torch.ones(2, 2)}
    save_file(tensors, str(folder / "model.safetensors"))
    save_file({"block31.layer3.layer": torch.tensor([7, 128], dtype=torch.uint8)}, str(folder / "opaque_records.safetensors"))
    documents = {
        "model_config.json": {"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_8_0"},
        "source_manifest.json": {"origin": "test_fixture"},
        "conversion_report.json": {"roundtrip": "byte_identical", "profile": "dlss_nr_310_8_0"},
        "numerics.json": {},
        "preprocessing.json": {},
    }
    for name, value in documents.items():
        (folder / name).write_text(json.dumps(value), encoding="utf-8")
    return folder


def make_validation_report(folder):
    from musubi_tuner.dlssnr.identity import file_sha256, implementation_identity, json_sha256

    report = {
        "schema": "dlssnr_forward_validation_v1",
        "profile": "dlss_nr_310_8_0",
        "numerics_profile": "train_surrogate",
        "model_sha256": file_sha256(folder / "model.safetensors"),
        "implementation_sha256": json_sha256(implementation_identity()),
        "reference_identity": {"test_fixture": "synthetic evidence, not native compatibility"},
        "float_validated": True,
        "checks": {"raw_head": True, "neural_preclamp": True, "rendered_proxy": True},
    }
    path = folder / "forward_validation_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_adapter_identity_covers_non_target_base_parameters():
    model = torch.nn.ModuleDict({"target": ChannelLinear(2, 2), "other": ChannelLinear(2, 2)})
    for parameter in model.parameters():
        torch.nn.init.ones_(parameter)
    before = base_target_sha256(model, ["target.weight"])
    with torch.no_grad():
        model["other"].weight.add_(1)
    assert base_target_sha256(model, ["target.weight"]) != before


def test_lane_constraint_does_not_rewrite_signed_zero_in_a_frozen_lora_base():
    from test_dlssnr_training import SmallNR, small_inject
    from musubi_tuner.training.dlssnr_trainer import clear_lane15_state

    model = SmallNR()
    with torch.no_grad():
        model.blocks["0"].input_adapter.weight[:, 15].fill_(-0.0)
    network = small_inject(model, {})
    before = base_target_sha256(model, network.target_names)
    optimizer = torch.optim.AdamW(network.parameters())
    clear_lane15_state(model, optimizer)
    assert base_target_sha256(model, network.target_names) == before


def test_canonical_save_preserves_metadata_and_opaque_bytes(tmp_path):
    artifacts = importlib.import_module("musubi_tuner.dlssnr.artifacts")
    source = tmp_path / "base"
    source.mkdir()
    model = torch.nn.Linear(2, 2, bias=False)
    save_file(
        {"weight": model.weight.detach(), "blocks.31.opaque.layer3": torch.tensor([7, 128], dtype=torch.uint8)},
        str(source / "model.safetensors"),
    )
    save_file({"block31.layer3.layer": torch.tensor([7, 128], dtype=torch.uint8)}, str(source / "opaque_records.safetensors"))
    (source / "model_config.json").write_text(json.dumps({"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_8_0"}))
    (source / "source_manifest.json").write_text('{"origin":"fixture"}')
    before = (source / "opaque_records.safetensors").read_bytes()
    metadata = {"experimental_surrogate": True, "temporal_trained": False}
    output = tmp_path / "final"
    artifacts.save_canonical(model, output, source_dir=source, metadata=metadata)
    assert (output / "opaque_records.safetensors").read_bytes() == before
    assert json.loads((output / "source_manifest.json").read_text()) == {"origin": "fixture"}
    tensors = load_file(output / "model.safetensors")
    assert tensors["blocks.31.opaque.layer3"].tolist() == [7, 128]
    assert json.loads((output / "training_metadata.json").read_text())["experimental_surrogate"] is True
    assert (source / "opaque_records.safetensors").read_bytes() == before


def test_normal_training_rejects_missing_conversion_provenance(tmp_path):
    artifacts = importlib.import_module("musubi_tuner.dlssnr.artifacts")
    save_file({"weight": torch.ones(2, 2)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model_config.json").write_text(json.dumps({"schema": "dlssnr_canonical_v1", "profile": "dlss_nr_310_8_0"}))
    with pytest.raises(ValueError, match="validation|conversion|provenance"):
        artifacts.inspect_canonical(tmp_path, development_smoke=False)


def test_normal_inspection_does_not_require_forward_evidence(tmp_path):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    identity = inspect_canonical(make_canonical(tmp_path))
    assert identity["model_sha256"]
    assert "conversion_report.json" in identity
    assert "forward_validation_report" not in identity


def test_normal_inspection_ignores_unrequested_report_sidecars(tmp_path):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    (tmp_path / "forward_validation_report.json").write_text("stale report", encoding="utf-8")
    assert "forward_validation_report" not in inspect_canonical(tmp_path)


@pytest.mark.parametrize("development_smoke", [False, True])
def test_requested_forward_evidence_is_required_even_in_smoke_mode(tmp_path, development_smoke):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    with pytest.raises(ValueError, match="forward validation"):
        inspect_canonical(tmp_path, development_smoke=development_smoke, require_forward_validation=True)


@pytest.mark.parametrize("development_smoke", [False, True])
@pytest.mark.parametrize("explicit_path", [False, True])
def test_requested_forward_evidence_is_verified_and_recorded(tmp_path, development_smoke, explicit_path):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical
    from musubi_tuner.dlssnr.identity import file_sha256

    make_canonical(tmp_path)
    report = make_validation_report(tmp_path)
    options = {"validation_report": report} if explicit_path else {"require_forward_validation": True}
    identity = inspect_canonical(tmp_path, development_smoke=development_smoke, **options)
    assert identity["forward_validation_report"] == file_sha256(report)


@pytest.mark.parametrize("development_smoke", [False, True])
def test_explicit_missing_forward_evidence_is_not_ignored(tmp_path, development_smoke):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    with pytest.raises(ValueError, match="forward validation"):
        inspect_canonical(tmp_path, development_smoke=development_smoke, validation_report=tmp_path / "missing.json")


@pytest.mark.parametrize("development_smoke", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("profile", "other"),
        ("numerics_profile", "native_reference"),
        ("model_sha256", "other"),
        ("implementation_sha256", "other"),
        ("reference_identity", {}),
        ("float_validated", False),
        ("checks", {"raw_head": True, "neural_preclamp": True, "rendered_proxy": False}),
    ],
)
def test_explicit_invalid_forward_evidence_is_rejected(tmp_path, development_smoke, field, value):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    path = make_validation_report(tmp_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    report[field] = value
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="forward validation"):
        inspect_canonical(tmp_path, development_smoke=development_smoke, validation_report=path)


@pytest.mark.parametrize(
    "missing",
    ["source_manifest.json", "conversion_report.json", "opaque_records.safetensors", "numerics.json", "preprocessing.json"],
)
def test_optional_evidence_does_not_relax_conversion_provenance(tmp_path, missing):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError, match="conversion/provenance"):
        inspect_canonical(tmp_path)


@pytest.mark.parametrize("field,value", [("roundtrip", "failed"), ("profile", "other")])
def test_optional_evidence_does_not_relax_conversion_roundtrip(tmp_path, field, value):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    path = tmp_path / "conversion_report.json"
    conversion = json.loads(path.read_text(encoding="utf-8"))
    conversion[field] = value
    path.write_text(json.dumps(conversion), encoding="utf-8")
    with pytest.raises(ValueError, match="source round-trip"):
        inspect_canonical(tmp_path)


@pytest.mark.parametrize("field", ["schema", "profile"])
def test_optional_evidence_does_not_relax_canonical_identity(tmp_path, field):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    make_canonical(tmp_path)
    path = tmp_path / "model_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config[field] = "other"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="schema/profile"):
        inspect_canonical(tmp_path)


def test_canonical_loader_rejects_hidden_unknown_keys_and_nonfinite_weights(tmp_path):
    from test_dlssnr_training import SmallNR

    model = SmallNR()
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for extra, corrupt in [(True, False), (False, True)]:
        tensors = {key: value.clone() for key, value in original.items()}
        if extra:
            tensors["unrecognized.opaque.weight"] = torch.zeros(1)
        if corrupt:
            tensors["blocks.70.head.rgb.weight"][0, 0] = float("nan")
        file = tmp_path / "bad.safetensors"
        save_file(tensors, str(file))
        with pytest.raises((ValueError, RuntimeError)):
            model.load_canonical(str(file))


def test_adapter_loader_rejects_mismatched_scaling(tmp_path):
    from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA, load_adapter, save_adapter

    first, second = DLSSNRLoRA(), DLSSNRLoRA()
    first.add("linear.weight", ChannelLinear(4, 4), 2, 2, 0)
    second.add("linear.weight", ChannelLinear(4, 4), 2, 4, 0)
    file = tmp_path / "adapter.safetensors"
    save_adapter(first, file, "base")
    with pytest.raises(ValueError, match="alpha|scal|configuration"):
        load_adapter(second, file, "base")
