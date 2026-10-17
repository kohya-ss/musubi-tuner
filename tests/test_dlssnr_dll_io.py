import importlib
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from test_dlssnr_dll import pe_image, weight_map
from test_dlssnr_native_quantization import tiny_record


def test_unpack_refuses_existing_output_without_touching_it(tmp_path):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        io.unpack_dll(tmp_path / "not-read.dll", output)
    assert marker.read_text() == "keep"


def test_unpack_rejects_unaudited_resource_before_creating_artifacts(tmp_path):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    dll = tmp_path / "unknown.dll"
    dll.write_bytes(pe_image(weight_map([("a", b"\x00\x00")])))
    with pytest.raises(ValueError, match="audited|310.8.0"):
        io.unpack_dll(dll, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("target", ["dll", "report"])
def test_pack_refuses_existing_outputs_before_reading_inputs(tmp_path, target):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    dll = tmp_path / "out.dll"
    occupied = dll if target == "dll" else Path(str(dll) + ".report.json")
    occupied.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        io.pack_dll(tmp_path / "not-read.dll", tmp_path / "not-read", dll)
    assert occupied.read_bytes() == b"keep"
    if target == "report":
        assert not dll.exists()


@pytest.mark.parametrize("kwargs", [{"mix": -0.1}, {"mix": 1.1}, {"strength": float("nan")}, {"lora_multiplier": 0.5}])
def test_invalid_multiplier_options_fail_before_creating_outputs(tmp_path, kwargs):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    with pytest.raises(ValueError):
        io.pack_dll(tmp_path / "not-read.dll", tmp_path / "not-read", tmp_path / "out.dll", **kwargs)
    assert list(tmp_path.iterdir()) == []


def small_canonical(tmp_path):
    from musubi_tuner.dlssnr.checkpoint import LoadedSource, repack_record
    from musubi_tuner.dlssnr.convert import canonical_config
    from musubi_tuner.dlssnr.profiles import PROFILE_ID

    record = tiny_record()
    tensors = {"weight": np.ones((128, 32), np.float32), "skip": np.array([1], np.float32)}
    blob = repack_record(record, tensors)
    digest = hashlib.sha256(blob).hexdigest()
    manifest = {
        "totals": {"blockCount": 71},
        "stages": [{"id": "enc32", "file": "enc32.bin", "packedByteLength": len(blob), "sha256": digest}],
        "tensors": [{"name": record.name, "stage": "enc32", "stageOffset": 0, "byteLength": len(blob)}],
    }
    source = LoadedSource(tmp_path, manifest, {"enc32": blob}, {"enc32": digest}, (record,), {record.name: blob})
    save_file(tensors, tmp_path / "model.safetensors")
    save_file({record.name: np.frombuffer(blob, np.uint8)}, tmp_path / "opaque_records.safetensors")
    documents = {
        "model_config.json": canonical_config((record,)),
        "source_manifest.json": manifest,
        "conversion_report.json": {"profile": PROFILE_ID, "roundtrip": "byte_identical", "stage_sha256": source.stage_sha256},
        "numerics.json": {},
        "preprocessing.json": {},
    }
    for name, value in documents.items():
        (tmp_path / name).write_text(json.dumps(value))
    return source


def test_existing_canonical_provenance_can_have_extra_metadata_and_different_source_filenames(tmp_path):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    source = small_canonical(tmp_path)
    path = tmp_path / "source_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["stages"][0].update(file="custom-name.bin", producer="old-extractor")
    manifest["tensors"][0]["note"] = "original provenance"
    path.write_text(json.dumps(manifest))
    assert io._validate_canonical(tmp_path, source)["model_sha256"]


@pytest.mark.parametrize("mutation", ["stage_hash", "offset", "conversion_hash", "opaque"])
def test_export_rejects_mismatched_source_provenance(tmp_path, mutation):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    source = small_canonical(tmp_path)
    if mutation == "opaque":
        row = next(iter(source.blobs))
        save_file({row: np.ones(len(source.blobs[row]), np.uint8)}, tmp_path / "opaque_records.safetensors")
    else:
        path = tmp_path / ("conversion_report.json" if mutation == "conversion_hash" else "source_manifest.json")
        value = json.loads(path.read_text())
        if mutation == "conversion_hash":
            value["stage_sha256"]["enc32"] = "wrong"
        elif mutation == "stage_hash":
            value["stages"][0]["sha256"] = "wrong"
        else:
            value["tensors"][0]["stageOffset"] = 1
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="template"):
        io._validate_canonical(tmp_path, source)


@pytest.fixture(scope="module")
def real_canonical(tmp_path_factory):
    source = os.environ.get("DLSSNR_DLL_PATH")
    if not source:
        pytest.skip("set DLSSNR_DLL_PATH to a user-supplied audited 310.8.0 DLL")
    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    output = tmp_path_factory.mktemp("dll-canonical") / "base"
    report = io.unpack_dll(Path(source), output)
    assert report["roundtrip"] == "byte_identical"
    return Path(source), output


def test_real_dll_unpacks_for_normal_training_and_rebuilds_bit_exact(real_canonical, tmp_path):
    from musubi_tuner.dlssnr.artifacts import inspect_canonical
    from musubi_tuner.dlssnr.identity import file_sha256

    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    source, model = real_canonical
    assert inspect_canonical(model)["model_sha256"]
    original_hash = file_sha256(source)
    output = tmp_path / "original.dll"
    report = io.pack_dll(source, model, output)
    assert report["whole_dll_byte_identical"] is True
    assert report["quantized_tensors_verified"] is True
    assert report["totals"]["exported_changed_values"] == 0
    assert file_sha256(source) == file_sha256(output) == original_hash
    assert json.loads(Path(str(output) + ".report.json").read_text())["output_sha256"] == original_hash


def test_real_trained_checkpoint_exports_only_expected_payload_changes(real_canonical, tmp_path):
    from musubi_tuner.dlssnr.checkpoint import unpack_record
    from musubi_tuner.dlssnr.dll import read_weights
    from musubi_tuner.dlssnr.profiles import build_records
    from musubi_tuner.dlssnr.storage import STORAGE_METADATA_KEY, load_canonical_tensors, storage_policy

    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    source, base = real_canonical
    model = tmp_path / "trained"
    shutil.copytree(base, model)
    profile = "dlss_nr_310_8_0"
    tensors = {name: value.numpy() for name, value in load_canonical_tensors(model / "model.safetensors", profile).items()}
    tensors["blocks.0.ffn.fc1.weight"][0, 0] = np.float32(1.1)
    tensors["blocks.0.ffn.skip_scale"][0] = np.float32(1.0006)
    tensors["blocks.0.attn.temperature"][0] = np.float32(0.3333)
    storage = storage_policy(profile, "fp32")
    save_file(tensors, model / "model.safetensors", metadata={STORAGE_METADATA_KEY: json.dumps(storage)})
    config_path = model / "model_config.json"
    config = json.loads(config_path.read_text())
    config["weight_storage"] = storage
    config_path.write_text(json.dumps(config))
    del tensors
    output = tmp_path / "trained.dll"
    report = io.pack_dll(source, model, output)
    original_bytes, exported_bytes = source.read_bytes(), output.read_bytes()
    original, exported = read_weights(original_bytes), read_weights(exported_bytes)
    assert exported_bytes[: original.offset] == original_bytes[: original.offset]
    assert exported_bytes[original.offset + original.size :] == original_bytes[original.offset + original.size :]
    for name, record in original.records.items():
        if name != "block0.layer0.layer":
            assert exported.records[name].payload == record.payload
    values = unpack_record(build_records()[0], bytes(exported.records["block0.layer0.layer"].payload))
    assert values["blocks.0.ffn.fc1.weight"][0, 0] == 1.125
    assert values["blocks.0.ffn.skip_scale"][0] == 1.0009765625
    assert values["blocks.0.attn.temperature"][0] == np.float32(0.3333)
    assert report["totals"]["exported_changed_values"] == 3
    assert report["whole_dll_byte_identical"] is False
    assert report["native_export_validated"] is False
    reset = tmp_path / "reset.dll"
    io.pack_dll(source, model, reset, mix=0)
    assert reset.read_bytes() == original_bytes


def test_real_lora_is_merged_with_its_own_multiplier_before_export(real_canonical, tmp_path):
    import torch
    from musubi_tuner.dlssnr.checkpoint import unpack_record
    from musubi_tuner.dlssnr.dll import read_weights
    from musubi_tuner.dlssnr.identity import file_sha256
    from musubi_tuner.dlssnr.model import NRModel
    from musubi_tuner.dlssnr.profiles import build_records
    from musubi_tuner.networks.lora_dlssnr import base_target_sha256, inject, save_adapter

    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    source, base = real_canonical
    model = NRModel()
    model.load_canonical(str(base / "model.safetensors"))
    network = inject(model, {"profile": "vit_only", "rank": 1, "alpha": 1, "dropout": 0})
    with torch.no_grad():
        network.adapters[0].lora_down.fill_(0.5)
        network.adapters[0].lora_up.fill_(0.5)
    adapter_path = tmp_path / "adapter.safetensors"
    save_adapter(network, adapter_path, base_target_sha256(model, network.target_names))
    target = network.target_names[0]
    base_weight = model.state_dict()[target].detach()
    merged = base_weight + 0.125
    blended = (base_weight.double() + 0.3 * (merged.double() - base_weight)).float()
    expected = (blended.double() * 0.7).float().to(torch.float8_e4m3fn).float().numpy().copy()
    early_rounded = merged.to(torch.float8_e4m3fn).float()
    wrong_blended = (base_weight.double() + 0.3 * (early_rounded.double() - base_weight)).float()
    wrong = (wrong_blended.double() * 0.7).float().to(torch.float8_e4m3fn).float().numpy()
    assert np.any(expected != wrong), "fixture must catch native rounding before mix/strength"
    before = file_sha256(base / "model.safetensors")
    del model, network
    output = tmp_path / "lora.dll"
    report = io.pack_dll(source, base, output, merge_lora=adapter_path, lora_multiplier=0.5, mix=0.3, strength=0.7)
    record = next(
        record for record in build_records() if any(view.name == target for region in record.regions for view in region.views)
    )
    actual = unpack_record(record, bytes(read_weights(output.read_bytes()).records[record.name].payload))[target]
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert report["lora"]["multiplier"] == 0.5
    assert report["lora"]["sha256"] == file_sha256(adapter_path)
    assert file_sha256(base / "model.safetensors") == before


def test_real_qat_artifact_roundtrip_matches_published_weights_and_surrogate(real_canonical, tmp_path):
    import torch
    from musubi_tuner.dlssnr.artifacts import save_canonical
    from musubi_tuner.dlssnr.checkpoint import unpack_record
    from musubi_tuner.dlssnr.dll import read_weights
    from musubi_tuner.dlssnr.model import NRModel
    from musubi_tuner.dlssnr.numerics import fp32_execution
    from musubi_tuner.dlssnr.pipeline import forward_frame
    from musubi_tuner.dlssnr.profiles import build_records
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy, with_native_weight_qat
    from musubi_tuner.dlssnr.weight_quantization import native_storage_kinds, native_weight_ste

    io = importlib.import_module("musubi_tuner.dlssnr.dll_io")
    source, base = real_canonical
    model = NRModel().eval()
    model.load_canonical(str(base / "model.safetensors"))
    configure_model_runtime(model, with_native_weight_qat(default_runtime_policy(), True))
    edits = {
        "blocks.0.ffn.fc1.weight": 1.0626,
        "blocks.0.input_adapter.weight": 1.0006,
        "blocks.0.attn.prior": -2.0007,
        "blocks.0.attn.temperature": 0.3333,
        "blocks.70.blend_scale": 0.5003,
    }
    parameters = dict(model.named_parameters())
    kinds = native_storage_kinds()
    expected = {}
    with torch.no_grad():
        for name, value in edits.items():
            parameters[name].flatten()[0] = value
            expected[name] = native_weight_ste(parameters[name], kinds[name]).clone()
    trained = tmp_path / "qat"
    save_canonical(model, trained, source_dir=base)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = torch.full((1, 3, 48, 48), 0.5, device=device)
    controls = torch.ones(1, 5, 48, 48, device=device)
    with torch.no_grad(), fp32_execution():
        raw = forward_frame(model.to(device), image, controls, 42)["raw_head"].cpu()
    del model, parameters
    output = tmp_path / "qat.dll"
    report = io.pack_dll(source, trained, output)
    assert report["quantized_tensors_verified"] is True
    # The public importer intentionally accepts only the audited original DLL.
    # Re-extract modified payloads with the same record decoder used by export.
    resource = read_weights(output.read_bytes())
    unpacked = {
        name: torch.from_numpy(value)
        for record in build_records()
        for name, value in unpack_record(record, bytes(resource.records[record.name].payload)).items()
        if not name.startswith("opaque.")
    }
    for name, wanted in expected.items():
        torch.testing.assert_close(unpacked[name].view(torch.int32), wanted.view(torch.int32), rtol=0, atol=0)
    restored = NRModel().eval()
    restored.load_state_dict(unpacked, strict=True)
    with torch.no_grad(), fp32_execution():
        actual = forward_frame(restored.to(device), image, controls, 42)["raw_head"].cpu()
    torch.testing.assert_close(actual, raw, rtol=0, atol=0)
