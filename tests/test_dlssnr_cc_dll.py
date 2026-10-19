import importlib
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from test_dlssnr_dll import pe_image, weight_map


def two_resource_pe(ht, cc):
    """Independent two-resource PE fixture, with identical record names in each map."""
    cc_start = (0x400 + len(ht) + 7) & ~7
    end = cc_start + len(cc)
    data = bytearray(end)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\0\0"
    struct.pack_into("<HH", data, 0x84, 0x8664, 1)
    struct.pack_into("<H", data, 0x94, 0xF0)
    struct.pack_into("<H", data, 0x98, 0x20B)
    struct.pack_into("<I", data, 0x98 + 108, 16)
    struct.pack_into("<II", data, 0x98 + 128, 0x1000, end - 0x200)
    data[0x188:0x190] = b".rsrc\0\0\0"
    struct.pack_into("<IIII", data, 0x190, end - 0x200, 0x1000, end - 0x200, 0x200)
    struct.pack_into("<HHII", data, 0x20C, 0, 1, 10, 0x80000020)
    struct.pack_into("<HH", data, 0x22C, 2, 0)
    struct.pack_into("<II", data, 0x230, 0x80000100, 0x80000050)
    struct.pack_into("<II", data, 0x238, 0x80000140, 0x80000070)
    struct.pack_into("<HHII", data, 0x25C, 0, 1, 0x409, 0x90)
    struct.pack_into("<HHII", data, 0x27C, 0, 1, 0x409, 0xA0)
    struct.pack_into("<IIII", data, 0x290, 0x1200, len(ht), 0, 0)
    struct.pack_into("<IIII", data, 0x2A0, 0x1000 + cc_start - 0x200, len(cc), 0, 0)
    for offset, name in ((0x300, "WEIGHTS_HT"), (0x340, "WEIGHTS_CC_SMALL_FP8")):
        encoded = name.encode("utf-16-le")
        struct.pack_into("<H", data, offset, len(name))
        data[offset + 2 : offset + 2 + len(encoded)] = encoded
    data[0x400 : 0x400 + len(ht)] = ht
    data[cc_start:] = cc
    return bytes(data) + b"certificate-and-overlay"


def test_explicit_cc_resource_selection_preserves_ht_default_and_other_bytes():
    native = importlib.import_module("musubi_tuner.dlssnr.dll")
    dll = two_resource_pe(weight_map([("same", b"\x11\x22")]), weight_map([("same", b"\xa1\xb2")]))
    assert bytes(native.read_weights(dll).records["same"].payload) == b"\x11\x22"
    cc = native.read_weights(dll, resource_name="WEIGHTS_CC_SMALL_FP8")
    assert bytes(cc.records["same"].payload) == b"\xa1\xb2"
    patched = cc.replace(dll, {"same": b"\xc3\xd4"})
    assert patched == dll.replace(b"\xa1\xb2", b"\xc3\xd4")
    assert bytes(native.read_weights(patched).records["same"].payload) == b"\x11\x22"
    with pytest.raises(ValueError, match="UNKNOWN"):
        native.read_weights(dll, resource_name="UNKNOWN")


@pytest.fixture
def cc_source(tmp_path, monkeypatch):
    io = importlib.import_module("musubi_tuner.dlssnr.cc_dll_io")
    cc = weight_map([("block0.layer0.layer", b"\x01\x80\x03\x04"), ("block30.layer0.blend_scale", b"\x00\x38")])
    # Substitute only the audit identity; all PE parsing and file I/O remain real.
    monkeypatch.setattr(io, "AUDITED_RESOURCE_SHA256", hashlib.sha256(cc).hexdigest())
    source = tmp_path / "source.dll"
    source.write_bytes(two_resource_pe(weight_map([("block0.layer0.layer", b"\x11\x22")]), cc))
    return io, source


def test_cc_unpack_and_original_repack_are_lossless(cc_source, tmp_path):
    io, source = cc_source
    original = source.read_bytes()
    folder = tmp_path / "records"
    report = io.unpack_cc_dll(source, folder)
    assert report["roundtrip"] == "byte_identical"
    assert report["record_count"] == 2
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["resource_name"] == "WEIGHTS_CC_SMALL_FP8"
    assert manifest["storage"] == "opaque_packed_records"
    assert manifest["training_ready"] is False
    row = manifest["records"][0]
    assert (row["name"], row["byte_length"], row["dimensions"]) == ("block0.layer0.layer", 4, [2])
    assert (folder / row["file"]).read_bytes() == b"\x01\x80\x03\x04"
    output = tmp_path / "repacked.dll"
    packed = io.pack_cc_dll(source, folder, output)
    assert output.read_bytes() == original == source.read_bytes()
    assert packed["whole_dll_byte_identical"] is True
    assert packed["changed_records"] == []
    assert packed["native_export_validated"] is False
    assert json.loads(Path(str(output) + ".report.json").read_text())["output_sha256"] == hashlib.sha256(original).hexdigest()


def test_cc_patch_changes_only_selected_payload_not_ht_or_metadata(cc_source, tmp_path):
    io, source = cc_source
    folder = tmp_path / "records"
    io.unpack_cc_dll(source, folder)
    (folder / "records/block30.layer0.blend_scale.bin").write_bytes(b"\x00\x00")
    output = tmp_path / "ablation.dll"
    report = io.pack_cc_dll(source, folder, output)
    parser = importlib.import_module("musubi_tuner.dlssnr.dll")
    original = source.read_bytes()
    cc = parser.read_weights(original, resource_name="WEIGHTS_CC_SMALL_FP8")
    offset = cc.records["block30.layer0.blend_scale"].offset
    assert output.read_bytes() == original[:offset] + b"\x00\x00" + original[offset + 2 :]
    assert report["changed_bytes"] == 1
    assert [row["name"] for row in report["changed_records"]] == ["block30.layer0.blend_scale"]
    assert report["payload_only_modified"] is True


@pytest.mark.parametrize("mutation", ["schema", "resource", "hash", "missing", "extra", "duplicate", "size", "metadata", "path"])
def test_cc_pack_rejects_tampered_manifest_before_publishing(cc_source, tmp_path, mutation):
    io, source = cc_source
    folder = tmp_path / "records"
    io.unpack_cc_dll(source, folder)
    path = folder / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "schema":
        manifest["schema"] = "dlssnr_canonical_v1"
    elif mutation == "resource":
        manifest["resource_name"] = "WEIGHTS_HT"
    elif mutation == "hash":
        manifest["resource_sha256"] = "0" * 64
    elif mutation == "missing":
        manifest["records"].pop()
    elif mutation == "extra":
        manifest["records"].append({"name": "unexpected"})
    elif mutation == "duplicate":
        manifest["records"][1] = manifest["records"][0]
    elif mutation == "size":
        manifest["records"][0]["byte_length"] = 2
    elif mutation == "metadata":
        manifest["records"][0]["dtype_code"] = 99
    else:
        manifest["records"][0]["file"] = "../outside.bin"
    path.write_text(json.dumps(manifest))
    output = tmp_path / "bad.dll"
    with pytest.raises(ValueError):
        io.pack_cc_dll(source, folder, output)
    assert not output.exists()
    assert not Path(str(output) + ".report.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "short", "long", "directory"])
def test_cc_pack_rejects_wrong_record_files(cc_source, tmp_path, mutation):
    io, source = cc_source
    folder = tmp_path / "records"
    io.unpack_cc_dll(source, folder)
    path = folder / "records/block0.layer0.layer.bin"
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        (folder / "records/extra.bin").write_bytes(b"\x00\x00")
    elif mutation == "directory":
        path.unlink()
        path.mkdir()
    else:
        path.write_bytes(b"\x00" * (2 if mutation == "short" else 6))
    output = tmp_path / "bad.dll"
    with pytest.raises((OSError, ValueError)):
        io.pack_cc_dll(source, folder, output)
    assert not output.exists()


@pytest.mark.parametrize("target", ["unpack", "dll", "report"])
def test_cc_outputs_never_overwrite_existing_paths(cc_source, tmp_path, target):
    io, source = cc_source
    folder, output = tmp_path / "records", tmp_path / "out.dll"
    occupied = folder if target == "unpack" else output if target == "dll" else Path(str(output) + ".report.json")
    occupied.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        if target == "unpack":
            io.unpack_cc_dll(source, folder)
        else:
            io.pack_cc_dll(tmp_path / "not-read.dll", folder, output)
    assert occupied.read_bytes() == b"keep"


def test_cc_import_rejects_unaudited_resource_without_creating_output(tmp_path):
    io = importlib.import_module("musubi_tuner.dlssnr.cc_dll_io")
    path = tmp_path / "unknown.dll"
    path.write_bytes(two_resource_pe(weight_map([("a", b"\x00\x00")]), weight_map([("b", b"\x00\x00")])))
    with pytest.raises(ValueError, match="audited|unsupported"):
        io.unpack_cc_dll(path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("script", ["dlssnr_unpack_model.py", "dlssnr_pack_model.py"])
def test_cc_cli_handles_bad_input_without_creating_output(tmp_path, script):
    root = Path(__file__).resolve().parents[1]
    bad = tmp_path / "bad.dll"
    bad.write_bytes(b"invalid")
    output = tmp_path / "output"
    args = ["--source_dll", str(bad)] if "unpack" in script else ["--template_dll", str(bad), "--record_dir", str(tmp_path)]
    result = subprocess.run(
        [sys.executable, str(root / script), *args, "--weights", "CC", "--output", str(output)], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "PE DLL" in result.stderr
    assert not output.exists()


def test_real_cc_cli_roundtrip_and_one_record_ablation(tmp_path):
    source = os.environ.get("DLSSNR_CC_DLL_PATH")
    if not source:
        pytest.skip("set DLSSNR_CC_DLL_PATH to the audited 310.9.16 DLL")
    from musubi_tuner.dlssnr.dll import read_weights

    root = Path(__file__).resolve().parents[1]
    bundle = tmp_path / "models"
    model = bundle / "CC"
    result = subprocess.run(
        [sys.executable, str(root / "dlssnr_unpack_model.py"), "--source_dll", source, "--output_dir", str(bundle)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (bundle / "HT/model.safetensors").is_file()
    ht_report = json.loads((bundle / "HT/conversion_report.json").read_text())
    assert ht_report["output_dir"] == str((bundle / "HT").resolve())
    assert ht_report["roundtrip"] == "byte_identical"
    index = json.loads((bundle / "models.json").read_text())
    assert set(index["models"]) == {"HT", "CC"}
    manifest = json.loads((model / "manifest.json").read_text())
    assert manifest["record_count"] == 77
    assert manifest["block_count"] == 31
    assert manifest["payload_bytes"] == 113887490
    before = Path(source).read_bytes()
    source_hash = hashlib.sha256(before).hexdigest()
    output = tmp_path / "unchanged.dll"
    command = [sys.executable, str(root / "dlssnr_pack_model.py"), "--template_dll", source]
    result = subprocess.run(
        [*command, "--model_dir", str(bundle), "--weights", "CC", "--output_dll", str(output)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert output.read_bytes() == before
    record = model / "records/block30.layer0.blend_scale.bin"
    assert len(record.read_bytes()) == 2
    record.write_bytes(b"\x00\x00" if record.read_bytes() != b"\x00\x00" else b"\x00\x38")
    changed = tmp_path / "changed.dll"
    result = subprocess.run([*command, "--model_dir", str(model), "--output_dll", str(changed)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    parsed = read_weights(before, resource_name="WEIGHTS_CC_SMALL_FP8")
    offset = parsed.records["block30.layer0.blend_scale"].offset
    assert changed.read_bytes() == before[:offset] + record.read_bytes() + before[offset + 2 :]
    ht_output = tmp_path / "ht_selected.dll"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "dlssnr_pack_model.py"),
            "--template_dll",
            str(changed),
            "--model_dir",
            str(bundle),
            "--weights",
            "HT",
            "--output_dll",
            str(ht_output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert ht_output.read_bytes() == changed.read_bytes()
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == source_hash


def test_existing_entry_points_can_explicitly_unpack_and_autodetect_cc(cc_source, tmp_path):
    _, source = cc_source
    io = importlib.import_module("musubi_tuner.dlssnr.dll_models")
    folder = tmp_path / "cc"
    report = io.unpack_models(source, folder, weights="CC")
    assert report["resource_name"] == "WEIGHTS_CC_SMALL_FP8"
    output = tmp_path / "out.dll"
    packed = io.pack_selected_model(source, folder, output)
    assert packed["resource_name"] == "WEIGHTS_CC_SMALL_FP8"
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("kwargs", [{"weights": "HT"}, {"mix": 0.5}, {"strength": 2}, {"merge_lora": "adapter.safetensors"}])
def test_cc_selection_rejects_wrong_resource_or_numeric_training_options(cc_source, tmp_path, kwargs):
    cc, source = cc_source
    folder = tmp_path / "cc"
    cc.unpack_cc_dll(source, folder)
    io = importlib.import_module("musubi_tuner.dlssnr.dll_models")
    with pytest.raises(ValueError, match="CC|resource"):
        io.pack_selected_model(source, folder, tmp_path / "out.dll", **kwargs)
    assert not (tmp_path / "out.dll").exists()


def test_bundle_selection_requires_an_explicit_weight_name(tmp_path):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_models")
    bundle = tmp_path / "models"
    bundle.mkdir()
    (bundle / "models.json").write_text(
        json.dumps({"schema": "dlssnr_weight_bundle_v1", "models": {"HT": {"directory": "HT"}, "CC": {"directory": "CC"}}})
    )
    with pytest.raises(ValueError, match="--weights"):
        io.pack_selected_model(tmp_path / "not-read.dll", bundle, tmp_path / "out.dll")


def test_multi_unpack_failure_does_not_publish_a_partial_bundle(cc_source, tmp_path, monkeypatch):
    _, source = cc_source
    io = importlib.import_module("musubi_tuner.dlssnr.dll_models")

    def fail_cc(*args, **kwargs):
        raise ValueError("CC validation failed")

    def fake_ht(source, output, *, save_precision):
        assert save_precision == "native"
        output.mkdir()
        (output / "conversion_report.json").write_text(json.dumps({"output_dir": str(output)}))
        return {"output_dir": str(output), "roundtrip": "byte_identical"}

    # The orchestration owns atomic publication, not either model's conversion.
    monkeypatch.setattr(io, "unpack_dll", fake_ht)
    monkeypatch.setattr(io, "unpack_cc_dll", fail_cc)
    output = tmp_path / "bundle"
    with pytest.raises(ValueError, match="CC validation"):
        io.unpack_models(source, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".dlssnr-models-*"))


@pytest.fixture
def fast_ht_conversion(monkeypatch):
    io = importlib.import_module("musubi_tuner.dlssnr.dll_models")

    def convert(source, output, *, save_precision):
        assert save_precision == "native"
        output.mkdir()
        report = {"output_dir": str(output), "roundtrip": "byte_identical", "resource_name": "WEIGHTS_HT"}
        (output / "conversion_report.json").write_text(json.dumps(report))
        (output / "model.safetensors").write_bytes(b"canonical-test-fixture")
        return report

    # HT conversion has its own tests; this fixture isolates bundle publication.
    monkeypatch.setattr(io, "unpack_dll", convert)
    return io


def test_multi_unpack_publishes_named_children_and_resolved_reports(cc_source, tmp_path, fast_ht_conversion):
    _, source = cc_source
    output = tmp_path / "models"
    report = fast_ht_conversion.unpack_models(source, output)
    assert set(report["models"]) == {"HT", "CC"}
    assert (output / "HT/model.safetensors").read_bytes() == b"canonical-test-fixture"
    assert (output / "CC/records/block30.layer0.blend_scale.bin").read_bytes() == b"\x00\x38"
    assert report["models"]["CC"]["output_dir"] == str((output / "CC").resolve())
    ht_report = json.loads((output / "HT/conversion_report.json").read_text())
    assert ht_report["output_dir"] == str((output / "HT").resolve())
    assert json.loads((output / "models.json").read_text())["models"]["CC"]["directory"] == "CC"


def test_single_ht_auto_unpack_keeps_the_legacy_output_root(tmp_path, fast_ht_conversion):
    source = tmp_path / "ht.dll"
    source.write_bytes(pe_image(weight_map([("a", b"\x01\x02")])))
    output = tmp_path / "model"
    report = fast_ht_conversion.unpack_models(source, output)
    assert report["resource_name"] == "WEIGHTS_HT"
    assert (output / "model.safetensors").is_file()
    assert not (output / "HT").exists()
    assert not (output / "models.json").exists()


def test_cc_rejects_record_symlink_outside_input_directory(cc_source, tmp_path):
    io, source = cc_source
    folder = tmp_path / "cc"
    io.unpack_cc_dll(source, folder)
    path = folder / "records/block30.layer0.blend_scale.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(ValueError, match="escapes"):
        io.pack_cc_dll(source, folder, tmp_path / "out.dll")
    assert outside.read_bytes() == b"\x00\x38"


def test_bundle_rejects_child_symlink_outside_bundle(cc_source, tmp_path):
    cc, source = cc_source
    outside = tmp_path / "outside"
    cc.unpack_cc_dll(source, outside)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "models.json").write_text(json.dumps({"schema": "dlssnr_weight_bundle_v1", "models": {"CC": {"directory": "CC"}}}))
    try:
        (bundle / "CC").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    io = importlib.import_module("musubi_tuner.dlssnr.dll_models")
    with pytest.raises(ValueError, match="escapes"):
        io.pack_selected_model(source, bundle, tmp_path / "out.dll", weights="CC")
