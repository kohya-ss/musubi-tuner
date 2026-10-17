"""Lossless CC_SMALL_FP8 packed-record experiments, not a canonical training format."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from musubi_tuner.dlssnr.dll import read_weights
from musubi_tuner.dlssnr.dll_io import _publish_new, _require_new, _sha256, _write_staged
from musubi_tuner.dlssnr.identity import file_sha256


RESOURCE_NAME = "WEIGHTS_CC_SMALL_FP8"
AUDITED_RESOURCE_SHA256 = "f743e8eb320f42bc9bdc992ee2bf51f7bb7c8688ced4ecf5ce6807ec4eef7962"
MANIFEST_SCHEMA = "dlssnr_cc_packed_records_v1"


def _load_template(path: Path):
    data = path.read_bytes()
    resource = read_weights(data, resource_name=RESOURCE_NAME)
    digest = _sha256(memoryview(data)[resource.offset : resource.offset + resource.size])
    if digest != AUDITED_RESOURCE_SHA256:
        raise ValueError(
            f"unsupported {RESOURCE_NAME} resource {digest}; expected audited 310.9.16 resource "
            f"{AUDITED_RESOURCE_SHA256}. Use an original CC resource, not an already modified one."
        )
    return data, resource


def _manifest(path: Path, data: bytes, resource) -> dict:
    records = []
    for name, record in resource.records.items():
        match = re.fullmatch(r"block(\d+)\.layer(\d+)\.([a-z_]+)", name)
        if match is None:
            raise ValueError(f"unsupported CC record name: {name}")
        records.append(
            {
                "name": name,
                "block": int(match[1]),
                "layer": int(match[2]),
                "parameter": match[3],
                "file": f"records/{name}.bin",
                "byte_length": len(record.payload),
                "payload_offset_in_resource": record.offset - resource.offset,
                "dtype_code": record.dtype_code,
                "metadata0": record.metadata0,
                "metadata1": record.metadata1,
                "dimensions": list(record.dimensions),
                "source_sha256": _sha256(record.payload),
            }
        )
    records.sort(key=lambda row: (row["block"], row["layer"], row["parameter"]))
    return {
        "schema": MANIFEST_SCHEMA,
        "resource_name": RESOURCE_NAME,
        "resource_sha256": AUDITED_RESOURCE_SHA256,
        "resource_bytes": resource.size,
        "resource_file_offset": resource.offset,
        "source_dll": {"path": str(path.resolve()), "sha256": _sha256(data)},
        "storage": "opaque_packed_records",
        "training_ready": False,
        "record_count": len(records),
        "block_count": len({row["block"] for row in records}),
        "payload_bytes": sum(row["byte_length"] for row in records),
        "records": records,
    }


def unpack_cc_dll(source_dll: str | Path, output_dir: str | Path) -> dict:
    """Write exact serialized payload bytes plus audited names and metadata."""
    source, output = Path(source_dll), Path(output_dir)
    _require_new(output)
    original, resource = _load_template(source)
    manifest = _manifest(source, original, resource)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dlssnr-cc-unpack-", dir=output.parent) as temporary:
        staged = Path(temporary) / "model"
        (staged / "records").mkdir(parents=True)
        restored = {}
        for row in manifest["records"]:
            path = staged / row["file"]
            _write_staged(path, bytes(resource.records[row["name"]].payload))
            restored[row["name"]] = path.read_bytes()
        if resource.replace(original, restored) != original:
            raise ValueError("CC extraction failed the whole-DLL byte roundtrip")
        manifest["byte_roundtrip_verified"] = True
        _write_staged(staged / "manifest.json", json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8"))
        if file_sha256(source) != manifest["source_dll"]["sha256"]:
            raise ValueError("source DLL changed during CC unpack")
        _require_new(output)
        staged.rename(output)
    return {
        "output_dir": str(output.resolve()),
        "resource_name": RESOURCE_NAME,
        "record_count": manifest["record_count"],
        "block_count": manifest["block_count"],
        "payload_bytes": manifest["payload_bytes"],
        "roundtrip": "byte_identical",
        "training_ready": False,
    }


def _load_records(directory: Path, expected: dict):
    root = directory.resolve()
    manifest_path = (root / "manifest.json").resolve()
    if not manifest_path.is_relative_to(root):
        raise ValueError("CC manifest path escapes the record directory")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise ValueError("CC record manifest must be an object")
    # Container paths/offsets may change; the complete CC resource identity must not.
    for field in (
        "schema",
        "resource_name",
        "resource_sha256",
        "resource_bytes",
        "storage",
        "training_ready",
        "record_count",
        "block_count",
        "payload_bytes",
        "records",
    ):
        if manifest.get(field) != expected[field]:
            raise ValueError(f"CC record manifest {field} does not match the audited template")
    records_dir = (root / "records").resolve()
    if not records_dir.is_relative_to(root):
        raise ValueError("CC records path escapes the record directory")
    filenames = {Path(row["file"]).name for row in expected["records"]}
    if {path.name for path in records_dir.iterdir()} != filenames:
        raise ValueError("CC record file inventory differs from the manifest")
    payloads, snapshots = {}, {manifest_path: _sha256(manifest_bytes)}
    for row in expected["records"]:
        path = (root / row["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"{row['name']}: record path escapes the record directory")
        with path.open("rb") as handle:
            blob = handle.read(row["byte_length"] + 1)
        if len(blob) != row["byte_length"]:
            raise ValueError(f"{row['name']}: replacement payload size differs from the template")
        payloads[row["name"]] = blob
        snapshots[path] = _sha256(blob)
    return payloads, snapshots


def pack_cc_dll(template_dll: str | Path, record_dir: str | Path, output_dll: str | Path) -> dict:
    """Replace only same-sized CC payloads, preserving HT and all surrounding bytes."""
    template, directory, output = Path(template_dll), Path(record_dir), Path(output_dll)
    report_path = Path(str(output) + ".report.json")
    _require_new(output)
    _require_new(report_path)
    original, resource = _load_template(template)
    manifest = _manifest(template, original, resource)
    payloads, snapshots = _load_records(directory, manifest)
    candidate = resource.replace(original, payloads)
    extracted = read_weights(candidate, resource_name=RESOURCE_NAME)
    changed = []
    for name, payload in payloads.items():
        if extracted.records[name].payload != payload:
            raise ValueError(f"{name}: CC re-extraction differs from the supplied payload")
        before = resource.records[name].payload
        if before != payload:
            changed.append(
                {
                    "name": name,
                    "byte_length": len(payload),
                    "changed_bytes": sum(left != right for left, right in zip(before, payload)),
                    "source_sha256": _sha256(before),
                    "output_sha256": _sha256(payload),
                }
            )
    if file_sha256(template) != manifest["source_dll"]["sha256"]:
        raise ValueError("template DLL changed during CC pack")
    for path, digest in snapshots.items():
        if file_sha256(path) != digest:
            raise ValueError(f"CC input changed during pack: {path}")
    report = {
        "schema": "dlssnr_cc_dll_export_v1",
        "resource_name": RESOURCE_NAME,
        "template_dll": str(template.resolve()),
        "template_sha256": manifest["source_dll"]["sha256"],
        "record_dir": str(directory.resolve()),
        "output_dll": str(output.resolve()),
        "output_sha256": _sha256(candidate),
        "source_resource_sha256": AUDITED_RESOURCE_SHA256,
        "resource_sha256": _sha256(memoryview(candidate)[resource.offset : resource.offset + resource.size]),
        "record_count": len(payloads),
        "changed_records": changed,
        "changed_bytes": sum(row["changed_bytes"] for row in changed),
        "whole_dll_byte_identical": candidate == original,
        "payload_only_modified": True,
        "native_export_validated": False,
        "native_load_test": "not_run",
        "training_ready": False,
        "limits": [
            "Opaque packed records, not homogeneous FP8/FP16 tensors or a canonical training checkpoint.",
            "Same-size replacements are byte-verified, not checked for valid numeric values or native behavior.",
            "HT, serialized metadata, other resources, PE checksum and signature bytes are preserved.",
            "Changed DLLs are not re-signed and may be rejected by a native loader.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dlssnr-cc-pack-", dir=output.parent) as temporary:
        staged_dll, staged_report = Path(temporary) / "model.dll", Path(temporary) / "report.json"
        _write_staged(staged_dll, candidate)
        _write_staged(staged_report, json.dumps(report, indent=2, allow_nan=False).encode("utf-8"))
        _publish_new(staged_dll, output)
        try:
            _publish_new(staged_report, report_path)
        except BaseException:
            output.unlink()
            raise
    return report
