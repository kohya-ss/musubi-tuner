"""Checked DLL/canonical import and export for audited HT and CC resources."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
from safetensors import safe_open

from musubi_tuner.dlssnr.checkpoint import LoadedSource
from musubi_tuner.dlssnr.convert import canonical_config, convert_source
from musubi_tuner.dlssnr.dll import read_weights
from musubi_tuner.dlssnr.identity import file_sha256
from musubi_tuner.dlssnr.native import repack_trained_record, validate_multipliers
from musubi_tuner.dlssnr.profiles import PROFILE_ID, CC_PROFILE_ID, build_records, canonical_names, get_profile, stage_bytes


logger = logging.getLogger(__name__)
AUDITED_RESOURCE_SHA256 = "836f445d06ecd2e59bb9f17b84b91c143396fd76ccda1c9dc7fe81d5edd548f4"


def _sha256(data) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_new(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _load_template(path: Path, profile: str = PROFILE_ID):
    topology = get_profile(profile)
    resource_name, audited_hash = "WEIGHTS_HT", AUDITED_RESOURCE_SHA256
    if profile == CC_PROFILE_ID:
        from musubi_tuner.dlssnr.cc_dll_io import AUDITED_RESOURCE_SHA256 as cc_hash, RESOURCE_NAME

        resource_name, audited_hash = RESOURCE_NAME, cc_hash
    data = path.read_bytes()
    resource = read_weights(data, resource_name=resource_name)
    resource_hash = _sha256(memoryview(data)[resource.offset : resource.offset + resource.size])
    if resource_hash != audited_hash:
        raise ValueError(
            f"unsupported {resource_name} resource {resource_hash}; expected audited {profile} "
            f"{audited_hash}. Use the original DLL, not an already repacked DLL."
        )
    records = build_records(profile)
    sizes = stage_bytes(profile)
    if set(resource.records) != {record.name for record in records}:
        raise ValueError("DLL record inventory does not match the canonical profile")
    blobs, stage_parts, entries = {}, {stage: [] for stage in sizes}, []
    offsets = dict.fromkeys(sizes, 0)
    for record in records:
        blob = bytes(resource.records[record.name].payload)
        if len(blob) != record.nbytes:
            raise ValueError(f"{record.name}: DLL record size differs from canonical profile")
        blobs[record.name] = blob
        stage_parts[record.stage].append(blob)
        entries.append(
            {
                "name": record.name,
                "block": record.block,
                "layer": record.layer,
                "parameter": record.parameter,
                "stage": record.stage,
                "stageOffset": offsets[record.stage],
                "byteLength": len(blob),
            }
        )
        offsets[record.stage] += len(blob)
    stages = {stage: b"".join(parts) for stage, parts in stage_parts.items()}
    hashes = {stage: _sha256(blob) for stage, blob in stages.items()}
    if {stage: len(blob) for stage, blob in stages.items()} != sizes:
        raise ValueError("DLL stage sizes differ from canonical profile")
    manifest = {
        "totals": {"blockCount": topology.block_count},
        "stages": [
            {"id": stage, "file": f"{stage}.bin", "packedByteLength": len(blob), "sha256": hashes[stage]}
            for stage, blob in stages.items()
        ],
        "tensors": entries,
        "dll": {
            "path": str(path.resolve()),
            "sha256": _sha256(data),
            "resource_sha256": resource_hash,
            "resource_name": resource_name,
            "resource_offset": resource.offset,
            "resource_bytes": resource.size,
        },
    }
    return data, resource, LoadedSource(path.parent.resolve(), manifest, stages, hashes, records, blobs, profile)


def unpack_dll(source_dll: str | Path, output_dir: str | Path, *, profile: str = PROFILE_ID, save_precision="native") -> dict:
    """Import into the existing training contract; never replace an existing directory."""
    source_path, output = Path(source_dll), Path(output_dir)
    _require_new(output)
    _, _, source = _load_template(source_path, profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dlssnr-unpack-", dir=output.parent) as temporary:
        staging = Path(temporary) / "model"
        report = convert_source(source, staging, verify=True, save_precision=save_precision)
        report.update(
            source_dll=source.manifest["dll"],
            output_dir=str(output.resolve()),
            resource_name=source.manifest["dll"]["resource_name"],
            training_ready=True,
        )
        (staging / "conversion_report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        if file_sha256(source_path) != source.manifest["dll"]["sha256"]:
            raise ValueError("source DLL changed during unpack")
        _require_new(output)
        staging.rename(output)
    return report


def _validate_canonical(model: Path, source: LoadedSource) -> dict:
    from musubi_tuner.dlssnr.artifacts import inspect_canonical

    identity = inspect_canonical(model)
    config = json.loads((model / "model_config.json").read_text(encoding="utf-8"))
    expected = canonical_config(source.records, profile=source.profile)
    for field in ("profile", "parameter_layout", "prior_layout", "tensors"):
        if config.get(field) != expected[field]:
            raise ValueError(f"canonical {field} differs from the supported layout")
    manifest = json.loads((model / "source_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("totals", {}).get("blockCount") != get_profile(source.profile).block_count:
        raise ValueError("canonical source block count does not match the template DLL")
    # Source filenames and extra extractor metadata are not weight identity.
    for section, fields in (
        ("stages", ("id", "packedByteLength", "sha256")),
        ("tensors", ("name", "stage", "stageOffset", "byteLength")),
    ):
        actual, expected_rows = manifest.get(section), source.manifest[section]
        if (
            not isinstance(actual, list)
            or len(actual) != len(expected_rows)
            or any(
                not isinstance(row, dict) or any(row.get(field) != expected_row[field] for field in fields)
                for row, expected_row in zip(actual, expected_rows)
            )
        ):
            raise ValueError(f"canonical source {section} do not match the template DLL")
    report = json.loads((model / "conversion_report.json").read_text(encoding="utf-8"))
    if report.get("stage_sha256") != source.stage_sha256:
        raise ValueError("canonical conversion source hashes do not match the template DLL")
    with safe_open(str(model / "opaque_records.safetensors"), framework="numpy") as handle:
        if set(handle.keys()) != set(source.blobs):
            raise ValueError("canonical opaque record inventory differs from the template DLL")
        for name, blob in source.blobs.items():
            value = handle.get_tensor(name)
            if value.dtype != np.uint8 or value.shape != (len(blob),) or value.tobytes() != blob:
                raise ValueError(f"{name}: canonical source record differs from the template DLL")
    return identity


def _pack_canonical(model: Path, source: LoadedSource, *, mix: float, strength: float):
    from musubi_tuner.dlssnr.storage import decode_stored_tensor, read_storage_policy
    from musubi_tuner.dlssnr.weight_quantization import native_storage_kinds

    identity = _validate_canonical(model, source)
    policy = read_storage_policy(model / "model.safetensors", source.profile)
    kinds = native_storage_kinds(source.profile)
    expected_names = set(canonical_names(source.records)) | {
        f"blocks.{block}.opaque.layer3" for block in get_profile(source.profile).vit_blocks
    }
    payloads, statistics = {}, []
    with safe_open(str(model / "model.safetensors"), framework="pt", device="cpu") as handle:
        names = set(handle.keys())
        if names != expected_names:
            raise ValueError(
                f"canonical tensor inventory mismatch; missing={sorted(expected_names - names)[:5]}, "
                f"unexpected={sorted(names - expected_names)[:5]}. Merge LoRA adapters before packing."
            )
        adapter = handle.get_tensor("blocks.0.input_adapter.weight").float().numpy()
        if adapter.shape != (32, 16) or np.any(adapter[:, 15] != 0):
            raise ValueError("input adapter lane 15 must remain zero")
        stage = None
        for record in source.records:
            if stage != record.stage:
                stage = record.stage
                logger.info("Packing %s", stage)
            keys = [view.name for region in record.regions for view in region.views]
            if any(region.kind == "opaque" for region in record.regions):
                keys.append(f"blocks.{record.block}.opaque.layer3")
            tensors = {name: decode_stored_tensor(name, handle.get_tensor(name), policy, kinds).numpy() for name in keys}
            payload, stats = repack_trained_record(record, source.blobs[record.name], tensors, mix=mix, strength=strength)
            payloads[record.name] = payload
            statistics.extend(stats)
    if file_sha256(model / "model.safetensors") != identity["model_sha256"]:
        raise ValueError("canonical weights changed during export")
    return payloads, statistics, identity


def _write_staged(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if file_sha256(path) != _sha256(data):
        raise OSError(f"staged output verification failed: {path}")


def _publish_new(staged: Path, destination: Path) -> None:
    # Hard links publish complete bytes without an overwrite race. Windows rename
    # also refuses existing destinations on filesystems without hard-link support.
    try:
        os.link(staged, destination)
    except OSError:
        if os.name != "nt":
            raise
        _require_new(destination)
        os.rename(staged, destination)


def pack_dll(
    template_dll: str | Path,
    model_dir: str | Path,
    output_dll: str | Path,
    *,
    mix: float = 1.0,
    strength: float = 1.0,
    merge_lora: str | Path | None = None,
    lora_multiplier: float = 1.0,
) -> dict:
    """Export a full/merged canonical checkpoint or merge one checked DLSS-NR LoRA first."""
    template, model, output = Path(template_dll), Path(model_dir), Path(output_dll)
    report_path = Path(str(output) + ".report.json")
    _require_new(output)
    _require_new(report_path)
    validate_multipliers(mix, strength)
    if not math.isfinite(lora_multiplier):
        raise ValueError("LoRA multiplier must be finite")
    if merge_lora is None and lora_multiplier != 1:
        raise ValueError("lora_multiplier requires merge_lora")
    from musubi_tuner.dlssnr.artifacts import read_canonical_profile

    original, resource, source = _load_template(template, read_canonical_profile(model))
    lora_info = None
    if merge_lora is not None:
        from musubi_tuner.networks.lora_dlssnr import merge_to_directory

        input_identity = _validate_canonical(model, source)
        adapter = Path(merge_lora)
        adapter_hash = file_sha256(adapter)
        with tempfile.TemporaryDirectory(prefix="dlssnr-merge-") as temporary:
            merged = Path(temporary) / "merged"
            # Mix/strength must run before the single native quantization in packing.
            merge_to_directory(model, adapter, merged, multiplier=lora_multiplier, save_precision="fp32")
            payloads, statistics, identity = _pack_canonical(merged, source, mix=mix, strength=strength)
        if file_sha256(adapter) != adapter_hash or file_sha256(model / "model.safetensors") != input_identity["model_sha256"]:
            raise ValueError("LoRA or base weights changed during export")
        lora_info = {
            "path": str(adapter.resolve()),
            "sha256": adapter_hash,
            "multiplier": lora_multiplier,
            "base_model_sha256": input_identity["model_sha256"],
            "merged_model_sha256": identity["model_sha256"],
        }
    else:
        payloads, statistics, identity = _pack_canonical(model, source, mix=mix, strength=strength)
    candidate = resource.replace(original, payloads)
    resource_name = source.manifest["dll"]["resource_name"]
    extracted = read_weights(candidate, resource_name=resource_name)
    for name, payload in payloads.items():
        if extracted.records[name].payload != payload:
            raise ValueError(f"{name}: DLL re-extraction differs from verified packed weights")
    if file_sha256(template) != source.manifest["dll"]["sha256"]:
        raise ValueError("template DLL changed during export")
    fields = (
        "values",
        "trained_changed_values",
        "blended_changed_values",
        "exported_changed_values",
        "rounded_values",
        "lost_update_values",
    )
    totals = {field: sum(row[field] for row in statistics) for field in fields}
    totals["max_abs_quantization_error"] = max(row["max_abs_quantization_error"] for row in statistics)
    report = {
        "schema": "dlssnr_dll_export_v1",
        "resource_name": resource_name,
        "profile": source.profile,
        "template_dll": str(template.resolve()),
        "template_sha256": _sha256(original),
        "model_dir": str(model.resolve()),
        "model_sha256": identity["model_sha256"],
        "output_dll": str(output.resolve()),
        "output_sha256": _sha256(candidate),
        "resource_sha256": _sha256(memoryview(candidate)[resource.offset : resource.offset + resource.size]),
        "resource_offset": resource.offset,
        "resource_bytes": resource.size,
        "mix": mix,
        "strength": strength,
        "lora": lora_info,
        "formula": "quantize_native(strength * (template + mix * (trained_or_merged - template)))",
        "rounding": "FP32 directly to native dtype, nearest ties to even; no rescaling or clipping",
        "whole_dll_byte_identical": candidate == original,
        "quantized_tensors_verified": True,
        "payload_only_modified": True,
        "native_export_validated": False,
        "native_load_test": "not_run",
        "totals": totals,
        "tensors": statistics,
        "limits": [
            "No native loader or image-quality certification.",
            "PE checksum and signature bytes are preserved, not updated; signature/load checks may reject changed DLLs.",
            "Canonical training layout is unchanged; MLX logical-v18 layout is not substituted.",
            "Opaque values are preserved and are not scaled by strength.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dlssnr-pack-", dir=output.parent) as temporary:
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
