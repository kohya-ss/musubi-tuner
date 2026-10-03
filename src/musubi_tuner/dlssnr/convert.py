"""Write a canonical FP32 checkpoint from a verified 310.8.0 source directory."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from musubi_tuner.dlssnr.checkpoint import (
    LoadedSource,
    logical_parameter_count,
    measure_fingerprint,
    repack_record,
    unpack_record,
)
from musubi_tuner.dlssnr.profiles import PROFILE_ID, block_channels, block_kind, build_records, canonical_names
from musubi_tuner.dlssnr.identity import file_sha256
from musubi_tuner.dlssnr.numerics import SURROGATE_FLAGS


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def verify_roundtrip(source: LoadedSource) -> None:
    for record in source.records:
        original = source.blobs[record.name]
        restored = repack_record(record, unpack_record(record, original))
        if restored != original:
            mismatch = next(index for index, (left, right) in enumerate(zip(restored, original)) if left != right)
            raise ValueError(f"{record.name}: repack mismatch at byte {mismatch}")


def canonical_config(records=None) -> dict:
    records = build_records() if records is None else records
    tensors = []
    for record in records:
        for region in record.regions:
            for view in region.views:
                if region.kind in ("e4", "f16frag"):
                    shape = [view.n1 - view.n0, view.k1 - view.k0]
                elif region.kind == "prior":
                    shape = [region.heads, 64, 64]
                else:
                    shape = [region.k]
                tensors.append(
                    {
                        "name": view.name,
                        "shape": shape,
                        "canonical_dtype": "float32",
                        "source_dtype": region.kind,
                        "group": view.group,
                        "record": record.name,
                        "offset": region.offset,
                        "k0": view.k0,
                        "k1": view.k1,
                        "n0": view.n0,
                        "n1": view.n1,
                    }
                )
    blocks = []
    seen: set[int] = set()
    for record in records:
        if record.block in seen:
            continue
        seen.add(record.block)
        blocks.append(
            {
                "block": record.block,
                "kind": block_kind(record.block),
                "channels": block_channels(record.block),
                "phase": record.phase,
                "stage": record.stage,
            }
        )
    return {
        "schema": "dlssnr_canonical_v1",
        "profile": PROFILE_ID,
        "parameter_layout": "pytorch_out_in",
        "prior_layout": "head_query_natural_key_natural",
        "blocks": blocks,
        "tensors": tensors,
        "frozen_input_lane": {"tensor": "blocks.0.input_adapter.weight", "in_feature": 15},
        "opaque_vit_layer3": [f"blocks.{block}.opaque.layer3" for block in range(31, 39)],
    }


def convert_source(source: LoadedSource, output_dir: str | Path, verify: bool) -> dict:
    if verify:
        verify_roundtrip(source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, np.ndarray] = {}
    opaque: dict[str, np.ndarray] = {}
    for record in source.records:
        unpacked = unpack_record(record, source.blobs[record.name])
        for name, value in unpacked.items():
            if name.startswith("opaque."):
                block = record.block
                if record.layer == 3 and record.stage == "vit":
                    tensors[f"blocks.{block}.opaque.layer3"] = value
                continue
            tensors[name] = value
        opaque[record.name] = np.frombuffer(source.blobs[record.name], dtype=np.uint8).copy()

    expected = canonical_names(source.records)
    missing = [name for name in expected if name not in tensors]
    if missing:
        raise RuntimeError(f"missing canonical tensors: {missing[:5]}")
    ordered = {name: tensors[name] for name in expected}
    for block in range(31, 39):
        key = f"blocks.{block}.opaque.layer3"
        ordered[key] = tensors[key]
    save_file(ordered, str(output / "model.safetensors"))
    save_file(opaque, str(output / "opaque_records.safetensors"))

    fingerprint = measure_fingerprint(source)
    report = {
        "schema": "dlssnr_conversion_report_v1",
        "profile": PROFILE_ID,
        "implementation_stage": "P0",
        "source_dir": str(source.directory),
        "stage_sha256": source.stage_sha256,
        "payload_bytes": sum(len(blob) for blob in source.stages.values()),
        "logical_parameters": logical_parameter_count(source.records),
        "roundtrip": "byte_identical" if verify else "not_run",
        "canonical_sha256": file_sha256(output / "model.safetensors"),
        "byte_roundtrip_verified": bool(verify),
        "layout_verified": False,
        "fingerprint": fingerprint,
        "trainer_commit": _git_commit(),
        "native_reference": "not_implemented",
        "train_surrogate": "implemented_unvalidated",
    }
    (output / "model_config.json").write_text(json.dumps(canonical_config(source.records), indent=2), encoding="utf-8")
    (output / "source_manifest.json").write_text(json.dumps(source.manifest, indent=2), encoding="utf-8")
    (output / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "preprocessing.json").write_text(
        json.dumps(
            {
                "schema": "dlssnr_preprocessing_v1",
                "implementation_stage": "P0",
                "executed": False,
                "centering": "f16(f16(f16(proxy) - 0.5) * 0.125)",
                "style_lane": "style_id / 128",
                "lane15": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "numerics.json").write_text(
        json.dumps(
            {
                "schema": "dlssnr_numerics_v1",
                "implementation_stage": "P0",
                "profiles": {
                    "native_reference": {"status": "not_implemented"},
                    "train_surrogate": {**SURROGATE_FLAGS, "status": "implemented_unvalidated"},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return report
