# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Load a DLSS-NR 310.8.0 model directory and convert it to canonical FP32 tensors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from musubi_tuner.dlssnr.packing import (
    decode_e4m3,
    encode_e4m3,
    pack_e4_codes,
    pack_f16_matrix,
    pack_prior,
    packed_f16_index,
    unpack_e4_codes,
    unpack_f16_matrix,
    unpack_prior,
)
from musubi_tuner.dlssnr.profiles import (
    PROFILE_ID,
    SOURCE_FINGERPRINT,
    STAGE_BYTES,
    RecordLayout,
    Region,
    build_records,
)


@dataclass
class LoadedSource:
    directory: Path
    manifest: dict
    stages: dict[str, bytes]
    stage_sha256: dict[str, str]
    records: tuple[RecordLayout, ...]
    blobs: dict[str, bytes]


def _reject_path(root: Path, relative: str) -> Path:
    if relative.startswith(("/", "\\")) or ":" in relative or ".." in Path(relative).parts:
        raise ValueError(f"manifest path escapes the source directory: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"manifest path escapes the source directory: {relative}")
    return path


def load_source(directory: str | Path) -> LoadedSource:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest.json in {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest["totals"]["blockCount"]) != 71:
        raise ValueError(f"profile {PROFILE_ID} requires 71 blocks, manifest says {manifest['totals']['blockCount']}")

    stages: dict[str, bytes] = {}
    hashes: dict[str, str] = {}
    seen_ids: list[str] = []
    for stage in manifest["stages"]:
        stage_id = stage["id"]
        seen_ids.append(stage_id)
        path = _reject_path(root, str(Path("model") / stage["file"]))
        blob = path.read_bytes()
        if len(blob) != int(stage["packedByteLength"]):
            raise ValueError(f"{stage_id}: file is {len(blob)} bytes, manifest says {stage['packedByteLength']}")
        if stage_id in STAGE_BYTES and len(blob) != STAGE_BYTES[stage_id]:
            raise ValueError(f"{stage_id}: expected {STAGE_BYTES[stage_id]} bytes for {PROFILE_ID}, got {len(blob)}")
        digest = hashlib.sha256(blob).hexdigest()
        if digest != stage["sha256"]:
            raise ValueError(f"{stage_id}: sha256 {digest} != manifest {stage['sha256']}")
        stages[stage_id] = blob
        hashes[stage_id] = digest
    if seen_ids != list(STAGE_BYTES):
        raise ValueError(f"stage order/ids {seen_ids} != {list(STAGE_BYTES)}")

    records = build_records()
    entries = manifest["tensors"]
    if len(entries) != len(records):
        raise ValueError(f"manifest has {len(entries)} tensors, profile has {len(records)}")
    blobs: dict[str, bytes] = {}
    cursor: dict[str, int] = {stage_id: 0 for stage_id in stages}
    for entry, record in zip(entries, records):
        if entry["name"] != record.name or entry["stage"] != record.stage:
            raise ValueError(f"manifest record {entry['name']} @ {entry['stage']} != profile {record.name} @ {record.stage}")
        if int(entry["byteLength"]) != record.nbytes:
            raise ValueError(f"{record.name}: manifest length {entry['byteLength']} != profile {record.nbytes}")
        if int(entry["stageOffset"]) != cursor[record.stage]:
            raise ValueError(f"{record.name}: stage offset {entry['stageOffset']} != contiguous {cursor[record.stage]}")
        start = int(entry["stageOffset"])
        stop = start + record.nbytes
        blobs[record.name] = stages[record.stage][start:stop]
        cursor[record.stage] = stop
    for stage_id, end in cursor.items():
        if end != len(stages[stage_id]):
            raise ValueError(f"{stage_id}: records cover {end} bytes, file is {len(stages[stage_id])}")
    return LoadedSource(root, manifest, stages, hashes, records, blobs)


def _vector(blob: bytes, dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(bytes(blob), dtype=dtype).astype(np.float32)


def unpack_record(record: RecordLayout, blob: bytes) -> dict[str, np.ndarray]:
    if len(blob) != record.nbytes:
        raise ValueError(f"{record.name}: got {len(blob)} bytes, layout is {record.nbytes}")
    tensors: dict[str, np.ndarray] = {}
    for region in record.regions:
        piece = blob[region.offset : region.offset + region.nbytes]
        if region.kind == "pad":
            if any(piece):
                raise ValueError(f"{record.name}: pad at {region.offset} is not zero")
            continue
        if region.kind == "opaque":
            tensors[f"opaque.{record.name}"] = np.frombuffer(piece, dtype=np.uint8).copy()
            continue
        if region.kind == "e4":
            values = decode_e4m3(unpack_e4_codes(piece, region.k, region.n))
            _take_matrix_views(tensors, region, values)
        elif region.kind == "f16frag":
            _take_matrix_views(tensors, region, unpack_f16_matrix(piece, region.k, region.n))
        elif region.kind == "f16":
            tensors[region.views[0].name] = _vector(piece, np.dtype("<f2"))
        elif region.kind == "f32":
            tensors[region.views[0].name] = _vector(piece, np.dtype("<f4"))
        elif region.kind == "prior":
            tensors[region.views[0].name] = unpack_prior(piece, region.heads)
        else:
            raise ValueError(f"unknown region {region.kind}")
    return tensors


def _take_matrix_views(tensors: dict[str, np.ndarray], region: Region, values_kn: np.ndarray) -> None:
    for view in region.views:
        tensors[view.name] = np.ascontiguousarray(values_kn[view.k0 : view.k1, view.n0 : view.n1].T)


def _place_views(region: Region, tensors: dict[str, np.ndarray]) -> np.ndarray:
    full = np.zeros((region.k, region.n), dtype=np.float32)
    for view in region.views:
        full[view.k0 : view.k1, view.n0 : view.n1] = tensors[view.name].T
    return full


def repack_record(record: RecordLayout, tensors: dict[str, np.ndarray]) -> bytes:
    out = bytearray(record.nbytes)
    for region in record.regions:
        if region.kind == "pad":
            continue
        dest = memoryview(out)[region.offset : region.offset + region.nbytes]
        if region.kind == "opaque":
            dest[:] = tensors[f"opaque.{record.name}"].tobytes()
        elif region.kind == "e4":
            dest[:] = pack_e4_codes(encode_e4m3(_place_views(region, tensors)))
        elif region.kind == "f16frag":
            packed = pack_f16_matrix(_place_views(region, tensors))
            if len(packed) != region.nbytes:
                raise ValueError(f"{record.name}: f16 fragment packed {len(packed)} != {region.nbytes}")
            dest[:] = packed
        elif region.kind == "f16":
            values = np.ascontiguousarray(tensors[region.views[0].name], dtype=np.float32)
            dest[:] = values.astype(np.float16).view(np.uint16).view(np.uint8).tobytes()
        elif region.kind == "f32":
            values = np.ascontiguousarray(tensors[region.views[0].name], dtype=np.float32)
            dest[:] = values.view(np.uint8).tobytes()
        elif region.kind == "prior":
            dest[:] = pack_prior(tensors[region.views[0].name])
        else:
            raise ValueError(f"unknown region {region.kind}")
    return bytes(out)


def _e4_code_stats(codes: np.ndarray) -> dict[str, float]:
    hist = np.bincount(codes.reshape(-1), minlength=256)
    values = np.abs(decode_e4m3(np.arange(256, dtype=np.uint8)))
    finite = np.isfinite(values)

    def count_at_least(threshold: float) -> int:
        return int(hist[finite & (values >= threshold - 1e-8)].sum())

    present = np.flatnonzero(hist)
    finite_codes = [int(code) for code in present.tolist() if (int(code) & 0x7F) != 0x7F]
    return {
        "count": int(codes.size),
        "nan": int(hist[(np.arange(256) & 0x7F) == 0x7F].sum()),
        "pos_zero": int(hist[0x00]),
        "neg_zero": int(hist[0x80]),
        "max_abs": float(max((values[code] for code in finite_codes), default=0.0)),
        "ge_0_125": count_at_least(0.125),
        "ge_0_5": count_at_least(0.5),
    }


def measure_fingerprint(source: LoadedSource) -> dict[str, float | int]:
    e4_parts: list[np.ndarray] = []
    pad_bytes = 0
    prior_values: list[np.ndarray] = []
    temperatures: list[np.ndarray] = []
    from_vit_max = 0.0
    for record in source.records:
        blob = source.blobs[record.name]
        for region in record.regions:
            piece = blob[region.offset : region.offset + region.nbytes]
            if region.kind == "pad":
                pad_bytes += region.nbytes
                if any(piece):
                    raise ValueError(f"{record.name}: nonzero pad at {region.offset}")
            elif region.kind == "e4":
                codes = unpack_e4_codes(piece, region.k, region.n)
                e4_parts.append(codes.reshape(-1))
                if record.name == "block39.layer0.layer" and region.offset == 0:
                    from_vit_max = float(np.max(np.abs(decode_e4m3(codes))))
            elif region.kind == "prior":
                prior_values.append(unpack_prior(piece, region.heads).reshape(-1))
            elif region.kind == "f32":
                temperatures.append(np.frombuffer(piece, dtype="<f4").copy())
    e4 = _e4_code_stats(np.concatenate(e4_parts))
    prior = np.concatenate(prior_values)
    temperature = np.concatenate(temperatures)
    return {
        "e4_count": e4["count"],
        "e4_nan": e4["nan"],
        "e4_pos_zero": e4["pos_zero"],
        "e4_neg_zero": e4["neg_zero"],
        "e4_max_abs": e4["max_abs"],
        "e4_abs_ge_0_125": e4["ge_0_125"],
        "e4_abs_ge_0_5": e4["ge_0_5"],
        "from_vit_max_abs": from_vit_max,
        "prior_count": int(prior.size),
        "prior_min": float(prior.min()),
        "prior_max": float(prior.max()),
        "prior_zeros": int(np.count_nonzero(prior == 0.0)),
        "temperature_count": int(temperature.size),
        "temperature_min": float(temperature.min()),
        "temperature_max": float(temperature.max()),
        "pad_bytes": pad_bytes,
    }


def assert_source_fingerprint(measured: dict) -> None:
    expected = SOURCE_FINGERPRINT
    for key in (
        "e4_count",
        "e4_nan",
        "e4_pos_zero",
        "e4_neg_zero",
        "e4_abs_ge_0_125",
        "e4_abs_ge_0_5",
        "prior_count",
        "prior_zeros",
        "temperature_count",
        "pad_bytes",
    ):
        if int(measured[key]) != int(expected[key]):
            raise ValueError(f"fingerprint {key}: measured {measured[key]} != {expected[key]}")
    for key in ("e4_max_abs", "from_vit_max_abs", "prior_min", "prior_max"):
        if float(measured[key]) != float(expected[key]):
            raise ValueError(f"fingerprint {key}: measured {measured[key]} != {expected[key]}")
    for key in ("temperature_min", "temperature_max"):
        if abs(float(measured[key]) - float(expected[key])) > 1e-4:
            raise ValueError(f"fingerprint {key}: measured {measured[key]} != {expected[key]}")


def assert_auxiliary_fingerprint(source: LoadedSource) -> None:
    """Checks that are not a raw code histogram: adapter lane 15, head padding, blend scale."""
    by_name = {record.name: record for record in source.records}
    adapter_record = by_name["block0.layer0.layer"]
    adapter = unpack_record(adapter_record, source.blobs[adapter_record.name])["blocks.0.input_adapter.weight"]
    if adapter.shape != (32, 16):
        raise ValueError(f"input adapter shape {adapter.shape}")
    if int(np.count_nonzero(adapter[:, 15] == 0.0)) != 32 or int(np.count_nonzero(adapter == 0.0)) != 32:
        raise ValueError("input adapter lane 15 is not the only all-zero input column")

    head_record = by_name["block70.layer0.layer"]
    head_region = next(region for region in head_record.regions if region.kind == "f16frag")
    head_blob = source.blobs[head_record.name][head_region.offset : head_region.offset + head_region.nbytes]
    halves = np.frombuffer(head_blob, dtype="<u2")
    rows = np.arange(32, dtype=np.int32)
    cols = np.arange(4, dtype=np.int32)
    live = packed_f16_index(rows[:, None], cols[None, :], 4).reshape(-1)
    unused = np.ones(halves.size, dtype=bool)
    unused[live] = False
    if int(unused.sum()) != SOURCE_FINGERPRINT["head_unused_halves"] or np.any(halves[unused] != 0):
        raise ValueError("head fragment padding is not 384 zero halves")

    blend = np.frombuffer(source.blobs["block70.layer0.blend_scale"], dtype="<f2")
    expected = np.float32(SOURCE_FINGERPRINT["blend_scale"]).astype(np.float16)
    if blend.shape != (1,) or blend[0] != expected:
        raise ValueError(f"blend_scale {float(blend[0])} != {float(expected)}")


def logical_parameter_count(records: tuple[RecordLayout, ...] | None = None) -> int:
    records = records if records is not None else build_records()
    return sum(region.logical_count() for record in records for region in record.regions)
