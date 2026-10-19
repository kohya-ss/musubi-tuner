"""Canonical DLSS-NR weights to native storage, without changing the training graph."""

from __future__ import annotations

import math

import numpy as np

from musubi_tuner.dlssnr.checkpoint import repack_record, unpack_record
from musubi_tuner.dlssnr.packing import decode_e4m3
from musubi_tuner.dlssnr.profiles import RecordLayout


_E4_VALUES = decode_e4m3(np.arange(127, dtype=np.uint8))
_E4_MIDPOINTS = (_E4_VALUES[:-1] + _E4_VALUES[1:]) * np.float32(0.5)


def quantize_tensor(values: np.ndarray, kind: str) -> np.ndarray:
    """Direct FP32 -> native round-to-nearest-even; never rescale or silently clip."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError("native weights must be finite")
    magnitude = np.abs(values)
    if kind == "e4":
        if np.any(magnitude > 448):
            raise ValueError("weight is outside the E4M3FN range [-448, 448]")
        index = np.searchsorted(_E4_MIDPOINTS, magnitude)
        tie = magnitude == _E4_MIDPOINTS[np.minimum(index, len(_E4_MIDPOINTS) - 1)]
        index += tie & ((index & 1) != 0)
        return np.copysign(_E4_VALUES[index], values)
    if kind in ("f16", "f16frag", "prior"):
        if np.any(magnitude > 65504):
            raise ValueError("weight is outside the FP16 finite range [-65504, 65504]")
        return values.astype(np.float16).astype(np.float32)
    if kind == "f32":
        return values
    raise ValueError(f"unsupported native tensor kind {kind}")


def validate_multipliers(mix: float, strength: float) -> None:
    if not math.isfinite(mix) or not 0 <= mix <= 1:
        raise ValueError("mix must be finite and in [0, 1]")
    if not math.isfinite(strength):
        raise ValueError("strength must be finite")


def _same_bits(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return left.view(np.uint32) == right.view(np.uint32)


def quantization_statistics(name, kind, source, trained, blended, quantized) -> dict:
    """Shared byte-level accounting for DLL export and training diagnostics."""
    unchanged = _same_bits(trained, source)
    error = quantized.astype(np.float64) - blended
    blended_changed = ~_same_bits(blended, source)
    exported_changed = ~_same_bits(quantized, source)
    return {
        "name": name,
        "storage": kind,
        "values": int(trained.size),
        "trained_changed_values": int(np.count_nonzero(~unchanged)),
        "blended_changed_values": int(np.count_nonzero(blended_changed)),
        "exported_changed_values": int(np.count_nonzero(exported_changed)),
        "rounded_values": int(np.count_nonzero(~_same_bits(quantized, blended))),
        "lost_update_values": int(np.count_nonzero(blended_changed & ~exported_changed)),
        "max_abs_quantization_error": float(np.max(np.abs(error))),
        "mean_abs_quantization_error": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def repack_trained_record(
    record: RecordLayout, original: bytes, tensors: dict[str, np.ndarray], *, mix: float = 1.0, strength: float = 1.0
) -> tuple[bytes, list[dict]]:
    """Mix, scale, quantize, and verify one record, preserving opaque bytes and signed zeros."""
    validate_multipliers(mix, strength)
    base = unpack_record(record, original)
    if repack_record(record, base) != original:
        raise ValueError(f"{record.name}: template does not pass canonical byte round-trip")
    kinds = {view.name: region.kind for region in record.regions for view in region.views}
    prepared, statistics = {}, []
    for name, source in base.items():
        canonical_name = f"blocks.{record.block}.opaque.layer3" if name.startswith("opaque.") else name
        if canonical_name not in tensors:
            raise ValueError(f"missing canonical tensor {canonical_name}")
        value = np.asarray(tensors[canonical_name])
        if value.shape != source.shape or value.dtype != source.dtype:
            raise ValueError(f"{canonical_name}: expected {source.shape} {source.dtype}, got {value.shape} {value.dtype}")
        if name.startswith("opaque."):
            if not np.array_equal(value, source):
                raise ValueError(f"{canonical_name}: opaque bytes must match the template")
            prepared[name] = source
            continue
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name}: weights must be finite")
        unchanged = _same_bits(value, source)
        if mix == 0:
            blended = source.copy()
        elif mix == 1:
            blended = value.copy()
        else:
            blended = (source.astype(np.float64) + mix * (value.astype(np.float64) - source)).astype(np.float32)
            blended[unchanged] = source[unchanged]
        if strength != 1:
            with np.errstate(over="ignore", invalid="ignore"):
                blended = (blended.astype(np.float64) * strength).astype(np.float32)
        try:
            quantized = quantize_tensor(blended, kinds[name])
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        prepared[name] = quantized
        statistics.append(quantization_statistics(name, kinds[name], source, value, blended, quantized))
    payload = repack_record(record, prepared)
    restored = unpack_record(record, payload)
    for name, value in prepared.items():
        if restored[name].tobytes() != value.tobytes():
            raise ValueError(f"{name}: quantized payload failed decode verification")
    return payload, statistics
