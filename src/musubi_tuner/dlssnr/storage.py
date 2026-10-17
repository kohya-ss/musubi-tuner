"""On-disk precision is independent of execution precision and frozen-base quantization."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors import safe_open

from musubi_tuner.dlssnr.profiles import get_profile
from musubi_tuner.dlssnr.weight_quantization import STORAGE_DTYPES, native_storage_kinds

STORAGE_SCHEMA = "dlssnr_weight_storage_v1"
STORAGE_METADATA_KEY = "dlssnr_weight_storage"


def storage_policy(profile: str, precision: str = "native") -> dict:
    get_profile(profile)
    if precision not in ("native", "fp32"):
        raise ValueError("weight storage precision must be native or fp32")
    return {"schema": STORAGE_SCHEMA, "profile": profile, "precision": precision}


def read_storage_policy(path: str | Path, profile: str) -> dict:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        header = (handle.metadata() or {}).get(STORAGE_METADATA_KEY)
    policy = json.loads(header) if header is not None else None
    config_path = path.parent / "model_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("profile") != profile:
            raise ValueError("weight storage profile differs from the model config")
        if config.get("weight_storage") != policy:
            raise ValueError("weight storage policy differs between model config and weight header")
    if policy is None:
        return storage_policy(profile, "fp32")
    if not isinstance(policy, dict) or policy != storage_policy(profile, policy.get("precision")):
        raise ValueError("invalid weight storage schema or profile")
    return policy


def decode_stored_tensor(name: str, value: torch.Tensor, policy: dict, kinds: dict) -> torch.Tensor:
    opaque = {f"blocks.{block}.opaque.layer3" for block in get_profile(policy["profile"]).vit_blocks}
    if name in opaque:
        if value.dtype != torch.uint8 or value.shape != (2,):
            raise ValueError(f"{name}: opaque storage must contain exactly two bytes")
        return value
    if policy["precision"] == "native":
        if name not in kinds:
            raise ValueError(f"{name}: unknown native storage tensor")
        expected = STORAGE_DTYPES[kinds[name]]
    else:
        expected = torch.float32
    if value.dtype != expected:
        raise ValueError(f"{name}: storage dtype must be {expected}, got {value.dtype}")
    decoded = value.float()
    if not torch.isfinite(decoded).all():
        raise ValueError(f"{name}: stored parameters must be finite")
    return decoded


def load_canonical_tensors(path: str | Path, profile: str) -> dict[str, torch.Tensor]:
    policy = read_storage_policy(path, profile)
    kinds = native_storage_kinds(profile)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {name: decode_stored_tensor(name, handle.get_tensor(name), policy, kinds) for name in handle.keys()}


@torch.no_grad()
def encode_canonical_tensors(tensors: dict[str, torch.Tensor], profile: str, precision: str = "native"):
    policy = storage_policy(profile, precision)
    kinds = native_storage_kinds(profile)
    opaque = {f"blocks.{block}.opaque.layer3" for block in get_profile(profile).vit_blocks}
    encoded = {}
    report = {"precision": precision, "values": 0, "rounded_values": 0, "stored_bytes": 0, "max_abs_error": 0.0}
    for name, original in tensors.items():
        value = original.detach().cpu().contiguous()
        if name in opaque:
            stored = decode_stored_tensor(name, value, policy, kinds)
        else:
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name}: canonical parameters must be finite FP32 before storage")
            if precision == "native" and name not in kinds:
                raise ValueError(f"{name}: unknown native storage tensor")
            dtype = STORAGE_DTYPES[kinds[name]] if precision == "native" else torch.float32
            if not (value.abs() <= torch.finfo(dtype).max).all():
                raise ValueError(f"{name}: parameters exceed the {dtype} storage range")
            stored = value.to(dtype)
            error = (stored.float() - value).abs()
            report["values"] += value.numel()
            report["rounded_values"] += torch.count_nonzero(error).item()
            if error.numel():
                report["max_abs_error"] = max(report["max_abs_error"], error.max().item())
        encoded[name] = stored
        report["stored_bytes"] += stored.numel() * stored.element_size()
    return encoded, policy, report
