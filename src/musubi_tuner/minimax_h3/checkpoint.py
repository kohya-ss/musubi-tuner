from __future__ import annotations

from collections.abc import Callable, Iterable
import json
from pathlib import Path
from typing import TypeVar

from accelerate import init_empty_weights
import torch
import torch.nn as nn

from musubi_tuner.modules.convrot_int8_utils import (
    ConvRotInt8Artifact,
    canonicalize_convrot_int8_key,
    inspect_convrot_int8_artifact,
    prepare_convrot_int8_model,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


ModuleT = TypeVar("ModuleT", bound=nn.Module)


def resolve_safetensors_files(path: str | Path) -> list[Path]:
    path = Path(path).resolve()
    if path.is_file():
        if path.suffix != ".safetensors":
            raise ValueError(f"Expected a safetensors checkpoint, got {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"Checkpoint path does not exist: {path}")

    index_files = sorted(path.glob("*.safetensors.index.json"))
    if len(index_files) > 1:
        raise ValueError(f"Multiple safetensors indexes found in {path}: {index_files}")
    if index_files:
        try:
            index = json.loads(index_files[0].read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"Invalid safetensors index: {index_files[0]}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Safetensors index has no weights: {index_files[0]}")

        files = []
        for filename in sorted(set(weight_map.values())):
            if not isinstance(filename, str):
                raise ValueError(f"Invalid shard filename in {index_files[0]}: {filename!r}")
            shard = (path / filename).resolve()
            if not shard.is_relative_to(path) or not shard.is_file():
                raise ValueError(f"Missing or unsafe safetensors shard in {index_files[0]}: {filename}")
            files.append(shard)
        return files

    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise ValueError(f"No safetensors checkpoint found in {path}")
    if len(files) > 1:
        raise ValueError(f"Multiple safetensors files without an index found in {path}: {files}")
    return [files[0].resolve()]


def _normalize_checkpoint_key(key: str, prefixes: tuple[str, ...]) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _normalize_loaded_key(
    key: str,
    key_prefixes: tuple[str, ...],
    key_transform: Callable[[str], str] | None,
) -> str:
    key = _normalize_checkpoint_key(key, key_prefixes)
    if key_transform is not None:
        key = key_transform(key)
    return canonicalize_convrot_int8_key(key)


def inspect_safetensors_convrot_int8(
    files: Iterable[str | Path],
    *,
    key_prefixes: tuple[str, ...] = (),
    key_transform: Callable[[str], str] | None = None,
    disable_mmap: bool = False,
) -> ConvRotInt8Artifact | None:
    return inspect_convrot_int8_artifact(
        files,
        key_normalizer=lambda key: _normalize_loaded_key(key, key_prefixes, key_transform),
        disable_numpy_memmap=disable_mmap,
    )


def load_safetensors_metadata(files: Iterable[str | Path]) -> dict[str, str]:
    merged = {}
    for file in files:
        path = Path(file).resolve()
        with MemoryEfficientSafeOpen(str(path)) as handle:
            metadata = handle.metadata() or {}
        for key, value in metadata.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"Conflicting MiniMax-H3 checkpoint metadata {key!r} in {path}")
            merged[key] = value
    return merged


def _format_key_mismatch(
    missing: Iterable[str],
    unexpected: Iterable[str],
    shape_mismatches: Iterable[str],
    dtype_mismatches: Iterable[str],
) -> str:
    missing = sorted(missing)
    unexpected = sorted(unexpected)
    shape_mismatches = sorted(shape_mismatches)
    dtype_mismatches = sorted(dtype_mismatches)
    return (
        "MiniMax-H3 checkpoint key mismatch: "
        f"missing={missing[:20]}, unexpected={unexpected[:20]}, shape_mismatches={shape_mismatches[:20]}, "
        f"dtype_mismatches={dtype_mismatches[:20]}"
    )


def load_safetensors_module(
    factory: Callable[[], ModuleT],
    files: Iterable[str | Path],
    *,
    device: str | torch.device,
    dtype: torch.dtype | None,
    key_prefixes: tuple[str, ...] = (),
    key_transform: Callable[[str], str] | None = None,
    strict_dtype: bool = False,
    convrot_artifact: ConvRotInt8Artifact | None = None,
    convrot_bwd_mode: str = "bf16",
    disable_mmap: bool = False,
) -> ModuleT:
    files = [Path(path).resolve() for path in files]
    if not files:
        raise ValueError("MiniMax-H3 checkpoint file list is empty")

    with init_empty_weights():
        model = factory()
        if convrot_artifact is not None:
            prepare_convrot_int8_model(model, convrot_artifact, bwd_mode=convrot_bwd_mode)

    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    seen_normalized_keys = set()
    loaded_model_keys = set()
    consumed_control_keys = set()
    unexpected = set()
    shape_mismatches = []
    dtype_mismatches = []

    for path in files:
        shard = {}
        with MemoryEfficientSafeOpen(str(path), disable_numpy_memmap=disable_mmap) as handle:
            raw_keys = handle.keys()
            for raw_key in raw_keys:
                key = _normalize_loaded_key(raw_key, key_prefixes, key_transform)
                if convrot_artifact is None and key.endswith((".scale_weight", ".comfy_quant")):
                    raise ValueError(f"ConvRot INT8 tensors require artifact inspection before loading: {path}:{raw_key}")
                if key in seen_normalized_keys:
                    raise ValueError(f"Duplicate MiniMax-H3 checkpoint key {key!r} in {path}")
                seen_normalized_keys.add(key)

                if key.endswith(".comfy_quant"):
                    if key not in convrot_artifact.control_keys:
                        unexpected.add(key)
                    else:
                        consumed_control_keys.add(key)
                    continue
                if key not in expected_keys:
                    unexpected.add(key)
                    continue

                tensor = handle.get_tensor(raw_key, device=torch.device("cpu"), dtype=None)
                if tensor.shape != expected_state[key].shape:
                    shape_mismatches.append(f"{key}: expected {tuple(expected_state[key].shape)}, got {tuple(tensor.shape)}")
                    continue

                if convrot_artifact is not None:
                    if key in convrot_artifact.weight_keys:
                        if tensor.dtype is not torch.int8:
                            dtype_mismatches.append(f"{key}: expected torch.int8, got {tensor.dtype}")
                            continue
                    elif key in convrot_artifact.scale_keys:
                        if tensor.dtype is not torch.float32:
                            dtype_mismatches.append(f"{key}: expected torch.float32, got {tensor.dtype}")
                            continue
                    elif tensor.is_floating_point():
                        target_dtype = dtype if dtype is not None else expected_state[key].dtype
                        if target_dtype in {torch.float16, torch.bfloat16}:
                            if tensor.dtype not in {torch.float16, torch.bfloat16}:
                                dtype_mismatches.append(
                                    f"{key}: expected source torch.float16 or torch.bfloat16 for {target_dtype} compute, got {tensor.dtype}"
                                )
                                continue
                        elif target_dtype is torch.float32:
                            if tensor.dtype is not torch.float32:
                                dtype_mismatches.append(f"{key}: expected fixed torch.float32, got {tensor.dtype}")
                                continue
                        elif tensor.dtype is not target_dtype:
                            dtype_mismatches.append(f"{key}: expected {target_dtype}, got {tensor.dtype}")
                            continue
                        if tensor.dtype is not target_dtype:
                            tensor = tensor.to(dtype=target_dtype)
                    elif tensor.dtype != expected_state[key].dtype:
                        dtype_mismatches.append(f"{key}: expected {expected_state[key].dtype}, got {tensor.dtype}")
                        continue
                else:
                    if strict_dtype and tensor.dtype != expected_state[key].dtype:
                        dtype_mismatches.append(f"{key}: expected {expected_state[key].dtype}, got {tensor.dtype}")
                        continue
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype=dtype)

                shard[key] = tensor
                loaded_model_keys.add(key)
        if shard:
            model.load_state_dict(shard, strict=False, assign=True)

    missing = expected_keys - loaded_model_keys
    if convrot_artifact is not None:
        missing_controls = convrot_artifact.control_keys - consumed_control_keys
        if missing_controls:
            raise ValueError(f"MiniMax-H3 ConvRot control tensors were not consumed: {sorted(missing_controls)[:20]}")
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise ValueError(_format_key_mismatch(missing, unexpected, shape_mismatches, dtype_mismatches))

    model.to(device=torch.device(device))
    model.eval()
    return model
