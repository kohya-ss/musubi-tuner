from __future__ import annotations

from collections.abc import Iterable
import hashlib
from pathlib import Path

import torch

from musubi_tuner.minimax_h3.media import fingerprint_file
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, get_split_weight_filenames


def resolve_safetensors_files(path: str | Path) -> list[Path]:
    """Resolve a checkpoint path to its safetensors files.

    Repo convention: the path must be a ``.safetensors`` file; a
    ``...00001-of-00004.safetensors`` style first shard expands to all shards.
    """
    path = Path(path)
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"MiniMax-H3 checkpoints must be a .safetensors file (pass the first shard for split checkpoints): {path}")
    split_filenames = get_split_weight_filenames(str(path))
    if split_filenames is not None:
        return [Path(filename) for filename in split_filenames]
    return [path]


def fingerprint_checkpoint(path: str | Path) -> str:
    """Identity of a (possibly sharded) checkpoint for cache-staleness checks: shard names + file fingerprints."""
    files = resolve_safetensors_files(Path(path).resolve())
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprint_file(file).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


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


def strip_key_prefixes(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Strip the first matching prefix from every key (fixed provenance rules, e.g. ``vae.``)."""

    def strip(key: str) -> str:
        for prefix in prefixes:
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    stripped = {strip(key): value for key, value in state_dict.items()}
    if len(stripped) != len(state_dict):
        raise ValueError(f"MiniMax-H3 checkpoint keys collide after stripping prefixes {prefixes}")
    return stripped
