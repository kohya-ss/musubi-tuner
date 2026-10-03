"""Content identities shared by manifests, canonical artifacts and training state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_identity() -> dict:
    root = Path(__file__).parent
    names = (
        "geometry.py",
        "noise.py",
        "preprocess.py",
        "arithmetic.py",
        "numerics.py",
        "runtime.py",
        "attention.py",
        "fp8.py",
        "model.py",
        "pipeline.py",
        "temporal.py",
        "profiles.py",
    )
    return {name: file_sha256(root / name) for name in names}
