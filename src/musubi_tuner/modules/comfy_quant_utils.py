"""Shared helpers for ComfyUI ``.comfy_quant`` pre-quantized checkpoint specs.

Each quantized module in a ComfyUI checkpoint carries a tiny sibling tensor
``<module>.comfy_quant`` holding the uint8 bytes of a JSON object that declares the
module's quantization format. This module only decodes the JSON and classifies
formats so callers can route a checkpoint to the right loader; format-specific
validation and conversion live with the loaders (``convrot_int8_utils`` for ConvRot
INT8, ``nvfp4_utils`` for NVFP4 + per-row INT8 embeddings).
"""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Set, Union

import torch

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

COMFY_QUANT_SUFFIX = ".comfy_quant"
COMFY_WEIGHT_SCALE_SUFFIX = ".weight_scale"

# canonical labels returned by classify_comfy_quant_spec / detect_comfy_quant_formats
FORMAT_CONVROT_INT8 = "convrot_int8"
FORMAT_INT8_TENSORWISE = "int8_tensorwise"
FORMAT_NVFP4 = "nvfp4"


def decode_comfy_quant_spec(key: str, tensor: torch.Tensor) -> dict:
    """Decode a ``.comfy_quant`` tensor (uint8 bytes of JSON) into a dict, format-agnostic."""
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"Invalid comfy_quant tensor for {key}: expected 1D uint8, got {tensor.dtype} ndim={tensor.ndim}")
    try:
        spec = json.loads(bytes(tensor.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid comfy_quant JSON for {key}: {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid comfy_quant spec for {key}: expected a JSON object, got {type(spec).__name__}")
    return spec


def classify_comfy_quant_spec(spec: dict) -> str:
    """Map a decoded spec to a canonical format label (unknown formats pass through as-is)."""
    quant_format = spec.get("format")
    if quant_format == FORMAT_INT8_TENSORWISE and spec.get("convrot"):
        return FORMAT_CONVROT_INT8
    return str(quant_format)


def detect_comfy_quant_formats(files: Iterable[Union[str, Path]], *, disable_numpy_memmap: bool = False) -> Set[str]:
    """Collect the canonical format labels a checkpoint declares (empty set: not pre-quantized).

    Routing only — the spec tensors are tiny, so reading them here is cheap; per-format
    validation happens again inside the selected loader.
    """
    formats: Set[str] = set()
    for file in files:
        with MemoryEfficientSafeOpen(str(file), disable_numpy_memmap=disable_numpy_memmap) as f:
            for key in f.keys():
                if key.endswith(COMFY_QUANT_SUFFIX):
                    formats.add(classify_comfy_quant_spec(decode_comfy_quant_spec(key, f.get_tensor(key))))
    return formats
