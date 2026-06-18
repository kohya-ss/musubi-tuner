from __future__ import annotations

import os
import re
from typing import Any

import torch
from safetensors.torch import load_file, safe_open

from musubi_tuner.dataset.architectures import ARCHITECTURE_LENS_FULL
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.model_utils import dtype_to_str

LENS_TEXT_CACHE_PRECISIONS = ("auto", "bf16", "fp16", "fp32", "fp8", "nvfp4")
NVFP4_BLOCK_SIZE = 16
NVFP4_FP4_MAX = 6.0
NVFP4_FP8_MAX = 448.0
NVFP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def normalize_lens_cache_precision(value: str | None) -> str:
    if value is None:
        return "auto"
    value = value.lower()
    aliases = {
        "bfloat16": "bf16",
        "float16": "fp16",
        "float32": "fp32",
        "float": "fp32",
        "float8": "fp8",
        "float8_e4m3fn": "fp8",
        "fp8_e4m3fn": "fp8",
    }
    value = aliases.get(value, value)
    if value not in LENS_TEXT_CACHE_PRECISIONS:
        raise ValueError(f"Unsupported Lens text encoder cache precision: {value}")
    if value in ("fp8", "nvfp4") and getattr(torch, "float8_e4m3fn", None) is None:
        raise ValueError(f"Lens {value} text cache requires torch.float8_e4m3fn support.")
    return value


def save_lens_text_cache(item_info: Any, embeds: list[torch.Tensor], precision: str | None = "auto") -> None:
    precision = normalize_lens_cache_precision(precision)
    if not embeds:
        raise ValueError("embeds should not be empty")

    sd: dict[str, torch.Tensor] = {}
    metadata = {
        "architecture": ARCHITECTURE_LENS_FULL,
        "caption1": item_info.caption,
        "format_version": "1.0.1",
    }

    if precision == "nvfp4":
        metadata["lens_text_cache_precision"] = "nvfp4"
        for i, embed in enumerate(embeds):
            _validate_lens_embed(embed)
            base_key = f"varlen_lens_ctx_{i}"
            packed, block_scale, global_scale, shape = _quantize_nvfp4(embed)
            sd[f"{base_key}_nvfp4_packed"] = packed
            sd[f"{base_key}_nvfp4_block_scale"] = block_scale
            sd[f"{base_key}_nvfp4_global_scale"] = global_scale
            metadata[f"{base_key}_nvfp4_shape"] = ",".join(str(dim) for dim in shape)
    else:
        if precision != "auto":
            metadata["lens_text_cache_precision"] = precision
        for i, embed in enumerate(embeds):
            _validate_lens_embed(embed)
            cached = _convert_cache_tensor(embed, precision)
            dtype_str = dtype_to_str(cached.dtype)
            sd[f"varlen_lens_ctx_{i}_{dtype_str}"] = cached.detach().cpu()

    _replace_nan_with_zero(sd, item_info.text_encoder_output_cache_path)
    os.makedirs(os.path.dirname(item_info.text_encoder_output_cache_path), exist_ok=True)
    safetensors_utils.mem_eff_save_file(sd, item_info.text_encoder_output_cache_path, metadata=metadata)


def load_lens_text_cache(path: str) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
        keys = list(f.keys())

    if metadata.get("lens_text_cache_precision") == "nvfp4" or any(key.endswith("_nvfp4_packed") for key in keys):
        return _load_nvfp4_lens_text_cache(path, metadata, keys)

    return load_file(path)


def _validate_lens_embed(embed: torch.Tensor) -> None:
    if embed.dim() != 2:
        raise ValueError(f"Lens text embed should be 2D tensor (feature, hidden_size), got {embed.shape}")


def _convert_cache_tensor(embed: torch.Tensor, precision: str) -> torch.Tensor:
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
        "fp8": torch.float8_e4m3fn,
    }
    if precision == "auto":
        return embed.detach().cpu()
    return embed.detach().to(dtype_map[precision]).cpu()


def _replace_nan_with_zero(sd: dict[str, torch.Tensor], path: str) -> None:
    for key, value in sd.items():
        if value.is_floating_point() and torch.isnan(value.float()).any():
            value = value.clone()
            value[torch.isnan(value.float())] = 0
            sd[key] = value


def _quantize_nvfp4(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, ...]]:
    original_shape = tuple(tensor.shape)
    flat = tensor.detach().to(dtype=torch.float32, device="cpu").flatten()
    if flat.numel() == 0:
        packed = torch.empty((0, NVFP4_BLOCK_SIZE // 2), dtype=torch.uint8)
        block_scale = torch.empty((0,), dtype=torch.float8_e4m3fn)
        global_scale = torch.tensor([1.0], dtype=torch.float32)
        return packed, block_scale, global_scale, original_shape

    pad_len = (-flat.numel()) % NVFP4_BLOCK_SIZE
    if pad_len:
        flat = torch.nn.functional.pad(flat, (0, pad_len))
    blocks = flat.reshape(-1, NVFP4_BLOCK_SIZE)

    global_amax = blocks.abs().max()
    if global_amax == 0:
        global_scale = torch.tensor([1.0], dtype=torch.float32)
        block_scale = torch.ones(blocks.shape[0], dtype=torch.float8_e4m3fn)
        indices = torch.zeros_like(blocks, dtype=torch.uint8)
    else:
        global_scale_value = torch.clamp(global_amax / (NVFP4_FP8_MAX * NVFP4_FP4_MAX), min=1e-12)
        block_amax = blocks.abs().max(dim=1).values
        block_scale_f32 = torch.clamp((block_amax / NVFP4_FP4_MAX) / global_scale_value, min=1e-8, max=NVFP4_FP8_MAX)
        block_scale = block_scale_f32.to(torch.float8_e4m3fn)
        dequant_scale = block_scale.to(torch.float32)[:, None] * global_scale_value
        normalized = blocks / dequant_scale.clamp_min(1e-12)
        lut = torch.tensor(NVFP4_E2M1_VALUES, dtype=torch.float32)
        indices = torch.argmin((normalized.unsqueeze(-1) - lut).abs(), dim=-1).to(torch.uint8)
        global_scale = global_scale_value.reshape(1).to(torch.float32)

    packed = (indices[:, 0::2] << 4) | indices[:, 1::2]
    return packed.contiguous(), block_scale.contiguous(), global_scale.contiguous(), original_shape


def _load_nvfp4_lens_text_cache(path: str, metadata: dict[str, str], keys: list[str]) -> dict[str, torch.Tensor]:
    sd = load_file(path)
    output: dict[str, torch.Tensor] = {}
    packed_pattern = re.compile(r"^varlen_lens_ctx_(\d+)_nvfp4_packed$")
    packed_keys = sorted((key for key in keys if packed_pattern.match(key)), key=lambda key: int(packed_pattern.match(key).group(1)))

    for packed_key in packed_keys:
        match = packed_pattern.match(packed_key)
        assert match is not None
        idx = int(match.group(1))
        base_key = f"varlen_lens_ctx_{idx}"
        shape_key = f"{base_key}_nvfp4_shape"
        if shape_key not in metadata:
            raise ValueError(f"Missing Lens NVFP4 shape metadata: {shape_key}")
        shape = tuple(int(dim) for dim in metadata[shape_key].split(","))
        block_scale_key = f"{base_key}_nvfp4_block_scale"
        global_scale_key = f"{base_key}_nvfp4_global_scale"
        if block_scale_key not in sd or global_scale_key not in sd:
            raise ValueError(f"Missing Lens NVFP4 scale tensors for {base_key}")

        tensor = _dequantize_nvfp4(sd[packed_key], sd[block_scale_key], sd[global_scale_key], shape)
        output[f"{base_key}_bfloat16"] = tensor

    return output


def _dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        raise TypeError(f"Lens NVFP4 packed tensor must be uint8, got {packed.dtype}")
    if packed.ndim != 2 or packed.shape[1] != NVFP4_BLOCK_SIZE // 2:
        raise ValueError(f"Lens NVFP4 packed tensor must have shape [blocks, 8], got {tuple(packed.shape)}")

    high = packed >> 4
    low = packed & 0x0F
    indices = torch.stack((high, low), dim=-1).reshape(packed.shape[0], NVFP4_BLOCK_SIZE).to(torch.long)
    lut = torch.tensor(NVFP4_E2M1_VALUES, dtype=torch.float32)
    values = lut[indices]
    dequant = values * block_scale.to(torch.float32)[:, None] * global_scale.to(torch.float32).reshape(1)
    numel = 1
    for dim in shape:
        numel *= dim
    return dequant.flatten()[:numel].reshape(shape).to(torch.bfloat16)
