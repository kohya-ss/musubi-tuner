"""Frozen FP8 projection storage with canonical FP32 materialization identities."""

from __future__ import annotations

import hashlib

import torch

from musubi_tuner.dlssnr.model import ChannelLinear


def canonical_tensor_sha256(tensors) -> str:
    """Hash a name-sorted stream without retaining every dequantized matrix."""
    digest = hashlib.sha256()
    for name, value in tensors:
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def iter_canonical_tensors(model):
    named = dict(model.named_parameters())
    linears = {f"{name}.weight": module for name, module in model.named_modules() if isinstance(module, ChannelLinear)}
    for name in sorted(named.keys() | linears.keys()):
        yield name, linears[name].materialized_weight() if name in linears else named[name]


def materialize_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(device="cpu", dtype=torch.float32).contiguous() for name, value in iter_canonical_tensors(model)
    }


def quantize_frozen_base(model, *, scaled: bool) -> dict:
    from musubi_tuner.modules.fp8_optimization_utils import quantize_weight

    selected = [
        (f"{name}.weight", module)
        for name, module in model.named_modules()
        if isinstance(module, ChannelLinear) and "input_adapter" not in name and ".head." not in f"{name}."
    ]
    if not selected:
        raise ValueError("FP8 found no eligible frozen NR projections")
    for name, module in selected:
        if module.weight.requires_grad:
            raise ValueError(f"{name}: FP8 storage requires a frozen base, not a trainable matrix")
        if module.weight.dtype != torch.float32:
            raise ValueError(f"{name}: quantize a canonical FP32 base exactly once")
        if not torch.isfinite(module.weight).all():
            raise ValueError(f"{name}: non-finite base weight")
        if not scaled and module.weight.abs().max() > 448:
            raise ValueError(f"{name}: unscaled FP8 overflow; use --fp8_scaled for this base")
    original = canonical_tensor_sha256(iter_canonical_tensors(model))
    pending = []
    # Validate the complete conversion before replacing any parameter.
    for name, module in selected:
        if scaled:
            mode = "block" if module.weight.shape[1] % 64 == 0 else "channel"
            quantized, scale = quantize_weight(name, module.weight.detach(), torch.float8_e4m3fn, 448.0, -448.0, mode, 64)
            published = quantized.float()
            if scale.ndim == 3:
                published = (published.reshape(scale.shape[0], scale.shape[1], 64) * scale).reshape_as(quantized)
            else:
                published = published * scale
        else:
            quantized, scale = module.weight.detach().to(torch.float8_e4m3fn), None
            published = quantized.float()
        if not torch.isfinite(published).all():
            raise ValueError(f"{name}: FP8 overflow or non-finite materialized weight")
        pending.append((name, module, quantized, scale))
    for _, module, quantized, scale in pending:
        # Frozen storage is a buffer: DDP must not broadcast unsupported FP8 parameters.
        del module.weight
        module.register_buffer("weight", quantized)
        if scale is not None:
            module.register_buffer("scale_weight", scale)
    report = {
        "schema": "dlssnr_fp8_base_v1",
        "dtype": "float8_e4m3fn",
        "scaled": scaled,
        "quantization_mode": "block64_with_channel_fallback" if scaled else "cast",
        "block_size": 64 if scaled else None,
        "source_base_sha256": original,
        "effective_base_sha256": canonical_tensor_sha256(iter_canonical_tensors(model)),
        "targets": sorted(name for name, _ in selected),
        "scales_sha256": canonical_tensor_sha256(sorted((name, scale) for name, _, _, scale in pending if scale is not None)),
    }
    model.base_quantization = report
    return report
