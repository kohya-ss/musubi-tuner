"""Optional standard-softmax cores for the explicitly experimental NR profile."""

from __future__ import annotations

import importlib
from importlib import metadata

import torch
import torch.nn.functional as F

OPTIONAL_BACKENDS = {
    "flash_attn": ("flash_attn", "flash_attn_func", "flash-attn"),
    "xformers": ("xformers.ops", "memory_efficient_attention", "xformers"),
    "sage_attn": ("sageattention", "sageattn", "sageattention"),
}


def load_backend(backend):
    if backend == "sdpa":
        return F.scaled_dot_product_attention
    if backend not in OPTIONAL_BACKENDS:
        raise ValueError(f"unknown experimental attention backend {backend}")
    module, function, _ = OPTIONAL_BACKENDS[backend]
    try:
        return getattr(importlib.import_module(module), function)
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise RuntimeError(f"{backend} is unavailable or has an incompatible extension ABI: {error}") from error


def attention(query, key, value, *, backend, prior=None, compute_dtype=None):
    """[..., heads, queries/keys, 32] -> FP32, with no extra inverse-sqrt scale."""
    if backend == "sage_attn" and torch.is_grad_enabled() and any(item.requires_grad for item in (query, key, value)):
        raise RuntimeError("SageAttention is inference-only for NR; no verified genuine backward is available")
    if backend in ("flash_attn", "sage_attn") and prior is not None:
        raise ValueError(f"{backend} cannot discard NR window priors; choose attention_scope global")
    dtype = compute_dtype or query.dtype
    if backend in OPTIONAL_BACKENDS and query.device.type != "cuda":
        raise RuntimeError(f"{backend} requires CUDA")
    if backend in ("flash_attn", "sage_attn") and dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"{backend} requires fp16 or bf16 compute")
    shape = query.shape
    heads, length, width = shape[-3:]
    query = query.reshape(-1, heads, length, width).to(dtype).contiguous()
    key = key.reshape(-1, heads, key.shape[-2], width).to(dtype).contiguous()
    value = value.reshape(-1, heads, value.shape[-2], width).to(dtype).contiguous()
    bias = None if prior is None else prior.to(dtype).expand(query.shape[0], heads, length, key.shape[-2])
    function = load_backend(backend)
    with torch.autocast(query.device.type, enabled=False):
        if backend == "sdpa":
            result = function(query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False, scale=1.0)
        elif backend == "xformers":
            result = function(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_bias=bias.contiguous() if bias is not None else None,
                p=0.0,
                scale=1.0,
            ).transpose(1, 2)
        elif backend == "flash_attn":
            result = function(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=0.0,
                softmax_scale=1.0,
                causal=False,
            ).transpose(1, 2)
        else:
            result = function(query, key, value, tensor_layout="HND", is_causal=False, sm_scale=1.0)
    return result.reshape(shape).float()


def probe_backend(policy, device, *, training):
    """Exercise the requested core and, for training, its Q/K/V and prior backward."""
    backend = policy["attention_backend"]
    if backend == "native":
        return
    if training and backend == "sage_attn":
        raise ValueError("SageAttention is inference-only for NR; training backward is not verified")
    dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[policy["mixed_precision"]]
    generator = torch.Generator(device=device).manual_seed(271)
    cases = (False, True) if policy["attention_scope"] == "all" else (False,)
    try:
        for window in cases:
            count = 64 if window else 17
            inputs = [
                torch.randn(2, 2, count, 32, device=device, dtype=dtype, generator=generator).requires_grad_(training)
                for _ in range(3)
            ]
            prior = (
                torch.randn(2, count, count, device=device, dtype=dtype, generator=generator).requires_grad_(training)
                if window
                else None
            )
            with torch.set_grad_enabled(training):
                result = attention(*inputs, backend=backend, prior=prior)
                if not torch.isfinite(result).all():
                    raise RuntimeError("non-finite attention output")
                if training:
                    gradients = torch.autograd.grad(
                        result.square().mean(), inputs + ([prior] if prior is not None else []), allow_unused=True
                    )
                    if any(
                        value is None or not torch.isfinite(value).all() or not torch.count_nonzero(value) for value in gradients
                    ):
                        raise RuntimeError("missing, zero or non-finite attention/prior backward")
    except Exception as error:
        raise RuntimeError(f"{backend} does not support this NR dtype/scope/backward on {device}: {error}") from error


def backend_identity(backend):
    if backend in ("native", "sdpa"):
        return {"backend": backend, "torch": str(torch.__version__)}
    package = OPTIONAL_BACKENDS[backend][2]
    try:
        version = metadata.version(package)
    except metadata.PackageNotFoundError:
        version = "unversioned"
    return {"backend": backend, "package": package, "version": version, "torch": str(torch.__version__)}
