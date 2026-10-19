# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Published half/E4 values with explicit floating-point surrogate derivatives.

These reproduce scalar publications, not the native F13/F24 matrix accumulators.
Rounding has an identity backward; E4 saturation has the clamp's zero backward.
The exponential uses the continuous piecewise-linear mantissa as its backward.
Constants follow OpenDLSS-NR 9d08f418, common.glsl and the scalar CPU reference.
"""

from __future__ import annotations

import torch


class _Round(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, e4):
        rounded = value.to(torch.float16)
        if e4:
            rounded = rounded.to(torch.float8_e4m3fn)
        return rounded.to(value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


class _ForwardValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reference, surrogate):
        return reference

    @staticmethod
    def backward(ctx, gradient):
        return None, gradient


def half_ste(value: torch.Tensor) -> torch.Tensor:
    return _Round.apply(value, False)


def e4m3_ste(value: torch.Tensor) -> torch.Tensor:
    return _Round.apply(value.clamp(-448, 448), True)


def cubic_half(value: torch.Tensor) -> torch.Tensor:
    value = half_ste(value)
    bounded = value.clamp(-4, 4)
    inner = half_ste(-0.055908203125 * bounded.abs() + 0.447265625)
    polynomial = half_ste(bounded * inner + 0.89453125)
    return half_ste(value * polynomial)


def exp_weight(score: torch.Tensor, *, vit: bool = False) -> torch.Tensor:
    if vit:
        scale, bias, lower, upper, shift, offset = 0.08953857421875, 1.708984375, 1.439453125, 1.9775390625, 4, 0x4000
    else:
        scale, bias, lower, upper, shift, offset = 0.044921875, 1.30078125, 1.03125, 1.5693359375, 5, 0x8000
    with torch.no_grad():
        affine = (score * scale + bias).to(torch.float16).clamp(lower, upper)
        bits = (affine.view(torch.int16).to(torch.int32) << shift) + offset
        reference = bits.to(torch.int16).view(torch.float16).to(score.dtype)
    affine_float = (score * scale + bias).clamp(lower, upper)
    exponent = (2**shift) * (affine_float - 1) - 15
    integral = exponent.floor()
    surrogate = torch.exp2(integral) * (1 + exponent - integral)
    return _ForwardValue.apply(reference, surrogate)


def cosine_half(value: torch.Tensor) -> torch.Tensor:
    """A 32-lane norm with the half reduction tree; zero rows publish zero."""
    if value.shape[-1] != 32:
        raise ValueError("cosine_half requires 32 channels per head")
    value = half_ste(value)
    summed = half_ste(value[..., :16].square() + half_ste(value[..., 16:].square()))
    for stride in (8, 4, 2, 1):
        summed = half_ste(summed[..., :stride] + summed[..., stride : 2 * stride])
    zero = summed == 0
    reciprocal = half_ste(torch.rsqrt(torch.where(zero, torch.ones_like(summed), summed)))
    return torch.where(zero, torch.zeros_like(value), half_ste(value * reciprocal))


def sum64_half(value: torch.Tensor, *, window: bool = False) -> torch.Tensor:
    """Sum 64 weights in the runtime's lane-local order, publishing each add."""
    if value.shape[-1] != 64:
        raise ValueError("sum64_half requires 64 keys")
    if window:
        # Natural 8x8 tokens to physical 4x4-tiled key order.
        order = torch.arange(64, device=value.device).reshape(2, 4, 2, 4).permute(0, 2, 1, 3).reshape(64)
        value = value[..., order]
    chunks = value.reshape(*value.shape[:-1], 4, 16)
    paired = half_ste(chunks[..., :8] + chunks[..., 8:])
    total = paired[..., 0, :]
    for index in range(1, 4):
        total = half_ste(total + paired[..., index, :])
    even, odd = total[..., 0:1], total[..., 1:2]
    for index in (2, 4, 6):
        even = half_ste(even + total[..., index : index + 1])
        odd = half_ste(odd + total[..., index + 1 : index + 2])
    return half_ste(even + odd)
