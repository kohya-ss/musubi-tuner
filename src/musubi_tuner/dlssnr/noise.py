# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Per-pixel Gaussian lanes from the DLSS-NR preprocess hash.

This is a direct port of `shaders/preprocess.comp`. XOR order does not matter. Values are rounded to
float16 the way the shader publishes them. The function is not part of the autograd graph.
"""

from __future__ import annotations

import numpy as np


def _bits_to_f32(bits: int) -> np.float32:
    return np.array(bits, dtype=np.uint32).view(np.float32)


def _wrap_mul(values: np.ndarray, factor: int) -> np.ndarray:
    return np.multiply(values, np.uint32(factor), dtype=np.uint32)


def _hash_uniform(value: np.ndarray) -> np.ndarray:
    mixed = value.astype(np.uint32, copy=True)
    shift = (mixed >> np.uint32(28)) + np.uint32(4)
    mixed = np.bitwise_xor(np.right_shift(mixed, shift), mixed)
    mixed = _wrap_mul(mixed, 0x108EF2D9)
    integer = np.bitwise_xor(mixed >> np.uint32(30), mixed >> np.uint32(8)) + np.uint32(1)
    return integer.astype(np.float32) * _bits_to_f32(0x33800000)


def gaussian_lanes(width: int, height: int, seed: int) -> np.ndarray:
    """Return float32 [3, height, width] noise for one frame. Coordinates are the padded ones."""
    x = _wrap_mul(np.arange(width, dtype=np.uint32), 0x8DA6B343)[None, :]
    seed_term = np.array((np.uint64(seed) * np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF), dtype=np.uint32)
    ln2 = _bits_to_f32(0x3F317218)
    tau = _bits_to_f32(0x40C90FDB)
    result = np.empty((3, height, width), dtype=np.float32)
    # Bound temporary arrays without changing any per-pixel float operation or rounding.
    for top in range(0, height, 32):
        bottom = min(top + 32, height)
        y = _wrap_mul(np.arange(top, bottom, dtype=np.uint32), 0xD8163841)[:, None]
        base = np.bitwise_xor(np.bitwise_xor(x, y), seed_term)
        base = np.bitwise_xor(base, np.uint32(0x243F6A88))
        shift = (base >> np.uint32(28)) + np.uint32(4)
        base = np.bitwise_xor(np.right_shift(base, shift), base)
        base = _wrap_mul(base, 0x108EF2D9)
        base = np.bitwise_xor(base >> np.uint32(22), base)
        u0 = _hash_uniform(_wrap_mul(base, 0x2C9277B5) + np.uint32(0xAC564B05))
        u1 = _hash_uniform(_wrap_mul(base, 0xFA6DC5F9) + np.uint32(0x4712A88E))
        u2 = _hash_uniform(_wrap_mul(base, 0xCAA5B80D) + np.uint32(0x21DD796B))
        u3 = _hash_uniform(_wrap_mul(base, 0x83232C31) + np.uint32(0x3463E0AC))
        radius0 = np.sqrt(np.log2(np.clip(u0, 1e-20, None)) * ln2 * np.float32(-2.0))
        radius1 = np.sqrt(np.log2(np.clip(u2, 1e-20, None)) * ln2 * np.float32(-2.0))
        lanes = np.stack([radius0 * np.cos(u1 * tau), radius0 * np.sin(u1 * tau), radius1 * np.cos(u3 * tau)], axis=0).astype(
            np.float32
        )
        result[:, top:bottom] = lanes.astype(np.float16).astype(np.float32)
    return result
