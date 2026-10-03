# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Fragment addressing for the DLSS-NR 310.8.0 packed weight files.

Index formulas follow the OpenDLSS loader (packedWeightIndex, packedF16WeightIndex,
relativeBias). FP8 K rows are restored to natural channel order with
inversePackedInputIndex. This module does not run the network.
"""

from __future__ import annotations

import sys

import numpy as np

if sys.byteorder != "little":
    raise RuntimeError("DLSS-NR packing assumes a little-endian host")


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def packed_input_index(k: np.ndarray | int) -> np.ndarray:
    """Native within-32 activation permutation. Bit 4 of the group is fixed."""
    k = np.asarray(k, dtype=np.int32)
    base = k & np.int32(~31)
    within = k & np.int32(31)
    half = within & np.int32(16)
    quarter = within & np.int32(15)
    extra = np.where((quarter & np.int32(2)) != 0, np.int32(8), np.int32(0))
    return base + half + (quarter >> np.int32(2)) * np.int32(2) + (quarter & np.int32(1)) + extra


def inverse_packed_input_index(k: np.ndarray | int) -> np.ndarray:
    """Inverse of packed_input_index. Disk row k is read into natural row inverse(k)."""
    k = np.asarray(k, dtype=np.int32)
    base = k & np.int32(~31)
    within = k & np.int32(31)
    return (
        base
        + (within & np.int32(17))
        + ((within & np.int32(2)) << np.int32(1))
        + ((within & np.int32(4)) << np.int32(1))
        + ((within & np.int32(8)) >> np.int32(2))
    )


def packed_weight_index(k: np.ndarray, n: np.ndarray, output_channels: int) -> np.ndarray:
    """Byte index of E4M3 weight (k, n) inside one packed [K, N] matrix."""
    k = np.asarray(k, dtype=np.int32)
    n = np.asarray(n, dtype=np.int32)
    k_tile = k >> np.int32(5)
    k_in = k & np.int32(31)
    n_tile = n >> np.int32(7)
    n_in = n & np.int32(127)
    n_half = n_in >> np.int32(6)
    n_group = (n_in & np.int32(63)) >> np.int32(4)
    n_in_group = n_in & np.int32(15)
    lane = ((n_in_group & np.int32(7)) << np.int32(2)) | ((k_in & np.int32(15)) >> np.int32(2))
    byte_in_lane = ((n_in_group >> np.int32(3)) << np.int32(3)) | ((k_in >> np.int32(4)) << np.int32(2)) | (k_in & np.int32(3))
    return (
        k_tile * np.int32(output_channels * 32)
        + n_tile * np.int32(4096)
        + n_half * np.int32(2048)
        + n_group * np.int32(512)
        + lane * np.int32(16)
        + byte_in_lane
    )


def packed_f16_index(k: np.ndarray, n: np.ndarray, output_channels: int) -> np.ndarray:
    """Half index of an f16 GEMM weight inside m16n8k16 tiles."""
    k = np.asarray(k, dtype=np.int32)
    n = np.asarray(n, dtype=np.int32)
    n_tiles = (output_channels + 15) // 16
    tile = (k >> np.int32(4)) * np.int32(n_tiles) + (n >> np.int32(4))
    kk = k & np.int32(15)
    nn = n & np.int32(15)
    lane = ((nn & np.int32(7)) << np.int32(2)) | ((kk & np.int32(7)) >> np.int32(1))
    fragment = np.where(kk >= np.int32(8), np.int32(2), np.int32(0)) + (kk & np.int32(1))
    return tile * np.int32(256) + lane * np.int32(8) + ((nn >> np.int32(3)) & np.int32(1)) * np.int32(4) + fragment


def tiled_token(token: np.ndarray) -> np.ndarray:
    """Natural 8x8 window token -> 4x4-tiled physical token."""
    token = np.asarray(token, dtype=np.int32)
    x = token & np.int32(7)
    y = token >> np.int32(3)
    return (
        ((y >> np.int32(2)) * np.int32(32))
        + ((x >> np.int32(2)) * np.int32(16))
        + ((y & np.int32(3)) * np.int32(4))
        + (x & np.int32(3))
    )


def prior_half_index(physical_query: np.ndarray, physical_key: np.ndarray) -> np.ndarray:
    """Half index of one attention-prior element. Both axes are physical tokens."""
    q = np.asarray(physical_query, dtype=np.int32)
    k = np.asarray(physical_key, dtype=np.int32)
    m = q & np.int32(15)
    n = k & np.int32(15)
    lane = ((m & np.int32(7)) << np.int32(2)) | ((n & np.int32(7)) >> np.int32(1))
    fragment = np.where(m >= np.int32(8), np.int32(2), np.int32(0)) + (n & np.int32(1))
    tile_offset = (q >> np.int32(4)) * np.int32(1024) + (k >> np.int32(4)) * np.int32(256)
    lane_offset = lane * np.int32(8) + (n >> np.int32(3)) * np.int32(4)
    return tile_offset + lane_offset + fragment


def _as_uint8(blob: bytes | np.ndarray) -> np.ndarray:
    if isinstance(blob, np.ndarray):
        return np.ascontiguousarray(blob, dtype=np.uint8).reshape(-1)
    return np.frombuffer(blob, dtype=np.uint8)


def unpack_e4_codes(blob: bytes | np.ndarray, k_size: int, n_size: int) -> np.ndarray:
    """Return uint8 codes in natural [K, N] order."""
    raw = _as_uint8(blob)
    if raw.size != k_size * n_size:
        raise ValueError(f"E4M3 region is {raw.size} bytes, expected {k_size}x{n_size}")
    rows = inverse_packed_input_index(np.arange(k_size, dtype=np.int32))
    cols = np.arange(n_size, dtype=np.int32)
    index = packed_weight_index(rows[:, None], cols[None, :], n_size)
    return raw[index]


def pack_e4_codes(codes_kn: np.ndarray) -> bytes:
    """Pack natural [K, N] E4M3 codes back into fragment order."""
    codes = np.ascontiguousarray(codes_kn, dtype=np.uint8)
    k_size, n_size = codes.shape
    rows = inverse_packed_input_index(np.arange(k_size, dtype=np.int32))
    cols = np.arange(n_size, dtype=np.int32)
    index = packed_weight_index(rows[:, None], cols[None, :], n_size)
    out = np.zeros(k_size * n_size, dtype=np.uint8)
    out[index.reshape(-1)] = codes.reshape(-1)
    return out.tobytes()


def unpack_f16_matrix(blob: bytes | np.ndarray, k_size: int, n_size: int) -> np.ndarray:
    """Return float32 [K, N], including signed zeros from the stored halves."""
    raw = _as_uint8(blob)
    halves = raw.view(np.uint16)
    rows = np.arange(k_size, dtype=np.int32)
    cols = np.arange(n_size, dtype=np.int32)
    index = packed_f16_index(rows[:, None], cols[None, :], n_size)
    if int(index.max()) >= halves.size:
        raise ValueError(f"f16 fragment index {int(index.max())} outside {halves.size} halves")
    selected = halves[index]
    return selected.view(np.float16).astype(np.float32)


def pack_f16_matrix(values_kn: np.ndarray) -> bytes:
    """Pack float32 [K, N] exact half values into m16n8k16 fragment order."""
    values = np.ascontiguousarray(values_kn, dtype=np.float32)
    k_size, n_size = values.shape
    n_tiles = (n_size + 15) // 16
    k_tiles = (k_size + 15) // 16
    out = np.zeros(k_tiles * n_tiles * 256, dtype=np.uint16)
    rows = np.arange(k_size, dtype=np.int32)
    cols = np.arange(n_size, dtype=np.int32)
    index = packed_f16_index(rows[:, None], cols[None, :], n_size)
    out[index.reshape(-1)] = values.astype(np.float16).view(np.uint16).reshape(-1)
    return out.view(np.uint8).tobytes()


def _prior_natural_index() -> np.ndarray:
    query = np.arange(64, dtype=np.int32)
    key = np.arange(64, dtype=np.int32)
    qq, kk = np.meshgrid(query, key, indexing="ij")
    return prior_half_index(tiled_token(qq), tiled_token(kk))


_PRIOR_INDEX = _prior_natural_index()


def unpack_prior(blob: bytes | np.ndarray, heads: int) -> np.ndarray:
    """Return float32 [heads, 64 query, 64 key], both axes in natural token order."""
    raw = _as_uint8(blob)
    halves = raw.view(np.uint16)
    if halves.size != heads * 4096:
        raise ValueError(f"prior is {halves.size} halves, expected {heads * 4096}")
    out = np.empty((heads, 64, 64), dtype=np.float32)
    for head in range(heads):
        selected = halves[head * 4096 + _PRIOR_INDEX]
        out[head] = selected.view(np.float16).astype(np.float32)
    return out


def pack_prior(values: np.ndarray) -> bytes:
    """Pack [heads, 64, 64] natural-order prior back into fragment order."""
    values = np.ascontiguousarray(values, dtype=np.float32)
    heads = values.shape[0]
    out = np.zeros(heads * 4096, dtype=np.uint16)
    halves = values.astype(np.float16).view(np.uint16)
    for head in range(heads):
        out[head * 4096 + _PRIOR_INDEX.reshape(-1)] = halves[head].reshape(-1)
    return out.view(np.uint8).tobytes()


def decode_e4m3(codes: np.ndarray) -> np.ndarray:
    """E4M3FN codes to float32. 0x80 stays -0, and 0x7f/0xff stay NaN."""
    c = np.asarray(codes, dtype=np.uint8)
    sign = np.where((c & np.uint8(0x80)) != 0, np.float32(-1.0), np.float32(1.0))
    exp = ((c >> np.uint8(3)) & np.uint8(0x0F)).astype(np.int32)
    mant = (c & np.uint8(7)).astype(np.int32)
    subnormal = mant.astype(np.float32) * np.float32(2.0**-9)
    normal = (np.float32(1.0) + mant.astype(np.float32) / np.float32(8.0)) * np.exp2((exp - 7).astype(np.float32))
    finite = np.where(exp == 0, subnormal, normal).astype(np.float32)
    nan = (exp == 15) & (mant == 7)
    finite = np.where(nan, np.float32(np.nan), finite)
    return finite * sign


def _e4m3_encode_tables() -> tuple[np.ndarray, np.ndarray]:
    codes = np.arange(256, dtype=np.uint8)
    values = decode_e4m3(codes)
    order = np.argsort(values.view(np.uint32), kind="stable")
    return values.view(np.uint32)[order], codes[order]


_E4_BITS, _E4_CODES = _e4m3_encode_tables()


def encode_e4m3(values: np.ndarray) -> np.ndarray:
    """Map exact E4M3 float32 values back to codes. Other values raise."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    flat = bits.reshape(-1)
    pos = np.searchsorted(_E4_BITS, flat)
    if np.any(pos >= _E4_BITS.size) or np.any(_E4_BITS[np.minimum(pos, _E4_BITS.size - 1)] != flat):
        raise ValueError("value is not an exact E4M3FN code")
    return _E4_CODES[pos].reshape(bits.shape)
