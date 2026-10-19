"""Noise optimization must preserve the original hash and half publications."""

import numpy as np
import pytest

from musubi_tuner.dlssnr import noise


def reference_noise(width, height, seed):
    # Independent pre-optimization implementation, including integer promotion.
    def f32(bits):
        return np.array(bits, dtype=np.uint32).view(np.float32)

    def mul(values, factor):
        return (values.astype(np.uint64) * np.uint64(factor)).astype(np.uint32)

    def uniform(value):
        mixed = value.astype(np.uint32, copy=True)
        shift = (mixed >> np.uint32(28)).astype(np.int32) + 4
        mixed = np.bitwise_xor(np.right_shift(mixed, shift), mixed)
        mixed = mul(mixed, 0x108EF2D9)
        integer = np.bitwise_xor(mixed >> np.uint32(30), mixed >> np.uint32(8)) + np.uint32(1)
        return integer.astype(np.float32) * f32(0x33800000)

    xx, yy = np.meshgrid(np.arange(width, dtype=np.uint32), np.arange(height, dtype=np.uint32))
    seed_term = np.array((np.uint64(seed) * np.uint64(0x9E3779B9)) & np.uint64(0xFFFFFFFF), dtype=np.uint32)
    base = np.bitwise_xor(np.bitwise_xor(mul(xx, 0x8DA6B343), mul(yy, 0xD8163841)), seed_term)
    base = np.bitwise_xor(base, np.uint32(0x243F6A88))
    shift = (base >> np.uint32(28)).astype(np.int32) + 4
    base = np.bitwise_xor(np.right_shift(base, shift), base)
    base = mul(base, 0x108EF2D9)
    base = np.bitwise_xor(base >> np.uint32(22), base)
    u0 = uniform(mul(base, 0x2C9277B5) + np.uint32(0xAC564B05))
    u1 = uniform(mul(base, 0xFA6DC5F9) + np.uint32(0x4712A88E))
    u2 = uniform(mul(base, 0xCAA5B80D) + np.uint32(0x21DD796B))
    u3 = uniform(mul(base, 0x83232C31) + np.uint32(0x3463E0AC))
    radius0 = np.sqrt(np.log2(np.clip(u0, 1e-20, None)) * f32(0x3F317218) * np.float32(-2.0))
    radius1 = np.sqrt(np.log2(np.clip(u2, 1e-20, None)) * f32(0x3F317218) * np.float32(-2.0))
    lanes = np.stack(
        [radius0 * np.cos(u1 * f32(0x40C90FDB)), radius0 * np.sin(u1 * f32(0x40C90FDB)), radius1 * np.cos(u3 * f32(0x40C90FDB))]
    ).astype(np.float32)
    return lanes.astype(np.float16).astype(np.float32)


@pytest.mark.parametrize("size", [(1, 1), (17, 31), (65, 97), (320, 320), (1088, 1024)])
@pytest.mark.parametrize("seed", [0, 1, 1337, 0x80000000, 0xFFFFFFFF])
def test_noise_is_bitwise_equal_to_original_publication(size, seed):
    expected = reference_noise(*size, seed)
    actual = noise.gaussian_lanes(*size, seed)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_noise_limits_hash_working_set_instead_of_materializing_the_full_field(monkeypatch):
    sizes = []
    original = noise._hash_uniform

    def observed(value):
        sizes.append(value.size)
        return original(value)

    monkeypatch.setattr(noise, "_hash_uniform", observed)
    actual = noise.gaussian_lanes(65, 131, 7)
    assert max(sizes) <= 65 * 64, "Hash intermediates must stay bounded to a small row tile"
    np.testing.assert_array_equal(actual.view(np.uint32), reference_noise(65, 131, 7).view(np.uint32))


def test_hash_shifts_do_not_promote_uint32_arrays_to_int64(monkeypatch):
    dtypes = []
    original = np.right_shift

    def observed(left, right, *args, **kwargs):
        result = original(left, right, *args, **kwargs)
        dtypes.append(result.dtype)
        return result

    monkeypatch.setattr(np, "right_shift", observed)
    noise._hash_uniform(np.array([0, 1, 0x80000000, 0xFFFFFFFF], dtype=np.uint32))
    assert dtypes and all(dtype == np.uint32 for dtype in dtypes)


@pytest.mark.parametrize("seed", [-1, 2**64])
def test_noise_preserves_invalid_seed_rejection(seed):
    with pytest.raises(OverflowError):
        noise.gaussian_lanes(17, 31, seed)


def test_noise_changes_with_seed_without_consuming_numpy_rng():
    before = np.random.get_state()
    first = noise.gaussian_lanes(35, 67, 0)
    second = noise.gaussian_lanes(35, 67, 1)
    after = np.random.get_state()
    assert not np.array_equal(first, second)
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
