import importlib

import numpy as np
import pytest

from musubi_tuner.dlssnr.packing import decode_e4m3
from musubi_tuner.dlssnr.checkpoint import repack_record, unpack_record
from musubi_tuner.dlssnr.profiles import RecordLayout, Region, View


def test_fp8_rounds_ties_to_even_and_preserves_signed_zero():
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    values = np.array([0.0, -0.0, 1.0625, 1.1875, -1.0625, -1.1875, 2**-10, 3 * 2**-10, 448], dtype=np.float32)
    actual = native.quantize_tensor(values, "e4")
    expected = np.array([0.0, -0.0, 1.0, 1.25, -1.0, -1.25, 0.0, 2**-8, 448], dtype=np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_every_finite_fp8_code_is_preserved():
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    codes = np.arange(256, dtype=np.uint8)
    values = decode_e4m3(codes[(codes & 127) != 127])
    np.testing.assert_array_equal(native.quantize_tensor(values, "e4").view(np.uint32), values.view(np.uint32))


@pytest.mark.parametrize("kind", ["f16", "f16frag", "prior"])
def test_half_regions_round_once_and_preserve_signed_zero(kind):
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    values = np.array([1.00048828125, 1.00146484375, -0.0, 65504], dtype=np.float32)
    expected = np.array([1.0, 1.001953125, -0.0, 65504], dtype=np.float32)
    np.testing.assert_array_equal(native.quantize_tensor(values, kind).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("kind,value", [("e4", 448.1), ("f16", 65505), ("prior", -65505), ("f32", np.inf), ("e4", np.nan)])
def test_export_rejects_nonfinite_and_out_of_range_values(kind, value):
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    with pytest.raises(ValueError, match="finite|range"):
        native.quantize_tensor(np.array([value], dtype=np.float32), kind)


def tiny_record():
    return RecordLayout(
        "block1.layer0.layer",
        1,
        0,
        "layer",
        "enc32",
        0,
        (
            Region("e4", 0, 4096, k=32, n=128, views=(View("weight", "matrices", 0, 32, 0, 128),)),
            Region("f16", 4096, 2, k=1, views=(View("skip", "scales", 0, 1, 0, 1),)),
            Region("pad", 4098, 16),
        ),
    )


@pytest.mark.parametrize("mix,expected", [(0, 1.0), (0.5, 1.25), (1, 1.5)])
def test_mix_interpolates_the_base_and_trained_weights(mix, expected):
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    record = tiny_record()
    base = {"weight": np.full((128, 32), 1.0, np.float32), "skip": np.array([-0.0], np.float32)}
    original = repack_record(record, base)
    trained = {"weight": np.full((128, 32), 1.5, np.float32), "skip": base["skip"]}
    payload, stats = native.repack_trained_record(record, original, trained, mix=mix)
    restored = unpack_record(record, payload)
    assert np.all(restored["weight"] == expected)
    assert np.signbit(restored["skip"][0])
    assert stats[0]["trained_changed_values"] == 4096
    assert stats[0]["exported_changed_values"] == (0 if mix == 0 else 4096)
    if mix == 0:
        assert payload == original


@pytest.mark.parametrize("mix,strength,expected", [(0, 0.5, 0.5), (1, 0.5, 0.75), (0.5, 2, 2.5), (1, 0, 0)])
def test_strength_scales_all_numeric_weights_after_mix(mix, strength, expected):
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    record = tiny_record()
    base = {"weight": np.ones((128, 32), np.float32), "skip": np.array([1], np.float32)}
    trained = {"weight": np.full((128, 32), 1.5, np.float32), "skip": np.array([2], np.float32)}
    payload, _ = native.repack_trained_record(record, repack_record(record, base), trained, mix=mix, strength=strength)
    restored = unpack_record(record, payload)
    assert np.all(restored["weight"] == expected)
    assert restored["skip"][0] == (1 + mix) * strength


def test_report_exposes_updates_lost_to_quantization():
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    record = tiny_record()
    base = {"weight": np.ones((128, 32), np.float32), "skip": np.array([1], np.float32)}
    trained = {"weight": base["weight"] + np.float32(0.01), "skip": base["skip"]}
    payload, stats = native.repack_trained_record(record, repack_record(record, base), trained, strength=1)
    assert payload == repack_record(record, base)
    assert stats[0]["trained_changed_values"] == stats[0]["lost_update_values"] == 4096
    assert stats[0]["exported_changed_values"] == 0
    assert stats[0]["max_abs_quantization_error"] == pytest.approx(0.01)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "missing", "nan", "strength"])
def test_record_export_rejects_invalid_weights_before_packing(mutation):
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    record = tiny_record()
    base = {"weight": np.ones((128, 32), np.float32), "skip": np.array([1], np.float32)}
    original = repack_record(record, base)
    trained = dict(base)
    if mutation == "shape":
        trained["weight"] = trained["weight"][:1]
    elif mutation == "dtype":
        trained["weight"] = trained["weight"].astype(np.float16)
    elif mutation == "missing":
        del trained["skip"]
    elif mutation == "nan":
        trained["weight"][0, 0] = np.nan
    with pytest.raises(ValueError):
        native.repack_trained_record(record, original, trained, strength=np.inf if mutation == "strength" else 1)


def test_record_export_rejects_modified_opaque_bytes():
    native = importlib.import_module("musubi_tuner.dlssnr.native")
    record = RecordLayout("block31.layer3.layer", 31, 3, "layer", "vit", None, (Region("opaque", 0, 2),))
    with pytest.raises(ValueError, match="opaque"):
        native.repack_trained_record(record, b"\x00\x80", {"blocks.31.opaque.layer3": np.array([1, 128], np.uint8)}, strength=1)
