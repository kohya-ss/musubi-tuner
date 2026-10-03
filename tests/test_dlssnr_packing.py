import os
from pathlib import Path

import numpy as np
import pytest

from musubi_tuner.dlssnr.checkpoint import (
    assert_auxiliary_fingerprint,
    assert_source_fingerprint,
    load_source,
    logical_parameter_count,
    measure_fingerprint,
)
from musubi_tuner.dlssnr.convert import verify_roundtrip
from musubi_tuner.dlssnr.packing import (
    decode_e4m3,
    encode_e4m3,
    inverse_packed_input_index,
    pack_e4_codes,
    pack_f16_matrix,
    pack_prior,
    packed_f16_index,
    packed_input_index,
    packed_weight_index,
    prior_half_index,
    tiled_token,
    unpack_e4_codes,
    unpack_f16_matrix,
    unpack_prior,
)
from musubi_tuner.dlssnr.profiles import SOURCE_FINGERPRINT, block_channels, build_records


def test_fragment_indexes_match_the_loader_formulas():
    assert int(packed_weight_index(np.int32(1), np.int32(0), 16)) == 1
    assert int(packed_f16_index(np.int32(1), np.int32(0), 32)) == 1
    assert int(prior_half_index(np.int32(1), np.int32(0))) == 32


def test_channel_permutation_is_an_involution_on_each_group_of_32():
    index = np.arange(64, dtype=np.int32)
    assert np.array_equal(inverse_packed_input_index(packed_input_index(index)), index)
    assert np.array_equal(packed_input_index(inverse_packed_input_index(index)), index)
    changed = packed_input_index(np.arange(32, dtype=np.int32))
    assert changed[0] == 0 and changed[1] == 1 and changed[2] == 8


def test_prior_address_covers_each_half_once():
    query = np.arange(64, dtype=np.int32)
    key = np.arange(64, dtype=np.int32)
    qq, kk = np.meshgrid(query, key, indexing="ij")
    index = prior_half_index(tiled_token(qq), tiled_token(kk)).reshape(-1)
    assert index.min() == 0
    assert index.max() == 4095
    assert len(np.unique(index)) == 4096


def test_e4m3_code_roundtrip_keeps_signed_zero():
    codes = np.arange(256, dtype=np.uint8)
    finite = codes[(codes & 0x7F) != 0x7F]
    restored = encode_e4m3(decode_e4m3(finite))
    assert np.array_equal(restored, finite)
    matrix = np.zeros((32, 16), dtype=np.uint8)
    matrix[0, 0] = 0x80
    matrix[1, 2] = 0x3C
    matrix[31, 15] = 0x01
    assert np.array_equal(unpack_e4_codes(pack_e4_codes(matrix), 32, 16), matrix)


def test_e4m3_expert_byte_split_matches_one_matrix():
    rng = np.random.default_rng(0)
    whole = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    whole[(whole & 0x7F) == 0x7F] = 0
    packed = pack_e4_codes(whole)
    first = unpack_e4_codes(packed[: 64 * 128], 64, 128)
    second = unpack_e4_codes(packed[64 * 128 :], 64, 128)
    assert np.array_equal(first, whole[:64])
    assert np.array_equal(second, whole[64:])


def test_f16_fragment_and_prior_roundtrip_keep_signed_zero():
    adapter = np.zeros((16, 32), dtype=np.float32)
    adapter[0, 0] = np.float32(-0.0)
    adapter[15, 31] = np.float32(0.5)
    assert np.array_equal(unpack_f16_matrix(pack_f16_matrix(adapter), 16, 32).view(np.uint32), adapter.view(np.uint32))

    head = np.zeros((32, 4), dtype=np.float32)
    head[0, 3] = np.float32(-0.25)
    restored = unpack_f16_matrix(pack_f16_matrix(head), 32, 4)
    assert restored.shape == (32, 4)
    assert np.array_equal(restored.view(np.uint32), head.view(np.uint32))
    stored = np.frombuffer(pack_f16_matrix(head), dtype="<u2")
    assert stored.size == 512
    assert int(np.count_nonzero(stored)) == 1

    prior = np.zeros((2, 64, 64), dtype=np.float32)
    prior[1, 3, 5] = np.float32(-0.0)
    prior[0, 63, 0] = np.float32(-2.0)
    assert np.array_equal(unpack_prior(pack_prior(prior), 2).view(np.uint32), prior.view(np.uint32))


def test_profile_inventory_matches_the_audited_ledger():
    records = build_records()
    assert len(records) == 153
    assert sum(record.nbytes for record in records) == 147_683_778
    assert logical_parameter_count(records) == SOURCE_FINGERPRINT["logical_parameters_without_blend"] + 1
    block1 = next(record for record in records if record.name == "block1.layer0.layer")
    assert block1.nbytes == 20_672
    assert [(region.offset, region.nbytes, region.kind) for region in block1.regions] == [
        (0, 4096, "e4"),
        (4096, 4096, "e4"),
        (8192, 16, "pad"),
        (8208, 64, "f16"),
        (8272, 16, "pad"),
        (8288, 3072, "e4"),
        (11360, 8192, "prior"),
        (19552, 4, "f32"),
        (19556, 12, "pad"),
        (19568, 1024, "e4"),
        (20592, 64, "f16"),
        (20656, 16, "pad"),
    ]
    assert next(record.phase for record in records if record.block == 0) == 0
    assert next(record.phase for record in records if record.name == "block70.layer0.layer") == 1
    assert next(record.phase for record in records if record.block == 56) == 2
    assert block_channels(23) == 512
    assert block_channels(31) == 1024
    assert block_channels(39) is None
    assert block_channels(48) == 256
    assert block_channels(62) == 64
    assert next(record.phase for record in records if record.block == 39) is None
    names = [view.name for record in records for region in record.regions for view in region.views]
    assert len(names) == len(set(names))
    groups = {view.group for record in records for region in record.regions for view in region.views}
    assert groups == {"matrices", "priors", "scales", "input_rgb_head", "temporal_head", "temporal_blend"}


def _source_dir() -> Path | None:
    candidates = []
    if os.environ.get("DLSSNR_SOURCE_DIR"):
        candidates.append(Path(os.environ["DLSSNR_SOURCE_DIR"]))
    candidates.append(Path(__file__).resolve().parents[2] / "OpenDLSS-NR" / "models" / "nr")
    for candidate in candidates:
        if (candidate / "manifest.json").is_file():
            return candidate
    return None


@pytest.mark.skipif(_source_dir() is None, reason="DLSS-NR 310.8.0 weights are not on this machine")
def test_shipped_weights_match_profile_and_repack():
    source = load_source(_source_dir())
    fingerprint = measure_fingerprint(source)
    assert_source_fingerprint(fingerprint)
    assert_auxiliary_fingerprint(source)
    verify_roundtrip(source)
