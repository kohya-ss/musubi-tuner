import json

import numpy as np
import pytest
from PIL import Image

from musubi_tuner.dlssnr.dataset import load_single_frame_manifest, load_temporal_manifest


def write_manifest(tmp_path, *, frames=1, mutate=None):
    Image.fromarray(np.full((48, 48, 3), 120, np.uint8)).save(tmp_path / "source.png")
    Image.fromarray(np.full((48, 48, 3), 40, np.uint8)).save(tmp_path / "target.png")
    np.save(tmp_path / "controls.npy", np.zeros((5, 48, 48), np.float32))
    np.save(tmp_path / "motion.npy", np.zeros((48, 48, 2), np.float32))
    Image.fromarray(np.full((48, 48), 255, np.uint8)).save(tmp_path / "mask.png")
    row = {
        "schema": "dlssnr_pairs_v1",
        "sample_id": "one",
        "sequence_id": "scene",
        "source_encoding": "srgb_proxy",
        "target_encoding": "srgb_proxy",
        "controls_encoding": "dlssnr_lanes_10_14_v1",
        "motion_layout": "hwc",
        "frames": [],
    }
    for index in range(frames):
        frame = {
            "frame_index": index + 10,
            "input_path": "source.png",
            "target_path": "target.png",
            "controls_path": "controls.npy",
            "reset": index == 0,
        }
        if index:
            frame.update(motion_path="motion.npy", history_valid_path="mask.png", temporal_valid_path="mask.png")
        row["frames"].append(frame)
    if mutate:
        mutate(row)
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_manifest_is_lazy_and_fingerprints_referenced_file_contents(tmp_path, monkeypatch):
    path = write_manifest(tmp_path)
    from musubi_tuner.dlssnr import dataset

    def cannot_decode_yet(path):
        raise AssertionError("manifest indexing must not decode every image")

    with monkeypatch.context() as scope:
        scope.setattr(dataset, "_proxy_image", cannot_decode_yet)
        samples = load_single_frame_manifest(path, 48, 48)
        assert len(samples) == 1
    before = samples.fingerprint()
    assert samples[0]["frame_index"] == 10
    Image.fromarray(np.full((48, 48, 3), 60, np.uint8)).save(tmp_path / "target.png")
    assert samples.fingerprint() != before
    assert samples[0]["target"].mean().item() == pytest.approx(60 / 255)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        (lambda r: r.update(source_encoding="linear_hdr"), "source_encoding"),
        (lambda r: r.pop("controls_encoding"), "controls_encoding"),
        (lambda r: r["frames"][1].update(frame_index=10), "frame_index"),
        (lambda r: r["frames"][1].update(reset="false"), "reset"),
    ],
)
def test_manifest_rejects_bad_encoding_and_sequence_metadata(tmp_path, mutation, reason):
    path = write_manifest(tmp_path, frames=2, mutate=mutation)
    with pytest.raises(ValueError, match=reason):
        load_temporal_manifest(path, 48, 48, 2)


@pytest.mark.parametrize("second_id", ["one", "ONE", "oNe"])
def test_duplicate_sample_ids_are_rejected(tmp_path, second_id):
    path = write_manifest(tmp_path)
    first = path.read_text(encoding="utf-8")
    second = json.loads(first)
    second["sample_id"] = second_id
    path.write_text(first + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sample_id"):
        load_single_frame_manifest(path, 48, 48)


@pytest.mark.parametrize("sample_id", ["NUL", "con.txt", "COM1", "Lpt9", "frame.", "frame ", "frame\x00", "frame\t"])
def test_sample_ids_reject_nonportable_filenames(tmp_path, sample_id):
    path = write_manifest(tmp_path, mutate=lambda row: row.update(sample_id=sample_id))
    with pytest.raises(ValueError, match="sample_id"):
        load_single_frame_manifest(path, 48, 48)


def test_png_masks_are_supported_and_temporal_mask_is_optional_when_loss_is_disabled(tmp_path):
    path = write_manifest(tmp_path, frames=2, mutate=lambda r: r["frames"][1].pop("temporal_valid_path"))
    clips = load_temporal_manifest(path, 48, 48, 2, require_temporal_mask=False)
    sample = clips[0]
    assert sample["frames"][1]["history_valid"].all()
    assert sample["frames"][1]["frame_index"] == 11
    assert not sample["frames"][1]["temporal_valid"].any()


def test_nonfinite_controls_fail_before_training(tmp_path):
    path = write_manifest(tmp_path)
    np.save(tmp_path / "controls.npy", np.full((5, 48, 48), np.nan, np.float32))
    with pytest.raises(ValueError, match="finite"):
        samples = load_single_frame_manifest(path, 48, 48)
        samples[0]
