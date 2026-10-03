import json

import numpy as np
import pytest
import toml
import torch
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.dlssnr.config import load_dataset_config
from musubi_tuner.dlssnr.dataset import load_single_frame_manifest, load_temporal_manifest
from test_dlssnr_training import make_args, small_math  # noqa: F401


def write_pairs(tmp_path, sizes, *, frames=1, png=False):
    rows = []
    for index, (width, height) in enumerate(sizes):
        red = np.broadcast_to(np.linspace(0.1, 0.9, height, dtype=np.float32)[:, None], (height, width))
        rgb = np.stack((red, red, red))
        controls = np.stack((red, np.full_like(red, 0.5), red, red, red))
        source_name = f"source{index}.{'png' if png else 'npy'}"
        if png:
            Image.fromarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8)).save(tmp_path / source_name)
        else:
            np.save(tmp_path / source_name, rgb)
        np.save(tmp_path / f"target{index}.npy", rgb)
        np.save(tmp_path / f"controls{index}.npy", controls)
        np.save(tmp_path / f"motion{index}.npy", np.stack((np.full_like(red, 10), np.full_like(red, 8))))
        mask = np.ones((1, height, width), np.float32)
        mask[:, :, : width // 2] = 0
        np.save(tmp_path / f"valid{index}.npy", mask)
        np.save(tmp_path / f"loss{index}.npy", np.full((1, height, width), 0.75, np.float32))
        clip = []
        for frame_index in range(frames):
            frame = {
                "frame_index": frame_index,
                "reset": frame_index == 0,
                "input_path": source_name,
                "target_path": f"target{index}.npy",
                "controls_path": f"controls{index}.npy",
                "loss_mask_path": f"loss{index}.npy",
            }
            if frame_index:
                frame.update(
                    motion_path=f"motion{index}.npy",
                    history_valid_path=f"valid{index}.npy",
                    temporal_valid_path=f"valid{index}.npy",
                )
            clip.append(frame)
        rows.append(
            {
                "schema": "dlssnr_pairs_v1",
                "sample_id": f"sample{index}",
                "sequence_id": f"scene{index}",
                "source_encoding": "srgb_proxy",
                "target_encoding": "srgb_proxy",
                "controls_encoding": "dlssnr_lanes_10_14_v1",
                "motion_layout": "chw",
                "frames": clip,
            }
        )
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_dataset_bucket_options_override_general_without_becoming_training_flags(tmp_path):
    path = tmp_path / "dataset.toml"
    path.write_text(
        toml.dumps(
            {
                "general": {"resolution": 128, "enable_bucket": False, "bucket_no_upscale": False},
                "datasets": [{"train_manifest": "pairs.jsonl", "enable_bucket": True, "bucket_no_upscale": True}],
            }
        ),
        encoding="utf-8",
    )
    data = load_dataset_config(path)
    assert data["enable_bucket"] is True
    assert data["bucket_no_upscale"] is True
    assert data["bucket_size"] == [128, 128]


@pytest.mark.parametrize("field", ["enable_bucket", "bucket_no_upscale"])
def test_dataset_bucket_options_require_booleans(tmp_path, field):
    path = tmp_path / "dataset.toml"
    path.write_text(toml.dumps({"general": {field: "true"}, "datasets": [{"train_manifest": "pairs.jsonl"}]}))
    with pytest.raises(ValueError, match=field):
        load_dataset_config(path)


@pytest.mark.parametrize("png", [False, True])
def test_public_bucket_selector_drives_landscape_and_portrait_samples(tmp_path, png):
    path = write_pairs(tmp_path, [(320, 192), (192, 320), (128, 128)], png=png)
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=True)
    assert [tuple(samples[index]["source"].shape[-2:]) for index in range(3)] == [(96, 160), (160, 96), (128, 128)]
    assert samples.bucket_sizes == [(160, 96), (96, 160), (128, 128)]


def test_bucket_no_upscale_uses_native_size_rounded_down_to_nr_steps(tmp_path):
    path = write_pairs(tmp_path, [(79, 63)])
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=True, bucket_no_upscale=True)
    assert samples[0]["source"].shape == (3, 48, 64)


def test_nr_bucket_selector_excludes_axes_smaller_than_native_geometry(tmp_path):
    path = write_pairs(tmp_path, [(200, 128)])
    samples = load_single_frame_manifest(path, 48, 48, enable_bucket=True)
    assert samples[0]["source"].shape == (3, 48, 48)


def test_bucket_no_upscale_rejects_images_below_the_native_minimum(tmp_path):
    path = write_pairs(tmp_path, [(40, 64)])
    with pytest.raises(ValueError, match="unsupported size|minimum"):
        load_single_frame_manifest(path, 128, 128, enable_bucket=True, bucket_no_upscale=True)


def test_disabled_buckets_keep_strict_fixed_resolution_without_implicit_resize(tmp_path):
    path = write_pairs(tmp_path, [(200, 128)])
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=False)
    with pytest.raises(ValueError, match="match bucket"):
        samples[0]


def test_bucket_indexing_reads_headers_without_decoding_pixels(tmp_path, monkeypatch):
    from musubi_tuner.dlssnr import dataset

    path = write_pairs(tmp_path, [(320, 192)], png=True)

    def cannot_decode_yet(path):
        raise AssertionError("bucket indexing must not decode every image")

    monkeypatch.setattr(dataset, "_proxy_image", cannot_decode_yet)
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=True)
    assert samples.bucket_sizes == [(160, 96)]


@pytest.mark.parametrize("layout", ["chw", "hwc"])
def test_temporal_resize_and_crop_keep_pairs_and_pixel_motion_aligned(tmp_path, layout):
    path = write_pairs(tmp_path, [(200, 128)], frames=3)
    if layout == "hwc":
        row = json.loads(path.read_text())
        row["motion_layout"] = "hwc"
        path.write_text(json.dumps(row), encoding="utf-8")
        motion = np.load(tmp_path / "motion0.npy").transpose(1, 2, 0)
        np.save(tmp_path / "motion0.npy", motion)
    sample = load_temporal_manifest(path, 128, 128, 3, enable_bucket=True)[0]
    for frame in sample["frames"]:
        assert frame["source"].shape == (3, 96, 160)
        torch.testing.assert_close(frame["source"], frame["target"], rtol=0, atol=0)
        torch.testing.assert_close(frame["controls"][0], frame["source"][0], rtol=0, atol=0)
        torch.testing.assert_close(frame["source"], sample["frames"][0]["source"], rtol=0, atol=0)
        torch.testing.assert_close(frame["loss_mask"], torch.full((1, 96, 160), 0.75), rtol=0, atol=2e-7)
    for frame in sample["frames"][1:]:
        # Cover resize is 160 x 102, followed by a three-pixel center crop on y.
        torch.testing.assert_close(frame["motion"][0], torch.full((96, 160), 8.0), rtol=0, atol=1e-6)
        torch.testing.assert_close(frame["motion"][1], torch.full((96, 160), 6.375), rtol=0, atol=1e-6)
        assert frame["history_valid"].dtype == torch.bool
        assert not frame["history_valid"][:, :, :80].any()
        assert frame["history_valid"][:, :, 80:].all()
        assert torch.equal(frame["history_valid"], frame["temporal_valid"])


def test_mask_center_crop_uses_the_same_pixel_centers_as_rgb_resize(tmp_path):
    path = write_pairs(tmp_path, [(200, 128)], frames=2)
    mask = np.zeros((1, 128, 200), np.float32)
    mask[:, 32:96, 100:] = 1
    np.save(tmp_path / "valid0.npy", mask)
    frame = load_temporal_manifest(path, 128, 128, 2, enable_bucket=True)[0]["frames"][1]
    expected = torch.zeros(1, 96, 160, dtype=torch.bool)
    expected[:, 22:73, 80:] = True
    assert torch.equal(frame["history_valid"], expected)


def test_temporal_bucket_rejects_resolution_changes_within_a_clip(tmp_path):
    path = write_pairs(tmp_path, [(200, 128), (128, 200)], frames=2)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["frames"][1]["input_path"] = "source1.npy"
    path.write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="clip|resolution|size"):
        load_temporal_manifest(path, 128, 128, 2, enable_bucket=True)


def test_bucket_crop_rejects_a_loss_mask_with_no_remaining_supervision(tmp_path):
    path = write_pairs(tmp_path, [(200, 128)])
    mask = np.zeros((1, 128, 200), np.float32)
    mask[:, :1] = 1
    np.save(tmp_path / "loss0.npy", mask)
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=True)
    with pytest.raises(ValueError, match="supervised"):
        samples[0]


def test_motion_that_overflows_after_upscale_is_rejected(tmp_path):
    path = write_pairs(tmp_path, [(64, 64)], frames=2)
    np.save(tmp_path / "motion0.npy", np.full((2, 64, 64), 3e38, np.float32))
    samples = load_temporal_manifest(path, 128, 128, 2, enable_bucket=True)
    with pytest.raises(ValueError, match="finite"):
        samples[0]


def test_bucket_indexing_rejects_empty_source_dimensions(tmp_path):
    path = write_pairs(tmp_path, [(64, 64)])
    np.save(tmp_path / "source0.npy", np.zeros((3, 0, 64), np.float32))
    with pytest.raises(ValueError, match="size|dimension"):
        load_single_frame_manifest(path, 128, 128, enable_bucket=True)


def test_bucket_interpolation_keeps_proxy_and_loss_mask_in_their_declared_range(tmp_path):
    path = write_pairs(tmp_path, [(200, 128)])
    np.save(tmp_path / "source0.npy", np.ones((3, 128, 200), np.float32))
    np.save(tmp_path / "target0.npy", np.ones((3, 128, 200), np.float32))
    np.save(tmp_path / "loss0.npy", np.ones((1, 128, 200), np.float32))
    frame = load_single_frame_manifest(path, 128, 128, enable_bucket=True)[0]
    for name in ("source", "target", "loss_mask"):
        assert ((frame[name] >= 0) & (frame[name] <= 1)).all()


def test_misaligned_pairs_report_the_original_grid_not_the_area_budget(tmp_path):
    path = write_pairs(tmp_path, [(320, 192)])
    np.save(tmp_path / "target0.npy", np.zeros((3, 128, 128), np.float32))
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=True)
    with pytest.raises(ValueError, match="original.*320x192") as error:
        samples[0]
    assert "160x96" in str(error.value)


def test_bucket_plan_never_mixes_shapes_or_duplicates_small_bucket_tails(tmp_path):
    from musubi_tuner.dataset.bucket import BucketBatchManager
    from musubi_tuner.dlssnr.dataset import NRBatchPlan

    path = write_pairs(tmp_path, [(320, 192), (192, 320), (320, 192), (128, 128), (320, 192)])
    samples = load_single_frame_manifest(path, 128, 128, enable_bucket=True)
    plan = NRBatchPlan(samples, 2)
    assert isinstance(plan.manager, BucketBatchManager)
    assert len(plan) == 4
    indices = [plan.indices(index) for index in range(len(plan))]
    assert sorted(item for batch in indices for item in batch) == list(range(5))
    assert sorted(map(len, indices)) == [1, 1, 1, 2]
    for batch in indices:
        assert len({samples.bucket_sizes[index] for index in batch}) == 1
    assert plan.sample_count(4) == 5
    assert plan.sample_count(6) == 7
    assert plan.indices(4) == plan.indices(0)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
@pytest.mark.usefixtures("small_math")
def test_mixed_buckets_train_and_resume_exactly_with_partial_batches(tmp_path, lora, mode):
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path, lora=lora, accum=2, batch=2, mode=mode)
    path = write_pairs(
        tmp_path, [(320, 192), (192, 320), (320, 192), (128, 128), (320, 192)], frames=1 if mode == "single_frame" else 3
    )
    args.dataset_config.write_text(
        toml.dumps(
            {
                "general": {"resolution": 128, "batch_size": 2, "enable_bucket": True},
                "datasets": [{"train_manifest": path.name}],
            }
        ),
        encoding="utf-8",
    )
    args.max_train_steps = 3
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    run = tmp_path / "output/dlssnr"
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = {key: tensor.clone() for key, tensor in load_file(run / "final" / filename).items()}
    state = torch.load(run / "state-step000003/trainer_state.pt", weights_only=True)
    assert state["consumed_samples"] == 7
    args.resume = run / "state-step000001"
    train(args)
    actual = load_file(run / "final" / filename)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert json.loads((run / "run_config.json").read_text())["bucket_plan"]["samples_per_epoch"] == 5


@pytest.mark.usefixtures("small_math")
def test_validation_sequences_use_buckets_without_changing_training_rng(tmp_path):
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path, lora=True, accum=2, batch=2)
    path = write_pairs(tmp_path, [(320, 192), (192, 320)], frames=1)
    validation = json.loads(path.read_text().splitlines()[1])
    validation["sample_id"], validation["sequence_id"] = "validation", "validation"
    validation_path = tmp_path / "validation_pairs.jsonl"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    data = {"train_manifest": path.name}
    config = {"general": {"resolution": 128, "batch_size": 2, "enable_bucket": True}, "datasets": [data]}
    args.dataset_config.write_text(toml.dumps(config), encoding="utf-8")
    args.output_name = "plain"
    trainer.train_lora_from_args(args)
    data["validation_manifest"] = validation_path.name
    args.dataset_config.write_text(toml.dumps(config), encoding="utf-8")
    args.output_name, args.sample_every_n_steps = "evaluated", 1
    trainer.train_lora_from_args(args)
    plain = load_file(tmp_path / "output/plain/final/adapter.safetensors")
    evaluated = load_file(tmp_path / "output/evaluated/final/adapter.safetensors")
    for name in plain:
        torch.testing.assert_close(evaluated[name], plain[name], rtol=0, atol=0)
    report = json.loads((tmp_path / "output/evaluated/evaluation/step000002.json").read_text())
    assert report["baseline"]["validation"][0]["sample_id"] == "validation"
    assert report["candidate"]["validation"][0]["rgb_mae"] >= 0
