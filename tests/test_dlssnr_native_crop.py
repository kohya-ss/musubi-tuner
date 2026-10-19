"""NR no-upscale alignment crops must preserve the native paired pixel grid."""

import json

import numpy as np
import pytest
import toml
import torch
from PIL import Image
from safetensors.torch import load_file

from musubi_tuner.dlssnr.dataset import load_single_frame_manifest, load_temporal_manifest
from test_dlssnr_buckets import write_pairs
from test_dlssnr_training import make_args, small_math  # noqa: F401


def native_pairs(root, size, *, frames=1, png=False, layout="chw"):
    width, height = size
    path = write_pairs(root, [size], frames=frames, png=png)
    y, x = np.mgrid[:height, :width]
    checker = ((x + y) % 2).astype(np.float32)
    source = np.stack((checker, x.astype(np.float32) / (width - 1), y.astype(np.float32) / (height - 1)))
    if png:
        pixels = (source.transpose(1, 2, 0) * 255).astype(np.uint8)
        Image.fromarray(pixels).save(root / "source0.png")
        source = (pixels.astype(np.float32) / 255).transpose(2, 0, 1)
    else:
        np.save(root / "source0.npy", source)
    target = 1 - source
    controls = np.concatenate((source, checker[None] * 2 - 1, checker[None] / 128), axis=0)
    motion = np.stack(((x % 9).astype(np.float32) / 8 + 0.125, -(y % 7).astype(np.float32) / 4 - 0.25))
    history = (x % 4 >= 2)[None]
    temporal = (y % 5 >= 2)[None]
    loss = (x % 7).astype(np.float32)[None] / 8
    for name, value in (
        ("target0", target),
        ("controls0", controls),
        ("valid0", history),
        ("temporal0", temporal),
        ("loss0", loss),
    ):
        np.save(root / f"{name}.npy", value)
    np.save(root / "motion0.npy", motion if layout == "chw" else motion.transpose(1, 2, 0))
    row = json.loads(path.read_text())
    row["motion_layout"] = layout
    for frame in row["frames"][1:]:
        frame["temporal_valid_path"] = "temporal0.npy"
    path.write_text(json.dumps(row), encoding="utf-8")
    originals = dict(
        source=source,
        target=target,
        controls=controls,
        motion=motion,
        history_valid=history,
        temporal_valid=temporal,
        loss_mask=loss,
    )
    return path, {name: torch.from_numpy(value.copy()) for name, value in originals.items()}


@pytest.mark.parametrize("size", [(79, 63), (63, 79), (64, 63), (64, 48), (834, 1257)])
def test_in_budget_no_upscale_keeps_exact_pixels_in_the_center_crop(tmp_path, size):
    path, original = native_pairs(tmp_path, size)
    dataset = load_single_frame_manifest(path, 1024, 1024, enable_bucket=True, bucket_no_upscale=True)
    width, height = size
    bucket_width, bucket_height = width // 16 * 16, height // 16 * 16
    left, top = (width - bucket_width) // 2, (height - bucket_height) // 2
    frame = dataset[0]
    assert dataset.bucket_sizes == [(bucket_width, bucket_height)]
    for name in ("source", "target", "controls", "loss_mask"):
        expected = original[name][:, top : top + bucket_height, left : left + bucket_width]
        torch.testing.assert_close(frame[name], expected, rtol=0, atol=0)
    assert set(frame["source"][0].unique().tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("layout", ["chw", "hwc"])
@pytest.mark.parametrize("png", [False, True])
def test_native_crop_keeps_temporal_motion_masks_and_frame_alignment(tmp_path, layout, png):
    path, original = native_pairs(tmp_path, (79, 63), frames=3, png=png, layout=layout)
    dataset = load_temporal_manifest(path, 128, 128, 3, enable_bucket=True, bucket_no_upscale=True)
    frames = dataset[0]["frames"]
    streamed = list(dataset.iter_frames(0))
    for index, frame in enumerate(frames):
        for name, value in original.items():
            expected = value[:, 7:55, 7:71]
            if index == 0 and name in ("motion", "history_valid", "temporal_valid"):
                expected = torch.zeros_like(expected)
            torch.testing.assert_close(frame[name], expected, rtol=0, atol=0)
            torch.testing.assert_close(streamed[index][name], expected, rtol=0, atol=0)
    assert not frames[1]["motion"].eq(0).all()
    assert frames[1]["history_valid"].dtype == frames[1]["temporal_valid"].dtype == torch.bool


def test_exact_area_budget_also_uses_native_crop(tmp_path):
    path, original = native_pairs(tmp_path, (79, 63))
    frame = load_single_frame_manifest(path, 79, 63, enable_bucket=True, bucket_no_upscale=True)[0]
    torch.testing.assert_close(frame["source"], original["source"][:, 7:55, 7:71], rtol=0, atol=0)


@pytest.mark.parametrize("size,budget", [((79, 63), (64, 48)), ((200, 128), (128, 128))])
def test_over_budget_no_upscale_retains_cover_resize_and_motion_scaling(tmp_path, size, budget):
    path, original = native_pairs(tmp_path, size, frames=2)
    ordinary = load_temporal_manifest(path, *budget, 2, enable_bucket=True)[0]["frames"][1]
    capped = load_temporal_manifest(path, *budget, 2, enable_bucket=True, bucket_no_upscale=True)[0]["frames"][1]
    for name in original:
        torch.testing.assert_close(capped[name], ordinary[name], rtol=0, atol=0)
    assert ((capped["source"][0] > 0) & (capped["source"][0] < 1)).any()


def test_disabled_no_upscale_still_resizes_small_images_to_the_requested_budget(tmp_path):
    path, _ = native_pairs(tmp_path, (79, 63))
    frame = load_single_frame_manifest(path, 128, 128, enable_bucket=True, bucket_no_upscale=False)[0]
    assert frame["source"].shape[-1] > 79 and frame["source"].shape[-2] > 63
    assert ((frame["source"][0] > 0) & (frame["source"][0] < 1)).any()


def test_native_crop_rejects_labels_that_only_exist_in_the_trimmed_border(tmp_path):
    path, _ = native_pairs(tmp_path, (79, 63))
    mask = np.zeros((1, 63, 79), np.float32)
    mask[:, :7] = 1
    np.save(tmp_path / "loss0.npy", mask)
    dataset = load_single_frame_manifest(path, 128, 128, enable_bucket=True, bucket_no_upscale=True)
    with pytest.raises(ValueError, match="no supervised pixels"):
        dataset[0]


def test_native_crop_also_works_for_targetless_inference(tmp_path):
    path, original = native_pairs(tmp_path, (79, 63))
    row = json.loads(path.read_text())
    del row["frames"][0]["target_path"]
    del row["target_encoding"]
    path.write_text(json.dumps(row), encoding="utf-8")
    frame = load_single_frame_manifest(path, 128, 128, require_target=False, enable_bucket=True, bucket_no_upscale=True)[0]
    assert frame["target"] is None
    torch.testing.assert_close(frame["source"], original["source"][:, 7:55, 7:71], rtol=0, atol=0)


def test_native_crop_still_excludes_reprojection_beyond_the_new_history_boundary(tmp_path):
    from musubi_tuner.dlssnr.pipeline import _reproject

    path, _ = native_pairs(tmp_path, (79, 63), frames=2)
    previous, current = load_temporal_manifest(path, 128, 128, 2, enable_bucket=True, bucket_no_upscale=True)[0]["frames"]
    assert current["history_valid"][..., -1].all()
    warped, usable = _reproject(
        current["source"][None],
        previous["source"][None],
        current["motion"][None],
        current["history_valid"][None],
        torch.tensor([False]),
    )
    assert not usable[..., -1].any()
    torch.testing.assert_close(warped[..., -1], current["source"][None, ..., -1], rtol=0, atol=0)


@pytest.mark.usefixtures("small_math")
@pytest.mark.parametrize("lora,mode", [(False, "temporal"), (True, "single_frame")])
def test_native_cropped_data_trains_and_resumes_without_resampling(tmp_path, lora, mode):
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path, lora=lora, mode=mode)
    native_pairs(tmp_path, (79, 63), frames=3 if mode == "temporal" else 1)
    args.dataset_config.write_text(
        toml.dumps(
            {
                "general": {"resolution": 128, "enable_bucket": True, "bucket_no_upscale": True},
                "datasets": [{"train_manifest": "pairs.jsonl"}],
            }
        ),
        encoding="utf-8",
    )
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = load_file(folder / "final" / filename)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert state["identity"]["bucket_plan"]["buckets"][0]["resolution"] == [64, 48]
    assert "dataset.py" in state["identity"]["implementation"]
    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), expected, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    torch.testing.assert_close(resumed["optimizer"]["state"], state["optimizer"]["state"], rtol=0, atol=0)
