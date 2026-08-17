import numpy as np
import pytest
import torch

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.image_video_dataset import ItemInfo, VideoDataset
from musubi_tuner.detect_watermark_mask import (
    bounding_boxes,
    consistent_gradient,
    detect_watermark_region,
    make_border_mask,
    trimmed_temporal_std,
)
from musubi_tuner.training.trainer_base import reduce_loss


# region detection


def _synthetic_frames(height=64, width=96, n_frames=16, watermark_box=(50, 60, 80, 90)):
    """Moving content everywhere, plus a constant bright block (the "watermark")."""
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(n_frames, height, width)).astype(np.float32)
    y0, y1, x0, x1 = watermark_box
    frames[:, y0:y1, x0:x1] = 200.0
    return frames


def test_detect_watermark_region_finds_static_block():
    # the deviation signal on its own: an opaque block has a temporal std of 0
    frames = _synthetic_frames()
    watermark = detect_watermark_region(frames, dilate=0, min_area=0, gradient_threshold=0.0)

    assert watermark[50:60, 80:90].all()
    # nothing outside the block, the moving content has a high temporal std
    outside = watermark.copy()
    outside[50:60, 80:90] = False
    assert not outside.any()


def test_detect_watermark_region_corner_only_ignores_center():
    # a static block in the middle of the frame is not a watermark under --corner_only
    frames = _synthetic_frames(watermark_box=(28, 36, 44, 52))
    watermark = detect_watermark_region(frames, corner_only=True, dilate=0, min_area=0, gradient_threshold=0.0)
    assert not watermark.any()


def test_detect_watermark_region_tolerates_frames_without_the_watermark():
    # the watermark is missing from 3 of 30 frames (10%): a plain temporal std would be
    # dominated by those frames and miss it entirely
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(30, 64, 96)).astype(np.float32)
    frames[:, 50:60, 80:90] = 200.0
    frames[:3, 50:60, 80:90] = 20.0  # the watermark is not on screen in these frames

    strict = detect_watermark_region(frames, dilate=0, min_area=0, frame_tolerance=0.0, gradient_threshold=0.0)
    assert not strict[50:60, 80:90].any()

    tolerant = detect_watermark_region(frames, dilate=0, min_area=0, frame_tolerance=0.1, gradient_threshold=0.0)
    assert tolerant[50:60, 80:90].all()

    outside = tolerant.copy()
    outside[50:60, 80:90] = False
    assert not outside.any()  # tolerating dropped frames must not start matching moving content


def test_trimmed_temporal_std_matches_plain_std_without_tolerance():
    rng = np.random.default_rng(1)
    frames = rng.normal(128.0, 20.0, size=(12, 8, 8)).astype(np.float32)
    assert np.allclose(trimmed_temporal_std(frames, 0.0), frames.std(axis=0), atol=1e-4)


def test_trimmed_temporal_std_keeps_all_frames_of_a_constant_pixel():
    # ties must not be broken arbitrarily: an exactly constant pixel stays at std 0
    frames = np.full((10, 4, 4), 77.0, dtype=np.float32)
    assert np.allclose(trimmed_temporal_std(frames, 0.3), 0.0)


def _translucent_frames(alpha, n_frames=24, height=96, width=160):
    """Moving textured content with an alpha-blended static logo, plus a near-static region.

    The near-static patch is deliberate bait: it is what a "low temporal deviation means
    watermark" rule would wrongly pick up, so it must stay out of the result.
    """
    import cv2

    rng = np.random.default_rng(2)
    logo = np.zeros((height, width), np.uint8)
    cv2.rectangle(logo, (width - 45, height - 30), (width - 10, height - 10), 255, 2)
    logo = logo.astype(np.float32) / 255.0

    # near-static "sky", feathered into the moving content so it has no hard edge of its own
    # (a static hard edge is a consistent gradient, and would be picked up as a watermark)
    sky = np.zeros((height, width), np.float32)
    sky[:28, :44] = 1.0
    sky = cv2.GaussianBlur(sky, (0, 0), 6)

    frames = np.empty((n_frames, height, width), np.float32)
    for i in range(n_frames):
        content = cv2.GaussianBlur(rng.random((height, width)).astype(np.float32), (0, 0), 3)
        content = cv2.normalize(content, None, 0, 255, cv2.NORM_MINMAX)  # full-contrast texture
        content = np.roll(content, i * 7, axis=1)
        content = (1.0 - sky) * content + sky * (190.0 + rng.normal(0, 2, (height, width)))
        frames[i] = alpha * logo * 255.0 + (1.0 - alpha * logo) * np.clip(content, 0, 255)

    return frames, logo > 0.5


def test_translucent_watermark_is_missed_by_deviation_but_found_by_the_gradient_pass():
    frames, logo = _translucent_frames(alpha=0.3)

    # a 30% opacity overlay only scales the temporal deviation by 1 - alpha, nowhere near
    # the threshold, so the deviation pass alone cannot see it
    deviation_only = detect_watermark_region(frames, dilate=0, min_area=0, gradient_threshold=0.0)
    assert deviation_only[logo].mean() < 0.1

    both_signals = detect_watermark_region(frames, dilate=0, min_area=0)
    assert both_signals[logo].all()


def test_gradient_signal_does_not_claim_a_near_static_background():
    # a feathered static region has a consistent gradient too, but a far weaker one; the
    # gradient signal is isolated here because the deviation signal legitimately flags static
    # regions (that is what --corner_only and --max_coverage exist for)
    frames, _ = _translucent_frames(alpha=0.3)
    assert not consistent_gradient(frames, gradient_threshold=3.0)[:20, :36].any()


def test_detection_covers_the_interior_of_an_outlined_shape():
    # the gradient only marks the rectangle's edges; the bounding box has to cover the inside
    frames, _ = _translucent_frames(alpha=0.5)
    detected = detect_watermark_region(frames, dilate=0)

    height, width = detected.shape
    interior = np.zeros((height, width), bool)
    interior[height - 27 : height - 13, width - 42 : width - 13] = True
    assert detected[interior].all()


def test_bounding_boxes_covers_a_hollow_shape():
    import cv2

    ring = np.zeros((40, 40), np.uint8)
    cv2.circle(ring, (20, 20), 12, 1, 2)
    boxed = bounding_boxes(ring, min_area=0)

    assert boxed[20, 20] == 1  # the hole is covered
    assert boxed[8:33, 8:33].all()  # and so is the whole box around the circle
    assert boxed[0, 0] == 0  # but nothing outside it


def test_bounding_boxes_keeps_separate_regions_separate():
    # two logos in opposite corners must not merge into one box spanning the frame
    mask = np.zeros((40, 40), np.uint8)
    mask[2:6, 2:6] = 1
    mask[34:38, 34:38] = 1
    boxed = bounding_boxes(mask, min_area=0)

    assert boxed[2:6, 2:6].all() and boxed[34:38, 34:38].all()
    assert not boxed[18:22, 18:22].any()


def test_bounding_boxes_drops_regions_below_min_area():
    mask = np.zeros((40, 40), np.uint8)
    mask[2:4, 2:4] = 1  # 4 px
    mask[20:30, 20:30] = 1  # 100 px
    boxed = bounding_boxes(mask, min_area=32)

    assert not boxed[2:4, 2:4].any()
    assert boxed[20:30, 20:30].all()


def _detect_args(tmp_path, **overrides):
    from types import SimpleNamespace

    defaults = dict(
        video_dir=str(tmp_path),
        recursive=False,
        n_frames=16,
        threshold=8.0,
        corner_only=False,
        corner_margin=0.25,
        dilate=0,
        min_area=0,
        frame_tolerance=0.1,
        gradient_threshold=3.0,
        max_coverage=0.4,
        suffix="_wmask.png",
        output_dir=None,
        overwrite=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _write_synthetic_video(path, static_background: bool):
    import cv2

    rng = np.random.default_rng(0)
    height, width, n_frames = 64, 96, 24
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height))
    background = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    for _ in range(n_frames):
        frame = background.copy() if static_background else rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        frame[50:60, 80:90] = 200  # the static "watermark"
        writer.write(frame)
    writer.release()


def test_process_video_writes_a_mask(tmp_path):
    from musubi_tuner.detect_watermark_mask import process_video

    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, static_background=False)

    assert process_video(str(video_path), _detect_args(tmp_path)) == "written"
    assert (tmp_path / "clip_wmask.png").exists()


def test_process_video_refuses_to_mask_more_than_max_coverage(tmp_path):
    # a locked-off shot is temporally static everywhere, so the detection is meaningless:
    # the boundary must stop it from masking most of the frame
    from musubi_tuner.detect_watermark_mask import process_video

    video_path = tmp_path / "static.mp4"
    _write_synthetic_video(video_path, static_background=True)

    assert process_video(str(video_path), _detect_args(tmp_path)) == "over_max_coverage"
    assert not (tmp_path / "static_wmask.png").exists()

    # raising the limit lets the same detection through, so the guard is what blocked it
    assert process_video(str(video_path), _detect_args(tmp_path, max_coverage=1.0)) == "written"


def test_process_video_skips_an_existing_mask_unless_overwriting(tmp_path):
    from musubi_tuner.detect_watermark_mask import process_video

    video_path = tmp_path / "clip.mp4"
    _write_synthetic_video(video_path, static_background=False)
    (tmp_path / "clip_wmask.png").write_bytes(b"")

    assert process_video(str(video_path), _detect_args(tmp_path)) == "exists"
    assert process_video(str(video_path), _detect_args(tmp_path, overwrite=True)) == "written"


def test_make_border_mask_keeps_only_the_perimeter():
    border = make_border_mask(100, 100, 0.25)
    assert border[0, 0] and border[99, 99]
    assert not border[50, 50]
    assert border.mean() == pytest.approx(1.0 - 0.5 * 0.5)


# endregion detection

# region loss reduction


def test_reduce_loss_without_mask_is_plain_mean():
    loss = torch.rand(2, 4, 3, 8, 8)
    assert torch.allclose(reduce_loss(loss, {}), loss.mean())
    assert torch.allclose(reduce_loss(loss, None), loss.mean())
    assert torch.allclose(reduce_loss(loss, {"watermark_mask": None}), loss.mean())


def test_reduce_loss_with_open_mask_matches_mean():
    loss = torch.rand(2, 4, 3, 8, 8)
    mask = torch.ones(2, 64, 64)
    assert torch.allclose(reduce_loss(loss, {"watermark_mask": mask}), loss.mean(), atol=1e-6)


def test_reduce_loss_excludes_masked_region():
    # right half of the latent grid is masked out; put a huge loss there and check it is ignored
    loss = torch.ones(1, 4, 3, 8, 8)
    loss[..., 4:] = 1000.0

    mask = torch.ones(1, 64, 64)
    mask[:, :, 32:] = 0.0

    assert torch.allclose(reduce_loss(loss, {"watermark_mask": mask}), torch.tensor(1.0), atol=1e-4)


def test_reduce_loss_is_per_sample():
    loss = torch.ones(2, 1, 1, 4, 4)
    loss[1] = 3.0

    mask = torch.ones(2, 32, 32)
    mask[1] = 0.0  # second sample fully masked out -> only the first one contributes

    assert torch.allclose(reduce_loss(loss, {"watermark_mask": mask}), torch.tensor(1.0), atol=1e-4)


def test_reduce_loss_accepts_unbatched_mask_and_4d_loss():
    loss = torch.ones(2, 4, 8, 8)
    loss[..., 4:] = 5.0

    mask = torch.ones(64, 64)
    mask[:, 32:] = 0.0

    assert torch.allclose(reduce_loss(loss, {"watermark_mask": mask}), torch.tensor(1.0), atol=1e-4)


def test_reduce_loss_ignores_mask_for_non_spatial_loss():
    # patchified / sequence-shaped losses have no spatial layout: fall back to mean()
    loss = torch.rand(2, 256, 64)
    mask = torch.ones(2, 64, 64)
    assert torch.allclose(reduce_loss(loss, {"watermark_mask": mask}), loss.mean())


# endregion loss reduction

# region dataset plumbing


def _make_dataset(tmp_path, **kwargs):
    return VideoDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=False,
        bucket_no_upscale=False,
        target_frames=[1],
        video_directory=str(tmp_path),
        architecture="wan",
        **kwargs,
    )


def test_find_watermark_mask_path(tmp_path):
    dataset = _make_dataset(tmp_path)
    assert dataset.find_watermark_mask_path("clip", str(tmp_path)) is None

    mask_path = tmp_path / "clip_wmask.png"
    mask_path.write_bytes(b"")
    assert dataset.find_watermark_mask_path("clip", str(tmp_path)) == str(mask_path)


def test_find_watermark_mask_path_honours_custom_suffix(tmp_path):
    dataset = _make_dataset(tmp_path, watermark_mask_suffix="_logo.png")
    (tmp_path / "clip_wmask.png").write_bytes(b"")
    assert dataset.find_watermark_mask_path("clip", str(tmp_path)) is None

    (tmp_path / "clip_logo.png").write_bytes(b"")
    assert dataset.find_watermark_mask_path("clip", str(tmp_path)) == str(tmp_path / "clip_logo.png")


def test_find_watermark_mask_path_disabled(tmp_path):
    dataset = _make_dataset(tmp_path, watermark_mask_suffix=None)
    (tmp_path / "clip_wmask.png").write_bytes(b"")
    assert dataset.find_watermark_mask_path("clip", str(tmp_path)) is None


def _write_item(tmp_path, name, mask: bool):
    from safetensors.torch import save_file

    latent_path = tmp_path / f"{name}.safetensors"
    save_file({"latents_1x8x8_float32": torch.zeros(4, 1, 8, 8)}, str(latent_path))
    te_path = tmp_path / f"{name}_te.safetensors"
    save_file({"context": torch.zeros(4, 8)}, str(te_path))

    item = ItemInfo(name, "", (64, 64), (64, 64, 1), frame_count=1, latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(te_path)
    if mask:
        from PIL import Image

        mask_array = np.full((64, 64), 255, dtype=np.uint8)
        mask_array[:, 32:] = 0
        mask_path = tmp_path / f"{name}_wmask.png"
        Image.fromarray(mask_array).save(mask_path)
        item.watermark_mask_path = str(mask_path)
    return item


def test_batch_manager_omits_key_without_masks(tmp_path):
    item = _write_item(tmp_path, "no_mask", mask=False)
    batch = BucketBatchManager({(64, 64, 1): [item]}, batch_size=1)[0]
    assert "watermark_mask" not in batch


def test_batch_manager_loads_mask(tmp_path):
    item = _write_item(tmp_path, "with_mask", mask=True)
    batch = BucketBatchManager({(64, 64, 1): [item]}, batch_size=1)[0]

    mask = batch["watermark_mask"]
    assert mask.shape == (1, 64, 64)
    assert mask.dtype == torch.float32
    assert mask[0, :, :32].eq(1.0).all()
    assert mask[0, :, 32:].eq(0.0).all()


def test_batch_manager_fills_missing_masks_with_ones(tmp_path):
    items = [_write_item(tmp_path, "a", mask=True), _write_item(tmp_path, "b", mask=False)]
    batch = BucketBatchManager({(64, 64, 1): items}, batch_size=2)[0]

    mask = batch["watermark_mask"]
    assert mask.shape == (2, 64, 64)
    assert mask[1].eq(1.0).all()  # the item without a mask is trained on in full


# endregion dataset plumbing

# region end-to-end


def test_prepare_for_training_picks_up_the_mask_and_masks_the_loss(tmp_path):
    from PIL import Image
    from safetensors.torch import save_file

    # cache files as written by wan_cache_latents.py: "{stem}_{frame_pos}-{frame_count}_{WxH}_{arch}.safetensors"
    save_file({"latents_1x8x8_float32": torch.zeros(4, 1, 8, 8)}, str(tmp_path / "clip_00000-001_0064x0064_wan.safetensors"))
    save_file({"context": torch.zeros(4, 8)}, str(tmp_path / "clip_wan_te.safetensors"))

    # bottom-right quarter of the frame is the watermark
    mask_array = np.full((64, 64), 255, dtype=np.uint8)
    mask_array[32:, 32:] = 0
    Image.fromarray(mask_array).save(tmp_path / "clip_wmask.png")

    dataset = _make_dataset(tmp_path, cache_directory=str(tmp_path))
    dataset.prepare_for_training()
    batch = dataset.batch_manager[0]

    assert batch["watermark_mask"].shape == (1, 64, 64)

    # the masked quarter of the latent grid is ignored, so its loss does not reach the scalar
    loss = torch.ones(1, 16, 1, 8, 8)
    loss[..., 4:, 4:] = 1000.0
    assert torch.allclose(reduce_loss(loss, batch), torch.tensor(1.0), atol=1e-4)


def test_top_level_entrypoint_exists():
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "detect_watermark_mask.py"
    assert script.read_text(encoding="utf-8") == (
        'from musubi_tuner.detect_watermark_mask import main\n\nif __name__ == "__main__":\n    main()\n'
    )


# endregion end-to-end
