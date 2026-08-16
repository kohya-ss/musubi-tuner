import inspect
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_KREA2,
    ARCHITECTURE_MINIMAX_H3,
    ARCHITECTURE_MINIMAX_H3_FULL,
    ARCHITECTURE_QWEN_IMAGE,
    round_down_frame_count,
)
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.image_video_dataset import ImageDataset, VideoDataset
from musubi_tuner.training import trainer_base
from musubi_tuner.training.trainer_base import NetworkTrainer


def test_h3_architecture_and_bucket_step():
    assert ARCHITECTURE_MINIMAX_H3 == "mmh3"
    assert ARCHITECTURE_MINIMAX_H3_FULL == "minimax_h3"
    assert BucketSelector.ARCHITECTURE_STEPS_MAP[ARCHITECTURE_MINIMAX_H3] == 32


def test_h3_documentation_does_not_ask_users_to_set_source_fps():
    documentation = (ROOT / "docs" / "minimax_h3.md").read_text(encoding="utf-8")

    assert "source_fps = " not in documentation
    assert "`source_fps` is not needed" in documentation


@pytest.mark.parametrize(
    ("frames", "expected"),
    [(5, 5), (21, 5), (22, 22), (38, 22), (39, 39), (55, 39), (56, 56)],
)
def test_h3_frame_count_rounds_to_17n_plus_5(frames: int, expected: int):
    assert round_down_frame_count(frames, ARCHITECTURE_MINIMAX_H3, 4) == expected


def test_h3_frame_count_rejects_values_below_five():
    with pytest.raises(ValueError, match="at least 5 frames"):
        round_down_frame_count(4, ARCHITECTURE_MINIMAX_H3, 4)


def test_frame_helper_requires_stride_and_preserves_stride_one():
    assert inspect.signature(round_down_frame_count).parameters["vae_frame_stride"].default is inspect.Parameter.empty

    with pytest.raises(TypeError):
        round_down_frame_count(8, ARCHITECTURE_KREA2)

    assert round_down_frame_count(8, ARCHITECTURE_KREA2, 1) == 8
    assert round_down_frame_count(8, ARCHITECTURE_QWEN_IMAGE, 1) == 8


def test_h3_audio_dataset_survives_dataloader_worker_pickling(tmp_path: Path):
    # spawned DataLoader workers (Windows/macOS) pickle the dataset, including the AudioSpec
    from musubi_tuner.minimax_h3.media import H3_AUDIO_SPEC

    restored_spec = pickle.loads(pickle.dumps(H3_AUDIO_SPEC))
    assert restored_spec.samples_per_crop(22) == H3_AUDIO_SPEC.samples_per_crop(22) == 29600

    dataset = VideoDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        target_frames=[5],
        video_directory=str(tmp_path),
        cache_directory=str(tmp_path),
        architecture=ARCHITECTURE_MINIMAX_H3,
        audio_spec=H3_AUDIO_SPEC,
    )
    restored_dataset = pickle.loads(pickle.dumps(dataset))
    assert restored_dataset.audio_spec.samples_per_crop(5) == 6400


def test_h3_video_dataset_uses_24_fps_and_preserves_valid_target_frames(tmp_path: Path):
    dataset = VideoDataset(
        resolution=(768, 768),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        target_frames=[5, 22, 39, 56],
        video_directory=str(tmp_path),
        architecture=ARCHITECTURE_MINIMAX_H3,
    )

    assert dataset.target_fps == 24.0
    assert dataset.target_frames == (5, 22, 39, 56)


def test_h3_full_frame_extraction_preserves_valid_frame_count(tmp_path: Path):
    dataset = VideoDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        frame_extraction="full",
        target_frames=[5],
        max_frames=56,
        video_directory=str(tmp_path),
        cache_directory=str(tmp_path),
        architecture=ARCHITECTURE_MINIMAX_H3,
    )

    class FakeDatasource:
        has_control = False

        def set_bucket_selector(self, bucket_selector):
            self.bucket_selector = bucket_selector

        def set_source_and_target_fps(self, source_fps, target_fps):
            self.source_fps = source_fps
            self.target_fps = target_fps

        def __iter__(self):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            yield lambda: ("clip.mp4", [frame.copy() for _ in range(56)], "caption", None)

    dataset.datasource = FakeDatasource()

    batches = list(dataset.retrieve_latent_cache_batches(num_workers=1))

    assert len(batches) == 1
    bucket, items = batches[0]
    assert bucket == (64, 64, 56)
    assert items[0].frame_count == 56
    assert items[0].content.shape[0] == 56
    assert Path(items[0].text_encoder_output_cache_path).name == "clip_00000-056_mmh3_te.safetensors"


def _h3_image_dataset(tmp_path: Path, **overrides):
    parameters = dict(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        image_directory=str(tmp_path),
        cache_directory=str(tmp_path),
        architecture=ARCHITECTURE_MINIMAX_H3,
    )
    parameters.update(overrides)
    return ImageDataset(**parameters)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"multiple_target": True}, "multiple targets"),
        ({"no_resize_control": True}, "no_resize_control"),
        ({"control_resolution": (512, 512)}, "control_resolution"),
        ({"fp_1f_target_index": -1}, "nonnegative"),
        # time-annotated control validation: indices and controls must arrive together, with an
        # explicit target index and 1..2 nonnegative entries
        ({"control_directory": "controls"}, "require fp_1f_clean_indices"),
        ({"fp_1f_clean_indices": [0]}, "explicit fp_1f_target_index"),
        ({"fp_1f_clean_indices": [0], "fp_1f_target_index": 24}, "requires control images"),
        ({"fp_1f_clean_indices": [0, 1, 2], "fp_1f_target_index": 24}, "1 or 2 entries"),
        ({"fp_1f_clean_indices": [-1], "fp_1f_target_index": 24}, "nonnegative"),
    ],
)
def test_h3_image_dataset_rejects_unsupported_one_frame_features(tmp_path: Path, overrides: dict, message: str):
    with pytest.raises(ValueError, match=message):
        _h3_image_dataset(tmp_path, **overrides)


def test_h3_image_dataset_jsonl_control_paths_require_indices(tmp_path: Path):
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text('{"image_path": "target.png", "caption": "c", "control_path": "source.png"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="require fp_1f_clean_indices"):
        _h3_image_dataset(tmp_path, image_directory=None, image_jsonl_file=str(jsonl))


def test_h3_jsonl_datasource_exposes_control_paths_for_fingerprinting(tmp_path: Path):
    jsonl = tmp_path / "items.jsonl"
    jsonl.write_text(
        '{"image_path": "a.png", "caption": "c", "control_path": "s0.png", "control_path_1": "s1.png"}\n',
        encoding="utf-8",
    )
    dataset = _h3_image_dataset(
        tmp_path,
        image_directory=None,
        image_jsonl_file=str(jsonl),
        fp_1f_clean_indices=[0, 48],
        fp_1f_target_index=24,
    )

    assert dataset.datasource.get_control_paths() == {"a.png": ["s0.png", "s1.png"]}


def test_h3_image_dataset_with_controls_splits_buckets_and_forwards_indices(tmp_path: Path):
    dataset = _h3_image_dataset(
        tmp_path,
        control_directory=str(tmp_path / "ctrl"),
        fp_1f_clean_indices=[0],
        fp_1f_target_index=24,
    )
    assert dataset.has_control

    class FakeImageDatasource:
        has_control = True

        def __iter__(self):
            from PIL import Image

            image = Image.new("RGB", (64, 64))
            control = Image.new("RGB", (128, 128))  # resized to the bucket resolution
            yield lambda: (str(tmp_path / "target.png"), [image], "caption", [control])

    dataset.datasource = FakeImageDatasource()

    batches = list(dataset.retrieve_latent_cache_batches(num_workers=1))

    assert len(batches) == 1
    bucket, items = batches[0]
    # (W, H) + control count + per-control bucket-resized shape
    assert bucket == (64, 64, 1, 64, 64)
    assert items[0].fp_1f_clean_indices == [0]
    assert items[0].fp_1f_target_index == 24
    assert len(items[0].control_content) == 1
    assert items[0].control_content[0].shape == (64, 64, 3)


def test_h3_image_prepare_for_training_splits_buckets_by_control_count(tmp_path: Path):
    from safetensors.torch import save_file

    for stem in ("edit", "plain"):
        # the control-key scan reads real safetensors headers, so the caches must be well-formed
        save_file({"latents": torch.zeros(1)}, str(tmp_path / f"{stem}_0064x0064_mmh3.safetensors"))
        save_file({"rows": torch.zeros(1)}, str(tmp_path / f"{stem}_mmh3_te.safetensors"))
    with_controls = _h3_image_dataset(
        tmp_path,
        control_directory=str(tmp_path / "ctrl"),
        fp_1f_clean_indices=[0, 48],
        fp_1f_target_index=24,
    )
    without_controls = _h3_image_dataset(tmp_path)

    with_controls.prepare_for_training()
    without_controls.prepare_for_training()

    assert set(with_controls.batch_manager.buckets) == {(64, 64, 2)}
    assert set(without_controls.batch_manager.buckets) == {(64, 64)}


def test_h3_image_dataset_forwards_the_target_index_to_items(tmp_path: Path):
    dataset = _h3_image_dataset(tmp_path, fp_1f_target_index=24)

    class FakeImageDatasource:
        has_control = False

        def __iter__(self):
            from PIL import Image

            image = Image.new("RGB", (64, 64))
            yield lambda: (str(tmp_path / "portrait.png"), [image], "caption", None)

    dataset.datasource = FakeImageDatasource()

    batches = list(dataset.retrieve_latent_cache_batches(num_workers=1))

    assert len(batches) == 1
    bucket, items = batches[0]
    assert bucket == (64, 64)
    assert items[0].fp_1f_target_index == 24
    assert items[0].content.shape == (64, 64, 3)
    assert Path(items[0].latent_cache_path).name == "portrait_0064x0064_mmh3.safetensors"
    assert Path(items[0].text_encoder_output_cache_path).name == "portrait_mmh3_te.safetensors"


def test_h3_prepare_for_training_pairs_each_crop_with_its_own_text_cache(tmp_path: Path):
    dataset = VideoDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        target_frames=[5],
        video_directory=str(tmp_path),
        cache_directory=str(tmp_path),
        architecture=ARCHITECTURE_MINIMAX_H3,
    )
    latent_cache = tmp_path / "clip_00022-005_0064x0064_mmh3.safetensors"
    text_cache = tmp_path / "clip_00022-005_mmh3_te.safetensors"
    latent_cache.touch()
    text_cache.touch()

    dataset.prepare_for_training()

    assert dataset.num_train_items == 1
    item = next(iter(dataset.batch_manager.buckets.values()))[0]
    assert Path(item.text_encoder_output_cache_path) == text_cache


def test_h3_training_sample_preserves_valid_frame_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class H3Trainer(NetworkTrainer):
        @property
        def architecture(self):
            return ARCHITECTURE_MINIMAX_H3

        @property
        def architecture_full_name(self):
            return ARCHITECTURE_MINIMAX_H3_FULL

        def do_inference(self, *args, **kwargs):
            frame_count = args[10]
            captured["frame_count"] = frame_count
            return torch.zeros(1, 3, frame_count, 8, 8)

    class FakeAccelerator:
        device = torch.device("cpu")

        def get_tracker(self, name):
            raise ValueError(name)

    monkeypatch.setattr(trainer_base, "save_videos_grid", lambda *args, **kwargs: None)

    trainer = H3Trainer()
    trainer._i2v_training = False
    trainer._control_training = False
    trainer.default_guidance_scale = 1.0
    args = SimpleNamespace(output_name="sample")

    trainer.sample_image_inference(
        FakeAccelerator(),
        args,
        torch.nn.Identity(),
        torch.bfloat16,
        torch.nn.Identity(),
        str(tmp_path),
        {"frame_count": 56, "seed": 123},
        epoch=None,
        steps=0,
    )

    assert captured["frame_count"] == 56
