"""Dataset provenance for the MiniMax-H3 cache scripts: items carry the index of the dataset
they came from (stamped by the DatasetGroup), the per-dataset cache plan is looked up through
it, and the shared cache drivers let an architecture decide what "existing cache" means."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner import cache_latents, cache_text_encoder_outputs
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.image_video_dataset import DatasetGroup, ItemInfo, VideoDataset
from musubi_tuner.minimax_h3.cache_plan import (
    H3DatasetPlan,
    item_control_paths,
    item_crop_start,
    item_plan,
    item_record,
    plan_h3_datasets,
)
from musubi_tuner.minimax_h3.media import H3Record


class _FakeVideoDatasource:
    has_control = False
    audio_sources = None

    def set_bucket_selector(self, bucket_selector):
        pass

    def set_source_and_target_fps(self, source_fps, target_fps):
        pass

    def __iter__(self):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        fetch = lambda: ("clip.mp4", [frame.copy() for _ in range(5)], "caption", None)  # noqa: E731
        fetch.datasource_index = 0
        yield fetch


def _video_dataset(directory: Path) -> VideoDataset:
    directory.mkdir(exist_ok=True)
    dataset = VideoDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        target_frames=[5],
        video_directory=str(directory),
        cache_directory=str(directory),
        architecture=ARCHITECTURE_MINIMAX_H3,
    )
    dataset.datasource = _FakeVideoDatasource()
    return dataset


def test_dataset_group_stamps_the_dataset_index_that_items_carry(tmp_path: Path):
    datasets = [_video_dataset(tmp_path / "a"), _video_dataset(tmp_path / "b")]
    assert [dataset.dataset_index for dataset in datasets] == [None, None]

    DatasetGroup(datasets)

    assert [dataset.dataset_index for dataset in datasets] == [0, 1]
    _, items = next(iter(datasets[1].retrieve_latent_cache_batches(num_workers=1)))
    assert items[0].dataset_index == 1
    assert items[0].datasource_index == 0
    assert items[0].frame_pos == 0


def test_plan_requires_datasets_in_group_order(tmp_path: Path):
    dataset = _video_dataset(tmp_path / "a")
    with pytest.raises(ValueError, match="DatasetGroup position"):
        plan_h3_datasets([dataset], task="t2va", one_frame=False)


def _plan(*, is_image: bool, records=(), control_paths=None) -> H3DatasetPlan:
    return H3DatasetPlan(
        index=0,
        is_image=is_image,
        records=list(records),
        audio_sources=None,
        control_paths=control_paths or {},
        controls_as_references=False,
    )


def test_item_plan_checks_provenance_and_dataset_kind():
    plans = [_plan(is_image=False)]
    video_item = ItemInfo("clip.mp4", "caption", (64, 64), (64, 64), frame_count=5)
    with pytest.raises(ValueError, match="dataset provenance"):
        item_plan(plans, video_item)

    video_item.dataset_index = 0
    assert item_plan(plans, video_item) is plans[0]
    video_item.dataset_index = 1
    with pytest.raises(ValueError, match="dataset provenance"):
        item_plan(plans, video_item)

    # an image item (no frame_count) cannot come from a video dataset plan
    image_item = ItemInfo("image.png", "caption", (64, 64), (64, 64))
    image_item.dataset_index = 0
    with pytest.raises(ValueError, match="dataset kind"):
        item_plan(plans, image_item)
    assert item_plan([_plan(is_image=True)], image_item).is_image


def test_item_accessors_require_the_record_crop_and_control_provenance(tmp_path: Path):
    record = H3Record(video_path=tmp_path / "clip.mp4", caption="caption", references=())
    plan = _plan(is_image=False, records=[record])
    item = ItemInfo("clip.mp4", "caption", (64, 64), (64, 64), frame_count=5)

    with pytest.raises(ValueError, match="datasource provenance"):
        item_record(plan, item)
    with pytest.raises(ValueError, match="crop provenance"):
        item_crop_start(item)
    item.datasource_index = 0
    item.frame_pos = 17
    assert item_record(plan, item) is record
    assert item_crop_start(item) == 17

    image_plan = _plan(is_image=True, control_paths={"image.png": ["a.png", "b.png"]})
    image_item = ItemInfo("image.png", "caption", (64, 64), (64, 64))
    assert item_control_paths(image_plan, image_item, 2) == [Path("a.png").resolve(), Path("b.png").resolve()]
    with pytest.raises(ValueError, match="control paths"):
        item_control_paths(image_plan, image_item, 1)
    with pytest.raises(ValueError, match="control paths"):
        item_control_paths(image_plan, ItemInfo("other.png", "caption", (64, 64), (64, 64)), 2)


class _FakeCacheDataset:
    """The slice of BaseDataset the shared cache drivers touch."""

    def __init__(self, items: list[ItemInfo]):
        self.items = items

    def retrieve_latent_cache_batches(self, num_workers):
        yield (64, 64), list(self.items)

    def retrieve_text_encoder_output_cache_batches(self, num_workers):
        yield list(self.items)

    def get_all_latent_cache_files(self):
        return []


def _cache_items(tmp_path: Path) -> list[ItemInfo]:
    items = []
    for name in ("current", "stale"):
        item = ItemInfo(name, "caption", (64, 64), (64, 64), content=np.zeros((64, 64, 3), dtype=np.uint8))
        item.latent_cache_path = str(tmp_path / f"{name}_mmh3.safetensors")
        item.text_encoder_output_cache_path = str(tmp_path / f"{name}_mmh3_te.safetensors")
        items.append(item)
    return items


def test_latent_driver_skips_items_whose_cache_the_architecture_deems_current(tmp_path: Path):
    items = _cache_items(tmp_path)
    for item in items:
        Path(item.latent_cache_path).touch()  # both files exist; only the metadata check tells them apart
    args = SimpleNamespace(skip_existing=True, num_workers=1, batch_size=None, keep_cache=True)
    encoded = []

    cache_latents.encode_datasets(
        [_FakeCacheDataset(items)],
        lambda batch: encoded.extend(item.item_key for item in batch),
        args,
        cache_is_current=lambda item: item.item_key == "current",
    )
    assert encoded == ["stale"]

    # the default test stays "the cache file exists"
    encoded.clear()
    Path(items[1].latent_cache_path).unlink()
    cache_latents.encode_datasets([_FakeCacheDataset(items)], lambda batch: encoded.extend(item.item_key for item in batch), args)
    assert encoded == ["stale"]


def test_text_driver_skips_items_whose_cache_the_architecture_deems_current(tmp_path: Path):
    items = _cache_items(tmp_path)
    existing = {Path(item.text_encoder_output_cache_path).as_posix() for item in items}
    encoded = []

    def run(**kwargs):
        encoded.clear()
        cache_text_encoder_outputs.process_text_encoder_batches(
            1,
            True,
            None,
            [_FakeCacheDataset(items)],
            [{str(Path(path)) for path in existing}],
            [set()],
            lambda batch: encoded.extend(item.item_key for item in batch),
            **kwargs,
        )
        return list(encoded)

    assert run(cache_is_current=lambda item: item.item_key == "current") == ["stale"]
    # the default test stays set membership of the existing cache files
    assert run() == []
