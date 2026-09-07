"""The per-item extras accessor: architecture-specific consumers read the fields a dataset
carries beyond the shared schema through it, without knowing the datasource implementation."""

import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.architectures import ARCHITECTURE_QWEN_IMAGE
from musubi_tuner.dataset.datasources import (
    ImageDirectoryDatasource,
    ImageJsonlDatasource,
    ItemExtras,
    VideoDirectoryDatasource,
    VideoJsonlDatasource,
)
from musubi_tuner.dataset.image_video_dataset import ImageDataset


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_image_jsonl_extras_exclude_the_shared_schema_including_numbered_keys(tmp_path: Path):
    jsonl = tmp_path / "data" / "items.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "image_path": "a.png",
                "image_path_1": "a_1.png",
                "caption": "c",
                "control_path": "ctrl.png",
                "references": [{"type": "image", "path": "refs/face.png"}],
                "teacher_caption": "t",
                "custom": 1,
            },
            {"image_path": "b.png", "caption": "c"},
        ],
    )

    datasource = ImageJsonlDatasource(str(jsonl), control_count_per_image=None)

    first = datasource.get_item_extras(0)
    assert isinstance(first, ItemExtras)
    assert first.fields == {"references": [{"type": "image", "path": "refs/face.png"}], "teacher_caption": "t", "custom": 1}
    assert Path(first.base_directory) == jsonl.parent.resolve()
    assert first.label == "items.jsonl line 1"
    assert datasource.get_item_extras(1).fields == {}
    assert datasource.get_item_extras(1).label == "items.jsonl line 2"


def test_video_jsonl_extras_exclude_the_shared_schema(tmp_path: Path):
    jsonl = tmp_path / "videos.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "clip.mp4",
                "caption": "c",
                "control_path": "ctrl.mp4",
                "audio_path": "clip.wav",
                "references": [{"type": "video", "path": "ref.mp4"}],
            }
        ],
    )

    extras = VideoJsonlDatasource(str(jsonl)).get_item_extras(0)

    assert extras.fields == {"references": [{"type": "video", "path": "ref.mp4"}]}
    assert Path(extras.base_directory) == tmp_path.resolve()
    assert extras.label == "videos.jsonl line 1"


def test_directory_datasources_have_no_extras(tmp_path: Path):
    image = _touch(tmp_path / "images" / "a.png")
    (tmp_path / "images" / "a.txt").write_text("c", encoding="utf-8")
    video = _touch(tmp_path / "videos" / "a.mp4")
    (tmp_path / "videos" / "a.txt").write_text("c", encoding="utf-8")

    image_extras = ImageDirectoryDatasource(str(tmp_path / "images"), ".txt").get_item_extras(0)
    video_extras = VideoDirectoryDatasource(str(tmp_path / "videos"), ".txt").get_item_extras(0)

    assert image_extras == ItemExtras(fields={}, base_directory=str(tmp_path / "images"), label=str(image))
    assert video_extras == ItemExtras(fields={}, base_directory=str(tmp_path / "videos"), label=str(video))


def test_image_items_carry_their_datasource_index(tmp_path: Path):
    for name in ("first", "second"):
        Image.new("RGB", (64, 64)).save(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.txt").write_text(name, encoding="utf-8")

    dataset = ImageDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        image_directory=str(tmp_path),
        cache_directory=str(tmp_path / "cache"),
        architecture=ARCHITECTURE_QWEN_IMAGE,
    )

    fetchers = list(dataset.datasource)
    assert [fetcher.datasource_index for fetcher in fetchers] == [0, 1]

    items = [item for _, batch in dataset.retrieve_latent_cache_batches(num_workers=1) for item in batch]
    by_key = {item.item_key: item.datasource_index for item in items}
    assert by_key == {dataset.datasource.image_paths[0]: 0, dataset.datasource.image_paths[1]: 1}
