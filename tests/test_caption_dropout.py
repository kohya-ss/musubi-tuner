"""Tests for caption dropout: config parsing, cache-path helpers, dropout selection at load time."""

import pytest

from musubi_tuner.dataset.config_utils import BaseDatasetParams


def test_base_dataset_params_default_caption_dropout_rate_is_zero():
    params = BaseDatasetParams()
    assert params.caption_dropout_rate == 0.0


def test_base_dataset_params_accepts_caption_dropout_rate():
    params = BaseDatasetParams(caption_dropout_rate=0.1)
    assert params.caption_dropout_rate == 0.1


from musubi_tuner.dataset.image_video_dataset import EMPTY_CAPTION_CACHE_KEY, ImageDataset


def _make_image_dataset(tmp_path, caption_dropout_rate=0.0):
    return ImageDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=False,
        bucket_no_upscale=False,
        cache_directory=str(tmp_path),
        debug_dataset=False,
        architecture="wan",
        image_directory=str(tmp_path),
        caption_dropout_rate=caption_dropout_rate,
    )


def test_item_info_defaults_caption_dropout_fields():
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

    item = ItemInfo("key", "a caption", (64, 64))
    assert item.caption_dropout_rate == 0.0
    assert item.empty_text_encoder_output_cache_path is None


def test_base_dataset_stores_caption_dropout_rate(tmp_path):
    dataset = _make_image_dataset(tmp_path, caption_dropout_rate=0.15)
    assert dataset.caption_dropout_rate == 0.15


def test_get_empty_text_encoder_output_cache_path(tmp_path):
    dataset = _make_image_dataset(tmp_path)
    path = dataset.get_empty_text_encoder_output_cache_path()
    assert path == str(tmp_path / f"{EMPTY_CAPTION_CACHE_KEY}_wan_te.safetensors")


def test_get_empty_caption_item_info(tmp_path):
    dataset = _make_image_dataset(tmp_path)
    item = dataset.get_empty_caption_item_info()
    assert item.item_key == EMPTY_CAPTION_CACHE_KEY
    assert item.caption == ""
    assert item.text_encoder_output_cache_path == dataset.get_empty_text_encoder_output_cache_path()
