"""Tests for caption dropout: config parsing, cache-path helpers, dropout selection at load time."""

import pytest

from musubi_tuner.dataset.config_utils import BaseDatasetParams


def test_base_dataset_params_default_caption_dropout_rate_is_zero():
    params = BaseDatasetParams()
    assert params.caption_dropout_rate == 0.0


def test_base_dataset_params_accepts_caption_dropout_rate():
    params = BaseDatasetParams(caption_dropout_rate=0.1)
    assert params.caption_dropout_rate == 0.1


from musubi_tuner.dataset.image_video_dataset import EMPTY_CAPTION_CACHE_KEY, ImageDataset, ItemInfo


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


import torch
from safetensors.torch import save_file


def _write_minimal_wan_caches(tmp_path, item_key="item0", write_empty=True):
    # latent cache: filename encodes item_key + WxH + architecture
    latent = torch.zeros(16, 1, 8, 8)
    save_file({"latents_1x8x8_fp32": latent}, str(tmp_path / f"{item_key}_0064x0064_wan.safetensors"))
    # text-encoder cache for the real caption
    save_file({"varlen_t5_fp32": torch.zeros(4, 16)}, str(tmp_path / f"{item_key}_wan_te.safetensors"))
    if write_empty:
        save_file({"varlen_t5_fp32": torch.zeros(4, 16)}, str(tmp_path / f"{EMPTY_CAPTION_CACHE_KEY}_wan_te.safetensors"))


def test_prepare_for_training_sets_dropout_fields_on_items(tmp_path):
    _write_minimal_wan_caches(tmp_path)
    dataset = _make_image_dataset(tmp_path, caption_dropout_rate=0.2)
    dataset.prepare_for_training()
    bucket = next(iter(dataset.batch_manager.buckets.values()))
    assert bucket[0].caption_dropout_rate == 0.2
    assert bucket[0].empty_text_encoder_output_cache_path == dataset.get_empty_text_encoder_output_cache_path()


def test_prepare_for_training_raises_if_empty_cache_missing(tmp_path):
    _write_minimal_wan_caches(tmp_path, write_empty=False)
    dataset = _make_image_dataset(tmp_path, caption_dropout_rate=0.2)
    with pytest.raises(FileNotFoundError):
        dataset.prepare_for_training()


def test_prepare_for_training_leaves_dropout_fields_default_when_rate_zero(tmp_path):
    _write_minimal_wan_caches(tmp_path, write_empty=False)
    dataset = _make_image_dataset(tmp_path, caption_dropout_rate=0.0)
    dataset.prepare_for_training()  # must not raise even though empty cache is absent
    bucket = next(iter(dataset.batch_manager.buckets.values()))
    assert bucket[0].caption_dropout_rate == 0.0
    assert bucket[0].empty_text_encoder_output_cache_path is None


from unittest.mock import patch

from musubi_tuner.dataset.bucket import BucketBatchManager


def _item_with_dropout(tmp_path, rate, real_value, empty_value):
    real_path = tmp_path / "real_wan_te.safetensors"
    empty_path = tmp_path / "empty_wan_te.safetensors"
    save_file({"varlen_t5_float32": torch.full((1, 4), real_value)}, str(real_path))
    save_file({"varlen_t5_float32": torch.full((1, 4), empty_value)}, str(empty_path))

    latent_path = tmp_path / "real_0064x0064_wan.safetensors"
    save_file({"latents_1x8x8_fp32": torch.zeros(16, 1, 8, 8)}, str(latent_path))

    item = ItemInfo("real", "a caption", (64, 64), (64, 64), latent_cache_path=str(latent_path))
    item.text_encoder_output_cache_path = str(real_path)
    item.caption_dropout_rate = rate
    item.empty_text_encoder_output_cache_path = str(empty_path)
    return item


def test_bucket_batch_manager_uses_empty_cache_when_dropout_fires(tmp_path):
    item = _item_with_dropout(tmp_path, rate=0.5, real_value=1.0, empty_value=0.0)
    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    with patch("musubi_tuner.dataset.bucket.random.random", return_value=0.1):  # < 0.5 -> dropout fires
        batch = manager[0]
    assert torch.all(batch["t5"][0] == 0.0)


def test_bucket_batch_manager_uses_real_cache_when_dropout_does_not_fire(tmp_path):
    item = _item_with_dropout(tmp_path, rate=0.5, real_value=1.0, empty_value=0.0)
    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    with patch("musubi_tuner.dataset.bucket.random.random", return_value=0.9):  # >= 0.5 -> no dropout
        batch = manager[0]
    assert torch.all(batch["t5"][0] == 1.0)


def test_bucket_batch_manager_no_dropout_when_rate_zero(tmp_path):
    item = _item_with_dropout(tmp_path, rate=0.0, real_value=1.0, empty_value=0.0)
    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    with patch("musubi_tuner.dataset.bucket.random.random", return_value=0.0):  # would fire if rate were > 0
        batch = manager[0]
    assert torch.all(batch["t5"][0] == 1.0)
