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
