"""Tests for caption dropout: config parsing, cache-path helpers, dropout selection at load time."""

import pytest

from musubi_tuner.dataset.config_utils import BaseDatasetParams


def test_base_dataset_params_default_caption_dropout_rate_is_zero():
    params = BaseDatasetParams()
    assert params.caption_dropout_rate == 0.0


def test_base_dataset_params_accepts_caption_dropout_rate():
    params = BaseDatasetParams(caption_dropout_rate=0.1)
    assert params.caption_dropout_rate == 0.1
