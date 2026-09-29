"""Tests for Qwen-Image 2.1 cache writing and loading."""

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import (
    save_latent_cache_qwen_image_21,
    save_text_encoder_output_cache_qwen_image_21,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.qwen_image_21.qwen_image_21_utils import reference_fingerprints
from musubi_tuner.qwen_image_21_train_network import prepare_conditioning


@pytest.mark.parametrize("reference_count", [0, 1, 2])
def test_cache_round_trip_with_variable_text(tmp_path, reference_count):
    items = []
    references = [torch.randn(64, 1, 2, 4) for _ in range(reference_count)]
    hashes = reference_fingerprints([Image.new("RGBA", (64, 32), color) for color in ["red", "blue"][:reference_count]])
    for index, text_length in enumerate([3, 5]):
        item = ItemInfo(str(index), "edit", (32, 32), (32, 32), latent_cache_path=str(tmp_path / f"{index}_qi21.safetensors"))
        item.text_encoder_output_cache_path = str(tmp_path / f"{index}_qi21_te.safetensors")
        save_latent_cache_qwen_image_21(item, torch.randn(64, 1, 2, 2), references, hashes)
        save_text_encoder_output_cache_qwen_image_21(
            item,
            torch.randn(text_length, 4096),
            torch.arange(reference_count),
            torch.tensor([[2, 4]] * reference_count, dtype=torch.int64).reshape(-1, 2),
            hashes,
        )
        items.append(item)

    batch = BucketBatchManager({(32, 32): items}, batch_size=2)[0]
    text, lengths, tokens, shapes, slots = prepare_conditioning(batch, torch.device("cpu"), torch.float32)
    assert text.shape == (2, 5, 4096)
    assert lengths == [3, 5]
    assert len(tokens) == reference_count
    assert shapes == [(1, 2, 4)] * reference_count
    assert slots == [list(range(reference_count))] * 2
    for index, reference in enumerate(references):
        torch.testing.assert_close(batch[f"latents_control_{index}"], torch.stack([reference, reference]))


def test_text_recache_replaces_stale_keys(tmp_path):
    item = ItemInfo("sample", "edit", (32, 32), (32, 32))
    item.text_encoder_output_cache_path = str(tmp_path / "sample_qi21_te.safetensors")
    save_file(
        {"obsolete_float32": torch.zeros(1), "varlen_vl_embed_bfloat16": torch.zeros(1, 4096, dtype=torch.bfloat16)},
        item.text_encoder_output_cache_path,
    )
    save_text_encoder_output_cache_qwen_image_21(
        item,
        torch.ones(2, 4096),
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, 2, dtype=torch.int64),
        torch.empty(0, 32, dtype=torch.uint8),
    )
    state = load_file(item.text_encoder_output_cache_path)
    assert "obsolete_float32" not in state
    assert "varlen_vl_embed_bfloat16" not in state
    assert state["varlen_vl_embed_float32"].shape == (2, 4096)


def test_same_size_reference_order_rejected():
    images = [Image.new("RGBA", (32, 32), color) for color in ("red", "blue")]
    hashes = reference_fingerprints(images)
    batch = dict(
        vl_embed=[torch.randn(3, 4096)],
        latents_control_0=torch.randn(1, 64, 1, 2, 2),
        latents_control_1=torch.randn(1, 64, 1, 2, 2),
        image_slots=[torch.tensor([1, 2])],
        reference_grids=[torch.tensor([[2, 2], [2, 2]])],
        reference_hashes=[hashes],
        vlm_reference_hashes=[hashes.clone()],
    )
    prepare_conditioning(batch, "cpu", torch.float32)
    batch["vlm_reference_hashes"] = [reference_fingerprints(images[::-1])]
    with pytest.raises(ValueError, match="contents/order"):
        prepare_conditioning(batch, "cpu", torch.float32)
