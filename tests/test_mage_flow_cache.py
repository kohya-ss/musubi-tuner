import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import save_latent_cache_mage_flow, save_text_encoder_output_cache_mage_flow
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
from musubi_tuner.mage_flow.mage_vae import posterior_seed, sample_posterior
from musubi_tuner.mage_flow_cache_latents import encode_and_save_batch


class FakeMageVAE:
    device = torch.device("cpu")
    dtype = torch.float32
    latent_channels = 128
    downsample_factor = 16

    def encode_moments(self, pixels):
        height, width = pixels.shape[-2:]
        means = pixels.mean(dim=(1, 2, 3), keepdim=True).expand(-1, 128, height // 16, width // 16).clone()
        logvars = torch.zeros_like(means)
        return means, logvars


def make_item(tmp_path: Path, key: str, value: int, controls=()):
    image = np.full((16, 32, 3), value, dtype=np.uint8)
    item = ItemInfo(
        item_key=key,
        caption="caption",
        original_size=(32, 16),
        bucket_size=(32, 16),
        content=image,
        latent_cache_path=str(tmp_path / f"{key}.safetensors"),
    )
    item.control_content = [np.full((16, 32, 3), control, dtype=np.uint8) for control in controls] or None
    return item


def test_bucket_manager_keeps_varlen_text_and_groups_edit_control_shapes(tmp_path):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()

    def save_cached_item(key, text_length, control_shapes):
        item = ItemInfo(
            item_key=key,
            caption="caption",
            original_size=(32, 32),
            bucket_size=(32, 32),
            latent_cache_path=str(cache_dir / f"{key}_0032x0032_mfe.safetensors"),
        )
        item.text_encoder_output_cache_path = str(cache_dir / f"{key}_mfe_te.safetensors")
        controls = [torch.zeros(128, height, width, dtype=torch.bfloat16) for height, width in control_shapes]
        save_latent_cache_mage_flow(
            item,
            torch.zeros(128, 2, 2, dtype=torch.bfloat16),
            controls,
            is_edit=True,
        )
        save_text_encoder_output_cache_mage_flow(
            item,
            torch.zeros(text_length, 2560, dtype=torch.bfloat16),
            is_edit=True,
        )

    save_cached_item("a", 3, [(1, 2), (1, 1)])
    save_cached_item("b", 5, [(1, 2), (1, 1)])
    save_cached_item("c", 4, [(2, 1), (1, 1)])
    dataset = ImageDataset(
        resolution=(32, 32),
        caption_extension=None,
        batch_size=2,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        image_directory=str(image_dir),
        cache_directory=str(cache_dir),
        architecture="mfe",
    )

    dataset.prepare_for_training()

    assert len(dataset.batch_manager.buckets) == 2
    batches = [dataset.batch_manager[index] for index in range(len(dataset.batch_manager))]
    paired = next(batch for batch in batches if batch["latents"].shape[0] == 2)
    assert [embedding.shape[0] for embedding in paired["mage_flow_embed"]] == [3, 5]
    assert paired["latents_control_0"].shape == (2, 128, 1, 2)
    assert paired["latents_control_1"].shape == (2, 128, 1, 1)


def test_mage_training_cache_rejects_mode_metadata_mismatch(tmp_path):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    item = ItemInfo(
        item_key="renamed",
        caption="caption",
        original_size=(32, 32),
        bucket_size=(32, 32),
        latent_cache_path=str(cache_dir / "renamed_0032x0032_mfe.safetensors"),
    )
    item.text_encoder_output_cache_path = str(cache_dir / "renamed_mfe_te.safetensors")
    save_latent_cache_mage_flow(
        item,
        torch.zeros(128, 2, 2, dtype=torch.bfloat16),
        None,
        is_edit=False,
    )
    save_text_encoder_output_cache_mage_flow(
        item,
        torch.zeros(3, 2560, dtype=torch.bfloat16),
        is_edit=True,
    )
    dataset = ImageDataset(
        resolution=(32, 32),
        caption_extension=None,
        batch_size=1,
        num_repeats=1,
        enable_bucket=True,
        bucket_no_upscale=False,
        image_directory=str(image_dir),
        cache_directory=str(cache_dir),
        architecture="mfe",
    )

    with pytest.raises(ValueError, match=r"cache architecture mismatch.*expected mage_flow_edit.*got mage_flow"):
        dataset.prepare_for_training()


def test_posterior_seed_uses_stable_sha256_identity():
    identity = "mage_flow\0item-7\0target".encode("utf-8")
    expected = (int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") + 42) % (2**63 - 1)

    assert posterior_seed("mage_flow", "item-7", "target", 42) == expected
    assert posterior_seed("mage_flow", "item-7", "target", 42) == expected
    assert posterior_seed("mage_flow", "item-7", "control:0", 42) != expected


def test_sample_posterior_uses_only_the_supplied_generator():
    mean = torch.zeros(1, 2, 2, 2)
    logvar = torch.zeros_like(mean)
    torch.manual_seed(999)
    before = torch.random.get_rng_state()
    first = sample_posterior(mean, logvar, torch.Generator().manual_seed(123))
    after = torch.random.get_rng_state()
    second = sample_posterior(mean, logvar, torch.Generator().manual_seed(123))

    assert torch.equal(before, after)
    torch.testing.assert_close(first, second)


def test_cache_posterior_is_stable_across_item_order_and_batching(tmp_path):
    vae = FakeMageVAE()
    first_a = make_item(tmp_path / "first", "a", 10)
    first_b = make_item(tmp_path / "first", "b", 20)
    second_a = make_item(tmp_path / "second", "a", 10)
    second_b = make_item(tmp_path / "second", "b", 20)
    Path(first_a.latent_cache_path).parent.mkdir()
    Path(second_a.latent_cache_path).parent.mkdir()

    encode_and_save_batch(vae, [first_a, first_b], is_edit=False, seed=77)
    encode_and_save_batch(vae, [second_b], is_edit=False, seed=77)
    encode_and_save_batch(vae, [second_a], is_edit=False, seed=77)

    torch.testing.assert_close(
        load_file(first_a.latent_cache_path)["latents_1x1x2_bfloat16"],
        load_file(second_a.latent_cache_path)["latents_1x1x2_bfloat16"],
    )
    torch.testing.assert_close(
        load_file(first_b.latent_cache_path)["latents_1x1x2_bfloat16"],
        load_file(second_b.latent_cache_path)["latents_1x1x2_bfloat16"],
    )


def test_edit_cache_preserves_reference_order_and_separates_rng_roles(tmp_path):
    item = make_item(tmp_path, "edit", 80, controls=(80, 80, 80))

    encode_and_save_batch(FakeMageVAE(), [item], is_edit=True, seed=5)

    tensors = load_file(item.latent_cache_path)
    assert list(tensors) == [
        "latents_1x1x2_bfloat16",
        "latents_control_0_1x1x2_bfloat16",
        "latents_control_1_1x1x2_bfloat16",
        "latents_control_2_1x1x2_bfloat16",
    ]
    target = tensors["latents_1x1x2_bfloat16"]
    assert not torch.equal(target, tensors["latents_control_0_1x1x2_bfloat16"])
    assert not torch.equal(
        tensors["latents_control_0_1x1x2_bfloat16"],
        tensors["latents_control_1_1x1x2_bfloat16"],
    )
    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as handle:
        assert handle.metadata()["architecture"] == "mage_flow_edit"


@pytest.mark.parametrize(
    ("is_edit", "controls", "match"),
    [
        (False, (1,), "T2I"),
        (True, (), "between 1 and 3"),
        (True, (1, 2, 3, 4), "between 1 and 3"),
    ],
)
def test_cache_rejects_mode_and_reference_count_mismatches(tmp_path, is_edit, controls, match):
    item = make_item(tmp_path, "bad", 10, controls=controls)
    with pytest.raises(ValueError, match=match):
        encode_and_save_batch(FakeMageVAE(), [item], is_edit=is_edit, seed=0)


def test_cache_rejects_non_divisible_images_before_vae(tmp_path):
    item = make_item(tmp_path, "bad-size", 10)
    item.content = np.zeros((17, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="divisible by 16"):
        encode_and_save_batch(FakeMageVAE(), [item], is_edit=False, seed=0)


def test_cache_serializer_rejects_non_finite_or_wrong_channel_latents(tmp_path):
    item = make_item(tmp_path, "bad-latent", 10)
    with pytest.raises(ValueError, match="128 channels"):
        save_latent_cache_mage_flow(item, torch.zeros(4, 1, 2), None, is_edit=False)
    latent = torch.zeros(128, 1, 2, dtype=torch.bfloat16)
    latent[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        save_latent_cache_mage_flow(item, latent, None, is_edit=False)
