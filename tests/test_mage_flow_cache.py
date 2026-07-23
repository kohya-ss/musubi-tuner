import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.cache_io import save_latent_cache_mage_flow
from musubi_tuner.dataset.image_video_dataset import ItemInfo
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
