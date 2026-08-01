from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_KREA2,
    ARCHITECTURE_KREA2_EDIT,
    ARCHITECTURE_KREA2_EDIT_FULL,
)
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo
from musubi_tuner.krea2.krea2_mmdit import SingleMMDiTConfig, SingleStreamDiT
from musubi_tuner.krea2_cache_latents import (
    encode_and_save_batch,
    prepare_krea2_control_image,
    validate_krea2_edit_cache,
)
from musubi_tuner.krea2_cache_text_encoder_outputs import encode_and_save_batch as encode_text_and_save_batch
from musubi_tuner.krea2_train_network import Krea2NetworkTrainer


def _item(tmp_path, target_shape=(64, 64), control_shape=(64, 64)):
    target = np.zeros((*target_shape, 3), dtype=np.uint8)
    control = np.zeros((*control_shape, 3), dtype=np.uint8)
    item = ItemInfo("item.png", "edit this image", (target_shape[1], target_shape[0]), content=target)
    item.control_content = [control]
    item.latent_cache_path = str(tmp_path / "item_0064x0064_kr2e.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "item_kr2e_te.safetensors")
    return item


def test_match_target_res_preserves_reference_aspect_ratio(tmp_path):
    item = _item(tmp_path, target_shape=(1024, 1024), control_shape=(1080, 1920))
    item.match_target_res = True

    reference = prepare_krea2_control_image(item, item.control_content[0])

    assert reference.shape == (3, 768, 1360)


def test_default_control_resize_uses_exact_target_bucket(tmp_path):
    item = _item(tmp_path, target_shape=(64, 64), control_shape=(64, 128))
    item.match_target_res = False
    item.no_resize_control = False

    reference = prepare_krea2_control_image(item, item.control_content[0])

    assert reference.shape == (3, 64, 64)


def test_match_target_res_rejects_control_resolution(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        ImageDataset(
            resolution=(64, 64),
            caption_extension=".txt",
            batch_size=1,
            num_repeats=1,
            enable_bucket=True,
            bucket_no_upscale=False,
            image_directory=str(tmp_path),
            control_resolution=(64, 64),
            match_target_res=True,
            architecture=ARCHITECTURE_KREA2_EDIT,
        )


def test_krea2_edit_dataset_keeps_raw_reference_for_architecture_caches(tmp_path):
    targets = tmp_path / "targets"
    controls = tmp_path / "controls"
    cache = tmp_path / "cache"
    targets.mkdir()
    controls.mkdir()
    cache.mkdir()
    Image.fromarray(np.zeros((80, 80, 3), dtype=np.uint8)).save(targets / "item.png")
    Image.fromarray(np.zeros((32, 96, 3), dtype=np.uint8)).save(controls / "item.png")
    (targets / "item.txt").write_text("edit this image", encoding="utf-8")
    dataset = ImageDataset(
        resolution=(64, 64),
        caption_extension=".txt",
        batch_size=1,
        num_repeats=1,
        enable_bucket=False,
        bucket_no_upscale=False,
        image_directory=str(targets),
        control_directory=str(controls),
        cache_directory=str(cache),
        match_target_res=True,
        architecture=ARCHITECTURE_KREA2_EDIT,
    )

    _, batch = next(iter(dataset.retrieve_latent_cache_batches(num_workers=1)))

    assert batch[0].content.shape[:2] == (64, 64)
    assert batch[0].control_content[0].shape[:2] == (32, 96)
    assert batch[0].match_target_res is True


class _FakeVAE:
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self):
        self.posterior_flags = []

    def encode_pixels_to_latents(self, pixels, *, sample_posterior=False, generator=None):
        self.posterior_flags.append(sample_posterior)
        return torch.zeros(
            pixels.shape[0],
            1,
            pixels.shape[2],
            pixels.shape[3] // 8,
            pixels.shape[4] // 8,
            dtype=pixels.dtype,
        )


def test_edit_latent_cache_saves_reference_latents_and_policy(tmp_path):
    item = _item(tmp_path)
    item.match_target_res = True
    vae = _FakeVAE()

    encode_and_save_batch(vae, [item], edit=True)

    cached = load_file(item.latent_cache_path)
    assert any(key.startswith("latents_") and not key.startswith("latents_control_") for key in cached)
    assert any(key.startswith("latents_control_0_") for key in cached)
    assert vae.posterior_flags == [True, True]
    with safe_open(item.latent_cache_path, framework="pt", device="cpu") as cache:
        metadata = cache.metadata()
    assert metadata["architecture"] == ARCHITECTURE_KREA2_EDIT_FULL
    assert metadata["match_target_res"] == "true"
    assert validate_krea2_edit_cache(item) is True
    item.match_target_res = False
    assert validate_krea2_edit_cache(item) is False


def test_edit_text_cache_passes_reference_images(monkeypatch, tmp_path):
    item = _item(tmp_path, control_shape=(32, 64))
    captured = {}

    def fake_get_prompt_embeds(encoder, prompts, images=None):
        captured["prompts"] = prompts
        captured["images"] = images
        return torch.ones(1, 3, 2, 4), torch.ones(1, 3, dtype=torch.bool)

    monkeypatch.setattr(
        "musubi_tuner.krea2_cache_text_encoder_outputs.krea2_utils.get_krea2_prompt_embeds",
        fake_get_prompt_embeds,
    )

    encode_text_and_save_batch(object(), [item], edit=True)

    assert captured["prompts"] == [item.caption]
    assert len(captured["images"][0]) == 1
    assert captured["images"][0][0].shape == (3, 32, 64)
    cached = load_file(item.text_encoder_output_cache_path)
    assert any(key.startswith("varlen_krea2_vl_embed_") for key in cached)
    with safe_open(item.text_encoder_output_cache_path, framework="pt", device="cpu") as cache:
        assert cache.metadata()["architecture"] == ARCHITECTURE_KREA2_EDIT_FULL


class _CPUAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return nullcontext()


class _RecordingDiT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = SingleStreamDiT(
            SingleMMDiTConfig(
                features=16,
                tdim=8,
                txtdim=16,
                heads=1,
                multiplier=1,
                layers=2,
                patch=1,
                channels=1,
                txtlayers=2,
                txtheads=1,
                txtkvheads=1,
            ),
            attn_mode="torch",
        )
        self.config = self.inner.config
        self.last_kwargs = None

    def forward(self, **kwargs):
        self.last_kwargs = kwargs
        return self.inner(**kwargs)


def _training_batch(include_control=True):
    batch = {
        "latents": torch.randn(1, 1, 1, 2, 2),
        "krea2_vl_embed": [torch.randn(2, 2, 16)],
    }
    if include_control:
        batch["latents_control_0"] = torch.randn(1, 1, 1, 2, 2)
    return batch


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_edit_training_forward_appends_refs_and_backpropagates(gradient_checkpointing):
    trainer = Krea2NetworkTrainer()
    trainer.is_edit = True
    model = _RecordingDiT()
    if gradient_checkpointing:
        model.inner.enable_gradient_checkpointing()
        model.train()
    batch = _training_batch()
    noise = torch.randn_like(batch["latents"])
    noisy = torch.randn_like(batch["latents"])
    args = SimpleNamespace(gradient_checkpointing=gradient_checkpointing, kv_cache=True)

    output = trainer.call_dit(
        args,
        _CPUAccelerator(),
        model,
        batch["latents"],
        batch,
        noise,
        noisy,
        torch.tensor([500.0]),
        torch.float32,
    )
    loss = torch.nn.functional.mse_loss(output.pred, output.target)
    loss.backward()

    assert output.pred.shape == batch["latents"].shape
    assert model.last_kwargs["reflen"] == 4
    assert model.last_kwargs["isolate_refs"] is True
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
        for parameter in model.parameters()
    )


def test_edit_training_rejects_missing_reference_cache():
    trainer = Krea2NetworkTrainer()
    trainer.is_edit = True
    batch = _training_batch(include_control=False)

    with pytest.raises(ValueError, match="latents_control_0"):
        trainer.call_dit(
            SimpleNamespace(gradient_checkpointing=False, kv_cache=True),
            _CPUAccelerator(),
            _RecordingDiT(),
            batch["latents"],
            batch,
            torch.randn_like(batch["latents"]),
            torch.randn_like(batch["latents"]),
            torch.tensor([500.0]),
            torch.float32,
        )


class _TrainingSampleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=1, channels=1)
        self.input_dtypes = []

    def forward(self, img, context, t, pos, mask, reflen=0, **kwargs):
        self.input_dtypes.append((img.dtype, t.dtype))
        return torch.zeros_like(img[:, : img.shape[1] - reflen])


class _TrainingSampleVAE(torch.nn.Module):
    dtype = torch.float32

    def decode_to_pixels(self, latents):
        return torch.zeros(latents.shape[0], 3, latents.shape[-2], latents.shape[-1])


def test_edit_training_sample_uses_dit_dtype_for_reference_tokens(monkeypatch):
    monkeypatch.setattr(
        "musubi_tuner.krea2_train_network.krea2_sampling.load_reference_images",
        lambda paths: [torch.zeros(3, 16, 16)],
    )
    monkeypatch.setattr(
        "musubi_tuner.krea2_train_network.krea2_sampling.encode_reference_images",
        lambda vae, images, device, dtype: [torch.ones(1, 2, 2, dtype=dtype)],
    )
    trainer = Krea2NetworkTrainer()
    trainer.is_edit = True
    model = _TrainingSampleModel()

    pixels = trainer.do_inference(
        _CPUAccelerator(),
        SimpleNamespace(turbo_dit=None, kv_cache=False),
        {
            "krea2_vl_embed": torch.zeros(1, 1, 1),
            "control_image_path": ["reference.png"],
        },
        _TrainingSampleVAE(),
        torch.float32,
        model,
        0.0,
        1,
        16,
        16,
        1,
        torch.Generator(device="cpu").manual_seed(1),
        False,
        1.0,
        1.0,
    )

    assert pixels.shape == (1, 3, 1, 2, 2)
    assert model.input_dtypes == [(torch.float32, torch.float32)]


def test_krea2_trainer_selects_separate_edit_architecture():
    trainer = Krea2NetworkTrainer()
    trainer.is_edit = False
    assert trainer.architecture == ARCHITECTURE_KREA2
    trainer.is_edit = True
    assert trainer.architecture == ARCHITECTURE_KREA2_EDIT


def test_kv_cache_training_requires_edit_mode():
    trainer = Krea2NetworkTrainer()
    args = SimpleNamespace(
        edit=False,
        kv_cache=True,
        fp8_base=False,
        fp8_scaled=False,
        turbo_dit_cache=False,
        turbo_dit=None,
        blocks_to_swap=0,
        sample_prompts=None,
    )

    with pytest.raises(ValueError, match="requires --edit"):
        trainer.handle_model_specific_args(args)
