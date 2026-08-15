import argparse
import importlib
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser


class FakeVAE(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype), requires_grad=False)
        self.encode_input_dtype = None
        self.decode_input_dtype = None

    @property
    def dtype(self):
        return self.weight.dtype

    def encode(self, image):
        self.encode_input_dtype = image.dtype
        return torch.ones((1, 128, 1, 1), device=image.device, dtype=self.dtype)

    def decode(self, latent):
        self.decode_input_dtype = latent.dtype
        return torch.zeros((latent.shape[0], 3, 16, 16), device=latent.device, dtype=self.dtype)


@pytest.mark.parametrize("model_version", tuple(flux2_utils.FLUX2_MODEL_INFO))
@pytest.mark.parametrize(
    "trainable_dtype",
    [torch.float32, torch.bfloat16],
    ids=["fp32", "bf16"],
)
def test_full_loader_uses_selected_model_info_and_requested_dtype(monkeypatch, model_version, trainable_dtype):
    full_train = importlib.import_module("musubi_tuner.flux_2_train")
    trainer = full_train.Flux2Trainer()
    args = SimpleNamespace(
        model_version=model_version,
        mixed_precision="no",
        disable_numpy_memmap=True,
    )
    trainer.handle_model_specific_args(args)

    loaded_model = object()
    loader_kwargs = {}

    def fake_load_flow_model(**kwargs):
        loader_kwargs.update(kwargs)
        return loaded_model

    monkeypatch.setattr(full_train.flux2_utils, "load_flow_model", fake_load_flow_model)

    result = trainer.load_full_finetune_transformer(
        SimpleNamespace(device=torch.device("cpu")),
        args,
        "model.safetensors",
        "torch",
        False,
        torch.device("cpu"),
        trainable_dtype,
    )

    assert result is loaded_model
    assert loader_kwargs == {
        "device": torch.device("cpu"),
        "model_version_info": flux2_utils.FLUX2_MODEL_INFO[model_version],
        "dit_path": "model.safetensors",
        "attn_mode": "torch",
        "split_attn": False,
        "loading_device": torch.device("cpu"),
        "dit_weight_dtype": trainable_dtype,
        "fp8_scaled": False,
        "disable_numpy_memmap": True,
    }


def test_fp32_sampling_casts_model_inputs_but_preserves_vae_dtype(monkeypatch):
    network_module = importlib.import_module("musubi_tuner.flux_2_train_network")
    monkeypatch.setattr(
        network_module.flux2_utils,
        "preprocess_control_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 16, 16), dtype=torch.float32), None, None),
    )
    monkeypatch.setattr(network_module, "clean_memory_on_device", lambda _device: None)

    observed_dtypes = {}

    def fake_denoise_cfg(
        _model,
        img,
        _img_ids,
        txt,
        _txt_ids,
        uncond_txt,
        _uncond_txt_ids,
        **kwargs,
    ):
        observed_dtypes.update(
            {
                "latents": img.dtype,
                "context": txt.dtype,
                "negative_context": uncond_txt.dtype,
                "control_tokens": kwargs["img_cond_seq"].dtype,
            }
        )
        return img

    monkeypatch.setattr(network_module.flux2_utils, "denoise_cfg", fake_denoise_cfg)

    trainer = Flux2NetworkTrainer()
    trainer.handle_model_specific_args(SimpleNamespace(model_version="klein-base-4b", mixed_precision="no"))
    vae = FakeVAE(torch.bfloat16)
    pixels = trainer.do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(mixed_precision="no"),
        {
            "ctx_vec": torch.ones((1, 2, 4), dtype=torch.bfloat16),
            "negative_ctx_vec": torch.ones((1, 2, 4), dtype=torch.bfloat16),
            "control_image_path": ["control.png"],
        },
        vae,
        torch.float32,
        object(),
        discrete_flow_shift=0,
        sample_steps=1,
        width=16,
        height=16,
        frame_count=1,
        generator=torch.Generator().manual_seed(1),
        do_classifier_free_guidance=True,
        guidance_scale=4.0,
        cfg_scale=1.0,
    )

    assert pixels.dtype is torch.float32
    assert observed_dtypes == {
        "latents": torch.float32,
        "context": torch.float32,
        "negative_context": torch.float32,
        "control_tokens": torch.float32,
    }
    assert vae.encode_input_dtype is torch.bfloat16
    assert vae.decode_input_dtype is torch.bfloat16


def test_model_version_metadata_is_string_provenance_and_reload_stays_explicit():
    full_train = importlib.import_module("musubi_tuner.flux_2_train")
    selected_version = "klein-9b"
    metadata = full_train.Flux2Trainer().full_finetune_metadata(SimpleNamespace(model_version=selected_version))

    assert metadata == {"ss_flux_2_model_version": selected_version}
    assert all(isinstance(value, str) for value in metadata.values())

    parser = flux2_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args([]).model_version == "dev"
    assert parser.parse_args(["--model_version", selected_version]).model_version == selected_version
