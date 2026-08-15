import argparse
import importlib
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.flux_kontext_train_network import FluxKontextNetworkTrainer
from musubi_tuner.training.full_finetune import add_full_finetune_args, resolve_trainable_dtype


class RecordingFlux(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype))
        self.floating_input_dtypes = {}

    def forward(self, **inputs):
        self.floating_input_dtypes.update(
            {name: value.dtype for name, value in inputs.items() if isinstance(value, torch.Tensor) and value.is_floating_point()}
        )
        return torch.zeros_like(inputs["img"])


class FakeVAE(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), dtype=dtype), requires_grad=False)

    @property
    def dtype(self):
        return self.weight.dtype

    def encode(self, image):
        return torch.ones((1, 16, 2, 2), device=image.device, dtype=self.dtype)

    def decode(self, latent):
        return torch.zeros((latent.shape[0], 3, 16, 16), device=latent.device, dtype=self.dtype)


def test_full_loader_passes_default_trainable_dtype_directly(monkeypatch):
    full_train = importlib.import_module("musubi_tuner.flux_kontext_train")
    trainer = full_train.FluxKontextTrainer()
    loaded_model = object()
    loader_kwargs = {}

    def fake_load_flow_model(**kwargs):
        loader_kwargs.update(kwargs)
        return loaded_model

    monkeypatch.setattr(full_train.flux_utils, "load_flow_model", fake_load_flow_model)

    parser = add_full_finetune_args(argparse.ArgumentParser())
    trainable_dtype = resolve_trainable_dtype(parser.parse_args([]))
    result = trainer.load_full_finetune_transformer(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(dit="model.safetensors"),
        "model.safetensors",
        "torch",
        False,
        torch.device("cpu"),
        trainable_dtype,
    )

    assert result is loaded_model
    assert loader_kwargs == {
        "ckpt_path": "model.safetensors",
        "dtype": torch.float32,
        "device": torch.device("cpu"),
        "disable_mmap": True,
        "attn_mode": "torch",
        "split_attn": False,
        "loading_device": torch.device("cpu"),
        "fp8_scaled": False,
    }


@pytest.mark.parametrize(
    "dit_dtype",
    [torch.float32, torch.bfloat16],
    ids=["full-fp32", "lora-default-bf16"],
)
def test_sampling_uses_requested_model_dtype_and_preserves_position_ids(monkeypatch, dit_dtype):
    network_module = importlib.import_module("musubi_tuner.flux_kontext_train_network")
    monkeypatch.setattr(
        network_module.flux_utils,
        "preprocess_control_image",
        lambda *_args, **_kwargs: (torch.zeros((1, 3, 16, 16), dtype=torch.float32), None, None),
    )
    monkeypatch.setattr(network_module, "clean_memory_on_device", lambda _device: None)

    transformer = RecordingFlux(dit_dtype)
    trainer = FluxKontextNetworkTrainer()
    pixels = trainer.do_inference(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(mixed_precision="no"),
        {
            "t5_vec": torch.ones((1, 2, 4), dtype=torch.float32),
            "clip_l_pooler": torch.ones((1, 4), dtype=torch.float32),
            "control_image_path": ["control.png"],
        },
        FakeVAE(dit_dtype),
        dit_dtype,
        transformer,
        discrete_flow_shift=0,
        sample_steps=1,
        width=16,
        height=16,
        frame_count=1,
        generator=torch.Generator().manual_seed(1),
        do_classifier_free_guidance=False,
        guidance_scale=2.5,
        cfg_scale=1.0,
    )

    assert pixels.dtype is torch.float32
    assert transformer.weight.dtype is dit_dtype
    assert transformer.floating_input_dtypes
    assert transformer.floating_input_dtypes == {
        "img": dit_dtype,
        "img_ids": torch.float32,
        "txt": dit_dtype,
        "txt_ids": torch.float32,
        "y": dit_dtype,
        "timesteps": dit_dtype,
        "guidance": dit_dtype,
    }
