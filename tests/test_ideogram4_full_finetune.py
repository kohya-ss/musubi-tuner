import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4_train_network import Ideogram4NetworkTrainer


def _full_train_module():
    return importlib.import_module("musubi_tuner.ideogram4_train")


def _write_conditional_checkpoint(
    path: Path,
    tensor_key: str = "input_proj.weight",
    tensor_dtype: torch.dtype = torch.float32,
) -> None:
    save_file(
        {tensor_key: torch.ones(1, dtype=tensor_dtype)},
        str(path),
        metadata={"model_type": ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE},
    )


@pytest.mark.parametrize(
    ("tensor_key", "quantization_name"),
    [
        ("layers.0.attention.qkv.weight.quant_state.bitsandbytes__nf4", "bnb 4-bit"),
        ("layers.0.attention.qkv.weight_scale", "prequantized FP8"),
    ],
)
def test_quantized_conditional_header_is_rejected_before_model_construction(tmp_path, monkeypatch, tensor_key, quantization_name):
    full_train = _full_train_module()
    checkpoint = tmp_path / "quantized.safetensors"
    _write_conditional_checkpoint(checkpoint, tensor_key)
    constructed = []

    def fail_if_constructed(*args, **kwargs):
        constructed.append((args, kwargs))
        raise AssertionError("model construction must not start")

    monkeypatch.setattr(full_train.ideogram4_utils, "load_ideogram4_transformer", fail_if_constructed)

    with pytest.raises(ValueError, match=quantization_name):
        full_train.Ideogram4Trainer().validate_full_finetune_model_args(
            SimpleNamespace(
                dit=str(checkpoint),
                disable_numpy_memmap=False,
                use_unconditional_dit_for_lora_sampling=False,
            )
        )

    assert constructed == []


@pytest.mark.parametrize(
    ("tensor_dtype", "is_supported"),
    [
        (torch.float32, True),
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.int32, False),
    ],
    ids=["fp32", "fp16", "bf16", "int32"],
)
def test_plain_conditional_header_dtype_allowlist(tmp_path, tensor_dtype, is_supported):
    full_train = _full_train_module()
    checkpoint = tmp_path / f"plain-{tensor_dtype}.safetensors"
    _write_conditional_checkpoint(checkpoint, tensor_dtype=tensor_dtype)
    args = SimpleNamespace(
        dit=str(checkpoint),
        disable_numpy_memmap=False,
        use_unconditional_dit_for_lora_sampling=False,
    )

    if is_supported:
        full_train.Ideogram4Trainer().validate_full_finetune_model_args(args)
    else:
        with pytest.raises(ValueError, match=r"dtype.*F32.*F16.*BF16"):
            full_train.Ideogram4Trainer().validate_full_finetune_model_args(args)


@pytest.mark.skipif(not hasattr(torch, "float8_e4m3fn"), reason="PyTorch has no FP8 dtype")
def test_fp8_tensor_dtype_header_is_rejected_without_scale_key(tmp_path):
    full_train = _full_train_module()
    checkpoint = tmp_path / "dtype_only_fp8.safetensors"
    save_file(
        {"input_proj.weight": torch.ones(1).to(torch.float8_e4m3fn)},
        str(checkpoint),
        metadata={"model_type": ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE},
    )

    with pytest.raises(ValueError, match="prequantized FP8"):
        full_train.Ideogram4Trainer().validate_full_finetune_model_args(
            SimpleNamespace(
                dit=str(checkpoint),
                disable_numpy_memmap=False,
                use_unconditional_dit_for_lora_sampling=False,
            )
        )


def test_full_loader_passes_trainable_dtype_directly(monkeypatch):
    full_train = _full_train_module()
    loaded_model = object()
    loader_call = {}

    def fake_load_transformer(path, **kwargs):
        loader_call.update(path=path, **kwargs)
        return loaded_model

    monkeypatch.setattr(full_train.ideogram4_utils, "load_ideogram4_transformer", fake_load_transformer)

    result = full_train.Ideogram4Trainer().load_full_finetune_transformer(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(dit_dtype="bfloat16", disable_numpy_memmap=True),
        "conditional.safetensors",
        "torch",
        False,
        torch.device("cpu"),
        torch.float32,
    )

    assert result is loaded_model
    assert loader_call == {
        "path": "conditional.safetensors",
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "expected_model_type": ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE,
        "disable_mmap": True,
        "attn_mode": "torch",
        "split_attn": False,
    }


def test_lora_scaled_fp8_loader_uses_configured_compute_dtype(monkeypatch):
    loaded_model = object()
    loader_call = {}

    def fake_load_transformer(path, **kwargs):
        loader_call.update(path=path, **kwargs)
        return loaded_model

    monkeypatch.setattr(ideogram4_utils, "load_ideogram4_transformer", fake_load_transformer)

    result = Ideogram4NetworkTrainer().load_transformer(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(dit_dtype="bfloat16", disable_numpy_memmap=False),
        "conditional_fp8.safetensors",
        "torch",
        False,
        torch.device("cpu"),
        None,
    )

    assert result is loaded_model
    assert loader_call["dtype"] is torch.bfloat16


class _PreparedTransformer:
    def __init__(self):
        self.calls = []

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return torch.full_like(inputs["x"], 3.0)


class _WrapperAwareAccelerator:
    device = torch.device("cpu")

    def __init__(self, prepared_transformer, raw_transformer):
        self.prepared_transformer = prepared_transformer
        self.raw_transformer = raw_transformer
        self.unwrap_calls = []

    def unwrap_model(self, model, keep_fp32_wrapper=False):
        self.unwrap_calls.append((model, keep_fp32_wrapper))
        assert model is self.prepared_transformer
        return self.raw_transformer

    @staticmethod
    def autocast():
        return nullcontext()


def test_call_dit_reads_raw_config_but_forwards_through_prepared_wrapper():
    trainer = Ideogram4NetworkTrainer()
    prepared = _PreparedTransformer()
    raw = SimpleNamespace(config=SimpleNamespace(in_channels=4))
    accelerator = _WrapperAwareAccelerator(prepared, raw)
    latents = torch.zeros((1, 4, 1, 1), dtype=torch.float32)
    noise = torch.zeros_like(latents)

    output = trainer.call_dit(
        SimpleNamespace(gradient_checkpointing=False, timestep_sampling="sigma"),
        accelerator,
        prepared,
        latents,
        {"i4_llm_features": [torch.zeros((1, 8), dtype=torch.float32)]},
        noise,
        torch.zeros_like(latents),
        torch.tensor([500.0]),
        torch.float32,
    )

    assert accelerator.unwrap_calls == [(prepared, False)]
    assert len(prepared.calls) == 1
    assert torch.equal(output.pred, torch.full_like(latents, 3.0))


def test_prompt_encoding_uses_instance_sampling_policy(monkeypatch):
    network_module = importlib.import_module("musubi_tuner.ideogram4_train_network")
    trainer = Ideogram4NetworkTrainer()
    trainer.use_unconditional_dit_for_sampling = lambda _args: True
    encoded_prompts = []

    monkeypatch.setattr(network_module, "load_prompts", lambda _path: [{"prompt": "positive", "negative_prompt": "negative"}])
    monkeypatch.setattr(network_module.ideogram4_utils, "load_ideogram4_tokenizer", object)
    monkeypatch.setattr(network_module.ideogram4_utils, "load_ideogram4_text_encoder", lambda *args, **kwargs: object())

    def fake_encode(_tokenizer, _text_encoder, prompt, _device):
        encoded_prompts.append(prompt)
        return torch.zeros((1, 8), dtype=torch.float32)

    monkeypatch.setattr(network_module.ideogram4_utils, "encode_prompt_to_features", fake_encode)
    monkeypatch.setattr(network_module, "clean_memory_on_device", lambda _device: None)

    sample_parameters = trainer.process_sample_prompts(
        SimpleNamespace(
            text_encoder="text.safetensors",
            disable_numpy_memmap=False,
            unconditional_dit="unconditional.safetensors",
            use_unconditional_dit_for_lora_sampling=False,
            validate_caption_structure=False,
            warn_on_caption_issues=False,
        ),
        SimpleNamespace(device=torch.device("cpu")),
        "prompts.txt",
    )

    assert encoded_prompts == ["positive"]
    assert "i4_unconditional_llm_features" not in sample_parameters[0]


def test_lora_instance_policy_keeps_two_flag_opt_in():
    trainer = Ideogram4NetworkTrainer()

    assert (
        trainer.use_unconditional_dit_for_sampling(
            SimpleNamespace(
                unconditional_dit="unconditional.safetensors",
                use_unconditional_dit_for_lora_sampling=False,
            )
        )
        is False
    )
    assert (
        trainer.use_unconditional_dit_for_sampling(
            SimpleNamespace(
                unconditional_dit="unconditional.safetensors",
                use_unconditional_dit_for_lora_sampling=True,
            )
        )
        is True
    )
    assert (
        trainer.use_unconditional_dit_for_sampling(
            SimpleNamespace(unconditional_dit=None, use_unconditional_dit_for_lora_sampling=True)
        )
        is False
    )


def test_full_policy_uses_any_unconditional_path_and_freezes_model(monkeypatch):
    full_train = _full_train_module()
    trainer = full_train.Ideogram4Trainer()
    unconditional_model = torch.nn.Linear(1, 1)
    loader_call = {}

    def fake_load_transformer(path, **kwargs):
        loader_call.update(path=path, **kwargs)
        return unconditional_model

    monkeypatch.setattr(full_train.ideogram4_utils, "load_ideogram4_transformer", fake_load_transformer)
    args = SimpleNamespace(
        unconditional_dit="unconditional.safetensors",
        use_unconditional_dit_for_lora_sampling=False,
        disable_numpy_memmap=True,
    )

    assert trainer.use_unconditional_dit_for_sampling(args) is True
    assert (
        trainer.use_unconditional_dit_for_sampling(
            SimpleNamespace(unconditional_dit=None, use_unconditional_dit_for_lora_sampling=True)
        )
        is False
    )

    trainer.on_before_sample_images(
        SimpleNamespace(device=torch.device("cpu")),
        args,
        None,
        None,
        None,
        None,
        None,
        None,
        torch.float32,
    )

    assert trainer.unconditional_transformer is unconditional_model
    assert all(not parameter.requires_grad for parameter in unconditional_model.parameters())
    assert unconditional_model.training is False
    assert loader_call == {
        "path": "unconditional.safetensors",
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "expected_model_type": ideogram4_utils.IDEOGRAM4_UNCOND_MODEL_TYPE,
        "disable_mmap": True,
        "attn_mode": "torch",
        "split_attn": False,
    }


def test_full_rejects_lora_only_unconditional_switch(tmp_path):
    full_train = _full_train_module()
    checkpoint = tmp_path / "plain.safetensors"
    _write_conditional_checkpoint(checkpoint)

    with pytest.raises(ValueError, match="--use_unconditional_dit_for_lora_sampling"):
        full_train.Ideogram4Trainer().validate_full_finetune_model_args(
            SimpleNamespace(
                dit=str(checkpoint),
                disable_numpy_memmap=False,
                use_unconditional_dit_for_lora_sampling=True,
            )
        )


def test_full_metadata_preserves_native_conditional_model_type():
    full_train = _full_train_module()

    assert full_train.Ideogram4Trainer().full_finetune_metadata(SimpleNamespace()) == {
        "model_type": ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE
    }
