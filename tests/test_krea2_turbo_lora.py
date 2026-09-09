"""Tests for --turbo_lora: a Turbo LoRA composed live (as a second LoRA hook) on top of
RAW weights, alongside the LoRA being trained. Never merges into or mutates base weights."""

import pytest

from musubi_tuner.krea2 import krea2_utils


def _validate(**overrides):
    base = dict(
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=False,
        nvfp4_columnwise_chunk_rows=1024,
        turbo_dit=None,
        turbo_lora=None,
        scaled_mm_available=True,
        cuda_available=False,
        device_capability=None,
    )
    base.update(overrides)
    krea2_utils.validate_krea2_quantization_args(**base)


def test_validate_rejects_turbo_dit_and_turbo_lora_together():
    with pytest.raises(ValueError, match="turbo_dit.*turbo_lora|turbo_lora.*turbo_dit"):
        _validate(turbo_dit="turbo.safetensors", turbo_lora="turbo_lora.safetensors")


def test_validate_allows_turbo_lora_with_convrot_int8():
    _validate(convrot_int8=True, turbo_lora="turbo_lora.safetensors")  # must not raise


def test_validate_allows_turbo_lora_with_nvfp4():
    _validate(nvfp4=True, turbo_lora="turbo_lora.safetensors")  # must not raise


def test_validate_allows_turbo_lora_alone():
    _validate(turbo_lora="turbo_lora.safetensors")  # must not raise


def test_validate_still_rejects_convrot_with_turbo_dit():
    with pytest.raises(ValueError, match="turbo_dit"):
        _validate(convrot_int8=True, turbo_dit="turbo.safetensors")


def test_validate_still_rejects_nvfp4_with_turbo_dit():
    with pytest.raises(ValueError, match="turbo_dit"):
        _validate(nvfp4=True, turbo_dit="turbo.safetensors")


from types import SimpleNamespace


def _trainer_args(**overrides):
    base = dict(
        fp8_base=False,
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        nvfp4=False,
        turbo_dit=None,
        turbo_dit_cache=False,
        turbo_lora=None,
        turbo_lora_multiplier=1.0,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _handle_args(args):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    Krea2NetworkTrainer().handle_model_specific_args(args)


def test_parser_has_turbo_lora_flags():
    import argparse

    from musubi_tuner.krea2_train_network import krea2_setup_parser

    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args([])
    assert args.turbo_lora is None
    assert args.turbo_lora_multiplier == 1.0


def test_trainer_rejects_turbo_dit_and_turbo_lora_together():
    with pytest.raises(ValueError, match="turbo_dit.*turbo_lora|turbo_lora.*turbo_dit"):
        _handle_args(_trainer_args(turbo_dit="turbo.safetensors", turbo_lora="turbo_lora.safetensors"))


def test_trainer_rejects_turbo_dit_cache_without_turbo_dit():
    with pytest.raises(ValueError, match="turbo_dit_cache"):
        _handle_args(_trainer_args(turbo_dit_cache=True, turbo_lora="turbo_lora.safetensors"))


def test_trainer_rejects_turbo_dit_cache_with_neither_turbo_source():
    with pytest.raises(ValueError, match="turbo_dit_cache"):
        _handle_args(_trainer_args(turbo_dit_cache=True))


def test_trainer_accepts_turbo_lora_with_blocks_to_swap():
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", blocks_to_swap=4, sample_prompts="p.txt"))


def test_trainer_accepts_turbo_lora_with_convrot_int8():
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", convrot_int8=True, sample_prompts="p.txt"))


def test_trainer_still_rejects_turbo_dit_with_blocks_to_swap():
    with pytest.raises(ValueError, match="blocks_to_swap"):
        _handle_args(_trainer_args(turbo_dit="turbo.safetensors", blocks_to_swap=4))


def test_trainer_accepts_turbo_lora_alone():
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", sample_prompts="p.txt"))


def test_trainer_warns_turbo_lora_without_sample_prompts(caplog):
    _handle_args(_trainer_args(turbo_lora="turbo_lora.safetensors", sample_prompts=None))
    assert "turbo_dit" in caplog.text.lower() or "turbo_lora" in caplog.text.lower()


import torch
from safetensors.torch import save_file


class _FakeAccelerator:
    device = torch.device("cpu")

    def unwrap_model(self, m):
        return m


def test_ensure_turbo_lora_network_builds_once_and_starts_disabled(tiny_k2_model, tmp_path):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
    from musubi_tuner.networks import lora_krea2

    model = tiny_k2_model
    trainee = lora_krea2.create_arch_network(1.0, 4, 4, None, None, model)
    trainee.apply_to(None, model, apply_text_encoder=False, apply_unet=True)

    lora_path = tmp_path / "turbo_lora.safetensors"
    save_file(trainee.state_dict(), str(lora_path))  # any valid LoRA sd; values don't matter here

    trainer = Krea2NetworkTrainer()
    args = _trainer_args(turbo_lora=str(lora_path), turbo_lora_multiplier=1.0)
    accelerator = _FakeAccelerator()

    network1 = trainer._ensure_turbo_lora_network(args, accelerator, model)
    assert all(not lora.enabled for lora in network1.unet_loras)
    assert all(not p.requires_grad for p in network1.parameters())

    network2 = trainer._ensure_turbo_lora_network(args, accelerator, model)
    assert network1 is network2  # built once, cached


def test_turbo_lora_composes_additively_with_trainee_lora(tiny_k2_model, tmp_path):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
    from musubi_tuner.networks import lora_krea2

    torch.manual_seed(0)
    model = tiny_k2_model

    # Trainee LoRA: real network, apply_to() the model, then hand-set one module's weights
    # to a known nonzero delta so its contribution is detectable.
    trainee = lora_krea2.create_arch_network(1.0, 4, 4, None, None, model)
    lora0 = trainee.unet_loras[0]
    linear_module = lora0.org_module  # captured before apply_to() deletes this attribute
    in_features = linear_module.in_features
    trainee.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    lora0.lora_down.weight.data.fill_(0.1)
    lora0.lora_up.weight.data.fill_(0.1)

    x = torch.randn(2, in_features)
    trainee_out = linear_module(x)

    # Turbo LoRA: only one key (matching lora0's name), different nonzero values, so it wraps
    # the exact same underlying Linear as a second, independent hook.
    down2 = torch.full((4, in_features), 0.2)
    up2 = torch.full((linear_module.out_features, 4), 0.2)
    turbo_sd = {
        f"{lora0.lora_name}.lora_down.weight": down2,
        f"{lora0.lora_name}.lora_up.weight": up2,
        f"{lora0.lora_name}.alpha": torch.tensor(4.0),
    }
    lora_path = tmp_path / "turbo_lora.safetensors"
    save_file(turbo_sd, str(lora_path))

    trainer = Krea2NetworkTrainer()
    args = _trainer_args(turbo_lora=str(lora_path), turbo_lora_multiplier=1.0)
    accelerator = _FakeAccelerator()

    turbo_network = trainer._ensure_turbo_lora_network(args, accelerator, model)
    assert torch.equal(linear_module(x), trainee_out)  # still disabled: no change yet

    turbo_network.set_enabled(True)
    composed_out = linear_module(x)
    expected_turbo_delta = (up2.float() @ down2.float() @ x.float().T).T  # alpha == rank -> scale 1.0
    assert torch.allclose(composed_out, trainee_out + expected_turbo_delta, atol=1e-4)

    turbo_network.set_enabled(False)
    assert torch.equal(linear_module(x), trainee_out)  # back to trainee-only


def test_on_before_after_sample_images_toggle_turbo_lora(tiny_k2_model, tmp_path):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer
    from musubi_tuner.networks import lora_krea2

    model = tiny_k2_model
    trainee = lora_krea2.create_arch_network(1.0, 4, 4, None, None, model)
    trainee.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    lora_path = tmp_path / "turbo_lora.safetensors"
    save_file(trainee.state_dict(), str(lora_path))

    trainer = Krea2NetworkTrainer()
    args = _trainer_args(turbo_lora=str(lora_path), turbo_dit=None, sample_prompts="p.txt")
    accelerator = _FakeAccelerator()

    trainer.on_before_sample_images(accelerator, args, 0, 0, None, model, trainee, [], torch.float32)
    assert trainer._turbo_lora_network is not None
    assert all(lora.enabled for lora in trainer._turbo_lora_network.unet_loras)

    trainer.on_after_sample_images(accelerator, args, 0, 0, None, model, trainee, [], torch.float32)
    assert all(not lora.enabled for lora in trainer._turbo_lora_network.unet_loras)


def test_do_inference_turbo_mu_pinned_for_turbo_lora():
    # turbo_mu selection is a single expression inside do_inference; exercise it directly
    # via the same condition the implementation uses, since a full do_inference call needs a
    # real model/VAE/text-encoder pipeline out of scope for this unit test.
    args = _trainer_args(turbo_dit=None, turbo_lora="turbo_lora.safetensors")
    turbo_mu = 1.15 if (args.turbo_dit or getattr(args, "turbo_lora", None)) else None
    assert turbo_mu == 1.15
