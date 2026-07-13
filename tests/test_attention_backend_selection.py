from types import SimpleNamespace

import torch

from musubi_tuner.modules import attention
from musubi_tuner import krea2_train_network


def _mock_external_flash_ready(monkeypatch, *, native_available=False, capability=(12, 0)):
    monkeypatch.delenv("MUSUBI_DISABLE_EXTERNAL_FLASH_SDPA", raising=False)
    monkeypatch.setattr(attention, "flash_attn_func", object())
    monkeypatch.setattr(attention, "flash_attn_varlen_func", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: native_available)
    monkeypatch.setattr(attention, "_external_flash_supports_training", lambda device=None: True)


def test_external_flash_replaces_missing_native_sdpa_flash(monkeypatch):
    _mock_external_flash_ready(monkeypatch)

    assert attention.should_use_external_flash_for_sdpa()


def test_external_flash_does_not_replace_native_sdpa_flash(monkeypatch):
    _mock_external_flash_ready(monkeypatch, native_available=True)

    assert not attention.should_use_external_flash_for_sdpa()


def test_external_flash_sdpa_can_be_disabled(monkeypatch):
    _mock_external_flash_ready(monkeypatch)
    monkeypatch.setenv("MUSUBI_DISABLE_EXTERNAL_FLASH_SDPA", "1")

    assert not attention.should_use_external_flash_for_sdpa()


def test_external_flash_falls_back_when_training_probe_fails(monkeypatch):
    _mock_external_flash_ready(monkeypatch)
    monkeypatch.setattr(attention, "_external_flash_supports_training", lambda device=None: False)

    assert not attention.should_use_external_flash_for_sdpa()


def test_krea2_routes_sdpa_to_external_flash_when_needed(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "should_use_external_flash_for_sdpa", lambda: True)
    args = SimpleNamespace(
        fp8_base=True,
        fp8_scaled=True,
        sdpa=True,
        flash_attn=False,
        turbo_dit_cache=False,
        turbo_dit=None,
        blocks_to_swap=0,
        sample_prompts=None,
    )

    krea2_train_network.Krea2NetworkTrainer().handle_model_specific_args(args)

    assert not args.sdpa
    assert args.flash_attn
