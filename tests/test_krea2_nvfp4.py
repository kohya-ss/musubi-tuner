"""Tests for wiring NVFP4 pre-quantized loading into load_krea2_dit."""

import pytest

from musubi_tuner.krea2.krea2_utils import load_krea2_dit


def test_load_krea2_dit_rejects_multiple_quantizations():
    with pytest.raises(AssertionError, match="exclusive"):
        load_krea2_dit("unused.safetensors", fp8_scaled=True, nvfp4=True)


def test_load_krea2_dit_rejects_fp8_scaled_with_convrot_or_nvfp4():
    with pytest.raises(AssertionError, match="exclusive"):
        load_krea2_dit("unused.safetensors", fp8_scaled=True, convrot_int8=True)


def test_load_krea2_dit_rejects_nvfp4_with_lora_weights():
    with pytest.raises(AssertionError, match="lora_weights"):
        load_krea2_dit("unused.safetensors", nvfp4=True, lora_weights=[{}])


class _StopAfterPatchCall(Exception):
    """Raised by the fake patch fn below to short-circuit load_krea2_dit before load_state_dict,
    which would otherwise need a full, correctly-shaped state dict to succeed."""


def _make_tiny_nvfp4_artifact(tmp_path):
    """A single quantized Linear ('proj') in ComfyUI NVFP4 format -- enough for
    load_safetensors_with_lora_and_fp8 + NvFp4Quantizer to succeed before load_krea2_dit
    ever reaches dit.load_state_dict (which we short-circuit before, via a fake patch fn)."""
    import torch
    from safetensors.torch import save_file

    from musubi_tuner.modules.nvfp4_utils import _quantize_nvfp4_2d

    torch.manual_seed(0)
    weight = torch.randn(64, 32) * 0.02
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(weight)
    payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
    path = tmp_path / "artifact.safetensors"
    save_file(
        {
            "proj.weight": packed,
            "proj.weight_scale": block_scale,
            "proj.weight_scale_2": tensor_scale,
            "proj.comfy_quant": payload,
        },
        str(path),
    )
    return str(path)


def test_load_krea2_dit_threads_training_false_into_nvfp4_patch(monkeypatch, tmp_path):
    from musubi_tuner.krea2 import krea2_utils

    path = _make_tiny_nvfp4_artifact(tmp_path)
    captured = {}

    def fake_patch(model, sd, shapes, int8_mods, use_scaled_mm=False, training=False, calc_device=None, columnwise_chunk_rows=1024):
        captured["training"] = training
        raise _StopAfterPatchCall()

    monkeypatch.setattr(krea2_utils, "apply_nvfp4_monkey_patch", fake_patch)

    with pytest.raises(_StopAfterPatchCall):
        krea2_utils.load_krea2_dit(path, nvfp4=True, training=False, device="cpu", loading_device="cpu")

    assert captured["training"] is False


def test_load_krea2_dit_defaults_training_true_into_nvfp4_patch(monkeypatch, tmp_path):
    from musubi_tuner.krea2 import krea2_utils

    path = _make_tiny_nvfp4_artifact(tmp_path)
    captured = {}

    def fake_patch(model, sd, shapes, int8_mods, use_scaled_mm=False, training=False, calc_device=None, columnwise_chunk_rows=1024):
        captured["training"] = training
        raise _StopAfterPatchCall()

    monkeypatch.setattr(krea2_utils, "apply_nvfp4_monkey_patch", fake_patch)

    with pytest.raises(_StopAfterPatchCall):
        krea2_utils.load_krea2_dit(path, nvfp4=True, device="cpu", loading_device="cpu")  # training= omitted

    assert captured["training"] is True


def test_load_krea2_dit_nvfp4_lora_rejection_message_has_no_network_module_suggestion():
    """The trainer-oriented '--network_module' workaround phrasing doesn't apply to a
    trainer-less inference script; the message must not suggest it."""
    with pytest.raises(AssertionError) as exc_info:
        load_krea2_dit("unused.safetensors", nvfp4=True, lora_weights=[{}])
    assert "--network_module" not in str(exc_info.value)


import argparse

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser


def _base_args(**overrides):
    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args([])
    defaults = dict(
        fp8_base=False, fp8_scaled=False, convrot_int8=False, convrot_int8_bwd="bf16",
        nvfp4=False, turbo_dit=None, turbo_dit_cache=False, blocks_to_swap=0,
        block_swap_h2d_only=False,
    )
    for key, value in defaults.items():
        setattr(args, key, value)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_parser_has_nvfp4_flag():
    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args(["--nvfp4"])
    assert args.nvfp4 is True


def test_handle_model_specific_args_rejects_nvfp4_with_fp8():
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, fp8_base=True, fp8_scaled=True)
    with pytest.raises(ValueError, match="--nvfp4"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_nvfp4_with_convrot(monkeypatch):
    # nvfp4 + convrot_int8 together select mixed-format prequantized loading; must not raise.
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "get_device_capability", lambda: (10, 0))
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, convrot_int8=True)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_rejects_nvfp4_with_turbo_dit():
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, turbo_dit=True)
    with pytest.raises(ValueError, match="turbo_dit"):
        trainer.handle_model_specific_args(args)


from musubi_tuner import krea2_train_network


def test_handle_model_specific_args_rejects_nvfp4_with_block_swap_without_h2d_only(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=4, block_swap_h2d_only=False)
    with pytest.raises(ValueError, match="block_swap_h2d_only"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_nvfp4_with_block_swap_h2d_only(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=4, block_swap_h2d_only=True)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_allows_nvfp4_without_block_swap(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, block_swap_h2d_only=False)
    trainer.handle_model_specific_args(args)  # must not raise


def test_parser_has_nvfp4_columnwise_chunk_rows_flag():
    parser = argparse.ArgumentParser()
    krea2_setup_parser(parser)
    args = parser.parse_args([])
    assert args.nvfp4_columnwise_chunk_rows == 1024


def test_handle_model_specific_args_rejects_non_128_multiple_chunk_rows(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, nvfp4_columnwise_chunk_rows=1000)
    with pytest.raises(ValueError, match="nvfp4_columnwise_chunk_rows"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_128_multiple_chunk_rows(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, nvfp4_columnwise_chunk_rows=512)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_rejects_non_positive_chunk_rows(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True, blocks_to_swap=0, nvfp4_columnwise_chunk_rows=0)
    with pytest.raises(ValueError, match="nvfp4_columnwise_chunk_rows"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_rejects_nvfp4_on_non_blackwell_gpu(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "get_device_capability", lambda: (8, 9))
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    with pytest.raises(ValueError, match="Blackwell"):
        trainer.handle_model_specific_args(args)


def test_handle_model_specific_args_allows_nvfp4_on_blackwell_gpu(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "get_device_capability", lambda: (10, 0))
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_allows_nvfp4_when_cuda_not_yet_available(monkeypatch):
    # CLI validation can run before accelerate has placed the process on a GPU (e.g. a
    # multi-process launch's early arg-parsing phase) -- must not hard-fail just because CUDA
    # isn't visible yet at this point.
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: True)
    monkeypatch.setattr(krea2_train_network.torch.cuda, "is_available", lambda: False)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    trainer.handle_model_specific_args(args)  # must not raise


def test_handle_model_specific_args_rejects_nvfp4_without_scaled_mm_support(monkeypatch):
    monkeypatch.setattr(krea2_train_network, "nvfp4_scaled_mm_available", lambda: False)
    trainer = Krea2NetworkTrainer()
    args = _base_args(nvfp4=True)
    with pytest.raises(ValueError, match="PyTorch 2.10"):
        trainer.handle_model_specific_args(args)


def _make_tiny_mixed_artifact(tmp_path):
    """One NVFP4 Linear ('mlp_proj') + one prequantized ConvRot INT8 Linear ('attn_proj')."""
    import json

    import torch
    from safetensors.torch import save_file

    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
    from musubi_tuner.modules.nvfp4_utils import _quantize_nvfp4_2d

    torch.manual_seed(0)
    nvfp4_weight = torch.randn(64, 32) * 0.02
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(nvfp4_weight)
    nvfp4_payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)

    convrot_weight = torch.randn(64, 256) * 0.02
    q_weight, q_scale = quantize_int8_convrot_weight(convrot_weight, 256)
    convrot_spec = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode("utf-8")
    convrot_payload = torch.tensor(list(convrot_spec), dtype=torch.uint8)

    path = tmp_path / "mixed.safetensors"
    save_file(
        {
            "mlp_proj.weight": packed,
            "mlp_proj.weight_scale": block_scale,
            "mlp_proj.weight_scale_2": tensor_scale,
            "mlp_proj.comfy_quant": nvfp4_payload,
            "attn_proj.weight": q_weight,
            "attn_proj.weight_scale": q_scale,
            "attn_proj.comfy_quant": convrot_payload,
        },
        str(path),
    )
    return str(path)


def test_load_krea2_dit_applies_both_patches_for_mixed_nvfp4_convrot(monkeypatch, tmp_path):
    from musubi_tuner.krea2 import krea2_utils

    path = _make_tiny_mixed_artifact(tmp_path)
    captured = {}

    def fake_nvfp4_patch(model, sd, shapes, int8_mods, use_scaled_mm=False, training=False, calc_device=None, columnwise_chunk_rows=1024):
        captured["nvfp4_shapes"] = dict(shapes)
        return model

    def fake_convrot_patch(model, sd, bwd_mode="bf16", groupsize=256, groupsize_map=None):
        captured["groupsize_map"] = dict(groupsize_map) if groupsize_map else groupsize_map
        raise _StopAfterPatchCall()

    monkeypatch.setattr(krea2_utils, "apply_nvfp4_monkey_patch", fake_nvfp4_patch)
    monkeypatch.setattr(krea2_utils, "apply_convrot_int8_monkey_patch", fake_convrot_patch)

    with pytest.raises(_StopAfterPatchCall):
        krea2_utils.load_krea2_dit(path, nvfp4=True, convrot_int8=True, device="cpu", loading_device="cpu")

    assert captured["nvfp4_shapes"] == {"mlp_proj": (64, 32)}
    assert captured["groupsize_map"] == {"attn_proj": 256}


def _make_mixed_artifact_with_nvfp4_int8_embedding(tmp_path):
    """One ConvRot INT8 Linear ('attn_proj') + one NVFP4-owned int8_tensorwise embedding
    ('embed', no 'convrot' flag in its spec). Both formats emit a Musubi-layout
    '<module>.scale_weight' key, which is the collision apply_convrot_int8_monkey_patch
    must not mistake for its own when the caller passes groupsize_map (see the finding
    this test guards: scanning ALL '.scale_weight' keys in the merged mixed state dict
    would misclassify the NVFP4 embedding's scale as a ConvRot target)."""
    import json

    import torch
    from safetensors.torch import save_file

    from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight, quantize_int8_rowwise

    torch.manual_seed(0)
    convrot_weight = torch.randn(64, 256) * 0.02
    q_weight, q_scale = quantize_int8_convrot_weight(convrot_weight, 256)
    convrot_spec = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode("utf-8")
    convrot_payload = torch.tensor(list(convrot_spec), dtype=torch.uint8)

    embed_weight_f32 = torch.randn(16, 8) * 0.02
    embed_q, embed_scale = quantize_int8_rowwise(embed_weight_f32)
    embed_spec = json.dumps({"format": "int8_tensorwise"}).encode("utf-8")
    embed_payload = torch.tensor(list(embed_spec), dtype=torch.uint8)

    path = tmp_path / "mixed_with_int8_embedding.safetensors"
    save_file(
        {
            "attn_proj.weight": q_weight,
            "attn_proj.weight_scale": q_scale,
            "attn_proj.comfy_quant": convrot_payload,
            "embed.weight": embed_q,
            "embed.weight_scale": embed_scale,
            "embed.comfy_quant": embed_payload,
        },
        str(path),
    )
    return str(path)


def test_apply_convrot_int8_monkey_patch_ignores_nvfp4_owned_scale_weight_in_merged_state_dict(tmp_path):
    """Regression test: apply_convrot_int8_monkey_patch must not scan the full merged
    mixed-mode state dict for '.scale_weight' keys -- it must restrict discovery to
    groupsize_map when the caller passes one (every real call site does), so an NVFP4
    int8_tensorwise embedding's '<module>.scale_weight' key (same Musubi naming
    convention) in the merged dict is not misclassified as a ConvRot target."""
    from torch import nn

    from musubi_tuner.modules.comfy_quant_utils import FORMAT_CONVROT_INT8, FORMAT_INT8_TENSORWISE, FORMAT_NVFP4
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
    from musubi_tuner.modules.nvfp4_utils import NvFp4Quantizer

    path = _make_mixed_artifact_with_nvfp4_int8_embedding(tmp_path)

    convrot_quantizer = ConvRotInt8Quantizer(
        target_layer_keys=[],  # prequantized-only, mirrors krea2_utils's mixed-mode wiring
        foreign_formats={FORMAT_NVFP4, FORMAT_INT8_TENSORWISE},
    )
    convrot_sd = convrot_quantizer.load_and_quantize([path], calc_device="cpu")

    nvfp4_quantizer = NvFp4Quantizer(foreign_formats={FORMAT_CONVROT_INT8})
    nvfp4_sd = nvfp4_quantizer.load_and_quantize([path], calc_device="cpu")

    # Same merge order MixedQuantizer uses (dict.update per sub-quantizer): nvfp4 first,
    # then convrot_int8, so the merged dict contains both formats' Musubi-layout keys.
    merged_sd = {}
    merged_sd.update(nvfp4_sd)
    merged_sd.update(convrot_sd)
    assert "embed.scale_weight" in merged_sd  # NVFP4's renamed int8_tensorwise scale
    assert "attn_proj.scale_weight" in merged_sd  # ConvRot's scale

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_proj = nn.Linear(256, 64, bias=False)
            self.embed = nn.Embedding(16, 8)

    model = TinyModel()

    apply_convrot_int8_monkey_patch(model, merged_sd, groupsize_map=convrot_quantizer.module_groupsizes)

    assert model.convrot_int8_layer_count == 1
    assert not hasattr(model.embed, "_convrot_groupsize")
