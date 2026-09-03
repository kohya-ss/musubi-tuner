"""Tests for wiring NVFP4 + ConvRot INT8 loading into flux2_utils.load_flow_model.

Mirrors tests/test_krea2_nvfp4.py's approach: exclusivity asserts are hit before any file
I/O (a placeholder path is never opened), and the mixed-dispatch test builds a tiny
prequantized artifact then short-circuits load_flow_model via a fake patch fn before
load_state_dict (which would otherwise need a full, correctly-shaped state dict).
"""

import pytest

from musubi_tuner.flux_2 import flux2_utils


def test_load_flow_model_rejects_fp8_scaled_with_nvfp4():
    with pytest.raises(AssertionError, match="exclusive"):
        flux2_utils.load_flow_model(
            device="cpu",
            model_version_info=None,
            dit_path="unused.safetensors",
            attn_mode="torch",
            split_attn=False,
            loading_device="cpu",
            fp8_scaled=True,
            nvfp4=True,
        )


def test_load_flow_model_rejects_fp8_scaled_with_convrot_int8():
    with pytest.raises(AssertionError, match="exclusive"):
        flux2_utils.load_flow_model(
            device="cpu",
            model_version_info=None,
            dit_path="unused.safetensors",
            attn_mode="torch",
            split_attn=False,
            loading_device="cpu",
            fp8_scaled=True,
            convrot_int8=True,
        )


def test_load_flow_model_rejects_nvfp4_with_lora_weights_list():
    with pytest.raises(AssertionError, match="lora_weights_list"):
        flux2_utils.load_flow_model(
            device="cpu",
            model_version_info=None,
            dit_path="unused.safetensors",
            attn_mode="torch",
            split_attn=False,
            loading_device="cpu",
            nvfp4=True,
            lora_weights_list=[{}],
        )


class _StopAfterPatchCall(Exception):
    """Raised by the fake patch fns below to short-circuit load_flow_model before
    load_state_dict, which would otherwise need a full, correctly-shaped state dict."""


def _make_tiny_nvfp4_artifact(tmp_path):
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


def test_load_flow_model_threads_training_flag_into_nvfp4_patch(monkeypatch, tmp_path):
    path = _make_tiny_nvfp4_artifact(tmp_path)
    captured = {}

    def fake_patch(model, sd, shapes, int8_mods, use_scaled_mm=False, training=False, calc_device=None, columnwise_chunk_rows=1024):
        captured["training"] = training
        raise _StopAfterPatchCall()

    monkeypatch.setattr(flux2_utils, "apply_nvfp4_monkey_patch", fake_patch)

    with pytest.raises(_StopAfterPatchCall):
        flux2_utils.load_flow_model(
            device="cpu",
            model_version_info=flux2_utils.FLUX2_MODEL_INFO["klein-base-9b"],
            dit_path=path,
            attn_mode="torch",
            split_attn=False,
            loading_device="cpu",
            nvfp4=True,
            training=False,
        )

    assert captured["training"] is False


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


def test_load_flow_model_applies_both_patches_for_mixed_nvfp4_convrot(monkeypatch, tmp_path):
    path = _make_tiny_mixed_artifact(tmp_path)
    captured = {}

    def fake_nvfp4_patch(model, sd, shapes, int8_mods, use_scaled_mm=False, training=False, calc_device=None, columnwise_chunk_rows=1024):
        captured["nvfp4_shapes"] = dict(shapes)
        return model

    def fake_convrot_patch(model, sd, bwd_mode="bf16", groupsize=256, groupsize_map=None):
        captured["groupsize_map"] = dict(groupsize_map) if groupsize_map else groupsize_map
        raise _StopAfterPatchCall()

    monkeypatch.setattr(flux2_utils, "apply_nvfp4_monkey_patch", fake_nvfp4_patch)
    monkeypatch.setattr(flux2_utils, "apply_convrot_int8_monkey_patch", fake_convrot_patch)

    with pytest.raises(_StopAfterPatchCall):
        flux2_utils.load_flow_model(
            device="cpu",
            model_version_info=flux2_utils.FLUX2_MODEL_INFO["klein-base-9b"],
            dit_path=path,
            attn_mode="torch",
            split_attn=False,
            loading_device="cpu",
            nvfp4=True,
            convrot_int8=True,
        )

    assert captured["nvfp4_shapes"] == {"mlp_proj": (64, 32)}
    assert captured["groupsize_map"] == {"attn_proj": 256}
