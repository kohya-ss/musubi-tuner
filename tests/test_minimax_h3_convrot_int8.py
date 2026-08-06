"""Tests for MiniMax-H3 ConvRot INT8 base-weight quantization (R2).

Covers the H3-specific pieces on top of tests/test_krea2_convrot_int8.py: the
per-layer group size selection (256 for attn/mlp, 64 for adaln_proj whose
in_features is not a multiple of 256), the ComfyUI pre-quantized checkpoint
conversion (``weight``/``weight_scale``/``comfy_quant`` triplets), and the H3
transformer loader wiring (quantization scope, strict load, patched forward).
"""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from musubi_tuner.minimax_h3.model import (
    H3_CONVROT_INT8_ALLOWED_GROUPSIZES,
    H3_CONVROT_INT8_EXCLUDE_KEYS,
    H3_CONVROT_INT8_TARGET_KEYS,
    MiniMaxH3Config,
    MiniMaxH3Model,
    _load_h3_transformer_convrot_int8,
)
from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout
from musubi_tuner.modules.convrot_int8_utils import (
    ConvRotInt8Quantizer,
    parse_comfy_quant_spec,
    quantize_weight_convrot,
    select_convrot_groupsize,
)
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight


def _convrot_config(num_layers: int = 1) -> MiniMaxH3Config:
    # hidden_size=256 puts attn/mlp at group size 256 while time_embed_dim=64 puts
    # adaln_proj at group size 64 — the same mixed layout as the published checkpoint.
    return MiniMaxH3Config(
        hidden_size=256,
        num_layers=num_layers,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=256,
        text_dim=12,
        timestep_input_dim=4,
        time_embed_hidden_size=16,
        time_embed_dim=64,
        rope_inv_freq_len=16,
    )


def _make_quantizer() -> ConvRotInt8Quantizer:
    return ConvRotInt8Quantizer(
        H3_CONVROT_INT8_TARGET_KEYS,
        H3_CONVROT_INT8_EXCLUDE_KEYS,
        allowed_groupsizes=H3_CONVROT_INT8_ALLOWED_GROUPSIZES,
    )


def _save_tiny_bf16_checkpoint(path, config: MiniMaxH3Config, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = MiniMaxH3Model(config, dtype=torch.bfloat16)
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(path))
    return state


def _comfy_quant_tensor(groupsize: int) -> torch.Tensor:
    spec = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize}
    return torch.tensor(list(json.dumps(spec).encode("utf-8")), dtype=torch.uint8)


def _save_comfy_prequantized_checkpoint(path, bf16_state: dict, quantizer: ConvRotInt8Quantizer) -> dict:
    """Quantize the target layers of a bf16 state dict into the ComfyUI layout."""
    out = {}
    for key, value in bf16_state.items():
        if quantizer.is_target_key(key):
            result = quantize_weight_convrot(key, value, quantizer.allowed_groupsizes)
            assert result is not None, key
            wq, ws, gs = result
            module_path = key[: -len(".weight")]
            out[key] = wq
            out[module_path + ".weight_scale"] = ws
            out[module_path + ".comfy_quant"] = _comfy_quant_tensor(gs)
        else:
            out[key] = value.contiguous()
    save_file(out, str(path))
    return out


def _load_tiny_convrot(files, config: MiniMaxH3Config) -> MiniMaxH3Model:
    return _load_h3_transformer_convrot_int8(
        files,
        config,
        device=torch.device("cpu"),
        quant_device=torch.device("cpu"),
        bwd_mode="bf16",
        attn_mode="torch",
        split_attn=False,
        disable_mmap=False,
    )


def _t2_inputs(config: MiniMaxH3Config, text_length: int = 3) -> dict:
    layout = build_h3_layout(
        task="t2va",
        text_length=text_length,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )
    token_tags = torch.tensor([1, 0, 1][:text_length], dtype=torch.int64)
    return {
        "video_latents": torch.randn(1, 24, 2, 4, 4),
        "audio_latents": torch.randn(1, 32, 2, 8),
        "text_hidden_states": torch.randn(1, text_length, config.text_dim),
        "text_token_tags": token_tags.unsqueeze(0),
        "layout": layout,
        "model_t_video": torch.full((1,), 0.25),
        "model_t_audio": torch.full((1,), 0.75),
    }


# ---------------------------------------------------------------------------
# group size selection / comfy_quant parsing
# ---------------------------------------------------------------------------


def test_select_convrot_groupsize_reproduces_comfyui_choices():
    allowed = H3_CONVROT_INT8_ALLOWED_GROUPSIZES
    assert select_convrot_groupsize(5376, allowed) == 256  # qkv_proj / fc1
    assert select_convrot_groupsize(7168, allowed) == 256  # out_proj
    assert select_convrot_groupsize(14336, allowed) == 256  # fc2
    assert select_convrot_groupsize(2688, allowed) == 64  # adaln_proj
    assert select_convrot_groupsize(300, allowed) is None
    assert select_convrot_groupsize(2688, (256,)) is None  # Krea 2 default is unchanged


def test_quantizer_rejects_non_power_of_4_groupsize():
    with pytest.raises(ValueError, match="powers of 4"):
        ConvRotInt8Quantizer(allowed_groupsizes=(128,))


def test_parse_comfy_quant_spec_accepts_the_published_layout():
    spec = parse_comfy_quant_spec("x.comfy_quant", _comfy_quant_tensor(64))
    assert spec == {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 64}


@pytest.mark.parametrize(
    "spec",
    [
        {"format": "nvfp4"},
        {"format": "int8_tensorwise", "convrot": False, "convrot_groupsize": 256},
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 128},
    ],
)
def test_parse_comfy_quant_spec_rejects_unsupported_specs(spec):
    raw = torch.tensor(list(json.dumps(spec).encode("utf-8")), dtype=torch.uint8)
    with pytest.raises(ValueError):
        parse_comfy_quant_spec("x.comfy_quant", raw)


def test_parse_comfy_quant_spec_rejects_invalid_json():
    raw = torch.tensor(list(b"not json"), dtype=torch.uint8)
    with pytest.raises(ValueError, match="Invalid comfy_quant JSON"):
        parse_comfy_quant_spec("x.comfy_quant", raw)


# ---------------------------------------------------------------------------
# dynamic quantization of a tiny H3 checkpoint
# ---------------------------------------------------------------------------


def test_dynamic_quantization_covers_the_h3_scope_with_mixed_groupsizes(tmp_path):
    config = _convrot_config()
    checkpoint = tmp_path / "tiny_bf16.safetensors"
    _save_tiny_bf16_checkpoint(checkpoint, config)

    model = _load_tiny_convrot([checkpoint], config)

    block = model.blocks[0]
    for module, groupsize in (
        (block.attn.qkv_proj, 256),
        (block.attn.out_proj, 256),
        (block.mlp.fc1, 256),
        (block.mlp.fc2, 256),
        (block.adaln_proj.linear, 64),
    ):
        assert module.weight.dtype == torch.int8
        assert module.scale_weight.dtype == torch.float32
        assert module.scale_weight.shape == (module.weight.shape[0], 1)
        assert module._convrot_groupsize == groupsize
        assert not module.weight.requires_grad

    # token_refiner and final_layer stay BF16 and unpatched
    refiner_block = model.token_refiner.blocks[0]
    assert refiner_block.attn.qkv_proj.weight.dtype == torch.bfloat16
    assert not hasattr(refiner_block.attn.qkv_proj, "scale_weight")
    assert model.final_layer.adaln_proj.linear.weight.dtype == torch.bfloat16
    assert not hasattr(model.final_layer.adaln_proj.linear, "scale_weight")


def test_patched_tiny_model_forward_runs_and_stays_close_to_bf16(tmp_path):
    config = _convrot_config()
    checkpoint = tmp_path / "tiny_bf16.safetensors"
    _save_tiny_bf16_checkpoint(checkpoint, config)

    reference = MiniMaxH3Model(config, dtype=torch.bfloat16)
    reference.load_state_dict(load_file(str(checkpoint)), strict=True, assign=True)
    reference.eval().requires_grad_(False)

    model = _load_tiny_convrot([checkpoint], config)

    torch.manual_seed(1)
    inputs = _t2_inputs(config)
    with torch.no_grad():
        quantized_out = model(**inputs)
        reference_out = reference(**inputs)

    for quantized, ref in ((quantized_out.video, reference_out.video), (quantized_out.audio, reference_out.audio)):
        assert quantized.shape == ref.shape
        assert torch.isfinite(quantized).all()
        relerr = ((quantized.float() - ref.float()).norm() / ref.float().norm()).item()
        assert relerr < 0.15  # int8 weight + activation-free eager path on a random tiny model


# ---------------------------------------------------------------------------
# ComfyUI pre-quantized checkpoints
# ---------------------------------------------------------------------------


def test_prequantized_checkpoint_loads_bit_identical_to_dynamic_quantization(tmp_path):
    config = _convrot_config()
    bf16_path = tmp_path / "tiny_bf16.safetensors"
    prequant_path = tmp_path / "tiny_int8_convrot.safetensors"
    bf16_state = _save_tiny_bf16_checkpoint(bf16_path, config)
    _save_comfy_prequantized_checkpoint(prequant_path, bf16_state, _make_quantizer())

    dynamic_model = _load_tiny_convrot([bf16_path], config)
    prequant_model = _load_tiny_convrot([prequant_path], config)

    dynamic_sd = dynamic_model.state_dict()
    prequant_sd = prequant_model.state_dict()
    assert set(dynamic_sd) == set(prequant_sd)
    for key in dynamic_sd:
        assert dynamic_sd[key].dtype == prequant_sd[key].dtype, key
        assert torch.equal(dynamic_sd[key], prequant_sd[key]), key

    block = prequant_model.blocks[0]
    assert block.adaln_proj.linear._convrot_groupsize == 64
    assert block.mlp.fc1._convrot_groupsize == 256


def test_prequantized_checkpoint_with_lora_merge_hook_raises(tmp_path):
    config = _convrot_config()
    bf16_path = tmp_path / "tiny_bf16.safetensors"
    prequant_path = tmp_path / "tiny_int8_convrot.safetensors"
    bf16_state = _save_tiny_bf16_checkpoint(bf16_path, config)
    _save_comfy_prequantized_checkpoint(prequant_path, bf16_state, _make_quantizer())

    with pytest.raises(ValueError, match="Cannot merge LoRA"):
        _make_quantizer().load_and_quantize(
            [str(prequant_path)],
            calc_device=None,
            weight_hook=lambda key, value, keep_on_calc_device=False: value,
        )


def test_int8_weight_without_comfy_quant_spec_raises(tmp_path):
    path = tmp_path / "orphan_int8.safetensors"
    save_file({"blocks.0.mlp.fc1.weight": torch.zeros(8, 256, dtype=torch.int8)}, str(path))

    with pytest.raises(ValueError, match="comfy_quant"):
        _make_quantizer().load_and_quantize([str(path)], calc_device=None)


def test_weight_scale_without_comfy_quant_spec_raises(tmp_path):
    path = tmp_path / "orphan_scale.safetensors"
    save_file({"blocks.0.mlp.fc1.weight_scale": torch.ones(8, 1)}, str(path))

    with pytest.raises(ValueError, match="without a matching"):
        _make_quantizer().load_and_quantize([str(path)], calc_device=None)


def test_prequantized_module_groupsizes_follow_the_file_spec(tmp_path):
    # the file's spec wins even for a layer the dynamic rule would put at 256
    path = tmp_path / "gs64.safetensors"
    weight, scale = quantize_int8_convrot_weight(torch.randn(8, 256, dtype=torch.float32), 64)
    save_file(
        {
            "blocks.0.mlp.fc1.weight": weight,
            "blocks.0.mlp.fc1.weight_scale": scale,
            "blocks.0.mlp.fc1.comfy_quant": _comfy_quant_tensor(64),
        },
        str(path),
    )

    quantizer = _make_quantizer()
    state = quantizer.load_and_quantize([str(path)], calc_device=None)
    assert quantizer.module_groupsizes == {"blocks.0.mlp.fc1": 64}
    assert set(state) == {"blocks.0.mlp.fc1.weight", "blocks.0.mlp.fc1.scale_weight"}
    assert state["blocks.0.mlp.fc1.weight"].dtype == torch.int8
