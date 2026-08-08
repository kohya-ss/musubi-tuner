import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3 import model as h3_model
from musubi_tuner.minimax_h3.checkpoint import inspect_safetensors_convrot_int8, load_safetensors_module
from musubi_tuner.minimax_h3.model import (
    AdalnProj,
    FinalLayer,
    MiniMaxH3Config,
    MiniMaxH3Model,
    parse_h3_transformer_config,
)
from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Artifact, ConvRotInt8LayerSpec
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


def _tiny_config(*, num_layers: int = 2, pruned: bool = False) -> MiniMaxH3Config:
    return MiniMaxH3Config(
        hidden_size=16,
        num_layers=num_layers,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=24,
        text_dim=12,
        timestep_input_dim=4,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        rope_inv_freq_len=1,
        adaln_curve_grid=1025 if pruned else None,
    )


def _tiny_model(*, num_layers: int = 2, training: bool = True) -> MiniMaxH3Model:
    model = MiniMaxH3Model(_tiny_config(num_layers=num_layers), dtype=torch.float32)
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
    return model.train(training)


def _t2_layout(text_length: int = 3):
    return build_h3_layout(
        task="t2va",
        text_length=text_length,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )


def _t2_inputs(batch_size: int = 1, text_length: int = 3):
    token_tags = torch.tensor([1, 0, 1][:text_length], dtype=torch.int64)
    return {
        "video_latents": torch.randn(batch_size, 24, 2, 4, 4),
        "audio_latents": torch.randn(batch_size, 32, 2, 8),
        "text_hidden_states": torch.randn(batch_size, text_length, 12),
        "text_token_tags": token_tags.unsqueeze(0).expand(batch_size, -1).clone(),
        "layout": _t2_layout(text_length),
        "model_t_video": torch.full((batch_size,), 0.25),
        "model_t_audio": torch.full((batch_size,), 0.75),
    }


def test_released_config_and_meta_state_dict_match_published_bf16_header():
    config = MiniMaxH3Config()

    model = MiniMaxH3Model(config, dtype=torch.bfloat16, device=torch.device("meta"))
    state = model.state_dict()

    assert config.in_channels == 24
    assert config.audio_in_channels == 32
    assert config.hidden_size == 5376
    assert config.num_layers == 50
    assert config.num_attention_heads == 56
    assert config.attention_head_dim == 128
    assert config.text_dim == 5120
    assert len(state) == 535
    assert state["video_patch_proj.weight"].shape == (5376, 96)
    assert state["video_patch_proj.weight"].dtype == torch.float32
    assert state["audio_patch_proj.weight"].shape == (5376, 32)
    assert state["condition_proj.weight"].shape == (5376, 5120)
    assert state["condition_proj.weight"].dtype == torch.bfloat16
    assert state["blocks.0.attn.qkv_proj.weight"].shape == (21504, 5376)
    assert state["blocks.0.adaln_proj.linear.weight"].shape == (96768, 2688)
    assert state["final_layer.adaln_proj.linear.weight"].shape == (10752, 2688)
    assert state["final_layer.video_out.weight"].shape == (96, 5376)
    assert state["final_layer.video_out.weight"].dtype == torch.float32
    assert state["rope.inv_freq"].shape == (16,)


def test_pruned_config_builds_curve_table_without_time_embedder_state():
    config = _tiny_config(num_layers=1, pruned=True)
    model = MiniMaxH3Model(config, dtype=torch.bfloat16, device=torch.device("meta"))
    state = model.state_dict()

    assert config.is_pruned
    assert state["adaln_t_table"].shape == (1025, 8)
    assert state["adaln_t_table"].dtype is torch.float32
    assert not any(key.startswith("time_embedder.") for key in state)
    assert state["blocks.0.adaln_proj.linear.weight"].shape[-1] == 8
    assert state["final_layer.adaln_proj.linear.weight"].shape[-1] == 8
    assert not model.blocks[0].adaln_proj.apply_silu
    assert not model.final_layer.adaln_proj.apply_silu


@pytest.mark.parametrize(
    "overrides",
    [
        {"adaln_curve_grid": 1024, "time_embed_dim": 8},
        {"adaln_curve_grid": 1025, "time_embed_dim": 16},
    ],
)
def test_pruned_config_rejects_unknown_curve_shapes(overrides):
    with pytest.raises(ValueError, match="pruned AdaLN"):
        MiniMaxH3Config(**overrides)


def test_pruned_timestep_curve_interpolates_fp32_then_casts_to_compute_dtype():
    model = MiniMaxH3Model(_tiny_config(num_layers=1, pruned=True), dtype=torch.bfloat16)
    table = torch.arange(1025, dtype=torch.float32)[:, None].repeat(1, 8)
    with torch.no_grad():
        model.adaln_t_table.copy_(table)

    timesteps = torch.tensor([0.0, 0.30017, 1.0], dtype=torch.float32)
    actual = model._timestep_embeddings(timesteps, torch.device("cpu"))
    position = timesteps * 1024
    expected = position[:, None].repeat(1, 8).to(torch.bfloat16)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual, expected)
    assert not torch.equal(actual[1].float(), table[position[1].round().long()])


def test_pruned_timestep_curve_handles_bf16_as_represented_and_rejects_scalars():
    model = MiniMaxH3Model(_tiny_config(num_layers=1, pruned=True), dtype=torch.float32)
    table = torch.arange(1025, dtype=torch.float32)[:, None].repeat(1, 8)
    with torch.no_grad():
        model.adaln_t_table.copy_(table)
    timesteps = torch.tensor([0.13, 0.30017, 0.7773, 0.999], dtype=torch.bfloat16)

    actual = model._timestep_embeddings(timesteps, torch.device("cpu"))
    expected = (timesteps.float() * 1024)[:, None].repeat(1, 8)

    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="one-dimensional"):
        model._timestep_embeddings(torch.tensor(0.5), torch.device("cpu"))


def test_pruned_timestep_curve_honors_the_execution_device_contract():
    model = MiniMaxH3Model(_tiny_config(num_layers=1, pruned=True), dtype=torch.float32)

    actual = model._timestep_embeddings(torch.tensor([0.25]), torch.device("meta"))

    assert actual.device.type == "meta"


def test_pruned_forward_casts_curve_output_before_bf16_adaln_without_autocast():
    model = MiniMaxH3Model(_tiny_config(num_layers=1, pruned=True), dtype=torch.bfloat16).eval()
    with torch.no_grad():
        model.adaln_t_table.zero_()
    seen_dtypes = []
    hook = model.blocks[0].adaln_proj.linear.register_forward_pre_hook(lambda _module, inputs: seen_dtypes.append(inputs[0].dtype))

    with torch.no_grad():
        model(**_t2_inputs())
    hook.remove()

    assert seen_dtypes == [torch.bfloat16]


def test_model_compute_dtype_uses_an_unquantized_norm_sentinel():
    model = MiniMaxH3Model(_tiny_config(num_layers=1), dtype=torch.bfloat16)
    model.condition_proj.weight = nn.Parameter(
        torch.empty(model.condition_proj.weight.shape, dtype=torch.int8),
        requires_grad=False,
    )

    assert model.dtype is torch.bfloat16


def test_rope_inv_freq_has_no_synthesized_fallback(monkeypatch):
    def unexpected_log(_value):
        pytest.fail("rope.inv_freq must be loaded from the checkpoint, not synthesized")

    monkeypatch.setattr(h3_model.math, "log", unexpected_log)

    model = MiniMaxH3Model(_tiny_config(), dtype=torch.float32)

    assert model.rope.inv_freq.shape == (1,)


def test_published_transformer_metadata_is_parsed_strictly():
    released = {
        "hidden_size": 5376,
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "num_attention_heads": 56,
        "attention_head_dim": 128,
        "ffn_hidden_size": 14336,
        "latents_dim": 24,
        "audio_latents_dim": 32,
        "patch_size": [1, 2, 2],
        "text_dim": 5120,
        "timestep_input_dim": 256,
        "time_embed_hidden_size": 5376,
        "time_embed_dim": 2688,
        "adaln_out_features": 96768,
        "final_adaln_out_features": 10752,
        "rope_inv_freq_len": 16,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
        "image_model": "minimax_h3",
    }

    actual = parse_h3_transformer_config({"config": json.dumps({"transformer": released})})

    assert actual == MiniMaxH3Config()
    with pytest.raises(ValueError, match=r"hidden_size.*5376.*4096"):
        parse_h3_transformer_config({"config": json.dumps({"transformer": {**released, "hidden_size": 4096}})})
    marked = parse_h3_transformer_config(
        {
            "config": json.dumps({"transformer": released}),
            "format": "int8_tensorwise",
            "convrot": "true",
        }
    )
    assert marked == MiniMaxH3Config()


def test_tiny_model_rejects_batch_size_above_one_in_r1():
    model = _tiny_model()

    with pytest.raises(ValueError, match=r"R1 requires batch_size=1"):
        model(**_t2_inputs(batch_size=2))


def test_model_reuses_rotary_state_for_the_same_layout(monkeypatch):
    model = _tiny_model(num_layers=1, training=False)
    inputs = _t2_inputs(batch_size=1)
    calls = {"positions": 0, "rotation": 0}
    original_build_position_grid = h3_model.build_position_grid
    original_rotation_table = model._rotation_table

    def record_positions(*args, **kwargs):
        calls["positions"] += 1
        return original_build_position_grid(*args, **kwargs)

    def record_rotation(*args, **kwargs):
        calls["rotation"] += 1
        return original_rotation_table(*args, **kwargs)

    monkeypatch.setattr(h3_model, "build_position_grid", record_positions)
    monkeypatch.setattr(model, "_rotation_table", record_rotation)

    model(**inputs)
    model(
        **{
            **inputs,
            "model_t_video": torch.tensor([0.4]),
            "model_t_audio": torch.tensor([0.6]),
        }
    )

    assert calls == {"positions": 1, "rotation": 1}

    model.to("cpu")
    assert not model._rotary_cache


def test_rotary_cache_is_bounded_and_cleared_by_checkpoint_load():
    model = _tiny_model(num_layers=1, training=False)
    for text_length in (1, 2, 3):
        model._cached_rotation_table(
            _t2_layout(text_length),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    assert len(model._rotary_cache) == 2

    model.load_state_dict(model.state_dict())

    assert not model._rotary_cache


def test_model_accepts_ordered_ref2va_visual_and_audio_conditions():
    model = _tiny_model(num_layers=1)
    image = H3VideoGeometry(1, 2, 4)
    video = H3VideoGeometry(2, 4, 4)
    layout = build_h3_layout(
        task="ref2va",
        text_length=2,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
        references=(
            H3ReferenceGeometry("image", video=image),
            H3ReferenceGeometry("video", video=video, audio_frames=8),
            H3ReferenceGeometry("audio", audio_frames=2),
        ),
    )

    output = model(
        video_latents=torch.randn(1, 24, 2, 4, 4),
        audio_latents=torch.randn(1, 32, 2, 8),
        text_hidden_states=torch.randn(1, 2, 12),
        text_token_tags=torch.tensor([[0, 1]]),
        layout=layout,
        model_t_video=0.25,
        model_t_audio=0.75,
        visual_condition_latents=(
            torch.randn(1, 24, 1, 2, 4),
            torch.randn(1, 24, 2, 4, 4),
        ),
        audio_condition_latents=(
            torch.randn(1, 32, 2, 8),
            torch.randn(1, 32, 2, 2),
        ),
    )

    assert output.video.shape == (1, 24, 2, 4, 4)
    assert output.audio.shape == (1, 32, 2, 8)


def test_model_rejects_condition_geometry_even_when_the_packed_row_count_matches():
    model = _tiny_model(num_layers=1)
    layout = build_h3_layout(
        task="ref2va",
        text_length=1,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
        references=(H3ReferenceGeometry("image", video=H3VideoGeometry(1, 2, 8)),),
    )

    with pytest.raises(ValueError, match=r"ref_000_image geometry.*1x2x8.*1x4x4"):
        model(
            video_latents=torch.randn(1, 24, 2, 4, 4),
            audio_latents=torch.randn(1, 32, 2, 8),
            text_hidden_states=torch.randn(1, 1, 12),
            text_token_tags=torch.tensor([[1]]),
            layout=layout,
            model_t_video=0.25,
            model_t_audio=0.75,
            visual_condition_latents=(torch.randn(1, 24, 1, 4, 4),),
        )


def test_model_requires_text_token_tags_to_keep_the_batch_axis():
    model = _tiny_model(num_layers=1)
    inputs = _t2_inputs(batch_size=1)
    inputs["text_token_tags"] = torch.tensor([1, 0, 1], dtype=torch.int64)

    with pytest.raises(ValueError, match=r"\[1,3\]"):
        model(**inputs)


def test_block_adaln_rows_are_ordered_as_three_modalities_per_timestep():
    projection = AdalnProj(timestep_dim=1, hidden_size=1, expand=1, modalities=3, dtype=torch.float32)
    with torch.no_grad():
        projection.linear.weight.copy_(torch.tensor([[1.0], [2.0], [3.0]]))
        projection.linear.bias.zero_()

    (rows,) = projection(torch.tensor([[1.0], [10.0]]))

    silu = torch.nn.functional.silu(torch.tensor([1.0, 10.0]))
    expected = torch.cat((silu[0] * torch.tensor([1.0, 2.0, 3.0]), silu[1] * torch.tensor([1.0, 2.0, 3.0])))
    torch.testing.assert_close(rows[:, 0], expected)


def test_pruned_adaln_projects_curve_coordinates_without_silu():
    projection = AdalnProj(
        timestep_dim=2,
        hidden_size=1,
        expand=2,
        modalities=1,
        dtype=torch.float32,
        apply_silu=False,
    )
    with torch.no_grad():
        projection.linear.weight.copy_(torch.eye(2))
        projection.linear.bias.zero_()

    first, second = projection(torch.tensor([[-1.0, 2.0]]))

    torch.testing.assert_close(first, torch.tensor([[-1.0]]))
    torch.testing.assert_close(second, torch.tensor([[2.0]]))


def test_final_layer_uses_direct_time_rows_without_modality_offsets():
    layer = FinalLayer(
        hidden_size=2,
        timestep_dim=1,
        video_output_dim=1,
        audio_output_dim=1,
        dtype=torch.float32,
    )

    class FixedAdaLN(nn.Module):
        def forward(self, timestep_embeddings):
            del timestep_embeddings
            shift = torch.tensor([[10.0, 0.0], [20.0, 0.0]])
            scale = torch.zeros_like(shift)
            return shift, scale

    layer.norm = nn.Identity()
    layer.adaln_proj = FixedAdaLN()
    with torch.no_grad():
        layer.video_out.weight.copy_(torch.tensor([[1.0, 0.0]]))
        layer.video_out.bias.zero_()
        layer.audio_out.weight.copy_(torch.tensor([[1.0, 0.0]]))
        layer.audio_out.bias.zero_()

    video, audio = layer(
        torch.zeros(1, 4, 2),
        torch.zeros(2, 1),
        video_slice=slice(2, 4),
        audio_slice=slice(0, 2),
        video_timestep_index=1,
        audio_timestep_index=0,
    )

    torch.testing.assert_close(video, torch.full((1, 2, 1), 20.0))
    torch.testing.assert_close(audio, torch.full((1, 2, 1), 10.0))


def test_segment_modulation_matches_out_of_place_outputs_and_gradients():
    segments = ((0, 2, 0), (2, 4, 1))
    generator = torch.Generator().manual_seed(1234)
    values = (
        torch.randn(1, 4, 3, dtype=torch.float64, generator=generator),
        torch.randn(2, 3, dtype=torch.float64, generator=generator),
        torch.randn(2, 3, dtype=torch.float64, generator=generator),
        torch.randn(1, 4, 3, dtype=torch.float64, generator=generator),
        torch.randn(1, 4, 3, dtype=torch.float64, generator=generator),
        torch.randn(2, 3, dtype=torch.float64, generator=generator),
    )
    actual = [value.clone().requires_grad_() for value in values]
    expected = [value.clone().requires_grad_() for value in values]
    actual_norm = nn.RMSNorm(3, dtype=torch.float64)
    expected_norm = nn.RMSNorm(3, dtype=torch.float64)
    with torch.no_grad():
        norm_weight = torch.randn(3, dtype=torch.float64, generator=generator)
        actual_norm.weight.copy_(norm_weight)
        expected_norm.weight.copy_(norm_weight)

    source, shift, scale, residual, update, gate = actual
    actual_modulated = h3_model._mod_scale_shift(actual_norm(source), shift, scale, segments)
    actual_gated = h3_model._mod_gate(residual, update, gate, segments)

    source, shift, scale, residual, update, gate = expected
    expected_normalized = expected_norm(source)
    expected_modulated = torch.cat(
        [expected_normalized[:, start:stop] * (1.0 + scale[row]) + shift[row] for start, stop, row in segments],
        dim=1,
    )
    expected_gated = torch.cat(
        [residual[:, start:stop] + update[:, start:stop] * gate[row] for start, stop, row in segments],
        dim=1,
    )

    torch.testing.assert_close(actual_modulated, expected_modulated, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gated, expected_gated, rtol=1e-12, atol=1e-12)
    (actual_modulated.square().mean() + actual_gated.square().mean()).backward()
    (expected_modulated.square().mean() + expected_gated.square().mean()).backward()

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor.grad, expected_tensor.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_norm.weight.grad, expected_norm.weight.grad, rtol=1e-12, atol=1e-12)


def test_model_returns_native_positive_outputs_without_comfy_sign_or_audio_slope():
    model = _tiny_model(num_layers=1)

    class FixedFinal(nn.Module):
        def forward(
            self,
            hidden_states,
            timestep_embeddings,
            *,
            video_slice,
            audio_slice,
            video_timestep_index,
            audio_timestep_index,
        ):
            del timestep_embeddings, video_timestep_index, audio_timestep_index
            batch = hidden_states.shape[0]
            return (
                torch.full((batch, video_slice.stop - video_slice.start, 96), 3.0),
                torch.full((batch, audio_slice.stop - audio_slice.start, 32), 4.0),
            )

    model.final_layer = FixedFinal()

    output = model(**_t2_inputs(batch_size=1))

    torch.testing.assert_close(output.video, torch.full_like(output.video, 3.0))
    torch.testing.assert_close(output.audio, torch.full_like(output.audio, 4.0))


def test_block_swap_runs_wait_device_assertion_forward_and_submit_in_order(monkeypatch):
    events = []

    class FakeOffloader:
        def __init__(self, blocks, device):
            self.blocks = blocks
            self.device = device

        def prepare_block_devices_before_forward(self, blocks):
            events.append("prepare")
            for block in blocks:
                block.to(self.device)

        def wait_for_block(self, index):
            events.append(f"wait:{index}")

        def submit_move_blocks_forward(self, blocks, index):
            assert blocks is self.blocks
            events.append(f"submit:{index}")

        def set_forward_only(self, value):
            events.append(f"forward_only:{value}")

    captured = {}

    def fake_create_offloader(block_type, blocks, num_blocks, blocks_to_swap, config):
        captured.update(
            block_type=block_type,
            blocks=blocks,
            num_blocks=num_blocks,
            blocks_to_swap=blocks_to_swap,
            config=config,
        )
        return FakeOffloader(blocks, config.device)

    monkeypatch.setattr("musubi_tuner.minimax_h3.model.create_offloader", fake_create_offloader)
    model = _tiny_model(num_layers=4)
    model.blocks[0].register_buffer("required_scale", torch.ones(1))
    for index, block in enumerate(model.blocks):
        block.register_forward_pre_hook(lambda module, args, index=index: events.append(f"forward:{index}"))
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=True)

    model.enable_block_swap(1, config)
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    model.prepare_block_swap_before_forward()
    model.switch_block_swap_for_inference()
    model.switch_block_swap_for_training()
    events.clear()
    model(**_t2_inputs(batch_size=1))

    assert captured["block_type"] == "minimax-h3"
    assert captured["num_blocks"] == 4
    assert captured["blocks_to_swap"] == 1
    assert events == [
        "wait:0",
        "forward:0",
        "submit:0",
        "wait:1",
        "forward:1",
        "submit:1",
        "wait:2",
        "forward:2",
        "submit:2",
        "wait:3",
        "forward:3",
        "submit:3",
    ]
    assert model.blocks[0].required_scale.device.type == "cpu"


def test_gradient_checkpointing_interface_toggles_both_flags():
    model = _tiny_model(num_layers=1)

    model.enable_gradient_checkpointing(activation_cpu_offloading=True)
    assert model.gradient_checkpointing is True
    assert model.activation_cpu_offloading is True
    model.disable_gradient_checkpointing()
    assert model.gradient_checkpointing is False
    assert model.activation_cpu_offloading is False


def test_gradient_checkpointed_forward_and_backward_recompute_the_same_block():
    model = _tiny_model(num_layers=1)
    model.enable_gradient_checkpointing()

    output = model(**_t2_inputs(batch_size=1))
    (output.video.square().mean() + output.audio.square().mean()).backward()

    assert model.blocks[0].attn.qkv_proj.weight.grad is not None


def test_block_device_assertion_catches_parameters_left_off_execution_device():
    model = _tiny_model(num_layers=1)
    model._execution_device = torch.device("meta")

    with pytest.raises(RuntimeError, match=r"parameter.*cpu.*expected meta after wait"):
        model._assert_block_device(model.blocks[0], 0)


def test_checkpoint_loader_can_require_exact_published_dtypes(tmp_path: Path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.empty(2, 2, dtype=torch.float32))

    checkpoint = tmp_path / "wrong-dtype.safetensors"
    save_file({"weight": torch.zeros(2, 2, dtype=torch.bfloat16)}, checkpoint)

    with pytest.raises(ValueError, match=r"dtype_mismatches.*expected torch.float32.*torch.bfloat16"):
        load_safetensors_module(
            Tiny,
            [checkpoint],
            device="cpu",
            dtype=None,
            strict_dtype=True,
        )


def test_checkpoint_loader_rejects_missing_rope_inv_freq(tmp_path: Path):
    state = _tiny_model(num_layers=1).state_dict()
    del state["rope.inv_freq"]
    checkpoint = tmp_path / "missing-rope.safetensors"
    save_file(state, checkpoint)

    with pytest.raises(ValueError, match=r"missing=.*rope\.inv_freq"):
        load_safetensors_module(
            lambda: MiniMaxH3Model(_tiny_config(num_layers=1), dtype=torch.float32),
            [checkpoint],
            device="cpu",
            dtype=None,
        )


def test_checkpoint_loader_rejects_uninspected_quantized_weight_pairs(tmp_path: Path):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

    checkpoint = tmp_path / "quantized.safetensors"
    save_file(
        {
            "linear.weight": torch.zeros(2, 2, dtype=torch.int8),
            "linear.weight_scale": torch.ones(1),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="require artifact inspection"):
        load_safetensors_module(Tiny, [checkpoint], device="cpu", dtype=None)


def _convrot_payload(groupsize: int) -> torch.Tensor:
    raw = json.dumps(
        {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": groupsize},
        separators=(",", ":"),
    ).encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.uint8)


def _convrot_test_config(*, pruned: bool) -> MiniMaxH3Config:
    return MiniMaxH3Config(
        hidden_size=256,
        num_layers=1,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=128,
        ffn_hidden_size=256,
        text_dim=256,
        timestep_input_dim=64,
        time_embed_hidden_size=256,
        time_embed_dim=8 if pruned else 64,
        rope_inv_freq_len=16,
        adaln_curve_grid=1025 if pruned else None,
    )


def _synthetic_convrot_h3_state(*, pruned: bool, layers: dict[str, int]):
    config = _convrot_test_config(pruned=pruned)
    model = MiniMaxH3Model(config, dtype=torch.bfloat16)
    state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
    if pruned:
        for key, tensor in list(state.items()):
            if ".adaln_proj.linear." in key:
                state[key] = tensor.to(torch.float16)

    for module_path, groupsize in layers.items():
        weight_key = f"{module_path}.weight"
        weight = state.pop(weight_key)
        quantized, scale = quantize_int8_convrot_weight(weight.float(), groupsize)
        state[weight_key] = quantized
        state[f"{module_path}.weight_scale"] = scale
        state[f"{module_path}.comfy_quant"] = _convrot_payload(groupsize)
    return config, state


def _tiny_transformer_convrot_layers(*, pruned: bool) -> dict[str, int]:
    layers = {
        "blocks.0.attn.qkv_proj": 256,
        "blocks.0.attn.out_proj": 256,
        "blocks.0.mlp.fc1": 256,
        "blocks.0.mlp.fc2": 256,
    }
    if not pruned:
        layers["blocks.0.adaln_proj.linear"] = 64
    return layers


def test_load_h3_transformer_auto_detects_full_convrot_and_per_layer_groups(tmp_path: Path, monkeypatch):
    config, state = _synthetic_convrot_h3_state(
        pruned=False,
        layers=_tiny_transformer_convrot_layers(pruned=False),
    )
    checkpoint = tmp_path / "full-int8-convrot.safetensors"
    save_file(state, checkpoint, metadata={"config": "{}", "note": "contains unrelated INT8 and quantized text"})
    monkeypatch.setattr(h3_model, "parse_h3_transformer_config", lambda _metadata: config)

    loaded = h3_model.load_h3_transformer(checkpoint, device="cpu", dtype=torch.bfloat16)

    assert loaded.is_convrot_int8
    assert loaded.convrot_int8_layer_count == 5
    assert loaded.blocks[0].attn.qkv_proj.weight.dtype is torch.int8
    assert loaded.blocks[0].attn.qkv_proj.scale_weight.dtype is torch.float32
    assert "attn.qkv_proj.scale_weight" in dict(loaded.blocks[0].named_buffers())
    assert loaded.blocks[0].attn.qkv_proj._convrot_groupsize == 256
    assert loaded.blocks[0].adaln_proj.linear._convrot_groupsize == 64
    assert torch.equal(
        loaded.blocks[0].attn.qkv_proj.scale_weight,
        state["blocks.0.attn.qkv_proj.weight_scale"],
    )
    loaded._execution_device = torch.device("cpu")
    loaded._assert_block_device(loaded.blocks[0], 0)


def test_load_h3_transformer_auto_detects_pruned_and_converts_f16_adaln(tmp_path: Path, monkeypatch):
    config, state = _synthetic_convrot_h3_state(
        pruned=True,
        layers=_tiny_transformer_convrot_layers(pruned=True),
    )
    checkpoint = tmp_path / "pruned-int8-convrot.safetensors"
    save_file(state, checkpoint)
    monkeypatch.setattr(h3_model, "_published_pruned_h3_config", lambda: config)

    loaded = h3_model.load_h3_transformer(checkpoint, device="cpu", dtype=torch.bfloat16)

    assert loaded.config.is_pruned
    assert loaded.is_convrot_int8
    assert loaded.convrot_int8_layer_count == 4
    assert loaded.adaln_t_table.dtype is torch.float32
    assert loaded.video_patch_proj.weight.dtype is torch.float32
    assert loaded.final_layer.video_out.weight.dtype is torch.float32
    assert loaded.rope.inv_freq.dtype is torch.float32
    assert loaded.blocks[0].adaln_proj.linear.weight.dtype is torch.bfloat16
    assert state["blocks.0.adaln_proj.linear.weight"].dtype is torch.float16
    assert loaded.blocks[0].attn.qkv_proj.scale_weight.dtype is torch.float32


def test_full_classifier_rejects_nonpublished_convrot_layer(tmp_path: Path, monkeypatch):
    layers = _tiny_transformer_convrot_layers(pruned=False)
    layers["condition_proj"] = 4
    config, state = _synthetic_convrot_h3_state(pruned=False, layers=layers)
    checkpoint = tmp_path / "extra-convrot-layer.safetensors"
    save_file(state, checkpoint, metadata={"config": "{}"})
    artifact = inspect_safetensors_convrot_int8([checkpoint])
    monkeypatch.setattr(h3_model, "parse_h3_transformer_config", lambda _metadata: config)

    with pytest.raises(ValueError, match=r"ConvRot.*topology.*condition_proj"):
        h3_model._classify_h3_transformer([checkpoint], artifact)


def test_full_classifier_rejects_missing_published_convrot_layer(tmp_path: Path, monkeypatch):
    layers = _tiny_transformer_convrot_layers(pruned=False)
    del layers["blocks.0.mlp.fc2"]
    config, state = _synthetic_convrot_h3_state(pruned=False, layers=layers)
    checkpoint = tmp_path / "missing-convrot-layer.safetensors"
    save_file(state, checkpoint, metadata={"config": "{}"})
    artifact = inspect_safetensors_convrot_int8([checkpoint])
    monkeypatch.setattr(h3_model, "parse_h3_transformer_config", lambda _metadata: config)

    with pytest.raises(ValueError, match=r"ConvRot.*topology.*blocks.0.mlp.fc2"):
        h3_model._classify_h3_transformer([checkpoint], artifact)


def test_published_convrot_topology_requires_released_group_sizes():
    config = MiniMaxH3Config()
    layers = {}
    for index in range(config.num_layers):
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
            module_path = f"blocks.{index}.{suffix}"
            layers[module_path] = ConvRotInt8LayerSpec(
                module_path,
                f"{module_path}.weight",
                f"{module_path}.scale_weight",
                256,
            )
        module_path = f"blocks.{index}.adaln_proj.linear"
        layers[module_path] = ConvRotInt8LayerSpec(
            module_path,
            f"{module_path}.weight",
            f"{module_path}.scale_weight",
            64,
        )
    broken = layers["blocks.0.attn.qkv_proj"]
    layers[broken.module_path] = ConvRotInt8LayerSpec(
        broken.module_path,
        broken.weight_key,
        broken.scale_key,
        64,
    )

    with pytest.raises(ValueError, match=r"group size.*blocks.0.attn.qkv_proj.*256.*64"):
        h3_model._validate_h3_convrot_topology(config, ConvRotInt8Artifact(layers, frozenset()))


@pytest.mark.parametrize(
    ("table", "match"),
    [
        (torch.empty(1025, 8, dtype=torch.float16), "F32"),
        (torch.empty(1024, 8, dtype=torch.float32), r"\[1025, 8\]"),
        (torch.empty(1025, 16, dtype=torch.float32), r"\[1025, 8\]"),
    ],
)
def test_pruned_classifier_rejects_invalid_curve_table_before_construction(tmp_path: Path, table, match):
    _config, state = _synthetic_convrot_h3_state(
        pruned=True,
        layers={"blocks.0.attn.qkv_proj": 16},
    )
    state["adaln_t_table"] = table
    checkpoint = tmp_path / "bad-table.safetensors"
    save_file(state, checkpoint)
    artifact = inspect_safetensors_convrot_int8([checkpoint])

    with pytest.raises(ValueError, match=match):
        h3_model._classify_h3_transformer([checkpoint], artifact)


def test_pruned_classifier_rejects_mixed_curve_and_time_embedder_structures(tmp_path: Path):
    _config, state = _synthetic_convrot_h3_state(
        pruned=True,
        layers={"blocks.0.attn.qkv_proj": 16},
    )
    state["time_embedder.proj_in.weight"] = torch.empty(16, 4, dtype=torch.float32)
    checkpoint = tmp_path / "mixed-adaln.safetensors"
    save_file(state, checkpoint)
    artifact = inspect_safetensors_convrot_int8([checkpoint])

    with pytest.raises(ValueError, match="both.*adaln_t_table.*time_embedder|mixed"):
        h3_model._classify_h3_transformer([checkpoint], artifact)


def test_pruned_classifier_rejects_non_eight_wide_adaln(tmp_path: Path, monkeypatch):
    config, state = _synthetic_convrot_h3_state(
        pruned=True,
        layers={"blocks.0.attn.qkv_proj": 16},
    )
    state["blocks.0.adaln_proj.linear.weight"] = torch.empty(
        config.block_adaln_out_features,
        16,
        dtype=torch.float16,
    )
    checkpoint = tmp_path / "bad-adaln-width.safetensors"
    save_file(state, checkpoint)
    artifact = inspect_safetensors_convrot_int8([checkpoint])
    monkeypatch.setattr(h3_model, "_published_pruned_h3_config", lambda: config)

    with pytest.raises(ValueError, match=r"adaln_proj.*expected shape.*8"):
        h3_model._classify_h3_transformer([checkpoint], artifact)


def test_pruned_classifier_requires_convrot_artifact(tmp_path: Path):
    config = _tiny_config(num_layers=1, pruned=True)
    state = MiniMaxH3Model(config, dtype=torch.bfloat16).state_dict()
    checkpoint = tmp_path / "pruned-bf16.safetensors"
    save_file(state, checkpoint)

    with pytest.raises(ValueError, match="pruned.*ConvRot"):
        h3_model._classify_h3_transformer([checkpoint], None)
