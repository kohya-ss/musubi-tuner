import argparse
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model
from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import apply_convrot_int8_monkey_patch
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


def _stub(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_training_module(monkeypatch):
    target = "_isolated_minimax_h3_train_network"
    noop = lambda *args, **kwargs: None
    _stub(monkeypatch, "musubi_tuner.minimax_h3.audio_vae", load_audio_vae=noop)
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.generation_inputs",
        VIDEO_VAE_SPATIAL_RATIO=16,
        build_reference_geometries=noop,
        decode_generation_visuals=noop,
        encode_audio_conditions=noop,
        encode_visual_conditions=noop,
        load_generation_record=noop,
        module_device_dtype=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.media",
        H3_AUDIO_SPEC=object(),
        audio_latent_frames=noop,
        video_latent_frames=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.sampling",
        augment_condition_latents=noop,
        create_sampling_generator=noop,
        initialize_target_latents=noop,
        sample_joint_av=noop,
        synchronize_decoded_av=noop,
        write_joint_av=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.text_encoder",
        TEACHER_CONDITIONS_REF="ref",
        build_presentation=noop,
        encode_h3_presentation=noop,
        load_h3_processor=noop,
        load_h3_text_encoder=noop,
        load_h3_uncond_cache=noop,
        normalize_teacher_conditions=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.video_vae",
        VIDEO_VAE_DECODE_DTYPE=torch.float16,
        VIDEO_VAE_ENCODE_DTYPE=torch.bfloat16,
        load_video_vae=noop,
    )
    _stub(monkeypatch, "musubi_tuner.minimax_h3_cache_latents", PyAVH3MediaDecoder=object)

    def add_audio_train_args(parser):
        parser.add_argument("--audio_loss_weight", type=float, default=1.0)
        parser.add_argument("--video_only", action="store_true")
        return parser

    _stub(
        monkeypatch,
        "musubi_tuner.training.audio_loss",
        add_audio_train_args=add_audio_train_args,
        effective_audio_loss_weights=noop,
        log_audio_supervision_summary=noop,
        scan_audio_supervised_fraction=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.training.parser_common",
        read_config_from_file=lambda args, parser: args,
        setup_parser_common=lambda: argparse.ArgumentParser(),
    )
    _stub(monkeypatch, "musubi_tuner.training.sampling_prompts", load_prompts=noop)

    class NetworkTrainer:
        pass

    _stub(monkeypatch, "musubi_tuner.training.trainer_base", DiTOutput=SimpleNamespace, NetworkTrainer=NetworkTrainer)
    _stub(monkeypatch, "musubi_tuner.utils.device_utils", clean_memory_on_device=noop, synchronize_device=noop)
    model_utils = _stub(monkeypatch, "musubi_tuner.utils.model_utils", compile_transformer=noop)
    spec = importlib.util.spec_from_file_location(
        target,
        Path(__file__).resolve().parents[1] / "src" / "musubi_tuner" / "minimax_h3_train_network.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, target, module)
    spec.loader.exec_module(module)
    module.model_utils = model_utils
    return module


def _trainer_args(**overrides):
    values = {
        "convrot_int8": False,
        "convrot_int8_bwd": "bf16",
        "base_weights": None,
        "disable_numpy_memmap": False,
        "prune_adaln": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_training_parser_exposes_the_convrot_flags(monkeypatch):
    train = _load_training_module(monkeypatch)
    parser = train.minimax_h3_setup_parser(argparse.ArgumentParser())

    defaults = parser.parse_args(["--task", "t2va"])
    int8 = parser.parse_args(["--task", "t2va", "--convrot_int8", "--convrot_int8_bwd", "int8"])

    assert defaults.convrot_int8 is False
    assert defaults.convrot_int8_bwd == "bf16"
    assert int8.convrot_int8 is True
    assert int8.convrot_int8_bwd == "int8"


def test_training_detection_guards_merges_and_compile_policy(monkeypatch):
    train = _load_training_module(monkeypatch)
    trainer = train.MiniMaxH3NetworkTrainer()
    bf16 = SimpleNamespace(is_convrot_int8=False, blocks=[])
    int8 = SimpleNamespace(is_convrot_int8=True, blocks=[])
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    # int8 backward needs an INT8 base (flag or auto-detected pre-quantized checkpoint)
    with pytest.raises(ValueError, match="convrot_int8_bwd.*INT8"):
        trainer.on_transformer_loaded(_trainer_args(convrot_int8_bwd="int8"), accelerator, bf16)
    with pytest.raises(ValueError, match="base_weights.*INT8"):
        trainer.on_transformer_loaded(_trainer_args(base_weights=["base.safetensors"]), accelerator, int8)
    with pytest.raises(ValueError, match=r"int8.*CUDA"):
        trainer.on_transformer_loaded(_trainer_args(convrot_int8_bwd="int8"), accelerator, int8)

    captured = {}
    monkeypatch.setattr(
        train,
        "load_h3_transformer",
        lambda *args, **kwargs: captured.update(load=kwargs) or int8,
    )
    monkeypatch.setattr(
        train.model_utils,
        "compile_transformer",
        lambda *args, **kwargs: captured.update(compile=kwargs) or int8,
    )
    trainer.blocks_to_swap = 0
    args = _trainer_args(convrot_int8_bwd="int8")

    assert trainer.load_transformer(accelerator, args, "dit.safetensors", "torch", False, "cpu", torch.bfloat16) is int8
    assert trainer._convrot_int8_active is True
    assert trainer.compile_transformer(args, int8) is int8
    assert captured["load"]["convrot_int8_bwd"] == "int8"
    assert captured["compile"]["disable_linear"] is True


def _load_generation_module(monkeypatch):
    target = "_isolated_minimax_h3_generate_video"
    noop = lambda *args, **kwargs: None
    _stub(monkeypatch, "transformers", CLIPTextModel=torch.nn.Module)
    _stub(monkeypatch, "musubi_tuner.minimax_h3.audio_vae", load_audio_vae=noop)
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.generation_inputs",
        VIDEO_VAE_SPATIAL_RATIO=16,
        build_reference_geometries=noop,
        decode_generation_visuals=noop,
        encode_audio_conditions=noop,
        encode_visual_conditions=noop,
        load_generation_record=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3.video_vae",
        VIDEO_VAE_DECODE_DTYPE=torch.float16,
        VIDEO_VAE_ENCODE_DTYPE=torch.bfloat16,
        load_video_vae=noop,
    )
    _stub(
        monkeypatch,
        "musubi_tuner.minimax_h3_cache_latents",
        PyAVH3MediaDecoder=object,
        fingerprint_file=noop,
    )
    spec = importlib.util.spec_from_file_location(
        target,
        Path(__file__).resolve().parents[1] / "src" / "musubi_tuner" / "minimax_h3_generate_video.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, target, module)
    spec.loader.exec_module(module)
    return module


def test_generation_selects_merge_for_bf16_and_attachment_for_int8(monkeypatch):
    generate = _load_generation_module(monkeypatch)
    calls = []
    attached = [object()]
    monkeypatch.setattr(generate, "_merge_lora_weights", lambda transformer, args: calls.append(("merge", transformer)))
    monkeypatch.setattr(
        generate,
        "_apply_lora_weights",
        lambda transformer, args, device: calls.append(("attach", transformer, device)) or attached,
        raising=False,
    )
    device = torch.device("cpu")
    bf16 = SimpleNamespace(is_convrot_int8=False)
    int8 = SimpleNamespace(is_convrot_int8=True)

    # plain BF16 base: one-time destructive CPU merge
    args = SimpleNamespace(lora_weight=["adapter.safetensors"], convrot_int8=False)
    assert generate._configure_lora_weights(bf16, args, device, prequantized=False) == []
    # pre-quantized INT8 base (auto-detected): runtime additive branches
    assert generate._configure_lora_weights(int8, args, device, prequantized=True) is attached
    # BF16 base + --convrot_int8: merged during the streaming load, nothing to do here
    dynamic_args = SimpleNamespace(lora_weight=["adapter.safetensors"], convrot_int8=True)
    assert generate._configure_lora_weights(int8, dynamic_args, device, prequantized=False) == []
    # no LoRA: nothing happens on any route
    no_lora = SimpleNamespace(lora_weight=None, convrot_int8=False)
    assert generate._configure_lora_weights(bf16, no_lora, device, prequantized=False) == []
    # --lora_runtime_attach overrides both merge routes with runtime branches (the merge
    # rounds small-magnitude LoRAs -- e.g. teacher matching -- out of the BF16 weights)
    attach_args = SimpleNamespace(lora_weight=["adapter.safetensors"], convrot_int8=False, lora_runtime_attach=True)
    assert generate._configure_lora_weights(bf16, attach_args, device, prequantized=False) is attached
    attach_int8_args = SimpleNamespace(lora_weight=["adapter.safetensors"], convrot_int8=True, lora_runtime_attach=True)
    assert generate._configure_lora_weights(int8, attach_int8_args, device, prequantized=False) is attached
    assert calls == [("merge", bf16), ("attach", int8, device), ("attach", bf16, device), ("attach", int8, device)]


def _tiny_model(*, num_layers: int = 1):
    config = MiniMaxH3Config(
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
    )
    return MiniMaxH3Model(config, dtype=torch.float32)


def _prepare_int8_targets(model):
    target_paths = tuple(
        f"blocks.{index}.{suffix}"
        for index in range(len(model.blocks))
        for suffix in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
    )
    state_dict = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
    for module_path in target_paths:
        weight = state_dict.pop(f"{module_path}.weight")
        quantized_weight, scale = quantize_int8_convrot_weight(weight, 4)
        state_dict[f"{module_path}.weight"] = quantized_weight
        state_dict[f"{module_path}.scale_weight"] = scale
    apply_convrot_int8_monkey_patch(model, state_dict, groupsize_map={path: 4 for path in target_paths})
    model.requires_grad_(False)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return target_paths


def test_attached_lora_forward_does_not_mutate_int8_base(tmp_path: Path, monkeypatch):
    generate = _load_generation_module(monkeypatch)
    source = _tiny_model()
    source_network = generate.lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, source)
    source_network.apply_to(None, source, apply_text_encoder=False, apply_unet=True)
    with torch.no_grad():
        for name, parameter in source_network.named_parameters():
            if "lora_up" in name:
                parameter.fill_(0.1)
    lora_path = tmp_path / "adapter.safetensors"
    save_file({key: value.detach().contiguous() for key, value in source_network.state_dict().items()}, lora_path)

    model = _tiny_model()
    target_paths = _prepare_int8_targets(model)
    snapshots = {path: model.get_submodule(path).weight.detach().clone() for path in target_paths}
    args = SimpleNamespace(
        lora_weight=[str(lora_path)],
        lora_multiplier=[0.75],
        include_patterns=None,
        exclude_patterns=None,
    )

    networks = generate._apply_lora_weights(model, args, torch.device("cpu"))
    output = model.blocks[0].attn.qkv_proj(torch.randn(3, 16))

    assert len(networks) == 1
    assert output.shape == (3, 48)
    assert all(not parameter.requires_grad for parameter in networks[0].parameters())
    for path, expected in snapshots.items():
        assert torch.equal(model.get_submodule(path).weight, expected)


def test_lora_gradient_reaches_adapter_over_checkpointed_int8_base(monkeypatch):
    generate = _load_generation_module(monkeypatch)
    model = _tiny_model()
    target_paths = _prepare_int8_targets(model)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    model.train()
    network = generate.lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.prepare_optimizer_params(unet_lr=1e-4)
    layout = build_h3_layout(
        task="t2va",
        text_length=3,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )

    output = model(
        video_latents=torch.randn(1, 24, 2, 4, 4),
        audio_latents=torch.randn(1, 32, 2, 8),
        text_hidden_states=torch.randn(1, 3, 12),
        text_token_tags=torch.tensor([[1, 0, 1]]),
        layout=layout,
        model_t_video=torch.tensor(0.25),
        model_t_audio=torch.tensor(0.75),
    )
    (output.video.square().mean() + output.audio.square().mean()).backward()

    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in network.parameters())
    assert all(model.get_submodule(path).weight.grad is None for path in target_paths)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for real block swap")
@pytest.mark.parametrize("use_pinned_memory", [False, True])
def test_cuda_block_swap_keeps_convrot_scales_resident_during_lora_backward(monkeypatch, use_pinned_memory):
    generate = _load_generation_module(monkeypatch)
    device = torch.device("cuda")
    model = _tiny_model(num_layers=3)
    target_paths = _prepare_int8_targets(model)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    model.train()
    network = generate.lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    network.prepare_optimizer_params(unet_lr=1e-4)
    network.to(device)

    model.enable_block_swap(
        1,
        BlockSwapConfig(device=device, supports_backward=True, use_pinned_memory=use_pinned_memory),
    )
    model.move_to_device_except_swap_blocks(device)
    model.switch_block_swap_for_training()
    layout = build_h3_layout(
        task="t2va",
        text_length=3,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )

    output = model(
        video_latents=torch.randn(1, 24, 2, 4, 4, device=device),
        audio_latents=torch.randn(1, 32, 2, 8, device=device),
        text_hidden_states=torch.randn(1, 3, 12, device=device),
        text_token_tags=torch.tensor([[1, 0, 1]], device=device),
        layout=layout,
        model_t_video=torch.tensor([0.25], device=device),
        model_t_audio=torch.tensor([0.75], device=device),
    )
    (output.video.square().mean() + output.audio.square().mean()).backward()
    torch.cuda.synchronize(device)

    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in network.parameters())
    for path in target_paths:
        module = model.get_submodule(path)
        assert module.weight.dtype is torch.int8
        assert module.weight.grad is None
        assert module.scale_weight.dtype is torch.float32
        assert module.scale_weight.device.type == "cuda"
