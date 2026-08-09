from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import musubi_tuner.minimax_h3_train_network as h3_module
from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model
from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import apply_convrot_int8_monkey_patch
from musubi_tuner.minimax_h3_train_network import (
    H3SamplingResources,
    MiniMaxH3NetworkTrainer,
    minimax_h3_setup_parser,
)
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import DiTOutput


def test_process_batch_accumulates_observed_audio_supervision(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    for present in (1.0, 0.0):
        batch = _training_batch()
        batch["audio_present"] = torch.tensor([present], dtype=torch.float32)
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )

    assert trainer._audio_items_seen == 2
    assert trainer._audio_supervised_seen == 1
    assert trainer.extra_metadata(args)["ss_minimax_h3_supervised_audio_fraction"] == 0.5


def test_h3_rejects_the_single_vae_prompt_seam_with_a_pointer_to_prepare_sampling():
    with pytest.raises(NotImplementedError, match="prepare_sampling"):
        MiniMaxH3NetworkTrainer().process_sample_prompts(_trainer_args(), _Accelerator(), "prompts.json")


def test_h3_warns_after_the_first_epoch_when_no_real_audio_was_seen(caplog):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._audio_items_seen = 3
    trainer._audio_supervised_seen = 0
    caplog.set_level("WARNING")

    trainer.on_epoch_end(_trainer_args(), SimpleNamespace(is_main_process=True), None, None, 1)

    assert "audio loss is always 0" in caplog.text


@pytest.mark.parametrize(
    "overrides, items_seen, supervised_seen, epoch",
    [
        ({}, 3, 1, 1),  # real audio was seen
        ({}, 3, 0, 2),  # later epochs stay silent
        ({"video_only": True}, 3, 0, 1),
        ({"audio_loss_weight": 0.0}, 3, 0, 1),
        ({}, 0, 0, 1),  # nothing seen (no step ran on this rank)
    ],
)
def test_h3_epoch_end_stays_silent_unless_audio_supervision_was_expected(caplog, overrides, items_seen, supervised_seen, epoch):
    trainer = MiniMaxH3NetworkTrainer()
    trainer._audio_items_seen = items_seen
    trainer._audio_supervised_seen = supervised_seen
    caplog.set_level("WARNING")

    trainer.on_epoch_end(_trainer_args(**overrides), SimpleNamespace(is_main_process=True), None, None, epoch)

    assert caplog.text == ""


class _Accelerator:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)

    def autocast(self):
        return nullcontext()


class _RecordingTransformer:
    def __init__(self, video_prediction: float = 2.0, audio_prediction: float = -1.0):
        self.video_prediction = video_prediction
        self.audio_prediction = audio_prediction
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            video=torch.full_like(kwargs["video_latents"], self.video_prediction),
            audio=torch.full_like(kwargs["audio_latents"], self.audio_prediction),
        )


def _trainer_args(**overrides):
    values = {
        "timestep_sampling": "uniform",
        "weighting_scheme": "none",
        "discrete_flow_shift": 1.0,
        "h3_shift_video": 12.0,
        "h3_shift_audio": 3.0,
        "h3_visual_cond_clean": 0.999,
        "h3_audio_cond_clean": 1.0,
        "min_timestep": None,
        "max_timestep": None,
        "preserve_distribution_shape": False,
        "blocks_to_swap": 0,
        "sample_prompts": None,
        "gradient_checkpointing": False,
        "task": "t2va",
        "video_only": False,
        "audio_loss_weight": 1.0,
        "convrot_int8": False,
        "convrot_int8_bwd": "bf16",
        "base_weights": None,
        "xm_best_of_k": 1,
        "h3_video_best_of_k": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _training_batch(batch_size: int = 1, *, text_length: int = 3):
    return {
        "latents_audio": torch.full((batch_size, 32, 2, 8), 4.0),
        "audio_present": torch.ones(batch_size, dtype=torch.float32),
        "mmh3_hidden_states": [torch.full((text_length, 12), float(index)) for index in range(batch_size)],
        "mmh3_token_tags": [torch.tensor([1, 0, 1][:text_length], dtype=torch.int64) for _ in range(batch_size)],
        "timesteps": None,
    }


def test_h3_parser_defaults_to_the_only_supported_training_coordinates():
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--task", "t2va"])

    assert args.timestep_sampling == "uniform"
    assert args.weighting_scheme == "none"
    assert args.discrete_flow_shift == 1.0
    assert args.h3_shift_video == 12.0
    assert args.h3_shift_audio == 3.0
    assert args.h3_visual_cond_clean == 0.999
    assert args.h3_audio_cond_clean == 1.0
    assert args.network_module == "networks.lora_minimax_h3"
    assert args.video_only is False
    assert args.audio_loss_weight == 1.0
    assert args.convrot_int8 is False
    assert args.convrot_int8_bwd == "bf16"
    assert "--h3_video_only" not in parser.format_help()


def test_h3_parser_exposes_separate_video_best_of_k_option(tmp_path, monkeypatch):
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())
    assert parser.parse_args(["--task", "t2va"]).h3_video_best_of_k == 1
    assert parser.parse_args(["--task", "t2va", "--h3_video_best_of_k", "3"]).h3_video_best_of_k == 3

    config = tmp_path / "h3_best_of_k.toml"
    config.write_text('task = "t2va"\nh3_video_best_of_k = 4\n', encoding="utf-8")
    common_parser = minimax_h3_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["minimax_h3_train_network", "--config_file", str(config)])
    args = common_parser.parse_args()
    assert read_config_from_file(args, common_parser).h3_video_best_of_k == 4


def test_h3_best_of_k_validation_uses_the_h3_option_and_rejects_forward_xm():
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=r"--h3_video_best_of_k.*at least 1"):
        trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=0, xm_best_of_k=1))
    with pytest.raises(ValueError, match=r"not Forward XM.*--h3_video_best_of_k"):
        trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=1, xm_best_of_k=2))

    trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=2, xm_best_of_k=1))
    assert trainer._best_of_k_count == 2
    assert trainer._best_of_k_enabled is True


@pytest.mark.parametrize(
    ("option", "toml_value"),
    [
        ("xm_best_of_k", "0"),
        ("xm_best_of_k", "1.0"),
        ("xm_best_of_k", "true"),
        ("xm_best_of_k", '"1"'),
        ("h3_video_best_of_k", "0"),
        ("h3_video_best_of_k", "1.0"),
        ("h3_video_best_of_k", "true"),
        ("h3_video_best_of_k", '"1"'),
    ],
)
def test_h3_best_of_k_validation_rejects_invalid_toml_types_and_zero(tmp_path, monkeypatch, option, toml_value):
    config = tmp_path / "invalid_h3_best_of_k.toml"
    config.write_text(f'task = "t2va"\n{option} = {toml_value}\n', encoding="utf-8")
    parser = minimax_h3_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["minimax_h3_train_network", "--config_file", str(config)])
    args = read_config_from_file(parser.parse_args(), parser)
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=rf"--{option}.*(?:integer|at least 1)"):
        trainer._validate_and_init_best_of_k(args)

    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)


def test_h3_best_of_k_startup_log_names_the_distinct_objective(caplog):
    caplog.set_level("INFO")
    trainer = MiniMaxH3NetworkTrainer()
    trainer._validate_and_init_best_of_k(_trainer_args(h3_video_best_of_k=2, xm_best_of_k=1))

    assert "video-focused best-of-K heuristic" in caplog.text
    assert "not Forward XM" in caplog.text
    assert "selection objective: video only" in caplog.text
    assert "final objective: video + weighted audio" in caplog.text
    assert "K=2" in caplog.text
    assert "1.67x" in caplog.text
    assert "Forward XM enabled for MiniMax-H3" not in caplog.text


def test_h3_parser_accepts_int8_convrot_backward_mode():
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--task", "t2va", "--convrot_int8_bwd", "int8"])

    assert args.convrot_int8_bwd == "int8"


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"timestep_sampling": "sigma"}, "timestep_sampling"),
        ({"weighting_scheme": "sigma_sqrt"}, "weighting_scheme"),
        ({"discrete_flow_shift": 1.1}, "discrete_flow_shift"),
        ({"h3_shift_video": 0.0}, "h3_shift_video"),
        ({"h3_shift_audio": 101.0}, "h3_shift_audio"),
        ({"h3_visual_cond_clean": -0.1}, "h3_visual_cond_clean"),
        ({"h3_audio_cond_clean": 1.1}, "h3_audio_cond_clean"),
        ({"blocks_to_swap": 49}, "blocks_to_swap"),
    ],
)
def test_h3_trainer_rejects_training_knobs_with_the_wrong_coordinate_contract(override, message):
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=message):
        trainer.handle_model_specific_args(_trainer_args(**override))


def test_h3_trainer_allows_training_time_sample_prompts():
    trainer = MiniMaxH3NetworkTrainer()

    trainer.handle_model_specific_args(_trainer_args(sample_prompts="prompts.json"))


def test_h3_trainer_validates_backward_mode_and_destructive_merges_after_detection():
    trainer = MiniMaxH3NetworkTrainer()
    bf16 = SimpleNamespace(is_convrot_int8=False)
    int8 = SimpleNamespace(is_convrot_int8=True)

    with pytest.raises(ValueError, match="convrot_int8_bwd.*INT8"):
        trainer.on_transformer_loaded(_trainer_args(convrot_int8_bwd="int8"), None, bf16)
    with pytest.raises(ValueError, match="base_weights.*INT8"):
        trainer.on_transformer_loaded(_trainer_args(base_weights=["base.safetensors"]), None, int8)
    with pytest.raises(ValueError, match=r"int8.*CUDA"):
        trainer.on_transformer_loaded(
            _trainer_args(convrot_int8_bwd="int8"),
            SimpleNamespace(device=torch.device("cpu")),
            int8,
        )
    trainer.on_transformer_loaded(_trainer_args(), None, int8)


def test_h3_trainer_passes_backward_mode_to_loader_and_excludes_int8_linears_from_compile(monkeypatch):
    import musubi_tuner.minimax_h3_train_network as train

    captured = {}
    transformer = SimpleNamespace(blocks=[], is_convrot_int8=True)
    monkeypatch.setattr(
        train,
        "load_h3_transformer",
        lambda *args, **kwargs: captured.update(load=kwargs) or transformer,
    )
    monkeypatch.setattr(
        train.model_utils,
        "compile_transformer",
        lambda *args, **kwargs: captured.update(compile=kwargs) or transformer,
    )
    trainer = train.MiniMaxH3NetworkTrainer()
    trainer.blocks_to_swap = 0
    args = _trainer_args(convrot_int8_bwd="int8", disable_numpy_memmap=False)
    accelerator = SimpleNamespace(device=torch.device("cpu"))

    loaded = trainer.load_transformer(accelerator, args, "dit.safetensors", "torch", False, "cpu", torch.bfloat16)
    compiled = trainer.compile_transformer(args, transformer)

    assert loaded is transformer
    assert trainer._convrot_int8_active is True
    assert compiled is transformer
    assert captured["load"]["convrot_int8_bwd"] == "int8"
    assert captured["compile"]["disable_linear"] is True


def test_h3_parser_exposes_the_dual_vae_and_text_assets_needed_for_training_samples():
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(
        [
            "--task",
            "t2va",
            "--video_vae",
            "video.safetensors",
            "--audio_vae",
            "audio.safetensors",
            "--text_encoder",
            "qwen.safetensors",
        ]
    )

    assert args.video_vae == "video.safetensors"
    assert args.audio_vae == "audio.safetensors"
    assert args.text_encoder == "qwen.safetensors"
    assert args.processor == "Qwen/Qwen3-VL-32B-Instruct"
    assert args.processor_revision is None
    assert args.h3_allow_experimental_sample_duration is False


def test_h3_training_sample_uses_the_live_transformer_then_decodes_and_muxes_both_modalities(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_train_network as train

    events = []

    class Transformer:
        training = True

        def eval(self):
            events.append("transformer_eval")
            self.training = False
            return self

        def train(self, mode=True):
            events.append(("transformer_train", mode))
            self.training = mode
            return self

        def __call__(self, **kwargs):
            events.append("sample_live_transformer")
            return SimpleNamespace(
                video=torch.zeros_like(kwargs["video_latents"]),
                audio=torch.zeros_like(kwargs["audio_latents"]),
            )

    class VideoVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchor", torch.tensor(0.0, dtype=torch.float16))

        def to(self, *args, **kwargs):
            events.append("move_video_vae")
            return super().to(*args, **kwargs)

        def decode(self, latents):
            events.append("decode_video")
            assert latents.shape == (1, 24, 2, 4, 4)
            return torch.zeros(1, 3, 5, 8, 8)

    class AudioVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("anchor", torch.tensor(0.0, dtype=torch.float32))

        def to(self, *args, **kwargs):
            events.append("move_audio_vae")
            return super().to(*args, **kwargs)

        def decode(self, latents):
            events.append("decode_audio")
            assert latents.shape == (1, 32, 2, 8)
            return torch.zeros(1, 2, 6667)

    captured = {}
    monkeypatch.setattr(
        train,
        "write_joint_av",
        lambda decoded, output_path: captured.update(decoded=decoded, output_path=Path(output_path)),
    )
    trainer = train.MiniMaxH3NetworkTrainer()
    sample_resources = train.H3SamplingResources(video_vae=VideoVAE(), audio_vae=AudioVAE())
    layout = build_h3_layout(
        task="t2va",
        text_length=3,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )
    sample_parameter = {
        "enum": 0,
        "prompt": "joint sample",
        "sample_steps": 2,
        "width": 64,
        "height": 64,
        "frame_count": 5,
        "seed": 123,
        "h3_layout": layout,
        "h3_text_hidden_states": torch.zeros(1, 3, 12),
        "h3_text_token_tags": torch.tensor([[1, 0, 1]], dtype=torch.int64),
        "h3_visual_conditions": (),
        "h3_audio_conditions": (),
    }
    args = _trainer_args(
        output_dir=str(tmp_path),
        output_name="h3",
    )
    transformer = Transformer()

    output = trainer.sample_image_inference(
        _Accelerator(),
        args,
        transformer,
        torch.bfloat16,
        sample_resources,
        str(tmp_path),
        sample_parameter,
        None,
        12,
    )

    assert output == captured["output_path"]
    assert captured["output_path"].suffix == ".mp4"
    assert captured["decoded"].video.shape == (5, 8, 8, 3)
    assert captured["decoded"].audio.shape == (2, 6667)
    assert events.index("sample_live_transformer") < events.index("decode_video") < events.index("decode_audio")
    assert transformer.training is True


def test_prepare_training_samples_encodes_text_once_and_returns_both_vaes_as_sampling_resources(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_train_network as train

    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(
        json.dumps(
            [
                {
                    "prompt": "joint sample",
                    "width": 64,
                    "height": 64,
                    "frame_count": 23,
                    "sample_steps": 2,
                    "seed": 123,
                }
            ]
        ),
        encoding="utf-8",
    )
    asset_paths = {}
    for name in ("video_vae", "audio_vae", "text_encoder"):
        path = tmp_path / f"{name}.safetensors"
        path.touch()
        asset_paths[name] = str(path)
    args = _trainer_args(
        sample_prompts=str(prompt_file),
        processor="processor",
        processor_revision=None,
        h3_allow_experimental_sample_duration=True,
        disable_numpy_memmap=False,
        **asset_paths,
    )
    events = []

    class TextEncoder(torch.nn.Module):
        pass

    class VideoVAE(torch.nn.Module):
        vae_ratio = 16

    class AudioVAE(torch.nn.Module):
        pass

    record = SimpleNamespace(references=())
    monkeypatch.setattr(train, "PyAVH3MediaDecoder", lambda: object())
    monkeypatch.setattr(train, "load_h3_processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train,
        "load_h3_text_encoder",
        lambda *args, **kwargs: events.append("load_text_encoder") or TextEncoder(),
    )
    monkeypatch.setattr(
        train,
        "load_generation_record",
        lambda *args, **kwargs: record,
    )
    monkeypatch.setattr(train, "decode_generation_visuals", lambda *args, **kwargs: ({}, {}))
    monkeypatch.setattr(train, "build_presentation", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train,
        "encode_h3_presentation",
        lambda *args, **kwargs: (
            events.append("encode_text") or torch.zeros(3, 12),
            torch.tensor([1, 0, 1], dtype=torch.int64),
        ),
    )
    monkeypatch.setattr(
        train,
        "load_video_vae",
        lambda *args, **kwargs: events.append(("load_video_vae", kwargs["dtype"])) or VideoVAE(),
    )
    monkeypatch.setattr(
        train,
        "load_audio_vae",
        lambda *args, **kwargs: events.append("load_audio_vae") or AudioVAE(),
    )
    monkeypatch.setattr(train, "clean_memory_on_device", lambda *args, **kwargs: None)
    trainer = train.MiniMaxH3NetworkTrainer()

    sample_parameters, sample_resources = trainer.prepare_sampling(args, _Accelerator(), torch.bfloat16)

    assert isinstance(sample_resources, H3SamplingResources)
    assert events == ["load_text_encoder", "encode_text", ("load_video_vae", torch.float16), "load_audio_vae"]
    assert isinstance(sample_resources.video_vae, VideoVAE)
    assert isinstance(sample_resources.audio_vae, AudioVAE)
    assert len(sample_parameters) == 1
    parameter = sample_parameters[0]
    assert parameter["h3_layout"].task == "t2va"
    assert parameter["frame_count"] == 22
    assert parameter["h3_layout"].target_video == H3VideoGeometry(7, 4, 4)
    assert parameter["h3_layout"].target_audio_frames == 37
    assert parameter["h3_text_hidden_states"].shape == (1, 3, 12)
    assert parameter["h3_text_token_tags"].shape == (1, 3)
    assert parameter["h3_visual_conditions"] == ()
    assert parameter["h3_audio_conditions"] == ()
    assert not any(key.startswith("_h3_") for key in parameter)


def test_prepare_ref_training_sample_carries_ordered_visual_and_audio_conditions_into_the_layout(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_train_network as train

    reference_jsonl = tmp_path / "references.jsonl"
    reference_jsonl.touch()
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(
        json.dumps(
            [
                {
                    "reference_jsonl": str(reference_jsonl),
                    "reference_index": 0,
                    "width": 64,
                    "height": 64,
                    "frame_count": 5,
                    "sample_steps": 2,
                }
            ]
        ),
        encoding="utf-8",
    )
    asset_paths = {}
    for name in ("video_vae", "audio_vae", "text_encoder"):
        path = tmp_path / f"{name}.safetensors"
        path.touch()
        asset_paths[name] = str(path)
    args = _trainer_args(
        task="ref2va",
        sample_prompts=str(prompt_file),
        processor="processor",
        processor_revision=None,
        h3_allow_experimental_sample_duration=True,
        disable_numpy_memmap=False,
        **asset_paths,
    )

    class EmptyModule(torch.nn.Module):
        pass

    class VideoVAE(EmptyModule):
        vae_ratio = 16

        def __init__(self):
            super().__init__()
            self.register_buffer("dtype_probe", torch.zeros(1))

    reference = SimpleNamespace(type="video", path="reference.mp4", audio=object())
    record = SimpleNamespace(references=(reference,))
    visual = torch.zeros(1, 24, 2, 4, 4)
    audio = torch.zeros(1, 32, 2, 8)
    record_loads = []
    visual_decodes = []
    audio_frame_counts = []
    monkeypatch.setattr(train, "PyAVH3MediaDecoder", lambda: object())
    monkeypatch.setattr(train, "load_h3_processor", lambda *args, **kwargs: object())
    monkeypatch.setattr(train, "load_h3_text_encoder", lambda *args, **kwargs: EmptyModule())
    monkeypatch.setattr(
        train,
        "load_generation_record",
        lambda *args, **kwargs: record_loads.append(record) or record,
    )
    monkeypatch.setattr(
        train,
        "decode_generation_visuals",
        lambda request, loaded_record, decoder: (
            visual_decodes.append(loaded_record) or ({reference.path: torch.zeros(5, 64, 64, 3)}, {})
        ),
    )
    monkeypatch.setattr(train, "build_presentation", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        train,
        "encode_h3_presentation",
        lambda *args, **kwargs: (torch.zeros(3, 12), torch.tensor([1, 0, 1], dtype=torch.int64)),
    )
    video_vae_load_dtypes = []
    monkeypatch.setattr(
        train,
        "load_video_vae",
        lambda *args, **kwargs: video_vae_load_dtypes.append(kwargs["dtype"]) or VideoVAE(),
    )
    monkeypatch.setattr(train, "load_audio_vae", lambda *args, **kwargs: EmptyModule())
    monkeypatch.setattr(
        train,
        "encode_visual_conditions",
        lambda *args, **kwargs: ((visual,), (), {0: H3VideoGeometry(2, 4, 4)}),
    )

    def fake_encode_audio_conditions(request, loaded_record, decoder, audio_vae, *, reference_video_frame_counts):
        del request, decoder, audio_vae
        assert loaded_record is record
        audio_frame_counts.append(reference_video_frame_counts)
        return (audio,), {0: 8}

    monkeypatch.setattr(train, "encode_audio_conditions", fake_encode_audio_conditions)
    monkeypatch.setattr(train, "clean_memory_on_device", lambda *args, **kwargs: None)
    trainer = train.MiniMaxH3NetworkTrainer()

    sample_parameters, sample_resources = trainer.prepare_sampling(args, _Accelerator(), torch.bfloat16)

    parameter = sample_parameters[0]
    assert parameter["h3_layout"].task == "ref2va"
    assert parameter["h3_layout"].references == (H3ReferenceGeometry("video", video=H3VideoGeometry(2, 4, 4), audio_frames=8),)
    assert parameter["h3_visual_conditions"] == (visual,)
    assert parameter["h3_audio_conditions"] == (audio,)
    assert video_vae_load_dtypes == [torch.float32]
    assert sample_resources.video_vae.dtype_probe.dtype is torch.float16
    assert record_loads == [record]
    assert visual_decodes == [record, record]
    assert audio_frame_counts == [{0: 5}]


def test_process_batch_uses_one_shared_base_time_and_independent_audio_noise(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch()
    video_latents = torch.full((1, 24, 2, 4, 4), 5.0)
    video_noise = torch.full_like(video_latents, -2.0)
    real_randn_like = torch.randn_like

    def fixed_audio_noise(tensor, *positional, **kwargs):
        if tuple(tensor.shape) == (1, 32, 2, 8):
            return torch.full_like(tensor, 3.0)
        return real_randn_like(tensor, *positional, **kwargs)

    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", fixed_audio_noise)

    trainer._validate_and_init_best_of_k(args)
    rng_before = torch.random.get_rng_state().clone()
    loss, metrics = trainer._process_batch_for_training(
        args,
        _Accelerator(),
        transformer,
        None,
        batch,
        video_latents,
        video_noise,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    rng_after = torch.random.get_rng_state().clone()

    call = transformer.calls[0]
    assert call["model_t_video"].shape == torch.Size([])
    assert call["model_t_audio"].shape == torch.Size([])
    assert call["model_t_video"].item() == pytest.approx(0.2)
    assert call["model_t_audio"].item() == pytest.approx(0.5)
    assert torch.allclose(call["video_latents"], torch.full_like(video_latents, -0.6))
    assert torch.allclose(call["audio_latents"], torch.full_like(batch["latents_audio"], 3.5))
    video_target = video_latents - video_noise
    audio_target = batch["latents_audio"] - 3.0
    expected_video_loss = torch.nn.functional.mse_loss(torch.full_like(video_target, 2.0), video_target)
    expected_audio_loss = torch.nn.functional.mse_loss(torch.full_like(audio_target, -1.0), audio_target)
    assert torch.allclose(loss, expected_video_loss + expected_audio_loss, rtol=1e-5, atol=1e-8)
    assert set(metrics) == {"loss/video", "loss/audio"}
    assert metrics["loss/video"] == pytest.approx(expected_video_loss.item())
    assert metrics["loss/audio"] == pytest.approx(expected_audio_loss.item())
    assert torch.equal(rng_after, rng_before)
    assert "h3_video_best_of_k/candidate_loss_mean" not in metrics


class _ToyH3BestOfKTrainer(MiniMaxH3NetworkTrainer):
    def __init__(self, device="cpu"):
        super().__init__()
        self.video_parameter = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.audio_parameter = torch.nn.Parameter(torch.tensor(0.0, device=device))
        self.best_of_k_records = []

    def _call_training_dit(
        self,
        args,
        accelerator,
        transformer,
        batch,
        latents,
        video_noise,
        noisy_video,
        state,
        network_dtype,
    ):
        del args, transformer, batch
        is_candidate_zero = torch.count_nonzero(video_noise).item() == 0
        video_error = 2.0 if is_candidate_zero else 1.0
        audio_error = 0.0 if is_candidate_zero else 10.0
        cpu_mask = torch.rand((), device="cpu")
        device_mask = (
            torch.rand((), device=accelerator.device) if accelerator.device.type == "cuda" else torch.rand((), device="cpu")
        )
        self.best_of_k_records.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "video_noise": video_noise.detach().clone(),
                "noisy_video": noisy_video.detach().clone(),
                "audio_latents": state.audio_latents.detach().clone(),
                "audio_noise": state.audio_noise.detach().clone(),
                "noisy_audio": state.noisy_audio.detach().clone(),
                "base_time": state.base_time.detach().clone(),
                "model_t_video": state.model_t_video.detach().clone(),
                "model_t_audio": state.model_t_audio.detach().clone(),
                "visual_conditions": tuple(value.detach().clone() for value in state.visual_conditions),
                "audio_conditions": tuple(value.detach().clone() for value in state.audio_conditions),
                "audio_loss_weight": state.audio_loss_weight.detach().clone(),
                "cpu_mask": cpu_mask.detach().clone(),
                "device_mask": device_mask.detach().cpu().clone(),
            }
        )
        return DiTOutput(
            pred=torch.ones_like(latents, dtype=network_dtype) * video_error + self.video_parameter.to(network_dtype),
            target=torch.zeros_like(latents, dtype=network_dtype),
            extra={
                "audio_pred": torch.ones_like(state.audio_latents, dtype=network_dtype) * audio_error
                + self.audio_parameter.to(network_dtype),
                "audio_target": torch.zeros_like(state.audio_latents, dtype=network_dtype),
                "audio_loss_weight": state.audio_loss_weight,
            },
        )


def _run_h3_best_of_k(monkeypatch, trainer=None, device="cpu"):
    trainer = trainer or _ToyH3BestOfKTrainer(device)
    trainer._best_of_k_count = 2
    trainer._best_of_k_enabled = True
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_video_best_of_k=2,
    )
    batch = _training_batch()
    batch["latents_audio"] = batch["latents_audio"].to(device)
    batch["timesteps"] = [0.25]
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4, device=device)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8, device=device)
    latents = torch.zeros(1, 24, 2, 4, 4, device=device)
    candidate_zero = torch.zeros_like(latents)
    candidate_one = torch.ones_like(latents)
    monkeypatch.setattr(h3_module, "draw_candidate_noise", lambda reference, generator: candidate_one)
    loss, metrics = trainer.process_batch_best_of_k(
        args,
        _Accelerator(device),
        None,
        None,
        batch,
        latents,
        candidate_zero,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    return trainer, loss, metrics


def test_h3_best_of_k_rejects_zero_iteration_internal_state_before_final_forward():
    trainer = _ToyH3BestOfKTrainer()
    trainer._best_of_k_count = 0
    trainer._best_of_k_enabled = True
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_video_best_of_k=2,
    )
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    latents = torch.zeros(1, 24, 2, 4, 4)

    with pytest.raises(RuntimeError, match="candidate loop ran zero iterations"):
        trainer.process_batch_best_of_k(
            args,
            _Accelerator(),
            None,
            None,
            batch,
            latents,
            torch.zeros_like(latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )

    assert trainer.best_of_k_records == []


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_h3_best_of_k_selects_video_only_replays_rng_and_keeps_audio_gradient(monkeypatch, device):
    real_randn_like = torch.randn_like
    audio_noise_draws = 0
    captured = {}
    real_create = h3_module.create_candidate_generator

    def count_audio_noise(reference, *args, **kwargs):
        nonlocal audio_noise_draws
        if tuple(reference.shape) == (1, 32, 2, 8):
            audio_noise_draws += 1
        return real_randn_like(reference, *args, **kwargs)

    def capture_after_generator(reference):
        generator = real_create(reference)
        captured["cpu"] = torch.random.get_rng_state().clone()
        if reference.device.type == "cuda":
            captured["device"] = torch.cuda.get_rng_state(reference.device).clone()
        return generator

    monkeypatch.setattr(torch, "randn_like", count_audio_noise)
    monkeypatch.setattr(h3_module, "create_candidate_generator", capture_after_generator)
    torch.manual_seed(321)
    if device == "cuda":
        torch.cuda.manual_seed_all(321)
    trainer, loss, metrics = _run_h3_best_of_k(monkeypatch, device=device)
    post_cpu_state = torch.random.get_rng_state().clone()
    post_device_state = torch.cuda.get_rng_state(torch.device(device)).clone() if device == "cuda" else None
    records = trainer.best_of_k_records

    assert [record["grad_enabled"] for record in records] == [False, False, True]
    assert torch.count_nonzero(records[-1]["video_noise"]).item() > 0
    for key in (
        "audio_latents",
        "audio_noise",
        "noisy_audio",
        "base_time",
        "model_t_video",
        "model_t_audio",
        "audio_loss_weight",
    ):
        assert all(torch.equal(record[key], records[0][key]) for record in records[1:])
    for key in ("visual_conditions", "audio_conditions"):
        assert all(all(torch.equal(left, right) for left, right in zip(record[key], records[0][key])) for record in records[1:])
    assert not torch.equal(records[0]["noisy_video"], records[1]["noisy_video"])
    assert all(torch.equal(record["cpu_mask"], records[0]["cpu_mask"]) for record in records[1:])
    assert all(torch.equal(record["device_mask"], records[0]["device_mask"]) for record in records[1:])
    assert audio_noise_draws == 1
    assert loss.item() == pytest.approx(101.0)
    assert set(metrics) == {
        "loss/video",
        "loss/audio",
        "h3_video_best_of_k/candidate_loss_mean",
        "h3_video_best_of_k/selection_gain",
    }
    assert metrics["loss/video"].item() == pytest.approx(1.0)
    assert metrics["loss/audio"].item() == pytest.approx(100.0)
    assert metrics["h3_video_best_of_k/candidate_loss_mean"] == pytest.approx(2.5)
    assert metrics["h3_video_best_of_k/selection_gain"] == pytest.approx(3.0)
    assert not any(key.startswith("xm/") for key in metrics)

    torch.random.set_rng_state(captured["cpu"])
    if device == "cuda":
        torch.cuda.set_rng_state(captured["device"], torch.device(device))
    torch.rand((), device="cpu")
    torch.rand((), device=torch.device(device) if device == "cuda" else "cpu")
    assert torch.equal(torch.random.get_rng_state(), post_cpu_state)
    if device == "cuda":
        assert torch.equal(torch.cuda.get_rng_state(torch.device(device)), post_device_state)

    loss.backward()
    assert trainer.video_parameter.grad is not None
    assert trainer.audio_parameter.grad is not None
    assert trainer.video_parameter.grad.item() == pytest.approx(2.0)
    assert trainer.audio_parameter.grad.item() == pytest.approx(20.0)


def test_h3_best_of_k_rejects_nonfinite_video_candidate(monkeypatch):
    class _NaNH3Trainer(_ToyH3BestOfKTrainer):
        def __init__(self):
            super().__init__()
            self.component_calls = 0

        def _compute_per_sample_component_losses(self, output, network_dtype):
            video, audio = super()._compute_per_sample_component_losses(output, network_dtype)
            if self.component_calls == 1:
                video = torch.full_like(video, torch.nan)
            self.component_calls += 1
            return video, audio

    with pytest.raises(ValueError, match=r"MiniMax-H3.*candidate 1.*sample indices \[0\]"):
        _run_h3_best_of_k(monkeypatch, _NaNH3Trainer())


class _ProductionPathAccelerator:
    def __init__(self, device):
        self.device = torch.device(device)

    def autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()


class _TinyJointH3Transformer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.video_projection = torch.nn.Linear(1, 1, bias=False, device=device, dtype=torch.float32)
        self.audio_projection = torch.nn.Linear(1, 1, bias=False, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.video_projection.weight.fill_(1.0)
            self.audio_projection.weight.fill_(1.0)
        self.records = []

    def forward(self, **kwargs):
        cpu_mask = torch.rand((), device="cpu")
        device = kwargs["video_latents"].device
        device_mask = torch.rand((), device=device) if device.type == "cuda" else torch.rand((), device="cpu")
        video_prediction = self.video_projection(kwargs["video_latents"].reshape(-1, 1)).reshape_as(kwargs["video_latents"])
        audio_prediction = self.audio_projection(kwargs["audio_latents"].reshape(-1, 1)).reshape_as(kwargs["audio_latents"])
        self.records.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "autocast_enabled": torch.is_autocast_enabled(device.type),
                "video_input": kwargs["video_latents"].detach().clone(),
                "video_prediction": video_prediction.detach().clone(),
                "audio_input": kwargs["audio_latents"].detach().clone(),
                "audio_prediction": audio_prediction.detach().clone(),
                "model_t_video": kwargs["model_t_video"].detach().clone(),
                "model_t_audio": kwargs["model_t_audio"].detach().clone(),
                "visual_conditions": tuple(value.detach().clone() for value in kwargs["visual_condition_latents"]),
                "audio_conditions": tuple(value.detach().clone() for value in kwargs["audio_condition_latents"]),
                "cpu_mask": cpu_mask.detach().clone(),
                "device_mask": device_mask.detach().cpu().clone(),
            }
        )
        return SimpleNamespace(video=video_prediction, audio=audio_prediction)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_h3_best_of_k_real_production_path_dispatches_pairs_targets_and_keeps_joint_gradients(monkeypatch, device):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_video_best_of_k=2,
        xm_best_of_k=1,
    )
    trainer._validate_and_init_best_of_k(args)
    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (2, True)

    transformer = _TinyJointH3Transformer(device)
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    video_latents = torch.full((1, 24, 2, 4, 4), 2.0, device=device)
    candidate_zero = torch.zeros_like(video_latents)
    candidate_one = torch.ones_like(video_latents)
    fixed_audio_noise = torch.full((1, 32, 2, 8), 0.5, device=device)
    audio_noise_draws = 0
    real_randn_like = torch.randn_like

    def draw_fixed_audio_noise(reference, *positional, **kwargs):
        nonlocal audio_noise_draws
        if tuple(reference.shape) == tuple(fixed_audio_noise.shape):
            audio_noise_draws += 1
            return fixed_audio_noise.to(dtype=reference.dtype, device=reference.device)
        return real_randn_like(reference, *positional, **kwargs)

    monkeypatch.setattr(torch, "randn_like", draw_fixed_audio_noise)
    monkeypatch.setattr(h3_module, "draw_candidate_noise", lambda reference, generator: candidate_one)

    loss, metrics = trainer._process_batch_for_training(
        args,
        _ProductionPathAccelerator(device),
        transformer,
        None,
        batch,
        video_latents,
        candidate_zero,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    records = transformer.records

    assert type(trainer) is MiniMaxH3NetworkTrainer
    assert [record["grad_enabled"] for record in records] == [False, False, True]
    assert [record["autocast_enabled"] for record in records] == [device == "cuda"] * 3
    assert transformer.video_projection.weight.dtype == torch.float32
    assert transformer.audio_projection.weight.dtype == torch.float32
    if device == "cuda":
        assert all(record["video_prediction"].dtype == torch.float16 for record in records)
        assert all(record["audio_prediction"].dtype == torch.float16 for record in records)

    expected_noises = (candidate_zero, candidate_one)
    candidate_video_losses = []
    for record, expected_noise in zip(records[:2], expected_noises):
        sigma_video = 1.0 - record["model_t_video"]
        recovered_noise = (record["video_input"] - (1.0 - sigma_video) * video_latents) / sigma_video
        torch.testing.assert_close(recovered_noise, expected_noise)
        expected_target = video_latents - expected_noise
        candidate_video_losses.append(torch.nn.functional.mse_loss(record["video_prediction"].float(), expected_target.float()))

    assert candidate_video_losses[1] < candidate_video_losses[0]
    torch.testing.assert_close(records[-1]["video_input"], records[1]["video_input"])
    assert metrics["h3_video_best_of_k/candidate_loss_mean"] == pytest.approx(
        torch.stack(candidate_video_losses).mean().item(), rel=1e-5, abs=1e-6
    )
    assert metrics["h3_video_best_of_k/selection_gain"] == pytest.approx(
        (candidate_video_losses[0] - candidate_video_losses[1]).item(), rel=1e-5, abs=1e-6
    )

    assert audio_noise_draws == 1
    audio_latents = batch["latents_audio"].to(device)
    for record in records:
        sigma_audio = 1.0 - record["model_t_audio"]
        recovered_audio_noise = (record["audio_input"] - (1.0 - sigma_audio) * audio_latents) / sigma_audio
        torch.testing.assert_close(recovered_audio_noise, fixed_audio_noise)
    for key in ("audio_input", "model_t_video", "model_t_audio", "visual_conditions", "audio_conditions"):
        first = records[0][key]
        for record in records[1:]:
            if isinstance(first, tuple):
                assert all(torch.equal(left, right) for left, right in zip(record[key], first))
            else:
                assert torch.equal(record[key], first)
    assert all(torch.equal(record["cpu_mask"], records[0]["cpu_mask"]) for record in records[1:])
    assert all(torch.equal(record["device_mask"], records[0]["device_mask"]) for record in records[1:])

    final_video_target = video_latents - candidate_one
    final_audio_target = audio_latents - fixed_audio_noise
    expected_video_loss = torch.nn.functional.mse_loss(records[-1]["video_prediction"].float(), final_video_target.float())
    expected_audio_loss = torch.nn.functional.mse_loss(records[-1]["audio_prediction"].float(), final_audio_target.float())
    assert metrics["loss/video"].item() == pytest.approx(expected_video_loss.item(), rel=1e-5, abs=1e-6)
    assert metrics["loss/audio"].item() == pytest.approx(expected_audio_loss.item(), rel=1e-5, abs=1e-6)
    assert loss.item() == pytest.approx((expected_video_loss + expected_audio_loss).item(), rel=1e-5, abs=1e-6)

    loss.backward()
    for parameter in (transformer.video_projection.weight, transformer.audio_projection.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_process_batch_preserves_the_released_fp32_audio_cache_dtype(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch(batch_size=1)
    batch["latents_audio"] = batch["latents_audio"].to(torch.float32)
    video_latents = torch.zeros(1, 24, 2, 4, 4, dtype=torch.float16)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        None,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    assert transformer.calls[0]["video_latents"].dtype == torch.float16
    assert transformer.calls[0]["audio_latents"].dtype == torch.float32


def test_process_batch_uses_zero_weight_for_silence_placeholder_items(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer(audio_prediction=float("nan"))
    batch = _training_batch()
    batch["audio_present"] = torch.tensor([0.0], dtype=torch.float32)
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        None,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    assert torch.isfinite(loss)
    assert metrics["loss/audio"] == 0.0


def test_process_batch_video_only_disables_audio_loss_even_with_real_audio(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(video_only=True)
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer(audio_prediction=float("nan"))
    batch = _training_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        None,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    # the transformer still sees the (noised) real audio latents as attention context
    assert torch.isfinite(loss)
    assert metrics["loss/audio"] == 0.0
    assert torch.count_nonzero(transformer.calls[0]["audio_latents"]) > 0


def test_condition_noise_uses_a_fresh_global_draw_per_tensor_and_step(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(task="ref2va", h3_visual_cond_clean=0.5, h3_audio_cond_clean=0.5)
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch()
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    condition_draws = iter((1.0, 2.0, 3.0, 4.0))
    condition_shapes = []

    def fake_condition_noise(shape, **kwargs):
        condition_shapes.append(tuple(shape))
        return torch.full(shape, next(condition_draws), dtype=kwargs["dtype"], device=kwargs["device"])

    monkeypatch.setattr(torch, "randn", fake_condition_noise)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    for step in range(2):
        trainer.process_batch(
            args,
            _Accelerator(),
            transformer,
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            step,
        )

    first_call, second_call = transformer.calls
    first_visual = first_call["visual_condition_latents"][0]
    first_audio = first_call["audio_condition_latents"][0]
    second_visual = second_call["visual_condition_latents"][0]
    second_audio = second_call["audio_condition_latents"][0]
    assert condition_shapes == [(1, 24, 1, 4, 4), (1, 32, 2, 8)] * 2
    assert torch.equal(first_visual, torch.full_like(first_visual, 0.5))
    assert torch.equal(first_audio, torch.full_like(first_audio, 1.0))
    assert torch.equal(second_visual, torch.full_like(second_visual, 1.5))
    assert torch.equal(second_audio, torch.full_like(second_audio, 2.0))


def test_runtime_rejects_batch_size_above_one():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    video_latents = torch.zeros(2, 24, 2, 4, 4)
    batch = _training_batch()
    with pytest.raises(ValueError, match=r"R1 requires batch_size=1"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )


def test_runtime_rejects_more_than_one_timestep_for_the_single_item():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    batch = _training_batch()
    batch["timesteps"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="exactly one timestep"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )


@pytest.mark.parametrize(
    "present",
    [
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32),
        torch.tensor([0.0, 1.0], dtype=torch.float32),
    ],
)
def test_runtime_rejects_invalid_audio_present_before_transformer(present: torch.Tensor):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    batch = _training_batch()
    batch["audio_present"] = present
    transformer = _RecordingTransformer()
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    with pytest.raises(ValueError, match="audio_present"):
        trainer.process_batch(
            args,
            _Accelerator(),
            transformer,
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )

    assert transformer.calls == []


def test_runtime_rejects_a_batch_from_a_different_authoritative_task():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(task="ref2va")
    trainer.handle_model_specific_args(args)
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    with pytest.raises(ValueError, match=r"--task ref2va.*T2VA"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            _training_batch(batch_size=1),
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )


def test_t2va_without_conditions_does_not_draw_condition_noise(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))
    monkeypatch.setattr(
        torch,
        "randn",
        lambda *args, **kwargs: pytest.fail("T2VA without conditions must not draw condition noise"),
    )
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    trainer.process_batch(
        args,
        _Accelerator(),
        _RecordingTransformer(),
        None,
        _training_batch(batch_size=1),
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )


def test_h3_component_vectors_and_combiner_define_scalar_loss():
    trainer = MiniMaxH3NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([[1.0, 5.0]]),
        target=torch.tensor([[3.0, 1.0]]),
        extra={
            "audio_pred": torch.tensor([[0.0, 2.0]]),
            "audio_target": torch.tensor([[2.0, 2.0]]),
            "audio_loss_weight": torch.tensor([0.25], dtype=torch.float32),
        },
    )

    video, audio = trainer._compute_per_sample_component_losses(output, torch.float32)
    total = trainer._combine_per_sample_losses(video, audio, output.extra["audio_loss_weight"])
    canonical = trainer.compute_per_sample_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        None,
        torch.float32,
        torch.float32,
        0,
    )
    loss, metrics = trainer.compute_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        None,
        torch.float32,
        torch.float32,
        0,
    )

    assert video.shape == audio.shape == total.shape == (1,)
    torch.testing.assert_close(video, torch.tensor([10.0]))
    torch.testing.assert_close(audio, torch.tensor([2.0]))
    torch.testing.assert_close(total, torch.tensor([10.5]))
    torch.testing.assert_close(canonical, total)
    assert torch.allclose(loss, total.mean(), rtol=1e-5, atol=1e-8)
    assert set(metrics) == {"loss/video", "loss/audio"}


@pytest.mark.parametrize(
    ("pred", "target", "message"),
    [
        (torch.zeros(1, 2), torch.zeros(1, 1), "shapes must match"),
        (torch.zeros(0, 2), torch.zeros(0, 2), "non-empty batch"),
        (torch.zeros(1, 0), torch.zeros(1, 0), "at least one element"),
    ],
)
def test_h3_component_losses_reject_malformed_video_objectives(pred, target, message):
    output = DiTOutput(
        pred=pred,
        target=target,
        extra={"audio_loss_weight": torch.tensor([0.0], dtype=torch.float32)},
    )

    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer()._compute_per_sample_component_losses(output, torch.float32)


def test_h3_component_losses_reject_video_device_mismatch():
    output = DiTOutput(
        pred=torch.zeros(1, 2),
        target=torch.empty(1, 2, device="meta"),
        extra={"audio_loss_weight": torch.tensor([0.0], dtype=torch.float32)},
    )

    with pytest.raises(ValueError, match="devices must match"):
        MiniMaxH3NetworkTrainer()._compute_per_sample_component_losses(output, torch.float32)


@pytest.mark.parametrize(
    ("audio_pred", "audio_target", "message"),
    [
        (torch.zeros(1, 2), torch.zeros(1, 1), "shapes must match"),
        (torch.zeros(2, 1), torch.zeros(2, 1), "same leading batch axis"),
        (torch.zeros(1, 0), torch.zeros(1, 0), "at least one element"),
    ],
)
def test_h3_component_losses_reject_malformed_audio_objectives(audio_pred, audio_target, message):
    output = DiTOutput(
        pred=torch.zeros(1, 2),
        target=torch.zeros(1, 2),
        extra={
            "audio_pred": audio_pred,
            "audio_target": audio_target,
            "audio_loss_weight": torch.tensor([1.0], dtype=torch.float32),
        },
    )

    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer()._compute_per_sample_component_losses(output, torch.float32)


def test_h3_component_losses_reject_audio_device_mismatch():
    output = DiTOutput(
        pred=torch.zeros(1, 2),
        target=torch.zeros(1, 2),
        extra={
            "audio_pred": torch.zeros(1, 2),
            "audio_target": torch.empty(1, 2, device="meta"),
            "audio_loss_weight": torch.tensor([1.0], dtype=torch.float32),
        },
    )

    with pytest.raises(ValueError, match="devices must match"):
        MiniMaxH3NetworkTrainer()._compute_per_sample_component_losses(output, torch.float32)


@pytest.mark.parametrize(
    "weight",
    [
        None,
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor([1.0], dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float32),
        torch.tensor([float("inf")], dtype=torch.float32),
        torch.tensor([-1.0], dtype=torch.float32),
    ],
)
def test_h3_component_losses_require_exact_audio_weight_contract(weight):
    output = DiTOutput(pred=torch.zeros(1, 1), target=torch.zeros(1, 1), extra={"audio_loss_weight": weight})

    with pytest.raises(ValueError, match=r"finite nonnegative float32 tensor with shape \[1\]"):
        MiniMaxH3NetworkTrainer()._compute_per_sample_component_losses(output, torch.float32)


def test_compute_loss_is_video_mean_plus_weighted_audio_mean_mse():
    trainer = MiniMaxH3NetworkTrainer()

    def output():
        return DiTOutput(
            pred=torch.tensor([[1.0, 5.0]]),
            target=torch.tensor([[3.0, 1.0]]),
            extra={
                "audio_pred": torch.tensor([[0.0, 2.0]]),
                "audio_target": torch.tensor([[2.0, 2.0]]),
                "audio_loss_weight": torch.tensor([1.0], dtype=torch.float32),
            },
        )

    loss, metrics = trainer.compute_loss(_trainer_args(), output(), torch.tensor(0.25), object(), torch.bfloat16, torch.float32, 7)

    assert loss.item() == pytest.approx(12.0)
    assert metrics == {"loss/video": pytest.approx(10.0), "loss/audio": pytest.approx(2.0)}

    weighted = output()
    weighted.extra["audio_loss_weight"] = torch.tensor([0.5], dtype=torch.float32)
    loss, metrics = trainer.compute_loss(_trainer_args(), weighted, torch.tensor(0.25), object(), torch.bfloat16, torch.float32, 7)

    assert loss.item() == pytest.approx(11.0)
    assert metrics["loss/audio"] == pytest.approx(2.0)


def test_compute_loss_requires_the_audio_weight_tensor():
    trainer = MiniMaxH3NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([[1.0]]),
        target=torch.tensor([[3.0]]),
        extra={"audio_pred": torch.tensor([[0.0]]), "audio_target": torch.tensor([[2.0]])},
    )

    with pytest.raises(ValueError, match="audio loss weight"):
        trainer.compute_loss(_trainer_args(), output, torch.tensor(0.25), object(), torch.bfloat16, torch.float32, 7)


def test_compute_loss_skips_audio_expression_and_gradient_for_zero_weight():
    trainer = MiniMaxH3NetworkTrainer()
    video_pred = torch.tensor([[1.0, 5.0]], requires_grad=True)
    audio_pred = torch.tensor([[float("nan"), 2.0]], requires_grad=True)
    output = DiTOutput(
        pred=video_pred,
        target=torch.tensor([[3.0, 1.0]]),
        extra={
            "audio_pred": audio_pred,
            "audio_target": torch.tensor([[2.0, 2.0]]),
            "audio_loss_weight": torch.tensor([0.0], dtype=torch.float32),
        },
    )

    loss, metrics = trainer.compute_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        object(),
        torch.bfloat16,
        torch.float32,
        7,
    )
    loss.backward()

    assert loss.item() == pytest.approx(10.0)
    assert metrics["loss/video"] == pytest.approx(10.0)
    assert metrics["loss/audio"] == pytest.approx(0.0)
    assert video_pred.grad is not None
    assert audio_pred.grad is None


def test_compute_loss_does_not_touch_missing_audio_tensors_at_zero_weight():
    output = DiTOutput(
        pred=torch.tensor([[1.0, 5.0]]),
        target=torch.tensor([[3.0, 1.0]]),
        extra={"audio_loss_weight": torch.tensor([0.0], dtype=torch.float32)},
    )

    loss, metrics = MiniMaxH3NetworkTrainer().compute_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        object(),
        torch.bfloat16,
        torch.float32,
        7,
    )

    assert loss.item() == pytest.approx(10.0)
    assert metrics["loss/audio"] == pytest.approx(0.0)


def test_process_batch_rejects_caches_without_audio_present():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    batch = _training_batch()
    del batch["audio_present"]
    video_latents = torch.zeros(1, 24, 2, 4, 4)

    with pytest.raises(ValueError, match="audio_present.*re-run latent caching"):
        trainer.process_batch(
            args,
            _Accelerator(),
            _RecordingTransformer(),
            None,
            batch,
            video_latents,
            torch.zeros_like(video_latents),
            None,
            torch.bfloat16,
            torch.float32,
            None,
            0,
        )


def test_h3_training_metadata_records_task_scheduler_and_target_policy():
    trainer = MiniMaxH3NetworkTrainer()
    trainer._audio_items_seen = 4
    trainer._audio_supervised_seen = 1

    metadata = trainer.extra_metadata(_trainer_args(task="fl2va"))

    assert metadata == {
        "ss_minimax_h3_task": "fl2va",
        "ss_minimax_h3_base_family": "fl2va",
        "ss_minimax_h3_shift_video": 12.0,
        "ss_minimax_h3_shift_audio": 3.0,
        "ss_minimax_h3_visual_cond_clean": 0.999,
        "ss_minimax_h3_audio_cond_clean": 1.0,
        "ss_minimax_h3_loss_policy": "video_mean_plus_weighted_audio_mean",
        "ss_minimax_h3_audio_supervision": "presence_gated_training_weight",
        "ss_minimax_h3_supervised_audio_fraction": 0.25,
        "ss_minimax_h3_audio_loss_weight": 1.0,
        "ss_minimax_h3_video_only": False,
        "ss_minimax_h3_target_modules": "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2",
        "ss_minimax_h3_convrot_int8": False,
        "ss_minimax_h3_latent_cache_version": "2",
        "ss_minimax_h3_text_cache_version": "1",
    }


def test_t2va_metadata_distinguishes_task_from_the_fl2va_base_family():
    metadata = MiniMaxH3NetworkTrainer().extra_metadata(_trainer_args(task="t2va"))

    assert metadata["ss_minimax_h3_task"] == "t2va"
    assert metadata["ss_minimax_h3_base_family"] == "fl2va"


def test_h3_metadata_omits_the_audio_fraction_until_a_batch_has_been_observed():
    metadata = MiniMaxH3NetworkTrainer().extra_metadata(_trainer_args())

    assert "ss_minimax_h3_supervised_audio_fraction" not in metadata


def _tiny_model(num_layers: int = 2):
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
    model = MiniMaxH3Model(config, dtype=torch.float32)
    with torch.no_grad():
        model.rope.inv_freq.fill_(1.0)
    return model


def test_default_h3_lora_policy_targets_only_four_projections_in_main_blocks():
    model = _tiny_model(num_layers=2)

    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)

    assert {module.lora_name for module in network.unet_loras} == {
        "lora_unet_blocks_0_attn_qkv_proj",
        "lora_unet_blocks_0_attn_out_proj",
        "lora_unet_blocks_0_mlp_fc1",
        "lora_unet_blocks_0_mlp_fc2",
        "lora_unet_blocks_1_attn_qkv_proj",
        "lora_unet_blocks_1_attn_out_proj",
        "lora_unet_blocks_1_mlp_fc1",
        "lora_unet_blocks_1_mlp_fc2",
    }


def test_h3_lora_gets_gradients_with_checkpointing_and_block_swap(monkeypatch):
    class _Offloader:
        def __init__(self, blocks, device):
            self.blocks = blocks
            self.device = device

        def prepare_block_devices_before_forward(self, blocks):
            for block in blocks:
                block.to(self.device)

        def wait_for_block(self, index):
            return None

        def submit_move_blocks_forward(self, blocks, index):
            return None

        def set_forward_only(self, value):
            return None

    monkeypatch.setattr(
        "musubi_tuner.minimax_h3.model.create_offloader",
        lambda block_type, blocks, num_blocks, blocks_to_swap, config: _Offloader(blocks, config.device),
    )
    model = _tiny_model(num_layers=3)
    model.requires_grad_(False)
    model.enable_gradient_checkpointing()
    model.train()
    model.enable_block_swap(1, BlockSwapConfig(device=torch.device("cpu"), supports_backward=True))
    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    model.prepare_block_swap_before_forward()
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
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

    gradients = [parameter.grad for parameter in network.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)


def test_h3_lora_gets_gradients_over_frozen_int8_convrot_base_with_checkpointing():
    model = _tiny_model(num_layers=1)
    target_paths = (
        "blocks.0.attn.qkv_proj",
        "blocks.0.attn.out_proj",
        "blocks.0.mlp.fc1",
        "blocks.0.mlp.fc2",
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
    model.enable_gradient_checkpointing()
    model.train()
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
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

    gradients = [parameter.grad for parameter in network.parameters()]
    assert any(gradient is not None and torch.count_nonzero(gradient) for gradient in gradients)
    assert all(model.get_submodule(path).weight.grad is None for path in target_paths)
