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
from musubi_tuner.minimax_h3.packing import FRAME_RESCALE, H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import apply_convrot_int8_monkey_patch
from musubi_tuner.minimax_h3_train_network import (
    H3SamplingResources,
    MiniMaxH3NetworkTrainer,
    _apply_timestep_focus,
    _decomposed_flow_loss,
    _normalize_h3_sample_parameter,
    _prediction_geometry_log,
    minimax_h3_setup_parser,
)
from musubi_tuner.training.sampling_prompts import line_to_prompt_dict
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


def _parser_defaults() -> dict[str, object]:
    parser = minimax_h3_setup_parser(setup_parser_common())
    return {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help" and action.default is not argparse.SUPPRESS
    }


# the trainer reads plain argparse attributes, so the fake args start from the real parser defaults:
# a hand-written subset goes stale without a failing assertion whenever the shared training
# arguments grow, and only the values the tests actually choose are spelled out here
_TRAINER_DEFAULTS = _parser_defaults() | {
    "task": "t2va",  # required on the real command line
    "blocks_to_swap": 0,  # the parser leaves this None to mean "disabled"
}


def _trainer_args(**overrides):
    return SimpleNamespace(**(_TRAINER_DEFAULTS | overrides))


def _training_batch(batch_size: int = 1, *, text_length: int = 3):
    return {
        "latents_audio": torch.full((batch_size, 32, 2, 8), 4.0),
        "audio_present": torch.ones(batch_size, dtype=torch.float32),
        "mmh3_hidden_states": [torch.full((text_length, 12), float(index)) for index in range(batch_size)],
        "mmh3_token_tags": [torch.tensor([1, 0, 1][:text_length], dtype=torch.int64) for _ in range(batch_size)],
        "timesteps": None,
    }


def test_sample_prompt_line_parses_inline_refs_and_reference_jsonl():
    line = "a cat sings --ref refs/cat.png --ref refs/dance.mp4;audio=refs/song.wav --w 640 --h 384"

    prompt_dict = line_to_prompt_dict(line)

    assert prompt_dict["prompt"] == "a cat sings"
    assert prompt_dict["ref"] == ["refs/cat.png", "refs/dance.mp4;audio=refs/song.wav"]
    assert prompt_dict["width"] == 640
    assert prompt_dict["height"] == 384

    assert line_to_prompt_dict("a cat sings --rj refs/all.jsonl")["reference_jsonl"] == "refs/all.jsonl"


def test_h3_ref2va_sample_normalization_resolves_inline_refs_from_the_prompt_file_directory(tmp_path):
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.touch()
    face = tmp_path / "refs" / "face.png"
    face.parent.mkdir()
    face.touch()
    args = _trainer_args(task="ref2va", sample_prompts=str(prompt_file))

    sample = _normalize_h3_sample_parameter(args, {"prompt": "a cat sings", "ref": ["refs/face.png"]})

    assert sample["ref"] == ["refs/face.png"]
    assert Path(sample["ref_base_directory"]) == tmp_path.resolve()
    assert sample["reference_jsonl"] is None

    with pytest.raises(ValueError, match="does not exist"):
        _normalize_h3_sample_parameter(args, {"prompt": "a cat sings", "ref": ["refs/missing.png"]})
    with pytest.raises(ValueError, match="cannot combine"):
        _normalize_h3_sample_parameter(
            args, {"prompt": "a cat sings", "ref": ["refs/face.png"], "reference_jsonl": "refs/all.jsonl"}
        )
    with pytest.raises(ValueError, match="requires a prompt"):
        _normalize_h3_sample_parameter(args, {"ref": ["refs/face.png"]})
    with pytest.raises(ValueError, match="does not apply to --ref"):
        _normalize_h3_sample_parameter(args, {"prompt": "a cat sings", "ref": ["refs/face.png"], "reference_index": 1})
    with pytest.raises(ValueError, match="non-empty strings"):
        _normalize_h3_sample_parameter(args, {"prompt": "a cat sings", "ref": " "})

    for task in ("t2va", "fl2va"):
        with pytest.raises(ValueError, match="does not accept"):
            _normalize_h3_sample_parameter(
                _trainer_args(task=task, sample_prompts=str(prompt_file)),
                {"prompt": "a cat sings", "ref": ["refs/face.png"]},
            )


def test_h3_ref2va_sample_normalization_resolves_relative_reference_jsonl_from_the_prompt_file(tmp_path):
    prompt_directory = tmp_path / "sub"
    prompt_directory.mkdir()
    prompt_file = prompt_directory / "prompts.txt"
    prompt_file.touch()
    jsonl = prompt_directory / "refs.jsonl"
    jsonl.touch()
    args = _trainer_args(task="ref2va", sample_prompts=str(prompt_file))

    sample = _normalize_h3_sample_parameter(args, {"prompt": "p", "reference_jsonl": "refs.jsonl"})

    assert Path(sample["reference_jsonl"]) == jsonl.resolve()

    with pytest.raises(ValueError, match="does not exist"):
        _normalize_h3_sample_parameter(args, {"prompt": "p", "reference_jsonl": "nowhere.jsonl"})


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


def test_h3_parser_exposes_one_best_of_k_count_and_multiframe_stream(tmp_path, monkeypatch):
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())
    defaults = parser.parse_args(["--task", "t2va"])
    configured = parser.parse_args(["--task", "t2va", "--h3_best_of_k", "3", "--h3_best_of_k_stream", "audio"])
    assert (defaults.h3_best_of_k, defaults.h3_best_of_k_stream) == (1, "video")
    assert (configured.h3_best_of_k, configured.h3_best_of_k_stream) == (3, "audio")

    config = tmp_path / "h3_best_of_k.toml"
    config.write_text('task = "t2va"\nh3_best_of_k = 4\nh3_best_of_k_stream = "audio"\n', encoding="utf-8")
    common_parser = minimax_h3_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["minimax_h3_train_network", "--config_file", str(config)])
    args = common_parser.parse_args()
    loaded = read_config_from_file(args, common_parser)
    assert (loaded.h3_best_of_k, loaded.h3_best_of_k_stream) == (4, "audio")


@pytest.mark.parametrize(
    ("option", "guidance"),
    [
        ("--h3_video_best_of_k", "--h3_best_of_k_stream video"),
        ("--h3_audio_best_of_k", "--h3_best_of_k_stream audio"),
        ("--h3_image_best_of_k", "one-frame batches automatically search video"),
    ],
)
def test_removed_h3_best_of_k_cli_names_fail_with_hidden_migration_guidance(capsys, option, guidance):
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    for removed_args in ([option], [option, "3"]):
        with pytest.raises(SystemExit):
            parser.parse_args(["--task", "t2va", *removed_args])

        error = capsys.readouterr().err
        assert option in error
        assert "--h3_best_of_k" in error
        assert guidance in error
    assert option not in parser.format_help()


@pytest.mark.parametrize(
    ("key", "guidance"),
    [
        ("h3_video_best_of_k", "--h3_best_of_k_stream video"),
        ("h3_audio_best_of_k", "--h3_best_of_k_stream audio"),
        ("h3_image_best_of_k", "one-frame batches automatically search video"),
    ],
)
def test_removed_h3_best_of_k_toml_names_fail_with_the_same_migration_guidance(tmp_path, monkeypatch, key, guidance):
    config = tmp_path / "removed_h3_best_of_k.toml"
    config.write_text(f'task = "t2va"\n{key} = 3\n', encoding="utf-8")
    parser = minimax_h3_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["minimax_h3_train_network", "--config_file", str(config)])
    args = read_config_from_file(parser.parse_args(), parser)

    with pytest.raises(ValueError) as error:
        MiniMaxH3NetworkTrainer()._validate_and_init_best_of_k(args)

    assert f"--{key}" in str(error.value)
    assert "--h3_best_of_k" in str(error.value)
    assert guidance in str(error.value)


def test_h3_best_of_k_validation_uses_the_canonical_option_and_rejects_forward_xm():
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=r"--h3_best_of_k.*integer.*at least 1"):
        trainer._validate_and_init_best_of_k(_trainer_args(h3_best_of_k=0, xm_best_of_k=1))
    with pytest.raises(ValueError, match=r"--xm_best_of_k.*valid integer 1.*--h3_best_of_k"):
        trainer._validate_and_init_best_of_k(_trainer_args(h3_best_of_k=1, xm_best_of_k=2))

    trainer._validate_and_init_best_of_k(_trainer_args(h3_best_of_k=2, h3_best_of_k_stream="audio", xm_best_of_k=1))
    assert trainer._best_of_k_count == 2
    assert trainer._best_of_k_enabled is True
    assert trainer._h3_best_of_k_config.count == 2
    assert trainer._h3_best_of_k_config.multi_frame_stream == "audio"


def test_h3_best_of_k_rejects_teacher_matching_before_candidate_execution():
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=r"h3_teacher_matching.*--h3_best_of_k"):
        trainer._validate_and_init_best_of_k(_trainer_args(h3_best_of_k=2, h3_teacher_matching=True))

    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)


@pytest.mark.parametrize(
    ("option", "toml_value"),
    [
        ("xm_best_of_k", "0"),
        ("xm_best_of_k", "1.0"),
        ("xm_best_of_k", "true"),
        ("xm_best_of_k", '"1"'),
        ("h3_best_of_k", "0"),
        ("h3_best_of_k", "1.0"),
        ("h3_best_of_k", "true"),
        ("h3_best_of_k", '"1"'),
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


@pytest.mark.parametrize("toml_value", ['"image"', "true", "3"])
def test_h3_best_of_k_validation_rejects_invalid_toml_stream_values(tmp_path, monkeypatch, toml_value):
    config = tmp_path / "invalid_h3_best_of_k_stream.toml"
    config.write_text(f'task = "t2va"\nh3_best_of_k = 2\nh3_best_of_k_stream = {toml_value}\n', encoding="utf-8")
    parser = minimax_h3_setup_parser(setup_parser_common())
    monkeypatch.setattr(sys, "argv", ["minimax_h3_train_network", "--config_file", str(config)])
    args = read_config_from_file(parser.parse_args(), parser)
    trainer = MiniMaxH3NetworkTrainer()

    with pytest.raises(ValueError, match=r"--h3_best_of_k_stream.*video.*audio"):
        trainer._validate_and_init_best_of_k(args)

    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)


def test_h3_best_of_k_rejects_audio_stream_when_video_only_and_restores_default_state():
    trainer = MiniMaxH3NetworkTrainer()
    trainer._validate_and_init_best_of_k(_trainer_args(h3_best_of_k=2, h3_best_of_k_stream="video"))

    with pytest.raises(ValueError, match=r"--video_only.*--h3_best_of_k_stream audio"):
        trainer._validate_and_init_best_of_k(_trainer_args(h3_best_of_k=2, h3_best_of_k_stream="audio", video_only=True))

    assert (trainer._best_of_k_count, trainer._best_of_k_enabled) == (1, False)
    assert trainer._h3_best_of_k_config.count == 1
    assert trainer._h3_best_of_k_config.multi_frame_stream == "video"


def test_h3_best_of_k_startup_log_names_the_stream_and_static_fallbacks(caplog):
    caplog.set_level("INFO")
    trainer = MiniMaxH3NetworkTrainer()
    trainer._validate_and_init_best_of_k(
        _trainer_args(
            h3_best_of_k=2,
            h3_best_of_k_stream="audio",
            audio_loss_weight=0.0,
            one_frame=True,
            xm_best_of_k=1,
        )
    )

    assert "MiniMax-H3 best-of-K enabled" in caplog.text
    assert "multi-frame stream=audio" in caplog.text
    assert "audio_loss_weight is 0" in caplog.text
    assert "one-frame batches override" in caplog.text
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
    assert not any(key.startswith("h3_best_of_k/") for key in metrics)


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
        state,
        inputs,
        network_dtype,
        network=None,
    ):
        del args, transformer, batch, network
        is_candidate_zero = torch.count_nonzero(inputs.video_noise).item() == 0
        video_error = 2.0 if is_candidate_zero else 1.0
        audio_error = 0.0 if is_candidate_zero else 10.0
        cpu_mask = torch.rand((), device="cpu")
        device_mask = (
            torch.rand((), device=accelerator.device) if accelerator.device.type == "cuda" else torch.rand((), device="cpu")
        )
        self.best_of_k_records.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "video_noise": inputs.video_noise.detach().clone(),
                "noisy_video": inputs.noisy_video.detach().clone(),
                "audio_latents": state.audio_latents.detach().clone(),
                "audio_noise": inputs.audio_noise.detach().clone(),
                "noisy_audio": inputs.noisy_audio.detach().clone(),
                "base_time": state.base_time.detach().clone(),
                "model_t_video": state.model_t_video.detach().clone(),
                "model_t_audio": state.model_t_audio.detach().clone(),
                "visual_conditions": tuple(value.detach().clone() for value in state.visual_conditions),
                "audio_conditions": tuple(value.detach().clone() for value in state.audio_conditions),
                "audio_loss_weight": state.audio_loss_weight.detach().clone(),
                "layout": state.runtime.layout,
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


def _run_h3_best_of_k(monkeypatch, trainer=None, device="cpu", transformer=None):
    trainer = trainer or _ToyH3BestOfKTrainer(device)
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_best_of_k=2,
        h3_best_of_k_stream="video",
    )
    trainer._validate_and_init_best_of_k(args)
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
        transformer,
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


def test_h3_best_of_k_uses_forward_only_for_candidates_and_training_for_winner(monkeypatch):
    class _BlockSwapTransformer:
        def __init__(self):
            self.mode = "training"
            self.events = []

        def switch_block_swap_for_inference(self):
            assert self.mode == "training"
            self.mode = "forward-only"
            self.events.append("inference")

        def switch_block_swap_for_training(self):
            assert self.mode == "forward-only"
            self.mode = "training"
            self.events.append("training")

        def observe_forward(self):
            expected_mode = "training" if torch.is_grad_enabled() else "forward-only"
            assert self.mode == expected_mode
            self.events.append(f"forward:{expected_mode}")

    class _BlockSwapTrainer(_ToyH3BestOfKTrainer):
        def _call_training_dit(self, *args, **kwargs):
            transformer = args[2]
            transformer.observe_forward()
            return super()._call_training_dit(*args, **kwargs)

    transformer = _BlockSwapTransformer()
    _, loss, _ = _run_h3_best_of_k(monkeypatch, _BlockSwapTrainer(), transformer=transformer)

    assert transformer.events == [
        "inference",
        "forward:forward-only",
        "forward:forward-only",
        "training",
        "forward:training",
    ]
    loss.backward()


def test_h3_best_of_k_rejects_zero_iteration_internal_state_before_final_forward():
    trainer = _ToyH3BestOfKTrainer()
    trainer._best_of_k_count = 0
    trainer._best_of_k_enabled = True
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_best_of_k=2,
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
        "h3_best_of_k/video/candidate_loss_mean",
        "h3_best_of_k/video/selection_gain",
    }
    assert metrics["loss/video"].item() == pytest.approx(1.0)
    assert metrics["loss/audio"].item() == pytest.approx(100.0)
    assert metrics["h3_best_of_k/video/candidate_loss_mean"] == pytest.approx(2.5)
    assert metrics["h3_best_of_k/video/selection_gain"] == pytest.approx(3.0)
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


def test_h3_best_of_k_audio_search_changes_only_audio_and_ignores_the_better_composite_candidate(monkeypatch):
    class _OpposedObjectivesTrainer(_ToyH3BestOfKTrainer):
        def _call_training_dit(
            self,
            args,
            accelerator,
            transformer,
            batch,
            latents,
            state,
            inputs,
            network_dtype,
            network=None,
        ):
            super()._call_training_dit(
                args,
                accelerator,
                transformer,
                batch,
                latents,
                state,
                inputs,
                network_dtype,
                network,
            )
            candidate_zero = torch.count_nonzero(inputs.audio_noise).item() == 0
            video_error, audio_error = (10.0, 1.0) if candidate_zero else (0.0, 2.0)
            return DiTOutput(
                pred=torch.ones_like(latents) * video_error + self.video_parameter,
                target=torch.zeros_like(latents),
                extra={
                    "audio_pred": torch.ones_like(state.audio_latents) * audio_error + self.audio_parameter,
                    "audio_target": torch.zeros_like(state.audio_latents),
                    "audio_loss_weight": state.audio_loss_weight,
                },
            )

    trainer = _OpposedObjectivesTrainer()
    args = _trainer_args(h3_best_of_k=2, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(args)
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    latents = torch.zeros(1, 24, 2, 4, 4)
    video_noise = torch.zeros_like(latents)
    monkeypatch.setattr(torch, "randn_like", lambda reference, *args, **kwargs: torch.zeros_like(reference))
    drawn_shapes = []

    def candidate_one(reference, generator):
        del generator
        drawn_shapes.append(tuple(reference.shape))
        return torch.ones_like(reference)

    monkeypatch.setattr(h3_module, "draw_candidate_noise", candidate_one)

    loss, metrics = trainer.process_batch_best_of_k(
        args,
        _Accelerator(),
        None,
        None,
        batch,
        latents,
        video_noise,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    records = trainer.best_of_k_records
    assert [record["grad_enabled"] for record in records] == [False, False, True]
    assert drawn_shapes == [(1, 32, 2, 8)]
    assert all(torch.equal(record["video_noise"], video_noise) for record in records)
    assert torch.count_nonzero(records[0]["audio_noise"]).item() == 0
    assert torch.count_nonzero(records[1]["audio_noise"]).item() > 0
    assert torch.equal(records[2]["audio_noise"], records[0]["audio_noise"])
    assert all(torch.equal(record["noisy_video"], records[0]["noisy_video"]) for record in records[1:])
    assert not torch.equal(records[0]["noisy_audio"], records[1]["noisy_audio"])
    assert torch.equal(records[2]["noisy_audio"], records[0]["noisy_audio"])
    assert loss.item() == pytest.approx(101.0)
    assert set(metrics) == {
        "loss/video",
        "loss/audio",
        "h3_best_of_k/audio/candidate_loss_mean",
        "h3_best_of_k/audio/selection_gain",
    }
    assert metrics["h3_best_of_k/audio/candidate_loss_mean"] == pytest.approx(2.5)
    assert metrics["h3_best_of_k/audio/selection_gain"] == pytest.approx(0.0)


def test_h3_best_of_k_one_frame_overrides_audio_stream_and_keeps_silence_state_fixed(monkeypatch):
    trainer = _ToyH3BestOfKTrainer()
    args = _trainer_args(one_frame=True, h3_best_of_k=2, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(args)
    batch = _one_frame_batch(target_index=24)
    batch["timesteps"] = [0.25]
    latents = torch.zeros(1, 24, 1, 4, 4)
    monkeypatch.setattr(torch, "randn_like", lambda reference, *args, **kwargs: torch.zeros_like(reference))
    monkeypatch.setattr(h3_module, "draw_candidate_noise", lambda reference, generator: torch.ones_like(reference))

    loss, metrics = trainer.process_batch_best_of_k(
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

    records = trainer.best_of_k_records
    assert [record["layout"].target_video.frames for record in records] == [1, 1, 1]
    assert all(record["layout"].time_overrides.target_time == FRAME_RESCALE * 24 for record in records)
    for key in ("audio_noise", "noisy_audio", "audio_loss_weight"):
        assert all(torch.equal(record[key], records[0][key]) for record in records[1:])
    assert torch.equal(records[-1]["video_noise"], records[1]["video_noise"])
    assert torch.equal(loss, torch.tensor(1.0))
    assert set(metrics) == {
        "loss/video",
        "loss/audio",
        "h3_best_of_k/image/candidate_loss_mean",
        "h3_best_of_k/image/selection_gain",
    }


def test_h3_best_of_k_runtime_kind_does_not_leak_between_image_and_audio_batches(monkeypatch):
    trainer = _ToyH3BestOfKTrainer()
    args = _trainer_args(one_frame=True, h3_best_of_k=2, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(args)
    monkeypatch.setattr(torch, "randn_like", lambda reference, *args, **kwargs: torch.zeros_like(reference))
    monkeypatch.setattr(h3_module, "draw_candidate_noise", lambda reference, generator: torch.ones_like(reference))

    metric_prefixes = []
    for batch, latents in (
        (_one_frame_batch(target_index=24), torch.zeros(1, 24, 1, 4, 4)),
        (_training_batch(), torch.zeros(1, 24, 2, 4, 4)),
        (_one_frame_batch(target_index=24), torch.zeros(1, 24, 1, 4, 4)),
    ):
        batch["timesteps"] = [0.25]
        _, metrics = trainer.process_batch_best_of_k(
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
        metric_prefixes.append({key.rsplit("/", 1)[0] for key in metrics if key.startswith("h3_best_of_k/")})

    assert metric_prefixes == [
        {"h3_best_of_k/image"},
        {"h3_best_of_k/audio"},
        {"h3_best_of_k/image"},
    ]


@pytest.mark.parametrize(("audio_present", "audio_loss_weight"), [(0.0, 1.0), (1.0, 0.0)])
def test_h3_audio_best_of_k_zero_effective_weight_matches_one_prepared_ordinary_step(monkeypatch, audio_present, audio_loss_weight):
    args = _trainer_args(h3_best_of_k=2, h3_best_of_k_stream="audio", audio_loss_weight=audio_loss_weight)
    batch = _training_batch()
    batch["audio_present"] = torch.tensor([audio_present], dtype=torch.float32)
    batch["timesteps"] = [0.25]
    latents = torch.zeros(1, 24, 2, 4, 4)
    video_noise = torch.zeros_like(latents)
    monkeypatch.setattr(torch, "randn_like", lambda reference, *args, **kwargs: torch.zeros_like(reference))
    monkeypatch.setattr(
        h3_module,
        "create_candidate_generator",
        lambda reference: pytest.fail("zero-weight audio fallback must not create a candidate generator"),
    )

    trainer = _ToyH3BestOfKTrainer()
    trainer._validate_and_init_best_of_k(args)
    initial_rng = torch.random.get_rng_state().clone()
    loss, metrics = trainer.process_batch_best_of_k(
        args,
        _Accelerator(),
        None,
        None,
        batch,
        latents,
        video_noise,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    fallback_rng = torch.random.get_rng_state().clone()
    torch.random.set_rng_state(initial_rng)
    ordinary = _ToyH3BestOfKTrainer()
    ordinary_loss, ordinary_metrics = ordinary.process_batch(
        args,
        _Accelerator(),
        None,
        None,
        batch,
        latents,
        video_noise,
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )
    ordinary_rng = torch.random.get_rng_state().clone()

    assert len(trainer.best_of_k_records) == 1
    assert trainer.best_of_k_records[0]["grad_enabled"] is True
    assert torch.equal(loss, ordinary_loss)
    assert metrics.keys() == ordinary_metrics.keys() == {"loss/video", "loss/audio"}
    assert all(torch.equal(metrics[key], ordinary_metrics[key]) for key in metrics)
    assert torch.equal(fallback_rng, ordinary_rng)
    assert not any(key.startswith("h3_best_of_k/") for key in metrics)
    assert (trainer._audio_items_seen, trainer._audio_supervised_seen) == (1, int(audio_present))


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


def test_h3_audio_best_of_k_ignores_a_nonfinite_unselected_video_component(monkeypatch):
    class _NonfiniteVideoTrainer(_ToyH3BestOfKTrainer):
        def __init__(self):
            super().__init__()
            self.component_calls = 0

        def _compute_per_sample_component_losses(self, output, network_dtype):
            video, audio = super()._compute_per_sample_component_losses(output, network_dtype)
            if self.component_calls == 1:
                video = torch.full_like(video, torch.nan)
            self.component_calls += 1
            return video, audio

    trainer = _NonfiniteVideoTrainer()
    args = _trainer_args(h3_best_of_k=2, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(args)
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "randn_like", lambda reference, *args, **kwargs: torch.zeros_like(reference))
    monkeypatch.setattr(h3_module, "draw_candidate_noise", lambda reference, generator: torch.ones_like(reference))

    loss, metrics = trainer.process_batch_best_of_k(
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

    assert torch.isfinite(loss)
    assert set(metrics) == {
        "loss/video",
        "loss/audio",
        "h3_best_of_k/audio/candidate_loss_mean",
        "h3_best_of_k/audio/selection_gain",
    }


def test_h3_audio_best_of_k_rejects_a_nonfinite_selected_audio_component(monkeypatch):
    class _NonfiniteAudioTrainer(_ToyH3BestOfKTrainer):
        def __init__(self):
            super().__init__()
            self.component_calls = 0

        def _compute_per_sample_component_losses(self, output, network_dtype):
            video, audio = super()._compute_per_sample_component_losses(output, network_dtype)
            if self.component_calls == 1:
                audio = torch.full_like(audio, torch.inf)
            self.component_calls += 1
            return video, audio

    trainer = _NonfiniteAudioTrainer()
    args = _trainer_args(h3_best_of_k=2, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(args)
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "randn_like", lambda reference, *args, **kwargs: torch.zeros_like(reference))
    monkeypatch.setattr(h3_module, "draw_candidate_noise", lambda reference, generator: torch.ones_like(reference))

    with pytest.raises(ValueError, match=r"MiniMax-H3 audio.*candidate 1.*sample indices \[0\]"):
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
        h3_best_of_k=2,
        h3_best_of_k_stream="video",
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
    assert metrics["h3_best_of_k/video/candidate_loss_mean"] == pytest.approx(
        torch.stack(candidate_video_losses).mean().item(), rel=1e-5, abs=1e-6
    )
    assert metrics["h3_best_of_k/video/selection_gain"] == pytest.approx(
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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_h3_audio_best_of_k_real_production_path_pairs_audio_targets_and_keeps_video_fixed(monkeypatch, device):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(
        task="ref2va",
        h3_visual_cond_clean=0.5,
        h3_audio_cond_clean=0.5,
        h3_best_of_k=2,
        h3_best_of_k_stream="audio",
    )
    trainer._validate_and_init_best_of_k(args)

    transformer = _TinyJointH3Transformer(device)
    batch = _training_batch()
    batch["timesteps"] = [0.25]
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    video_latents = torch.full((1, 24, 2, 4, 4), 2.0, device=device)
    video_noise = torch.zeros_like(video_latents)
    fixed_audio_noise = torch.full((1, 32, 2, 8), 0.5, device=device)
    candidate_audio_noise = torch.ones_like(fixed_audio_noise)
    audio_noise_draws = 0
    candidate_reference_shapes = []
    real_randn_like = torch.randn_like

    def draw_fixed_audio_noise(reference, *positional, **kwargs):
        nonlocal audio_noise_draws
        if tuple(reference.shape) == tuple(fixed_audio_noise.shape):
            audio_noise_draws += 1
            return fixed_audio_noise.to(dtype=reference.dtype, device=reference.device)
        return real_randn_like(reference, *positional, **kwargs)

    def draw_candidate(reference, generator):
        del generator
        candidate_reference_shapes.append(tuple(reference.shape))
        return candidate_audio_noise.to(dtype=reference.dtype, device=reference.device)

    monkeypatch.setattr(torch, "randn_like", draw_fixed_audio_noise)
    monkeypatch.setattr(h3_module, "draw_candidate_noise", draw_candidate)

    loss, metrics = trainer._process_batch_for_training(
        args,
        _ProductionPathAccelerator(device),
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
    records = transformer.records

    assert type(trainer) is MiniMaxH3NetworkTrainer
    assert [record["grad_enabled"] for record in records] == [False, False, True]
    assert [record["autocast_enabled"] for record in records] == [device == "cuda"] * 3
    assert candidate_reference_shapes == [(1, 32, 2, 8)]
    assert audio_noise_draws == 1
    for record in records[1:]:
        torch.testing.assert_close(record["video_input"], records[0]["video_input"], rtol=0.0, atol=0.0)

    audio_latents = batch["latents_audio"].to(device)
    expected_noises = (fixed_audio_noise, candidate_audio_noise)
    candidate_audio_losses = []
    for record, expected_noise in zip(records[:2], expected_noises):
        sigma_audio = 1.0 - record["model_t_audio"]
        recovered_noise = (record["audio_input"] - (1.0 - sigma_audio) * audio_latents) / sigma_audio
        torch.testing.assert_close(recovered_noise, expected_noise)
        expected_target = audio_latents - expected_noise
        candidate_audio_losses.append(torch.nn.functional.mse_loss(record["audio_prediction"].float(), expected_target.float()))

    assert candidate_audio_losses[1] < candidate_audio_losses[0]
    torch.testing.assert_close(records[-1]["audio_input"], records[1]["audio_input"], rtol=0.0, atol=0.0)
    assert metrics["h3_best_of_k/audio/candidate_loss_mean"] == pytest.approx(
        torch.stack(candidate_audio_losses).mean().item(), rel=1e-5, abs=1e-6
    )
    assert metrics["h3_best_of_k/audio/selection_gain"] == pytest.approx(
        (candidate_audio_losses[0] - candidate_audio_losses[1]).item(), rel=1e-5, abs=1e-6
    )
    assert not any(key.startswith("h3_best_of_k/video/") for key in metrics)
    assert not any(key.startswith("h3_video_best_of_k/") for key in metrics)

    final_video_target = video_latents - video_noise
    final_audio_target = audio_latents - candidate_audio_noise
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


def _recording_randn(monkeypatch) -> list[torch.Tensor]:
    """Record every global-RNG normal draw while still returning real noise."""
    real_randn = torch.randn
    draws: list[torch.Tensor] = []

    def recording_randn(*args, **kwargs):
        noise = real_randn(*args, **kwargs)
        draws.append(noise)
        return noise

    monkeypatch.setattr(torch, "randn", recording_randn)
    return draws


def test_condition_noise_is_drawn_from_the_global_rng_per_condition_and_step(monkeypatch):
    # per-role condition seeds (visuals from seed, audio from seed + 1) made one item's audio noise
    # the next item's visual noise; training now draws from the global RNG like the target noise,
    # so every condition tensor and every step gets its own independent draw
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(task="ref2va", h3_visual_cond_clean=0.5, h3_audio_cond_clean=0.5)
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch()
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    draws = _recording_randn(monkeypatch)
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

    # one draw per condition tensor, visuals before audio, repeated for the second step
    assert [tuple(noise.shape) for noise in draws] == [(1, 24, 1, 4, 4), (1, 32, 2, 8)] * 2
    first_call, second_call = transformer.calls
    # the conditions are zeros here, so clean*x + (1-clean)*eps collapses to the scaled draw
    assert torch.equal(first_call["visual_condition_latents"][0], 0.5 * draws[0])
    assert torch.equal(first_call["audio_condition_latents"][0], 0.5 * draws[1])
    assert not torch.equal(draws[0], draws[2])
    assert not torch.equal(draws[1], draws[3])
    assert not torch.equal(first_call["visual_condition_latents"][0], second_call["visual_condition_latents"][0])
    assert not torch.equal(first_call["audio_condition_latents"][0], second_call["audio_condition_latents"][0])


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


def _one_frame_batch(target_index: int | None = 24):
    batch = {
        "latents_audio": torch.full((1, 32, 2, 2), 4.0),
        "audio_present": torch.zeros(1, dtype=torch.float32),
        "mmh3_hidden_states": [torch.full((3, 12), 0.0)],
        "mmh3_token_tags": [torch.tensor([1, 0, 1], dtype=torch.int64)],
        "timesteps": None,
    }
    if target_index is not None:
        batch["one_frame_target_index"] = torch.tensor([target_index], dtype=torch.int64)
    return batch


def _one_frame_process_batch(trainer, args, batch, transformer):
    video_latents = torch.zeros(1, 24, 1, 4, 4)
    return trainer.process_batch(
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


def test_one_frame_batch_builds_the_time_override_layout(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(one_frame=True)
    trainer.handle_model_specific_args(args)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    transformer = _RecordingTransformer()

    _one_frame_process_batch(trainer, args, _one_frame_batch(target_index=24), transformer)

    layout = transformer.calls[0]["layout"]
    assert layout.task == "t2va"
    assert layout.target_video.frames == 1
    assert layout.target_audio_frames == 2
    assert layout.time_overrides is not None
    assert layout.time_overrides.condition_times == ()
    assert layout.time_overrides.target_time == FRAME_RESCALE * 24
    # the silence placeholder stays excluded from audio supervision
    assert trainer._audio_items_seen == 1
    assert trainer._audio_supervised_seen == 0


def test_one_frame_batch_requires_the_training_flag():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)

    with pytest.raises(ValueError, match=r"pass --one_frame"):
        _one_frame_process_batch(trainer, args, _one_frame_batch(), _RecordingTransformer())


@pytest.mark.parametrize(
    "index",
    [None, torch.tensor(24, dtype=torch.int64), torch.tensor([24], dtype=torch.int32), torch.tensor([-1], dtype=torch.int64)],
)
def test_one_frame_batch_requires_a_valid_index_tensor(index):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(one_frame=True)
    trainer.handle_model_specific_args(args)
    batch = _one_frame_batch(target_index=None)
    if index is not None:
        batch["one_frame_target_index"] = index

    with pytest.raises(ValueError, match="one_frame_target_index|nonnegative"):
        _one_frame_process_batch(trainer, args, batch, _RecordingTransformer())


def test_one_frame_batch_rejects_condition_latents():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(one_frame=True)
    trainer.handle_model_specific_args(args)
    batch = _one_frame_batch()
    batch["latents_first"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_last"] = torch.zeros(1, 24, 1, 4, 4)

    with pytest.raises(ValueError, match="plain T2VA"):
        _one_frame_process_batch(trainer, args, batch, _RecordingTransformer())


def test_video_batch_rejects_a_stray_one_frame_index():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(one_frame=True)
    trainer.handle_model_specific_args(args)
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    batch = _training_batch()
    batch["one_frame_target_index"] = torch.tensor([0], dtype=torch.int64)

    with pytest.raises(ValueError, match="video batch cannot carry one_frame_target_index"):
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
    ("overrides", "message"),
    [
        ({"one_frame": True, "task": "fl2va"}, "requires --task t2va"),
        ({"one_frame": True, "task": "ref2va"}, "requires --task t2va"),
        ({"one_frame": True, "h3_teacher_matching": True}, "does not support --one_frame"),
    ],
)
def test_one_frame_training_flag_validations(overrides, message):
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_trainer_args(**overrides))


def test_one_frame_training_records_provenance_metadata():
    args = _trainer_args(one_frame=True)
    metadata = MiniMaxH3NetworkTrainer().extra_metadata(args)
    assert metadata["ss_minimax_h3_one_frame"] is True
    assert "ss_minimax_h3_one_frame" not in MiniMaxH3NetworkTrainer().extra_metadata(_trainer_args())


def test_h3_best_of_k_metadata_is_absent_at_k_one_and_records_active_configuration():
    trainer = MiniMaxH3NetworkTrainer()
    disabled_args = _trainer_args(h3_best_of_k=1, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(disabled_args)
    disabled = trainer.extra_metadata(disabled_args)
    assert "ss_minimax_h3_best_of_k" not in disabled
    assert "ss_minimax_h3_best_of_k_stream" not in disabled

    enabled_args = _trainer_args(h3_best_of_k=3, h3_best_of_k_stream="audio")
    trainer._validate_and_init_best_of_k(enabled_args)
    enabled = trainer.extra_metadata(enabled_args)
    assert enabled["ss_minimax_h3_best_of_k"] == 3
    assert enabled["ss_minimax_h3_best_of_k_stream"] == "audio"


def test_one_frame_sample_normalization_parses_the_of_option():
    args = _trainer_args(one_frame=True)

    sample = _normalize_h3_sample_parameter(
        args, {"prompt": "a lighthouse", "frame_count": 1, "one_frame": "target_index=24", "width": 64, "height": 64}
    )

    assert sample["frame_count"] == 1
    assert sample["one_frame_target_index"] == 24
    default = _normalize_h3_sample_parameter(args, {"prompt": "a lighthouse", "frame_count": 1})
    assert default["one_frame_target_index"] == 0


@pytest.mark.parametrize(
    ("args_overrides", "sample", "message"),
    [
        ({"task": "fl2va"}, {"prompt": "x", "frame_count": 1, "first_frame": "a.png", "last_frame": "b.png"}, "t2va only"),
        ({}, {"prompt": "x", "frame_count": 1, "one_frame": "target_index=0,control_index=0"}, "control_index"),
        ({}, {"prompt": "x", "frame_count": 124, "one_frame": "target_index=24"}, r"require --f 1"),
    ],
)
def test_one_frame_sample_normalization_rejects_invalid_requests(args_overrides, sample, message):
    args = _trainer_args(**args_overrides)

    with pytest.raises(ValueError, match=message):
        _normalize_h3_sample_parameter(args, sample)


def test_t2va_draws_no_condition_noise(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    draws = _recording_randn(monkeypatch)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))
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

    assert not draws, "T2VA has no conditions to augment, so it must not draw condition noise"


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


def test_h3_zero_audio_weight_makes_the_video_selection_score_the_exact_effective_objective():
    trainer = MiniMaxH3NetworkTrainer()
    output = DiTOutput(
        pred=torch.tensor([[1.0, 5.0]]),
        target=torch.tensor([[3.0, 1.0]]),
        extra={"audio_loss_weight": torch.tensor([0.0], dtype=torch.float32)},
    )

    video, audio = trainer._compute_per_sample_component_losses(output, torch.float32)
    effective = trainer.compute_per_sample_loss(
        _trainer_args(),
        output,
        torch.tensor(0.25),
        None,
        torch.float32,
        torch.float32,
        0,
    )

    assert torch.equal(audio, torch.zeros_like(audio))
    assert torch.equal(video, effective)


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


# --- guidance-distillation loss (contrastive guidance targets) ---


def _uncond_cache(tmp_path, *, rows: int = 2, width: int = 12, value: float = 0.0) -> str:
    from musubi_tuner.minimax_h3.text_encoder import save_h3_uncond_cache

    path = tmp_path / "uncond_space.safetensors"
    save_h3_uncond_cache(
        path,
        torch.full((rows, width), value),
        torch.ones(rows, dtype=torch.int64),
        metadata={"text": " "},
    )
    return str(path)


def test_h3_parser_defaults_leave_the_guidance_loss_off():
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--task", "t2va"])

    assert args.h3_guidance_loss_scale == 0.0
    assert args.h3_guidance_loss_scale_audio is None
    assert args.h3_guidance_loss_sigma_min == 0.0
    assert args.h3_guidance_loss_uncond_cache is None


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"h3_guidance_loss_scale": -1.0}, "h3_guidance_loss_scale"),
        ({"h3_guidance_loss_scale": 3.0, "h3_guidance_loss_scale_audio": -0.5}, "h3_guidance_loss_scale_audio"),
        ({"h3_guidance_loss_scale": 3.0, "h3_guidance_loss_sigma_min": 1.5}, "h3_guidance_loss_sigma_min"),
        ({"h3_guidance_loss_scale": 3.0}, "h3_guidance_loss_uncond_cache"),
    ],
)
def test_h3_guidance_loss_rejects_invalid_coordinates(overrides, message):
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_trainer_args(**overrides))


def test_h3_uncond_cache_round_trips_and_rejects_foreign_formats(tmp_path):
    from safetensors.torch import save_file

    from musubi_tuner.minimax_h3.text_encoder import load_h3_uncond_cache

    path = _uncond_cache(tmp_path, rows=2, width=12, value=0.5)
    hidden, tags, metadata = load_h3_uncond_cache(path)
    assert hidden.shape == (2, 12)
    assert torch.equal(tags, torch.ones(2, dtype=torch.int64))
    assert metadata["text"] == " "

    foreign = tmp_path / "foreign.safetensors"
    save_file({"hidden_states": torch.zeros(2, 12), "token_tags": torch.ones(2, dtype=torch.int64)}, str(foreign))
    with pytest.raises(ValueError, match="cache format"):
        load_h3_uncond_cache(foreign)


def test_guidance_loss_rewrites_both_targets_around_the_uncond_prediction(tmp_path, monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_guidance_loss_scale=3.0, h3_guidance_loss_uncond_cache=_uncond_cache(tmp_path))
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer(video_prediction=2.0, audio_prediction=-1.0)
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

    # two forwards: the no-grad uncond probe first, then the conditional pass
    assert len(transformer.calls) == 2
    uncond_call, cond_call = transformer.calls
    assert uncond_call["layout"].text_length == 2
    assert cond_call["layout"].text_length == 3
    assert uncond_call["text_hidden_states"].shape == (1, 2, 12)
    assert torch.equal(uncond_call["text_token_tags"], torch.ones(1, 2, dtype=torch.int64))
    # everything but the text condition is shared with the conditional pass
    assert torch.equal(uncond_call["video_latents"], cond_call["video_latents"])
    assert torch.equal(uncond_call["audio_latents"], cond_call["audio_latents"])
    assert uncond_call["model_t_video"] is cond_call["model_t_video"]

    # both fake forwards return the same constants, so uncond_video=2, uncond_audio=-1;
    # video target 0 -> 2 + 3*(0-2) = -4, audio target 4 -> -1 + 3*(4+1) = 14
    assert metrics["loss/video"] == pytest.approx(torch.nn.functional.mse_loss(torch.tensor(2.0), torch.tensor(-4.0)).item())
    assert metrics["loss/audio"] == pytest.approx(torch.nn.functional.mse_loss(torch.tensor(-1.0), torch.tensor(14.0)).item())
    assert metrics["guidance/applied"] == 1.0
    assert metrics["guidance/base_sigma"] == pytest.approx(0.25)
    assert metrics["guidance/video_gap_rms"] == pytest.approx(2.0)
    assert metrics["guidance/audio_gap_rms"] == pytest.approx(5.0)
    assert torch.isfinite(loss)


def test_guidance_loss_uses_nested_forward_only_scope_for_uncond_probe(tmp_path, monkeypatch):
    class _BlockSwapGuidanceTransformer(_RecordingTransformer):
        def __init__(self):
            super().__init__()
            self.mode = "training"
            self.events = []

        def switch_block_swap_for_inference(self):
            assert self.mode == "training"
            self.mode = "forward-only"
            self.events.append("inference")

        def switch_block_swap_for_training(self):
            assert self.mode == "forward-only"
            self.mode = "training"
            self.events.append("training")

        def __call__(self, **kwargs):
            expected_mode = "training" if torch.is_grad_enabled() else "forward-only"
            assert self.mode == expected_mode
            self.events.append(f"forward:{expected_mode}")
            return super().__call__(**kwargs)

    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_guidance_loss_scale=3.0, h3_guidance_loss_uncond_cache=_uncond_cache(tmp_path))
    trainer.handle_model_specific_args(args)
    transformer = _BlockSwapGuidanceTransformer()
    batch = _training_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    loss, _ = trainer.process_batch(
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

    assert transformer.events == ["inference", "forward:forward-only", "training", "forward:training"]
    assert torch.isfinite(loss)


def test_guidance_loss_uncond_layout_carries_the_one_frame_overrides(tmp_path, monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(
        one_frame=True,
        h3_guidance_loss_scale=3.0,
        h3_guidance_loss_uncond_cache=_uncond_cache(tmp_path),
    )
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    _, metrics = _one_frame_process_batch(trainer, args, _one_frame_batch(target_index=24), transformer)

    # the no-grad uncond probe first, then the conditional pass, both on one-frame layouts
    assert len(transformer.calls) == 2
    uncond_call, cond_call = transformer.calls
    assert uncond_call["layout"].text_length == 2
    assert uncond_call["layout"].target_video.frames == 1
    assert uncond_call["layout"].target_audio_frames == 2
    assert uncond_call["layout"].time_overrides == cond_call["layout"].time_overrides
    assert uncond_call["layout"].time_overrides.target_time == FRAME_RESCALE * 24
    assert metrics["guidance/applied"] == 1.0


def test_guidance_loss_audio_scale_can_differ_from_video(tmp_path, monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(
        h3_guidance_loss_scale=3.0,
        h3_guidance_loss_scale_audio=1.0,
        h3_guidance_loss_uncond_cache=_uncond_cache(tmp_path),
    )
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer(video_prediction=2.0, audio_prediction=-1.0)
    batch = _training_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    _, metrics = trainer.process_batch(
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

    # audio scale 1 keeps the audio target at the plain velocity: -1 + 1*(4+1) = 4
    assert metrics["loss/audio"] == pytest.approx(torch.nn.functional.mse_loss(torch.tensor(-1.0), torch.tensor(4.0)).item())


def test_guidance_loss_sigma_gate_skips_the_uncond_forward(tmp_path, monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(
        h3_guidance_loss_scale=3.0,
        h3_guidance_loss_sigma_min=0.5,
        h3_guidance_loss_uncond_cache=_uncond_cache(tmp_path),
    )
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer(video_prediction=2.0)
    batch = _training_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    _, metrics = trainer.process_batch(
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

    # base sigma 0.25 < 0.5: single conditional forward, plain velocity target
    assert len(transformer.calls) == 1
    assert metrics["guidance/applied"] == 0.0
    assert metrics["guidance/base_sigma"] == pytest.approx(0.25)
    assert "guidance/video_gap_rms" not in metrics
    assert metrics["loss/video"] == pytest.approx(torch.nn.functional.mse_loss(torch.tensor(2.0), torch.tensor(0.0)).item())


def test_guidance_loss_rejects_a_width_mismatch_against_the_text_cache(tmp_path, monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_guidance_loss_scale=3.0, h3_guidance_loss_uncond_cache=_uncond_cache(tmp_path, width=8))
    trainer.handle_model_specific_args(args)
    batch = _training_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))

    with pytest.raises(ValueError, match="width"):
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


def test_guidance_loss_metadata_is_recorded_only_when_active(tmp_path):
    trainer = MiniMaxH3NetworkTrainer()
    off = trainer.extra_metadata(_trainer_args())
    assert not any(key.startswith("ss_minimax_h3_guidance") for key in off)

    args = _trainer_args(h3_guidance_loss_scale=4.0, h3_guidance_loss_sigma_min=0.3)
    on = trainer.extra_metadata(args)
    assert on["ss_minimax_h3_guidance_loss_scale"] == 4.0
    assert on["ss_minimax_h3_guidance_loss_scale_audio"] == 4.0
    assert on["ss_minimax_h3_guidance_loss_sigma_min"] == 0.3


# --- teacher matching (FL2VA teacher targets for a T2VA student) ---


class _ToggleNetwork:
    def __init__(self):
        self.enabled = True
        self.calls = []

    def set_enabled(self, value):
        self.enabled = bool(value)
        self.calls.append(bool(value))


class _TeacherAwareTransformer(_RecordingTransformer):
    """Returns the teacher constants while the LoRA is disabled, the student constants otherwise."""

    def __init__(self, network, *, teacher_video: float = 3.0, teacher_audio: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.network = network
        self.teacher_video = teacher_video
        self.teacher_audio = teacher_audio

    def __call__(self, **kwargs):
        if not self.network.enabled:
            self.calls.append(kwargs)
            return SimpleNamespace(
                video=torch.full_like(kwargs["video_latents"], self.teacher_video),
                audio=torch.full_like(kwargs["audio_latents"], self.teacher_audio),
            )
        return super().__call__(**kwargs)


def _teacher_batch(*, text_length: int = 3, teacher_text_length: int = 5, teacher_width: int = 12):
    batch = _training_batch(text_length=text_length)
    batch["latents_first"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_last"] = torch.zeros(1, 24, 1, 4, 4)
    batch["mmh3_teacher_hidden_states"] = [torch.zeros(teacher_text_length, teacher_width)]
    batch["mmh3_teacher_token_tags"] = [torch.tensor([1, 0, 0, 1, 1][:teacher_text_length], dtype=torch.int64)]
    return batch


def _ref_teacher_batch(*, text_length: int = 3, teacher_text_length: int = 5, teacher_width: int = 12, include_fl: bool = False):
    # the ref teacher needs no first/last latents (a plain T2VA latent cache suffices);
    # include_fl mimics reusing an FL2VA latent cache, whose endpoint latents go unused
    batch = _training_batch(text_length=text_length)
    if include_fl:
        batch["latents_first"] = torch.zeros(1, 24, 1, 4, 4)
        batch["latents_last"] = torch.zeros(1, 24, 1, 4, 4)
    batch["mmh3_teacher_ref_hidden_states"] = [torch.zeros(teacher_text_length, teacher_width)]
    batch["mmh3_teacher_ref_token_tags"] = [torch.tensor([1, 0, 0, 1, 1][:teacher_text_length], dtype=torch.int64)]
    return batch


def _patch_deterministic_noise(monkeypatch):
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))
    monkeypatch.setattr(torch, "randn", lambda shape, **kwargs: torch.zeros(shape, dtype=kwargs.get("dtype")))


def test_h3_parser_defaults_leave_teacher_matching_off():
    parser = minimax_h3_setup_parser(argparse.ArgumentParser())

    args = parser.parse_args(["--task", "t2va"])

    assert args.h3_teacher_matching is False
    assert args.h3_teacher_conditions == "first,last"
    # the identity-decision band was measured at base sigma 0.6-0.75, so the default anchor starts at 0.75
    assert args.h3_teacher_condition_sigma_max == 0.75


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"h3_teacher_matching": True, "task": "fl2va"}, "t2va"),
        ({"h3_teacher_matching": True, "h3_guidance_loss_scale": 3.0}, "mutually exclusive"),
        ({"h3_teacher_matching": True, "h3_teacher_conditions": "first"}, "first,last"),
        ({"h3_teacher_matching": True, "h3_teacher_condition_sigma_max": 1.5}, "h3_teacher_condition_sigma_max"),
        ({"h3_teacher_matching": True, "h3_teacher_condition_sigma_max": -0.1}, "h3_teacher_condition_sigma_max"),
    ],
)
def test_h3_teacher_matching_rejects_invalid_configurations(overrides, message):
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_trainer_args(**overrides))


def test_h3_teacher_conditions_normalizes_whitespace():
    MiniMaxH3NetworkTrainer().handle_model_specific_args(
        _trainer_args(h3_teacher_matching=True, h3_teacher_conditions=" first , last ")
    )


def test_teacher_matching_replaces_both_targets_with_the_frozen_base_predictions(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True)
    trainer.handle_model_specific_args(args)
    network = _ToggleNetwork()
    transformer = _TeacherAwareTransformer(
        network, teacher_video=3.0, teacher_audio=0.5, video_prediction=2.0, audio_prediction=-1.0
    )
    batch = _teacher_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    _patch_deterministic_noise(monkeypatch)

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        network,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    # two forwards: the no-grad LoRA-disabled teacher first, then the student pass
    assert len(transformer.calls) == 2
    teacher_call, student_call = transformer.calls
    assert teacher_call["layout"].task == "fl2va"
    assert teacher_call["layout"].text_length == 5
    assert len(teacher_call["visual_condition_latents"]) == 2
    assert teacher_call["text_hidden_states"].shape == (1, 5, 12)
    assert student_call["layout"].task == "t2va"
    assert student_call["layout"].text_length == 3
    assert len(student_call["visual_condition_latents"]) == 0
    # the LoRA is disabled exactly for the teacher forward and restored afterwards
    assert network.calls == [False, True]
    assert network.enabled is True
    # everything but the conditioning is shared between the two passes
    assert torch.equal(teacher_call["video_latents"], student_call["video_latents"])
    assert torch.equal(teacher_call["audio_latents"], student_call["audio_latents"])
    assert teacher_call["model_t_video"] is student_call["model_t_video"]

    # both targets are the teacher predictions: student 2.0 vs teacher 3.0, audio -1.0 vs 0.5
    # the decomposed teacher-matching loss equals the MSE up to float32 rounding of the norm path
    assert metrics["loss/video"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["loss/audio"] == pytest.approx(2.25, rel=1e-4)
    assert metrics["teacher/base_sigma"] == pytest.approx(0.25)
    assert metrics["teacher/conditioned"] == 1.0
    # flow targets are 0 (video) and 4 (audio), so the logged teacher deviations are 3.0 and 3.5
    assert metrics["teacher/video_flow_gap_rms"] == pytest.approx(3.0)
    assert metrics["teacher/audio_flow_gap_rms"] == pytest.approx(3.5)
    # direction/magnitude decomposition: video 2.0 vs 3.0 is parallel at 2/3 the norm,
    # audio -1.0 vs 0.5 is anti-parallel at twice the norm
    assert metrics["teacher/video_cos"] == pytest.approx(1.0)
    assert metrics["teacher/video_norm_ratio"] == pytest.approx(2.0 / 3.0)
    assert metrics["teacher/audio_cos"] == pytest.approx(-1.0)
    assert metrics["teacher/audio_norm_ratio"] == pytest.approx(2.0)
    # constant residuals (-1.0 video, -1.5 audio) are pure DC
    assert metrics["teacher/video_residual_dc_rms"] == pytest.approx(1.0)
    assert metrics["teacher/video_residual_ac_rms"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["teacher/audio_residual_dc_rms"] == pytest.approx(1.5)
    assert metrics["teacher/audio_residual_ac_rms"] == pytest.approx(0.0, abs=1e-6)
    assert torch.isfinite(loss)


def test_timestep_focus_remaps_a_uniform_draw_into_the_band_mixture():
    u = torch.linspace(0.0, 0.999, 1000)

    out = _apply_timestep_focus(u, 0.4, 0.8, 0.5)

    assert out[u < 0.5].min() >= 0.4 and out[u < 0.5].max() < 0.8  # focused draws stay in the band
    assert out[u >= 0.5].min() >= 0.0 and out[u >= 0.5].max() <= 1.0  # the rest stays uniform over [0,1)
    in_band = ((out >= 0.4) & (out < 0.8)).float().mean().item()
    assert in_band == pytest.approx(0.5 + 0.5 * 0.4, abs=0.02)  # density = prob + (1-prob)*(max-min)
    torch.testing.assert_close(_apply_timestep_focus(u, 0.4, 0.8, 0.0), u)  # prob 0 = identity


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"h3_timestep_focus_prob": 1.5}, "focus_prob"),
        ({"h3_timestep_focus_prob": 0.5, "h3_timestep_focus_min": 0.8, "h3_timestep_focus_max": 0.4}, "min < max"),
        ({"h3_timestep_focus_prob": 0.5, "min_timestep": 100}, "min_timestep"),
        ({"h3_teacher_loss_dc_weight": 0.0}, "h3_teacher_matching"),
        ({"h3_teacher_loss_mag_weight": 0.5}, "h3_teacher_matching"),
        ({"h3_teacher_preservation_weight": 2.0}, "h3_teacher_matching"),
        ({"h3_teacher_matching": True, "h3_teacher_loss_mag_weight": -1.0}, "nonnegative"),
        ({"h3_teacher_matching": True, "h3_teacher_preservation_weight": -0.5}, "nonnegative"),
    ],
)
def test_teacher_loss_and_timestep_focus_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        MiniMaxH3NetworkTrainer().handle_model_specific_args(_trainer_args(**overrides))


def _dc_split_output(conditioned: float) -> DiTOutput:
    target = torch.zeros(1, 2, 2, 2, 2)
    pred = torch.zeros(1, 2, 2, 2, 2)
    # channel 0: constant +2 offset (pure DC, energy 2.0); channel 1: +/-1 pattern (pure AC, energy 0.5)
    pred[:, 0] += 2.0
    pred[:, 1, ..., 0] += 1.0
    pred[:, 1, ..., 1] -= 1.0
    extra = {
        "audio_pred": None,
        "audio_target": None,
        "audio_loss_weight": torch.tensor([0.0]),
        "guidance_log": {"teacher/conditioned": torch.tensor(conditioned)},
    }
    return DiTOutput(pred=pred, target=target, extra=extra)


def test_compute_loss_attenuates_the_video_residual_dc_component_on_teaching_steps():
    trainer = MiniMaxH3NetworkTrainer()

    for dc_weight, expected in ((1.0, 2.5), (0.25, 1.0), (0.0, 0.5)):
        loss, logs = trainer.compute_loss(
            _trainer_args(h3_teacher_matching=True, h3_teacher_loss_dc_weight=dc_weight),
            _dc_split_output(conditioned=1.0),
            None,
            None,
            torch.bfloat16,
            torch.float32,
            0,
        )
        assert logs["loss/video"].item() == pytest.approx(expected)
        assert loss.item() == pytest.approx(expected)


def test_compute_loss_keeps_full_dc_and_applies_the_preservation_weight_on_anchor_steps():
    trainer = MiniMaxH3NetworkTrainer()

    loss, logs = trainer.compute_loss(
        _trainer_args(h3_teacher_matching=True, h3_teacher_loss_dc_weight=0.0, h3_teacher_preservation_weight=2.0),
        _dc_split_output(conditioned=0.0),
        None,
        None,
        torch.bfloat16,
        torch.float32,
        0,
    )

    # the anchor step ignores the DC attenuation (full MSE value) and doubles the returned loss;
    # loss/video is logged unweighted so sigma-binned reads stay comparable
    assert logs["loss/video"].item() == pytest.approx(2.5)
    assert loss.item() == pytest.approx(5.0)


def test_preservation_density_compensation_restores_the_anchor_share_under_focus():
    from musubi_tuner.minimax_h3_train_network import _preservation_density_compensation

    # run3 coordinates: anchor width 0.25, focus band [0.4,0.8) at prob 0.5 leaves the anchor 0.1875
    assert _preservation_density_compensation(0.75, 0.4, 0.8, 0.5) == pytest.approx(0.25 / 0.1875)
    assert _preservation_density_compensation(0.75, 0.4, 0.8, 0.0) == 1.0  # focus off = no correction
    assert _preservation_density_compensation(1.0, 0.4, 0.8, 0.5) == 1.0  # no anchor band
    # focus band fully below the anchor: the anchor thins to (1-p)*width, compensation 1/(1-p)
    assert _preservation_density_compensation(0.8, 0.4, 0.7, 0.5) == pytest.approx(2.0)


def test_decomposed_flow_loss_keeps_the_mse_value_but_splits_the_gradient_geometry():
    generator = torch.Generator().manual_seed(0)
    pred = torch.randn(1, 3, 2, 2, 2, generator=generator)
    target = torch.randn(1, 3, 2, 2, 2, generator=generator)

    # at unit weights the value equals the MSE exactly (only the gradients differ)
    unit = _decomposed_flow_loss(pred.clone().requires_grad_(True), target, 1.0, 1.0)
    torch.testing.assert_close(unit, torch.nn.functional.mse_loss(pred, target))

    # direction-only gradient is purely rotational: orthogonal to the prediction
    rotational = pred.clone().requires_grad_(True)
    _decomposed_flow_loss(rotational, target, 0.0, 1.0).backward()
    grad = rotational.grad.flatten()
    radial_unit = pred.flatten() / pred.flatten().norm()
    assert abs(torch.dot(grad, radial_unit).item()) < 1e-5 * grad.norm().item()

    # magnitude-only gradient is purely radial: no rotational component
    radial = pred.clone().requires_grad_(True)
    _decomposed_flow_loss(radial, target, 1.0, 0.0).backward()
    grad = radial.grad.flatten()
    tangential = grad - torch.dot(grad, radial_unit) * radial_unit
    assert tangential.norm().item() < 1e-5 * grad.norm().item()


def test_extra_metadata_records_teacher_loss_shape_and_timestep_focus():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(
        h3_teacher_matching=True,
        h3_teacher_loss_dc_weight=0.2,
        h3_teacher_loss_mag_weight=0.5,
        h3_teacher_preservation_weight=1.5,
        h3_timestep_focus_prob=0.5,
    )

    metadata = trainer.extra_metadata(args)

    assert metadata["ss_minimax_h3_teacher_loss"] == "decomposed_mag_dir"
    assert metadata["ss_minimax_h3_teacher_loss_dc_weight"] == 0.2
    assert metadata["ss_minimax_h3_teacher_loss_mag_weight"] == 0.5
    assert metadata["ss_minimax_h3_teacher_preservation_weight"] == 1.5
    assert metadata["ss_minimax_h3_timestep_focus_min"] == 0.4
    assert metadata["ss_minimax_h3_timestep_focus_max"] == 0.8
    assert metadata["ss_minimax_h3_timestep_focus_prob"] == 0.5

    plain = trainer.extra_metadata(_trainer_args())
    assert "ss_minimax_h3_teacher_loss" not in plain
    assert "ss_minimax_h3_teacher_loss_dc_weight" not in plain
    assert "ss_minimax_h3_timestep_focus_prob" not in plain


def test_prediction_geometry_log_splits_the_residual_into_style_dc_and_content_ac():
    prediction = torch.zeros(1, 2, 2, 2, 2)
    target = torch.zeros(1, 2, 2, 2, 2)
    # channel 0: constant +2 offset (pure DC); channel 1: zero-mean +/-1 pattern (pure AC)
    prediction[:, 0] += 2.0
    prediction[:, 1, ..., 0] += 1.0
    prediction[:, 1, ..., 1] -= 1.0

    metrics = _prediction_geometry_log("video", prediction, target)

    # DC energy: channel 0 contributes 2^2 over half the channels -> rms sqrt(4/2)
    assert metrics["teacher/video_residual_dc_rms"] == pytest.approx((4.0 / 2.0) ** 0.5)
    # AC energy: channel 1 contributes 1^2 everywhere over half the elements -> rms sqrt(1/2)
    assert metrics["teacher/video_residual_ac_rms"] == pytest.approx((1.0 / 2.0) ** 0.5)
    # the split conserves the residual energy: rms^2 = dc_rms^2 + ac_rms^2
    residual_rms = (prediction - target).pow(2).mean().sqrt()
    assert metrics["teacher/video_residual_dc_rms"] ** 2 + metrics["teacher/video_residual_ac_rms"] ** 2 == pytest.approx(
        residual_rms.item() ** 2
    )


def test_teacher_condition_sigma_max_switches_the_teacher_to_a_preservation_anchor(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    # the drawn base sigma (0.25) lies above the threshold, so the teacher must drop the
    # endpoint conditions and run on the student's own text and layout
    args = _trainer_args(h3_teacher_matching=True, h3_teacher_condition_sigma_max=0.2)
    trainer.handle_model_specific_args(args)
    network = _ToggleNetwork()
    transformer = _TeacherAwareTransformer(
        network, teacher_video=3.0, teacher_audio=0.5, video_prediction=2.0, audio_prediction=-1.0
    )
    batch = _teacher_batch()
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    _patch_deterministic_noise(monkeypatch)

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        network,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    teacher_call, student_call = transformer.calls
    assert teacher_call["layout"].task == "t2va"
    assert teacher_call["layout"] is student_call["layout"]
    assert teacher_call["text_hidden_states"].shape == (1, 3, 12)
    assert len(teacher_call["visual_condition_latents"]) == 0
    # the LoRA is still disabled for the anchor forward, and the targets are its predictions
    assert network.calls == [False, True]
    assert metrics["teacher/conditioned"] == 0.0
    # the decomposed teacher-matching loss equals the MSE up to float32 rounding of the norm path
    assert metrics["loss/video"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["loss/audio"] == pytest.approx(2.25, rel=1e-4)
    assert torch.isfinite(loss)


def _teacher_matching_process_batch(trainer, args, batch, *, network, transformer=None):
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    return trainer.process_batch(
        args,
        _Accelerator(),
        transformer if transformer is not None else _RecordingTransformer(),
        network,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )


def test_teacher_matching_requires_teacher_text_rows(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True)
    trainer.handle_model_specific_args(args)
    batch = _teacher_batch()
    del batch["mmh3_teacher_hidden_states"]
    del batch["mmh3_teacher_token_tags"]

    with pytest.raises(ValueError, match="teacher text rows"):
        _teacher_matching_process_batch(trainer, args, batch, network=_ToggleNetwork())


def test_teacher_matching_requires_fl2va_latent_caches(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True)
    trainer.handle_model_specific_args(args)
    batch = _teacher_batch()
    del batch["latents_first"]
    del batch["latents_last"]

    with pytest.raises(ValueError, match="FL2VA-style latent caches"):
        _teacher_matching_process_batch(trainer, args, batch, network=_ToggleNetwork())


def test_teacher_text_rows_are_rejected_without_the_teacher_matching_flag():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)

    with pytest.raises(ValueError, match="--h3_teacher_matching"):
        _teacher_matching_process_batch(trainer, args, _teacher_batch(), network=None)


def test_teacher_matching_rejects_a_teacher_text_width_mismatch(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True)
    trainer.handle_model_specific_args(args)

    with pytest.raises(ValueError, match="width"):
        _teacher_matching_process_batch(trainer, args, _teacher_batch(teacher_width=8), network=_ToggleNetwork())


# --- ref teacher (Ref2VA self-reference teacher for a T2VA student) ---


def test_h3_teacher_conditions_accepts_ref_and_records_it_in_metadata():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True, h3_teacher_conditions=" ref ")
    trainer.handle_model_specific_args(args)

    metadata = trainer.extra_metadata(args)
    assert metadata["ss_minimax_h3_teacher_conditions"] == "ref"


def test_ref_teacher_matching_runs_the_teacher_on_the_self_reference_layout(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True, h3_teacher_conditions="ref")
    trainer.handle_model_specific_args(args)
    network = _ToggleNetwork()
    transformer = _TeacherAwareTransformer(
        network, teacher_video=3.0, teacher_audio=0.5, video_prediction=2.0, audio_prediction=-1.0
    )
    batch = _ref_teacher_batch()
    video_latents = torch.full((1, 24, 2, 4, 4), 2.0)
    _patch_deterministic_noise(monkeypatch)

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        network,
        batch,
        video_latents,
        torch.zeros_like(video_latents),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    # two forwards: the no-grad LoRA-disabled teacher on the Ref2VA layout, then the student pass
    assert len(transformer.calls) == 2
    teacher_call, student_call = transformer.calls
    assert teacher_call["layout"].task == "ref2va"
    assert teacher_call["layout"].text_length == 5
    assert len(teacher_call["layout"].references) == 1
    reference = teacher_call["layout"].references[0]
    assert reference.kind == "video"
    assert (reference.video.frames, reference.video.height, reference.video.width) == (2, 4, 4)
    assert reference.audio_frames == batch["latents_audio"].shape[-1]
    # the reference conditions are the cached target latents with the standard clean augmentation
    assert len(teacher_call["visual_condition_latents"]) == 1
    torch.testing.assert_close(teacher_call["visual_condition_latents"][0], torch.full_like(video_latents, 2.0 * 0.999))
    assert len(teacher_call["audio_condition_latents"]) == 1
    torch.testing.assert_close(teacher_call["audio_condition_latents"][0], batch["latents_audio"])  # audio clean 1.0
    assert student_call["layout"].task == "t2va"
    assert len(student_call["visual_condition_latents"]) == 0
    assert len(student_call["audio_condition_latents"]) == 0
    assert network.calls == [False, True]
    # both targets are the teacher predictions: student 2.0 vs teacher 3.0, audio -1.0 vs 0.5
    assert metrics["teacher/conditioned"] == 1.0
    assert metrics["loss/video"] == pytest.approx(1.0, rel=1e-4)
    assert metrics["loss/audio"] == pytest.approx(2.25, rel=1e-4)
    assert torch.isfinite(loss)


def test_ref_teacher_works_without_fl_latents_and_ignores_them_when_present(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True, h3_teacher_conditions="ref")
    trainer.handle_model_specific_args(args)
    _patch_deterministic_noise(monkeypatch)

    # an FL2VA latent cache can be reused: the endpoint latents never reach the teacher forward
    network = _ToggleNetwork()
    transformer = _TeacherAwareTransformer(network)
    _teacher_matching_process_batch(trainer, args, _ref_teacher_batch(include_fl=True), network=network, transformer=transformer)
    teacher_call = transformer.calls[0]
    assert teacher_call["layout"].task == "ref2va"
    assert len(teacher_call["visual_condition_latents"]) == 1  # the self-reference only


def test_ref_teacher_switches_to_the_preservation_anchor_above_sigma_max(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    # the drawn base sigma (0.25) lies above the threshold, so the teacher must drop the
    # reference conditions and run on the student's own text and layout
    args = _trainer_args(h3_teacher_matching=True, h3_teacher_conditions="ref", h3_teacher_condition_sigma_max=0.2)
    trainer.handle_model_specific_args(args)
    network = _ToggleNetwork()
    transformer = _TeacherAwareTransformer(network)
    _patch_deterministic_noise(monkeypatch)

    _, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        transformer,
        network,
        _ref_teacher_batch(),
        torch.zeros(1, 24, 2, 4, 4),
        torch.zeros(1, 24, 2, 4, 4),
        None,
        torch.bfloat16,
        torch.float32,
        None,
        0,
    )

    teacher_call, student_call = transformer.calls
    assert teacher_call["layout"] is student_call["layout"]
    assert len(teacher_call["visual_condition_latents"]) == 0
    assert len(teacher_call["audio_condition_latents"]) == 0
    assert metrics["teacher/conditioned"] == 0.0


@pytest.mark.parametrize(
    "conditions, batch_factory, message",
    [
        # the text cache kind must match the configured teacher conditions: distinct tensor
        # keys per kind turn a cache/flag mismatch into a hard error instead of a silent desync
        ("ref", _teacher_batch, "first,last teacher rows"),
        ("ref", _training_batch, "reference teacher text rows"),
        ("first,last", lambda: _ref_teacher_batch(include_fl=True), "ref teacher rows"),
    ],
)
def test_teacher_mode_and_text_cache_kind_must_match(conditions, batch_factory, message):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True, h3_teacher_conditions=conditions)
    trainer.handle_model_specific_args(args)

    with pytest.raises(ValueError, match=message):
        _teacher_matching_process_batch(trainer, args, batch_factory(), network=_ToggleNetwork())


def test_ref_teacher_text_rows_are_rejected_without_the_teacher_matching_flag():
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)

    with pytest.raises(ValueError, match="--h3_teacher_matching"):
        _teacher_matching_process_batch(trainer, args, _ref_teacher_batch(), network=None)


def test_teacher_matching_requires_the_lora_network(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True)
    trainer.handle_model_specific_args(args)
    _patch_deterministic_noise(monkeypatch)

    with pytest.raises(RuntimeError, match="LoRA network"):
        _teacher_matching_process_batch(trainer, args, _teacher_batch(), network=None)


def test_teacher_matching_metadata_is_recorded_only_when_active():
    trainer = MiniMaxH3NetworkTrainer()
    off = trainer.extra_metadata(_trainer_args())
    assert not any(key.startswith("ss_minimax_h3_teacher") for key in off)

    on = trainer.extra_metadata(
        _trainer_args(h3_teacher_matching=True, h3_teacher_conditions=" first , last ", h3_teacher_condition_sigma_max=0.5)
    )
    assert on["ss_minimax_h3_teacher_matching"] is True
    assert on["ss_minimax_h3_teacher_conditions"] == "first,last"
    assert on["ss_minimax_h3_teacher_condition_sigma_max"] == 0.5


def test_lora_set_enabled_bypasses_training_modules():
    # regression guard for the teacher-matching smoke bug: only LoRAInfModule honored
    # `enabled`, so set_enabled(False) silently kept the LoRA active in training forwards
    model = _tiny_model(num_layers=1)
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    for lora in network.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=1.0)
    proj = model.blocks[0].attn.qkv_proj
    x = torch.randn(2, proj.weight.shape[1])

    with torch.no_grad():
        adapted = proj(x)
        network.set_enabled(False)
        disabled = proj(x)
        network.set_enabled(True)
        restored = proj(x)

    base = torch.nn.functional.linear(x, proj.weight, proj.bias)
    assert not torch.allclose(adapted, base)
    torch.testing.assert_close(disabled, base)
    torch.testing.assert_close(restored, adapted)


def test_teacher_matching_bypasses_the_lora_on_a_real_network(monkeypatch):
    # end-to-end against the real tiny model and a real LoRA network: the lora_down
    # projections must fire only in the student forward, never in the teacher forward
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(h3_teacher_matching=True)
    trainer.handle_model_specific_args(args)
    model = _tiny_model(num_layers=1)
    model.requires_grad_(False)
    network = lora_minimax_h3.create_arch_network(1.0, 2, 2.0, None, None, model)
    network.apply_to(None, model, apply_text_encoder=False, apply_unet=True)
    for lora in network.unet_loras:
        torch.nn.init.normal_(lora.lora_up.weight, std=1.0)
    lora_down_calls = []
    network.unet_loras[0].lora_down.register_forward_hook(lambda module, inputs, output: lora_down_calls.append(1))
    _patch_deterministic_noise(monkeypatch)

    loss, metrics = trainer.process_batch(
        args,
        _Accelerator(),
        model,
        network,
        _teacher_batch(),
        torch.zeros(1, 24, 2, 4, 4),
        torch.zeros(1, 24, 2, 4, 4),
        None,
        torch.float32,
        torch.float32,
        None,
        0,
    )

    assert len(lora_down_calls) == 1
    assert all(lora.enabled for lora in network.unet_loras)
    assert torch.isfinite(loss)
    assert metrics["teacher/base_sigma"] == pytest.approx(0.25)
