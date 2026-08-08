from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.model import MiniMaxH3Config, MiniMaxH3Model
from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.modules.convrot_int8_kernels import quantize_int8_convrot_weight
from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Artifact, ConvRotInt8LayerSpec, prepare_convrot_int8_model
from musubi_tuner.dataset.cache_io import AUDIO_PRESENT_KEY
from musubi_tuner.minimax_h3_train_network import (
    MiniMaxH3NetworkTrainer,
    minimax_h3_setup_parser,
    validate_h3_dataset_batch_size,
)
from musubi_tuner.training.audio_loss import scan_audio_supervised_fraction
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer


def _dataset_group(*batch_sizes: int):
    return SimpleNamespace(
        datasets=[SimpleNamespace(batch_manager=SimpleNamespace(batch_size=batch_size)) for batch_size in batch_sizes]
    )


def test_h3_dataset_accepts_batch_size_one_without_inspecting_cache_items():
    validate_h3_dataset_batch_size(_dataset_group(1, 1))


def test_h3_dataset_rejects_real_batches_and_points_to_gradient_accumulation():
    with pytest.raises(ValueError, match=r"dataset 1.*batch_size=2.*gradient accumulation"):
        validate_h3_dataset_batch_size(_dataset_group(1, 2))


def _presence_cache(path: Path, *, present: torch.Tensor | None) -> Path:
    tensors = {"latents_2x4x4_float32": torch.zeros(24, 2, 4, 4)}
    if present is not None:
        tensors[AUDIO_PRESENT_KEY] = present
    save_file(tensors, path)
    return path.resolve()


def _audio_presence_dataset_group(entries: list[Path]):
    items = [SimpleNamespace(latent_cache_path=str(path)) for path in entries]
    manager = SimpleNamespace(batch_size=1, buckets={(64, 64, 5): items})
    dataset = SimpleNamespace(batch_manager=manager)
    return SimpleNamespace(datasets=[dataset])


def test_audio_supervision_scan_counts_repeats_and_opens_each_unique_cache_once(monkeypatch, tmp_path: Path):
    from musubi_tuner.training import audio_loss

    supervised = _presence_cache(tmp_path / "supervised.safetensors", present=torch.tensor(1.0, dtype=torch.float32))
    unsupervised = _presence_cache(tmp_path / "unsupervised.safetensors", present=torch.tensor(0.0, dtype=torch.float32))
    dataset_group = _audio_presence_dataset_group([supervised, supervised, unsupervised])
    opened = []
    real_safe_open = audio_loss.safe_open

    def counted_safe_open(path, *args, **kwargs):
        opened.append(Path(path).resolve())
        return real_safe_open(path, *args, **kwargs)

    monkeypatch.setattr(audio_loss, "safe_open", counted_safe_open)

    fraction = scan_audio_supervised_fraction(dataset_group)

    assert fraction == pytest.approx(2 / 3)
    assert opened.count(supervised) == 1
    assert opened.count(unsupervised) == 1


def test_audio_supervision_scan_rejects_pre_audio_present_caches(tmp_path: Path):
    legacy = _presence_cache(tmp_path / "legacy.safetensors", present=None)

    with pytest.raises(ValueError, match="re-run latent caching"):
        scan_audio_supervised_fraction(_audio_presence_dataset_group([legacy]))


@pytest.mark.parametrize(
    "present",
    [
        torch.tensor([0.0], dtype=torch.float32),
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(float("nan"), dtype=torch.float32),
        torch.tensor(0.5, dtype=torch.float32),
    ],
)
def test_audio_supervision_scan_rejects_invalid_presence_entries(tmp_path: Path, present: torch.Tensor):
    path = _presence_cache(tmp_path / "invalid.safetensors", present=present)

    with pytest.raises(ValueError, match="audio_present"):
        scan_audio_supervised_fraction(_audio_presence_dataset_group([path]))


def test_h3_dataset_build_scans_audio_supervision_before_returning(monkeypatch, tmp_path: Path, caplog):
    path = _presence_cache(tmp_path / "unsupervised.safetensors", present=torch.tensor(0.0, dtype=torch.float32))
    dataset_group = _audio_presence_dataset_group([path])
    sentinel_collator = object()
    sentinel_epoch = object()
    monkeypatch.setattr(
        NetworkTrainer,
        "_build_dataset",
        lambda self, args: (dataset_group, sentinel_collator, sentinel_epoch),
    )
    caplog.set_level("INFO")
    trainer = MiniMaxH3NetworkTrainer()

    result = trainer._build_dataset(_trainer_args())

    assert result == (dataset_group, sentinel_collator, sentinel_epoch)
    assert trainer._supervised_audio_fraction == 0.0
    assert "supervised_audio_fraction=0.000000" in caplog.text


class _Accelerator:
    device = torch.device("cpu")

    @staticmethod
    def autocast():
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
        "blocks_to_swap": 0,
        "sample_prompts": None,
        "gradient_checkpointing": False,
        "task": "t2va",
        "video_only": False,
        "audio_loss_weight": 1.0,
        "convrot_int8": False,
        "convrot_int8_bwd": "bf16",
        "base_weights": None,
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
    trainer._sampling_video_vae = VideoVAE()
    trainer._sampling_audio_vae = AudioVAE()
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
        None,
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


def test_prepare_training_samples_encodes_text_once_and_owns_both_vaes_without_using_the_shared_vae(tmp_path, monkeypatch):
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

    sample_parameters, shared_vae = trainer._prepare_sampling(args, _Accelerator(), torch.bfloat16)

    assert shared_vae is None
    assert events == ["load_text_encoder", "encode_text", ("load_video_vae", torch.float16), "load_audio_vae"]
    assert isinstance(trainer._sampling_video_vae, VideoVAE)
    assert isinstance(trainer._sampling_audio_vae, AudioVAE)
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

    sample_parameters, shared_vae = trainer._prepare_sampling(args, _Accelerator(), torch.bfloat16)

    assert shared_vae is None
    parameter = sample_parameters[0]
    assert parameter["h3_layout"].task == "ref2va"
    assert parameter["h3_layout"].references == (H3ReferenceGeometry("video", video=H3VideoGeometry(2, 4, 4), audio_frames=8),)
    assert parameter["h3_visual_conditions"] == (visual,)
    assert parameter["h3_audio_conditions"] == (audio,)
    assert video_vae_load_dtypes == [torch.float32]
    assert trainer._sampling_video_vae.dtype_probe.dtype is torch.float16
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

    loss, metrics = trainer.process_batch(
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
    assert loss == pytest.approx((expected_video_loss + expected_audio_loss).item())
    assert metrics["loss/video"] == pytest.approx(expected_video_loss.item())
    assert metrics["loss/audio"] == pytest.approx(expected_audio_loss.item())


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


def _cpu_noise(shape, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def test_condition_noise_restarts_per_role_uses_audio_seed_plus_one_and_changes_per_step(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args(task="ref2va", h3_visual_cond_clean=0.5, h3_audio_cond_clean=0.5)
    trainer.handle_model_specific_args(args)
    transformer = _RecordingTransformer()
    batch = _training_batch()
    batch["latents_ref_000_image"] = torch.zeros(1, 24, 1, 4, 4)
    batch["latents_ref_001_audio"] = torch.zeros(1, 32, 2, 8)
    video_latents = torch.zeros(1, 24, 2, 4, 4)
    seeds = iter((torch.tensor([100]), torch.tensor([300])))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: next(seeds))
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
    assert torch.equal(first_visual[0], 0.5 * _cpu_noise((24, 1, 4, 4), 100))
    assert torch.equal(first_audio[0], 0.5 * _cpu_noise((32, 2, 8), 101))
    assert not torch.equal(first_visual, second_call["visual_condition_latents"][0])
    assert not torch.equal(first_audio, second_call["audio_condition_latents"][0])


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


def test_t2va_does_not_advance_the_condition_seed_stream(monkeypatch):
    trainer = MiniMaxH3NetworkTrainer()
    args = _trainer_args()
    trainer.handle_model_specific_args(args)
    monkeypatch.setattr(torch, "rand", lambda shape, **kwargs: torch.tensor([0.25], device=kwargs.get("device")))
    monkeypatch.setattr(torch, "randn_like", lambda tensor, *args, **kwargs: torch.zeros_like(tensor))
    monkeypatch.setattr(
        torch,
        "randint",
        lambda *args, **kwargs: pytest.fail("T2VA without conditions must not draw a condition seed"),
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


def test_compute_loss_is_video_mean_plus_weighted_audio_mean_mse():
    trainer = MiniMaxH3NetworkTrainer()

    def output():
        return DiTOutput(
            pred=torch.tensor([1.0, 5.0]),
            target=torch.tensor([3.0, 1.0]),
            extra={
                "audio_pred": torch.tensor([0.0, 2.0]),
                "audio_target": torch.tensor([2.0, 2.0]),
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
        pred=torch.tensor([1.0]),
        target=torch.tensor([3.0]),
        extra={"audio_pred": torch.tensor([0.0]), "audio_target": torch.tensor([2.0])},
    )

    with pytest.raises(ValueError, match="audio loss weight"):
        trainer.compute_loss(_trainer_args(), output, torch.tensor(0.25), object(), torch.bfloat16, torch.float32, 7)


def test_compute_loss_skips_audio_expression_and_gradient_for_zero_weight():
    trainer = MiniMaxH3NetworkTrainer()
    video_pred = torch.tensor([1.0, 5.0], requires_grad=True)
    audio_pred = torch.tensor([float("nan"), 2.0], requires_grad=True)
    output = DiTOutput(
        pred=video_pred,
        target=torch.tensor([3.0, 1.0]),
        extra={
            "audio_pred": audio_pred,
            "audio_target": torch.tensor([2.0, 2.0]),
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
    trainer._supervised_audio_fraction = 0.25

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
    quantized = {}
    layers = {}
    for module_path in target_paths:
        module = model.get_submodule(module_path)
        quantized[module_path] = quantize_int8_convrot_weight(module.weight.detach(), 4)
        layers[module_path] = ConvRotInt8LayerSpec(
            module_path,
            f"{module_path}.weight",
            f"{module_path}.scale_weight",
            4,
        )
    prepare_convrot_int8_model(model, ConvRotInt8Artifact(layers, frozenset()))
    with torch.no_grad():
        for module_path, (weight, scale) in quantized.items():
            module = model.get_submodule(module_path)
            module.weight.copy_(weight)
            module.scale_weight.copy_(scale)

    model.requires_grad_(False)
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
