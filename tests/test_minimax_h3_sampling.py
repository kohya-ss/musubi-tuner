from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3.sampling import (
    augment_condition_latents,
    build_shifted_schedule,
    decode_joint_av,
    initialize_target_latents,
    sample_joint_av,
    write_joint_av,
)
from musubi_tuner.minimax_h3_generate_video import (
    load_cached_text_conditioning,
    run_generation,
    validate_generation_args,
)


def _layout():
    return build_h3_layout(
        task="t2va",
        text_length=3,
        target_video=H3VideoGeometry(2, 4, 4),
        target_audio_frames=8,
    )


def _cpu_noise(shape, generator):
    return torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")


def test_shifted_schedules_share_one_descending_base_grid_but_keep_modality_shifts():
    schedule = build_shifted_schedule(2, video_shift=12.0, audio_shift=3.0)

    torch.testing.assert_close(schedule.base, torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64))
    torch.testing.assert_close(schedule.video, torch.tensor([1.0, 12.0 / 13.0, 0.0], dtype=torch.float64))
    torch.testing.assert_close(schedule.audio, torch.tensor([1.0, 0.75, 0.0], dtype=torch.float64))
    assert not torch.equal(schedule.video, schedule.audio)


@pytest.mark.parametrize("shift", (0.0, 101.0))
def test_shifted_schedule_rejects_out_of_contract_shifts(shift):
    with pytest.raises(ValueError, match="shift"):
        build_shifted_schedule(2, video_shift=shift, audio_shift=3.0)


def test_target_initialization_draws_video_then_audio_from_one_request_generator():
    video, audio = initialize_target_latents(
        video_shape=(1, 24, 2, 4, 4),
        audio_shape=(1, 32, 2, 8),
        seed=123,
        device=torch.device("cpu"),
        video_dtype=torch.float16,
        audio_dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu").manual_seed(123)
    expected_video = _cpu_noise((1, 24, 2, 4, 4), generator).to(torch.float16)
    expected_audio = _cpu_noise((1, 32, 2, 8), generator)

    assert torch.equal(video, expected_video)
    assert torch.equal(audio, expected_audio)


def test_condition_augmentation_resets_visual_stream_and_uses_seed_plus_one_for_audio():
    visuals = (torch.zeros(1, 24, 1, 4, 4), torch.zeros(1, 24, 1, 4, 4))
    audios = (torch.zeros(1, 32, 2, 8),)

    augmented_visuals, augmented_audios = augment_condition_latents(
        visuals,
        audios,
        seed=456,
        visual_clean=0.5,
        audio_clean=0.5,
        device=torch.device("cpu"),
    )

    expected_visual = 0.5 * _cpu_noise((1, 24, 1, 4, 4), torch.Generator(device="cpu").manual_seed(456))
    expected_audio = 0.5 * _cpu_noise((1, 32, 2, 8), torch.Generator(device="cpu").manual_seed(457))
    assert torch.equal(augmented_visuals[0], expected_visual)
    assert torch.equal(augmented_visuals[1], expected_visual)
    assert torch.equal(augmented_audios[0], expected_audio)

    changed_visuals, changed_audios = augment_condition_latents(
        visuals,
        audios,
        seed=457,
        visual_clean=0.5,
        audio_clean=0.5,
        device=torch.device("cpu"),
    )
    assert not torch.equal(changed_visuals[0], augmented_visuals[0])
    assert not torch.equal(changed_audios[0], augmented_audios[0])


def test_condition_augmentation_does_not_change_target_noise_sequence():
    expected = initialize_target_latents(
        video_shape=(1, 24, 2, 4, 4),
        audio_shape=(1, 32, 2, 8),
        seed=789,
        device=torch.device("cpu"),
        video_dtype=torch.float32,
        audio_dtype=torch.float32,
    )
    augment_condition_latents(
        (torch.zeros(1, 24, 1, 4, 4),),
        (torch.zeros(1, 32, 2, 8),),
        seed=789,
        visual_clean=0.5,
        audio_clean=0.5,
        device=torch.device("cpu"),
    )
    actual = initialize_target_latents(
        video_shape=(1, 24, 2, 4, 4),
        audio_shape=(1, 32, 2, 8),
        seed=789,
        device=torch.device("cpu"),
        video_dtype=torch.float32,
        audio_dtype=torch.float32,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_joint_sampler_uses_native_dataward_predictions_and_each_sigma_delta():
    class Transformer:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                video=torch.full_like(kwargs["video_latents"], 2.0),
                audio=torch.full_like(kwargs["audio_latents"], 3.0),
            )

    transformer = Transformer()
    initial_video = torch.full((1, 24, 2, 4, 4), 5.0)
    initial_audio = torch.full((1, 32, 2, 8), 7.0)

    result = sample_joint_av(
        transformer,
        layout=_layout(),
        text_hidden_states=torch.zeros(1, 3, 12),
        text_token_tags=torch.tensor([[1, 0, 1]]),
        initial_video=initial_video,
        initial_audio=initial_audio,
        steps=2,
        video_shift=12.0,
        audio_shift=3.0,
    )

    assert len(transformer.calls) == 2
    assert transformer.calls[0]["model_t_video"].item() == pytest.approx(0.0)
    assert transformer.calls[0]["model_t_audio"].item() == pytest.approx(0.0)
    assert transformer.calls[1]["model_t_video"].item() == pytest.approx(1.0 / 13.0)
    assert transformer.calls[1]["model_t_audio"].item() == pytest.approx(0.25)
    torch.testing.assert_close(
        transformer.calls[1]["video_latents"],
        initial_video + (1.0 - 12.0 / 13.0) * 2.0,
    )
    torch.testing.assert_close(
        transformer.calls[1]["audio_latents"],
        initial_audio + (1.0 - 0.75) * 3.0,
    )
    torch.testing.assert_close(result.video, initial_video + 2.0)
    torch.testing.assert_close(result.audio, initial_audio + 3.0)


def test_joint_decode_trims_video_and_audio_to_one_planned_duration():
    class VideoVAE:
        def decode(self, latents):
            assert latents.shape == (1, 24, 2, 4, 4)
            return torch.linspace(-1.0, 1.0, 1 * 3 * 6 * 8 * 8).reshape(1, 3, 6, 8, 8)

    class AudioVAE:
        sample_rate = 32000

        def decode(self, latents):
            assert latents.shape == (1, 32, 2, 8)
            return torch.linspace(-1.0, 1.0, 2 * 8000).reshape(1, 2, 8000)

    decoded = decode_joint_av(
        VideoVAE(),
        AudioVAE(),
        SimpleNamespace(video=torch.zeros(1, 24, 2, 4, 4), audio=torch.zeros(1, 32, 2, 8)),
        frame_count=5,
    )

    assert decoded.video.shape == (5, 8, 8, 3)
    assert decoded.video.dtype == torch.uint8
    assert decoded.audio.shape == (2, 6667)
    assert decoded.audio.dtype == torch.float32
    assert decoded.fps == 24
    assert decoded.sample_rate == 32000


def test_joint_output_uses_a_replaceable_mux_boundary(tmp_path):
    captured = {}

    def muxer(video, audio, output_path, *, fps, sample_rate):
        captured.update(video=video, audio=audio, output_path=output_path, fps=fps, sample_rate=sample_rate)

    decoded = SimpleNamespace(
        video=torch.zeros(5, 8, 8, 3, dtype=torch.uint8),
        audio=torch.zeros(2, 6667),
        fps=24,
        sample_rate=32000,
    )
    output_path = tmp_path / "result.mp4"

    write_joint_av(decoded, output_path, muxer=muxer)

    assert captured == {
        "video": decoded.video,
        "audio": decoded.audio,
        "output_path": output_path,
        "fps": 24,
        "sample_rate": 32000,
    }


def _generation_args(tmp_path, *, task="t2va", **overrides):
    paths = {}
    for name in ("dit", "video_vae", "audio_vae", "text_encoder"):
        path = tmp_path / f"{name}.safetensors"
        path.touch()
        paths[name] = str(path)
    values = {
        **paths,
        "task": task,
        "prompt": "a test prompt",
        "text_cache": None,
        "processor": "Qwen/Qwen3-VL-32B-Instruct",
        "first_frame": None,
        "last_frame": None,
        "reference_jsonl": None,
        "reference_index": 0,
        "width": 64,
        "height": 64,
        "frame_count": 124,
        "allow_experimental_duration": False,
        "steps": 2,
        "seed": 1,
        "output": str(tmp_path / "output.mp4"),
        "blocks_to_swap": 0,
        "h3_shift_video": 12.0,
        "h3_shift_audio": 3.0,
        "h3_visual_cond_clean": 0.999,
        "h3_audio_cond_clean": 1.0,
        "lora_weight": None,
        "lora_multiplier": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("field", ("width", "height"))
def test_generation_validation_rejects_non_32_aligned_axes(tmp_path, field):
    with pytest.raises(ValueError, match="divisible by 32"):
        validate_generation_args(_generation_args(tmp_path, **{field: 63}))


def test_generation_validation_enforces_task_inputs_and_block_swap_range(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.touch()
    last.touch()
    validate_generation_args(_generation_args(tmp_path, task="fl2va", first_frame=str(first), last_frame=str(last)))

    with pytest.raises(ValueError, match="reference_jsonl"):
        validate_generation_args(_generation_args(tmp_path, task="ref2va"))
    with pytest.raises(ValueError, match="blocks_to_swap"):
        validate_generation_args(_generation_args(tmp_path, blocks_to_swap=49))


def test_cached_text_conditioning_validates_task_width_and_tags(tmp_path):
    path = tmp_path / "conditioning.safetensors"
    hidden = torch.zeros(3, 5120, dtype=torch.bfloat16)
    tags = torch.tensor([1, 0, 1], dtype=torch.int64)
    save_file(
        {
            "varlen_mmh3_hidden_states_bfloat16": hidden,
            "varlen_mmh3_token_tags_int64": tags,
        },
        str(path),
        metadata={
            "task": "t2va",
            "frame_count": "124",
            "presentation_fingerprint": "sha256:presentation",
            "hidden_state_convention": "index50-after-50-layers-pre-final-norm",
            "token_tag_algorithm": "expanded-vision-span-with-flanks-v1",
            "text_width": "5120",
            "max_text_rows": "32768",
        },
    )

    actual_hidden, actual_tags = load_cached_text_conditioning(
        path,
        task="t2va",
        frame_count=124,
        presentation_identity="sha256:presentation",
    )

    assert actual_hidden.shape == (1, 3, 5120)
    assert actual_hidden.dtype == torch.bfloat16
    assert torch.equal(actual_tags, tags)
    with pytest.raises(ValueError, match=r"task.*ref2va.*t2va"):
        load_cached_text_conditioning(path, task="ref2va")
    with pytest.raises(ValueError, match=r"frame count.*141.*124"):
        load_cached_text_conditioning(path, task="t2va", frame_count=141)
    with pytest.raises(ValueError, match="presentation fingerprint"):
        load_cached_text_conditioning(
            path,
            task="t2va",
            presentation_identity="sha256:different",
        )


def test_generation_text_cache_requires_an_identifiable_presentation(tmp_path):
    text_cache = tmp_path / "conditioning.safetensors"
    text_cache.touch()

    with pytest.raises(ValueError, match="T2VA requires --prompt"):
        validate_generation_args(
            _generation_args(
                tmp_path,
                text_cache=str(text_cache),
                text_encoder=None,
                prompt=None,
            )
        )
    validate_generation_args(_generation_args(tmp_path, text_cache=str(text_cache), text_encoder=None))
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.touch()
    last.touch()
    with pytest.raises(ValueError, match="FL2VA.*text_cache"):
        validate_generation_args(
            _generation_args(
                tmp_path,
                task="fl2va",
                text_cache=str(text_cache),
                text_encoder=None,
                first_frame=str(first),
                last_frame=str(last),
            )
        )


def test_generation_orchestrates_t2va_sampling_decode_and_mux_without_co_resident_vaes(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_generate_video as generate

    args = _generation_args(
        tmp_path,
        frame_count=5,
        allow_experimental_duration=True,
        output=str(tmp_path / "result.mp4"),
        device="cpu",
        attn_mode="torch",
        split_attn=False,
        use_pinned_memory_for_block_swap=False,
        include_patterns=None,
        exclude_patterns=None,
        disable_numpy_memmap=False,
        processor_revision=None,
    )
    events = []

    class Transformer:
        offloader = None

        def to(self, device):
            events.append(("transformer", str(device)))
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            assert value is False
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(
                video=torch.zeros_like(kwargs["video_latents"]),
                audio=torch.zeros_like(kwargs["audio_latents"]),
            )

    class VideoVAE:
        def decode(self, latents):
            events.append(("decode_video", tuple(latents.shape)))
            return torch.zeros(1, 3, 5, 4, 4)

    class AudioVAE:
        def decode(self, latents):
            events.append(("decode_audio", tuple(latents.shape)))
            return torch.zeros(1, 2, 6667)

    monkeypatch.setattr(
        generate,
        "_encode_text",
        lambda *unused: (torch.zeros(1, 3, 5120, dtype=torch.bfloat16), torch.ones(3, dtype=torch.int64)),
    )
    monkeypatch.setattr(generate, "load_h3_transformer", lambda *unused, **kwargs: Transformer())
    monkeypatch.setattr(
        generate,
        "load_video_vae",
        lambda *unused, **kwargs: events.append(("load_video_vae", str(kwargs["device"]), kwargs["dtype"])) or VideoVAE(),
    )
    monkeypatch.setattr(
        generate,
        "load_audio_vae",
        lambda *unused, **kwargs: events.append(("load_audio_vae", str(kwargs["device"]))) or AudioVAE(),
    )
    captured = {}
    monkeypatch.setattr(
        generate,
        "write_joint_av",
        lambda decoded, output: captured.update(decoded=decoded, output=output),
    )

    output = run_generation(args)

    assert output == Path(args.output)
    assert [event[0] for event in events] == [
        "transformer",
        "load_video_vae",
        "decode_video",
        "load_audio_vae",
        "decode_audio",
    ]
    assert next(event for event in events if event[0] == "load_video_vae")[2] is torch.float16
    assert captured["decoded"].video.shape == (5, 4, 4, 3)
    assert captured["decoded"].audio.shape == (2, 6667)
    assert captured["output"] == args.output
