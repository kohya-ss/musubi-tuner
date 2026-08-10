from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3.generation_inputs import decode_generation_visuals
from musubi_tuner.minimax_h3.media import H3Record
from musubi_tuner.minimax_h3.sampling import (
    augment_condition_latents,
    build_shifted_schedule,
    create_sampling_generator,
    decode_joint_av,
    initialize_target_latents,
    sample_joint_av,
    write_joint_av,
)
from musubi_tuner.minimax_h3_generate_video import (
    load_cached_text_conditioning,
    run_generation,
    save_selected_frame,
    setup_parser,
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
        generator=create_sampling_generator(123),
        device=torch.device("cpu"),
        video_dtype=torch.float16,
        audio_dtype=torch.float32,
    )
    generator = torch.Generator(device="cpu").manual_seed(123)
    expected_video = _cpu_noise((1, 24, 2, 4, 4), generator).to(torch.float16)
    expected_audio = _cpu_noise((1, 32, 2, 8), generator)

    assert torch.equal(video, expected_video)
    assert torch.equal(audio, expected_audio)


def test_condition_augmentation_draws_visuals_then_audio_from_the_same_request_stream():
    visuals = (torch.zeros(1, 24, 1, 4, 4), torch.zeros(1, 24, 1, 4, 4))
    audios = (torch.zeros(1, 32, 2, 8),)

    augmented_visuals, augmented_audios = augment_condition_latents(
        visuals,
        audios,
        generator=create_sampling_generator(456),
        visual_clean=0.5,
        audio_clean=0.5,
        device=torch.device("cpu"),
    )

    # sequential draws from the one request stream: each condition tensor gets its own noise, and
    # the conditions are zeros here, so clean*x + (1-clean)*eps collapses to the scaled draw
    expected = torch.Generator(device="cpu").manual_seed(456)
    assert torch.equal(augmented_visuals[0], 0.5 * _cpu_noise((1, 24, 1, 4, 4), expected))
    assert torch.equal(augmented_visuals[1], 0.5 * _cpu_noise((1, 24, 1, 4, 4), expected))
    assert torch.equal(augmented_audios[0], 0.5 * _cpu_noise((1, 32, 2, 8), expected))
    assert not torch.equal(augmented_visuals[0], augmented_visuals[1])


def test_condition_noise_does_not_alias_across_consecutive_request_seeds():
    # per-role seed offsets (visuals from seed, audio from seed + 1) made one request's audio noise
    # the next request's visual noise; the shared per-request stream removes that alias
    def augmented(seed: int):
        return augment_condition_latents(
            (torch.zeros(1, 32, 2, 8),),
            (torch.zeros(1, 32, 2, 8),),
            generator=create_sampling_generator(seed),
            visual_clean=0.5,
            audio_clean=0.5,
            device=torch.device("cpu"),
        )

    visuals, audios = augmented(456)
    next_visuals, next_audios = augmented(457)

    assert not torch.equal(audios[0], next_visuals[0])
    assert not torch.equal(visuals[0], next_visuals[0])
    assert not torch.equal(audios[0], next_audios[0])


def test_one_request_generator_feeds_target_noise_then_condition_noise_in_call_order():
    def request(seed: int):
        generator = create_sampling_generator(seed)
        video, audio = initialize_target_latents(
            video_shape=(1, 24, 2, 4, 4),
            audio_shape=(1, 32, 2, 8),
            generator=generator,
            device=torch.device("cpu"),
            video_dtype=torch.float32,
            audio_dtype=torch.float32,
        )
        conditions = augment_condition_latents(
            (torch.zeros(1, 24, 1, 4, 4),),
            (torch.zeros(1, 32, 2, 8),),
            generator=generator,
            visual_clean=0.5,
            audio_clean=0.5,
            device=torch.device("cpu"),
        )
        return video, audio, conditions

    expected = torch.Generator(device="cpu").manual_seed(789)
    video, audio, (visuals, audios) = request(789)

    assert torch.equal(video, _cpu_noise((1, 24, 2, 4, 4), expected))
    assert torch.equal(audio, _cpu_noise((1, 32, 2, 8), expected))
    assert torch.equal(visuals[0], 0.5 * _cpu_noise((1, 24, 1, 4, 4), expected))
    assert torch.equal(audios[0], 0.5 * _cpu_noise((1, 32, 2, 8), expected))
    # the same seed replays the whole request, and a different one changes every tensor
    replayed_video, replayed_audio, (replayed_visuals, _) = request(789)
    other_video, _, (other_visuals, _) = request(790)
    assert torch.equal(video, replayed_video)
    assert torch.equal(audio, replayed_audio)
    assert torch.equal(visuals[0], replayed_visuals[0])
    assert not torch.equal(video, other_video)
    assert not torch.equal(visuals[0], other_visuals[0])


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
        "h3_image_mode": "none",
        "h3_select_frame": 0,
        "lora_weight": None,
        "lora_multiplier": None,
        "convrot_int8": False,
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


def test_generation_validation_normalizes_first_image_mode(tmp_path):
    first = tmp_path / "first.png"
    first.touch()
    args = _generation_args(
        tmp_path,
        task="fl2va",
        first_frame=str(first),
        frame_count=None,
        output=str(tmp_path / "result.png"),
        h3_image_mode="first",
    )

    validate_generation_args(args)

    assert args.frame_count == 5
    assert args.last_frame == str(first)
    assert args.allow_experimental_duration is True


def test_image_output_does_not_require_unused_audio_vae_checkpoint(tmp_path):
    first = tmp_path / "first.png"
    first.touch()
    args = _generation_args(
        tmp_path,
        task="fl2va",
        first_frame=str(first),
        frame_count=None,
        output=str(tmp_path / "result.png"),
        h3_image_mode="first",
        audio_vae=None,
    )

    validate_generation_args(args)


def test_generation_validation_enforces_image_mode_contract(tmp_path):
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.touch()
    last.touch()

    with pytest.raises(ValueError, match="does not accept --last_frame"):
        validate_generation_args(
            _generation_args(
                tmp_path,
                task="fl2va",
                first_frame=str(first),
                last_frame=str(last),
                output=str(tmp_path / "result.png"),
                h3_image_mode="first",
            )
        )
    with pytest.raises(ValueError, match="last_frame"):
        validate_generation_args(
            _generation_args(
                tmp_path,
                task="fl2va",
                first_frame=str(first),
                last_frame=None,
                output=str(tmp_path / "result.png"),
                h3_image_mode="first_last",
            )
        )
    with pytest.raises(ValueError, match=r"\.mp4"):
        validate_generation_args(_generation_args(tmp_path, output=str(tmp_path / "result.png")))


def test_save_selected_frame_clamps_warns_and_writes_image(tmp_path, caplog):
    video = torch.zeros(2, 4, 4, 3, dtype=torch.uint8)
    video[1, :, :, 0] = 255
    output = tmp_path / "frame.png"

    with caplog.at_level(logging.WARNING):
        save_selected_frame(video, output, 99)

    assert output.exists()
    assert "clamping to 1" in caplog.text


def test_generation_parser_uses_same_text_visual_pixel_cap_as_cache_cli():
    args = setup_parser().parse_args(
        [
            "--task",
            "t2va",
            "--dit",
            "dit.safetensors",
            "--video_vae",
            "video.safetensors",
            "--audio_vae",
            "audio.safetensors",
            "--output",
            "output.mp4",
        ]
    )

    assert args.h3_text_visual_max_pixels == 1024 * 1024


def test_image_generation_caps_text_visual_without_changing_vae_geometry(tmp_path):
    from PIL import Image

    source = tmp_path / "source.png"
    Image.new("RGB", (3000, 2000)).save(source)
    args = SimpleNamespace(
        task="fl2va",
        first_frame=str(source),
        last_frame=str(source),
        width=64,
        height=96,
        h3_text_visual_max_pixels=1024 * 1024,
    )
    record = H3Record(Path("."), "caption", (), 0)

    raw_visuals, text_visuals = decode_generation_visuals(args, record, decoder=None)

    assert raw_visuals["first"].shape == (1, 96, 64, 3)
    assert text_visuals["first"].frames.shape == (1, 832, 1248, 3)


def test_cached_text_conditioning_validates_task_width_and_tags(tmp_path):
    path = tmp_path / "conditioning.safetensors"
    hidden = torch.zeros(3, 5120, dtype=torch.bfloat16)
    tags = torch.tensor([1, 0, 1], dtype=torch.int64)
    tensors = {
        "varlen_mmh3_hidden_states_bfloat16": hidden,
        "varlen_mmh3_token_tags_int64": tags,
    }
    save_file(
        tensors,
        str(path),
        metadata={
            "task": "t2va",
            "cache_format": "minimax-h3-text-v2",
            "presentation_fingerprint": "sha256:presentation",
        },
    )

    actual_hidden, actual_tags = load_cached_text_conditioning(
        path,
        task="t2va",
        presentation_identity="sha256:presentation",
    )

    assert actual_hidden.shape == (1, 3, 5120)
    assert actual_hidden.dtype == torch.bfloat16
    assert torch.equal(actual_tags, tags)
    with pytest.raises(ValueError, match=r"task.*ref2va.*t2va"):
        load_cached_text_conditioning(path, task="ref2va")
    with pytest.raises(ValueError, match="presentation fingerprint"):
        load_cached_text_conditioning(
            path,
            task="t2va",
            presentation_identity="sha256:different",
        )

    stale = tmp_path / "stale.safetensors"
    save_file(
        tensors,
        str(stale),
        metadata={"task": "t2va", "presentation_fingerprint": "sha256:presentation"},
    )
    with pytest.raises(ValueError, match="text cache format"):
        load_cached_text_conditioning(stale, task="t2va", presentation_identity="sha256:presentation")


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
    )
    # the pre-quantization probe reads the DiT file headers; the stub DiT here is not
    # a real safetensors file, so report an ordinary (non-pre-quantized) checkpoint
    monkeypatch.setattr(generate, "resolve_safetensors_files", lambda path: [path])
    monkeypatch.setattr(generate, "has_comfy_quant_tensors", lambda files, **kwargs: False)
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

    output = generate.run_generation(args)

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


def test_generation_orchestrates_image_output_without_audio_decode(tmp_path, monkeypatch):
    import musubi_tuner.minimax_h3_generate_video as generate

    first = tmp_path / "first.png"
    first.touch()
    args = _generation_args(
        tmp_path,
        task="fl2va",
        first_frame=str(first),
        frame_count=None,
        output=str(tmp_path / "result.png"),
        h3_image_mode="first",
        device="cpu",
        attn_mode="torch",
        split_attn=False,
        use_pinned_memory_for_block_swap=False,
        include_patterns=None,
        exclude_patterns=None,
        disable_numpy_memmap=False,
        processor_revision=None,
    )
    monkeypatch.setattr(generate, "resolve_safetensors_files", lambda path: [path])
    monkeypatch.setattr(generate, "has_comfy_quant_tensors", lambda files, **kwargs: False)
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
        vae_ratio = 16

        def decode(self, latents):
            events.append(("decode_video", tuple(latents.shape)))
            return torch.zeros(1, 3, 5, 4, 4)

    monkeypatch.setattr(
        generate,
        "decode_generation_visuals",
        lambda *unused: ({"first": torch.zeros(1, 64, 64, 3), "last": torch.zeros(1, 64, 64, 3)}, {}),
    )
    monkeypatch.setattr(
        generate,
        "encode_visual_conditions",
        lambda *unused: (
            (torch.zeros(1, 24, 1, 4, 4), torch.zeros(1, 24, 1, 4, 4)),
            (H3VideoGeometry(1, 4, 4), H3VideoGeometry(1, 4, 4)),
            {},
        ),
    )
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
        generate, "load_audio_vae", lambda *unused, **kwargs: pytest.fail("audio VAE should not decode image output")
    )
    captured = {}
    monkeypatch.setattr(
        generate,
        "save_selected_frame",
        lambda decoded_video, output, frame: captured.update(video=decoded_video, output=output, frame=frame),
    )

    output = run_generation(args)

    assert output == Path(args.output)
    assert [event[0] for event in events] == ["load_video_vae", "transformer", "load_video_vae", "decode_video"]
    assert captured["video"].shape == (5, 4, 4, 3)
    assert captured["output"] == args.output
    assert captured["frame"] == 0
