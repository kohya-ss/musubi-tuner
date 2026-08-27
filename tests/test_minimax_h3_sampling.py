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
    create_sampling_generator,
    decode_joint_av,
    initialize_target_latents,
    sample_joint_av,
    write_joint_av,
)
from musubi_tuner.minimax_h3.generation_inputs import load_generation_record, parse_one_frame_options
from musubi_tuner.minimax_h3.packing import FRAME_RESCALE, H3TimeOverrides
from musubi_tuner.minimax_h3.sampling import write_image
from musubi_tuner.minimax_h3_generate_video import (
    _one_frame_time_overrides,
    load_cached_text_conditioning,
    validate_generation_args,
)

# the parser moved to generation_inputs so the trainer's sample prompts share it
_parse_one_frame_options = parse_one_frame_options


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


def test_joint_sampler_reports_the_per_step_clean_estimate():
    class Transformer:
        def __call__(self, **kwargs):
            return SimpleNamespace(
                video=torch.full_like(kwargs["video_latents"], 2.0),
                audio=torch.full_like(kwargs["audio_latents"], 3.0),
            )

    captured = []
    result = sample_joint_av(
        Transformer(),
        layout=_layout(),
        text_hidden_states=torch.zeros(1, 3, 12),
        text_token_tags=torch.tensor([[1, 0, 1]]),
        initial_video=torch.full((1, 24, 2, 4, 4), 5.0),
        initial_audio=torch.full((1, 32, 2, 8), 7.0),
        steps=2,
        video_shift=12.0,
        audio_shift=3.0,
        x0_callback=lambda index, video, audio: captured.append((index, video, audio)),
    )

    assert [index for index, _, _ in captured] == [0, 1]
    # under a constant velocity, x0_hat = x_t + sigma * v equals the trajectory endpoint at every step
    for _, video, audio in captured:
        torch.testing.assert_close(video, result.video)
        torch.testing.assert_close(audio, result.audio)


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
        "ref": None,
        "one_frame": None,
        "width": 64,
        "height": 64,
        "frame_count": 124,
        "output_fps": 24,
        "stretch_keep_bands": 0,
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
        "convrot_int8": False,
        "prune_adaln": False,
        "trajectory_dir": None,
        "trajectory_stride": 1,
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


def test_generation_validation_accepts_inline_refs_exclusively_with_reference_jsonl(tmp_path):
    validate_generation_args(_generation_args(tmp_path, task="ref2va", ref=["face.png"]))

    jsonl = tmp_path / "refs.jsonl"
    jsonl.touch()
    with pytest.raises(ValueError, match="exactly one of"):
        validate_generation_args(_generation_args(tmp_path, task="ref2va", ref=["face.png"], reference_jsonl=str(jsonl)))
    with pytest.raises(ValueError, match="requires --prompt"):
        validate_generation_args(_generation_args(tmp_path, task="ref2va", ref=["face.png"], prompt=None))
    with pytest.raises(ValueError, match="reference_index"):
        validate_generation_args(_generation_args(tmp_path, task="ref2va", ref=["face.png"], reference_index=1))
    with pytest.raises(ValueError, match="T2VA does not accept"):
        validate_generation_args(_generation_args(tmp_path, task="t2va", ref=["face.png"]))
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.touch()
    last.touch()
    with pytest.raises(ValueError, match="FL2VA does not accept"):
        validate_generation_args(
            _generation_args(tmp_path, task="fl2va", first_frame=str(first), last_frame=str(last), ref=["face.png"])
        )


def test_load_generation_record_builds_inline_ref_records_without_a_jsonl(tmp_path):
    refs_directory = tmp_path / "refs"
    refs_directory.mkdir()
    face = refs_directory / "face.png"
    style = refs_directory / "style.webp"
    face.touch()
    style.touch()

    record = load_generation_record(
        SimpleNamespace(
            task="ref2va",
            prompt="a cat sings",
            ref=["refs/face.png", "refs/style.webp"],
            ref_base_directory=str(tmp_path),
            reference_jsonl=None,
            reference_index=0,
        )
    )

    assert record.caption == "a cat sings"
    assert [reference.type for reference in record.references] == ["image", "image"]
    assert [reference.path for reference in record.references] == [face.resolve(), style.resolve()]


def test_parse_one_frame_options_accepts_indices_and_rejects_malformed_specs():
    assert _parse_one_frame_options("target_index=24,control_index=0;240") == (24, (0, 240))
    assert _parse_one_frame_options("control_index=7") == (0, (7,))
    assert _parse_one_frame_options("target_index=3") == (3, None)

    for spec, message in (
        ("target_index", "key=value"),
        ("speed=2", "unknown option"),
        ("target_index=1,target_index=2", "duplicate option"),
        ("target_index=-1", "nonnegative"),
        ("control_index=a", "must be an integer"),
    ):
        with pytest.raises(ValueError, match=message):
            _parse_one_frame_options(spec)


def test_one_frame_time_overrides_map_pixel_frame_indices_to_rotary_units():
    args = SimpleNamespace(frame_count=1, one_frame="target_index=24,control_index=0;240")
    assert _one_frame_time_overrides(args) == H3TimeOverrides(
        condition_times=(0.0, FRAME_RESCALE * 240),
        target_time=FRAME_RESCALE * 24,
    )

    defaults = _one_frame_time_overrides(SimpleNamespace(frame_count=1, one_frame=None))
    assert defaults == H3TimeOverrides(condition_times=(), target_time=0.0)
    assert _one_frame_time_overrides(SimpleNamespace(frame_count=124, one_frame=None)) is None


def test_generation_validation_gates_the_one_frame_mode(tmp_path):
    png = str(tmp_path / "output.png")
    validate_generation_args(_generation_args(tmp_path, frame_count=1, output=png))
    validate_generation_args(_generation_args(tmp_path, frame_count=1, output=png, one_frame="target_index=240"))

    with pytest.raises(ValueError, match="must use .png"):
        validate_generation_args(_generation_args(tmp_path, frame_count=1))
    with pytest.raises(ValueError, match="require --frame_count 1"):
        validate_generation_args(_generation_args(tmp_path, one_frame="target_index=1"))
    with pytest.raises(ValueError, match="control_index applies only to FL2VA"):
        validate_generation_args(_generation_args(tmp_path, frame_count=1, output=png, one_frame="control_index=0"))

    first = tmp_path / "first.png"
    first.touch()
    validate_generation_args(
        _generation_args(
            tmp_path,
            task="fl2va",
            frame_count=1,
            output=png,
            first_frame=str(first),
            one_frame="target_index=24,control_index=0",
        )
    )
    with pytest.raises(ValueError, match="one entry per provided frame"):
        validate_generation_args(_generation_args(tmp_path, task="fl2va", frame_count=1, output=png, first_frame=str(first)))
    with pytest.raises(ValueError, match="one entry per provided frame"):
        validate_generation_args(
            _generation_args(
                tmp_path,
                task="fl2va",
                frame_count=1,
                output=png,
                first_frame=str(first),
                one_frame="control_index=0;240",
            )
        )
    with pytest.raises(ValueError, match="requires --first_frame and/or --last_frame"):
        validate_generation_args(_generation_args(tmp_path, task="fl2va", frame_count=1, output=png, one_frame="control_index=0"))


def test_mux_encodes_above_the_1mbps_pyav_default(tmp_path):
    # regression: without an explicit CRF, PyAV's libx264 default is ~1 Mbps ABR, which crushes
    # fine detail in evaluation outputs. Noise frames at CRF 16 must blow far past that cap.
    from musubi_tuner.minimax_h3.sampling import mux_audio_video

    generator = torch.Generator().manual_seed(0)
    video = torch.randint(0, 256, (12, 256, 256, 3), generator=generator, dtype=torch.uint8)
    audio = torch.zeros(2, 16000)
    output = tmp_path / "noise.mp4"

    mux_audio_video(video, audio, output, fps=24, sample_rate=32000)

    bits_per_second = output.stat().st_size * 8 / (12 / 24)
    assert bits_per_second > 3_000_000


def test_write_image_saves_a_single_uint8_frame(tmp_path):
    frame = torch.zeros(4, 6, 3, dtype=torch.uint8)
    frame[:, :, 1] = 200
    output = tmp_path / "sub" / "image.png"

    write_image(frame, output)

    from PIL import Image

    with Image.open(output) as image:
        assert image.size == (6, 4)
        assert image.getpixel((0, 0)) == (0, 200, 0)

    with pytest.raises(ValueError, match=r"uint8 \[H,W,3\]"):
        write_image(frame.float(), tmp_path / "bad.png")


def test_cached_text_conditioning_validates_task_format_and_fingerprint(tmp_path):
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
    assert captured["output"] == Path(args.output)


def test_generation_trajectory_dump_writes_sigma_schedule_and_per_step_videos(tmp_path, monkeypatch):
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
        trajectory_dir=str(tmp_path / "trajectory"),
    )
    monkeypatch.setattr(generate, "resolve_safetensors_files", lambda path: [path])
    monkeypatch.setattr(generate, "has_comfy_quant_tensors", lambda files, **kwargs: False)

    class Transformer:
        offloader = None

        def to(self, device):
            return self

        def eval(self):
            return self

        def requires_grad_(self, value):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(
                video=torch.zeros_like(kwargs["video_latents"]),
                audio=torch.zeros_like(kwargs["audio_latents"]),
            )

    decode_calls = []

    class VideoVAE:
        def decode(self, latents):
            decode_calls.append(tuple(latents.shape))
            return torch.zeros(1, 3, 5, 32, 32)

    class AudioVAE:
        def decode(self, latents):
            return torch.zeros(1, 2, 6667)

    monkeypatch.setattr(
        generate,
        "_encode_text",
        lambda *unused: (torch.zeros(1, 3, 5120, dtype=torch.bfloat16), torch.ones(3, dtype=torch.int64)),
    )
    monkeypatch.setattr(generate, "load_h3_transformer", lambda *unused, **kwargs: Transformer())
    monkeypatch.setattr(generate, "load_video_vae", lambda *unused, **kwargs: VideoVAE())
    monkeypatch.setattr(generate, "load_audio_vae", lambda *unused, **kwargs: AudioVAE())
    monkeypatch.setattr(generate, "write_joint_av", lambda decoded, output: None)

    generate.run_generation(args)

    trajectory_dir = tmp_path / "trajectory"
    schedule_lines = (trajectory_dir / "sigma_schedule.csv").read_text(encoding="utf-8").splitlines()
    assert schedule_lines[0] == "step,base_sigma,sigma_video,sigma_audio"
    assert schedule_lines[1] == "0,1.000000,1.000000,1.000000"
    assert schedule_lines[2] == "1,0.500000,0.923077,0.750000"
    assert len(schedule_lines) == 1 + args.steps
    step_files = sorted(path.name for path in trajectory_dir.glob("*.mp4"))
    assert step_files == ["step000_base1.0000_sigv1.0000.mp4", "step001_base0.5000_sigv0.9231.mp4"]
    # the final output decode plus one decode per dumped step
    assert len(decode_calls) == 1 + args.steps
