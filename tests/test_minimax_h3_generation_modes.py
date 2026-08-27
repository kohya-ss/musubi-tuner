from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import musubi_tuner.minimax_h3_generate_video as generate
from musubi_tuner.minimax_h3_generate_video import (
    apply_overrides,
    parse_prompt_line,
    validate_prompt_args,
    validate_session_args,
)


def _session_args(tmp_path, *, task="t2va", **overrides):
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
        "output_name": None,
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
        "interactive": False,
        "from_file": None,
        "latent_path": None,
        "bell": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_parse_prompt_line_maps_inline_options_and_collects_refs():
    overrides = parse_prompt_line(
        "a cat sings --w 768 --h 1344 --f 1 --d 42 --s 20 --fs 10.5 --fsa 2.5"
        " --i first.png --ei last.png --of target_index=24,control_index=0;12 --o cat.png"
    )
    assert overrides == {
        "prompt": "a cat sings",
        "width": 768,
        "height": 1344,
        "frame_count": 1,
        "seed": 42,
        "steps": 20,
        "h3_shift_video": 10.5,
        "h3_shift_audio": 2.5,
        "first_frame": "first.png",
        "last_frame": "last.png",
        "one_frame": "target_index=24,control_index=0;12",
        "output_name": "cat.png",
    }

    refs = parse_prompt_line("a duet --ref face.png --ref voice.mp4;audio=voice.wav")
    assert refs == {"prompt": "a duet", "ref": ["face.png", "voice.mp4;audio=voice.wav"]}

    # a line starting with "--" carries only options, so the CLI --prompt stays in effect
    # (and an invalid value still reaches validation instead of becoming the prompt text)
    assert parse_prompt_line("--w 0") == {"width": 0}
    assert parse_prompt_line("--d 43 --s 20") == {"seed": 43, "steps": 20}

    # the literal "\n" becomes a newline, so the multi-line official prompt format fits on one line
    assert parse_prompt_line("line one\\nline two --d 1") == {"prompt": "line one\nline two", "seed": 1}

    with pytest.raises(ValueError, match="unknown option --x"):
        parse_prompt_line("a cat --x 1")


def test_apply_overrides_keeps_the_base_args_untouched_and_resets_output_name():
    base = SimpleNamespace(prompt="base", width=64, ref=["session.png"], output_name="stale.png")
    prompt_args = apply_overrides(base, {"prompt": "override", "ref": ["line.png"]})
    assert prompt_args.prompt == "override"
    assert prompt_args.ref == ["line.png"]
    assert prompt_args.output_name is None
    assert prompt_args.width == 64
    assert base.prompt == "base" and base.ref == ["session.png"] and base.output_name == "stale.png"


def test_session_validation_enforces_mode_exclusivity_and_multi_prompt_restrictions(tmp_path):
    prompts = tmp_path / "prompts.txt"
    prompts.touch()
    latents = tmp_path / "latents.safetensors"
    latents.touch()

    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_session_args(_session_args(tmp_path, interactive=True, from_file=str(prompts)))
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_session_args(_session_args(tmp_path, from_file=str(prompts), latent_path=[str(latents)]))

    cache = tmp_path / "text.safetensors"
    cache.touch()
    with pytest.raises(ValueError, match="do not accept --text_cache"):
        validate_session_args(_session_args(tmp_path, interactive=True, text_cache=str(cache)))
    with pytest.raises(ValueError, match="do not accept --trajectory_dir"):
        validate_session_args(_session_args(tmp_path, from_file=str(prompts), trajectory_dir=str(tmp_path)))
    with pytest.raises(ValueError, match="requires --text_encoder"):
        validate_session_args(_session_args(tmp_path, interactive=True, text_encoder=None))
    validate_session_args(_session_args(tmp_path, interactive=True))
    validate_session_args(_session_args(tmp_path, from_file=str(prompts)))

    # decode-only mode needs just the latents and the video VAE
    validate_session_args(_session_args(tmp_path, latent_path=[str(latents)], task=None, dit=None, text_encoder=None))
    with pytest.raises(ValueError, match="requires --video_vae"):
        validate_session_args(_session_args(tmp_path, latent_path=[str(latents)], video_vae=None))

    with pytest.raises(ValueError, match="requires --task"):
        validate_session_args(_session_args(tmp_path, task=None))


def test_prompt_validation_checks_output_name_suffixes_in_directory_modes(tmp_path):
    args = _session_args(tmp_path, output=str(tmp_path))
    validate_prompt_args(args, directory_output=True)

    args.output_name = "clip.mp4"
    validate_prompt_args(args, directory_output=True)
    args.output_name = "clip.txt"
    with pytest.raises(ValueError, match="must use .mp4"):
        validate_prompt_args(args, directory_output=True)

    image_args = _session_args(tmp_path, frame_count=1, output=str(tmp_path))
    image_args.output_name = "image.png"
    validate_prompt_args(image_args, directory_output=True)
    image_args.output_name = "image.mp4"
    with pytest.raises(ValueError, match="must use .png"):
        validate_prompt_args(image_args, directory_output=True)


def test_one_frame_fl2va_control_index_error_reports_the_counts(tmp_path):
    first = tmp_path / "first.png"
    first.touch()
    args = _session_args(
        tmp_path,
        task="fl2va",
        frame_count=1,
        first_frame=str(first),
        one_frame="target_index=6,control_index=0;123",
        output=str(tmp_path / "out.png"),
    )

    with pytest.raises(ValueError, match=r"got 2 control_index entries for 1 condition frames \(first_frame\)"):
        validate_prompt_args(args)


def test_from_file_records_invalid_lines_without_aborting_the_batch(tmp_path, monkeypatch, caplog):
    counters = {"text": 0, "transformer": 0, "video_vae": 0, "audio_vae": 0}
    _stub_generation_models(monkeypatch, counters)
    written = []
    monkeypatch.setattr(generate, "write_joint_av", lambda decoded, output: written.append(Path(output)))

    args = _batch_args(tmp_path, ["# comment", "", "a cat sings --d 11", "a dog barks --w 0"])
    with caplog.at_level(logging.ERROR, logger=generate.logger.name):
        generate.process_from_file(args, torch.device("cpu"))

    # the bad width line is file line 4; it is recorded as a failed item while the valid prompt still runs
    assert len(written) == 1
    assert written[0].name.endswith("_11.mp4")
    messages = [record.getMessage() for record in caplog.records]
    assert any("line 4 validation" in message and "divisible by 32" in message for message in messages)


def test_latent_file_roundtrip_preserves_tensors_and_rejects_missing_audio(tmp_path):
    args = _session_args(tmp_path, frame_count=39, allow_experimental_duration=True)
    video = torch.randn(1, 24, 3, 4, 4)
    audio = torch.randn(1, 32, 2, 65)

    path = generate._save_latent_file(tmp_path / "video_latent.safetensors", video, audio, args, 7)
    loaded_video, loaded_audio, frame_count, metadata = generate._load_latent_file(path)
    assert torch.equal(loaded_video, video)
    assert torch.equal(loaded_audio, audio)
    assert frame_count == 39
    assert metadata["seeds"] == "7"
    assert metadata["prompt"] == "a test prompt"

    image_args = _session_args(tmp_path, frame_count=1)
    image_path = generate._save_latent_file(tmp_path / "image_latent.safetensors", video, None, image_args, 8)
    _, loaded_audio, frame_count, _ = generate._load_latent_file(image_path)
    assert loaded_audio is None
    assert frame_count == 1

    broken = generate._save_latent_file(tmp_path / "broken_latent.safetensors", video, None, args, 9)
    with pytest.raises(ValueError, match="missing latent_audio"):
        generate._load_latent_file(broken)


class _StubTransformer:
    offloader = None
    blocks: list = []

    def to(self, device):
        return self

    def prepare_block_swap_before_forward(self):
        return None

    def eval(self):
        return self

    def requires_grad_(self, value):
        return self

    def __call__(self, **kwargs):
        return SimpleNamespace(
            video=torch.zeros_like(kwargs["video_latents"]),
            audio=torch.zeros_like(kwargs["audio_latents"]),
        )


class _StubVAE:
    def __init__(self, decoded):
        self._decoded = decoded

    def to(self, device):
        return self

    def decode(self, latents):
        return self._decoded()


def _stub_generation_models(monkeypatch, counters):
    monkeypatch.setattr(generate, "resolve_safetensors_files", lambda path: [path])
    monkeypatch.setattr(generate, "has_comfy_quant_tensors", lambda files, **kwargs: False)
    monkeypatch.setattr(
        generate,
        "_encode_text",
        lambda *unused: (
            counters.__setitem__("text", counters["text"] + 1),
            (torch.zeros(1, 3, 5120, dtype=torch.bfloat16), torch.ones(3, dtype=torch.int64)),
        )[1],
    )
    monkeypatch.setattr(
        generate,
        "load_h3_transformer",
        lambda *unused, **kwargs: counters.__setitem__("transformer", counters["transformer"] + 1) or _StubTransformer(),
    )
    monkeypatch.setattr(
        generate,
        "load_video_vae",
        lambda *unused, **kwargs: (
            counters.__setitem__("video_vae", counters["video_vae"] + 1) or _StubVAE(lambda: torch.zeros(1, 3, 5, 4, 4))
        ),
    )
    monkeypatch.setattr(
        generate,
        "load_audio_vae",
        lambda *unused, **kwargs: (
            counters.__setitem__("audio_vae", counters["audio_vae"] + 1) or _StubVAE(lambda: torch.zeros(1, 2, 6667))
        ),
    )


def _batch_args(tmp_path, prompt_lines):
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(prompt_lines), encoding="utf-8")
    output_dir = tmp_path / "outputs"
    return _session_args(
        tmp_path,
        frame_count=5,
        allow_experimental_duration=True,
        output=str(output_dir),
        from_file=str(prompts),
        device="cpu",
        attn_mode="torch",
        split_attn=False,
        use_pinned_memory_for_block_swap=False,
        include_patterns=None,
        exclude_patterns=None,
        disable_numpy_memmap=False,
        seed=3,
    )


def test_from_file_batch_loads_each_model_family_once_and_removes_latents_on_success(tmp_path, monkeypatch):
    counters = {"text": 0, "transformer": 0, "video_vae": 0, "audio_vae": 0}
    _stub_generation_models(monkeypatch, counters)
    written = []
    monkeypatch.setattr(generate, "write_joint_av", lambda decoded, output: written.append(Path(output)))

    args = _batch_args(tmp_path, ["# comment", "", "a cat sings --d 11", "a dog barks --d 22 --s 3"])
    generate.process_from_file(args, torch.device("cpu"))

    assert counters == {"text": 2, "transformer": 1, "video_vae": 1, "audio_vae": 1}
    assert len(written) == 2
    assert sorted(path.name.split("_")[-1] for path in written) == ["11.mp4", "22.mp4"]
    # intermediate latents were written during sampling and removed after their decode
    assert list((tmp_path / "outputs").glob("*_latent.safetensors")) == []


def test_from_file_batch_keeps_the_latents_of_a_failed_decode(tmp_path, monkeypatch):
    counters = {"text": 0, "transformer": 0, "video_vae": 0, "audio_vae": 0}
    _stub_generation_models(monkeypatch, counters)

    def failing_mux(decoded, output):
        raise RuntimeError("mux exploded")

    monkeypatch.setattr(generate, "write_joint_av", failing_mux)

    args = _batch_args(tmp_path, ["a cat sings --d 11"])
    generate.process_from_file(args, torch.device("cpu"))

    kept = list((tmp_path / "outputs").glob("*_latent.safetensors"))
    assert len(kept) == 1
    _, audio_latents, frame_count, metadata = generate._load_latent_file(kept[0])
    assert audio_latents is not None
    assert frame_count == 5
    assert metadata["seeds"] == "11"


def test_compile_wraps_the_dit_once_with_training_parity_exclusions(tmp_path, monkeypatch):
    counters = {"text": 0, "transformer": 0, "video_vae": 0, "audio_vae": 0}
    _stub_generation_models(monkeypatch, counters)
    monkeypatch.setattr(generate, "write_joint_av", lambda decoded, output: None)
    compiled = []
    monkeypatch.setattr(
        generate,
        "compile_transformer",
        lambda args, transformer, target_blocks, disable_linear: compiled.append(disable_linear) or transformer,
    )

    args = _session_args(
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
    args.compile = True
    generate.run_generation(args, torch.device("cpu"))

    # blocks_to_swap=0 and a stub without is_convrot_int8 keep the Linears compiled
    assert compiled == [False]


def test_latent_decode_mode_loads_only_the_vaes(tmp_path, monkeypatch):
    counters = {"text": 0, "transformer": 0, "video_vae": 0, "audio_vae": 0}
    _stub_generation_models(monkeypatch, counters)
    written = []
    monkeypatch.setattr(generate, "write_joint_av", lambda decoded, output: written.append(Path(output)))
    monkeypatch.setattr(generate, "write_image", lambda frame, output: written.append(Path(output)))

    video_args = _session_args(tmp_path, frame_count=5, allow_experimental_duration=True)
    video_file = generate._save_latent_file(
        tmp_path / "video_latent.safetensors", torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 9), video_args, 5
    )
    image_args = _session_args(tmp_path, frame_count=1)
    image_file = generate._save_latent_file(tmp_path / "image_latent.safetensors", torch.randn(1, 24, 1, 4, 4), None, image_args, 6)

    args = _session_args(
        tmp_path,
        output=str(tmp_path / "decoded"),
        latent_path=[str(video_file), str(image_file)],
        disable_numpy_memmap=False,
    )
    generate.process_latent_decode(args, torch.device("cpu"))

    assert counters["text"] == 0 and counters["transformer"] == 0
    assert counters["video_vae"] == 1 and counters["audio_vae"] == 1
    assert [path.suffix for path in written] == [".mp4", ".png"]
    assert all(path.parent == tmp_path / "decoded" for path in written)
