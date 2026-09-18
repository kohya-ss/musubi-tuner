"""The generation request contract shared by minimax_h3_generate_video.py and the trainer's
training-time samples (minimax_h3/generation_inputs.py): one request type, one validator, one
prompt-line vocabulary, and parser groups with one set of defaults."""

from __future__ import annotations

import argparse

import pytest

from musubi_tuner.minimax_h3.generation_inputs import (
    H3GenerationRequest,
    build_generation_layout,
    request_from_args,
    request_overrides,
    validate_generation_request,
)
from musubi_tuner.minimax_h3.packing import H3VideoGeometry
from musubi_tuner.minimax_h3.sampling import DEFAULT_AUDIO_SHIFT, DEFAULT_VIDEO_SHIFT
from musubi_tuner.minimax_h3_generate_video import setup_parser as generation_parser
from musubi_tuner.minimax_h3_train_network import minimax_h3_setup_parser
from musubi_tuner.training.sampling_prompts import line_to_prompt_dict


def test_generation_cli_defaults_are_the_request_defaults():
    args = generation_parser().parse_args(["--task", "t2va", "--save_path", "out.mp4"])

    assert request_from_args(args) == H3GenerationRequest(task="t2va")

    # the house flags land on the request field names: --video_size is height then width
    args = generation_parser().parse_args(
        ["--task", "t2va", "--save_path", "out", "--video_size", "1344", "768", "--video_length", "39", "--infer_steps", "7"]
    )
    request = request_from_args(args)
    assert (request.height, request.width, request.frame_count, request.steps) == (1344, 768, 39, 7)
    assert not hasattr(args, "video_size")


def test_trainer_and_generation_share_the_sampler_and_text_encoder_options():
    train = vars(minimax_h3_setup_parser(argparse.ArgumentParser()).parse_args(["--task", "t2va"]))
    generate = vars(generation_parser().parse_args(["--task", "t2va", "--save_path", "out.mp4"]))

    for name in (
        "h3_shift_video",
        "h3_shift_audio",
        "h3_visual_cond_clean",
        "h3_audio_cond_clean",
        "video_vae",
        "audio_vae",
        "text_encoder",
        "nvfp4_scaled_mm",
        "text_encoder_blocks_to_swap",
        "text_encoder_attn_mode",
    ):
        assert train[name] == generate[name], name
    assert train["h3_shift_video"] == DEFAULT_VIDEO_SHIFT
    assert train["h3_shift_audio"] == DEFAULT_AUDIO_SHIFT


def test_prompt_line_vocabulary_maps_onto_the_request_fields():
    prompt_dict = line_to_prompt_dict(
        "a cat sings --w 768 --h 1344 --f 1 --d 42 --s 20 --fs 10.5 --fsa 2.5 --ofps 12 --skb 3"
        " --i first.png --ei last.png --ci c.png --of target_index=24 --rj refs.jsonl --ref face.png --o out.png"
    )

    assert prompt_dict.pop("output_name") == "out.png"
    assert request_overrides(prompt_dict) == {
        "prompt": "a cat sings",
        "width": 768,
        "height": 1344,
        "frame_count": 1,
        "seed": 42,
        "steps": 20,
        "h3_shift_video": 10.5,
        "h3_shift_audio": 2.5,
        "output_fps": 12,
        "stretch_keep_bands": 3,
        "first_frame": "first.png",
        "last_frame": "last.png",
        "condition_image": ["c.png"],
        "one_frame_inference": "target_index=24",
        "reference_jsonl": "refs.jsonl",
        "ref": ["face.png"],
    }
    # request field names pass through (prompt files written in the request vocabulary), the
    # caller's fields and unrelated keys do not, and values are coerced to the field types
    assert request_overrides({"first_frame": "a.png", "steps": "3", "task": "fl2va", "enum": 1, "subset": {}}) == {
        "first_frame": "a.png",
        "steps": 3,
    }
    assert request_overrides({"prompt": "line one\\nline two"}) == {"prompt": "line one\nline two"}
    for key in ("negative_prompt", "cfg_scale", "guidance_scale", "control_video_path"):
        with pytest.raises(ValueError, match=key):
            request_overrides({key: "x"})


def test_prompt_text_is_never_read_as_an_option(caplog):
    # "d 12 monkeys" used to set the seed: only the " --"-separated parts are options
    assert line_to_prompt_dict("d 12 monkeys --w 64") == {"prompt": "d 12 monkeys", "width": 64}
    with caplog.at_level("WARNING"):
        assert line_to_prompt_dict("a cat --x 1 --d 2") == {"prompt": "a cat", "seed": 2}
    assert any("--x 1" in record.getMessage() for record in caplog.records)


def test_request_validation_covers_the_canvas_timeline_and_task_inputs(tmp_path):
    validate_generation_request(H3GenerationRequest(task="t2va", prompt="p"))
    with pytest.raises(ValueError, match="divisible by 32"):
        validate_generation_request(H3GenerationRequest(task="t2va", prompt="p", width=100))
    with pytest.raises(ValueError, match=r"17\*n\+5"):
        validate_generation_request(H3GenerationRequest(task="t2va", prompt="p", frame_count=100))
    with pytest.raises(ValueError, match="released 5-15s"):
        validate_generation_request(H3GenerationRequest(task="t2va", prompt="p", frame_count=39))
    validate_generation_request(H3GenerationRequest(task="t2va", prompt="p", frame_count=39, allow_experimental_duration=True))
    with pytest.raises(ValueError, match="--infer_steps must be positive"):
        validate_generation_request(H3GenerationRequest(task="t2va", prompt="p", steps=0))
    with pytest.raises(ValueError, match="h3_shift_audio"):
        validate_generation_request(H3GenerationRequest(task="t2va", prompt="p", h3_shift_audio=0.0))
    with pytest.raises(ValueError, match="requires --prompt"):
        validate_generation_request(H3GenerationRequest(task="t2va"))
    with pytest.raises(ValueError, match="--task must be one of"):
        validate_generation_request(H3GenerationRequest(task="i2v", prompt="p"))

    first = tmp_path / "first.png"
    first.touch()
    validate_generation_request(H3GenerationRequest(task="fl2va", prompt="p", first_frame=str(first)))
    with pytest.raises(ValueError, match="does not exist"):
        validate_generation_request(H3GenerationRequest(task="fl2va", prompt="p", last_frame=str(tmp_path / "missing.png")))


def test_video_fl2va_layout_carries_the_given_anchor_roles():
    # a lone last frame keeps its end-of-video anchor (L2VA): the request's entries select the roles
    request = H3GenerationRequest(task="fl2va", prompt="p", last_frame="last.png", width=64, height=64, frame_count=39)

    layout = build_generation_layout(request, text_length=3, visual_geometries=(H3VideoGeometry(1, 4, 4),))

    assert [segment.role for segment in layout.segments] == ["text", "last", "target_audio", "target_video"]
    assert layout.target_video == H3VideoGeometry(12, 4, 4)
