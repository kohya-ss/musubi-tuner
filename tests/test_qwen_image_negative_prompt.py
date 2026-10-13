"""Tests for the negative prompt used by `qwen_image_generate_image.prepare_text_inputs`.

`--negative_prompt` defaults to None, but the unconditional embedding is always
computed (and used for CFG with the default `--guidance_scale 4.0`). Passing None to
the embedder raised `TypeError: 'NoneType' object is not iterable` (issue #1080).
A missing negative prompt now falls back to " ", the same default used by the
Qwen-Image sample-image path in `qwen_image_train_network.py`.
"""

import argparse

import torch

from musubi_tuner import qwen_image_generate_image as gen
from musubi_tuner.qwen_image import qwen_image_utils


class _StubTextEncoder:
    device = torch.device("cpu")
    dtype = torch.bfloat16

    def to(self, *args, **kwargs):
        return self


def _make_args(**overrides) -> argparse.Namespace:
    args = argparse.Namespace(
        prompt="a cat",
        negative_prompt=None,
        guidance_scale=4.0,
        is_edit=False,
        is_layered=False,
        automatic_prompt_lang_for_layered=None,
        image_size=[64, 64],
        control_image_path=None,
        text_encoder_cpu=True,
        model_version="original",
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _shared_models():
    return {"tokenizer": object(), "text_encoder": _StubTextEncoder(), "vl_processor": object()}


def _record_prompts(monkeypatch, name):
    seen = []

    def fake_embed(*args, **kwargs):
        seen.append(args[2])  # (tokenizer_or_processor, text_encoder, prompt, ...)
        return torch.zeros(1, 4, 8), torch.ones(1, 4)

    monkeypatch.setattr(qwen_image_utils, name, fake_embed)
    return seen


def test_missing_negative_prompt_defaults_to_space(monkeypatch):
    seen = _record_prompts(monkeypatch, "get_qwen_prompt_embeds")

    arg_c, arg_null = gen.prepare_text_inputs(_make_args(), None, torch.device("cpu"), _shared_models())

    assert seen == ["a cat", " "]
    assert arg_c["prompt"] == "a cat"
    assert arg_null["prompt"] == " "


def test_explicit_negative_prompt_is_passed_through(monkeypatch):
    seen = _record_prompts(monkeypatch, "get_qwen_prompt_embeds")

    _, arg_null = gen.prepare_text_inputs(_make_args(negative_prompt="blurry"), None, torch.device("cpu"), _shared_models())

    assert seen == ["a cat", "blurry"]
    assert arg_null["prompt"] == "blurry"


def test_explicit_empty_negative_prompt_is_kept(monkeypatch):
    seen = _record_prompts(monkeypatch, "get_qwen_prompt_embeds")

    gen.prepare_text_inputs(_make_args(negative_prompt=""), None, torch.device("cpu"), _shared_models())

    assert seen == ["a cat", ""]


def test_missing_negative_prompt_defaults_to_space_for_edit(monkeypatch):
    seen = _record_prompts(monkeypatch, "get_qwen_prompt_embeds_with_image")

    _, arg_null = gen.prepare_text_inputs(
        _make_args(is_edit=True, model_version="edit-2511"), None, torch.device("cpu"), _shared_models()
    )

    assert seen == ["a cat", " "]
    assert arg_null["prompt"] == " "
