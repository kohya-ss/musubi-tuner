from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.media import H3AudioSource, H3Record, H3Reference
from musubi_tuner.minimax_h3.text_encoder import (
    IMAGE_PLACEHOLDER,
    VIDEO_PLACEHOLDER,
    H3TextVisual,
    build_presentation,
    build_token_tags,
    extract_layer_50_pre_norm,
    normalize_h3_text_encoder_key,
    validate_text_rows,
)
from musubi_tuner.minimax_h3_cache_text_encoder_outputs import _text_cache_metadata


def _record(tmp_path: Path, references=()) -> H3Record:
    return H3Record(
        video_path=tmp_path / "target.mp4",
        caption="A bright scene with clear sound.",
        target_audio=H3AudioSource(tmp_path / "target.wav", embedded=False),
        references=tuple(references),
        jsonl_line=1,
    )


def _visual(frame_count: int, timestamps=None) -> H3TextVisual:
    return H3TextVisual(
        frames=torch.zeros(frame_count, 32, 32, 3),
        timestamps=None if timestamps is None else tuple(timestamps),
    )


def test_t2va_and_fl2va_presentations_are_non_chat_golden_strings(tmp_path: Path):
    record = _record(tmp_path)

    t2va = build_presentation(record, "t2va")
    fl2va = build_presentation(record, "fl2va", {"first": _visual(1), "last": _visual(1)})

    assert t2va.text == record.caption
    assert t2va.images == ()
    assert t2va.videos == ()
    assert fl2va.text == f"<Picture 1>: {IMAGE_PLACEHOLDER}<Picture 2>: {IMAGE_PLACEHOLDER}{record.caption}"
    assert len(fl2va.images) == 2
    assert fl2va.videos == ()


def test_ref2va_presentation_preserves_jsonl_order_and_timestamp_format(tmp_path: Path):
    image = tmp_path / "face.png"
    video = tmp_path / "motion.mp4"
    soundtrack = tmp_path / "motion.wav"
    voice = tmp_path / "voice.wav"
    references = (
        H3Reference(type="image", path=image),
        H3Reference(
            type="video",
            path=video,
            audio=H3AudioSource(soundtrack, embedded=False),
            duration_seconds=2.0,
        ),
        H3Reference(
            type="audio",
            path=voice,
            audio=H3AudioSource(voice, embedded=False),
            duration_seconds=1.0,
        ),
    )
    record = _record(tmp_path, references)

    presentation = build_presentation(
        record,
        "ref2va",
        {
            image: _visual(1),
            video: _visual(3, [0.0, 0.5, 1.0]),
        },
    )

    assert presentation.text == (
        f"<Picture 1>: {IMAGE_PLACEHOLDER}"
        f"<Audio 1>: <Video 1>: <0.2 seconds>{VIDEO_PLACEHOLDER}"
        f"<1.0 seconds>{VIDEO_PLACEHOLDER}"
        f"<Audio 2>: {record.caption}"
    )
    assert presentation.processor_text == (
        f"<Picture 1>: {IMAGE_PLACEHOLDER}<Audio 1>: <Video 1>: {VIDEO_PLACEHOLDER}<Audio 2>: {record.caption}"
    )
    assert len(presentation.images) == 1
    assert [tuple(video_block.shape) for video_block in presentation.videos] == [
        (3, 32, 32, 3),
    ]


def test_layer_50_uses_hidden_state_index_50_where_zero_is_embeddings():
    hidden_states = tuple(torch.full((1, 2, 4), float(index)) for index in range(51))
    output = SimpleNamespace(hidden_states=hidden_states, last_hidden_state=torch.full((1, 2, 4), -1.0))
    model = SimpleNamespace(language_model=SimpleNamespace(layers=[object() for _ in range(64)]))

    actual = extract_layer_50_pre_norm(output, model)

    torch.testing.assert_close(actual, hidden_states[50])


def test_truncated_50_layer_output_prefers_captured_pre_norm_state():
    captured = torch.full((1, 2, 4), 50.0)
    output = SimpleNamespace(
        hidden_states=None,
        last_hidden_state=torch.full((1, 2, 4), -50.0),
        h3_layer_50_pre_norm=captured,
    )
    model = SimpleNamespace(language_model=SimpleNamespace(layers=[object() for _ in range(50)]))

    actual = extract_layer_50_pre_norm(output, model)

    assert actual is captured


def test_token_tags_cover_expanded_vision_rows_and_both_flanking_tokens():
    processed = {
        "input_ids": torch.tensor(
            [[10, 151652, 151655, 151655, 151653, 20, 151652, 151656, 151653, 30]],
            dtype=torch.int64,
        )
    }

    tags = build_token_tags(processed)

    torch.testing.assert_close(tags, torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.int64))
    assert 2 not in tags


def test_text_row_limit_reports_modality_counts_and_bf16_payload():
    hidden_states = torch.empty(32769, 5120, dtype=torch.bfloat16, device="meta")
    token_tags = torch.ones(32769, dtype=torch.int64)
    token_tags[:3] = 0

    with pytest.raises(
        ValueError,
        match=r"32769.*vision_rows=3.*text_rows=32766.*320\.0 MiB",
    ):
        validate_text_rows(hidden_states, token_tags)


def test_text_row_validation_rejects_audio_tags():
    hidden_states = torch.zeros(3, 5120, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_text_rows(hidden_states, torch.tensor([1, 2, 0], dtype=torch.int64))


def test_released_text_checkpoint_key_mapping_keeps_visual_tower_separate():
    assert normalize_h3_text_encoder_key("model.layers.49.mlp.down_proj.weight") == (
        "language_model.layers.49.mlp.down_proj.weight"
    )
    assert normalize_h3_text_encoder_key("model.embed_tokens.weight") == "language_model.embed_tokens.weight"
    assert normalize_h3_text_encoder_key("visual.patch_embed.proj.weight") == "visual.patch_embed.proj.weight"


def test_text_cache_metadata_distinguishes_requested_storage_dtype():
    common = {
        "task": "t2va",
        "crop_start": 0,
        "frame_count": 22,
        "processor_identity": "processor",
        "text_encoder_identity": "encoder",
        "presentation_identity": "presentation",
    }

    bf16 = _text_cache_metadata(**common, cache_dtype="bf16")
    float32 = _text_cache_metadata(**common, cache_dtype="float32")

    assert bf16["cache_dtype"] == "bf16"
    assert float32["cache_dtype"] == "float32"
    assert bf16 != float32
