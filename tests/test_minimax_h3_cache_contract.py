import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.media import (
    H3MediaInfo,
    audio_latent_frames,
    load_h3_jsonl_records,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import (
    save_latent_cache_minimax_h3,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo


@pytest.mark.parametrize(
    ("frames", "video_frames", "audio_frames"),
    [(5, 2, 8), (22, 7, 37), (39, 12, 65), (56, 17, 93)],
)
def test_h3_geometry_uses_exact_integer_identity(frames: int, video_frames: int, audio_frames: int):
    assert video_latent_frames(frames) == video_frames
    assert audio_latent_frames(frames) == audio_frames
    assert waveform_samples(audio_frames) == audio_frames * 800


@pytest.mark.parametrize("frames", [0, 4, 6, 21, 23])
def test_h3_geometry_rejects_non_17n_plus_5_frames(frames: int):
    with pytest.raises(ValueError, match=r"17\*n\+5"):
        video_latent_frames(frames)
    with pytest.raises(ValueError, match=r"17\*n\+5"):
        audio_latent_frames(frames)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def test_ref2va_jsonl_preserves_reference_order_and_canonicalizes_paths(tmp_path: Path):
    video = _touch(tmp_path / "target.mp4")
    target_audio = _touch(tmp_path / "target.wav")
    image = _touch(tmp_path / "refs" / "face.png")
    reference_video = _touch(tmp_path / "refs" / "motion.mp4")
    reference_audio = _touch(tmp_path / "refs" / "motion.wav")
    voice = _touch(tmp_path / "refs" / "voice.flac")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "caption": "scene and sound",
                "audio_path": "target.wav",
                "references": [
                    {"type": "image", "path": "refs/face.png"},
                    {"type": "video", "path": "refs/motion.mp4", "audio_path": "refs/motion.wav"},
                    {"type": "audio", "path": "refs/voice.flac"},
                ],
            }
        ],
    )
    media = {
        video: H3MediaInfo(has_audio=False, duration_seconds=8.0),
        target_audio: H3MediaInfo(has_audio=True, duration_seconds=8.0),
        image: H3MediaInfo(has_audio=False, duration_seconds=None),
        reference_video: H3MediaInfo(has_audio=True, duration_seconds=6.0),
        reference_audio: H3MediaInfo(has_audio=True, duration_seconds=6.0),
        voice: H3MediaInfo(has_audio=True, duration_seconds=4.0),
    }

    records = load_h3_jsonl_records(jsonl, "ref2va", media.__getitem__)

    assert len(records) == 1
    record = records[0]
    assert record.video_path == video
    assert record.target_audio.path == target_audio
    assert record.target_audio.embedded is False
    assert [reference.type for reference in record.references] == ["image", "video", "audio"]
    assert [reference.path for reference in record.references] == [image, reference_video, voice]
    assert record.references[1].audio is not None
    assert record.references[1].audio.path == reference_audio
    assert record.references[1].audio.embedded is False
    assert record.references[2].audio is not None
    assert record.references[2].audio.path == voice


def test_target_audio_resolution_prefers_one_same_stem_sidecar(tmp_path: Path):
    video = _touch(tmp_path / "clip.mp4")
    sidecar = _touch(tmp_path / "clip.wav")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [{"video_path": "clip.mp4", "caption": "caption"}])
    media = {
        video: H3MediaInfo(has_audio=True, duration_seconds=5.0),
        sidecar: H3MediaInfo(has_audio=True, duration_seconds=5.0),
    }

    record = load_h3_jsonl_records(jsonl, "t2va", media.__getitem__)[0]

    assert record.target_audio.path == sidecar
    assert record.target_audio.embedded is False


def test_target_audio_resolution_rejects_ambiguous_sidecars(tmp_path: Path):
    video = _touch(tmp_path / "clip.mp4")
    _touch(tmp_path / "clip.wav")
    _touch(tmp_path / "clip.flac")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [{"video_path": "clip.mp4", "caption": "caption"}])

    with pytest.raises(ValueError, match="Multiple same-stem audio sidecars"):
        load_h3_jsonl_records(
            jsonl,
            "t2va",
            lambda path: H3MediaInfo(has_audio=path == video, duration_seconds=5.0),
        )


def test_explicit_target_audio_decode_failure_does_not_fall_back(tmp_path: Path):
    video = _touch(tmp_path / "clip.mp4")
    explicit = _touch(tmp_path / "broken.wav")
    _touch(tmp_path / "clip.wav")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [{"video_path": "clip.mp4", "caption": "caption", "audio_path": "broken.wav"}],
    )

    def probe(path: Path) -> H3MediaInfo:
        if path == explicit:
            raise ValueError("decode failed")
        return H3MediaInfo(has_audio=path == video, duration_seconds=5.0)

    with pytest.raises(ValueError, match="Explicit target audio.*decode failed"):
        load_h3_jsonl_records(jsonl, "t2va", probe)


def test_target_audio_is_required(tmp_path: Path):
    video = _touch(tmp_path / "clip.mp4")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [{"video_path": "clip.mp4", "caption": "caption"}])

    with pytest.raises(ValueError, match="Missing target audio"):
        load_h3_jsonl_records(
            jsonl,
            "t2va",
            lambda path: H3MediaInfo(has_audio=False, duration_seconds=5.0),
        )


@pytest.mark.parametrize(
    ("references", "message"),
    [
        ([{"type": "image", "path": f"image_{index}.png"} for index in range(10)], "at most 9 image"),
        ([{"type": "video", "path": f"video_{index}.mp4"} for index in range(4)], "at most 3 video"),
        ([{"type": "audio", "path": f"audio_{index}.wav"} for index in range(4)], "at most 3 audio-bearing"),
        ([{"type": "image", "path": f"image_{index}.png"} for index in range(13)], "at most 12 reference"),
    ],
)
def test_ref2va_reference_limits_fail_before_model_work(tmp_path: Path, references: list[dict], message: str):
    _touch(tmp_path / "target.mp4")
    _touch(tmp_path / "target.wav")
    for reference in references:
        _touch(tmp_path / reference["path"])
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "audio_path": "target.wav",
                "caption": "caption",
                "references": references,
            }
        ],
    )

    with pytest.raises(ValueError, match=message):
        load_h3_jsonl_records(
            jsonl,
            "ref2va",
            lambda path: H3MediaInfo(has_audio=path.suffix in {".wav", ".mp4"}, duration_seconds=5.0),
        )


def test_ref2va_requires_a_visual_reference(tmp_path: Path):
    _touch(tmp_path / "target.mp4")
    _touch(tmp_path / "target.wav")
    _touch(tmp_path / "voice.wav")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "audio_path": "target.wav",
                "caption": "caption",
                "references": [{"type": "audio", "path": "voice.wav"}],
            }
        ],
    )

    with pytest.raises(ValueError, match="at least one visual"):
        load_h3_jsonl_records(
            jsonl,
            "ref2va",
            lambda path: H3MediaInfo(has_audio=True, duration_seconds=5.0),
        )


@pytest.mark.parametrize("duration", [1.99, 15.01])
def test_ref2va_video_reference_duration_is_two_through_fifteen_seconds(tmp_path: Path, duration: float):
    _touch(tmp_path / "target.mp4")
    _touch(tmp_path / "target.wav")
    reference_video = _touch(tmp_path / "reference.mp4")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "audio_path": "target.wav",
                "caption": "caption",
                "references": [{"type": "video", "path": "reference.mp4"}],
            }
        ],
    )

    def probe(path: Path) -> H3MediaInfo:
        if path == reference_video:
            return H3MediaInfo(has_audio=False, duration_seconds=duration)
        return H3MediaInfo(has_audio=True, duration_seconds=5.0)

    with pytest.raises(ValueError, match="between 2 and 15 seconds"):
        load_h3_jsonl_records(jsonl, "ref2va", probe)


def _h3_item(tmp_path: Path) -> ItemInfo:
    item = ItemInfo(
        item_key="clip.mp4",
        caption="caption",
        original_size=(64, 64),
        bucket_size=(64, 64, 5),
        frame_count=5,
    )
    item.latent_cache_path = str(tmp_path / "clip_00000-005_0064x0064_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "clip_mmh3_te.safetensors")
    return item


def test_h3_cache_keys_round_trip_through_existing_bucket_collator(tmp_path: Path):
    item = _h3_item(tmp_path)
    latent_tensors = {
        "latents_2x4x4_bfloat16": torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        "latents_audio_32x2x8_float32": torch.zeros(32, 2, 8),
        "latents_first_1x4x4_float16": torch.ones(24, 1, 4, 4, dtype=torch.float16),
    }
    text_tensors = {
        "varlen_mmh3_hidden_states_bfloat16": torch.zeros(3, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_token_tags_int64": torch.tensor([1, 0, 1], dtype=torch.int64),
    }
    save_latent_cache_minimax_h3(item, latent_tensors, {"task": "fl2va"})
    save_text_encoder_output_cache_minimax_h3(item, text_tensors, {"task": "fl2va"})

    manager = BucketBatchManager({(64, 64, 5): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["latents"].shape == (1, 24, 2, 4, 4)
    assert batch["latents_audio"].shape == (1, 32, 2, 8)
    assert batch["latents_first"].shape == (1, 24, 1, 4, 4)
    assert isinstance(batch["mmh3_hidden_states"], list)
    assert batch["mmh3_hidden_states"][0].shape == (3, 5120)
    assert isinstance(batch["mmh3_token_tags"], list)
    torch.testing.assert_close(batch["mmh3_token_tags"][0], torch.tensor([1, 0, 1], dtype=torch.int64))


def test_h3_latent_writer_rejects_transposed_audio_layout(tmp_path: Path):
    item = _h3_item(tmp_path)
    tensors = {
        "latents_2x4x4_bfloat16": torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        "latents_audio_32x2x8_float32": torch.zeros(2, 32, 8),
    }

    with pytest.raises(ValueError, match=r"audio latent \[32,2,A\]"):
        save_latent_cache_minimax_h3(item, tensors)


@pytest.mark.parametrize(
    "tags",
    [torch.tensor([1, 0, 1], dtype=torch.int32), torch.tensor([1, 2, 1], dtype=torch.int64)],
)
def test_h3_text_writer_rejects_invalid_token_tags(tmp_path: Path, tags: torch.Tensor):
    item = _h3_item(tmp_path)
    tensors = {
        "varlen_mmh3_hidden_states_bfloat16": torch.zeros(3, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_token_tags_int64": tags,
    }

    with pytest.raises(ValueError, match="token tags"):
        save_text_encoder_output_cache_minimax_h3(item, tensors)
