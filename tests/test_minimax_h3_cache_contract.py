import json
import logging
from pathlib import Path
import sys

import pytest
from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.media import (
    H3AudioSource,
    H3MediaInfo,
    H3Record,
    H3Reference,
    audio_latent_frames,
    h3_records_from_datasource,
    load_h3_jsonl_records,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3_cache_latents import (
    build_latent_tensors,
    cache_metadata_matches,
    log_audio_presence_summary,
    record_media_paths,
    setup_parser,
)
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import (
    AUDIO_PRESENT_KEY,
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
        image: H3MediaInfo(has_audio=False, duration_seconds=None),
        reference_video: H3MediaInfo(has_audio=True, duration_seconds=6.0),
        reference_audio: H3MediaInfo(has_audio=True, duration_seconds=6.0),
        voice: H3MediaInfo(has_audio=True, duration_seconds=4.0),
    }

    records = load_h3_jsonl_records(jsonl, "ref2va", media.__getitem__)

    assert len(records) == 1
    record = records[0]
    assert record.video_path == video
    assert [reference.type for reference in record.references] == ["image", "video", "audio"]
    assert [reference.path for reference in record.references] == [image, reference_video, voice]
    assert record.references[1].audio is not None
    assert record.references[1].audio.path == reference_audio
    assert record.references[1].audio.embedded is False
    assert record.references[2].audio is not None
    assert record.references[2].audio.path == voice
    assert record_media_paths(record) == {video, image, reference_video, reference_audio, voice}


def test_records_from_jsonl_datasource_share_the_parsed_data(tmp_path: Path):
    video = _touch(tmp_path / "target.mp4")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [{"video_path": "target.mp4", "caption": "caption"}])

    from musubi_tuner.dataset.datasources import VideoJsonlDatasource

    datasource = VideoJsonlDatasource(str(jsonl))
    records = h3_records_from_datasource(datasource, "t2va", lambda path: H3MediaInfo(has_audio=False, duration_seconds=5.0))

    assert len(records) == len(datasource.data) == 1
    assert records[0].video_path == video
    assert records[0].caption == "caption"
    assert records[0].references == ()


def test_records_from_directory_datasource_use_captions_and_resolved_paths(tmp_path: Path):
    video = _touch(tmp_path / "clip.mp4")

    class FakeDirectoryDatasource:
        def __len__(self):
            return 1

        def get_caption(self, idx):
            return str(video), "caption"

    records = h3_records_from_datasource(FakeDirectoryDatasource(), "t2va")

    assert records == [H3Record(video_path=video, caption="caption", references=(), jsonl_line=0)]

    with pytest.raises(ValueError, match="Ref2VA requires video_jsonl_file"):
        h3_records_from_datasource(FakeDirectoryDatasource(), "ref2va")


def test_ref2va_reference_audio_resolution_and_media_paths(tmp_path: Path):
    video = _touch(tmp_path / "target.mp4")
    reference_video = _touch(tmp_path / "reference.mp4")
    reference_audio = _touch(tmp_path / "reference.wav")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "caption": "caption",
                "references": [{"type": "video", "path": "reference.mp4", "audio_path": "reference.wav"}],
            }
        ],
    )
    probed = []

    def probe(path: Path) -> H3MediaInfo:
        probed.append(path)
        return H3MediaInfo(has_audio=path == reference_audio, duration_seconds=5.0)

    record = load_h3_jsonl_records(jsonl, "ref2va", probe)[0]

    assert probed == [reference_video, reference_audio]
    assert record.references[0].audio == H3AudioSource(path=reference_audio, embedded=False)
    assert record_media_paths(record) == {video, reference_video, reference_audio}


def test_ref2va_null_audio_path_makes_video_reference_visual_only(tmp_path: Path):
    video = _touch(tmp_path / "target.mp4")
    motion = _touch(tmp_path / "motion.mp4")
    voices = [_touch(tmp_path / f"voice_{index}.wav") for index in range(3)]
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "caption": "caption",
                "references": [
                    # the motion video has an embedded audio track, but null opts out; without
                    # the opt-out this record would exceed the 3 audio-bearing reference limit
                    {"type": "video", "path": "motion.mp4", "audio_path": None},
                    *({"type": "audio", "path": f"voice_{index}.wav"} for index in range(3)),
                ],
            }
        ],
    )

    record = load_h3_jsonl_records(
        jsonl,
        "ref2va",
        lambda path: H3MediaInfo(has_audio=True, duration_seconds=5.0),
    )[0]

    assert record.references[0].type == "video"
    assert record.references[0].audio is None
    assert record_media_paths(record) == {video, motion, *voices}


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        ({"type": "image", "path": "face.png", "audio_path": None}, "image cannot have audio_path"),
        ({"type": "audio", "path": "voice.wav", "audio_path": None}, "audio uses path, not audio_path"),
    ],
)
def test_ref2va_null_audio_path_is_rejected_on_non_video_references(tmp_path: Path, reference: dict, message: str):
    _touch(tmp_path / "target.mp4")
    _touch(tmp_path / "face.png")
    _touch(tmp_path / reference["path"])
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
                "caption": "caption",
                "references": [{"type": "image", "path": "face.png"}, reference],
            }
        ],
    )

    with pytest.raises(ValueError, match=message):
        load_h3_jsonl_records(
            jsonl,
            "ref2va",
            lambda path: H3MediaInfo(has_audio=True, duration_seconds=5.0),
        )


def test_audio_presence_summary_is_aggregated(caplog):
    caplog.set_level(logging.INFO)

    log_audio_presence_summary({True: 3, False: 9})

    assert "real_audio=3 missing_audio=9 supervised_audio_fraction=0.250000" in caplog.text


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
    for reference in references:
        _touch(tmp_path / reference["path"])
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
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
    _touch(tmp_path / "voice.wav")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
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
    reference_video = _touch(tmp_path / "reference.mp4")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "video_path": "target.mp4",
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
        AUDIO_PRESENT_KEY: torch.tensor(1.0, dtype=torch.float32),
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
    torch.testing.assert_close(batch["audio_present"], torch.tensor([1.0]))
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


def test_h3_latent_writer_requires_binary_float32_audio_present_scalar(tmp_path: Path):
    item = _h3_item(tmp_path)
    base = {
        "latents_2x4x4_bfloat16": torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        "latents_audio_32x2x8_float32": torch.zeros(32, 2, 8),
    }

    with pytest.raises(ValueError, match=AUDIO_PRESENT_KEY):
        save_latent_cache_minimax_h3(item, base, {"task": "t2va"})

    invalid = (
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(float("nan"), dtype=torch.float32),
        torch.tensor(0.5, dtype=torch.float32),
    )
    for scalar in invalid:
        with pytest.raises(ValueError, match=AUDIO_PRESENT_KEY):
            save_latent_cache_minimax_h3(
                item,
                {**base, AUDIO_PRESENT_KEY: scalar},
                {"task": "t2va"},
            )


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


def test_h3_teacher_text_rows_round_trip_through_the_bucket_collator(tmp_path: Path):
    item = _h3_item(tmp_path)
    latent_tensors = {
        "latents_2x4x4_bfloat16": torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        "latents_audio_32x2x8_float32": torch.zeros(32, 2, 8),
        AUDIO_PRESENT_KEY: torch.tensor(1.0, dtype=torch.float32),
        "latents_first_1x4x4_float16": torch.ones(24, 1, 4, 4, dtype=torch.float16),
        "latents_last_1x4x4_float16": torch.ones(24, 1, 4, 4, dtype=torch.float16),
    }
    text_tensors = {
        "varlen_mmh3_hidden_states_bfloat16": torch.zeros(3, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_token_tags_int64": torch.tensor([1, 0, 1], dtype=torch.int64),
        "varlen_mmh3_teacher_hidden_states_bfloat16": torch.zeros(5, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_teacher_token_tags_int64": torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64),
    }
    save_latent_cache_minimax_h3(item, latent_tensors, {"task": "fl2va"})
    save_text_encoder_output_cache_minimax_h3(item, text_tensors, {"task": "t2va", "teacher_conditions": "first,last"})

    manager = BucketBatchManager({(64, 64, 5): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["mmh3_hidden_states"][0].shape == (3, 5120)
    assert isinstance(batch["mmh3_teacher_hidden_states"], list)
    assert batch["mmh3_teacher_hidden_states"][0].shape == (5, 5120)
    torch.testing.assert_close(batch["mmh3_teacher_token_tags"][0], torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64))


def test_h3_text_writer_rejects_a_one_sided_teacher_pair(tmp_path: Path):
    item = _h3_item(tmp_path)
    student = {
        "varlen_mmh3_hidden_states_bfloat16": torch.zeros(3, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_token_tags_int64": torch.tensor([1, 0, 1], dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="teacher"):
        save_text_encoder_output_cache_minimax_h3(
            item,
            {**student, "varlen_mmh3_teacher_hidden_states_bfloat16": torch.zeros(5, 5120, dtype=torch.bfloat16)},
        )
    with pytest.raises(ValueError, match="teacher"):
        save_text_encoder_output_cache_minimax_h3(
            item,
            {**student, "varlen_mmh3_teacher_token_tags_int64": torch.ones(5, dtype=torch.int64)},
        )


def test_h3_text_writer_validates_teacher_rows_like_student_rows(tmp_path: Path):
    item = _h3_item(tmp_path)
    tensors = {
        "varlen_mmh3_hidden_states_bfloat16": torch.zeros(3, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_token_tags_int64": torch.tensor([1, 0, 1], dtype=torch.int64),
        "varlen_mmh3_teacher_hidden_states_bfloat16": torch.zeros(5, 5120, dtype=torch.bfloat16),
        "varlen_mmh3_teacher_token_tags_int64": torch.tensor([1, 2, 0, 1, 1], dtype=torch.int64),
    }

    with pytest.raises(ValueError, match="token tags"):
        save_text_encoder_output_cache_minimax_h3(item, tensors)


class _FakeH3VideoVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("latents_mean", torch.zeros(24))
        self.register_buffer("latents_std", torch.ones(24))
        self.calls = []

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        self.calls.append(pixels.detach().cpu())
        frame_count = pixels.shape[2]
        latent_frames = 1 if frame_count == 1 else video_latent_frames(frame_count)
        return torch.zeros(
            pixels.shape[0],
            48,
            latent_frames,
            pixels.shape[3] // 16,
            pixels.shape[4] // 16,
            device=pixels.device,
        )


class _FakeH3AudioVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        self.calls.append(waveform.detach().cpu())
        latent_frames = (waveform.shape[-1] + 799) // 800
        return torch.zeros(waveform.shape[0], 32, 2, latent_frames, device=waveform.device)


class _FakeH3MediaDecoder:
    def __init__(self, visuals=None, audio_lengths=None):
        self.visuals = visuals or {}
        self.audio_lengths = audio_lengths or {}
        self.audio_calls = []
        self.visual_calls = []

    def decode_audio(self, source, *, start_sample, sample_count, require_exact):
        self.audio_calls.append((source, start_sample, sample_count, require_exact))
        length = self.audio_lengths.get(source.path, sample_count)
        return torch.zeros(2, length)

    def decode_reference_visual(self, reference, *, target_frame_count, target_size):
        self.visual_calls.append((reference, target_frame_count, target_size))
        return self.visuals[reference.path]


def _cache_record(tmp_path: Path, references=()) -> H3Record:
    video = _touch(tmp_path / "target.mp4")
    return H3Record(
        video_path=video,
        caption="scene and sound",
        references=tuple(references),
        jsonl_line=1,
    )


def test_build_fl2va_latents_encodes_the_provided_audio_window(tmp_path: Path):
    record = _cache_record(tmp_path)
    frames = torch.zeros(5, 64, 64, 3, dtype=torch.uint8)
    frames[-1] = 255
    waveform = torch.linspace(-0.5, 0.5, 2 * 6400).reshape(2, 6400)
    video_vae = _FakeH3VideoVAE()
    audio_vae = _FakeH3AudioVAE()
    decoder = _FakeH3MediaDecoder()

    payload = build_latent_tensors(
        record=record,
        task="fl2va",
        target_frames=frames,
        target_waveform=waveform,
        audio_present=True,
        crop_start_frame=3,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cache_seed=123,
        media_decoder=decoder,
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={record.video_path: "target-video"},
        allow_experimental_duration=True,
    )

    assert set(payload.tensors) == {
        "latents_2x4x4_float32",
        "latents_audio_32x2x8_float32",
        AUDIO_PRESENT_KEY,
        "latents_first_1x4x4_float32",
        "latents_last_1x4x4_float32",
    }
    assert payload.tensors[AUDIO_PRESENT_KEY].shape == torch.Size([])
    assert payload.tensors[AUDIO_PRESENT_KEY].dtype == torch.float32
    assert payload.tensors[AUDIO_PRESENT_KEY].item() == 1.0
    assert decoder.audio_calls == []
    assert len(audio_vae.calls) == 1
    torch.testing.assert_close(audio_vae.calls[0], waveform.unsqueeze(0))
    assert [call.shape for call in video_vae.calls] == [
        (1, 3, 5, 64, 64),
        (1, 3, 1, 64, 64),
        (1, 3, 1, 64, 64),
    ]
    torch.testing.assert_close(video_vae.calls[1], torch.full_like(video_vae.calls[1], -1.0))
    torch.testing.assert_close(video_vae.calls[2], torch.full_like(video_vae.calls[2], 1.0))
    assert payload.metadata["task"] == "fl2va"
    assert payload.metadata["cache_seed"] == "123"
    assert payload.metadata["crop_start_frame"] == "3"
    assert payload.metadata["cache_format"] == "minimax-h3-latent-v2"
    assert payload.metadata["video_vae_fingerprint"] == "video-fingerprint"
    assert payload.metadata["audio_vae_fingerprint"] == "audio-fingerprint"
    assert json.loads(payload.metadata["media_fingerprints"]) == {str(record.video_path): "target-video"}
    assert set(payload.metadata) == {
        "task",
        "cache_seed",
        "crop_start_frame",
        "cache_format",
        "video_vae_fingerprint",
        "audio_vae_fingerprint",
        "media_fingerprints",
    }


def test_missing_target_audio_encodes_silence_with_presence_zero(tmp_path: Path):
    record = _cache_record(tmp_path)
    video_vae = _FakeH3VideoVAE()
    audio_vae = _FakeH3AudioVAE()
    decoder = _FakeH3MediaDecoder()

    payload = build_latent_tensors(
        record=record,
        task="t2va",
        target_frames=torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
        target_waveform=torch.zeros(2, 6400),
        audio_present=False,
        crop_start_frame=0,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cache_seed=0,
        media_decoder=decoder,
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={record.video_path: "target-video"},
        allow_experimental_duration=True,
    )

    assert decoder.audio_calls == []
    assert len(audio_vae.calls) == 1
    assert audio_vae.calls[0].shape == (1, 2, 6400)
    assert torch.count_nonzero(audio_vae.calls[0]) == 0
    assert payload.tensors[AUDIO_PRESENT_KEY].item() == 0.0


def test_silence_placeholder_must_be_all_zeros(tmp_path: Path):
    record = _cache_record(tmp_path)

    with pytest.raises(ValueError, match="all zeros"):
        build_latent_tensors(
            record=record,
            task="t2va",
            target_frames=torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
            target_waveform=torch.ones(2, 6400),
            audio_present=False,
            crop_start_frame=0,
            video_vae=_FakeH3VideoVAE(),
            audio_vae=_FakeH3AudioVAE(),
            cache_seed=0,
            media_decoder=_FakeH3MediaDecoder(),
            video_vae_fingerprint="video-fingerprint",
            audio_vae_fingerprint="audio-fingerprint",
            media_fingerprints={record.video_path: "target-video"},
            allow_experimental_duration=True,
        )


def test_target_waveform_length_must_match_the_crop(tmp_path: Path):
    record = _cache_record(tmp_path)

    with pytest.raises(ValueError, match=r"must be \[2,6400\]"):
        build_latent_tensors(
            record=record,
            task="t2va",
            target_frames=torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
            target_waveform=torch.zeros(2, 6000),
            audio_present=True,
            crop_start_frame=0,
            video_vae=_FakeH3VideoVAE(),
            audio_vae=_FakeH3AudioVAE(),
            cache_seed=0,
            media_decoder=_FakeH3MediaDecoder(),
            video_vae_fingerprint="video-fingerprint",
            audio_vae_fingerprint="audio-fingerprint",
            media_fingerprints={record.video_path: "target-video"},
            allow_experimental_duration=True,
        )


def test_h3_skip_existing_requires_all_cache_identity_metadata(tmp_path: Path):
    cache_path = tmp_path / "cache.safetensors"
    save_file(
        {"latents_2x4x4_float32": torch.zeros(24, 2, 4, 4)},
        cache_path,
        metadata={"task": "t2va", "video_vae_fingerprint": "old", "cache_seed": "0"},
    )

    assert cache_metadata_matches(cache_path, {"task": "t2va", "video_vae_fingerprint": "old"})
    assert not cache_metadata_matches(cache_path, {"task": "t2va", "video_vae_fingerprint": "new"})
    assert not cache_metadata_matches(cache_path, {"task": "t2va", "audio_vae_fingerprint": "missing"})
    assert cache_metadata_matches(cache_path, {"cache_seed": "0"})
    assert not cache_metadata_matches(cache_path, {"cache_seed": "1"})


def test_h3_latent_cache_parser_exposes_only_the_two_explicit_vae_paths():
    help_text = setup_parser().format_help()

    assert "--video_vae" in help_text
    assert "--audio_vae" in help_text
    assert "--h3_video_only" not in help_text
    assert "--vae VAE" not in help_text
    assert "--vae_dtype" not in help_text


def test_build_ref2va_latents_preserves_ordered_numbered_roles(tmp_path: Path):
    image = _touch(tmp_path / "face.png")
    reference_video = _touch(tmp_path / "motion.mp4")
    reference_video_audio = _touch(tmp_path / "motion.wav")
    voice = _touch(tmp_path / "voice.wav")
    references = (
        H3Reference(type="image", path=image),
        H3Reference(
            type="video",
            path=reference_video,
            audio=H3AudioSource(path=reference_video_audio, embedded=False),
            duration_seconds=4.0,
        ),
        H3Reference(
            type="audio",
            path=voice,
            audio=H3AudioSource(path=voice, embedded=False),
            duration_seconds=1.0,
        ),
    )
    record = _cache_record(tmp_path, references)
    decoder = _FakeH3MediaDecoder(
        visuals={
            image: torch.zeros(1, 32, 64, 3, dtype=torch.uint8),
            reference_video: torch.zeros(5, 64, 32, 3, dtype=torch.uint8),
        },
        audio_lengths={voice: 1600},
    )
    video_vae = _FakeH3VideoVAE()
    audio_vae = _FakeH3AudioVAE()

    payload = build_latent_tensors(
        record=record,
        task="ref2va",
        target_frames=torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
        target_waveform=torch.zeros(2, 6400),
        audio_present=True,
        crop_start_frame=0,
        video_vae=video_vae,
        audio_vae=audio_vae,
        cache_seed=7,
        media_decoder=decoder,
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={path: path.name for path in {record.video_path, image, reference_video, reference_video_audio, voice}},
        allow_experimental_duration=True,
    )

    assert set(payload.tensors) == {
        "latents_2x4x4_float32",
        "latents_audio_32x2x8_float32",
        AUDIO_PRESENT_KEY,
        "latents_ref_000_image_1x2x4_float32",
        "latents_ref_001_video_2x4x2_float32",
        "latents_ref_001_audio_32x2x8_float32",
        "latents_ref_002_audio_32x2x2_float32",
    }
    assert [call[0].path for call in decoder.audio_calls] == [reference_video_audio, voice]
    assert decoder.audio_calls[0][1:] == (0, 6400, True)
    assert decoder.audio_calls[1][1:] == (0, 6400, False)
    assert json.loads(payload.metadata["media_fingerprints"]) == {
        str(path): path.name for path in {record.video_path, image, reference_video, reference_video_audio, voice}
    }


def test_build_ref2va_revalidates_limits_before_any_model_work(tmp_path: Path):
    references = tuple(H3Reference(type="image", path=_touch(tmp_path / f"image_{index}.png")) for index in range(10))
    record = _cache_record(tmp_path, references)
    video_vae = _FakeH3VideoVAE()
    audio_vae = _FakeH3AudioVAE()
    decoder = _FakeH3MediaDecoder()

    with pytest.raises(ValueError, match="at most 9 image"):
        build_latent_tensors(
            record=record,
            task="ref2va",
            target_frames=torch.zeros(5, 64, 64, 3, dtype=torch.uint8),
            target_waveform=torch.zeros(2, 6400),
            audio_present=True,
            crop_start_frame=0,
            video_vae=video_vae,
            audio_vae=audio_vae,
            cache_seed=0,
            media_decoder=decoder,
            video_vae_fingerprint="video-fingerprint",
            audio_vae_fingerprint="audio-fingerprint",
            media_fingerprints={},
            allow_experimental_duration=True,
        )

    assert video_vae.calls == []
    assert audio_vae.calls == []
    assert decoder.audio_calls == []
    assert decoder.visual_calls == []
