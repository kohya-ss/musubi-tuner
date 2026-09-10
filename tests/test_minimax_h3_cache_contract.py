import json
import logging
from pathlib import Path
import sys

import pytest
from safetensors.torch import save_file
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.dataset.datasources import ImageDirectoryDatasource, ImageJsonlDatasource, ItemExtras, VideoDirectoryDatasource
from musubi_tuner.minimax_h3.media import (
    ONE_FRAME_REFERENCE_FRAME_CAP,
    H3AudioSource,
    H3MediaInfo,
    H3Record,
    H3Reference,
    audio_latent_frames,
    h3_records_from_datasource,
    load_h3_jsonl_records,
    parse_inline_references,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3.cache_plan import cache_metadata_matches
from musubi_tuner.minimax_h3_cache_latents import (
    build_latent_tensors,
    build_one_frame_latent_tensors,
    encode_one_frame_silence_latent,
    log_audio_presence_summary,
    record_media_paths,
    setup_parser,
)
from musubi_tuner.dataset.bucket import BucketBatchManager
from musubi_tuner.dataset.cache_io import (
    AUDIO_PRESENT_KEY,
    ONE_FRAME_CONTROL_INDICES_KEY,
    ONE_FRAME_TARGET_INDEX_KEY,
    save_latent_cache_minimax_h3,
    save_text_encoder_output_cache_minimax_h3,
)
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen


def _saved_keys(path: str) -> set[str]:
    with MemoryEfficientSafeOpen(path) as f:
        return set(f.keys())


def _save_text_rows(item: ItemInfo, rows: int = 3, tags=None, metadata=None, **teacher) -> None:
    save_text_encoder_output_cache_minimax_h3(
        item,
        hidden_states=torch.zeros(rows, 5120, dtype=torch.bfloat16),
        token_tags=torch.tensor([1, 0, 1][:rows], dtype=torch.int64) if tags is None else tags,
        metadata=metadata,
        **teacher,
    )


_TEACHER_ROWS = dict(
    teacher_hidden_states=torch.zeros(5, 5120, dtype=torch.bfloat16),
    teacher_token_tags=torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64),
)


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


def test_inline_references_infer_types_from_extensions_and_resolve_relative_paths(tmp_path: Path):
    image = _touch(tmp_path / "refs" / "face.png")
    motion = _touch(tmp_path / "refs" / "motion.mp4")
    song = _touch(tmp_path / "refs" / "song.wav")
    voice = _touch(tmp_path / "refs" / "voice.flac")
    clip = _touch(tmp_path / "refs" / "clip")  # no extension -> video
    media = {
        motion: H3MediaInfo(has_audio=False, duration_seconds=6.0),
        song: H3MediaInfo(has_audio=True, duration_seconds=6.0),
        voice: H3MediaInfo(has_audio=True, duration_seconds=4.0),
        clip: H3MediaInfo(has_audio=True, duration_seconds=3.0),
    }

    references = parse_inline_references(
        ["refs/face.png", "refs/motion.mp4;audio=refs/song.wav", "refs/voice.flac", "refs/clip"],
        tmp_path,
        media.__getitem__,
    )

    assert [reference.type for reference in references] == ["image", "video", "audio", "video"]
    assert [reference.path for reference in references] == [image, motion, voice, clip]
    assert references[1].audio == H3AudioSource(path=song, embedded=False)
    assert references[2].audio == H3AudioSource(path=voice, embedded=False)
    assert references[3].audio == H3AudioSource(path=clip, embedded=True)


def test_inline_reference_type_override_and_spec_errors(tmp_path: Path):
    still = _touch(tmp_path / "still.png")
    _touch(tmp_path / "voice.wav")

    def probe(path):
        del path
        return H3MediaInfo(has_audio=False, duration_seconds=6.0)

    references = parse_inline_references(["still.png;type=video"], tmp_path, probe)
    assert references[0].type == "video"
    assert references[0].path == still

    for spec, message in (
        ("still.png;fast", "key=value"),
        ("still.png;size=2", "unknown inline reference option"),
        ("still.png;type=image;type=video", "duplicate inline reference option"),
        ("still.png;type=photo", "must be image, video, or audio"),
        (";type=image", "must start with a path"),
        ("still.png;audio=voice.wav", "image cannot have audio_path"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_inline_references([spec], tmp_path, probe)

    # the reference-count validation (at least one visual) runs before the per-item checks
    with pytest.raises(ValueError, match="audio uses path"):
        parse_inline_references(["still.png", "voice.wav;audio=voice.wav"], tmp_path, probe)


def test_inline_references_share_the_jsonl_count_and_duration_rules(tmp_path: Path):
    _touch(tmp_path / "still.png")
    _touch(tmp_path / "voice.wav")
    _touch(tmp_path / "long.mp4")

    def probe(path):
        return H3MediaInfo(has_audio=True, duration_seconds=30.0 if path.suffix == ".mp4" else 4.0)

    with pytest.raises(ValueError, match="at most 9 image references"):
        parse_inline_references(["still.png"] * 10, tmp_path, probe)
    with pytest.raises(ValueError, match="at least one visual reference"):
        parse_inline_references(["voice.wav"], tmp_path, probe)
    with pytest.raises(ValueError, match="between 2 and 15 seconds"):
        parse_inline_references(["long.mp4"], tmp_path, probe)
    with pytest.raises(ValueError, match="does not exist"):
        parse_inline_references(["missing.png"], tmp_path, probe)


def test_records_from_jsonl_datasource_share_the_parsed_data(tmp_path: Path):
    video = _touch(tmp_path / "target.mp4")
    jsonl = tmp_path / "data.jsonl"
    _write_jsonl(jsonl, [{"video_path": "target.mp4", "caption": "caption"}])

    from musubi_tuner.dataset.datasources import VideoJsonlDatasource

    datasource = VideoJsonlDatasource(str(jsonl))
    records = h3_records_from_datasource(datasource, "t2va", lambda path: H3MediaInfo(has_audio=False, duration_seconds=5.0))

    assert len(records) == len(datasource) == 1
    assert records[0].video_path == video
    assert records[0].caption == "caption"
    assert records[0].references == ()
    assert records[0].label == "data.jsonl line 1"


def test_records_from_directory_datasource_use_captions_and_resolved_paths(tmp_path: Path):
    video = _touch(tmp_path / "clip.mp4")
    (tmp_path / "clip.txt").write_text("caption", encoding="utf-8")

    datasource = VideoDirectoryDatasource(str(tmp_path), ".txt")
    records = h3_records_from_datasource(datasource, "t2va")

    assert records == [H3Record(video_path=video, caption="caption", references=(), label=str(video))]

    # a video directory item has no place for references, so Ref2VA needs a record-based dataset
    with pytest.raises(ValueError, match="Ref2VA requires per-item references"):
        h3_records_from_datasource(datasource, "ref2va")


def test_records_read_only_the_shared_accessors_of_the_datasource(tmp_path: Path):
    """The H3 record builder must not depend on the datasource implementation (no isinstance,
    no JSONL internals): a minimal datasource exposing the shared accessors is enough."""
    video = _touch(tmp_path / "clip.mp4")
    face = _touch(tmp_path / "refs" / "face.png")

    class MinimalDatasource:
        def __len__(self):
            return 1

        def get_caption(self, idx):
            return str(video), "caption"

        def get_item_extras(self, idx):
            return ItemExtras(
                fields={"references": [{"type": "image", "path": "refs/face.png"}], "teacher_caption": "teacher"},
                base_directory=str(tmp_path),
                label="custom item 1",
            )

    (record,) = h3_records_from_datasource(MinimalDatasource(), "ref2va")

    assert record.video_path == video
    assert [(reference.type, reference.path) for reference in record.references] == [("image", face)]
    assert record.teacher_caption == "teacher"
    assert record.label == "custom item 1"
    assert record.context == "H3 custom item 1"

    with pytest.raises(ValueError, match="H3 custom item 1: references require task ref2va"):
        h3_records_from_datasource(MinimalDatasource(), "t2va")


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
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        target_audio=torch.zeros(32, 2, 8),
        audio_present=True,
        visual_conditions={"first": torch.ones(24, 1, 4, 4, dtype=torch.float16)},
        metadata={"task": "fl2va"},
    )
    _save_text_rows(item, metadata={"task": "fl2va"})

    # the writer names the entries (`latents[_<role>]_<shape>_<dtype>`, varlen text rows):
    # existing caches depend on these exact names
    assert _saved_keys(item.latent_cache_path) == {
        "latents_2x4x4_bfloat16",
        "latents_audio_32x2x8_float32",
        AUDIO_PRESENT_KEY,
        "latents_first_1x4x4_float16",
    }
    assert _saved_keys(item.text_encoder_output_cache_path) == {
        "varlen_mmh3_hidden_states_bfloat16",
        "varlen_mmh3_token_tags_int64",
    }

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

    with pytest.raises(ValueError, match=r"audio latent \[32,2,A\]"):
        save_latent_cache_minimax_h3(
            item,
            target_video=torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
            target_audio=torch.zeros(2, 32, 8),
            audio_present=True,
        )


def test_h3_latent_writer_records_audio_presence_as_the_binary_scalar_entry(tmp_path: Path):
    item = _h3_item(tmp_path)
    for audio_present in (True, False):
        save_latent_cache_minimax_h3(
            item,
            target_video=torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
            target_audio=torch.zeros(32, 2, 8),
            audio_present=audio_present,
            metadata={"task": "t2va"},
        )
        with MemoryEfficientSafeOpen(item.latent_cache_path) as f:
            scalar = f.get_tensor(AUDIO_PRESENT_KEY)
        assert scalar.dtype == torch.float32 and scalar.shape == torch.Size([])
        assert scalar.item() == (1.0 if audio_present else 0.0)


@pytest.mark.parametrize(
    ("conditions", "message"),
    [
        ({"visual_conditions": {"cond_0": torch.zeros(24, 1, 4, 4)}}, "Unsupported MiniMax-H3 condition role"),
        ({"visual_conditions": {"ref_000_audio": torch.zeros(24, 1, 4, 4)}}, "carries audio rows"),
        ({"audio_conditions": {"ref_000_video": torch.zeros(32, 2, 8)}}, "carries a visual latent"),
        ({"visual_conditions": {"last": torch.zeros(1, 24, 4, 4)}}, r"visual condition last latent must be \[24,F,H,W\]"),
        ({"audio_conditions": {"ref_001_audio": torch.zeros(32, 8)}}, r"audio condition ref_001_audio latent \[32,2,A\]"),
    ],
)
def test_h3_latent_writer_rejects_conditions_outside_the_role_vocabulary(tmp_path: Path, conditions: dict, message: str):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match=message):
        save_latent_cache_minimax_h3(
            item,
            target_video=torch.zeros(24, 2, 4, 4),
            target_audio=torch.zeros(32, 2, 8),
            audio_present=True,
            **conditions,
        )


@pytest.mark.parametrize(
    "tags",
    [torch.tensor([1, 0, 1], dtype=torch.int32), torch.tensor([1, 2, 1], dtype=torch.int64)],
)
def test_h3_text_writer_rejects_invalid_token_tags(tmp_path: Path, tags: torch.Tensor):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match="token tags"):
        _save_text_rows(item, tags=tags)


def test_h3_teacher_text_rows_round_trip_through_the_bucket_collator(tmp_path: Path):
    item = _h3_item(tmp_path)
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        target_audio=torch.zeros(32, 2, 8),
        audio_present=True,
        visual_conditions={
            "first": torch.ones(24, 1, 4, 4, dtype=torch.float16),
            "last": torch.ones(24, 1, 4, 4, dtype=torch.float16),
        },
        metadata={"task": "fl2va"},
    )
    _save_text_rows(item, metadata={"task": "t2va", "teacher_conditions": "first,last"}, teacher_kind="first,last", **_TEACHER_ROWS)

    assert _saved_keys(item.text_encoder_output_cache_path) == {
        "varlen_mmh3_hidden_states_bfloat16",
        "varlen_mmh3_token_tags_int64",
        "varlen_mmh3_teacher_hidden_states_bfloat16",
        "varlen_mmh3_teacher_token_tags_int64",
    }

    manager = BucketBatchManager({(64, 64, 5): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["mmh3_hidden_states"][0].shape == (3, 5120)
    assert isinstance(batch["mmh3_teacher_hidden_states"], list)
    assert batch["mmh3_teacher_hidden_states"][0].shape == (5, 5120)
    torch.testing.assert_close(batch["mmh3_teacher_token_tags"][0], torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64))


def test_h3_text_writer_rejects_a_one_sided_teacher_pair(tmp_path: Path):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match="teacher"):
        _save_text_rows(item, teacher_kind="first,last", teacher_hidden_states=torch.zeros(5, 5120, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="teacher"):
        _save_text_rows(item, teacher_kind="ref", teacher_token_tags=torch.ones(5, dtype=torch.int64))
    with pytest.raises(ValueError, match="teacher"):
        _save_text_rows(item, **_TEACHER_ROWS)


def test_h3_text_writer_rejects_an_unknown_teacher_kind(tmp_path: Path):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match="teacher kind"):
        _save_text_rows(item, teacher_kind="first", **_TEACHER_ROWS)


def test_h3_ref_teacher_text_rows_round_trip_through_the_bucket_collator(tmp_path: Path):
    # the ref teacher needs no endpoint condition latents: a plain T2VA latent cache suffices
    item = _h3_item(tmp_path)
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 2, 4, 4, dtype=torch.bfloat16),
        target_audio=torch.zeros(32, 2, 8),
        audio_present=True,
        metadata={"task": "t2va"},
    )
    _save_text_rows(item, metadata={"task": "t2va", "teacher_conditions": "ref"}, teacher_kind="ref", **_TEACHER_ROWS)

    assert _saved_keys(item.text_encoder_output_cache_path) >= {
        "varlen_mmh3_teacher_ref_hidden_states_bfloat16",
        "varlen_mmh3_teacher_ref_token_tags_int64",
    }

    manager = BucketBatchManager({(64, 64, 5): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["mmh3_hidden_states"][0].shape == (3, 5120)
    assert "mmh3_teacher_hidden_states" not in batch
    assert isinstance(batch["mmh3_teacher_ref_hidden_states"], list)
    assert batch["mmh3_teacher_ref_hidden_states"][0].shape == (5, 5120)
    torch.testing.assert_close(batch["mmh3_teacher_ref_token_tags"][0], torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64))


def test_h3_subject_ref_teacher_text_keys_round_trip_through_the_bucket_collator(tmp_path: Path):
    # the subject-reference teacher rows live next to a one-frame ref2va latent cache; the writer
    # must accept the third teacher kind (smoke S1 regression: it only knew first,last and ref)
    item = ItemInfo("view", "sks girl", (64, 64), (64, 64))
    item.latent_cache_path = str(tmp_path / "view_0064x0064_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "view_mmh3_te.safetensors")
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 1, 4, 4),
        target_audio=torch.zeros(32, 2, 2),
        audio_present=False,
        visual_conditions={"ref_000_image": torch.ones(24, 1, 4, 4)},
        one_frame_target_index=0,
        metadata={"task": "ref2va", "one_frame": "1"},
    )
    _save_text_rows(
        item,
        tags=torch.tensor([1, 1, 1], dtype=torch.int64),
        metadata={"task": "t2va", "teacher_conditions": "subject_ref"},
        teacher_kind="subject_ref",
        **_TEACHER_ROWS,
    )

    assert _saved_keys(item.text_encoder_output_cache_path) >= {
        "varlen_mmh3_teacher_subject_ref_hidden_states_bfloat16",
        "varlen_mmh3_teacher_subject_ref_token_tags_int64",
    }

    batch = BucketBatchManager({(64, 64): [item]}, batch_size=1)[0]

    assert batch["latents_ref_000_image"].shape == (1, 24, 1, 4, 4)
    assert "mmh3_teacher_hidden_states" not in batch and "mmh3_teacher_ref_hidden_states" not in batch
    assert batch["mmh3_teacher_subject_ref_hidden_states"][0].shape == (5, 5120)
    torch.testing.assert_close(batch["mmh3_teacher_subject_ref_token_tags"][0], torch.tensor([1, 0, 0, 1, 1], dtype=torch.int64))


def test_h3_text_writer_validates_teacher_rows_like_student_rows(tmp_path: Path):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match="token tags"):
        _save_text_rows(
            item,
            teacher_kind="first,last",
            teacher_hidden_states=torch.zeros(5, 5120, dtype=torch.bfloat16),
            teacher_token_tags=torch.tensor([1, 2, 0, 1, 1], dtype=torch.int64),
        )
    with pytest.raises(ValueError, match=r"\[L,5120\]"):
        _save_text_rows(
            item,
            teacher_kind="first,last",
            teacher_hidden_states=torch.zeros(5, 4096, dtype=torch.bfloat16),
            teacher_token_tags=torch.ones(5, dtype=torch.int64),
        )


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
        label="items.jsonl line 1",
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

    assert payload.target_video.shape == (24, 2, 4, 4) and payload.target_video.dtype == torch.float32
    assert payload.target_audio.shape == (32, 2, 8) and payload.target_audio.dtype == torch.float32
    assert payload.audio_present is True
    assert {role: tuple(latent.shape) for role, latent in payload.visual_conditions.items()} == {
        "first": (24, 1, 4, 4),
        "last": (24, 1, 4, 4),
    }
    assert payload.audio_conditions == {}
    assert payload.one_frame_target_index is None and payload.one_frame_control_indices is None
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
    assert payload.audio_present is False


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

    assert payload.target_video.shape == (24, 2, 4, 4)
    assert payload.target_audio.shape == (32, 2, 8)
    assert {role: tuple(latent.shape) for role, latent in payload.visual_conditions.items()} == {
        "ref_000_image": (24, 1, 2, 4),
        "ref_001_video": (24, 2, 4, 2),
    }
    assert {role: tuple(latent.shape) for role, latent in payload.audio_conditions.items()} == {
        "ref_001_audio": (32, 2, 8),
        "ref_002_audio": (32, 2, 2),
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


def test_one_frame_silence_latent_is_the_two_frame_placeholder():
    audio_vae = _FakeH3AudioVAE()

    latent = encode_one_frame_silence_latent(audio_vae)

    assert latent.shape == (32, 2, 2)
    assert len(audio_vae.calls) == 1
    assert audio_vae.calls[0].shape == (1, 2, 1600)
    assert torch.count_nonzero(audio_vae.calls[0]) == 0


def test_build_one_frame_latents_pack_silence_and_the_target_index(tmp_path: Path):
    image_path = _touch(tmp_path / "portrait.png")
    video_vae = _FakeH3VideoVAE()
    silence = torch.zeros(32, 2, 2)

    payload = build_one_frame_latent_tensors(
        image_frames=torch.zeros(64, 64, 3, dtype=torch.uint8),
        target_index=24,
        video_vae=video_vae,
        silence_audio_latent=silence,
        cache_seed=123,
        item_key=str(image_path),
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={image_path: "portrait-image"},
    )

    assert payload.target_video.shape == (24, 1, 4, 4)
    assert payload.target_audio is silence
    assert payload.audio_present is False
    assert payload.visual_conditions == {} and payload.audio_conditions == {}
    assert payload.one_frame_target_index == 24
    assert payload.one_frame_control_indices is None
    assert [call.shape for call in video_vae.calls] == [(1, 3, 1, 64, 64)]
    assert payload.metadata["task"] == "t2va"
    assert payload.metadata["crop_start_frame"] == "0"
    assert payload.metadata["one_frame"] == "1"
    assert payload.metadata["one_frame_target_index"] == "24"
    assert json.loads(payload.metadata["media_fingerprints"]) == {str(image_path): "portrait-image"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_index": -1}, "nonnegative"),
        ({"image_frames": torch.zeros(60, 64, 3, dtype=torch.uint8)}, "divisible by 32"),
        ({"image_frames": torch.zeros(2, 64, 64, 3, dtype=torch.uint8)}, "single"),
        ({"silence_audio_latent": torch.zeros(32, 2, 8)}, r"\[32,2,2\]"),
    ],
)
def test_build_one_frame_latents_reject_invalid_inputs(tmp_path: Path, overrides: dict, message: str):
    image_path = _touch(tmp_path / "portrait.png")
    inputs = dict(
        image_frames=torch.zeros(64, 64, 3, dtype=torch.uint8),
        target_index=0,
        video_vae=_FakeH3VideoVAE(),
        silence_audio_latent=torch.zeros(32, 2, 2),
        cache_seed=0,
        item_key=str(image_path),
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={image_path: "portrait-image"},
    )
    inputs.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_one_frame_latent_tensors(**inputs)


def test_one_frame_cache_keys_round_trip_through_the_bucket_collator(tmp_path: Path):
    item = ItemInfo("portrait", "an image caption", (64, 64), (64, 64))
    item.latent_cache_path = str(tmp_path / "portrait_0064x0064_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "portrait_mmh3_te.safetensors")
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 1, 4, 4),
        target_audio=torch.zeros(32, 2, 2),
        audio_present=False,
        one_frame_target_index=24,
        metadata={"task": "t2va", "one_frame": "1"},
    )
    _save_text_rows(item, tags=torch.tensor([1, 1, 1], dtype=torch.int64), metadata={"task": "t2va"})

    assert _saved_keys(item.latent_cache_path) == {
        "latents_1x4x4_float32",
        "latents_audio_32x2x2_float32",
        AUDIO_PRESENT_KEY,
        ONE_FRAME_TARGET_INDEX_KEY,
    }

    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["latents"].shape == (1, 24, 1, 4, 4)
    assert batch["latents_audio"].shape == (1, 32, 2, 2)
    torch.testing.assert_close(batch["audio_present"], torch.tensor([0.0]))
    torch.testing.assert_close(batch["one_frame_target_index"], torch.tensor([24], dtype=torch.int64))


def test_h3_latent_writer_rejects_invalid_one_frame_target_indices(tmp_path: Path):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match="target index must be nonnegative"):
        save_latent_cache_minimax_h3(
            item,
            target_video=torch.zeros(24, 1, 4, 4),
            target_audio=torch.zeros(32, 2, 2),
            audio_present=False,
            one_frame_target_index=-1,
            metadata={"task": "t2va"},
        )


@pytest.mark.parametrize("control_indices", [[0], [0, 48], [0, 24, 48]])
def test_build_one_frame_latents_pack_controls_and_their_indices(tmp_path: Path, control_indices: list[int]):
    image_path = _touch(tmp_path / "target.png")
    video_vae = _FakeH3VideoVAE()
    controls = [torch.full((64, 64, 3), 32 * (index + 1), dtype=torch.uint8) for index in range(len(control_indices))]

    payload = build_one_frame_latent_tensors(
        image_frames=torch.zeros(64, 64, 3, dtype=torch.uint8),
        target_index=24,
        video_vae=video_vae,
        silence_audio_latent=torch.zeros(32, 2, 2),
        cache_seed=123,
        item_key=str(image_path),
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={image_path: "target-image"},
        control_frames=controls,
        control_indices=control_indices,
    )

    # one-frame conditions are the ordered cond_{i} slots (any count), never the video first/last roles
    expected_roles = tuple(f"cond_{index:03d}" for index in range(len(control_indices)))
    assert payload.target_video.shape == (24, 1, 4, 4)
    assert {role: tuple(latent.shape) for role, latent in payload.visual_conditions.items()} == {
        role: (24, 1, 4, 4) for role in expected_roles
    }
    assert payload.audio_conditions == {}
    assert payload.one_frame_target_index == 24
    assert payload.one_frame_control_indices == tuple(control_indices)
    # target encode + one condition encode per control
    assert [call.shape for call in video_vae.calls] == [(1, 3, 1, 64, 64)] * (1 + len(control_indices))
    assert payload.metadata["task"] == "fl2va"
    assert payload.metadata["one_frame"] == "1"
    assert payload.metadata["one_frame_format"] == "minimax-h3-one-frame-v2"
    assert payload.metadata["one_frame_control_indices"] == ";".join(str(index) for index in control_indices)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"control_frames": [torch.zeros(64, 64, 3, dtype=torch.uint8)]}, "together"),
        ({"control_indices": [0]}, "together"),
        ({"control_frames": [], "control_indices": []}, "at least one control image"),
        (
            {"control_frames": [torch.zeros(64, 64, 3, dtype=torch.uint8)], "control_indices": [0, 48]},
            "does not match",
        ),
        (
            {"control_frames": [torch.zeros(32, 32, 3, dtype=torch.uint8)], "control_indices": [0]},
            "does not match the target",
        ),
        (
            {"control_frames": [torch.zeros(64, 64, 3, dtype=torch.uint8)], "control_indices": [-1]},
            "nonnegative",
        ),
    ],
)
def test_build_one_frame_latents_reject_invalid_controls(tmp_path: Path, overrides: dict, message: str):
    image_path = _touch(tmp_path / "target.png")
    inputs = dict(
        image_frames=torch.zeros(64, 64, 3, dtype=torch.uint8),
        target_index=24,
        video_vae=_FakeH3VideoVAE(),
        silence_audio_latent=torch.zeros(32, 2, 2),
        cache_seed=0,
        item_key=str(image_path),
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={image_path: "target-image"},
    )
    inputs.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_one_frame_latent_tensors(**inputs)


def test_one_frame_control_cache_keys_round_trip_through_the_bucket_collator(tmp_path: Path):
    item = ItemInfo("edit", "an editing caption", (64, 64), (64, 64, 1))
    item.latent_cache_path = str(tmp_path / "edit_0064x0064_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "edit_mmh3_te.safetensors")
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 1, 4, 4),
        target_audio=torch.zeros(32, 2, 2),
        audio_present=False,
        visual_conditions={"cond_000": torch.ones(24, 1, 4, 4)},
        one_frame_target_index=24,
        one_frame_control_indices=[0],
        metadata={"task": "fl2va", "one_frame": "1"},
    )
    _save_text_rows(item, tags=torch.tensor([1, 1, 1], dtype=torch.int64), metadata={"task": "fl2va"})

    assert _saved_keys(item.latent_cache_path) == {
        "latents_1x4x4_float32",
        "latents_cond_000_1x4x4_float32",
        "latents_audio_32x2x2_float32",
        AUDIO_PRESENT_KEY,
        ONE_FRAME_TARGET_INDEX_KEY,
        ONE_FRAME_CONTROL_INDICES_KEY,
    }

    manager = BucketBatchManager({(64, 64, 1): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["latents"].shape == (1, 24, 1, 4, 4)
    assert batch["latents_cond_000"].shape == (1, 24, 1, 4, 4)
    torch.testing.assert_close(batch["one_frame_target_index"], torch.tensor([24], dtype=torch.int64))
    torch.testing.assert_close(batch["one_frame_control_indices"], torch.tensor([[0]], dtype=torch.int64))


def _one_frame_reference_record(tmp_path: Path, *, with_video: bool = True, with_audio_reference: bool = False) -> H3Record:
    image = _touch(tmp_path / "refs" / "face.png")
    references = [H3Reference(type="image", path=image)]
    if with_video:
        video = _touch(tmp_path / "refs" / "motion.mp4")
        references.append(
            H3Reference(type="video", path=video, audio=H3AudioSource(path=video, embedded=True), duration_seconds=4.0)
        )
    if with_audio_reference:
        voice = _touch(tmp_path / "refs" / "voice.wav")
        references.append(
            H3Reference(type="audio", path=voice, audio=H3AudioSource(path=voice, embedded=False), duration_seconds=1.0)
        )
    return H3Record(
        video_path=_touch(tmp_path / "target.png"),
        caption="a novel view of the character",
        references=tuple(references),
        label="items.jsonl line 1",
    )


def test_build_one_frame_latents_pack_references_under_numbered_roles(tmp_path: Path):
    record = _one_frame_reference_record(tmp_path)
    image, video = (reference.path for reference in record.references)
    decoder = _FakeH3MediaDecoder(
        visuals={
            image: torch.zeros(1, 32, 64, 3, dtype=torch.uint8),
            video: torch.zeros(5, 64, 32, 3, dtype=torch.uint8),
        }
    )
    video_vae = _FakeH3VideoVAE()
    audio_vae = _FakeH3AudioVAE()

    payload = build_one_frame_latent_tensors(
        image_frames=torch.zeros(64, 64, 3, dtype=torch.uint8),
        target_index=24,
        video_vae=video_vae,
        silence_audio_latent=torch.zeros(32, 2, 2),
        cache_seed=123,
        item_key=str(record.video_path),
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={record.video_path: "target-image", image: "face", video: "motion"},
        record=record,
        audio_vae=audio_vae,
        media_decoder=decoder,
    )

    assert payload.target_video.shape == (24, 1, 4, 4)
    assert payload.target_audio.shape == (32, 2, 2)
    assert payload.audio_present is False
    assert {role: tuple(latent.shape) for role, latent in payload.visual_conditions.items()} == {
        "ref_000_image": (24, 1, 2, 4),
        "ref_001_video": (24, 2, 4, 2),
    }
    assert {role: tuple(latent.shape) for role, latent in payload.audio_conditions.items()} == {"ref_001_audio": (32, 2, 8)}
    assert payload.one_frame_target_index == 24
    assert payload.one_frame_control_indices is None
    # references are decoded with the one-frame policy: images capped to the target area,
    # videos to the released 15 s span (not a target duration, which a single frame lacks)
    assert [(call[1], call[2]) for call in decoder.visual_calls] == [(ONE_FRAME_REFERENCE_FRAME_CAP, (64, 64))] * 2
    # the reference video keeps its own audio duration
    assert decoder.audio_calls[0][1:] == (0, 6400, True)
    assert [call.shape for call in video_vae.calls] == [(1, 3, 1, 64, 64), (1, 3, 1, 32, 64), (1, 3, 5, 64, 32)]
    assert payload.metadata["task"] == "ref2va"
    assert payload.metadata["one_frame"] == "1"
    assert payload.metadata["one_frame_target_index"] == "24"
    assert "one_frame_control_indices" not in payload.metadata
    assert json.loads(payload.metadata["media_fingerprints"]) == {
        str(record.video_path): "target-image",
        str(image): "face",
        str(video): "motion",
    }


@pytest.mark.parametrize(
    ("record_kwargs", "overrides", "message"),
    [
        ({"with_audio_reference": True}, {}, "standalone audio references"),
        ({}, {"control_frames": [torch.zeros(64, 64, 3, dtype=torch.uint8)], "control_indices": [0]}, "cannot combine"),
        ({}, {"media_decoder": None}, "media decoder"),
        ({}, {"audio_vae": None}, "audio VAE"),
    ],
)
def test_build_one_frame_latents_reject_invalid_references(tmp_path: Path, record_kwargs: dict, overrides: dict, message: str):
    record = _one_frame_reference_record(tmp_path, **record_kwargs)
    decoder = _FakeH3MediaDecoder(
        visuals={reference.path: torch.zeros(1, 64, 64, 3, dtype=torch.uint8) for reference in record.references}
    )
    video_vae = _FakeH3VideoVAE()
    inputs = dict(
        image_frames=torch.zeros(64, 64, 3, dtype=torch.uint8),
        target_index=24,
        video_vae=video_vae,
        silence_audio_latent=torch.zeros(32, 2, 2),
        cache_seed=0,
        item_key=str(record.video_path),
        video_vae_fingerprint="video-fingerprint",
        audio_vae_fingerprint="audio-fingerprint",
        media_fingerprints={},
        record=record,
        audio_vae=_FakeH3AudioVAE(),
        media_decoder=decoder,
    )
    inputs.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_one_frame_latent_tensors(**inputs)

    assert video_vae.calls == []


def test_one_frame_reference_cache_keys_round_trip_through_the_bucket_collator(tmp_path: Path):
    item = ItemInfo("view", "a reference caption", (64, 64), (64, 64))
    item.latent_cache_path = str(tmp_path / "view_0064x0064_mmh3.safetensors")
    item.text_encoder_output_cache_path = str(tmp_path / "view_mmh3_te.safetensors")
    save_latent_cache_minimax_h3(
        item,
        target_video=torch.zeros(24, 1, 4, 4),
        target_audio=torch.zeros(32, 2, 2),
        audio_present=False,
        visual_conditions={"ref_000_image": torch.ones(24, 1, 4, 4), "ref_001_video": torch.ones(24, 2, 4, 4)},
        audio_conditions={"ref_001_audio": torch.zeros(32, 2, 8)},
        one_frame_target_index=24,
        metadata={"task": "ref2va", "one_frame": "1"},
    )
    _save_text_rows(item, metadata={"task": "ref2va"})

    assert _saved_keys(item.latent_cache_path) == {
        "latents_1x4x4_float32",
        "latents_ref_000_image_1x4x4_float32",
        "latents_ref_001_video_2x4x4_float32",
        "latents_ref_001_audio_32x2x8_float32",
        "latents_audio_32x2x2_float32",
        AUDIO_PRESENT_KEY,
        ONE_FRAME_TARGET_INDEX_KEY,
    }

    manager = BucketBatchManager({(64, 64): [item]}, batch_size=1)
    batch = manager[0]

    assert batch["latents"].shape == (1, 24, 1, 4, 4)
    assert batch["latents_ref_000_image"].shape == (1, 24, 1, 4, 4)
    assert batch["latents_ref_001_video"].shape == (1, 24, 2, 4, 4)
    assert batch["latents_ref_001_audio"].shape == (1, 32, 2, 8)
    torch.testing.assert_close(batch["one_frame_target_index"], torch.tensor([24], dtype=torch.int64))


def test_h3_image_records_come_from_the_image_jsonl_aligned_with_datasource_indices(tmp_path: Path):
    target = _touch(tmp_path / "data" / "target.png")
    face = _touch(tmp_path / "data" / "refs" / "face.png")
    jsonl = tmp_path / "data" / "items.jsonl"
    _write_jsonl(
        jsonl,
        [
            {
                "image_path": str(target),
                "caption": "a novel view",
                "references": [{"type": "image", "path": "refs/face.png"}],
            }
        ],
    )
    datasource = ImageJsonlDatasource(str(jsonl), control_count_per_image=1)

    records = h3_records_from_datasource(datasource, "ref2va")

    # one record per datasource index (ItemInfo.datasource_index), references resolved from the JSONL directory
    assert len(records) == 1
    record = records[0]
    assert record.video_path == target
    assert record.caption == "a novel view"
    assert [(reference.type, reference.path) for reference in record.references] == [("image", face)]
    assert record.label == "items.jsonl line 1"

    with pytest.raises(ValueError, match="H3 items.jsonl line 1: references require task ref2va"):
        h3_records_from_datasource(datasource, "t2va")


def test_h3_records_carry_the_optional_teacher_caption(tmp_path: Path):
    target = _touch(tmp_path / "target.png")
    face = _touch(tmp_path / "face.png")
    video = _touch(tmp_path / "clip.mp4")
    jsonl = tmp_path / "items.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"image_path": str(target), "caption": "c", "references": [{"type": "image", "path": "face.png"}]},
            {
                "image_path": str(face),
                "caption": "c",
                "teacher_caption": "subject_definitions:\n<Subject 1> ...",
                "references": [{"type": "image", "path": "target.png"}],
            },
        ],
    )

    records = h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), control_count_per_image=1), "ref2va")

    assert records[0].teacher_caption is None
    assert records[1].teacher_caption == "subject_definitions:\n<Subject 1> ..."

    video_jsonl = tmp_path / "videos.jsonl"
    _write_jsonl(
        video_jsonl,
        [{"video_path": "clip.mp4", "caption": "c", "teacher_caption": "t", "references": [{"type": "image", "path": "face.png"}]}],
    )
    (video_record,) = load_h3_jsonl_records(video_jsonl, "ref2va", lambda path: H3MediaInfo(has_audio=False, duration_seconds=6.0))
    assert video_record.teacher_caption == "t"
    assert video_record.video_path == video

    _write_jsonl(
        jsonl,
        [{"image_path": str(target), "caption": "c", "teacher_caption": "", "references": [{"type": "image", "path": "face.png"}]}],
    )
    with pytest.raises(ValueError, match="teacher_caption must be a non-empty string"):
        h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), control_count_per_image=1), "ref2va")


def _image_with_controls_datasource(tmp_path: Path, control_count: int) -> ImageDirectoryDatasource:
    """An image_directory / control_directory pair: target.png with target_{i}.png controls in order."""
    images = tmp_path / "images"
    controls = tmp_path / "controls"
    _touch(images / "target.png")
    (images / "target.txt").write_text("a character in a pose on a background", encoding="utf-8")
    for index in range(control_count):
        _touch(controls / f"target_{index}.png")
    return ImageDirectoryDatasource(str(images), ".txt", str(controls), None, False)


def test_h3_control_images_become_ordered_image_references_for_ref2va(tmp_path: Path):
    datasource = _image_with_controls_datasource(tmp_path, 3)  # character, pose, background

    (record,) = h3_records_from_datasource(datasource, "ref2va", control_images_as_references=True)

    assert record.video_path == (tmp_path / "images" / "target.png").resolve()
    assert [(reference.type, reference.path.name) for reference in record.references] == [
        ("image", "target_0.png"),
        ("image", "target_1.png"),
        ("image", "target_2.png"),
    ]
    assert record.teacher_caption is None

    # the same control images are timed FL2VA controls (not references) for the other tasks
    (fl_record,) = h3_records_from_datasource(datasource, "fl2va", control_images_as_references=True)
    assert fl_record.references == ()
    # ... and without the opt-in the directory dataset still cannot provide references
    with pytest.raises(ValueError, match="control images as untimed references"):
        h3_records_from_datasource(datasource, "ref2va")


def test_h3_jsonl_control_paths_become_references_unless_the_record_has_its_own(tmp_path: Path):
    target = _touch(tmp_path / "target.png")
    char = _touch(tmp_path / "char.png")
    pose = _touch(tmp_path / "pose.png")
    face = _touch(tmp_path / "refs" / "face.png")
    jsonl = tmp_path / "items.jsonl"
    _write_jsonl(
        jsonl,
        [
            {"image_path": str(target), "caption": "c", "control_path_0": str(char), "control_path_1": str(pose)},
            {"image_path": str(char), "caption": "c", "references": [{"type": "image", "path": "refs/face.png"}]},
        ],
    )

    records = h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), None), "ref2va", control_images_as_references=True)

    assert [reference.path for reference in records[0].references] == [char, pose]
    assert [reference.path for reference in records[1].references] == [face]

    # a record cannot carry both control images and references
    _write_jsonl(
        jsonl,
        [
            {
                "image_path": str(target),
                "caption": "c",
                "control_path": str(char),
                "references": [{"type": "image", "path": "refs/face.png"}],
            }
        ],
    )
    with pytest.raises(ValueError, match="items.jsonl line 1: cannot combine control images with references"):
        h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), None), "ref2va", control_images_as_references=True)

    # the Ref2VA limits apply to control-derived references too
    _write_jsonl(jsonl, [{"image_path": str(target), "caption": "c", **{f"control_path_{i}": str(char) for i in range(10)}}])
    with pytest.raises(ValueError, match="at most 9 image references"):
        h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), None), "ref2va", control_images_as_references=True)


@pytest.mark.parametrize(
    ("has_control", "indices", "task", "record_task", "message"),
    [
        (False, None, "t2va", None, None),
        (False, None, "fl2va", None, "requires image datasets with control images"),
        (False, None, "ref2va", None, None),  # references come from the JSONL records (checked at record building)
        (True, [0], "fl2va", None, None),
        (True, [0], "t2va", None, "time-annotated control images .* require --task fl2va"),
        (True, [0], "ref2va", None, "time-annotated control images .* require --task fl2va"),
        (True, None, "ref2va", None, None),
        (True, None, "t2va", "ref2va", None),  # subject-reference teacher: the records are built as ref2va
        (True, None, "t2va", None, "untimed references"),
        (True, None, "fl2va", None, "untimed references"),
    ],
)
def test_h3_image_dataset_task_matrix(has_control, indices, task, record_task, message):
    from types import SimpleNamespace

    from musubi_tuner.minimax_h3.cache_plan import validate_h3_image_dataset_task

    dataset = SimpleNamespace(has_control=has_control, fp_1f_clean_indices=indices)
    if message is None:
        validate_h3_image_dataset_task(dataset, task, True, record_task)
    else:
        with pytest.raises(ValueError, match=message):
            validate_h3_image_dataset_task(dataset, task, True, record_task)
    with pytest.raises(ValueError, match="require --one_frame"):
        validate_h3_image_dataset_task(dataset, task, False, record_task)


def test_h3_image_records_require_references_for_ref2va_and_reject_duplicates(tmp_path: Path):
    directory = tmp_path / "images"
    image = _touch(directory / "plain.png")
    (directory / "plain.txt").write_text("plain caption", encoding="utf-8")
    directory_datasource = ImageDirectoryDatasource(str(directory), ".txt", None, 1, False)

    # image directories build plain records (t2va / fl2va) but cannot carry references
    assert h3_records_from_datasource(directory_datasource, "t2va") == [
        H3Record(video_path=image, caption="plain caption", references=(), label=str(image))
    ]
    with pytest.raises(ValueError, match="Ref2VA requires per-item references"):
        h3_records_from_datasource(directory_datasource, "ref2va")

    target = _touch(tmp_path / "target.png")
    face = _touch(tmp_path / "face.png")
    jsonl = tmp_path / "items.jsonl"
    line = {"image_path": str(target), "caption": "c", "references": [{"type": "image", "path": str(face)}]}
    _write_jsonl(jsonl, [line, line])

    with pytest.raises(ValueError, match="items.jsonl line 2: duplicate target"):
        h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), control_count_per_image=1), "ref2va")

    # a record without references among records that have them fails on its own line
    _write_jsonl(jsonl, [line, {"image_path": str(face), "caption": "c"}])
    with pytest.raises(ValueError, match="items.jsonl line 2: Ref2VA requires at least one visual reference"):
        h3_records_from_datasource(ImageJsonlDatasource(str(jsonl), control_count_per_image=1), "ref2va")


def test_one_frame_format_tag_makes_skip_existing_rebuild_pre_cond_caches(tmp_path: Path):
    from safetensors.torch import save_file

    from musubi_tuner.minimax_h3_cache_latents import build_latent_metadata

    expected = build_latent_metadata(
        task="fl2va",
        crop_start_frame=0,
        cache_seed=0,
        video_vae_fingerprint="v",
        audio_vae_fingerprint="a",
        media_fingerprints={},
        one_frame_target_index=24,
        one_frame_control_indices=[0],
    )
    assert expected["one_frame_format"] == "minimax-h3-one-frame-v2"

    # a cache written before the ordered cond_ slots carries every other key but not the one-frame format tag
    legacy = {key: value for key, value in expected.items() if key != "one_frame_format"}
    path = tmp_path / "legacy.safetensors"
    save_file({"latents_1x4x4_float32": torch.zeros(24, 1, 4, 4)}, str(path), metadata=legacy)
    assert not cache_metadata_matches(path, expected)
    save_file({"latents_1x4x4_float32": torch.zeros(24, 1, 4, 4)}, str(path), metadata=expected)
    assert cache_metadata_matches(path, expected)


@pytest.mark.parametrize(
    ("one_frame", "message"),
    [
        ({"one_frame_target_index": 24, "one_frame_control_indices": [-1]}, "control indices must be nonnegative"),
        ({"one_frame_target_index": 24, "one_frame_control_indices": []}, "at least one entry"),
        ({"one_frame_control_indices": [0]}, "require the one-frame target index"),
    ],
)
def test_h3_latent_writer_rejects_invalid_one_frame_control_indices(tmp_path: Path, one_frame: dict, message: str):
    item = _h3_item(tmp_path)

    with pytest.raises(ValueError, match=message):
        save_latent_cache_minimax_h3(
            item,
            target_video=torch.zeros(24, 1, 4, 4),
            target_audio=torch.zeros(32, 2, 2),
            audio_present=False,
            visual_conditions={"cond_000": torch.zeros(24, 1, 4, 4)},
            metadata={"task": "fl2va"},
            **one_frame,
        )
