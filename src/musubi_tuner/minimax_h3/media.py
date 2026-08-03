from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal, Optional

import av


H3Task = Literal["t2va", "fl2va", "ref2va"]
H3ReferenceType = Literal["image", "video", "audio"]
H3MediaProbe = Callable[[Path], "H3MediaInfo"]

AUDIO_SIDECAR_EXTENSIONS = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})


@dataclass(frozen=True)
class H3MediaInfo:
    has_audio: bool
    duration_seconds: Optional[float]


@dataclass(frozen=True)
class H3AudioSource:
    path: Path
    embedded: bool


@dataclass(frozen=True)
class H3Reference:
    type: H3ReferenceType
    path: Path
    audio: Optional[H3AudioSource] = None
    duration_seconds: Optional[float] = None


@dataclass(frozen=True)
class H3Record:
    video_path: Path
    caption: str
    target_audio: H3AudioSource
    references: tuple[H3Reference, ...]
    jsonl_line: int


def _validate_frame_count(frame_count: int) -> None:
    if frame_count < 5 or (frame_count - 5) % 17 != 0:
        raise ValueError(f"Invalid MiniMax-H3 frame count {frame_count}; expected 17*n+5")


def video_latent_frames(frame_count: int) -> int:
    _validate_frame_count(frame_count)
    return 5 * ((frame_count - 5) // 17) + 2


def audio_latent_frames(frame_count: int) -> int:
    _validate_frame_count(frame_count)
    return (10 * frame_count + 3) // 6


def waveform_samples(audio_frames: int) -> int:
    if audio_frames <= 0:
        raise ValueError(f"Audio latent frame count must be positive, got {audio_frames}")
    return audio_frames * 800


def probe_h3_media(path: Path) -> H3MediaInfo:
    with av.open(str(path)) as container:
        has_audio = bool(container.streams.audio)
        durations = []
        for stream in (*container.streams.video, *container.streams.audio):
            if stream.duration is not None and stream.time_base is not None:
                durations.append(float(stream.duration * stream.time_base))
        if durations:
            duration_seconds = max(durations)
        elif container.duration is not None:
            duration_seconds = float(container.duration / av.time_base)
        else:
            duration_seconds = None
    return H3MediaInfo(has_audio=has_audio, duration_seconds=duration_seconds)


def _resolve_existing_path(value: object, base_directory: Path, field: str, line_number: int) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"H3 JSONL line {line_number}: {field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base_directory / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"H3 JSONL line {line_number}: {field} does not exist: {path}")
    return path


def _probe_required_audio(path: Path, probe: H3MediaProbe, label: str, line_number: int) -> H3MediaInfo:
    try:
        info = probe(path)
    except Exception as error:
        raise ValueError(f"H3 JSONL line {line_number}: {label} failed to decode: {error}") from error
    if not info.has_audio:
        raise ValueError(f"H3 JSONL line {line_number}: {label} contains no audio stream: {path}")
    return info


def _same_stem_audio_sidecars(video_path: Path) -> list[Path]:
    return sorted(
        (
            candidate.resolve()
            for candidate in video_path.parent.iterdir()
            if candidate.is_file() and candidate.stem == video_path.stem and candidate.suffix.lower() in AUDIO_SIDECAR_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def _resolve_target_audio(
    data: dict,
    video_path: Path,
    base_directory: Path,
    line_number: int,
    probe: H3MediaProbe,
) -> H3AudioSource:
    if data.get("audio_path") is not None:
        path = _resolve_existing_path(data["audio_path"], base_directory, "audio_path", line_number)
        _probe_required_audio(path, probe, "Explicit target audio", line_number)
        return H3AudioSource(path=path, embedded=False)

    sidecars = _same_stem_audio_sidecars(video_path)
    if len(sidecars) > 1:
        formatted = ", ".join(str(path) for path in sidecars)
        raise ValueError(f"H3 JSONL line {line_number}: Multiple same-stem audio sidecars found: {formatted}")
    if sidecars:
        _probe_required_audio(sidecars[0], probe, "Target audio sidecar", line_number)
        return H3AudioSource(path=sidecars[0], embedded=False)

    try:
        video_info = probe(video_path)
    except Exception as error:
        raise ValueError(f"H3 JSONL line {line_number}: Target video failed to decode: {error}") from error
    if video_info.has_audio:
        return H3AudioSource(path=video_path, embedded=True)
    raise ValueError(f"H3 JSONL line {line_number}: Missing target audio for {video_path}")


def make_h3_directory_record(
    video_path: str | Path,
    caption: str,
    probe: H3MediaProbe = probe_h3_media,
) -> H3Record:
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise ValueError(f"MiniMax-H3 target video does not exist: {video_path}")
    if not isinstance(caption, str):
        raise ValueError("MiniMax-H3 directory caption must be a string")
    target_audio = _resolve_target_audio({}, video_path, video_path.parent, 0, probe)
    return H3Record(
        video_path=video_path,
        caption=caption,
        target_audio=target_audio,
        references=(),
        jsonl_line=0,
    )


def _validate_reference_counts(references: list, line_number: int) -> None:
    if len(references) > 12:
        raise ValueError(f"H3 JSONL line {line_number}: Ref2VA allows at most 12 reference items")

    types = [reference.get("type") if isinstance(reference, dict) else None for reference in references]
    unsupported = [reference_type for reference_type in types if reference_type not in {"image", "video", "audio"}]
    if unsupported:
        raise ValueError(f"H3 JSONL line {line_number}: Unsupported reference type: {unsupported[0]!r}")
    if types.count("image") > 9:
        raise ValueError(f"H3 JSONL line {line_number}: Ref2VA allows at most 9 image references")
    if types.count("video") > 3:
        raise ValueError(f"H3 JSONL line {line_number}: Ref2VA allows at most 3 video references")
    if types.count("audio") > 3:
        raise ValueError(f"H3 JSONL line {line_number}: Ref2VA allows at most 3 audio-bearing references")
    if not any(reference_type in {"image", "video"} for reference_type in types):
        raise ValueError(f"H3 JSONL line {line_number}: Ref2VA requires at least one visual reference")


def _parse_references(
    raw_references: object,
    base_directory: Path,
    line_number: int,
    probe: H3MediaProbe,
) -> tuple[H3Reference, ...]:
    if not isinstance(raw_references, list):
        raise ValueError(f"H3 JSONL line {line_number}: references must be a list")
    _validate_reference_counts(raw_references, line_number)

    references = []
    audio_bearing_count = 0
    for index, raw_reference in enumerate(raw_references):
        reference_type = raw_reference["type"]
        field_prefix = f"references[{index}]"
        path = _resolve_existing_path(raw_reference.get("path"), base_directory, f"{field_prefix}.path", line_number)

        if reference_type == "image":
            if raw_reference.get("audio_path") is not None:
                raise ValueError(f"H3 JSONL line {line_number}: {field_prefix} image cannot have audio_path")
            references.append(H3Reference(type="image", path=path))
            continue

        if reference_type == "audio":
            if raw_reference.get("audio_path") is not None:
                raise ValueError(f"H3 JSONL line {line_number}: {field_prefix} audio uses path, not audio_path")
            info = _probe_required_audio(path, probe, f"{field_prefix} audio", line_number)
            references.append(
                H3Reference(
                    type="audio",
                    path=path,
                    audio=H3AudioSource(path=path, embedded=False),
                    duration_seconds=info.duration_seconds,
                )
            )
            audio_bearing_count += 1
            continue

        try:
            video_info = probe(path)
        except Exception as error:
            raise ValueError(f"H3 JSONL line {line_number}: {field_prefix} video failed to decode: {error}") from error
        duration = video_info.duration_seconds
        if duration is None or duration < 2.0 or duration > 15.0:
            raise ValueError(f"H3 JSONL line {line_number}: {field_prefix} video must be between 2 and 15 seconds; got {duration}")

        audio = None
        if raw_reference.get("audio_path") is not None:
            audio_path = _resolve_existing_path(
                raw_reference["audio_path"], base_directory, f"{field_prefix}.audio_path", line_number
            )
            _probe_required_audio(audio_path, probe, f"Explicit {field_prefix} audio", line_number)
            audio = H3AudioSource(path=audio_path, embedded=False)
        elif video_info.has_audio:
            audio = H3AudioSource(path=path, embedded=True)
        if audio is not None:
            audio_bearing_count += 1
        references.append(H3Reference(type="video", path=path, audio=audio, duration_seconds=duration))

    if audio_bearing_count > 3:
        raise ValueError(f"H3 JSONL line {line_number}: Ref2VA allows at most 3 audio-bearing references")
    return tuple(references)


def load_h3_jsonl_records(
    jsonl_path: str | Path,
    task: H3Task,
    probe: H3MediaProbe = probe_h3_media,
) -> list[H3Record]:
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")

    jsonl_path = Path(jsonl_path).resolve()
    if not jsonl_path.is_file():
        raise ValueError(f"MiniMax-H3 JSONL does not exist: {jsonl_path}")
    base_directory = jsonl_path.parent
    records = []

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"H3 JSONL line {line_number}: invalid JSON: {error.msg}") from error
            if not isinstance(data, dict):
                raise ValueError(f"H3 JSONL line {line_number}: each record must be an object")

            video_path = _resolve_existing_path(data.get("video_path"), base_directory, "video_path", line_number)
            caption = data.get("caption")
            if not isinstance(caption, str):
                raise ValueError(f"H3 JSONL line {line_number}: caption must be a string")
            target_audio = _resolve_target_audio(data, video_path, base_directory, line_number, probe)

            raw_references = data.get("references", [])
            if task == "ref2va":
                references = _parse_references(raw_references, base_directory, line_number, probe)
            else:
                if raw_references:
                    raise ValueError(f"H3 JSONL line {line_number}: references require task ref2va")
                references = ()

            records.append(
                H3Record(
                    video_path=video_path,
                    caption=caption,
                    target_audio=target_audio,
                    references=references,
                    jsonl_line=line_number,
                )
            )

    if not records:
        raise ValueError(f"MiniMax-H3 JSONL contains no records: {jsonl_path}")
    return records
