from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Callable, Literal, Optional

import av

from musubi_tuner.dataset.audio_utils import AudioSource as H3AudioSource
from musubi_tuner.dataset.audio_utils import AudioSpec
from musubi_tuner.dataset.datasources import VideoDatasource, VideoJsonlDatasource


H3Task = Literal["t2va", "fl2va", "ref2va"]
H3ReferenceType = Literal["image", "video", "audio"]
H3MediaProbe = Callable[[Path], "H3MediaInfo"]

TARGET_FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_TERMINAL_TOLERANCE_SAMPLES = 800


@dataclass(frozen=True)
class H3MediaInfo:
    has_audio: bool
    duration_seconds: Optional[float]


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
    references: tuple[H3Reference, ...]
    jsonl_line: int


def _validate_frame_count(frame_count: int) -> None:
    if frame_count < 5 or (frame_count - 5) % 17 != 0:
        raise ValueError(f"Invalid MiniMax-H3 frame count {frame_count}; expected 17*n+5")


def video_latent_frames(frame_count: int) -> int:
    _validate_frame_count(frame_count)
    return 5 * ((frame_count - 5) // 17) + 2


def audio_latent_frames(frame_count: int, *, output_fps: int = TARGET_FPS) -> int:
    _validate_frame_count(frame_count)
    if isinstance(output_fps, bool) or not isinstance(output_fps, int) or output_fps <= 0:
        raise ValueError(f"MiniMax-H3 output fps must be a positive integer, got {output_fps!r}")
    if output_fps == TARGET_FPS:
        return (10 * frame_count + 3) // 6
    # 40 Hz audio latents over the real duration frame_count/output_fps seconds;
    # reduces to the released (10*f+3)//6 mapping at 24 fps
    return int((Fraction(10 * frame_count * TARGET_FPS, output_fps) + 3) // 6)


def waveform_samples(audio_frames: int) -> int:
    if audio_frames <= 0:
        raise ValueError(f"Audio latent frame count must be positive, got {audio_frames}")
    return audio_frames * 800


def h3_samples_per_crop(frame_count: int) -> int:
    # module-level (not a lambda) so the spec stays picklable for spawned DataLoader workers
    return waveform_samples(audio_latent_frames(frame_count))


# passed to the shared dataset layer so that it decodes and windows target audio for us
H3_AUDIO_SPEC = AudioSpec(
    sample_rate=AUDIO_SAMPLE_RATE,
    channels=2,
    samples_per_crop=h3_samples_per_crop,
    codec_pad_tolerance=AUDIO_TERMINAL_TOLERANCE_SAMPLES,
)


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


def _resolve_existing_path(value: object, base_directory: Path, field: str, context: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}: {field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base_directory / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{context}: {field} does not exist: {path}")
    return path


def _probe_required_audio(path: Path, probe: H3MediaProbe, label: str, context: str) -> H3MediaInfo:
    try:
        info = probe(path)
    except Exception as error:
        raise ValueError(f"{context}: {label} failed to decode: {error}") from error
    if not info.has_audio:
        raise ValueError(f"{context}: {label} contains no audio stream: {path}")
    return info


def _validate_reference_counts(references: list, context: str) -> None:
    if len(references) > 12:
        raise ValueError(f"{context}: Ref2VA allows at most 12 reference items")

    types = [reference.get("type") if isinstance(reference, dict) else None for reference in references]
    unsupported = [reference_type for reference_type in types if reference_type not in {"image", "video", "audio"}]
    if unsupported:
        raise ValueError(f"{context}: Unsupported reference type: {unsupported[0]!r}")
    if types.count("image") > 9:
        raise ValueError(f"{context}: Ref2VA allows at most 9 image references")
    if types.count("video") > 3:
        raise ValueError(f"{context}: Ref2VA allows at most 3 video references")
    if types.count("audio") > 3:
        raise ValueError(f"{context}: Ref2VA allows at most 3 audio-bearing references")
    if not any(reference_type in {"image", "video"} for reference_type in types):
        raise ValueError(f"{context}: Ref2VA requires at least one visual reference")


def _parse_references(
    raw_references: object,
    base_directory: Path,
    context: str,
    probe: H3MediaProbe,
) -> tuple[H3Reference, ...]:
    if not isinstance(raw_references, list):
        raise ValueError(f"{context}: references must be a list")
    _validate_reference_counts(raw_references, context)

    references = []
    audio_bearing_count = 0
    for index, raw_reference in enumerate(raw_references):
        reference_type = raw_reference["type"]
        field_prefix = f"references[{index}]"
        path = _resolve_existing_path(raw_reference.get("path"), base_directory, f"{field_prefix}.path", context)

        if reference_type == "image":
            if "audio_path" in raw_reference:
                raise ValueError(f"{context}: {field_prefix} image cannot have audio_path")
            references.append(H3Reference(type="image", path=path))
            continue

        if reference_type == "audio":
            if "audio_path" in raw_reference:
                raise ValueError(f"{context}: {field_prefix} audio uses path, not audio_path")
            info = _probe_required_audio(path, probe, f"{field_prefix} audio", context)
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
            raise ValueError(f"{context}: {field_prefix} video failed to decode: {error}") from error
        duration = video_info.duration_seconds
        if duration is None or duration < 2.0 or duration > 15.0:
            raise ValueError(f"{context}: {field_prefix} video must be between 2 and 15 seconds; got {duration}")

        # an explicit "audio_path": null makes the reference visual-only (e.g. a motion
        # reference), suppressing the video's embedded audio track
        audio = None
        if "audio_path" not in raw_reference:
            if video_info.has_audio:
                audio = H3AudioSource(path=path, embedded=True)
        elif raw_reference["audio_path"] is not None:
            audio_path = _resolve_existing_path(raw_reference["audio_path"], base_directory, f"{field_prefix}.audio_path", context)
            _probe_required_audio(audio_path, probe, f"Explicit {field_prefix} audio", context)
            audio = H3AudioSource(path=audio_path, embedded=False)
        if audio is not None:
            audio_bearing_count += 1
        references.append(H3Reference(type="video", path=path, audio=audio, duration_seconds=duration))

    if audio_bearing_count > 3:
        raise ValueError(f"{context}: Ref2VA allows at most 3 audio-bearing references")
    return tuple(references)


INLINE_REFERENCE_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
INLINE_REFERENCE_AUDIO_SUFFIXES = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})


def _inline_reference_data(spec: str, context: str) -> dict:
    """Parses one inline reference spec `path[;type=...][;audio=...]` into the JSONL dict shape."""
    parts = spec.split(";")
    path = parts[0].strip()
    if not path:
        raise ValueError(f"{context}: inline reference must start with a path: {spec!r}")

    options: dict[str, str] = {}
    for part in parts[1:]:
        key, separator, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(f"{context}: inline reference options must be key=value, got {part!r} in {spec!r}")
        if key not in {"type", "audio"}:
            raise ValueError(f"{context}: unknown inline reference option {key!r} in {spec!r} (allowed: type, audio)")
        if key in options:
            raise ValueError(f"{context}: duplicate inline reference option {key!r} in {spec!r}")
        options[key] = value

    reference_type = options.get("type")
    if reference_type is None:
        suffix = Path(path).suffix.lower()
        if suffix in INLINE_REFERENCE_IMAGE_SUFFIXES:
            reference_type = "image"
        elif suffix in INLINE_REFERENCE_AUDIO_SUFFIXES:
            reference_type = "audio"
        else:
            reference_type = "video"
    elif reference_type not in {"image", "video", "audio"}:
        raise ValueError(f"{context}: inline reference type must be image, video, or audio, got {reference_type!r}")

    data: dict = {"type": reference_type, "path": path}
    if "audio" in options:
        # the JSONL-shape validation rejects audio_path on image and audio references
        data["audio_path"] = options["audio"]
    return data


def parse_inline_references(
    specs: list[str] | tuple[str, ...],
    base_directory: Path,
    probe: H3MediaProbe = probe_h3_media,
    context: str = "H3 --ref",
) -> tuple[H3Reference, ...]:
    """Parses inline `--ref` specs with exactly the JSONL `references` validation rules.

    Relative paths resolve from base_directory (the caller decides: CWD for the CLI,
    the prompt file's directory for sample-prompt lines).
    """
    raw_references = [_inline_reference_data(spec, f"{context}[{index}]") for index, spec in enumerate(specs)]
    return _parse_references(raw_references, base_directory, context, probe)


def _record_from_jsonl_data(
    data: object,
    base_directory: Path,
    line_number: int,
    task: H3Task,
    probe: H3MediaProbe,
) -> H3Record:
    if not isinstance(data, dict):
        raise ValueError(f"H3 JSONL line {line_number}: each record must be an object")

    video_path = _resolve_existing_path(data.get("video_path"), base_directory, "video_path", line_number)
    caption = data.get("caption")
    if not isinstance(caption, str):
        raise ValueError(f"H3 JSONL line {line_number}: caption must be a string")

    raw_references = data.get("references", [])
    if task == "ref2va":
        references = _parse_references(raw_references, base_directory, line_number, probe)
    else:
        if raw_references:
            raise ValueError(f"H3 JSONL line {line_number}: references require task ref2va")
        references = ()

    return H3Record(video_path=video_path, caption=caption, references=references, jsonl_line=line_number)


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
            records.append(_record_from_jsonl_data(data, base_directory, line_number, task, probe))

    if not records:
        raise ValueError(f"MiniMax-H3 JSONL contains no records: {jsonl_path}")
    return records


def h3_records_from_datasource(
    datasource: VideoDatasource,
    task: H3Task,
    probe: H3MediaProbe = probe_h3_media,
) -> list[H3Record]:
    """Builds H3 records from the datasource's already-parsed data (no JSONL re-read).

    Records align with datasource indices, so cache items reference them through
    ItemInfo.datasource_index.
    """
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")

    if isinstance(datasource, VideoJsonlDatasource):
        base_directory = Path(datasource.video_jsonl_file).resolve().parent
        records = [
            _record_from_jsonl_data(data, base_directory, index + 1, task, probe) for index, data in enumerate(datasource.data)
        ]
        if not records:
            raise ValueError(f"MiniMax-H3 JSONL contains no records: {datasource.video_jsonl_file}")
        return records

    if task == "ref2va":
        raise ValueError("MiniMax-H3 Ref2VA requires video_jsonl_file")

    records = []
    for index in range(len(datasource)):
        video_path, caption = datasource.get_caption(index)
        video_path = Path(video_path).resolve()
        if not video_path.is_file():
            raise ValueError(f"MiniMax-H3 target video does not exist: {video_path}")
        if not isinstance(caption, str):
            raise ValueError("MiniMax-H3 directory caption must be a string")
        records.append(H3Record(video_path=video_path, caption=caption, references=(), jsonl_line=0))
    return records
