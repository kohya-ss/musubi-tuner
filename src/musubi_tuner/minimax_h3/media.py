from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Protocol, Sequence

import av
import numpy as np
from PIL import Image
import torch

from musubi_tuner.dataset.audio_utils import AudioSource as H3AudioSource
from musubi_tuner.dataset.audio_utils import AudioSpec, decode_audio, slice_audio_window
from musubi_tuner.dataset.datasources import ContentDatasource
from musubi_tuner.dataset.media_utils import load_video


H3Task = Literal["t2va", "fl2va", "ref2va"]
H3ReferenceType = Literal["image", "video", "audio"]
H3MediaProbe = Callable[[Path], "H3MediaInfo"]

TARGET_FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_TERMINAL_TOLERANCE_SAMPLES = 800
# with a one-frame (image) target, reference videos keep their full released span instead of
# being capped by the target duration (shared by generation and the one-frame caches)
ONE_FRAME_REFERENCE_FRAME_CAP = 15 * TARGET_FPS
# released reference canvas: short edge and pixel budget of decoded reference videos, on a
# 32-pixel grid (the caches and generation decode references through the same policy)
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
# reference videos enter the Qwen3-VL presentation as 2 fps frame samples (the released
# text-visual clock); the stride converts from the native 24 fps decode
TEXT_VISUAL_FPS = 2
TEXT_VISUAL_FRAME_STRIDE = TARGET_FPS // TEXT_VISUAL_FPS


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
    # where the record came from (e.g. "items.jsonl line 3"), for error messages only
    label: str = ""
    # optional per-item `teacher_caption`: the caption of the subject-reference teacher presentation
    # (text cache only); None means the teacher wraps `caption` with the boilerplate declaration
    teacher_caption: Optional[str] = None

    @property
    def context(self) -> str:
        return f"H3 {self.label}" if self.label else "H3 record"


def _parse_teacher_caption(fields: Mapping[str, object], context: str) -> Optional[str]:
    teacher_caption = fields.get("teacher_caption")
    if teacher_caption is not None and (not isinstance(teacher_caption, str) or not teacher_caption.strip()):
        raise ValueError(f"{context}: teacher_caption must be a non-empty string when present")
    return teacher_caption


def validate_subject_reference_record(record: H3Record, context: str) -> None:
    """The subject-reference teacher (v1) takes 1..9 image references; video/audio references
    need per-item role declarations (motion? voice?) and are deferred."""
    if not record.references:
        raise ValueError(f"{context}: the subject-reference teacher requires at least one image reference")
    unsupported = [reference.type for reference in record.references if reference.type != "image"]
    if unsupported:
        raise ValueError(
            f"{context}: the subject-reference teacher supports image references only (got {unsupported[0]!r};"
            " video/audio references are not supported yet)"
        )


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


def round_to_canvas_multiple(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, round(value / multiple) * multiple)


def adapt_reference_canvas(width: int, height: int) -> tuple[int, int]:
    """The released reference-video canvas for a source aspect ratio: BASE_SHORT_EDGE on the
    short side, scaled down to MAX_PIXELS, on the CANVAS_MULTIPLE grid."""
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return round_to_canvas_multiple(nominal_width), round_to_canvas_multiple(nominal_height)


def resize_frames(frames: Sequence[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    """LANCZOS-resize decoded RGB(A) frames to exactly (width, height) as a uint8 [F,H,W,3] tensor."""
    width, height = size
    resized = [
        torch.from_numpy(np.asarray(Image.fromarray(frame[..., :3]).resize((width, height), Image.Resampling.LANCZOS)).copy())
        for frame in frames
    ]
    return torch.stack(resized)


class H3MediaDecoder(Protocol):
    def decode_audio(
        self,
        source: H3AudioSource,
        *,
        start_sample: int,
        sample_count: int,
        require_exact: bool,
    ) -> torch.Tensor: ...

    def decode_reference_visual(
        self,
        reference: H3Reference,
        *,
        target_frame_count: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor: ...


class PyAVH3MediaDecoder:
    """Decodes MiniMax-H3 reference media (target media is decoded by the shared dataset layer)."""

    def __init__(self, terminal_tolerance_samples: int = AUDIO_TERMINAL_TOLERANCE_SAMPLES):
        self.terminal_tolerance_samples = terminal_tolerance_samples

    def decode_audio(
        self,
        source: H3AudioSource,
        *,
        start_sample: int,
        sample_count: int,
        require_exact: bool,
    ) -> torch.Tensor:
        if start_sample < 0 or sample_count <= 0:
            raise ValueError("MiniMax-H3 audio window must have a nonnegative start and positive length")
        waveform = decode_audio(source, sample_rate=AUDIO_SAMPLE_RATE, channels=2)
        return slice_audio_window(
            waveform,
            start_sample=start_sample,
            sample_count=sample_count,
            pad_tolerance=self.terminal_tolerance_samples,
            require_exact=require_exact,
            context=str(source.path),
        )

    def decode_reference_visual(
        self,
        reference: H3Reference,
        *,
        target_frame_count: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if reference.type == "image":
            with Image.open(reference.path) as image:
                frame = np.asarray(image.convert("RGB"))
            height, width = frame.shape[:2]
            target_area = target_size[0] * target_size[1]
            scale = min(1.0, math.sqrt(target_area / (width * height)))
            size = round_to_canvas_multiple(width * scale), round_to_canvas_multiple(height * scale)
            return resize_frames([frame], size)

        if reference.type != "video":
            raise ValueError(f"Reference type {reference.type!r} has no visual stream")
        frames = load_video(str(reference.path), target_fps=TARGET_FPS, fps_resample_mode="timestamps")
        usable_frames = min(len(frames), target_frame_count)
        if usable_frames < 5:
            raise ValueError(f"MiniMax-H3 reference video requires at least 5 frames: {reference.path}")
        usable_frames = 5 + ((usable_frames - 5) // 17) * 17
        frames = frames[:usable_frames]
        source_height, source_width = frames[0].shape[:2]
        width, height = adapt_reference_canvas(source_width, source_height)
        if source_width * source_height < width * height:
            width = round_to_canvas_multiple(source_width)
            height = round_to_canvas_multiple(source_height)
        return resize_frames(frames, (width, height))


def prepare_pixels(frames: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Decoded [F,H,W,C] pixels (uint8, or floats in [0,1]) to the VAE's [1,3,F,H,W] in [-1,1]; alpha is dropped."""
    frames = torch.as_tensor(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"MiniMax-H3 decoded video must be [F,H,W,C], got {tuple(frames.shape)}")
    frames = frames[..., :3]
    if frames.dtype == torch.uint8:
        frames = frames.float().div_(127.5).sub_(1.0)
    elif frames.is_floating_point():
        if not torch.all((frames >= 0) & (frames <= 1)):
            raise ValueError("Floating MiniMax-H3 decoded pixels must be in [0,1]")
        frames = frames.float().mul_(2.0).sub_(1.0)
    else:
        raise ValueError(f"Unsupported MiniMax-H3 decoded pixel dtype: {frames.dtype}")
    return frames.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def module_device_dtype(module: torch.nn.Module, fallback_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    """Device and floating dtype of a module's first floating tensor (the VAEs move between devices)."""
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), fallback_dtype


def fingerprint_file(path: str | Path) -> str:
    """Lightweight file identity (size + mtime) for cache-staleness checks; deliberately not a content hash."""
    stat = Path(path).resolve().stat()
    return f"stat:{stat.st_size}:{stat.st_mtime_ns}"


def reject_one_frame_audio_references(record: H3Record) -> None:
    """Standalone audio references have no window with a one-frame target (it is defined by the
    target duration); video references keep their embedded audio. Shared by generation, the
    one-frame caches, and training-time samples."""
    if any(reference.type == "audio" for reference in record.references):
        raise ValueError(
            "MiniMax-H3 one-frame targets do not accept standalone audio references"
            " (their window is defined by the target duration); video references keep their embedded audio"
        )


def _record_from_fields(
    *,
    target_path: Path,
    caption: object,
    fields: Mapping[str, object],
    base_directory: Path,
    label: str,
    task: H3Task,
    probe: H3MediaProbe,
    control_images: Sequence[str] | None = None,
) -> H3Record:
    """Builds a record from a validated target path plus the item's H3-specific fields
    (``references``, ``teacher_caption``); relative reference paths resolve from base_directory.

    With ``control_images`` (the item's control image paths, as the dataset layer opens them,
    i.e. relative to the working directory) and task ref2va, a record without a ``references``
    field takes those images as its ordered image references; a record cannot have both.
    """
    context = f"H3 {label}"
    if not isinstance(caption, str):
        raise ValueError(f"{context}: caption must be a string")

    raw_references = fields.get("references", [])
    if task == "ref2va":
        if control_images and "references" in fields:
            raise ValueError(f"{context}: cannot combine control images with references")
        if control_images:
            raw_references = [{"type": "image", "path": str(Path(path).expanduser().resolve())} for path in control_images]
        references = _parse_references(raw_references, base_directory, context, probe)
    else:
        if raw_references:
            raise ValueError(f"{context}: references require task ref2va")
        references = ()

    return H3Record(
        video_path=target_path,
        caption=caption,
        references=references,
        label=label,
        teacher_caption=_parse_teacher_caption(fields, context),
    )


def load_h3_jsonl_records(
    jsonl_path: str | Path,
    task: H3Task,
    probe: H3MediaProbe = probe_h3_media,
) -> list[H3Record]:
    """Reads a standalone Ref2VA JSONL (generation --reference_jsonl); relative paths resolve from its directory."""
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
            label = f"{jsonl_path.name} line {line_number}"
            context = f"H3 {label}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{context}: invalid JSON: {error.msg}") from error
            if not isinstance(data, dict):
                raise ValueError(f"{context}: each record must be an object")
            records.append(
                _record_from_fields(
                    target_path=_resolve_existing_path(data.get("video_path"), base_directory, "video_path", context),
                    caption=data.get("caption"),
                    fields=data,
                    base_directory=base_directory,
                    label=label,
                    task=task,
                    probe=probe,
                )
            )

    if not records:
        raise ValueError(f"MiniMax-H3 JSONL contains no records: {jsonl_path}")
    return records


def h3_records_from_datasource(
    datasource: ContentDatasource,
    task: H3Task,
    probe: H3MediaProbe = probe_h3_media,
    *,
    control_images_as_references: bool = False,
) -> list[H3Record]:
    """Builds the H3 records of a dataset's datasource (image or video), aligned with the
    datasource indices so cache items find theirs through ItemInfo.datasource_index.

    The target path and caption come from the shared accessor; the H3-specific fields
    (``references``, ``teacher_caption``) come from the item extras, which only record-based
    datasources (``video_jsonl_file`` / ``image_jsonl_file``) can carry. With
    ``control_images_as_references`` (image datasets), an item's control images
    (``control_directory`` / ``control_path``) become its ordered image references for Ref2VA
    instead, so Ref2VA requires one of the two. The target path is resolved the way the
    dataset layer opens it.
    """
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
    if len(datasource) == 0:
        raise ValueError("MiniMax-H3 dataset contains no items")

    control_paths: Mapping[str, Sequence[str]] = {}
    if control_images_as_references and task == "ref2va":
        control_paths = datasource.get_control_paths()
    if task == "ref2va" and not control_paths:
        if not any("references" in datasource.get_item_extras(index).fields for index in range(len(datasource))):
            raise ValueError(
                "MiniMax-H3 Ref2VA requires per-item references: video_jsonl_file / image_jsonl_file records with"
                " a references field, or (image datasets) control images as untimed references"
            )

    records = []
    seen_targets: dict[Path, str] = {}
    for index in range(len(datasource)):
        target_path, caption = datasource.get_caption(index)
        extras = datasource.get_item_extras(index)
        context = f"H3 {extras.label}"
        if not isinstance(target_path, str) or not target_path.strip():
            raise ValueError(f"{context}: target path must be a non-empty path")
        target = Path(target_path).expanduser().resolve()
        if not target.is_file():
            raise ValueError(f"{context}: target does not exist: {target}")
        if target in seen_targets:
            raise ValueError(f"{context}: duplicate target {target} (also {seen_targets[target]})")
        seen_targets[target] = extras.label
        records.append(
            _record_from_fields(
                target_path=target,
                caption=caption,
                fields=extras.fields,
                base_directory=Path(extras.base_directory),
                label=extras.label,
                task=task,
                probe=probe,
                control_images=control_paths.get(target_path),
            )
        )
    return records
