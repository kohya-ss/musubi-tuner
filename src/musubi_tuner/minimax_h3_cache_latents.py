from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from types import MethodType
from typing import Protocol

import av
import numpy as np
from PIL import Image
from safetensors import safe_open
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_latent_cache_minimax_h3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, VideoDataset
from musubi_tuner.dataset.media_utils import resize_image_to_bucket
from musubi_tuner.minimax_h3.audio_vae import encode_audio_mode, load_audio_vae
from musubi_tuner.minimax_h3.checkpoint import resolve_safetensors_files
from musubi_tuner.minimax_h3.media import (
    H3AudioSource,
    H3Record,
    H3Reference,
    H3Task,
    audio_latent_frames,
    load_h3_jsonl_records,
    make_h3_directory_record,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3.video_vae import (
    VIDEO_VAE_ENCODE_DTYPE,
    encode_video_condition,
    encode_video_target,
    load_video_vae,
)
from musubi_tuner.utils.model_utils import dtype_to_str


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TARGET_FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_TERMINAL_TOLERANCE_SAMPLES = 800
AUDIO_TIMESTAMP_TOLERANCE_SAMPLES = 2
CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
TARGET_AUDIO_POLICIES = frozenset({"real-supervised", "missing-unsupervised", "video-only-unsupervised"})


@dataclass(frozen=True)
class H3LatentCachePayload:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str]


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


def resample_frame_indices(
    timestamps: Sequence[float],
    *,
    source_frame_duration: float,
    target_fps: int = TARGET_FPS,
) -> list[int]:
    if not timestamps:
        return []
    if source_frame_duration <= 0 or target_fps <= 0:
        raise ValueError("Video frame durations and target FPS must be positive")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Video timestamps must be nondecreasing")

    origin = timestamps[0]
    normalized = [timestamp - origin for timestamp in timestamps]
    duration = normalized[-1] + source_frame_duration
    target_count = max(1, math.ceil(duration * target_fps - 1e-9))
    indices = []
    for target_index in range(target_count):
        target_time = target_index / target_fps
        right = bisect_left(normalized, target_time)
        if right == 0:
            source_index = 0
        elif right == len(normalized):
            source_index = len(normalized) - 1
        else:
            left = right - 1
            left_distance = target_time - normalized[left]
            right_distance = normalized[right] - target_time
            source_index = left if left_distance <= right_distance + 1e-12 else right
        indices.append(source_index)
    return indices


def _decode_video_frames(path: Path) -> tuple[list[np.ndarray], list[float], float]:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"MiniMax-H3 visual source has no video stream: {path}")
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate is not None else TARGET_FPS
        source_frame_duration = 1.0 / average_rate if average_rate > 0 else 1.0 / TARGET_FPS
        frames = []
        timestamps = []
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
            else:
                timestamp = index * source_frame_duration
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(timestamp)
    if not frames:
        raise ValueError(f"MiniMax-H3 visual source decoded no frames: {path}")
    return frames, timestamps, source_frame_duration


def _video_at_24fps(path: Path) -> list[np.ndarray]:
    frames, timestamps, source_frame_duration = _decode_video_frames(path)
    indices = resample_frame_indices(timestamps, source_frame_duration=source_frame_duration)
    return [frames[index] for index in indices]


def _round_to_multiple(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _adapt_canvas(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return _round_to_multiple(nominal_width), _round_to_multiple(nominal_height)


def _resize_frames(frames: Sequence[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    width, height = size
    resized = [
        torch.from_numpy(np.asarray(Image.fromarray(frame[..., :3]).resize((width, height), Image.Resampling.LANCZOS)).copy())
        for frame in frames
    ]
    return torch.stack(resized)


class PyAVH3MediaDecoder:
    def __init__(self, terminal_tolerance_samples: int = AUDIO_TERMINAL_TOLERANCE_SAMPLES):
        self.terminal_tolerance_samples = terminal_tolerance_samples

    def decode_target_video(
        self,
        path: str | Path,
        start_frame: int | None,
        end_frame: int | None,
        bucket_selector,
    ) -> list[np.ndarray]:
        frames = _video_at_24fps(Path(path).resolve())
        frames = frames[slice(start_frame, end_frame)]
        if not frames:
            raise ValueError(f"MiniMax-H3 target crop decoded no frames: {path}")
        bucket_resolution = bucket_selector.get_bucket_resolution((frames[0].shape[1], frames[0].shape[0]))
        return [resize_image_to_bucket(frame, bucket_resolution) for frame in frames]

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

        chunks = []
        next_start_sample = 0
        with av.open(str(source.path)) as container:
            if not container.streams.audio:
                raise ValueError(f"MiniMax-H3 audio source has no audio stream: {source.path}")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout="stereo", rate=AUDIO_SAMPLE_RATE)
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunk = _audio_frame_to_tensor(resampled)
                    chunk_start = _audio_frame_start_sample(resampled, next_start_sample)
                    chunks.append((chunk_start, chunk))
                    next_start_sample = chunk_start + chunk.shape[1]
            for resampled in resampler.resample(None):
                chunk = _audio_frame_to_tensor(resampled)
                chunk_start = _audio_frame_start_sample(resampled, next_start_sample)
                chunks.append((chunk_start, chunk))
                next_start_sample = chunk_start + chunk.shape[1]

        if not chunks:
            raise ValueError(f"MiniMax-H3 audio source decoded no samples: {source.path}")
        waveform = assemble_audio_chunks(chunks)
        window = waveform[:, start_sample : start_sample + sample_count]
        if require_exact and window.shape[1] < sample_count:
            deficit = sample_count - window.shape[1]
            if deficit > self.terminal_tolerance_samples:
                raise ValueError(
                    f"MiniMax-H3 audio source is materially short at sample {start_sample}: "
                    f"need {sample_count}, decoded {window.shape[1]}"
                )
            window = torch.nn.functional.pad(window, (0, deficit))
        if window.shape[1] == 0:
            raise ValueError(f"MiniMax-H3 audio window is empty at sample {start_sample}: {source.path}")
        return window.contiguous()

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
            size = _round_to_multiple(width * scale), _round_to_multiple(height * scale)
            return _resize_frames([frame], size)

        if reference.type != "video":
            raise ValueError(f"Reference type {reference.type!r} has no visual stream")
        frames = _video_at_24fps(reference.path)
        usable_frames = min(len(frames), target_frame_count)
        if usable_frames < 5:
            raise ValueError(f"MiniMax-H3 reference video requires at least 5 frames: {reference.path}")
        usable_frames = 5 + ((usable_frames - 5) // 17) * 17
        frames = frames[:usable_frames]
        source_height, source_width = frames[0].shape[:2]
        width, height = _adapt_canvas(source_width, source_height)
        if source_width * source_height < width * height:
            width = _round_to_multiple(source_width)
            height = _round_to_multiple(source_height)
        return _resize_frames(frames, (width, height))


def _audio_frame_to_tensor(frame: av.AudioFrame) -> torch.Tensor:
    array = frame.to_ndarray()
    if array.ndim != 2:
        raise ValueError(f"Unexpected PyAV audio frame shape: {array.shape}")
    if array.shape[0] != 2 and array.shape[1] == 2:
        array = array.T
    if array.shape[0] != 2:
        raise ValueError(f"PyAV stereo resampler returned shape {array.shape}")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _audio_frame_start_sample(frame: av.AudioFrame, fallback: int) -> int:
    if frame.pts is None or frame.time_base is None:
        return fallback
    return round(frame.pts * frame.time_base * AUDIO_SAMPLE_RATE)


def assemble_audio_chunks(
    chunks: Sequence[tuple[int, torch.Tensor]],
    *,
    timestamp_tolerance_samples: int = AUDIO_TIMESTAMP_TOLERANCE_SAMPLES,
) -> torch.Tensor:
    if not chunks:
        raise ValueError("MiniMax-H3 audio chunk list is empty")
    if timestamp_tolerance_samples < 0:
        raise ValueError("MiniMax-H3 audio timestamp tolerance must be nonnegative")

    expected_start = chunks[0][0]
    assembled = []
    for start_sample, chunk in chunks:
        if chunk.ndim != 2 or chunk.shape[0] != 2:
            raise ValueError(f"MiniMax-H3 audio chunk must be stereo [2,L], got {tuple(chunk.shape)}")
        delta = start_sample - expected_start
        if abs(delta) > timestamp_tolerance_samples:
            raise ValueError(f"MiniMax-H3 audio stream is discontinuous: expected sample {expected_start}, got {start_sample}")
        if delta > 0:
            assembled.append(torch.zeros(2, delta, dtype=chunk.dtype, device=chunk.device))
        elif delta < 0:
            chunk = chunk[:, -delta:]
        if chunk.shape[1] > 0:
            assembled.append(chunk)
        expected_start = start_sample + chunk.shape[1] + max(0, -delta)
    return torch.cat(assembled, dim=1)


def _validate_task_record(record: H3Record, task: H3Task) -> None:
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
    references = record.references
    if task != "ref2va":
        if references:
            raise ValueError(f"MiniMax-H3 task {task} does not accept references")
        return

    if len(references) > 12:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 12 reference items")
    image_count = sum(reference.type == "image" for reference in references)
    video_count = sum(reference.type == "video" for reference in references)
    audio_bearing_count = sum(reference.audio is not None for reference in references)
    if image_count > 9:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 9 image references")
    if video_count > 3:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 3 video references")
    if audio_bearing_count > 3:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 3 audio-bearing references")
    if image_count + video_count == 0:
        raise ValueError("MiniMax-H3 Ref2VA requires at least one visual reference")


def _model_device_dtype(model: torch.nn.Module, fallback_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), fallback_dtype


def _prepare_pixels(frames: torch.Tensor | np.ndarray) -> torch.Tensor:
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


def _encode_target_video(video_vae, pixels: torch.Tensor, cache_seed: int, item_key: str) -> torch.Tensor:
    device, dtype = _model_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    return encode_video_target(video_vae, pixels.to(device=device, dtype=dtype), cache_seed, item_key)


def _encode_condition_video(video_vae, pixels: torch.Tensor) -> torch.Tensor:
    device, dtype = _model_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    return encode_video_condition(video_vae, pixels.to(device=device, dtype=dtype))


def _encode_audio(audio_vae, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] != 2:
        raise ValueError(f"MiniMax-H3 decoded audio must be stereo [2,L], got {tuple(waveform.shape)}")
    device, dtype = _model_device_dtype(audio_vae, torch.float32)
    return encode_audio_mode(audio_vae, waveform.unsqueeze(0).to(device=device, dtype=dtype))


def _visual_key(role: str, latent: torch.Tensor) -> str:
    if latent.ndim != 4 or latent.shape[0] != 24:
        raise ValueError(f"MiniMax-H3 visual latent must be [24,F,H,W], got {tuple(latent.shape)}")
    _, frames, height, width = latent.shape
    dtype_name = dtype_to_str(latent.dtype)
    return f"latents_{role + '_' if role else ''}{frames}x{height}x{width}_{dtype_name}"


def _audio_key(role: str, latent: torch.Tensor) -> str:
    if latent.ndim != 3 or latent.shape[:2] != (32, 2):
        raise ValueError(f"MiniMax-H3 audio latent must be [32,2,A], got {tuple(latent.shape)}")
    dtype_name = dtype_to_str(latent.dtype)
    return f"latents_{role + '_' if role else 'audio_'}32x2x{latent.shape[2]}_{dtype_name}"


def _audio_start_sample(crop_start_frame: int) -> int:
    return (crop_start_frame * AUDIO_SAMPLE_RATE + TARGET_FPS // 2) // TARGET_FPS


def _media_fingerprint_metadata(fingerprints: Mapping[Path, str]) -> str:
    normalized = {str(Path(path).resolve()): value for path, value in fingerprints.items()}
    return json.dumps(dict(sorted(normalized.items())), ensure_ascii=True, separators=(",", ":"))


def _target_audio_policy(record: H3Record, *, video_only: bool) -> str:
    if video_only:
        return "video-only-unsupervised"
    if record.target_audio is None:
        return "missing-unsupervised"
    return "real-supervised"


def _audio_policy_is_supervised(policy: str | None, *, legacy_missing: bool) -> bool | None:
    if policy is None:
        return True if legacy_missing else None
    if policy not in TARGET_AUDIO_POLICIES:
        return None
    return policy == "real-supervised"


def build_latent_metadata(
    *,
    record: H3Record,
    task: H3Task,
    crop_start_frame: int,
    frame_count: int,
    height: int,
    width: int,
    cache_seed: int,
    video_vae_fingerprint: str,
    audio_vae_fingerprint: str,
    media_fingerprints: Mapping[Path, str],
    video_only: bool = False,
) -> dict[str, str]:
    reference_kinds = [
        reference.type + ("+audio" if reference.type == "video" and reference.audio is not None else "")
        for reference in record.references
    ]
    return {
        "task": task,
        "cache_seed": str(cache_seed),
        "crop_start_frame": str(crop_start_frame),
        "audio_start_seconds": str(Fraction(crop_start_frame, TARGET_FPS)),
        "target_geometry": f"{frame_count}x{height}x{width}",
        "reference_kinds": json.dumps(reference_kinds, ensure_ascii=True, separators=(",", ":")),
        "posterior_policy": "video_vae=fp32;target=seeded;conditions=seed42-fp16;audio=mode",
        "normalization": "released-minimax-h3-v1",
        "video_vae_fingerprint": video_vae_fingerprint,
        "audio_vae_fingerprint": audio_vae_fingerprint,
        "media_fingerprints": _media_fingerprint_metadata(media_fingerprints),
        "target_audio_policy": _target_audio_policy(record, video_only=video_only),
    }


def cache_metadata_matches(path: str | Path, expected: Mapping[str, str]) -> bool:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual = handle.metadata() or {}
    except Exception as error:
        logger.warning("Unable to read MiniMax-H3 cache metadata from %s: %s", path, error)
        return False
    for key, value in expected.items():
        if key != "target_audio_policy":
            if actual.get(key) != value:
                return False
            continue
        expected_supervised = _audio_policy_is_supervised(value, legacy_missing=False)
        actual_supervised = _audio_policy_is_supervised(actual.get(key), legacy_missing=True)
        if expected_supervised is None or actual_supervised is None or expected_supervised != actual_supervised:
            return False
    return True


def build_latent_tensors(
    *,
    record: H3Record,
    task: H3Task,
    target_frames: torch.Tensor | np.ndarray,
    crop_start_frame: int,
    video_vae,
    audio_vae,
    cache_seed: int,
    media_decoder: H3MediaDecoder,
    video_vae_fingerprint: str,
    audio_vae_fingerprint: str,
    media_fingerprints: Mapping[Path, str],
    allow_experimental_duration: bool = False,
    video_only: bool = False,
) -> H3LatentCachePayload:
    _validate_task_record(record, task)
    if crop_start_frame < 0:
        raise ValueError(f"MiniMax-H3 crop start must be nonnegative, got {crop_start_frame}")

    target_frames = torch.as_tensor(target_frames)
    if target_frames.ndim != 4:
        raise ValueError(f"MiniMax-H3 target frames must be [F,H,W,C], got {tuple(target_frames.shape)}")
    frame_count, height, width = target_frames.shape[:3]
    expected_video_frames = video_latent_frames(frame_count)
    expected_audio_frames = audio_latent_frames(frame_count)
    if width % 32 or height % 32:
        raise ValueError(f"MiniMax-H3 target axes must be divisible by 32, got {width}x{height}")
    duration = Fraction(frame_count, TARGET_FPS)
    if not allow_experimental_duration and not (Fraction(5, 1) <= duration <= Fraction(15, 1)):
        raise ValueError(
            f"MiniMax-H3 target duration {float(duration):.3f}s is outside the released 5-15s range; "
            "pass --allow_experimental_duration to proceed"
        )

    target_pixels = _prepare_pixels(target_frames)
    canonical_item_key = f"{record.video_path}#{crop_start_frame}:{frame_count}"
    target_video = _encode_target_video(video_vae, target_pixels, cache_seed, canonical_item_key)[0]
    if target_video.shape[1] != expected_video_frames:
        raise ValueError(f"MiniMax-H3 video VAE returned {target_video.shape[1]} frames, expected {expected_video_frames}")

    target_samples = waveform_samples(expected_audio_frames)
    start_sample = _audio_start_sample(crop_start_frame)
    target_audio_policy = _target_audio_policy(record, video_only=video_only)
    if target_audio_policy == "real-supervised":
        target_audio_source = record.target_audio
        if target_audio_source is None:
            raise ValueError("MiniMax-H3 real-supervised audio policy requires a target audio source")
        target_waveform = media_decoder.decode_audio(
            target_audio_source,
            start_sample=start_sample,
            sample_count=target_samples,
            require_exact=True,
        )
        audio_loss_weight = 1.0
    else:
        target_waveform = torch.zeros((2, target_samples), dtype=torch.float32)
        audio_loss_weight = 0.0
    target_audio = _encode_audio(audio_vae, target_waveform)[0]
    if target_audio.shape[2] != expected_audio_frames:
        raise ValueError(f"MiniMax-H3 audio VAE returned {target_audio.shape[2]} frames, expected {expected_audio_frames}")

    tensors = {
        _visual_key("", target_video): target_video,
        _audio_key("", target_audio): target_audio,
        "mmh3_audio_loss_weight_float32": torch.tensor(audio_loss_weight, dtype=torch.float32),
    }
    if task == "fl2va":
        for role, frame in (("first", target_frames[:1]), ("last", target_frames[-1:])):
            condition = _encode_condition_video(video_vae, _prepare_pixels(frame))[0]
            tensors[_visual_key(role, condition)] = condition
    elif task == "ref2va":
        for index, reference in enumerate(record.references):
            role_prefix = f"ref_{index:03d}"
            visual_frames = None
            if reference.type in {"image", "video"}:
                visual_frames = media_decoder.decode_reference_visual(
                    reference,
                    target_frame_count=frame_count,
                    target_size=(width, height),
                )
                if reference.type == "video":
                    video_latent_frames(visual_frames.shape[0])
                condition = _encode_condition_video(video_vae, _prepare_pixels(visual_frames))[0]
                tensors[_visual_key(f"{role_prefix}_{reference.type}", condition)] = condition

            if reference.audio is not None:
                if reference.type == "video":
                    reference_audio_frames = audio_latent_frames(visual_frames.shape[0])
                    reference_samples = waveform_samples(reference_audio_frames)
                    require_exact = True
                else:
                    reference_samples = target_samples
                    require_exact = False
                waveform = media_decoder.decode_audio(
                    reference.audio,
                    start_sample=0,
                    sample_count=reference_samples,
                    require_exact=require_exact,
                )
                audio_latent = _encode_audio(audio_vae, waveform)[0]
                tensors[_audio_key(f"{role_prefix}_audio", audio_latent)] = audio_latent

    metadata = build_latent_metadata(
        record=record,
        task=task,
        crop_start_frame=crop_start_frame,
        frame_count=frame_count,
        height=height,
        width=width,
        cache_seed=cache_seed,
        video_vae_fingerprint=video_vae_fingerprint,
        audio_vae_fingerprint=audio_vae_fingerprint,
        media_fingerprints=media_fingerprints,
        video_only=video_only,
    )
    return H3LatentCachePayload(tensors=tensors, metadata=metadata)


_fingerprint_cache: dict[tuple[Path, int, int], str] = {}


def fingerprint_file(path: str | Path) -> str:
    path = Path(path).resolve()
    stat = path.stat()
    cache_key = (path, stat.st_size, stat.st_mtime_ns)
    cached = _fingerprint_cache.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    fingerprint = f"sha256:{digest.hexdigest()}"
    _fingerprint_cache[cache_key] = fingerprint
    return fingerprint


def fingerprint_checkpoint(path: str | Path) -> str:
    path = Path(path).resolve()
    files = resolve_safetensors_files(path)
    digest = hashlib.sha256()
    for file in files:
        relative_name = file.name if path.is_file() else file.relative_to(path).as_posix()
        digest.update(relative_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprint_file(file).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def record_media_paths(record: H3Record) -> set[Path]:
    paths = {record.video_path}
    if record.target_audio is not None:
        paths.add(record.target_audio.path)
    for reference in record.references:
        paths.add(reference.path)
        if reference.audio is not None:
            paths.add(reference.audio.path)
    return paths


def records_for_dataset(dataset: VideoDataset, task: H3Task, *, video_only: bool = False) -> list[H3Record]:
    if dataset.control_directory is not None or dataset.has_control:
        raise ValueError("MiniMax-H3 R1 does not use the shared control-video fields")
    if dataset.video_jsonl_file is not None:
        records = load_h3_jsonl_records(dataset.video_jsonl_file, task, video_only=video_only)
        if len(records) != len(dataset.datasource.data):
            raise ValueError("MiniMax-H3 JSONL record count changed while building the dataset")
        for data, record in zip(dataset.datasource.data, records):
            data["video_path"] = str(record.video_path)
            data["caption"] = record.caption
        return records
    if task == "ref2va":
        raise ValueError("MiniMax-H3 Ref2VA requires video_jsonl_file")
    records = []
    canonical_paths = []
    for index in range(len(dataset.datasource)):
        video_path, caption = dataset.datasource.get_caption(index)
        record = make_h3_directory_record(video_path, caption, video_only=video_only)
        records.append(record)
        canonical_paths.append(str(record.video_path))
    dataset.datasource.video_paths = canonical_paths
    return records


def warn_missing_target_audio(
    records: Sequence[H3Record],
    *,
    video_only: bool,
    limit: int = 10,
) -> None:
    if video_only:
        return
    if limit < 0:
        raise ValueError("MiniMax-H3 missing-audio warning limit must be nonnegative")
    missing = [record for record in records if record.target_audio is None]
    for record in missing[:limit]:
        logger.warning(
            "MiniMax-H3 target has no audio; caching an unsupervised silence placeholder: %s",
            record.video_path,
        )


def log_target_audio_summary(policy_counts: Mapping[str, int]) -> None:
    unknown = set(policy_counts) - TARGET_AUDIO_POLICIES
    if unknown or any(not isinstance(count, int) or count < 0 for count in policy_counts.values()):
        raise ValueError(f"Invalid MiniMax-H3 target-audio policy counts: {dict(policy_counts)}")
    real_audio = policy_counts.get("real-supervised", 0)
    missing_audio = policy_counts.get("missing-unsupervised", 0)
    video_only_count = policy_counts.get("video-only-unsupervised", 0)
    total = real_audio + missing_audio + video_only_count
    supervised_fraction = real_audio / total if total else 0.0
    logger.info(
        "MiniMax-H3 target-audio cache summary: real_audio=%d missing_audio=%d video_only=%d supervised_audio_fraction=%.6f",
        real_audio,
        missing_audio,
        video_only_count,
        supervised_fraction,
    )


def record_for_item(item: ItemInfo, records: Sequence[H3Record]) -> tuple[H3Record, int]:
    item_path = Path(item.item_key).resolve()
    for record in records:
        target = record.video_path
        if item_path.parent != target.parent or item_path.suffix.lower() != target.suffix.lower():
            continue
        pattern = rf"^{re.escape(target.stem)}_(\d+)-(\d+)$"
        match = re.fullmatch(pattern, item_path.stem)
        if match is None:
            continue
        crop_start = int(match.group(1))
        encoded_count = int(match.group(2))
        if item.frame_count != encoded_count:
            raise ValueError(
                f"MiniMax-H3 crop item frame count mismatch for {item.item_key}: item={item.frame_count}, key={encoded_count}"
            )
        return record, crop_start
    raise ValueError(f"MiniMax-H3 cache item has no canonical media record: {item.item_key}")


def install_h3_video_decoder(dataset: VideoDataset, decoder: PyAVH3MediaDecoder) -> None:
    def get_video_data_from_path(
        datasource,
        video_path,
        start_frame=None,
        end_frame=None,
        bucket_selector=None,
    ):
        if bucket_selector is None:
            bucket_selector = datasource.bucket_selector
        return decoder.decode_target_video(video_path, start_frame, end_frame, bucket_selector)

    dataset.datasource.get_video_data_from_path = MethodType(get_video_data_from_path, dataset.datasource)


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_latents.setup_parser_common(include_vae=False)
    parser.add_argument("--video_vae", type=str, required=True, help="MiniMax-H3 video VAE safetensors path or directory")
    parser.add_argument("--audio_vae", type=str, required=True, help="MiniMax-H3 audio VAE safetensors path or directory")
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument("--cache_seed", type=int, default=0, help="seed used for reproducible target-video posterior samples")
    parser.add_argument(
        "--h3_video_only",
        action="store_true",
        help="ignore all target audio and cache unsupervised Audio-VAE silence placeholders",
    )
    parser.add_argument(
        "--allow_experimental_duration",
        action="store_true",
        help="allow target crops outside the released 5-15 second duration range",
    )
    parser.add_argument("--disable_mmap", action="store_true", help="disable memory-mapped safetensors loading")
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Loading dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_H3)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets
    if not all(isinstance(dataset, VideoDataset) for dataset in datasets):
        raise ValueError("MiniMax-H3 latent caching accepts only video datasets")

    decoder = PyAVH3MediaDecoder()
    records = []
    for dataset in datasets:
        dataset_records = records_for_dataset(dataset, args.task, video_only=args.h3_video_only)
        install_h3_video_decoder(dataset, decoder)
        records.extend(dataset_records)
    warn_missing_target_audio(records, video_only=args.h3_video_only)

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets,
            args.debug_mode,
            args.console_width,
            args.console_back,
            args.console_num_images,
            fps=TARGET_FPS,
        )
        return

    logger.info("Fingerprinting MiniMax-H3 VAE checkpoints")
    video_vae_fingerprint = fingerprint_checkpoint(args.video_vae)
    audio_vae_fingerprint = fingerprint_checkpoint(args.audio_vae)
    media_fingerprints = {path: fingerprint_file(path) for record in records for path in record_media_paths(record)}

    logger.info("Loading MiniMax-H3 video VAE from %s", args.video_vae)
    video_vae = load_video_vae(
        args.video_vae,
        device=device,
        dtype=VIDEO_VAE_ENCODE_DTYPE,
        disable_mmap=args.disable_mmap,
    )
    logger.info("Loading MiniMax-H3 audio VAE from %s", args.audio_vae)
    audio_vae = load_audio_vae(args.audio_vae, device=device, dtype=torch.float32, disable_mmap=args.disable_mmap)

    skip_matching_cache = args.skip_existing
    args.skip_existing = False
    cache_policy_counts: Counter[str] = Counter()

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            record, crop_start = record_for_item(item, records)
            cache_policy_counts[_target_audio_policy(record, video_only=args.h3_video_only)] += 1
            record_fingerprints = {path: media_fingerprints[path] for path in record_media_paths(record)}
            frame_count, height, width = item.content.shape[:3]
            expected_metadata = build_latent_metadata(
                record=record,
                task=args.task,
                crop_start_frame=crop_start,
                frame_count=frame_count,
                height=height,
                width=width,
                cache_seed=args.cache_seed,
                video_vae_fingerprint=video_vae_fingerprint,
                audio_vae_fingerprint=audio_vae_fingerprint,
                media_fingerprints=record_fingerprints,
                video_only=args.h3_video_only,
            )
            if skip_matching_cache and Path(item.latent_cache_path).is_file():
                if cache_metadata_matches(item.latent_cache_path, expected_metadata):
                    logger.info("Skipping matching MiniMax-H3 latent cache: %s", item.latent_cache_path)
                    continue
                logger.info("Rebuilding stale MiniMax-H3 latent cache: %s", item.latent_cache_path)
            payload = build_latent_tensors(
                record=record,
                task=args.task,
                target_frames=item.content,
                crop_start_frame=crop_start,
                video_vae=video_vae,
                audio_vae=audio_vae,
                cache_seed=args.cache_seed,
                media_decoder=decoder,
                video_vae_fingerprint=video_vae_fingerprint,
                audio_vae_fingerprint=audio_vae_fingerprint,
                media_fingerprints=record_fingerprints,
                allow_experimental_duration=args.allow_experimental_duration,
                video_only=args.h3_video_only,
            )
            logger.info("Saving MiniMax-H3 latent cache for %s to %s", item.item_key, item.latent_cache_path)
            save_latent_cache_minimax_h3(item, payload.tensors, payload.metadata)

    cache_latents.encode_datasets(datasets, encode, args)
    log_target_audio_summary(cache_policy_counts)


if __name__ == "__main__":
    main()
