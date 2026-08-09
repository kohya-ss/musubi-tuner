from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable, Optional, Sequence

import av
import numpy as np
import torch

import logging

logger = logging.getLogger(__name__)

AUDIO_SIDECAR_EXTENSIONS = frozenset({".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav"})

# tolerance for timestamp gaps/overlaps between decoded audio chunks (codec jitter)
DEFAULT_TIMESTAMP_TOLERANCE_SAMPLES = 2
# tolerance for pts oscillation around the decode-order sample positions (muxers that stamp
# audio pts from a wall clock, e.g. screen captures); bounded oscillation with no net drift
# means the samples themselves are contiguous and only the timestamps wobble
DEFAULT_PTS_JITTER_TOLERANCE_SECONDS = 0.1
# tolerance for missing samples at the end of a stream (codec priming/padding); shortfalls
# within this tolerance are zero-padded, larger shortfalls are an error
DEFAULT_CODEC_PAD_TOLERANCE_SAMPLES = 800

_CHANNEL_LAYOUTS = {1: "mono", 2: "stereo"}


@dataclass(frozen=True)
class AudioSource:
    path: Path
    embedded: bool  # True if the audio is an audio track inside the video container


@dataclass(frozen=True)
class AudioSpec:
    """Architecture-specific audio dataset parameters.

    Entry scripts of audio-capable architectures construct one and pass it to
    `generate_dataset_group_by_blueprint`; the shared dataset layer stays
    architecture-agnostic. `samples_per_crop` maps a video crop's frame count to the
    number of waveform samples the architecture's audio latent grid requires.

    `samples_per_crop` must be a picklable module-level function (not a lambda or
    closure): datasets carry the spec into DataLoader workers, which are spawned
    processes on Windows/macOS and pickle their arguments.
    """

    sample_rate: int
    channels: int
    samples_per_crop: Callable[[int], int]
    codec_pad_tolerance: int = DEFAULT_CODEC_PAD_TOLERANCE_SAMPLES

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError(f"Audio sample rate must be positive, got {self.sample_rate}")
        if self.channels not in _CHANNEL_LAYOUTS:
            raise ValueError(f"Audio channels must be one of {sorted(_CHANNEL_LAYOUTS)}, got {self.channels}")
        if self.codec_pad_tolerance < 0:
            raise ValueError(f"Audio codec pad tolerance must be nonnegative, got {self.codec_pad_tolerance}")


def probe_audio(path: str | Path) -> bool:
    """Returns True if the media file contains at least one audio stream."""
    with av.open(str(path)) as container:
        return bool(container.streams.audio)


def sidecar_audio_paths(video_path: str | Path) -> list[Path]:
    """Returns same-stem audio files next to the video, sorted by name."""
    video_path = Path(video_path)
    return sorted(
        (
            candidate.resolve()
            for candidate in video_path.parent.iterdir()
            if candidate.is_file() and candidate.stem == video_path.stem and candidate.suffix.lower() in AUDIO_SIDECAR_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def resolve_audio_source(video_path: str | Path, explicit_path: Optional[str | Path] = None) -> Optional[AudioSource]:
    """Resolves the audio source for a video item.

    Priority: explicit path > same-stem sidecar file > audio track embedded in the video.
    Returns None if the item has no audio. An explicit path that does not exist or has no
    audio stream is an error; so is more than one sidecar candidate.
    """
    video_path = Path(video_path).resolve()

    if explicit_path is not None:
        path = Path(explicit_path).resolve()
        if not path.is_file():
            raise ValueError(f"Explicit audio path does not exist: {path}")
        if not probe_audio(path):
            raise ValueError(f"Explicit audio source contains no audio stream: {path}")
        return AudioSource(path=path, embedded=False)

    sidecars = sidecar_audio_paths(video_path)
    if len(sidecars) > 1:
        formatted = ", ".join(str(path) for path in sidecars)
        raise ValueError(f"Multiple same-stem audio sidecars found for {video_path}: {formatted}")
    if sidecars:
        if not probe_audio(sidecars[0]):
            raise ValueError(f"Audio sidecar contains no audio stream: {sidecars[0]}")
        return AudioSource(path=sidecars[0], embedded=False)

    if video_path.is_file() and probe_audio(video_path):
        return AudioSource(path=video_path, embedded=True)
    return None


def _audio_frame_to_tensor(frame: av.AudioFrame, channels: int) -> torch.Tensor:
    array = frame.to_ndarray()
    if array.ndim != 2:
        raise ValueError(f"Unexpected PyAV audio frame shape: {array.shape}")
    if array.shape[0] != channels and array.shape[1] == channels:
        array = array.T
    if array.shape[0] != channels:
        raise ValueError(f"PyAV resampler returned shape {array.shape}, expected {channels} channels")
    return torch.from_numpy(np.asarray(array, dtype=np.float32).copy())


def _audio_frame_start_sample(frame: av.AudioFrame, sample_rate: int, fallback: int) -> int:
    if frame.pts is None or frame.time_base is None:
        return fallback
    return round(frame.pts * frame.time_base * sample_rate)


def assemble_audio_chunks(
    chunks: Sequence[tuple[int, torch.Tensor]],
    *,
    channels: int,
    timestamp_tolerance_samples: int = DEFAULT_TIMESTAMP_TOLERANCE_SAMPLES,
    pts_jitter_tolerance_samples: int = 0,
) -> torch.Tensor:
    """Concatenates (start_sample, [C, L]) chunks into one waveform.

    Small gaps within the tolerance are zero-filled and small overlaps are trimmed;
    larger discontinuities are an error. Timestamps that only oscillate around the
    decode-order sample positions and return to them by the last chunk carry no missing
    or duplicated samples — the stream is contiguous and merely stamped with a jittery
    clock — so the chunks are concatenated in decode order, ignoring pts. A real gap or
    overlap shifts all subsequent timestamps permanently and never takes this path.
    `pts_jitter_tolerance_samples` bounds the oscillation; 0 disables the recovery.
    """
    if not chunks:
        raise ValueError("Audio chunk list is empty")
    if timestamp_tolerance_samples < 0:
        raise ValueError("Audio timestamp tolerance must be nonnegative")
    if pts_jitter_tolerance_samples < 0:
        raise ValueError("Audio pts jitter tolerance must be nonnegative")
    for _, chunk in chunks:
        if chunk.ndim != 2 or chunk.shape[0] != channels:
            raise ValueError(f"Audio chunk must be [{channels},L], got {tuple(chunk.shape)}")

    if pts_jitter_tolerance_samples > 0:
        deviations = []
        nominal_start = chunks[0][0]
        for start_sample, chunk in chunks:
            deviations.append(start_sample - nominal_start)
            nominal_start += chunk.shape[1]
        peak_deviation = max(abs(deviation) for deviation in deviations)
        if peak_deviation <= pts_jitter_tolerance_samples and abs(deviations[-1]) <= timestamp_tolerance_samples:
            if peak_deviation > timestamp_tolerance_samples:
                logger.debug(
                    f"Audio pts jitter up to {peak_deviation} samples with no net drift; concatenating chunks in decode order"
                )
            return torch.cat([chunk for _, chunk in chunks], dim=1)

    expected_start = chunks[0][0]
    assembled = []
    for start_sample, chunk in chunks:
        delta = start_sample - expected_start
        if abs(delta) > timestamp_tolerance_samples:
            raise ValueError(f"Audio stream is discontinuous: expected sample {expected_start}, got {start_sample}")
        if delta > 0:
            assembled.append(torch.zeros(channels, delta, dtype=chunk.dtype, device=chunk.device))
        elif delta < 0:
            chunk = chunk[:, -delta:]
        if chunk.shape[1] > 0:
            assembled.append(chunk)
        expected_start = start_sample + chunk.shape[1] + max(0, -delta)
    return torch.cat(assembled, dim=1)


def decode_audio(source: AudioSource, *, sample_rate: int, channels: int) -> torch.Tensor:
    """Decodes the full audio stream to a [channels, samples] float32 waveform."""
    layout = _CHANNEL_LAYOUTS.get(channels)
    if layout is None:
        raise ValueError(f"Audio channels must be one of {sorted(_CHANNEL_LAYOUTS)}, got {channels}")

    chunks = []
    next_start_sample = 0
    with av.open(str(source.path)) as container:
        if not container.streams.audio:
            raise ValueError(f"Audio source has no audio stream: {source.path}")
        stream = container.streams.audio[0]
        # containers with a coarse timestamp grid (e.g. Matroska's 1 ms time base) quantize
        # chunk timestamps; two consecutive chunks can be quantized in opposite directions,
        # so allow up to one full tick of jitter when assembling chunks
        timestamp_tolerance = DEFAULT_TIMESTAMP_TOLERANCE_SAMPLES
        if stream.time_base is not None:
            tick_samples = float(stream.time_base) * sample_rate
            timestamp_tolerance = max(timestamp_tolerance, math.ceil(tick_samples) + 1)
        resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunk = _audio_frame_to_tensor(resampled, channels)
                chunk_start = _audio_frame_start_sample(resampled, sample_rate, next_start_sample)
                chunks.append((chunk_start, chunk))
                next_start_sample = chunk_start + chunk.shape[1]
        for resampled in resampler.resample(None):
            chunk = _audio_frame_to_tensor(resampled, channels)
            chunk_start = _audio_frame_start_sample(resampled, sample_rate, next_start_sample)
            chunks.append((chunk_start, chunk))
            next_start_sample = chunk_start + chunk.shape[1]

    if not chunks:
        raise ValueError(f"Audio source decoded no samples: {source.path}")
    return assemble_audio_chunks(
        chunks,
        channels=channels,
        timestamp_tolerance_samples=timestamp_tolerance,
        pts_jitter_tolerance_samples=round(sample_rate * DEFAULT_PTS_JITTER_TOLERANCE_SECONDS),
    ).contiguous()


def slice_audio_window(
    waveform: torch.Tensor,
    *,
    start_sample: int,
    sample_count: int,
    pad_tolerance: int = DEFAULT_CODEC_PAD_TOLERANCE_SAMPLES,
    require_exact: bool = True,
    context: str = "",
) -> torch.Tensor:
    """Extracts [C, sample_count] from a [C, L] waveform.

    A terminal shortfall within pad_tolerance is zero-padded (codec priming/padding);
    a larger shortfall is an error when require_exact is True.
    """
    if waveform.ndim != 2:
        raise ValueError(f"Audio waveform must be [C, L], got {tuple(waveform.shape)}")
    if start_sample < 0 or sample_count <= 0:
        raise ValueError("Audio window must have a nonnegative start and positive length")

    suffix = f": {context}" if context else ""
    window = waveform[:, start_sample : start_sample + sample_count]
    if require_exact and window.shape[1] < sample_count:
        deficit = sample_count - window.shape[1]
        if deficit > pad_tolerance:
            raise ValueError(
                f"Audio source is materially short at sample {start_sample}: need {sample_count}, got {window.shape[1]}{suffix}"
            )
        window = torch.nn.functional.pad(window, (0, deficit))
    if window.shape[1] == 0:
        raise ValueError(f"Audio window is empty at sample {start_sample}{suffix}")
    return window.contiguous()


def audio_window_start(crop_start_frame: int, fps: int, sample_rate: int) -> int:
    """Maps a video crop start frame (at integer fps) to the nearest waveform sample."""
    if crop_start_frame < 0:
        raise ValueError(f"Crop start frame must be nonnegative, got {crop_start_frame}")
    if fps <= 0 or sample_rate <= 0:
        raise ValueError("fps and sample_rate must be positive")
    return (crop_start_frame * sample_rate + fps // 2) // fps
