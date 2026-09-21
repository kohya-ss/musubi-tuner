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
# Chunk timestamps that wobble in both directions within this range around the decode-order
# sample positions come from a muxer stamping pts with a jittery clock (screen and USB captures
# on a 10 ms timer): the samples are contiguous and are concatenated in decode order. Below the
# range a one-off pts step cannot be told from the wobble; concatenation absorbs it as a shift of
# at most the range, too small to matter for audio-visual alignment.
DEFAULT_PTS_JITTER_RANGE_SECONDS = 0.025
# Larger or one-directional discontinuities (a cut at a non-frame boundary, capture stalls, an
# audio clock drifting against the muxer clock) are repaired in place so that the audio stays
# aligned to the video timestamps: a gap up to this size is zero-filled at its pts position
# and an overlap up to this size is trimmed; anything larger is an error.
DEFAULT_MAX_GAP_FILL_SECONDS = 0.05
DEFAULT_MAX_OVERLAP_TRIM_SECONDS = 0.025
# a training window whose in-place repairs add up to more than this is rejected: that much
# missing audio inside one clip is not worth teaching
DEFAULT_MAX_WINDOW_REPAIR_SECONDS = 0.2
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


@dataclass(frozen=True)
class AudioRepair:
    """One in-place timeline repair made while assembling decoded chunks.

    `position` is the sample index in the assembled waveform where the repair sits; `filled`
    zero samples were inserted for a pts gap, `trimmed` samples were dropped for a pts overlap.
    """

    position: int
    filled: int = 0
    trimmed: int = 0


@dataclass(frozen=True)
class DecodedAudio:
    """A [C, L] waveform plus the timeline repairs that went into it."""

    waveform: torch.Tensor
    repairs: tuple[AudioRepair, ...] = ()

    def repaired_samples(self, start_sample: int, sample_count: int) -> int:
        """Total repaired samples whose position falls inside the window."""
        return repaired_samples_in_window(self.repairs, start_sample, sample_count)


def repaired_samples_in_window(repairs: Sequence[AudioRepair], start_sample: int, sample_count: int) -> int:
    end = start_sample + sample_count
    return sum(repair.filled + repair.trimmed for repair in repairs if start_sample <= repair.position < end)


def window_repair_limit(sample_rate: int) -> int:
    """The default per-window cap on repaired samples for a stream at `sample_rate`."""
    return round(sample_rate * DEFAULT_MAX_WINDOW_REPAIR_SECONDS)


def _is_pts_jitter(deviations: Sequence[int], jitter_range: int, tolerance: int) -> bool:
    """True if chunk timestamps merely wobble around the decode-order positions.

    Jitter is bounded (the deviations span at most `jitter_range`) and goes both ways: the
    per-chunk steps include at least one gap and at least one overlap beyond the tolerance. A
    one-directional series is a drift or a cut, whose samples really are missing or duplicated.
    """
    if max(deviations) - min(deviations) > jitter_range:
        return False
    steps = [later - earlier for earlier, later in zip(deviations, deviations[1:])]
    return any(step > tolerance for step in steps) and any(step < -tolerance for step in steps)


def assemble_audio_chunks(
    chunks: Sequence[tuple[int, torch.Tensor]],
    *,
    channels: int,
    timestamp_tolerance_samples: int = DEFAULT_TIMESTAMP_TOLERANCE_SAMPLES,
    pts_jitter_range_samples: int = 0,
    max_gap_fill_samples: int = 0,
    max_overlap_trim_samples: int = 0,
    sample_rate: Optional[int] = None,
    context: str = "",
) -> DecodedAudio:
    """Concatenates (start_sample, [C, L]) chunks into one waveform.

    Timestamps that only wobble around the decode-order sample positions (see `_is_pts_jitter`)
    carry no missing or duplicated samples: the stream is contiguous and merely stamped with a
    jittery clock, so the chunks are concatenated in decode order, ignoring pts.
    `pts_jitter_range_samples` bounds the wobble; 0 disables this path.

    Otherwise every chunk is placed at its pts position: a gap up to `max_gap_fill_samples` is
    zero-filled and an overlap up to `max_overlap_trim_samples` is trimmed, each recorded as an
    `AudioRepair` (gaps and overlaps within `timestamp_tolerance_samples` are handled the same
    way but not recorded). A larger discontinuity is an error. `sample_rate` and `context` only
    make the error message readable.
    """
    if not chunks:
        raise ValueError("Audio chunk list is empty")
    if timestamp_tolerance_samples < 0:
        raise ValueError("Audio timestamp tolerance must be nonnegative")
    if pts_jitter_range_samples < 0 or max_gap_fill_samples < 0 or max_overlap_trim_samples < 0:
        raise ValueError("Audio pts jitter range and repair limits must be nonnegative")
    for _, chunk in chunks:
        if chunk.ndim != 2 or chunk.shape[0] != channels:
            raise ValueError(f"Audio chunk must be [{channels},L], got {tuple(chunk.shape)}")

    if pts_jitter_range_samples > 0:
        deviations = []
        nominal_start = chunks[0][0]
        for start_sample, chunk in chunks:
            deviations.append(start_sample - nominal_start)
            nominal_start += chunk.shape[1]
        if _is_pts_jitter(deviations, pts_jitter_range_samples, timestamp_tolerance_samples):
            logger.debug(
                f"Audio pts jitter spanning {max(deviations) - min(deviations)} samples; concatenating chunks in decode order"
            )
            return DecodedAudio(torch.cat([chunk for _, chunk in chunks], dim=1))

    def describe(samples: int, signed: bool = True) -> str:
        sign = "+" if signed else ""
        if sample_rate is None:
            return f"{samples:{sign}d} samples"
        return f"{samples * 1000 / sample_rate:{sign}.1f} ms ({samples:{sign}d} samples)"

    suffix = f": {context}" if context else ""
    expected_start = chunks[0][0]
    assembled = []
    repairs = []
    position = 0
    for start_sample, chunk in chunks:
        delta = start_sample - expected_start
        if abs(delta) > timestamp_tolerance_samples:
            limit = max_gap_fill_samples if delta > 0 else max_overlap_trim_samples
            if abs(delta) > limit:
                where = f"sample {position}" if sample_rate is None else f"{position / sample_rate:.3f}s"
                raise ValueError(
                    f"Audio stream is discontinuous at {where}: pts jumps by {describe(delta)}, beyond the repairable"
                    f" {describe(limit, signed=False)}{suffix}. Re-encode the audio so its timestamps are contiguous"
                    " (e.g. ffmpeg -c:v copy -af aresample=async=1:min_hard_comp=0.001 -c:a aac) or drop the audio."
                )
            if delta > 0:
                repairs.append(AudioRepair(position, filled=delta))
            else:
                repairs.append(AudioRepair(position, trimmed=min(-delta, chunk.shape[1])))
        if delta > 0:
            assembled.append(torch.zeros(channels, delta, dtype=chunk.dtype, device=chunk.device))
            position += delta
        elif delta < 0:
            chunk = chunk[:, -delta:]
        if chunk.shape[1] > 0:
            assembled.append(chunk)
            position += chunk.shape[1]
        expected_start = start_sample + chunk.shape[1] + max(0, -delta)
    return DecodedAudio(torch.cat(assembled, dim=1), tuple(repairs))


def decode_audio(source: AudioSource, *, sample_rate: int, channels: int) -> DecodedAudio:
    """Decodes the full audio stream to a [channels, samples] float32 waveform.

    Timeline defects are handled as `assemble_audio_chunks` describes; repairs beyond the
    timestamp tolerance are logged per file and returned for the per-window check in
    `slice_audio_window`.
    """
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
    decoded = assemble_audio_chunks(
        chunks,
        channels=channels,
        timestamp_tolerance_samples=timestamp_tolerance,
        pts_jitter_range_samples=round(sample_rate * DEFAULT_PTS_JITTER_RANGE_SECONDS),
        max_gap_fill_samples=round(sample_rate * DEFAULT_MAX_GAP_FILL_SECONDS),
        max_overlap_trim_samples=round(sample_rate * DEFAULT_MAX_OVERLAP_TRIM_SECONDS),
        sample_rate=sample_rate,
        context=str(source.path),
    )
    if decoded.repairs:
        gaps = [repair.filled for repair in decoded.repairs if repair.filled]
        overlaps = [repair.trimmed for repair in decoded.repairs if repair.trimmed]
        to_ms = 1000 / sample_rate
        logger.warning(
            f"Audio pts discontinuities repaired in {source.path}: {len(gaps)} gaps zero-filled"
            f" ({sum(gaps) * to_ms:.1f} ms total, largest {max(gaps, default=0) * to_ms:.1f} ms),"
            f" {len(overlaps)} overlaps trimmed ({sum(overlaps) * to_ms:.1f} ms total)"
        )
    return DecodedAudio(decoded.waveform.contiguous(), decoded.repairs)


def slice_audio_window(
    waveform: torch.Tensor,
    *,
    start_sample: int,
    sample_count: int,
    pad_tolerance: int = DEFAULT_CODEC_PAD_TOLERANCE_SAMPLES,
    require_exact: bool = True,
    context: str = "",
    repairs: Sequence[AudioRepair] = (),
    max_repair_samples: Optional[int] = None,
) -> torch.Tensor:
    """Extracts [C, sample_count] from a [C, L] waveform.

    A terminal shortfall within pad_tolerance is zero-padded (codec priming/padding);
    a larger shortfall is an error when require_exact is True. When the waveform's timeline
    `repairs` are given with `max_repair_samples`, a window containing more repaired samples
    than that is an error: the audio inside it is too broken to teach.
    """
    if waveform.ndim != 2:
        raise ValueError(f"Audio waveform must be [C, L], got {tuple(waveform.shape)}")
    if start_sample < 0 or sample_count <= 0:
        raise ValueError("Audio window must have a nonnegative start and positive length")

    suffix = f": {context}" if context else ""
    if repairs and max_repair_samples is not None:
        repaired = repaired_samples_in_window(repairs, start_sample, sample_count)
        if repaired > max_repair_samples:
            raise ValueError(
                f"Audio window at sample {start_sample} contains {repaired} repaired samples"
                f" (zero-filled gaps and trimmed overlaps), beyond the {max_repair_samples} allowed{suffix}"
            )
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
