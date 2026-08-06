# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for Musubi from Hugging Face Diffusers PR #14355 at commit
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
# (modular_pipelines/minimax_h3/packing.py and packing_ref2va.py).
# ComfyUI is used only as an independent numerical reference.

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from musubi_tuner.minimax_h3.media import H3Task

VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32
STEREO_CHANNELS = 2
VIDEO_PATCH_SIZE = (1, 2, 2)
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FRAME_RESCALE = 5.0 / 3.0

H3ReferenceKind = Literal["image", "audio", "video"]
H3SegmentKind = Literal["text", "visual_condition", "audio_condition", "target_audio", "target_video"]


@dataclass(frozen=True)
class H3VideoGeometry:
    frames: int
    height: int
    width: int

    def __post_init__(self) -> None:
        if self.frames <= 0:
            raise ValueError(f"MiniMax-H3 video geometry needs positive frames, got {self.frames}")
        if self.height <= 0 or self.width <= 0 or self.height % 2 or self.width % 2:
            raise ValueError(f"MiniMax-H3 latent video axes must be positive multiples of 2, got {self.height}x{self.width}")

    @property
    def frame_rows(self) -> int:
        return (self.height // 2) * (self.width // 2)

    @property
    def row_count(self) -> int:
        return self.frames * self.frame_rows


@dataclass(frozen=True)
class H3ReferenceGeometry:
    kind: H3ReferenceKind
    video: H3VideoGeometry | None = None
    audio_frames: int = 0

    def __post_init__(self) -> None:
        if self.kind not in {"image", "audio", "video"}:
            raise ValueError(f"Unsupported MiniMax-H3 reference kind: {self.kind}")
        if self.audio_frames < 0:
            raise ValueError(f"MiniMax-H3 reference audio length must be nonnegative, got {self.audio_frames}")
        if self.kind == "image":
            if self.video is None or self.video.frames != 1 or self.audio_frames:
                raise ValueError("MiniMax-H3 image references require one visual frame and no audio rows")
        elif self.kind == "audio":
            if self.video is not None or self.audio_frames <= 0:
                raise ValueError("MiniMax-H3 audio references require audio rows and no visual geometry")
        else:
            if self.video is None:
                raise ValueError("MiniMax-H3 video references require visual geometry")
            _validate_video_latent_frames(self.video.frames, "reference video")
            if self.audio_frames:
                expected = _expected_audio_frames(self.video.frames)
                if self.audio_frames != expected:
                    raise ValueError(f"MiniMax-H3 reference video audio has {self.audio_frames} frames, expected {expected}")


@dataclass(frozen=True)
class H3RowSegment:
    role: str
    kind: H3SegmentKind
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"Invalid MiniMax-H3 row segment {self.role}: [{self.start}, {self.stop})")

    @property
    def row_count(self) -> int:
        return self.stop - self.start

    @property
    def row_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class H3PackedLayout:
    task: H3Task
    text_length: int
    target_video: H3VideoGeometry
    target_audio_frames: int
    visual_conditions: tuple[H3VideoGeometry, ...]
    references: tuple[H3ReferenceGeometry, ...]
    segments: tuple[H3RowSegment, ...]
    row_count: int

    def segment(self, role: str) -> H3RowSegment:
        matches = [segment for segment in self.segments if segment.role == role]
        if len(matches) != 1:
            raise KeyError(f"MiniMax-H3 layout expected exactly one {role!r} segment, found {len(matches)}")
        return matches[0]

    @property
    def target_video_segment(self) -> H3RowSegment:
        return self.segment("target_video")

    @property
    def target_audio_segment(self) -> H3RowSegment:
        return self.segment("target_audio")


@dataclass(frozen=True)
class H3TimestepRows:
    unique_timesteps: torch.Tensor
    row_timesteps: torch.Tensor
    row_timestep_indices: torch.Tensor
    token_tags: torch.Tensor
    block_adaln_indices: torch.Tensor
    block_segments: tuple[tuple[int, int, int], ...]
    video_timestep_index: int
    audio_timestep_index: int


def _coerce_video_geometry(value: H3VideoGeometry | Sequence[int], label: str) -> H3VideoGeometry:
    if isinstance(value, H3VideoGeometry):
        return value
    if len(value) != 3:
        raise ValueError(f"MiniMax-H3 {label} geometry must be (frames,height,width), got {tuple(value)}")
    return H3VideoGeometry(*(int(part) for part in value))


def _validate_video_latent_frames(frames: int, label: str) -> None:
    if frames < 2 or (frames - 2) % 5:
        raise ValueError(f"MiniMax-H3 {label} latent frames must be 5*n+2, got {frames}")


def _expected_audio_frames(video_latent_frames: int) -> int:
    _validate_video_latent_frames(video_latent_frames, "target video")
    n = (video_latent_frames - 2) // 5
    pixel_frames = 17 * n + 5
    return (10 * pixel_frames + 3) // 6


def pack_video_rows(latents: torch.Tensor) -> torch.Tensor:
    """Patch H3 video latents while preserving the leading batch axis."""
    unbatched = latents.ndim == 4
    if unbatched:
        latents = latents.unsqueeze(0)
    if latents.ndim != 5:
        raise ValueError(f"MiniMax-H3 video latents must be [B,24,F,H,W], got {tuple(latents.shape)}")
    batch, channels, frames, height, width = latents.shape
    if channels != VIDEO_CHANNELS:
        raise ValueError(f"MiniMax-H3 video latents need {VIDEO_CHANNELS} channels, got {channels}")
    patch_t, patch_h, patch_w = VIDEO_PATCH_SIZE
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"MiniMax-H3 video latent shape {frames}x{height}x{width} is not divisible by patch {VIDEO_PATCH_SIZE}")
    rows = latents.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(
        batch,
        (frames // patch_t) * (height // patch_h) * (width // patch_w),
        channels * patch_t * patch_h * patch_w,
    )
    return rows[0] if unbatched else rows


def pack_audio_rows(latents: torch.Tensor) -> torch.Tensor:
    """Pack stereo H3 audio in channel-major order."""
    unbatched = latents.ndim == 3
    if unbatched:
        latents = latents.unsqueeze(0)
    if latents.ndim != 4 or latents.shape[1] != AUDIO_CHANNELS or latents.shape[2] != STEREO_CHANNELS:
        raise ValueError(f"MiniMax-H3 audio latents must be [B,32,2,A], got {tuple(latents.shape)}")
    batch, channels, stereo, frames = latents.shape
    rows = latents.permute(0, 2, 3, 1).reshape(batch, stereo * frames, channels)
    return rows[0] if unbatched else rows


def build_h3_layout(
    *,
    task: H3Task,
    text_length: int,
    target_video: H3VideoGeometry | Sequence[int],
    target_audio_frames: int,
    visual_conditions: Sequence[H3VideoGeometry | Sequence[int]] = (),
    references: Sequence[H3ReferenceGeometry] = (),
) -> H3PackedLayout:
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
    if text_length <= 0:
        raise ValueError(f"MiniMax-H3 text length must be positive, got {text_length}")
    target_video = _coerce_video_geometry(target_video, "target video")
    _validate_video_latent_frames(target_video.frames, "target video")
    expected_audio_frames = _expected_audio_frames(target_video.frames)
    if target_audio_frames != expected_audio_frames:
        raise ValueError(
            f"MiniMax-H3 target audio has {target_audio_frames} frames, expected {expected_audio_frames} "
            f"for {target_video.frames} video latent frames"
        )

    visual_conditions = tuple(
        _coerce_video_geometry(condition, f"visual condition {index}") for index, condition in enumerate(visual_conditions)
    )
    references = tuple(references)
    if task == "t2va" and (visual_conditions or references):
        raise ValueError("MiniMax-H3 T2VA layout does not accept condition rows")
    if task == "fl2va":
        if references or len(visual_conditions) != 2:
            raise ValueError("MiniMax-H3 FL2VA layout requires exactly first and last visual conditions")
        for role, condition in zip(("first", "last"), visual_conditions):
            if condition != H3VideoGeometry(1, target_video.height, target_video.width):
                raise ValueError(f"MiniMax-H3 FL2VA {role} condition must be one target-sized latent frame, got {condition}")
    if task == "ref2va":
        if visual_conditions or not references:
            raise ValueError("MiniMax-H3 Ref2VA layout requires ordered references and no FL2VA conditions")
        if not any(reference.kind in {"image", "video"} for reference in references):
            raise ValueError("MiniMax-H3 Ref2VA layout requires at least one visual reference")

    segments = []
    row = 0

    def append(role: str, kind: H3SegmentKind, row_count: int) -> None:
        nonlocal row
        segments.append(H3RowSegment(role, kind, row, row + row_count))
        row += row_count

    append("text", "text", text_length)
    if task == "fl2va":
        append("first", "visual_condition", visual_conditions[0].row_count)
        append("last", "visual_condition", visual_conditions[1].row_count)
    elif task == "ref2va":
        for index, reference in enumerate(references):
            prefix = f"ref_{index:03d}"
            if reference.kind == "image":
                append(f"{prefix}_image", "visual_condition", reference.video.row_count)
            elif reference.kind == "audio":
                append(f"{prefix}_audio", "audio_condition", reference.audio_frames * STEREO_CHANNELS)
            else:
                if reference.audio_frames:
                    append(f"{prefix}_audio", "audio_condition", reference.audio_frames * STEREO_CHANNELS)
                append(f"{prefix}_video", "visual_condition", reference.video.row_count)
    append("target_audio", "target_audio", target_audio_frames * STEREO_CHANNELS)
    append("target_video", "target_video", target_video.row_count)

    return H3PackedLayout(
        task=task,
        text_length=text_length,
        target_video=target_video,
        target_audio_frames=target_audio_frames,
        visual_conditions=visual_conditions,
        references=references,
        segments=tuple(segments),
        row_count=row,
    )


def _axis_from_sqrt_area(dimension: int, sqrt_area: float) -> torch.Tensor:
    ratio = dimension / sqrt_area
    count = dimension // 2
    return (torch.arange(count, dtype=torch.float64) * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(video: H3VideoGeometry) -> tuple[torch.Tensor, torch.Tensor]:
    sqrt_area = math.sqrt(video.height * video.width)
    height_axis = _axis_from_sqrt_area(video.height, sqrt_area)
    width_axis = _axis_from_sqrt_area(video.width, sqrt_area)
    height, width = torch.meshgrid(height_axis, width_axis, indexing="ij")
    return torch.stack((height.reshape(-1), width.reshape(-1)), dim=-1), width_axis


def _video_t_spans(frames: int) -> list[float]:
    return [FRAME_RESCALE * FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)] for index in range(frames)]


def _video_grid(video: H3VideoGeometry, cursor: float) -> torch.Tensor:
    spans = torch.tensor(_video_t_spans(video.frames), dtype=torch.float64)
    times = cursor + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))
    frame, _ = _frame_grid(video)
    grid = torch.empty(video.frames, frame.shape[0], 3, dtype=torch.float64)
    grid[..., 0] = times[:, None]
    grid[..., 1:] = frame[None]
    return grid.reshape(-1, 3)


def _audio_grid(cursor: float, frames: int, width_low: float, width_high: float) -> torch.Tensor:
    grid = torch.zeros(frames * STEREO_CHANNELS, 3, dtype=torch.float64)
    grid[:, 0] = (cursor + torch.arange(frames, dtype=torch.float64)).repeat(STEREO_CHANNELS)
    grid[:frames, 2] = width_low
    grid[frames:, 2] = width_high
    return grid


def build_position_grid(layout: H3PackedLayout, *, device: torch.device | str | None = None) -> torch.Tensor:
    positions = torch.empty(layout.row_count, 3, dtype=torch.float64)
    text = layout.segment("text")
    positions[text.row_slice] = 0
    positions[text.row_slice, 0] = torch.arange(layout.text_length, dtype=torch.float64)

    target_frame, target_width = _frame_grid(layout.target_video)
    target_width_endpoints = float(target_width[0]), float(target_width[-1])
    cursor = float(layout.text_length)

    if layout.task == "fl2va":
        first = layout.segment("first")
        positions[first.row_slice, 0] = cursor
        positions[first.row_slice, 1:] = target_frame
        last = layout.segment("last")
        last_time = cursor + sum(_video_t_spans(layout.target_video.frames)) - FRAME_RESCALE
        positions[last.row_slice, 0] = last_time
        positions[last.row_slice, 1:] = target_frame
    elif layout.task == "ref2va":
        for index, reference in enumerate(layout.references):
            prefix = f"ref_{index:03d}"
            if reference.kind == "image":
                segment = layout.segment(f"{prefix}_image")
                frame, _ = _frame_grid(reference.video)
                positions[segment.row_slice, 0] = cursor
                positions[segment.row_slice, 1:] = frame
                cursor += 1.0
            elif reference.kind == "audio":
                segment = layout.segment(f"{prefix}_audio")
                positions[segment.row_slice] = _audio_grid(cursor, reference.audio_frames, *target_width_endpoints)
                cursor += float(reference.audio_frames)
            else:
                frame, width = _frame_grid(reference.video)
                if reference.audio_frames:
                    audio = layout.segment(f"{prefix}_audio")
                    positions[audio.row_slice] = _audio_grid(
                        cursor,
                        reference.audio_frames,
                        float(width[0]),
                        float(width[-1]),
                    )
                video = layout.segment(f"{prefix}_video")
                positions[video.row_slice] = _video_grid(reference.video, cursor)
                cursor += max(float(reference.audio_frames), sum(_video_t_spans(reference.video.frames)))

    target_audio = layout.target_audio_segment
    positions[target_audio.row_slice] = _audio_grid(
        cursor,
        layout.target_audio_frames,
        *target_width_endpoints,
    )
    target_video = layout.target_video_segment
    positions[target_video.row_slice] = _video_grid(layout.target_video, cursor)
    positions = positions.unsqueeze(0)
    return positions.to(device=device) if device is not None else positions


def _single_model_time(value: float | torch.Tensor, label: str) -> float:
    tensor = torch.as_tensor(value).detach().reshape(-1)
    if tensor.numel() != 1:
        raise ValueError(f"MiniMax-H3 R1 {label} must contain exactly one value")
    scalar = float(tensor[0].item())
    if not math.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
        raise ValueError(f"MiniMax-H3 {label} must be finite and in [0,1], got {scalar}")
    return scalar


def _validate_clean_coefficient(value: float, label: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"MiniMax-H3 {label} must be finite and in [0,1], got {value}")
    return value


def _build_modulation_segments(indices: torch.Tensor) -> tuple[tuple[int, int, int], ...]:
    change_points = torch.nonzero(indices[1:] != indices[:-1], as_tuple=False).flatten() + 1
    boundaries = torch.cat((indices.new_tensor([0]), change_points, indices.new_tensor([indices.numel()])))
    return tuple(
        tuple(row)
        for row in torch.stack(
            (boundaries[:-1], boundaries[1:], indices[boundaries[:-1]]),
            dim=1,
        ).tolist()
    )


def build_timestep_rows(
    layout: H3PackedLayout,
    *,
    text_token_tags: torch.Tensor,
    model_t_video: float | torch.Tensor,
    model_t_audio: float | torch.Tensor,
    visual_condition_clean: float = 0.999,
    audio_condition_clean: float = 1.0,
) -> H3TimestepRows:
    video_time = _single_model_time(model_t_video, "video model time")
    audio_time = _single_model_time(model_t_audio, "audio model time")
    visual_clean = _validate_clean_coefficient(visual_condition_clean, "visual condition clean coefficient")
    audio_clean = _validate_clean_coefficient(audio_condition_clean, "audio condition clean coefficient")

    text_token_tags = torch.as_tensor(text_token_tags)
    if text_token_tags.dtype != torch.int64 or text_token_tags.shape != (1, layout.text_length):
        raise ValueError(f"MiniMax-H3 text token tags must be int64 [1,{layout.text_length}]")
    if not torch.all((text_token_tags == 0) | (text_token_tags == 1)):
        raise ValueError("MiniMax-H3 text token tags may contain only 0 and 1")
    text_token_tags = text_token_tags.detach().cpu()

    segment_times = {}
    for segment in layout.segments:
        if segment.kind in {"text", "target_video"}:
            segment_times[segment.role] = video_time
        elif segment.kind == "target_audio":
            segment_times[segment.role] = audio_time
        elif segment.kind == "visual_condition":
            segment_times[segment.role] = max(video_time, visual_clean)
        else:
            segment_times[segment.role] = max(audio_time, audio_clean)

    unique_values = sorted(set(segment_times.values()))
    timestep_index = {value: index for index, value in enumerate(unique_values)}
    row_timesteps = torch.empty((1, layout.row_count), dtype=torch.float32)
    row_timestep_indices = torch.empty((1, layout.row_count), dtype=torch.int64)
    token_tags = torch.empty((1, layout.row_count), dtype=torch.int64)
    for segment in layout.segments:
        value = segment_times[segment.role]
        row_timesteps[:, segment.row_slice] = value
        row_timestep_indices[:, segment.row_slice] = timestep_index[value]
        if segment.kind == "text":
            token_tags[:, segment.row_slice] = text_token_tags
        elif segment.kind in {"visual_condition", "target_video"}:
            token_tags[:, segment.row_slice] = 0
        else:
            token_tags[:, segment.row_slice] = 2

    block_adaln_indices = 3 * row_timestep_indices + token_tags
    block_segments = _build_modulation_segments(block_adaln_indices[0])

    return H3TimestepRows(
        unique_timesteps=torch.tensor(unique_values, dtype=torch.float32),
        row_timesteps=row_timesteps,
        row_timestep_indices=row_timestep_indices,
        token_tags=token_tags,
        block_adaln_indices=block_adaln_indices,
        block_segments=block_segments,
        video_timestep_index=timestep_index[video_time],
        audio_timestep_index=timestep_index[audio_time],
    )


def _unpack_video_rows(rows: torch.Tensor, video: H3VideoGeometry) -> torch.Tensor:
    unbatched = rows.ndim == 2
    if unbatched:
        rows = rows.unsqueeze(0)
    expected_shape = (video.row_count, VIDEO_CHANNELS * math.prod(VIDEO_PATCH_SIZE))
    if rows.ndim != 3 or tuple(rows.shape[1:]) != expected_shape:
        raise ValueError(f"MiniMax-H3 target video rows must be [B,{expected_shape[0]},{expected_shape[1]}]")
    batch = rows.shape[0]
    patch_t, patch_h, patch_w = VIDEO_PATCH_SIZE
    unpacked = rows.reshape(
        batch,
        video.frames // patch_t,
        video.height // patch_h,
        video.width // patch_w,
        VIDEO_CHANNELS,
        patch_t,
        patch_h,
        patch_w,
    )
    unpacked = unpacked.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(
        batch,
        VIDEO_CHANNELS,
        video.frames,
        video.height,
        video.width,
    )
    return unpacked[0] if unbatched else unpacked


def _unpack_audio_rows(rows: torch.Tensor, audio_frames: int) -> torch.Tensor:
    unbatched = rows.ndim == 2
    if unbatched:
        rows = rows.unsqueeze(0)
    expected_shape = (audio_frames * STEREO_CHANNELS, AUDIO_CHANNELS)
    if rows.ndim != 3 or tuple(rows.shape[1:]) != expected_shape:
        raise ValueError(f"MiniMax-H3 target audio rows must be [B,{expected_shape[0]},{expected_shape[1]}]")
    batch = rows.shape[0]
    unpacked = rows.reshape(batch, STEREO_CHANNELS, audio_frames, AUDIO_CHANNELS).permute(0, 3, 1, 2)
    return unpacked[0] if unbatched else unpacked


def unpack_targets(
    layout: H3PackedLayout,
    target_video_rows: torch.Tensor,
    target_audio_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_video_rows.ndim != target_audio_rows.ndim or target_video_rows.ndim not in {2, 3}:
        raise ValueError("MiniMax-H3 target video/audio rows must both be batched or both be unbatched")
    if target_video_rows.ndim == 3 and target_video_rows.shape[0] != target_audio_rows.shape[0]:
        raise ValueError("MiniMax-H3 target video/audio rows must have matching batch axes")
    return (
        _unpack_video_rows(target_video_rows, layout.target_video),
        _unpack_audio_rows(target_audio_rows, layout.target_audio_frames),
    )
