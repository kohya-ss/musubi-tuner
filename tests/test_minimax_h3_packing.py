import math
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3.packing import (
    H3ReferenceGeometry,
    H3VideoGeometry,
    build_h3_layout,
    build_position_grid,
    build_timestep_rows,
    pack_audio_rows,
    pack_video_rows,
    unpack_targets,
)


TARGET_VIDEO = H3VideoGeometry(frames=2, height=4, width=4)


def _axis(dimension: int, sqrt_area: float) -> torch.Tensor:
    ratio = dimension / sqrt_area
    count = dimension // 2
    return (torch.arange(count, dtype=torch.float64) * (ratio / count) + (1.0 - ratio) / 2.0) * 32.0


def _frame_grid(video: H3VideoGeometry) -> torch.Tensor:
    sqrt_area = math.sqrt(video.height * video.width)
    height, width = torch.meshgrid(
        _axis(video.height, sqrt_area),
        _axis(video.width, sqrt_area),
        indexing="ij",
    )
    return torch.stack((height.reshape(-1), width.reshape(-1)), dim=-1)


def _video_grid(video: H3VideoGeometry, cursor: float) -> torch.Tensor:
    spans = torch.tensor(
        [5.0 / 3.0 * (1, 4, 4, 4, 4)[index % 5] for index in range(video.frames)],
        dtype=torch.float64,
    )
    times = cursor + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))
    frame = _frame_grid(video)
    grid = torch.empty(video.frames, frame.shape[0], 3, dtype=torch.float64)
    grid[..., 0] = times[:, None]
    grid[..., 1:] = frame[None]
    return grid.reshape(-1, 3)


def _audio_grid(cursor: float, frames: int, width_low: float, width_high: float) -> torch.Tensor:
    grid = torch.zeros(frames * 2, 3, dtype=torch.float64)
    grid[:, 0] = (cursor + torch.arange(frames, dtype=torch.float64)).repeat(2)
    grid[:frames, 2] = width_low
    grid[frames:, 2] = width_high
    return grid


def test_video_rows_preserve_batch_and_patch_order():
    latents = torch.arange(2 * 24 * 2 * 4 * 4, dtype=torch.float32).reshape(2, 24, 2, 4, 4)

    rows = pack_video_rows(latents)

    expected_rows = []
    for batch in range(2):
        batch_rows = []
        for frame in range(2):
            for patch_y in range(2):
                for patch_x in range(2):
                    values = []
                    for channel in range(24):
                        for y in range(patch_y * 2, patch_y * 2 + 2):
                            for x in range(patch_x * 2, patch_x * 2 + 2):
                                values.append(latents[batch, channel, frame, y, x])
                    batch_rows.append(torch.stack(values))
        expected_rows.append(torch.stack(batch_rows))

    torch.testing.assert_close(rows, torch.stack(expected_rows))
    assert rows.shape == (2, 8, 96)


def test_audio_rows_are_channel_major():
    latents = torch.arange(2 * 32 * 2 * 3).reshape(2, 32, 2, 3)

    rows = pack_audio_rows(latents)

    expected = latents.permute(0, 2, 3, 1).reshape(2, 6, 32)
    torch.testing.assert_close(rows, expected)


@pytest.mark.parametrize(
    ("task", "conditions", "references", "expected"),
    [
        (
            "t2va",
            (),
            (),
            [("text", "text", 3), ("target_audio", "target_audio", 16), ("target_video", "target_video", 8)],
        ),
        (
            "fl2va",
            (H3VideoGeometry(1, 4, 4), H3VideoGeometry(1, 4, 4)),
            (),
            [
                ("text", "text", 3),
                ("first", "visual_condition", 4),
                ("last", "visual_condition", 4),
                ("target_audio", "target_audio", 16),
                ("target_video", "target_video", 8),
            ],
        ),
        (
            "ref2va",
            (),
            (
                H3ReferenceGeometry("image", video=H3VideoGeometry(1, 2, 4)),
                H3ReferenceGeometry("video", video=H3VideoGeometry(2, 4, 8), audio_frames=8),
                H3ReferenceGeometry("audio", audio_frames=2),
            ),
            [
                ("text", "text", 3),
                ("ref_000_image", "visual_condition", 2),
                ("ref_001_audio", "audio_condition", 16),
                ("ref_001_video", "visual_condition", 16),
                ("ref_002_audio", "audio_condition", 4),
                ("target_audio", "target_audio", 16),
                ("target_video", "target_video", 8),
            ],
        ),
    ],
)
def test_layout_has_explicit_semantic_order_and_target_slices(task, conditions, references, expected):
    layout = build_h3_layout(
        task=task,
        text_length=3,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
        visual_conditions=conditions,
        references=references,
    )

    assert [(segment.role, segment.kind, segment.row_count) for segment in layout.segments] == expected
    assert layout.row_count == sum(row_count for _, _, row_count in expected)
    assert layout.target_audio_segment.stop == layout.target_video_segment.start
    assert layout.target_video_segment.stop == layout.row_count


def test_layout_rejects_target_audio_that_does_not_match_video_duration():
    with pytest.raises(ValueError, match=r"target audio.*expected 8"):
        build_h3_layout(
            task="t2va",
            text_length=1,
            target_video=TARGET_VIDEO,
            target_audio_frames=7,
        )


def test_timestep_plan_preserves_text_tags_and_uses_final_layer_time_indices_directly():
    layout = build_h3_layout(
        task="fl2va",
        text_length=3,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
        visual_conditions=(H3VideoGeometry(1, 4, 4), H3VideoGeometry(1, 4, 4)),
    )

    plan = build_timestep_rows(
        layout,
        text_token_tags=torch.tensor([[1, 0, 1]]),
        model_t_video=torch.tensor([0.25]),
        model_t_audio=torch.tensor([0.75]),
        visual_condition_clean=0.9,
        audio_condition_clean=1.0,
    )

    torch.testing.assert_close(plan.unique_timesteps, torch.tensor([0.25, 0.75, 0.9]))
    assert plan.unique_timesteps.ndim == 1
    assert plan.unique_timesteps.dtype is torch.float32
    assert plan.row_timesteps.shape == (1, layout.row_count)
    assert plan.row_timestep_indices.shape == (1, layout.row_count)
    assert plan.token_tags.shape == (1, layout.row_count)
    assert plan.block_segments[0] == (0, 1, 1)
    assert all(len(segment) == 3 and all(isinstance(value, int) for value in segment) for segment in plan.block_segments)
    torch.testing.assert_close(plan.token_tags[0, :3], torch.tensor([1, 0, 1]))
    torch.testing.assert_close(plan.block_adaln_indices[0, :3], torch.tensor([1, 0, 1]))
    condition = layout.segment("first")
    torch.testing.assert_close(
        plan.block_adaln_indices[0, condition.start : condition.stop],
        torch.full((condition.row_count,), 6),
    )
    torch.testing.assert_close(
        plan.block_adaln_indices[0, layout.target_audio_segment.start : layout.target_audio_segment.stop],
        torch.full((layout.target_audio_segment.row_count,), 5),
    )
    assert plan.video_timestep_index == 0
    assert plan.audio_timestep_index == 1


def test_position_grid_matches_full_fl2va_clock_and_does_not_advance_target_cursor():
    layout = build_h3_layout(
        task="fl2va",
        text_length=2,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
        visual_conditions=(H3VideoGeometry(1, 4, 4), H3VideoGeometry(1, 4, 4)),
    )

    actual = build_position_grid(layout)

    frame = _frame_grid(TARGET_VIDEO)
    text = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    first = torch.column_stack((torch.full((4,), 2.0), frame))
    last_time = 2.0 + (5.0 / 3.0 + 20.0 / 3.0) - 5.0 / 3.0
    last = torch.column_stack((torch.full((4,), last_time), frame))
    target_width = _axis(4, 4.0)
    audio = _audio_grid(2.0, 8, float(target_width[0]), float(target_width[-1]))
    video = _video_grid(TARGET_VIDEO, 2.0)
    expected = torch.cat((text, first, last, audio, video))

    assert actual.dtype == torch.float64
    assert actual.shape == (1, layout.row_count, 3)
    torch.testing.assert_close(actual[0], expected)


def test_position_grid_uses_the_full_five_frame_temporal_cycle():
    target = H3VideoGeometry(7, 2, 2)
    layout = build_h3_layout(
        task="t2va",
        text_length=1,
        target_video=target,
        target_audio_frames=37,
    )

    positions = build_position_grid(layout)
    video_times = positions[0, layout.target_video_segment.row_slice, 0]
    spans = torch.tensor([1, 4, 4, 4, 4, 1, 4], dtype=torch.float64) * (5.0 / 3.0)
    expected = 1.0 + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))

    torch.testing.assert_close(video_times, expected, rtol=0, atol=0)


def test_position_grid_matches_mixed_reference_cursor_order():
    image = H3VideoGeometry(1, 2, 4)
    video_reference = H3VideoGeometry(2, 4, 8)
    references = (
        H3ReferenceGeometry("image", video=image),
        H3ReferenceGeometry("video", video=video_reference, audio_frames=8),
        H3ReferenceGeometry("audio", audio_frames=2),
    )
    layout = build_h3_layout(
        task="ref2va",
        text_length=2,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
        references=references,
    )

    actual = build_position_grid(layout)

    text = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    image_grid = torch.column_stack((torch.full((2,), 2.0), _frame_grid(image)))
    video_cursor = 3.0
    reference_width = _axis(video_reference.width, math.sqrt(video_reference.height * video_reference.width))
    video_audio = _audio_grid(video_cursor, 8, float(reference_width[0]), float(reference_width[-1]))
    video_grid = _video_grid(video_reference, video_cursor)
    standalone_cursor = video_cursor + max(8.0, 5.0 / 3.0 + 20.0 / 3.0)
    target_cursor = standalone_cursor + 2.0
    target_width = _axis(TARGET_VIDEO.width, math.sqrt(TARGET_VIDEO.height * TARGET_VIDEO.width))
    standalone_audio = _audio_grid(standalone_cursor, 2, float(target_width[0]), float(target_width[-1]))
    target_audio = _audio_grid(target_cursor, 8, float(target_width[0]), float(target_width[-1]))
    target_video = _video_grid(TARGET_VIDEO, target_cursor)
    expected = torch.cat((text, image_grid, video_audio, video_grid, standalone_audio, target_audio, target_video))

    torch.testing.assert_close(actual[0], expected)
    assert torch.all(actual[0, layout.target_audio_segment.start : layout.target_audio_segment.stop, 0] >= target_cursor)


def test_target_rows_round_trip_back_to_video_and_audio_latents():
    layout = build_h3_layout(
        task="t2va",
        text_length=1,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
    )
    video = torch.randn(2, 24, 2, 4, 4)
    audio = torch.randn(2, 32, 2, 8)

    unpacked_video, unpacked_audio = unpack_targets(layout, pack_video_rows(video), pack_audio_rows(audio))

    torch.testing.assert_close(unpacked_video, video)
    torch.testing.assert_close(unpacked_audio, audio)


def test_timestep_plan_rejects_more_than_one_time_for_r1():
    layout = build_h3_layout(
        task="t2va",
        text_length=1,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
    )

    with pytest.raises(ValueError, match="exactly one value"):
        build_timestep_rows(
            layout,
            text_token_tags=torch.tensor([[1]]),
            model_t_video=torch.tensor([0.25, 0.5]),
            model_t_audio=torch.tensor([0.75]),
        )


def test_unbatched_target_rows_round_trip_without_confusing_row_counts_for_batch_axes():
    layout = build_h3_layout(
        task="t2va",
        text_length=1,
        target_video=TARGET_VIDEO,
        target_audio_frames=8,
    )
    video = torch.randn(24, 2, 4, 4)
    audio = torch.randn(32, 2, 8)

    unpacked_video, unpacked_audio = unpack_targets(layout, pack_video_rows(video), pack_audio_rows(audio))

    torch.testing.assert_close(unpacked_video, video)
    torch.testing.assert_close(unpacked_audio, audio)
