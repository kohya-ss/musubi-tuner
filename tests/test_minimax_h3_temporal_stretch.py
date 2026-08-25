from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import musubi_tuner.minimax_h3_generate_video as generate
from musubi_tuner.minimax_h3.media import audio_latent_frames, video_latent_frames
from musubi_tuner.minimax_h3.model import MiniMaxH3Model
from musubi_tuner.minimax_h3.packing import (
    FRAME_RESCALE,
    H3VideoGeometry,
    _expected_audio_frames,
    build_h3_layout,
    build_position_grid,
)


def _layout(output_fps=24, temporal_fine_bands=0, frame_count=124, task="t2va", **kwargs):
    return build_h3_layout(
        task=task,
        text_length=10,
        target_video=H3VideoGeometry(video_latent_frames(frame_count), 42, 24),
        target_audio_frames=audio_latent_frames(frame_count, output_fps=output_fps),
        output_fps=output_fps,
        temporal_fine_bands=temporal_fine_bands,
        **kwargs,
    )


def _target_video_times(layout, grid):
    segment = layout.target_video_segment
    times = grid[0, segment.row_slice, 0]
    return times.view(layout.target_video.frames, layout.target_video.frame_rows)[:, 0]


def test_expected_audio_frames_match_the_media_helper_across_rates():
    for fps in (24, 20, 16, 12, 8, 6, 1):
        for frame_count in (5, 22, 124, 141, 243, 345):
            assert _expected_audio_frames(video_latent_frames(frame_count), fps) == audio_latent_frames(frame_count, output_fps=fps)


def test_audio_latent_frames_validates_the_rate_before_the_native_fast_path():
    for bad in (24.0, 12.0, 0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            audio_latent_frames(124, output_fps=bad)


def test_stretched_target_times_scale_exactly_and_audio_rows_keep_the_native_rate():
    native = _layout()
    stretched = _layout(output_fps=12)
    native_times = _target_video_times(native, build_position_grid(native))
    stretched_times = _target_video_times(stretched, build_position_grid(stretched))
    assert torch.allclose(stretched_times - 10.0, (native_times - 10.0) * 2.0)
    # the audio grid stays one rotary unit per latent frame; only the frame count grows
    grid = build_position_grid(stretched)
    audio_times = grid[0, stretched.target_audio_segment.row_slice, 0]
    per_channel = audio_times.view(2, stretched.target_audio_frames)
    assert torch.equal(per_channel[0].diff(), torch.ones(stretched.target_audio_frames - 1, dtype=torch.float64))
    # video and audio cover the same stretched duration to within one audio frame
    video_span = float(stretched_times[-1] - 10.0) + FRAME_RESCALE * 4 * 2
    assert math.isclose(video_span, stretched.target_audio_frames, abs_tol=1.0)


def test_fl2va_last_anchor_follows_the_stretched_timeline():
    layout = _layout(
        output_fps=12,
        task="fl2va",
        visual_conditions=[H3VideoGeometry(1, 42, 24)] * 2,
    )
    grid = build_position_grid(layout)
    last_anchor = float(grid[0, layout.segment("last").start, 0]) - 10.0
    assert last_anchor == pytest.approx((124 - 1) * FRAME_RESCALE * 2, abs=1e-9)


def test_keep_bands_rotate_by_the_unstretched_grid_and_spare_the_spatial_axes():
    inv_freq = torch.tensor([10000.0 ** (-k / 16) for k in range(16)], dtype=torch.float32)
    dummy = SimpleNamespace(rope=SimpleNamespace(inv_freq=inv_freq))
    stretched = _layout(output_fps=12, temporal_fine_bands=3)
    grid = build_position_grid(stretched)
    fine_grid = build_position_grid(replace(stretched, output_fps=24, temporal_fine_bands=0))

    hybrid = MiniMaxH3Model._rotation_table(dummy, grid, torch.float64, fine_position_ids=fine_grid, fine_bands=3)
    all_stretched = MiniMaxH3Model._rotation_table(dummy, grid, torch.float64)
    all_fine = MiniMaxH3Model._rotation_table(dummy, fine_grid, torch.float64)

    # pair layout: [temporal 16 | height 16 | width 16] truncated to the head pair budget
    assert torch.equal(hybrid[..., :3, :, :], all_fine[..., :3, :, :])
    assert torch.equal(hybrid[..., 3:16, :, :], all_stretched[..., 3:16, :, :])
    assert torch.equal(hybrid[..., 16:, :, :], all_stretched[..., 16:, :, :])
    # non-target rows (text) are identical in both grids, so the splice is a no-op there
    assert torch.equal(hybrid[:, :10], all_stretched[:, :10])

    with pytest.raises(ValueError, match="at least one"):
        MiniMaxH3Model._rotation_table(dummy, grid, torch.float64, fine_position_ids=fine_grid, fine_bands=16)


def test_layout_validation_rejects_inconsistent_stretch_combinations():
    with pytest.raises(ValueError, match="one-frame"):
        build_h3_layout(
            task="t2va",
            text_length=10,
            target_video=H3VideoGeometry(1, 42, 24),
            target_audio_frames=2,
            one_frame=True,
            output_fps=12,
        )
    with pytest.raises(ValueError, match="require an active temporal stretch"):
        _layout(output_fps=24, temporal_fine_bands=3)
    with pytest.raises(ValueError, match="at 12 fps"):
        build_h3_layout(
            task="t2va",
            text_length=10,
            target_video=H3VideoGeometry(video_latent_frames(124), 42, 24),
            target_audio_frames=audio_latent_frames(124),
            output_fps=12,
        )
    # stretched and native layouts must never share a rotary cache entry
    assert hash(_layout()) != hash(_layout(output_fps=12))
    assert hash(_layout(output_fps=12)) != hash(_layout(output_fps=12, temporal_fine_bands=3))


def _prompt_args(**overrides):
    args = generate.setup_parser().parse_args(["--task", "t2va", "--output", "out.mp4", "--prompt", "p", "--frame_count", "124"])
    args.output_name = None
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_prompt_validation_bounds_the_stretch_arguments():
    generate.validate_prompt_args(_prompt_args(output_fps=12, stretch_keep_bands=3, allow_experimental_duration=True))
    with pytest.raises(ValueError, match=r"--output_fps must be in \[1,24\]"):
        generate.validate_prompt_args(_prompt_args(output_fps=48))
    with pytest.raises(ValueError, match=r"--stretch_keep_bands must be in \[0,15\]"):
        generate.validate_prompt_args(_prompt_args(output_fps=12, stretch_keep_bands=16, allow_experimental_duration=True))
    with pytest.raises(ValueError, match="requires an --output_fps below"):
        generate.validate_prompt_args(_prompt_args(stretch_keep_bands=3))
    with pytest.raises(ValueError, match="one-frame"):
        generate.validate_prompt_args(_prompt_args(frame_count=1, output_fps=12, output="out.png"))
    # the released 5-15 s duration gate reads the real (stretched) duration
    generate.validate_prompt_args(_prompt_args(output_fps=12))  # 124/12 = 10.3 s
    with pytest.raises(ValueError, match="outside the released 5-15s range"):
        generate.validate_prompt_args(_prompt_args(frame_count=22, output_fps=12))  # 1.8 s


def test_prompt_line_aliases_cover_the_stretch_options():
    overrides = generate.parse_prompt_line("hello --ofps 12 --skb 3")
    assert overrides == {"prompt": "hello", "output_fps": 12, "stretch_keep_bands": 3}


def test_output_fps_metadata_is_authoritative_and_validated(tmp_path, caplog):
    source = tmp_path / "latent.safetensors"
    assert generate._parse_output_fps_metadata(source, {}, 24) == 24
    assert generate._parse_output_fps_metadata(source, {"output_fps": "12"}, 24) == 12
    with caplog.at_level("WARNING", logger=generate.logger.name):
        assert generate._parse_output_fps_metadata(source, {"output_fps": "12"}, 16) == 12
    assert any("ignoring --output_fps 16" in record.getMessage() for record in caplog.records)
    for bad in ("", "12.0", "0", "48"):
        with pytest.raises(ValueError):
            generate._parse_output_fps_metadata(source, {"output_fps": bad}, 24)
