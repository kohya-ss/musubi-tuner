# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Valid-rectangle to padded-field geometry for DLSS-NR 310.8.0.

The rules match OpenDLSS `Geometry::fromValid`, including the extra width step and the refusal of
sizes whose level 0 is not a whole number of 8-pixel windows.
"""

from __future__ import annotations

from dataclasses import dataclass

from musubi_tuner.dlssnr.packing import align_up


@dataclass(frozen=True)
class Geometry:
    valid_width: int
    valid_height: int
    full_width: int
    full_height: int
    levels: tuple[tuple[int, int], ...]  # six (width, height), level 0 is the first pool

    @property
    def full_size(self) -> tuple[int, int]:
        return self.full_height, self.full_width


def _reductions(valid: int) -> int:
    reductions = 0
    size = valid
    for level in range(6):
        half = align_up((size + 1) // 2, 4)
        if half < size:
            reductions += 1
        if level == 0 and half % 8 != 0:
            reductions += 1
        size = half
    return reductions


def resolve_geometry(valid_width: int, valid_height: int) -> Geometry:
    if valid_width < 1 or valid_height < 1:
        raise ValueError(f"invalid size {valid_width}x{valid_height}")
    align_w = 1 << _reductions(valid_width)
    align_h = 1 << _reductions(valid_height)
    full_w = max(320, align_up(valid_width, align_w))
    full_h = max(320, align_up(valid_height, align_h))
    if full_w % (4 * align_w) == 0 and full_h % (4 * align_h) == 0:
        full_w += align_w
    levels: list[tuple[int, int]] = []
    width, height = full_w, full_h
    for _ in range(6):
        width = align_up((width + 1) // 2, 4)
        height = align_up((height + 1) // 2, 4)
        levels.append((width, height))
    level0_w, level0_h = levels[0]
    if level0_w % 8 or level0_h % 8 or valid_width < 33 or valid_height < 33:
        raise ValueError(
            f"unsupported size {valid_width}x{valid_height}: level 0 is {level0_w}x{level0_h}; "
            "each axis must be at least 33 and level 0 must be a whole number of 8-pixel windows"
        )
    return Geometry(valid_width, valid_height, full_w, full_h, tuple(levels))
