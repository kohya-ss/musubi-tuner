from __future__ import annotations


H3_IMAGE_MODES = frozenset({"none", "first", "first_last"})
H3_TEXT_VISUAL_MAX_PIXELS = 1024 * 1024


def normalize_h3_image_mode(mode: str) -> str:
    normalized = mode.replace("-", "_")
    if normalized not in H3_IMAGE_MODES:
        raise ValueError(f"Unsupported MiniMax-H3 image mode {mode!r}; expected one of {sorted(H3_IMAGE_MODES)}")
    return normalized


def validate_h3_frame_count(frame_count: int) -> int:
    frame_count = int(frame_count)
    if frame_count < 5 or (frame_count - 5) % 17 != 0:
        raise ValueError(f"Invalid MiniMax-H3 frame count {frame_count}; expected 17*n+5")
    return frame_count
