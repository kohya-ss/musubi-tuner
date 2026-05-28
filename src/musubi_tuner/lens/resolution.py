"""Resolution buckets for Lens inference."""

from __future__ import annotations

from typing import Dict, Tuple


RESOLUTION_BUCKETS: Dict[int, Dict[str, Tuple[int, int]]] = {
    1024: {
        "1:2": (1472, 736),
        "9:16": (1376, 768),
        "2:3": (1248, 832),
        "3:4": (1152, 864),
        "1:1": (1024, 1024),
        "4:3": (864, 1152),
        "3:2": (832, 1248),
        "16:9": (768, 1376),
        "2:1": (736, 1472),
    },
    1440: {
        "1:2": (2080, 1040),
        "9:16": (1936, 1088),
        "2:3": (1760, 1168),
        "3:4": (1616, 1216),
        "1:1": (1440, 1440),
        "4:3": (1216, 1616),
        "3:2": (1168, 1760),
        "16:9": (1088, 1936),
        "2:1": (1040, 2080),
    },
}

SUPPORTED_BASE_RESOLUTIONS = tuple(RESOLUTION_BUCKETS.keys())
SUPPORTED_ASPECT_RATIOS = tuple(RESOLUTION_BUCKETS[1024].keys())


def resolve_resolution(base_resolution: int, aspect_ratio: str) -> Tuple[int, int]:
    if base_resolution not in RESOLUTION_BUCKETS:
        raise ValueError(f"Unsupported base_resolution={base_resolution}. Supported: {SUPPORTED_BASE_RESOLUTIONS}")
    table = RESOLUTION_BUCKETS[base_resolution]
    if aspect_ratio not in table:
        raise ValueError(f"Unsupported aspect_ratio={aspect_ratio!r}. Supported: {SUPPORTED_ASPECT_RATIOS}")
    return table[aspect_ratio]
