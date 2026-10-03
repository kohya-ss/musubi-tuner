# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Build the 16-lane field tensor from a valid-rectangle proxy.

Lane order matches the network. Controls must already be encoded lanes 10-14; this function does not
map UI sliders. Color and noise are published through float16. Controls are placed as given.
"""

from __future__ import annotations

import numpy as np
import torch

from musubi_tuner.dlssnr.geometry import Geometry
from musubi_tuner.dlssnr.noise import gaussian_lanes


def center_proxy(proxy: torch.Tensor) -> torch.Tensor:
    """Three round-to-nearest-even float16 steps: f16((f16(proxy) - 0.5) * 0.125).

    Backward uses a straight-through estimate of (proxy - 0.5) * 0.125 so a history tensor can
    still train the previous frame. The forward value stays on the float16 grid.
    """
    value = proxy.to(dtype=torch.float16)
    half = torch.tensor(0.5, dtype=torch.float16, device=proxy.device)
    scale = torch.tensor(0.125, dtype=torch.float16, device=proxy.device)
    value = (value - half).to(dtype=torch.float16)
    value = (value * scale).to(dtype=torch.float16)
    rounded = value.to(dtype=torch.float32)
    if not proxy.requires_grad:
        return rounded
    estimate = (proxy - 0.5) * 0.125
    return rounded.detach() + estimate - estimate.detach()


def _mirror_index(length: int, valid: int) -> np.ndarray:
    index = np.arange(length, dtype=np.int64)
    reflected = np.where(index < valid, index, 2 * valid - index - 2)
    return np.clip(reflected, 0, valid - 1)


def mirror_to_field(image: torch.Tensor, geometry: Geometry) -> torch.Tensor:
    """image is [B, C, valid_h, valid_w]. Outside the rectangle, sample the mirrored valid pixel."""
    if image.shape[-2] != geometry.valid_height or image.shape[-1] != geometry.valid_width:
        raise ValueError(
            f"image is {image.shape[-1]}x{image.shape[-2]}, geometry is {geometry.valid_width}x{geometry.valid_height}"
        )
    rows = _mirror_index(geometry.full_height, geometry.valid_height)
    cols = _mirror_index(geometry.full_width, geometry.valid_width)
    row_t = torch.as_tensor(rows, device=image.device)
    col_t = torch.as_tensor(cols, device=image.device)
    return image[:, :, row_t][:, :, :, col_t]


def build_features(
    proxy_valid: torch.Tensor,
    controls_valid: torch.Tensor,
    geometry: Geometry,
    frame_seed: int,
    history_rgb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return [B, 16, full_h, full_w].

    `history_rgb` is the reprojected previous output on the valid rectangle, in proxy units. None copies
    the current centered proxy into lanes 7-9.
    """
    if controls_valid.shape[-3] != 5:
        raise ValueError(f"controls must have 5 lanes, got {tuple(controls_valid.shape)}")
    if controls_valid.shape[0] != proxy_valid.shape[0]:
        raise ValueError("controls batch does not match proxy batch")
    proxy_field = mirror_to_field(proxy_valid, geometry)
    centered = center_proxy(proxy_field)
    controls = mirror_to_field(controls_valid, geometry)
    batch, _, height, width = centered.shape
    features = centered.new_zeros(batch, 16, height, width)
    seeds = list(range(frame_seed, frame_seed + batch)) if isinstance(frame_seed, int) else list(frame_seed)
    if len(seeds) != batch:
        raise ValueError(f"expected {batch} frame seeds, got {len(seeds)}")
    for item, seed in enumerate(seeds):
        noise = torch.from_numpy(gaussian_lanes(width, height, int(seed))).to(device=centered.device, dtype=centered.dtype)
        features[item, 0:3] = noise
    features[:, 3] = 1.0
    features[:, 4:7] = centered
    if history_rgb is None:
        features[:, 7:10] = centered
    else:
        features[:, 7:10] = center_proxy(mirror_to_field(history_rgb, geometry))
    features[:, 10:15] = controls
    features[:, 15] = 0.0
    return features
