# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""History warp and closed-loop rollout.

The sampler is bilinear, the named surrogate for the native five-tap Catmull-Rom. Motion is
current-to-previous in valid-rectangle pixels, x right and y down. Coordinates use pixel centers
and align_corners=False. Samples whose previous pixel center leaves the rectangle are marked
invalid and must not be read as black history.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

HISTORY_SAMPLER = "bilinear"
SEED_POLICY = "dlssnr_seed_v1"


def stable_frame_seed(global_seed: int, epoch: int, sample_id: str, frame_index: int, crop_id: int = 0) -> int:
    """FNV-1a over explicit integers and the UTF-8 sample id. Not Python's salted hash."""
    digest = 2166136261
    for part in (global_seed, epoch, crop_id, frame_index):
        digest ^= part & 0xFFFFFFFF
        digest = (digest * 16777619) & 0xFFFFFFFF
    for byte in sample_id.encode("utf-8"):
        digest ^= byte
        digest = (digest * 16777619) & 0xFFFFFFFF
    return int(digest)


def warp_bilinear(image: torch.Tensor, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Warp `image` by current-to-previous motion.

    image is [B,C,H,W], motion is [B,2,H,W]. Returns the sampled image and a [B,1,H,W] mask that is
    true where the previous pixel center lies inside the rectangle. Outside samples are zeroed.
    """
    if motion.shape[-3] != 2 or motion.shape[-2:] != image.shape[-2:]:
        raise ValueError(f"motion {tuple(motion.shape)} does not match image {tuple(image.shape)}")
    _, _, height, width = image.shape
    y = torch.arange(height, device=image.device, dtype=image.dtype).view(1, height, 1)
    x = torch.arange(width, device=image.device, dtype=image.dtype).view(1, 1, width)
    previous_x = x + motion[:, 0]
    previous_y = y + motion[:, 1]
    grid_x = 2 * (previous_x + 0.5) / width - 1
    grid_y = 2 * (previous_y + 0.5) / height - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    inside = (previous_x >= 0) & (previous_x <= (width - 1)) & (previous_y >= 0) & (previous_y <= (height - 1))
    inside = inside[:, None]
    return sampled * inside.to(dtype=sampled.dtype), inside


def publish_history(rendered: torch.Tensor) -> torch.Tensor:
    """Surrogate history is the FP32 rendered proxy. Native instead truncates toward zero to float16."""
    return rendered
