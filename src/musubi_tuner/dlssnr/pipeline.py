# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""One NR frame. History is an explicit tensor; a reset or a missing sample does not blend."""

from __future__ import annotations

import torch

from musubi_tuner.dlssnr.geometry import Geometry, resolve_geometry
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.preprocess import build_features
from musubi_tuner.dlssnr.temporal import publish_history, warp_bilinear


def forward_frame(
    model: NRModel,
    source_proxy: torch.Tensor,
    controls: torch.Tensor,
    frame_seed: int,
    geometry: Geometry | None = None,
    history: torch.Tensor | None = None,
    motion: torch.Tensor | None = None,
    history_valid: torch.Tensor | None = None,
    reset: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | Geometry]:
    """source_proxy and controls are the valid rectangle, NCHW and N5HW.

    `history` is the previous rendered proxy. `motion` is current-to-previous [B,2,H,W]. `reset` is [B] bool.
    """
    if source_proxy.ndim != 4 or source_proxy.shape[1] != 3:
        raise ValueError(f"source_proxy must be [B,3,H,W], got {tuple(source_proxy.shape)}")
    if geometry is None:
        geometry = resolve_geometry(int(source_proxy.shape[-1]), int(source_proxy.shape[-2]))
    history_rgb, usable = _reproject(source_proxy, history, motion, history_valid, reset)
    features = build_features(
        source_proxy, controls, geometry, frame_seed, history_rgb=history_rgb if history is not None else None
    )
    raw_field = model(features, geometry)
    raw = raw_field[:, :, : geometry.valid_height, : geometry.valid_width]
    preclamp = source_proxy + raw[:, 0:3] / 4
    neural = preclamp.clamp(0, 1)
    blend = model.blocks["70"].blend_scale
    if history is None or usable is None:
        weight = neural.new_zeros(source_proxy.shape[0], 1, geometry.valid_height, geometry.valid_width)
        rendered = neural
    else:
        weight = (torch.sigmoid(raw[:, 3:4]) * blend).clamp(0, 1) * usable.to(dtype=neural.dtype)
        rendered = (1 - weight) * neural + weight * history_rgb
    return {
        "raw_head": raw,
        "neural_preclamp": preclamp,
        "neural_proxy": neural,
        "rendered_proxy": rendered,
        "blend_weight": weight,
        "next_history": publish_history(rendered),
        "geometry": geometry,
    }


@torch.no_grad()
def rollout_sequence(
    model: NRModel,
    source: torch.Tensor,
    controls: torch.Tensor,
    motion: torch.Tensor,
    reset: torch.Tensor,
    history_valid: torch.Tensor,
    seeds: list[int],
) -> torch.Tensor:
    """Closed loop over one clip. source is [T,3,H,W]. Each frame is detached, so the graph does not grow."""
    if source.ndim != 4:
        raise ValueError("rollout_sequence expects one clip [T,3,H,W]")
    history = None
    rendered = []
    for index in range(source.shape[0]):
        use_history = history if not bool(reset[index]) else None
        outputs = forward_frame(
            model,
            source[index : index + 1],
            controls[index : index + 1],
            seeds[index],
            history=use_history,
            motion=None if use_history is None else motion[index : index + 1],
            history_valid=None if use_history is None else history_valid[index : index + 1],
            reset=reset[index : index + 1],
        )
        rendered.append(outputs["rendered_proxy"])
        history = outputs["next_history"].detach()
    return torch.cat(rendered, dim=0)


def _reproject(source, history, motion, history_valid, reset):
    if history is None:
        return None, None
    if motion is None:
        raise ValueError("motion is required when history is present")
    warped, inside = warp_bilinear(history, motion)
    usable = inside
    if history_valid is not None:
        usable = usable & history_valid.to(dtype=torch.bool)
    if reset is not None:
        usable = usable & ~reset.to(dtype=torch.bool).view(-1, 1, 1, 1)
    history_rgb = torch.where(usable, warped, source)
    return history_rgb, usable
