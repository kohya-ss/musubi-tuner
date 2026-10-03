"""Differentiable NR clip losses, independent of the optimizer and Accelerate."""

from __future__ import annotations

import torch

from musubi_tuner.dlssnr.losses import _rho
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.temporal import warp_bilinear


def _supervised_mask(batch, burn_in):
    source = batch["source"]
    mask = batch.get("loss_mask", torch.ones_like(source[..., :1, :, :]))
    return mask if source.ndim == 4 else torch.cat(list(mask[:, burn_in:].unbind(1)), dim=0)


def _temporal_mask(batch, index):
    motion = batch["motion"][:, index]
    height, width = motion.shape[-2:]
    x = torch.arange(width, device=motion.device).view(1, 1, width) + motion[:, 0]
    y = torch.arange(height, device=motion.device).view(1, height, 1) + motion[:, 1]
    inside = ((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1))[:, None]
    active = (~batch["reset"][:, index]).view(-1, 1, 1, 1)
    return batch["temporal_valid"][:, index].float() * inside * active


def loss_denominators(batch, burn_in):
    mask = _supervised_mask(batch, burn_in)
    count = 3 * mask.sum()
    edge = 3 * ((mask[..., 1:] * mask[..., :-1]).sum() + (mask[..., 1:, :] * mask[..., :-1, :]).sum())
    temporal = mask.new_zeros(())
    if batch["source"].ndim == 5:
        for index in range(burn_in + 1, batch["source"].shape[1]):
            temporal = temporal + 3 * _temporal_mask(batch, index).sum()
    return {"pre": float(count), "out": float(count), "edge": float(edge), "temporal": float(temporal)}


def _clip_outputs(model, batch, seeds, burn_in):
    history = None
    outputs = []
    frame_count = batch["source"].shape[1]
    if burn_in < 1 or burn_in >= frame_count:
        raise ValueError("burn_in must leave at least one differentiable frame")
    for index in range(frame_count):
        with torch.set_grad_enabled(torch.is_grad_enabled() and index >= burn_in):
            frame = forward_frame(
                model,
                batch["source"][:, index],
                batch["controls"][:, index],
                [item[index] for item in seeds],
                history=history,
                motion=batch["motion"][:, index] if history is not None else None,
                history_valid=batch["history_valid"][:, index] if history is not None else None,
                reset=batch["reset"][:, index],
            )
        history = frame["next_history"].detach() if index < burn_in else frame["next_history"]
        if index >= burn_in:
            outputs.append(frame)
    return outputs


def training_loss(model, batch, seeds, weights, burn_in=0, normalizers=None):
    is_clip = batch["source"].ndim == 5
    outputs = (
        _clip_outputs(model, batch, seeds, burn_in)
        if is_clip
        else [forward_frame(model, batch["source"], batch["controls"], seeds)]
    )
    preclamp = torch.cat([frame["neural_preclamp"] for frame in outputs], dim=0)
    rendered = torch.cat([frame["rendered_proxy"] for frame in outputs], dim=0)
    target = torch.cat(list(batch["target"][:, burn_in:].unbind(1)), dim=0) if is_clip else batch["target"]
    mask = _supervised_mask(batch, burn_in)
    terms = {"pre": (_rho(preclamp - target) * mask).sum(), "out": (_rho(rendered - target) * mask).sum()}
    edge = rendered.new_zeros(())
    for dimension in (-1, -2):
        valid = mask[..., 1:] * mask[..., :-1] if dimension == -1 else mask[..., 1:, :] * mask[..., :-1, :]
        edge = edge + (_rho(rendered.diff(dim=dimension) - target.diff(dim=dimension)) * valid).sum()
    terms["edge"] = edge
    terms["temporal"] = rendered.new_zeros(())
    if is_clip and weights.get("temporal", 0):
        for offset in range(1, len(outputs)):
            index = burn_in + offset
            previous, _ = warp_bilinear(outputs[offset - 1]["rendered_proxy"], batch["motion"][:, index])
            previous_target, _ = warp_bilinear(batch["target"][:, index - 1], batch["motion"][:, index])
            error = (outputs[offset]["rendered_proxy"] - previous) - (batch["target"][:, index] - previous_target)
            terms["temporal"] = terms["temporal"] + (_rho(error) * _temporal_mask(batch, index)).sum()
    denominators = normalizers if normalizers is not None else loss_denominators(batch, burn_in)
    if denominators["pre"] <= 0:
        raise ValueError("loss mask has no supervised pixels")
    losses = {name: value / denominators[name] if denominators[name] > 0 else value * 0 for name, value in terms.items()}
    total = sum(weights.get(name, 0.0) * value for name, value in losses.items())
    if not torch.isfinite(total):
        raise RuntimeError("non-finite NR loss")
    metrics = {
        f"loss/{name}": float(value.detach()) for name, value in losses.items() if name != "temporal" or weights.get(name, 0)
    }
    metrics["loss"] = float(total.detach())
    metrics["blend_max"] = max(float(frame["blend_weight"].detach().max()) for frame in outputs)
    return total, metrics
