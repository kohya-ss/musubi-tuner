"""Streaming paired-frame and student-history evaluation in proxy RGB space."""

from __future__ import annotations

import math

import torch

from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.temporal import stable_frame_seed, warp_bilinear


@torch.no_grad()
def evaluate(model, datasets, seed, device):
    reports = {}
    for kind, dataset in datasets.items():
        cases = []
        for sample_index, row in enumerate(dataset.rows):
            history = previous_target = None
            sums = {
                name: 0.0
                for name in ("absolute", "squared", "count", "pre", "blend", "pixels", "saturation", "temporal", "temporal_count")
            }
            for frame in dataset.iter_frames(sample_index):
                tensors = {name: value.unsqueeze(0).to(device) for name, value in frame.items() if isinstance(value, torch.Tensor)}
                reset = frame["reset"]
                if reset:
                    history = previous_target = None
                output = forward_frame(
                    model,
                    tensors["source"],
                    tensors["controls"],
                    stable_frame_seed(seed, 0, row["sample_id"], frame["frame_index"], row.get("crop_id", 0)),
                    history=history,
                    motion=tensors["motion"] if history is not None else None,
                    history_valid=tensors["history_valid"] if history is not None else None,
                    reset=torch.tensor([reset], device=device),
                )
                rendered = output["rendered_proxy"]
                if not all(
                    torch.isfinite(output[name]).all() for name in ("raw_head", "neural_preclamp", "rendered_proxy", "blend_weight")
                ):
                    raise RuntimeError(f"non-finite evaluation output: {row['sample_id']} frame {frame['frame_index']}")
                target, mask = tensors["target"], tensors["loss_mask"]
                error = rendered - target
                sums["absolute"] += float((error.abs() * mask).sum())
                sums["squared"] += float((error.square() * mask).sum())
                sums["count"] += float(mask.sum()) * 3
                sums["pre"] += float(((output["neural_preclamp"] - target).abs() * mask).sum())
                sums["blend"] += float((output["blend_weight"] * mask).sum())
                sums["pixels"] += float(mask.sum())
                saturated = (output["neural_preclamp"] < 0) | (output["neural_preclamp"] > 1)
                sums["saturation"] += float((saturated * mask).sum())
                if history is not None:
                    previous, inside = warp_bilinear(history, tensors["motion"])
                    target_previous, _ = warp_bilinear(previous_target, tensors["motion"])
                    temporal_mask = tensors["temporal_valid"] * inside
                    delta = (rendered - previous) - (target - target_previous)
                    sums["temporal"] += float((delta.abs() * temporal_mask).sum())
                    sums["temporal_count"] += float(temporal_mask.sum()) * 3
                history, previous_target = output["next_history"].detach(), target
            mse = sums["squared"] / sums["count"]
            cases.append(
                {
                    "sample_id": row["sample_id"],
                    "frames": len(row["frames"]),
                    "rgb_mae": sums["absolute"] / sums["count"],
                    "rgb_psnr": -10 * math.log10(mse) if mse > 0 else None,
                    "preclamp_mae": sums["pre"] / sums["count"],
                    "blend_mean": sums["blend"] / sums["pixels"],
                    "saturation_fraction": sums["saturation"] / sums["count"],
                    "temporal_mae": sums["temporal"] / sums["temporal_count"] if sums["temporal_count"] else None,
                }
            )
        reports[kind] = cases
    return reports
