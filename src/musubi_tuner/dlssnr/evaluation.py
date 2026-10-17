"""Streaming paired-frame and student-history evaluation in proxy RGB space."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from musubi_tuner.dlssnr.content_metrics import ContentMetrics
from musubi_tuner.dlssnr.detail_metrics import DetailDiagnostics, NOISE_SEED_XOR
from musubi_tuner.dlssnr.losses import masked_temporal_residual
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.runtime import native_weight_runtime
from musubi_tuner.dlssnr.temporal import stable_frame_seed, warp_bilinear


@torch.no_grad()
def evaluate(model, datasets, seed, device, *, compare_native=False, detail_diagnostics=False, content_metric=None):
    reports = {}
    for kind, dataset in datasets.items():
        synthetic_protocol = getattr(dataset, "synthetic_protocol", None)
        temporal_support = "joint_loss_mask" if synthetic_protocol is not None else None
        cases = []
        for sample_index, row in enumerate(dataset.rows):
            previous_target = previous_mask = None
            modes = (False, True) if compare_native else (False,)
            histories = [None for _ in modes]
            totals = [_empty_sums() for _ in modes]
            details = [DetailDiagnostics() for _ in modes] if detail_diagnostics else None
            content = [ContentMetrics() for _ in modes] if content_metric is not None else None
            gap = {name: 0.0 for name in ("absolute", "pre", "count", "max")}
            for frame in dataset.iter_frames(sample_index):
                tensors = {name: value.unsqueeze(0).to(device) for name, value in frame.items() if isinstance(value, torch.Tensor)}
                reset = frame["reset"]
                if reset:
                    histories = [None for _ in modes]
                    previous_target = previous_mask = None
                frame_seed = stable_frame_seed(seed, 0, row["sample_id"], frame["frame_index"], row.get("crop_id", 0))
                outputs = []
                for index, native in enumerate(modes):
                    history = histories[index]
                    frame_args = {
                        "history": history,
                        "motion": tensors["motion"] if history is not None else None,
                        "history_valid": tensors["history_valid"] if history is not None else None,
                        "reset": torch.tensor([reset], device=device),
                    }
                    with native_weight_runtime(model) if native else nullcontext():
                        output = forward_frame(model, tensors["source"], tensors["controls"], frame_seed, **frame_args)
                        _validate_output(output, row, frame)
                        if details is not None:
                            # Probe only the current noise lanes; never feed the alternate
                            # output into either runtime's primary history.
                            alternate = forward_frame(
                                model, tensors["source"], tensors["controls"], frame_seed ^ NOISE_SEED_XOR, **frame_args
                            )
                            _validate_output(alternate, row, frame, alternate=True)
                            details[index].add(output, alternate, tensors)
                            del alternate
                    _accumulate(
                        totals[index],
                        output,
                        tensors,
                        history,
                        previous_target,
                        previous_mask=previous_mask,
                        temporal_support=temporal_support,
                    )
                    if content is not None:
                        scores = content_metric(output["rendered_proxy"], tensors["source"], tensors["loss_mask"])
                        content[index].add(scores, tensors["loss_mask"])
                    histories[index] = output["next_history"].detach()
                    outputs.append(output)
                if compare_native:
                    mask = tensors["loss_mask"]
                    error = (outputs[0]["rendered_proxy"] - outputs[1]["rendered_proxy"]).abs()
                    gap["absolute"] += float((error * mask).sum())
                    gap["pre"] += float(((outputs[0]["neural_preclamp"] - outputs[1]["neural_preclamp"]).abs() * mask).sum())
                    gap["count"] += float(mask.sum()) * 3
                    gap["max"] = max(gap["max"], float(torch.where(mask > 0, error, 0).max()))
                previous_target = tensors["target"]
                previous_mask = tensors["loss_mask"]
            case = _metrics(totals[0], row)
            if synthetic_protocol is not None:
                case["synthetic_temporal"] = dict(synthetic_protocol)
            if details is not None:
                case["detail_diagnostics"] = details[0].metrics()
            if content is not None:
                case["content_preservation"] = {**content[0].metrics(), "protocol": content_metric.identity}
            if compare_native:
                case["native"] = _metrics(totals[1], row)
                if synthetic_protocol is not None:
                    case["native"]["synthetic_temporal"] = dict(synthetic_protocol)
                if details is not None:
                    case["native"]["detail_diagnostics"] = details[1].metrics()
                if content is not None:
                    case["native"]["content_preservation"] = {**content[1].metrics(), "protocol": content_metric.identity}
                case["native_gap"] = {
                    "rgb_mae": gap["absolute"] / gap["count"],
                    "preclamp_mae": gap["pre"] / gap["count"],
                    "rgb_max_abs": gap["max"],
                }
            cases.append(case)
        reports[kind] = cases
    return reports


def _validate_output(output, row, frame, *, alternate=False):
    if not all(torch.isfinite(output[name]).all() for name in ("raw_head", "neural_preclamp", "rendered_proxy", "blend_weight")):
        label = " alternate" if alternate else ""
        raise RuntimeError(f"non-finite evaluation output: {row['sample_id']} frame {frame['frame_index']}{label}")


def _empty_sums():
    return {
        name: 0.0 for name in ("absolute", "squared", "count", "pre", "blend", "pixels", "saturation", "temporal", "temporal_count")
    }


def _accumulate(sums, output, tensors, history, previous_target, *, previous_mask=None, temporal_support=None):
    if temporal_support not in (None, "joint_loss_mask"):
        raise ValueError("unsupported evaluation temporal_support policy")
    rendered, target, mask = output["rendered_proxy"], tensors["target"], tensors["loss_mask"]
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
        if temporal_support == "joint_loss_mask":
            if previous_mask is None:
                raise ValueError("joint temporal evaluation requires the previous loss mask")
            delta, temporal_mask = masked_temporal_residual(
                rendered,
                history,
                target,
                previous_target,
                mask,
                previous_mask,
                tensors["motion"],
                tensors["temporal_valid"],
                torch.zeros(rendered.shape[0], dtype=torch.bool, device=rendered.device),
            )
        else:
            previous, inside = warp_bilinear(history, tensors["motion"])
            target_previous, _ = warp_bilinear(previous_target, tensors["motion"])
            temporal_mask = tensors["temporal_valid"] * inside
            delta = (rendered - previous) - (target - target_previous)
        sums["temporal"] += float((delta.abs() * temporal_mask).sum())
        sums["temporal_count"] += float(temporal_mask.sum()) * 3


def _metrics(sums, row):
    if sums["count"] <= 0:
        raise ValueError(f"evaluation sample {row['sample_id']} has no supervised pixels")
    mse = sums["squared"] / sums["count"]
    return {
        "sample_id": row["sample_id"],
        "frames": len(row["frames"]),
        "rgb_mae": sums["absolute"] / sums["count"],
        "rgb_psnr": -10 * math.log10(mse) if mse > 0 else None,
        "preclamp_mae": sums["pre"] / sums["count"],
        "blend_mean": sums["blend"] / sums["pixels"],
        "saturation_fraction": sums["saturation"] / sums["count"],
        "temporal_mae": sums["temporal"] / sums["temporal_count"] if sums["temporal_count"] else None,
    }
