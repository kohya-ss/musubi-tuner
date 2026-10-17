"""Source-only, matched-noise single-frame control response measurements."""

from __future__ import annotations

import csv
import logging
import math
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

from musubi_tuner.dlssnr.artifacts import write_json
from musubi_tuner.dlssnr.config import _integer, _number
from musubi_tuner.dlssnr.content_metrics import ContentMetrics
from musubi_tuner.dlssnr.controls import fixed_control_tensor, resolve_fixed_controls
from musubi_tuner.dlssnr.dataset import load_single_frame_manifest
from musubi_tuner.dlssnr.detail_metrics import (
    HIGHPASS_SIGMAS,
    NOISE_SEED_XOR,
    RATIO_ENERGY_FLOOR,
    _highpass_energy_sum,
    _masked_sum,
    diagnostic_protocol,
)
from musubi_tuner.dlssnr.evaluation import _validate_output
from musubi_tuner.dlssnr.fp8 import canonical_tensor_sha256, iter_canonical_tensors
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.identity import file_sha256, implementation_identity
from musubi_tuner.dlssnr.infer import save_png
from musubi_tuner.dlssnr.losses import masked_gaussian_lowpass
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.runtime import default_runtime_policy, native_weight_runtime, with_native_weight_qat
from musubi_tuner.dlssnr.temporal import SEED_POLICY, stable_frame_seed
from musubi_tuner.training.dlssnr_services import evaluation_mode

logger = logging.getLogger(__name__)


def build_scan_config(
    bucket_width,
    bucket_height,
    *,
    tone_values=(0, 0.5, 1),
    structure_values=(0, 0.5, 1),
    style=0,
    skin=-1,
    auto_mask=True,
    seed=0,
    lowpass_sigma=6.0,
    compare_native=False,
    content_preservation=False,
):
    for value, name in ((bucket_width, "bucket_width"), (bucket_height, "bucket_height")):
        _integer(value, name)
    resolve_geometry(bucket_width, bucket_height)
    _integer(seed, "seed", 0)
    _number(lowpass_sigma, "lowpass_sigma", positive=True)
    if lowpass_sigma > 32:
        raise ValueError("lowpass_sigma must be <= 32")
    if type(compare_native) is not bool:
        raise ValueError("compare_native must be a boolean")
    if type(content_preservation) is not bool:
        raise ValueError("content_preservation must be a boolean")
    for name, values in (("tone_values", tone_values), ("structure_values", structure_values)):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{name} must be a nonempty list")
    points, seen = [], set()
    for tone_index, tone in enumerate(tone_values):
        for structure_index, structure in enumerate(structure_values):
            requested = resolve_fixed_controls(
                {
                    "nr_tone": tone,
                    "nr_structure": structure,
                    "nr_style": style,
                    "nr_skin": skin,
                    "nr_auto_mask": auto_mask,
                }
            )
            for name in ("nr_tone", "nr_structure", "nr_skin"):
                requested[name] = float(requested[name])
            lanes = fixed_control_tensor(requested, 1, 1)[:, 0, 0].tolist()
            if tuple(lanes) in seen:
                raise ValueError("duplicate control point after FP16 lane encoding; use a coarser grid")
            seen.add(tuple(lanes))
            points.append(
                {
                    "id": f"tone{tone_index:03d}_structure{structure_index:03d}",
                    "requested": requested,
                    "encoded_lanes": lanes,
                }
            )
    config = {
        "resolution": [bucket_width, bucket_height],
        "seed": seed,
        "lowpass_sigma": float(lowpass_sigma),
        "compare_native": compare_native,
        "points": points,
    }
    if content_preservation:
        config["content_preservation"] = True
    return config


@torch.no_grad()
def response_metrics(source, output, alternate, mask, *, sigma):
    if source.ndim != 4 or source.shape[1] != 3 or mask.shape != source[:, :1].shape:
        raise ValueError("response metrics require BCHW RGB and a matching B1HW mask")
    if not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all() or not mask.any():
        raise ValueError("response metrics mask must be finite in [0,1] with nonempty support")
    images = [source, *(frame[name] for frame in (output, alternate) for name in ("rendered_proxy", "neural_preclamp"))]
    if any(image.shape != source.shape or not torch.isfinite(image).all() for image in images):
        raise ValueError("response images must have matching shapes and finite values")
    with fp32_execution(), torch.autocast(source.device.type, enabled=False):
        source, mask = source.float(), mask.float()
        rendered, preclamp = output["rendered_proxy"].float(), output["neural_preclamp"].float()
        delta = rendered - source
        lowpass = masked_gaussian_lowpass(delta, mask, sigma)
        noise = rendered - alternate["rendered_proxy"].float()
        count = 3 * float(mask.sum(dtype=torch.float64))
        measured = {
            "valid_rgb_values": count,
            "rgb_mae_vs_input": _masked_sum(delta.abs(), mask) / count,
            "preclamp_mae_vs_input": _masked_sum((preclamp - source).abs(), mask) / count,
            "lowpass_delta_rms": math.sqrt(_masked_sum(lowpass.square(), mask) / count),
            "highpass_delta_rms": math.sqrt(_masked_sum((delta - lowpass).square(), mask) / count),
            "saturation_fraction": _masked_sum(((preclamp < 0) | (preclamp > 1)).float(), mask) / count,
            "noise_rgb_mae": _masked_sum(noise.abs(), mask) / count,
            "noise_preclamp_mae": _masked_sum((preclamp - alternate["neural_preclamp"].float()).abs(), mask) / count,
        }
        for highpass_sigma in HIGHPASS_SIGMAS:
            band = f"sigma_{highpass_sigma:g}px"
            before = _highpass_energy_sum(source, mask, highpass_sigma) / count
            after = _highpass_energy_sum(rendered, mask, highpass_sigma) / count
            measured[f"hf_input_energy_{band}"] = before
            measured[f"hf_output_energy_{band}"] = after
            measured[f"hf_output_to_input_{band}"] = after / before if before > RATIO_ENERGY_FLOOR else None
            measured[f"noise_hf_rms_{band}"] = math.sqrt(_highpass_energy_sum(noise, mask, highpass_sigma) / count)
    if not all(value is None or math.isfinite(value) for value in measured.values()):
        raise RuntimeError("non-finite control response metrics")
    return measured


def _write_csv(path, measurements):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(measurements[0]))
        writer.writeheader()
        for row in measurements:
            # Quoting CSV cells does not stop spreadsheet formula evaluation.
            writer.writerow(
                {
                    key: "'" + value if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")) else value
                    for key, value in row.items()
                }
            )


@fp32_execution()
@torch.no_grad()
def scan_controls(model, manifest, output_dir, config, *, content_metric=None):
    """Scan a canonical/merged model using a config from build_scan_config.

    scan_report.json is written last, and marks a complete scan. Failed runs may
    retain partial PNGs; existing output directories are never overwritten.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"scan output directory already exists: {output_dir}")
    use_content = config.get("content_preservation", False)
    if use_content and content_metric is None:
        raise ValueError("enabled content preservation requires a frozen feature backend")
    if any(getattr(module, "_dlssnr_lora_target", None) for module in model.modules()):
        raise ValueError("scan a canonical model; merge the LoRA adapter before scanning")
    width, height = config["resolution"]
    dataset = load_single_frame_manifest(manifest, width, height, require_target=False, fixed_controls={})
    for index in range(len(dataset)):
        if not dataset[index]["loss_mask"].any():
            raise ValueError(f"sample {dataset.rows[index]['sample_id']}: scan mask has no valid pixels")
    device = next(model.parameters()).device
    policy = deepcopy(getattr(model, "runtime_policy", default_runtime_policy()))
    runtimes = {"configured": policy}
    if config["compare_native"]:
        runtimes["native"] = with_native_weight_qat(default_runtime_policy(), True)
    implementation = implementation_identity()
    for name in ("control_scan.py", "controls.py", "dataset.py", "detail_metrics.py", "losses.py", "infer.py", "evaluation.py"):
        implementation[name] = file_sha256(Path(__file__).parent / name)
    report = {
        "schema": "dlssnr_control_scan_v1",
        "native_equivalent": False,
        "config": deepcopy(config),
        "protocol": {
            "history": "single_frame_reset_no_history",
            "reference": "input_proxy_rgb",
            "controls": "fixed_maps_override_manifest_controls",
            "seed_policy": SEED_POLICY,
            "noise_pairing": "same_primary_and_alternate_seeds_for_all_controls_and_runtimes",
            "detail": {**diagnostic_protocol(), "noise_history": "none_single_frame"},
            "frequency_split_sigma_px": config["lowpass_sigma"],
            "frequency_split": "masked_gaussian_delta_lowpass_and_delta_minus_lowpass",
            "mask": "loss_mask_or_all_pixels",
            "png": "8bit_srgb_proxy_metrics_before_png_rounding",
        },
        "runtimes": runtimes,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "source_identity": deepcopy(getattr(model, "source_identity", None)),
        "runtime_provenance": deepcopy(getattr(model, "runtime_provenance", None)),
        "base_quantization": deepcopy(getattr(model, "base_quantization", None)),
        "model_parameters_sha256": canonical_tensor_sha256(iter_canonical_tensors(model)),
        "manifest_sha256": file_sha256(manifest),
        "data_identity": dataset.fingerprint(),
        "implementation": implementation,
        "inputs": [],
        "measurements": [],
    }
    if use_content:
        report["content_preservation"] = content_metric.identity
        for name in ("content_metrics.py", "dino_loss.py"):
            implementation[name] = file_sha256(Path(__file__).parent / name)
    output_dir.mkdir(parents=True, exist_ok=False)
    with evaluation_mode(model), torch.autocast(device.type, enabled=False):
        for index in range(len(dataset)):
            sample = dataset[index]
            source = sample["source"].unsqueeze(0).to(device)
            mask = sample["loss_mask"].unsqueeze(0).to(device)
            frame_seed = stable_frame_seed(config["seed"], 0, sample["sample_id"], sample["frame_index"], sample["crop_id"])
            image_root = Path("images") / sample["sample_id"]
            input_path = image_root / "input.png"
            save_png(sample["source"], output_dir / input_path)
            report["inputs"].append(
                {
                    "sample_id": sample["sample_id"],
                    "sequence_id": sample["sequence_id"],
                    "frame_index": sample["frame_index"],
                    "crop_id": sample["crop_id"],
                    "image": input_path.as_posix(),
                }
            )
            for runtime in runtimes:
                with native_weight_runtime(model) if runtime == "native" else nullcontext():
                    for point in config["points"]:
                        controls = torch.tensor(point["encoded_lanes"], device=device, dtype=torch.float32)
                        controls = controls.view(1, 5, 1, 1).expand(1, 5, height, width)
                        result = forward_frame(model, source, controls, frame_seed)
                        _validate_output(result, sample, sample)
                        alternate = forward_frame(model, source, controls, frame_seed ^ NOISE_SEED_XOR)
                        _validate_output(alternate, sample, sample, alternate=True)
                        metrics = response_metrics(source, result, alternate, mask, sigma=config["lowpass_sigma"])
                        if use_content:
                            content = ContentMetrics()
                            content.add(content_metric(result["rendered_proxy"], source, mask), mask)
                            metrics["dinov3_patch_mse_vs_input"] = content.metrics()["dinov3_patch_mse"]
                        image_path = image_root / runtime / f"{point['id']}.png"
                        save_png(result["rendered_proxy"][0], output_dir / image_path)
                        requested = point["requested"]
                        report["measurements"].append(
                            {
                                "sample_id": sample["sample_id"],
                                "frame_index": sample["frame_index"],
                                "case_id": point["id"],
                                "tone": requested["nr_tone"],
                                "structure": requested["nr_structure"],
                                "style": requested["nr_style"],
                                "skin": requested["nr_skin"],
                                "auto_mask": requested["nr_auto_mask"],
                                "runtime": runtime,
                                "frame_seed": frame_seed,
                                "alternate_seed": frame_seed ^ NOISE_SEED_XOR,
                                **metrics,
                                "image": image_path.as_posix(),
                            }
                        )
                        logger.info("NR control scan %s %s %s", sample["sample_id"], runtime, point["id"])
                        del result, alternate
    if dataset.fingerprint() != report["data_identity"]:
        raise RuntimeError("scan input files changed during measurement; rerun with stable inputs")
    _write_csv(output_dir / "scan_metrics.csv", report["measurements"])
    write_json(output_dir / "scan_report.json", report)
    return report
