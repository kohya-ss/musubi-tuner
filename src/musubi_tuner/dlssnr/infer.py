"""Surrogate inference. Outputs are proxy-space RGB, before any DLL color grade."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from musubi_tuner.dlssnr.dataset import load_single_frame_manifest, load_temporal_manifest
from musubi_tuner.dlssnr.model import NRModel
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.pipeline import forward_frame
from musubi_tuner.dlssnr.temporal import stable_frame_seed
from musubi_tuner.dlssnr.runtime import (
    default_runtime_policy,
    configure_model_runtime,
    validate_runtime_device,
    validate_runtime_policy,
)


def require_surrogate(numerics_profile: str) -> None:
    if numerics_profile not in ("train_surrogate", "train_experimental"):
        raise ValueError(f"only train_surrogate/train_experimental inference is implemented, got {numerics_profile}")


def load_model(model_dir: str | Path, device: str = "auto", *, runtime_overrides=None) -> NRModel:
    from musubi_tuner.dlssnr.artifacts import inspect_canonical, read_artifact_runtime

    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    source_identity = inspect_canonical(model_dir, development_smoke=True)
    saved = read_artifact_runtime(model_dir)
    overrides = {key: value for key, value in (runtime_overrides or {}).items() if value is not None}
    if unknown := set(overrides) - (set(default_runtime_policy()) - {"schema"}):
        raise ValueError(f"unsupported inference runtime overrides: {sorted(unknown)}")
    policy = {**saved, **overrides}
    if overrides.get("fp8_base") is False and "fp8_scaled" not in overrides:
        policy["fp8_scaled"] = False
    validate_runtime_policy(policy)
    validate_runtime_device(policy, torch.device(device))
    model = NRModel().to(dtype=torch.float32)
    model.load_canonical(str(Path(model_dir) / "model.safetensors"))
    model.requires_grad_(False)
    if policy["fp8_base"]:
        from musubi_tuner.dlssnr.fp8 import quantize_frozen_base

        quantize_frozen_base(model, scaled=policy["fp8_scaled"])
    model.to(device).eval()
    configure_model_runtime(model, policy, training=False)
    model.source_identity = source_identity
    model.runtime_provenance = {"saved_policy": saved, "overrides": overrides, "matches_saved_policy": policy == saved}
    return model


def add_runtime_arguments(parser):
    parser.add_argument("--numerics_profile", choices=["train_surrogate", "train_experimental"], default=None)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--fp8_base", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fp8_scaled", action=argparse.BooleanOptionalAction, default=None)
    backends = parser.add_mutually_exclusive_group()
    backends.add_argument("--attention_backend", choices=["native", "sdpa", "flash_attn", "xformers", "sage_attn"], default=None)
    for backend in ("sdpa", "flash_attn", "xformers", "sage_attn"):
        backends.add_argument(f"--{backend}", dest="attention_backend", action="store_const", const=backend)
    parser.add_argument("--attention_scope", choices=["all", "global"], default=None)


def runtime_overrides_from_args(args):
    return {
        name: getattr(args, name)
        for name in ("numerics_profile", "mixed_precision", "fp8_base", "fp8_scaled", "attention_backend", "attention_scope")
        if getattr(args, name) is not None
    }


def _write_inference_metadata(model, output, seed, width, height, manifest):
    from musubi_tuner.dlssnr.artifacts import write_json
    from musubi_tuner.dlssnr.identity import file_sha256

    write_json(
        output / "inference_metadata.json",
        {
            "schema": "dlssnr_inference_v1",
            "runtime_policy": getattr(model, "runtime_policy", default_runtime_policy()),
            "runtime_provenance": getattr(model, "runtime_provenance", None),
            "source_identity": getattr(model, "source_identity", None),
            "seed": seed,
            "resolution": [width, height],
            "manifest_sha256": file_sha256(manifest),
            "native_equivalent": False,
        },
    )


def save_png(image: torch.Tensor, path: Path) -> None:
    array = image.detach().clamp(0, 1).mul(255).round().to(dtype=torch.uint8).permute(1, 2, 0).cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@fp32_execution()
def generate_stills(
    model, manifest: str | Path, bucket_width: int, bucket_height: int, output_dir: str | Path, seed: int = 0
) -> list[Path]:
    samples = load_single_frame_manifest(manifest, bucket_width, bucket_height, require_target=False)
    output = Path(output_dir)
    written = []
    device = next(model.parameters()).device
    for sample in samples:
        frame_seed = stable_frame_seed(seed, 0, sample["sample_id"], sample["frame_index"], sample["crop_id"])
        with torch.no_grad():
            rendered = forward_frame(
                model, sample["source"].unsqueeze(0).to(device), sample["controls"].unsqueeze(0).to(device), frame_seed
            )["rendered_proxy"][0]
        path = output / f"{sample['sample_id']}.png"
        save_png(rendered, path)
        written.append(path)
    _write_inference_metadata(model, output, seed, bucket_width, bucket_height, manifest)
    return written


@fp32_execution()
def generate_sequence(
    model, manifest: str | Path, bucket_width: int, bucket_height: int, output_dir: str | Path, seed: int = 0
) -> list[Path]:
    clips = load_temporal_manifest(manifest, bucket_width, bucket_height, None, require_target=False)
    if len(clips) != 1:
        raise ValueError("video inference takes one clip; split the manifest")
    clip = clips.rows[0]
    device = next(model.parameters()).device
    output = Path(output_dir) / clip["sample_id"]
    written = []
    history = None
    with torch.no_grad():
        for frame in clips.iter_frames(0):
            if frame["reset"]:
                history = None
            tensors = {name: value.unsqueeze(0).to(device) for name, value in frame.items() if isinstance(value, torch.Tensor)}
            result = forward_frame(
                model,
                tensors["source"],
                tensors["controls"],
                stable_frame_seed(seed, 0, clip["sample_id"], frame["frame_index"], clip.get("crop_id", 0)),
                history=history,
                motion=tensors["motion"] if history is not None else None,
                history_valid=tensors["history_valid"] if history is not None else None,
            )
            path = output / f"frame_{frame['frame_index']:04d}.png"
            save_png(result["rendered_proxy"][0], path)
            written.append(path)
            history = result["next_history"].detach()
    _write_inference_metadata(model, output, seed, bucket_width, bucket_height, manifest)
    return written
