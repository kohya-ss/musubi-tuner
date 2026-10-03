"""Measure one NR runtime policy per fresh process, without exporting trained weights."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from musubi_tuner.dlssnr.artifacts import inspect_canonical, write_json
from musubi_tuner.dlssnr.config import validate_lora
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.infer import add_runtime_arguments
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.dlssnr.profiles import PROFILE_ID
from musubi_tuner.dlssnr.runtime import runtime_policy, validate_runtime_device, validate_runtime_policy
from musubi_tuner.training.dlssnr_services import (
    assert_finite_parameters,
    capture_rng,
    create_accelerator,
    optimizer_update,
    restore_rng,
)
from musubi_tuner.training.dlssnr_trainer import NRSupervisedTrainer, NRTrainModule, clear_lane15_state


def setup_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    add_runtime_arguments(parser)
    parser.set_defaults(
        numerics_profile="train_surrogate",
        mixed_precision="no",
        fp8_base=False,
        fp8_scaled=False,
        attention_backend="native",
        attention_scope="all",
    )
    return parser


def benchmark_config(args):
    resolve_geometry(args.width, args.height)
    for name in ("batch_size", "warmup", "steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be at least 1")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    learning_rate = args.learning_rate if args.learning_rate is not None else (1e-4 if args.lora else 1e-5)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if args.fp8_base and not args.lora:
        raise ValueError("FP8 is only supported for a frozen LoRA base")
    config = {
        "model": {
            "model_dir": str(args.model_dir.resolve()),
            "profile": PROFILE_ID,
            "numerics_profile": args.numerics_profile,
            "attention_backend": args.attention_backend,
            "attention_scope": args.attention_scope,
        },
        "data": {"bucket_size": [args.width, args.height], "synthetic": True},
        "precision": {
            "mixed_precision": args.mixed_precision,
            "master_dtype": "float32",
            "fp8_base": args.fp8_base,
            "fp8_scaled": args.fp8_scaled,
        },
        "training": {
            "mode": "single_frame",
            "seed": args.seed,
            "device": "cuda",
            "gradient_accumulation_steps": 1,
            "gradient_checkpointing": args.gradient_checkpointing,
            "max_overflow_retries": 16,
        },
        "optimizer": {
            "type": "AdamW",
            "args": ["weight_decay=0.0"],
            "learning_rate": learning_rate,
            "lr_scheduler": "constant",
            "max_grad_norm": 0.0,
        },
        "parameter_groups": {"prior_lr_multiplier": 0.1, "scale_lr_multiplier": 0.1, "temporal_blend_lr_multiplier": 0.1},
        "loss": {"pre": 1.0, "out": 1.0, "edge": 0.05, "temporal": 0.0},
    }
    if args.lora:
        config["lora"] = {"profile": "vit_only", "rank": args.network_dim, "alpha": args.network_dim, "dropout": 0.0}
        validate_lora(config["lora"])
    validate_runtime_policy(runtime_policy(config), training=True)
    return config


@fp32_execution()
def measure(args):
    config = benchmark_config(args)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("run this single-GPU benchmark in a fresh, non-distributed process")
    source_identity = inspect_canonical(args.model_dir)
    accelerator = create_accelerator(config["training"], config["precision"])
    policy = runtime_policy(config)
    try:
        validate_runtime_device(policy, accelerator.device, training=True)
        model, network, optimizer, base_identity = NRSupervisedTrainer(config, lora=args.lora)._initialize_model(policy)
        owner = NRTrainModule(model, config["loss"], network=network)
        wrapped, optimizer = accelerator.prepare(owner, optimizer)
        source = torch.linspace(0.2, 0.8, args.width * args.height, device=accelerator.device).reshape(
            1, 1, args.height, args.width
        )
        source = source.expand(args.batch_size, 3, args.height, args.width).contiguous()
        controls = torch.zeros(args.batch_size, 5, args.height, args.width, device=accelerator.device)
        controls[:, 1] = 0.5
        batch = {"source": source, "target": source * 0.9, "controls": controls}
        seeds = [args.seed + index for index in range(args.batch_size)]
        times, losses, retries = [], [], []
        for index in range(args.warmup + args.steps):
            if index == args.warmup:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(accelerator.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(accelerator.device)
            rng = capture_rng(accelerator.device)
            torch.cuda.synchronize(accelerator.device)
            start = time.perf_counter()
            for attempt in range(config["training"]["max_overflow_retries"] + 1):
                optimizer.zero_grad(set_to_none=True)
                if attempt:
                    restore_rng(rng)
                loss, metrics = wrapped(batch, seeds)
                accelerator.backward(loss)
                del loss
                model.enforce_lane15()
                if optimizer_update(accelerator, owner, optimizer, 0.0):
                    break
            else:
                raise RuntimeError("FP16 overflow retry budget exhausted")
            clear_lane15_state(model, optimizer)
            assert_finite_parameters(owner)
            torch.cuda.synchronize(accelerator.device)
            if index >= args.warmup:
                times.append(time.perf_counter() - start)
                losses.append(metrics["loss"])
                retries.append(attempt)
        return {
            "schema": "dlssnr_runtime_benchmark_v1",
            "status": "ok",
            "runtime_policy": policy,
            "source_identity": source_identity,
            "base_parameters_sha256": base_identity,
            "workload": {
                "kind": "single_frame_ramp_proxy_v1",
                "resolution": [args.width, args.height],
                "batch_size": args.batch_size,
                "seed": args.seed,
                "warmup_updates": args.warmup,
                "measured_updates": args.steps,
                "lora": args.lora,
                "rank": args.network_dim if args.lora else None,
            },
            "optimizer": config["optimizer"],
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(accelerator.device),
            "losses": losses,
            "overflow_retries": retries,
            "seconds_per_update": times,
            "median_seconds_per_update": statistics.median(times),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(accelerator.device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(accelerator.device) / 2**20,
            "model_storage_mib": sum(value.numel() * value.element_size() for value in (*owner.parameters(), *owner.buffers()))
            / 2**20,
            "native_equivalent": False,
        }
    finally:
        accelerator.end_training()
        accelerator.free_memory()


def main():
    args = setup_parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"benchmark output already exists: {args.output}")
    try:
        result = measure(args)
    except torch.cuda.OutOfMemoryError as error:
        write_json(
            args.output,
            {
                "schema": "dlssnr_runtime_benchmark_v1",
                "status": "out_of_memory",
                "error": str(error),
                "runtime_policy": runtime_policy(benchmark_config(args)),
                "resolution": [args.width, args.height],
                "lora": args.lora,
            },
        )
        raise
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
