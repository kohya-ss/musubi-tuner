"""Validate benchmark workload/policy rather than silently measuring another mode."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "tools/benchmark_dlssnr_runtime.py"
    spec = importlib.util.spec_from_file_location("benchmark_dlssnr_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_resolves_exact_requested_runtime_and_workload():
    module = _module()
    args = module.setup_parser().parse_args(
        [
            "--model_dir",
            "canonical",
            "--output",
            "measurement.json",
            "--width",
            "512",
            "--height",
            "512",
            "--lora",
            "--network_dim",
            "8",
            "--numerics_profile",
            "train_experimental",
            "--mixed_precision",
            "bf16",
            "--gradient_checkpointing",
            "--fp8_base",
            "--fp8_scaled",
            "--sdpa",
            "--warmup",
            "1",
            "--steps",
            "2",
        ]
    )
    config = module.benchmark_config(args)
    assert config["training"]["gradient_checkpointing"] is True
    assert config["precision"]["mixed_precision"] == "bf16"
    assert config["precision"]["fp8_base"] and config["precision"]["fp8_scaled"]
    assert config["model"]["attention_backend"] == "sdpa"
    assert config["lora"]["rank"] == 8
    assert config["optimizer"]["type"] == "AdamW"
    assert config["data"]["bucket_size"] == [512, 512]


@pytest.mark.parametrize(
    "options",
    [
        ["--steps", "0"],
        ["--warmup", "0"],
        ["--batch_size", "0"],
        ["--width", "0"],
        ["--learning_rate", "nan"],
        ["--fp8_base", "--numerics_profile", "train_experimental"],
        ["--mixed_precision", "bf16"],
        ["--lora", "--network_dim", "0"],
    ],
)
def test_benchmark_rejects_invalid_or_unacknowledged_workloads(options):
    module = _module()
    args = module.setup_parser().parse_args(["--model_dir", "canonical", "--output", "measurement.json", *options])
    with pytest.raises(ValueError):
        module.benchmark_config(args)
