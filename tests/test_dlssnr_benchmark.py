"""Validate benchmark workload/policy rather than silently measuring another mode."""

import importlib.util
from pathlib import Path

import pytest
import torch


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
            "--gradient_checkpointing_min_pixels",
            "20000",
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
    assert config["training"]["gradient_checkpointing_min_pixels"] == 20000
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
        ["--gradient_checkpointing_min_pixels", "20000"],
        ["--gradient_checkpointing", "--gradient_checkpointing_min_pixels", "-1"],
    ],
)
def test_benchmark_rejects_invalid_or_unacknowledged_workloads(options):
    module = _module()
    args = module.setup_parser().parse_args(["--model_dir", "canonical", "--output", "measurement.json", *options])
    with pytest.raises(ValueError):
        module.benchmark_config(args)


def test_benchmark_does_not_silently_ignore_native_weight_qat():
    from musubi_tuner.dlssnr.runtime import runtime_policy

    module = _module()
    args = module.setup_parser().parse_args(["--model_dir", "canonical", "--output", "measurement.json", "--native_weight_qat"])
    assert runtime_policy(module.benchmark_config(args)).get("native_weight_qat") is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_benchmark_prevalidates_qat_once_per_attempt(tmp_path, monkeypatch):
    from accelerate.state import AcceleratorState
    from test_dlssnr_artifacts import make_canonical
    from test_dlssnr_qat_validation import SmallCheckpointNR

    from musubi_tuner.dlssnr.weight_quantization import native_weight_validation
    from musubi_tuner.training import dlssnr_trainer as trainer

    module = _module()
    model_dir = make_canonical(tmp_path / "base", SmallCheckpointNR())
    monkeypatch.setattr(trainer, "NRModel", SmallCheckpointNR)
    attempts = []

    def observed(model):
        attempts.append(True)
        return native_weight_validation(model)

    monkeypatch.setattr(module, "native_weight_validation", observed, raising=False)
    args = module.setup_parser().parse_args(
        [
            "--model_dir",
            str(model_dir),
            "--output",
            str(tmp_path / "result.json"),
            "--width",
            "48",
            "--height",
            "48",
            "--warmup",
            "1",
            "--steps",
            "1",
            "--native_weight_qat",
        ]
    )
    AcceleratorState._reset_state(reset_partial_state=True)
    try:
        result = module.measure(args)
        assert len(attempts) == 2
        assert result["status"] == "ok"
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
