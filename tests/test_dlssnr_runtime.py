"""Explicit precision boundaries and successful-update state transactions."""

import json

import pytest
import torch
from accelerate.state import AcceleratorState
from safetensors.torch import load_file

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.model import ChannelLinear
from musubi_tuner.training import dlssnr_trainer as trainer
from test_dlssnr_config import make_args as config_args
from test_dlssnr_training import SmallNR, make_args, small_math  # noqa: F401


@pytest.fixture(autouse=True)
def reset_accelerator():
    AcceleratorState._reset_state(reset_partial_state=True)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def test_runtime_defaults_identify_the_unchanged_baseline(tmp_path):
    from musubi_tuner.dlssnr.runtime import runtime_policy

    policy = runtime_policy(build_train_config(config_args(tmp_path)))
    assert policy["numerics_profile"] == "train_surrogate"
    assert policy["mixed_precision"] == "no"
    assert policy["attention_backend"] == "native"
    assert not policy["gradient_checkpointing"]
    assert not policy["fp8_base"]


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_experimental_precision_accepts_canonical_training_without_smoke(tmp_path, precision):
    args = config_args(tmp_path, ["--numerics_profile", "train_experimental", "--mixed_precision", precision])
    args.development_smoke = False
    args.model_dir = tmp_path / "converted"
    config = build_train_config(args)
    assert config["precision"]["mixed_precision"] == precision
    assert config["precision"]["master_dtype"] == "float32"
    assert config["training"]["max_overflow_retries"] == 16


@pytest.mark.parametrize(
    "options,match",
    [
        (["--mixed_precision", "bf16"], "train_experimental"),
        (["--numerics_profile", "train_experimental", "--deployment_target", "native_roundtrip"], "float_runtime"),
        (["--max_overflow_retries", "-1"], "max_overflow_retries"),
        (["--numerics_profile", "train_experimental", "--mixed_precision", "bf16", "--device", "cpu"], "CUDA"),
    ],
)
def test_precision_policy_rejects_invalid_combinations(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        build_train_config(config_args(tmp_path, options))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_projection_executes_low_precision_but_keeps_master_and_output_fp32(tmp_path, precision):
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, runtime_policy

    model = SmallNR().cuda()
    config = build_train_config(config_args(tmp_path, ["--numerics_profile", "train_experimental", "--mixed_precision", precision]))
    configure_model_runtime(model, runtime_policy(config), training=True)
    module = model.blocks["0"].input_adapter
    source = torch.linspace(0.13, 0.93, 16 * 5 * 5, device="cuda").reshape(1, 16, 5, 5)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    with torch.autocast("cuda", dtype=dtype):
        expected = torch.nn.functional.conv2d(source, module.weight[:, :, None, None]).float()
    actual = module(source)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    assert module.weight.dtype == module.weight.grad.dtype == torch.float32
    assert torch.isfinite(module.weight.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
@pytest.mark.usefixtures("small_math")
def test_amp_runner_resumes_scaler_weights_and_successful_counters(tmp_path, precision, lora, mode):
    args = make_args(tmp_path, lora=lora, mode=mode)
    args.device, args.numerics_profile, args.mixed_precision = "cuda", "train_experimental", precision
    args.max_grad_norm = 0.01
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = {name: value.clone() for name, value in load_file(folder / "final" / filename).items()}
    saved = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert saved["schema"] == "dlssnr_train_state_v3"
    scaler = saved["rank_states"][0]["scaler"]
    assert (scaler is not None) == (precision == "fp16")
    if scaler is not None:
        assert scaler["scale"] > 0
    metadata = json.loads((folder / "run_config.json").read_text())
    assert metadata["runtime_policy"]["mixed_precision"] == precision
    assert metadata["numerics"]["profile"] == "train_experimental"
    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), expected, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert resumed["consumed_samples"] == 4
    assert resumed["global_update"] == 2
    assert resumed["rank_states"][0]["scaler"] == scaler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.usefixtures("small_math")
def test_fp16_overflow_retries_same_batch_without_advancing_rng_or_counters(tmp_path, monkeypatch):
    from musubi_tuner.training import dlssnr_services as services

    real_create = services.create_accelerator
    for name, initial_scale in (("clean", 1.0), ("overflow", 2**32)):

        def create(training, precision=None, initial_scale=initial_scale):
            accelerator = real_create(training, precision)
            accelerator.scaler.load_state_dict(
                {
                    "scale": float(initial_scale),
                    "growth_factor": 2.0,
                    "backoff_factor": 0.5,
                    "growth_interval": 2000,
                    "_growth_tracker": 0,
                }
            )
            return accelerator

        monkeypatch.setattr(trainer, "create_accelerator", create)
        args = make_args(tmp_path / name, lora=True)
        args.device, args.numerics_profile, args.mixed_precision = "cuda", "train_experimental", "fp16"
        args.max_overflow_retries = 40
        trainer.train_lora_from_args(args)
        AcceleratorState._reset_state(reset_partial_state=True)
    clean = tmp_path / "clean/output/dlssnr"
    overflow = tmp_path / "overflow/output/dlssnr"
    left, right = [load_file(folder / "final/adapter.safetensors") for folder in (clean, overflow)]
    # Loss scaling changes rounding, not the selected samples or dropout RNG stream.
    torch.testing.assert_close(left, right, rtol=0.02, atol=5e-5)
    first = torch.load(clean / "state-step000002/trainer_state.pt", weights_only=True)
    last = torch.load(overflow / "state-step000002/trainer_state.pt", weights_only=True)
    assert last["global_update"] == first["global_update"] == 2
    assert last["consumed_samples"] == first["consumed_samples"] == 4
    torch.testing.assert_close(last["rank_states"][0]["rng"]["torch"], first["rank_states"][0]["rng"]["torch"], rtol=0, atol=0)
    torch.testing.assert_close(last["rank_states"][0]["rng"]["cuda"], first["rank_states"][0]["rng"]["cuda"], rtol=0, atol=0)
    records = [json.loads(line) for line in (overflow / "metrics.jsonl").read_text().splitlines()]
    assert len(records) == 2 and sum(item["overflow_retries"] for item in records) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.usefixtures("small_math")
def test_fp16_unrecoverable_gradient_never_saves_a_success(tmp_path, monkeypatch):
    def corrupt_model():
        model = SmallNR()
        model.blocks["70"].head.rgb.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        return model

    monkeypatch.setattr(trainer, "NRModel", corrupt_model)
    args = make_args(tmp_path)
    args.device, args.numerics_profile, args.mixed_precision = "cuda", "train_experimental", "fp16"
    args.max_overflow_retries = 1
    with pytest.raises(RuntimeError, match="overflow|non-finite"):
        trainer.train_from_args(args)
    assert not list(args.output_dir.rglob("trainer_state.pt"))
    assert not (args.output_dir / args.output_name / "final").exists()


def test_default_projection_resists_unrequested_outer_autocast():
    linear = ChannelLinear(3, 16)
    torch.nn.init.normal_(linear.weight, std=0.1)
    source = torch.rand(1, 16, 4, 4)
    expected = linear(source)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = linear(source)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
