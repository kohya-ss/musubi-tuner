"""Deployment proxies retain independent history and never mutate training state."""

import importlib
import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.evaluation import evaluate
from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy
from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject
from test_dlssnr_training import make_args, small_math  # noqa: F401


def test_quantization_report_counts_lost_updates_and_code_flips():
    quantization = importlib.import_module("musubi_tuner.dlssnr.weight_quantization")
    model = TinyFP8NR()
    weight = model.blocks["31"].ffn.fc1.weight
    with torch.no_grad():
        weight.fill_(1)
    reference = quantization.capture_native_reference(model)
    assert reference["blocks.31.ffn.fc1.weight"].dtype == torch.float8_e4m3fn
    for delta, expected in [(0.01, 0), (0.07, weight.numel())]:
        with torch.no_grad():
            weight.fill_(1 + delta)
        report = quantization.native_quantization_report(model, reference)
        row = next(item for item in report["tensors"] if item["name"] == "blocks.31.ffn.fc1.weight")
        assert row["exported_changed_values"] == expected
        assert row["lost_update_values"] == weight.numel() - expected
        assert report["by_storage"]["e4"]["flip_fraction"] == expected / (weight.numel() * 2)
    assert reference["blocks.31.ffn.fc1.weight"].float().eq(1).all()


def test_quantization_report_includes_lora_delta_not_just_frozen_base():
    quantization = importlib.import_module("musubi_tuner.dlssnr.weight_quantization")
    model = TinyFP8NR()
    with torch.no_grad():
        model.blocks["31"].ffn.fc1.weight.fill_(1)
    network = tiny_fp8_inject(model, {})
    reference = quantization.capture_native_reference(model, network)
    with torch.no_grad():
        network.adapters[0].lora_down.fill_(1)
        network.adapters[0].lora_up.fill_(0.035)
    report = quantization.native_quantization_report(model, reference, network)
    row = next(item for item in report["tensors"] if item["name"] == network.target_names[0])
    assert row["exported_changed_values"] == 2048
    assert row["mean_abs_quantization_error"] == pytest.approx(0.055, abs=1e-6)


def test_native_runtime_restores_flags_after_exception():
    runtime = importlib.import_module("musubi_tuner.dlssnr.runtime")
    from musubi_tuner.dlssnr.model import WindowAttn

    model = TinyFP8NR()
    model.blocks["0"].attn = WindowAttn(32, 0)
    policy = {**default_runtime_policy(), "numerics_profile": "train_experimental", "attention_backend": "sdpa"}
    configure_model_runtime(model, policy, training=True)
    model.gradient_checkpointing = True
    parameters = {name: id(value) for name, value in model.named_parameters()}
    with pytest.raises(RuntimeError, match="interrupted"):
        with runtime.native_weight_runtime(model):
            assert model.blocks["0"].attn.attention_backend == "native"
            assert model.blocks["0"].input_adapter.native_weight_kind == "f16frag"
            assert model.runtime_policy["mixed_precision"] == "no"
            raise RuntimeError("interrupted")
    assert model.runtime_policy == policy
    assert model.gradient_checkpointing is True
    assert not model.native_weight_qat
    assert model.blocks["0"].attn.attention_backend == "sdpa"
    assert model.blocks["0"].input_adapter.native_weight_kind is None
    assert {name: id(value) for name, value in model.named_parameters()} == parameters


@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
def test_native_evaluation_matches_exported_weights_with_independent_history(tmp_path, mode):
    from musubi_tuner.dlssnr.native import quantize_tensor
    from musubi_tuner.dlssnr.weight_quantization import native_storage_kinds
    from musubi_tuner.training.dlssnr_trainer import _datasets

    args = make_args(tmp_path, evaluate=True, mode=mode)
    _, datasets = _datasets(build_train_config(args))
    model = TinyFP8NR().eval()
    configure_model_runtime(model, default_runtime_policy())
    exported = TinyFP8NR().eval()
    kinds = native_storage_kinds()
    exported.load_state_dict(
        {name: torch.from_numpy(quantize_tensor(value.numpy(), kinds[name])) for name, value in model.state_dict().items()}
    )
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    expected = evaluate(exported, datasets, 42, torch.device("cpu"))
    actual = evaluate(model, datasets, 42, torch.device("cpu"), compare_native=True)
    torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert model.runtime_policy == default_runtime_policy()
    for kind in datasets:
        for observed, wanted in zip(actual[kind], expected[kind]):
            assert observed["native"] == wanted
            assert observed["native_gap"]["rgb_mae"] > 0


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.usefixtures("small_math")
def test_qat_training_resumes_and_saves_native_evidence(tmp_path, lora, shuffle):
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path, lora=lora, evaluate=True)
    args.native_weight_qat = args.eval_native = True
    args.shuffle_dataset = shuffle
    if lora:
        args.network_dropout = 0
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    folder = args.output_dir / args.output_name
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected = {name: value.clone() for name, value in load_file(folder / "final" / filename).items()}
    report = json.loads((folder / "final/native_quantization.json").read_text())
    assert report["totals"]["values"] > 0
    assert report["totals"]["exported_changed_values"] > 0
    evaluation = json.loads((folder / "evaluation/step000002.json").read_text())
    assert evaluation["candidate"]["validation"][0]["native_gap"]["rgb_mae"] == 0
    assert evaluation["native_evaluation"]["native_equivalent"] is False
    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), expected, rtol=0, atol=0)
    assert json.loads((folder / "final/native_quantization.json").read_text()) == report
