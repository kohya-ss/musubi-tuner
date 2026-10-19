"""Batched validation is attempt-local and must never cache quantized weights."""

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject
from test_dlssnr_qat import qat_policy
from test_dlssnr_training import SmallNR, make_args, small_math  # noqa: F401
from torch.utils.checkpoint import checkpoint

from musubi_tuner.dlssnr import weight_quantization as quantization
from musubi_tuner.dlssnr.model import DenseFFN, NRModel, _run_block
from musubi_tuner.dlssnr.runtime import configure_model_runtime
from musubi_tuner.training.dlssnr_ema import NRParameterEMA


class SmallCheckpointNR(SmallNR):
    """Keep two resolution levels and real checkpoint/QAT math in runner tests."""

    def __init__(self, profile="dlss_nr_310_8_0"):
        super().__init__(profile)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_min_pixels = 0
        for index in (31, 32):
            block = torch.nn.Module()
            block.ffn, block.attn = DenseFFN(32, 64), torch.nn.Identity()
            for name, parameter in block.named_parameters():
                if name.endswith("skip_scale"):
                    torch.nn.init.ones_(parameter)
                else:
                    torch.nn.init.normal_(parameter, std=0.03)
            self.blocks[str(index)] = block

    enable_gradient_checkpointing = NRModel.enable_gradient_checkpointing

    def forward(self, features, geometry):
        features = features[..., : geometry.valid_height, : geometry.valid_width]
        features = self.blocks["0"].input_adapter(features)
        high = _run_block(
            self.blocks["31"], features, self.gradient_checkpointing, min_pixels=self.gradient_checkpointing_min_pixels
        )
        low = _run_block(
            self.blocks["32"], high[..., ::2, ::2], self.gradient_checkpointing, min_pixels=self.gradient_checkpointing_min_pixels
        )
        features = high + torch.nn.functional.interpolate(low, size=high.shape[-2:], mode="nearest")
        return torch.cat((self.blocks["70"].head.rgb(features), self.blocks["70"].head.logit(features)), dim=1)


def _assert_exact(actual, expected):
    assert type(actual) is type(expected)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_exact(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_exact(left, right)
    else:
        assert actual == expected


def validation(model):
    factory = getattr(quantization, "native_weight_validation", None)
    assert callable(factory), "Attempt-local batched QAT validation is not implemented"
    return factory(model)


def make_model():
    torch.manual_seed(43)
    model = TinyFP8NR()
    configure_model_runtime(model, qat_policy(), training=True)
    return model


def test_batched_validation_uses_one_host_check_through_checkpoint_replay():
    model = make_model()
    source = torch.rand(1, 16, 4, 5)
    expected = checkpoint(lambda value: model(value, None).square(), source, use_reentrant=False)
    expected.sum().backward()
    gradients = {name: value.grad.clone() for name, value in model.named_parameters() if value.grad is not None}
    model.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile, validation(model):
        actual = checkpoint(lambda value: model(value, None).square(), source, use_reentrant=False)
        actual.sum().backward()
    host_checks = sum(item.count for item in profile.key_averages() if item.key == "aten::is_nonzero")
    assert host_checks == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        {name: value.grad for name, value in model.named_parameters() if value.grad is not None}, gradients, rtol=0, atol=0
    )


@pytest.mark.parametrize("bad", [448.1, float("nan"), float("inf")])
def test_preflight_rejects_invalid_weights_before_any_projection(bad):
    model = make_model()
    with torch.no_grad():
        model.blocks["31"].ffn.fc2.weight[0, 0] = bad
    visits = []
    hook = model.blocks["0"].input_adapter.register_forward_pre_hook(lambda *_: visits.append(True))
    try:
        with pytest.raises(ValueError, match="finite|range"), validation(model):
            model(torch.ones(1, 16, 2, 2), None)
        assert not visits
    finally:
        hook.remove()


def test_scope_rechecks_mutated_weights_and_restores_normal_checks_after_error():
    model = make_model()
    layer = model.blocks["31"].ffn.fc1
    source = torch.ones(1, 32)
    with pytest.raises(ValueError, match="finite|range"), validation(model):
        layer(source)
        with torch.no_grad():
            layer.weight.fill_(449)
        layer(source)
    with pytest.raises(ValueError, match="finite|range"):
        layer(source)


def test_next_attempt_and_restored_state_cannot_reuse_previous_checks():
    model = make_model()
    valid = copy.deepcopy(model.state_dict())
    with validation(model):
        model(torch.ones(1, 16, 2, 2), None)
    invalid = copy.deepcopy(valid)
    invalid["blocks.31.ffn.fc1.weight"].fill_(449)
    model.load_state_dict(invalid)
    with pytest.raises(ValueError, match="finite|range"), validation(model):
        pass
    model.load_state_dict(valid)
    with validation(model):
        actual = model(torch.ones(1, 16, 2, 2), None)
    expected = model(torch.ones(1, 16, 2, 2), None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.state_dict().keys() == valid.keys()


def test_ema_swap_invalidates_live_validation_and_failure_restores_weights():
    model = make_model()
    ema = NRParameterEMA(model, 0.5)
    before = copy.deepcopy(model.state_dict())
    ema.shadow["blocks.31.ffn.fc1.weight"].fill_(449)
    with validation(model):
        with pytest.raises(ValueError, match="finite|range"), ema.average_parameters():
            model(torch.ones(1, 16, 2, 2), None)
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)
        actual = model(torch.ones(1, 16, 2, 2), None)
    torch.testing.assert_close(actual, model(torch.ones(1, 16, 2, 2), None), rtol=0, atol=0)


def test_lora_fused_weights_keep_the_original_validation_path():
    model = make_model()
    network = tiny_fp8_inject(model, {})
    with torch.no_grad():
        network.adapters[0].lora_down.fill_(1)
        network.adapters[0].lora_up.fill_(500)
    with pytest.raises(ValueError, match="finite|range"), validation(model):
        model(torch.ones(1, 16, 2, 2), None)


def test_updated_weights_are_requantized_in_each_new_attempt():
    model = make_model()
    layer = model.blocks["31"].ffn.fc1
    with torch.no_grad():
        layer.weight.fill_(1.0)
    with validation(model):
        first = layer.published_weight(layer.weight).detach().clone()
    with torch.no_grad():
        layer.weight.fill_(1.07)
    with validation(model):
        second = layer.published_weight(layer.weight)
    assert torch.equal(first, torch.ones_like(first))
    assert torch.equal(second, torch.full_like(second, 1.125))


@pytest.mark.usefixtures("small_math")
def test_runner_revalidates_each_retry_and_closes_scope_before_optimizer_and_ema(tmp_path, monkeypatch):
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path)
    args.native_weight_qat, args.ema_decay = True, 0.5
    args.output_name = "clean"
    trainer.train_from_args(args)
    expected = torch.load(args.output_dir / "clean/state-step000002/trainer_state.pt", weights_only=True)
    preflights, attempts = [], []
    original_update = trainer.optimizer_update

    def observed(model):
        scope = quantization.native_weight_validation(model)
        preflights.append(True)
        return scope

    def retry_once(accelerator, owner, optimizer, max_norm):
        for module in owner.modules():
            for attribute in ("_native_weight_validation", "_native_blend_validation"):
                assert getattr(module, attribute, None) is None, "Validation must end before updating weights"
        attempts.append(True)
        return False if len(attempts) == 1 else original_update(accelerator, owner, optimizer, max_norm)

    monkeypatch.setattr(trainer, "native_weight_validation", observed, raising=False)
    monkeypatch.setattr(trainer, "optimizer_update", retry_once)
    args.output_name = "retried"
    trainer.train_from_args(args)
    assert len(preflights) == len(attempts) == 3
    actual = torch.load(args.output_dir / "retried/state-step000002/trainer_state.pt", weights_only=True)
    assert actual["global_update"] == actual["ema"]["num_updates"] == 2
    assert actual["consumed_samples"] == expected["consumed_samples"] == 4
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert actual["scheduler"] == expected["scheduler"]
    for key in ("state",):
        torch.testing.assert_close(actual["optimizer"][key], expected["optimizer"][key], rtol=0, atol=0)
    torch.testing.assert_close(actual["ema"]["shadow"], expected["ema"]["shadow"], rtol=0, atol=0)
    for key in ("torch", "cuda"):
        torch.testing.assert_close(actual["rank_states"][0]["rng"][key], expected["rank_states"][0]["rng"][key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))]
)
@pytest.mark.usefixtures("small_math")
def test_qat_selective_checkpoint_and_ema_match_full_checkpoint_and_exact_resume(tmp_path, monkeypatch, device):
    from musubi_tuner.training import dlssnr_trainer as trainer

    # Prime checkpoint's lazy framework imports before comparing the seeded Python RNG streams.
    checkpoint(lambda value: value.square(), torch.ones(1, requires_grad=True), use_reentrant=False).sum().backward()
    monkeypatch.setattr(trainer, "NRModel", SmallCheckpointNR)
    args = make_args(tmp_path)
    args.device = device
    if device == "cuda":
        args.numerics_profile, args.mixed_precision = "train_experimental", "fp16"
    args.native_weight_qat, args.ema_decay, args.gradient_checkpointing = True, 0.5, True
    args.lr_scheduler = "linear"
    states, products = [], []
    for name, threshold, resume in (("full", 0, False), ("selective", 1000, False), ("selective", 1000, True)):
        args.output_name, args.gradient_checkpointing_min_pixels = name, threshold
        folder = args.output_dir / name
        args.resume = folder / "state-step000001" if resume else None
        trainer.train_from_args(args)
        states.append(torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True))
        products.append([load_file(folder / suffix / "model.safetensors") for suffix in ("final", "final/ema")])
    for state, product in zip(states[1:], products[1:]):
        torch.testing.assert_close(product, products[0], rtol=0, atol=0)
        for key in ("optimizer", "scheduler", "ema", "rank_states"):
            _assert_exact(state[key], states[0][key])
        assert state["global_update"] == state["ema"]["num_updates"] == 2
        assert state["consumed_samples"] == 4
    args.gradient_checkpointing_min_pixels = 0
    with pytest.raises(ValueError, match="identity|runtime|config"):
        trainer.train_from_args(args)


def test_qat_selective_checkpoint_resume_in_a_fresh_process(tmp_path):
    script = """
import sys
from pathlib import Path
from test_dlssnr_qat_validation import SmallCheckpointNR
from test_dlssnr_training import make_args
from musubi_tuner.training import dlssnr_trainer as trainer
trainer.NRModel = SmallCheckpointNR
args = make_args(Path(sys.argv[1]))
args.native_weight_qat, args.ema_decay, args.gradient_checkpointing = True, 0.5, True
args.gradient_checkpointing_min_pixels = 1000
args.lr_scheduler = 'linear'
if sys.argv[2] == 'resume':
    args.resume = args.output_dir / args.output_name / 'state-step000001'
trainer.train_from_args(args)
"""
    states, products = [], []
    for mode in ("initial", "resume"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode],
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        folder = tmp_path / "output/dlssnr"
        states.append(torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True))
        products.append([load_file(folder / suffix / "model.safetensors") for suffix in ("final", "final/ema")])
    _assert_exact(states[1], states[0])
    torch.testing.assert_close(products[1], products[0], rtol=0, atol=0)
