"""Optional FP32 parameter EMA must not alter the live training trajectory."""

import copy
import importlib
import importlib.util
import json

import pytest
import torch
from safetensors.torch import load_file

from musubi_tuner.dlssnr.config import build_train_config, config_sha256
from test_dlssnr_config import make_args as config_args
from test_dlssnr_training import make_args, small_math  # noqa: F401


def ema_class():
    name = "musubi_tuner.training.dlssnr_ema"
    assert importlib.util.find_spec(name) is not None, "NR parameter EMA is not implemented"
    return importlib.import_module(name).NRParameterEMA


def tiny_module():
    module = torch.nn.Module()
    module.weight = torch.nn.Parameter(torch.tensor([0.0, 2.0]))
    module.frozen = torch.nn.Parameter(torch.tensor([9.0]), requires_grad=False)
    module.register_buffer("counter", torch.tensor(3))
    return module


def assert_ema_state(actual, expected):
    assert {key: value for key, value in actual.items() if key != "shadow"} == {
        key: value for key, value in expected.items() if key != "shadow"
    }
    torch.testing.assert_close(actual["shadow"], expected["shadow"], rtol=0, atol=0)


@pytest.mark.parametrize("lora", [False, True])
def test_ema_cli_is_disabled_until_a_decay_is_supplied(tmp_path, lora):
    args = config_args(tmp_path, lora=lora)
    assert args.ema_decay is None
    plain = build_train_config(args, lora=lora)
    assert "ema_decay" not in plain["training"]
    enabled = build_train_config(config_args(tmp_path, ["--ema_decay", "0.999"], lora=lora), lora=lora)
    assert enabled["training"]["ema_decay"] == 0.999
    assert config_sha256(enabled) != config_sha256(plain)


@pytest.mark.parametrize("value", [0.0, -0.1, 1.0, 1.1, float("nan"), float("inf"), True])
def test_ema_rejects_invalid_decay_before_training(tmp_path, value):
    args = config_args(tmp_path)
    args.ema_decay = value
    with pytest.raises(ValueError, match="ema_decay"):
        build_train_config(args)


def test_ema_averages_only_trainable_fp32_parameters_from_initial_values():
    module = tiny_module()
    rng = torch.get_rng_state().clone()
    ema = ema_class()(module, 0.5)
    assert ema.num_updates == 0
    assert set(ema.shadow) == {"weight"}
    with torch.no_grad():
        module.weight.copy_(torch.tensor([2.0, 4.0]))
    ema.update()
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([1.0, 3.0]), rtol=0, atol=0)
    with torch.no_grad():
        module.weight.copy_(torch.tensor([4.0, 6.0]))
    ema.update()
    assert ema.num_updates == 2
    torch.testing.assert_close(ema.shadow["weight"], torch.tensor([2.5, 4.5]), rtol=0, atol=0)
    torch.testing.assert_close(module.weight, torch.tensor([4.0, 6.0]), rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    state = ema.state_dict()
    assert state["schema"] == "dlssnr_parameter_ema_v1"
    assert state["decay"] == 0.5 and state["num_updates"] == 2
    assert state["shadow"]["weight"].device.type == "cpu"
    assert state["shadow"]["weight"].dtype == torch.float32
    state["shadow"]["weight"].zero_()
    assert ema.shadow["weight"].ne(0).all()


def test_ema_swap_restores_parameters_and_gradients_even_after_failure():
    module = tiny_module()
    ema = ema_class()(module, 0.5)
    with torch.no_grad():
        module.weight.add_(4)
    module.weight.grad = torch.tensor([3.0, 5.0])
    ema.update()
    before = {name: value.clone() for name, value in module.state_dict().items()}
    parameter_id = id(module.weight)
    gradient = module.weight.grad.clone()
    rng = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="evaluation failed"):
        with ema.average_parameters():
            torch.testing.assert_close(module.weight, ema.shadow["weight"], rtol=0, atol=0)
            assert id(module.weight) == parameter_id
            with pytest.raises(RuntimeError, match="active"):
                ema.update()
            with pytest.raises(RuntimeError, match="active"):
                with ema.average_parameters():
                    pass
            raise RuntimeError("evaluation failed")
    torch.testing.assert_close(module.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(module.weight.grad, gradient, rtol=0, atol=0)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert id(module.weight) == parameter_id
    ema.update()
    assert ema.num_updates == 2


@pytest.mark.parametrize("corruption", ["schema", "decay", "updates", "keys", "shape", "dtype", "nonfinite"])
def test_ema_restore_validates_all_state_before_replacing_shadows(corruption):
    ema = ema_class()(tiny_module(), 0.5)
    ema.update()
    saved = ema.state_dict()
    state = copy.deepcopy(saved)
    if corruption == "schema":
        state["schema"] = "unknown"
    elif corruption == "decay":
        state["decay"] = 0.9
    elif corruption == "updates":
        state["num_updates"] = 2
    elif corruption == "keys":
        state["shadow"]["unknown"] = torch.zeros(1)
    elif corruption == "shape":
        state["shadow"]["weight"] = torch.zeros(3)
    elif corruption == "dtype":
        state["shadow"]["weight"] = state["shadow"]["weight"].half()
    else:
        state["shadow"]["weight"][0] = float("nan")
    with pytest.raises(ValueError, match="EMA"):
        ema.load_state_dict(state, expected_updates=1)
    assert_ema_state(ema.state_dict(), saved)


def test_ema_restored_state_continues_the_same_recurrence():
    module = tiny_module()
    first = ema_class()(module, 0.75)
    with torch.no_grad():
        module.weight.add_(2)
    first.update()
    second = ema_class()(module, 0.75)
    second.load_state_dict(first.state_dict(), expected_updates=1)
    with torch.no_grad():
        module.weight.add_(4)
    first.update()
    second.update()
    assert_ema_state(first.state_dict(), second.state_dict())


def test_ema_refuses_non_fp32_trainable_parameters():
    with pytest.raises(ValueError, match="FP32"):
        ema_class()(tiny_module().half(), 0.5)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("mode", ["single_frame", "temporal"])
@pytest.mark.parametrize("qat", [False, True])
@pytest.mark.usefixtures("small_math")
def test_ema_products_evaluation_and_exact_resume_preserve_raw_training(tmp_path, lora, mode, qat):
    from musubi_tuner.dlssnr.runtime import runtime_policy
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path, lora=lora, mode=mode, evaluate=True)
    args.native_weight_qat = args.eval_native = qat
    if lora and qat:
        args.network_dropout = 0
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    args.output_name = "plain"
    train(args)
    plain = args.output_dir / args.output_name
    filename = "adapter.safetensors" if lora else "model.safetensors"
    expected_raw = load_file(plain / "final" / filename)
    plain_state = torch.load(plain / "state-step000002/trainer_state.pt", weights_only=True)
    assert "ema" not in plain_state
    assert not (plain / "final/ema").exists()
    assert "ema_candidate" not in json.loads((plain / "evaluation/step000002.json").read_text())

    args.output_name, args.ema_decay = "averaged", 0.5
    train(args)
    folder = args.output_dir / args.output_name
    assert (folder / "final/ema" / filename).is_file(), "EMA must be a separate, deployable product"
    torch.testing.assert_close(load_file(folder / "final" / filename), expected_raw, rtol=0, atol=0)
    first = torch.load(folder / "state-step000001/trainer_state.pt", weights_only=True)
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert first["ema"]["num_updates"] == 1
    assert state["ema"]["num_updates"] == state["global_update"] == 2
    assert state["scheduler"] == plain_state["scheduler"]
    assert state["optimizer"]["param_groups"] == plain_state["optimizer"]["param_groups"]
    torch.testing.assert_close(state["optimizer"]["state"], plain_state["optimizer"]["state"], rtol=0, atol=0)
    torch.testing.assert_close(state["rank_states"][0]["rng"]["torch"], plain_state["rank_states"][0]["rng"]["torch"])
    averaged = load_file(folder / "final/ema" / filename)
    assert any(not torch.equal(averaged[name], value) for name, value in expected_raw.items())
    for name, value in state["ema"]["shadow"].items():
        key = name.split(".", 1)[1]
        expected = torch.lerp(first["ema"]["shadow"][name], expected_raw[key], 0.5)
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
        torch.testing.assert_close(averaged[key], value, rtol=0, atol=0)
    if not lora:
        assert averaged["blocks.0.input_adapter.weight"][:, 15].eq(0).all()
        if mode == "single_frame":
            for name in ("blocks.70.head.logit.weight", "blocks.70.blend_scale"):
                torch.testing.assert_close(averaged[name], expected_raw[name], rtol=0, atol=0)
    for variant, directory in (("raw", folder / "final"), ("ema", folder / "final/ema")):
        metadata = json.loads((directory / "training_metadata.json").read_text())
        assert metadata["weight_variant"] == variant
        assert metadata["ema"]["num_updates"] == 2
        assert metadata["ema"]["scope"] == ("lora_parameters" if lora else "trainable_parameters")
        if qat:
            report = json.loads((directory / "native_quantization.json").read_text())
            assert report["totals"]["values"] > 0
    evaluation = json.loads((folder / "evaluation/step000002.json").read_text())
    assert evaluation["ema"]["num_updates"] == 2
    plain_evaluation = json.loads((plain / "evaluation/step000002.json").read_text())
    assert evaluation["candidate"] == plain_evaluation["candidate"]
    config = build_train_config(args, lora=lora)
    model, network, _, _ = trainer.NRSupervisedTrainer(config, lora=lora)._initialize_model(runtime_policy(config))
    (network if lora else model).load_state_dict(averaged)
    model.eval()
    if network is not None:
        network.eval()
    _, validation = trainer._datasets(config)
    expected_evaluation = trainer.evaluate(model, validation, args.seed, torch.device("cpu"), compare_native=qat)
    assert evaluation["ema_candidate"] == expected_evaluation

    args.resume = folder / "state-step000001"
    train(args)
    torch.testing.assert_close(load_file(folder / "final" / filename), expected_raw, rtol=0, atol=0)
    torch.testing.assert_close(load_file(folder / "final/ema" / filename), averaged, rtol=0, atol=0)
    resumed = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert_ema_state(resumed["ema"], state["ema"])
    torch.testing.assert_close(resumed["rank_states"][0]["rng"]["torch"], state["rank_states"][0]["rng"]["torch"])
    assert json.loads((folder / "evaluation/step000002.json").read_text()) == evaluation
    args.ema_decay = 0.75
    with pytest.raises(ValueError, match="identity"):
        train(args)
    args.ema_decay = None
    with pytest.raises(ValueError, match="identity"):
        train(args)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.usefixtures("small_math")
def test_ema_skips_overflow_retries_and_accumulation_microsteps(tmp_path, monkeypatch, lora):
    from musubi_tuner.training import dlssnr_trainer as trainer

    args = make_args(tmp_path, lora=lora, accum=4)
    args.ema_decay = 0.5
    train = trainer.train_lora_from_args if lora else trainer.train_from_args
    train(args)
    original = torch.load(args.output_dir / args.output_name / "state-step000002/trainer_state.pt", weights_only=True)
    assert "ema" in original, "EMA must be included in the optimizer-boundary state"
    update = trainer.optimizer_update
    attempts = []

    def overflow_once(*values):
        attempts.append(True)
        return False if len(attempts) == 1 else update(*values)

    monkeypatch.setattr(trainer, "optimizer_update", overflow_once)
    args.output_name = "retried"
    train(args)
    restored = torch.load(args.output_dir / args.output_name / "state-step000002/trainer_state.pt", weights_only=True)
    assert len(attempts) == 3
    assert restored["ema"]["num_updates"] == 2
    assert_ema_state(restored["ema"], original["ema"])


@pytest.mark.parametrize("requested", [False, True])
def test_restore_rejects_missing_or_unrequested_ema_state(requested):
    from musubi_tuner.training.dlssnr_state import restore_state

    module = tiny_module()
    ema = ema_class()(module, 0.5)
    payload = {"global_update": 0}
    if not requested:
        payload["ema"] = ema.state_dict()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="EMA"):
        restore_state(payload, optimizer, ema=ema if requested else None)


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.usefixtures("small_math")
def test_ema_with_qat_preserves_frozen_fp8_base_and_adapter_metadata(tmp_path, monkeypatch, scaled):
    from safetensors import safe_open
    from musubi_tuner.networks import lora_dlssnr
    from musubi_tuner.training import dlssnr_trainer as trainer
    from test_dlssnr_fp8 import TinyFP8NR, tiny_fp8_inject

    monkeypatch.setattr(trainer, "NRModel", TinyFP8NR)
    monkeypatch.setattr(lora_dlssnr, "inject", tiny_fp8_inject)
    args = make_args(tmp_path, lora=True, evaluate=True)
    args.ema_decay, args.network_dropout = 0.5, 0
    args.numerics_profile, args.fp8_base, args.fp8_scaled = "train_experimental", True, scaled
    args.native_weight_qat = args.eval_native = True
    trainer.train_lora_from_args(args)
    folder = args.output_dir / args.output_name
    state = torch.load(folder / "state-step000002/trainer_state.pt", weights_only=True)
    assert "ema" in state, "EMA state must cover only the trainable LoRA parameters"
    assert all(name.startswith("network.") for name in state["ema"]["shadow"])
    with safe_open(folder / "final/adapter.safetensors", framework="pt") as handle:
        original_metadata = handle.metadata()
    with safe_open(folder / "final/ema/adapter.safetensors", framework="pt") as handle:
        assert handle.metadata() == original_metadata
    averaged = load_file(folder / "final/ema/adapter.safetensors")
    args.resume = folder / "state-step000001"
    trainer.train_lora_from_args(args)
    torch.testing.assert_close(load_file(folder / "final/ema/adapter.safetensors"), averaged, rtol=0, atol=0)
