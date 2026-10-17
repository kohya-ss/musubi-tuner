from argparse import Namespace

import pytest
import torch

from musubi_tuner.training.optimizer_setup import create_optimizer


def optimizer_args(name, options):
    return Namespace(optimizer_type=name, optimizer_args=options, learning_rate=0.01, lr_scheduler="constant", max_grad_norm=0.0)


@pytest.mark.parametrize("caller", ["shared", "diffusion_trainer"])
@pytest.mark.parametrize("name", ["SGD", "AdamW", "torch.optim.AdamW"])
def test_shared_optimizer_preserves_torch_updates_and_group_learning_rates(caller, name):
    from musubi_tuner.training.trainer_base import NetworkTrainer

    factory = create_optimizer if caller == "shared" else NetworkTrainer().get_optimizer
    left, right = torch.nn.Parameter(torch.tensor([1.0, -2.0])), torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    options = {"weight_decay": 0.02, **({"momentum": 0.9} if name == "SGD" else {"betas": (0.8, 0.95)})}
    args = optimizer_args(name, [f"{key}={value!r}" for key, value in options.items()])
    logged_name, _, actual, train, evaluate = factory(args, [{"params": [left], "lr": 0.001}])
    reference = getattr(torch.optim, name.split(".")[-1])([{"params": [right], "lr": 0.001}], lr=0.01, **options)
    assert logged_name == f"{type(reference).__module__}.{type(reference).__name__}"
    train()
    for _ in range(2):
        left.grad, right.grad = torch.tensor([0.4, -0.2]), torch.tensor([0.4, -0.2])
        actual.step()
        reference.step()
    evaluate()
    torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_shared_adafactor_retains_relative_step_behavior_for_other_architectures():
    parameter = torch.nn.Parameter(torch.ones(2, 2))
    args = optimizer_args("Adafactor", None)
    _, _, optimizer, _, _ = create_optimizer(args, [parameter])
    assert args.learning_rate is None
    assert args.lr_scheduler == "adafactor:0.01"
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter, torch.ones_like(parameter))


def test_nr_rejects_closure_based_optimizers():
    from musubi_tuner.training.dlssnr_services import create_nr_optimizer

    config = {"type": "LBFGS", "args": [], "learning_rate": 0.01, "lr_scheduler": "constant", "max_grad_norm": 0.0}
    with pytest.raises(ValueError, match="closure"):
        create_nr_optimizer([torch.nn.Parameter(torch.ones(2))], config)


def test_frozen_lane_clears_sgd_momentum_as_well_as_adam_state():
    from musubi_tuner.training.dlssnr_trainer import clear_lane15_state
    from test_dlssnr_training import SmallNR

    model = SmallNR()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    weight = model.blocks["0"].input_adapter.weight
    optimizer.state[weight]["momentum_buffer"] = torch.ones_like(weight)
    optimizer.state[weight]["quantized_codes"] = torch.full_like(weight, 127, dtype=torch.uint8)
    clear_lane15_state(model, optimizer)
    assert torch.count_nonzero(optimizer.state[weight]["momentum_buffer"][:, 15]) == 0
    assert torch.all(optimizer.state[weight]["momentum_buffer"][:, :15] == 1)
    assert torch.all(optimizer.state[weight]["quantized_codes"] == 127)
