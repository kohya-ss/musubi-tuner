import torch

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer

from .test_self_flow_call_dit import FakeAccelerator, make_args


def test_rep_proj_merged_into_first_group(tiny_model):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args()
    lora_params = [{"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 1e-4}]
    out = trainer.extra_trainable_params(args, FakeAccelerator(), None, tiny_model, lora_params)
    assert len(out) == 1
    # 2 Linear layers x (weight + bias) = 4 extra params
    assert len(list(out[0]["params"])) == 1 + 4
    assert trainer.rep_proj is not None
    first_linear = trainer.rep_proj[0]
    assert first_linear.in_features == tiny_model.hidden_size


def test_rep_proj_creates_group_when_empty(tiny_model):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args()
    out = trainer.extra_trainable_params(args, FakeAccelerator(), None, tiny_model, [])
    assert len(out) == 1
    assert len(list(out[0]["params"])) == 4


def test_no_self_flow_passthrough(tiny_model):
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    params = [{"params": []}]
    assert trainer.extra_trainable_params(args, FakeAccelerator(), None, tiny_model, params) is params
    assert trainer.rep_proj is None
