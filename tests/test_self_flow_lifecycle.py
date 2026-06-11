import torch

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer

from .test_self_flow_call_dit import FakeAccelerator, make_args


class StubNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.ones(4))


class PreparingAccelerator(FakeAccelerator):
    def prepare(self, m):
        return m


def test_on_train_start_snapshots_ema():
    trainer = Flux2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args()
    net = StubNetwork()
    trainer.on_train_start(args, PreparingAccelerator(), net, None, None)
    assert torch.equal(trainer.ema_lora_state["lora_w"], torch.ones(4))
    # snapshot is a copy, not a view
    net.lora_w.data.fill_(5.0)
    assert torch.equal(trainer.ema_lora_state["lora_w"], torch.ones(4))


def test_on_post_optimizer_step_lerps_only_on_sync():
    trainer = Flux2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(ema_decay=0.9)
    net = StubNetwork()
    trainer.on_train_start(args, PreparingAccelerator(), net, None, None)

    net.lora_w.data.fill_(2.0)
    trainer.on_post_optimizer_step(args, PreparingAccelerator(), net, None, sync_gradients=False, global_step=1)
    assert torch.equal(trainer.ema_lora_state["lora_w"], torch.ones(4))  # unchanged

    trainer.on_post_optimizer_step(args, PreparingAccelerator(), net, None, sync_gradients=True, global_step=1)
    assert torch.allclose(trainer.ema_lora_state["lora_w"], torch.full((4,), 1.1))


def test_no_self_flow_is_noop():
    trainer = Flux2SelfFlowNetworkTrainer()
    args = make_args(self_flow=False)
    trainer.on_train_start(args, PreparingAccelerator(), StubNetwork(), None, None)
    assert trainer.ema_lora_state is None
