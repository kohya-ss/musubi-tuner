import torch

from musubi_tuner.flux_2_train_network_self_flow import Flux2SelfFlowNetworkTrainer

from .test_self_flow_call_dit import FakeAccelerator, make_args


class StubNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_w = torch.nn.Parameter(torch.ones(4))
        self._load_calls: list[str] = []
        # Registry of path -> weight value for load_weights stub
        self._weight_registry: dict[str, float] = {}

    def load_weights(self, path: str):
        """Stub: set lora_w to registered value for this path (or 0.0 if unknown)."""
        self._load_calls.append(path)
        val = self._weight_registry.get(path, 0.0)
        self.lora_w.data.fill_(val)
        return f"loaded from {path}"


class PreparingAccelerator(FakeAccelerator):
    def prepare(self, m):
        return m

    def print(self, *args, **kwargs):
        pass


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


def test_network_weights_ema_without_network_weights_raises():
    """--network_weights_ema requires --network_weights to be set."""
    trainer = Flux2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)
    args = make_args(network_weights_ema="ema.safetensors", network_weights=None)
    import pytest

    with pytest.raises(ValueError, match="network_weights"):
        trainer.on_train_start(args, PreparingAccelerator(), StubNetwork(), None, None)


def test_network_weights_ema_with_both_set_restores_order():
    """With both --network_weights_ema and --network_weights:
    - ema_lora_state equals EMA file weights
    - live network weights equal the student file weights afterward (restore order)
    """
    trainer = Flux2SelfFlowNetworkTrainer()
    trainer.rep_proj = torch.nn.Linear(4, 4)

    student_path = "student.safetensors"
    ema_path = "ema.safetensors"

    args = make_args(network_weights_ema=ema_path, network_weights=student_path)

    net = StubNetwork()
    # EMA file -> lora_w=2.0, student file -> lora_w=3.0
    net._weight_registry[ema_path] = 2.0
    net._weight_registry[student_path] = 3.0

    trainer.on_train_start(args, PreparingAccelerator(), net, None, None)

    # EMA state should reflect the EMA file weights
    assert torch.allclose(trainer.ema_lora_state["lora_w"], torch.full((4,), 2.0))
    # Live network weights must be the student weights (restored last)
    assert torch.allclose(net.lora_w.detach(), torch.full((4,), 3.0))
