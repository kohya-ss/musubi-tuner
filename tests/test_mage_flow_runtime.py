import pytest
import torch

import musubi_tuner.mage_flow.model as model_module
import musubi_tuner.mage_flow_train_network as train_module
from musubi_tuner.mage_flow.model import MageFlow
from musubi_tuner.mage_flow.utils import MageFlowConfig, pack_training_batch
from musubi_tuner.mage_flow_train_network import MageFlowNetworkTrainer
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig


def _config(depth=3, checkpoint=False):
    return MageFlowConfig(
        in_channels=4,
        out_channels=4,
        context_in_dim=6,
        hidden_size=16,
        depth=depth,
        num_heads=2,
        axes_dim=(2, 2, 4),
        text_max_length=16,
        checkpoint=checkpoint,
    )


class _FakeOffloader:
    def __init__(self):
        self.waited = []
        self.submitted = []
        self.forward_only = None
        self.prepared = 0

    def wait_for_block(self, index):
        self.waited.append(index)

    def submit_move_blocks_forward(self, blocks, index):
        assert len(blocks) == 3
        self.submitted.append(index)

    def set_forward_only(self, value):
        self.forward_only = value

    def prepare_block_devices_before_forward(self, blocks):
        assert len(blocks) == 3
        self.prepared += 1


def test_block_swap_bounds_and_forward_dispatch(monkeypatch):
    fake = _FakeOffloader()
    monkeypatch.setattr(model_module, "create_offloader", lambda *_args, **_kwargs: fake)
    model = MageFlow(_config()).eval()
    config = BlockSwapConfig(device=torch.device("cpu"), supports_backward=False)

    with pytest.raises(ValueError, match="0 through 1"):
        model.enable_block_swap(2, config)
    model.enable_block_swap(1, config)
    model.prepare_block_swap_before_forward()
    packed = pack_training_batch(
        torch.randn(1, 4, 2, 2),
        [torch.randn(2, 6)],
        torch.tensor([0.5]),
    )
    with torch.no_grad():
        model(packed)

    assert fake.prepared == 1
    assert fake.waited == [0, 1, 2]
    assert fake.submitted == [0, 1, 2]
    model.switch_block_swap_for_inference()
    assert fake.forward_only is True
    model.switch_block_swap_for_training()
    assert fake.forward_only is False


def test_runtime_hooks_are_noops_without_swap_and_checkpoint_aliases_work():
    model = MageFlow(_config())

    model.move_to_device_except_swap_blocks(torch.device("cpu"))
    model.prepare_block_swap_before_forward()
    model.switch_block_swap_for_inference()
    model.switch_block_swap_for_training()
    model.enable_gradient_checkpointing(cpu_offload=False)
    assert model.checkpoint is True
    model.disable_gradient_checkpointing()
    assert model.checkpoint is False


def test_trainer_compile_targets_only_repeated_blocks(monkeypatch):
    captured = {}

    def fake_compile(args, model, groups, disable_linear):
        captured.update(args=args, model=model, groups=groups, disable_linear=disable_linear)
        return model

    monkeypatch.setattr(train_module.model_utils, "compile_transformer", fake_compile)
    trainer = MageFlowNetworkTrainer()
    trainer.blocks_to_swap = 0
    model = MageFlow(_config())
    args = object()

    assert trainer.compile_transformer(args, model) is model
    assert captured["groups"] == [model.transformer_blocks]
    assert captured["disable_linear"] is False
