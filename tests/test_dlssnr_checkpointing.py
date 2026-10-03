"""Checkpointing must save activations, not change the NR program or its RNG."""

import copy

import pytest
import torch

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.geometry import resolve_geometry
from musubi_tuner.dlssnr.model import DenseFFN, NRModel, WindowAttn, _run_window_stage
from musubi_tuner.networks.lora_dlssnr import DLSSNRLoRA
from test_dlssnr_config import make_args


def _blocks():
    torch.manual_seed(19)
    blocks = torch.nn.ModuleDict()
    for index in range(2):
        block = torch.nn.Module()
        block.ffn = DenseFFN(32, 128)
        block.attn = WindowAttn(32, index)
        for name, value in block.named_parameters():
            if name.endswith(("skip_scale", "temperature")):
                torch.nn.init.ones_(value)
            else:
                torch.nn.init.normal_(value, std=0.03)
        blocks[str(index)] = block
    return blocks


def test_checkpoint_flag_is_effective_and_default_stays_disabled(tmp_path):
    assert not build_train_config(make_args(tmp_path))["training"]["gradient_checkpointing"]
    config = build_train_config(make_args(tmp_path, ["--gradient_checkpointing"]))
    assert config["training"]["gradient_checkpointing"]


@pytest.mark.parametrize("lora", [False, True])
def test_bound_blocks_replay_forward_gradients_and_dropout_rng(lora):
    blocks = _blocks()
    if lora:
        network = DLSSNRLoRA()
        for index in range(2):
            network.add(f"{index}.ffn.fc1.weight", blocks[str(index)].ffn.fc1, 2, 2, 0.3)
        for adapter in network.adapters:
            torch.nn.init.normal_(adapter.lora_up, std=0.02)
        blocks.requires_grad_(False)
        owner = network
    else:
        owner = blocks
    initial = copy.deepcopy(owner.state_dict())
    source = torch.rand(1, 32, 9, 11)  # Frozen input must not discard LoRA gradients.
    results = []
    for enabled in (False, True):
        owner.load_state_dict(initial)
        owner.zero_grad(set_to_none=True)
        torch.manual_seed(71)
        output = _run_window_stage(blocks, source, range(2), checkpointing=enabled)
        output.square().mean().backward()
        gradients = {name: value.grad.clone() for name, value in owner.named_parameters() if value.requires_grad}
        assert gradients and any(torch.count_nonzero(value) for value in gradients.values())
        optimizer = torch.optim.SGD(owner.parameters(), lr=0.01)
        optimizer.step()
        results.append((output.detach(), gradients, copy.deepcopy(owner.state_dict()), torch.rand(7)))
    for expected, actual in zip(results[0], results[1]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_real_model_checkpoints_stem_encoder_vit_decoder_and_final_block():
    torch.manual_seed(12)
    model = NRModel()
    model.enable_gradient_checkpointing()
    model.freeze_single_frame()
    geometry = resolve_geometry(48, 48)
    source = torch.rand(1, 16, geometry.full_height, geometry.full_width)
    visits = {str(index): 0 for index in (0, 4, 14, 30, 31, 38, 40, 48, 62, 66, 70)}
    hooks = []
    for key in visits:

        def count(module, inputs, key=key):
            visits[key] += 1

        hooks.append(model.blocks[key].ffn.register_forward_pre_hook(count))
    try:
        model(source, geometry).square().mean().backward()
        assert all(count >= 2 for count in visits.values()), visits
        assert torch.count_nonzero(model.blocks["70"].head.rgb.weight.grad)
        model.zero_grad(set_to_none=True)
        visits.update({key: 0 for key in visits})
        model.eval()
        model(source, geometry).sum().backward()
        assert all(count == 1 for count in visits.values()), visits
        model.train()
        visits.update({key: 0 for key in visits})
        with torch.no_grad():
            model(source, geometry)
        assert all(count == 1 for count in visits.values()), visits
    finally:
        for hook in hooks:
            hook.remove()


def test_checkpointing_reduces_saved_activation_storage():
    blocks = _blocks()
    source = torch.rand(1, 32, 32, 32)
    weights = {value.untyped_storage().data_ptr() for value in blocks.parameters()}
    retained = []
    outputs = []
    for enabled in (False, True):
        saved = {}

        def pack(value):
            storage = value.untyped_storage()
            if storage.data_ptr() not in weights:
                saved[storage.data_ptr()] = storage.nbytes()
            return value

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
            output = _run_window_stage(blocks, source, range(2), checkpointing=enabled)
        outputs.append(output.detach())
        retained.append(sum(saved.values()))
        del output
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    assert retained[1] < retained[0] / 2, retained


def test_reconfiguring_checkpointing_off_stops_recomputation():
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, default_runtime_policy

    class TinyCheckpoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = _blocks()
            self.gradient_checkpointing = False

        enable_gradient_checkpointing = NRModel.enable_gradient_checkpointing

        def forward(self, source):
            return _run_window_stage(self.blocks, source, range(2), checkpointing=self.gradient_checkpointing)

    model = TinyCheckpoint()
    visits = []
    hook = model.blocks["0"].ffn.register_forward_pre_hook(lambda *_: visits.append(1))
    try:
        for enabled, expected_visits in ((True, 2), (False, 1)):
            visits.clear()
            configure_model_runtime(model, {**default_runtime_policy(), "gradient_checkpointing": enabled}, training=True)
            model(torch.rand(1, 32, 4, 4)).sum().backward()
            assert len(visits) == expected_visits
            model.zero_grad(set_to_none=True)
    finally:
        hook.remove()
