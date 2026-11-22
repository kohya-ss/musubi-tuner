"""In-training sample generation must not apply the LoRA network's dropout (issue #1162).

The LoRA modules hook the forward of the transformer's layers, but they are children of the
network, not of the transformer, so the ``transformer.eval()`` call in ``sample_image_inference``
does not reach them. These tests run the ``--sample_at_first`` path of
``NetworkTrainer._run_training_loop`` on a tiny CPU model and check the LoRA output seen while
sampling.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from musubi_tuner.networks import lora as lora_module
from musubi_tuner.training.parser_common import setup_parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer


class _TinyDiT(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16)

    def forward(self, x):
        return self.proj(x)

    # called by NetworkTrainer.sample_images
    def switch_block_swap_for_inference(self):
        pass

    def switch_block_swap_for_training(self):
        pass


class _FakeAccelerator:
    device = torch.device("cpu")
    is_main_process = True
    is_local_main_process = True
    trackers = []

    def unwrap_model(self, model):
        return model

    def print(self, *args, **kwargs):
        pass

    def init_trackers(self, *args, **kwargs):
        pass

    def autocast(self):
        return contextlib.nullcontext()


class _StopTraining(Exception):
    pass


class _ProbeTrainer(NetworkTrainer):
    """Records what do_inference sees and stops the run right after the first sample."""

    def __init__(self, network, x):
        super().__init__()
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.network = network
        self.x = x
        self.network_training = None
        self.outputs = []

    @property
    def architecture(self):
        return "test"

    @property
    def architecture_full_name(self):
        return "test"

    def do_inference(self, accelerator, args, sample_parameter, vae, dit_dtype, transformer, *rest, **kwargs):
        self.network_training = self.network.training
        self.outputs = [transformer(self.x) for _ in range(2)]
        return None  # nothing to save

    def on_after_sample_images(self, *args, **kwargs):
        raise _StopTraining


def _make_network(transformer, **dropout_kwargs):
    network = lora_module.create_network([_TinyDiT.__name__], "lora_unet", 1.0, 4, 4, None, None, transformer, **dropout_kwargs)
    network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
    for lora in network.unet_loras:
        nn.init.normal_(lora.lora_up.weight)  # lora_up starts at zero, which would hide any dropout
    return network


def _run_until_first_sample(trainer, transformer, network, tmp_path):
    args, _ = setup_parser_common().parse_known_args([])
    args.sample_at_first = True
    args.output_dir = str(tmp_path)
    args.output_name = "test"
    args.full_fp16 = args.full_bf16 = False  # set in _prepare_with_accelerator in a real run

    with pytest.raises(_StopTraining):
        trainer._run_training_loop(
            args,
            _FakeAccelerator(),
            session_id=0,
            training_started_at=0.0,
            train_dataset_group=SimpleNamespace(
                num_train_items=1, datasets=[SimpleNamespace(batch_size=1, get_metadata=lambda: {})]
            ),
            train_dataloader=[None],
            current_epoch=SimpleNamespace(value=0),
            transformer=transformer,
            network=network,
            training_model=network,
            optimizer=None,
            optimizer_name="AdamW",
            optimizer_args="",
            optimizer_train_fn=lambda: None,
            optimizer_eval_fn=lambda: None,
            lr_scheduler=None,
            lr_descriptions=None,
            sample_resources=None,
            sample_parameters=[{"prompt": "test", "seed": 42}],
            dit_dtype=torch.float32,
            network_dtype=torch.float32,
        )


@pytest.mark.parametrize(
    "dropout_kwargs",
    [
        {"neuron_dropout": 0.5},  # --network_dropout
        {"rank_dropout": "0.5"},  # --network_args rank_dropout=...
        {"module_dropout": "1.0"},  # --network_args module_dropout=..., 1.0 always skips the LoRA
    ],
    ids=["neuron_dropout", "rank_dropout", "module_dropout"],
)
def test_network_dropout_is_disabled_while_sampling(tmp_path, dropout_kwargs):
    torch.manual_seed(0)
    transformer = _TinyDiT()
    network = _make_network(transformer, **dropout_kwargs)
    x = torch.randn(4, 16)

    with torch.no_grad():
        network.eval()
        expected = transformer(x)  # what standalone inference with these weights gives
        network.train()
        assert not torch.equal(transformer(x), expected)  # dropout is active in training mode

    trainer = _ProbeTrainer(network, x)
    _run_until_first_sample(trainer, transformer, network, tmp_path)

    assert trainer.network_training is False
    for output in trainer.outputs:
        assert torch.equal(output, expected)

    # training mode is restored for the rest of the run
    assert all(module.training for module in network.modules())
    with torch.no_grad():
        assert not torch.equal(transformer(x), expected)
