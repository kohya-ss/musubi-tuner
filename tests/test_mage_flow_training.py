from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.mage_flow.sampling import build_scheduler, euler_step
from musubi_tuner.mage_flow.training import sigma_from_training_timesteps, unpack_target_predictions
from musubi_tuner.mage_flow_train_network import MageFlowNetworkTrainer


class _FakeAccelerator:
    device = torch.device("cpu")

    @staticmethod
    def unwrap_model(model):
        return model

    @staticmethod
    def autocast():
        return nullcontext()


class _EchoTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_packed = None

    def forward(self, packed):
        self.last_packed = packed
        return packed.image_tokens


def test_euler_step_uses_official_epsilon_minus_z_direction():
    latent = torch.tensor([5.0])
    velocity = torch.tensor([3.0])

    stepped = euler_step(latent, velocity, sigma=1.0, next_sigma=0.0)

    torch.testing.assert_close(stepped, torch.tensor([2.0]))


def test_scheduler_uses_static_shift_and_terminal_zero():
    scheduler = build_scheduler(num_steps=4, device="cpu", shift=6.0)
    base = torch.tensor([1.0, 0.75, 0.5, 0.25])
    expected = 6.0 * base / (1.0 + 5.0 * base)

    torch.testing.assert_close(scheduler.sigmas, torch.cat([expected, torch.zeros(1)]))


def test_training_timesteps_recover_exact_sigmas():
    timesteps = torch.tensor([201.0, 801.0])

    torch.testing.assert_close(sigma_from_training_timesteps(timesteps), torch.tensor([0.2, 0.8]))


def test_unpack_predictions_selects_only_each_target():
    from musubi_tuner.mage_flow.utils import pack_training_batch

    targets = [torch.zeros(2, 2, 3), torch.zeros(2, 1, 2)]
    refs = [[torch.zeros(2, 1, 1)], [torch.zeros(2, 2, 1)]]
    text = [torch.zeros(2, 5), torch.zeros(1, 5)]
    packed = pack_training_batch(targets, text, torch.tensor([0.2, 0.8]), refs)
    prediction = torch.arange(packed.image_tokens.numel(), dtype=torch.float32).reshape_as(packed.image_tokens)

    unpacked = unpack_target_predictions(prediction, packed)

    assert [tuple(item.shape) for item in unpacked] == [(2, 2, 3), (2, 1, 2)]
    first_flat = prediction[0, :6].reshape(2, 3, 2).permute(2, 0, 1)
    second_start = int(packed.image_cu_seqlens[1])
    second_flat = prediction[0, second_start : second_start + 2].reshape(1, 2, 2).permute(2, 0, 1)
    torch.testing.assert_close(unpacked[0], first_flat)
    torch.testing.assert_close(unpacked[1], second_flat)


def test_trainer_packs_clean_edit_refs_and_losses_only_target():
    trainer = MageFlowNetworkTrainer()
    trainer.is_edit = True
    transformer = _EchoTransformer()
    latents = torch.full((2, 4, 2, 3), 10.0)
    noisy = torch.stack([torch.full((4, 2, 3), 2.0), torch.full((4, 2, 3), 8.0)])
    noise = torch.full_like(latents, 13.0)
    ref = torch.stack([torch.full((4, 1, 2), 20.0), torch.full((4, 1, 2), 30.0)])
    batch = {
        "mage_flow_embed": [torch.zeros(3, 6), torch.zeros(2, 6)],
        "latents_control_0": ref,
    }
    args = SimpleNamespace(gradient_checkpointing=False)

    output = trainer.call_dit(
        args,
        _FakeAccelerator(),
        transformer,
        latents,
        batch,
        noise,
        noisy,
        torch.tensor([201.0, 801.0]),
        torch.float32,
    )

    torch.testing.assert_close(output.pred, noisy)
    torch.testing.assert_close(output.target, noise - latents)
    torch.testing.assert_close(transformer.last_packed.timesteps, torch.tensor([0.2, 0.8]))
    image_cu = transformer.last_packed.image_cu_seqlens.tolist()
    for sample_index, (start, end) in enumerate(zip(image_cu, image_cu[1:])):
        target_len = 6
        packed_ref = transformer.last_packed.image_tokens[0, start + target_len : end]
        expected_ref = ref[sample_index].permute(1, 2, 0).reshape(-1, 4)
        torch.testing.assert_close(packed_ref, expected_ref)


def test_trainer_rejects_missing_or_noncontiguous_edit_references():
    trainer = MageFlowNetworkTrainer()
    trainer.is_edit = True
    common = dict(
        args=SimpleNamespace(gradient_checkpointing=False),
        accelerator=_FakeAccelerator(),
        transformer_arg=_EchoTransformer(),
        latents=torch.zeros(1, 4, 2, 2),
        noise=torch.ones(1, 4, 2, 2),
        noisy_model_input=torch.zeros(1, 4, 2, 2),
        timesteps=torch.tensor([501.0]),
        network_dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="between 1 and 3"):
        trainer.call_dit(batch={"mage_flow_embed": [torch.zeros(1, 6)]}, **common)
    with pytest.raises(ValueError, match="contiguous"):
        trainer.call_dit(
            batch={
                "mage_flow_embed": [torch.zeros(1, 6)],
                "latents_control_0": torch.zeros(1, 4, 2, 2),
                "latents_control_2": torch.zeros(1, 4, 2, 2),
            },
            **common,
        )
