from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.mage_flow.sampling import build_scheduler, euler_step, sample_latents
from musubi_tuner.mage_flow.training import sigma_from_training_timesteps, unpack_target_predictions
import musubi_tuner.mage_flow_train_network as train_module
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


class _ZeroVelocity(torch.nn.Module):
    def forward(self, packed):
        return torch.zeros_like(packed.image_tokens)


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


def test_fixed_seed_sampling_is_deterministic():
    kwargs = {
        "transformer": _ZeroVelocity(),
        "text_tokens": [torch.zeros(2, 5)],
        "latent_shapes": [(2, 3)],
        "steps": 2,
        "seeds": [42],
        "channels": 4,
        "device": "cpu",
        "dtype": torch.float32,
    }

    first = sample_latents(**kwargs)
    second = sample_latents(**kwargs)
    different = sample_latents(**{**kwargs, "seeds": [43]})

    torch.testing.assert_close(first[0], second[0])
    assert not torch.equal(first[0], different[0])


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


def test_parser_defaults_to_the_fixed_mage_flow_lora_and_official_training_schedule():
    parser = train_module.mage_flow_setup_parser(train_module.setup_parser_common())
    args = parser.parse_args([])

    assert args.network_module == "musubi_tuner.networks.lora_mage_flow"
    assert args.timestep_sampling == "shift"
    assert args.discrete_flow_shift == 6.0
    assert args.weighting_scheme == "none"


def test_trainer_rejects_plain_fp8_and_non_mage_network_module():
    trainer = MageFlowNetworkTrainer()
    common = {
        "is_edit": False,
        "mixed_precision": "bf16",
        "fp8_base": False,
        "fp8_scaled": False,
        "network_module": "musubi_tuner.networks.lora_mage_flow",
        "sage_attn": False,
        "xformers": False,
        "flash3": False,
        "sdpa": True,
        "flash_attn": False,
        "blocks_to_swap": 0,
    }
    with pytest.raises(ValueError, match="fp8_scaled"):
        trainer.handle_model_specific_args(SimpleNamespace(**{**common, "fp8_base": True}))
    with pytest.raises(ValueError, match="LoRA-only"):
        trainer.handle_model_specific_args(SimpleNamespace(**{**common, "network_module": "some.other.network"}))
    with pytest.raises(ValueError, match="0 through 10"):
        trainer.handle_model_specific_args(SimpleNamespace(**{**common, "blocks_to_swap": 11}))


def test_training_sample_vae_load_requires_decoder(monkeypatch):
    captured = {}

    def fake_loader(path, **kwargs):
        captured.update(path=path, **kwargs)
        return object()

    monkeypatch.setattr(train_module, "load_mage_vae", fake_loader)
    args = SimpleNamespace(vae="vae.safetensors")

    result = MageFlowNetworkTrainer().load_vae(args, torch.bfloat16, args.vae)

    assert result is not None
    assert captured == {
        "path": "vae.safetensors",
        "device": "cpu",
        "dtype": torch.bfloat16,
        "require_decoder": True,
    }
