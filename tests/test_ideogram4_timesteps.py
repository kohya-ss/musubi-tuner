import argparse
from types import SimpleNamespace

import torch

from musubi_tuner.training import parser_common, timesteps
from musubi_tuner.training.trainer_base import NetworkTrainer


def test_ideogram4_shift_matches_resolution_aware_logit_normal():
    u = torch.tensor([0.5, 0.5], dtype=torch.float32)

    t_512 = timesteps.compute_ideogram4_shift_timestep(u, token_grid_height=32, token_grid_width=32)
    t_1024 = timesteps.compute_ideogram4_shift_timestep(u, token_grid_height=64, token_grid_width=64)

    assert torch.allclose(t_512, torch.full_like(t_512, 0.5))
    assert torch.allclose(t_1024, torch.full_like(t_1024, 2.0 / 3.0))


def test_ideogram4_shift_uses_official_std_in_logit_space():
    u = torch.tensor([0.8413447460685429], dtype=torch.float32)

    actual = timesteps.compute_ideogram4_shift_timestep(u, token_grid_height=32, token_grid_width=32)

    assert torch.allclose(actual, torch.tensor([torch.sigmoid(torch.tensor(1.5)).item()]))


def test_parser_accepts_ideogram4_shift_timestep_sampling():
    parser = argparse.ArgumentParser()
    parser_common._add_timestep_args(parser)

    args = parser.parse_args(["--timestep_sampling", "ideogram4_shift"])

    assert args.timestep_sampling == "ideogram4_shift"


def test_trainer_ideogram4_shift_samples_from_token_grid_resolution():
    trainer = NetworkTrainer()
    args = SimpleNamespace(
        timestep_sampling="ideogram4_shift",
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
    )
    latents = torch.zeros(2, 128, 64, 64, dtype=torch.float32)
    noise = torch.ones_like(latents)

    noisy_model_input, sampled_timesteps = trainer.get_noisy_model_input_and_timesteps(
        args,
        noise,
        latents,
        [0.5, 0.5],
        None,
        torch.device("cpu"),
        torch.float32,
    )

    assert torch.allclose(noisy_model_input, torch.full_like(noisy_model_input, 2.0 / 3.0))
    assert torch.allclose(sampled_timesteps, torch.full_like(sampled_timesteps, 2000.0 / 3.0 + 1.0))
