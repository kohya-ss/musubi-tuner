import argparse
from types import SimpleNamespace

import torch

from musubi_tuner.training import parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer


def test_parser_accepts_uniform_shift_timestep_sampling():
    parser = argparse.ArgumentParser()
    parser_common._add_timestep_args(parser)

    args = parser.parse_args(["--timestep_sampling", "uniform_shift"])

    assert args.timestep_sampling == "uniform_shift"


def test_uniform_shift_applies_static_flow_shift_to_uniform_samples():
    trainer = NetworkTrainer()
    args = SimpleNamespace(
        timestep_sampling="uniform_shift",
        discrete_flow_shift=6.0,
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
        sigmoid_scale=1.0,
    )
    latents = torch.zeros(4, 1, 1, 1)
    noise = torch.ones_like(latents)

    noisy_model_input, sampled_timesteps = trainer.get_noisy_model_input_and_timesteps(
        args,
        noise,
        latents,
        [0.0, 0.05, 0.5, 1.0],
        None,
        torch.device("cpu"),
        torch.float32,
    )

    expected = torch.tensor([0.0, 0.24, 6.0 / 7.0, 1.0])
    torch.testing.assert_close(noisy_model_input[:, 0, 0, 0], expected)
    torch.testing.assert_close(sampled_timesteps, expected * 1000.0 + 1.0)


def test_uniform_remains_unshifted_when_discrete_flow_shift_is_set():
    trainer = NetworkTrainer()
    args = SimpleNamespace(
        timestep_sampling="uniform",
        discrete_flow_shift=6.0,
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
        sigmoid_scale=1.0,
    )
    latents = torch.zeros(1, 1, 1, 1)
    noise = torch.ones_like(latents)

    noisy_model_input, _ = trainer.get_noisy_model_input_and_timesteps(
        args,
        noise,
        latents,
        [0.5],
        None,
        torch.device("cpu"),
        torch.float32,
    )

    torch.testing.assert_close(noisy_model_input, torch.full_like(noisy_model_input, 0.5))
