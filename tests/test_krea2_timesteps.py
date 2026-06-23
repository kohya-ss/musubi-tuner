"""Tests for the `krea2_shift` timestep-sampling mode.

krea2_shift reproduces the K2 inference time-shift schedule during training: the
shift is resolution-aware, interpolated linearly in image-sequence length between
(256px -> mu 0.5) and (1280px -> mu 1.15), with seqlen = (res / 16)**2 (Qwen-Image
VAE f8 x patch 2). The endpoints x1=256, x2=6400 distinguish it from flux_shift
(x2=4096), so the high end saturates at 1280px instead of 1024px.

See krea2_sampling.timesteps (inference) and trainer_base.compute_sampling_timesteps.
"""

import argparse
import math
from types import SimpleNamespace

import torch

from musubi_tuner.training import parser_common
from musubi_tuner.training.trainer_base import NetworkTrainer
from musubi_tuner.utils import train_utils


def _expected_shifted_t(latent_h, latent_w, q=0.5):
    """Replicate the krea2_shift transform for a uniform sample `q`.

    With q=0.5 the logit-normal draw is 0 -> sigmoid(0)=0.5, so the result is
    shift / (1 + shift) regardless of sigmoid_scale, giving a deterministic check.
    """
    mu = train_utils.get_lin_function(x1=256, y1=0.5, x2=6400, y2=1.15)((latent_h // 2) * (latent_w // 2))
    shift = math.exp(mu)
    z = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * q - 1.0)).item()  # 0 at q=0.5
    t = 1.0 / (1.0 + math.exp(-z))
    return (t * shift) / (1 + (shift - 1) * t)


def test_parser_accepts_krea2_shift_timestep_sampling():
    parser = argparse.ArgumentParser()
    parser_common._add_timestep_args(parser)

    args = parser.parse_args(["--timestep_sampling", "krea2_shift"])

    assert args.timestep_sampling == "krea2_shift"


def test_krea2_shift_matches_inference_endpoints_at_1024():
    trainer = NetworkTrainer()
    args = SimpleNamespace(
        timestep_sampling="krea2_shift",
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
        sigmoid_scale=1.0,
    )
    # 1024x1024 image -> 128x128 latent (f8); seqlen (128//2)**2 = 4096 -> mu 0.90625.
    latents = torch.zeros(2, 16, 1, 128, 128, dtype=torch.float32)
    noise = torch.ones_like(latents)

    noisy_model_input, sampled_timesteps = trainer.get_noisy_model_input_and_timesteps(
        args, noise, latents, [0.5, 0.5], None, torch.device("cpu"), torch.float32
    )

    # noise=1, latents=0 -> noisy_model_input == t (the shifted timestep in [0, 1]).
    expected_t = _expected_shifted_t(128, 128)
    assert math.isclose(expected_t, 2.474871 / (1 + 2.474871), rel_tol=1e-4)  # ~0.7122, anchors the endpoints
    assert torch.allclose(noisy_model_input, torch.full_like(noisy_model_input, expected_t), atol=1e-5)
    assert torch.allclose(sampled_timesteps, torch.full_like(sampled_timesteps, expected_t * 1000.0 + 1.0), atol=1e-2)


def test_krea2_shift_is_resolution_aware():
    trainer = NetworkTrainer()
    args = SimpleNamespace(
        timestep_sampling="krea2_shift",
        min_timestep=None,
        max_timestep=None,
        preserve_distribution_shape=False,
        sigmoid_scale=1.0,
    )

    def shifted_t_for(latent_side):
        latents = torch.zeros(1, 16, 1, latent_side, latent_side, dtype=torch.float32)
        noise = torch.ones_like(latents)
        out, _ = trainer.get_noisy_model_input_and_timesteps(
            args, noise, latents, [0.5], None, torch.device("cpu"), torch.float32
        )
        return out.flatten()[0].item()

    t_256 = shifted_t_for(32)  # 256px image
    t_1024 = shifted_t_for(128)  # 1024px image
    t_1280 = shifted_t_for(160)  # 1280px image (the maxres endpoint)

    # Larger resolution -> stronger shift -> larger t (noisier midpoint), monotonic.
    assert t_256 < t_1024 < t_1280
    assert math.isclose(t_256, _expected_shifted_t(32, 32), rel_tol=1e-5)
    assert math.isclose(t_1280, _expected_shifted_t(160, 160), rel_tol=1e-5)
