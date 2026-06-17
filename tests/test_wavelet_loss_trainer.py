"""Integration tests for the FLUX.2 wavelet-loss trainer wiring.

These exercise the musubi-side glue (arg parsing, x0 recovery, compute_loss
combination, metadata), not the wavelet_loss package internals.
"""

import argparse

import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.flux_2_train_network_wavelet_loss import (
    _parse_band_weights,
    wavelet_loss_setup_parser,
    Flux2WaveletLossNetworkTrainer,
)
from musubi_tuner.training.trainer_base import DiTOutput


def test_parse_band_weights_key_value():
    result = _parse_band_weights("ll=0.1,lh=0.01,hl=0.02,hh=0.05")
    assert result == {"ll": 0.1, "lh": 0.01, "hl": 0.02, "hh": 0.05}


def test_parse_band_weights_json():
    result = _parse_band_weights('{"ll": 0.1, "hh": 0.05}')
    assert result == {"ll": 0.1, "hh": 0.05}


def test_parse_band_weights_none():
    assert _parse_band_weights(None) is None


def _wavelet_only_parser() -> argparse.ArgumentParser:
    # Standalone parser with only the wavelet args, to test them in isolation.
    return wavelet_loss_setup_parser(argparse.ArgumentParser())


def test_parser_defaults():
    parser = _wavelet_only_parser()
    args = parser.parse_args([])
    assert args.wavelet_loss is False
    assert args.wavelet_loss_alpha == 0.1
    assert args.wavelet_loss_transform == "swt"
    assert args.wavelet_loss_wavelet == "sym7"
    assert args.wavelet_loss_level == 1
    assert args.wavelet_loss_type is None
    assert args.wavelet_loss_band_weights is None


def test_parser_band_weights_parsed():
    parser = _wavelet_only_parser()
    args = parser.parse_args(["--wavelet_loss", "--wavelet_loss_band_weights", "ll=0.1,hh=0.05"])
    assert args.wavelet_loss is True
    assert args.wavelet_loss_band_weights == {"ll": 0.1, "hh": 0.05}


def test_parser_does_not_define_dropped_args():
    parser = _wavelet_only_parser()
    args = parser.parse_args([])
    for dropped in (
        "wavelet_loss_primary",
        "wavelet_loss_timestep_intensity",
        "wavelet_loss_use_snr_aware_huber",
        "wavelet_loss_min_snr_beta",
    ):
        assert not hasattr(args, dropped), f"dropped arg leaked: {dropped}"


def test_handle_model_specific_args_requires_package(monkeypatch):
    import musubi_tuner.flux_2_train_network_wavelet_loss as mod

    monkeypatch.setattr(mod, "WaveletLoss", None)
    trainer = Flux2WaveletLossNetworkTrainer()
    args = argparse.Namespace(
        wavelet_loss=True,
        model_version="flux2-dev",  # any valid key; only reached if import guard passes
    )
    # The guard must fire before any model-version logic.
    with pytest.raises(ImportError):
        trainer.handle_model_specific_args(args)


pytest.importorskip("wavelet_loss")


class _FakeScheduler:
    """Minimal stand-in for get_sigmas: needs .sigmas and .timesteps."""

    def __init__(self, timesteps: torch.Tensor, sigmas: torch.Tensor):
        self.timesteps = timesteps
        self.sigmas = sigmas


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        wavelet_loss=True,
        wavelet_loss_alpha=0.1,
        wavelet_loss_type=None,
        wavelet_loss_transform="swt",
        wavelet_loss_wavelet="sym7",
        wavelet_loss_level=1,
        wavelet_loss_band_weights=None,
        wavelet_loss_band_level_weights=None,
        wavelet_loss_quaternion_component_weights=None,
        wavelet_loss_ll_level_threshold=None,
        wavelet_loss_normalize_bands=None,
        wavelet_loss_metrics=False,
        weighting_scheme="none",
        loss_type="l2",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _setup_trainer_and_batch():
    from musubi_tuner.flux_2_train_network_wavelet_loss import WaveletLoss as _WL

    torch.manual_seed(0)
    b, c, h, w = 1, 4, 16, 16
    latents = torch.randn(b, c, h, w)
    noise = torch.randn(b, c, h, w)

    # schedule with a single timestep present so get_sigmas resolves cleanly
    timesteps = torch.tensor([2])
    schedule_ts = torch.tensor([0, 1, 2, 3])
    schedule_sigmas = torch.tensor([0.0, 0.25, 0.5, 0.75])
    scheduler = _FakeScheduler(schedule_ts, schedule_sigmas)
    sigma = 0.5  # sigmas[index of timestep 2]

    noisy = (1.0 - sigma) * latents + sigma * noise
    target = noise - latents  # velocity target (matches Flux2 call_dit)

    trainer = Flux2WaveletLossNetworkTrainer()
    args = _make_args()
    trainer.wavelet_loss = _WL(
        transform_type="swt", wavelet="sym7", level=1, ll_level_threshold=None, device=torch.device("cpu")
    )
    return trainer, args, latents, noise, noisy, target, timesteps, scheduler, sigma


def test_x0_target_recovers_latents():
    _, _, latents, noise, noisy, target, _, _, sigma = _setup_trainer_and_batch()
    x0_target = noisy - sigma * target
    assert torch.allclose(x0_target, latents, atol=1e-5)
    # perfect prediction (pred == target) recovers latents too
    x0_pred = noisy - sigma * target
    assert torch.allclose(x0_pred, latents, atol=1e-5)


def test_compute_loss_with_wavelet_returns_metrics():
    trainer, args, latents, noise, noisy, target, timesteps, scheduler, _ = _setup_trainer_and_batch()
    pred = target + 0.1 * torch.randn_like(target)  # imperfect prediction
    output = DiTOutput(pred=pred, target=target, extra={"noisy_model_input": noisy})

    loss, metrics = trainer.compute_loss(
        args, output, timesteps, scheduler, torch.float32, torch.float32, global_step=0
    )
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert len(metrics) > 0
    assert all(k.startswith("wavelet_loss/") for k in metrics)


def test_compute_loss_disabled_equals_weighted_mse():
    trainer, args, latents, noise, noisy, target, timesteps, scheduler, _ = _setup_trainer_and_batch()
    args.wavelet_loss = False
    trainer.wavelet_loss = None
    pred = target + 0.1 * torch.randn_like(target)
    output = DiTOutput(pred=pred, target=target, extra={"noisy_model_input": noisy})

    loss, metrics = trainer.compute_loss(
        args, output, timesteps, scheduler, torch.float32, torch.float32, global_step=0
    )
    expected = F.mse_loss(pred, target).detach()
    assert metrics == {}
    assert torch.allclose(loss, expected, atol=1e-6)
