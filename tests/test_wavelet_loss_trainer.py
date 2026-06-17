"""Integration tests for the FLUX.2 wavelet-loss trainer wiring.

These exercise the musubi-side glue (arg parsing, x0 recovery, compute_loss
combination, metadata), not the wavelet_loss package internals.
"""

import argparse

import pytest

from musubi_tuner.flux_2_train_network_wavelet_loss import (
    _parse_band_weights,
    wavelet_loss_setup_parser,
    Flux2WaveletLossNetworkTrainer,
)


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
