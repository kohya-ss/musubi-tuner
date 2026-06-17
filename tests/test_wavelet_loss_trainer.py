"""Integration tests for the FLUX.2 wavelet-loss trainer wiring.

These exercise the musubi-side glue (arg parsing, x0 recovery, compute_loss
combination, metadata), not the wavelet_loss package internals.
"""

from musubi_tuner.flux_2_train_network_wavelet_loss import _parse_band_weights


def test_parse_band_weights_key_value():
    result = _parse_band_weights("ll=0.1,lh=0.01,hl=0.02,hh=0.05")
    assert result == {"ll": 0.1, "lh": 0.01, "hl": 0.02, "hh": 0.05}


def test_parse_band_weights_json():
    result = _parse_band_weights('{"ll": 0.1, "hh": 0.05}')
    assert result == {"ll": 0.1, "hh": 0.05}


def test_parse_band_weights_none():
    assert _parse_band_weights(None) is None
