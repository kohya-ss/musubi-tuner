"""Wavelet-loss training entry point for FLUX.2.

Augments the standard FLUX.2 flow-matching loss with a frequency-domain
auxiliary term computed by the optional ``wavelet_loss`` package.

The wavelet loss operates on *estimated clean latents* (x0) recovered from the
model's velocity prediction and the velocity target via the flow-matching
identity (``noisy = (1-sigma)*latents + sigma*noise``,
``target = noise - latents``):

    x0_pred   = noisy_model_input - sigma * pred     (approximately latents)
    x0_target = noisy_model_input - sigma * target   (exactly latents)

This gives the wavelet transform meaningful frequency structure to penalise and
makes the auxiliary loss reach zero at a perfect prediction.

Usage (extends a normal FLUX.2 training command):

    accelerate launch flux_2_train_network_wavelet_loss.py \\
        --wavelet_loss \\
        --wavelet_loss_alpha 0.1 \\
        --wavelet_loss_transform swt \\
        --wavelet_loss_level 2 \\
        <...normal FLUX.2 training args...>

The wavelet term combines additively with the base MSE:
``loss = mse.mean() + alpha * wavelet_loss``.
"""

import argparse
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from musubi_tuner.flux_2_train_network import Flux2NetworkTrainer, flux2_setup_parser
from musubi_tuner.hv_train_network import (
    DiTOutput,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.training.timesteps import compute_loss_weighting_for_sd3, get_sigmas

try:
    from wavelet_loss import WaveletLoss
except ImportError:
    WaveletLoss = None  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _parse_band_weights(weights_str: Optional[str]) -> Optional[dict[str, float]]:
    """Parse ``ll=0.1,lh=0.01,hl=0.01,hh=0.05`` or a JSON/literal dict string."""
    if weights_str is None:
        return None
    import ast
    import json as _json

    if weights_str.strip().startswith("{"):
        try:
            return ast.literal_eval(weights_str)
        except (ValueError, SyntaxError):
            return _json.loads(weights_str.replace("'", '"'))

    result = {}
    for pair in weights_str.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k.strip()] = float(v.strip())
    return result
