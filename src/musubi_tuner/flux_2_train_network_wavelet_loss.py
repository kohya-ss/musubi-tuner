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


class Flux2WaveletLossNetworkTrainer(Flux2NetworkTrainer):
    """FLUX.2 + wavelet-domain auxiliary loss.

    Owned state:
    - ``self.wavelet_loss``: ``WaveletLoss`` module, constructed in
      ``on_train_start`` when ``--wavelet_loss`` is set. Holds the wavelet
      filters as registered buffers so they move to the correct device with
      ``.to(device)``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.wavelet_loss: Optional["WaveletLoss"] = None  # type: ignore[type-arg]

    def handle_model_specific_args(self, args: argparse.Namespace) -> None:
        # Check the optional-dependency guard FIRST so a clear error is raised
        # before any FLUX.2-specific model-version logic runs.
        if args.wavelet_loss and WaveletLoss is None:
            raise ImportError(
                "wavelet-loss package is not installed. Install it with: pip install -e /path/to/wavelet-loss"
            )
        super().handle_model_specific_args(args)

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        """Delegate to parent and stash ``noisy_model_input`` for compute_loss.

        ``output.pred`` and ``output.target`` (both velocity space) are already
        on the returned ``DiTOutput``; only ``noisy_model_input`` needs stashing
        to recover the clean latents in ``compute_loss``.
        """
        output = super().call_dit(
            args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs
        )
        output.extra["noisy_model_input"] = noisy_model_input
        return output

    def on_train_start(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        network,
        transformer,
        optimizer,
    ) -> None:
        """Construct and move the WaveletLoss module to the training device."""
        if not args.wavelet_loss:
            return

        assert WaveletLoss is not None, "wavelet-loss package not installed"
        device = accelerator.device
        self.wavelet_loss = WaveletLoss(
            transform_type=args.wavelet_loss_transform,
            wavelet=args.wavelet_loss_wavelet,
            level=args.wavelet_loss_level,
            band_weights=args.wavelet_loss_band_weights,
            band_level_weights=args.wavelet_loss_band_level_weights,
            quaternion_component_weights=args.wavelet_loss_quaternion_component_weights,
            ll_level_threshold=args.wavelet_loss_ll_level_threshold,
            metrics=args.wavelet_loss_metrics,
            normalize_bands=args.wavelet_loss_normalize_bands,
            device=device,
        )
        self.wavelet_loss.to(device)

        logger.info("Wavelet loss enabled:")
        logger.info(f"\tTransform: {args.wavelet_loss_transform}")
        logger.info(f"\tWavelet:   {args.wavelet_loss_wavelet}")
        logger.info(f"\tLevel:     {args.wavelet_loss_level}")
        logger.info(f"\tAlpha:     {args.wavelet_loss_alpha}")
        if args.wavelet_loss_band_weights:
            logger.info(f"\tBand weights: {args.wavelet_loss_band_weights}")


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


def wavelet_loss_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Wavelet-loss-specific CLI arguments."""
    parser.add_argument("--wavelet_loss", action="store_true", help="Enable wavelet auxiliary loss. Default: False")
    parser.add_argument("--wavelet_loss_alpha", type=float, default=0.1, help="Wavelet loss weight. Default: 0.1")
    parser.add_argument(
        "--wavelet_loss_type",
        default=None,
        help="Loss function for wavelet bands: l1, l2, huber, smooth_l1. Defaults to --loss_type.",
    )
    parser.add_argument(
        "--wavelet_loss_transform",
        default="swt",
        choices=["dwt", "swt", "qwt"],
        help="Wavelet transform: dwt (discrete), swt (stationary), qwt (quaternion). Default: swt",
    )
    parser.add_argument("--wavelet_loss_wavelet", default="sym7", help="Wavelet family (e.g. sym7, db4). Default: sym7")
    parser.add_argument(
        "--wavelet_loss_level",
        type=int,
        default=1,
        help="Decomposition levels. Level 1 captures coarse structure; higher levels add detail. Default: 1",
    )
    parser.add_argument(
        "--wavelet_loss_band_weights",
        type=_parse_band_weights,
        default=None,
        help="Per-band weights as ll=0.1,lh=0.01,hl=0.01,hh=0.05 or JSON dict. Default: library defaults.",
    )
    parser.add_argument(
        "--wavelet_loss_band_level_weights",
        type=_parse_band_weights,
        default=None,
        help="Per-band-per-level weights as ll1=0.1,lh1=0.01,hh2=0.05 etc. Overrides --wavelet_loss_band_weights.",
    )
    parser.add_argument(
        "--wavelet_loss_quaternion_component_weights",
        type=_parse_band_weights,
        default=None,
        help="QWT component weights as r=1.0,i=0.7,j=0.7,k=0.5. Only used with --wavelet_loss_transform qwt.",
    )
    parser.add_argument(
        "--wavelet_loss_ll_level_threshold",
        type=int,
        default=None,
        help="Level at which to include LL (low-frequency) band. -1 = last level only. Default: None (use all).",
    )
    parser.add_argument(
        "--wavelet_loss_normalize_bands",
        action="store_true",
        default=None,
        help="Normalise each wavelet band before computing the loss.",
    )
    parser.add_argument(
        "--wavelet_loss_metrics",
        action="store_true",
        help="Log detailed per-band wavelet metrics each step (adds overhead). Default: False",
    )
    return parser
