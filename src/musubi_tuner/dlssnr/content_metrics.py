"""Optional, evaluation-only DINOv3 content drift relative to the input."""

from copy import deepcopy
import math

import torch

from musubi_tuner.dlssnr.dino_loss import NRDinoLoss, create_dino_loss
from musubi_tuner.dlssnr.numerics import fp32_execution
from musubi_tuner.training.dlssnr_services import evaluation_mode

CONTENT_SETTINGS = {"model_type": "small", "layer": -4, "resize": 224, "use_gram": False, "use_norm": True}


def create_content_metric(dino_loss=None):
    if dino_loss is not None and all(dino_loss.settings[name] == CONTENT_SETTINGS[name] for name in ("model_type", "layer")):
        # The extractor does not use Gram/normalization/resize settings. Keep a
        # separate wrapper so evaluation cannot change the training objective.
        feature_loss = NRDinoLoss(dino_loss.backend, CONTENT_SETTINGS, provenance=deepcopy(dino_loss.identity["provenance"]))
    else:
        feature_loss = create_dino_loss(CONTENT_SETTINGS)
    return NRContentMetric(feature_loss)


def _check_mask(mask):
    if mask.ndim != 4 or mask.shape[1] != 1 or not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
        raise ValueError("content metric mask must be finite B1HW in [0,1]")


class NRContentMetric(torch.nn.Module):
    def __init__(self, feature_loss):
        super().__init__()
        if feature_loss.settings["use_gram"] is not False or feature_loss.settings["use_norm"] is not True:
            raise ValueError("content preservation requires normalized spatial patch MSE, not Gram statistics")
        self.feature_loss = feature_loss.requires_grad_(False).float()
        self.identity = {
            "schema": "dlssnr_content_preservation_v1",
            "metric": "normalized_spatial_patch_mse",
            "reference": "input_proxy_rgb",
            "feature_model": deepcopy(feature_loss.identity),
            "frame_weight": "original_valid_rgb_mass",
            "precision": "float32_environment_tf32_float64_frame_reduction",
        }
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, output, source, mask):
        _check_mask(mask)
        if output.ndim != 4 or output.shape[1] != 3 or source.shape != output.shape or mask.shape != output[:, :1].shape:
            raise ValueError("content metric requires matching BCHW RGB and B1HW mask")
        if any(not torch.isfinite(torch.where(mask > 0, image, 0)).all() for image in (output, source)):
            raise ValueError("content metric images must be finite on valid pixels")
        with evaluation_mode(self.feature_loss), fp32_execution():
            values = self.feature_loss(output, source, mask)
            if values.shape != (output.shape[0],) or not torch.isfinite(values).all() or (values < 0).any():
                raise RuntimeError("non-finite or invalid content preservation scores")
            return values.detach()


class ContentMetrics:
    """Accumulate frame scores by valid RGB mass, including fractional masks."""

    def __init__(self):
        self.weighted_sum = 0.0
        self.count = 0.0

    @torch.no_grad()
    def add(self, scores, mask):
        _check_mask(mask)
        if scores.shape != (mask.shape[0],) or not torch.isfinite(scores).all() or (scores < 0).any():
            raise ValueError("content scores must be finite nonnegative per-image values")
        mass = 3 * mask.double().sum((1, 2, 3))
        self.weighted_sum += float((scores.detach().double() * mass).sum())
        self.count += float(mass.sum())

    def metrics(self):
        if not math.isfinite(self.weighted_sum) or not math.isfinite(self.count):
            raise RuntimeError("non-finite content preservation summary")
        if self.count <= 0:
            raise ValueError("content preservation has no valid pixels")
        return {"dinov3_patch_mse": self.weighted_sum / self.count, "valid_rgb_values": self.count}
