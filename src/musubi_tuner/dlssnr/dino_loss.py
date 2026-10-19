"""Optional frozen SenseCraft DINOv3 features with NR mask and reduction semantics."""

from __future__ import annotations

import hashlib
import inspect
from importlib.metadata import version

import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.fp8 import canonical_tensor_sha256


def create_dino_loss(settings):
    """Load only on opt-in, without consuming the student's random stream."""
    from musubi_tuner.training.dlssnr_services import capture_rng, restore_rng

    rng = capture_rng()
    try:
        return _load_dino_loss(settings)
    finally:
        restore_rng(rng)


def _source_sha256(cls):
    return hashlib.sha256(inspect.getsource(inspect.getmodule(cls)).encode("utf-8")).hexdigest()


def _load_dino_loss(settings):
    try:
        from sensecraft.loss import ViTDinoV3PerceptualLoss
        from sensecraft.loss.gram_dinov3 import ModelType
    except ImportError as error:
        raise ImportError('NR DINOv3 loss requires its optional dependency: uv pip install ".[dinov3]"') from error

    model_type = ModelType[settings["model_type"].upper()]
    # SenseCraft 0.3.11 re-resolves negative indices after truncating. Resolve once
    # against the full model, then pass an absolute layer to its feature extractor.
    backend = ViTDinoV3PerceptualLoss(
        model_type=model_type,
        input_range=(0, 1),
        loss_layer=-1,
        use_gram=settings["use_gram"],
        use_norm=settings["use_norm"],
    )
    depth = len(backend.model.layer)
    layer = settings["layer"]
    resolved = depth + layer if layer < 0 else layer
    if not 0 <= resolved < depth:
        raise ValueError(f"--dino_loss_layer must be in [{-depth}, {depth - 1}] for {settings['model_type']}")
    backend.model.layer = backend.model.layer[: resolved + 1]
    backend.num_layers = resolved + 1
    backend.loss_layer = resolved
    backend.model.set_attn_implementation("eager")
    provenance = {
        "provider": "sensecraft_vit_dinov3",
        "model_id": model_type.value,
        "model_config": backend.model.config.to_dict(),
        "resolved_layer": resolved,
        "attention_backend": "eager",
        "sensecraft_version": version("sensecraft"),
        "transformers_version": version("transformers"),
        "sensecraft_source_sha256": _source_sha256(type(backend)),
        "transformers_source_sha256": _source_sha256(type(backend.model)),
    }
    return NRDinoLoss(backend, settings, provenance=provenance)


class NRDinoLoss(torch.nn.Module):
    """Return per-image patch losses; the trainer owns cross-rank pixel weighting."""

    def __init__(self, backend, settings, *, provenance):
        super().__init__()
        self.backend = backend.requires_grad_(False).float()
        self.settings = dict(settings)
        self.patch_size = int(backend.model.config.patch_size)
        self.prefix_tokens = 1 + int(backend.model.config.num_register_tokens)
        self.identity = {
            "schema": "dlssnr_dinov3_patch_v1",
            "settings": self.settings,
            "provenance": provenance,
            "weights_sha256": canonical_tensor_sha256(sorted([*backend.named_parameters(), *backend.named_buffers()])),
            "preprocessing": "masked_antialiased_downscale_neutral_padding_v1",
            "reduction": "valid_rgb_mass_weighted_images",
            "input_range": list(backend.input_range),
            "patch_size": self.patch_size,
            "prefix_tokens": self.prefix_tokens,
        }
        self.eval()

    def train(self, mode=True):
        # Parent NRTrainModule.train() must never enable teacher dropout or RoPE augmentation.
        return super().train(False)

    def _prepare(self, image, mask, resized_mask, size):
        # Sanitize before resizing: excluded values must not bleed into retained pixels.
        image = torch.where(mask > 0, image.float(), 0).clamp(0, 1) * mask
        if size != image.shape[-2:]:
            image = F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=True)
        image = image / torch.where(resized_mask > 0, resized_mask, 1)
        image = torch.where(resized_mask > 0, image, 0.5).clamp(0, 1)
        padding = (0, -size[1] % self.patch_size, 0, -size[0] % self.patch_size)
        return F.pad(image, padding, value=0.5)

    def _features(self, image, patch_count):
        features = self.backend.dinov3_fwd(self.backend.normalize_input(image))
        if features.ndim != 3 or features.shape[1] != self.prefix_tokens + patch_count:
            raise ValueError("DINOv3 feature tokens do not match the image patch grid")
        features = features[:, self.prefix_tokens :]
        return F.normalize(features, dim=-1) if self.settings["use_norm"] else features

    def forward(self, prediction, target, mask):
        if prediction.ndim != 4 or prediction.shape[1] != 3 or target.shape != prediction.shape:
            raise ValueError("DINOv3 requires matching BCHW RGB images")
        if mask.shape != prediction[:, :1].shape:
            raise ValueError("DINOv3 requires a B1HW loss mask")
        with torch.autocast(prediction.device.type, enabled=False):
            mask = mask.detach().float()
            height, width = prediction.shape[-2:]
            scale = min(1.0, self.settings["resize"] / max(height, width))
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
            resized_mask = mask
            if size != (height, width):
                resized_mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False, antialias=True)
            prepared = self._prepare(prediction, mask, resized_mask, size)
            padding = (0, -size[1] % self.patch_size, 0, -size[0] % self.patch_size)
            weights = F.avg_pool2d(F.pad(resized_mask, padding), self.patch_size).flatten(1)
            pred_features = self._features(prepared, weights.shape[1])
            with torch.no_grad():
                target_features = self._features(self._prepare(target.detach(), mask, resized_mask, size), weights.shape[1])
            mass = weights.sum(dim=1)
            denominator = torch.where(mass > 0, mass, 1)
            if self.settings["use_gram"]:
                root_weights = weights.sqrt().unsqueeze(-1)
                pred_gram = self.backend.gram_matrix(pred_features * root_weights, normalize=False)
                target_gram = self.backend.gram_matrix(target_features * root_weights, normalize=False)
                return (pred_gram - target_gram).abs().mean(dim=(1, 2)) / denominator
            error = (pred_features - target_features).square().mean(dim=-1)
            return (error * weights).sum(dim=1) / denominator
