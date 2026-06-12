from typing import Tuple

import torch

from musubi_tuner.ideogram4.latent_norm import get_latent_norm
from musubi_tuner.ideogram4.pipeline import patchify_latents, unpatchify_latents


PATCH_SIZE = 2


def get_ideogram4_latent_shift_scale(
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load Ideogram4 latent normalization tensors.

    Ideogram4 does not feed raw VAE latents directly to the transformer.
    It patchifies them and applies per-channel shift/scale normalization.
    """
    shift, scale = get_latent_norm()

    shift = shift.view(1, -1, 1, 1).to(device=device, dtype=dtype)
    scale = scale.view(1, -1, 1, 1).to(device=device, dtype=dtype)

    return shift, scale


@torch.no_grad()
def encode_images_to_ideogram4_latents(
    vae,
    images: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """
    Encode image tensors into Ideogram4 transformer-ready latents.

    Input:
      images: (B, 3, H, W), expected in the range used by the VAE code,
              usually [-1, 1].

    Output:
      latents: (B, 128, H/16, W/16)

    Steps:
      1. VAE encoder -> moments
      2. take mean latent
      3. patchify 2x2: (B, 32, H/8, W/8) -> (B, 128, H/16, W/16)
      4. normalize with Ideogram shift/scale
    """
    device = torch.device(device)

    if next(vae.parameters()).device != device:
        vae.to(device)

    images = images.to(device=device, dtype=dtype)

    ae_channels = vae.params.z_channels

    moments = vae.encoder(images)
    mean = moments[:, :ae_channels]

    patched = patchify_latents(mean, PATCH_SIZE)

    shift, scale = get_ideogram4_latent_shift_scale(device=device, dtype=patched.dtype)
    latents = (patched - shift) / scale

    return latents


@torch.no_grad()
def decode_ideogram4_latents_to_images(
    vae,
    latents: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """
    Decode Ideogram4 transformer latents back into image tensors.

    Input:
      latents: (B, 128, H/16, W/16)

    Output:
      images: decoded VAE output tensor, usually in [-1, 1]-style range.

    Steps:
      1. denormalize with shift/scale
      2. unpatchify: (B, 128, H/16, W/16) -> (B, 32, H/8, W/8)
      3. VAE decoder
    """
    device = torch.device(device)

    if next(vae.parameters()).device != device:
        vae.to(device)

    latents = latents.to(device=device, dtype=dtype)

    shift, scale = get_ideogram4_latent_shift_scale(device=device, dtype=latents.dtype)
    patched = latents * scale + shift

    z = unpatchify_latents(patched, PATCH_SIZE)

    images = vae.decoder(z)

    return images


def make_flow_noisy_latents(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    Flow-matching interpolation.

    AI Toolkit / Musubi convention here:
      t = 0 -> clean
      t = 1 -> pure noise

    Formula:
      z_t = (1 - t) * clean + t * noise
    """
    while t.ndim < clean_latents.ndim:
        t = t.view(*t.shape, *([1] * (clean_latents.ndim - t.ndim)))

    return (1.0 - t) * clean_latents + t * noise


def make_flow_target(
    clean_latents: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """
    Ideogram4 / AI Toolkit training target after predict_velocity correction.

    Target:
      noise - clean
    """
    return (noise - clean_latents).detach()
