"""Flux2 KL autoencoder.

Ideogram 4 uses the same KL autoencoder as FLUX.2, so the building blocks
(``AutoEncoderParams``/``Encoder``/``Decoder``/...) are imported directly from
``flux_2.flux2_models`` to avoid duplication. Only the top-level ``AutoEncoder``
wrapper differs: Ideogram 4 packs the patchified latent as ``(pi pj c)`` (vs
FLUX.2's ``(c pi pj)``) and does NOT apply the checkpoint's BatchNorm in
encode/decode (latent (de)normalization is done with the standalone constants in
latent_norm.py). ``convert_diffusers_state_dict`` is also Ideogram 4 specific.
"""

from __future__ import annotations

import math
import re

import torch
from einops import rearrange
from torch import Tensor, nn

from musubi_tuner.flux_2.flux2_models import AutoEncoderParams, Encoder, Decoder

__all__ = ["AutoEncoderParams", "AutoEncoder", "convert_diffusers_state_dict"]


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.ps = [2, 2]
        # NOTE: `bn` holds BatchNorm statistics shipped in the checkpoint, but it is NOT used by the
        # caching/training/inference pipeline. Latent (de)normalization is done with the standalone
        # LATENT_SHIFT/LATENT_SCALE constants in latent_norm.py, which are the authoritative values
        # (verified by correct generation) and do NOT match bn's running stats. `bn` is retained only
        # so the checkpoint's bn.* keys load via the default strict state_dict load.
        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]
        return rearrange(
            mean,
            "... c (i pi) (j pj) -> ... (pi pj c) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )

    def decode(self, z: Tensor) -> Tensor:
        z = rearrange(
            z,
            "... (pi pj c) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        return self.decoder(z)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


_NUM_RESOLUTIONS = 4


def convert_diffusers_state_dict(src: dict[str, Tensor]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    attn_substrings = (".mid.attn_1.",)
    for src_key, tensor in src.items():
        dst_key = _rewrite_diffusers_key(src_key)
        if dst_key is None:
            raise KeyError(f"Unrecognized diffusers VAE state-dict key: {src_key}")
        if any(s in dst_key for s in attn_substrings) and dst_key.endswith(".weight") and tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).unsqueeze(-1)
        out[dst_key] = tensor
    return out


def _rewrite_diffusers_key(key: str) -> str | None:
    if key.startswith("bn."):
        return key

    if key.startswith("quant_conv."):
        return key.replace("quant_conv.", "encoder.quant_conv.", 1)
    if key.startswith("post_quant_conv."):
        return key.replace("post_quant_conv.", "decoder.post_quant_conv.", 1)

    if key == "encoder.conv_norm_out.weight":
        return "encoder.norm_out.weight"
    if key == "encoder.conv_norm_out.bias":
        return "encoder.norm_out.bias"
    if key == "decoder.conv_norm_out.weight":
        return "decoder.norm_out.weight"
    if key == "decoder.conv_norm_out.bias":
        return "decoder.norm_out.bias"

    m = re.match(r"^(encoder|decoder)\.mid_block\.resnets\.(\d+)\.(.+)$", key)
    if m:
        side, idx, rest = m.group(1), int(m.group(2)), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"{side}.mid.block_{idx + 1}.{rest}"
    m = re.match(r"^(encoder|decoder)\.mid_block\.attentions\.0\.(.+)$", key)
    if m:
        side, rest = m.group(1), m.group(2)
        rest = (
            rest.replace("group_norm.", "norm.")
            .replace("to_q.", "q.")
            .replace("to_k.", "k.")
            .replace("to_v.", "v.")
            .replace("to_out.0.", "proj_out.")
        )
        return f"{side}.mid.attn_1.{rest}"

    m = re.match(r"^encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        level, res_idx, rest = m.group(1), m.group(2), m.group(3)
        rest = rest.replace("conv_shortcut", "nin_shortcut")
        return f"encoder.down.{level}.block.{res_idx}.{rest}"
    m = re.match(r"^encoder\.down_blocks\.(\d+)\.downsamplers\.0\.conv\.(.+)$", key)
    if m:
        return f"encoder.down.{m.group(1)}.downsample.conv.{m.group(2)}"

    m = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        res_idx = m.group(2)
        rest = m.group(3).replace("conv_shortcut", "nin_shortcut")
        return f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.block.{res_idx}.{rest}"
    m = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.conv\.(.+)$", key)
    if m:
        diffusers_idx = int(m.group(1))
        return f"decoder.up.{_NUM_RESOLUTIONS - 1 - diffusers_idx}.upsample.conv.{m.group(2)}"

    if key.startswith(("encoder.conv_in.", "encoder.conv_out.", "decoder.conv_in.", "decoder.conv_out.")):
        return key

    return None
