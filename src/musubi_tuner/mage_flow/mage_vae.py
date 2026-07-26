"""MageVAE inference modules adapted from Microsoft Mage commit ea7109b.

Copyright (c) 2026 Microsoft. Licensed under the MIT License.
"""

from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .utils import ComponentValidationError


def posterior_seed(architecture: str, item_key: str, role: str, seed: int) -> int:
    identity = f"{architecture}\0{item_key}\0{role}".encode("utf-8")
    digest_value = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big", signed=False)
    return (digest_value + int(seed)) % (2**63 - 1)


def sample_posterior(mean: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if mean.shape != logvar.shape:
        raise ValueError(f"posterior mean/logvar shapes must match, got {mean.shape} and {logvar.shape}")
    epsilon = torch.randn(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
    return mean + torch.exp(0.5 * logvar.clamp(min=-20.0, max=10.0)) * epsilon


def nonlinearity(x):
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


def modulate(x, shift, scale):
    if x.dim() == 4:
        batch, channels = x.shape[:2]
        return x * (1 + scale.view(batch, channels, 1, 1)) + shift.view(batch, channels, 1, 1)
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class _EncoderLayerNorm2d(LayerNorm2d):
    pass


class _VaeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        dtype = x.dtype
        value = x.float()
        value = value * torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * value.to(dtype)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t[:, None].float() * frequencies[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        embedding = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class BottleneckPatchEmbed(nn.Module):
    def __init__(self, patch_size=16, in_chans=3, pca_dim=128, embed_dim=384, bias=True):
        super().__init__()
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim + embed_dim, embed_dim, kernel_size=1, bias=bias)

    def forward(self, x, cond):
        return self.proj2(torch.cat([self.proj1(x), cond], dim=1))


class DiCoBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_size, hidden_size, 1), nn.Sigmoid())
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1)
        self.norm1 = LayerNorm2d(hidden_size, affine=False)
        self.norm2 = LayerNorm2d(hidden_size, affine=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, inp, conditioning):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(conditioning).chunk(6, dim=1)
        value = modulate(self.norm1(inp), shift_msa, scale_msa)
        value = F.gelu(self.conv2(self.conv1(value)))
        value = self.conv3(value * self.ca(value))
        value = inp + gate_msa[..., None, None] * value
        return value + gate_mlp[..., None, None] * self.conv5(F.gelu(self.conv4(modulate(self.norm2(value), shift_mlp, scale_mlp))))


class _EncoderDiCoBlock(nn.Module):
    def __init__(self, hidden_size, mlp_ratio=4.0):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, padding=1, groups=hidden_size)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_size, hidden_size, 1), nn.Sigmoid())
        ffn = int(mlp_ratio * hidden_size)
        self.conv4 = nn.Conv2d(hidden_size, ffn, 1)
        self.conv5 = nn.Conv2d(ffn, hidden_size, 1)
        self.norm1 = _EncoderLayerNorm2d(hidden_size)
        self.norm2 = _EncoderLayerNorm2d(hidden_size)

    def forward(self, inp):
        value = self.norm1(inp)
        value = F.gelu(self.conv2(self.conv1(value)))
        value = self.conv3(value * self.ca(value))
        value = inp + value
        return value + self.conv5(F.gelu(self.conv4(self.norm2(value))))


class NerfEmbedder(nn.Module):
    def __init__(self, in_channels, hidden_size_input, max_freqs=8):
        super().__init__()
        self.max_freqs = max_freqs
        self.embedder = nn.Sequential(nn.Linear(in_channels + max_freqs**2, hidden_size_input))

    @lru_cache
    def fetch_pos(self, patch_size, device, dtype):
        position = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        position_y, position_x = torch.meshgrid(position, position, indexing="ij")
        position_x = position_x.reshape(-1, 1, 1)
        position_y = position_y.reshape(-1, 1, 1)
        frequencies = torch.linspace(0, self.max_freqs, self.max_freqs, dtype=dtype, device=device)
        frequency_x = frequencies[None, :, None]
        frequency_y = frequencies[None, None, :]
        coefficients = (1 + frequency_x * frequency_y) ** -1
        dct_x = torch.cos(position_x * frequency_x * torch.pi)
        dct_y = torch.cos(position_y * frequency_y * torch.pi)
        return (dct_x * dct_y * coefficients).view(1, -1, self.max_freqs**2)

    def forward(self, x):
        batch, patch_area, _ = x.shape
        patch_size = int(patch_area**0.5)
        dct = self.fetch_pos(patch_size, x.device, x.dtype).expand(batch, -1, -1)
        return self.embedder(torch.cat([x, dct], dim=-1))


class NerfFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = _VaeRMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        return self.linear(self.norm(x))


class _MLPResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(channels, channels), nn.SiLU(), nn.Linear(channels, channels))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels))

    def forward(self, x, conditioning):
        shift, scale, gate = self.adaLN_modulation(conditioning).chunk(3, dim=-1)
        hidden = self.in_ln(x) * (1 + scale) + shift
        return x + gate * self.mlp(hidden)


class SimpleMLPAdaLN(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, patch_size):
        super().__init__()
        del out_channels
        self.patch_size = patch_size
        self.cond_embed = nn.Linear(z_channels, patch_size**2 * model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        self.res_blocks = nn.ModuleList(_MLPResBlock(model_channels) for _ in range(num_res_blocks))

    def forward(self, x, conditioning):
        x = self.input_proj(x)
        conditioning = self.cond_embed(conditioning).reshape(conditioning.shape[0], self.patch_size**2, -1)
        for block in self.res_blocks:
            x = block(x, conditioning)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, dropout=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        hidden = self.conv1(nonlinearity(self.norm1(x)))
        hidden = self.conv2(self.dropout(nonlinearity(self.norm2(hidden))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden


class AttnBlock(nn.Module):
    def __init__(self, in_channels, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.norm = Normalize(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        normalized = self.norm(x)
        query, key, value = self.q(normalized), self.k(normalized), self.v(normalized)
        patch_size = self.patch_size
        batch, channels, height, width = query.shape
        pad_height = (patch_size - height % patch_size) % patch_size
        pad_width = (patch_size - width % patch_size) % patch_size
        if pad_height or pad_width:
            query = F.pad(query, (0, pad_width, 0, pad_height), mode="replicate")
            key = F.pad(key, (0, pad_width, 0, pad_height), mode="replicate")
            value = F.pad(value, (0, pad_width, 0, pad_height), mode="replicate")
        padded_height, padded_width = query.shape[-2:]
        patch_rows, patch_columns = padded_height // patch_size, padded_width // patch_size
        patch_count = patch_rows * patch_columns

        def to_patches(tensor):
            return (
                tensor.reshape(batch, channels, patch_rows, patch_size, patch_columns, patch_size)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch * patch_count, channels, patch_size * patch_size)
            )

        query, key, value = to_patches(query), to_patches(key), to_patches(value)
        weights = torch.bmm(query.permute(0, 2, 1), key) * channels**-0.5
        weights = F.softmax(weights, dim=2).permute(0, 2, 1)
        hidden = (
            torch.bmm(value, weights)
            .reshape(batch, patch_rows, patch_columns, channels, patch_size, patch_size)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, padded_height, padded_width)
        )
        if pad_height or pad_width:
            hidden = hidden[:, :, :height, :width]
        return x + self.proj_out(hidden)


class _Decoder(nn.Module):
    def __init__(self, out_ch=384, z_ch=128):
        super().__init__()
        self.conv_in = nn.Conv2d(z_ch, out_ch, 3, padding=1)
        self.block = nn.Sequential(
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
            AttnBlock(out_ch, patch_size=32),
            ResnetBlock(in_channels=out_ch, out_channels=out_ch),
        )
        self.norm_out = Normalize(out_ch)
        self.conv_out = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.ada = nn.Identity()

    def forward(self, z):
        hidden = self.block(self.conv_in(z))
        return self.ada(self.conv_out(nonlinearity(self.norm_out(hidden))))


class _DConvEncoder(nn.Module):
    def __init__(
        self,
        z_ch=128,
        hidden_size=384,
        num_blocks=21,
        patch_size=16,
        mlp_ratio=4.0,
        head_size=768,
        num_head_blocks=2,
        out_ch_mult=2,
    ):
        super().__init__()
        self.z_ch = z_ch
        self.patch_size = patch_size
        self.patch_cond_embed = nn.Conv2d(3, head_size, kernel_size=patch_size, stride=patch_size)
        self.head_blocks = nn.ModuleList([_EncoderDiCoBlock(head_size, mlp_ratio=mlp_ratio) for _ in range(num_head_blocks)])
        self.proj_down = nn.Conv2d(head_size, hidden_size, 1)
        self.z_proj = nn.Conv2d(z_ch, hidden_size, 1)
        self.fuse_proj = nn.Conv2d(hidden_size * 2, hidden_size, 1)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.blocks = nn.ModuleList([DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_blocks)])
        self.norm_out = LayerNorm2d(hidden_size)
        self.proj_out = nn.Conv2d(hidden_size, z_ch * out_ch_mult, 1)

    def forward_pred(self, z_t, timestep, image):
        conditioning = self.patch_cond_embed(image)
        for block in self.head_blocks:
            conditioning = block(conditioning)
        conditioning = self.proj_down(conditioning)
        hidden = self.fuse_proj(torch.cat([conditioning, self.z_proj(z_t)], dim=1))
        timestep_embedding = self.t_embedder(timestep.view(-1))
        for block in self.blocks:
            hidden = block(hidden, timestep_embedding)
        return self.proj_out(self.norm_out(hidden))


class _YEmbedder(nn.Module):
    def __init__(self, ch=384, z_ch=128):
        super().__init__()
        self.decoder = _Decoder(out_ch=ch, z_ch=z_ch)


class _DConvDenoiser(nn.Module):
    def __init__(
        self,
        patch_size=16,
        in_channels=3,
        hidden_size=384,
        hidden_size_x=32,
        mlp_ratio=4.0,
        num_blocks=24,
        num_cond_blocks=21,
        bottleneck_dim=128,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder_x = nn.Conv2d(hidden_size, hidden_size_x * patch_size**2, 1)
        self.x_embedder = NerfEmbedder(in_channels + hidden_size_x, hidden_size_x, max_freqs=8)
        self.s_embedder = BottleneckPatchEmbed(patch_size, in_channels, bottleneck_dim, hidden_size, bias=True)
        self.blocks = nn.ModuleList([DiCoBlock(hidden_size, mlp_ratio=mlp_ratio) for _ in range(num_cond_blocks)])
        self.dec_net = SimpleMLPAdaLN(
            in_channels=hidden_size_x,
            model_channels=hidden_size_x,
            out_channels=in_channels,
            z_channels=hidden_size,
            num_res_blocks=num_blocks - num_cond_blocks,
            patch_size=patch_size,
        )
        self.final_layer = NerfFinalLayer(hidden_size_x, in_channels)
        self.y_embedder = _YEmbedder(ch=hidden_size, z_ch=bottleneck_dim)

    def forward(self, x, timestep, conditioning):
        batch, _, height, width = x.shape
        timestep_embedding = self.t_embedder(timestep.view(-1))
        hidden = self.s_embedder(x, conditioning)
        for block in self.blocks:
            hidden = block(hidden, timestep_embedding)
        length = hidden.shape[-2] * hidden.shape[-1]
        hidden = hidden.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        image = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        image = torch.cat([image, self.y_embedder_x(conditioning).flatten(2)], dim=1)
        image = image.reshape(batch, -1, self.patch_size**2, length).permute(0, 3, 2, 1).flatten(0, 1)
        image = self.x_embedder(image)
        image = self.final_layer(self.dec_net(image, hidden))
        image = image.transpose(1, 2).reshape(batch, length, -1)
        return F.fold(
            image.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )


def _module_shapes(module_type) -> dict[str, tuple[int, ...]]:
    with torch.device("meta"):
        module = module_type()
    return {key: tuple(value.shape) for key, value in module.state_dict().items()}


def _inspect_vae(path: str | Path, require_decoder: bool):
    component_path = Path(path)
    if not component_path.exists():
        raise ComponentValidationError(f"vae component file does not exist: {component_path}")
    if not component_path.is_file():
        raise ComponentValidationError(f"vae component path must be one regular file: {component_path}")
    if component_path.suffix.lower() != ".safetensors":
        raise ComponentValidationError(f"vae component must be one .safetensors file: {component_path}")

    with safe_open(component_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        official = any(key.startswith("student.dconv_encoder.") for key in keys)
        canonical = any(key.startswith("dconv_encoder.") for key in keys)
        if official == canonical:
            raise ComponentValidationError("MageVAE checkpoint must use one exact official or canonical key layout")
        encoder_prefix = "student.dconv_encoder." if official else "dconv_encoder."
        decoder_prefix = "pipeline." if official else "decoder_model."
        encoder_shapes = {
            key[len(encoder_prefix) :]: tuple(handle.get_slice(key).get_shape()) for key in keys if key.startswith(encoder_prefix)
        }
        decoder_shapes = {
            key[len(decoder_prefix) :]: tuple(handle.get_slice(key).get_shape())
            for key in keys
            if key.startswith(decoder_prefix)
            and not key[len(decoder_prefix) :].startswith(("y_embedder.encoder.", "y_embedder.bottleneck."))
        }
        dtypes = {key: handle.get_slice(key).get_dtype() for key in keys}

    expected_anchor = (256, 384, 1, 1)
    if encoder_shapes.get("proj_out.weight") != expected_anchor:
        raise ComponentValidationError(
            "MageVAE encoder expected packed mean+logvar proj_out.weight shape "
            f"{expected_anchor}, got {encoder_shapes.get('proj_out.weight')}"
        )
    expected_encoder = _module_shapes(_DConvEncoder)
    expected_decoder = _module_shapes(_DConvDenoiser) if require_decoder else {}
    missing_encoder = sorted(set(expected_encoder) - set(encoder_shapes))
    unexpected_encoder = sorted(set(encoder_shapes) - set(expected_encoder))
    mismatched_encoder = sorted(
        key for key in set(expected_encoder) & set(encoder_shapes) if expected_encoder[key] != encoder_shapes[key]
    )
    missing_decoder = sorted(set(expected_decoder) - set(decoder_shapes))
    unexpected_decoder = sorted(set(decoder_shapes) - set(expected_decoder)) if require_decoder else []
    mismatched_decoder = sorted(
        key for key in set(expected_decoder) & set(decoder_shapes) if expected_decoder[key] != decoder_shapes[key]
    )
    allowed_dtypes = {"BF16", "F16", "F32"}
    bad_dtypes = [f"{key}:{dtype}" for key, dtype in dtypes.items() if dtype not in allowed_dtypes]
    recognized_keys = {encoder_prefix + key for key in expected_encoder}
    if require_decoder:
        recognized_keys.update(decoder_prefix + key for key in expected_decoder)
    else:
        recognized_keys.update(key for key in keys if key.startswith(decoder_prefix))
    if official:
        recognized_keys.update(
            key for key in keys if key.startswith(("pipeline.y_embedder.encoder.", "pipeline.y_embedder.bottleneck."))
        )
    unknown_keys = sorted(set(keys) - recognized_keys)
    if (
        missing_encoder
        or unexpected_encoder
        or mismatched_encoder
        or missing_decoder
        or unexpected_decoder
        or mismatched_decoder
        or bad_dtypes
        or unknown_keys
    ):
        raise ComponentValidationError(
            "MageVAE structural mismatch; "
            f"missing encoder={missing_encoder[:10]}; unexpected encoder={unexpected_encoder[:10]}; "
            f"shape encoder={mismatched_encoder[:10]}; missing decoder={missing_decoder[:10]}; "
            f"unexpected decoder={unexpected_decoder[:10]}; shape decoder={mismatched_decoder[:10]}; "
            f"dtype={bad_dtypes[:10]}; unknown={unknown_keys[:10]}"
        )
    return component_path, encoder_prefix, decoder_prefix, expected_encoder, expected_decoder


def _load_prefixed_tensors(path: Path, prefix: str, expected: dict[str, tuple[int, ...]]) -> dict[str, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(prefix + key) for key in expected}


class MageVAE(nn.Module):
    latent_channels = 128
    downsample_factor = 16

    def __init__(self, encoder: _DConvEncoder, decoder: _DConvDenoiser | None):
        super().__init__()
        self.dconv_encoder = encoder
        self.decoder_model = decoder

    @torch.no_grad()
    def encode_moments(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]
        if height % self.downsample_factor or width % self.downsample_factor:
            raise ValueError(f"H, W must be multiples of 16, got ({height}, {width})")
        batch = x.shape[0]
        z_t = torch.zeros(
            batch,
            self.latent_channels,
            height // self.downsample_factor,
            width // self.downsample_factor,
            device=x.device,
            dtype=x.dtype,
        )
        timestep = torch.zeros(batch, device=x.device, dtype=x.dtype)
        output = self.dconv_encoder.forward_pred(z_t, timestep, x)
        return output[:, : self.latent_channels], output[:, self.latent_channels :].clamp(-20.0, 10.0)

    @torch.no_grad()
    def encode(self, x: torch.Tensor, generators: list[torch.Generator] | None = None) -> torch.Tensor:
        mean, logvar = self.encode_moments(x)
        if generators is None:
            return mean
        if len(generators) != mean.shape[0]:
            raise ValueError("one posterior generator is required per image")
        return torch.cat(
            [
                sample_posterior(mean[index : index + 1], logvar[index : index + 1], generators[index])
                for index in range(len(generators))
            ]
        )

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if self.decoder_model is None:
            raise RuntimeError("MageVAE was loaded encoder-only; decoding is unavailable")
        conditioning = self.decoder_model.y_embedder.decoder(z)
        batch = z.shape[0]
        height = z.shape[2] * self.downsample_factor
        width = z.shape[3] * self.downsample_factor
        noise = torch.zeros(batch, 3, height, width, device=z.device, dtype=z.dtype)
        timestep = torch.zeros(batch, device=z.device, dtype=z.dtype)
        return self.decoder_model(noise, timestep, conditioning)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


@torch.no_grad()
def decode_mage_vae_latents(vae: MageVAE, latents: torch.Tensor) -> torch.Tensor:
    device = vae.device
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        return vae.decode(latents.to(device=device, dtype=torch.float32))


def load_mage_vae(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    require_decoder: bool = True,
) -> MageVAE:
    component_path, encoder_prefix, decoder_prefix, encoder_shapes, decoder_shapes = _inspect_vae(path, require_decoder)
    encoder_state = _load_prefixed_tensors(component_path, encoder_prefix, encoder_shapes)
    with torch.device("meta"):
        encoder = _DConvEncoder()
    encoder.load_state_dict(encoder_state, strict=True, assign=True)
    decoder = None
    if require_decoder:
        decoder_state = _load_prefixed_tensors(component_path, decoder_prefix, decoder_shapes)
        with torch.device("meta"):
            decoder = _DConvDenoiser()
        decoder.load_state_dict(decoder_state, strict=True, assign=True)
    model = MageVAE(encoder, decoder).to(device=device, dtype=dtype)
    model.requires_grad_(False)
    model.eval()
    return model


__all__ = ["MageVAE", "decode_mage_vae_latents", "load_mage_vae", "posterior_seed", "sample_posterior"]
