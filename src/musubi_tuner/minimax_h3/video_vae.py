# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for Musubi from Hugging Face Diffusers PR #14355 at commit
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc. Musubi keeps the published
# checkpoint names and adds cache-specific posterior sampling wrappers.
# ComfyUI is used only as an independent numerical reference.

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VIDEO_VAE_ENCODE_DTYPE = torch.float32
VIDEO_VAE_DECODE_DTYPE = torch.float16

LATENTS_MEAN = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
]

LATENTS_STD = [
    1.2223774194717407,
    1.2767263650894165,
    1.68317747116088865,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.96531379222869875,
    1.05698859691619875,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049925,
    0.7197399735450745,
    0.69362932443618775,
    2.961095094680786,
    2.7694199085235595,
    3.0496184825897215,
    2.1088054180145265,
    3.276226282119751,
    3.1627357006073,
    2.28168129920959475,
    2.6127843856811525,
]


class CausalConv3d(nn.Conv3d):
    """Spatially reflected, temporally causal 3D convolution."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.causal_padding = (padding,) * 3 if isinstance(padding, int) else tuple(padding)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        temporal, height, width = self.causal_padding
        if height or width:
            hidden_states = F.pad(hidden_states, (width, width, height, height, 0, 0), mode="reflect")
        if temporal:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, temporal * 2, 0), mode="constant")
        return F.conv3d(
            hidden_states,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class TemporalIsolatedGroupNorm(nn.GroupNorm):
    """Compute group-normalization statistics independently for each frame."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 5:
            return super().forward(hidden_states)
        batch_size, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * frames, channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, frames, channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


def group_norm_3d(num_channels: int) -> TemporalIsolatedGroupNorm:
    return TemporalIsolatedGroupNorm(32, num_channels, eps=1e-6, affine=True)


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_stride=1, space_stride=2):
        super().__init__()
        self.space_stride = space_stride
        self.conv = CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(time_stride, space_stride, space_stride),
            padding=(1, 0, 0),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.space_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode="reflect")
        return self.conv(hidden_states)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = group_norm_3d(in_channels)
        self.conv1 = CausalConv3d(in_channels, self.out_channels, kernel_size=3, padding=1)
        self.norm2 = group_norm_3d(self.out_channels)
        self.conv2 = CausalConv3d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.nin_shortcut = (
            CausalConv3d(in_channels, self.out_channels, kernel_size=1) if in_channels != self.out_channels else None
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(F.silu(self.norm1(hidden_states)))
        hidden_states = self.conv2(F.silu(self.norm2(hidden_states)))
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class EncoderFCN3D(nn.Module):
    def __init__(self, ch, ch_mult, space_down, time_down, num_res_blocks, in_channels, z_channels, double_z=True):
        super().__init__()
        num_levels = len(ch_mult)
        layers_per_level = [num_res_blocks] * num_levels if isinstance(num_res_blocks, int) else list(num_res_blocks)
        block_channels = [ch * multiplier for multiplier in ch_mult]
        input_channels = [block_channels[0], *block_channels[:-1]]

        self.num_levels = num_levels
        self.num_res_blocks = layers_per_level
        self.conv_in = CausalConv3d(in_channels, input_channels[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList()
        for level in range(num_levels):
            down = nn.Module()
            down.block = nn.ModuleList(
                [
                    ResnetBlock3D(
                        input_channels[level] if layer == 0 else block_channels[level],
                        block_channels[level],
                    )
                    for layer in range(layers_per_level[level])
                ]
            )
            if space_down[level] * time_down[level] > 1:
                down.downsample = Downsample3D(
                    block_channels[level],
                    block_channels[level],
                    time_stride=time_down[level],
                    space_stride=space_down[level],
                )
            self.down.append(down)

        self.norm_out = group_norm_3d(block_channels[-1])
        output_channels = 2 * z_channels if double_z else z_channels
        self.conv_out = CausalConv3d(block_channels[-1], output_channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down in self.down:
            for block in down.block:
                hidden_states = block(hidden_states)
            if hasattr(down, "downsample"):
                hidden_states = down.downsample(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


def create_token_ids(patch_dims, device, dtype=torch.float32):
    grids = [2.0 * (torch.arange(0.5, size, dtype=dtype, device=device) / size) - 1.0 for size in patch_dims]
    return torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, len(patch_dims) - 1).unsqueeze(0)


class RotaryEmbeddingND(nn.Module):
    def __init__(self, dim, rotary_base=100.0, n_dim=3):
        super().__init__()
        if dim % (2 * n_dim):
            raise ValueError(f"Rotary dimension {dim} must be divisible by {2 * n_dim}")
        inv_freq = 1.0 / rotary_base ** torch.arange(0, 1, 2 * n_dim / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None].float() * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


def _apply_rotary_emb(hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cosine, sine = (value.to(hidden_states.dtype) for value in rotary_emb)
    rotary_dim = cosine.shape[-1]
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return torch.cat((rotary * cosine + rotated * sine, passthrough), dim=-1)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, bias=True):
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.w1(hidden_states).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * value)


class Attention(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = heads * dim_head
        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(inner_dim, inner_dim * 3, bias=bias)
        self.to_out = nn.Linear(inner_dim, inner_dim, bias=bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        qkv = self.to_qkv(hidden_states).view(batch_size, sequence_length, self.heads, 3 * self.dim_head)
        query, key, value = qkv.chunk(3, dim=-1)
        query = self.norm_q(query.float()).to(query.dtype)
        key = self.norm_k(key.float()).to(key.dtype)
        if rotary_emb is not None:
            query = _apply_rotary_emb(query, rotary_emb)
            key = _apply_rotary_emb(key, rotary_emb)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, sequence_length, -1)
        return self.to_out(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(self, heads, dim_head, bias=True, eps=1e-5):
        super().__init__()
        dim = heads * dim_head
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.attn = Attention(heads=heads, dim_head=dim_head, bias=bias, eps=eps)
        self.scale1 = nn.Parameter(torch.empty(dim))
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True, eps=eps)
        self.ff = FeedForward(dim=dim, bias=bias)
        self.scale2 = nn.Parameter(torch.empty(dim))

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        normalized = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.norm1.weight.float(),
            self.norm1.eps,
        ).to(hidden_states.dtype)
        hidden_states = hidden_states + self.attn(normalized, rotary_emb) * self.scale1
        normalized = F.rms_norm(
            hidden_states.float(),
            (hidden_states.shape[-1],),
            self.norm2.weight.float(),
            self.norm2.eps,
        ).to(hidden_states.dtype)
        return hidden_states + self.ff(normalized) * self.scale2


class ViT3DDecoder(nn.Module):
    def __init__(
        self,
        patch_size=16,
        patch_size_t=4,
        in_channels=24,
        out_channels=3,
        num_layers=36,
        heads=32,
        dim_head=64,
        rope_theta=100.0,
        rope_dim_ratio=0.75,
        bias=True,
        eps=1e-5,
        num_register_tokens=4,
    ):
        super().__init__()
        dim = heads * dim_head
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.pos_embed = RotaryEmbeddingND(int(dim_head * rope_dim_ratio), rope_theta, n_dim=3)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.empty(1, num_register_tokens, dim))
        self.register_buffer("mask_token", torch.empty(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(heads=heads, dim_head=dim_head, bias=bias, eps=eps) for _ in range(num_layers)]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, frames * height * width, channels)
        hidden_states = self.x_embedder(hidden_states)
        num_patches = hidden_states.shape[1]
        hidden_states = torch.cat(
            (
                hidden_states,
                self.register_tokens.expand(batch_size, -1, -1),
                torch.zeros_like(hidden_states[:, :1]),
            ),
            dim=1,
        )

        position_ids = create_token_ids((frames, height, width), hidden_states.device)
        position_ids = position_ids.expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        rotary_emb = self.pos_embed(torch.cat((position_ids, suffix_ids), dim=1))
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.proj_out(self.norm_out(hidden_states))[:, :num_patches]
        hidden_states = hidden_states.view(
            batch_size,
            frames,
            height,
            width,
            self.out_channels,
            self.patch_size_t,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            frames * self.patch_size_t,
            height * self.patch_size,
            width * self.patch_size,
        )


class MiniMaxH3VideoVAE(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_ch=3,
        ch=128,
        embed_dim=24,
        z_channels=24,
        ch_mult=(1, 2, 2, 4, 4, 8),
        num_res_blocks=2,
        space_down=(2, 2, 2, 2, 1, 1),
        time_down=(1, 2, 2, 1, 1, 1),
        clip_length=17,
        token_drop=3,
        tile_size=256,
        tile_overlap_min=64,
        tiling=True,
    ):
        super().__init__()
        self.vae_ratio = math.prod(space_down)
        self.vae_ratio_t = math.prod(time_down)
        self.clip_length = clip_length
        self.token_drop = token_drop
        self.frame_pre_padding = (-clip_length) % self.vae_ratio_t
        self.tokens_chunk_size = math.ceil(clip_length / self.vae_ratio_t)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.vae_ratio_t - self.frame_pre_padding, 0)
        self.use_tiling = tiling
        self.tile_size = tile_size
        self.tile_overlap_min = tile_overlap_min

        self.encoder = EncoderFCN3D(
            ch=ch,
            ch_mult=list(ch_mult),
            space_down=list(space_down),
            time_down=list(time_down),
            num_res_blocks=num_res_blocks,
            in_channels=in_channels,
            z_channels=z_channels,
            double_z=True,
        )
        self.quant_conv = nn.Conv3d(2 * z_channels, 2 * embed_dim, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(embed_dim, z_channels, kernel_size=1)
        self.decoder = ViT3DDecoder(
            patch_size=self.vae_ratio,
            patch_size_t=self.vae_ratio_t,
            in_channels=z_channels,
            out_channels=out_ch,
        )
        self.register_buffer("latents_mean", torch.tensor(LATENTS_MEAN))
        self.register_buffer("latents_std", torch.tensor(LATENTS_STD))
        self.register_buffer("pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)

    def _split_tiles(self, length: int) -> tuple[list[int], list[int], list[int]]:
        if self.tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / self.tile_size)
        while self.tile_size * num_tiles - self.tile_overlap_min * (num_tiles - 1) < length:
            num_tiles += 1
        overlaps = [self.tile_overlap_min] * (num_tiles - 1)
        remaining = self.tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.vae_ratio):
            overlaps[index % (num_tiles - 1)] += self.vae_ratio
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + self.tile_size - overlaps[index])
        return starts, [self.tile_size] * num_tiles, overlaps

    @staticmethod
    def _blend(first: torch.Tensor, second: torch.Tensor, extent: int, dim: int) -> torch.Tensor:
        extent = min(first.shape[dim], second.shape[dim], extent)
        positions = torch.arange(extent, device=second.device, dtype=second.dtype)
        shape = [1] * first.ndim
        shape[dim] = extent
        first_weight = (1 - positions / extent).view(shape)
        second_weight = (positions / extent).view(shape)
        first_slice = [slice(None)] * first.ndim
        first_slice[dim] = slice(-extent, None)
        second_slice = [slice(None)] * second.ndim
        second_slice[dim] = slice(0, extent)
        blended = first[tuple(first_slice)] * first_weight + second[tuple(second_slice)] * second_weight
        if extent == second.shape[dim]:
            return blended
        rest = [slice(None)] * second.ndim
        rest[dim] = slice(extent, None)
        return torch.cat((blended, second[tuple(rest)]), dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        stitched_rows = []
        for row_index, row in enumerate(tiles):
            stitched_row = []
            for column_index, tile in enumerate(row):
                if row_index:
                    tile = self._blend(tiles[row_index - 1][column_index], tile, height_overlaps[row_index - 1], -2)
                if column_index:
                    tile = self._blend(row[column_index - 1], tile, width_overlaps[column_index - 1], -1)
                if row_index < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                stitched_row.append(tile)
            stitched_rows.append(torch.cat(stitched_row, dim=-1))
        return torch.cat(stitched_rows, dim=-2)

    def _encode_clip(self, pixels: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(pixels))
        height_starts, height_lengths, height_overlaps = self._split_tiles(pixels.shape[-2])
        width_starts, width_lengths, width_overlaps = self._split_tiles(pixels.shape[-1])
        tiles = [
            [
                self.quant_conv(
                    self.encoder(
                        pixels[
                            ...,
                            top : top + tile_height,
                            left : left + tile_width,
                        ]
                    )
                )
                for left, tile_width in zip(width_starts, width_lengths)
            ]
            for top, tile_height in zip(height_starts, height_lengths)
        ]
        latent_height_overlaps = [value // self.vae_ratio for value in height_overlaps]
        latent_width_overlaps = [value // self.vae_ratio for value in width_overlaps]
        return self._stitch_tiles(tiles, latent_height_overlaps, latent_width_overlaps)

    def _decode_clip(self, latents: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(latents))
        height_starts, height_lengths, height_overlaps = self._split_tiles(latents.shape[-2] * self.vae_ratio)
        width_starts, width_lengths, width_overlaps = self._split_tiles(latents.shape[-1] * self.vae_ratio)
        tiles = [
            [
                self.decoder(
                    self.post_quant_conv(
                        latents[
                            ...,
                            top // self.vae_ratio : (top + tile_height) // self.vae_ratio,
                            left // self.vae_ratio : (left + tile_width) // self.vae_ratio,
                        ]
                    )
                )
                for left, tile_width in zip(width_starts, width_lengths)
            ]
            for top, tile_height in zip(height_starts, height_lengths)
        ]
        return self._stitch_tiles(tiles, height_overlaps, width_overlaps)

    def _encode_video(self, pixels: torch.Tensor) -> torch.Tensor:
        frame_count = pixels.shape[2]
        if frame_count % self.clip_length:
            padding = pixels[:, :, -1:].repeat(1, 1, (-frame_count) % self.clip_length, 1, 1)
            pixels = torch.cat((pixels, padding), dim=2)
        moments = torch.cat(
            [
                self._encode_clip(pixels[:, :, start : start + self.clip_length])
                for start in range(0, pixels.shape[2], self.clip_length)
            ],
            dim=2,
        )
        return moments[:, :, : -self.token_drop] if self.token_drop else moments

    def _decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        chunk_tokens = self.tokens_chunk_size
        chunk_frames = chunk_tokens * self.vae_ratio_t
        padded_token_count = latents.shape[2] + self.token_drop
        pad_tokens = (-padded_token_count) % chunk_tokens
        num_chunks = (padded_token_count + pad_tokens) // chunk_tokens - int(self.token_drop > 0)
        if num_chunks < 1:
            pad_tokens += chunk_tokens
            num_chunks = 1
        if pad_tokens:
            latents = torch.cat((latents, latents[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)), dim=2)

        decoded = []
        overlap = None
        for index in range(num_chunks):
            start = index * chunk_tokens
            clip = self._decode_clip(latents[:, :, start : start + chunk_tokens + self.token_overlap])
            for part_index in range(int(self.token_drop > 0) + 1):
                frame_start = part_index * chunk_frames
                part = clip[:, :, frame_start : frame_start + chunk_frames]
                part = part[:, :, self.frame_pre_padding :]
                if part_index == 0:
                    if overlap is not None:
                        part = self._blend(overlap, part, self.frame_overlap, dim=-3)
                    decoded.append(part)
                else:
                    overlap = part
        if overlap is not None:
            decoded.append(overlap)
        pixels = torch.cat(decoded, dim=2)

        if pad_tokens:
            intra_tail = self.clip_length % self.vae_ratio_t
            tokens_before_padding = latents.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (tokens_before_padding + offset) % chunk_tokens == 0 else self.vae_ratio_t
                for offset in range(pad_tokens)
            )
            pixels = pixels[:, :, :-pad_frames]
        return pixels

    def encode_moments(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = pixels.unsqueeze(2)
        pixels = (pixels + 1.0) * 0.5
        pixels = (pixels - self.pixel_mean.to(pixels)) / self.pixel_std.to(pixels)
        if pixels.shape[2] == 1:
            moments = self._encode_clip(pixels)[:, :, -1:]
        else:
            moments = self._encode_video(pixels)
        return moments.float()

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        mean = self.encode_moments(pixels).chunk(2, dim=1)[0]
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(mean)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(mean)
        return (mean - latent_mean) / latent_std

    def encode_tiled(self, pixels: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.encode(pixels)

    def decode_tiled(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return self.decode(latents)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latent_mean = self.latents_mean.view(1, -1, 1, 1, 1).to(latents)
        latent_std = self.latents_std.view(1, -1, 1, 1, 1).to(latents)
        latents = latents * latent_std + latent_mean
        if latents.shape[2] == 1:
            pixels = self._decode_clip(latents)[:, :, -1:]
        else:
            pixels = self._decode_video(latents)
        pixels = pixels.float() * self.pixel_std.to(pixels) + self.pixel_mean.to(pixels)
        return pixels.clamp(0.0, 1.0) * 2.0 - 1.0


def _video_posterior_sample(vae, pixels: torch.Tensor, generator: torch.Generator, fp16_roundtrip: bool) -> torch.Tensor:
    moments = vae.encode_moments(pixels)
    if moments.ndim != 5 or moments.shape[1] != 48:
        raise ValueError(f"Expected MiniMax-H3 video moments [B,48,T,H,W], got {tuple(moments.shape)}")
    mean, logvar = moments.float().chunk(2, dim=1)
    noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32, device="cpu").to(mean.device)
    sample = mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * noise
    if fp16_roundtrip:
        sample = sample.to(torch.float16).to(torch.float32)
    latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(sample)
    latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(sample)
    return (sample - latents_mean) / latents_std


@torch.no_grad()
def encode_video_target(vae, pixels: torch.Tensor, cache_seed: int, canonical_item_key: str) -> torch.Tensor:
    digest = hashlib.sha256(f"{cache_seed}\0{canonical_item_key}".encode()).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**63)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return _video_posterior_sample(vae, pixels, generator, fp16_roundtrip=False)


@torch.no_grad()
def encode_video_condition(vae, pixels: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(42)
    return _video_posterior_sample(vae, pixels, generator, fp16_roundtrip=True)


def load_video_vae(
    path,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = VIDEO_VAE_DECODE_DTYPE,
    disable_mmap: bool = False,
) -> MiniMaxH3VideoVAE:
    from accelerate import init_empty_weights

    from musubi_tuner.minimax_h3.checkpoint import strip_key_prefixes
    from musubi_tuner.utils.safetensors_utils import load_safetensors

    with init_empty_weights():
        vae = MiniMaxH3VideoVAE()
    # loading straight to the target device avoids a resident full-model CPU copy
    device = torch.device(device)
    sd = load_safetensors(str(path), device=device, disable_mmap=True, disable_numpy_memmap=disable_mmap)
    sd = strip_key_prefixes(sd, ("first_stage_model.", "video_vae.", "vae."))
    for key in sd.keys():
        if sd[key].is_floating_point():
            sd[key] = sd[key].to(dtype)
    vae.load_state_dict(sd, strict=True, assign=True)
    vae.to(device)
    vae.eval()
    return vae
