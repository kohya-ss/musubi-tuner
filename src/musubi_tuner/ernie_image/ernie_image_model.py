# Copyright 2025 Baidu ERNIE-Image Team and The HuggingFace Team. All rights reserved.
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
# This file has been modified from the original version.
# Original implementation: HuggingFace Diffusers (ErnieImageTransformer2DModel)
# Modifications: Copied and modified for Musubi Tuner project.
#   - Removed Diffusers dependencies (ConfigMixin, ModelMixin, etc.)
#   - Replaced attention with musubi_tuner.modules.attention for flash/xformers/sageattn support
#   - Converted from sequence-first [S, B, H] to batch-first [B, S, H] layout

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.attention import AttentionParams, attention
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader

logger = logging.getLogger(__name__)


# --- Embeddings ---


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool = False, downscale_freq_shift: float = 0.0):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.num_channels // 2
        exponent = -math.log(10000) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        exponent = exponent / (half_dim - self.downscale_freq_shift)
        emb = timesteps.float().unsqueeze(-1) * exponent.exp().unsqueeze(0)
        if self.flip_sin_to_cos:
            emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)
        else:
            emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.num_channels % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(out_channels, out_channels)

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.linear_2(self.act(self.linear_1(sample)))


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float32, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    return out.float()


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: Tuple[int, int, int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(3)], dim=-1)
        return emb  # [B, S, head_dim//2]


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    # x_in: [B, S, heads, head_dim], freqs_cis: [B, S, head_dim//2]
    freqs_cis = freqs_cis.unsqueeze(2)  # [B, S, 1, head_dim//2]
    # Duplicate frequencies for pairs: [θ0,θ0,θ1,θ1,...]
    freqs_cis = torch.stack([freqs_cis, freqs_cis], dim=-1).reshape(*freqs_cis.shape[:-1], -1)  # [B, S, 1, head_dim]
    rot_dim = freqs_cis.shape[-1]
    x, x_pass = x_in[..., :rot_dim], x_in[..., rot_dim:]
    cos_ = torch.cos(freqs_cis).to(x.dtype)
    sin_ = torch.sin(freqs_cis).to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated = torch.cat((-x2, x1), dim=-1)
    return torch.cat((x * cos_ + x_rotated * sin_, x_pass), dim=-1)


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, H, W]
        batch_size, dim, height, width = x.shape
        return x.reshape(batch_size, dim, height * width).transpose(1, 2).contiguous()  # [B, H*W, D]


# --- Attention ---


class ErnieImageAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        bias: bool = False,
        qk_norm: bool = True,
        eps: float = 1e-5,
        out_bias: bool = True,
    ):
        super().__init__()
        self.n_heads = heads
        self.head_dim = dim_head
        inner_dim = dim_head * heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, query_dim, bias=out_bias)])

        self.norm_q = nn.RMSNorm(dim_head, eps=eps) if qk_norm else None
        self.norm_k = nn.RMSNorm(dim_head, eps=eps) if qk_norm else None

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))  # [B, S, heads, head_dim]
        key = key.unflatten(-1, (self.n_heads, -1))
        value = value.unflatten(-1, (self.n_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        dtype = value.dtype
        query, key = query.to(dtype), key.to(dtype)

        qkv = [query, key, value]
        del query, key, value
        hidden_states = attention(qkv, attn_params=attn_params)
        del qkv

        hidden_states = hidden_states.to(dtype)
        return self.to_out[0](hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, hidden_states, freqs_cis, attn_params, use_reentrant=False)
        else:
            return self._forward(hidden_states, freqs_cis, attn_params)


# --- Feed-forward ---


class ErnieImageFeedForward(nn.Module):
    def __init__(self, hidden_size: int, ffn_hidden_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.linear_fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.up_proj(x) * F.gelu(self.gate_proj(x)))


# --- Transformer Block ---


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        ffn_hidden_size: int,
        eps: float = 1e-6,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.adaLN_sa_ln = RMSNorm(hidden_size, eps=eps)
        self.self_attention = ErnieImageAttention(
            query_dim=hidden_size,
            dim_head=hidden_size // num_heads,
            heads=num_heads,
            qk_norm=qk_layernorm,
            eps=eps,
            bias=False,
            out_bias=False,
        )
        self.adaLN_mlp_ln = RMSNorm(hidden_size, eps=eps)
        self.mlp = ErnieImageFeedForward(hidden_size, ffn_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor],
        temb: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        attn_params: Optional[AttentionParams] = None,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        residual = x
        x = self.adaLN_sa_ln(x)
        x = (x.float() * (1 + scale_msa.float()) + shift_msa.float()).to(x.dtype)
        attn_out = self.self_attention(x, freqs_cis=freqs_cis, attn_params=attn_params)
        x = residual + (gate_msa.float() * attn_out.float()).to(x.dtype)

        residual = x
        x = self.adaLN_mlp_ln(x)
        x = (x.float() * (1 + scale_mlp.float()) + shift_mlp.float()).to(x.dtype)
        x = residual + (gate_mlp.float() * self.mlp(x).float()).to(x.dtype)
        return x


# --- Final Layer ---


class ErnieImageAdaLNContinuous(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=eps)
        self.linear = nn.Linear(hidden_size, hidden_size * 2)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(conditioning).chunk(2, dim=-1)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x


# --- Main Model ---


ERNIE_IMAGE_ARCHITECTURE = "ernie-image"


class ErnieImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        num_attention_heads: int = 32,
        num_layers: int = 36,
        ffn_hidden_size: int = 12288,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        text_in_dim: int = 3072,
        rope_theta: int = 256,
        rope_axes_dim: Tuple[int, int, int] = (32, 48, 48),
        eps: float = 1e-6,
        qk_layernorm: bool = True,
        attn_mode: str = "torch",
        split_attn: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_in_dim = text_in_dim
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.x_embedder = ErnieImagePatchEmbedDynamic(in_channels, hidden_size, patch_size)
        self.text_proj = nn.Linear(text_in_dim, hidden_size, bias=False) if text_in_dim != hidden_size else None
        self.time_proj = Timesteps(hidden_size, flip_sin_to_cos=False, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(hidden_size, hidden_size)
        self.pos_embed = ErnieImageEmbedND3(dim=self.head_dim, theta=rope_theta, axes_dim=rope_axes_dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

        self.layers = nn.ModuleList(
            [
                ErnieImageSharedAdaLNBlock(hidden_size, num_attention_heads, ffn_hidden_size, eps, qk_layernorm=qk_layernorm)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = ErnieImageAdaLNContinuous(hidden_size, eps)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        nn.init.zeros_(self.final_linear.weight)
        nn.init.zeros_(self.final_linear.bias)

        self.gradient_checkpointing = False

        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = num_layers

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        # TODO: implement cpu_offload support
        # Only wrap at the block level; avoid nested checkpointing inside self_attention.
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks

        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} blocks. Requested {self.blocks_to_swap} blocks."
        )

        self.offloader = ModelOffloader(
            "layer", self.layers, len(self.layers), self.blocks_to_swap, supports_backward, device, use_pinned_memory
        )
        print(
            f"ERNIE-Image: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks to device {device}. "
            f"Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("ERNIE-Image: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("ERNIE-Image: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_layers = self.layers
            self.layers = nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.layers = save_layers

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_bth: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> torch.Tensor:
        device, dtype = hidden_states.device, hidden_states.dtype
        B, C, H, W = hidden_states.shape
        p = self.patch_size
        Hp, Wp = H // p, W // p
        N_img = Hp * Wp

        # Patch embed image
        img_tokens = self.x_embedder(hidden_states)  # [B, N_img, hidden_size]

        # Project text
        if self.text_proj is not None and text_bth.numel() > 0:
            text_bth = self.text_proj(text_bth)
        Tmax = text_bth.shape[1]

        # Concatenate image + text tokens
        x = torch.cat([img_tokens, text_bth], dim=1)  # [B, N_img + Tmax, hidden_size]

        # Position IDs: [B, N_img + Tmax, 3] for (seq_idx, y, x)
        # Text IDs
        text_ids = (
            torch.cat(
                [
                    torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
                    torch.zeros((B, Tmax, 2), device=device),
                ],
                dim=-1,
            )
            if Tmax > 0
            else torch.zeros((B, 0, 3), device=device)
        )
        # Image IDs
        grid_yx = torch.stack(
            torch.meshgrid(
                torch.arange(Hp, device=device, dtype=torch.float32),
                torch.arange(Wp, device=device, dtype=torch.float32),
                indexing="ij",
            ),
            dim=-1,
        ).reshape(-1, 2)
        image_ids = torch.cat(
            [text_lens.float().view(B, 1, 1).expand(-1, N_img, -1), grid_yx.view(1, N_img, 2).expand(B, -1, -1)],
            dim=-1,
        )

        # RoPE
        all_ids = torch.cat([image_ids, text_ids], dim=1)  # [B, N_img + Tmax, 3]
        freqs_cis = self.pos_embed(all_ids)  # [B, N_img + Tmax, head_dim//2]

        # Attention mask for text padding
        valid_text = (
            torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
            if Tmax > 0
            else torch.zeros((B, 0), device=device, dtype=torch.bool)
        )
        attn_params = AttentionParams.create_attention_params_from_mask(
            self.attn_mode, self.split_attn, N_img, valid_text
        )

        # Timestep conditioning
        sample = self.time_proj(timestep)
        sample = sample.to(dtype=dtype)
        c = self.time_embedding(sample)

        # AdaLN modulation: 6 vectors broadcast to all tokens
        mod = self.adaLN_modulation(c)  # [B, 6 * hidden_size]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        # Expand to [B, 1, hidden_size] for broadcasting over sequence dim
        temb = (
            shift_msa.unsqueeze(1),
            scale_msa.unsqueeze(1),
            gate_msa.unsqueeze(1),
            shift_mlp.unsqueeze(1),
            scale_mlp.unsqueeze(1),
            gate_mlp.unsqueeze(1),
        )

        # Transformer layers
        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = checkpoint(layer, x, freqs_cis, temb, attn_params, use_reentrant=False)
            else:
                x = layer(x, freqs_cis, temb, attn_params)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, index)

        # Final norm + projection (only image tokens)
        x = self.final_norm(x, c).type_as(x)
        patches = self.final_linear(x[:, :N_img])  # [B, N_img, p*p*out_channels]

        # Unpatchify
        output = (
            patches.view(B, Hp, Wp, p, p, self.out_channels)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, self.out_channels, H, W)
        )
        return output
