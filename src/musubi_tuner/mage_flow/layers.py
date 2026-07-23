"""NR-MMDiT layers adapted from Microsoft Mage at commit ea7109b.

Copyright (c) 2026 Microsoft. Licensed under the MIT License.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.normalization import RMSNorm

from .attention import packed_attention


def apply_rotary_emb_mageflow(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(x_complex * freqs_cis.unsqueeze(1)).flatten(-2)
    return rotated.to(dtype=x.dtype)


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    if timesteps.ndim != 1:
        raise ValueError("timesteps must be a rank-1 tensor")
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)
    frequencies = torch.exp(exponent).to(timesteps.dtype)
    embedding = scale * timesteps[:, None].float() * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=-1)
    if flip_sin_to_cos:
        embedding = torch.cat([embedding[:, half_dim:], embedding[:, :half_dim]], dim=-1)
    if embedding_dim % 2:
        embedding = torch.nn.functional.pad(embedding, (0, 1))
    return embedding


class Timesteps(nn.Module):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )


class MageFlowTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        projected = self.time_proj(timestep)
        return self.timestep_embedder(projected.to(dtype=hidden_states.dtype))


class MageFlowEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: tuple[int, ...], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = tuple(axes_dim)
        positive = torch.arange(4096)
        negative = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [self.rope_params(positive, axis_dim, theta) for axis_dim in self.axes_dim],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [self.rope_params(negative, axis_dim, theta) for axis_dim in self.axes_dim],
            dim=1,
        )
        self.scale_rope = scale_rope
        self.video_freq_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

    @staticmethod
    def rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        if dim % 2:
            raise ValueError("each RoPE axis dimension must be even")
        frequencies = torch.outer(
            index,
            1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim)),
        )
        return torch.polar(torch.ones_like(frequencies), frequencies)

    def forward(
        self,
        image_shapes: list[list[tuple[int, int, int]]],
        device: torch.device,
    ) -> torch.Tensor:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        packed_frequencies = []
        for sample_shapes in image_shapes:
            for frame_index, (frames, height, width) in enumerate(sample_shapes):
                key = (frames, height, width, frame_index)
                if key not in self.video_freq_cache:
                    self.video_freq_cache[key] = self._compute_video_freqs(frames, height, width, frame_index).cpu()
                packed_frequencies.append(self.video_freq_cache[key].to(device))
        if not packed_frequencies:
            raise ValueError("image_shapes must contain at least one image")
        return torch.cat(packed_frequencies, dim=0)

    def _compute_video_freqs(self, frames: int, height: int, width: int, frame_index: int = 0) -> torch.Tensor:
        if max(frame_index + frames, height // 2 + height % 2, width // 2 + width % 2) > 4096:
            raise ValueError("Mage-Flow RoPE supports axis indices below 4096")
        positive = self.pos_freqs.split([axis // 2 for axis in self.axes_dim], dim=1)
        negative = self.neg_freqs.split([axis // 2 for axis in self.axes_dim], dim=1)
        frame_freqs = positive[0][frame_index : frame_index + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1)
        if self.scale_rope:
            height_freqs = torch.cat(
                [negative[1][-(height - height // 2) :], positive[1][: height // 2]],
                dim=0,
            )
            width_freqs = torch.cat(
                [negative[2][-(width - width // 2) :], positive[2][: width // 2]],
                dim=0,
            )
        else:
            height_freqs = positive[1][:height]
            width_freqs = positive[2][:width]
        height_freqs = height_freqs.view(1, height, 1, -1).expand(frames, height, width, -1)
        width_freqs = width_freqs.view(1, 1, width, -1).expand(frames, height, width, -1)
        return torch.cat([frame_freqs, height_freqs, width_freqs], dim=-1).reshape(frames * height * width, -1).clone()


class Attention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        *,
        added_kv_proj_dim: int,
        out_dim: int,
        bias: bool = True,
        eps: float = 1e-6,
        backend: str = "sdpa",
    ):
        super().__init__()
        self.inner_dim = out_dim
        self.heads = heads
        self.dim_head = dim_head
        self.backend = backend
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.to_q = nn.Linear(query_dim, out_dim, bias=bias)
        self.to_k = nn.Linear(query_dim, out_dim, bias=bias)
        self.to_v = nn.Linear(query_dim, out_dim, bias=bias)
        self.add_q_proj = nn.Linear(added_kv_proj_dim, out_dim, bias=True)
        self.add_k_proj = nn.Linear(added_kv_proj_dim, out_dim, bias=True)
        self.add_v_proj = nn.Linear(added_kv_proj_dim, out_dim, bias=True)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)
        self.to_out = nn.ModuleList([nn.Linear(out_dim, out_dim, bias=True), nn.Dropout(0.0)])
        self.to_add_out = nn.Linear(out_dim, out_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *,
        txt_cu_lens: torch.Tensor,
        img_cu_lens: torch.Tensor,
        image_rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_query = self.to_q(hidden_states).unflatten(-1, (self.heads, self.dim_head)).flatten(0, 1)
        image_key = self.to_k(hidden_states).unflatten(-1, (self.heads, self.dim_head)).flatten(0, 1)
        image_value = self.to_v(hidden_states).unflatten(-1, (self.heads, self.dim_head)).flatten(0, 1)
        text_query = self.add_q_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.dim_head)).flatten(0, 1)
        text_key = self.add_k_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.dim_head)).flatten(0, 1)
        text_value = self.add_v_proj(encoder_hidden_states).unflatten(-1, (self.heads, self.dim_head)).flatten(0, 1)

        image_query = apply_rotary_emb_mageflow(self.norm_q(image_query), image_rotary_emb)
        image_key = apply_rotary_emb_mageflow(self.norm_k(image_key), image_rotary_emb)
        text_query = self.norm_added_q(text_query)
        text_key = self.norm_added_k(text_key)

        image_lengths = img_cu_lens[1:] - img_cu_lens[:-1]
        text_lengths = txt_cu_lens[1:] - txt_cu_lens[:-1]
        joint_lengths = text_lengths + image_lengths
        joint_cu = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=joint_lengths.device),
                torch.cumsum(joint_lengths, dim=0, dtype=torch.int32),
            ]
        )
        batch_size = text_lengths.numel()
        sample_indices = torch.arange(batch_size, device=joint_lengths.device)
        text_sample_ids = torch.repeat_interleave(sample_indices, text_lengths)
        image_sample_ids = torch.repeat_interleave(sample_indices, image_lengths)
        text_positions = torch.arange(text_query.shape[0], device=joint_lengths.device) - txt_cu_lens[text_sample_ids]
        image_positions = torch.arange(image_query.shape[0], device=joint_lengths.device) - img_cu_lens[image_sample_ids]
        text_dest = joint_cu[text_sample_ids] + text_positions
        image_dest = joint_cu[image_sample_ids] + text_lengths[image_sample_ids] + image_positions

        total_tokens = int(joint_cu[-1].item())
        joint_query = torch.empty((total_tokens, self.heads, self.dim_head), dtype=image_query.dtype, device=image_query.device)
        joint_key = torch.empty_like(joint_query)
        joint_value = torch.empty_like(joint_query)
        joint_query[text_dest], joint_query[image_dest] = text_query, image_query
        joint_key[text_dest], joint_key[image_dest] = text_key, image_key
        joint_value[text_dest], joint_value[image_dest] = text_value, image_value

        attended = packed_attention(joint_query, joint_key, joint_value, joint_cu, backend=self.backend)
        image_output = attended[image_dest].flatten(1)
        text_output = attended[text_dest].flatten(1)
        image_output = self.to_out[1](self.to_out[0](image_output)).view_as(hidden_states)
        text_output = self.to_add_out(text_output).view_as(encoder_hidden_states)
        return image_output, text_output


class MageFlowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        attention_backend: str = "sdpa",
    ):
        super().__init__()
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            added_kv_proj_dim=dim,
            out_dim=dim,
            bias=True,
            eps=eps,
            backend=attention_backend,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    @staticmethod
    def _modulate(
        x: torch.Tensor,
        mod_params: torch.Tensor,
        cu_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        if cu_lens is None:
            return x * (1 + scale) + shift, gate
        if x.shape[0] != 1:
            raise ValueError("packed modulation expects a leading singleton batch dimension")
        lengths = cu_lens[1:] - cu_lens[:-1]
        shift_tokens = shift.repeat_interleave(lengths, dim=0)
        scale_tokens = scale.repeat_interleave(lengths, dim=0)
        gate_tokens = gate.repeat_interleave(lengths, dim=0)
        flattened = x.view(-1, x.shape[-1])
        return (flattened * (1 + scale_tokens) + shift_tokens).view_as(x), gate_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        txt_cu_lens: torch.Tensor,
        img_cu_lens: torch.Tensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del joint_attention_kwargs
        image_mod_1, image_mod_2 = self.img_mod(temb).chunk(2, dim=-1)
        text_mod_1, text_mod_2 = self.txt_mod(temb).chunk(2, dim=-1)
        image_input, image_gate_1 = self._modulate(self.img_norm1(hidden_states), image_mod_1, img_cu_lens)
        text_input, text_gate_1 = self._modulate(self.txt_norm1(encoder_hidden_states), text_mod_1, txt_cu_lens)
        image_attention, text_attention = self.attn(
            image_input,
            text_input,
            image_rotary_emb=image_rotary_emb,
            txt_cu_lens=txt_cu_lens,
            img_cu_lens=img_cu_lens,
        )
        hidden_states = hidden_states + image_gate_1 * image_attention
        encoder_hidden_states = encoder_hidden_states + text_gate_1 * text_attention
        image_input, image_gate_2 = self._modulate(self.img_norm2(hidden_states), image_mod_2, img_cu_lens)
        text_input, text_gate_2 = self._modulate(self.txt_norm2(encoder_hidden_states), text_mod_2, txt_cu_lens)
        hidden_states = hidden_states + image_gate_2 * self.img_mlp(image_input)
        encoder_hidden_states = encoder_hidden_states + text_gate_2 * self.txt_mlp(text_input)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clamp(-65504, 65504)
            encoder_hidden_states = encoder_hidden_states.clamp(-65504, 65504)
        return encoder_hidden_states, hidden_states


class AdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        self.norm = nn.LayerNorm(embedding_dim, eps, elementwise_affine, bias)

    def forward(
        self,
        x: torch.Tensor,
        conditioning_embedding: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embedding = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = embedding.chunk(2, dim=-1)
        if cu_seqlens is None:
            return self.norm(x) * (1 + scale) + shift
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        scale_tokens = scale.repeat_interleave(lengths, dim=0)
        shift_tokens = shift.repeat_interleave(lengths, dim=0)
        flattened = x.view(-1, x.shape[-1])
        return (self.norm(flattened) * (1 + scale_tokens) + shift_tokens).view_as(x)


__all__ = [
    "AdaLayerNormContinuous",
    "Attention",
    "MageFlowEmbedRope",
    "MageFlowTimestepProjEmbeddings",
    "MageFlowTransformerBlock",
    "Timesteps",
    "apply_rotary_emb_mageflow",
    "get_timestep_embedding",
]
