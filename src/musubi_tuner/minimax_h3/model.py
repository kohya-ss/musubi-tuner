# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
# (models/transformers/transformer_minimax_h3.py).
# ComfyUI is used only as an independent numerical reference.

from __future__ import annotations

import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from musubi_tuner.minimax_h3.checkpoint import (
    load_safetensors_metadata,
    load_safetensors_module,
    resolve_safetensors_files,
)
from musubi_tuner.minimax_h3.packing import (
    AUDIO_CHANNELS,
    VIDEO_CHANNELS,
    VIDEO_PATCH_SIZE,
    H3PackedLayout,
    build_position_grid,
    build_timestep_rows,
    pack_audio_rows,
    pack_video_rows,
    unpack_targets,
)
from musubi_tuner.modules.attention import AttentionParams, attention
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

_ROTARY_CACHE_SIZE = 2


@dataclass(frozen=True)
class MiniMaxH3Config:
    in_channels: int = 24
    audio_in_channels: int = 32
    hidden_size: int = 5376
    num_layers: int = 50
    token_refiner_num_layers: int = 2
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120
    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688
    rope_inv_freq_len: int = 16
    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5
    adaln_curve_grid: int | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "in_channels",
            "audio_in_channels",
            "hidden_size",
            "num_layers",
            "token_refiner_num_layers",
            "num_attention_heads",
            "attention_head_dim",
            "ffn_hidden_size",
            "text_dim",
            "timestep_input_dim",
            "time_embed_hidden_size",
            "time_embed_dim",
            "rope_inv_freq_len",
        )
        for field in integer_fields:
            if getattr(self, field) <= 0:
                raise ValueError(f"MiniMax-H3 {field} must be positive")
        if self.in_channels != VIDEO_CHANNELS or self.audio_in_channels != AUDIO_CHANNELS:
            raise ValueError("MiniMax-H3 R1 requires 24 video channels and 32 audio channels")
        if self.patch_size != VIDEO_PATCH_SIZE:
            raise ValueError(f"MiniMax-H3 R1 requires video patch size {VIDEO_PATCH_SIZE}")
        if self.timestep_input_dim % 2:
            raise ValueError("MiniMax-H3 timestep input width must be even")
        if self.rope_inv_freq_len * 6 > self.attention_head_dim:
            raise ValueError("MiniMax-H3 rotary width exceeds the attention head width")
        if self.adaln_curve_grid is not None:
            if self.adaln_curve_grid != 1025:
                raise ValueError("MiniMax-H3 pruned AdaLN requires exactly 1025 curve rows")
            if self.time_embed_dim != 8:
                raise ValueError("MiniMax-H3 pruned AdaLN requires time_embed_dim=8")

    @property
    def attention_inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def video_patch_dim(self) -> int:
        return self.in_channels * math.prod(self.patch_size)

    @property
    def block_adaln_out_features(self) -> int:
        return 6 * self.hidden_size * 3

    @property
    def final_adaln_out_features(self) -> int:
        return 2 * self.hidden_size

    @property
    def is_pruned(self) -> bool:
        return self.adaln_curve_grid is not None


_PUBLISHED_CONFIG_FIELDS = {
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
    "image_model": "minimax_h3",
}


def parse_h3_transformer_config(metadata: Mapping[str, str]) -> MiniMaxH3Config:
    artifact_markers = " ".join(f"{key}={value}" for key, value in metadata.items() if key != "config").lower()
    if any(marker in artifact_markers for marker in ("convrot", "int8", "fp8", "quantized")):
        raise ValueError("Quantized or ConvRot MiniMax-H3 artifacts are deferred to R2")
    raw_config = metadata.get("config")
    if raw_config is None:
        raise ValueError("MiniMax-H3 transformer checkpoint is missing config metadata")
    try:
        root = json.loads(raw_config)
        transformer = root["transformer"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("MiniMax-H3 transformer checkpoint has invalid config metadata") from error
    if not isinstance(transformer, dict):
        raise TypeError("MiniMax-H3 transformer config must be an object")
    for field, expected in _PUBLISHED_CONFIG_FIELDS.items():
        if field not in transformer:
            raise ValueError(f"MiniMax-H3 transformer config is missing {field}")
        actual = transformer[field]
        if actual != expected:
            raise ValueError(f"MiniMax-H3 transformer config {field} expected {expected}, got {actual}")
    if transformer.get("adaln_curve_grid") is not None:
        raise ValueError("Pruned MiniMax-H3 AdaLN artifacts are deferred to R2")
    return MiniMaxH3Config()


class TimeEmbedder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.proj_in = nn.Linear(input_dim, hidden_size, dtype=torch.float32, device=device)
        self.proj_out = nn.Linear(hidden_size, output_dim, dtype=torch.float32, device=device)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.input_dim // 2
        frequencies = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
        angles = timesteps.to(torch.float32)[:, None] * frequencies[None]
        embedding = torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1)
        return self.proj_out(F.silu(self.proj_in(embedding)))


class AdalnProj(nn.Module):
    def __init__(
        self,
        timestep_dim: int,
        hidden_size: int,
        expand: int,
        modalities: int,
        *,
        dtype: torch.dtype,
        apply_silu: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.expand = expand
        self.modalities = modalities
        self.hidden_size = hidden_size
        self.apply_silu = apply_silu
        self.linear = nn.Linear(
            timestep_dim,
            expand * hidden_size * modalities,
            dtype=dtype,
            device=device,
        )

    def forward(self, timestep_embeddings: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.apply_silu:
            timestep_embeddings = F.silu(timestep_embeddings)
        projected = self.linear(timestep_embeddings)
        projected = projected.reshape(projected.shape[0] * self.modalities, self.expand * self.hidden_size)
        return projected.chunk(self.expand, dim=-1)


def _apply_rope_split_half(hidden_states: torch.Tensor, rotation_table: torch.Tensor) -> torch.Tensor:
    pairs = rotation_table.shape[-3]
    rotary = torch.stack((hidden_states[..., :pairs], hidden_states[..., pairs : 2 * pairs]), dim=-1)
    rotary = torch.matmul(rotation_table, rotary.unsqueeze(-1)).squeeze(-1)
    rotary = torch.cat((rotary[..., 0], rotary[..., 1]), dim=-1)
    return torch.cat((rotary, hidden_states[..., 2 * pairs :]), dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qk_norm_eps: float,
        *,
        attn_mode: str,
        split_attn: bool,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.attn_mode = "torch" if attn_mode == "sdpa" else attn_mode
        self.split_attn = split_attn
        self.qkv_proj = nn.Linear(hidden_size, self.inner_dim * 3, bias=False, dtype=dtype, device=device)
        self.q_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps, dtype=dtype, device=device)
        self.k_norm = nn.RMSNorm(head_dim, eps=qk_norm_eps, dtype=dtype, device=device)
        self.out_proj = nn.Linear(self.inner_dim, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor, rotation_table: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query, key, value = self.qkv_proj(hidden_states).split(self.inner_dim, dim=-1)
        query = self.q_norm(query.reshape(batch_size, sequence_length, self.num_heads, self.head_dim))
        key = self.k_norm(key.reshape(batch_size, sequence_length, self.num_heads, self.head_dim))
        value = value.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        if rotation_table is not None:
            query = _apply_rope_split_half(query, rotation_table)
            key = _apply_rope_split_half(key, rotation_table)

        if self.attn_mode == "flash3":
            try:
                from musubi_tuner.wan.modules.attention import flash_attention as wan_flash_attention
            except ImportError as error:
                raise RuntimeError("FlashAttention 3 was selected but its runtime is unavailable") from error
            output = wan_flash_attention(
                [query, key, value],
                attn_mode="flash3",
                split_attn=False,
                dtype=hidden_states.dtype if hidden_states.dtype in {torch.float16, torch.bfloat16} else torch.bfloat16,
            )
            output = output.reshape(batch_size, sequence_length, self.inner_dim)
        else:
            params = AttentionParams.create_attention_params(self.attn_mode, self.split_attn)
            output = attention([query, key, value], attn_params=params)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size * 2, bias=False, dtype=dtype, device=device)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, values = self.fc1(hidden_states).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * values)


class RefinerBlock(nn.Module):
    def __init__(self, config: MiniMaxH3Config, *, attn_mode: str, split_attn: bool, dtype, device) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            attn_mode=attn_mode,
            split_attn=split_attn,
            dtype=dtype,
            device=device,
        )
        self.mlp = MLP(config.hidden_size, config.ffn_hidden_size, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states))
        return hidden_states + self.mlp(self.norm2(hidden_states))


class TokenRefiner(nn.Module):
    def __init__(self, config: MiniMaxH3Config, *, attn_mode: str, split_attn: bool, dtype, device) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                RefinerBlock(config, attn_mode=attn_mode, split_attn=split_attn, dtype=dtype, device=device)
                for _ in range(config.token_refiner_num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(config.hidden_size, eps=config.final_norm_eps, dtype=dtype, device=device)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(hidden_states)
        return self.final_norm(hidden_states)


def _mod_scale_shift(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    segments: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    # Callers pass a fresh norm output; these disjoint slice updates retain trainable AdaLN gradients.
    for start, stop, row in segments:
        hidden_states[:, start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return hidden_states


def _mod_gate(
    residual: torch.Tensor,
    update: torch.Tensor,
    gate: torch.Tensor,
    segments: tuple[tuple[int, int, int], ...],
) -> torch.Tensor:
    output = residual.clone()
    for start, stop, row in segments:
        output[:, start:stop].add_(update[:, start:stop] * gate[row])
    return output


class DiTBlock(nn.Module):
    def __init__(self, config: MiniMaxH3Config, *, attn_mode: str, split_attn: bool, dtype, device) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.norm_eps, dtype=dtype, device=device)
        self.attn = Attention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_head_dim,
            config.qk_norm_eps,
            attn_mode=attn_mode,
            split_attn=split_attn,
            dtype=dtype,
            device=device,
        )
        self.mlp = MLP(config.hidden_size, config.ffn_hidden_size, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(
            config.time_embed_dim,
            config.hidden_size,
            expand=6,
            modalities=3,
            dtype=dtype,
            apply_silu=not config.is_pruned,
            device=device,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embeddings: torch.Tensor,
        modulation_segments: tuple[tuple[int, int, int], ...],
        rotation_table: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(timestep_embeddings)
        normalized = _mod_scale_shift(self.norm1(hidden_states), shift_msa, scale_msa, modulation_segments)
        hidden_states = _mod_gate(
            hidden_states,
            self.attn(normalized, rotation_table),
            gate_msa,
            modulation_segments,
        )
        normalized = _mod_scale_shift(self.norm2(hidden_states), shift_mlp, scale_mlp, modulation_segments)
        return _mod_gate(hidden_states, self.mlp(normalized), gate_mlp, modulation_segments)


class FinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        timestep_dim: int,
        video_output_dim: int,
        audio_output_dim: int,
        *,
        norm_eps: float = 1e-5,
        dtype: torch.dtype,
        apply_silu: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps, dtype=dtype, device=device)
        self.adaln_proj = AdalnProj(
            timestep_dim,
            hidden_size,
            expand=2,
            modalities=1,
            dtype=dtype,
            apply_silu=apply_silu,
            device=device,
        )
        self.video_out = nn.Linear(hidden_size, video_output_dim, dtype=torch.float32, device=device)
        self.audio_out = nn.Linear(hidden_size, audio_output_dim, dtype=torch.float32, device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embeddings: torch.Tensor,
        *,
        video_slice: slice,
        audio_slice: slice,
        video_timestep_index: int,
        audio_timestep_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(timestep_embeddings)
        video = self.norm(hidden_states[:, video_slice])
        video = video * (1.0 + scale[video_timestep_index]) + shift[video_timestep_index]
        audio = self.norm(hidden_states[:, audio_slice])
        audio = audio * (1.0 + scale[audio_timestep_index]) + shift[audio_timestep_index]
        return self.video_out(video.to(torch.float32)), self.audio_out(audio.to(torch.float32))


@dataclass(frozen=True)
class MiniMaxH3Output:
    video: torch.Tensor
    audio: torch.Tensor


class MiniMaxH3Model(nn.Module):
    def __init__(
        self,
        config: MiniMaxH3Config,
        *,
        attn_mode: str = "torch",
        split_attn: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if attn_mode not in {"torch", "sdpa", "flash", "flash3", "sageattn", "xformers"}:
            raise ValueError(f"Unsupported MiniMax-H3 attention mode: {attn_mode}")
        self.config = config
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.video_patch_proj = nn.Linear(config.video_patch_dim, config.hidden_size, dtype=torch.float32, device=device)
        self.audio_patch_proj = nn.Linear(config.audio_in_channels, config.hidden_size, dtype=torch.float32, device=device)
        self.condition_proj = nn.Linear(config.text_dim, config.hidden_size, dtype=dtype, device=device)
        if config.is_pruned:
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.empty(config.adaln_curve_grid, config.time_embed_dim, dtype=torch.float32, device=device),
            )
        else:
            self.time_embedder = TimeEmbedder(
                config.timestep_input_dim,
                config.time_embed_hidden_size,
                config.time_embed_dim,
                device=device,
            )
        self.rope = nn.Module()
        inv_freq = torch.empty(config.rope_inv_freq_len, dtype=torch.float32, device=device)
        self.rope.register_buffer("inv_freq", inv_freq)
        self.token_refiner = TokenRefiner(
            config,
            attn_mode=attn_mode,
            split_attn=split_attn,
            dtype=dtype,
            device=device,
        )
        self.blocks = nn.ModuleList(
            [
                DiTBlock(config, attn_mode=attn_mode, split_attn=split_attn, dtype=dtype, device=device)
                for _ in range(config.num_layers)
            ]
        )
        self.final_layer = FinalLayer(
            config.hidden_size,
            config.time_embed_dim,
            config.video_patch_dim,
            config.audio_in_channels,
            norm_eps=config.final_norm_eps,
            dtype=dtype,
            apply_silu=not config.is_pruned,
            device=device,
        )

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = 0
        self.offloader = None
        self._execution_device = torch.device(device) if device is not None else None
        self._rotary_cache: OrderedDict[tuple[H3PackedLayout, torch.device, torch.dtype], torch.Tensor] = OrderedDict()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        if hasattr(self, "_rotary_cache"):
            self._rotary_cache.clear()
        return result

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self._rotary_cache.clear()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def device(self) -> torch.device:
        return self.video_patch_proj.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.condition_proj.weight.dtype

    def _timestep_embeddings(
        self,
        unique_timesteps: torch.Tensor,
        execution_device: torch.device,
    ) -> torch.Tensor:
        if unique_timesteps.ndim != 1:
            raise ValueError("MiniMax-H3 unique timesteps must be a one-dimensional tensor")
        if not self.config.is_pruned:
            return self.time_embedder(unique_timesteps.to(execution_device)).to(self.dtype)

        table = self.adaln_t_table
        timesteps_fp32 = unique_timesteps.to(device=table.device, dtype=torch.float32)
        position = timesteps_fp32.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        fraction = position - lower.to(position.dtype)
        embedding_fp32 = torch.lerp(table.float()[lower], table.float()[lower + 1], fraction[:, None])
        return embedding_fp32.to(self.dtype)

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig) -> None:
        if num_blocks <= 0:
            raise ValueError("MiniMax-H3 blocks to swap must be positive")
        if num_blocks > len(self.blocks) - 2:
            raise ValueError(f"MiniMax-H3 cannot swap more than {len(self.blocks) - 2} of {len(self.blocks)} blocks")
        self.blocks_to_swap = num_blocks
        self._execution_device = torch.device(config.device)
        self.offloader = create_offloader(
            "minimax-h3",
            self.blocks,
            len(self.blocks),
            num_blocks,
            config,
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        device = torch.device(device)
        if self.blocks_to_swap:
            saved_blocks = self.blocks
            self.blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = saved_blocks
        self._execution_device = device

    def prepare_block_swap_before_forward(self) -> None:
        if not self.blocks_to_swap:
            return
        if self.offloader is None:
            raise RuntimeError("MiniMax-H3 block swap has no offloader")
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def switch_block_swap_for_inference(self) -> None:
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def _assert_block_device(self, block: nn.Module, index: int) -> None:
        expected = self._execution_device
        if expected is None:
            return

        def matches(actual: torch.device) -> bool:
            return actual.type == expected.type and (expected.index is None or actual.index == expected.index)

        for name, parameter in block.named_parameters():
            if not matches(parameter.device):
                raise RuntimeError(
                    f"MiniMax-H3 block {index} parameter {name} is on {parameter.device}, expected {expected} after wait"
                )
        for name, buffer in block.named_buffers():
            if not matches(buffer.device):
                raise RuntimeError(f"MiniMax-H3 block {index} buffer {name} is on {buffer.device}, expected {expected} after wait")

    def _rotation_table(self, position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        positions = position_ids.to(device=self.rope.inv_freq.device, dtype=torch.float32)
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("MiniMax-H3 position ids must be [B,S,3]")
        per_axis = positions.unsqueeze(-1) * self.rope.inv_freq.reshape(1, 1, 1, -1)
        temporal, height, width = per_axis.unbind(dim=2)
        half = torch.cat((temporal, height, width), dim=-1)
        angles = torch.cat((half, half), dim=-1)
        pair_count = angles.shape[-1] // 2
        angles = angles[..., :pair_count]
        cosine, sine = torch.cos(angles), torch.sin(angles)
        return (
            torch.stack((cosine, -sine, sine, cosine), dim=-1)
            .reshape(
                angles.shape[0],
                angles.shape[1],
                1,
                pair_count,
                2,
                2,
            )
            .to(dtype)
        )

    def _cached_rotation_table(
        self,
        layout: H3PackedLayout,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (layout, torch.device(device), dtype)
        cached = self._rotary_cache.get(key)
        if cached is not None:
            self._rotary_cache.move_to_end(key)
            return cached
        position_ids = build_position_grid(layout, device=device)
        rotation_table = self._rotation_table(position_ids, dtype)
        self._rotary_cache[key] = rotation_table
        while len(self._rotary_cache) > _ROTARY_CACHE_SIZE:
            self._rotary_cache.popitem(last=False)
        return rotation_table

    @staticmethod
    def _validate_text_tags(tags: torch.Tensor, batch_size: int, text_length: int) -> torch.Tensor:
        tags = torch.as_tensor(tags)
        if tags.shape != (batch_size, text_length):
            raise ValueError(f"MiniMax-H3 text token tags must be [{batch_size},{text_length}]")
        return tags

    @staticmethod
    def _validate_condition_rows(
        latents: Sequence[torch.Tensor],
        segments,
        pack,
        *,
        batch_size: int,
        label: str,
    ) -> list[torch.Tensor]:
        if len(latents) != len(segments):
            raise ValueError(f"MiniMax-H3 expected {len(segments)} {label} tensors, got {len(latents)}")
        rows = []
        for tensor, segment in zip(latents, segments):
            if tensor.shape[0] != batch_size:
                raise ValueError(f"MiniMax-H3 {segment.role} batch size does not match the targets")
            packed = pack(tensor)
            if packed.shape[1] != segment.row_count:
                raise ValueError(f"MiniMax-H3 {segment.role} has {packed.shape[1]} packed rows, expected {segment.row_count}")
            rows.append(packed)
        return rows

    def forward(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        text_hidden_states: torch.Tensor,
        text_token_tags: torch.Tensor,
        layout: H3PackedLayout,
        model_t_video: float | torch.Tensor,
        model_t_audio: float | torch.Tensor,
        visual_condition_latents: Sequence[torch.Tensor] = (),
        audio_condition_latents: Sequence[torch.Tensor] = (),
        visual_condition_clean: float = 0.999,
        audio_condition_clean: float = 1.0,
    ) -> MiniMaxH3Output:
        if video_latents.ndim != 5 or tuple(video_latents.shape[2:]) != (
            layout.target_video.frames,
            layout.target_video.height,
            layout.target_video.width,
        ):
            raise ValueError("MiniMax-H3 target video tensor does not match the packed layout")
        if audio_latents.ndim != 4 or tuple(audio_latents.shape[1:]) != (
            self.config.audio_in_channels,
            2,
            layout.target_audio_frames,
        ):
            raise ValueError("MiniMax-H3 target audio tensor does not match the packed layout")
        batch_size = video_latents.shape[0]
        if batch_size != 1:
            raise ValueError(f"MiniMax-H3 R1 requires batch_size=1, got {batch_size}; use gradient accumulation")
        if audio_latents.shape[0] != batch_size:
            raise ValueError("MiniMax-H3 target video and audio batch sizes differ")
        if text_hidden_states.shape != (batch_size, layout.text_length, self.config.text_dim):
            raise ValueError(f"MiniMax-H3 text hidden states must be [{batch_size},{layout.text_length},{self.config.text_dim}]")
        text_token_tags = self._validate_text_tags(text_token_tags, batch_size, layout.text_length)

        execution_device = self.video_patch_proj.weight.device
        video_dtype = video_latents.dtype
        audio_dtype = audio_latents.dtype
        video_latents = video_latents.to(execution_device)
        audio_latents = audio_latents.to(execution_device)
        text_hidden_states = text_hidden_states.to(execution_device)
        visual_condition_latents = tuple(tensor.to(execution_device) for tensor in visual_condition_latents)
        audio_condition_latents = tuple(tensor.to(execution_device) for tensor in audio_condition_latents)

        visual_segments = tuple(segment for segment in layout.segments if segment.kind == "visual_condition")
        audio_segments = tuple(segment for segment in layout.segments if segment.kind == "audio_condition")
        if layout.task == "fl2va":
            expected_visual_geometries = layout.visual_conditions
        elif layout.task == "ref2va":
            expected_visual_geometries = tuple(
                reference.video for reference in layout.references if reference.kind in {"image", "video"}
            )
        else:
            expected_visual_geometries = ()
        expected_audio_frames = (
            tuple(reference.audio_frames for reference in layout.references if reference.audio_frames)
            if layout.task == "ref2va"
            else ()
        )
        for tensor, segment, geometry in zip(
            visual_condition_latents,
            visual_segments,
            expected_visual_geometries,
        ):
            actual = tuple(tensor.shape[2:]) if tensor.ndim == 5 else tuple(tensor.shape)
            expected = (geometry.frames, geometry.height, geometry.width)
            if actual != expected:
                raise ValueError(
                    f"MiniMax-H3 {segment.role} geometry expected {'x'.join(map(str, expected))}, got {'x'.join(map(str, actual))}"
                )
        for tensor, segment, frames in zip(audio_condition_latents, audio_segments, expected_audio_frames):
            actual = tuple(tensor.shape[1:]) if tensor.ndim == 4 else tuple(tensor.shape)
            expected = (self.config.audio_in_channels, 2, frames)
            if actual != expected:
                raise ValueError(
                    f"MiniMax-H3 {segment.role} geometry expected {'x'.join(map(str, expected))}, got {'x'.join(map(str, actual))}"
                )
        visual_rows = self._validate_condition_rows(
            visual_condition_latents,
            visual_segments,
            pack_video_rows,
            batch_size=batch_size,
            label="visual condition",
        )
        audio_rows = self._validate_condition_rows(
            audio_condition_latents,
            audio_segments,
            pack_audio_rows,
            batch_size=batch_size,
            label="audio condition",
        )
        visual_rows.append(pack_video_rows(video_latents))
        audio_rows.append(pack_audio_rows(audio_latents))
        projected_video = self.video_patch_proj(torch.cat(visual_rows, dim=1).to(torch.float32)).to(self.dtype)
        projected_audio = self.audio_patch_proj(torch.cat(audio_rows, dim=1).to(torch.float32)).to(self.dtype)

        text = self.condition_proj(text_hidden_states.to(self.dtype))
        text = self.token_refiner(text)
        parts = []
        visual_offset = 0
        audio_offset = 0
        for segment in layout.segments:
            if segment.kind == "text":
                parts.append(text)
            elif segment.kind in {"visual_condition", "target_video"}:
                parts.append(projected_video[:, visual_offset : visual_offset + segment.row_count])
                visual_offset += segment.row_count
            else:
                parts.append(projected_audio[:, audio_offset : audio_offset + segment.row_count])
                audio_offset += segment.row_count
        hidden_states = torch.cat(parts, dim=1)
        if hidden_states.shape[1] != layout.row_count:
            raise RuntimeError("MiniMax-H3 packed residual stream length drifted from its layout")

        timestep_rows = build_timestep_rows(
            layout,
            text_token_tags=text_token_tags,
            model_t_video=model_t_video,
            model_t_audio=model_t_audio,
            visual_condition_clean=visual_condition_clean,
            audio_condition_clean=audio_condition_clean,
        )
        timestep_embeddings = self._timestep_embeddings(timestep_rows.unique_timesteps, execution_device)
        rotation_table = self._cached_rotation_table(
            layout,
            device=execution_device,
            dtype=hidden_states.dtype,
        )

        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
                self._assert_block_device(block, index)
            if self.gradient_checkpointing and self.training:
                forward_fn = block
                if self.activation_cpu_offloading:
                    forward_fn = create_cpu_offloading_wrapper(forward_fn, execution_device)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    forward_fn,
                    hidden_states,
                    timestep_embeddings,
                    timestep_rows.block_segments,
                    rotation_table,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    timestep_embeddings,
                    timestep_rows.block_segments,
                    rotation_table,
                )
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, index)

        video_rows, audio_rows = self.final_layer(
            hidden_states,
            timestep_embeddings,
            video_slice=layout.target_video_segment.row_slice,
            audio_slice=layout.target_audio_segment.row_slice,
            video_timestep_index=timestep_rows.video_timestep_index,
            audio_timestep_index=timestep_rows.audio_timestep_index,
        )
        video, audio = unpack_targets(layout, video_rows, audio_rows)
        return MiniMaxH3Output(video=video.to(video_dtype), audio=audio.to(audio_dtype))


def load_h3_transformer(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "torch",
    split_attn: bool = False,
    disable_mmap: bool = False,
) -> MiniMaxH3Model:
    if dtype != torch.bfloat16:
        raise ValueError("MiniMax-H3 R1 accepts only BF16 transformer checkpoints")
    files = resolve_safetensors_files(checkpoint_path)
    config = parse_h3_transformer_config(load_safetensors_metadata(files))
    return load_safetensors_module(
        lambda: MiniMaxH3Model(config, attn_mode=attn_mode, split_attn=split_attn, dtype=dtype),
        files,
        device=device,
        dtype=None,
        strict_dtype=True,
        disable_mmap=disable_mmap,
    )
