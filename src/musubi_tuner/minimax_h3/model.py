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
import logging
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from musubi_tuner.minimax_h3.checkpoint import (
    load_safetensors_metadata,
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
from musubi_tuner.modules.convrot_int8_utils import canonicalize_convrot_int8_key, has_comfy_quant_tensors
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, WeightTransformHooks

logger = logging.getLogger(__name__)

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
    # pruned-AdaLN artifacts: the sinusoidal time embedder is replaced by a published
    # FP32 [adaln_curve_grid, time_embed_dim] lookup table interpolated over t in [0, 1]
    adaln_curve_grid: int | None = None
    # self-pruned AdaLN (--prune_adaln): the sinusoidal time embedder is retained and its
    # SiLU'd output is projected onto a load-time mean-centered rank-r SVD basis; AdaLN
    # projections take the r basis coefficients plus a constant channel carrying the mean
    adaln_rank: int | None = None

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
            if self.adaln_rank is not None:
                raise ValueError("MiniMax-H3 pruned AdaLN curve table and self-pruned basis are mutually exclusive")
            if self.adaln_curve_grid != 1025:
                raise ValueError("MiniMax-H3 pruned AdaLN requires exactly 1025 curve rows")
            if self.time_embed_dim != 8:
                raise ValueError("MiniMax-H3 pruned AdaLN requires time_embed_dim=8")
        if self.adaln_rank is not None and self.adaln_rank <= 0:
            raise ValueError("MiniMax-H3 self-pruned AdaLN rank must be positive")

    @property
    def is_pruned(self) -> bool:
        return self.adaln_curve_grid is not None or self.adaln_rank is not None

    @property
    def adaln_in_features(self) -> int:
        if self.adaln_rank is not None:
            # r basis coefficients plus the constant channel carrying the curve mean
            return self.adaln_rank + 1
        return self.time_embed_dim

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


def parse_h3_transformer_config(metadata: Mapping[str, str], *, allow_convrot_int8: bool = False) -> MiniMaxH3Config:
    artifact_markers = " ".join(f"{key}={value}" for key, value in metadata.items() if key != "config").lower()
    blocked_markers = ("fp8", "nvfp4") if allow_convrot_int8 else ("convrot", "int8", "fp8", "nvfp4", "quantized")
    if any(marker in artifact_markers for marker in blocked_markers):
        raise ValueError(
            "Unsupported quantized MiniMax-H3 checkpoint. ConvRot INT8 checkpoints are detected from their"
            " tensor structure (or pass --convrot_int8 to quantize a BF16 checkpoint); other quantized"
            " formats (fp8/NVFP4) are not supported."
        )
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
        raise ValueError("Pruned MiniMax-H3 AdaLN must be classified from its tensor structure")
    return MiniMaxH3Config()


def _read_h3_tensor_headers(files: Sequence[str | Path]) -> dict[str, tuple]:
    """Read tensor headers (dtype string, shape tuple) across shards, keyed by canonical name."""
    headers: dict[str, tuple] = {}
    for file in files:
        checkpoint_path = str(Path(file).resolve())
        with MemoryEfficientSafeOpen(checkpoint_path) as handle:
            for raw_key in handle.keys():
                key = canonicalize_convrot_int8_key(raw_key)
                if key in headers:
                    raise ValueError(f"Duplicate MiniMax-H3 tensor header {key!r} in {checkpoint_path}")
                raw_header = handle.header[raw_key]
                headers[key] = (raw_header["dtype"], tuple(raw_header["shape"]))
    return headers


def _published_pruned_h3_config() -> MiniMaxH3Config:
    return MiniMaxH3Config(time_embed_dim=8, adaln_curve_grid=1025)


def _require_pruned_adaln_header(headers: Mapping[str, tuple], key: str, expected_shape: tuple) -> None:
    header = headers.get(key)
    if header is None:
        raise ValueError(f"Pruned MiniMax-H3 checkpoint is missing {key}")
    if header[1] != expected_shape:
        raise ValueError(f"Pruned MiniMax-H3 {key} expected shape {expected_shape}, got {header[1]}")


def classify_h3_transformer(
    files: Sequence[str | Path],
    *,
    allow_convrot_int8: bool = False,
) -> MiniMaxH3Config:
    """Classify a transformer checkpoint as the published full or pruned-AdaLN structure.

    Pruned artifacts (published as ConvRot INT8 and as BF16) are recognized structurally:
    the ``adaln_t_table`` FP32 [1025, 8] curve table replaces the sinusoidal time
    embedder, and every AdaLN projection takes the 8-wide interpolated embedding
    directly (no SiLU). Full checkpoints keep the published-config metadata validation.
    """
    headers = _read_h3_tensor_headers(files)
    table = headers.get("adaln_t_table")
    if table is None:
        return parse_h3_transformer_config(load_safetensors_metadata(files), allow_convrot_int8=allow_convrot_int8)
    if any(key.startswith("time_embedder.") for key in headers):
        raise ValueError("MiniMax-H3 checkpoint has mixed adaln_t_table and time_embedder structures")
    table_dtype, table_shape = table
    if table_dtype != "F32":
        raise ValueError(f"Pruned MiniMax-H3 adaln_t_table must be F32, got {table_dtype}")
    if table_shape != (1025, 8):
        raise ValueError(f"Pruned MiniMax-H3 adaln_t_table must have shape [1025, 8], got {table_shape}")
    config = _published_pruned_h3_config()
    for index in range(config.num_layers):
        prefix = f"blocks.{index}.adaln_proj.linear"
        _require_pruned_adaln_header(headers, f"{prefix}.weight", (config.block_adaln_out_features, config.time_embed_dim))
        _require_pruned_adaln_header(headers, f"{prefix}.bias", (config.block_adaln_out_features,))
    _require_pruned_adaln_header(
        headers, "final_layer.adaln_proj.linear.weight", (config.final_adaln_out_features, config.time_embed_dim)
    )
    _require_pruned_adaln_header(headers, "final_layer.adaln_proj.linear.bias", (config.final_adaln_out_features,))
    return config


# --prune_adaln: rank 8 matches the published pruned artifacts; the grid samples the
# model time t in [0, 1] densely for the mean-centered SVD (the curve is smooth, so the
# basis is insensitive to the exact grid size beyond a few hundred points)
H3_PRUNE_ADALN_RANK = 8
H3_PRUNE_ADALN_GRID = 4096

_H3_ADALN_WEIGHT_SUFFIX = ".adaln_proj.linear.weight"
_H3_TIME_EMBEDDER_PREFIX = "time_embedder."


def read_h3_time_embedder_state(files: Sequence[str | Path]) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for file in files:
        with MemoryEfficientSafeOpen(str(Path(file).resolve())) as handle:
            for key in handle.keys():
                if key.startswith(_H3_TIME_EMBEDDER_PREFIX):
                    state[key] = handle.get_tensor(key)
    return state


def compute_pruned_adaln_basis(
    time_embedder_state: Mapping[str, torch.Tensor],
    config: MiniMaxH3Config,
    *,
    rank: int = H3_PRUNE_ADALN_RANK,
    grid: int = H3_PRUNE_ADALN_GRID,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-centered rank-``rank`` SVD basis of the SiLU'd time-embedding curve.

    Returns FP32 ``(basis [time_embed_dim, rank], mean [time_embed_dim])`` computed over
    a uniform ``grid``-point sampling of the model time t in [0, 1]. The SiLU that the
    full model applies per block is baked into the curve, so pruned projections consume
    the coefficients directly.
    """
    embedder = TimeEmbedder(config.timestep_input_dim, config.time_embed_hidden_size, config.time_embed_dim)
    embedder.load_state_dict({key.removeprefix(_H3_TIME_EMBEDDER_PREFIX): value for key, value in time_embedder_state.items()})
    with torch.no_grad():
        curve = F.silu(embedder(torch.linspace(0.0, 1.0, grid, dtype=torch.float32)))
    mean = curve.mean(dim=0)
    _, _, v_rows = torch.linalg.svd(curve - mean, full_matrices=False)
    return v_rows[:rank].T.contiguous(), mean


def build_prune_adaln_hooks(basis: torch.Tensor, mean: torch.Tensor) -> WeightTransformHooks:
    """Streaming rewrite W [out, D] -> W @ [basis | mean] [out, rank + 1] in BF16.

    A pure per-tensor transform: biases are untouched because the constant coefficient
    channel applies the mean term at runtime. The rank + 1 input width is not divisible
    by any ConvRot group size, so --convrot_int8 skips the rewritten projections, the
    same scope as the published pruned INT8 artifacts.
    """
    projection = torch.cat((basis, mean[:, None]), dim=1)

    def split_hook(key: str, tensor: torch.Tensor | None):
        if not key.endswith(_H3_ADALN_WEIGHT_SUFFIX):
            return None, None
        if tensor is None:
            return [key], None
        pruned = (tensor.to(torch.float32) @ projection.to(tensor.device)).to(torch.bfloat16)
        return [key], [pruned]

    return WeightTransformHooks(split_hook=split_hook)


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
        # pruned-AdaLN checkpoints bake the activation into the curve table, so the
        # 8-wide interpolated embedding feeds the projection directly
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
        query = query.to(value)
        key = key.to(value)

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
            config.adaln_in_features,
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
        if config.adaln_curve_grid is not None:
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
            if config.adaln_rank is not None:
                # load-time mean-centered SVD basis of the SiLU'd time-embedding curve
                self.register_buffer(
                    "adaln_basis",
                    torch.empty(config.time_embed_dim, config.adaln_rank, dtype=torch.float32, device=device),
                )
                self.register_buffer(
                    "adaln_mean",
                    torch.empty(config.time_embed_dim, dtype=torch.float32, device=device),
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
            config.adaln_in_features,
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
        # norms are never quantized, so this sentinel stays the compute dtype even when a
        # checkpoint quantizes more Linears than the published ConvRot INT8 scope
        return self.blocks[0].norm1.weight.dtype

    def _timestep_embeddings(self, unique_timesteps: torch.Tensor, execution_device: torch.device) -> torch.Tensor:
        if unique_timesteps.ndim != 1:
            raise ValueError("MiniMax-H3 unique timesteps must be a one-dimensional tensor")
        if not self.config.is_pruned:
            return self.time_embedder(unique_timesteps.to(execution_device)).to(self.dtype)

        if self.config.adaln_rank is not None:
            # self-pruned AdaLN: the exact FP32 time-embedding curve projected onto the
            # load-time basis; the trailing constant channel applies the curve mean via
            # the per-block projections (their weights are W @ [basis | mean])
            embedding = self.time_embedder(unique_timesteps.to(execution_device))
            centered = F.silu(embedding) - self.adaln_mean
            coefficients = F.pad(centered @ self.adaln_basis, (0, 1), value=1.0)
            return coefficients.to(self.dtype)

        # pruned AdaLN: FP32 linear interpolation over the published [1025, 8] curve
        # table; the model time t = 1 - sigma lives in [0, 1] and maps onto the grid
        table = self.adaln_t_table
        timesteps_fp32 = unique_timesteps.to(device=table.device, dtype=torch.float32)
        position = timesteps_fp32.clamp(0.0, 1.0) * (table.shape[0] - 1)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        fraction = position - lower.to(position.dtype)
        embedding_fp32 = torch.lerp(table[lower], table[lower + 1], fraction[:, None])
        return embedding_fp32.to(device=execution_device, dtype=self.dtype)

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

        if hidden_states.device != execution_device:
            hidden_states = hidden_states.to(execution_device)

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


# ConvRot INT8 scope mirrors the ComfyUI distribution: the five per-block Linears
# (attn qkv/out, mlp fc1/fc2, adaln_proj). token_refiner blocks and the final layer
# stay BF16. adaln_proj's in_features (2688) is not a multiple of 256, so the group
# size falls back to 64 there — the same choice ComfyUI publishes in `comfy_quant`.
H3_CONVROT_INT8_TARGET_KEYS = ["attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear"]
H3_CONVROT_INT8_EXCLUDE_KEYS = ["token_refiner.", "final_layer."]
H3_CONVROT_INT8_ALLOWED_GROUPSIZES = (256, 64)


def _load_h3_transformer_convrot_int8(
    files: list[Path],
    config: MiniMaxH3Config,
    *,
    device: torch.device,
    quant_device: torch.device,
    bwd_mode: str,
    attn_mode: str,
    split_attn: bool,
    disable_mmap: bool,
    lora_weights: list[dict] | None = None,
    lora_multipliers: list[float] | None = None,
    prune_hooks: WeightTransformHooks | None = None,
    prune_state: dict[str, torch.Tensor] | None = None,
) -> MiniMaxH3Model:
    """Load the transformer with ConvRot INT8 base weights.

    BF16 checkpoints are quantized on the fly on ``quant_device`` (LoRA weights, if any,
    are merged into the BF16 weights first); ComfyUI pre-quantized checkpoints are
    converted to the Musubi layout during the same streaming pass. ``device`` is where
    the weights end up ("cpu" under block swap). ``prune_hooks`` (--prune_adaln) rewrites
    the AdaLN projections before the quantization decision, which then skips them.
    """
    from accelerate import init_empty_weights

    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

    with init_empty_weights():
        model = MiniMaxH3Model(config, attn_mode=attn_mode, split_attn=split_attn, dtype=torch.bfloat16)

    quantizer = ConvRotInt8Quantizer(
        H3_CONVROT_INT8_TARGET_KEYS,
        H3_CONVROT_INT8_EXCLUDE_KEYS,
        allowed_groupsizes=H3_CONVROT_INT8_ALLOWED_GROUPSIZES,
    )
    sd = load_safetensors_with_lora_and_fp8(
        model_files=[str(path) for path in files],
        lora_weights_list=lora_weights,
        lora_multipliers=lora_multipliers,
        fp8_optimization=False,
        calc_device=quant_device,
        move_to_device=(device == quant_device),
        dit_weight_dtype=None,
        disable_numpy_memmap=disable_mmap,
        weight_transform_hooks=prune_hooks,
        quantizer=quantizer,
    )
    if prune_state is not None:
        sd.update(prune_state)
    apply_convrot_int8_monkey_patch(model, sd, bwd_mode=bwd_mode, groupsize_map=quantizer.module_groupsizes)
    # int8 tensors cannot require grad, and load_state_dict(assign=True) re-wraps incoming
    # tensors with the meta params' requires_grad (default True); the base is frozen anyway.
    model.requires_grad_(False)
    for key in sd.keys():
        # the H3 DiT stores only BF16 compute weights and FP32 fixed-precision islands;
        # F16 tensors (e.g. pruned-artifact AdaLN projections) convert to the compute dtype
        if sd[key].dtype == torch.float16:
            sd[key] = sd[key].to(torch.bfloat16)
        if device.type != "cpu":
            sd[key] = sd[key].to(device)
    # strict load keeps the R1 key verification: missing/unexpected keys raise
    model.load_state_dict(sd, strict=True, assign=True)
    model.eval()
    return model


def _load_h3_transformer_bf16(
    files: Sequence[str | Path],
    config: MiniMaxH3Config,
    *,
    device: torch.device | str,
    attn_mode: str,
    split_attn: bool,
    disable_mmap: bool,
    prune_hooks: WeightTransformHooks | None = None,
    prune_state: dict[str, torch.Tensor] | None = None,
) -> MiniMaxH3Model:
    from accelerate import init_empty_weights

    from musubi_tuner.utils.safetensors_utils import load_safetensors

    with init_empty_weights():
        model = MiniMaxH3Model(config, attn_mode=attn_mode, split_attn=split_attn, dtype=torch.bfloat16)

    # load straight to the target device: on CUDA the per-tensor memmap path avoids a
    # resident full-model CPU copy (pages stay file-backed) and the .to(device) below
    # becomes a no-op; under block swap the caller passes "cpu" and nothing changes
    device = torch.device(device)
    sd: dict[str, torch.Tensor] = {}
    if prune_hooks is not None:
        # streaming per-tensor load so each full [out, D] AdaLN weight is rewritten to
        # [out, rank + 1] as it arrives; peak memory stays near the pruned model size
        from musubi_tuner.utils.lora_utils import load_safetensors_with_fp8_optimization_and_hook

        sd = load_safetensors_with_fp8_optimization_and_hook(
            [str(file) for file in files],
            False,
            device,
            move_to_device=True,
            weight_transform_hooks=prune_hooks,
            disable_numpy_memmap=disable_mmap,
        )
        sd.update(prune_state or {})
    else:
        for file in files:
            sd.update(load_safetensors(str(file), device=device, disable_mmap=True, disable_numpy_memmap=disable_mmap))
    # pruned artifacts publish the AdaLN projections in F16; convert them to the BF16
    # compute dtype (the same treatment as the ConvRot INT8 loader)
    if config.is_pruned:
        for key, tensor in sd.items():
            if ".adaln_proj.linear." in key and tensor.dtype == torch.float16:
                sd[key] = tensor.to(torch.bfloat16)
    # the model definition is the dtype source of truth: BF16 compute weights plus FP32
    # fixed-precision islands must arrive exactly as published, without silent conversion
    expected_state = model.state_dict()
    dtype_mismatches = sorted(
        f"{key}: expected {expected_state[key].dtype}, got {tensor.dtype}"
        for key, tensor in sd.items()
        if key in expected_state and tensor.dtype != expected_state[key].dtype
    )
    if dtype_mismatches:
        raise ValueError(f"MiniMax-H3 checkpoint dtype mismatch: {dtype_mismatches[:20]}")
    model.load_state_dict(sd, strict=True, assign=True)
    model.to(device)
    model.eval()
    return model


def load_h3_transformer(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    attn_mode: str = "torch",
    split_attn: bool = False,
    disable_mmap: bool = False,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    quant_device: torch.device | str | None = None,
    lora_weights: list[dict] | None = None,
    lora_multipliers: list[float] | None = None,
    prune_adaln: bool = False,
) -> MiniMaxH3Model:
    if dtype != torch.bfloat16:
        raise ValueError("MiniMax-H3 accepts only BF16 transformer checkpoints")
    files = resolve_safetensors_files(checkpoint_path)
    # Pre-quantized ConvRot INT8 artifacts (full or pruned) and pruned BF16 artifacts are
    # detected from their tensor structure; --convrot_int8 additionally quantizes BF16
    # checkpoints (full or pruned) on the fly — pruned AdaLN projections are 8-wide and
    # fall outside every ConvRot group size, so the quantizer skips them automatically.
    prequantized = has_comfy_quant_tensors(files, disable_numpy_memmap=disable_mmap)
    use_convrot_int8 = convrot_int8 or prequantized
    config = classify_h3_transformer(files, allow_convrot_int8=use_convrot_int8)
    prune_hooks: WeightTransformHooks | None = None
    prune_state: dict[str, torch.Tensor] | None = None
    if prune_adaln:
        if config.is_pruned:
            logger.info("MiniMax-H3 checkpoint is already pruned; --prune_adaln has no effect")
        elif prequantized:
            raise ValueError(
                "--prune_adaln requires a BF16 MiniMax-H3 checkpoint; pre-quantized ConvRot INT8 checkpoints"
                " cannot be pruned at load time (use the published pruned artifact instead)"
            )
        else:
            config = replace(config, adaln_rank=H3_PRUNE_ADALN_RANK)
            basis, mean = compute_pruned_adaln_basis(read_h3_time_embedder_state(files), config)
            prune_hooks = build_prune_adaln_hooks(basis, mean)
            prune_state = {"adaln_basis": basis, "adaln_mean": mean}
            logger.info(
                f"Pruning MiniMax-H3 AdaLN projections at load time (mean-centered rank-{H3_PRUNE_ADALN_RANK} basis,"
                " time embedder retained)"
            )
    if use_convrot_int8:
        device = torch.device(device)
        return _load_h3_transformer_convrot_int8(
            [Path(checkpoint_path)],  # unexpanded: the streaming loader expands split shards itself
            config,
            device=device,
            quant_device=device if quant_device is None else torch.device(quant_device),
            bwd_mode=convrot_int8_bwd,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_mmap=disable_mmap,
            lora_weights=lora_weights,
            lora_multipliers=lora_multipliers,
            prune_hooks=prune_hooks,
            prune_state=prune_state,
        )
    if lora_weights:
        raise ValueError("MiniMax-H3 load-time LoRA merge is only wired for the ConvRot INT8 path")
    return _load_h3_transformer_bf16(
        files,
        config,
        device=device,
        attn_mode=attn_mode,
        split_attn=split_attn,
        disable_mmap=disable_mmap,
        prune_hooks=prune_hooks,
        prune_state=prune_state,
    )
