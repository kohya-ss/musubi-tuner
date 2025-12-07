# Copyright (c) 2025 Z-Image Team
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
# Original implementation: https://github.com/Tongyi-MAI/Z-Image
# Modifications: Copied and modified for Musubi Tuner project.

"""Z-Image Transformer."""

import math
from typing import Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights

from musubi_tuner.modules.attention import AttentionParams, attention
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import WeightTransformHooks

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.zimage import zimage_config
from musubi_tuner.zimage.zimage_config import (
    ADALN_EMBED_DIM,
    FREQUENCY_EMBEDDING_SIZE,
    MAX_PERIOD,
    ROPE_AXES_DIMS,
    ROPE_AXES_LENS,
    ROPE_THETA,
)


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=FREQUENCY_EMBEDDING_SIZE):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=MAX_PERIOD):
        with torch.amp.autocast("cuda", enabled=False):
            half = dim // 2
            freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # original implementation. kept for reference
        # output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        # return output * self.weight

        # cast to float32 for numerical stability
        x_f = x.float()
        w_f = self.weight.float()
        out = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * w_f).to(x.dtype)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)


def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


class ZImageAttention(nn.Module):
    _attention_backend = None

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, qk_norm: bool = True, eps: float = 1e-5, use_16bit: bool = False):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.use_16bit = use_16bit

        self.to_q = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.to_k = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_v = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(n_heads * self.head_dim, dim, bias=False)])

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def _forward(
        self, hidden_states: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, attn_params: Optional[AttentionParams] = None
    ) -> torch.Tensor:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = query.unflatten(-1, (self.n_heads, -1))  # [B, seq_len, n_heads, head_dim]
        key = key.unflatten(-1, (self.n_kv_heads, -1))
        value = value.unflatten(-1, (self.n_kv_heads, -1))

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)

        # query.dtype is float32. It is imcompatible with FlashAttention, so we convert to the original dtype if use_16bit is set.
        dtype = query.dtype if not self.use_16bit else value.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Call attention
        qkv = [query, key, value]
        del query, key, value
        hidden_states = attention(qkv, attn_params=attn_params)
        del qkv

        hidden_states = hidden_states.to(dtype)

        output = self.to_out[0](hidden_states)
        return output

    def forward(
        self, hidden_states: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, attn_params: Optional[AttentionParams] = None
    ) -> torch.Tensor:
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, hidden_states, freqs_cis, attn_params, use_reentrant=False)
        else:
            return self._forward(hidden_states, freqs_cis, attn_params)


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
        use_16bit: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // n_heads
        self.layer_id = layer_id
        self.modulation = modulation

        self.attention = ZImageAttention(dim, n_heads, n_kv_heads, qk_norm, norm_eps, use_16bit=use_16bit)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        if modulation:
            self.adaLN_modulation = nn.ModuleList([nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)])

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        self.feed_forward.enable_gradient_checkpointing()
        self.attention.enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.feed_forward.disable_gradient_checkpointing()
        self.attention.disable_gradient_checkpointing()

    def _forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
            del adaln_input
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis=freqs_cis, attn_params=attn_params)
            del scale_msa
            x = x + gate_msa * self.attention_norm2(attn_out)
            del gate_msa
            x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
            del scale_mlp, gate_mlp
        else:
            attn_out = self.attention(self.attention_norm1(x), freqs_cis=freqs_cis, attn_params=attn_params)
            x = x + self.attention_norm2(attn_out)
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ):
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.feed_forward.w1.weight.device)
            return checkpoint(forward_fn, x, freqs_cis, adaln_input, attn_params, use_reentrant=False)
        else:
            return self._forward(x, freqs_cis, adaln_input, attn_params)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(self, theta: float = ROPE_THETA, axes_dims: List[int] = ROPE_AXES_DIMS, axes_lens: List[int] = ROPE_AXES_LENS):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens)
        self.freqs_cis = None

    @staticmethod
    def precompute_freqs_cis(dim: List[int], end: List[int], theta: float = ROPE_THETA):
        with torch.device("cpu"):
            freqs_cis = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                freqs_cis_i = torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64)
                freqs_cis.append(freqs_cis_i)
            return freqs_cis

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.freqs_cis is None:
            # [torch.Size([1536, 16]), torch.Size([512, 24]), torch.Size([512, 24])]
            self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)  # keep on cpu

        result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            result.append(self.freqs_cis[i].to(device)[index])
        return torch.cat(result, dim=-1)


class ZImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=ROPE_THETA,
        t_scale=1000.0,
        axes_dims=ROPE_AXES_DIMS,
        axes_lens=ROPE_AXES_LENS,
        attn_mode: str = "torch",
        split_attn: bool = False,
        use_16bit_for_attention: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads
        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        assert len(all_patch_size) == len(all_f_patch_size)

        all_x_embedder = {}
        all_final_layer = {}
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder
            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)

        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=True, use_16bit=use_16bit_for_attention
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, modulation=False, use_16bit=use_16bit_for_attention
                )
                for layer_id in range(n_refiner_layers)
            ]
        )

        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm, use_16bit=use_16bit_for_attention)
                for layer_id in range(n_layers)
            ]
        )

        head_dim = dim // n_heads
        assert head_dim == sum(axes_dims)
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None

        self.offloader = None
        self.num_blocks = n_layers

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = cpu_offload

        for block in self.noise_refiner + self.context_refiner + self.layers:
            block.enable_gradient_checkpointing(activation_cpu_offloading=cpu_offload)

        print(f"Z-Image: Gradient checkpointing enabled. CPU offload: {cpu_offload}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        for block in self.noise_refiner + self.context_refiner + self.layers:
            block.disable_gradient_checkpointing()

        print("Z-Image: Gradient checkpointing disabled.")

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks

        assert self.blocks_to_swap <= self.num_blocks - 2, (
            f"Cannot swap more than {self.num_blocks - 2} double blocks. Requested {self.blocks_to_swap} double blocks."
        )

        self.offloader = ModelOffloader(
            "double", self.layers, len(self.layers), self.blocks_to_swap, supports_backward, device, use_pinned_memory
        )
        print(
            f"Z-Image: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks to device {device}. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("Z-Image: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("Z-Image: Block swap set to forward and backward.")

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

    def unpatchify(self, x: torch.Tensor, size: Tuple[int, int, int], patch_size: int, f_patch_size: int) -> torch.Tensor:
        """
        Unpatchify the latent tensor back to image/video format.

        Args:
            x: [B, seq_len, patch_dim] tensor
            size: (F, H, W) tuple of the original latent size
            patch_size: spatial patch size (pH = pW)
            f_patch_size: temporal patch size (pF)

        Returns:
            [B, C, F, H, W] tensor
        """
        pH = pW = patch_size
        pF = f_patch_size
        F_size, H_size, W_size = size
        B = x.shape[0]
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        ori_len = F_tokens * H_tokens * W_tokens

        # Take only the original image tokens (exclude caption part if any)
        x = x[:, :ori_len]  # [B, ori_len, patch_dim]

        x = x.view(B, F_tokens, H_tokens, W_tokens, pF, pH, pW, self.out_channels)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)  # [B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW]
        x = x.reshape(B, self.out_channels, F_size, H_size, W_size)
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)
        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify(self, x: torch.Tensor, patch_size: int, f_patch_size: int) -> torch.Tensor:
        """
        Patchify the latent tensor.

        Args:
            x: [B, C, F, H, W] tensor
            patch_size: spatial patch size (pH = pW)
            f_patch_size: temporal patch size (pF)

        Returns:
            [B, seq_len, patch_dim] tensor where seq_len = (F/pF) * (H/pH) * (W/pW)
            and patch_dim = pF * pH * pW * C
        """
        pH = pW = patch_size
        pF = f_patch_size
        B, C, F_size, H_size, W_size = x.shape
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW

        x = x.view(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1)  # [B, F_tokens, H_tokens, W_tokens, pF, pH, pW, C]
        x = x.reshape(B, F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return x

    def create_image_position_ids(
        self, F_tokens: int, H_tokens: int, W_tokens: int, cap_seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create position IDs for image patches.

        Args:
            F_tokens: number of frame tokens
            H_tokens: number of height tokens
            W_tokens: number of width tokens
            cap_seq_len: caption sequence length (for offset)
            device: device to create tensor on

        Returns:
            [seq_len, 3] tensor of position IDs
        """
        # Image positions start after caption positions
        # Position format: (cap_seq_len + 1 + f_idx, h_idx, w_idx). [F_tokens * H_tokens * W_tokens, 3]
        return self.create_coordinate_grid(
            size=(F_tokens, H_tokens, W_tokens), start=(cap_seq_len + 1, 0, 0), device=device
        ).flatten(0, 2)

    def create_caption_position_ids(self, cap_seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create position IDs for caption tokens.

        Args:
            cap_seq_len: caption sequence length
            device: device to create tensor on

        Returns:
            [cap_seq_len, 3] tensor of position IDs
        """
        # Caption positions: (i + 1, 0, 0) for i in range(cap_seq_len). [cap_seq_len, 3]
        return self.create_coordinate_grid(size=(cap_seq_len, 1, 1), start=(1, 0, 0), device=device).flatten(0, 2)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        patch_size: int = 2,
        f_patch_size: int = 1,
    ) -> torch.Tensor:
        """
        Forward pass of the Z-Image Transformer.

        Args:
            x: Latent tensor [B, C, F, H, W]
            t: Timestep tensor [B]
            cap_feats: Caption features [B, cap_seq_len, cap_feat_dim]
            cap_mask: Caption mask [B, cap_seq_len], True for valid tokens
            patch_size: Spatial patch size (default: 2)
            f_patch_size: Temporal patch size (default: 1)

        Returns:
            Output tensor [B, C, F, H, W]
        """
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        B, C, F_size, H_size, W_size = x.shape
        device = x.device
        cap_seq_len = cap_feats.shape[1]

        # Timestep embedding
        t = t * self.t_scale  # 0-1 to 0-1000
        adaln_input = self.t_embedder(t)

        # Patchify and embed x
        pH = pW = patch_size
        pF = f_patch_size
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        x_seq_len = F_tokens * H_tokens * W_tokens

        x = self.patchify(x, patch_size, f_patch_size)  # [B, x_seq_len, patch_dim]
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)  # [B, x_seq_len, dim]

        adaln_input = adaln_input.type_as(x)

        # Create position IDs and RoPE for x (same for all samples since images have same size)
        x_pos_ids = self.create_image_position_ids(F_tokens, H_tokens, W_tokens, cap_seq_len, device)
        x_freqs_cis = self.rope_embedder(x_pos_ids)  # [x_seq_len, head_dim]
        del x_pos_ids
        x_freqs_cis = x_freqs_cis.unsqueeze(0).expand(B, -1, -1)  # [B, x_seq_len, head_dim]

        # Apply noise refiner
        noise_refiner_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, None)
        for layer in self.noise_refiner:
            x = layer(x, x_freqs_cis, adaln_input, attn_params=noise_refiner_attn_params)

        # Embed caption features
        cap_feats = self.cap_embedder(cap_feats)  # [B, cap_seq_len, dim]

        # Apply cap_pad_token to masked positions
        if cap_mask is not None:
            cap_pad_mask = ~cap_mask  # True for padding positions
            cap_feats = cap_feats.masked_fill(cap_pad_mask.unsqueeze(-1), 0.0)
            cap_feats = cap_feats + self.cap_pad_token * cap_pad_mask.unsqueeze(-1).float()

        # Create position IDs and RoPE for captions
        cap_pos_ids = self.create_caption_position_ids(cap_seq_len, device)
        cap_freqs_cis = self.rope_embedder(cap_pos_ids)  # [cap_seq_len, head_dim]
        del cap_pos_ids
        cap_freqs_cis = cap_freqs_cis.unsqueeze(0).expand(B, -1, -1)  # [B, cap_seq_len, head_dim]

        # Apply context refiner
        context_refiner_attn_params = AttentionParams.create_attention_params_from_mask(
            self.attn_mode, self.split_attn, 0, cap_mask
        )
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_freqs_cis, attn_params=context_refiner_attn_params)

        # Concatenate x and cap_feats for unified processing
        # Order: [x tokens, caption tokens]
        unified = torch.cat([x, cap_feats], dim=1)  # [B, x_seq_len + cap_seq_len, dim]
        del x, cap_feats
        unified_freqs_cis = torch.cat([x_freqs_cis, cap_freqs_cis], dim=1)  # [B, x_seq_len + cap_seq_len, head_dim]
        del x_freqs_cis, cap_freqs_cis

        # Apply main transformer layers
        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, x_seq_len, cap_mask)
        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)

            unified = layer(unified, unified_freqs_cis, adaln_input, attn_params=attn_params)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.layers, index)

        unified = unified.to(device)  # ensure unified is on the correct device when activation CPU offloading is used

        # Apply final layer
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)

        # Unpatchify (takes only the first x_seq_len tokens)
        x = self.unpatchify(unified, (F_size, H_size, W_size), patch_size, f_patch_size)

        return x


FP8_OPTIMIZATION_TARGET_KEYS = [".layers."]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_embedder", "final_layer"]


def create_model(
    attn_mode: str, split_attn: bool, dtype: Optional[torch.dtype], use_16bit_for_attention: bool = False
) -> ZImageTransformer2DModel:
    with init_empty_weights():
        logger.info("Creating ZImageTransformer2DModel")
        model = ZImageTransformer2DModel(
            all_patch_size=tuple(zimage_config.DEFAULT_TRANSFORMER_PATCH_SIZE),
            all_f_patch_size=tuple(zimage_config.DEFAULT_TRANSFORMER_F_PATCH_SIZE),
            in_channels=zimage_config.DEFAULT_TRANSFORMER_IN_CHANNELS,
            dim=zimage_config.DEFAULT_TRANSFORMER_DIM,
            n_layers=zimage_config.DEFAULT_TRANSFORMER_N_LAYERS,
            n_refiner_layers=zimage_config.DEFAULT_TRANSFORMER_N_REFINER_LAYERS,
            n_heads=zimage_config.DEFAULT_TRANSFORMER_N_HEADS,
            n_kv_heads=zimage_config.DEFAULT_TRANSFORMER_N_KV_HEADS,
            norm_eps=zimage_config.DEFAULT_TRANSFORMER_NORM_EPS,
            qk_norm=zimage_config.DEFAULT_TRANSFORMER_QK_NORM,
            cap_feat_dim=zimage_config.DEFAULT_TRANSFORMER_CAP_FEAT_DIM,
            rope_theta=zimage_config.ROPE_THETA,
            t_scale=zimage_config.DEFAULT_TRANSFORMER_T_SCALE,
            axes_dims=zimage_config.ROPE_AXES_DIMS,
            axes_lens=zimage_config.ROPE_AXES_LENS,
            attn_mode=attn_mode,
            split_attn=split_attn,
            use_16bit_for_attention=use_16bit_for_attention,
        )
        if dtype is not None:
            model.to(dtype)
    return model


def load_zimage_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[List[float]] = None,
    disable_numpy_memmap: bool = False,
    use_16bit_for_attention: bool = False,
) -> ZImageTransformer2DModel:
    """
    Load a Z-Image model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
        disable_numpy_memmap (bool): Whether to disable numpy memory mapping when loading weights.
        use_16bit_for_attention (bool): Whether to use 16-bit precision for attention computations.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    model = create_model(attn_mode, split_attn, dit_weight_dtype, use_16bit_for_attention=use_16bit_for_attention)

    replace_keys = {
        "all_final_layer.2-1.linear": "final_layer.linear",
        "all_final_layer.2-1.adaLN_modulation": "final_layer.adaLN_modulation",
        "all_x_embedder.2-1.bias": "x_embedder.bias",
        "all_x_embedder.2-1.weight": "x_embedder.weight",
        ".attention.to_out.0.bias": ".attention.out.bias",
        ".attention.norm_k.weight": ".attention.k_norm.weight",
        ".attention.norm_q.weight": ".attention.q_norm.weight",
        ".attention.to_out.0.weight": ".attention.out.weight",
    }
    replace_keys_reverse = {v: k for k, v in replace_keys.items()}

    def comfyui_z_image_weight_split_hook(
        key: str, value: Optional[torch.Tensor]
    ) -> Tuple[Optional[list[str]], Optional[list[torch.Tensor]]]:
        # Use split hook for key conversion
        for k, v in replace_keys_reverse.items():
            if k in key:
                key = key.replace(k, v)
                return [key], [value] if value is not None else None

        # convert to separate Q, K, V weights/biases
        if "attention.qkv.weight" in key:
            new_keys = [key.replace("qkv", module) for module in ["to_q", "to_k", "to_v"]]
            return new_keys, list(torch.chunk(value, 3, dim=0)) if value is not None else None

        return None, None

    hooks = WeightTransformHooks(split_hook=comfyui_z_image_weight_split_hook)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=hooks,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    return model
