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


def select_per_token(
    value_noisy: torch.Tensor,
    value_clean: torch.Tensor,
    noise_mask: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    noise_mask_expanded = noise_mask.unsqueeze(-1)
    return torch.where(
        noise_mask_expanded == 1,
        value_noisy.unsqueeze(1).expand(-1, seq_len, -1),
        value_clean.unsqueeze(1).expand(-1, seq_len, -1),
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


def clamp_fp16(x):
    if x.dtype == torch.float16:
        return torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
    return x


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

    def _forward(self, x, apply_fp16_downscale=False):
        x3 = self.w3(x)
        if x.dtype == torch.float16 and apply_fp16_downscale:
            x3.div_(32)
        return self.w2(clamp_fp16(F.silu(self.w1(x)) * x3))

    def forward(self, x, apply_fp16_downscale=False):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x, apply_fp16_downscale)


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
        if output.dtype == torch.float16:
            output.div_(4)
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
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ):
        if self.modulation:
            seq_len = x.shape[1]
            if noise_mask is not None:
                assert adaln_noisy is not None and adaln_clean is not None
                mod_noisy = self.adaLN_modulation[0](adaln_noisy)
                mod_clean = self.adaLN_modulation[0](adaln_clean)

                scale_msa_noisy, gate_msa_noisy, scale_mlp_noisy, gate_mlp_noisy = mod_noisy.chunk(4, dim=1)
                scale_msa_clean, gate_msa_clean, scale_mlp_clean, gate_mlp_clean = mod_clean.chunk(4, dim=1)

                gate_msa_noisy, gate_mlp_noisy = gate_msa_noisy.tanh(), gate_mlp_noisy.tanh()
                gate_msa_clean, gate_mlp_clean = gate_msa_clean.tanh(), gate_mlp_clean.tanh()

                scale_msa_noisy, scale_mlp_noisy = 1.0 + scale_msa_noisy, 1.0 + scale_mlp_noisy
                scale_msa_clean, scale_mlp_clean = 1.0 + scale_msa_clean, 1.0 + scale_mlp_clean

                scale_msa = select_per_token(scale_msa_noisy, scale_msa_clean, noise_mask, seq_len)
                scale_mlp = select_per_token(scale_mlp_noisy, scale_mlp_clean, noise_mask, seq_len)
                gate_msa = select_per_token(gate_msa_noisy, gate_msa_clean, noise_mask, seq_len)
                gate_mlp = select_per_token(gate_mlp_noisy, gate_mlp_clean, noise_mask, seq_len)
            else:
                assert adaln_input is not None
                scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation[0](adaln_input).unsqueeze(1).chunk(4, dim=2)
                del adaln_input
                gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
                scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis=freqs_cis, attn_params=attn_params)
            del scale_msa
            x = x + gate_msa * self.attention_norm2(clamp_fp16(attn_out))
            del gate_msa
            x = x + gate_mlp * self.ffn_norm2(
                clamp_fp16(self.feed_forward(self.ffn_norm1(x) * scale_mlp, apply_fp16_downscale=True))
            )
            del scale_mlp, gate_mlp
        else:
            attn_out = self.attention(self.attention_norm1(x), freqs_cis=freqs_cis, attn_params=attn_params)
            x = x + self.attention_norm2(clamp_fp16(attn_out))
            x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

        return x

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        adaln_input: Optional[torch.Tensor] = None,
        noise_mask: Optional[torch.Tensor] = None,
        adaln_noisy: Optional[torch.Tensor] = None,
        adaln_clean: Optional[torch.Tensor] = None,
        attn_params: Optional[AttentionParams] = None,
    ):
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.feed_forward.w1.weight.device)
            return checkpoint(
                forward_fn,
                x,
                freqs_cis,
                adaln_input,
                noise_mask,
                adaln_noisy,
                adaln_clean,
                attn_params,
                use_reentrant=False,
            )
        else:
            return self._forward(x, freqs_cis, adaln_input, noise_mask, adaln_noisy, adaln_clean, attn_params)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c=None, noise_mask=None, c_noisy=None, c_clean=None):
        seq_len = x.shape[1]
        if noise_mask is not None:
            assert c_noisy is not None and c_clean is not None
            scale_noisy = 1.0 + self.adaLN_modulation(c_noisy)
            scale_clean = 1.0 + self.adaLN_modulation(c_clean)
            scale = select_per_token(scale_noisy, scale_clean, noise_mask, seq_len)
        else:
            assert c is not None
            scale = 1.0 + self.adaLN_modulation(c)
            scale = scale.unsqueeze(1)

        x = self.norm_final(x) * scale
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
        siglip_feat_dim: Optional[int] = None,
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

        if siglip_feat_dim is not None:
            self.siglip_embedder = nn.Sequential(
                RMSNorm(siglip_feat_dim, eps=norm_eps),
                nn.Linear(siglip_feat_dim, dim, bias=True),
            )
            self.siglip_refiner = nn.ModuleList(
                [
                    ZImageTransformerBlock(
                        2000 + layer_id,
                        dim,
                        n_heads,
                        n_kv_heads,
                        norm_eps,
                        qk_norm,
                        modulation=False,
                        use_16bit=use_16bit_for_attention,
                    )
                    for layer_id in range(n_refiner_layers)
                ]
            )
            self.siglip_pad_token = nn.Parameter(torch.empty((1, dim)))
        else:
            self.siglip_embedder = None
            self.siglip_refiner = None
            self.siglip_pad_token = None

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

    def unpatchify_omni(
        self,
        x: torch.Tensor,
        size: List[List[Optional[Tuple[int, int, int]]]],
        patch_size: int,
        f_patch_size: int,
        x_pos_offsets: List[Tuple[int, int]],
    ) -> torch.Tensor:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = x.shape[0]
        result = []
        for i in range(bsz):
            unified_x = x[i, x_pos_offsets[i][0] : x_pos_offsets[i][1]]
            cu_len = 0
            x_item = None
            for j in range(len(size[i])):
                if size[i][j] is None:
                    cu_len += zimage_config.SEQ_MULTI_OF
                    continue

                F_size, H_size, W_size = size[i][j]
                F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
                ori_len = F_tokens * H_tokens * W_tokens
                pad_len = (-ori_len) % zimage_config.SEQ_MULTI_OF

                x_item = (
                    unified_x[cu_len : cu_len + ori_len]
                    .view(F_tokens, H_tokens, W_tokens, pF, pH, pW, self.out_channels)
                    .permute(6, 0, 3, 1, 4, 2, 5)
                    .reshape(self.out_channels, F_size, H_size, W_size)
                )
                cu_len += ori_len + pad_len

            assert x_item is not None
            result.append(x_item)

        return torch.stack(result, dim=0)

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

    def _patchify_image(self, image: torch.Tensor, patch_size: int, f_patch_size: int):
        pH = pW = patch_size
        pF = f_patch_size
        C, F_size, H_size, W_size = image.size()
        F_tokens, H_tokens, W_tokens = F_size // pF, H_size // pH, W_size // pW
        image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        return image, (F_size, H_size, W_size), (F_tokens, H_tokens, W_tokens)

    def _pad_with_ids(
        self,
        feat: torch.Tensor,
        pos_grid_size: Tuple[int, int, int],
        pos_start: Tuple[int, int, int],
        device: torch.device,
        noise_mask_val: Optional[int] = None,
    ):
        ori_len = len(feat)
        pad_len = (-ori_len) % zimage_config.SEQ_MULTI_OF
        total_len = ori_len + pad_len

        ori_pos_ids = self.create_coordinate_grid(size=pos_grid_size, start=pos_start, device=device).flatten(0, 2)
        if pad_len > 0:
            pad_pos_ids = (
                self.create_coordinate_grid(size=(1, 1, 1), start=(0, 0, 0), device=device).flatten(0, 2).repeat(pad_len, 1)
            )
            pos_ids = torch.cat([ori_pos_ids, pad_pos_ids], dim=0)
            padded_feat = torch.cat([feat, feat[-1:].repeat(pad_len, 1)], dim=0)
            pad_mask = torch.cat(
                [
                    torch.zeros(ori_len, dtype=torch.bool, device=device),
                    torch.ones(pad_len, dtype=torch.bool, device=device),
                ]
            )
        else:
            pos_ids = ori_pos_ids
            padded_feat = feat
            pad_mask = torch.zeros(ori_len, dtype=torch.bool, device=device)

        noise_mask = [noise_mask_val] * total_len if noise_mask_val is not None else None
        return padded_feat, pos_ids, pad_mask, total_len, noise_mask

    def patchify_and_embed_omni(
        self,
        all_x: List[List[Optional[torch.Tensor]]],
        all_cap_feats: List[List[torch.Tensor]],
        all_siglip_feats: Optional[List[Optional[List[Optional[torch.Tensor]]]]],
        patch_size: int,
        f_patch_size: int,
        images_noise_mask: Optional[List[List[int]]],
    ):
        bsz = len(all_x)
        device = all_x[0][-1].device if all_x[0][-1] is not None else all_x[0][0].device
        dtype = all_x[0][-1].dtype if all_x[0][-1] is not None else all_x[0][0].dtype
        patch_dim = patch_size * patch_size * f_patch_size * self.in_channels
        siglip_feat_dim = None
        if self.siglip_embedder is not None:
            siglip_feat_dim = int(self.siglip_embedder[0].weight.shape[0])

        if images_noise_mask is None:
            images_noise_mask = [[1] * len(all_x[i]) for i in range(bsz)]

        all_x_out, all_x_size, all_x_pos_ids, all_x_pad_mask, all_x_len, all_x_noise_mask = [], [], [], [], [], []
        all_cap_out, all_cap_pos_ids, all_cap_pad_mask, all_cap_len, all_cap_noise_mask = [], [], [], [], []
        all_sig_out, all_sig_pos_ids, all_sig_pad_mask, all_sig_len, all_sig_noise_mask = [], [], [], [], []

        for i in range(bsz):
            num_images = len(all_x[i])

            cap_feats_list, cap_pos_list, cap_mask_list, cap_lens, cap_noise = [], [], [], [], []
            cap_end_pos = []
            cap_cu_len = 1
            for j, cap_item in enumerate(all_cap_feats[i]):
                noise_val = images_noise_mask[i][j] if j < len(images_noise_mask[i]) else 1
                cap_out, cap_pos, cap_mask, cap_len, cap_nm = self._pad_with_ids(
                    cap_item,
                    (len(cap_item) + (-len(cap_item)) % zimage_config.SEQ_MULTI_OF, 1, 1),
                    (cap_cu_len, 0, 0),
                    device,
                    noise_val,
                )
                cap_feats_list.append(cap_out)
                cap_pos_list.append(cap_pos)
                cap_mask_list.append(cap_mask)
                cap_lens.append(cap_len)
                cap_noise.extend(cap_nm)
                cap_cu_len += len(cap_item)
                cap_end_pos.append(cap_cu_len)
                cap_cu_len += 2

            all_cap_out.append(torch.cat(cap_feats_list, dim=0))
            all_cap_pos_ids.append(torch.cat(cap_pos_list, dim=0))
            all_cap_pad_mask.append(torch.cat(cap_mask_list, dim=0))
            all_cap_len.append(cap_lens)
            all_cap_noise_mask.append(cap_noise)

            x_feats_list, x_pos_list, x_mask_list, x_lens, x_sizes, x_noise = [], [], [], [], [], []
            for j, x_item in enumerate(all_x[i]):
                noise_val = images_noise_mask[i][j]
                if x_item is not None:
                    x_patches, size, (F_t, H_t, W_t) = self._patchify_image(x_item, patch_size, f_patch_size)
                    x_out, x_pos, x_mask, x_len, x_nm = self._pad_with_ids(
                        x_patches, (F_t, H_t, W_t), (cap_end_pos[j], 0, 0), device, noise_val
                    )
                    x_sizes.append(size)
                else:
                    x_len = zimage_config.SEQ_MULTI_OF
                    x_out = torch.zeros((x_len, patch_dim), dtype=dtype, device=device)
                    x_pos = self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(x_len, 1)
                    x_mask = torch.ones(x_len, dtype=torch.bool, device=device)
                    x_nm = [noise_val] * x_len
                    x_sizes.append(None)

                x_feats_list.append(x_out)
                x_pos_list.append(x_pos)
                x_mask_list.append(x_mask)
                x_lens.append(x_len)
                x_noise.extend(x_nm)

            all_x_out.append(torch.cat(x_feats_list, dim=0))
            all_x_pos_ids.append(torch.cat(x_pos_list, dim=0))
            all_x_pad_mask.append(torch.cat(x_mask_list, dim=0))
            all_x_size.append(x_sizes)
            all_x_len.append(x_lens)
            all_x_noise_mask.append(x_noise)

            if all_siglip_feats is None or all_siglip_feats[i] is None:
                all_sig_out.append(None)
                all_sig_pos_ids.append(None)
                all_sig_pad_mask.append(None)
                all_sig_len.append([0] * num_images)
                all_sig_noise_mask.append(None)
            else:
                sig_feats_list, sig_pos_list, sig_mask_list, sig_lens, sig_noise = [], [], [], [], []
                for j, sig_item in enumerate(all_siglip_feats[i]):
                    noise_val = images_noise_mask[i][j]
                    if sig_item is not None:
                        sig_H, sig_W, sig_C = sig_item.size()
                        sig_flat = sig_item.permute(2, 0, 1).reshape(sig_H * sig_W, sig_C)
                        sig_out, sig_pos, sig_mask, sig_len, sig_nm = self._pad_with_ids(
                            sig_flat,
                            (1, sig_H, sig_W),
                            (cap_end_pos[j] + 1, 0, 0),
                            device,
                            noise_val,
                        )
                        if x_sizes[j] is not None:
                            sig_pos = sig_pos.float()
                            sig_pos[..., 1] = sig_pos[..., 1] / max(sig_H - 1, 1) * (x_sizes[j][1] - 1)
                            sig_pos[..., 2] = sig_pos[..., 2] / max(sig_W - 1, 1) * (x_sizes[j][2] - 1)
                            sig_pos = sig_pos.to(torch.int32)
                    else:
                        sig_len = zimage_config.SEQ_MULTI_OF
                        assert siglip_feat_dim is not None
                        sig_out = torch.zeros((sig_len, siglip_feat_dim), dtype=dtype, device=device)
                        sig_pos = self.create_coordinate_grid((1, 1, 1), (0, 0, 0), device).flatten(0, 2).repeat(sig_len, 1)
                        sig_mask = torch.ones(sig_len, dtype=torch.bool, device=device)
                        sig_nm = [noise_val] * sig_len

                    sig_feats_list.append(sig_out)
                    sig_pos_list.append(sig_pos)
                    sig_mask_list.append(sig_mask)
                    sig_lens.append(sig_len)
                    sig_noise.extend(sig_nm)

                all_sig_out.append(torch.cat(sig_feats_list, dim=0))
                all_sig_pos_ids.append(torch.cat(sig_pos_list, dim=0))
                all_sig_pad_mask.append(torch.cat(sig_mask_list, dim=0))
                all_sig_len.append(sig_lens)
                all_sig_noise_mask.append(sig_noise)

        all_x_pos_offsets = [(sum(all_cap_len[i]), sum(all_cap_len[i]) + sum(all_x_len[i])) for i in range(bsz)]

        return (
            all_x_out,
            all_cap_out,
            all_sig_out,
            all_x_size,
            all_x_pos_ids,
            all_cap_pos_ids,
            all_sig_pos_ids,
            all_x_pad_mask,
            all_cap_pad_mask,
            all_sig_pad_mask,
            all_x_pos_offsets,
            all_x_noise_mask,
            all_cap_noise_mask,
            all_sig_noise_mask,
        )

    def _prepare_sequence(
        self,
        feats: List[torch.Tensor],
        pos_ids: List[torch.Tensor],
        inner_pad_mask: List[torch.Tensor],
        pad_token: torch.nn.Parameter,
        noise_mask: Optional[List[List[int]]],
        device: torch.device,
    ):
        item_seqlens = [len(f) for f in feats]
        max_seqlen = max(item_seqlens)
        bsz = len(feats)

        feats_cat = torch.cat(feats, dim=0)
        pad_mask_cat = torch.cat(inner_pad_mask)
        if pad_mask_cat.any():
            token = pad_token.squeeze(0) if pad_token.ndim == 2 and pad_token.shape[0] == 1 else pad_token
            feats_cat[pad_mask_cat] = token.to(dtype=feats_cat.dtype).unsqueeze(0).expand(int(pad_mask_cat.sum().item()), -1)
        feats = list(feats_cat.split(item_seqlens, dim=0))
        freqs_cis = list(self.rope_embedder(torch.cat(pos_ids, dim=0)).split([len(p) for p in pos_ids], dim=0))

        feats = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True, padding_value=0.0)
        freqs_cis = torch.nn.utils.rnn.pad_sequence(freqs_cis, batch_first=True, padding_value=0.0)[:, : feats.shape[1]]

        attn_params = AttentionParams.create_attention_params(self.attn_mode, True)
        attn_params.seqlens = torch.tensor(item_seqlens, device=device, dtype=torch.int32)
        attn_params.max_seqlen = max_seqlen

        noise_mask_tensor = None
        if noise_mask is not None:
            noise_mask_tensor = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(m, dtype=torch.long, device=device) for m in noise_mask],
                batch_first=True,
                padding_value=0,
            )[:, : feats.shape[1]]

        return feats, freqs_cis, attn_params, noise_mask_tensor

    def _build_unified_sequence(
        self,
        x: torch.Tensor,
        x_freqs: torch.Tensor,
        x_seqlens: List[int],
        x_noise_mask: Optional[List[List[int]]],
        cap: torch.Tensor,
        cap_freqs: torch.Tensor,
        cap_seqlens: List[int],
        cap_noise_mask: Optional[List[List[int]]],
        siglip: Optional[torch.Tensor],
        siglip_freqs: Optional[torch.Tensor],
        siglip_seqlens: Optional[List[int]],
        siglip_noise_mask: Optional[List[List[int]]],
        omni_mode: bool,
        device: torch.device,
    ):
        bsz = len(x_seqlens)
        unified = []
        unified_freqs = []
        unified_noise_mask = []

        for i in range(bsz):
            x_len, cap_len = x_seqlens[i], cap_seqlens[i]
            if omni_mode:
                if siglip is not None and siglip_seqlens is not None:
                    sig_len = siglip_seqlens[i]
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len], siglip[i][:sig_len]]))
                    unified_freqs.append(torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len], siglip_freqs[i][:sig_len]]))
                    unified_noise_mask.append(
                        torch.tensor(cap_noise_mask[i] + x_noise_mask[i] + siglip_noise_mask[i], dtype=torch.long, device=device)
                    )
                else:
                    unified.append(torch.cat([cap[i][:cap_len], x[i][:x_len]]))
                    unified_freqs.append(torch.cat([cap_freqs[i][:cap_len], x_freqs[i][:x_len]]))
                    unified_noise_mask.append(torch.tensor(cap_noise_mask[i] + x_noise_mask[i], dtype=torch.long, device=device))
            else:
                unified.append(torch.cat([x[i][:x_len], cap[i][:cap_len]]))
                unified_freqs.append(torch.cat([x_freqs[i][:x_len], cap_freqs[i][:cap_len]]))

        if omni_mode:
            if siglip is not None and siglip_seqlens is not None:
                unified_seqlens = [a + b + c for a, b, c in zip(cap_seqlens, x_seqlens, siglip_seqlens)]
            else:
                unified_seqlens = [a + b for a, b in zip(cap_seqlens, x_seqlens)]
        else:
            unified_seqlens = [a + b for a, b in zip(x_seqlens, cap_seqlens)]

        max_seqlen = max(unified_seqlens)

        unified = torch.nn.utils.rnn.pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = torch.nn.utils.rnn.pad_sequence(unified_freqs, batch_first=True, padding_value=0.0)

        attn_params = AttentionParams.create_attention_params(self.attn_mode, True)
        attn_params.seqlens = torch.tensor(unified_seqlens, device=device, dtype=torch.int32)
        attn_params.max_seqlen = max_seqlen

        noise_mask_tensor = None
        if omni_mode:
            noise_mask_tensor = torch.nn.utils.rnn.pad_sequence(unified_noise_mask, batch_first=True, padding_value=0)[
                :, : unified.shape[1]
            ]

        return unified, unified_freqs, attn_params, noise_mask_tensor

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
        x: Union[torch.Tensor, List[List[Optional[torch.Tensor]]]],
        t: torch.Tensor,
        cap_feats: Union[torch.Tensor, List[List[torch.Tensor]]],
        cap_mask: Optional[torch.Tensor],
        patch_size: int = 2,
        f_patch_size: int = 1,
        siglip_feats: Optional[List[Optional[List[Optional[torch.Tensor]]]]] = None,
        image_noise_mask: Optional[List[List[int]]] = None,
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

        omni_mode = isinstance(x, list)
        if omni_mode:
            device = x[0][-1].device if x[0][-1] is not None else x[0][0].device
            t_noisy = self.t_embedder(t * self.t_scale).type_as(x[0][-1] if x[0][-1] is not None else x[0][0])
            t_clean = self.t_embedder(torch.ones_like(t) * self.t_scale).type_as(x[0][-1] if x[0][-1] is not None else x[0][0])
            adaln_input = None

            (
                x_list,
                cap_list,
                sig_list,
                x_size,
                x_pos_ids,
                cap_pos_ids,
                sig_pos_ids,
                x_pad_mask,
                cap_pad_mask,
                sig_pad_mask,
                x_pos_offsets,
                x_noise_mask,
                cap_noise_mask,
                sig_noise_mask,
            ) = self.patchify_and_embed_omni(x, cap_feats, siglip_feats, patch_size, f_patch_size, image_noise_mask)

            x_seqlens = [len(xi) for xi in x_list]
            x_emb = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](torch.cat(x_list, dim=0))
            x_emb = list(x_emb.split(x_seqlens, dim=0))
            x_emb, x_freqs, x_attn_params, x_noise_tensor = self._prepare_sequence(
                x_emb, x_pos_ids, x_pad_mask, self.x_pad_token, x_noise_mask, device
            )

            for layer in self.noise_refiner:
                x_emb = layer(
                    x_emb,
                    x_freqs,
                    adaln_input,
                    noise_mask=x_noise_tensor,
                    adaln_noisy=t_noisy,
                    adaln_clean=t_clean,
                    attn_params=x_attn_params,
                )

            cap_seqlens = [len(ci) for ci in cap_list]
            cap_emb = self.cap_embedder(torch.cat(cap_list, dim=0))
            cap_emb = list(cap_emb.split(cap_seqlens, dim=0))
            cap_emb, cap_freqs, cap_attn_params, _ = self._prepare_sequence(
                cap_emb, cap_pos_ids, cap_pad_mask, self.cap_pad_token, None, device
            )
            for layer in self.context_refiner:
                cap_emb = layer(cap_emb, cap_freqs, attn_params=cap_attn_params)

            sig_seqlens = sig_freqs = sig_emb = sig_attn_params = None
            if sig_list is not None and sig_list[0] is not None and self.siglip_embedder is not None:
                sig_seqlens = [len(si) for si in sig_list]
                sig_emb = self.siglip_embedder(torch.cat(sig_list, dim=0))
                sig_emb = list(sig_emb.split(sig_seqlens, dim=0))
                sig_emb, sig_freqs, sig_attn_params, _ = self._prepare_sequence(
                    sig_emb, sig_pos_ids, sig_pad_mask, self.siglip_pad_token, None, device
                )
                for layer in self.siglip_refiner:
                    sig_emb = layer(sig_emb, sig_freqs, attn_params=sig_attn_params)

            unified, unified_freqs, unified_attn_params, unified_noise_tensor = self._build_unified_sequence(
                x_emb,
                x_freqs,
                x_attn_params.seqlens.cpu().tolist(),
                x_noise_mask,
                cap_emb,
                cap_freqs,
                cap_attn_params.seqlens.cpu().tolist(),
                cap_noise_mask,
                sig_emb,
                sig_freqs,
                sig_seqlens,
                sig_noise_mask,
                True,
                device,
            )

            for index, layer in enumerate(self.layers):
                if self.blocks_to_swap:
                    self.offloader.wait_for_block(index)
                unified = layer(
                    unified,
                    unified_freqs,
                    adaln_input,
                    noise_mask=unified_noise_tensor,
                    adaln_noisy=t_noisy,
                    adaln_clean=t_clean,
                    attn_params=unified_attn_params,
                )
                if self.blocks_to_swap:
                    self.offloader.submit_move_blocks_forward(self.layers, index)

            unified = unified.to(device)
            unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
                unified, noise_mask=unified_noise_tensor, c_noisy=t_noisy, c_clean=t_clean
            )
            x_out = self.unpatchify_omni(unified, x_size, patch_size, f_patch_size, x_pos_offsets)
            return x_out

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
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, c=adaln_input)

        # Unpatchify (takes only the first x_seq_len tokens)
        x = self.unpatchify(unified, (F_size, H_size, W_size), patch_size, f_patch_size)

        return x


FP8_OPTIMIZATION_TARGET_KEYS = ["layers.", "noise_refiner.", "context_refiner.", "siglip_refiner."]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_modulation", ".norm_", "_norm"]


def create_model(
    attn_mode: str,
    split_attn: bool,
    dtype: Optional[torch.dtype],
    siglip_feat_dim: Optional[int] = None,
    use_default_siglip_feat_dim: bool = True,
    use_16bit_for_attention: bool = False,
) -> ZImageTransformer2DModel:
    if use_default_siglip_feat_dim and siglip_feat_dim is None:
        siglip_feat_dim = getattr(zimage_config, "DEFAULT_TRANSFORMER_SIGLIP_FEAT_DIM", None)
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
            siglip_feat_dim=siglip_feat_dim,
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

    siglip_feat_dim = None
    if "siglip_embedder.1.weight" in sd and sd["siglip_embedder.1.weight"] is not None:
        w = sd["siglip_embedder.1.weight"]
        if w.ndim == 2:
            siglip_feat_dim = int(w.shape[1])
    elif "siglip_embedder.0.weight" in sd and sd["siglip_embedder.0.weight"] is not None:
        w = sd["siglip_embedder.0.weight"]
        if w.ndim == 2:
            siglip_feat_dim = int(w.shape[1])
    else:
        for k, v in sd.items():
            if "siglip_embedder" in k and k.endswith(".weight") and v is not None and getattr(v, "ndim", 0) == 2:
                siglip_feat_dim = int(v.shape[1])
                break

    model = create_model(
        attn_mode,
        split_attn,
        dit_weight_dtype,
        siglip_feat_dim=siglip_feat_dim,
        use_default_siglip_feat_dim=False,
        use_16bit_for_attention=use_16bit_for_attention,
    )

    # TODO cast weights to mixed precision dtype when fp8_scaled is True, and original weights are in fp32

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
