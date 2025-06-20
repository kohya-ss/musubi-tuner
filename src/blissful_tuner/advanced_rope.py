#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 19:25:53 2025
Advanced rope functions for Blissful Tuner extension
License: Apache 2.0

@author: blyss
"""
import torch
import torch.nn as nn
from einops import rearrange
from typing import List
from blissful_tuner.hvw_posemb_layers import get_nd_rotary_pos_embed


# From ComfyUI
def apply_rope_comfy(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


# From WanVideoWrapper
def rope_riflex(pos, dim, theta, L_test, k, temporal):
    assert dim % 2 == 0
    device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta**scale)
    # RIFLEX modification - adjust last frequency component if L_test and k are provided
    if temporal and k > 0 and L_test:
        omega[k - 1] = 0.9 * 2 * torch.pi / L_test
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.to(dtype=torch.float32, device=pos.device)


class EmbedND_RifleX(nn.Module):
    def __init__(self: nn.Module, dim: int, theta: float, axes_dim: List[int], num_frames: int, k: int):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim
        self.num_frames = num_frames
        self.k = k

    def forward(self, ids):
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope_riflex(ids[..., i], self.axes_dim[i], self.theta, self.num_frames, self.k, temporal=True if i == 0 else False) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)


# Modified from HunyuanVideo Wrapper
def get_rotary_pos_embed_riflex(vae_ver, transformer, latent_video_length, height, width, k=0):
    if "884" in vae_ver:
        latents_size = [(latent_video_length - 1) // 4 + 1, height // 8, width // 8]
    elif "888" in vae_ver:
        latents_size = [(latent_video_length - 1) // 8 + 1, height // 8, width // 8]
    else:
        latents_size = [latent_video_length, height // 8, width // 8]

    target_ndim = 3
    ndim = 5 - 2
    rope_theta = 256  # 225
    patch_size = transformer.patch_size
    rope_dim_list = transformer.rope_dim_list
    hidden_size = transformer.hidden_size
    heads_num = transformer.heads_num
    head_dim = hidden_size // heads_num

    if isinstance(patch_size, int):
        assert all(s % patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [s // patch_size for s in latents_size]
    elif isinstance(patch_size, list):
        assert all(
            s % patch_size[idx] == 0
            for idx, s in enumerate(latents_size)
        ), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
            f"but got {latents_size}."
        )
        rope_sizes = [
            s // patch_size[idx] for idx, s in enumerate(latents_size)
        ]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis

    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert (
        sum(rope_dim_list) == head_dim
    ), "sum(rope_dim_list) should equal to head_dim of attention layer"
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list,
        rope_sizes,
        theta=rope_theta,
        use_real=True,
        theta_rescale_factor=1,
        num_frames=latent_video_length,
        k=k,
    )
    return freqs_cos, freqs_sin
