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
