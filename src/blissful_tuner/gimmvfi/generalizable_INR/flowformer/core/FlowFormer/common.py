import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from ..utils.utils import coords_grid, bilinear_sampler, indexing
from loguru import logger

import math


def nerf_encoding(x, L=6, NORMALIZE_FACOR=1 / 300):
    """
    x is of shape [*, 2]. The last dimension are two coordinates (x and y).
    """
    freq_bands = 2.0 ** torch.linspace(0, L, L - 1).to(x.device)
    return torch.cat(
        [
            x * NORMALIZE_FACOR,
            torch.sin(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
            torch.cos(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
            torch.sin(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR),
            torch.cos(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR),
        ],
        dim=-1,
    )


def sampler_gaussian(latent, mean, std, image_size, point_num=25):
    # latent [B, H*W, D]
    # mean [B, 2, H, W]
    # std [B, 1, H, W]
    H, W = image_size
    B, HW, D = latent.shape
    STD_MAX = 20
    latent = rearrange(
        latent, "b (h w) c -> b c h w", h=H, w=W
    )  # latent = latent.view(B, H, W, D).permute(0, 3, 1, 2)
    mean = mean.permute(0, 2, 3, 1)  # [B, H, W, 2]

    dx = torch.linspace(-1, 1, int(point_num**0.5))
    dy = torch.linspace(-1, 1, int(point_num**0.5))
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]
    delta_3sigma = (
        F.sigmoid(std.permute(0, 2, 3, 1).reshape(B * HW, 1, 1, 1))
        * STD_MAX
        * delta
        * 3
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    centroid = mean.reshape(B * H * W, 1, 1, 2)
    coords = centroid + delta_3sigma

    coords = rearrange(coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W)
    sampled_latents = bilinear_sampler(
        latent, coords
    )  # [B*H*W, dim, point_num**0.5, point_num**0.5]
    sampled_latents = sampled_latents.permute(0, 2, 3, 1)
    sampled_weights = -(torch.sum(delta.pow(2), dim=-1))

    return sampled_latents, sampled_weights


def sampler_gaussian_zy(
    latent, mean, std, image_size, point_num=25, return_deltaXY=False, beta=1
):
    # latent [B, H*W, D]
    # mean [B, 2, H, W]
    # std [B, 1, H, W]
    H, W = image_size
    B, HW, D = latent.shape
    latent = rearrange(
        latent, "b (h w) c -> b c h w", h=H, w=W
    )  # latent = latent.view(B, H, W, D).permute(0, 3, 1, 2)
    mean = mean.permute(0, 2, 3, 1)  # [B, H, W, 2]

    dx = torch.linspace(-1, 1, int(point_num**0.5))
    dy = torch.linspace(-1, 1, int(point_num**0.5))
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]
    delta_3sigma = (
        std.permute(0, 2, 3, 1).reshape(B * HW, 1, 1, 1) * delta * 3
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    centroid = mean.reshape(B * H * W, 1, 1, 2)
    coords = centroid + delta_3sigma

    coords = rearrange(coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W)
    sampled_latents = bilinear_sampler(
        latent, coords
    )  # [B*H*W, dim, point_num**0.5, point_num**0.5]
    sampled_latents = sampled_latents.permute(0, 2, 3, 1)
    sampled_weights = -(torch.sum(delta.pow(2), dim=-1)) / beta

    if return_deltaXY:
        return sampled_latents, sampled_weights, delta_3sigma
    else:
        return sampled_latents, sampled_weights


def sampler_gaussian(latent, mean, std, image_size, point_num=25, return_deltaXY=False):
    # latent [B, H*W, D]
    # mean [B, 2, H, W]
    # std [B, 1, H, W]
    H, W = image_size
    B, HW, D = latent.shape
    STD_MAX = 20
    latent = rearrange(
        latent, "b (h w) c -> b c h w", h=H, w=W
    )  # latent = latent.view(B, H, W, D).permute(0, 3, 1, 2)
    mean = mean.permute(0, 2, 3, 1)  # [B, H, W, 2]

    dx = torch.linspace(-1, 1, int(point_num**0.5))
    dy = torch.linspace(-1, 1, int(point_num**0.5))
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]
    delta_3sigma = (
        F.sigmoid(std.permute(0, 2, 3, 1).reshape(B * HW, 1, 1, 1))
        * STD_MAX
        * delta
        * 3
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    centroid = mean.reshape(B * H * W, 1, 1, 2)
    coords = centroid + delta_3sigma

    coords = rearrange(coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W)
    sampled_latents = bilinear_sampler(
        latent, coords
    )  # [B*H*W, dim, point_num**0.5, point_num**0.5]
    sampled_latents = sampled_latents.permute(0, 2, 3, 1)
    sampled_weights = -(torch.sum(delta.pow(2), dim=-1))

    if return_deltaXY:
        return sampled_latents, sampled_weights, delta_3sigma
    else:
        return sampled_latents, sampled_weights


def sampler_gaussian_fix(latent, mean, image_size, point_num=49):
    # latent [B, H*W, D]
    # mean [B, 2, H, W]
    H, W = image_size
    B, HW, D = latent.shape
    STD_MAX = 20
    latent = rearrange(
        latent, "b (h w) c -> b c h w", h=H, w=W
    )  # latent = latent.view(B, H, W, D).permute(0, 3, 1, 2)
    mean = mean.permute(0, 2, 3, 1)  # [B, H, W, 2]

    radius = int((int(point_num**0.5) - 1) / 2)

    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    centroid = mean.reshape(B * H * W, 1, 1, 2)
    coords = centroid + delta

    coords = rearrange(coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W)
    sampled_latents = bilinear_sampler(
        latent, coords
    )  # [B*H*W, dim, point_num**0.5, point_num**0.5]
    sampled_latents = sampled_latents.permute(0, 2, 3, 1)
    sampled_weights = -(torch.sum(delta.pow(2), dim=-1)) / point_num  # smooth term

    return sampled_latents, sampled_weights


def sampler_gaussian_fix_pyramid(
    latent, feat_pyramid, scale_weight, mean, image_size, point_num=25
):
    # latent [B, H*W, D]
    # mean [B, 2, H, W]
    # scale weight [B, H*W, layer_num]

    H, W = image_size
    B, HW, D = latent.shape
    STD_MAX = 20
    latent = rearrange(
        latent, "b (h w) c -> b c h w", h=H, w=W
    )  # latent = latent.view(B, H, W, D).permute(0, 3, 1, 2)
    mean = mean.permute(0, 2, 3, 1)  # [B, H, W, 2]

    radius = int((int(point_num**0.5) - 1) / 2)

    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    sampled_latents = []
    for i in range(len(feat_pyramid)):
        centroid = mean.reshape(B * H * W, 1, 1, 2)
        coords = (centroid + delta) / 2**i
        coords = rearrange(
            coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W
        )
        sampled_latents.append(bilinear_sampler(feat_pyramid[i], coords))

    sampled_latents = torch.stack(
        sampled_latents, dim=1
    )  # [B, layer_num, dim, H*W, point_num]
    sampled_latents = sampled_latents.permute(
        0, 3, 4, 2, 1
    )  # [B, H*W, point_num, dim, layer_num]
    scale_weight = F.softmax(scale_weight, dim=2)  # [B, H*W, layer_num]
    vis_out = scale_weight
    scale_weight = torch.unsqueeze(
        torch.unsqueeze(scale_weight, dim=2), dim=2
    )  # [B, HW, 1, 1, layer_num]

    weighted_latent = torch.sum(
        sampled_latents * scale_weight, dim=-1
    )  # [B, H*W, point_num, dim]

    sampled_weights = -(torch.sum(delta.pow(2), dim=-1)) / point_num  # smooth term

    return weighted_latent, sampled_weights, vis_out


def sampler_gaussian_pyramid(
    latent, feat_pyramid, scale_weight, mean, std, image_size, point_num=25
):
    # latent [B, H*W, D]
    # mean [B, 2, H, W]
    # scale weight [B, H*W, layer_num]

    H, W = image_size
    B, HW, D = latent.shape
    STD_MAX = 20
    latent = rearrange(
        latent, "b (h w) c -> b c h w", h=H, w=W
    )  # latent = latent.view(B, H, W, D).permute(0, 3, 1, 2)
    mean = mean.permute(0, 2, 3, 1)  # [B, H, W, 2]

    radius = int((int(point_num**0.5) - 1) / 2)

    dx = torch.linspace(-1, 1, int(point_num**0.5))
    dy = torch.linspace(-1, 1, int(point_num**0.5))
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]
    delta_3sigma = (
        std.permute(0, 2, 3, 1).reshape(B * HW, 1, 1, 1) * delta * 3
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    sampled_latents = []
    for i in range(len(feat_pyramid)):
        centroid = mean.reshape(B * H * W, 1, 1, 2)
        coords = (centroid + delta_3sigma) / 2**i
        coords = rearrange(
            coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W
        )
        sampled_latents.append(bilinear_sampler(feat_pyramid[i], coords))

    sampled_latents = torch.stack(
        sampled_latents, dim=1
    )  # [B, layer_num, dim, H*W, point_num]
    sampled_latents = sampled_latents.permute(
        0, 3, 4, 2, 1
    )  # [B, H*W, point_num, dim, layer_num]
    scale_weight = F.softmax(scale_weight, dim=2)  # [B, H*W, layer_num]
    vis_out = scale_weight
    scale_weight = torch.unsqueeze(
        torch.unsqueeze(scale_weight, dim=2), dim=2
    )  # [B, HW, 1, 1, layer_num]

    weighted_latent = torch.sum(
        sampled_latents * scale_weight, dim=-1
    )  # [B, H*W, point_num, dim]

    sampled_weights = -(torch.sum(delta.pow(2), dim=-1)) / point_num  # smooth term

    return weighted_latent, sampled_weights, vis_out


def sampler_gaussian_fix_MH(latent, mean, image_size, point_num=25):
    """different heads have different mean"""
    # latent [B, H*W, D]
    # mean [B, 2, H, W, heands]

    H, W = image_size
    B, HW, D = latent.shape
    _, _, _, _, HEADS = mean.shape
    STD_MAX = 20
    latent = rearrange(latent, "b (h w) c -> b c h w", h=H, w=W)
    mean = mean.permute(0, 2, 3, 4, 1)  # [B, H, W, heads, 2]

    radius = int((int(point_num**0.5) - 1) / 2)

    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = (
        torch.stack(torch.meshgrid(dy, dx), axis=-1)
        .to(mean.device)
        .repeat(HEADS, 1, 1, 1)
    )  # [HEADS, point_num**0.5, point_num**0.5, 2]

    centroid = mean.reshape(B * H * W, HEADS, 1, 1, 2)
    coords = centroid + delta
    coords = rearrange(
        coords, "(b h w) H r1 r2 c -> b (h w H) (r1 r2) c", b=B, h=H, w=W, H=HEADS
    )
    sampled_latents = bilinear_sampler(latent, coords)  # [B, dim, H*W*HEADS, pointnum]
    sampled_latents = sampled_latents.permute(
        0, 2, 3, 1
    )  # [B, H*W*HEADS, pointnum, dim]
    sampled_weights = -(torch.sum(delta.pow(2), dim=-1)) / point_num  # smooth term
    return sampled_latents, sampled_weights


def sampler_gaussian_fix_pyramid_MH(
    latent, feat_pyramid, scale_head_weight, mean, image_size, point_num=25
):
    # latent [B, H*W, D]
    # mean [B, 2, H, W, heands]
    # scale_head weight [B, H*W, layer_num*heads]

    H, W = image_size
    B, HW, D = latent.shape
    _, _, _, _, HEADS = mean.shape

    latent = rearrange(latent, "b (h w) c -> b c h w", h=H, w=W)
    mean = mean.permute(0, 2, 3, 4, 1)  # [B, H, W, heads, 2]

    radius = int((int(point_num**0.5) - 1) / 2)

    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        mean.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    sampled_latents = []
    centroid = mean.reshape(B * H * W, HEADS, 1, 1, 2)
    for i in range(len(feat_pyramid)):
        coords = (centroid) / 2**i + delta
        coords = rearrange(
            coords, "(b h w) H r1 r2 c -> b (h w H) (r1 r2) c", b=B, h=H, w=W, H=HEADS
        )
        sampled_latents.append(
            bilinear_sampler(feat_pyramid[i], coords)
        )  # [B, dim, H*W*HEADS, point_num]

    sampled_latents = torch.stack(
        sampled_latents, dim=1
    )  # [B, layer_num, dim, H*W*HEADS, point_num]
    sampled_latents = sampled_latents.permute(
        0, 3, 4, 2, 1
    )  # [B, H*W*HEADS, point_num, dim, layer_num]

    scale_head_weight = scale_head_weight.reshape(B, H * W * HEADS, -1)
    scale_head_weight = F.softmax(scale_head_weight, dim=2)  # [B, H*W*HEADS, layer_num]
    scale_head_weight = torch.unsqueeze(
        torch.unsqueeze(scale_head_weight, dim=2), dim=2
    )  # [B, H*W*HEADS, 1, 1, layer_num]

    weighted_latent = torch.sum(
        sampled_latents * scale_head_weight, dim=-1
    )  # [B, H*W*HEADS, point_num, dim]

    sampled_weights = -(torch.sum(delta.pow(2), dim=-1)) / point_num  # smooth term

    return weighted_latent, sampled_weights


def sampler(feat, center, window_size):
    # feat [B, C, H, W]
    # center [B, 2, H, W]
    center = center.permute(0, 2, 3, 1)  # [B, H, W, 2]
    B, H, W, C = center.shape

    radius = window_size // 2
    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        center.device
    )  # [B*H*W, window_size, point_num**0.5, 2]

    center = center.reshape(B * H * W, 1, 1, 2)
    coords = center + delta

    coords = rearrange(coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W)
    sampled_latents = bilinear_sampler(
        feat, coords
    )  # [B*H*W, dim, window_size, window_size]
    # sampled_latents = sampled_latents.permute(0, 2, 3, 1)

    return sampled_latents


def retrieve_tokens(feat, center, window_size, sampler):
    # feat [B, C, H, W]
    # center [B, 2, H, W]
    radius = window_size // 2
    dx = torch.linspace(-radius, radius, 2 * radius + 1)
    dy = torch.linspace(-radius, radius, 2 * radius + 1)
    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(
        center.device
    )  # [B*H*W, point_num**0.5, point_num**0.5, 2]

    B, H, W, C = center.shape
    centroid = center.reshape(B * H * W, 1, 1, 2)
    coords = centroid + delta

    coords = rearrange(coords, "(b h w) r1 r2 c -> b (h w) (r1 r2) c", b=B, h=H, w=W)
    if sampler == "nn":
        sampled_latents = indexing(feat, coords)
    elif sampler == "bilinear":
        sampled_latents = bilinear_sampler(feat, coords)
    else:
        raise ValueError("invalid sampler")
    # [B, dim, H*W, point_num]

    return sampled_latents


def pyramid_retrieve_tokens(
    feat_pyramid, center, image_size, window_sizes, sampler="bilinear"
):
    center = center.permute(0, 2, 3, 1)  # [B, H, W, 2]
    sampled_latents_pyramid = []
    for idx in range(len(window_sizes)):
        sampled_latents_pyramid.append(
            retrieve_tokens(feat_pyramid[idx], center, window_sizes[idx], sampler)
        )
        center = center / 2

    return torch.cat(sampled_latents_pyramid, dim=-1)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim=22, out_dim=1, innter_dim=96, depth=5):
        super().__init__()
        self.FC1 = nn.Linear(in_dim, innter_dim)
        self.FC_out = nn.Linear(innter_dim, out_dim)
        self.relu = torch.nn.LeakyReLU(0.2)
        self.FC_inter = nn.ModuleList(
            [nn.Linear(innter_dim, innter_dim) for i in range(depth)]
        )

    def forward(self, x):
        x = self.FC1(x)
        x = self.relu(x)
        for inter_fc in self.FC_inter:
            x = inter_fc(x)
            x = self.relu(x)
        x = self.FC_out(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, num_kv_tokens, cfg, rpe_bias=None, use_rpe=False):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.num_kv_tokens = num_kv_tokens
        self.scale = (dim / heads) ** -0.5
        self.rpe = cfg.rpe
        self.attend = nn.Softmax(dim=-1)
        self.use_rpe = use_rpe

        if use_rpe:
            if rpe_bias is None:
                if self.rpe == "element-wise":
                    self.rpe_bias = nn.Parameter(
                        torch.zeros(heads, self.num_kv_tokens, dim // heads)
                    )
                elif self.rpe == "head-wise":
                    self.rpe_bias = nn.Parameter(
                        torch.zeros(1, heads, 1, self.num_kv_tokens)
                    )
                elif self.rpe == "token-wise":
                    self.rpe_bias = nn.Parameter(
                        torch.zeros(1, 1, 1, self.num_kv_tokens)
                    )  # 81 is point_num
                elif self.rpe == "implicit":
                    pass
                    # self.implicit_pe_fn = MLP(in_dim=22, out_dim=self.dim, innter_dim=int(self.dim//2.4), depth=2)
                    # raise ValueError('Implicit Encoding Not Implemented')
                elif self.rpe == "element-wise-value":
                    self.rpe_bias = nn.Parameter(
                        torch.zeros(heads, self.num_kv_tokens, dim // heads)
                    )
                    self.rpe_value = nn.Parameter(torch.randn(self.num_kv_tokens, dim))
                else:
                    raise ValueError("Not Implemented")
            else:
                self.rpe_bias = rpe_bias

    def attend_with_rpe(self, Q, K, rpe_bias):
        Q = rearrange(Q, "b i (heads d) -> b heads i d", heads=self.heads)
        K = rearrange(K, "b j (heads d) -> b heads j d", heads=self.heads)

        dots = (
            einsum("bhid, bhjd -> bhij", Q, K) * self.scale
        )  # (b hw) heads 1 pointnum
        if self.use_rpe:
            if self.rpe == "element-wise":
                rpe_bias_weight = (
                    einsum("bhid, hjd -> bhij", Q, rpe_bias) * self.scale
                )  # (b hw) heads 1 pointnum
                dots = dots + rpe_bias_weight
            elif self.rpe == "implicit":
                pass
                rpe_bias_weight = (
                    einsum("bhid, bhjd -> bhij", Q, rpe_bias) * self.scale
                )  # (b hw) heads 1 pointnum
                dots = dots + rpe_bias_weight
            elif self.rpe == "head-wise" or self.rpe == "token-wise":
                dots = dots + rpe_bias

        return self.attend(dots), dots

    def forward(self, Q, K, V, rpe_bias=None):
        if self.use_rpe:
            if rpe_bias is None or self.rpe == "element-wise":
                rpe_bias = self.rpe_bias
            else:
                rpe_bias = rearrange(
                    rpe_bias, "b hw pn (heads d) -> (b hw) heads pn d", heads=self.heads
                )
            attn, dots = self.attend_with_rpe(Q, K, rpe_bias)
        else:
            attn, dots = self.attend_with_rpe(Q, K, None)
        B, HW, _ = Q.shape

        if V is not None:
            V = rearrange(V, "b j (heads d) -> b heads j d", heads=self.heads)

            out = einsum("bhij, bhjd -> bhid", attn, V)
            out = rearrange(out, "b heads hw d -> b hw (heads d)", b=B, hw=HW)
        else:
            out = None

        # dots = torch.squeeze(dots, 2)
        # dots = rearrange(dots, '(b hw) heads d -> b hw (heads d)', b=B, hw=HW)

        return out, dots
