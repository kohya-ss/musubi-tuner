""" Twins
A PyTorch impl of : `Twins: Revisiting the Design of Spatial Attention in Vision Transformers`
    - https://arxiv.org/pdf/2104.13840.pdf
Code/weights from https://github.com/Meituan-AutoML/Twins, original copyright/license info below
"""
# --------------------------------------------------------
# Twins
# Copyright (c) 2021 Meituan
# Licensed under The Apache 2.0 License [see LICENSE for details]
# Written by Xinjie Li, Xiangxiang Chu
# --------------------------------------------------------
import math
from copy import deepcopy
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import Mlp, DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Attention
from timm.models.helpers import build_model_with_cfg#, overlay_external_default_cfg
from .attention import MultiHeadAttention, LinearPositionEmbeddingSine
from ...utils.utils import coords_grid, bilinear_sampler, upflow8


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embeds.0.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "twins_pcpvt_small": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_small-e70e7e7a.pth",
    ),
    "twins_pcpvt_base": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_base-e5ecb09b.pth",
    ),
    "twins_pcpvt_large": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_pcpvt_large-d273f802.pth",
    ),
    "twins_svt_small": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_small-42e5f78c.pth",
    ),
    "twins_svt_base": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_base-c2265010.pth",
    ),
    "twins_svt_large": _cfg(
        url="https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vt3p-weights/twins_svt_large-90f6aaa9.pth",
    ),
}

Size_ = Tuple[int, int]


class GroupAttnRPEContext(nn.Module):
    """Latent cost tokens attend to different group"""

    def __init__(
        self,
        dim,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        ws=1,
        cfg=None,
        vert_c_dim=0,
    ):
        super(GroupAttnRPEContext, self).__init__()
        assert ws != 1
        assert cfg is not None
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        assert (
            cfg.cost_latent_token_num % 5 == 0
        ), "cost_latent_token_num should be divided by 5."
        assert vert_c_dim > 0, "vert_c_dim should not be 0"

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.vert_c_dim = vert_c_dim

        self.cfg = cfg

        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        C_qk = C + self.vert_c_dim
        H, W = size
        batch_num = B // 5

        context = context.repeat(B // context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H * W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)

        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        padded_N = Hp * Wp

        coords = coords_grid(B, Hp, Wp).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)
        coords_enc = coords_enc.reshape(B, Hp, Wp, C_qk)

        q = (
            self.q(x_qk + coords_enc)
            .reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads)
            .transpose(2, 3)
        )
        q = q.reshape(
            B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4)

        v = self.v(x)
        k = self.k(x_qk + coords_enc)
        # concate and do shifting operation together
        kv = torch.cat([k, v], dim=-1)
        kv_up = torch.cat(
            [
                kv[:batch_num, self.ws : Hp, :, :],
                kv[:batch_num, Hp - self.ws : Hp, :, :],
            ],
            dim=1,
        )
        kv_down = torch.cat(
            [
                kv[batch_num : batch_num * 2, : self.ws, :, :],
                kv[batch_num : batch_num * 2, : Hp - self.ws, :, :],
            ],
            dim=1,
        )
        kv_left = torch.cat(
            [
                kv[batch_num * 2 : batch_num * 3, :, self.ws : Wp, :],
                kv[batch_num * 2 : batch_num * 3, :, Wp - self.ws : Wp, :],
            ],
            dim=2,
        )
        kv_right = torch.cat(
            [
                kv[batch_num * 3 : batch_num * 4, :, : self.ws, :],
                kv[batch_num * 3 : batch_num * 4, :, : Wp - self.ws, :],
            ],
            dim=2,
        )
        kv_center = kv[batch_num * 4 : batch_num * 5, :, :, :]
        kv_shifted = torch.cat([kv_up, kv_down, kv_left, kv_right, kv_center], dim=0)
        k, v = torch.split(kv_shifted, [self.dim, self.dim], dim=-1)

        k = k.reshape(
            B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads
        ).transpose(2, 3)
        k = k.reshape(
            B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4)

        v = v.reshape(
            B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads
        ).transpose(2, 3)
        v = v.reshape(
            B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GroupAttnRPE(nn.Module):
    """Latent cost tokens attend to different group"""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1, cfg=None):
        super(GroupAttnRPE, self).__init__()
        assert ws != 1
        assert cfg is not None
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        assert (
            cfg.cost_latent_token_num % 5 == 0
        ), "cost_latent_token_num should be divided by 5."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.cfg = cfg

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        batch_num = B // 5
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        padded_N = Hp * Wp

        coords = coords_grid(B, Hp, Wp).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        coords_enc = coords_enc.reshape(B, Hp, Wp, C)

        q = (
            self.q(x + coords_enc)
            .reshape(B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads)
            .transpose(2, 3)
        )
        q = q.reshape(
            B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4)

        v = self.v(x)
        k = self.k(x + coords_enc)
        # concate and do shifting operation together
        kv = torch.cat([k, v], dim=-1)
        kv_up = torch.cat(
            [
                kv[:batch_num, self.ws : Hp, :, :],
                kv[:batch_num, Hp - self.ws : Hp, :, :],
            ],
            dim=1,
        )
        kv_down = torch.cat(
            [
                kv[batch_num : batch_num * 2, : self.ws, :, :],
                kv[batch_num : batch_num * 2, : Hp - self.ws, :, :],
            ],
            dim=1,
        )
        kv_left = torch.cat(
            [
                kv[batch_num * 2 : batch_num * 3, :, self.ws : Wp, :],
                kv[batch_num * 2 : batch_num * 3, :, Wp - self.ws : Wp, :],
            ],
            dim=2,
        )
        kv_right = torch.cat(
            [
                kv[batch_num * 3 : batch_num * 4, :, : self.ws, :],
                kv[batch_num * 3 : batch_num * 4, :, : Wp - self.ws, :],
            ],
            dim=2,
        )
        kv_center = kv[batch_num * 4 : batch_num * 5, :, :, :]
        kv_shifted = torch.cat([kv_up, kv_down, kv_left, kv_right, kv_center], dim=0)
        k, v = torch.split(kv_shifted, [self.dim, self.dim], dim=-1)

        k = k.reshape(
            B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads
        ).transpose(2, 3)
        k = k.reshape(
            B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4)

        v = v.reshape(
            B, _h, self.ws, _w, self.ws, self.num_heads, C // self.num_heads
        ).transpose(2, 3)
        v = v.reshape(
            B, _h * _w, self.ws * self.ws, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LocallyGroupedAttnRPEContext(nn.Module):
    """LSA: self attention within a group"""

    def __init__(
        self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1, vert_c_dim=0
    ):
        assert ws != 1
        super(LocallyGroupedAttnRPEContext, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.vert_c_dim = vert_c_dim

        self.context_proj = nn.Linear(256, vert_c_dim)
        # context are not added to value
        self.q = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        C_qk = C + self.vert_c_dim

        context = context.repeat(B // context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H * W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)

        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)

        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        x_qk = x_qk.reshape(B, _h, self.ws, _w, self.ws, C_qk).transpose(2, 3)

        v = (
            self.v(x)
            .reshape(
                B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)[0]
        )

        coords = coords_grid(B, self.ws, self.ws).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk).view(
            B, self.ws, self.ws, C_qk
        )
        # coords_enc:   B, ws, ws, C
        # x:            B, _h, _w, self.ws, self.ws, C
        x_qk = x_qk + coords_enc[:, None, None, :, :, :]

        q = (
            self.q(x_qk)
            .reshape(
                B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)[0]
        )
        k = (
            self.k(x_qk)
            .reshape(
                B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)[0]
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttnRPEContext(nn.Module):
    """GSA: using a  key to summarize the information for a group to be efficient."""

    def __init__(
        self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1, vert_c_dim=0
    ):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.vert_c_dim = vert_c_dim
        self.context_proj = nn.Linear(256, vert_c_dim)
        self.q = nn.Linear(dim + vert_c_dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr_key = nn.Conv2d(
                dim + vert_c_dim, dim, kernel_size=sr_ratio, stride=sr_ratio
            )
            self.sr_value = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        C_qk = C + self.vert_c_dim
        H, W = size
        context = context.repeat(B // context.shape[0], 1, 1, 1)
        context = context.view(B, -1, H * W).permute(0, 2, 1)
        context = self.context_proj(context)
        context = context.view(B, H, W, -1)
        x = x.view(B, H, W, C)
        x_qk = torch.cat([x, context], dim=-1)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        x_qk = F.pad(x_qk, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape
        padded_size = (Hp, Wp)
        padded_N = Hp * Wp
        x = x.view(B, -1, C)
        x_qk = x_qk.view(B, -1, C_qk)

        coords = coords_grid(B, *padded_size).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C_qk)
        # coords_enc:   B, Hp*Wp, C
        # x:            B, Hp*Wp, C
        q = (
            self.q(x_qk + coords_enc)
            .reshape(B, padded_N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_key is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
            x_qk = x_qk.permute(0, 2, 1).reshape(B, C_qk, *padded_size)
            x = self.sr_value(x).reshape(B, C, -1).permute(0, 2, 1)
            x_qk = self.sr_key(x_qk).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
            x_qk = self.norm(x_qk)

        coords = coords_grid(
            B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio
        ).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        # align the coordinate of local and global
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = (
            self.k(x_qk + coords_enc)
            .reshape(
                B,
                (padded_size[0] // self.sr_ratio) * (padded_size[1] // self.sr_ratio),
                self.num_heads,
                C // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(
                B,
                (padded_size[0] // self.sr_ratio) * (padded_size[1] // self.sr_ratio),
                self.num_heads,
                C // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LocallyGroupedAttnRPE(nn.Module):
    """LSA: self attention within a group"""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1):
        assert ws != 1
        super(LocallyGroupedAttnRPE, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_, context=None):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        v = (
            self.v(x)
            .reshape(
                B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)[0]
        )

        coords = coords_grid(B, self.ws, self.ws).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C).view(
            B, self.ws, self.ws, C
        )
        # coords_enc:   B, ws, ws, C
        # x:            B, _h, _w, self.ws, self.ws, C
        x = x + coords_enc[:, None, None, :, :, :]

        q = (
            self.q(x)
            .reshape(
                B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)[0]
        )
        k = (
            self.k(x)
            .reshape(
                B, _h * _w, self.ws * self.ws, 1, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)[0]
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttnRPE(nn.Module):
    """GSA: using a  key to summarize the information for a group to be efficient."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_, context=None):
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.sr_ratio - W % self.sr_ratio) % self.sr_ratio
        pad_b = (self.sr_ratio - H % self.sr_ratio) % self.sr_ratio
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        padded_size = (Hp, Wp)
        padded_N = Hp * Wp
        x = x.view(B, -1, C)

        coords = coords_grid(B, *padded_size).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        # coords_enc:   B, Hp*Wp, C
        # x:            B, Hp*Wp, C
        q = (
            self.q(x + coords_enc)
            .reshape(B, padded_N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *padded_size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)

        coords = coords_grid(
            B, padded_size[0] // self.sr_ratio, padded_size[1] // self.sr_ratio
        ).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        # align the coordinate of local and global
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = (
            self.k(x + coords_enc)
            .reshape(
                B,
                (padded_size[0] // self.sr_ratio) * (padded_size[1] // self.sr_ratio),
                self.num_heads,
                C // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(
                B,
                (padded_size[0] // self.sr_ratio) * (padded_size[1] // self.sr_ratio),
                self.num_heads,
                C // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Hp, Wp, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossGlobalSubSampleAttnRPE(nn.Module):
    """GSA: using a  key to summarize the information for a group to be efficient."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, tgt, size: Size_):
        B, N, C = x.shape
        coords = coords_grid(B, *size).to(x.device)
        coords = coords.view(B, 2, -1).permute(0, 2, 1)
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        # coords_enc:   B, H*W, C
        # x:            B, H*W, C
        q = (
            self.q(x + coords_enc)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr is not None:
            tgt = tgt.permute(0, 2, 1).reshape(B, C, *size)
            tgt = self.sr(tgt).reshape(B, C, -1).permute(0, 2, 1)
            tgt = self.norm(tgt)
        coords = coords_grid(B, size[0] // self.sr_ratio, size[1] // self.sr_ratio).to(
            x.device
        )
        coords = coords.view(B, 2, -1).permute(0, 2, 1) * self.sr_ratio
        # align the coordinate of local and global
        coords_enc = LinearPositionEmbeddingSine(coords, dim=C)
        k = (
            self.k(tgt + coords_enc)
            .reshape(
                B,
                (size[0] // self.sr_ratio) * (size[1] // self.sr_ratio),
                self.num_heads,
                C // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(tgt)
            .reshape(
                B,
                (size[0] // self.sr_ratio) * (size[1] // self.sr_ratio),
                self.num_heads,
                C // self.num_heads,
            )
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LocallyGroupedAttn(nn.Module):
    """LSA: self attention within a group"""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, ws=1):
        assert ws != 1
        super(LocallyGroupedAttn, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.ws = ws

    def forward(self, x, size: Size_):
        # There are two implementations for this function, zero padding or mask. We don't observe obvious difference for
        # both. You can choose any one, we recommend forward_padding because it's neat. However,
        # the masking implementation is more reasonable and accurate.
        B, N, C = x.shape
        H, W = size
        x = x.view(B, H, W, C)
        pad_l = pad_t = 0
        pad_r = (self.ws - W % self.ws) % self.ws
        pad_b = (self.ws - H % self.ws) % self.ws
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        _h, _w = Hp // self.ws, Wp // self.ws
        x = x.reshape(B, _h, self.ws, _w, self.ws, C).transpose(2, 3)
        qkv = (
            self.qkv(x)
            .reshape(
                B, _h * _w, self.ws * self.ws, 3, self.num_heads, C // self.num_heads
            )
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = (attn @ v).transpose(2, 3).reshape(B, _h, _w, self.ws, self.ws, C)
        x = attn.transpose(2, 3).reshape(B, _h * self.ws, _w * self.ws, C)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSubSampleAttn(nn.Module):
    """GSA: using a  key to summarize the information for a group to be efficient."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr is not None:
            x = x.permute(0, 2, 1).reshape(B, C, *size)
            x = self.sr(x).reshape(B, C, -1).permute(0, 2, 1)
            x = self.norm(x)
        kv = (
            self.kv(x)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossGlobalSubSampleAttn(nn.Module):
    """GSA: using a  key to summarize the information for a group to be efficient."""

    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0, sr_ratio=1):
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None

    def forward(self, x, tgt, size: Size_):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr is not None:
            tgt = tgt.permute(0, 2, 1).reshape(B, C, *size)
            tgt = self.sr(tgt).reshape(B, C, -1).permute(0, 2, 1)
            tgt = self.norm(tgt)
        kv = (
            self.kv(tgt)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CrossBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        ws=None,
        with_rpe=True,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossGlobalSubSampleAttnRPE(
            dim, num_heads, attn_drop, drop, sr_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, src, tgt, size: Size_):
        src_shortcut, tgt_shortcut = src, tgt

        src, tgt = self.norm1(src), self.norm1(tgt)
        src = src_shortcut + self.drop_path(self.attn(src, tgt, size))
        tgt = tgt_shortcut + self.drop_path(self.attn(tgt, src, size))

        src = src + self.drop_path(self.mlp(self.norm2(src)))
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))
        return src, tgt


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        ws=None,
        with_rpe=False,
        vert_c_dim=0,
        groupattention=False,
        cfg=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if groupattention:
            assert with_rpe, "Not implementing groupattention without rpe"
            if vert_c_dim > 0:
                self.attn = GroupAttnRPEContext(
                    dim, num_heads, attn_drop, drop, ws, cfg, vert_c_dim
                )
            else:
                self.attn = GroupAttnRPE(dim, num_heads, attn_drop, drop, ws, cfg)
        elif ws is None:
            self.attn = Attention(dim, num_heads, False, None, attn_drop, drop)
        elif ws == 1:
            if with_rpe:
                if vert_c_dim > 0:
                    self.attn = GlobalSubSampleAttnRPEContext(
                        dim, num_heads, attn_drop, drop, sr_ratio, vert_c_dim
                    )
                else:
                    self.attn = GlobalSubSampleAttnRPE(
                        dim, num_heads, attn_drop, drop, sr_ratio
                    )
            else:
                self.attn = GlobalSubSampleAttn(
                    dim, num_heads, attn_drop, drop, sr_ratio
                )
        else:
            if with_rpe:
                if vert_c_dim > 0:
                    self.attn = LocallyGroupedAttnRPEContext(
                        dim, num_heads, attn_drop, drop, ws, vert_c_dim
                    )
                else:
                    self.attn = LocallyGroupedAttnRPE(
                        dim, num_heads, attn_drop, drop, ws
                    )
            else:
                self.attn = LocallyGroupedAttn(dim, num_heads, attn_drop, drop, ws)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, size: Size_, context=None):
        x = x + self.drop_path(self.attn(self.norm1(x), size, context))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PosConv(nn.Module):
    # PEG  from https://arxiv.org/abs/2102.10882
    def __init__(self, in_chans, embed_dim=768, stride=1):
        super(PosConv, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim),
        )
        self.stride = stride

    def forward(self, x, size: Size_):
        B, N, C = x.shape
        cnn_feat_token = x.transpose(1, 2).view(B, C, *size)
        x = self.proj(cnn_feat_token)
        if self.stride == 1:
            x += cnn_feat_token
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert (
            img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0
        ), f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, Size_]:
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        out_size = (H // self.patch_size[0], W // self.patch_size[1])

        return x, out_size


class Twins(nn.Module):
    """Twins Vision Transfomer (Revisiting Spatial Attention)
    Adapted from PVT (PyramidVisionTransformer) class at https://github.com/whai362/PVT.git
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dims=(64, 128, 256, 512),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(4, 4, 4, 4),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        wss=None,
        block_cls=Block,
        init_weight=True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_features = embed_dims[-1]

        img_size = to_2tuple(img_size)
        prev_chs = in_chans
        self.patch_embeds = nn.ModuleList()
        self.pos_drops = nn.ModuleList()
        for i in range(len(depths)):
            self.patch_embeds.append(
                PatchEmbed(img_size, patch_size, prev_chs, embed_dims[i])
            )
            self.pos_drops.append(nn.Dropout(p=drop_rate))
            prev_chs = embed_dims[i]
            img_size = tuple(t // patch_size for t in img_size)
            patch_size = 2

        self.blocks = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.ModuleList(
                [
                    block_cls(
                        dim=embed_dims[k],
                        num_heads=num_heads[k],
                        mlp_ratio=mlp_ratios[k],
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[cur + i],
                        norm_layer=norm_layer,
                        sr_ratio=sr_ratios[k],
                        ws=1 if wss is None or i % 2 == 1 else wss[k],
                    )
                    for i in range(depths[k])
                ]
            )
            self.blocks.append(_block)
            cur += depths[k]

        self.pos_block = nn.ModuleList(
            [PosConv(embed_dim, embed_dim) for embed_dim in embed_dims]
        )

        self.norm = norm_layer(self.num_features)

        # classification head
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        # init weights
        if init_weight:
            self.apply(self._init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set(["pos_block." + n for n, p in self.pos_block.named_parameters()])

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()

    def forward_features(self, x):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(self.patch_embeds, self.pos_drops, self.blocks, self.pos_block)
        ):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.norm(x)
        return x.mean(dim=1)  # GAP here

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# def _create_twins(variant, pretrained=False, **kwargs):
#     if kwargs.get('features_only', None):
#         raise RuntimeError('features_only not implemented for Vision Transformer models.')

#     model = build_model_with_cfg(
#         Twins, variant, pretrained,
#         default_cfg=default_cfgs[variant],
#         **kwargs)
#     return model


# @register_model
# def twins_pcpvt_small(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#         depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
#     return _create_twins('twins_pcpvt_small', pretrained=pretrained, **model_kwargs)


# @register_model
# def twins_pcpvt_base(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#         depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
#     return _create_twins('twins_pcpvt_base', pretrained=pretrained, **model_kwargs)


# @register_model
# def twins_pcpvt_large(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#         depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1], **kwargs)
#     return _create_twins('twins_pcpvt_large', pretrained=pretrained, **model_kwargs)


# @register_model
# def twins_svt_small(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4],
#         depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
#     return _create_twins('twins_svt_small', pretrained=pretrained, **model_kwargs)


# @register_model
# def twins_svt_base(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4],
#         depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
#     return _create_twins('twins_svt_base', pretrained=pretrained, **model_kwargs)


# @register_model
# def twins_svt_large(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
#         depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], **kwargs)
#     return _create_twins('twins_svt_large', pretrained=pretrained, **model_kwargs)

# @register_model
# def twins_svt_large_context(pretrained=False, **kwargs):
#     model_kwargs = dict(
#         patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
#         depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1], in_chans=6, init_weight=False, **kwargs)
#     return _create_twins('twins_svt_large', pretrained=pretrained, **model_kwargs)
# # def twins_svt_large_context(pretrained=False, **kwargs):
# #     model_kwargs = dict(
# #         patch_size=4, embed_dims=[128, 256], num_heads=[4, 8], mlp_ratios=[4, 4],
# #         depths=[2, 2], wss=[7, 7], sr_ratios=[8, 4], in_chans=6, init_weight=False, **kwargs)
# #     return _create_twins('twins_svt_large', pretrained=pretrained, **model_kwargs)
