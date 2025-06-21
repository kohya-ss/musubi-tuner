import loguru
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from ...utils.utils import coords_grid, bilinear_sampler, upflow8
from .attention import (
    MultiHeadAttention,
    LinearPositionEmbeddingSine,
    ExpPositionEmbeddingSine,
)
from typing import Optional, Tuple

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .gru import BasicUpdateBlock, GMAUpdateBlock
from .gma import Attention


def initialize_flow(img):
    """Flow is represented as difference between two means flow = mean1 - mean0"""
    N, C, H, W = img.shape
    mean = coords_grid(N, H, W).to(img.device)
    mean_init = coords_grid(N, H, W).to(img.device)

    # optical flow computed as difference: flow = mean1 - mean0
    return mean, mean_init


class CrossAttentionLayer(nn.Module):
    # def __init__(self, dim, cfg, num_heads=8, attn_drop=0., proj_drop=0., drop_path=0., dropout=0.):
    def __init__(
        self,
        qk_dim,
        v_dim,
        query_token_dim,
        tgt_token_dim,
        add_flow_token=True,
        num_heads=8,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        dropout=0.0,
        pe="linear",
    ):
        super(CrossAttentionLayer, self).__init__()

        head_dim = qk_dim // num_heads
        self.scale = head_dim**-0.5
        self.query_token_dim = query_token_dim
        self.pe = pe

        self.norm1 = nn.LayerNorm(query_token_dim)
        self.norm2 = nn.LayerNorm(query_token_dim)
        self.multi_head_attn = MultiHeadAttention(qk_dim, num_heads)
        self.q, self.k, self.v = (
            nn.Linear(query_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, qk_dim, bias=True),
            nn.Linear(tgt_token_dim, v_dim, bias=True),
        )

        self.proj = nn.Linear(v_dim * 2, query_token_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.ffn = nn.Sequential(
            nn.Linear(query_token_dim, query_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(query_token_dim, query_token_dim),
            nn.Dropout(dropout),
        )
        self.add_flow_token = add_flow_token
        self.dim = qk_dim

    def forward(self, query, key, value, memory, query_coord, patch_size, size_h3w3):
        """
        query_coord [B, 2, H1, W1]
        """
        B, _, H1, W1 = query_coord.shape

        if key is None and value is None:
            key = self.k(memory)
            value = self.v(memory)

        # [B, 2, H1, W1] -> [BH1W1, 1, 2]
        query_coord = query_coord.contiguous()
        query_coord = (
            query_coord.view(B, 2, -1)
            .permute(0, 2, 1)[:, :, None, :]
            .contiguous()
            .view(B * H1 * W1, 1, 2)
        )
        if self.pe == "linear":
            query_coord_enc = LinearPositionEmbeddingSine(query_coord, dim=self.dim)
        elif self.pe == "exp":
            query_coord_enc = ExpPositionEmbeddingSine(query_coord, dim=self.dim)

        short_cut = query
        query = self.norm1(query)

        if self.add_flow_token:
            q = self.q(query + query_coord_enc)
        else:
            q = self.q(query_coord_enc)
        k, v = key, value

        x = self.multi_head_attn(q, k, v)

        x = self.proj(torch.cat([x, short_cut], dim=2))
        x = short_cut + self.proj_drop(x)

        x = x + self.drop_path(self.ffn(self.norm2(x)))

        return x, k, v


class MemoryDecoderLayer(nn.Module):
    def __init__(self, dim, cfg):
        super(MemoryDecoderLayer, self).__init__()
        self.cfg = cfg
        self.patch_size = cfg.patch_size  # for converting coords into H2', W2' space

        query_token_dim, tgt_token_dim = cfg.query_latent_dim, cfg.cost_latent_dim
        qk_dim, v_dim = query_token_dim, query_token_dim
        self.cross_attend = CrossAttentionLayer(
            qk_dim,
            v_dim,
            query_token_dim,
            tgt_token_dim,
            add_flow_token=cfg.add_flow_token,
            dropout=cfg.dropout,
        )

    def forward(self, query, key, value, memory, coords1, size, size_h3w3):
        """
        x:      [B*H1*W1, 1, C]
        memory: [B*H1*W1, H2'*W2', C]
        coords1 [B, 2, H2, W2]
        size: B, C, H1, W1
        1. Note that here coords0 and coords1 are in H2, W2 space.
           Should first convert it into H2', W2' space.
        2. We assume the upper-left point to be [0, 0], instead of letting center of upper-left patch to be [0, 0]
        """
        x_global, k, v = self.cross_attend(
            query, key, value, memory, coords1, self.patch_size, size_h3w3
        )
        B, C, H1, W1 = size
        C = self.cfg.query_latent_dim
        x_global = x_global.view(B, H1, W1, C).permute(0, 3, 1, 2)
        return x_global, k, v


class ReverseCostExtractor(nn.Module):
    def __init__(self, cfg):
        super(ReverseCostExtractor, self).__init__()
        self.cfg = cfg

    def forward(self, cost_maps, coords0, coords1):
        """
        cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
        coords      -   B, 2, H1, W1
        """
        BH1W1, heads, H2, W2 = cost_maps.shape
        B, _, H1, W1 = coords1.shape

        assert (H1 == H2) and (W1 == W2)
        assert BH1W1 == B * H1 * W1

        cost_maps = cost_maps.reshape(B, H1 * W1 * heads, H2, W2)
        coords = coords1.permute(0, 2, 3, 1)
        corr = bilinear_sampler(cost_maps, coords)  # [B, H1*W1*heads, H2, W2]
        corr = rearrange(
            corr,
            "b (h1 w1 heads) h2 w2 -> (b h2 w2) heads h1 w1",
            b=B,
            heads=heads,
            h1=H1,
            w1=W1,
            h2=H2,
            w2=W2,
        )

        r = 4
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords0.device)
        centroid = coords0.permute(0, 2, 3, 1).reshape(BH1W1, 1, 1, 2)
        delta = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(corr, coords)
        corr = corr.view(B, H1, W1, -1).permute(0, 3, 1, 2)
        return corr


class MemoryDecoder(nn.Module):
    def __init__(self, cfg):
        super(MemoryDecoder, self).__init__()
        dim = self.dim = cfg.query_latent_dim
        self.cfg = cfg

        self.flow_token_encoder = nn.Sequential(
            nn.Conv2d(81 * cfg.cost_heads_num, dim, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1),
        )
        self.proj = nn.Conv2d(256, 256, 1)
        self.depth = cfg.decoder_depth
        self.decoder_layer = MemoryDecoderLayer(dim, cfg)

        if self.cfg.gma:
            self.update_block = GMAUpdateBlock(self.cfg, hidden_dim=128)
            self.att = Attention(
                args=self.cfg, dim=128, heads=1, max_pos_size=160, dim_head=128
            )
        else:
            self.update_block = BasicUpdateBlock(self.cfg, hidden_dim=128)

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination"""
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def encode_flow_token(self, cost_maps, coords):
        """
        cost_maps   -   B*H1*W1, cost_heads_num, H2, W2
        coords      -   B, 2, H1, W1
        """
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        r = 4
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords = centroid + delta
        corr = bilinear_sampler(cost_maps, coords)
        corr = corr.view(batch, h1, w1, -1).permute(0, 3, 1, 2)
        return corr

    def forward(self, cost_memory, context, data={}, flow_init=None, iters=None):
        """
        memory: [B*H1*W1, H2'*W2', C]
        context: [B, D, H1, W1]
        """
        cost_maps = data["cost_maps"]
        coords0, coords1 = initialize_flow(context)

        if flow_init is not None:
            # print("[Using warm start]")
            coords1 = coords1 + flow_init

        # flow = coords1

        flow_predictions = []

        context = self.proj(context)
        net, inp = torch.split(context, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        if self.cfg.gma:
            attention = self.att(inp)

        size = net.shape
        key, value = None, None
        if iters is None:
            iters = self.depth
        for idx in range(iters):
            coords1 = coords1.detach()

            cost_forward = self.encode_flow_token(cost_maps, coords1)
            # cost_backward = self.reverse_cost_extractor(cost_maps, coords0, coords1)

            query = self.flow_token_encoder(cost_forward)
            query = (
                query.permute(0, 2, 3, 1)
                .contiguous()
                .view(size[0] * size[2] * size[3], 1, self.dim)
            )
            cost_global, key, value = self.decoder_layer(
                query, key, value, cost_memory, coords1, size, data["H3W3"]
            )
            if self.cfg.only_global:
                corr = cost_global
            else:
                corr = torch.cat([cost_global, cost_forward], dim=1)

            flow = coords1 - coords0

            if self.cfg.gma:
                net, up_mask, delta_flow = self.update_block(
                    net, inp, corr, flow, attention
                )
            else:
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # flow = delta_flow
            coords1 = coords1 + delta_flow
            flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        # if self.training:
        #     return flow_predictions
        # else:
        return flow_predictions[-1], coords1 - coords0
