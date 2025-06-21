# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# amt: https://github.com/MCG-NKU/AMT
# raft: https://github.com/princeton-vl/RAFT
# --------------------------------------------------------

import torch
import torch.nn.functional as F
from .utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class BidirCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.corr_pyramid_T = []

        corr = BidirCorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr_T = corr.clone().permute(0, 4, 5, 3, 1, 2)

        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        corr_T = corr_T.reshape(batch * h2 * w2, dim, h1, w1)

        self.corr_pyramid.append(corr)
        self.corr_pyramid_T.append(corr_T)

        for _ in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            corr_T = F.avg_pool2d(corr_T, 2, stride=2)
            self.corr_pyramid.append(corr)
            self.corr_pyramid_T.append(corr_T)

    def __call__(self, coords0, coords1):
        r = self.radius
        coords0 = coords0.permute(0, 2, 3, 1)
        coords1 = coords1.permute(0, 2, 3, 1)
        assert (
            coords0.shape == coords1.shape
        ), f"coords0 shape: [{coords0.shape}] is not equal to [{coords1.shape}]"
        batch, h1, w1, _ = coords0.shape

        out_pyramid = []
        out_pyramid_T = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            corr_T = self.corr_pyramid_T[i]

            dx = torch.linspace(-r, r, 2 * r + 1, device=coords0.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords0.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)

            centroid_lvl_0 = coords0.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            centroid_lvl_1 = coords1.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            coords_lvl_0 = centroid_lvl_0 + delta_lvl
            coords_lvl_1 = centroid_lvl_1 + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl_0)
            corr_T = bilinear_sampler(corr_T, coords_lvl_1)
            corr = corr.view(batch, h1, w1, -1)
            corr_T = corr_T.view(batch, h1, w1, -1)
            out_pyramid.append(corr)
            out_pyramid_T.append(corr_T)

        out = torch.cat(out_pyramid, dim=-1)
        out_T = torch.cat(out_pyramid_T, dim=-1)
        return (
            out.permute(0, 3, 1, 2).contiguous().float(),
            out_T.permute(0, 3, 1, 2).contiguous().float(),
        )

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            (corr,) = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            dy = torch.linspace(-r, r, 2 * r + 1, device=coords.device)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())
