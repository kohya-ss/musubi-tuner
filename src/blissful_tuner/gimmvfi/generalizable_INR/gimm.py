# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# motif: https://github.com/sichun233746/MoTIF
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import GIMMConfig
from .modules.coord_sampler import CoordSampler3D
from .modules.fi_components import LateralBlock
from .modules.hyponet import HypoNet
from .modules.fi_utils import warp

from .modules.softsplat import softsplat


class GIMM(nn.Module):
    Config = GIMMConfig

    def __init__(self, config: GIMMConfig):
        super().__init__()
        self.config = config = config.copy()
        self.hyponet_config = config.hyponet
        self.coord_sampler = CoordSampler3D(config.coord_range)
        self.fwarp_type = config.fwarp_type

        # Motion Encoder
        channel = 32
        in_dim = 2
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_dim, channel // 2, 3, 1, 1, bias=True, groups=1),
            nn.Conv2d(channel // 2, channel, 3, 1, 1, bias=True, groups=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            LateralBlock(channel),
            LateralBlock(channel),
            LateralBlock(channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channel, channel // 2, 3, 1, 1, padding_mode="reflect", bias=True
            ),
        )

        # Latent Refiner
        channel = 64
        in_dim = 64
        self.res_conv = nn.Sequential(
            nn.Conv2d(in_dim, channel // 2, 3, 1, 1, bias=True, groups=1),
            nn.Conv2d(channel // 2, channel, 3, 1, 1, bias=True, groups=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            LateralBlock(channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                channel, channel // 2, 3, 1, 1, padding_mode="reflect", bias=True
            ),
        )
        self.g_filter = torch.nn.Parameter(
            torch.FloatTensor(
                [
                    [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
                    [1.0 / 8.0, 1.0 / 4.0, 1.0 / 8.0],
                    [1.0 / 16.0, 1.0 / 8.0, 1.0 / 16.0],
                ]
            ).reshape(1, 1, 1, 3, 3),
            requires_grad=False,
        )
        self.alpha_v = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)
        self.alpha_fe = torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=True)

        self.hyponet = HypoNet(config.hyponet, add_coord_dim=32)

    def cal_splatting_weights(self, raft_flow01, raft_flow10):
        batch_size = raft_flow01.shape[0]
        raft_flows = torch.cat([raft_flow01, raft_flow10], dim=0)

        ## flow variance metric
        sqaure_mean, mean_square = torch.split(
            F.conv3d(
                F.pad(
                    torch.cat([raft_flows**2, raft_flows], 1),
                    (1, 1, 1, 1),
                    mode="reflect",
                ).unsqueeze(1),
                self.g_filter,
            ).squeeze(1),
            2,
            dim=1,
        )
        var = (
            (sqaure_mean - mean_square**2)
            .clamp(1e-9, None)
            .sqrt()
            .mean(1)
            .unsqueeze(1)
        )
        var01 = var[:batch_size]
        var10 = var[batch_size:]

        ## flow warp metirc
        f01_warp = -warp(raft_flow10, raft_flow01)
        f10_warp = -warp(raft_flow01, raft_flow10)
        err01 = (
            torch.nn.functional.l1_loss(
                input=f01_warp, target=raft_flow01, reduction="none"
            )
            .mean(1)
            .unsqueeze(1)
        )
        err02 = (
            torch.nn.functional.l1_loss(
                input=f10_warp, target=raft_flow10, reduction="none"
            )
            .mean(1)
            .unsqueeze(1)
        )

        weights1 = 1 / (1 + err01 * self.alpha_fe) + 1 / (1 + var01 * self.alpha_v)
        weights2 = 1 / (1 + err02 * self.alpha_fe) + 1 / (1 + var10 * self.alpha_v)

        return weights1, weights2

    def forward(
        self, xs, coord=None, keep_xs_shape=True, ori_flow=None, timesteps=None
    ):
        coord = self.sample_coord_input(xs) if coord is None else coord
        raft_flow01 = ori_flow[:, :, 0]
        raft_flow10 = ori_flow[:, :, 1]

        # calculate splatting metrics
        weights1, weights2 = self.cal_splatting_weights(raft_flow01, raft_flow10)
        # b,c,h,w
        pixel_latent_0 = self.cnn_encoder(xs[:, :, 0])
        pixel_latent_1 = self.cnn_encoder(xs[:, :, 1])
        pixel_latent = []

        modulation_params_dict = None
        strtype = self.fwarp_type
        if isinstance(timesteps, list):
            assert isinstance(coord, list)
            assert len(timesteps) == len(coord)
            for i, cur_t in enumerate(timesteps):
                cur_t = cur_t.reshape(-1, 1, 1, 1)
                tmp_pixel_latent_0 = softsplat(
                    tenIn=pixel_latent_0,
                    tenFlow=raft_flow01 * cur_t,
                    tenMetric=weights1,
                    strMode=strtype + "-zeroeps",
                )
                tmp_pixel_latent_1 = softsplat(
                    tenIn=pixel_latent_1,
                    tenFlow=raft_flow10 * (1 - cur_t),
                    tenMetric=weights2,
                    strMode=strtype + "-zeroeps",
                )
                tmp_pixel_latent = torch.cat(
                    [tmp_pixel_latent_0, tmp_pixel_latent_1], dim=1
                )
                tmp_pixel_latent = tmp_pixel_latent + self.res_conv(
                    torch.cat([pixel_latent_0, pixel_latent_1, tmp_pixel_latent], dim=1)
                )
                pixel_latent.append(tmp_pixel_latent.permute(0, 2, 3, 1))

            all_outputs = []
            for idx, c in enumerate(coord):
                outputs = self.hyponet(
                    c,
                    modulation_params_dict=modulation_params_dict,
                    pixel_latent=pixel_latent[idx],
                )
                if keep_xs_shape:
                    permute_idx_range = [i for i in range(1, xs.ndim - 1)]
                    outputs = outputs.permute(0, -1, *permute_idx_range)
                all_outputs.append(outputs)
            return all_outputs

        else:
            cur_t = timesteps.reshape(-1, 1, 1, 1)
            tmp_pixel_latent_0 = softsplat(
                tenIn=pixel_latent_0,
                tenFlow=raft_flow01 * cur_t,
                tenMetric=weights1,
                strMode=strtype + "-zeroeps",
            )
            tmp_pixel_latent_1 = softsplat(
                tenIn=pixel_latent_1,
                tenFlow=raft_flow10 * (1 - cur_t),
                tenMetric=weights2,
                strMode=strtype + "-zeroeps",
            )
            tmp_pixel_latent = torch.cat(
                [tmp_pixel_latent_0, tmp_pixel_latent_1], dim=1
            )
            tmp_pixel_latent = tmp_pixel_latent + self.res_conv(
                torch.cat([pixel_latent_0, pixel_latent_1, tmp_pixel_latent], dim=1)
            )
            pixel_latent = tmp_pixel_latent.permute(0, 2, 3, 1)

            # predict all pixels of coord after applying the modulation_parms into hyponet
            outputs = self.hyponet(
                coord,
                modulation_params_dict=modulation_params_dict,
                pixel_latent=pixel_latent,
            )
            if keep_xs_shape:
                permute_idx_range = [i for i in range(1, xs.ndim - 1)]
                outputs = outputs.permute(0, -1, *permute_idx_range)
            return outputs

    def compute_loss(self, preds, targets, reduction="mean", single=False):
        assert reduction in ["mean", "sum", "none"]
        batch_size = preds.shape[0]
        sample_mses = 0
        assert preds.shape[2] == 1
        assert targets.shape[2] == 1
        for i in range(preds.shape[2]):
            sample_mses += torch.reshape(
                (preds[:, :, i] - targets[:, :, i]) ** 2, (batch_size, -1)
            ).mean(dim=-1)
        sample_mses = sample_mses / preds.shape[2]
        if reduction == "mean":
            total_loss = sample_mses.mean()
            psnr = (-10 * torch.log10(sample_mses)).mean()
        elif reduction == "sum":
            total_loss = sample_mses.sum()
            psnr = (-10 * torch.log10(sample_mses)).sum()
        else:
            total_loss = sample_mses
            psnr = -10 * torch.log10(sample_mses)

        return {"loss_total": total_loss, "mse": total_loss, "psnr": psnr}

    def sample_coord_input(
        self,
        batch_size,
        s_shape,
        t_ids,
        coord_range=None,
        upsample_ratio=1.0,
        device=None,
    ):
        assert device is not None
        assert coord_range is None
        coord_inputs = self.coord_sampler(
            batch_size, s_shape, t_ids, coord_range, upsample_ratio, device
        )
        return coord_inputs
