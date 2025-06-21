# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# amt: https://github.com/MCG-NKU/AMT
# motif: https://github.com/sichun233746/MoTIF
# --------------------------------------------------------

import torch
import torch.nn as nn
from .fi_utils import warp, resize


class LateralBlock(nn.Module):
    def __init__(self, dim):
        super(LateralBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=True),
        )

    def forward(self, x):
        res = x
        x = self.layers(x)
        return x + res


def convrelu(
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=True,
):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        ),
        nn.PReLU(out_channels),
    )


def multi_flow_combine(
    comb_block, img0, img1, flow0, flow1, mask=None, img_res=None, mean=None
):
    assert mean is None
    b, c, h, w = flow0.shape
    num_flows = c // 2
    flow0 = flow0.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)
    flow1 = flow1.reshape(b, num_flows, 2, h, w).reshape(-1, 2, h, w)

    mask = (
        mask.reshape(b, num_flows, 1, h, w).reshape(-1, 1, h, w)
        if mask is not None
        else None
    )
    img_res = (
        img_res.reshape(b, num_flows, 3, h, w).reshape(-1, 3, h, w)
        if img_res is not None
        else 0
    )
    img0 = torch.stack([img0] * num_flows, 1).reshape(-1, 3, h, w)
    img1 = torch.stack([img1] * num_flows, 1).reshape(-1, 3, h, w)
    mean = (
        torch.stack([mean] * num_flows, 1).reshape(-1, 1, 1, 1)
        if mean is not None
        else 0
    )

    img0_warp = warp(img0, flow0)
    img1_warp = warp(img1, flow1)
    img_warps = mask * img0_warp + (1 - mask) * img1_warp + mean + img_res
    img_warps = img_warps.reshape(b, num_flows, 3, h, w)

    res = comb_block(img_warps.view(b, -1, h, w))
    imgt_pred = img_warps.mean(1) + res

    imgt_pred = (imgt_pred + 1.0) / 2

    return imgt_pred


class ResBlock(nn.Module):
    def __init__(self, in_channels, side_channels, bias=True):
        super(ResBlock, self).__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.PReLU(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.PReLU(side_channels),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.PReLU(in_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                side_channels,
                side_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            nn.PReLU(side_channels),
        )
        self.conv5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )
        self.prelu = nn.PReLU(in_channels)

    def forward(self, x):
        out = self.conv1(x)

        res_feat = out[:, : -self.side_channels, ...]
        side_feat = out[:, -self.side_channels :, :, :]
        side_feat = self.conv2(side_feat)
        out = self.conv3(torch.cat([res_feat, side_feat], 1))

        res_feat = out[:, : -self.side_channels, ...]
        side_feat = out[:, -self.side_channels :, :, :]
        side_feat = self.conv4(side_feat)
        out = self.conv5(torch.cat([res_feat, side_feat], 1))

        out = self.prelu(x + out)
        return out


class BasicUpdateBlock(nn.Module):
    def __init__(
        self,
        cdim,
        hidden_dim,
        flow_dim,
        corr_dim,
        corr_dim2,
        fc_dim,
        corr_levels=4,
        radius=3,
        scale_factor=None,
        out_num=1,
    ):
        super(BasicUpdateBlock, self).__init__()
        cor_planes = corr_levels * (2 * radius + 1) ** 2

        self.scale_factor = scale_factor
        self.convc1 = nn.Conv2d(2 * cor_planes, corr_dim, 1, padding=0)
        self.convc2 = nn.Conv2d(corr_dim, corr_dim2, 3, padding=1)
        self.convf1 = nn.Conv2d(4, flow_dim * 2, 7, padding=3)
        self.convf2 = nn.Conv2d(flow_dim * 2, flow_dim, 3, padding=1)
        self.conv = nn.Conv2d(flow_dim + corr_dim2, fc_dim, 3, padding=1)

        self.gru = nn.Sequential(
            nn.Conv2d(fc_dim + 4 + cdim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, cdim, 3, padding=1),
        )

        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(hidden_dim, 4 * out_num, 3, padding=1),
        )

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, net, flow, corr):
        net = (
            resize(net, 1 / self.scale_factor) if self.scale_factor is not None else net
        )
        cor = self.lrelu(self.convc1(corr))
        cor = self.lrelu(self.convc2(cor))
        flo = self.lrelu(self.convf1(flow))
        flo = self.lrelu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        inp = self.lrelu(self.conv(cor_flo))
        inp = torch.cat([inp, flow, net], dim=1)

        out = self.gru(inp)
        delta_net = self.feat_head(out)
        delta_flow = self.flow_head(out)

        if self.scale_factor is not None:
            delta_net = resize(delta_net, scale_factor=self.scale_factor)
            delta_flow = self.scale_factor * resize(
                delta_flow, scale_factor=self.scale_factor
            )
        return delta_net, delta_flow


def get_bn():
    return nn.BatchNorm2d


class NewInitDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch):
        super().__init__()
        norm_layer = get_bn()

        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            convrelu(in_ch // 4, in_ch // 4, 5, 1, 2),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 2),
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),
            norm_layer(in_ch // 2),
            nn.ReLU(inplace=True),
        )

        in_ch = in_ch // 2
        self.convblock = nn.Sequential(
            convrelu(in_ch * 2 + 16, in_ch, kernel_size=1, padding=0),
            ResBlock(in_ch, skip_ch),
            ResBlock(in_ch, skip_ch),
            ResBlock(in_ch, skip_ch),
            nn.Conv2d(in_ch, in_ch + 5, 3, 1, 1, 1, 1, True),
        )

    def forward(self, f0, f1, flow0_in, flow1_in, img0=None, img1=None):
        f0 = self.upsample(f0)
        f1 = self.upsample(f1)
        f0_warp_ks = warp(f0, flow0_in)
        f1_warp_ks = warp(f1, flow1_in)

        f_in = torch.cat([f0_warp_ks, f1_warp_ks, flow0_in, flow1_in], dim=1)

        assert img0 is not None
        assert img1 is not None
        scale_factor = f_in.shape[2] / img0.shape[2]
        img0 = resize(img0, scale_factor=scale_factor)
        img1 = resize(img1, scale_factor=scale_factor)
        warped_img0 = warp(img0, flow0_in)
        warped_img1 = warp(img1, flow1_in)
        f_in = torch.cat([f_in, img0, img1, warped_img0, warped_img1], dim=1)

        out = self.convblock(f_in)
        ft_ = out[:, 4:, ...]
        flow0 = flow0_in + out[:, :2, ...]
        flow1 = flow1_in + out[:, 2:4, ...]
        return flow0, flow1, ft_


class NewMultiFlowDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, num_flows=3):
        super(NewMultiFlowDecoder, self).__init__()
        norm_layer = get_bn()

        self.upsample = nn.Sequential(
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            convrelu(in_ch // (4 * 4), in_ch // 4, 5, 1, 2),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 4),
            convrelu(in_ch // 4, in_ch // 2),
            nn.Conv2d(in_ch // 2, in_ch // 2, kernel_size=1),
            norm_layer(in_ch // 2),
            nn.ReLU(inplace=True),
        )

        self.num_flows = num_flows
        ch_factor = 2
        self.convblock = nn.Sequential(
            convrelu(in_ch * ch_factor + 17, in_ch * ch_factor),
            ResBlock(in_ch * ch_factor, skip_ch),
            ResBlock(in_ch * ch_factor, skip_ch),
            ResBlock(in_ch * ch_factor, skip_ch),
            nn.Conv2d(in_ch * ch_factor, 8 * num_flows, kernel_size=3, padding=1),
        )

    def forward(self, ft_, f0, f1, flow0, flow1, mask=None, img0=None, img1=None):
        f0 = self.upsample(f0)
        # print([f1.shape,f0.shape])
        f1 = self.upsample(f1)
        n = self.num_flows
        flow0 = 4.0 * resize(flow0, scale_factor=4.0)
        flow1 = 4.0 * resize(flow1, scale_factor=4.0)

        ft_ = resize(ft_, scale_factor=4.0)
        mask = resize(mask, scale_factor=4.0)
        f0_warp = warp(f0, flow0)
        f1_warp = warp(f1, flow1)

        f_in = torch.cat([ft_, f0_warp, f1_warp, flow0, flow1], 1)

        assert mask is not None
        f_in = torch.cat([f_in, mask], 1)

        assert img0 is not None
        assert img1 is not None
        warped_img0 = warp(img0, flow0)
        warped_img1 = warp(img1, flow1)
        f_in = torch.cat([f_in, img0, img1, warped_img0, warped_img1], dim=1)

        out = self.convblock(f_in)
        delta_flow0, delta_flow1, delta_mask, img_res = torch.split(
            out, [2 * n, 2 * n, n, 3 * n], 1
        )
        mask = delta_mask + mask.repeat(1, self.num_flows, 1, 1)
        mask = torch.sigmoid(mask)
        flow0 = delta_flow0 + flow0.repeat(1, self.num_flows, 1, 1)
        flow1 = delta_flow1 + flow1.repeat(1, self.num_flows, 1, 1)

        return flow0, flow1, mask, img_res
