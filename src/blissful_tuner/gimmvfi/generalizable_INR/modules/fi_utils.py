# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# raft: https://github.com/princeton-vl/RAFT
# ema-vfi: https://github.com/MCG-NJU/EMA-VFI
# --------------------------------------------------------

import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


def warp(tenInput, tenFlow):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device)
            .view(1, 1, 1, tenFlow.shape[3])
            .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        )
        tenVertical = (
            torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device)
            .view(1, 1, tenFlow.shape[2], 1)
            .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        )
        backwarp_tenGrid[k] = torch.cat([tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        1,
    )

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(
        input=tenInput,
        grid=g,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def normalize_flow(flows):
    # FIXME: MULTI-DIMENSION
    flow_scaler = torch.max(torch.abs(flows).flatten(1), dim=-1)[0].reshape(
        -1, 1, 1, 1, 1
    )
    flows = flows / flow_scaler  # [-1,1]
    # # Adapt to [0,1]
    flows = (flows + 1.0) / 2.0
    return flows, flow_scaler


def unnormalize_flow(flows, flow_scaler):
    return (flows * 2.0 - 1.0) * flow_scaler


def resize(x, scale_factor):
    return F.interpolate(
        x, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def build_coord(img):
    N, C, H, W = img.shape
    coords = coords_grid(N, H // 8, W // 8)
    return coords
