# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# ginr-ipc: https://github.com/kakaobrain/ginr-ipc
# --------------------------------------------------------

import torch
import torch.nn as nn


class CoordSampler3D(nn.Module):
    def __init__(self, coord_range, t_coord_only=False):
        super().__init__()
        self.coord_range = coord_range
        self.t_coord_only = t_coord_only

    def shape2coordinate(
        self,
        batch_size,
        spatial_shape,
        t_ids,
        coord_range=(-1.0, 1.0),
        upsample_ratio=1,
        device=None,
    ):
        coords = []
        assert isinstance(t_ids, list)
        _coords = torch.tensor(t_ids, device=device) / 1.0
        coords.append(_coords.to(torch.float32))
        for num_s in spatial_shape:
            num_s = int(num_s * upsample_ratio)
            _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
            _coords = coord_range[0] + (coord_range[1] - coord_range[0]) * _coords
            coords.append(_coords)
        coords = torch.meshgrid(*coords, indexing="ij")
        coords = torch.stack(coords, dim=-1)
        ones_like_shape = (1,) * coords.ndim
        coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
        return coords  # (B,T,H,W,3)

    def batchshape2coordinate(
        self,
        batch_size,
        spatial_shape,
        t_ids,
        coord_range=(-1.0, 1.0),
        upsample_ratio=1,
        device=None,
    ):
        coords = []
        _coords = torch.tensor(1, device=device)
        coords.append(_coords.to(torch.float32))
        for num_s in spatial_shape:
            num_s = int(num_s * upsample_ratio)
            _coords = (0.5 + torch.arange(num_s, device=device)) / num_s
            _coords = coord_range[0] + (coord_range[1] - coord_range[0]) * _coords
            coords.append(_coords)
        coords = torch.meshgrid(*coords, indexing="ij")
        coords = torch.stack(coords, dim=-1)
        ones_like_shape = (1,) * coords.ndim
        # Now coords b,1,h,w,3, coords[...,0]=1.
        coords = coords.unsqueeze(0).repeat(batch_size, *ones_like_shape)
        # assign per-sample timestep within the batch
        coords[..., :1] = coords[..., :1] * t_ids.reshape(-1, 1, 1, 1, 1)
        return coords

    def forward(
        self,
        batch_size,
        s_shape,
        t_ids,
        coord_range=None,
        upsample_ratio=1.0,
        device=None,
    ):
        coord_range = self.coord_range if coord_range is None else coord_range
        if isinstance(t_ids, list):
            coords = self.shape2coordinate(
                batch_size, s_shape, t_ids, coord_range, upsample_ratio, device
            )
        elif isinstance(t_ids, torch.Tensor):
            coords = self.batchshape2coordinate(
                batch_size, s_shape, t_ids, coord_range, upsample_ratio, device
            )
        if self.t_coord_only:
            coords = coords[..., :1]
        return coords
