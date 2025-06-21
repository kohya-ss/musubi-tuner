# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from torch import nn
import torch


# define siren layer & Siren model
class Sine(nn.Module):
    """Sine activation with scaling.

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


# Damping activation from http://arxiv.org/abs/2306.15242
class Damping(nn.Module):
    """Sine activation with sublinear factor

    Args:
        w0 (float): Omega_0 parameter from SIREN paper.
    """

    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        x = torch.clamp(x, min=1e-30)
        return torch.sin(self.w0 * x) * torch.sqrt(x.abs())
