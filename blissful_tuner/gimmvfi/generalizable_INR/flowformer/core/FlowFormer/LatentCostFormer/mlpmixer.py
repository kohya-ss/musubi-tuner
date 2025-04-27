from torch import nn
from einops.layers.torch import Rearrange, Reduce
from functools import partial
import numpy as np


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.0, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )


class MLPMixerLayer(nn.Module):
    def __init__(self, dim, cfg, drop_path=0.0, dropout=0.0):
        super(MLPMixerLayer, self).__init__()

        # print(f"use mlp mixer layer")
        K = cfg.cost_latent_token_num
        expansion_factor = cfg.mlp_expansion_factor
        chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

        self.mlpmixer = nn.Sequential(
            PreNormResidual(dim, FeedForward(K, expansion_factor, dropout, chan_first)),
            PreNormResidual(
                dim, FeedForward(dim, expansion_factor, dropout, chan_last)
            ),
        )

    def compute_params(self):
        num = 0
        for param in self.mlpmixer.parameters():
            num += np.prod(param.size())

        return num

    def forward(self, x):
        """
        x: [BH1W1, K, D]
        """

        return self.mlpmixer(x)
