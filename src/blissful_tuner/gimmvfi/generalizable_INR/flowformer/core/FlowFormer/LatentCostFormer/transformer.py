import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange

from ...utils.utils import coords_grid, bilinear_sampler, upflow8
from ..common import (
    FeedForward,
    pyramid_retrieve_tokens,
    sampler,
    sampler_gaussian_fix,
    retrieve_tokens,
    MultiHeadAttention,
    MLP,
)
from ..encoders import twins_svt_large_context, twins_svt_large
from ...position_encoding import PositionEncodingSine, LinearPositionEncoding
from .twins import PosConv
from .encoder import MemoryEncoder
from .decoder import MemoryDecoder
from .cnn import BasicEncoder


class FlowFormer(nn.Module):
    def __init__(self, cfg):
        super(FlowFormer, self).__init__()
        self.cfg = cfg

        self.memory_encoder = MemoryEncoder(cfg)
        self.memory_decoder = MemoryDecoder(cfg)
        if cfg.cnet == "twins":
            self.context_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.cnet == "basicencoder":
            self.context_encoder = BasicEncoder(output_dim=256, norm_fn="instance")

    def build_coord(self, img):
        N, C, H, W = img.shape
        coords = coords_grid(N, H // 8, W // 8)
        return coords

    def forward(
        self, image1, image2, output=None, flow_init=None, return_feat=False, iters=None
    ):
        # Following https://github.com/princeton-vl/RAFT/
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        data = {}

        if self.cfg.context_concat:
            context = self.context_encoder(torch.cat([image1, image2], dim=1))
        else:
            if return_feat:
                context, cfeat = self.context_encoder(image1, return_feat=return_feat)
            else:
                context = self.context_encoder(image1)
        if return_feat:
            cost_memory, ffeat = self.memory_encoder(
                image1, image2, data, context, return_feat=return_feat
            )
        else:
            cost_memory = self.memory_encoder(image1, image2, data, context)

        flow_predictions = self.memory_decoder(
            cost_memory, context, data, flow_init=flow_init, iters=iters
        )

        if return_feat:
            return flow_predictions, cfeat, ffeat
        return flow_predictions
