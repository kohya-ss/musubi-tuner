import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock, SmallUpdateBlock
from .extractor import BasicEncoder, SmallEncoder
from .corr import BidirCorrBlock, AlternateCorrBlock
from .utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


# BiRAFT
class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
            self.corr_levels = 4
            self.corr_radius = 3

        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4
            self.corr_levels = 4
            self.corr_radius = 4

        if "dropout" not in args._get_kwargs():
            self.args.dropout = 0

        if "alternate_corr" not in args._get_kwargs():
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(
                output_dim=128, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = SmallEncoder(
                output_dim=hdim + cdim, norm_fn="none", dropout=args.dropout
            )
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(
                output_dim=256, norm_fn="instance", dropout=args.dropout
            )
            self.cnet = BasicEncoder(
                output_dim=hdim + cdim, norm_fn="batch", dropout=args.dropout
            )
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def build_coord(self, img):
        N, C, H, W = img.shape
        coords = coords_grid(N, H // 8, W // 8, device=img.device)
        return coords

    def initialize_flow(self, img, img2):
        """Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        assert img.shape == img2.shape
        N, C, H, W = img.shape
        coords01 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords02 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords1 = coords_grid(N, H // 8, W // 8, device=img.device)
        coords2 = coords_grid(N, H // 8, W // 8, device=img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords01, coords02, coords1, coords2

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

    def get_corr_fn(self, image1, image2, projector=None):
        # run the feature network
        with autocast(enabled=self.args.mixed_precision):
            fmaps, feats = self.fnet([image1, image2], return_feature=True)
        fmap1, fmap2 = fmaps
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        corr_fn1 = None
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            if projector is not None:
                corr_fn1 = AlternateCorrBlock(
                    projector(feats[-1][0]),
                    projector(feats[-1][1]),
                    radius=self.args.corr_radius,
                )
        else:
            corr_fn = BidirCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
            if projector is not None:
                corr_fn1 = BidirCorrBlock(
                    projector(feats[-1][0]),
                    projector(feats[-1][1]),
                    radius=self.args.corr_radius,
                )
        if corr_fn1 is None:
            return corr_fn, corr_fn
        else:
            return corr_fn, corr_fn1

    def get_corr_fn_from_feat(self, fmap1, fmap2):
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.args.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        else:
            corr_fn = BidirCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        return corr_fn

    def forward(
        self,
        image1,
        image2,
        iters=12,
        flow_init=None,
        upsample=True,
        test_mode=False,
        corr_fn=None,
        mif=False,
    ):
        """Estimate optical flow between pair of frames"""
        assert flow_init is None

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        if corr_fn is None:
            corr_fn, _ = self.get_corr_fn(image1, image2)

        # # run the feature network
        # with autocast(enabled=self.args.mixed_precision):
        #     fmap1, fmap2 = self.fnet([image1, image2])

        # fmap1 = fmap1.float()
        # fmap2 = fmap2.float()
        # if self.args.alternate_corr:
        #     corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        # else:
        #     corr_fn = BidirCorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            # for image1
            cnet1, features1 = self.cnet(image1, return_feature=True, mif=mif)
            net1, inp1 = torch.split(cnet1, [hdim, cdim], dim=1)
            net1 = torch.tanh(net1)
            inp1 = torch.relu(inp1)
            # for image2
            cnet2, features2 = self.cnet(image2, return_feature=True, mif=mif)
            net2, inp2 = torch.split(cnet2, [hdim, cdim], dim=1)
            net2 = torch.tanh(net2)
            inp2 = torch.relu(inp2)

        coords01, coords02, coords1, coords2 = self.initialize_flow(image1, image2)

        # if flow_init is not None:
        #     coords1 = coords1 + flow_init

        # flow_predictions1 = []
        # flow_predictions2 = []
        for itr in range(iters):
            coords1 = coords1.detach()
            coords2 = coords2.detach()
            corr1, corr2 = corr_fn(coords1, coords2)  # index correlation volume

            flow1 = coords1 - coords01
            flow2 = coords2 - coords02

            with autocast(enabled=self.args.mixed_precision):
                net1, up_mask1, delta_flow1 = self.update_block(
                    net1, inp1, corr1, flow1
                )
                net2, up_mask2, delta_flow2 = self.update_block(
                    net2, inp2, corr2, flow2
                )

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow1
            coords2 = coords2 + delta_flow2
            flow_low1 = coords1 - coords01
            flow_low2 = coords2 - coords02
            # upsample predictions
            if up_mask1 is None:
                flow_up1 = upflow8(coords1 - coords01)
                flow_up2 = upflow8(coords2 - coords02)
            else:
                flow_up1 = self.upsample_flow(coords1 - coords01, up_mask1)
                flow_up2 = self.upsample_flow(coords2 - coords02, up_mask2)

            # flow_predictions.append(flow_up)
        return flow_up1, flow_up2, flow_low1, flow_low2, features1, features2
        # if test_mode:
        #     return coords1 - coords0, flow_up

        # return flow_predictions
