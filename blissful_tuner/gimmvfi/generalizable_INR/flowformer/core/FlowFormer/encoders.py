import torch
import torch.nn as nn
import timm
import numpy as np


class twins_svt_large(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model("twins_svt_large", pretrained=pretrained)

        del self.svt.head
        del self.svt.patch_embeds[2]
        del self.svt.patch_embeds[2]
        del self.svt.blocks[2]
        del self.svt.blocks[2]
        del self.svt.pos_block[2]
        del self.svt.pos_block[2]
        self.svt.norm.weight.requires_grad = False
        self.svt.norm.bias.requires_grad = False

    def forward(self, x, data=None, layer=2, return_feat=False):
        B = x.shape[0]
        if return_feat:
            feat = []
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()
            if return_feat:
                feat.append(x)
            if i == layer - 1:
                break
        if return_feat:
            return x, feat
        return x

    def compute_params(self, layer=2):
        num = 0
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):
            for param in embed.parameters():
                num += np.prod(param.size())

            for param in drop.parameters():
                num += np.prod(param.size())

            for param in blocks.parameters():
                num += np.prod(param.size())

            for param in pos_blk.parameters():
                num += np.prod(param.size())

            if i == layer - 1:
                break

        for param in self.svt.head.parameters():
            num += np.prod(param.size())

        return num


class twins_svt_large_context(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.svt = timm.create_model("twins_svt_large_context", pretrained=pretrained)

    def forward(self, x, data=None, layer=2):
        B = x.shape[0]
        for i, (embed, drop, blocks, pos_blk) in enumerate(
            zip(
                self.svt.patch_embeds,
                self.svt.pos_drops,
                self.svt.blocks,
                self.svt.pos_block,
            )
        ):
            x, size = embed(x)
            x = drop(x)
            for j, blk in enumerate(blocks):
                x = blk(x, size)
                if j == 0:
                    x = pos_blk(x, size)
            if i < len(self.svt.depths) - 1:
                x = x.reshape(B, *size, -1).permute(0, 3, 1, 2).contiguous()

            if i == layer - 1:
                break

        return x


if __name__ == "__main__":
    m = twins_svt_large()
    input = torch.randn(2, 3, 400, 800)
    out = m.extract_feature(input)
    print(out.shape)
