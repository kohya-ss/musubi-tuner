# TODO reimplement this file to avoid license issues

# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.


from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from safetensors.torch import load_file
from accelerate import init_empty_weights

from musubi_tuner.qwen_image.qwen_image_autoencoder_kl import DiagonalGaussianDistribution

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MEMORY_LIMIT = 512 * 1024**2  # 512MB
# MEMORY_LIMIT = 64 * 1024**2  # 64MB


# Optimized implementation of CogVideoXSafeConv3d
# https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L38
class PatchCausalConv3d(nn.Conv3d):
    r"""Causal Conv3d with efficient patch processing for large tensors."""

    def find_split_indices(self, seq_len, part_num):
        ideal_interval = seq_len / part_num
        possible_indices = list(range(0, seq_len, self.stride[0]))
        selected_indices = []

        for i in range(1, part_num):
            closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
            if closest not in selected_indices:
                selected_indices.append(closest)

        merged_indices = []
        prev_idx = 0
        for idx in selected_indices:
            if idx - prev_idx >= self.kernel_size[0]:
                merged_indices.append(idx)
                prev_idx = idx

        return merged_indices

    def forward(self, input):
        T = input.shape[2]  # input: NCTHW
        # memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3 # original value
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / MEMORY_LIMIT  # modified to use MEMORY_LIMIT
        part_num = int(memory_count / 2) + 1

        # T is large enough, and memory is sufficient, and we can split into multiple parts
        if T > self.kernel_size[0] and memory_count > 0.6 and part_num >= 2:
            if part_num > T // self.kernel_size[0]:
                part_num = T // self.kernel_size[0]
            kernel_size = self.kernel_size[0]
            split_indices = self.find_split_indices(T, part_num)

            # # original implementation. kept for reference
            # input_chunks = torch.tensor_split(input, split_indices, dim=2) if len(split_indices) > 0 else [input]
            # if kernel_size > 1:
            #     input_chunks = [input_chunks[0]] + [
            #         torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
            #         for i in range(1, len(input_chunks))
            #     ]

            # optimized implementation
            if len(split_indices) == 0 or kernel_size == 1:
                input_chunks = torch.tensor_split(input, split_indices, dim=2) if len(split_indices) > 0 else [input]
            else:
                boundaries = [0] + split_indices + [T]
                input_chunks = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    overlap_start = max(start - kernel_size + 1, 0)
                    if i == 0:
                        input_chunks.append(input[:, :, start:end])
                    else:
                        input_chunks.append(input[:, :, overlap_start:end])

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class PatchConv3d(nn.Conv3d):
    r"""Conv3d with efficient patch processing for large tensors."""

    def forward(self, input):
        assert self.kernel_size[0] == 1 and self.kernel_size[1] == 1 and self.kernel_size[2] == 1, (
            "PatchConv3d only supports kernel_size=1 for now."
        )
        assert self.stride[0] == 1 and self.stride[1] == 1 and self.stride[2] == 1, "PatchConv3d only supports stride=1 for now."
        assert self.padding[0] == 0 and self.padding[1] == 0 and self.padding[2] == 0, (
            "PatchConv3d only supports padding=0 for now."
        )

        T = input.shape[2]  # input: NCTHW
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / MEMORY_LIMIT
        part_num = int(memory_count / 2) + 1

        # T is large enough, and memory is sufficient, and we can split into multiple parts.
        if T > self.kernel_size[0] and memory_count > 0.6 and part_num >= 2:
            input_chunks = torch.tensor_split(input, part_num, dim=2)
            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class RMS_norm(nn.Module):
    """Root Mean Square Layer Normalization for Channel-First"""

    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class CausalConv3d(nn.Module):
    """Causal Conv3d with configurable padding for temporal axis."""

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode="replicate",
        disable_causal=False,
        enable_patch_conv=False,
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2)
        else:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        if enable_patch_conv:
            self.conv = PatchCausalConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        else:
            self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    """Prepare a causal attention mask for 3D videos.

    Args:
        n_frame (int): Number of frames (temporal length).
        n_hw (int): Product of height and width.
        dtype: Desired mask dtype.
        device: Device for the mask.
        batch_size (int, optional): If set, expands for batch.

    Returns:
        torch.Tensor: Causal attention mask.
    """
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class AttnBlock(nn.Module):
    """Self-attention block for 3D video tensors."""

    def __init__(self, in_channels: int, enable_patch_conv: bool = False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMS_norm(in_channels)

        conv_module = PatchConv3d if enable_patch_conv else nn.Conv3d
        self.q = conv_module(in_channels, in_channels, kernel_size=1)
        self.k = conv_module(in_channels, in_channels, kernel_size=1)
        self.v = conv_module(in_channels, in_channels, kernel_size=1)
        self.proj_out = conv_module(in_channels, in_channels, kernel_size=1)

    def sliced_attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        seq_hw = h * w
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()

        out = torch.empty_like(q)
        for frame in range(f):
            frm_start = frame * seq_hw
            frm_end = frm_start + seq_hw
            q_slice = q[:, :, frm_start:frm_end, :]
            k_slice = k[:, :, :frm_end, :]
            v_slice = v[:, :, :frm_end, :]
            out[:, :, frm_start:frm_end, :] = nn.functional.scaled_dot_product_attention(q_slice, k_slice, v_slice)

        out = rearrange(out, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w)
        return out

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = prepare_causal_attention_mask(f, h * w, h_.dtype, h_.device, batch_size=b)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        # return x + self.proj_out(self.attention(x))
        return x + self.proj_out(self.sliced_attention(x))


class ResnetBlock(nn.Module):
    """ResNet-style block for 3D video tensors."""

    def __init__(self, in_channels: int, out_channels: int, enable_patch_conv: bool = False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = RMS_norm(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)

        self.norm2 = RMS_norm(out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)
        if self.in_channels != self.out_channels:
            conv_module = PatchConv3d if enable_patch_conv else nn.Conv3d
            self.nin_shortcut = conv_module(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = F.silu(h, inplace=True)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h, inplace=True)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True, enable_patch_conv: bool = False):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3, enable_patch_conv=enable_patch_conv)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def _forward_fast(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape

            # x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W)
            if self.group_size <= 2:
                x_first = x_first[:, :, 0]
            elif self.group_size == 4:
                x_first = x_first[:, :, 0].add_(x_first[:, :, 1]).mul_(0.5)  # manual mean for group_size=4
            elif self.group_size == 8:
                # manual mean for group_size=8
                x_first = x_first[:, :, 0].add_(x_first[:, :, 1]).add_(x_first[:, :, 2]).add_(x_first[:, :, 3]).mul_(0.25)
            else:
                assert False, f"Unsupported group_size: {self.group_size}"

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape

            # x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W)
            if self.group_size == 1:
                x_next = x_next[:, :, 0]
            elif self.group_size == 2:
                x_next = x_next[:, :, 0].add_(x_next[:, :, 1]).mul_(0.5)  # manual mean for group_size=2
            elif self.group_size == 4:
                # manual mean for group_size=4
                x_next = x_next[:, :, 0].add_(x_next[:, :, 1]).add_(x_next[:, :, 2]).add_(x_next[:, :, 3]).mul_(0.25)
            elif self.group_size == 8:
                # manual mean for group_size=8
                x_next = (
                    x_next[:, :, 0]
                    .add_(x_next[:, :, 1])
                    .add_(x_next[:, :, 2])
                    .add_(x_next[:, :, 3])
                    .add_(x_next[:, :, 4])
                    .add_(x_next[:, :, 5])
                    .add_(x_next[:, :, 6])
                    .add_(x_next[:, :, 7])
                    .mul_(0.125)
                )
            else:
                assert False, f"Unsupported group_size: {self.group_size}"

            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape

            # shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W)
            if self.group_size == 1:
                shortcut = shortcut[:, :, 0]
            elif self.group_size == 2:
                shortcut = shortcut[:, :, 0].add_(shortcut[:, :, 1]).mul_(0.5)  # manual mean for group_size=2
            elif self.group_size == 4:
                # manual mean for group_size=4
                shortcut = shortcut[:, :, 0].add_(shortcut[:, :, 1]).add_(shortcut[:, :, 2]).add_(shortcut[:, :, 3]).mul_(0.25)
            elif self.group_size == 8:
                # manual mean for group_size=8
                shortcut = (
                    shortcut[:, :, 0]
                    .add_(shortcut[:, :, 1])
                    .add_(shortcut[:, :, 2])
                    .add_(shortcut[:, :, 3])
                    .add_(shortcut[:, :, 4])
                    .add_(shortcut[:, :, 5])
                    .add_(shortcut[:, :, 6])
                    .add_(shortcut[:, :, 7])
                    .mul_(0.125)
                )
            else:
                assert False, f"Unsupported group_size: {self.group_size}"

        return h + shortcut

    def _forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_downsample else 1
        h = self.conv(x)
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut

    def forward(self, x: Tensor):
        return self._forward_fast(x)


class Upsample(nn.Module):
    """Hierarchical upsampling with temporal/ spatial support."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True, enable_patch_conv: bool = False):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3, enable_patch_conv=enable_patch_conv)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor):
        r1 = 2 if self.add_temporal_upsample else 1
        h = self.conv(x)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)

        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    """Hierarchical video encoder with temporal and spatial factorization."""

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
        enable_patch_conv: bool = False,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, enable_patch_conv=enable_patch_conv)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, enable_patch_conv=enable_patch_conv))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(i_level >= np.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = Downsample(block_in, block_out, add_temporal_downsample, enable_patch_conv=enable_patch_conv)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv)

        # end
        self.norm_out = RMS_norm(block_in)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder."""

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
            if hasattr(self.down[i_level], "downsample"):
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = F.silu(h, inplace=True)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """Hierarchical video decoder with upsampling factories."""

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
        enable_patch_conv: bool = False,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, enable_patch_conv=enable_patch_conv)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv)
        self.mid.attn_1 = AttnBlock(block_in, enable_patch_conv=enable_patch_conv)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, enable_patch_conv=enable_patch_conv))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = Upsample(block_in, block_out, add_temporal_upsample, enable_patch_conv=enable_patch_conv)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = RMS_norm(block_in)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass through the decoder."""

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = F.silu(h, inplace=True)
        h = self.conv_out(h)
        return h


class AutoencoderKLConv3D(nn.Module):
    """KL regularized 3D Conv VAE with advanced tiling and slicing strategies."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        enable_patch_conv: bool = False,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.latent_channels = latent_channels

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
            enable_patch_conv=enable_patch_conv,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
            enable_patch_conv=enable_patch_conv,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.2):
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // self.ffactor_spatial
        self.tile_overlap_factor = tile_overlap_factor

        assert (self.tile_latent_min_size * self.tile_overlap_factor).is_integer(), (
            "self.tile_latent_min_size multiplied by tile_overlap_factor must be an integer"
        )

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b horizontally into a at blend_extent region."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b vertically into a at blend_extent region."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        """Tiled spatial encoding for large inputs via overlapping."""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def encode(self, x: Tensor) -> Tuple[DiagonalGaussianDistribution]:
        def _encode(x):
            if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)
        return (posterior,)

    def decode(self, z: Tensor) -> Tuple[Tensor]:
        # return_dict is not supported

        def _decode(z):
            if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        return (decoded,)

    def forward(
        self, sample: torch.Tensor, sample_posterior: bool = False, return_posterior: bool = True, return_dict: bool = True
    ):
        """Forward autoencoder pass. Returns both reconstruction and optionally the posterior."""
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z)[0]
        return (dec, posterior)


VAE_LATENT_CHANNELS = 32
VAE_SCALING_FACTOR = 1.03682


def load_vae_from_checkpoint(
    ckpt_path: str,
    device: torch.device,
    dtype: torch.dtype,
    sample_size: int = 256,
    enable_patch_conv: bool = False,
) -> AutoencoderKLConv3D:
    """Load the VAE model from a checkpoint file.

    Args:
        ckpt_path (str): Path to the checkpoint file.
        device (torch.device): Device to load the model onto.
        dtype (torch.dtype): Data type for the model parameters.
        sample_size (int): Sample size for the VAE model. 0 means no tiling (caller can call set_tile_sample_min_size() and enable_tiling() if needed).
        enable_patch_conv (bool): Whether to enable patch convolution.

    Returns:
        AutoencoderKLConv3D: Loaded VAE model.
    """
    """
{
  "_class_name": "AutoencoderKLConv3D",
  "_diffusers_version": "0.35.0",
  "block_out_channels": [
    128,
    256,
    512,
    1024,
    1024
  ],
  "downsample_match_channel": true,
  "ffactor_spatial": 16,
  "ffactor_temporal": 4,
  "in_channels": 3,
  "latent_channels": 32,
  "layers_per_block": 2,
  "out_channels": 3,
  "sample_size": 256,
  "sample_tsize": 64,
  "scaling_factor": 1.03682,
  "shift_factor": null,
  "upsample_match_channel": true
}
    """
    logger.info(f"Loading VAE from checkpoint: {ckpt_path}")
    if ckpt_path.endswith(".safetensors"):
        vae_state_dict = load_file(ckpt_path, device="cpu")
    else:
        vae_state_dict = torch.load(ckpt_path, map_location="cpu")

    logger.info("Initializing VAE model with empty weights")
    enable_tiling = sample_size != 0
    sample_size = sample_size if sample_size != 0 else 256
    with init_empty_weights():
        vae = AutoencoderKLConv3D(
            in_channels=3,
            out_channels=3,
            latent_channels=VAE_LATENT_CHANNELS,
            block_out_channels=(128, 256, 512, 1024, 1024),
            layers_per_block=2,
            ffactor_spatial=16,
            ffactor_temporal=4,
            sample_size=sample_size,
            sample_tsize=64,
            scaling_factor=VAE_SCALING_FACTOR,
            shift_factor=None,
            downsample_match_channel=True,
            upsample_match_channel=True,
            enable_patch_conv=enable_patch_conv,
        )
    info = vae.load_state_dict(vae_state_dict, strict=True, assign=True)
    logger.info(f"VAE loaded with info: {info}")
    vae.to(device=device, dtype=dtype)
    if enable_tiling:
        logger.info(f"Enabling VAE tiling with sample size: {sample_size}")
        vae.enable_tiling()
    return vae


if __name__ == "__main__":
    # Example usage
    import sys
    import av

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = sys.argv[1]
    safetensors_or_mp4 = sys.argv[2]
    spatial_size = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    enable_tiling = len(sys.argv) > 3
    vae = load_vae_from_checkpoint(vae_path, device=device, dtype=torch.float16, sample_size=spatial_size, enable_patch_conv=True)
    vae.eval()
    if enable_tiling:
        print(f"Enabling VAE tiling with spatial size: {spatial_size}")
        vae.enable_tiling()

    if safetensors_or_mp4.endswith(".safetensors"):
        latent = load_file(safetensors_or_mp4, device="cpu")
        if "latent" in latent:
            latent = latent["latent"]
        if len(latent.shape) == 4:
            latent = latent.unsqueeze(0)
        print(f"Loaded latent shape: {latent.shape}")
    else:
        container = av.open(safetensors_or_mp4, mode="r")
        frames = []
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            frames.append(img)
        video_np = np.stack(frames, axis=0)  # T H W C
        video_np = video_np.astype("float32") / 127.5 - 1.0
        video_np = video_np.transpose(3, 0, 1, 2)  # C T H W
        video_tensor = torch.from_numpy(video_np).unsqueeze(0).to(device=device, dtype=torch.float16)  # 1 C T H W

        print(f"Video tensor shape: {video_tensor.shape}")

        with torch.no_grad():
            print("Encoding video to latent space...")
            posterior = vae.encode(video_tensor)[0]
            latent = posterior.mode()

        print(f"Latent shape after encoding: {latent.shape}")

    latent = latent.to(device=device, dtype=torch.float16)
    with torch.no_grad():
        print("Decoding latent to reconstruct video...")
        recon = vae.decode(latent)[0]
    recon = recon.cpu().numpy().transpose(0, 2, 3, 4, 1)  # B T H W C
    recon = (recon * 127.5 + 127.5).clip(0, 255).astype("uint8")
    print(f"Reconstructed video shape: {recon.shape}")

    H = recon.shape[2] - recon.shape[2] % 2
    W = recon.shape[3] - recon.shape[3] % 2
    if (H, W) != (recon.shape[2], recon.shape[3]):
        print(f"Cropping frames to even size: {recon.shape[2]}x{recon.shape[3]} -> {H}x{W}")
        recon = recon[:, :, :H, :W, :]

    for b in range(recon.shape[0]):
        output_path = f"recon_{b}.mp4"
        container = av.open(output_path, mode="w")
        stream = container.add_stream("libx264", rate=24)
        stream.width = W
        stream.height = H
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 4000000

        for t in range(recon.shape[1]):
            frame = av.VideoFrame.from_ndarray(recon[b, t], format="rgb24")
            packet = stream.encode(frame)
            if packet:
                container.mux(packet)

        # Flush stream
        packet = stream.encode(None)
        if packet:
            container.mux(packet)

        container.close()
        print(f"Saved reconstructed video to {output_path}")
