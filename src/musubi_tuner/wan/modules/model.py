# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from accelerate import init_empty_weights

import logging

from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from musubi_tuner.utils.device_utils import clean_memory_on_device

from musubi_tuner.wan.modules.attention import flash_attention
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

__all__ = ["WanModel"]


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# @amp.autocast(enabled=False)
# no autocast is needed for rope_apply, because it is already in float64
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    device_type = x.device.type
    with torch.amp.autocast(device_type=device_type, enabled=False):
        n, c = x.size(2), x.size(3) // 2

        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w

            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
            freqs_i = torch.cat(
                [
                    freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)

            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

            # append to collection
            output.append(x_i)
        return torch.stack(output).float()


def calculate_freqs_i(fhw, c, freqs, f_indices=None):
    """f_indices is used to select specific frames for rotary embedding. e.g. [0,8] (with start image) or [0,8,20] (with start and end images)"""
    f, h, w = fhw[:3]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    if f_indices is None:
        freqs_f = freqs[0][:f]
    else:
        logger.info(f"Using f_indices: {f_indices} for rotary embedding. fhw: {fhw}")
        freqs_f = freqs[0][f_indices]

    freqs_i = torch.cat(
        [
            freqs_f.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        ],
        dim=-1,
    ).reshape(f * h * w, 1, -1)
    return freqs_i


# inplace version of rope_apply
def rope_apply_inplace_cached(x, grid_sizes, freqs_list):
    # with torch.amp.autocast(device_type=device_type, enabled=False):
    rope_dtype = torch.float64  # float32 does not reduce memory usage significantly

    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(rope_dtype).reshape(seq_len, n, -1, 2))
        freqs_i = freqs_list[i]

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        # x_i = torch.cat([x_i, x[i, seq_len:]])

        # inplace update
        x[i, :seq_len] = x_i.to(x.dtype)

    return x


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # return self._norm(x.float()).type_as(x) * self.weight
        # support fp8
        return self._norm(x.float()).type_as(x) * self.weight.to(x.dtype)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    # def forward(self, x):
    #     r"""
    #     Args:
    #         x(Tensor): Shape [B, L, C]
    #     """
    #     # inplace version, also supports fp8 -> does not have significant performance improvement
    #     original_dtype = x.dtype
    #     x = x.float()
    #     y = x.pow(2).mean(dim=-1, keepdim=True)
    #     y.add_(self.eps)
    #     y.rsqrt_()
    #     x *= y
    #     x = x.to(original_dtype)
    #     x *= self.weight.to(original_dtype)
    #     return x


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # # query, key, value function
        # def qkv_fn(x):
        #     q = self.norm_q(self.q(x)).view(b, s, n, d)
        #     k = self.norm_k(self.k(x)).view(b, s, n, d)
        #     v = self.v(x).view(b, s, n, d)
        #     return q, k, v
        # q, k, v = qkv_fn(x)
        # del x
        # query, key, value function

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        del x
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, s, n, d)
        k = k.view(b, s, n, d)
        v = v.view(b, s, n, d)

        rope_apply_inplace_cached(q, grid_sizes, freqs)
        rope_apply_inplace_cached(k, grid_sizes, freqs)
        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(
            qkv, k_lens=seq_lens, window_size=self.window_size, attn_mode=self.attn_mode, split_attn=self.split_attn
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        # q = self.norm_q(self.q(x)).view(b, -1, n, d)
        # k = self.norm_k(self.k(context)).view(b, -1, n, d)
        # v = self.v(context).view(b, -1, n, d)
        q = self.q(x)
        del x
        k = self.k(context)
        v = self.v(context)
        del context
        q = self.norm_q(q)
        k = self.norm_k(k)
        q = q.view(b, -1, n, d)
        k = k.view(b, -1, n, d)
        v = v.view(b, -1, n, d)

        # compute attention
        qkv = [q, k, v]
        del q, k, v
        x = flash_attention(qkv, k_lens=context_lens, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, attn_mode="torch", split_attn=False):
        super().__init__(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn)

        self.k_img = nn.Linear(dim, dim)
        self.v_img = nn.Linear(dim, dim)
        # self.alpha = nn.Parameter(torch.zeros((1, )))
        self.norm_k_img = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        context_img = context[:, :257]
        context = context[:, 257:]
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.q(x)
        del x
        q = self.norm_q(q)
        q = q.view(b, -1, n, d)
        k = self.k(context)
        k = self.norm_k(k).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        del context

        # compute attention
        qkv = [q, k, v]
        del k, v
        x = flash_attention(qkv, k_lens=context_lens, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # compute query, key, value
        k_img = self.norm_k_img(self.k_img(context_img)).view(b, -1, n, d)
        v_img = self.v_img(context_img).view(b, -1, n, d)
        del context_img

        # compute attention
        qkv = [q, k_img, v_img]
        del q, k_img, v_img
        img_x = flash_attention(qkv, k_lens=None, attn_mode=self.attn_mode, split_attn=self.split_attn)

        # output
        x = x.flatten(2)
        img_x = img_x.flatten(2)
        if self.training:
            x = x + img_x  # avoid inplace
        else:
            x += img_x
        del img_x

        x = self.o(x)
        return x


# For v2.1
WAN_CROSSATTENTION_CLASSES = {
    "t2v_cross_attn": WanCrossAttention,
    "i2v_cross_attn": WanI2VCrossAttention,
}


class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode="torch",
        split_attn=False,
        model_version="2.1",  # New!
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.model_version = model_version  # New!

        # layers
        if model_version == "2.1":
            cross_attn_class = WAN_CROSSATTENTION_CLASSES[cross_attn_type]
        elif model_version == "2.2":
            cross_attn_class = WanCrossAttention  # For Wan2.2, we use the same cross-attention class

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps, attn_mode, split_attn)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = cross_attn_class(dim, num_heads, (-1, -1), qk_norm, eps, attn_mode, split_attn)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C] for 2.1, [B, L, 6, C] for 2.2
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        org_dtype = x.dtype
        assert e.dtype == torch.float32
        if self.model_version == "2.1":
            e = self.modulation.to(torch.float32) + e
            e = e.chunk(6, dim=1)
            assert e[0].dtype == torch.float32

            # self-attention
            # y = self.self_attn((self.norm1(x).float() * (1 + e[1]) + e[0]).to(org_dtype), seq_lens, grid_sizes, freqs)
            y = self.self_attn(torch.addcmul(e[0], self.norm1(x).float(), (1 + e[1])).to(org_dtype), seq_lens, grid_sizes, freqs)
            # x = (x + y.to(torch.float32) * e[2]).to(org_dtype)
            x = torch.addcmul(x, y.to(torch.float32), e[2]).to(org_dtype)
            del y

            # cross-attention & ffn
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            del context
            # y = self.ffn((self.norm2(x).float() * (1 + e[4]) + e[3]).to(org_dtype))
            y = self.ffn(torch.addcmul(e[3], self.norm2(x).float(), (1 + e[4])).to(org_dtype))
            # x = (x + y.to(torch.float32) * e[5]).to(org_dtype)
            x = torch.addcmul(x, y.to(torch.float32), e[5]).to(org_dtype)
            del y
        else:  # For Wan2.2
            e = self.modulation.to(torch.float32) + e
            e = e.chunk(6, dim=2)  # e is [B, L, 6, C] for 2.2
            assert e[0].dtype == torch.float32

            # self-attention
            # y = self.self_attn(
            #     (self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)).to(org_dtype), seq_lens, grid_sizes, freqs
            # )
            y = self.self_attn(
                torch.addcmul(e[0].squeeze(2), self.norm1(x).float(), (1 + e[1].squeeze(2))).to(org_dtype), seq_lens, grid_sizes, freqs
            )
            # x = (x + y.to(torch.float32) * e[2].squeeze(2)).to(org_dtype)
            x = torch.addcmul(x, y.to(torch.float32), e[2].squeeze(2)).to(org_dtype)
            del y

            # cross-attention & ffn
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            del context
            # y = self.ffn((self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2)).to(org_dtype))
            y = self.ffn(torch.addcmul(e[3].squeeze(2), self.norm2(x).float(), (1 + e[4].squeeze(2))).to(org_dtype))
            # x = (x + y.to(torch.float32) * e[5].squeeze(2)).to(org_dtype)
            x = torch.addcmul(x, y.to(torch.float32), e[5].squeeze(2)).to(org_dtype)
            del y

        return x

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens):
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.modulation.device)
            return checkpoint(forward_fn, x, e, seq_lens, grid_sizes, freqs, context, context_lens, use_reentrant=False)
        return self._forward(x, e, seq_lens, grid_sizes, freqs, context, context_lens)


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6, model_version="2.1"):  # New!
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.model_version = model_version  # New!

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, C] for 2.1, [B, L, 6, C] for 2.2
        """
        assert e.dtype == torch.float32
        if self.model_version == "2.1":
            e = (self.modulation.to(torch.float32) + e.unsqueeze(1)).chunk(2, dim=1)
            # x = self.head(self.norm(x) * (1 + e[1]) + e[0])
            x = self.head(torch.addcmul(e[0], self.norm(x), (1 + e[1])))
        else:  # For Wan2.2
            e = (self.modulation.unsqueeze(0).to(torch.float32) + e.unsqueeze(2)).chunk(2, dim=2)
            # x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
            x = self.head(torch.addcmul(e[0].squeeze(2), self.norm(x), (1 + e[1].squeeze(2))))

        return x


FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER = 257 * 2


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim, flf_pos_emb=False):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim),
            torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(),
            torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim),
        )
        if flf_pos_emb:  # NOTE: we only use this for `flf2v`
            self.emb_pos = nn.Parameter(torch.zeros(1, FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER, 1280))
        else:
            self.emb_pos = None

    def forward(self, image_embeds):
        if self.emb_pos is not None:  # for `flf2v`
            bs, n, d = image_embeds.shape
            image_embeds = image_embeds.view(-1, 2 * n, d)
            image_embeds = image_embeds + self.emb_pos
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = [
    "norm",
    "patch_embedding",
    "text_embedding",
    "time_embedding",
    "time_projection",
    "head",
    "modulation",
    "img_emb",
]


class WanModel(nn.Module):  # ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    # @register_to_config
    def __init__(
        self,
        model_type="t2v",
        model_version="2.1",  # New!
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        attn_mode=None,
        split_attn=False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            model_version (`str`, *optional*, defaults to '2.1'):
                Version of the model, e.g., '2.1' or '2.2'. This is used to determine the modulation strategy.
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v", "flf2v"], f"Invalid model_type: {model_type}. Must be one of ['t2v', 'i2v', 'flf2v']."
        self.model_type = model_type
        self.model_version = model_version  # New!

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.attn_mode = attn_mode if attn_mode is not None else "torch"
        self.split_attn = split_attn

        # embeddings
        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.force_v2_1_time_embedding = False  # Override to use 2.1 style time embedding for 2.2 model

        # blocks
        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    cross_attn_type,
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    attn_mode,
                    split_attn,
                    model_version=self.model_version,  # New!
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps, model_version=self.model_version)  # New!

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1
        )
        self.freqs_fhw = {}

        if self.model_version == "2.1" and (model_type == "i2v" or model_type == "flf2v"):
            self.img_emb = MLPProj(1280, dim, flf_pos_emb=model_type == "flf2v")

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        # offloading
        self.blocks_to_swap = None
        self.offloader = None

    @property
    def dtype(self):
        return self.patch_embedding.weight.dtype

    @property
    def device(self):
        return self.patch_embedding.weight.device

    def set_time_embedding_v2_1(self, force_v2_1_time_embedding: bool):
        self.force_v2_1_time_embedding = force_v2_1_time_embedding
        if force_v2_1_time_embedding:
            logger.info("WanModel: Using 2.1 style time embedding for time_projection.")

    def fp8_optimization(
        self, state_dict: dict[str, torch.Tensor], device: torch.device, move_to_device: bool, use_scaled_mm: bool = False
    ) -> int:
        """
        Optimize the model state_dict with fp8.

        Args:
            state_dict (dict[str, torch.Tensor]):
                The state_dict of the model.
            device (torch.device):
                The device to calculate the weight.
            move_to_device (bool):
                Whether to move the weight to the device after optimization.
        """
        # inplace optimization
        state_dict = optimize_state_dict_with_fp8(
            state_dict, device, FP8_OPTIMIZATION_TARGET_KEYS, FP8_OPTIMIZATION_EXCLUDE_KEYS, move_to_device=move_to_device
        )

        # apply monkey patching
        apply_fp8_monkey_patch(self, state_dict, use_scaled_mm=use_scaled_mm)

        return state_dict

    def enable_gradient_checkpointing(self, activation_cpu_offloading=False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

        for block in self.blocks:
            block.enable_gradient_checkpointing(activation_cpu_offloading)

        print(f"WanModel: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        for block in self.blocks:
            block.disable_gradient_checkpointing()

        print(f"WanModel: Gradient checkpointing disabled.")

    def enable_block_swap(self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)

        assert (
            self.blocks_to_swap <= self.num_blocks - 1
        ), f"Cannot swap more than {self.num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."

        self.offloader = ModelOffloader(
            "wan_attn_block", self.blocks, self.num_blocks, self.blocks_to_swap, supports_backward, device, use_pinned_memory  # , debug=True
        )
        print(
            f"WanModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print(f"WanModel: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None, skip_block_indices=None, f_indices=None):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            skip_block_indices (List[int], *optional*):
                Indices of blocks to skip during forward pass
            f_indices (List[List[int]], *optional*):
                Indices of frames used for rotary embeddings, list of lists for each video in the batch

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        # remove assertions to work with Fun-Control T2V
        # if self.model_type == "i2v":
        #     assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            y = None

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  # x[0].shape = [1, 5120, F, H, W]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])  # list of [F, H, W]

        freqs_list = []
        for i, fhw in enumerate(grid_sizes):
            fhw = tuple(fhw.tolist())
            if f_indices is not None:
                fhw = tuple(list(fhw) + f_indices[i])  # add f_indices to fhw for cache key
            if fhw not in self.freqs_fhw:
                c = self.dim // self.num_heads // 2
                self.freqs_fhw[fhw] = calculate_freqs_i(fhw, c, self.freqs, None if f_indices is None else f_indices[i])
            freqs_list.append(self.freqs_fhw[fhw])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length exceeds maximum allowed length {seq_len}. Got {seq_lens.max()}"
        x = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
            if self.model_version == "2.1" or self.force_v2_1_time_embedding:  # For Wan2.1
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
                e0 = self.time_projection(e).unflatten(1, (6, self.dim))
                # e0: torch.Size([1, 6, 5120]), e: torch.Size([1, 5120]), t: torch.Size([1])

                if self.model_version != "2.1":  # Reshape to be compatible with 2.2 blocks
                    e0 = e0.unsqueeze(1)
                    e = e.unsqueeze(1)
                    t = t.unsqueeze(1).expand(-1, seq_len)
            else:  # For Wan2.2
                if t.dim() == 1:
                    # t = t.expand(t.size(0), seq_len) # this should be a bug in the original code
                    t = t.unsqueeze(1).expand(-1, seq_len)
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                # e0: torch.Size([1, 14040, 6, 5120]), e: torch.Size([1, 14040, 5120]), t: torch.Size([14040])

        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        if type(context) is list:
            context = torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        context = self.text_embedding(context)

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)
            clip_fea = None
            context_clip = None

        # arguments
        kwargs = dict(e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=freqs_list, context=context, context_lens=context_lens)

        if self.blocks_to_swap:
            clean_memory_on_device(device)

        # print(f"x: {x.shape}, e: {e0.shape}, context: {context.shape}, seq_lens: {seq_lens}")
        input_device = x.device
        for block_idx, block in enumerate(self.blocks):
            is_block_skipped = skip_block_indices is not None and block_idx in skip_block_indices

            if self.blocks_to_swap and not is_block_skipped:
                self.offloader.wait_for_block(block_idx)

            if not is_block_skipped:
                x = block(x, **kwargs)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, block_idx)

        if x.device != input_device:
            x = x.to(input_device)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


def detect_wan_sd_dtype(path: str) -> torch.dtype:
    # get dtype from model weights
    with MemoryEfficientSafeOpen(path) as f:
        keys = set(f.keys())
        key1 = "model.diffusion_model.blocks.0.cross_attn.k.weight"  # 1.3B
        key2 = "blocks.0.cross_attn.k.weight"  # 14B
        if key1 in keys:
            dit_dtype = f.get_tensor(key1).dtype
        elif key2 in keys:
            dit_dtype = f.get_tensor(key2).dtype
        else:
            raise ValueError(f"Could not find the dtype in the model weights: {path}")
    logger.info(f"Detected DiT dtype: {dit_dtype}")
    return dit_dtype


def load_wan_model(
    config: any,
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[List[float]] = None,
    use_scaled_mm: bool = False,
    disable_numpy_memmap: bool = False,
) -> WanModel:
    """
    Load a WAN model from the specified checkpoint.

    Args:
        config (any): Configuration object containing model parameters.
        device (Union[str, torch.device]): Device to load the model on.
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
        use_scaled_mm (bool): Whether to use scaled matrix multiplication for fp8.
        disable_numpy_memmap (bool): Whether to disable numpy memmap when loading weights.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        logger.info(
            f"Creating WanModel. I2V: {config.i2v}, FLF2V: {config.flf2v}, V2.2: {config.v2_2}, device: {device}, loading_device: {loading_device}, fp8_scaled: {fp8_scaled}"
        )
        model = WanModel(
            model_type="i2v" if config.i2v else ("flf2v" if config.flf2v else "t2v"),
            model_version="2.1" if not config.v2_2 else "2.2",
            dim=config.dim,
            eps=config.eps,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            in_dim=config.in_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            out_dim=config.out_dim,
            text_len=config.text_len,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    # remove "model.diffusion_model." prefix: 1.3B model has this prefix
    for key in list(sd.keys()):
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=use_scaled_mm)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    if dit_weight_dtype is not None:
        # cast model weights to the specified dtype. This makes sure that the model is in the correct dtype
        logger.info(f"Casting model weights to {dit_weight_dtype}")
        model = model.to(dit_weight_dtype)
    logger.info(f"Loaded DiT model from {dit_path}, info={info}")

    return model
