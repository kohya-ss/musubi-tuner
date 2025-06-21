import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from einops.layers.torch import Rearrange
from einops import rearrange


class BroadMultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(BroadMultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = rearrange(Q.squeeze(), "i (heads d) -> heads i d", heads=self.heads)
        K = rearrange(K, "b j (heads d) -> b heads j d", heads=self.heads)

        dots = einsum("hid, bhjd -> bhij", Q, K) * self.scale  # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, _, _ = K.shape
        _, N, _ = Q.shape

        V = rearrange(V, "b j (heads d) -> b heads j d", heads=self.heads)

        out = einsum("bhij, bhjd -> bhid", attn, V)
        out = rearrange(out, "b heads n d -> b n (heads d)", b=B, n=N)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttention, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K):
        Q = rearrange(Q, "b i (heads d) -> b heads i d", heads=self.heads)
        K = rearrange(K, "b j (heads d) -> b heads j d", heads=self.heads)

        dots = (
            einsum("bhid, bhjd -> bhij", Q, K) * self.scale
        )  # (b hw) heads 1 pointnum

        return self.attend(dots)

    def forward(self, Q, K, V):
        attn = self.attend_with_rpe(Q, K)
        B, HW, _ = Q.shape

        V = rearrange(V, "b j (heads d) -> b heads j d", heads=self.heads)

        out = einsum("bhij, bhjd -> bhid", attn, V)
        out = rearrange(out, "b heads hw d -> b hw (heads d)", b=B, hw=HW)

        return out


# class MultiHeadAttentionRelative_encoder(nn.Module):
#     def __init__(self, dim, heads):
#         super(MultiHeadAttentionRelative, self).__init__()
#         self.dim = dim
#         self.heads = heads
#         self.scale = (dim/heads) ** -0.5
#         self.attend = nn.Softmax(dim=-1)

#     def attend_with_rpe(self, Q, K, Q_r, K_r):
#         """
#             Q: [BH1W1, H3W3, dim]
#             K: [BH1W1, H3W3, dim]
#             Q_r: [BH1W1, H3W3, H3W3, dim]
#             K_r: [BH1W1, H3W3, H3W3, dim]
#         """

#         Q = rearrange(Q, 'b i (heads d) -> b heads i d', heads=self.heads) # [BH1W1, heads, H3W3, dim]
#         K = rearrange(K, 'b j (heads d) -> b heads j d', heads=self.heads) # [BH1W1, heads, H3W3, dim]
#         K_r = rearrange(K_r, 'b j (heads d) -> b heads j d', heads=self.heads) # [BH1W1, heads, H3W3, dim]
#         Q_r = rearrange(Q_r, 'b j (heads d) -> b heads j d', heads=self.heads) # [BH1W1, heads, H3W3, dim]

#         # context-context similarity
#         c_c = einsum('bhid, bhjd -> bhij', Q, K) * self.scale # [(B H1W1) heads H3W3 H3W3]
#         # context-position similarity
#         c_p = einsum('bhid, bhjd -> bhij', Q, K_r) * self.scale # [(B H1W1) heads 1 H3W3]
#         # position-context similarity
#         p_c = einsum('bhijd, bhikd -> bhijk', Q_r[:,:,:,None,:], K[:,:,:,None,:])
#         p_c = torch.squeeze(p_c, dim=4)
#         p_c = p_c.permute(0, 1, 3, 2)
#         dots = c_c + c_p + p_c
#         return self.attend(dots)

#     def forward(self, Q, K, V, Q_r, K_r):
#         attn = self.attend_with_rpe(Q, K, Q_r, K_r)
#         B, HW, _ = Q.shape

#         V = rearrange(V, 'b j (heads d) -> b heads j d', heads=self.heads)

#         out = einsum('bhij, bhjd -> bhid', attn, V)
#         out = rearrange(out, 'b heads hw d -> b hw (heads d)', b=B, hw=HW)

#         return out


class MultiHeadAttentionRelative(nn.Module):
    def __init__(self, dim, heads):
        super(MultiHeadAttentionRelative, self).__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim / heads) ** -0.5
        self.attend = nn.Softmax(dim=-1)

    def attend_with_rpe(self, Q, K, Q_r, K_r):
        """
        Q: [BH1W1, 1, dim]
        K: [BH1W1, H3W3, dim]
        Q_r: [BH1W1, H3W3, dim]
        K_r: [BH1W1, H3W3, dim]
        """

        Q = rearrange(
            Q, "b i (heads d) -> b heads i d", heads=self.heads
        )  # [BH1W1, heads, 1, dim]
        K = rearrange(
            K, "b j (heads d) -> b heads j d", heads=self.heads
        )  # [BH1W1, heads, H3W3, dim]
        K_r = rearrange(
            K_r, "b j (heads d) -> b heads j d", heads=self.heads
        )  # [BH1W1, heads, H3W3, dim]
        Q_r = rearrange(
            Q_r, "b j (heads d) -> b heads j d", heads=self.heads
        )  # [BH1W1, heads, H3W3, dim]

        # context-context similarity
        c_c = einsum("bhid, bhjd -> bhij", Q, K) * self.scale  # [(B H1W1) heads 1 H3W3]
        # context-position similarity
        c_p = (
            einsum("bhid, bhjd -> bhij", Q, K_r) * self.scale
        )  # [(B H1W1) heads 1 H3W3]
        # position-context similarity
        p_c = (
            einsum("bhijd, bhikd -> bhijk", Q_r[:, :, :, None, :], K[:, :, :, None, :])
            * self.scale
        )
        p_c = torch.squeeze(p_c, dim=4)
        p_c = p_c.permute(0, 1, 3, 2)
        dots = c_c + c_p + p_c
        return self.attend(dots)

    def forward(self, Q, K, V, Q_r, K_r):
        attn = self.attend_with_rpe(Q, K, Q_r, K_r)
        B, HW, _ = Q.shape

        V = rearrange(V, "b j (heads d) -> b heads j d", heads=self.heads)

        out = einsum("bhij, bhjd -> bhid", attn, V)
        out = rearrange(out, "b heads hw d -> b hw (heads d)", b=B, hw=HW)

        return out


def LinearPositionEmbeddingSine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim // 4 - 1, dim // 4).to(x.device)
    return torch.cat(
        [
            torch.sin(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
            torch.cos(3.14 * x[..., -2:-1] * freq_bands * NORMALIZE_FACOR),
            torch.sin(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR),
            torch.cos(3.14 * x[..., -1:] * freq_bands * NORMALIZE_FACOR),
        ],
        dim=-1,
    )


def ExpPositionEmbeddingSine(x, dim=128, NORMALIZE_FACOR=1 / 200):
    # 200 should be enough for a 8x downsampled image
    # assume x to be [_, _, 2]
    freq_bands = torch.linspace(0, dim // 4 - 1, dim // 4).to(x.device)
    return torch.cat(
        [
            torch.sin(x[..., -2:-1] * (NORMALIZE_FACOR * 2**freq_bands)),
            torch.cos(x[..., -2:-1] * (NORMALIZE_FACOR * 2**freq_bands)),
            torch.sin(x[..., -1:] * (NORMALIZE_FACOR * 2**freq_bands)),
            torch.cos(x[..., -1:] * (NORMALIZE_FACOR * 2**freq_bands)),
        ],
        dim=-1,
    )
