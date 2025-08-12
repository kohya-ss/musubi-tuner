# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import Optional
import torch

try:
    import flash_attn_interface

    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn

    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    import sageattention

    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False

try:
    import xformers.ops as xops

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


import warnings

__all__ = [
    "flash_attention",
    "attention",
]


def flash_attention(
    qkv,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
    attn_mode: Optional[str] = "torch",
    split_attn: bool = False,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    q, k, v = qkv
    qkv.clear()

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    # assert q.device.type == "cuda" and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # We cannot test Flash attention 3 in musubi tuner, so keep the original code.
    # Customized code (except for flash attention 3) is not supported q_lens and k_lens.
    if attn_mode != "flash3" and attn_mode != "sageattn":
        assert q_lens is None, "q_lens is not supported except for flash attention 3."
        assert k_lens is None or (
            min(k_lens) == max(k_lens) and k_lens[0] == lk
        ), f"k_lens is not supported except for flash attention 3 or sage attention. k_lens={k_lens}, lk={lk}."

    # SDPA
    if attn_mode == "torch" or attn_mode == "sdpa":
        assert not deterministic, "deterministic is not supported in scaled_dot_product_attention."
        if q_scale is not None:
            q = q * q_scale
        q = half(q.transpose(1, 2))
        k = half(k.transpose(1, 2))
        v = half(v.transpose(1, 2))

        if not split_attn:
            q = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale
            )
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = torch.nn.functional.scaled_dot_product_attention(
                    q[i : i + 1], k[i : i + 1], v[i : i + 1], is_causal=causal, dropout_p=dropout_p, scale=softmax_scale
                )

        del q, k, v
        x = x.transpose(1, 2).contiguous()
        return x.type(out_dtype)

    # flash attention 2
    if attn_mode == "flash" or attn_mode == "flash2":
        if q_scale is not None:
            q = q * q_scale
        q = half(q)
        k = half(k)
        v = half(v)

        if not split_attn:
            q = flash_attn.flash_attn_func(q, k, v, dropout_p, softmax_scale, causal, window_size, deterministic=deterministic)
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = flash_attn.flash_attn_func(
                    q[i : i + 1],
                    k[i : i + 1],
                    v[i : i + 1],
                    dropout_p,
                    softmax_scale,
                    causal,
                    window_size,
                    deterministic=deterministic,
                )
        del q, k, v
        return x.type(out_dtype)

    # xformers
    if attn_mode == "xformers":
        assert not deterministic, "deterministic is not supported in xformers."
        assert not causal, "causal is not supported in xformers."
        if q_scale is not None:
            q = q * q_scale
        q = half(q)
        k = half(k)
        v = half(v)

        if not split_attn:
            q = xops.memory_efficient_attention(q, k, v, p=dropout_p, scale=softmax_scale)
            x = q
        else:
            x = torch.empty_like(q)
            for i in range(q.size(0)):
                x[i : i + 1] = xops.memory_efficient_attention(
                    q[i : i + 1], k[i : i + 1], v[i : i + 1], p=dropout_p, scale=softmax_scale
                )

        del q, k, v
        return x.type(out_dtype)

    # sage attention with fixed length seems to cause NaN in I2V inference.
    # # sage attention
    # if attn_mode == "sageattn":
    #     print("Using sage attention")
    #     assert not deterministic, "deterministic is not supported in sage attention."
    #     if q_scale is not None:
    #         q = q * q_scale
    #     q, k, v = half(q), half(k), half(v)
    #     x = sageattention.sageattn(q, k, v, "NHD", is_causal=causal, sm_scale=softmax_scale)
    #     del q, k, v
    #     return x.type(out_dtype)

    assert not split_attn, "split_attn is not supported in flash attention 3 or sage attention."

    # preprocess query: in Wan 2.1, q_lens is always None.
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor([lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor([lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
    else:
        # Note: in Wan 2.1, all k_lens are same if we have same image size in the batch.
        if min(k_lens) == max(k_lens) and k.shape[1] == k_lens[0]:
            # B, L, N, C -> BN, L, C
            k = half(k.flatten(0, 1))
            v = half(v.flatten(0, 1))
        else:
            k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
            v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    # if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
    #     warnings.warn("Flash attention 3 is not available, use flash attention 2 instead.")

    # apply attention
    # if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
    if attn_mode == "flash3":
        # Not tested yet in musubi tuner.
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic,
        )[0].unflatten(0, (b, lq))
    # elif (version is None or version == 2) and FLASH_ATTN_2_AVAILABLE:
    #     # assert FLASH_ATTN_2_AVAILABLE
    #     x = flash_attn.flash_attn_varlen_func(
    #         q=q,
    #         k=k,
    #         v=v,
    #         cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
    #         cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
    #         max_seqlen_q=lq,
    #         max_seqlen_k=lk,
    #         dropout_p=dropout_p,
    #         softmax_scale=softmax_scale,
    #         causal=causal,
    #         window_size=window_size,
    #         deterministic=deterministic,
    #     ).unflatten(0, (b, lq))
    # elif version is None and SAGE_ATTN_AVAILABLE:
    elif attn_mode == "sageattn":
        # print("Using sage attention")
        assert not causal, "SAGE attention does not support causal attention."
        x = sageattention.sageattn_varlen(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            sm_scale=softmax_scale,
        ).unflatten(0, (b, lq))
    else:
        raise ValueError(f"Unknown attention mode: {attn_mode}")

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.0,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                "Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance."
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous()
        return out
