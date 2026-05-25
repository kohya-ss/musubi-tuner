# Unified attention function supporting various implementations

from dataclasses import dataclass
import torch
from typing import Optional, Union

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None
    flash_attn_qkvpacked_func = None
    flash_attn_varlen_qkvpacked_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device)
            for i in range(batch_size):
                cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]  # end of valid tokens for query
                cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen  # end of all tokens for query

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

            # attention bias for xformers
            if attn_mode == "xformers":
                seqlens_list = seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode == "torch":
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, cu_seqlens, max_seqlen)


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: One of:
            - Query tensor [B, L, H, D] (k and v must be provided separately)
            - List [q, k, v] of tensors each [B, L, H, D]
            - Packed QKV tensor [B, L, 3, H, D] (used directly by flash attention)
        k: Key tensor [B, L, H, D]. Required when qkv_or_q is a plain query tensor.
        v: Value tensor [B, L, H, D]. Required when qkv_or_q is a plain query tensor.
        attn_params: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    # qkv_packed holds [B, L, 3, H, D] when the input was already packed.
    # For flash attention this avoids a re-stack; other backends unpack via unbind views.
    qkv_packed: Optional[torch.Tensor] = None

    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        del qkv_or_q
    elif isinstance(qkv_or_q, torch.Tensor) and qkv_or_q.ndim == 5:
        # Packed QKV tensor [B, L, 3, H, D]
        assert qkv_or_q.shape[2] == 3, f"5D input must be packed QKV [B, L, 3, H, D], got shape {tuple(qkv_or_q.shape)}"
        qkv_packed = qkv_or_q
        del qkv_or_q
        q, k, v = qkv_packed.unbind(dim=2)  # views: each [B, L, H, D]
    else:
        q: torch.Tensor = qkv_or_q
        del qkv_or_q
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence
    seqlen_trimmed = False
    # Trim if all seqlens are the same, for attention modes other than flash or sageattn (which can handle masks efficiently)
    if (
        not attn_params.split_attn
        and attn_params.attention_mask is not None
        and attn_params.seqlens is not None
        and (attn_params.attn_mode != "flash" and attn_params.attn_mode != "sageattn")
    ):
        if torch.all(attn_params.seqlens == attn_params.seqlens[0]):
            seqlen = attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA and sageattn with fixed length
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)  # do not in-place modify
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
        k = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))]
        v = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = torch.nn.functional.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "xformers":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "sageattn":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v)  # B, L, H, D. No dropout support
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            q = q.view(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.view(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.view(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen
            )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                if qkv_packed is not None:
                    # Packed input: slice directly, no unbind/re-stack needed
                    qkv_i = qkv_packed[i : i + 1, : attn_params.seqlens[i]]  # [1, L_i, 3, H, D]
                    x_i = flash_attn_qkvpacked_func(qkv_i, drop_rate)  # [1, L_i, H, D]
                else:
                    x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)  # [1, L_i, H, D]
                    q[i] = None
                    k[i] = None
                    v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # [1, max_seqlen, H, D]
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
            qkv_packed = None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            if qkv_packed is not None:
                x = flash_attn_qkvpacked_func(qkv_packed, drop_rate)  # [B, L, H, D]
                qkv_packed = None
            else:
                x = flash_attn_func(q, k, v, drop_rate)  # [B, L, H, D]
                q, k, v = None, None, None
        else:
            # Varlen
            batch_size, seqlen = q.shape[0], q.shape[1]
            if qkv_packed is not None:
                # View to [B*L, 3, H, D] — zero-copy since qkv_packed is contiguous
                qkv_flat = qkv_packed.view(batch_size * seqlen, 3, *qkv_packed.shape[3:])
                qkv_packed = None
                # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
                x = flash_attn_varlen_qkvpacked_func(
                    qkv_flat, attn_params.cu_seqlens, attn_params.max_seqlen, drop_rate
                )
                qkv_flat = None
            else:
                q = q.view(batch_size * seqlen, *q.shape[2:])  # [B*L, H, D]
                k = k.view(batch_size * seqlen, *k.shape[2:])
                v = v.view(batch_size * seqlen, *v.shape[2:])
                # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
                x = flash_attn_varlen_func(
                    q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen, drop_rate
                )
                q, k, v = None, None, None

            # Reshape [B*L, H, D] → [B, L, H, D]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x
