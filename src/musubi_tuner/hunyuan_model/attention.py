import math

import torch
import torch.nn.functional as F

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    print("Trying to import sageattention")
    from sageattention import sageattn_varlen, sageattn

    print("Successfully imported sageattention")
except ImportError:
    print("Failed to import sageattention")
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "flash_fixlen": (
        lambda x: x,
        lambda x: x,
    ),
    "sageattn": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "sageattn_fixlen": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xformers": (
        lambda x: x,
        lambda x: x,
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


def attention(
    q_or_qkv_list,
    k=None,
    v=None,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    total_len=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    q, k, v = q_or_qkv_list if type(q_or_qkv_list) == list else (q_or_qkv_list, k, v)
    if type(q_or_qkv_list) == list:
        q_or_qkv_list.clear()
    split_attn = total_len is not None
    if (split_attn or cu_seqlens_q is None) and mode == "sageattn":
        mode = "sageattn_fixlen"
    elif (split_attn or cu_seqlens_q is None) and mode == "flash":
        mode = "flash_fixlen"
    # print(f"Attention mode: {mode}, split_attn: {split_attn}")
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]

    # trim the sequence length to the actual length instead of attn_mask
    if split_attn:
        trimmed_len = q.shape[1] - total_len
        q = [q[i : i + 1, : total_len[i]] for i in range(len(q))]
        k = [k[i : i + 1, : total_len[i]] for i in range(len(k))]
        v = [v[i : i + 1, : total_len[i]] for i in range(len(v))]
        q = [pre_attn_layout(q_i) for q_i in q]
        k = [pre_attn_layout(k_i) for k_i in k]
        v = [pre_attn_layout(v_i) for v_i in v]
        # print(
        #     f"Trimming the sequence length to {total_len},trimmed_len: {trimmed_len}, q.shape: {[q_i.shape for q_i in q]}, mode: {mode}"
        # )
    else:
        q = pre_attn_layout(q)
        k = pre_attn_layout(k)
        v = pre_attn_layout(v)

    if mode == "torch":
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = F.scaled_dot_product_attention(q[i], k[i], v[i], dropout_p=drop_rate, is_causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            if attn_mask is not None and attn_mask.dtype != torch.bool:
                attn_mask = attn_mask.to(q.dtype)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)
            del q, k, v
            del attn_mask

    elif mode == "xformers":
        # B, M, H, K: M is the sequence length, H is the number of heads, K is the dimension of the heads -> it is same as input dimension
        # currently only support batch_size = 1
        assert split_attn or cu_seqlens_q is None, "Xformers only supports splitting"
        if split_attn:
            x = []
            for i in range(len(q)):
                x_i = xops.memory_efficient_attention(q[i], k[i], v[i], p=drop_rate)  # , causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            x = xops.memory_efficient_attention(q, k, v, p=drop_rate)
            del q, k, v

    elif mode == "flash":
        x = flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        del q, k, v
        # x with shape [(bxs), a, d]
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]

    elif mode == "flash_fixlen":
        if split_attn:
            x = []
            for i in range(len(q)):
                # q: (batch_size, seqlen, nheads, headdim), k: (batch_size, seqlen, nheads_k, headdim), v: (batch_size, seqlen, nheads_k, headdim)
                x_i = flash_attn_func(q[i], k[i], v[i], dropout_p=drop_rate, causal=causal)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            x = flash_attn_func(q, k, v, dropout_p=drop_rate, causal=causal)
            del q, k, v  # this causes error in compiled mode with fullgraph=True

    elif mode == "sageattn":
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
        del q, k, v
        # x with shape [(bxs), a, d]
        x = x.view(batch_size, max_seqlen_q, x.shape[-2], x.shape[-1])  # reshape x to [b, s, a, d]

    elif mode == "sageattn_fixlen":
        if split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # (batch_size, seq_len, head_num, head_dim)
                q[i], k[i], v[i] = None, None, None
                x.append(x_i)
            del q, k, v
        else:
            x = sageattn(q, k, v)
            del q, k, v

    elif mode == "vanilla":
        assert not split_attn, "Vanilla attention does not support trimming"
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    if split_attn:
        x = [post_attn_layout(x_i) for x_i in x]
        for i in range(len(x)):
            x[i] = F.pad(x[i], (0, 0, 0, 0, 0, trimmed_len[i]))
        x = torch.cat(x, dim=0)
    else:
        x = post_attn_layout(x)

    b, s, a, d = x.shape
    x = x.reshape(b, s, -1)
    return x


def parallel_attention(hybrid_seq_parallel_attn, q, k, v, img_q_len, img_kv_len, cu_seqlens_q, cu_seqlens_kv):
    attn1 = hybrid_seq_parallel_attn(
        None,
        q[:, :img_q_len, :, :],
        k[:, :img_kv_len, :, :],
        v[:, :img_kv_len, :, :],
        dropout_p=0.0,
        causal=False,
        joint_tensor_query=q[:, img_q_len : cu_seqlens_q[1]],
        joint_tensor_key=k[:, img_kv_len : cu_seqlens_kv[1]],
        joint_tensor_value=v[:, img_kv_len : cu_seqlens_kv[1]],
        joint_strategy="rear",
    )
    if flash_attn.__version__ >= "2.7.0":
        attn2, *_ = _flash_attn_forward(
            q[:, cu_seqlens_q[1] :],
            k[:, cu_seqlens_kv[1] :],
            v[:, cu_seqlens_kv[1] :],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size_left=-1,
            window_size_right=-1,
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    else:
        attn2, *_ = _flash_attn_forward(
            q[:, cu_seqlens_q[1] :],
            k[:, cu_seqlens_kv[1] :],
            v[:, cu_seqlens_kv[1] :],
            dropout_p=0.0,
            softmax_scale=q.shape[-1] ** (-0.5),
            causal=False,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            return_softmax=False,
        )
    attn = torch.cat([attn1, attn2], dim=1)
    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn
