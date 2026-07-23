from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn.functional as F


def _validate_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> tuple[list[int], list[int]]:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed attention q, k, and v must have shape [tokens, heads, head_dim]")
    if k.shape != v.shape:
        raise ValueError(f"packed attention k and v shapes must match, got {tuple(k.shape)} and {tuple(v.shape)}")
    if q.shape[1:] != k.shape[1:]:
        raise ValueError("packed attention q and k/v must have matching head and head-dimension sizes")

    def validate_cu(name: str, cu: torch.Tensor, total: int) -> list[int]:
        if cu.ndim != 1 or cu.numel() < 2:
            raise ValueError(f"{name} must be rank 1 with at least two entries")
        if cu.dtype != torch.int32:
            raise ValueError(f"{name} must use torch.int32")
        values = cu.detach().cpu().tolist()
        if values[0] != 0:
            raise ValueError(f"{name} must start at zero")
        if values[-1] != total:
            raise ValueError(f"{name} must end at token count {total}, got {values[-1]}")
        if any(end <= start for start, end in zip(values, values[1:])):
            raise ValueError(f"{name} must be strictly increasing")
        return values

    q_cu = validate_cu("cu_seqlens_q", cu_seqlens_q, q.shape[0])
    k_cu = validate_cu("cu_seqlens_k", cu_seqlens_k, k.shape[0])
    if len(q_cu) != len(k_cu):
        raise ValueError("query and key cumulative lengths must contain the same number of segments")
    return q_cu, k_cu


def _sdpa_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_cu: list[int],
    k_cu: list[int],
    *,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (q_start, q_end, k_start, k_end) in enumerate(zip(q_cu, q_cu[1:], k_cu, k_cu[1:])):
        groups[(q_end - q_start, k_end - k_start)].append(index)

    output = torch.empty_like(q)
    for (query_length, key_length), indices in groups.items():
        q_batch = torch.stack([q[q_cu[index] : q_cu[index + 1]] for index in indices], dim=0).transpose(1, 2)
        k_batch = torch.stack([k[k_cu[index] : k_cu[index + 1]] for index in indices], dim=0).transpose(1, 2)
        v_batch = torch.stack([v[k_cu[index] : k_cu[index + 1]] for index in indices], dim=0).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            q_batch,
            k_batch,
            v_batch,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        ).transpose(1, 2)
        if attended.shape[1] != query_length or k_batch.shape[2] != key_length:
            raise RuntimeError("SDPA returned an unexpected packed segment shape")
        for batch_index, segment_index in enumerate(indices):
            output[q_cu[segment_index] : q_cu[segment_index + 1]] = attended[batch_index]
    return output


def _flash_attention_2_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    q_cu: list[int],
    k_cu: list[int],
    *,
    softmax_scale: float | None,
    causal: bool,
) -> torch.Tensor:
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise ImportError(
            "FlashAttention 2 was requested for Mage-Flow but flash_attn is not installed; "
            "use --attn_mode sdpa or install a compatible optional flash-attn build"
        ) from exc
    return flash_attn_varlen_func(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max(end - start for start, end in zip(q_cu, q_cu[1:])),
        max_seqlen_k=max(end - start for start, end in zip(k_cu, k_cu[1:])),
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
    )


def packed_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor | None = None,
    *,
    backend: str = "sdpa",
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    """Run isolated variable-length attention over packed token segments.

    Equal-sized segments are reshaped into a real batch and dispatched through
    one SDPA call. Heterogeneous inputs are grouped by `(query_len, key_len)`;
    no path attends over the unmasked concatenated sequence.
    """

    if cu_seqlens_k is None:
        cu_seqlens_k = cu_seqlens_q
    q_cu, k_cu = _validate_qkv(q, k, v, cu_seqlens_q, cu_seqlens_k)
    normalized_backend = backend.lower().strip()
    if normalized_backend in {"sdpa", "torch", "torch_sdpa"}:
        return _sdpa_packed(q, k, v, q_cu, k_cu, softmax_scale=softmax_scale, causal=causal)
    if normalized_backend in {"flash2", "fa2", "flash_attention_2", "flash_attn_2"}:
        return _flash_attention_2_packed(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            q_cu,
            k_cu,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    raise ValueError(f"unknown Mage-Flow attention backend {backend!r}; expected 'sdpa' or 'flash2'")


__all__ = ["packed_attention"]
