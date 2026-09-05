from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musubi_tuner.minimax_h3 import model as h3_model
from musubi_tuner.minimax_h3.model import _INT32_SAFE_EXTENT, _element_extent, Attention


def _fused_value_view(batch: int, seq: int, heads: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    inner = heads * head_dim
    fused = torch.arange(batch * seq * 3 * inner, dtype=torch.float32).reshape(batch, seq, 3 * inner)
    value = fused.split(inner, dim=-1)[2].reshape(batch, seq, heads, head_dim)
    return fused, value


def test_element_extent_of_fused_value_view_spans_the_buffer():
    fused, value = _fused_value_view(batch=1, seq=4, heads=2, head_dim=3)
    assert not value.is_contiguous()
    # the value view's last element is the last element of the fused buffer
    assert _element_extent(value) == fused.numel()
    assert _element_extent(fused) == fused.numel()
    assert _element_extent(value.contiguous()) == value.numel()


def test_element_extent_h3_scale_decision_boundary():
    # H3 dims: heads 56 x dim 128, fused width 21504. The strided value view crosses the
    # int32 line between the released 277f (~85k rows, clean in practice) and 362f
    # (~109k rows, corrupted by SDPA's memory-efficient backend before the guard).
    def fake_value_view(seq: int) -> SimpleNamespace:
        inner = 56 * 128
        return SimpleNamespace(
            storage_offset=lambda: 2 * inner,
            shape=(1, seq, 56, 128),
            stride=lambda: (seq * 3 * inner, 3 * inner, 128, 1),
        )

    # the view ends at the fused buffer's end, so its extent is exactly seq * 21504:
    # 99,864 * 21504 = 2,147,475,456 is the last size inside int32
    assert _element_extent(fake_value_view(85_000)) <= _INT32_SAFE_EXTENT
    assert _element_extent(fake_value_view(99_864)) <= _INT32_SAFE_EXTENT
    assert _element_extent(fake_value_view(99_865)) > _INT32_SAFE_EXTENT
    assert _element_extent(fake_value_view(110_500)) > _INT32_SAFE_EXTENT


def test_attention_forward_materializes_oversized_value_view(monkeypatch):
    torch.manual_seed(0)
    attention = Attention(
        hidden_size=8,
        num_heads=2,
        head_dim=4,
        qk_norm_eps=1e-6,
        attn_mode="sdpa",
        split_attn=False,
        dtype=torch.float32,
    )
    hidden = torch.randn(1, 5, 8)
    reference = attention(hidden)
    # force the guard on and make sure the contiguous() path is taken and harmless
    monkeypatch.setattr(h3_model, "_INT32_SAFE_EXTENT", 0)
    contiguized = attention(hidden)
    assert torch.allclose(reference, contiguized, atol=1e-6)
