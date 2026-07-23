import pytest
import torch
import torch.nn.functional as F

from musubi_tuner.mage_flow.attention import packed_attention


def test_equal_lengths_use_one_batched_sdpa_call(monkeypatch):
    calls = []

    def fake_sdpa(q, k, v, **kwargs):
        calls.append((q.shape, k.shape, v.shape, kwargs))
        return v

    monkeypatch.setattr(F, "scaled_dot_product_attention", fake_sdpa)
    q = torch.randn(6, 2, 4)
    k = torch.randn(6, 2, 4)
    v = torch.randn(6, 2, 4)

    out = packed_attention(q, k, v, torch.tensor([0, 3, 6], dtype=torch.int32))

    assert len(calls) == 1
    assert calls[0][0] == (2, 2, 3, 4)
    assert calls[0][1] == (2, 2, 3, 4)
    assert calls[0][3]["attn_mask"] is None
    assert calls[0][3]["is_causal"] is False
    assert out.shape == q.shape
    torch.testing.assert_close(out, v)


def test_heterogeneous_segments_cannot_cross_attend():
    torch.manual_seed(10)
    q = torch.randn(6, 2, 4)
    k = torch.randn(6, 2, 4)
    v = torch.randn(6, 2, 4)
    cu = torch.tensor([0, 2, 6], dtype=torch.int32)

    baseline = packed_attention(q, k, v, cu)
    changed_values = v.clone()
    changed_values[2:] = 1000
    changed = packed_attention(q, k, changed_values, cu)

    torch.testing.assert_close(baseline[:2], changed[:2])
    assert not torch.allclose(baseline[2:], changed[2:])


def test_heterogeneous_equal_length_groups_are_batched(monkeypatch):
    calls = []
    real_sdpa = F.scaled_dot_product_attention

    def recording_sdpa(q, k, v, **kwargs):
        calls.append(q.shape)
        return real_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", recording_sdpa)
    q = torch.randn(8, 2, 4)
    cu = torch.tensor([0, 2, 4, 8], dtype=torch.int32)

    packed_attention(q, q, q, cu)

    assert sorted(calls) == sorted([torch.Size([2, 2, 2, 4]), torch.Size([1, 2, 4, 4])])


@pytest.mark.parametrize(
    ("cu", "match"),
    [
        (torch.tensor([1, 3], dtype=torch.int32), "start"),
        (torch.tensor([0, 2], dtype=torch.int64), "int32"),
        (torch.tensor([0, 3, 2], dtype=torch.int32), "increasing"),
        (torch.tensor([0, 1], dtype=torch.int32), "token count"),
    ],
)
def test_attention_rejects_invalid_boundaries(cu, match):
    q = torch.randn(2, 1, 4)
    with pytest.raises(ValueError, match=match):
        packed_attention(q, q, q, cu)


def test_attention_rejects_unknown_backend():
    q = torch.randn(2, 1, 4)
    with pytest.raises(ValueError, match="backend"):
        packed_attention(q, q, q, torch.tensor([0, 2], dtype=torch.int32), backend="magic")
