"""Regression tests for krea2_sampling.gather_valid_text.

Background (the bug this locks): the Qwen3-VL conditioner pads the prompt to
max_length and then appends the template suffix, so its attention mask is
[valid prompt, pad, valid suffix] — the valid tokens are NOT a contiguous prefix.
The shared attention (cu_seqlens / trim) assumes valid == leading prefix, so an
interior-padded sequence fed straight through dropped the suffix conditioning and
produced corrupted (psychedelic) images. gather_valid_text() compacts the valid
tokens to a contiguous prefix; dropping the padding is lossless because text tokens
get zero RoPE position and padding is masked out.

These tests are pure-tensor (no GPU, no weights) so they can run anywhere.
"""

import torch

from musubi_tuner.krea2.krea2_sampling import gather_valid_text


def _make_tokens(seq, dim=4, layers=2):
    """Distinct, non-zero per-token feature blocks so we can track identity/order."""
    base = torch.arange(1, seq + 1, dtype=torch.float32).view(seq, 1, 1)
    return base * torch.ones(seq, layers, dim, dtype=torch.float32)


def test_interior_padding_is_compacted_to_a_prefix_losslessly():
    # mask = [valid prompt, PAD, valid suffix] — valid tokens are interior, not a prefix.
    txt = _make_tokens(seq=5).unsqueeze(0)  # (1, 5, L, D)
    mask = torch.tensor([[True, True, False, False, True]])

    out, newmask = gather_valid_text(txt, mask)

    # 3 valid tokens collapse to a leading prefix; the rest is zero padding.
    assert newmask.tolist() == [[True, True, True]]
    # Order preserved: tokens 1, 2 (prompt) then token 5 (suffix), padding dropped.
    expected = torch.stack([txt[0, 0], txt[0, 1], txt[0, 4]]).unsqueeze(0)
    assert torch.equal(out, expected)


def test_already_prefix_valid_is_unchanged_up_to_trim():
    txt = _make_tokens(seq=4).unsqueeze(0)
    mask = torch.tensor([[True, True, False, False]])

    out, newmask = gather_valid_text(txt, mask)

    assert newmask.tolist() == [[True, True]]
    assert torch.equal(out, txt[:, :2])


def test_batch_right_pads_to_max_valid_length():
    # Row 0 has 1 valid token, row 1 has 3 (with interior padding).
    txt = _make_tokens(seq=5).unsqueeze(0).repeat(2, 1, 1, 1)  # (2, 5, L, D)
    mask = torch.tensor(
        [
            [True, False, False, False, False],
            [True, False, True, False, True],
        ]
    )

    out, newmask = gather_valid_text(txt, mask)

    # Padded to the batch max of 3 valid tokens.
    assert out.shape[1] == 3
    assert newmask.tolist() == [[True, False, False], [True, True, True]]
    # Row 0: single valid token then zero padding.
    assert torch.equal(out[0, 0], txt[0, 0])
    assert torch.count_nonzero(out[0, 1:]) == 0
    # Row 1: valid tokens 1, 3, 5 compacted in order.
    assert torch.equal(out[1], torch.stack([txt[1, 0], txt[1, 2], txt[1, 4]]))


def test_preserves_dtype_and_device():
    txt = _make_tokens(seq=3).unsqueeze(0).to(torch.bfloat16)
    mask = torch.tensor([[True, False, True]])

    out, newmask = gather_valid_text(txt, mask)

    assert out.dtype == torch.bfloat16
    assert out.device == txt.device
    assert newmask.dtype == torch.bool
