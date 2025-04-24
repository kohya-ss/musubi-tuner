#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:51:05 2025
Prompt weighting for WanVideo
Adapted and heavily modified from https://github.com/xhinker/sd_embed
License: Apache 2.0
@author: blyss
"""
from transformers import T5EncoderModel, T5Tokenizer
import torch
import re
from typing import Tuple, List


def get_prompts_tokens_with_weights_t5(
    t5_tokenizer: T5Tokenizer, prompt: str
) -> Tuple[List[int], List[float]]:
    """
    Tokenize `prompt` once and return
      - input_ids:    [token_id, …]
      - token_weights: [weight, …] aligned 1:1 with input_ids
    honoring the (text,weight) chunks produced by parse_prompt_attention.
    """
    if not prompt:
        prompt = "empty"

    # 1) break into (text, weight) chunks
    chunks = parse_prompt_attention(prompt)

    # 2) build the concatenated string and record each chunk's char spans
    full_text = ""
    spans: List[Tuple[int,int,float]] = []  # [(start, end, weight), …]
    cursor = 0
    for text, w in chunks:
        full_text += text
        spans.append((cursor, cursor + len(text), w))
        cursor += len(text)

    # 3) run the tokenizer once, capturing char offsets
    encoded = t5_tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    ids = encoded["input_ids"][0].tolist()
    offsets = encoded["offset_mapping"][0].tolist()  # list of (start,end) per token

    # 4) map each token to its chunk-weight by looking at its start offset
    token_weights: List[float] = []
    for (s, e) in offsets:
        # special tokens often have (0,0) offsets; give them weight=1
        if s == e == 0:
            token_weights.append(1.0)
            continue

        # find the span that covers `s`
        for (st, en, w) in spans:
            if st <= s < en:
                token_weights.append(w)
                break
        else:
            # fallback
            token_weights.append(1.0)

    return ids, token_weights


def get_weighted_prompt_embeds_t5(
    prompt: str,
    t5: T5EncoderModel,
    device: torch.device,
    max_len: int = None
) -> torch.Tensor:
    """
    Produce a single tensor of shape (1, seq_len, hidden_dim) where
    each token embedding is scaled by its prompt-weight.

    Returns:
        weighted_embeds: torch.FloatTensor[1, seq_len, hidden_dim]
    """
    tokenizer = t5.tokenizer
    # 1) grab flat ids + weights
    ids_flat, weights_flat = get_prompts_tokens_with_weights_t5(tokenizer, prompt)

    # 2) optionally truncate
    if max_len is not None:
        ids_flat = ids_flat[:max_len]
        weights_flat = weights_flat[:max_len]

    # 3) batchify
    ids = torch.tensor([ids_flat], device=device)          # (1, L)
    attn_mask = (ids != tokenizer.pad_token_id).long()     # (1, L)

    # 4) encode
    with torch.no_grad():
        outputs = t5.encoder(input_ids=ids, attention_mask=attn_mask)
        hidden = outputs.last_hidden_state                 # (1, L, D)

    # 5) apply per-token weights
    weight = torch.tensor(weights_flat, device=device)      # (L,)
    weighted = hidden * weight.unsqueeze(0).unsqueeze(-1)   # broadcast to (1,L,D)
    weighted = weighted.squeeze(0)    # → (L, D)
    return [weighted]


re_attention = re.compile(r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""", re.X)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text: str) -> List[Tuple[str, float]]:
    """
    Parses a string with attention tokens and returns a list of (text, weight) pairs.
    """
    res: List[Tuple[str, float]] = []
    round_brackets: List[int] = []
    square_brackets: List[int] = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start: int, multiplier: float):
        for p in range(start, len(res)):
            chunk, w = res[p]
            res[p] = (chunk, w * multiplier)

    for m in re_attention.finditer(text):
        tok = m.group(0)
        weight = m.group(1)

        if tok.startswith('\\'):
            res.append((tok[1:], 1.0))
        elif tok == '(':
            round_brackets.append(len(res))
        elif tok == '[':
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif tok == ')' and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif tok == ']' and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, tok)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(("BREAK", -1.0))
                res.append((part, 1.0))

    # close any unclosed brackets
    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if not res:
        res = [("", 1.0)]

    # merge consecutive runs with identical weights
    i = 0
    while i + 1 < len(res):
        text_i, w_i = res[i]
        text_j, w_j = res[i + 1]
        if w_i == w_j:
            # merge
            res[i] = (text_i + text_j, w_i)
            res.pop(i + 1)
        else:
            i += 1

    return res
