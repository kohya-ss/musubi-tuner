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


def get_prompts_tokens_with_weights_t5(t5_tokenizer: T5Tokenizer, prompt: str) -> Tuple[List[int], List[float]]:
    """
    Returns two lists:
      • ids_flat  = [int, int, …]    one raw token‑ID per real wordpiece
      • weights_flat    = [float, …]       matching weight per token
    """
    if not prompt:
        prompt = "empty"

    texts_and_weights = parse_prompt_attention(prompt)
    ids_flat, weights_flat = [], []

    for chunk, weight in texts_and_weights:
        # this returns tensor of (1, N)
        ids_tensor = t5_tokenizer(chunk, add_special_tokens=True, padding=False)
        # squeeze off the batch dim → shape (N,)
        token_ids = ids_tensor.squeeze(0).tolist()

        ids_flat.extend(token_ids)
        weights_flat.extend([weight] * len(token_ids))
    return ids_flat, weights_flat


def get_weighted_prompt_embeds_t5(prompt: str, t5: T5EncoderModel, device: torch.device, max_len: int = None) -> List[torch.Tensor]:
    # 1) build flat lists of ids & weights
    ids_flat, weights_flat = get_prompts_tokens_with_weights_t5(t5.tokenizer, prompt)

    # 2) optionally truncate
    if max_len is not None:
        ids_flat = ids_flat[:max_len]
        weights_flat = weights_flat[:max_len]

    # 3) wrap into a single batch‑dim Tensor
    ids = torch.tensor([ids_flat], dtype=torch.long, device=device)  # (1, seq_len)
    mask = (ids != 0).long()  # (1, seq_len)

    # 4) encode and drop batch dim
    hidden = t5.model(ids, mask)        # returns (1, seq_len, hidden_dim)
    hidden = hidden.squeeze(0)          # now (seq_len, hidden_dim)

    # 5) apply per‑token weights
    weight = torch.tensor(weights_flat, device=device)                      # (seq_len,)
    hidden = hidden * weight[:, None]  # broadcast → (seq_len, hidden_dim)

    return [hidden]


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
