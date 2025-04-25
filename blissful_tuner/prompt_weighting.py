#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:51:05 2025
Prompt weighting for WanVideo
Adapted and heavily modified from https://github.com/xhinker/sd_embed
License: Apache 2.0
@author: blyss
"""
from transformers import T5Model
import torch
import re
from typing import Tuple, List


def get_weighted_prompt_embeds_t5(
    prompt: str,
    t5: T5Model,
    device: torch.device,
    max_len: int = None
) -> List[torch.Tensor]:

    # Split positive prompts and process each with weights
    prompts_raw = [p.strip() for p in prompt.split('|')]
    prompts = []
    all_weights = []

    for p in prompts_raw:
        cleaned_prompt, weights = parse_prompt_weights(p)
        prompts.append(cleaned_prompt)
        all_weights.append(weights)
    t5.model.to(device)
    context = t5(prompts, device)

    # Apply weights to embeddings if any were extracted
    for i, weights in enumerate(all_weights):
        for text, weight in weights.items():
            print(f"Applying weight {weight} to prompt: {text}")
            if len(weights) > 0:
                context[i] = context[i] * weight

    return context


def parse_prompt_weights(prompt: str) -> Tuple[str, dict]:
    """Extract text and weights from prompts with (text:weight) format"""
    # Parse all instances of (text:weight) in the prompt
    pattern = r'\((.*?):([\d\.]+)\)'
    matches = re.findall(pattern, prompt)

    # Replace each match with just the text part
    cleaned_prompt = prompt
    weights = {}

    for match in matches:
        text, weight = match
        orig_text = f"({text}:{weight})"
        cleaned_prompt = cleaned_prompt.replace(orig_text, text)
        weights[text] = float(weight)

    return cleaned_prompt, weights
