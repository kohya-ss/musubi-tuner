#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 12:51:05 2025
Extensions for prompt related stuff for Blissful Tuner
License: Apache 2.0
@author: blyss
"""
import os
import random
from transformers import T5Model
import torch
import re
from typing import Tuple, List, Union
from blissful_tuner.utils import BlissfulLogger
logger = BlissfulLogger(__name__, "#8e00ed")


def wildcard_replace(wildcard: str, wildcard_location: str) -> str:
    """
    Replace a single __wildcard__ by picking a weighted random entry
    from the file `{wildcard}.txt` in `wildcard_location`.

    File format (one entry per line):
        option_name       → weight = 1.0
        option_name:2.5   → weight = 2.5

    Weights are used *relatively*, so option A with weight=2.0
    is twice as likely as option B with weight=1.0.

    Raises:
      FileNotFoundError if the file isn't there.
      ValueError for malformed weights or empty files.
    """
    path = os.path.join(wildcard_location, f"{wildcard}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Wildcard file not found: {path}")

    options = []
    weights = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if ":" in line:
                name, w_str = line.split(":", 1)
                name = name.strip()
                try:
                    weight = float(w_str.strip())
                except ValueError:
                    raise ValueError(
                        f"Invalid weight '{w_str}' in {path} on line: {raw!r}"
                    )
            else:
                name = line
                weight = 1.0

            if name:
                options.append(name)
                weights.append(weight)

    if not options:
        raise ValueError(f"No valid options found in wildcard file: {path}")

    # random.choices uses weights relatively
    return random.choices(options, weights=weights, k=1)[0]


def process_wildcards(
    prompts: Union[str, List[str]],
    wildcard_location: str,
    max_depth: int = 50
) -> Union[str, List[str]]:
    """
    Recursively replace __keys__ in prompt(s) via wildcard_replace(key).

    Args:
        prompts:       A single prompt string or list of prompt strings.
        wildcard_replace:  Function that maps a key (no underscores) to its replacement str.
        max_depth:     Maximum recursion depth before bailing out.

    Returns:
        The same type as `prompts`, with all __key__ markers replaced.
    """
    # Normalize to list
    single = isinstance(prompts, str)
    prompt_list = [prompts] if single else list(prompts)

    pattern = re.compile(r"__([^_]+?)__")

    def replace_in_one(prompt: str) -> str:
        replacements = []
        depth = 0
        while depth < max_depth:
            # find all wildcard markers in this prompt
            matches = pattern.findall(prompt)
            if not matches:
                break
            # for each unique key, get replacement and do a global sub
            for key in set(matches):
                replacement = wildcard_replace(key, wildcard_location)
                prompt = re.sub(f"__{re.escape(key)}__", replacement, prompt)
                replacements.append(f"{key} -> {replacement}")
            depth += 1

        if depth >= max_depth:
            raise RecursionError(f"Wildcard recursion exceeded {max_depth} levels in prompt: {prompt}")
        if len(replacements) != 0:
            replacement_string = ", ".join(replacements)
            logger.info(f"Wildcard replacements: {replacement_string}")
        return prompt

    # Process each prompt
    processed = [replace_in_one(p) for p in prompt_list]

    # Return same type as input
    return processed[0] if single else processed


class MiniT5Wrapper():
    """A mini wrapper for the T5 to make managing prompt weighting in Musubi easier"""

    def __init__(self, device: torch.device, dtype: torch.dtype, t5: T5Model):
        self.device = device
        self.dtype = dtype
        self.t5 = t5
        self.model = t5.model
        self.times_called = 0

    def __call__(
        self,
        prompt: Union[str, List[str]],
        device: torch.device,
        max_len: int = None
    ) -> List[torch.Tensor]:
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniT5Wrapper expects a single prompt at a time (wrapped as a list). Got multiple prompts.")
            prompt = prompt[0]
        if self.times_called == 0:  # Only print this notice once even if called multiple times
            logger.info("Weighting prompts...")
        # Split positive prompts and process each with weights
        prompts_raw = [p.strip() for p in prompt.split('|')]
        prompts = []
        all_weights = []

        for p in prompts_raw:
            cleaned_prompt, weights = self.parse_prompt_weights(p)
            prompts.append(cleaned_prompt)
            all_weights.append(weights)
        context = self.t5(prompts, device)

        # Apply weights to embeddings if any were extracted
        for i, weights in enumerate(all_weights):
            for text, weight in weights.items():
                logger.info(f"Applying weight ({weight}) to promptchunk: '{text}'")
                if len(weights) > 0:
                    context[i] = context[i] * weight
        self.times_called += 1
        return context

    def parse_prompt_weights(self, prompt: str) -> Tuple[str, dict]:
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
