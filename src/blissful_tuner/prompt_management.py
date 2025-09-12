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
import argparse
from contextlib import nullcontext
from transformers import T5Model, T5Tokenizer
import torch
import re
from typing import Tuple, List, Union, Optional
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")


def load_t5(args, config, device):
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_tokenizer)
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.t5,
        fp8=args.fp8_t5,
    )
    text_encoder.model.to(device)
    if hasattr(args, "prompt_weighting") and args.prompt_weighting:
        text_encoder = MiniT5Wrapper(device, config.t5_dtype, text_encoder)
    return text_encoder


def rescale_text_encoders_hunyuan(llm_scale: float, clip_scale: float, transformer: torch.nn.Module) -> torch.nn.Module:
    logger.info(f"Scaling relative TE influence to LLM:{llm_scale}; CLIP:{clip_scale}")
    clip_multiplier = float(clip_scale)
    llm_multiplier = float(llm_scale)
    # Scale CLIP influence
    if hasattr(transformer, "txt_in"):
        txt_in = transformer.txt_in
        if hasattr(txt_in, "c_embedder"):
            original_c_embedder_forward = txt_in.c_embedder.forward

            def scaled_c_embedder_forward(*args, **kwargs):
                output = original_c_embedder_forward(*args, **kwargs)
                return output * clip_multiplier

            txt_in.c_embedder.forward = scaled_c_embedder_forward
            # Scale LLM influence
            if hasattr(txt_in, "individual_token_refiner"):
                for i, block in enumerate(txt_in.individual_token_refiner.blocks):
                    original_block_forward = block.forward

                    def scaled_block_forward(*args, **kwargs):
                        output = original_block_forward(*args, **kwargs)
                        return output * llm_multiplier

                    block.forward = scaled_block_forward
    return transformer


def wildcard_replace(wildcard: str, wildcard_location: str) -> str:
    """
    Replace a single __wildcard__ by picking a weighted random entry
    from the file `{wildcard}.txt` in `wildcard_location`.

    Supports subdirectories (e.g. "colors/Autumn") but forbids:
      - Absolute paths (leading '/')
      - Parent traversal ('..')
    """
    # 1) Sanitize the wildcard key
    if os.path.isabs(wildcard):
        raise ValueError(f"Absolute paths not allowed in wildcard: {wildcard!r}")
    if ".." in wildcard.split(os.sep):
        raise ValueError(f"Parent-directory traversal not allowed in wildcard: {wildcard!r}")

    # 2) Build and resolve the real path
    base_dir = os.path.abspath(wildcard_location)
    candidate = os.path.abspath(os.path.join(base_dir, f"{wildcard}.txt"))

    # 3) Ensure it's still inside base_dir
    if not (candidate == base_dir or candidate.startswith(base_dir + os.sep)):
        raise ValueError(f"Wildcard path escapes base directory: {candidate}")

    # 4) Load options & weights
    options: List[str] = []
    weights: List[float] = []
    if not os.path.isfile(candidate):
        raise FileNotFoundError(f"Wildcard file not found: {candidate}")

    with open(candidate, "r", encoding="utf-8") as f:
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
                    raise ValueError(f"Invalid weight '{w_str}' in {candidate!r} on line: {raw!r}")
            else:
                name = line
                weight = 1.0

            if name:
                options.append(name)
                weights.append(weight)

    if not options:
        raise ValueError(f"No valid options found in wildcard file: {candidate}")

    # 5) Pick one by relative weights
    return random.choices(options, weights=weights, k=1)[0]


def process_wildcards(prompts: Union[str, List[str]], wildcard_location: str, max_depth: int = 50) -> Union[str, List[str]]:
    """
    Recursively replace __keys__ in prompt(s) via wildcard_replace(key).
    Args:
        prompts:       A single prompt string or list of prompt strings.
        max_depth:     Maximum recursion depth before bailing out.
    Returns:
        The same type as `prompts`, with all __key__ markers replaced.
    """
    single = isinstance(prompts, str)
    prompt_list = [prompts] if single else list(prompts)

    pattern = re.compile(r"__([^_]+?)__")

    def replace_in_one(prompt: str) -> str:  # Helper function so we can make a listcomp
        depth = 0
        while depth < max_depth:
            # Replace each match independently, allowing repeats by chance
            replacements: List[str] = []

            def _cb(m: re.Match) -> str:
                key = m.group(1)
                val = wildcard_replace(key, wildcard_location)  # <- one draw per occurrence
                replacements.append(f"'{key}' -> '{val}'")
                return val

            new_prompt, nsubs = pattern.subn(_cb, prompt)
            if nsubs == 0:
                break

            if replacements:
                reps = ", ".join(replacements)
                logger.info(f"Wildcard replacements: {reps}")

            prompt = new_prompt
            depth += 1

        if depth >= max_depth:
            raise RecursionError(f"Wildcard recursion exceeded {max_depth} levels in prompt: {prompt}")

        return prompt

    processed = [replace_in_one(p) for p in prompt_list]
    return processed[0] if single else processed


def parse_prompt_weights(prompt: str) -> Tuple[List[str], List[float]]:
    """
    Split a diffusion prompt into (text, weight) pairs.

    Supports:
      • `(text:1.3)`   → explicit weight 1.3
      • `(text)`       → implicit “emphasis” weight 1.1
      • '(text:-2.0)'  → inversion weight
      • bare text      → default weight 1.0
    Everything is returned in the order it appears so `parts[i]` lines up with `weights[i]`.
    """
    # 1️ find every (…) group, where the :weight part is optional
    token_re = re.compile(r"\(([^:()]+?)(?::([+-]?\d*\.?\d+))?\)")

    parts: List[str] = []
    weights: List[float] = []

    idx = 0  # keeps track of how far we’ve scanned
    for m in token_re.finditer(prompt):
        # text that sits *before* this (…) block → default 1.0 weight
        if m.start() > idx:
            for seg in prompt[idx : m.start()].split(","):
                seg = seg.strip()
                if seg:
                    parts.append(seg)
                    weights.append(1.0)

        inner, w = m.group(1).strip(), m.group(2)
        parts.append(inner)
        if w is None:  # “(something)” with no :weight
            weights.append(1.1)
        else:
            weights.append(float(w))  # user-supplied weight

        idx = m.end()

    # tail text after the final (…) block
    if idx < len(prompt):
        for seg in prompt[idx:].split(","):
            seg = seg.strip()
            if seg:
                parts.append(seg)
                weights.append(1.0)
    return parts, weights


class MiniT5Wrapper:
    """A mini wrapper for the T5 to make managing prompt weighting in Musubi easier"""

    def __init__(self, device: torch.device, dtype: torch.dtype, t5: T5Model, tokenizer: T5Tokenizer, mode="wan"):
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.pad_to_max = False if mode == "wan" else True
        self.model = t5
        self.tokenizer = tokenizer
        self.times_called = 0

    def to(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Pass through for wrapped model"""
        if device and dtype:
            self.model.to(device, dtype)
        elif device:
            self.model.to(device)
        elif dtype:
            self.model.to(dtype)

    def __call__(self, prompt: Union[str, List[str]], device: torch.device, max_len: Optional[int] = None) -> Union[List[torch.Tensor], torch.tensor]:
        # ----- 0. normalise the input ------------------------------------------------
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError("MiniT5Wrapper expects exactly one prompt at a time")
            prompt = prompt[0]

        if self.times_called == 0:  # only once for noisy logs
            logger.info("Weighting prompts for T5…")
        self.times_called += 1

        # ----- 1. split “(text:weight)” chunks ---------------------------------------
        parts, weights = parse_prompt_weights(prompt)

        # ----- 2. tokenise each chunk so we know its span --------------------------
        all_ids: list[int] = []
        tok_weights: list[float] = []

        for text, w in zip(parts, weights):
            if w != 1.0:
                logger.info(f"{'Upweight' if w > 1.0 else 'Downweight' if w > 0.0 else 'Invert'} promptchunk '{text}' to '{w}x'")
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,  # add EOS once at the end
                return_attention_mask=False,
            )
            all_ids.extend(ids)
            tok_weights.extend([w] * len(ids))

        # truncate / crop if user asked for it
        if max_len is not None:
            all_ids = all_ids[: max_len - 1]  # leave room for EOS
            tok_weights = tok_weights[: max_len - 1]

        # final EOS token (T5 has no BOS; PAD is 0)
        eos_id = self.tokenizer.eos_token_id  # usually "1"
        all_ids.append(eos_id)
        tok_weights.append(1.0)  # EOS should stay neutral

        if self.pad_to_max:  # Padding comes after EOS, Flux seems to like it
            cur_len = len(all_ids)
            if cur_len < 512:
                for _ in range(512 - cur_len):
                    all_ids.append(self.tokenizer.pad_token_id)
                    tok_weights.append(1.0)

        # ----- 3. build tensors ------------------------------------------------------
        ids = torch.tensor(all_ids, dtype=torch.long, device=device).unsqueeze(0)
        mask = ids.ne(self.tokenizer.pad_token_id).int()  # 1 where real, not used for Flux

        weight_vec = torch.tensor(tok_weights, dtype=self.dtype, device=device)
        weight_vec = weight_vec.unsqueeze(0).unsqueeze(-1)  # shape [1, seq, 1]

        # ----- 4. encode & apply weights --------------------------------------------
        if self.mode == "wan":
            context = self.model(ids, mask)
        elif self.mode == "flux":
            context = self.model(ids, attention_mask=None, output_hidden_states=False)["last_hidden_state"]

        # After model, context is 1, seq_len, 4096 for wan or 1, 512, 4096 for Flux
        context = context * weight_vec  # scale token-wise
        # ----- 5. output ------------------------------------------------------------
        if self.mode == "wan":
            # trim to actual length & wrap the list Wan wants
            seq_len = mask.sum(dim=1).long().item()  # number of *real* tokens
            return [context[0, :seq_len]]

        return context  # Just return as is for Flux


def prepare_wan_special_inputs(args: argparse.Namespace, device: torch.device, config, t5) -> Union[dict, dict]:
    """Prepare special model inputs for Wan models in any mode"""
    perp_neg_context = nag_context = None
    if args.perp_neg or args.nag_prompt:
        if t5 is None:
            logger.info("Loading T5 text encoder")
            t5 = load_t5(args, config, device)
        t5.model.to(device)
        if args.perp_neg:
            logger.info("Encoding unconditional context for perpendicular negative")
            with (
                torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype) if args.fp8_t5 else nullcontext(),
                torch.no_grad(),
            ):
                perp_neg_context = t5("", device)  # Encode a blank prompt for true unconditional guidance
        if args.nag_prompt:
            logger.info("Encoding a separate negative prompt for NAG")
            with (
                torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype) if args.fp8_t5 else nullcontext(),
                torch.no_grad(),
            ):
                nag_context = t5(args.nag_prompt, device)

    return perp_neg_context, nag_context
