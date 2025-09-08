#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Blissful Tuner extension
License: Apache 2.0
Created on Sat Apr 12 14:09:37 2025

@author: blyss
"""

import random
import hashlib
import torch
import safetensors
import numpy as np
from typing import Union, Dict, Tuple, Optional, Type

try:
    from blissful_logger import BlissfulLogger
except ImportError:
    from blissful_tuner.blissful_logger import BlissfulLogger


# Adapted from ComfyUI
def load_torch_file(
    ckpt: str, safe_load: bool = True, device: Optional[Union[str, torch.device]] = None, return_metadata: bool = False
) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]]:
    if device is None:
        device = torch.device("cpu")
    metadata = None
    if ckpt.lower().endswith(".safetensors") or ckpt.lower().endswith(".sft"):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {}
                for k in f.keys():
                    sd[k] = f.get_tensor(k)
                if return_metadata:
                    metadata = f.metadata()
        except Exception as e:
            if len(e.args) > 0:
                message = e.args[0]
                if "HeaderTooLarge" in message:
                    raise ValueError(
                        "{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(
                            message, ckpt
                        )
                    )
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError(
                        "{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(
                            message, ckpt
                        )
                    )
            raise e
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=safe_load)

        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            if len(pl_sd) == 1:
                key = list(pl_sd.keys())[0]
                sd = pl_sd[key]
                if not isinstance(sd, dict):
                    sd = pl_sd
            else:
                sd = pl_sd
    return (sd, metadata) if return_metadata else sd


# Below here, Blyss wrote it!


def str_to_dtype(dtype_str: str):
    dtype_mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "float": torch.float32,
        "fp8": torch.float8_e4m3fn,
        "float8": torch.float8_e4m3fn,
    }
    if dtype_str in dtype_mapping:
        return dtype_mapping[dtype_str]
    else:
        error_out(ValueError, f"Unknown dtype string '{dtype_str}'")


def setup_compute_context(
    device: Optional[Union[torch.device, str]] = None, dtype: Optional[Union[torch.dtype, str]] = None
) -> Tuple[torch.device, torch.dtype]:
    logger = BlissfulLogger(__name__, "#8e00ed")

    if device is None:
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
    elif isinstance(device, str):
        device = torch.device(device)

    if dtype is None:
        dtype = torch.float32
    elif isinstance(dtype, str):
        dtype = str_to_dtype(dtype)

    torch.set_float32_matmul_precision("high")
    if dtype == torch.float16 or dtype == torch.bfloat16:
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            logger.info("FP16 accumulation enabled.")
    return device, dtype


def string_to_seed(s: str, bits: int = 63, silent: bool = False) -> int:
    """
    Turn any string into a reproducible integer in [0, 2**bits) with a hash and some other logic.

    Args:
        s:           Input string
        bits:        Number of bits for the final seed (PyTorch accepts up to 63 safely, numpy likes 32)
    Returns:
        A non-negative int < 2**bits
    """
    logger = BlissfulLogger(__name__, "#8e00ed")
    digest = hashlib.sha256(s.encode("utf-8")).digest()
    crypto = int.from_bytes(digest, byteorder="big")
    mask = (1 << bits) - 1
    algo = 0
    for i, char in enumerate(s):
        char_val = ord(char)
        if i % 2 == 0:
            algo += char_val
        elif i % 3 == 0:
            algo -= char_val
        elif i % 5 == 0:
            algo /= char_val
        else:
            char_val_str = str(char_val)
            for digit in char_val_str:
                algo *= int(digit) if digit != "0" else 0.31415  # Prevents us from ascending to infinity
    if algo == float("inf"):  # In case we somehow still do
        algo = len(s) + (314159 * ord(s[len(s) // 2])) - ord(s[len(s) // 4])
    seed = (abs(crypto - int(algo))) & mask
    if not silent:
        logger.info(f"Seed '{seed}' was generated from string '{s}'")
    return seed


def power_seed(seed: Union[int, str] = None) -> int:
    """
    Sets the random seed for reproducibility.
    """
    logger = BlissfulLogger(__name__, "#8e00ed")
    if seed is None:
        seed = random.getrandbits(32)
    else:
        try:
            seed = int(seed)
            msg = f"Seed '{seed}' was set globally!"
        except ValueError:
            s = seed
            seed = string_to_seed(seed, bits=32, silent=True)
            msg = f"Seed '{seed}' was generated from string '{s}' and set globally!"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(msg)
    return seed


def error_out(error: Type[Exception], message: str) -> None:
    logger = BlissfulLogger(__name__, "#8e00ed")
    logger.error(message, levelmod=1)
    raise error(message)
