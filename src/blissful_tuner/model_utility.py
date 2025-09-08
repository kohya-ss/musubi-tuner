#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model inspection and conversion utility for Blissful Tuner Extension

License: Apache 2.0
Created on Wed Apr 23 10:19:19 2025
@author: blyss
"""

import os
import argparse
import torch
import safetensors
from safetensors.torch import save_file
from rich_argparse import RichHelpFormatter
from rich.traceback import install as install_rich_tracebacks
from blissful_tuner.blissful_logger import BlissfulLogger
logger = BlissfulLogger(__name__, "#8e00ed")

install_rich_tracebacks()

parser = argparse.ArgumentParser(
    description="Utility for inspecting model structure and converting between dtypes and key naming. Supports loading single safetensors or sharded, saving is single safetensors",
    formatter_class=RichHelpFormatter,
)
parser.add_argument("--input", required=True, help="Checkpoint file or directory of shards to convert/inspect")
parser.add_argument(
    "--convert",
    type=str,
    default=None,
    help="/path/to/output.safetensors, If provided, the model will be loaded, processed and written to this file",
)
parser.add_argument(
    "--inspect",
    action="store_true",
    help="If provided, will print out the keys in the model's state dict along with their dtype and shape. Also runs the same processes as --convert so can be used as a dry run.",
)
parser.add_argument("--target_keys", nargs="*", type=str, default=None, help="Keys to target for dtype conversion")
parser.add_argument("--exclude_keys", nargs="*", type=str, default=None, help="Keys to exclude for dtype conversion")
parser.add_argument("--fully_exclude_keys", nargs="*", type=str, default=None, help="Keys to exclude from copying at all")
parser.add_argument("--strip_prefix", type=str, default=None, help="If specified and matched, prefix will be stripped for keys in state dict")
parser.add_argument(
    "--weights_only",
    action="store_false",
    help="Whether to load the model using 'weights_only' which can be safer. Default is true, don't change unless needed",
)
parser.add_argument(
    "--dtype",
    type=str,
    help="Datatype to convert tensors to when using --convert. "
    "If --target_keys or --exclude_keys is specified, only target keys that aren't excluded will have their dtype converted. "
    "This can be useful for creating mixed precision models!",
)
args = parser.parse_args()


def load_torch_file(ckpt, weights_only=True, device=None, return_metadata=False):
    """
    Load a single checkpoint file or all shards in a directory.
    - If `ckpt` is a dir, iterates over supported files, loads each, and merges.
    - Returns state_dict (and metadata if return_metadata=True and single file).
    """
    if device is None:
        device = torch.device("cpu")

    # --- shard support ---
    if os.path.isdir(ckpt):
        all_sd = {}
        for fname in sorted(os.listdir(ckpt)):
            path = os.path.join(ckpt, fname)
            # only load supported extensions
            if not os.path.isfile(path):
                continue
            if not path.lower().endswith((".safetensors", ".sft", ".pt", ".pth", ".bin")):
                continue
            # load each shard (we ignore metadata for shards)
            shard_sd = load_torch_file(path, weights_only, device, return_metadata=False)
            all_sd.update(shard_sd)
        return (all_sd, None) if return_metadata else all_sd

    # --- single file ---
    metadata = None
    if ckpt.lower().endswith((".safetensors", ".sft")):
        try:
            with safetensors.safe_open(ckpt, framework="pt", device=device.type) as f:
                sd = {k: f.get_tensor(k) for k in f.keys()}
                metadata = f.metadata() if return_metadata else None
        except Exception as e:
            raise ValueError(f"Safetensors load failed: {e}\nFile: {ckpt}")
    else:
        pl_sd = torch.load(ckpt, map_location=device, weights_only=weights_only)
        sd = pl_sd.get("state_dict", pl_sd)

    return (sd, metadata) if return_metadata else sd


logger.info("Loading checkpoint...")
checkpoint = load_torch_file(args.input, args.weights_only)

dtype_mapping = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

if args.convert is not None and os.path.exists(args.convert):
    confirm = input(f"{args.convert} exists. Overwrite? [y/N]: ").strip().lower()
    if confirm != "y":
        logger.info("Aborting.")
        exit()

converted_state_dict = {}
keys_to_process = checkpoint.keys()
dtypes_in_model = {}
for key in keys_to_process:
    is_fully_excluded = args.fully_exclude_keys is not None and any(pattern in key for pattern in args.fully_exclude_keys)
    if is_fully_excluded:
        continue  # Skip this key
    is_target = args.target_keys is None or any(pattern in key for pattern in args.target_keys)
    is_excluded = args.exclude_keys is not None and any(pattern in key for pattern in args.exclude_keys)
    is_target = is_target and not is_excluded
    value = checkpoint[key]
    if is_target:
        if args.strip_prefix is not None and key.startswith(args.strip_prefix):
            logger.info(f"'{key}' had it's prefix stripped per rules!")
            key = key.replace(args.strip_prefix, "")
    dtype_to_use = dtype_mapping.get(args.dtype.lower(), value.dtype) if args.dtype else value.dtype
    final_dtype = dtype_to_use if is_target else value.dtype
    if args.convert:
        converted_state_dict[key] = value.to(final_dtype)
    if final_dtype not in dtypes_in_model:
        dtypes_in_model[final_dtype] = 1
    else:
        dtypes_in_model[final_dtype] += 1
    if args.convert or (args.inspect and is_target):
        logger.info(f"'{key}': shape={value.shape} dtype='{final_dtype}'")
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


logger.info(f"Tensor and dtypes in model: {dtypes_in_model}")
if args.convert:
    output_file = args.convert.replace(".pth", ".safetensors").replace(".pt", ".safetensors")
    logger.info(f"Saving converted tensors to '{output_file}'...")
    save_file(converted_state_dict, output_file)
