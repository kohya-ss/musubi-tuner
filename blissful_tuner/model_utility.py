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
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="Convert any model checkpoint (single file or shard directory) to safetensors with dtype cast."
)
parser.add_argument(
    "--input",
    required=True,
    help="Checkpoint file or directory of shards to convert/inspect"
)
parser.add_argument("--convert", type=str, default=None)
parser.add_argument("--inspect", action="store_true")
parser.add_argument("--key_target", type=str)
parser.add_argument("--weights_only", type=str, default="true")
parser.add_argument("--dtype", type=str)
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
            if not path.lower().endswith((".safetensors", ".sft", ".pt", ".pth")):
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


print("Loading checkpoint...")
weights_only = args.weights_only.lower() == "true"
checkpoint = load_torch_file(args.input, weights_only)

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
        print("Aborting.")
        exit()

converted_state_dict = {}
keys_to_process = (
    [k for k in checkpoint if args.key_target in k] if args.key_target else checkpoint.keys()
)
dtypes_in_model = {}
for key in tqdm(keys_to_process, desc="Processing tensors"):
    value = checkpoint[key]
    if args.inspect:
        print(f"{key}: {value.shape} ({value.dtype})")
    dtype_to_use = (
        dtype_mapping.get(args.dtype.lower(), value.dtype)
        if args.dtype
        else value.dtype
    )
    if dtype_to_use not in dtypes_in_model:
        dtypes_in_model[dtype_to_use] = 1
    else:
        dtypes_in_model[dtype_to_use] += 1
    if args.convert:
        converted_state_dict[key] = value.to(dtype_to_use)


print(f"Dtypes in model: {dtypes_in_model}")
if args.convert:
    output_file = (
        args.convert.replace(".pth", ".safetensors")
        .replace(".pt", ".safetensors")
    )
    print(f"Saving converted tensors to '{output_file}'...")
    save_file(converted_state_dict, output_file)
