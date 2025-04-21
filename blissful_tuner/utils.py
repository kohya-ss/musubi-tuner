#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for Blissful Tuner extension
License: Apache 2.0
Created on Sat Apr 12 14:09:37 2025

@author: blyss
"""
import argparse
import torch
import safetensors
from typing import List, Union, Dict, Tuple, Optional


# Adapted from ComfyUI
def load_torch_file(
    ckpt: str,
    safe_load: bool = False,
    device: Union[str, torch.device] = None,
    return_metadata: bool = False
) -> Union[
    Dict[str, torch.Tensor],
    Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]
]:
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
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt or invalid. Make sure this is actually a safetensors file and not a ckpt or pt or other filetype.".format(message, ckpt))
                if "MetadataIncompleteBuffer" in message:
                    raise ValueError("{}\n\nFile path: {}\n\nThe safetensors file is corrupt/incomplete. Check the file size and make sure you have copied/downloaded it correctly.".format(message, ckpt))
            raise e
    else:

        pl_sd = torch.load(ckpt, map_location=device, weights_only=True)

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


# Adapted from WanVideoWrapper
def add_noise_to_reference_video(image: torch.Tensor, ratio: float = None) -> torch.Tensor:
    sigma = torch.ones((image.shape[0],)).to(image.device, image.dtype) * ratio
    image_noise = torch.randn_like(image) * sigma[:, None, None, None]
    image_noise = torch.where(image == -1, torch.zeros_like(image), image_noise)
    image = image + image_noise
    return image


# Blyss wrote it!
def parse_scheduled_cfg(schedule: str, infer_steps: int) -> List[int]:
    """
    Parse a schedule string like "1-10,20,!5,e~3" into a sorted list of steps.

    - "start-end" includes all steps in [start, end]
    - "e~n"    includes every nth step (n, 2n, ...) up to infer_steps
    - "x"      includes the single step x
    - Prefix "!" on any token to exclude those steps instead of including them.

    Raises argparse.ArgumentTypeError on malformed tokens or out-of-range steps.
    """
    included = set()
    excluded = set()

    for raw in schedule.split(","):
        token = raw.strip()
        if not token:
            continue  # skip empty tokens

        # exclusion if it starts with "!"
        if token.startswith("!"):
            target = "exclude"
            token = token[1:]
        else:
            target = "include"

        # modulus syntax: e.g. "e~3"
        if token.startswith("e~"):
            num_str = token[2:]
            try:
                n = int(num_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid modulus in '{raw}'")
            if n < 1:
                raise argparse.ArgumentTypeError(f"Modulus must be ≥ 1 in '{raw}'")

            steps = range(n, infer_steps + 1, n)

        # range syntax: e.g. "5-10"
        elif "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"Malformed range '{raw}'")
            start_str, end_str = parts
            try:
                start = int(start_str)
                end = int(end_str)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Non‑integer in range '{raw}'")
            if start < 1 or end < 1:
                raise argparse.ArgumentTypeError(f"Steps must be ≥ 1 in '{raw}'")
            if start > end:
                raise argparse.ArgumentTypeError(f"Start > end in '{raw}'")
            if end > infer_steps:
                raise argparse.ArgumentTypeError(f"End > infer_steps ({infer_steps}) in '{raw}'")

            steps = range(start, end + 1)

        # single‑step syntax: e.g. "7"
        else:
            try:
                step = int(token)
            except ValueError:
                raise argparse.ArgumentTypeError(f"Invalid token '{raw}'")
            if step < 1 or step > infer_steps:
                raise argparse.ArgumentTypeError(f"Step {step} out of range 1–{infer_steps} in '{raw}'")

            steps = [step]

        # apply include/exclude
        if target == "include":
            included.update(steps)
        else:
            excluded.update(steps)

    # final steps = included minus excluded, sorted
    return sorted(s for s in included if s not in excluded and s <= infer_steps)
