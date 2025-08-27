#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:23:28 2025
Various advanced guidance methods for Blissful Tuner
License: Apache 2.0
@author: blyss
"""
import argparse
from typing import List
import torch


def nag(z_positive, z_negative, nag_scale, nag_tau, nag_alpha):
    z_tilde = z_positive * nag_scale - z_negative * (nag_scale - 1)  # Revised implementation
    # z_tilde = z_positive + self.nag_scale * (z_positive - z_negative)  # Paper implementation
    norm_positive = torch.norm(z_positive, p=1, dim=-1, keepdim=True).expand(*z_positive.shape)
    norm_tilde = torch.norm(z_tilde, p=1, dim=-1, keepdim=True).expand(*z_tilde.shape)

    ratio = norm_tilde / norm_positive
    z_hat = z_tilde * torch.minimum(ratio, ratio.new_ones(1) * nag_tau) / ratio
    z_guidance = z_hat * nag_alpha + z_positive * (1 - nag_alpha)
    return z_guidance


def perpendicular_negative_cfg(cond: torch.Tensor, uncond: torch.Tensor, nocond: torch.Tensor, negative_scale: float, guidance_scale: float) -> torch.Tensor:
    """Perpendicular negative CFG"""
    pos_cond = cond - nocond
    neg_cond = uncond - nocond
    perp_neg_cond = neg_cond - ((torch.mul(neg_cond, pos_cond).sum()) / (torch.norm(pos_cond)**2)) * pos_cond
    perp_neg_cond *= negative_scale
    noise_pred = nocond + guidance_scale * (pos_cond - perp_neg_cond)
    return noise_pred


def apply_zerostar_scaling(cond: torch.Tensor, uncond: torch.Tensor, guidance_scale: float) -> torch.Tensor:
    """Function to apply CFGZero* scaling"""
    batch_size = cond.shape[0]
    positive_flat = cond.view(batch_size, -1)
    negative_flat = uncond.view(batch_size, -1)
    alpha = optimized_scale(positive_flat, negative_flat)
    alpha = alpha.view(batch_size, *([1] * (len(cond.shape) - 1)))
    alpha = alpha.to(cond.dtype)
    # CFG formula modified with alpha
    noise_pred = uncond * alpha + guidance_scale * (cond - uncond * alpha)
    return noise_pred


def optimized_scale(positive_flat: torch.Tensor, negative_flat: torch.Tensor) -> torch.Tensor:
    """Computes optimized scaling factors for CFG"""
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8

    # st_star = v_cond^T * v_uncond / ||v_uncond||^2
    st_star = dot_product / squared_norm

    return st_star


def parse_scheduled_cfg(schedule: str, infer_steps: int, guidance_scale: int) -> List[int]:
    """
    Parse a schedule string like "1-10,20,!5,e~3" into a sorted list of steps.

    - "start-end" includes all steps in [start, end]
    - "e~n"    includes every nth step (n, 2n, ...) up to infer_steps
    - "x"      includes the single step x
    - Prefix "!" on any token to exclude those steps instead of including them.
    - Postfix ":float" e.g. ":6.0" to any step or range to specify a guidance_scale override for that step

    Raises argparse.ArgumentTypeError on malformed tokens or out-of-range steps.
    """
    excluded = set()
    guidance_scale_dict = {}

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

        weight = guidance_scale
        if ":" in token:
            token, float_part = token.rsplit(":", 1)
            weight = float(float_part)

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
            for step in steps:
                guidance_scale_dict[step] = weight
        else:
            excluded.update(steps)

    for step in excluded:
        guidance_scale_dict.pop(step, None)
    return guidance_scale_dict
