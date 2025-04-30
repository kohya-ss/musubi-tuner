#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:23:28 2025
CFGZero* implementation for Blissful Tuner extension based on https://github.com/WeichenFan/CFG-Zero-star/blob/main/models/wan/wan_pipeline.py
License: Apache 2.0
@author: blyss
"""
import torch
from typing import Optional


def apply_zerostar(cond: torch.Tensor, uncond: torch.Tensor, current_step: int, guidance_scale: float, use_scaling: Optional[bool] = True, zero_init_steps: Optional[int] = -1) -> torch.Tensor:
    """Function to apply CFGZero* scaling and zero init"""
    if (current_step <= zero_init_steps):
        return cond * 0
    if not use_scaling:
        # CFG formula
        noise_pred = uncond + guidance_scale * (cond - uncond)
    else:
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
