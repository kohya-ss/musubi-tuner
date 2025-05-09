#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:23:28 2025
CFGZero* implementation for Blissful Tuner extension based on https://github.com/WeichenFan/CFG-Zero-star/blob/main/models/wan/wan_pipeline.py
License: Apache 2.0
@author: blyss
"""
import torch


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
