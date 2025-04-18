#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 17:23:28 2025

@author: blyss
"""
import torch


def apply_zerostar(cond, uncond, current_step, guidance_scale, use_zero_init=True, zero_init_steps=0):
    # https://github.com/WeichenFan/CFG-Zero-star as ported from KjNodes
    if (current_step <= zero_init_steps) and use_zero_init:
        return cond * 0

    batch_size = cond.shape[0]

    positive_flat = cond.view(batch_size, -1)
    negative_flat = uncond.view(batch_size, -1)

    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
    alpha = dot_product / squared_norm
    alpha = alpha.view(batch_size, *([1] * (len(cond.shape) - 1)))
    print(f"ZeroStar: Alpha is {torch.mean(alpha)}")
    noise_pred = uncond * alpha + guidance_scale * (cond - uncond * alpha)
    return noise_pred
