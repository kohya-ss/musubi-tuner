# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

# This is an adaptation of Magcache from https://github.com/Zehong-Ma/MagCache/
import numpy as np
import torch
from types import MethodType


def nearest_interp(src_array, target_length):
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


def set_magcache_params(dit, mag_ratios, num_steps, no_cfg):
    print("using Magcache")
    dit.forward = MethodType(magcache_forward, dit)
    dit.cnt = 0
    dit.num_steps = num_steps * 2
    dit.magcache_thresh = 0.12
    dit.K = 2
    dit.accumulated_err = [0.0, 0.0]
    dit.accumulated_steps = [0, 0]
    dit.accumulated_ratio = [1.0, 1.0]
    dit.retention_ratio = 0.2
    dit.residual_cache = [None, None]
    dit.mag_ratios = np.array([1.0] * 2 + mag_ratios)
    dit.no_cfg = no_cfg

    if len(dit.mag_ratios) != num_steps * 2:
        print(f"interpolate MAG RATIOS: curr len {len(dit.mag_ratios)}")
        mag_ratio_con = nearest_interp(dit.mag_ratios[0::2], num_steps)
        mag_ratio_ucon = nearest_interp(dit.mag_ratios[1::2], num_steps)
        interpolated_mag_ratios = np.concatenate([mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
        dit.mag_ratios = interpolated_mag_ratios


def magcache_forward(
    self,
    x,
    text_embed,
    pooled_text_embed,
    time,
    visual_rope_pos,
    text_rope_pos,
    scale_factor=(1.0, 1.0, 1.0),
    sparse_params=None,
    attention_mask=None,
):
    text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
        text_embed, time, pooled_text_embed, x, text_rope_pos
    )

    for text_transformer_block in self.text_transformer_blocks:
        text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)

    visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
        visual_embed, visual_rope_pos, scale_factor, sparse_params
    )

    skip_forward = False
    ori_visual_embed = visual_embed

    if self.cnt >= int(self.num_steps * self.retention_ratio):
        cur_mag_ratio = self.mag_ratios[self.cnt]
        self.accumulated_ratio[self.cnt % 2] = self.accumulated_ratio[self.cnt % 2] * cur_mag_ratio
        self.accumulated_steps[self.cnt % 2] += 1
        cur_skip_err = np.abs(1 - self.accumulated_ratio[self.cnt % 2])
        self.accumulated_err[self.cnt % 2] += cur_skip_err

        if self.accumulated_err[self.cnt % 2] < self.magcache_thresh and self.accumulated_steps[self.cnt % 2] <= self.K:
            skip_forward = True
            residual_visual_embed = self.residual_cache[self.cnt % 2]
        else:
            self.accumulated_err[self.cnt % 2] = 0
            self.accumulated_steps[self.cnt % 2] = 0
            self.accumulated_ratio[self.cnt % 2] = 1.0

    if skip_forward:
        visual_embed = visual_embed + residual_visual_embed
    else:
        for visual_transformer_block in self.visual_transformer_blocks:
            visual_embed = visual_transformer_block(
                visual_embed, text_embed, time_embed, visual_rope, sparse_params, attention_mask
            )
        residual_visual_embed = visual_embed - ori_visual_embed

    self.residual_cache[self.cnt % 2] = residual_visual_embed

    x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)

    if self.no_cfg:
        self.cnt += 2
    else:
        self.cnt += 1

    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_ratio = [1.0, 1.0]
        self.accumulated_err = [0.0, 0.0]
        self.accumulated_steps = [0, 0]
    return x
