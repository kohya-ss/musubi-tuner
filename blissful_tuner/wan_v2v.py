#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:01:35 2025

@author: blyss
"""
import argparse
import random
from typing import Tuple
import torch
import numpy as np
import torchvision.transforms.functional as TF
from easydict import EasyDict
from wan.modules.vae import WanVAE
from blissful_tuner.video_processing_common import BlissfulVideoProcessor
from blissful_tuner.utils import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")


def prepare_v2v_noise(
    args: argparse.Namespace,
    config: EasyDict,
    timesteps: torch.Tensor,
    device: torch.device,
    vae: WanVAE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare (noise, timesteps) for Wan Video-to-Video.

    Returns:
      - noise:      [C, F, H_lat, W_lat] latent-space tensor
      - timesteps:  possibly shortened timesteps tensor
    """
    # 1) Possibly shorten the schedule
    steps = args.infer_steps
    if args.v2v_denoise <= 1.0:
        steps = int(steps * args.v2v_denoise)
        timesteps = timesteps[-(steps):]
    elif args.v2v_denoise > 1.0:
        raise ValueError("--v2v_noise cannot be greater than 1.0!")
    args.infer_steps = steps
    if args.cfg_schedule is not None:
        invalid_steps = [step for step in args.cfg_schedule.keys() if int(step) > steps]
        for step in invalid_steps:
            args.cfg_schedule.pop(step, None)


    # 2) Compute target latent-grid dimensions
    height_px, width_px = args.video_size
    total_frames = args.video_length
    max_area = width_px * height_px
    aspect = height_px / width_px

    lat_h = int(
        round(
            np.sqrt(max_area * aspect)
            // config.vae_stride[1]
            // config.patch_size[1]
            * config.patch_size[1]
        )
    )
    lat_w = int(
        round(
            np.sqrt(max_area / aspect)
            // config.vae_stride[2]
            // config.patch_size[2]
            * config.patch_size[2]
        )
    )
    height_px = lat_h * config.vae_stride[1]
    width_px  = lat_w * config.vae_stride[2]
    lat_f = (total_frames - 1) // config.vae_stride[0] + 1

    # 3) Load raw frames
    vp = BlissfulVideoProcessor(device, vae.dtype)
    vp.prepare_files_and_path(args.video_path, None)
    raw_frames, _, _, _ = vp.load_frames(make_rgb=True)  # list of np.ndarray or PIL.Image

    # 4) Build a [1, C, T, H_px, W_px] video tensor
    frame_tensors = []
    for arr in raw_frames:
        t = TF.to_tensor(arr)                                   # [C, H_px, W_px], 0–1
        t = TF.resize(t, [height_px, width_px], interpolation=TF.InterpolationMode.BICUBIC)
        t = t.sub_(0.5).div_(0.5).to(device)                     # normalize to [-1,1]
        frame_tensors.append(t)
    video = torch.stack(frame_tensors, dim=0)                   # [T, C, H_px, W_px]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)              # [1, C, T, H_px, W_px]

    # 5) Encode the entire video in one go to latent space
    vae.to_device(device)
    with torch.autocast(device_type=str(device), dtype=vae.dtype), torch.no_grad():
        latent_list = vae.encode(video)           # returns list of length B=1
    input_samples = latent_list[0]                              # [C, T, H_lat, W_lat]
    vae.to_device("cpu")
    # 6) Prepare RNG & noise tensor
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device).manual_seed(seed)
    else:
        seed_g = torch.manual_seed(seed)

    noise = torch.randn(
        input_samples.shape[0],    # C
        lat_f,                     # T
        lat_h,                     # H_lat
        lat_w,                     # W_lat
        dtype=torch.float32,
        generator=seed_g,
        device=device if not args.cpu_noise else "cpu",
    ).to(device)

    # 7) Ensure input_samples has exactly T = noise.shape[1] frames
    in_f = input_samples.shape[1]
    tgt_f = noise.shape[1]

    if in_f < tgt_f:
        pad_count = tgt_f - in_f
        logger.info(f"Padding input at the {args.v2v_pad_mode}")
        if args.v2v_pad_mode == "front":
            # repeat the first frame pad_count times at the front
            pad = input_samples[:, :1].repeat(1, pad_count, 1, 1)
            input_samples = torch.cat([pad, input_samples], dim=1)
        elif args.v2v_pad_mode == "end":
            # repeat the last frame pad_count times at the end
            pad = input_samples[:, -1:].repeat(1, pad_count, 1, 1)
            input_samples = torch.cat([input_samples, pad], dim=1)
        else:
            raise ValueError(f"Unknown v2v_pad_mode: {args.v2v_pad_mode!r}")

    elif in_f > tgt_f:
        # too many frames: truncate the tail
        input_samples = input_samples[:, :tgt_f]

    # 8) Blend noise & input according to updated timestep schedule
    latent_timestep = timesteps[:1].to(noise) / 1000
    noise = noise * latent_timestep + (1 - latent_timestep) * input_samples
    return noise, timesteps
