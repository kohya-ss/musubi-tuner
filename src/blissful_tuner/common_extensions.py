#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:01:35 2025

@author: blyss
"""

import os
import argparse
from datetime import datetime
from typing import Tuple, Optional, Any
import torch
import numpy as np
from einops import rearrange
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from easydict import EasyDict
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.utils.device_utils import clean_memory_on_device
from blissful_tuner.blissful_core import get_current_model_type, get_current_version
from blissful_tuner.video_processing_common import BlissfulVideoProcessor
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")


def prepare_metadata(args: argparse.Namespace, seed_override: Optional[Any] = None) -> dict:
    seed = args.seed if seed_override is None else seed_override
    attr_list = [
        "prompt",
        "infer_steps",
        "guidance_scale",
        "flow_shift",
        "hidden_state_skip_layer",
        "apply_final_norm",
        "sample_solver",
        "scheduler",
        "fps",
        "task",
        "embedded_cfg_scale",
        "negative_prompt",
        "prompt_2",
        "cfg_schedule",
        "nag_scale",
        "nag_tau",
        "nag_alpha",
        "nag_prompt",
        "perp_neg",
        "cfgzerostar_scaling",
        "cfgzerostar_init_steps",
        "te_multiplier",
        "riflex_index",
    ]
    metadata = {
        "bt_model_type": f"{get_current_model_type()}",
        "bt_seeds": f"{seed}",
        "bt_creation_timestamp": f"{datetime.now()}",
        "bt_tunerver": f"{get_current_version()}",
    }

    if hasattr(args, "task"):  # Use task arg present for Wan to suss out model version
        if args.task.lower() in ["t2v-a14b", "i2v-a14b"]:
            metadata["bt_model_type"] = "Wan 2.2"
        else:
            metadata["bt_model_type"] = "Wan 2.1"

    for attr in attr_list:
        if hasattr(args, attr):  # Many are model specific
            value = getattr(args, attr)
            attr = "scheduler" if attr == "sample_solver" else attr  # Normalize naming convention for metadata
            if (
                (attr == "riflex_index" and value == 0)
                or (attr == "cfgzerostar_init_steps" and value == -1)
                or (attr == "cfgzerostar_scaling" and value is False)
            ):
                continue  # Don't put them in at their "off" values
            if (
                attr == "nag_tau" or attr == "nag_alpha"
            ) and args.nag_scale is None:  # We can assume nag_scale exists if either of these two pass the previous hasattr
                continue  # These default to nonzero floats but are only applicable if nag_scale is not None
            value = str(value) if value is not None else None  # Might be dict, list, etc so make sure to string it
            if value is not None:  # No point in passing through Nonetypes
                metadata[f"bt_{attr}"] = value

    if args.lora_weight:
        for i, lora_weight in enumerate(args.lora_weight):  # Type list even if only 1
            lora_weight = os.path.basename(lora_weight)  # Weed out the path for privacy etc and just use the LoRA's name
            metadata[f"bt_lora_{i}"] = f"{lora_weight}: {args.lora_multiplier[i]}"

    if hasattr(args, "lora_weight_high_noise") and args.lora_weight_high_noise:  # Same same... but different.
        for i, lora_weight_high_noise in enumerate(args.lora_weight_high_noise):
            lora_weight_high_noise = os.path.basename(lora_weight_high_noise)
            metadata[f"bt_lora_high_{i}"] = f"{lora_weight_high_noise}: {args.lora_multiplier_high_noise[i]}"
    return metadata


def save_media_advanced(
    medias: torch.Tensor,
    output_video: str,
    args: argparse.Namespace,
    rescale: bool = False,
    n_rows: int = 1,
    metadata: Optional[dict] = None,
):
    "Function for saving Musubi Tuner outputs with more codec and container types"

    # 1) rearrange so we iterate over time, everything that reaches this point should have shape B C T H W
    medias = rearrange(medias, "b c t h w -> t b c h w")

    VideoProcessor = BlissfulVideoProcessor()
    VideoProcessor.prepare_files_and_path(
        input_file_path=None,
        output_file_path=output_video,
        codec=args.codec if hasattr(args, "codec") else None,
        container=args.container if hasattr(args, "container") else None,
    )

    outputs = []
    for media in medias:
        # 2) tile frames into one grid [C, H, W]
        grid = torchvision.utils.make_grid(media, nrow=n_rows)
        # 3) convert to an OpenCV-ready numpy array
        np_img = VideoProcessor.tensor_to_np_image(grid, rescale=rescale)
        outputs.append(np_img)

    VideoProcessor.write_np_images_to_output(
        outputs,
        args.fps if hasattr(args, "fps") else 1.0,
        args.keep_pngs if hasattr(args, "keep_pngs") else False,
        metadata=metadata,
    )


def prepare_i2i_noise(
    base_noise: torch.Tensor, args: argparse.Namespace, config: EasyDict, timesteps: torch.Tensor, device: torch.device, vae: WanVAE
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare (noise, timesteps) for Wan Video-to-Video.

    Returns:
      - noise:      [C, F, H_lat, W_lat] latent-space tensor
      - timesteps:  possibly shortened timesteps tensor
    """
    # 1) Possibly shorten the schedule
    if args.denoise_strength > 1.0:
        raise ValueError("--denoise must be < 1.0!")
    if args.noise_mode.lower() == "traditional":
        steps = int(args.infer_steps * args.denoise_strength)
        timesteps = timesteps[-(steps):]
        logger.info(
            f"Modifying timestep schedule to run {args.denoise_strength * 100}% of the process. Noise added: {timesteps.float()[0] / 10:.2f}%"
        )
    elif args.noise_mode.lower() == "direct":
        normalized_ts = timesteps.float() / 1000
        start_idx = torch.argmin(torch.abs(normalized_ts - args.denoise_strength))
        timesteps = timesteps[start_idx:]
        logger.info(
            f"Modifying timestep schedule to add as close to {args.denoise_strength * 100}% noise as possible. Noise added: {timesteps.float()[0] / 10:.2f}%"
        )
    args.infer_steps = len(timesteps)

    output_height, output_width = args.video_size
    output_video_area = output_height * output_width
    aspect_ratio = output_height / output_width

    latent_height = int(
        round(np.sqrt(output_video_area * aspect_ratio) / config.vae_stride[1] / config.patch_size[1] * config.patch_size[1])
    )
    latent_width = int(
        round(np.sqrt(output_video_area / aspect_ratio) / config.vae_stride[2] / config.patch_size[2] * config.patch_size[2])
    )
    computed_height = latent_height * config.vae_stride[1]
    computed_width = latent_width * config.vae_stride[2]

    # 3) Load raw frames
    img = Image.open(args.i2i_path).convert("RGB")

    # 4) Build a [1, C, T, H_px, W_px] video tensor
    t = TF.to_tensor(img)  # [C, H, W], 0–1
    t = TF.resize(t, [computed_height, computed_width], interpolation=TF.InterpolationMode.BICUBIC)
    t = t.sub_(0.5).div_(0.5).to(device)  # normalize to [-1,1]
    t = t.unsqueeze(1)

    # 5) Encode the entire video in one go to latent space
    vae.to_device(device)
    logger.info("Encoding input image to latent space for i2i")
    with torch.autocast(device_type=str(device), dtype=vae.dtype), torch.no_grad():
        latent_list = vae.encode([t])  # returns list of length B=1
    input_samples = latent_list[0]  # [C, T, H_lat, W_lat]
    vae.to_device("cpu")

    # 8) Blend noise & input according to updated timestep schedule
    timestep_noise_percent = timesteps[:1].to(base_noise) / 1000
    noise = base_noise * timestep_noise_percent + (1 - timestep_noise_percent) * input_samples
    if args.i2_extra_noise is not None:
        logger.info((f"Adding {100 * args.i2_extra_noise}% extra noise to i2i image"))
        noise += args.i2_extra_noise * base_noise

    clean_memory_on_device(device)
    return noise, timesteps


def prepare_v2v_noise(
    base_noise: torch.Tensor,
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
    if args.denoise_strength > 1.0:
        raise ValueError("--denoise must be < 1.0!")
    if args.noise_mode.lower() == "traditional":
        steps = int(args.infer_steps * args.denoise_strength)
        timesteps = timesteps[-(steps):]
        logger.info(
            f"Modifying timestep schedule to run {args.denoise_strength * 100}% of the process. Noise added: {timesteps.float()[0] / 10:.2f}%"
        )
    elif args.noise_mode.lower() == "direct":
        normalized_ts = timesteps.float() / 1000
        start_idx = torch.argmin(torch.abs(normalized_ts - args.denoise_strength))
        timesteps = timesteps[start_idx:]
        logger.info(
            f"Modifying timestep schedule to add as close to {args.denoise_strength * 100}% noise as possible. Noise added: {timesteps.float()[0] / 10:.2f}%"
        )
    args.infer_steps = len(timesteps)

    output_height, output_width = args.video_size
    output_video_area = output_height * output_width
    aspect_ratio = output_height / output_width

    latent_height = int(
        round(np.sqrt(output_video_area * aspect_ratio) / config.vae_stride[1] / config.patch_size[1] * config.patch_size[1])
    )
    latent_width = int(
        round(np.sqrt(output_video_area / aspect_ratio) / config.vae_stride[2] / config.patch_size[2] * config.patch_size[2])
    )
    computed_height = latent_height * config.vae_stride[1]
    computed_width = latent_width * config.vae_stride[2]

    # 3) Load raw frames
    vp = BlissfulVideoProcessor(device, vae.dtype, will_write_video=False)
    vp.prepare_files_and_path(args.video_path, None)
    raw_frames, _, _, _ = vp.load_frames(make_rgb=True)  # list of np.ndarray or PIL.Image

    # 4) Build a [1, C, T, H_px, W_px] video tensor
    frame_tensors = []
    for arr in raw_frames:
        t = TF.to_tensor(arr)  # [C, H, W], 0–1
        t = TF.resize(t, [computed_height, computed_width], interpolation=TF.InterpolationMode.BICUBIC)
        t = t.sub_(0.5).div_(0.5).to(device)  # normalize to [-1,1]
        frame_tensors.append(t)
    video = torch.stack(frame_tensors, dim=0)  # [F, C, H, W]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, F, H, W]

    # 5) Encode the entire video in one go to latent space
    vae.to_device(device)
    logger.info("Encoding input video to latent space")
    with torch.autocast(device_type=str(device), dtype=vae.dtype), torch.no_grad():
        latent_list = vae.encode(video)  # returns list of length B=1
    input_samples = latent_list[0]  # [C, T, H_lat, W_lat]
    vae.to_device("cpu")

    # 7) Ensure input_samples has exactly T = noise.shape[1] frames
    in_f = input_samples.shape[1]
    tgt_f = base_noise.shape[1]

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
    timestep_noise_percent = timesteps[:1].to(base_noise) / 1000
    noise = base_noise * timestep_noise_percent + (1 - timestep_noise_percent) * input_samples
    if args.v2_extra_noise is not None:
        logger.info((f"Adding {100 * args.v2_extra_noise}% extra noise to V2V frames"))
        noise += args.v2_extra_noise * base_noise

    clean_memory_on_device(device)
    return noise, timesteps
