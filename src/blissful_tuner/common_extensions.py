#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  1 13:01:35 2025

@author: blyss
"""
import os
import argparse
import threading
from datetime import datetime
from typing import Tuple, Optional, Any
from pynput import keyboard
import torch
import numpy as np
from einops import rearrange
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
from easydict import EasyDict
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.utils.device_utils import clean_memory_on_device
from blissful_tuner.blissful_args import get_current_model_type, get_current_version
from blissful_tuner.video_processing_common import BlissfulVideoProcessor
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")


class BlissfulKeyboardManager:
    def __init__(self):
        self.early_exit_requested = False
        self.hotkey = keyboard.HotKey(
            keyboard.HotKey.parse("<ctrl>+q"),
            self.request_exit
        )

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
        logger.info("Keyboard manager initialized! Press CTRL+Q for early exit!")

    @property
    def exit_requested(self):
        return self.early_exit_requested

    def _on_press(self, key):
        # Feed every key press into the HotKey
        self.hotkey.press(key)

    def _on_release(self, key):
        # And feed every release too (HotKey needs both)
        self.hotkey.release(key)

    def terminate(self):
        if self.listener is not None:
            self.listener.stop()

    def request_exit(self):
        # This method will be called when <ctrl>+q is hit
        self.early_exit_requested = True
        logger.info("Early exit requested! Will exit after this step completes!")
        self.listener.stop()


class BlissfulThreadManager():
    def __init__(self, max_live_threads):
        self.managed_threads = []
        self.thread_count = 0
        self.max_threads = max_live_threads

    def spawn_thread(self, thread_target, thread_args):
        if self.thread_count >= self.max_threads:
            i = 0
            to_del = []
            while self.thread_count >= self.max_threads:
                thread_dict = self.managed_threads[i]
                logger.info(f"Joining thread #{thread_dict['id']} with target '{thread_dict['target']}'...")
                thread_dict["handle"].join()
                to_del.append(i)  # Don't delete while iterating against
                self.thread_count -= 1
                i += 1
            for delable in to_del:
                del self.managed_threads[delable]
        logger.info(f"Spawn thread #{self.thread_count} with target'{thread_target}'")
        thread_handle = threading.Thread(target=thread_target, args=thread_args)
        thread_dict = {
            "handle": thread_handle,
            "id": self.thread_count,
            "target": thread_target,
            "args": thread_args
        }
        self.managed_threads.append(thread_dict)
        thread_handle.start()
        self.thread_count += 1
        return thread_handle

    def cleanup_threads(self):
        for thread_dict in self.managed_threads:
            logger.info(f"Joining thread #{thread_dict['id']} with target '{thread_dict['target']}'...")
            thread_dict["handle"].join(timeout=3)


def prepare_metadata(args: argparse.Namespace, seed_override: Optional[Any] = None) -> dict:
    seed = args.seed if seed_override is None else seed_override
    attr_list = ["prompt", "infer_steps", "guidance_scale", "flow_shift",
                 "hidden_state_skip_layer", "apply_final_norm",
                 "fps", "task", "embedded_cfg_scale", "negative_prompt", "cfg_schedule"]
    metadata = {
        "bt_model_type": f"{get_current_model_type()}",
        "bt_seeds": f"{seed}",
        "bt_creation_timestamp": f"{datetime.now()}",
        "bt_tunerver": f"{get_current_version()}"
    }

    for attr in attr_list:
        if hasattr(args, attr):
            value = getattr(args, attr)
            value = str(value) if value is not None else "N/A"
            metadata[f"bt_{attr}"] = value

    if args.lora_weight:
        for i, lora_weight in enumerate(args.lora_weight):
            lora_weight = os.path.basename(lora_weight)
            metadata[f"bt_lora_{i}"] = f"{lora_weight}: {args.lora_multiplier[i]}"
    return metadata


def save_videos_grid_advanced(
    videos: torch.Tensor,
    output_video: str,
    args: argparse.Namespace,
    rescale: bool = False,
    n_rows: int = 1,
    metadata: Optional[dict] = None
):
    "Function for saving Musubi Tuner outputs with more codec and container types"

    # 1) rearrange so we iterate over time
    videos = rearrange(videos, "b c t h w -> t b c h w")

    VideoProcessor = BlissfulVideoProcessor()
    VideoProcessor.prepare_files_and_path(
        input_file_path=None,
        output_file_path=output_video,
        codec=args.codec,
        container=args.container
    )

    outputs = []
    for video in videos:
        # 2) tile frames into one grid [C, H, W]
        grid = torchvision.utils.make_grid(video, nrow=n_rows)
        # 3) convert to an OpenCV-ready numpy array
        np_img = VideoProcessor.tensor_to_np_image(grid, rescale=rescale)
        outputs.append(np_img)

    VideoProcessor.write_np_images_to_output(outputs, args.fps, args.keep_pngs, metadata=metadata)


def prepare_i2i_noise(
    base_noise: torch.Tensor,
    args: argparse.Namespace,
    config: EasyDict,
    timesteps: torch.Tensor,
    device: torch.device,
    vae: WanVAE
) -> torch.Tensor:
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
        logger.info(f"Modifying timestep schedule to run {args.denoise_strength * 100}% of the process. Noise added: {timesteps.float()[0] / 10:.2f}%")
    elif args.noise_mode.lower() == "direct":
        normalized_ts = timesteps.float() / 1000
        start_idx = torch.argmin(torch.abs(normalized_ts - args.denoise_strength))
        timesteps = timesteps[start_idx:]
        logger.info(f"Modifying timestep schedule to add as close to {args.denoise_strength * 100}% noise as possible. Noise added: {timesteps.float()[0] / 10:.2f}%")
    args.infer_steps = len(timesteps)

    output_height, output_width = args.video_size
    output_video_area = output_height * output_width
    aspect_ratio = output_height / output_width

    latent_height = int(
        round(
            np.sqrt(output_video_area * aspect_ratio)
            / config.vae_stride[1]
            / config.patch_size[1]
            * config.patch_size[1]
        )
    )
    latent_width = int(
        round(
            np.sqrt(output_video_area / aspect_ratio)
            / config.vae_stride[2]
            / config.patch_size[2]
            * config.patch_size[2]
        )
    )
    computed_height = latent_height * config.vae_stride[1]
    computed_width = latent_width * config.vae_stride[2]

    # 3) Load raw frames
    img = Image.open(args.i2i_path).convert("RGB")

    # 4) Build a [1, C, T, H_px, W_px] video tensor
    t = TF.to_tensor(img)                                   # [C, H, W], 0–1
    t = TF.resize(t, [computed_height, computed_width], interpolation=TF.InterpolationMode.BICUBIC)
    t = t.sub_(0.5).div_(0.5).to(device)                     # normalize to [-1,1]
    t = t.unsqueeze(1)

    # 5) Encode the entire video in one go to latent space
    vae.to_device(device)
    logger.info("Encoding input image to latent space for i2i")
    with torch.autocast(device_type=str(device), dtype=vae.dtype), torch.no_grad():
        latent_list = vae.encode([t])           # returns list of length B=1
    input_samples = latent_list[0]                # [C, T, H_lat, W_lat]
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
        logger.info(f"Modifying timestep schedule to run {args.denoise_strength * 100}% of the process. Noise added: {timesteps.float()[0] / 10:.2f}%")
    elif args.noise_mode.lower() == "direct":
        normalized_ts = timesteps.float() / 1000
        start_idx = torch.argmin(torch.abs(normalized_ts - args.denoise_strength))
        timesteps = timesteps[start_idx:]
        logger.info(f"Modifying timestep schedule to add as close to {args.denoise_strength * 100}% noise as possible. Noise added: {timesteps.float()[0] / 10:.2f}%")
    args.infer_steps = len(timesteps)

    output_height, output_width = args.video_size
    output_video_area = output_height * output_width
    aspect_ratio = output_height / output_width

    latent_height = int(
        round(
            np.sqrt(output_video_area * aspect_ratio)
            / config.vae_stride[1]
            / config.patch_size[1]
            * config.patch_size[1]
        )
    )
    latent_width = int(
        round(
            np.sqrt(output_video_area / aspect_ratio)
            / config.vae_stride[2]
            / config.patch_size[2]
            * config.patch_size[2]
        )
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
        t = TF.to_tensor(arr)                                   # [C, H, W], 0–1
        t = TF.resize(t, [computed_height, computed_width], interpolation=TF.InterpolationMode.BICUBIC)
        t = t.sub_(0.5).div_(0.5).to(device)                     # normalize to [-1,1]
        frame_tensors.append(t)
    video = torch.stack(frame_tensors, dim=0)                   # [F, C, H, W]
    video = video.permute(1, 0, 2, 3).unsqueeze(0)              # [1, C, F, H, W]

    # 5) Encode the entire video in one go to latent space
    vae.to_device(device)
    logger.info("Encoding input video to latent space")
    with torch.autocast(device_type=str(device), dtype=vae.dtype), torch.no_grad():
        latent_list = vae.encode(video)           # returns list of length B=1
    input_samples = latent_list[0]                # [C, T, H_lat, W_lat]
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
