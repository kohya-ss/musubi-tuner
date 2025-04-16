#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Rate Interpolation using GIMM-VFI

Created on Mon Apr 14 12:23:15 2025
@author: blyss
"""

import argparse
import os
import random
import subprocess
import shutil  # for cross-platform directory removal

import torch
import yaml
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

# Importing necessary modules from our project
from blissful_tuner.gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from blissful_tuner.gimmvfi.generalizable_INR.gimmvfi_f import GIMMVFI_F
from blissful_tuner.gimmvfi.generalizable_INR.configs import GIMMVFIConfig
from blissful_tuner.gimmvfi.generalizable_INR.raft import RAFT
from blissful_tuner.gimmvfi.generalizable_INR.flowformer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from blissful_tuner.gimmvfi.generalizable_INR.flowformer.configs.submission import get_cfg
from blissful_tuner.gimmvfi.utils.flow_viz import flow_to_image
from blissful_tuner.gimmvfi.utils.utils import InputPadder, RaftArgs, easydict_to_dict
from blissful_tuner.utils import load_torch_file

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


def set_seed(seed=None):
    """
    Sets the random seed for reproducibility.
    If no seed is given, a random 32-bit integer seed is generated.
    Args:
        seed (int, optional): The seed value to set.
    Returns:
        int: The seed value used.
    """
    if seed is None:
        seed = random.getrandbits(32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def load_model(model_path, mode="gimmvfi_r", precision="fp32", torch_compile=False):
    """
    Loads the GIMM-VFI model along with its required flow estimator.

    Depending on the mode ("gimmvfi_r" or "gimmvfi_f") a different configuration,
    checkpoint, and flow estimation network are loaded.

    Args:
        model_path (str): Path to the directory containing model files.
        mode (str): The model type ("gimmvfi_r" or "gimmvfi_f").
        precision (str): Precision setting (not used in this snippet).
        torch_compile (bool): Whether to compile the model via torch.compile.

    Returns:
        torch.nn.Module: The fully loaded and prepared model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Select proper configuration, checkpoint, and flow model based on mode.
    if "gimmvfi_r" in mode:
        config_path = os.path.join(model_path, "gimmvfi_r_arb.yaml")
        flow_model_filename = "raft-things_fp32.safetensors"
        checkpoint = os.path.join(model_path, "gimmvfi_r_arb_lpips_fp32.safetensors")
    elif "gimmvfi_f" in mode:
        config_path = os.path.join(model_path, "gimmvfi_f_arb.yaml")
        checkpoint = os.path.join(model_path, "gimmvfi_f_arb_lpips_fp32.safetensors")
        flow_model_filename = "flowformer_sintel_fp32.safetensors"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    flow_model_path = os.path.join(model_path, flow_model_filename)

    # Load and merge YAML configuration
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = easydict_to_dict(config)
    config = OmegaConf.create(config)
    arch_defaults = GIMMVFIConfig.create(config.arch)
    config = OmegaConf.merge(arch_defaults, config.arch)

    # Initialize the model and its associated flow estimator
    if "gimmvfi_r" in mode:
        model = GIMMVFI_R(config)
        # Setup RAFT as flow estimator
        raft_args = RaftArgs(small=False, mixed_precision=False, alternate_corr=False)
        raft_model = RAFT(raft_args)
        raft_sd = load_torch_file(flow_model_path)
        raft_model.load_state_dict(raft_sd, strict=True)
        raft_model.to(device)
        flow_estimator = raft_model
    else:  # mode == "gimmvfi_f"
        model = GIMMVFI_F(config)
        cfg = get_cfg()
        flowformer = FlowFormer(cfg.latentcostformer)
        flowformer_sd = load_torch_file(flow_model_path)
        flowformer.load_state_dict(flowformer_sd, strict=True)
        flow_estimator = flowformer.to(device)

    # Load main model checkpoint
    sd = load_torch_file(checkpoint)
    model.load_state_dict(sd, strict=False)

    # Attach the flow estimator to the model, set evaluation mode, and move to device
    model.flow_estimator = flow_estimator
    model = model.eval().to(device)

    if torch_compile:
        model = torch.compile(model)
    return model


def load_video_frames(video_path):
    """
    Loads all frames from the provided video file and converts each frame into
    a PyTorch tensor of shape [1, channels, height, width].

    Args:
        video_path (str): Path to the input video file.

    Returns:
        tuple:
            - List[torch.Tensor]: A list of frame tensors.
            - float: The original video frame rate (FPS).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV reads in BGR format; convert to RGB.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL image then to a NumPy array to mimic original processing
        img = Image.fromarray(rgb_frame)
        raw_img = np.array(img.convert("RGB"))

        # Convert the NumPy array to a torch tensor with shape (C, H, W) normalized to [0,1]
        img_tensor = torch.from_numpy(raw_img.copy()).permute(2, 0, 1) / 255.0
        # Add a batch dimension: final shape is [1, C, H, W]
        frames.append(img_tensor.to(torch.float).unsqueeze(0))

    cap.release()
    return frames, fps


def images_to_video(imgs, output_video_path, fps):
    """
    Converts a list of image arrays to a video file using ffmpeg.

    Args:
        imgs (List[np.array]): List of image arrays (uint8).
        output_video_path (str): Path to save the output video.
        fps (float): Frame rate for the output video.
    """
    # Determine image dimensions from the first image
    height, width, _ = imgs[0].shape

    # Create a temporary directory to store frame images.
    frame_dir = os.path.join(os.path.dirname(output_video_path), "frames")
    os.makedirs(frame_dir, exist_ok=True)

    # Save each image as a PNG file in the temporary directory.
    for idx, img in enumerate(imgs):
        image_path = os.path.join(frame_dir, f"{idx:04d}.png")
        cv2.imwrite(image_path, img)

    # Run ffmpeg command to create video from the saved frames.
    subprocess.run(
        [
            "ffmpeg",
            "-framerate",
            f"{fps}",
            "-i",
            os.path.join(frame_dir, "%04d.png"),
            "-c:v",
            "prores_ks",
            "-profile:v", "3",
            "-pix_fmt",
            "yuv422p10le",
            "-colorspace", "1",
            "-color_primaries", "1",
            "-color_trc", "1",
            output_video_path,
        ],
        check=True
    )


    # Remove the temporary directory using shutil for cross-platform compatibility.
    shutil.rmtree(frame_dir)


def tensor_to_image(tensor):
    """
    Converts a single frame tensor to an 8-bit uint8 image array.

    Args:
        tensor (torch.Tensor): A tensor of shape [1, C, H, W].

    Returns:
        np.array: An image array in H x W x C format (BGR order).
    """
    # Remove the batch dimension, move to CPU, convert to HWC format, scale to 255, and convert color order.
    img = (tensor.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
    return img[:, :, ::-1].astype(np.uint8)  # Convert RGB to BGR for OpenCV


def interpolate(model, frames, ds_factor, N):
    """
    Interpolates frames using the provided model.

    Args:
        model (torch.nn.Module): The loaded interpolation model.
        frames (List[torch.Tensor]): List of input frame tensors.
        ds_factor (float): Downsampling factor used by the model.
        N (int): Number of interpolation steps between two frames.

    Returns:
        tuple:
            - List[np.array]: A list of interpolated image arrays (uint8).
            - List[np.array]: A list of flow visualization images (uint8) corresponding to the interpolation.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    interpolated_images = []  # Will contain the final interpolated images.
    flows = []  # Will contain flow visualization images.
    start = 0
    end = len(frames) - 1

    # Process each adjacent pair of frames.
    for j in tqdm(range(start, end), desc="Interpolating frames"):
        I0 = frames[j]
        I2 = frames[j + 1]

        # For the very first frame, add it directly.
        if j == start:
            interpolated_images.append(tensor_to_image(I0))

        # Pad both images so that their dimensions are divisible by 32.
        padder = InputPadder(I0.shape, 32)
        I0_padded, I2_padded = padder.pad(I0, I2)
        # Concatenate along a new dimension to create a tensor of shape [batch, 2, C, H, W]
        xs = torch.cat((I0_padded.unsqueeze(2), I2_padded.unsqueeze(2)), dim=2).to(device, non_blocking=True)

        # Ensure model is in eval mode (should be the case) and zero gradients (if any)
        model.eval()
        model.zero_grad()

        batch_size = xs.shape[0]
        s_shape = xs.shape[-2:]

        with torch.no_grad():
            # Prepare coordinate inputs and timesteps for interpolation.
            coord_inputs = [
                (
                    model.sample_coord_input(
                        batch_size,
                        s_shape,
                        [1 / N * i],
                        device=xs.device,
                        upsample_ratio=ds_factor,
                    ),
                    None,
                )
                for i in range(1, N)
            ]
            timesteps = [
                i / N * torch.ones(batch_size, device=xs.device, dtype=torch.float)
                for i in range(1, N)
            ]
            all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
            # Unpad the outputs to get back to original image size.
            out_frames = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
            out_flowts = [padder.unpad(f) for f in all_outputs["flowt"]]

        # Convert flow tensors to visual images.
        flowt_images = [
            flow_to_image(
                flowt.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
                convert_to_bgr=True,
            )
            for flowt in out_flowts
        ]

        # Convert each interpolated frame tensor to an image array.
        I1_pred_images = [
            tensor_to_image(I1_pred[0])
            for I1_pred in out_frames
        ]

        # Append the interpolated frames and corresponding flow images.
        for i in range(N - 1):
            interpolated_images.append(I1_pred_images[i])
            flows.append(flowt_images[i])

        # Append the next original frame.
        interpolated_images.append(tensor_to_image(I2))

    # Optionally, remove the last frame if not desired (here we return images[:-1])
    return interpolated_images[:-1], flows


def main():
    parser = argparse.ArgumentParser(description="Frame rate interpolation using GIMM-VFI")
    parser.add_argument("--model", required=True, help="Path to the checkpoint directory")
    parser.add_argument("--input", required=True, help="Input video file to interpolate")
    parser.add_argument("--output", type=str, default=None, help="Output video file path")
    parser.add_argument("--flow_output", type=str, default=None, help="Flow visualization video file path")
    parser.add_argument("--ds_factor", type=float, default=1.0, help="Downsampling factor")
    parser.add_argument("--mode", type=str, default="gimmvfi_f", help="Model mode: 'gimmvfi_r' or 'gimmvfi_f' for RAFT or FlowFormer version respectively")
    parser.add_argument("--N", type=int, default=8, help="Interpolation steps between frames")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    args = parser.parse_args()

    # Determine the output path based on the input file if not provided.
    if args.output is None:
        basename = os.path.basename(args.input)
        basename_no_ext, extension = os.path.splitext(basename)
        dir_name = os.path.dirname(args.input)
        args.output = os.path.join(dir_name, f"{basename_no_ext}_vfi.mkv")

    # Check for overwriting an existing file.
    if os.path.exists(args.output):
        confirm = input(f"{args.output} exists. Overwrite? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborting.")
            exit()

    print(f"Output video will be saved to: {args.output}")

    # Load the interpolation model and the video frames.
    model = load_model(args.model, args.mode)
    frames, fps = load_video_frames(args.input)
    new_fps = fps * args.N  # Adjust the frame rate according to the interpolation

    # Set seed for reproducibility.
    set_seed(args.seed)

    # Perform the frame interpolation.
    frames_interpolated, flows = interpolate(model, frames, args.ds_factor, args.N)

    # Save the interpolated video.
    images_to_video(frames_interpolated, args.output, fps=new_fps)
    if args.flow_output is not None:
        images_to_video(flows, args.flow_output, fps=new_fps)


if __name__ == "__main__":
    main()
