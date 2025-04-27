#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Frame Rate Interpolation using GIMM-VFI
-----------------------------------
This specific code file as well as all files in ./blissful_tuner/gimmvfi and subfolders (all GIMM-VFI related code) licensed:

S-Lab License 1.0
Copyright 2024 S-Lab

Redistribution and use for non-commercial purpose in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

In the event that redistribution and/or use for commercial purpose in source or binary forms, with or without modification is required, please contact the contributor(s) of the work.
---------------------------------------
Created on Mon Apr 14 12:23:15 2025
@author: blyss
"""

import os
import warnings
from typing import List
import torch
import yaml
from tqdm import tqdm
from omegaconf import OmegaConf

# Importing necessary modules from our project
from gimmvfi.generalizable_INR.gimmvfi_r import GIMMVFI_R
from gimmvfi.generalizable_INR.gimmvfi_f import GIMMVFI_F
from gimmvfi.generalizable_INR.configs import GIMMVFIConfig
from gimmvfi.generalizable_INR.raft import RAFT
from gimmvfi.generalizable_INR.flowformer.core.FlowFormer.LatentCostFormer.transformer import FlowFormer
from gimmvfi.generalizable_INR.flowformer.configs.submission import get_cfg
from gimmvfi.utils.utils import InputPadder, RaftArgs, easydict_to_dict
from utils import load_torch_file, setup_compute_context
from video_processing_common import BlissfulVideoProcessor, setup_parser_video_common, set_seed
warnings.filterwarnings("ignore")


def load_model(model_path: str, device: torch.device, dtype: torch.dtype, mode: str = "gimmvfi_r") -> torch.nn.Module:
    """
    Loads the GIMM-VFI model along with its required flow estimator.

    Depending on the mode ("gimmvfi_r" or "gimmvfi_f") a different configuration,
    checkpoint, and flow estimation network are loaded.
    """

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
        flow_estimator = raft_model.to(device, dtype)
    else:  # mode == "gimmvfi_f"
        model = GIMMVFI_F(config)
        cfg = get_cfg()
        flowformer = FlowFormer(cfg.latentcostformer)
        flowformer_sd = load_torch_file(flow_model_path)
        flowformer.load_state_dict(flowformer_sd, strict=True)
        flow_estimator = flowformer.to(device, dtype)

    # Load main model checkpoint
    sd = load_torch_file(checkpoint)
    model.load_state_dict(sd, strict=False)

    # Attach the flow estimator to the model, set evaluation mode, and move to device
    model.flow_estimator = flow_estimator
    model = model.eval().to(device, dtype)

    return model


def interpolate(model: torch.nn.Module, frames: List[torch.Tensor], ds_factor: float, N: int, VideoProcessor: BlissfulVideoProcessor):
    """
    Interpolates frames using the provided model.

    Args:
        model: The loaded interpolation model.
        frames: List of input frame tensors.
        ds_factor: Downsampling factor used by the model.
        N: Number of interpolation steps between two frames.
    """
    device = VideoProcessor.device
    dtype = VideoProcessor.dtype
    start = 0
    end = len(frames) - 1

    # Process each adjacent pair of frames.
    for j in tqdm(range(start, end), desc="Interpolating frames"):
        I0 = frames[j]
        I2 = frames[j + 1]

        # For the very first frame, add it directly.
        if j == start:
            VideoProcessor.write_np_or_tensor_to_png(I0)

        # Pad both images so that their dimensions are divisible by 32.
        padder = InputPadder(I0.shape, 32)
        I0_padded, I2_padded = padder.pad(I0, I2)
        # Concatenate along a new dimension to create a tensor of shape [batch, 2, C, H, W]
        xs = torch.cat((I0_padded.unsqueeze(2), I2_padded.unsqueeze(2)), dim=2).to(device, dtype, non_blocking=True)

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
                i / N * torch.ones(batch_size, device=xs.device, dtype=dtype)
                for i in range(1, N)
            ]
            if dtype != torch.float32:
                with torch.autocast(device_type=str(device), dtype=dtype):
                    all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
            else:
                all_outputs = model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
            # Unpad the outputs to get back to original image size.
            out_frames = [padder.unpad(im) for im in all_outputs["imgt_pred"]]

        # Convert each interpolated frame tensor to an image array.
        I1_pred_images = [I1_pred[0] for I1_pred in out_frames]

        # Append the interpolated frames and corresponding flow images.
        for i in range(N - 1):
            VideoProcessor.write_np_or_tensor_to_png(I1_pred_images[i])

        # Append the next original frame.
        VideoProcessor.write_np_or_tensor_to_png(I2)


def main():
    parser = setup_parser_video_common(description="Frame rate interpolation using GIMM-VFI")
    parser.add_argument("--ds_factor", type=float, default=1.0, help="Downsampling factor")
    parser.add_argument("--mode", type=str, default="gimmvfi_f", help="Model mode: 'gimmvfi_r' or 'gimmvfi_f' for RAFT or FlowFormer version respectively")
    parser.add_argument("--factor", type=int, default=2, help="Interpolation steps between frames")
    args = parser.parse_args()
    device, dtype = setup_compute_context(None, args.dtype)
    VideoProcessor = BlissfulVideoProcessor(device, dtype)
    VideoProcessor.prepare_files_and_path(args.input, args.output, "VFI", args.codec, args.container)
    model = load_model(args.model, device, dtype, args.mode)
    frames, fps, _, _ = VideoProcessor.load_frames(make_rgb=True)
    frames = VideoProcessor.np_image_to_tensor(frames)
    new_fps = fps * args.factor  # Adjust the frame rate according to the interpolation

    # Set seed for reproducibility.
    set_seed(args.seed)

    # Perform the frame interpolation.
    interpolate(model, frames, args.ds_factor, args.factor, VideoProcessor)

    # Save the interpolated video.
    VideoProcessor.write_buffered_frames_to_output(new_fps, args.keep_pngs)


if __name__ == "__main__":
    main()
