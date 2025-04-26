#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Upscaler for Blissful Tuner Extension

License: Apache 2.0
Created on Wed Apr 23 10:19:19 2025
@author: blyss
"""

from typing import List
import torch
import numpy as np
from tqdm import tqdm

from swinir.network_swinir import SwinIR
from spandrel import ImageModelDescriptor, ModelLoader
from video_processing_common import BlissfulVideoProcessor, set_seed, setup_parser_video_common
from utils import setup_compute_context, load_torch_file, BlissfulLogger
logger = BlissfulLogger(__name__, "green")


def upscale_frames_swin(
    model: torch.nn.Module,
    frames: List[np.ndarray],
    VideoProcessor: BlissfulVideoProcessor
) -> List[np.ndarray]:
    """
    Upscale a list of RGB frames using a compiled SwinIR model.

    Args:
        model: Loaded SwinIR upsampler.
        frames: List of H×W×3 float32 RGB arrays in [0,1].
        device: torch device (cpu or cuda).
        dtype: torch.dtype to use for computation.

    Returns:
        List of upscaled H'×W'×3 uint8 BGR frames.
    """
    window_size = 8
    for img in tqdm(frames, desc="Upscaling SwinIR"):
        # Mark step for CUDA graph capture if enabled
        torch.compiler.cudagraph_mark_step_begin()

        # Convert HWC RGB → CHW tensor
        tensor = VideoProcessor.np_image_to_tensor(img)

        # Pad to window multiple
        _, _, h, w = tensor.shape
        h_pad = ((h + window_size - 1) // window_size) * window_size - h
        w_pad = ((w + window_size - 1) // window_size) * window_size - w
        tensor = torch.cat([tensor, torch.flip(tensor, [2])], 2)[:, :, : h + h_pad, :]
        tensor = torch.cat([tensor, torch.flip(tensor, [3])], 3)[:, :, :, : w + w_pad]

        # Inference
        with torch.no_grad():
            out = model(tensor)

        # Post-process: NCHW → HWC BGR uint8
        VideoProcessor.write_np_or_tensor_to_png(out)


def load_swin_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Instantiate and load weights into a SwinIR model.

    Args:
        model_path: Path to checkpoint (.pth or safetensors).
        device: torch device.
        dtype: torch dtype.
    Returns:
        SwinIR model in eval() on device and dtype.
    """
    logger.info(f"Loading SwinIR model ({dtype})…")
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6] * 9,
        embed_dim=240,
        num_heads=[8] * 9,
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='3conv',
    )
    ckpt = load_torch_file(model_path)
    key = 'params_ema' if 'params_ema' in ckpt else None
    model.load_state_dict(ckpt[key] if key else ckpt, strict=True)
    model.to(device, dtype).eval()
    return model


def load_esrgan_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """
    Load an ESRGAN (or RRDBNet) style model via Spandrel loader.

    Args:
        model_path: Path to ESRGAN checkpoint.
        device: torch device.
        dtype: torch dtype.
    Returns:
        Model ready for inference.
    """
    logger.info(f"Loading ESRGAN model ({dtype})…")
    descriptor = ModelLoader().load_from_file(model_path)
    assert isinstance(descriptor, ImageModelDescriptor)
    model = descriptor.model.eval().to(device, dtype)
    return model


def main() -> None:
    """
    Parse CLI args, load input, model, and run upscaling pipeline.
    """
    parser = setup_parser_video_common(description="Video upscaling using SwinIR or ESRGAN models")
    parser.add_argument(
        "--scale", type=float, default=2,
        help="Final scale multiplier for output resolution"
    )
    parser.add_argument(
        "--mode", choices=["swinir", "esrgan"], default="swinir",
        help="Model architecture to use"
    )
    args = parser.parse_args()
    args.mode = args.mode.lower()
    # Map string → torch.dtype
    device, dtype = setup_compute_context(None, args.dtype)
    VideoProcessor = BlissfulVideoProcessor(device, dtype)
    VideoProcessor.prepare_files_and_path(args.input, args.output, args.mode.upper())

    frames, fps, w, h = VideoProcessor.load_frames(make_rgb=True)
    set_seed(args.seed)
    # Load and run model
    if args.mode == "swinir":
        model = load_swin_model(args.model, device, dtype)
        upscale_frames_swin(model, frames, VideoProcessor)
    else:
        model = load_esrgan_model(args.model, device, dtype)
        logger.info("Processing with ESRGAN...")
        for frame in tqdm(frames, desc="Upscaling ESRGAN"):
            inp = VideoProcessor.np_image_to_tensor(frame)
            with torch.no_grad():
                sr = model(inp)
            VideoProcessor.write_np_or_tensor_to_png(sr)

    # Write video
    logger.info("Encoding output video...")
    out_w, out_h = int(w * args.scale), int(h * args.scale)
    VideoProcessor.write_buffered_frames_to_output(fps, args.keep_pngs, (out_w, out_h))
    logger.info("Done!")


if __name__ == "__main__":
    main()
