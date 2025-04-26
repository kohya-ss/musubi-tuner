#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common video processing utilities for Blissful Tuner extension.

License: Apache-2.0
Created on Thu Apr 24 11:29:37 2025
Author: Blyss
"""
import argparse
from rich_argparse import RichHelpFormatter
import glob
import os
import random
import shutil
import subprocess
from einops import rearrange
import torchvision
from typing import List, Tuple, Union, Optional
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
import torch
from pathlib import Path
try:
    from blissful_tuner.utils import BlissfulLogger, string_to_seed
except ImportError:  # This is needed so we can import either within blissful_tuner directory or base musubi directory
    from utils import BlissfulLogger, string_to_seed


logger = BlissfulLogger(__name__, "#8e00ed")


def set_seed(seed: Union[int, str] = None) -> int:
    """
    Sets the random seed for reproducibility.
    """
    if seed is None:
        seed = random.getrandbits(32)
    else:
        try:
            seed = int(seed)
        except ValueError:
            seed = string_to_seed(seed, bits=32)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def setup_parser_video_common(description: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, formatter_class=RichHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to the model(directory for GIMM-VFI, .safetensors otherwise)")
    parser.add_argument("--input", required=True, help="Input video/image to process")
    parser.add_argument("--dtype", type=str, default="fp32", help="Datatype to use")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file path, default is same path as input. Extension may be changed to match chosen settings!"
    )
    parser.add_argument("--seed", type=str, default=None, help="Seed for reproducibility")
    parser.add_argument("--keep_pngs", action="store_true", help="Also keep individual frames as PNGs")
    parser.add_argument(
        "--codec", choices=["prores", "h264", "h265"], default="prores",
        help="Codec to use, choose from 'prores', 'h264', or 'h265'. Ignored for images."
    )
    parser.add_argument(
        "--container", choices=["mkv", "mp4"], default="mkv",
        help="Container format to use, choose from 'mkv' or 'mp4'. Note prores can only go in MKV! Ignored for images."
    )
    return parser


class BlissfulVideoProcessor:
    """
    Manager for working with images and video in generative AI workloads
    """

    def __init__(self, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> None:
        """
        Initialize with a target device and dtype for tensor operations.

        Args:
            device: torch.device (e.g. cuda or cpu).
            dtype: torch.dtype (e.g. torch.float32, torch.float16).
        """
        self.device = device if device is not None else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.png_idx = 0
        self.frame_dir = ""
        self.input_file_path = ""
        self.output_file_path = ""
        self.output_directory = ""
        self.new_ext = ".mkv"
        self.codec = "prores"

    def prepare_files_and_path(
        self,
        input_file_path: Optional[str] = None,
        output_file_path: Optional[str] = None,
        modifier: Optional[str] = "",
        codec: Optional[str] = "prores",
        container: Optional[str] = "mkv",
    ) -> Tuple[str, str]:
        """
        Determine and confirm input/output paths, generating a default output
        name if none provided, and set up the frames directory path and codec/container.

        Args:
            input_file_path: Path to the source video.
            output_file_path: Desired output path or None to auto-generate.
            modifier: Suffix to append to the basename when auto-generating.
            codec: The video codec to use(ignored for images)
            container: The container format to use(ignored for images)

        Returns:
            A tuple of (input_file_path, output_file_path).
        """
        def _is_image_file(path: Path) -> bool:
            try:
                with Image.open(path) as img:
                    img.verify()
                return True
            except (UnidentifiedImageError, OSError):
                return False
        if codec is not None:
            if codec.lower() in ["prores", "h264", "h265"]:
                self.codec = codec.lower()
            else:
                raise ValueError("Invalid codec requested {codec}! Expected 'prores', 'h264', or 'h265'!")
        if container is not None:
            if container.lower() == "mkv":
                self.new_ext = ".mkv"
            elif container.lower() == "mp4":
                if self.codec != "prores":
                    self.new_ext = ".mp4"
                else:
                    logger.warning("Prores can only be written into an mkv but mp4 was passed! Selecting mkv and continuing...")
            else:
                raise ValueError("Invalid container format {container}! Expected 'mkv' or 'mp4'!")
        if input_file_path is not None:
            basename = os.path.basename(input_file_path)
            name, ext = os.path.splitext(basename)
            output_dir = os.path.dirname(input_file_path)
            is_image = _is_image_file(input_file_path)
            if is_image:
                self.new_ext = ".png"
                self.codec = "png"
        elif output_file_path is not None:
            output_dir = os.path.dirname(output_file_path)
        else:
            raise ValueError("At least one of input_file_path or output_file_path must be provided!")

        if not output_file_path:
            output_file_path = os.path.join(output_dir, f"{name}_{modifier}{self.new_ext}")
        o_basename = os.path.basename(output_file_path)
        o_name, o_ext = os.path.splitext(o_basename)
        o_output_dir = os.path.dirname(output_file_path)
        if o_ext != self.new_ext:
            logger.warning(f"Extension '{o_ext[-3:]}' not valid for output! Updating to '{self.new_ext[-3:]}'...")
            output_file_path = os.path.join(o_output_dir, f"{o_name}{self.new_ext}")

        if os.path.exists(output_file_path):
            choice = input(f"{output_file_path} exists. F for 'fix' by appending _! Overwrite?[y/N/f]: ").strip().lower()
            if choice == 'f':
                base = o_name
                while os.path.exists(output_file_path):
                    base += '_'
                    output_file_path = os.path.join(o_output_dir, f"{base}{self.new_ext}")
            elif choice != 'y':
                logger.info("Aborted.")
                exit()

        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.output_directory = output_dir
        self.frame_dir = os.path.join(self.output_directory, 'frames')
        if os.path.exists(self.frame_dir):
            while os.path.exists(self.frame_dir):
                self.frame_dir += "_"

        logger.info(f"Output will be saved to: {self.output_file_path} using {self.codec}!")
        return self.input_file_path, self.output_file_path

    def np_image_to_tensor(
        self,
        image: Union[np.ndarray, List[np.ndarray]]
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Convert a single H×W×3 numpy image or list of images (RGB uint8 or float32)
        into torch tensors of shape 1×3×H×W in [0,1], on the configured device and dtype.

        Args:
            image: An RGB image array or list of arrays.

        Returns:
            A torch.Tensor or list of torch.Tensors.
        """
        def _convert(img: np.ndarray) -> torch.Tensor:
            arr = img.astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr.transpose(2, 0, 1))
            return tensor.unsqueeze(0).to(self.device, self.dtype)

        if isinstance(image, np.ndarray):
            return _convert(image)
        return [_convert(img) for img in image]

    def tensor_to_np_image(
        self,
        tensor: Union[torch.Tensor, List[torch.Tensor]],
        rescale: bool = False
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convert a 1×3×H×W or 3×H×W torch tensor (RGB float in [0,1] or [-1,1])
        into H×W×3 uint8 BGR images suitable for OpenCV (and do rescale if needed).

        Args:
            tensor:   A torch.Tensor or list of torch.Tensors.
            rescale:  If True, assumes the tensor is in [-1,1] and remaps to [0,1].
        Returns:
            A numpy BGR image or list of images.
        """
        def _convert(t: torch.Tensor) -> np.ndarray:
            # 1) Bring to CPU, float, clamp
            t = t.detach().cpu().float()
            # 2) Optional range shift from [-1,1] to [0,1]
            if rescale:
                t = (t + 1.0) / 2.0
            t = t.clamp(0.0, 1.0)

            # 3) Normalize shape to [1,3,H,W]
            if t.ndim == 3:            # [3,H,W]
                t = t.unsqueeze(0)     # -> [1,3,H,W]
            elif t.ndim != 4 or t.shape[1] != 3:
                raise ValueError(f"Unexpected tensor shape: {tuple(t.shape)}")

            # 4) Squeeze batch, permute to H×W×C, scale to 0–255
            t = t.squeeze(0)                         # [3,H,W]
            img = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)        # [H,W,3]

            # 5) Flip RGB→BGR for OpenCV
            return img[..., ::-1]

        if isinstance(tensor, torch.Tensor):
            return _convert(tensor)
        else:
            return [_convert(t) for t in tensor]

    def load_frames(
        self,
        make_rgb: Optional[bool] = False
    ) -> Tuple[List[np.ndarray], float, int, int]:
        """
        Load all frames from the input video/image as uint8 BGR or RGB numpy arrays.

        Args:
            make_rgb: If True, convert frames to RGB.

        Returns:
            frames: List of H×W×3 image arrays.
            fps: Frame rate of the video.
            width: Original width.
            height: Original height.
        """
        cap = cv2.VideoCapture(self.input_file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: List[np.ndarray] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if make_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()
        return frames, fps, width, height

    def write_np_or_tensor_to_png(
        self,
        img: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """
        Write a single frame (numpy BGR or tensor) to the frames directory as PNG.

        Args:
            img: A BGR uint8 image array or a tensor to convert.
        """
        if isinstance(img, torch.Tensor):
            img = self.tensor_to_np_image(img)
        if self.png_idx == 0:
            os.makedirs(self.frame_dir, exist_ok=False)
        path = os.path.join(self.frame_dir, f"{self.png_idx:06d}.png")
        cv2.imwrite(path, img)
        self.png_idx += 1

    def write_np_images_to_output(
        self,
        imgs: List[np.ndarray],
        fps: Optional[float] = 1,
        keep_frames: Optional[bool] = False,
        rescale: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Dump a list of BGR frames as PNGs

        Args:
            imgs: List of H×W×3 uint8 BGR frames.
            fps: Output frame rate.
            rescale: To resize the output
            keep_frames: If True, do not delete PNGs afterward.
        """
        os.makedirs(self.frame_dir, exist_ok=False)
        for idx, img in enumerate(imgs):
            path = os.path.join(self.frame_dir, f"{idx:06d}.png")
            cv2.imwrite(path, img)
        self.write_buffered_frames_to_output(fps, keep_frames, rescale)

    def write_buffered_frames_to_output(
        self,
        fps: Optional[float] = 1,
        keep_frames: Optional[bool] = False,
        rescale: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Encode the PNG sequence in the frames directory to a video via ffmpeg,
        or—if there's only one frame—just write out an (optionally-rescaled) PNG.
        """
        # 1) get all the PNGs
        pattern = os.path.join(self.frame_dir, "*.png")
        png_paths = sorted(glob.glob(pattern))

        # 2) single-image case
        if len(png_paths) == 1:
            src = png_paths[0]

            if rescale is None:
                # just copy the original
                shutil.copy(src, self.output_file_path)
            else:
                # PIL approach: open, resize, save
                width, height = rescale
                with Image.open(src) as img:
                    # LANCZOS gives a high-quality down/upscale
                    img = img.resize((width, height), Image.LANCZOS)
                    img.save(self.output_file_path)
        else:
            # 3) multi‐frame → video
            codec_args = self._get_ffmpeg_codec_args()
            cmd = [
                "ffmpeg", "-framerate", str(fps),
                "-i", os.path.join(self.frame_dir, "%06d.png"),
            ] + codec_args

            if rescale is not None:
                w, h = rescale
                cmd += ["-vf", f"scale={w}:{h}"]

            # overwrite without prompt
            cmd += ["-y", self.output_file_path]

            subprocess.run(cmd, check=True)
        if not keep_frames:
            shutil.rmtree(self.frame_dir, ignore_errors=True)

    def _get_ffmpeg_codec_args(self) -> List[str]:
        """
        Return the ffmpeg args for codec/quality based on self.codec.
        """
        if self.codec == "prores":
            # prores_ks profile 3 + broadcast-safe colors
            return [
                "-c:v", "prores_ks",
                "-profile:v", "3",
                "-pix_fmt", "yuv422p10le",
                "-colorspace", "1",
                "-color_primaries", "1",
                "-color_trc", "1",
            ]
        elif self.codec == "h264":
            # libx264
            return [
                "-c:v", "libx264",
                "-preset", "slow",
                "-crf", "16",
                "-pix_fmt", "yuv420p",
            ]
        elif self.codec == "h265":
            # libx265
            return [
                "-c:v", "libx265",
                "-preset", "slow",
                "-crf", "16",
                "-pix_fmt", "yuv420p",
            ]
        else:
            raise ValueError(f"Unsupported codec: {self.codec}")


def save_videos_grid_advanced(
    videos: torch.Tensor,
    output_video: str,
    codec: str,
    container: str,
    rescale: bool = False,
    fps: int = 24,
    n_rows: int = 1,
    keep_frames: bool = False
):

    # 1) rearrange so we iterate over time
    videos = rearrange(videos, "b c t h w -> t b c h w")

    VideoProcessor = BlissfulVideoProcessor()
    VideoProcessor.prepare_files_and_path(
        input_file_path=None,
        output_file_path=output_video,
        codec=codec,
        container=container
    )

    outputs = []
    for video in videos:
        # 2) tile frames into one grid [C, H, W]
        grid = torchvision.utils.make_grid(video, nrow=n_rows)
        # 3) convert to an OpenCV-ready numpy array
        np_img = VideoProcessor.tensor_to_np_image(grid, rescale=rescale)
        outputs.append(np_img)

    # 4) write them out
    VideoProcessor.write_np_images_to_output(outputs, fps, keep_frames)
