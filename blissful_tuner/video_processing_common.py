#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Common video processing utilities for Blissful Tuner extension.

License: Apache-2.0
Created on Thu Apr 24 11:29:37 2025
Author: Blyss
"""

import os
import shutil
import subprocess
from typing import List, Tuple, Union

import cv2
import numpy as np
import torch


class BlissfulVideoProcessor:
    """
    Helper for loading video frames, converting between numpy images and torch tensors,
    and encoding sequences of frames to a ProRes video via ffmpeg.
    """

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        """
        Initialize with a target device and dtype for tensor operations.

        Args:
            device: torch.device (e.g. cuda or cpu).
            dtype: torch.dtype (e.g. torch.float32, torch.float16).
        """
        self.device = device
        self.dtype = dtype
        self.png_idx = 0
        self.frame_dir = ""
        self.input_file_path = ""
        self.output_file_path = ""
        self.output_directory = ""

    def prepare_files_and_path(
        self,
        input_file_path: str,
        output_file_path: Union[str, None],
        modifier: str
    ) -> Tuple[str, str]:
        """
        Determine and confirm input/output paths, generating a default output
        name if none provided, and set up the frames directory path.

        Args:
            input_file_path: Path to the source video.
            output_file_path: Desired output path or None to auto-generate.
            modifier: Suffix to append to the basename when auto-generating.

        Returns:
            A tuple of (input_file_path, output_file_path).
        """
        basename = os.path.basename(input_file_path)
        name, ext = os.path.splitext(basename)
        output_dir = os.path.dirname(input_file_path)

        if not output_file_path:
            output_file_path = os.path.join(output_dir, f"{name}_{modifier}{ext or '.mkv'}")

        if os.path.exists(output_file_path):
            choice = input(f"{output_file_path} exists. F for 'fix' by appending _! Overwrite?[y/N/f]: ").strip().lower()
            if choice == 'f':
                base = name
                while os.path.exists(output_file_path):
                    base += '_'
                    output_file_path = os.path.join(output_dir, f"{base}_{modifier}" + (ext or '.mkv'))
            elif choice != 'y':
                print("Aborted.")
                exit()

        self.input_file_path = input_file_path
        self.output_file_path = output_file_path
        self.output_directory = output_dir
        self.frame_dir = os.path.join(self.output_directory, 'frames')
        if os.path.exists(self.frame_dir):
            while os.path.exists(self.frame_dir):
                self.frame_dir += "_"

        print(f"Output video will be saved to: {self.output_file_path}")
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
        tensor: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Convert a 1×3×H×W torch tensor or list thereof (RGB float in [0,1])
        into H×W×3 uint8 BGR images suitable for OpenCV.

        Args:
            tensor: A torch.Tensor or list of torch.Tensors.

        Returns:
            A numpy BGR image or list of images.
        """
        def _convert(t: torch.Tensor) -> np.ndarray:
            t = t.squeeze(0).detach().cpu().clamp(0, 1).float()
            img = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            return img[..., ::-1]

        if isinstance(tensor, torch.Tensor):
            return _convert(tensor)
        return [_convert(t) for t in tensor]

    def load_video_frames(
        self,
        make_rgb: bool = False
    ) -> Tuple[List[np.ndarray], float, int, int]:
        """
        Load all frames from the input video as uint8 BGR or RGB numpy arrays.

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

    def write_np_images_to_video(
        self,
        imgs: List[np.ndarray],
        fps: float,
        out_width: int = None,
        out_height: int = None,
        keep_frames: bool = False
    ) -> None:
        """
        Dump a list of BGR frames as PNGs then encode them to ProRes video.

        Args:
            imgs: List of H×W×3 uint8 BGR frames.
            fps: Output frame rate.
            out_width: Optional scaled width.
            out_height: Optional scaled height.
            keep_frames: If True, do not delete PNGs afterward.
        """
        os.makedirs(self.frame_dir, exist_ok=False)
        for idx, img in enumerate(imgs):
            path = os.path.join(self.frame_dir, f"{idx:06d}.png")
            cv2.imwrite(path, img)
        self.write_buffered_pngs_to_video(fps, out_width, out_height, keep_frames)

    def write_buffered_pngs_to_video(
        self,
        fps: float,
        out_width: int = None,
        out_height: int = None,
        keep_frames: bool = False
    ) -> None:
        """
        Encode the PNG sequence in the frames directory to a ProRes video via ffmpeg.

        Args:
            fps: Frame rate.
            out_width: Scaled width (optional).
            out_height: Scaled height (optional).
            keep_frames: If True, preserve PNGs.
        """
        cmd = [
            "ffmpeg", "-framerate", str(fps),
            "-i", os.path.join(self.frame_dir, "%06d.png"),
            "-c:v", "prores_ks", "-profile:v", "3",
            "-pix_fmt", "yuv422p10le",
            "-colorspace", "1", "-color_primaries", "1", "-color_trc", "1",
            "-y"
        ]
        if out_width is not None and out_height is not None:
            cmd += ["-vf", f"scale={out_width}:{out_height}"]
        cmd.append(self.output_file_path)
        subprocess.run(cmd, check=True)
        if not keep_frames:
            shutil.rmtree(self.frame_dir, ignore_errors=True)
