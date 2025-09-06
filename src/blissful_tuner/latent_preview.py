#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent preview for Blissful Tuner extension
License: Apache 2.0
Created on Mon Mar 10 16:47:29 2025

@author: blyss
"""
import argparse
import os
from typing import List
import torch
import av
from PIL import Image
from .taehv import TAEHV
from .taesd import TAESD
from .utils import load_torch_file
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")


class LatentPreviewer():
    @torch.inference_mode()
    def __init__(
        self,
        args: argparse.Namespace,
        original_latents: torch.Tensor,
        scheduler: any,
        device: torch.device,
        dtype: torch.dtype,
        model_type: str = "hunyuan"
    ) -> None:
        self.mode = "latent2rgb" if args.preview_vae is None else "taehv" if model_type != "flux" else "taesd"
        logger.info(f"Initializing latent previewer with mode {self.mode}...")
        self.subtract_noise = False
        self.args = args
        self.model_type = model_type
        self.device = device
        self.dtype = dtype if dtype != torch.float8_e4m3fn else torch.float16
        self.scheduler = None
        self.sigmas = None
        self.warning_done = False

        if self.model_type not in ["hunyuan", "wan", "framepack", "flux"]:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        if model_type != "framepack" and original_latents is not None:  # Framepack will send in na clean latent, others will be noisy
            self.original_latents = original_latents.to(self.device)
            self.subtract_noise = True
            if scheduler is not None:
                self.sigmas = scheduler.sigmas
                self.scheduler = scheduler

        if "tae" in self.mode:
            if os.path.exists(args.preview_vae):
                tae_sd = load_torch_file(args.preview_vae, safe_load=True, device=args.device)
            else:
                raise FileNotFoundError(f"{args.preview_vae} was not found!")
            logger.info(f"Loading TAE: {args.preview_vae}...")
            tae_class = TAEHV if self.mode == "taehv" else TAESD
            self.decoder = self.decode_taehv if self.mode == "taehv" else self.decode_taesd
            self.tae = tae_class(tae_sd).to("cpu", self.dtype)  # Offload for VRAM and match datatype
        elif self.mode == "latent2rgb":
            self.decoder = self.decode_latent2rgb

    @torch.inference_mode()
    def preview(self, noisy_latents: torch.Tensor, step: int = None) -> None:
        self.clean_cache()
        if self.model_type == "wan":
            noisy_latents = noisy_latents.unsqueeze(0)  # F, C, H, W -> B, F, C, H, W
        elif self.model_type in ["hunyuan", "framepack"]:
            pass  # already B, F, C, H, W
        denoisy_latents = self.subtract_original_and_normalize(noisy_latents, step) if self.subtract_noise else noisy_latents
        decoded = self.decoder(denoisy_latents)  # returned as F, C, H, W

        # Upscale if we used latent2rgb so output is same size as expected
        scale_factor = 8 if self.mode == "latent2rgb" else None
        if scale_factor is not None:
            upscaled = torch.nn.functional.interpolate(
                decoded,
                scale_factor=scale_factor,
                mode="bicubic",
                align_corners=False
            )
        else:
            upscaled = decoded

        _, _, h, w = upscaled.shape
        self.write_preview(upscaled, w, h)
        self.clean_cache()

    def clean_cache(self):
        if self.device == "cuda" or self.device == torch.device("cuda"):
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def subtract_original_and_normalize(self, noisy_latents: torch.Tensor, step: int = None):
        device = noisy_latents.device
        noise = self.original_latents
        if self.scheduler is not None and self.sigmas is not None:
            noise_remaining = self.sigmas[self.scheduler.step_index]  # get step directly from scheduler
            if hasattr(self.scheduler, "last_noise") and self.scheduler.last_noise is not None:
                noise = self.scheduler.last_noise  # Some schedulers e.g. LCM change the noise/use additional noise.
        elif step is not None and self.sigmas is not None:
            noise_remaining = self.sigmas[step]
        else:
            if not self.warning_done:
                self.warning_done = True
            logger.warning("No sigmas provided to previewer so preview may be incorrect/bad!")
        # Subtract the portion of original latents
        denoisy_latents = noisy_latents - (noise.to(device) * noise_remaining)
        normalized_denoisy_latents = (denoisy_latents - denoisy_latents.mean()) / (denoisy_latents.std() + 1e-9)
        return normalized_denoisy_latents

    @torch.inference_mode()
    def write_preview(self, frames: List[torch.Tensor], width: int, height: int) -> None:
        target = os.path.join(self.args.save_path, "latent_preview.mp4")
        # Check if we only have a single frame.
        if frames.shape[0] == 1:
            # Clamp, scale, convert to byte and move to CPU
            frame = frames[0].clamp(0, 1).mul(255).byte().cpu()
            # Permute from (3, H, W) to (H, W, 3) for PIL.
            frame_np = frame.permute(1, 2, 0).numpy()
            # Change the target filename from .mp4 to .png
            target_img = target.replace(".mp4", ".png")
            Image.fromarray(frame_np).save(target_img)
            return

        # Otherwise, write out as a video.
        if hasattr(self.args, "fps"):
            out_fps = self.args.fps // 4 if self.mode == "latent2rgb" else self.args.fps
        else:
            out_fps = 24 // 4 if self.mode == "latent2rgb" else 24
            logger.warning(f"Requested to write out a video but no fps in args! Probably shouldn't see this but we'll set it to {out_fps} and keep going")
        container = av.open(target, mode="w")
        stream = container.add_stream("libx264", rate=out_fps)
        stream.pix_fmt = "yuv420p"
        stream.width = width
        stream.height = height

        # Loop through each frame.
        for frame in frames:
            # Clamp to [0,1], scale, convert to byte and move to CPU.
            frame = frame.clamp(0, 1).mul(255).byte().cpu()
            # Permute from (3, H, W) -> (H, W, 3) for AV.
            frame_np = frame.permute(1, 2, 0).numpy()
            video_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(video_frame):
                container.mux(packet)

        # Flush out any remaining packets and close.
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    @torch.inference_mode()
    def decode_taehv(self, latents: torch.Tensor):
        """
        Decodes latents with the TAEHV model, returns shape (F, C, H, W).
        """
        self.tae.to(self.device)  # Onload
        latents_permuted = latents.permute(0, 2, 1, 3, 4)  # Reordered to B, F, C, H, W for TAE
        latents_permuted = latents_permuted.to(device=self.device, dtype=self.dtype)
        decoded = self.tae.decode_video(latents_permuted, parallel=False, show_progress_bar=False)
        self.tae.to("cpu")  # Offload
        return decoded.squeeze(0)  # squeeze off batch dimension as next step doesn't want it

    @torch.inference_mode()
    def decode_taesd(self, latents: torch.Tensor):
        self.tae.to(self.device)  # Onload
        latents = latents.to(device=self.device, dtype=self.dtype)
        decoded = self.tae.decoder(latents).clamp(0, 1)
        self.tae.to("cpu")  # Offload
        return decoded

    @torch.inference_mode()
    def decode_latent2rgb(self, latents: torch.Tensor):
        """
        Decodes latents to RGB using linear transform, returns shape (F, 3, H, W).
        """
        model_params = {
            "hunyuan": {
                "rgb_factors": [
                    [-0.0395, -0.0331,  0.0445],
                    [ 0.0696,  0.0795,  0.0518],
                    [ 0.0135, -0.0945, -0.0282],
                    [ 0.0108, -0.0250, -0.0765],
                    [-0.0209,  0.0032,  0.0224],
                    [-0.0804, -0.0254, -0.0639],
                    [-0.0991,  0.0271, -0.0669],
                    [-0.0646, -0.0422, -0.0400],
                    [-0.0696, -0.0595, -0.0894],
                    [-0.0799, -0.0208, -0.0375],
                    [ 0.1166,  0.1627,  0.0962],
                    [ 0.1165,  0.0432,  0.0407],
                    [-0.2315, -0.1920, -0.1355],
                    [-0.0270,  0.0401, -0.0821],
                    [-0.0616, -0.0997, -0.0727],
                    [ 0.0249, -0.0469, -0.1703]
                ],
                "bias": [0.0259, -0.0192, -0.0761],
            },
            "wan": {
                "rgb_factors": [
                    [-0.1299, -0.1692,  0.2932],
                    [ 0.0671,  0.0406,  0.0442],
                    [ 0.3568,  0.2548,  0.1747],
                    [ 0.0372,  0.2344,  0.1420],
                    [ 0.0313,  0.0189, -0.0328],
                    [ 0.0296, -0.0956, -0.0665],
                    [-0.3477, -0.4059, -0.2925],
                    [ 0.0166,  0.1902,  0.1975],
                    [-0.0412,  0.0267, -0.1364],
                    [-0.1293,  0.0740,  0.1636],
                    [ 0.0680,  0.3019,  0.1128],
                    [ 0.0032,  0.0581,  0.0639],
                    [-0.1251,  0.0927,  0.1699],
                    [ 0.0060, -0.0633,  0.0005],
                    [ 0.3477,  0.2275,  0.2950],
                    [ 0.1984,  0.0913,  0.1861]
                ],
                "bias": [-0.1835, -0.0868, -0.3360],
            },
            "flux": {
                "rgb_factors": [
                    [-0.0346,  0.0244,  0.0681],
                    [ 0.0034,  0.0210,  0.0687],
                    [ 0.0275, -0.0668, -0.0433],
                    [-0.0174,  0.0160,  0.0617],
                    [ 0.0859,  0.0721,  0.0329],
                    [ 0.0004,  0.0383,  0.0115],
                    [ 0.0405,  0.0861,  0.0915],
                    [-0.0236, -0.0185, -0.0259],
                    [-0.0245,  0.0250,  0.1180],
                    [ 0.1008,  0.0755, -0.0421],
                    [-0.0515,  0.0201,  0.0011],
                    [ 0.0428, -0.0012, -0.0036],
                    [ 0.0817,  0.0765,  0.0749],
                    [-0.1264, -0.0522, -0.1103],
                    [-0.0280, -0.0881, -0.0499],
                    [-0.1262, -0.0982, -0.0778]
                ],
                "bias": [-0.0329, -0.0718, -0.0851]
            }
        }
        latent_rgb_factors = model_params[self.model_type]["rgb_factors"] if self.model_type != "framepack" else model_params["hunyuan"]
        latent_rgb_factors_bias = model_params[self.model_type]["bias"]

        # Prepare linear transform
        latent_rgb_factors = torch.tensor(
            latent_rgb_factors,
            device=latents.device,
            dtype=latents.dtype
        ).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(
            latent_rgb_factors_bias,
            device=latents.device,
            dtype=latents.dtype
        )

        # For each frame, apply the linear transform
        latent_images = []
        if self.model_type == "flux":
            latents = latents[:, :, None, :, :]  # Will be B, C, H, W so make a fake frame dim to make life easy
        for t in range(latents.shape[2]):  # B, C, F, H, W
            extracted = latents[:, :, t, :, :][0].permute(1, 2, 0)  # shape = (H, W, C) after .permute(1,2,0)
            rgb = torch.nn.functional.linear(extracted, latent_rgb_factors, bias=latent_rgb_factors_bias)  # shape = (H, W, 3) after linear
            latent_images.append(rgb)

        # Stack frames into (F, H, W, 3)
        latent_images = torch.stack(latent_images, dim=0)

        # Normalize to [0..1]
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        if latent_images_max > latent_images_min:
            latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)

        # Permute to (F, 3, H, W) before returning
        latent_images = latent_images.permute(0, 3, 1, 2)
        return latent_images
