#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:47:29 2025

@author: blyss
"""
import torch
import av
from PIL import Image
import numpy as np

rgb_factors = {}
rgb_factors_bias = {}
# For HunyuanVideo
rgb_factors['hunyuan'] = [
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
    ]
rgb_factors_bias['hunyuan'] = [0.0259, -0.0192, -0.0761]

#For WanVideo
rgb_factors['wan'] = [
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
    ]
rgb_factors_bias['wan'] = [-0.1835, -0.0868, -0.3360]

def latent_preview(noisy_latents, original_latents, timesteps, current_step, args, modeltype="hunyuan"):
    """
    Function for previewing latents

    Parameters
    ----------
    noisy_latents : torch.tensor
        Latents at current timestep, BCFHW
    original_latents : torch.tensor
        Latents at step 0, BCFHW
    timesteps : torch.tensor
        Denoising schedule e.g timesteps
    current_step : int
        Current step we are on.
    args: args
        The commandline args.

    Returns
    -------
    None.

    """
    timesteps_percent = timesteps / 1000
    noise_remaining = timesteps_percent[current_step]
    denoisy_latents = noisy_latents - (original_latents * noise_remaining)  # Subtract original noise remaining
    latents = (denoisy_latents - denoisy_latents.mean()) / (denoisy_latents.std() + 1e-5)  # Normalize

    # Hard-coded linear transform to approximate latents -> RGB, specific to HunyuanVideo, other transforms and their bias can be found in ComfyUI's latent_formats.py
    
    latent_rgb_factors = rgb_factors[modeltype]
    latent_rgb_factors_bias = rgb_factors_bias[modeltype]

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

    latent_images = []
    # shape is (B, C, F, H, W)
    for t in range(latents.shape[2]):
        latent = latents[:, :, t, :, :]
        latent = latent[0].permute(1, 2, 0)
        latent_image = torch.nn.functional.linear(
            latent,
            latent_rgb_factors,
            bias=latent_rgb_factors_bias
        )
        latent_images.append(latent_image)
    latent_images = torch.stack(latent_images, dim=0)

    # Normalize to [0..1]
    latent_images_min = latent_images.min()
    latent_images_max = latent_images.max()
    if latent_images_max > latent_images_min:  # avoid divide-by-zero
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)
    latent_images = latent_images.clamp(0.0, 1.0)

    latent_images_np = latent_images.float().cpu().numpy()

    # Convert to uint8 [0..255]
    latent_images_np = (latent_images_np * 255).astype(np.uint8)

    # Latents are 1/8 size, 1/4 framerate
    scale_factor = 8
    fps = int(args.fps / 4)

    F, H, W, C = latent_images_np.shape
    upscaled_width = W * scale_factor
    upscaled_height = H * scale_factor
    if args.video_length > 1 and F > 1:  # We have more than one frame so write a video
        container = av.open("latent_preview.mp4", mode="w")
        stream = container.add_stream("libx264", rate=fps)
        stream.pix_fmt = "yuv420p"
        stream.width = upscaled_width
        stream.height = upscaled_height

        # Loop over each frame
        for i in range(F):
            frame_rgb = latent_images_np[i]
            pil_image = Image.fromarray(frame_rgb)
            pil_image_upscaled = pil_image.resize(
                (upscaled_width, upscaled_height),
                resample=Image.LANCZOS
            )

            frame_rgb_upscaled = np.array(pil_image_upscaled)
            video_frame = av.VideoFrame.from_ndarray(frame_rgb_upscaled, format="rgb24")

            for packet in stream.encode(video_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
    else:  # Single frame so save as image
        frame_rgb = latent_images_np[0]

        pil_image = Image.fromarray(frame_rgb)
        pil_image_upscaled = pil_image.resize(
            (upscaled_width, upscaled_height),
            resample=Image.LANCZOS
        )

        pil_image_upscaled.save("latent_preview.png")

    with open("./previewflag", "w", encoding="utf-8") as previewflag:  # Kludge to tell UI to update preview
        previewflag.write("Updated")
