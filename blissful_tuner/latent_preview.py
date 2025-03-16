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


def latent_preview(noisy_latents, original_latents, timesteps, current_step, args, model_type="hunyuan"):
    """
    Function for previewing latents

    Parameters
    ----------
    noisy_latents : torch.tensor
        Latents at current timestep
    original_latents : torch.tensor
        Latents at step 0
    timesteps : torch.tensor
        Denoising schedule e.g timesteps
    current_step : int
        Current step we are on.
    args: args
        The commandline args.
    model_type:
        The type of model we are previewing for

    Returns
    -------
    None.

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
            "frame_axis": 2,  # 5 dimensions total, B, C, F, H, W
            "extract": lambda x, t: x[:, :, t, :, :][0].permute(1, 2, 0),
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
            "frame_axis": 1,  # 4 dimensions total, C, F, H, W
            "extract": lambda x, t: x[:, t, :, :].permute(1, 2, 0),
        },
    }

    # Validate model_type
    if model_type not in model_params:
        raise ValueError(f"Unsupported model type: {model_type}")

    frame_axis = model_params[model_type]["frame_axis"]
    extract_fn = model_params[model_type]["extract"]
    latent_rgb_factors = model_params[model_type]["rgb_factors"]
    latent_rgb_factors_bias = model_params[model_type]["bias"]

    timesteps_percent = timesteps / 1000
    noise_remaining = timesteps_percent[current_step].to(device=noisy_latents.device)  # Make sure to keep things on same device as original
    denoisy_latents = noisy_latents - (original_latents.to(device=noisy_latents.device) * noise_remaining)  # Subtract original noise remaining
    latents = (denoisy_latents - denoisy_latents.mean()) / (denoisy_latents.std() + 1e-5).to(device=noisy_latents.device)  # Normalize

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

    # Iterate over all the frames inside the latents and put them through the linear transform to rgb
    latent_images = [
        torch.nn.functional.linear(extract_fn(latents, t), latent_rgb_factors, bias=latent_rgb_factors_bias)
        for t in range(latents.shape[frame_axis])
    ]

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

    # Latents are 1/8 size, 1/4 framerate, this could be updated later for other scaling if necessary
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
