import argparse
import logging
import math
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, ARCHITECTURE_QWEN_IMAGE, save_latent_cache_qwen_image
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image import qwen_image_model, qwen_image_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_qwen_image(batch: List[ItemInfo]) -> tuple[torch.Tensor]:
    # item.content: target image (H, W, C)

    # Stack batch into target tensor (B,H,W,C) in RGB order and control images list of tensors (H, W, C)
    contents = []
    for item in batch:
        contents.append(torch.from_numpy(item.content))  # target image

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    return contents


def encode_and_save_batch(vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, batch: List[ItemInfo]):
    # item.content: target image (H, W, C)
    contents = preprocess_contents_qwen_image(batch)  # (B, C, H, W)
    contents = contents.unsqueeze(2)  # (B, C, 1, H, W), Qwen-Image VAE needs F axis

    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(contents.to(vae.device, dtype=vae.dtype))

    # # debugging: decode and visualize the latents
    # with torch.no_grad():
    #     pixels = vae.decode_to_pixels(latents)  # 0 to 1
    #     print(pixels.min(), pixels.max(), pixels.shape)
    #     pixels = pixels.to(torch.float32).cpu()
    #     pixels = (pixels * 255).clamp(0, 255).to(torch.uint8)  # convert to uint8
    #     pixels = pixels[0].permute(1, 2, 0)  # C, H, W -> H, W, C
    #     pixels = pixels.numpy()
    #     import cv2
    #     cv2.imshow("Decoded Pixels", pixels)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        target_latent = latents[b]  # 1, C, H, W. Target latents for this image (ground truth)

        print(f"Saving cache for item {item.item_key} at {item.latent_cache_path}, target latents shape: {target_latent.shape}")

        save_latent_cache_qwen_image(item_info=item, latent=target_latent)


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    # parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in Qwen-Image.")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "VAE checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = qwen_image_utils.load_vae(args.vae, device=device, disable_mmap=True)
    vae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
