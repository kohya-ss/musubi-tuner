import argparse
import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_KREA2,
    ItemInfo,
    save_latent_cache_krea2,
)
from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_krea2(batch: List[ItemInfo]) -> torch.Tensor:
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content))

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, C, H, W
    contents = contents.unsqueeze(2)  # (B, C, 1, H, W)
    contents = contents / 127.5 - 1.0
    return contents


def encode_and_save_batch(vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, batch: List[ItemInfo]):
    contents = preprocess_contents_krea2(batch)

    with torch.no_grad():
        latents = []
        for l in range(contents.shape[2]):
            lat = vae.encode_pixels_to_latents(contents[:, :, l:l+1, :, :].to(vae.device, dtype=vae.dtype))
            latents.append(lat)
        latents = torch.cat(latents, dim=2)  # (B, C, F, H, W)

    for b, item in enumerate(batch):
        target_latent = latents[b]  # C, 1, H, W
        print(f"Saving cache for item {item.item_key} at {item.latent_cache_path}, latents shape: {target_latent.shape}")
        save_latent_cache_krea2(item_info=item, latent=target_latent)


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in Krea2.")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    architecture = ARCHITECTURE_KREA2

    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "VAE checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = krea2_utils.load_vae(args.vae, device=device, disable_mmap=True)
    vae.to(device)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args, supports_alpha=False)


if __name__ == "__main__":
    main()
