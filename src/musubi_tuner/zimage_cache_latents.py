"""
Cache latents for Z-Image architecture.

This script encodes images using Z-Image's VAE and caches the latent representations
for faster training. Unlike other architectures, Z-Image does not support control images,
so only the target image latents are cached.
"""

import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, ARCHITECTURE_Z_IMAGE, save_latent_cache_z_image
from musubi_tuner.zimage import zimage_autoencoder
from musubi_tuner.zimage.zimage_autoencoder import AutoencoderKL
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_zimage(batch: List[ItemInfo]) -> torch.Tensor:
    """
    Preprocess batch contents for Z-Image VAE encoding.

    Args:
        batch: List of ItemInfo containing target images

    Returns:
        torch.Tensor: Preprocessed image tensor (B, C, H, W) normalized to [-1, 1]
    """
    contents = []
    for item in batch:
        # item.content: target image (H, W, C) in RGB order, uint8
        content = torch.from_numpy(item.content)
        if content.shape[-1] == 4:  # RGBA
            content = content[..., :3]  # remove alpha channel
        contents.append(content)

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents.float() / 127.5 - 1.0  # normalize to [-1, 1]

    return contents


def encode_and_save_batch(vae: AutoencoderKL, batch: List[ItemInfo]):
    """
    Encode a batch of images and save their latent representations.

    Args:
        vae: Z-Image VAE model (AutoencoderKL)
        batch: List of ItemInfo containing images to encode
    """
    contents = preprocess_contents_zimage(batch)

    h, w = contents.shape[2], contents.shape[3]
    if h < 16 or w < 16:
        item = batch[0]
        raise ValueError(f"Image size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    with torch.no_grad():
        # Move to VAE device and dtype
        contents = contents.to(vae.device, dtype=vae.dtype)

        # Encode using VAE - returns DiagonalGaussianDistribution
        posterior = vae.encode(contents)

        # Use mode() for deterministic latents (mean of the distribution)
        # This is preferred for training as it provides consistent latents
        latents = posterior.mode()

    # Save cache for each item in the batch
    for b, item in enumerate(batch):
        latent = latents[b]  # C, H, W

        logger.debug(f"Saving cache for item {item.item_key} at {item.latent_cache_path}. Latent shape: {latent.shape}")

        save_latent_cache_z_image(item_info=item, latent=latent)


def main():
    parser = cache_latents.setup_parser_common()
    # Z-Image VAE doesn't need special tiling options like HunyuanVideo
    # but we can add them if needed in the future

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.vae_dtype is not None:
        logger.warning("VAE dtype is specified but Z-Image VAE always uses float32 for better precision.")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return

    assert args.vae is not None, "VAE checkpoint is required (--vae)"

    logger.info(f"Loading Z-Image VAE from {args.vae}")
    vae = zimage_autoencoder.load_autoencoder_kl(args.vae, device=device, disable_mmap=True)
    vae.eval()
    logger.info(f"Loaded Z-Image VAE, dtype: {vae.dtype}")

    # Encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    # Reuse core loop from cache_latents
    cache_latents.encode_datasets(datasets, encode, args)

    logger.info("Done!")


if __name__ == "__main__":
    main()
