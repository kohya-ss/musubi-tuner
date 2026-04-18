"""
Cache latents for ERNIE-Image architecture.

Encodes images using FLUX.2 VAE (shared with ERNIE-Image), then patchifies
and BN-normalizes, caching the result for training.
"""

import logging
from typing import List

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.cache_io import save_latent_cache_ernie_image
from musubi_tuner.dataset.architectures import ARCHITECTURE_ERNIE_IMAGE
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.ernie_image import ernie_image_utils
from musubi_tuner.flux_2 import flux2_utils, flux2_models
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents(batch: List[ItemInfo]) -> torch.Tensor:
    contents = []
    for item in batch:
        content = torch.from_numpy(item.content)
        if content.shape[-1] == 4:
            content = content[..., :3]
        contents.append(content)

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, C, H, W
    contents = contents.float() / 127.5 - 1.0
    return contents


def encode_and_save_batch(vae: flux2_models.AutoEncoder, batch: List[ItemInfo]):
    contents = preprocess_contents(batch)

    h, w = contents.shape[2], contents.shape[3]
    if h < 16 or w < 16:
        item = batch[0]
        raise ValueError(f"Image size too small: {item.item_key}, size: {item.original_size}")

    with torch.no_grad():
        contents = contents.to(vae.device, dtype=vae.dtype)
        latents = vae.encode(contents)  # [B, 128, H/16, W/16] - already patchified and BN-normalized by VAE

    for b, item in enumerate(batch):
        latent = latents[b]  # [C, H, W]
        logger.debug(f"Saving cache for {item.item_key}. Latent shape: {latent.shape}")
        save_latent_cache_ernie_image(item_info=item, latent=latent)


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_ERNIE_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return

    assert args.vae is not None, "VAE checkpoint is required (--vae)"

    logger.info(f"Loading VAE from {args.vae}")
    vae = flux2_utils.load_ae(args.vae, dtype=torch.bfloat16, device=device, disable_mmap=True)
    vae.eval()
    logger.info(f"Loaded VAE, dtype: {vae.dtype}")

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args)
    logger.info("Done!")


if __name__ == "__main__":
    main()
