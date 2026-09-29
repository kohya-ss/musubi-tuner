"""Latent caching for Qwen-Image 2.1."""

import argparse
import logging

import torch

from musubi_tuner import cache_latents
from musubi_tuner.dataset.cache_io import save_latent_cache_qwen_image_21
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.qwen_image21 import utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae, batch: list[ItemInfo]) -> None:
    """Encode and save target and control image latents."""
    for item in batch:
        if isinstance(item.content, list):
            raise ValueError("Only one target image per item is supported")
        latent = utils.encode_image(vae, item.content)[0]
        references = [utils.encode_image(vae, image)[0] for image in item.control_content or []]
        save_latent_cache_qwen_image_21(item, latent, references, utils.reference_fingerprints(item.control_content))


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_latents.setup_parser_common()
    parser.add_argument("--vae_tiling", action="store_true", help="Enable spatial tiling for the Qwen-Image 2.1 VAE")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False
    datasets = utils.load_datasets(args)
    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return
    if not args.vae:
        parser.error("--vae is required (Qwen-Image 2.1 Diffusers VAE directory or safetensors)")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading VAE model from %s", args.vae)
    vae = utils.load_vae(args.vae, device, str_to_dtype(args.vae_dtype) if args.vae_dtype else torch.bfloat16, args.vae_tiling)
    logger.info("Encoding latents on %s", device)
    cache_latents.encode_datasets(datasets, lambda batch: encode_and_save_batch(vae, batch), args, supports_alpha=True)


if __name__ == "__main__":
    main()
