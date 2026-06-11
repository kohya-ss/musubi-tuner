import argparse
import logging
import os

import torch

from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit", type=str, required=True, help="conditional Ideogram 4 DiT safetensors path")
    parser.add_argument("--unconditional_dit", type=str, required=True, help="unconditional Ideogram 4 DiT safetensors path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL BF16 text encoder safetensors path")
    parser.add_argument("--vae", type=str, required=True, help="Flux2 VAE safetensors path")
    parser.add_argument("--prompt", type=str, required=True, help="prompt")
    parser.add_argument("--negative_prompt", type=str, default=None, help="ignored for Ideogram 4 asymmetric CFG v1")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size as height width")
    parser.add_argument("--sampler_preset", type=str, default="V4_DEFAULT_20", choices=sorted(PRESETS.keys()))
    parser.add_argument("--initial_sigma", type=float, default=1.004, help="override the first denoising sigma")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_path", type=str, required=True, help="output image path or directory")
    parser.add_argument("--device", type=str, default=None, help="device, default cuda if available")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="compute dtype for models")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable numpy memmap while loading safetensors")
    parser.add_argument("--warn_on_caption_issues", action="store_true", help="warn instead of failing on caption verifier issues")
    return parser


def _save_images(images, save_path: str):
    root, ext = os.path.splitext(save_path)
    if len(images) == 1 and ext:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        images[0].save(save_path)
        logger.info(f"Saved image: {save_path}")
        return

    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(save_path, f"ideogram4_{i:02d}.png")
        image.save(path)
        logger.info(f"Saved image: {path}")


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.negative_prompt is not None:
        logger.warning("Ideogram 4 v1 uses official asymmetric CFG; --negative_prompt is ignored.")

    height, width = args.image_size
    device = torch.device(args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype)

    ideogram4_utils.validate_prompt(args.prompt, warn_only=args.warn_on_caption_issues)

    logger.info("Loading Ideogram 4 tokenizer and text encoder")
    tokenizer = ideogram4_utils.load_ideogram4_tokenizer()
    text_encoder = ideogram4_utils.load_ideogram4_text_encoder(
        args.text_encoder,
        device=device,
        dtype=dtype,
        disable_mmap=args.disable_numpy_memmap,
    )
    text_features = [ideogram4_utils.encode_prompt_to_features(tokenizer, text_encoder, args.prompt, device)]
    del tokenizer, text_encoder
    clean_memory_on_device(device)

    logger.info("Loading conditional DiT")
    conditional_transformer = ideogram4_utils.load_ideogram4_transformer(
        args.dit,
        device=device,
        dtype=dtype,
        expected_model_type=ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE,
        disable_mmap=args.disable_numpy_memmap,
    )
    logger.info("Loading unconditional DiT")
    unconditional_transformer = ideogram4_utils.load_ideogram4_transformer(
        args.unconditional_dit,
        device=device,
        dtype=dtype,
        expected_model_type=ideogram4_utils.IDEOGRAM4_UNCOND_MODEL_TYPE,
        disable_mmap=args.disable_numpy_memmap,
    )
    logger.info("Loading VAE")
    autoencoder = ideogram4_utils.load_ideogram4_autoencoder(
        args.vae,
        device=device,
        dtype=dtype,
        disable_mmap=args.disable_numpy_memmap,
    )

    images = ideogram4_utils.generate_images(
        conditional_transformer=conditional_transformer,
        unconditional_transformer=unconditional_transformer,
        autoencoder=autoencoder,
        text_features=text_features,
        height=height,
        width=width,
        sampler_preset=args.sampler_preset,
        device=device,
        seed=args.seed,
        initial_sigma=args.initial_sigma,
    )
    _save_images(images, args.save_path)


if __name__ == "__main__":
    main()
