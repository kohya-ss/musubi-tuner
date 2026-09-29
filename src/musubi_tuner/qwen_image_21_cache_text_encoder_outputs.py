"""Text encoder output caching for Qwen-Image 2.1."""

import argparse
import logging

import torch

from musubi_tuner import cache_text_encoder_outputs
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_qwen_image_21
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.qwen_image21 import utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(processor, encoder, batch: list[ItemInfo]) -> None:
    """Encode and save text features and reference image metadata."""
    for item_index, item in enumerate(batch):
        control_shapes = [getattr(image, "shape", getattr(image, "size", None)) for image in item.control_content or []]
        logger.info(
            "Item %d: %s, prompt: %s, control images: %s",
            item_index,
            item.item_key,
            item.caption,
            control_shapes,
        )
        embed, slots, grids = utils.encode_prompt(processor, encoder, item.caption, item.control_content)
        save_text_encoder_output_cache_qwen_image_21(item, embed, slots, grids, utils.reference_fingerprints(item.control_content))


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser.add_argument(
        "--text_encoder", required=True, help="Local Qwen3-VL-8B text_encoder directory or BF16/FP32 safetensors file"
    )
    parser.add_argument(
        "--text_encoder_dtype", default="bfloat16", choices=["bfloat16", "float32"], help="Text encoder compute dtype"
    )
    parser.add_argument("--fp8_vl", action="store_true", help="Store Qwen3-VL language block linears in FP8")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    datasets = utils.load_datasets(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading Qwen3-VL text encoder: %s", args.text_encoder)
    processor, encoder = utils.load_text_encoder(
        args.text_encoder, device=device, dtype=str_to_dtype(args.text_encoder_dtype), fp8_vl=args.fp8_vl
    )
    logger.info("Encoding with Qwen3-VL on %s", device)
    existing, expected = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    def encode(batch):
        encode_and_save_batch(processor, encoder, batch)

    # Use the same image resize and crop as latent caching.
    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers or 1,
        args.skip_existing,
        args.batch_size,
        datasets,
        existing,
        expected,
        encode,
        requires_content=True,
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, existing, expected, args.keep_cache)


if __name__ == "__main__":
    main()
