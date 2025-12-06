"""
Cache text encoder outputs for Z-Image architecture.

This script encodes text prompts using Z-Image's Qwen3 text encoder and caches
the embeddings for faster training. Z-Image uses only a single text encoder (Qwen3),
making this simpler than other architectures that use multiple encoders.
"""

import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE, ItemInfo, save_text_encoder_output_cache_z_image
from musubi_tuner.zimage import zimage_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo], device: torch.device):
    """
    Encode a batch of prompts and save their text encoder outputs.

    Args:
        tokenizer: Qwen3 tokenizer
        text_encoder: Qwen3 text encoder model
        batch: List of ItemInfo containing captions to encode
        device: Device to use for encoding
    """
    prompts = [item.caption for item in batch]

    # Encode prompts using Qwen3
    # get_text_embeds returns (prompt_embeds, prompt_masks)
    # prompt_embeds: (B, seq_len, hidden_size)
    # prompt_masks: (B, seq_len) boolean mask
    prompt_embeds, prompt_masks = zimage_utils.get_text_embeds(tokenizer, text_encoder, prompts)

    # Move to CPU for saving
    prompt_embeds = prompt_embeds.cpu()

    # Save each item's embedding
    # We save variable-length embeddings (trimmed to actual text length) to save space
    for item, embed, mask in zip(batch, prompt_embeds, prompt_masks):
        # Trim to actual text length based on attention mask
        actual_length = int(mask.sum().item())
        embed_trimmed = embed[:actual_length]  # (actual_length, hidden_size)

        save_text_encoder_output_cache_z_image(item, embed_trimmed)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = zimage_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # Prepare cache files and paths
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Determine dtype for text encoder
    if args.fp8_llm:
        te_dtype = torch.float8_e4m3fn
        logger.info("Using fp8 for Qwen3 text encoder")
    else:
        te_dtype = torch.bfloat16
        logger.info("Using bfloat16 for Qwen3 text encoder")

    # Load Qwen3 tokenizer and text encoder
    logger.info(f"Loading Qwen3 text encoder from {args.text_encoder}")
    tokenizer, text_encoder = zimage_utils.load_qwen3(args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True)
    text_encoder.eval()

    # Encode with Qwen3 text encoder
    logger.info("Encoding prompts with Qwen3 text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder
        encode_and_save_batch(tokenizer, text_encoder, batch, device)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )

    # Clean up
    del tokenizer, text_encoder

    # Remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )

    logger.info("Done!")


def zimage_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Z-Image specific arguments to the parser."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3 text encoder checkpoint path or directory",
    )
    parser.add_argument(
        "--fp8_llm",
        action="store_true",
        help="Use fp8 precision for Qwen3 text encoder (reduces memory usage)",
    )
    return parser


if __name__ == "__main__":
    main()
