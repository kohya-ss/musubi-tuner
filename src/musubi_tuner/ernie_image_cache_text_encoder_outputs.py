"""
Cache text encoder outputs for ERNIE-Image architecture.

Encodes text prompts using Mistral3 text encoder and caches the embeddings
(hidden_states[-2]) for faster training.
"""

import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.architectures import ARCHITECTURE_ERNIE_IMAGE
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_ernie_image
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.ernie_image import ernie_image_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(tokenizer, text_encoder, batch: list[ItemInfo], device: torch.device):
    prompts = [item.caption for item in batch]

    for item, prompt in zip(batch, prompts):
        text_hiddens = ernie_image_utils.encode_text(tokenizer, text_encoder, prompt)
        embed = text_hiddens[0].cpu()  # [T, D]
        save_text_encoder_output_cache_ernie_image(item, embed)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = ernie_image_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_ERNIE_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    te_dtype = torch.bfloat16
    logger.info(f"Loading text encoder from {args.text_encoder}")
    tokenizer, text_encoder = ernie_image_utils.load_text_encoder(
        args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True, tokenizer_id=args.tokenizer
    )
    text_encoder.eval()

    logger.info("Encoding prompts with Mistral3 text encoder")

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

    del tokenizer, text_encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )

    logger.info("Done!")


def ernie_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Mistral3 text encoder .safetensors path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path (defaults to 'baidu/ERNIE-Image')")
    return parser


if __name__ == "__main__":
    main()
