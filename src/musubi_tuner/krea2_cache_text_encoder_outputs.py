"""Cache text encoder (Qwen3-VL-4B) outputs for Krea 2 (K2) training.

K2's text encoder returns a stack of *selected* Qwen3-VL hidden-state layers
(shape (B, seq, 12, 2560)) plus an attention mask. The layerwise fusion
(TextFusionTransformer) is trainable and lives inside the DiT, so we cache the raw
selected-layer stack. Padding tokens are dropped per item (varlen) — K2 gives text
tokens zero RoPE position and masks padding in attention, so this is lossless for
the image outputs.
"""

import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache_krea2
from musubi_tuner.dataset.architectures import ARCHITECTURE_KREA2
from musubi_tuner.krea2 import krea2_utils

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(encoder, batch: list[ItemInfo]):
    prompts = [item.caption for item in batch]
    for i, item in enumerate(batch):
        print(f"Item {i}: {item.item_key}, prompt: {item.caption}")

    hiddens, mask = krea2_utils.get_krea2_prompt_embeds(encoder, prompts)  # (B, seq, L, D), (B, seq)

    # Save per item, dropping padding tokens (varlen).
    for item, hidden_i, mask_i in zip(batch, hiddens, mask):
        valid = mask_i.bool()
        embed_i = hidden_i[valid]  # (valid_len, L, D)
        save_text_encoder_output_cache_krea2(item, embed_i)


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3-VL-4B text encoder safetensors path (official or ComfyUI key layout)",
    )
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for the text encoder, default is bfloat16")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    te_dtype = torch.bfloat16
    if args.text_encoder_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype

        te_dtype = str_to_dtype(args.text_encoder_dtype)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_KREA2)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=te_dtype, device=device)

    logger.info("Encoding with Qwen3-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal encoder
        encode_and_save_batch(encoder, batch)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
