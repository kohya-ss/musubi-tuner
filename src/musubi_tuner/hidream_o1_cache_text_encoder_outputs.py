import argparse
import logging

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HIDREAM_O1,
    ItemInfo,
    save_text_encoder_output_cache_hidream_o1,
)
from musubi_tuner.hidream_o1 import hidream_o1_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(processor, embedding_weight: torch.Tensor | None, device: torch.device, batch: list[ItemInfo]):
    for item in batch:
        pixel_values = None
        image_grid_thw = None
        if item.control_content is not None:
            input_ids, pixel_values, image_grid_thw = hidream_o1_utils.build_i2i_input_tensors(
                item.caption, item.control_content, processor
            )
        else:
            input_ids = hidream_o1_utils.build_t2i_input_ids(item.caption, processor)
        input_embeds = None
        if embedding_weight is not None:
            input_embeds = hidream_o1_utils.build_text_input_embeds(input_ids, embedding_weight, device).cpu()
        save_text_encoder_output_cache_hidream_o1(
            item, input_ids, input_embeds, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )


def hidream_o1_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "dev"], help="HiDream-O1 model variant")
    parser.add_argument("--dit", type=str, default=None, help="HiDream-O1 single checkpoint for optional cached text embeddings")
    parser.add_argument("--fp8_te", action="store_true", help="Cache HiDream-O1 text token embeddings in fp8 precision")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = hidream_o1_setup_parser(parser)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device {device} for tokenizer/processor setup")

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HIDREAM_O1)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading HiDream-O1 processor for model_type={args.model_type}")
    processor = hidream_o1_utils.load_processor(model_type=args.model_type)
    if args.fp8_te and args.dit is None:
        raise ValueError("--fp8_te requires --dit so text embeddings can be cached from the single checkpoint.")

    embedding_weight = None
    if args.dit is not None:
        te_dtype = torch.float8_e4m3fn if args.fp8_te else torch.bfloat16
        logger.info(f"Loading HiDream-O1 text embedding weight from {args.dit}, dtype={te_dtype}")
        embedding_weight = hidream_o1_utils.load_text_embedding_weight(args.dit, te_dtype, device)

    def encode_for_text_encoder(batch: list[ItemInfo], embedding_weight: torch.Tensor | None = embedding_weight):
        encode_and_save_batch(processor, embedding_weight, device, batch)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=any(getattr(dataset, "has_control", False) for dataset in datasets),
    )

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )

    if embedding_weight is not None:
        del embedding_weight
    logger.info("Done!")


if __name__ == "__main__":
    main()
