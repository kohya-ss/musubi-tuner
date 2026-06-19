import argparse
import logging
from typing import List

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4, ItemInfo, save_text_encoder_output_cache_ideogram4
from musubi_tuner.ideogram4 import ideogram4_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _resolve_cache_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if name == "fp8_e4m3fn":
        return ideogram4_utils.fp8_cache_dtype()
    raise ValueError(f"Unsupported text cache dtype: {name}")


def encode_and_save_batch(
    tokenizer,
    text_encoder,
    batch: List[ItemInfo],
    device: torch.device,
    cache_dtype_name: str,
    validate_caption_structure: bool,
    warn_only: bool,
):
    cache_dtype = _resolve_cache_dtype(cache_dtype_name)
    for item in batch:
        if validate_caption_structure:
            ideogram4_utils.validate_prompt(item.caption, warn_only=warn_only)
        with torch.no_grad():
            features = ideogram4_utils.encode_prompt_to_features(tokenizer, text_encoder, item.caption, device)
        mb_per_image, gb_per_1k = ideogram4_utils.dtype_cache_cost(features.shape[0], cache_dtype)
        logger.info(
            f"Saving Ideogram 4 text cache for {item.item_key}: tokens={features.shape[0]}, dtype={cache_dtype_name}, "
            f"approx {mb_per_image:.1f} MB/image, {gb_per_1k:.1f} GB/1k images"
        )
        save_text_encoder_output_cache_ideogram4(item, features.to(cache_dtype))


def setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL BF16 text encoder safetensors path")
    parser.add_argument(
        "--text_cache_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp8_e4m3fn", "float32"],
        help="dtype for cached Ideogram 4 text features",
    )
    parser.add_argument(
        "--validate_caption_structure",
        action="store_true",
        help="validate official structured JSON captions before caching; ordinary prompt captions are accepted by default",
    )
    parser.add_argument(
        "--warn_on_caption_issues",
        action="store_true",
        help="warn instead of failing on structured-caption issues when --validate_caption_structure is enabled",
    )
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = setup_parser(parser)
    args = parser.parse_args()

    device = args.device if hasattr(args, "device") and args.device else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    text_encoder_dtype = torch.bfloat16

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_IDEOGRAM4)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    tokenizer = ideogram4_utils.load_ideogram4_tokenizer()
    text_encoder_model = ideogram4_utils.load_ideogram4_text_encoder(
        args.text_encoder,
        device=device,
        dtype=text_encoder_dtype,
        disable_mmap=args.disable_numpy_memmap if hasattr(args, "disable_numpy_memmap") else False,
    )

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(
            tokenizer,
            text_encoder_model,
            batch,
            device,
            args.text_cache_dtype,
            args.validate_caption_structure,
            args.warn_on_caption_issues,
        )

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode,
        requires_content=False,
    )
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
