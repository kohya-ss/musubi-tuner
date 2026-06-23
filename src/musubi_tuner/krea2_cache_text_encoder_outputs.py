import argparse
import logging
from typing import Optional

import torch
import accelerate

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_KREA2,
    ItemInfo,
    save_text_encoder_output_cache_krea2,
)

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.krea2 import krea2_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    processor,
    text_encoder,
    batch: list[ItemInfo],
    device: torch.device,
    accelerator: Optional[accelerate.Accelerator],
):
    prompts = [item.caption for item in batch]

    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                hidden, mask = krea2_utils.get_krea2_prompt_embeds(
                    processor, text_encoder, prompts, device=device, dtype=text_encoder.dtype
                )
        else:
            hidden, mask = krea2_utils.get_krea2_prompt_embeds(
                processor, text_encoder, prompts, device=device, dtype=text_encoder.dtype
            )

    for item, (hidden_i, mask_i) in zip(batch, zip(hidden, mask)):
        txt_len = mask_i.to(dtype=torch.bool).sum().item()
        hidden_i = hidden_i[:txt_len]
        save_text_encoder_output_cache_krea2(item, hidden_i)


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="Text Encoder (Qwen3-VL 4B) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    return parser


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    architecture = ARCHITECTURE_KREA2

    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    accelerator = None
    if args.fp8_vl:
        accelerator = accelerate.Accelerator(mixed_precision="bf16")

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    logger.info(f"Loading Qwen3-VL text encoder: {args.text_encoder}")
    processor, text_encoder = krea2_utils.load_text_encoder(args.text_encoder, dtype=vl_dtype, device=device)

    logger.info("Encoding with Qwen3-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal text_encoder
        encode_and_save_batch(processor, text_encoder, batch, device, accelerator)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=False,
    )
    del text_encoder

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


if __name__ == "__main__":
    main()
