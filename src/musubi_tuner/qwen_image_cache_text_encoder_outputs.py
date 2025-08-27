import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
import accelerate
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_QWEN_IMAGE, ItemInfo, save_text_encoder_output_cache_qwen_image

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging

from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer: Qwen2Tokenizer,
    text_encoder: Qwen2_5_VLForConditionalGeneration,
    vl_processor: Optional[Qwen2VLProcessor],
    batch: list[ItemInfo],
    device: torch.device,
    accelerator: Optional[accelerate.Accelerator],
):
    is_edit = vl_processor is not None
    prompts = [item.caption for item in batch]
    # print(prompts)

    # prepare images
    if is_edit:
        images = []
        for item in batch:
            assert (
                item.control_content is not None and len(item.control_content) > 0
            ), f"Item {item.item_key} must have control content for Qwen-Image-Edit"
            control_content = item.control_content[0]  # np.ndarray, 0-255
            control_content = control_content[..., :3]  # ensure RGB, remove alpha if present
            images.append(control_content)  # vl_processor accepts PIL.Image and np.ndarray

    # encode prompt
    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                if not is_edit:
                    embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, prompts)
                else:
                    embed, mask = qwen_image_utils.get_qwen_prompt_embeds_with_image(vl_processor, text_encoder, prompts, images)
                if embed.dtype == torch.float8_e4m3fn:  # T5 returns bf16, but QwenVL-2.5 returns fp8
                    embed = embed.to(torch.bfloat16)

        else:
            if not is_edit:
                embed, mask = qwen_image_utils.get_qwen_prompt_embeds(tokenizer, text_encoder, prompts)
            else:
                embed, mask = qwen_image_utils.get_qwen_prompt_embeds_with_image(vl_processor, text_encoder, prompts, images)

    # save prompt cache
    for item, (embed, mask) in zip(batch, zip(embed, mask)):
        txt_len = mask.to(dtype=torch.bool).sum().item()  # length of the text in the batch
        embed = embed[:txt_len]
        save_text_encoder_output_cache_qwen_image(item, embed)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()
    is_edit = args.edit

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_QWEN_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # define accelerator for fp8 inference
    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    accelerator = None
    if args.fp8_vl:
        accelerator = accelerate.Accelerator(mixed_precision="bf16")

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load Qwen2.5-VL
    logger.info(f"Loading Qwen2.5-VL: {args.text_encoder}")
    tokenizer, text_encoder = qwen_image_utils.load_qwen2_5_vl(
        ckpt_path=args.text_encoder, dtype=vl_dtype, device=device, disable_mmap=True
    )

    # Load Qwen2VLProcessor
    if is_edit:
        logger.info("Loading Qwen2.5-VL Processor for Edit")
        vl_processor = qwen_image_utils.load_vl_processor()
    else:
        vl_processor = None

    # Encode with Qwen2.5-VL
    logger.info("Encoding with Qwen2.5-VL")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal tokenizer, text_encoder, vl_processor, device, accelerator
        encode_and_save_batch(tokenizer, text_encoder, vl_processor, batch, device, accelerator)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
        requires_content=is_edit,
    )
    del text_encoder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="Text Encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument("--edit", action="store_true", help="cache Text Encoder outputs for Qwen-Image-Edit")
    return parser


if __name__ == "__main__":
    main()
