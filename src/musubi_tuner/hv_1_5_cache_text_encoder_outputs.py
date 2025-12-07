import argparse
from typing import Optional

import torch
import accelerate
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Stack

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ItemInfo,
    save_text_encoder_output_cache_hunyuan_video_1_5,
)

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.hunyuan_video_1_5 import hunyuan_video_1_5_text_encoder
from musubi_tuner.qwen_image import qwen_image_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer_vlm: Qwen2Tokenizer,
    text_encoder_vlm: Qwen2_5_VLForConditionalGeneration,
    tokenizer_byt5: T5Tokenizer,
    text_encoder_byt5: T5Stack,
    batch: list[ItemInfo],
    device: torch.device,
    accelerator: Optional[accelerate.Accelerator],
):
    prompts = [item.caption for item in batch]

    # encode prompt with Qwen2.5-VL
    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                embed_vlm, mask_vlm = hunyuan_video_1_5_text_encoder.get_qwen_prompt_embeds(
                    tokenizer_vlm, text_encoder_vlm, prompts
                )
                if embed_vlm.dtype == torch.float8_e4m3fn:
                    embed_vlm = embed_vlm.to(torch.bfloat16)
        else:
            embed_vlm, mask_vlm = hunyuan_video_1_5_text_encoder.get_qwen_prompt_embeds(tokenizer_vlm, text_encoder_vlm, prompts)

    # encode prompt with BYT5 (for each prompt in batch)
    embed_byt5_list = []
    mask_byt5_list = []
    with torch.no_grad():
        for prompt in prompts:
            embed_byt5, mask_byt5 = hunyuan_video_1_5_text_encoder.get_glyph_prompt_embeds(
                tokenizer_byt5, text_encoder_byt5, prompt
            )
            embed_byt5_list.append(embed_byt5[0])  # remove batch dim
            mask_byt5_list.append(mask_byt5[0])  # remove batch dim

    # save prompt cache
    for i, item in enumerate(batch):
        embed_i = embed_vlm[i]
        mask_i = mask_vlm[i]

        # extract valid length for VLM embedding
        vlm_len = mask_i.to(dtype=torch.bool).sum().item()
        embed_i = embed_i[:vlm_len]

        # get BYT5 embedding for this item
        byt5_len = mask_byt5_list[i].to(dtype=torch.bool).sum().item()  # may be zero
        embed_byt5_i = embed_byt5_list[i][:byt5_len]

        save_text_encoder_output_cache_hunyuan_video_1_5(item, embed_i, embed_byt5_i)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = hunyuan_video_1_5_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HUNYUAN_VIDEO_1_5)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # define accelerator for fp8 inference
    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    accelerator = None
    if args.fp8_vl:
        accelerator = accelerate.Accelerator(mixed_precision="bf16")

    # prepare cache files and paths: all_cache_files_for_dataset = existing cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load Qwen2.5-VL
    logger.info(f"Loading Qwen2.5-VL: {args.text_encoder}")
    tokenizer_vlm, text_encoder_vlm = qwen_image_utils.load_qwen2_5_vl(
        ckpt_path=args.text_encoder, dtype=vl_dtype, device=device, disable_mmap=True
    )

    # Load BYT5
    logger.info(f"Loading BYT5: {args.byt5}")
    tokenizer_byt5, text_encoder_byt5 = hunyuan_video_1_5_text_encoder.load_byt5(
        ckpt_path=args.byt5, dtype=torch.float16, device=device, disable_mmap=True
    )

    # Encode with Qwen2.5-VL and BYT5
    logger.info("Encoding with Qwen2.5-VL and BYT5")

    def encode_for_text_encoders(batch: list[ItemInfo]):
        nonlocal tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5, device, accelerator
        encode_and_save_batch(tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5, batch, device, accelerator)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoders,
    )
    del text_encoder_vlm, text_encoder_byt5

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def hunyuan_video_1_5_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="Text Encoder (Qwen2.5-VL) checkpoint path")
    parser.add_argument("--byt5", type=str, default=None, required=True, help="BYT5 text encoder checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Qwen2.5-VL model")

    return parser


if __name__ == "__main__":
    main()
