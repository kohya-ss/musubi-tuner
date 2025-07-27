import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel, CLIPTokenizer, T5Tokenizer
import accelerate

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_FLUX_KONTEXT,
    ItemInfo,
    save_text_encoder_output_cache_flux_kontext,
)

from musubi_tuner.flux import flux_models
from musubi_tuner.flux import flux_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging

from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    tokenizer1: T5Tokenizer,
    text_encoder1: T5EncoderModel,
    tokenizer2: CLIPTokenizer,
    text_encoder2: CLIPTextModel,
    batch: list[ItemInfo],
    device: torch.device,
):
    prompts = [item.caption for item in batch]
    # print(prompts)

    # encode prompt
    t5_tokens = tokenizer1(
        prompts,
        max_length=flux_models.T5XXL_MAX_LENGTH,
        padding="max_length",
        return_length=False,
        return_overflowing_tokens=False,
        truncation=True,
        return_tensors="pt",
    )["input_ids"]
    l_tokens = tokenizer2(prompts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]

    with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
        t5_vec = text_encoder1(input_ids=t5_tokens.to(text_encoder1.device), attention_mask=None, output_hidden_states=False)[
            "last_hidden_state"
        ]
        assert torch.isnan(t5_vec).any() == False, "T5 vector contains NaN values"
        t5_vec = t5_vec.cpu()

    with torch.autocast(device_type=device.type, dtype=text_encoder2.dtype), torch.no_grad():
        clip_l_pooler = text_encoder2(l_tokens.to(text_encoder2.device))["pooler_output"]
        clip_l_pooler = clip_l_pooler.cpu()

    # save prompt cache
    for item, t5_vec, clip_ctx in zip(batch, t5_vec, clip_l_pooler):
        save_text_encoder_output_cache_flux_kontext(item, t5_vec, clip_ctx)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = flux_kontext_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_KONTEXT)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load T5 and CLIP text encoders
    t5_dtype = torch.float8e4m3fn if args.fp8_t5 else torch.bfloat16
    tokenizer1, text_encoder1 = flux_utils.load_t5xxl(args.text_encoder1, dtype=t5_dtype, device=device, disable_mmap=True)
    tokenizer2, text_encoder2 = flux_utils.load_clip_l(args.text_encoder2, dtype=torch.bfloat16, device=device, disable_mmap=True)

    # Encode with T5 and CLIP text encoders
    logger.info(f"Encoding with T5 and CLIP text encoders")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        encode_and_save_batch(tokenizer1, text_encoder1, tokenizer2, text_encoder2, batch, device)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_encoder1
    del text_encoder2

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def flux_kontext_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder1", type=str, default=None, required=True, help="text encoder (T5XXL) checkpoint path")
    parser.add_argument("--text_encoder2", type=str, default=None, required=True, help="text encoder 2 (CLIP-L) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    return parser


if __name__ == "__main__":
    main()
