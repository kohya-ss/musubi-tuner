import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_HUNYUAN_VIDEO,
    BaseDataset,
    ItemInfo,
    save_text_encoder_output_cache,
)
from musubi_tuner.hunyuan_model import text_encoder as text_encoder_module
from musubi_tuner.hunyuan_model.text_encoder import TextEncoder

import logging

from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_prompt(text_encoder: TextEncoder, prompt: Union[str, list[str]]):
    data_type = "video"  # video only, image is not supported
    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)

    return prompt_outputs.hidden_state, prompt_outputs.attention_mask


def encode_and_save_batch(
    text_encoder: TextEncoder, batch: list[ItemInfo], is_llm: bool, accelerator: Optional[accelerate.Accelerator]
):
    prompts = [item.caption for item in batch]
    # print(prompts)

    # encode prompt
    if accelerator is not None:
        with accelerator.autocast():
            prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)
    else:
        prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)

    # # convert to fp16 if needed
    # if prompt_embeds.dtype == torch.float32 and text_encoder.dtype != torch.float32:
    #     prompt_embeds = prompt_embeds.to(text_encoder.dtype)

    # save prompt cache
    for item, embed, mask in zip(batch, prompt_embeds, prompt_mask):
        save_text_encoder_output_cache(item, embed, mask, is_llm)


def prepare_cache_files_and_paths(datasets: list[BaseDataset]):
    all_cache_files_for_dataset = []  # exisiting cache files
    all_cache_paths_for_dataset = []  # all cache paths in the dataset
    for dataset in datasets:
        all_cache_files = [os.path.normpath(file) for file in dataset.get_all_text_encoder_output_cache_files()]
        all_cache_files = set(all_cache_files)
        all_cache_files_for_dataset.append(all_cache_files)

        all_cache_paths_for_dataset.append(set())
    return all_cache_files_for_dataset, all_cache_paths_for_dataset


def process_text_encoder_batches(
    num_workers: Optional[int],
    skip_existing: bool,
    batch_size: int,
    datasets: list[BaseDataset],
    all_cache_files_for_dataset: list[set],
    all_cache_paths_for_dataset: list[set],
    encode: callable,
    requires_content: Optional[bool] = False,
):
    """
    Architecture independent processing of text encoder batches.
    """

    num_workers = num_workers if num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]

        if not requires_content:
            batches = dataset.retrieve_text_encoder_output_cache_batches(num_workers)  # return captions only
        else:
            batches = dataset.retrieve_latent_cache_batches(num_workers)  # return captions and images/videos

        for batch in tqdm(batches):
            # update cache files (it's ok if we update it multiple times)
            if requires_content:
                batch = batch[1]  # batch is (key, items), so use items
            all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])

            # skip existing cache files
            if skip_existing:
                filtered_batch = [
                    item for item in batch if not os.path.normpath(item.text_encoder_output_cache_path) in all_cache_files
                ]
                # print(f"Filtered {len(batch) - len(filtered_batch)} existing cache files")
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = batch_size if batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])


def post_process_cache_files(
    datasets: list[BaseDataset], all_cache_files_for_dataset: list[set], all_cache_paths_for_dataset: list[set], keep_cache: bool
):
    for i, dataset in enumerate(datasets):
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for cache_file in all_cache_files:
            if cache_file not in all_cache_paths:
                if keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HUNYUAN_VIDEO)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # define accelerator for fp8 inference
    accelerator = None
    if args.fp8_llm:
        accelerator = accelerate.Accelerator(mixed_precision="fp16")

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = prepare_cache_files_and_paths(datasets)

    # Load Text Encoder 1
    text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)
    logger.info(f"loading text encoder 1: {args.text_encoder1}")
    text_encoder_1 = text_encoder_module.load_text_encoder_1(args.text_encoder1, device, args.fp8_llm, text_encoder_dtype)
    text_encoder_1.to(device=device)

    # Encode with Text Encoder 1 (LLM)
    logger.info("Encoding with Text Encoder 1")

    def encode_for_text_encoder_1(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder_1, batch, is_llm=True, accelerator=accelerator)

    process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder_1,
    )
    del text_encoder_1

    # Load Text Encoder 2
    logger.info(f"loading text encoder 2: {args.text_encoder2}")
    text_encoder_2 = text_encoder_module.load_text_encoder_2(args.text_encoder2, device, text_encoder_dtype)
    text_encoder_2.to(device=device)

    # Encode with Text Encoder 2
    logger.info("Encoding with Text Encoder 2")

    def encode_for_text_encoder_2(batch: list[ItemInfo]):
        encode_and_save_batch(text_encoder_2, batch, is_llm=False, accelerator=None)

    process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder_2,
    )
    del text_encoder_2

    # remove cache files not in dataset
    post_process_cache_files(datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache)


def setup_parser_common():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    return parser


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    return parser


if __name__ == "__main__":
    main()
