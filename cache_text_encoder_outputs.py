import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache
from hunyuan_model import text_encoder as text_encoder_module
from hunyuan_model.text_encoder import TextEncoder

import logging

from utils.model_utils import str_to_dtype

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


def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint["train_dataset_group"],
        training=False
    )
    val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint["val_dataset_group"],
        training=False
    )
    datasets = train_dataset_group.datasets + val_dataset_group.datasets

    # define accelerator for fp8 inference
    accelerator = None
    if args.fp8_llm:
        accelerator = accelerate.Accelerator(mixed_precision="fp16")

    # define encode function
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)

    all_cache_files_for_dataset = []  # exisiting cache files
    all_cache_paths_for_dataset = []  # all cache paths in the dataset
    for dataset in datasets:
        all_cache_files = [os.path.normpath(file) for file in dataset.get_all_text_encoder_output_cache_files()]
        all_cache_files = set(all_cache_files)
        all_cache_files_for_dataset.append(all_cache_files)

        all_cache_paths_for_dataset.append(set())

    def encode_for_text_encoder(text_encoder: TextEncoder, is_llm: bool):
        for i, dataset in enumerate(datasets):
            logger.info(f"Encoding dataset [{i}]")
            all_cache_files = all_cache_files_for_dataset[i]
            all_cache_paths = all_cache_paths_for_dataset[i]
            for batch in tqdm(dataset.retrieve_text_encoder_output_cache_batches(num_workers)):
                # update cache files (it's ok if we update it multiple times)
                all_cache_paths.update([os.path.normpath(item.text_encoder_output_cache_path) for item in batch])

                # skip existing cache files
                if args.skip_existing:
                    filtered_batch = [
                        item for item in batch if not os.path.normpath(item.text_encoder_output_cache_path) in all_cache_files
                    ]
                    # print(f"Filtered {len(batch) - len(filtered_batch)} existing cache files")
                    if len(filtered_batch) == 0:
                        continue
                    batch = filtered_batch

                bs = args.batch_size if args.batch_size is not None else len(batch)
                for i in range(0, len(batch), bs):
                    encode_and_save_batch(text_encoder, batch[i : i + bs], is_llm, accelerator)

    # Load Text Encoder 1
    text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)
    logger.info(f"loading text encoder 1: {args.text_encoder1}")
    text_encoder_1 = text_encoder_module.load_text_encoder_1(args.text_encoder1, device, args.fp8_llm, text_encoder_dtype)
    text_encoder_1.to(device=device)

    # Encode with Text Encoder 1
    logger.info("Encoding with Text Encoder 1")
    encode_for_text_encoder(text_encoder_1, is_llm=True)
    del text_encoder_1

    # Load Text Encoder 2
    logger.info(f"loading text encoder 2: {args.text_encoder2}")
    text_encoder_2 = text_encoder_module.load_text_encoder_2(args.text_encoder2, device, text_encoder_dtype)
    text_encoder_2.to(device=device)

    # Encode with Text Encoder 2
    logger.info("Encoding with Text Encoder 2")
    encode_for_text_encoder(text_encoder_2, is_llm=False)
    del text_encoder_2

    # remove cache files not in dataset
    for i, dataset in enumerate(datasets):
        all_cache_files = all_cache_files_for_dataset[i]
        all_cache_paths = all_cache_paths_for_dataset[i]
        for cache_file in all_cache_files:
            if cache_file not in all_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="data type for Text Encoder, default is float16")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)