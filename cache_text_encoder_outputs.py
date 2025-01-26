import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm
import accelerate

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.image_video_dataset import ItemInfo, save_text_encoder_output_cache
from hunyuan_model import text_encoder as text_encoder_module
from hunyuan_model.text_encoder import TextEncoder

import logging
from utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_prompt(text_encoder: TextEncoder, prompt: Union[str, list[str]]):
    data_type = "video"  # typically 'video'; can be changed if needed
    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)

    return prompt_outputs.hidden_state, prompt_outputs.attention_mask


def encode_and_save_batch(
    text_encoder: TextEncoder, batch: list[ItemInfo], is_llm: bool, accelerator: Optional[accelerate.Accelerator]
):
    # aggregate prompts
    prompts = [item.caption for item in batch]

    # encode
    if accelerator is not None:
        with accelerator.autocast():
            prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)
    else:
        prompt_embeds, prompt_mask = encode_prompt(text_encoder, prompts)

    # save
    for item, embed, mask in zip(batch, prompt_embeds, prompt_mask):
        save_text_encoder_output_cache(item, embed, mask, is_llm)


def main(args):
    # pick device
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # load config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args)

    # parse train + val dataset groups
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint["train_dataset_group"], training=False
    )
    val_dataset_group = config_utils.generate_dataset_group_by_blueprint(
        blueprint["val_dataset_group"], training=False
    )

    # combine
    all_datasets = train_dataset_group.datasets + val_dataset_group.datasets

    # optional: if you had a debug mode, you could do e.g.:
    # if args.debug_mode:
    #     debug_display_function(all_datasets, ...)
    #     return

    # optional: set up accelerator for fp8, if desired
    accelerator = None
    if args.fp8_llm:
        accelerator = accelerate.Accelerator(mixed_precision="fp16")

    # prepare batch encoding function
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)

    # each dataset can produce text-encoder batches
    all_cache_files_for_dataset = []
    all_cache_paths_for_dataset = []
    for dataset in all_datasets:
        cache_files = dataset.get_all_text_encoder_output_cache_files()
        cache_files = [os.path.normpath(f) for f in cache_files]
        all_cache_files_for_dataset.append(set(cache_files))
        all_cache_paths_for_dataset.append(set())

    # helper to encode with a given text encoder
    def encode_for_text_encoder(text_encoder: TextEncoder, is_llm: bool):
        for i, dataset in enumerate(all_datasets):
            logger.info(f"Encoding dataset [{i}] with text encoder {('LLM' if is_llm else 'CLIPL')}")
            existing_cache_files = all_cache_files_for_dataset[i]
            new_seen_cache = all_cache_paths_for_dataset[i]

            # stream batches
            for batch in tqdm(dataset.retrieve_text_encoder_output_cache_batches(num_workers)):
                # record these paths
                new_seen_cache.update(os.path.normpath(item.text_encoder_output_cache_path) for item in batch)

                # skip existing
                if args.skip_existing:
                    filtered_batch = [
                        item for item in batch
                        if os.path.normpath(item.text_encoder_output_cache_path) not in existing_cache_files
                    ]
                    if not filtered_batch:
                        continue
                    batch = filtered_batch

                # chunk into mini-batches for memory
                bs = args.batch_size or len(batch)
                for start_idx in range(0, len(batch), bs):
                    sub_batch = batch[start_idx : start_idx + bs]
                    encode_and_save_batch(text_encoder, sub_batch, is_llm, accelerator)

    # load + encode with text encoder 1
    text_encoder_dtype = torch.float16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)
    logger.info(f"Loading text encoder 1 from {args.text_encoder1}")
    text_encoder_1 = text_encoder_module.load_text_encoder_1(args.text_encoder1, device, args.fp8_llm, text_encoder_dtype)
    text_encoder_1.to(device=device)
    encode_for_text_encoder(text_encoder_1, is_llm=True)
    del text_encoder_1

    # load + encode with text encoder 2
    logger.info(f"Loading text encoder 2 from {args.text_encoder2}")
    text_encoder_2 = text_encoder_module.load_text_encoder_2(args.text_encoder2, device, text_encoder_dtype)
    text_encoder_2.to(device=device)
    encode_for_text_encoder(text_encoder_2, is_llm=False)
    del text_encoder_2

    # remove old cache files not in dataset
    for i, dataset in enumerate(all_datasets):
        existing_cache_files = all_cache_files_for_dataset[i]
        new_seen_cache = all_cache_paths_for_dataset[i]
        for cache_file in existing_cache_files:
            if cache_file not in new_seen_cache:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 path or dir")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 path or dir")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cpu). Default auto.")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="FP precision, default float16")
    parser.add_argument("--fp8_llm", action="store_true", help="Use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers. Default = CPU-1")
    parser.add_argument("--skip_existing", action="store_true", help="Skip items with existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="Keep old cache files not in the dataset")
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    main(args)