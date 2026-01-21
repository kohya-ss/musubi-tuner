import argparse

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_FLUX_2,
    ItemInfo,
    save_text_encoder_output_cache_flux_2,
)

from musubi_tuner.flux_2 import flux2_utils
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    text_embedder: torch.nn.Module,
    guidance_distilled: bool,
    batch: list[ItemInfo],
    device: torch.device,
):
    prompts = [item.caption for item in batch]
    with torch.autocast(device_type=device.type, dtype=text_embedder.dtype), torch.no_grad():
        ctx_vec = text_embedder(prompts)
        ## TODO train with guidance ?
        # if guidance_distilled:
        #     ctx_vec = text_embedder(prompts)
        # else:
        #     if len(prompts) > 1:
        #         raise NotImplementedError("Only works with batch size 1")
        #     ctx_empty = text_embedder([""]).to(torch.bfloat16)
        #     ctx_prompt = text_embedder(prompts).to(torch.bfloat16)
        #     ctx_vec = torch.cat([ctx_empty, ctx_prompt], dim=0)
        ctx_vec = ctx_vec.cpu()  # [1, 512, 15360]

    # save prompt cache
    for item, _ctx_vec in zip(batch, ctx_vec):
        save_text_encoder_output_cache_flux_2(item, _ctx_vec)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = flux_2_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_2)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load Mistral 3 text encoder
    m3_dtype = torch.float8e4m3fn if args.fp8_m3 else torch.bfloat16
    text_embedder = flux2_utils.load_textembedder(
        args.model_version,
        args.text_encoder,
        dtype=m3_dtype,
        device=device,
        disable_mmap=True,
    )

    # Encode with Mistral 3 text encoder
    logger.info("Encoding with Mistral 3 text encoder")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal text_embedder
        encode_and_save_batch(
            text_embedder,
            flux2_utils.FLUX2_MODEL_INFO[args.model_version]["guidance_distilled"],
            batch,
            device,
        )

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_embedder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def flux_2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, default=None, required=True, help="text encoder (mistral 3) checkpoint path")
    parser.add_argument("--fp8_m3", action="store_true", help="use fp8 for Text Encoder model")
    flux2_utils.add_model_version_args(parser)
    return parser


if __name__ == "__main__":
    main()
