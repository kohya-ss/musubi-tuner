import argparse
from typing import Optional
import torch
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
import accelerate

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_WAN, ItemInfo, save_text_encoder_output_cache_wan

# for t5 config: all Wan2.1 models have the same config for t5
from musubi_tuner.wan.configs import wan_t2v_14B
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.wan.modules.t5 import T5EncoderModel

from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")


def encode_and_save_batch(
    text_encoder: T5EncoderModel, batch: list[ItemInfo], device: torch.device, accelerator: Optional[accelerate.Accelerator]
):
    prompts = [item.caption for item in batch]
    # print(prompts)

    # encode prompt
    with torch.no_grad():
        if accelerator is not None:
            with accelerator.autocast():
                context = text_encoder(prompts, device)
        else:
            context = text_encoder(prompts, device)

    # save prompt cache
    for item, ctx in zip(batch, context):
        save_text_encoder_output_cache_wan(item, ctx)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    # define accelerator for fp8 inference
    config = wan_t2v_14B.t2v_14B  # all Wan2.1 models have the same config for t5
    accelerator = None
    if args.fp8_t5:
        accelerator = accelerate.Accelerator(mixed_precision="bf16" if config.t5_dtype == torch.bfloat16 else "fp16")

    # prepare cache files and paths: all_cache_files_for_dataset = exisiting cache files, all_cache_paths_for_dataset = all cache paths in the dataset
    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    # Load T5
    logger.info(f"Loading T5: {args.t5}")
    text_encoder = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=device, weight_path=args.t5, fp8=args.fp8_t5
    )

    # Encode with T5
    logger.info("Encoding with T5")

    def encode_for_text_encoder(batch: list[ItemInfo]):
        nonlocal text_encoder, device, accelerator
        encode_and_save_batch(text_encoder, batch, device, accelerator)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files_for_dataset,
        all_cache_paths_for_dataset,
        encode_for_text_encoder,
    )
    del text_encoder

    # remove cache files not in dataset
    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--t5", type=str, default=None, required=True, help="text encoder (T5) checkpoint path")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    return parser


if __name__ == "__main__":
    main()
