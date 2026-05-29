import argparse
import logging

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LENS, ItemInfo, save_text_encoder_output_cache_lens
from musubi_tuner.lens import lens_text_encoder
import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(text_embedder: lens_text_encoder.LensTextEmbedder, batch: list[ItemInfo], device: torch.device):
    prompts = [item.caption for item in batch]
    autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype
    with torch.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
        layer_features, mask = text_embedder(prompts)
        layer_features = [feat.cpu() for feat in layer_features]
        mask = mask.cpu()

    for i, item in enumerate(batch):
        valid_len = int(mask[i].sum().item())
        trimmed = [feat[i, :valid_len].contiguous() for feat in layer_features]
        save_text_encoder_output_cache_lens(item, trimmed)


def main():
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser = lens_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LENS)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    all_cache_files_for_dataset, all_cache_paths_for_dataset = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)

    te_dtype = torch.bfloat16 if args.text_encoder_dtype is None else str_to_dtype(args.text_encoder_dtype)
    text_embedder = lens_text_encoder.load_lens_text_embedder(
        args.text_encoder,
        dtype=te_dtype,
        device=device,
        disable_mmap=args.disable_numpy_memmap,
    )

    def encode_for_text_encoder(batch: list[ItemInfo], text_embedder=text_embedder):
        encode_and_save_batch(text_embedder, batch, device)

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

    cache_text_encoder_outputs.post_process_cache_files(
        datasets, all_cache_files_for_dataset, all_cache_paths_for_dataset, args.keep_cache
    )


def lens_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--text_encoder", type=str, required=True, help="Lens Comfy GPT-OSS text encoder safetensors path")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="text encoder dtype, default bfloat16")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors")
    return parser


if __name__ == "__main__":
    main()
