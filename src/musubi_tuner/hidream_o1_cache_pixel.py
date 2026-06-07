import logging
from typing import List

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HIDREAM_O1, ItemInfo, save_pixel_cache_hidream_o1
from musubi_tuner.hidream_o1 import hidream_o1_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(batch: List[ItemInfo]):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch], dim=0)
    pixels = hidream_o1_utils.preprocess_image_tensor(contents)
    pixel_tokens = hidream_o1_utils.patchify_pixels_grid(pixels)

    for item, tokens in zip(batch, pixel_tokens):
        control_tokens = None
        if item.control_content is not None:
            controls = item.control_content if isinstance(item.control_content, list) else [item.control_content]
            # Each control image for one sample can have a different shape (e.g. reference images of
            # different aspect ratios). The cache format stores controls individually and the model
            # reads them per-control with their own patch shape, so preprocess each one separately
            # instead of stacking (torch.stack would require a uniform shape across controls).
            control_tokens = []
            for control in controls:
                control_pixels = hidream_o1_utils.preprocess_image_tensor(torch.from_numpy(control).unsqueeze(0))
                control_tokens.append(hidream_o1_utils.patchify_pixels_grid(control_pixels)[0].to(torch.bfloat16))  # H,W,D
        logger.debug(f"Saving HiDream-O1 pixel-token cache for item {item.item_key}: {tuple(tokens.shape)}")
        save_pixel_cache_hidream_o1(item, tokens.to(torch.bfloat16), control_tokens)


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HIDREAM_O1)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return

    cache_latents.encode_datasets(datasets, encode_and_save_batch, args)
    logger.info("Done!")


if __name__ == "__main__":
    main()
