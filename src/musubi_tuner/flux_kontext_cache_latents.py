import logging
from typing import List

import numpy as np
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    ARCHITECTURE_FLUX_KONTEXT,
    save_latent_cache_flux_kontext,
)
from musubi_tuner.flux import flux_utils
from musubi_tuner.flux import flux_models
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_flux_kontext(batch: List[ItemInfo]) -> tuple[torch.Tensor, List[List[np.ndarray]]]:
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C), the length of the list is 1 for FLUX.1 Kontext

    # Stack batch into target tensor (B,H,W,C) in RGB order and control images list of tensors (H, W, C)
    contents = []
    controls = []
    for item in batch:
        contents.append(torch.from_numpy(item.content))  # target image

        if isinstance(item.control_content[0], np.ndarray):
            control_image = item.control_content[0]  # np.ndarray
            control_image = control_image[..., :3]  # ensure RGB, remove alpha if present
        else:
            control_image = item.control_content[0]  # PIL.Image
            control_image = control_image.convert("RGB")  # convert to RGB if RGBA
        controls.append(torch.from_numpy(np.array(control_image)))

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    # we can stack controls because they are all the same size (bucketed)
    controls = torch.stack(controls, dim=0)  # B, H, W, C
    controls = controls.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    controls = controls / 127.5 - 1.0  # normalize to [-1, 1]

    return contents, controls


def encode_and_save_batch(ae: flux_models.AutoEncoder, batch: List[ItemInfo]):
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)

    # assert all items in the batch have the one control content
    if not all(len(item.control_content) == 1 for item in batch):
        raise ValueError("FLUX.1 Kontext requires exactly one control content per item.")

    # _, _, contents, content_masks = preprocess_contents(batch)
    contents, controls = preprocess_contents_flux_kontext(batch)

    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))  # B, C, H, W
        control_latents = ae.encode(controls.to(ae.device, dtype=ae.dtype))  # B, C, H, W

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        target_latent = latents[b]  # C, H, W. Target latents for this image (ground truth)
        control_latent = control_latents[b]  # C, H, W

        print(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}. control latents shape: {control_latent.shape}, target latents shape: {target_latent.shape}"
        )

        # save cache (file path is inside item.latent_cache_path pattern), remove batch dim
        save_latent_cache_flux_kontext(
            item_info=item,
            latent=target_latent,  # Ground truth for this image
            control_latent=control_latent,  # Control latent for this image
        )


# def flux_kontext_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
#     return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    # parser = flux_kontext_setup_parser(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in FLUX.1 Kontext.")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_KONTEXT)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "ae checkpoint is required"

    logger.info(f"Loading AE model from {args.vae}")
    ae = flux_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)
    ae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
