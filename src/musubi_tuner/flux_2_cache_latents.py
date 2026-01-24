import logging
from typing import List, Optional
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_flux_2
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2 import flux2_models
from musubi_tuner.flux_2.flux2_utils import Flux2ModelInfo
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Module-level variable to hold model_version_info for encode_and_save_batch
_model_version_info: Optional[Flux2ModelInfo] = None


def preprocess_contents_flux_2(batch: List[ItemInfo]) -> tuple[torch.Tensor, List[List[np.ndarray]]]:
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C), optional

    # Stack batch into target tensor (B,H,W,C) in RGB order and control images list of tensors (H, W, C)
    contents = []
    controls = []
    for item in batch:
        contents.append(torch.from_numpy(item.content))  # target image

        if item.control_content is not None and len(item.control_content) > 0:
            if len(item.control_content) > 1:
                limit_pixels = 1024**2
            elif len(item.control_content) == 1:
                limit_pixels = 2024**2
            else:
                limit_pixels = None
            img_ctx = [(Image.fromarray(cc) if isinstance(cc, np.ndarray) else cc).convert("RGB") for cc in item.control_content]

            img_ctx_prep = flux2_utils.default_prep(
                img=img_ctx,
                limit_pixels=limit_pixels,
            )
            controls.append(img_ctx_prep)

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    if not controls:
        controls = None

    return contents, controls


def encode_and_save_batch(ae: flux2_models.AutoEncoder, batch: List[ItemInfo]):
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)

    contents, controls = preprocess_contents_flux_2(batch)

    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))  # B, C, H, W
        if controls is not None:
            control_latents = [[ae.encode(c.to(ae.device, dtype=ae.dtype).unsqueeze(0))[0] for c in cl] for cl in controls]
        else:
            control_latents = None

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        target_latent = latents[b]  # C, H, W. Target latents for this image (ground truth)
        control_latent = control_latents[b] if control_latents is not None else None  # C, H, W

        # Process mask for this item if it has one
        mask_weights_i = None
        if item.mask_content is not None:
            # mask_content is (H, W) grayscale numpy array with values 0-255
            mask = torch.from_numpy(item.mask_content).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            # Normalize mask from 0-255 to 0-1
            mask = mask.float() / 255.0

            # Downsample mask to latent space dimensions using area interpolation
            lat_h, lat_w = target_latent.shape[-2:]
            mask = F.interpolate(mask, size=(lat_h, lat_w), mode="area")  # (1, 1, lat_h, lat_w)

            mask_weights_i = mask  # Keep as (1, 1, H, W) for layout="video" with F=1

        print(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}, target latents shape: {target_latent.shape}, "
            f"control latents shape: {[cl.shape for cl in control_latent] if control_latent is not None else None}"
            f"{f', mask shape: {mask_weights_i.shape}' if mask_weights_i is not None else ''}"
        )

        # save cache (file path is inside item.latent_cache_path pattern), remove batch dim
        save_latent_cache_flux_2(
            item_info=item,
            latent=target_latent,  # Ground truth for this image
            control_latent=control_latent,  # Control latent for this image
            arch_full=_model_version_info.architecture_full,  # e.g., "flux_2_dev", "flux_2_klein_4b"
            mask_weights=mask_weights_i,  # Mask weights for mask-weighted loss training
        )


def main():
    global _model_version_info

    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    flux2_utils.add_model_version_args(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    # Get model version info (dataclass with architecture info)
    _model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
    logger.info(f"Model version: {args.model_version}, architecture: {_model_version_info.architecture}")

    # VAE dtype (defaults to float32)
    vae_dtype = torch.float32 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    logger.info(f"Using VAE dtype: {vae_dtype}")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=_model_version_info.architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "ae checkpoint is required"

    logger.info(f"Loading AE model from {args.vae}")
    ae = flux2_utils.load_ae(args.vae, dtype=vae_dtype, device=device, disable_mmap=True)
    ae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
