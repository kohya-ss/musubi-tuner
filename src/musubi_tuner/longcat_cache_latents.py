import argparse
import os
from contextlib import nullcontext
from typing import List, Sequence, cast

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_LONGCAT,
    BaseDataset,
    ItemInfo,
    save_latent_cache_longcat,
)
from musubi_tuner.longcat_video.configs import LONGCAT_LATENTS_MEAN, LONGCAT_LATENTS_STD
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.wan.modules.vae import WanVAE

import musubi_tuner.cache_latents as cache_latents

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _amp_context(device: torch.device, dtype: torch.dtype):
    if device.type in {"cuda", "xpu"}:
        try:
            from torch.amp import autocast as torch_autocast  # type: ignore[attr-defined]

            return torch_autocast(device_type=device.type, dtype=dtype)
        except (ImportError, AttributeError):
            from torch.cuda.amp import autocast as torch_autocast

            return torch_autocast(dtype=dtype)
    return nullcontext()


def _latent_dist(result):
    if isinstance(result, tuple):
        return result[0]
    return result.latent_dist  # type: ignore[attr-defined]


LongcatVAE = WanVAE


def _load_longcat_vae(vae_path: str, device: torch.device, vae_dtype: torch.dtype) -> LongcatVAE:
    vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype)
    vae.to(device)
    vae.to(vae_dtype)
    vae.eval()
    return vae


def _normalize_longcat_latents(latents: torch.Tensor) -> torch.Tensor:
    device = latents.device
    dtype = latents.dtype
    latents_fp32 = latents.to(torch.float32)
    mean = torch.tensor(LONGCAT_LATENTS_MEAN, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LONGCAT_LATENTS_STD, dtype=torch.float32, device=device).view(1, -1, 1, 1, 1).clamp_min(1e-6)
    normalized = (latents_fp32 - mean) / std
    return normalized.to(dtype)


def encode_and_save_batch(vae: LongcatVAE, batch: List[ItemInfo], store_first_frame: bool) -> None:
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if contents.ndim == 4:
        contents = contents.unsqueeze(1)

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()
    contents = contents.to(device=vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0

    height, width = contents.shape[-2:]
    if height < 8 or width < 8:
        item = batch[0]
        raise ValueError(
            f"Image or video size too small: {item.item_key} (and others), size: {item.original_size}"
        )

    with _amp_context(vae.device, vae.dtype), torch.no_grad():
        latents = vae.encode(contents)
        if isinstance(latents, list):
            latents = torch.stack(latents, dim=0)
        latents = latents.to(device=vae.device, dtype=vae.dtype)
        latents = _normalize_longcat_latents(latents)

    image_latents = None
    if store_first_frame:
        first_frame = contents[:, :, :1, :, :]
        with _amp_context(vae.device, vae.dtype), torch.no_grad():
            image_latents = vae.encode(first_frame)
            if isinstance(image_latents, list):
                image_latents = torch.stack(image_latents, dim=0)
            image_latents = image_latents.to(device=vae.device, dtype=vae.dtype)
            image_latents = _normalize_longcat_latents(image_latents)

    for idx, item in enumerate(batch):
        latent_tensor = latents[idx]
        image_latent_tensor = image_latents[idx] if image_latents is not None else None
        save_latent_cache_longcat(item, latent_tensor, image_latent=image_latent_tensor)


def _load_datasets(args) -> Sequence[BaseDataset]:
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_LONGCAT)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    return cast(Sequence[BaseDataset], dataset_group.datasets)


def main() -> None:
    parser = cache_latents.setup_parser_common()
    parser = longcat_setup_parser(parser)
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = _load_datasets(args)

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            list(datasets), args.debug_mode, args.console_width, args.console_back, args.console_num_images
        )
        return

    assert args.vae is not None, "VAE checkpoint is required"

    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae = _load_longcat_vae(args.vae, device, vae_dtype)

    if args.vae_tile_min_size is not None:
        if isinstance(vae, AutoencoderKLWan):
            vae.enable_tiling(args.vae_tile_min_size, args.vae_tile_min_size)
        else:
            logger.warning("VAE tiling requested but not supported for this VAE loader.")

    def encode_fn(batch: List[ItemInfo]) -> None:
        encode_and_save_batch(vae, batch, args.i2v)

    cache_latents.encode_datasets(list(datasets), encode_fn, args)


def longcat_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--i2v", action="store_true", help="Store first-frame latent for image-to-video conditioning")
    parser.add_argument(
        "--vae_tile_min_size",
        type=int,
        default=None,
        help="Minimum spatial size for LongCat VAE tiling (height & width).",
    )
    return parser


if __name__ == "__main__":
    main()
