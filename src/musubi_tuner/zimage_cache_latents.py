"""
Cache latents for Z-Image architecture.

This script encodes images using Z-Image's VAE and caches the latent representations
for faster training. Unlike other architectures, Z-Image does not support control images,
so only the target image latents are cached.
"""

import argparse
import logging
import math
from typing import List, Optional, Tuple

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_Z_IMAGE, ItemInfo, save_latent_cache_z_image
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.zimage import zimage_autoencoder
from musubi_tuner.zimage.zimage_autoencoder import AutoencoderKL
from musubi_tuner.zimage.zimage_utils import load_image_encoders

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_zimage(batch: List[ItemInfo]) -> torch.Tensor:
    """
    Preprocess batch contents for Z-Image VAE encoding.

    Args:
        batch: List of ItemInfo containing target images

    Returns:
        torch.Tensor: Preprocessed image tensor (B, C, H, W) normalized to [-1, 1]
    """
    contents = []
    for item in batch:
        # item.content: target image (H, W, C) in RGB order, uint8
        content = torch.from_numpy(item.content)
        if content.shape[-1] == 4:  # RGBA
            content = content[..., :3]  # remove alpha channel
        contents.append(content)

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents.float() / 127.5 - 1.0  # normalize to [-1, 1]

    return contents


def _control_images_to_list(control_content) -> list:
    if control_content is None:
        return []
    if isinstance(control_content, list):
        return control_content
    return [control_content]


def _siglip_last_hidden_to_grid(last_hidden_state: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [tokens, C]
    seq_len = int(last_hidden_state.shape[0])
    c = int(last_hidden_state.shape[1])

    g = int(math.isqrt(seq_len))
    if g * g == seq_len:
        tokens = last_hidden_state
        h = w = g
    else:
        g = int(math.isqrt(seq_len - 1))
        if g * g != (seq_len - 1):
            raise ValueError(f"SigLIP last_hidden_state length is not a square: {seq_len}")
        tokens = last_hidden_state[1:]
        h = w = g

    return tokens.reshape(h, w, c)


def encode_and_save_batch(vae: AutoencoderKL, image_encoder_assets: Optional[Tuple], batch: List[ItemInfo], i2v: bool = False):
    """
    Encode a batch of images and save their latent representations.

    Args:
        vae: Z-Image VAE model (AutoencoderKL)
        batch: List of ItemInfo containing images to encode
    """
    contents = preprocess_contents_zimage(batch)

    h, w = contents.shape[2], contents.shape[3]
    if h < 16 or w < 16:
        item = batch[0]
        raise ValueError(f"Image size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    with torch.no_grad():
        # Move to VAE device and dtype
        contents = contents.to(vae.device, dtype=vae.dtype)

        # Encode using VAE - returns DiagonalGaussianDistribution
        posterior = vae.encode(contents)

        # Use mode() for deterministic latents (mean of the distribution)
        # This is preferred for training as it provides consistent latents
        latents = posterior.mode()

    control_latents: Optional[list[list[torch.Tensor]]] = None
    siglip_features: Optional[list[list[torch.Tensor]]] = None
    if i2v:
        assert image_encoder_assets is not None, "--i2v requires --image_encoder to be set."
        feature_extractor, image_encoder = image_encoder_assets
        control_latents = []
        siglip_features = []

        for item in batch:
            ctrl_imgs = _control_images_to_list(item.control_content)
            if len(ctrl_imgs) == 0:
                control_latents.append([])
                siglip_features.append([])
                continue

            item_control_latents = []
            item_siglip = []
            for ci in ctrl_imgs:
                ci_t = torch.from_numpy(ci)
                if ci_t.shape[-1] == 4:
                    ci_t = ci_t[..., :3]
                ci_t = ci_t.permute(2, 0, 1).float() / 127.5 - 1.0
                ci_t = ci_t.unsqueeze(0).to(vae.device, dtype=vae.dtype)

                with torch.no_grad():
                    post = vae.encode(ci_t)
                    cl = post.mode()[0]
                item_control_latents.append(cl)

                ci_np = ci[..., :3] if ci.shape[-1] == 4 else ci
                with torch.no_grad():
                    vision_feature = hf_clip_vision_encode(ci_np, feature_extractor, image_encoder)
                    last_hidden = vision_feature.last_hidden_state[0]  # [tokens, C]
                grid = _siglip_last_hidden_to_grid(last_hidden)
                item_siglip.append(grid)

            control_latents.append(item_control_latents)
            siglip_features.append(item_siglip)

    # Save cache for each item in the batch
    for b, item in enumerate(batch):
        latent = latents[b]  # C, H, W

        logger.debug(f"Saving cache for item {item.item_key} at {item.latent_cache_path}. Latent shape: {latent.shape}")

        if i2v:
            cl = None if control_latents is None else control_latents[b]
            sf = None if siglip_features is None else siglip_features[b]
            save_latent_cache_z_image(item_info=item, latent=latent, control_latent=cl, siglip_feature=sf)
        else:
            save_latent_cache_z_image(item_info=item, latent=latent)


def zimage_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--i2v",
        action="store_true",
        help="Cache control-image latents and SigLIP features for OmniBase/edit training",
    )
    parser.add_argument(
        "--image_encoder",
        type=str,
        default=None,
        help="Directory/path of SigLIP Image Encoder (required if --i2v is set)",
    )
    return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = zimage_setup_parser(parser)
    # Z-Image VAE doesn't need special tiling options like HunyuanVideo
    # but we can add them if needed in the future

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.vae_dtype is not None:
        logger.warning("VAE dtype is specified but Z-Image VAE always uses float32 for better precision.")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_Z_IMAGE)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    has_control_dataset = any(
        (hasattr(ds, "has_control") and bool(ds.has_control))
        or (hasattr(ds, "control_directory") and ds.control_directory is not None)
        for ds in datasets
    )
    if not args.i2v and has_control_dataset:
        logger.info("Dataset config has control data. Enabling --i2v.")
        args.i2v = True

    if args.i2v:
        assert args.image_encoder is not None, "--i2v requires --image_encoder to be set."
    elif args.image_encoder is not None:
        logger.info("--image_encoder is set but --i2v is not set. Enabling --i2v.")
        args.i2v = True

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=1
        )
        return

    assert args.vae is not None, "VAE checkpoint is required (--vae)"

    logger.info(f"Loading Z-Image VAE from {args.vae}")
    vae = zimage_autoencoder.load_autoencoder_kl(args.vae, device=device, disable_mmap=True)
    vae.eval()
    logger.info(f"Loaded Z-Image VAE, dtype: {vae.dtype}")

    if args.i2v:
        feature_extractor, image_encoder = load_image_encoders(args)
        image_encoder.to(device)
        image_encoder_assets = (feature_extractor, image_encoder)
    else:
        image_encoder_assets = None

    # Encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, image_encoder_assets, batch, args.i2v)

    # Reuse core loop from cache_latents
    cache_latents.encode_datasets(datasets, encode, args)

    logger.info("Done!")


if __name__ == "__main__":
    main()
