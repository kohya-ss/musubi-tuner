"""
Cache latents for Z-Image architecture.

This script encodes images using Z-Image's VAE and caches the latent representations
for faster training.

Standard mode: Cache target image latents only
OmniBase mode: Cache target + control latents + SigLIP2 features (auto-detected from control_directory)
"""

import logging
from typing import List

import numpy as np
import torch
from PIL import Image

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    ARCHITECTURE_Z_IMAGE,
    save_latent_cache_z_image,
    has_omnibase_cache,
)
from musubi_tuner.zimage import zimage_autoencoder
from musubi_tuner.zimage.zimage_autoencoder import AutoencoderKL
from musubi_tuner.zimage import zimage_utils
import musubi_tuner.cache_latents as cache_latents

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


def preprocess_control_image(control_array: np.ndarray) -> torch.Tensor:
    """
    Preprocess a single control image for VAE encoding.

    Args:
        control_array: Control image as numpy array (H, W, C) in RGB order, uint8

    Returns:
        torch.Tensor: Preprocessed tensor (C, H, W) normalized to [-1, 1]
    """
    content = torch.from_numpy(control_array)
    if content.shape[-1] == 4:  # RGBA
        content = content[..., :3]  # remove alpha channel
    content = content.permute(2, 0, 1)  # H, W, C -> C, H, W
    content = content.float() / 127.5 - 1.0  # normalize to [-1, 1]
    return content


def extract_siglip_features(
    vision_model,
    processor,
    control_array: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract SigLIP2 features from a control image and convert to spatial grid.

    Args:
        vision_model: SigLIP2 vision model
        processor: SigLIP2 processor for image preprocessing
        control_array: Control image as numpy array (H, W, C) in RGB order, uint8
        device: Device to run inference on

    Returns:
        torch.Tensor: SigLIP features as spatial grid [H_sig, W_sig, D_sig]
    """
    # Convert numpy array to PIL Image for processor
    if control_array.shape[-1] == 4:  # RGBA
        control_array = control_array[..., :3]
    pil_image = Image.fromarray(control_array)

    # Preprocess image using SigLIP2 processor
    inputs = processor(images=pil_image, return_tensors="pt")

    # Move to device and cast to model dtype for efficiency (avoids fp32→fp16 conversions)
    model_dtype = next(vision_model.parameters()).dtype
    inputs = {k: v.to(device, dtype=model_dtype) if v.is_floating_point() else v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = vision_model(**inputs)
        last_hidden = outputs.last_hidden_state[0]  # [num_tokens, C]

    # Convert to spatial grid (handles CLS token if present)
    return zimage_utils.siglip_last_hidden_to_grid(last_hidden)


def encode_and_save_batch(
    vae: AutoencoderKL,
    batch: List[ItemInfo],
    vision_model=None,
    processor=None,
):
    """
    Encode a batch of images and save their latent representations.

    Standard mode: Encodes and saves target latents only
    OmniBase mode: Also encodes control latents and extracts SigLIP2 features

    Args:
        vae: Z-Image VAE model (AutoencoderKL)
        batch: List of ItemInfo containing images to encode
        vision_model: Optional SigLIP2 vision model for OmniBase
        processor: Optional SigLIP2 processor for OmniBase
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

    # Save cache for each item in the batch
    for b, item in enumerate(batch):
        latent = latents[b]  # C, H, W

        control_latents = None
        siglip_features = None

        # OmniBase: Process control images if present
        if item.control_content is not None:
            control_latents = []
            # Only create siglip_features list if BOTH vision_model and processor are available
            # (avoids 1:1 assertion failure when processor is None)
            siglip_features = [] if (vision_model is not None and processor is not None) else None

            # control_content can be a single np.ndarray or list[np.ndarray]
            controls = item.control_content
            if not isinstance(controls, list):
                controls = [controls]

            for ctrl_array in controls:
                # Encode control image with VAE (must be under no_grad to avoid graph building)
                ctrl_tensor = preprocess_control_image(ctrl_array)
                ctrl_tensor = ctrl_tensor.unsqueeze(0).to(vae.device, dtype=vae.dtype)  # [1, C, H, W]

                with torch.no_grad():
                    ctrl_posterior = vae.encode(ctrl_tensor)
                    ctrl_latent = ctrl_posterior.mode()[0]  # [C, H, W]
                control_latents.append(ctrl_latent)

                # Extract SigLIP features if encoder available (already has no_grad inside)
                if vision_model is not None and processor is not None:
                    sig_feat = extract_siglip_features(vision_model, processor, ctrl_array, vae.device)
                    siglip_features.append(sig_feat)

        logger.debug(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}. "
            f"Latent shape: {latent.shape}, "
            f"Control latents: {len(control_latents) if control_latents else 0}, "
            f"SigLIP features: {len(siglip_features) if siglip_features else 0}"
        )

        save_latent_cache_z_image(
            item_info=item,
            latent=latent,
            control_latents=control_latents,
            siglip_features=siglip_features,
        )


def main():
    parser = cache_latents.setup_parser_common()

    # OmniBase-specific arguments
    parser.add_argument(
        "--image_encoder",
        type=str,
        default=None,
        help="Path to SigLIP2 encoder for OmniBase I2I caching. "
        "Can be a HuggingFace repo ID (e.g., 'google/siglip2-base-patch16-256') "
        "or a local directory. If not provided, control images cached without SigLIP features.",
    )
    parser.add_argument(
        "--image_encoder_dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32"],
        help="Data type for SigLIP2 encoder. If not specified, auto-selects based on device.",
    )

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

    # Check if any dataset has control images (OmniBase mode)
    has_control_data = any(hasattr(ds, "has_control") and ds.has_control for ds in datasets)
    if has_control_data:
        logger.info("Detected control_directory in dataset config - OmniBase mode enabled")

        # Warn about --skip_existing potentially keeping stale non-OmniBase caches
        if args.skip_existing:
            logger.warning(
                "WARNING: --skip_existing is set with control_directory. "
                "If existing cache files were created without OmniBase support, "
                "they will be kept and will NOT contain control latents or SigLIP features. "
                "Consider removing old cache files or running without --skip_existing for a clean re-cache."
            )
            # Sample existing cache files directly (avoid spinning up dataset threadpool)
            import os
            import glob

            sample_stale_caches = []
            for ds in datasets:
                cache_dir = getattr(ds, "cache_directory", None)
                if cache_dir and os.path.isdir(cache_dir):
                    # Z-Image caches end with _zi.safetensors
                    cache_files = glob.glob(os.path.join(cache_dir, "*_zi.safetensors"))[:5]
                    for cache_file in cache_files:
                        if not has_omnibase_cache(cache_file):
                            sample_stale_caches.append(cache_file)
                            break  # Found one stale cache, that's enough
                if sample_stale_caches:
                    break

            if sample_stale_caches:
                logger.warning(
                    f"Found existing cache files WITHOUT OmniBase data (e.g., {sample_stale_caches[0]}). "
                    "These will be skipped and training will fail to use control images for these samples!"
                )

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

    # Load SigLIP2 encoder if specified (OmniBase)
    vision_model = None
    processor = None
    if args.image_encoder is not None:
        encoder_dtype = None
        if args.image_encoder_dtype is not None:
            encoder_dtype = getattr(torch, args.image_encoder_dtype)

        logger.info(f"Loading SigLIP2 encoder from {args.image_encoder}")
        vision_model, processor = zimage_utils.load_siglip2_encoder(
            encoder_path=args.image_encoder,
            device=device,
            dtype=encoder_dtype,
        )
        if vision_model is not None:
            logger.info(f"Loaded SigLIP2 encoder, dtype: {next(vision_model.parameters()).dtype}")
        else:
            logger.warning("Failed to load SigLIP2 encoder. Control images will be cached without SigLIP features.")
    elif has_control_data:
        logger.warning(
            "Control images detected but --image_encoder not specified. "
            "Control latents will be cached but SigLIP features will NOT be available. "
            "Consider specifying --image_encoder for full OmniBase support."
        )

    # Encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch, vision_model=vision_model, processor=processor)

    # Reuse core loop from cache_latents
    cache_latents.encode_datasets(datasets, encode, args)

    logger.info("Done!")


if __name__ == "__main__":
    main()
