"""Cache target and optional clean-reference latents for Krea 2 training.

K2 uses the *same* Qwen-Image VAE as musubi's qwen_image integration, and the same
latent normalization (`(raw - mean) / std`), so this reuses `qwen_image_utils.load_vae`
and `encode_pixels_to_latents` directly.
"""

import logging
from typing import List

import numpy as np
import torch
from safetensors import safe_open

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_krea2
from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_KREA2,
    ARCHITECTURE_KREA2_EDIT,
    ARCHITECTURE_KREA2_EDIT_FULL,
)
from musubi_tuner.dataset.media_utils import resize_image_to_bucket
from musubi_tuner.krea2.krea2_sampling import resize_reference_image
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

KREA2_CONTROL_IMAGE_MAX_PIXELS = 1024 * 1024


def _control_resolution_string(item: ItemInfo) -> str:
    resolution = item.control_resolution
    return "none" if resolution is None else f"{resolution[0]}x{resolution[1]}"


def krea2_edit_cache_metadata(item: ItemInfo) -> dict[str, str]:
    return {
        "krea2_edit": "true",
        "match_target_res": str(bool(item.match_target_res)).lower(),
        "no_resize_control": str(bool(item.no_resize_control)).lower(),
        "control_resolution": _control_resolution_string(item),
        "control_image_max_pixels": str(KREA2_CONTROL_IMAGE_MAX_PIXELS),
    }


def validate_krea2_edit_cache(item: ItemInfo) -> bool:
    """Return whether an existing Edit latent cache matches this dataset policy."""
    try:
        with safe_open(item.latent_cache_path, framework="pt", device="cpu") as cache:
            metadata = cache.metadata() or {}
    except Exception as error:
        logger.warning(f"Cannot read existing Krea 2 Edit cache {item.latent_cache_path}: {error}; rebuilding it")
        return False

    expected = {"architecture": ARCHITECTURE_KREA2_EDIT_FULL, **krea2_edit_cache_metadata(item)}
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if mismatches:
        logger.warning(
            f"Existing Krea 2 Edit cache policy does not match dataset config for {item.item_key}: {mismatches}; rebuilding it"
        )
        return False
    return True


def prepare_krea2_control_image(item: ItemInfo, control: np.ndarray) -> torch.Tensor:
    """Prepare one raw RGB reference for the VAE according to the dataset policy."""
    control = control[..., :3]
    target = item.content[0] if isinstance(item.content, list) else item.content
    target_height, target_width = target.shape[:2]

    if item.match_target_res:
        image = torch.from_numpy(np.array(control, copy=True)).permute(2, 0, 1).float().div_(255.0)
        return resize_reference_image(
            image,
            target_height * target_width,
            snap=16,
            antialias=False,
            allow_upscale=True,
        )

    if not item.no_resize_control and item.control_resolution is None:
        # Musubi's ordinary control behavior: exact target bucket shape, using
        # aspect-preserving resize followed by a center crop.
        resized = resize_image_to_bucket(control, (target_width, target_height))
        return torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float().div_(255.0)

    pixel_budget = (
        item.control_resolution[0] * item.control_resolution[1]
        if item.control_resolution is not None
        else KREA2_CONTROL_IMAGE_MAX_PIXELS
    )
    image = torch.from_numpy(np.array(control, copy=True)).permute(2, 0, 1).float().div_(255.0)
    return resize_reference_image(
        image,
        pixel_budget,
        snap=16,
        antialias=False,
        allow_upscale=item.control_resolution is not None and not item.no_resize_control,
    )


def encode_and_save_batch(
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    batch: List[ItemInfo],
    *,
    edit: bool = False,
):
    # Stack batch into (B, C, 1, H, W) in RGB order, normalized to [-1, 1].
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content))
    contents = torch.stack(contents, dim=0)  # (B, H, W, C)
    contents = contents.permute(0, 3, 1, 2)  # (B, C, H, W)
    contents = contents.unsqueeze(2)  # (B, C, 1, H, W), Qwen-Image VAE needs F axis
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(
            contents.to(vae.device, dtype=vae.dtype),
            sample_posterior=edit,
        )  # (B, C, 1, H, W)

        control_latents: list[list[torch.Tensor]] | None = None
        if edit:
            for item in batch:
                if item.control_content is None or len(item.control_content) == 0:
                    raise ValueError(f"Krea 2 Edit item {item.item_key} has no reference/control image")

            control_latents = [[] for _ in batch]
            num_controls = len(batch[0].control_content)
            if any(len(item.control_content) != num_controls for item in batch):
                raise ValueError("Krea 2 Edit cache batch contains different reference-image counts")

            for control_index in range(num_controls):
                controls = [prepare_krea2_control_image(item, item.control_content[control_index]) for item in batch]
                shapes = {tuple(control.shape) for control in controls}
                if len(shapes) != 1:
                    raise ValueError(f"Krea 2 Edit control {control_index} has incompatible batch shapes: {sorted(shapes)}")
                control_pixels = torch.stack(controls).unsqueeze(2).mul(2).sub(1)
                encoded = vae.encode_pixels_to_latents(
                    control_pixels.to(vae.device, dtype=vae.dtype),
                    sample_posterior=True,
                )
                for batch_index in range(len(batch)):
                    control_latents[batch_index].append(encoded[batch_index])

    for b, item in enumerate(batch):
        target_latent = latents[b]  # (C, 1, H, W)
        item_controls = control_latents[b] if control_latents is not None else None
        print(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}, "
            f"latents shape: {target_latent.shape}, control latents: "
            f"{[tuple(control.shape) for control in item_controls] if item_controls is not None else None}"
        )
        save_latent_cache_krea2(
            item_info=item,
            latent=target_latent,
            control_latent=item_controls,
            edit=edit,
            extra_metadata=krea2_edit_cache_metadata(item) if edit else None,
        )


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    parser.add_argument("--edit", action="store_true", help="cache Krea 2 Edit target/reference latents")

    args = parser.parse_args()

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in Krea 2 (uses the Qwen-Image VAE default).")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    architecture = ARCHITECTURE_KREA2_EDIT if args.edit else ARCHITECTURE_KREA2
    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets
    if args.edit:
        missing_control = [index for index, dataset in enumerate(datasets) if not dataset.has_control]
        if missing_control:
            raise ValueError(f"Krea 2 --edit requires control images in every dataset; missing in datasets {missing_control}")

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "VAE checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = qwen_image_utils.load_vae(args.vae, 3, device=device, disable_mmap=True)
    vae.to(device)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch, edit=args.edit)

    cache_latents.encode_datasets(
        datasets,
        encode,
        args,
        existing_cache_validator=validate_krea2_edit_cache if args.edit else None,
    )


if __name__ == "__main__":
    main()
