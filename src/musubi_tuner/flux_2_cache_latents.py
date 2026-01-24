import logging
from typing import List

import numpy as np
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_flux_2
from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2 import flux2_models
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_flux_2(batch: List[ItemInfo]) -> tuple[torch.Tensor, List[List[np.ndarray]]]:
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C), optional

    # Stack batch into target tensor (B,H,W,C) in RGB order and control images list of tensors (H, W, C)
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content))  # target image

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    controls = []
    for item in batch:
        if item.control_content is not None and len(item.control_content) > 0:
            controls.append([torch.from_numpy(cc[..., :3]) for cc in item.control_content])  # ensure RGB, remove alpha if present

    if len(controls) > 0:  # controls is list of list of (H, W, C), where H, W can vary
        controls = [[c.permute(2, 0, 1) for c in cl] for cl in controls]  # list of list of (H, W, C) -> list of list of (C, H, W)
        controls = [[c / 127.5 - 1.0 for c in cl] for cl in controls]  # normalize to [-1, 1]
    else:
        controls = None

    return contents, controls


def encode_and_save_batch(ae: flux2_models.AutoEncoder, batch: List[ItemInfo], arch_full: str):
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)

    contents, controls = preprocess_contents_flux_2(batch)

    with torch.no_grad():
        latents = ae.encode(contents.to(ae.device, dtype=ae.dtype))  # B, C, H, W
        if controls is not None:
            control_latents = [[ae.encode(c.to(ae.device, dtype=ae.dtype).unsqueeze(0))[0] for c in cl] for cl in controls]
            # now control_latents is list of list of (C, H, W) tensors
        else:
            control_latents = None

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        target_latent = latents[b]  # C, H, W. Target latents for this image (ground truth)
        control_latent = control_latents[b] if control_latents is not None else None  # list of (C, H, W) tensors or None

        print(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}, target latents shape: {target_latent.shape}, "
            f"control latents shape: {[cl.shape for cl in control_latent] if control_latent is not None else None}"
        )

        # save cache (file path is inside item.latent_cache_path pattern)
        save_latent_cache_flux_2(
            item_info=item,
            latent=target_latent,  # Ground truth for this image
            control_latent=control_latent,  # Control latent for this image
            arch_full=arch_full,
        )


def main():
    parser = cache_latents.setup_parser_common()
    flux2_utils.add_model_version_args(parser)

    args = parser.parse_args()
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=model_version_info.architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "ae checkpoint is required"

    logger.info(f"Loading AE model from {args.vae}")
    vae_dtype = torch.float32 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    ae = flux2_utils.load_ae(args.vae, dtype=vae_dtype, device=device, disable_mmap=True)
    ae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch, model_version_info.architecture_full)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
