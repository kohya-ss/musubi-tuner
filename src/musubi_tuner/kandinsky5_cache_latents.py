import logging
from typing import List, Optional

import numpy as np
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ItemInfo,
    ARCHITECTURE_KANDINSKY5,
    save_latent_cache_kandinsky5,
)
import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.kandinsky5.models.vae import build_vae

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae, batch: List[ItemInfo]):
    if len(batch) == 0:
        return

    videos = []
    controls = []
    for item in batch:
        content = item.content
        if content is None:
            raise ValueError(f"Content not loaded for item {item.item_key}")

        data = np.stack(content, axis=0) if isinstance(content, list) else content
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported content type for item {item.item_key}: {type(content)}")

        video = torch.from_numpy(data)
        if video.dim() == 3:  # H, W, C -> add temporal dimension
            video = video.unsqueeze(0)
        videos.append(video)

        if item.control_content is not None:
            ctrl = item.control_content
            ctrl = np.stack(ctrl, axis=0) if isinstance(ctrl, list) else ctrl
            ctrl_tensor = torch.from_numpy(ctrl)
            if ctrl_tensor.dim() == 3:  # H, W, C -> add temporal dimension
                ctrl_tensor = ctrl_tensor.unsqueeze(0)
            controls.append(ctrl_tensor)
        else:
            controls.append(None)

    inputs = torch.stack(videos, dim=0)  # B, F, H, W, C
    inputs = inputs.to(device=vae.device, dtype=vae.dtype)
    inputs = inputs.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    inputs = inputs / 127.5 - 1.0

    scaling_factor = getattr(vae.config, "scaling_factor", 1.0)

    with torch.no_grad():
        encoded = vae.encode(inputs)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample()
        elif isinstance(encoded, tuple):
            latents = encoded[0]
        else:
            latents = encoded
    latents = latents * scaling_factor

    latents = latents.cpu()

    # encode control latents if present
    control_latents = []
    for ctrl in controls:
        if ctrl is None:
            control_latents.append(None)
            continue
        ctrl = ctrl.to(device=vae.device, dtype=vae.dtype)
        ctrl = ctrl.permute(3, 0, 1, 2).contiguous()  # F, H, W, C -> C, F, H, W
        with torch.no_grad():
            encoded_ctrl = vae.encode(ctrl.unsqueeze(0))
            if hasattr(encoded_ctrl, "latent_dist"):
                ctrl_latent = encoded_ctrl.latent_dist.sample().squeeze(0)
            elif isinstance(encoded_ctrl, tuple):
                ctrl_latent = encoded_ctrl[0].squeeze(0)
            else:
                ctrl_latent = encoded_ctrl.squeeze(0)
        control_latents.append((ctrl_latent * scaling_factor).cpu())

    for idx, (item, latent) in enumerate(zip(batch, latents)):
        image_latent = latent[:, :1, :, :].clone() if latent.dim() == 4 else None  # first frame as source
        ctrl_latent = control_latents[idx]
        logger.info(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}. latents shape: {latent.shape}, "
            f"image_latent: {None if image_latent is None else image_latent.shape}, "
            f"control_latent: {None if ctrl_latent is None else ctrl_latent.shape}"
        )
        save_latent_cache_kandinsky5(
            item_info=item,
            latent=latent,
            image_latent=image_latent,
            control_latent=ctrl_latent,
        )


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_KANDINSKY5)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "VAE checkpoint is required"

    # Build VAE (Hunyuan-based 3D VAE) from Kandinsky weights
    vae_conf = type("VAEConf", (), {"name": "hunyuan", "checkpoint_path": args.vae})
    vae = build_vae(vae_conf)

    # Apply vae_dtype if specified
    if args.vae_dtype is not None:
        from musubi_tuner.utils.model_utils import str_to_dtype
        vae_dtype = str_to_dtype(args.vae_dtype)
        vae = vae.to(vae_dtype)

    vae.to(device)
    vae.eval()
    logger.info(f"Loaded VAE. dtype: {vae.dtype}, device: {vae.device}")

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
