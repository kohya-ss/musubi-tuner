import argparse
import logging
from typing import List

import numpy as np
import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_QWEN_IMAGE_LAYERED,
    ItemInfo,
    save_latent_cache_qwen_image,
)
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def preprocess_contents_qwen_image(batch: List[ItemInfo], is_layered: bool) -> tuple[torch.Tensor]:
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C), optional

    # Stack batch into target tensor (B,H,W,C) in RGB order and control images list of tensors (H, W, C)
    # Currently control images must have same size as target images. The length of control images can vary.
    contents = []
    controls = []
    n_channels = 4 if is_layered else 3  # layered model uses 4 channels for control images (RGBA), otherwise 3 channels (RGB
    for item in batch:
        content = item.content
        if is_layered and content.shape[2] == 3:
            # add alpha channel with full opacity: Qwen-Image-Layered VAE expects 4 channels
            h, w, _ = content.shape
            alpha_channel = np.ones((h, w, 1), dtype=content.dtype) * 255
            content = np.concatenate([content, alpha_channel], axis=2)  # H, W, 4

        contents.append(torch.from_numpy(content))  # target image

        if item.control_content is not None and len(item.control_content) > 0:
            controls.append([torch.from_numpy(cc[..., :n_channels]) for cc in item.control_content])

    contents = torch.stack(contents, dim=0)  # B, H, W, C
    contents = contents.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    if len(controls) > 0:  # controls is list of list of (H, W, C), where H, W can vary
        controls = [[c.permute(2, 0, 1) for c in cl] for cl in controls]  # list of list of (H, W, C) -> list of list of (C, H, W)
        controls = [[c / 127.5 - 1.0 for c in cl] for cl in controls]  # normalize to [-1, 1]
    else:
        controls = None

    return contents, controls


def encode_and_save_batch(vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, batch: List[ItemInfo], is_layered: bool):
    # item.content: target image (H, W, C)
    contents, controls = preprocess_contents_qwen_image(batch, is_layered)  # B, C, H, W and list of (C, F, H, W)
    contents = contents.unsqueeze(2)  # (B, C, 1, H, W), Qwen-Image VAE needs F axis

    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(contents.to(vae.device, dtype=vae.dtype))
        if controls is not None:
            control_latents = [
                [vae.encode_pixels_to_latents(c.to(vae.device, dtype=vae.dtype).unsqueeze(0))[0] for c in cl] for cl in controls
            ]
            # now control_latents is list of list of (C, 1, H, W) tensors
        else:
            control_latents = None

    # debugging: decode and visualize the latents
    """
    import cv2

    for batch_idx in range(latents.shape[0]):

        def decode(lat):
            with torch.no_grad():
                pixels = vae.decode_to_pixels(lat)
            print(pixels.min(), pixels.max(), pixels.shape)
            pixels = pixels.to(torch.float32).cpu()
            pixels = (pixels * 255).clamp(0, 255).to(torch.uint8)  # convert to uint8
            pixels = pixels[0].permute(1, 2, 0)  # C, H, W -> H, W, C
            pixels = pixels.numpy()
            pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)  # RGB to BGR for OpenCV
            return pixels

        lat_img = decode(latents[batch_idx : batch_idx + 1])  # 0 to 1
        ctrl_imgs = [] if control_latents is None else [decode(cl) for cl in control_latents[batch_idx]]  # list of (H, W, C)

        cv2.imshow("Decoded Pixels", lat_img)
        for i, ci in enumerate(ctrl_imgs):
            cv2.imshow(f"Control Decoded Pixels {i}", ci)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    """

    # save cache for each item in the batch
    for b, item in enumerate(batch):
        target_latent = latents[b]  # 1, C, H, W. Target latents for this image (ground truth)
        control_latent = control_latents[b] if control_latents is not None else None  # list of (1, C, H, W) or None

        if is_layered:
            # for layered model, control latents are treated as list of output layers: target_latent is base + layers from control_latent
            original_target_latent = target_latent
            target_latent = torch.concat([target_latent] + control_latent[0], dim=0)  # (num_layers+1, C, H, W)
            
            # (C, num_layers+1, H, W), because saving function expects (C, F, H, W) format (video latents)
            target_latent = target_latent.transpose(0, 1)
            
            control_latent = [original_target_latent]

        print(
            f"Saving cache for item {item.item_key} at {item.latent_cache_path}, target latents shape: {target_latent.shape}, "
            f"control latents shape: {[cl.shape for cl in control_latent] if control_latent is not None else None}"
        )

        save_latent_cache_qwen_image(item_info=item, latent=target_latent, control_latent=control_latent)


def qwen_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    qwen_image_utils.add_model_version_args(parser)
    return parser


def main():
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    parser = qwen_image_setup_parser(parser)

    args = parser.parse_args()
    qwen_image_utils.resolve_model_version_args(args)

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in Qwen-Image.")

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    if args.is_edit:
        architecture = ARCHITECTURE_QWEN_IMAGE_EDIT
    elif args.is_layered:
        architecture = ARCHITECTURE_QWEN_IMAGE_LAYERED
    else:
        architecture = ARCHITECTURE_QWEN_IMAGE

    blueprint = blueprint_generator.generate(user_config, args, architecture=architecture)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "VAE checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    input_channels = 4 if args.is_layered else 3
    vae = qwen_image_utils.load_vae(args.vae, input_channels, device=device, disable_mmap=True)
    vae.to(device)

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch, args.is_layered)

    # reuse core loop from cache_latents with no change
    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
