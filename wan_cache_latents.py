import argparse
import os
import glob
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from PIL import Image

import logging

from dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, ARCHITECTURE_WAN
from utils.model_utils import str_to_dtype
from wan.configs import wan_i2v_14B
from wan.modules.vae import WanVAE
from wan.modules.clip import CLIPModel
import cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae: WanVAE, clip: Optional[CLIPModel], batch: list[ItemInfo]):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    h, w = contents.shape[3], contents.shape[4]
    if h < 8 or w < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    # print(f"encode batch: {contents.shape}")
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        latent = vae.encode(contents)  # list of Tensor[C, F, H, W]
    latent = torch.stack(latent, dim=0)  # B, C, F, H, W
    latent = latent.to(vae.dtype)  # convert to bfloat16, we are not sure if this is correct

    if clip is not None:
        # extract first frame of contents
        images = contents[:, :, 0:1, :, :]  # B, C, F, H, W, non contiguous view is fine

        with torch.amp.autocast(device_type=clip.device.type, dtype=torch.float16), torch.no_grad():
            clip_context = clip.visual(images)
        clip_context = clip_context.to(torch.float16)  # convert to fp16

        # encode image latent for I2V
        B, _, _, lat_h, lat_w = latent.shape
        F = contents.shape[2]

        # Create mask for the required number of frames
        msk = torch.ones(1, F, lat_h, lat_w, dtype=vae.dtype, device=vae.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)  # 1, F, 4, H, W -> 1, 4, F, H, W
        msk = msk.repeat(B, 1, 1, 1, 1)  # B, 4, F, H, W

        # Zero padding for the required number of frames only
        padding_frames = F - 1  # The first frame is the input image
        images_resized = torch.concat([images, torch.zeros(B, 3, padding_frames, h, w, device=vae.device)], dim=2)
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            y = vae.encode(images_resized)
        y = torch.stack(y, dim=0)  # B, C, F, H, W

        y = y[:, :, :F]  # may be not needed
        y = y.to(vae.dtype)  # convert to bfloat16
        y = torch.concat([msk, y], dim=1)  # B, 4 + C, F, H, W

    else:
        clip_context = None
        y = None

    # control videos
    if batch[0].control_content is not None:
        control_contents = torch.stack([torch.from_numpy(item.control_content) for item in batch])
        if len(control_contents.shape) == 4:
            control_contents = control_contents.unsqueeze(1)
        control_contents = control_contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
        control_contents = control_contents.to(vae.device, dtype=vae.dtype)
        control_contents = control_contents / 127.5 - 1.0  # normalize to [-1, 1]
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            control_latent = vae.encode(control_contents)  # list of Tensor[C, F, H, W]
        control_latent = torch.stack(control_latent, dim=0)  # B, C, F, H, W
        control_latent = control_latent.to(vae.dtype)  # convert to bfloat16
    else:
        control_latent = None

    # # debug: decode and save
    # with torch.no_grad():
    #     latent_to_decode = latent / vae.config.scaling_factor
    #     images = vae.decode(latent_to_decode, return_dict=False)[0]
    #     images = (images / 2 + 0.5).clamp(0, 1)
    #     images = images.cpu().float().numpy()
    #     images = (images * 255).astype(np.uint8)
    #     images = images.transpose(0, 2, 3, 4, 1)  # B, C, F, H, W -> B, F, H, W, C
    #     for b in range(images.shape[0]):
    #         for f in range(images.shape[1]):
    #             fln = os.path.splitext(os.path.basename(batch[b].item_key))[0]
    #             img = Image.fromarray(images[b, f])
    #             img.save(f"./logs/decode_{fln}_{b}_{f:03d}.jpg")

    for i, item in enumerate(batch):
        l = latent[i]
        cctx = clip_context[i] if clip is not None else None
        y_i = y[i] if clip is not None else None
        control_latent_i = control_latent[i] if control_latent is not None else None
        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache_wan(item, l, cctx, y_i, control_latent_i)


def main(args):
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_WAN)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "vae checkpoint is required"

    vae_path = args.vae

    logger.info(f"Loading VAE model from {vae_path}")
    vae_dtype = torch.bfloat16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device)

    if args.clip is not None:
        clip_dtype = wan_i2v_14B.i2v_14B["clip_dtype"]
        clip = CLIPModel(dtype=clip_dtype, device=device, weight_path=args.clip)
    else:
        clip = None

    # Encode images
    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, clip, one_batch)

    cache_latents.encode_datasets(datasets, encode, args)


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training I2V model, this is required",
    )
    return parser


if __name__ == "__main__":
    parser = cache_latents.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()
    main(args)
