import argparse
from typing import Optional

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer

import logging

from musubi_tuner.dataset.image_video_dataset import ItemInfo, save_latent_cache_wan, ARCHITECTURE_WAN
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.wan.configs import wan_i2v_14B
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.wan.modules.clip import CLIPModel
import musubi_tuner.cache_latents as cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

black_image_latents = {}  # global variable for black image latent, used in encode_and_save_batch_one_frame. key: tuple for shape


def encode_and_save_batch(vae: WanVAE, clip: Optional[CLIPModel], i2v: bool, batch: list[ItemInfo], one_frame: bool = False):
    if one_frame:
        encode_and_save_batch_one_frame(vae, clip, batch)
        return

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

    if i2v:
        # extract first frame of contents
        images = contents[:, :, 0:1, :, :]  # B, C, F, H, W, non contiguous view is fine

        if clip is not None:
            with torch.amp.autocast(device_type=clip.device.type, dtype=torch.float16), torch.no_grad():
                clip_context = clip.visual(images)
            clip_context = clip_context.to(torch.float16)  # convert to fp16
        else:
            clip_context = None

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

    # control videos/images
    if batch[0].control_content is not None:
        # Check if control_content is a list (for images) or ndarray (for videos)
        if isinstance(batch[0].control_content, list):
            # For images with control images: control_content is list[np.ndarray]
            # We take the first control image from each item
            control_contents = torch.stack([torch.from_numpy(item.control_content[0]) for item in batch])
        else:
            # For videos with control videos: control_content is np.ndarray
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
        y_i = y[i] if i2v else None
        control_latent_i = control_latent[i] if control_latent is not None else None
        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache_wan(item, l, cctx, y_i, control_latent_i)


def encode_and_save_batch_one_frame(vae: WanVAE, clip: Optional[CLIPModel], batch: list[ItemInfo]):
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)
    assert clip is not None, "clip is required for one frame training"

    # contents: control_content + content
    _, _, contents, content_masks = cache_latents.preprocess_contents(batch)
    contents = contents.to(vae.device, dtype=vae.dtype)  # B, C, F, H, W
    assert contents.shape[2] >= 2, "One frame training requires at least 1 control frame and 1 target frame"

    # print(f"encode batch: {contents.shape}")
    with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
        # VAE encode: we need to encode one frame at a time because VAE encoder has stride=4 for the time dimension except for the first frame.
        latent = []
        for bi in range(contents.shape[0]):
            c = contents[bi : bi + 1]  # B, C, F, H, W, b=1
            l = []
            for f in range(c.shape[2]):  # iterate over frames
                cf = c[:, :, f : f + 1, :, :]  # B, C, 1, H, W
                l.append(vae.encode(cf)[0].unsqueeze(0))  # list of [C, 1, H, W] to [1, C, 1, H, W]
            latent.append(torch.cat(l, dim=2))  # B, C, F, H, W
        latent = torch.cat(latent, dim=0)  # B, C, F, H, W

    latent = latent.to(vae.dtype)  # convert to bfloat16, we are not sure if this is correct
    control_latent = latent[:, :, :-1, :, :]
    target_latent = latent[:, :, -1:, :, :]

    # Create black image latent for the target frame
    global black_image_latents
    shape = (1, contents.shape[1], 1, contents.shape[3], contents.shape[4])  # B=1, C, F=1, H, W
    if shape not in black_image_latents:
        with torch.amp.autocast(device_type=vae.device.type, dtype=vae.dtype), torch.no_grad():
            black_image_latent = vae.encode(torch.zeros(shape, device=vae.device, dtype=vae.dtype))[0]
        black_image_latent = black_image_latent.to(device="cpu", dtype=vae.dtype)
        black_image_latents[shape] = black_image_latent  # store for future use
    black_image_latent = black_image_latents[shape]  # [C, 1, H, W]

    # Vision encoding per‑item (once): use first content (first control content) because it is the start image
    num_control_images = contents.shape[2] - 1  # number of control images
    if num_control_images > 2:
        logger.error(f"One frame training requires 1 or 2 control images, but found {num_control_images} in {batch[0].item_key}. ")
        raise ValueError(
            f"One frame training requires 1 or 2 control images, but found {num_control_images} in {batch[0].item_key}."
        )

    images = contents[:, :, 0:num_control_images, :, :]  # B, C, F, H, W
    clip_context = []
    for i in range(images.shape[0]):
        with torch.amp.autocast(device_type=clip.device.type, dtype=torch.float16), torch.no_grad():
            clip_context.append(clip.visual(images[i : i + 1]))
    clip_context = torch.stack(clip_context, dim=0)  # B, num_control_images, N, D
    clip_context = clip_context.to(torch.float16)  # convert to fp16

    B, C, _, lat_h, lat_w = latent.shape
    for i, item in enumerate(batch):
        latent = target_latent[i]  # C, 1, H, W
        F = contents.shape[2]  # number of frames
        y = torch.zeros((4 + C, F, lat_h, lat_w), dtype=vae.dtype, device=vae.device)  # conditioning
        l = torch.zeros((C, F, lat_h, lat_w), dtype=vae.dtype, device=vae.device)  # training latent

        # Create latent and mask for the required number of frames
        control_latent_indices = item.fp_1f_clean_indices
        target_and_control_latent_indices = control_latent_indices + [item.fp_1f_target_index]
        f_indices = sorted(target_and_control_latent_indices)

        ci = 0
        for j, index in enumerate(f_indices):
            if index == item.fp_1f_target_index:
                # print(f"Set target latent. latent shape: {latent.shape}, black_image_latent shape: {black_image_latent.shape}")
                y[4:, j : j + 1, :, :] = black_image_latent
                l[:, j : j + 1, :, :] = latent  # set target latent
            else:
                # print(f"Set control latent. control_latent shape: {control_latent[i, :, ci, :, :].shape}")
                y[:4, j, :, :] = 1.0  # set mask to 1.0 for the clean latent frames
                y[4:, j, :, :] = control_latent[i, :, ci, :, :]  # set control latent
                l[:, j, :, :] = control_latent[i, :, ci, :, :]  # also set control latent to training latent
                ci += 1  # increment control latent index

        cctx = clip_context[i]

        logger.info(f"Saving cache for item: {item.item_key} at {item.latent_cache_path}")
        logger.info(f"  control_latent_indices: {control_latent_indices}, fp_1f_target_index: {item.fp_1f_target_index}")
        logger.info(f"  y shape: {y.shape}, mask: {y[0, :, 0, 0]}, l shape: {l.shape}, clip_context shape: {cctx.shape}")
        logger.info(f"  f_indices: {f_indices}")

        save_latent_cache_wan(item, l, cctx, y, None, f_indices=f_indices)


def main():
    parser = cache_latents.setup_parser_common()
    parser = wan_setup_parser(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.clip is not None:
        args.i2v = True

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
        encode_and_save_batch(vae, clip, args.i2v, one_batch, args.one_frame)

    cache_latents.encode_datasets(datasets, encode, args)


def wan_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument(
        "--i2v",
        action="store_true",
        help="I2V model, encode the first frame as control image. If clip is set, this is automatically set to True",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        help="text encoder (CLIP) checkpoint path, optional. If training Wan2.1 I2V model, this is required",
    )
    parser.add_argument(
        "--one_frame",
        action="store_true",
        help="Generate cache for one frame training (single frame, single section).",
    )
    return parser


if __name__ == "__main__":
    main()
