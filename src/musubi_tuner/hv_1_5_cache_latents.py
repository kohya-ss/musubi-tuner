import argparse
from typing import Optional

import torch

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HUNYUAN_VIDEO_1_5, ItemInfo, save_latent_cache_hunyuan_video_1_5
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.frame_pack.framepack_utils import load_image_encoders
from musubi_tuner.hunyuan_video_1_5 import hunyuan_video_1_5_vae
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_vae import AutoencoderKLConv3D
from musubi_tuner.utils.model_utils import str_to_dtype
import musubi_tuner.cache_latents as cache_latents

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    vae: AutoencoderKLConv3D, image_encoder_assets: Optional[tuple], batch: list[ItemInfo], i2v: bool = False
):
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    h, w = contents.shape[3], contents.shape[4]
    if h < 16 or w < 16:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    # VAE requires a lot of VRAM, so process one by one.
    latents_list = []
    for i in range(contents.shape[0]):
        content = contents[i : i + 1, :, :, :, :]  # 1, C, F, H, W
        with torch.autocast(device_type=vae.device.type, dtype=vae.dtype, enabled=True), torch.no_grad():
            latent = vae.encode(content)[0].mode()
            # latent = latent * vae.scaling_factor  # no scaling here, saved in VAE latent space directly
        latents_list.append(latent)
    latents = torch.cat(latents_list, dim=0)  # B, C, F, H, W

    cond_latents_list = None
    vision_features = None
    if i2v:
        # extract first frame of contents
        images = contents[:, :, 0:1, :, :]  # B, C, 1, H, W. normalized.
        lat_f, lat_h, lat_w = latents.shape[2], latents.shape[3], latents.shape[4]

        # make i2v cond_latents: 1 frame latent + mask channel.
        cond_latents_list = []
        vision_features = []
        for i in range(images.shape[0]):
            first_frame = images[i : i + 1, :, 0:1, :, :]  # 1, C, 1, H, W. normalized.
            with torch.autocast(device_type=vae.device.type, dtype=vae.dtype, enabled=True), torch.no_grad():
                cond_latents = vae.encode(first_frame)[0].mode()
                # cond_latents = cond_latents * vae.scaling_factor  # no scaling here, saved in VAE latent space directly

                latents_concat = torch.zeros(
                    1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS, lat_f, lat_h, lat_w, dtype=torch.float32, device=vae.device
                )
                latents_concat[:, :, 0:1, :, :] = cond_latents

                # latent_mask = torch.zeros(lat_f, device=vae.device)
                # latent_mask[0] = 1.0
                # mask_concat = torch.ones(1, 1, lat_f, lat_h, lat_w, device=vae.device) * latent_mask[None, None, :, None, None]
                mask_concat = torch.zeros(1, 1, lat_f, lat_h, lat_w, device=vae.device)
                mask_concat[:, :, 0:1, :, :] = 1.0

                cond_latents = torch.concat([latents_concat, mask_concat], dim=1)  # 1, C+1, F, H, W
                cond_latents_list.append(cond_latents[0])  # remove batch dim

            # extract vision feature from first frame
            first_frame_np = batch[i].content[0]  # H, W, C, uint8
            feature_extractor, image_encoder = image_encoder_assets
            with torch.no_grad():
                vision_feature = hf_clip_vision_encode(first_frame_np, feature_extractor, image_encoder)
                image_encoder_last_hidden_state = vision_feature.last_hidden_state  # float16
            vision_features.append(image_encoder_last_hidden_state[0])  # remove batch dim

    for i, item in enumerate(batch):
        latent = latents[i]
        cond_latent = None if cond_latents_list is None else cond_latents_list[i]
        vision_feature = None if vision_features is None else vision_features[i]
        save_latent_cache_hunyuan_video_1_5(item, latent, cond_latent, vision_feature)


def main():
    parser = cache_latents.setup_parser_common()
    parser = hv1_5_setup_parser(parser)

    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    if args.i2v:
        assert args.image_encoder is not None, "--i2v requires --image_encoder to be set."
    elif args.image_encoder is not None:
        logger.info("--image_encoder is set but --i2v is not set. Enabling --i2v.")
        args.i2v = True

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HUNYUAN_VIDEO_1_5)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "vae checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae = hunyuan_video_1_5_vae.load_vae_from_checkpoint(
        args.vae, device, vae_dtype, sample_size=args.vae_sample_size, enable_patch_conv=args.vae_enable_patch_conv
    )
    vae.eval()

    if args.i2v:
        feature_extractor, image_encoder = load_image_encoders(args)
        image_encoder.to(device)
        image_encoder_assets = (feature_extractor, image_encoder)
    else:
        image_encoder_assets = None

    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, image_encoder_assets, one_batch, args.i2v)

    cache_latents.encode_datasets(datasets, encode, args)


def hv1_5_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--vae_sample_size",
        type=int,
        default=128,
        help="VAE sample size (height/width). Default 128; set 256 if VRAM is sufficient for better quality; set 0 to disable tiling.",
    )
    parser.add_argument(
        "--vae_enable_patch_conv",
        action="store_true",
        help="Enable patch-based convolution in VAE for memory optimization",
    )
    parser.add_argument(
        "--i2v",
        action="store_true",
        help="Cache image features and conditional latents for I2V training/inference",
    )
    parser.add_argument(
        "--image_encoder", type=str, default=None, help="Directory/path of SigLIP Image Encoder (required if --i2v is set)"
    )
    return parser


if __name__ == "__main__":
    main()
