import argparse
import os
from typing import Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from PIL import Image

import logging

from musubi_tuner.dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache, ARCHITECTURE_HUNYUAN_VIDEO
from musubi_tuner.hunyuan_model.vae import load_vae
from musubi_tuner.hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def show_image(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],
    control_image: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
) -> int:
    import cv2

    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")

    if control_image is not None:
        if isinstance(control_image, np.ndarray):
            control_image = [control_image]
        for ci in control_image:
            imgs.append(ci)
        print(f"Number of control images: {len(control_image)}")

    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"Image {i + 1} of {len(imgs)}: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        cv2_img = np.array(img) if isinstance(img, Image.Image) else img
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv2_img)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k == ord("q") or k == ord("d"):
            return k
    return k


def show_console(
    image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]],
    width: int,
    back: str,
    interactive: bool = False,
    control_image: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
) -> int:
    from ascii_magic import from_pillow_image, Back

    back = None
    if back is not None:
        back = getattr(Back, back.upper())

    k = None
    imgs = (
        [image]
        if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image)
        else [image[0], image[-1]]
    )
    if len(imgs) > 1:
        print(f"Number of images: {len(image)}")

    if control_image is not None:
        if isinstance(control_image, np.ndarray):
            control_image = [control_image]
        for ci in control_image:
            imgs.append(ci)
        print(f"Number of control images: {len(control_image)}")

    for i, img in enumerate(imgs):
        if len(imgs) > 1:
            print(f"Image {i + 1} of {len(imgs)}: {img.shape}")
        else:
            print(f"Image: {img.shape}")
        pil_img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        ascii_img = from_pillow_image(pil_img)
        ascii_img.to_terminal(columns=width, back=back)

        if interactive:
            k = input("Press q to quit, d to next dataset, other key to next: ")
            if k == "q" or k == "d":
                return ord(k)

    if not interactive:
        return ord(" ")
    return ord(k) if k else ord(" ")


def save_video(image: Union[list[Union[Image.Image, np.ndarray], Union[Image.Image, np.ndarray]]], cache_path: str, fps: int = 24):
    import av

    directory = os.path.dirname(cache_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if (isinstance(image, np.ndarray) and len(image.shape) == 3) or isinstance(image, Image.Image):
        # save image
        image_path = cache_path.replace(".safetensors", ".jpg")
        img = image if isinstance(image, Image.Image) else Image.fromarray(image)
        img.save(image_path)
        print(f"Saved image: {image_path}")
    else:
        imgs = image
        print(f"Number of images: {len(imgs)}")
        # save video
        video_path = cache_path.replace(".safetensors", ".mp4")
        height, width = imgs[0].shape[0:2]

        # create output container
        container = av.open(video_path, mode="w")

        # create video stream
        codec = "libx264"
        pixel_format = "yuv420p"
        stream = container.add_stream(codec, rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = pixel_format
        stream.bit_rate = 1000000  # 1Mbit/s for preview quality

        for frame_img in imgs:
            if isinstance(frame_img, Image.Image):
                frame = av.VideoFrame.from_image(frame_img)
            else:
                frame = av.VideoFrame.from_ndarray(frame_img, format="rgb24")
            packets = stream.encode(frame)
            for packet in packets:
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()

        print(f"Saved video: {video_path}")


def show_datasets(
    datasets: list[BaseDataset],
    debug_mode: str,
    console_width: int,
    console_back: str,
    console_num_images: Optional[int],
    fps: int = 24,
):
    if debug_mode != "video":
        print("d: next dataset, q: quit")

    num_workers = max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        print(f"Dataset [{i}]")
        batch_index = 0
        num_images_to_show = console_num_images
        k = None
        for key, batch in dataset.retrieve_latent_cache_batches(num_workers):
            print(f"bucket resolution: {key}, count: {len(batch)}")
            for j, item_info in enumerate(batch):
                item_info: ItemInfo
                print(f"{batch_index}-{j}: {item_info}")
                if debug_mode == "image":
                    k = show_image(item_info.content, item_info.control_content)
                elif debug_mode == "console":
                    k = show_console(
                        item_info.content, console_width, console_back, console_num_images is None, item_info.control_content
                    )
                    if num_images_to_show is not None:
                        num_images_to_show -= 1
                        if num_images_to_show == 0:
                            k = ord("d")  # next dataset
                elif debug_mode == "video":
                    save_video(item_info.content, item_info.latent_cache_path, fps)
                    k = None  # save next video

                if k == ord("q"):
                    return
                elif k == ord("d"):
                    break
            if k == ord("d"):
                break
            batch_index += 1


def preprocess_contents(batch: list[ItemInfo]) -> tuple[int, int, torch.Tensor]:
    # item.content: target image (H, W, C)
    # item.control_content: list of images (H, W, C)

    # Stack batch into tensor (B,F,H,W,C) in RGB order. The numbers of control content for each item are the same.
    contents = []
    content_masks: list[list[Optional[torch.Tensor]]] = []
    for item in batch:
        item_contents = item.control_content + [item.content]

        item_masks = []
        for i, c in enumerate(item_contents):
            if c.shape[-1] == 4:  # RGBA
                item_contents[i] = c[..., :3]  # remove alpha channel from content

                alpha = c[..., 3]  # extract alpha channel
                mask_image = Image.fromarray(alpha, mode="L")
                width, height = mask_image.size
                mask_image = mask_image.resize((width // 8, height // 8), Image.LANCZOS)
                mask_image = np.array(mask_image)  # PIL to numpy, HWC
                mask_image = torch.from_numpy(mask_image).float() / 255.0  # 0 to 1.0, HWC
                mask_image = mask_image.squeeze(-1)  # HWC -> HW
                mask_image = mask_image.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # HW -> 111HW (BCFHW)
                mask_image = mask_image.to(torch.float32)
                content_mask = mask_image
            else:
                content_mask = None

            item_masks.append(content_mask)

        item_contents = [torch.from_numpy(c) for c in item_contents]
        contents.append(torch.stack(item_contents, dim=0))  # list of [F, H, W, C]
        content_masks.append(item_masks)

    contents = torch.stack(contents, dim=0)  # B, F, H, W, C. F is control frames + target frame

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    height, width = contents.shape[-2], contents.shape[-1]
    if height < 8 or width < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    return height, width, contents, content_masks


def encode_and_save_batch(vae: AutoencoderKLCausal3D, batch: list[ItemInfo]):
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
    with torch.no_grad():
        latent = vae.encode(contents).latent_dist.sample()
        # latent = latent * vae.config.scaling_factor

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

    for item, l in zip(batch, latent):
        # print(f"save latent cache: {item.latent_cache_path}, latent shape: {l.shape}")
        save_latent_cache(item, l)


def encode_datasets(datasets: list[BaseDataset], encode: callable, args: argparse.Namespace, supports_alpha: bool = False):
    """Common function to encode datasets. This function is called from multiple architecture scripts."""
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            batch: list[ItemInfo] = batch
            if not supports_alpha:
                # make sure content has 3 channels
                for item in batch:
                    if isinstance(item.content, np.ndarray):
                        if item.content.shape[-1] == 4:
                            item.content = item.content[..., :3]
                    else:
                        item.content = [img[..., :3] if img.shape[-1] == 4 else img for img in item.content]

            all_latent_cache_paths.extend([item.latent_cache_path for item in batch])

            if args.skip_existing:
                filtered_batch = [item for item in batch if not os.path.exists(item.latent_cache_path)]
                if len(filtered_batch) == 0:
                    continue
                batch = filtered_batch

            bs = args.batch_size if args.batch_size is not None else len(batch)
            for i in range(0, len(batch), bs):
                encode(batch[i : i + bs])

        # normalize paths
        all_latent_cache_paths = [os.path.normpath(p) for p in all_latent_cache_paths]
        all_latent_cache_paths = set(all_latent_cache_paths)

        # remove old cache files not in the dataset
        all_cache_files = dataset.get_all_latent_cache_files()
        for cache_file in all_cache_files:
            if os.path.normpath(cache_file) not in all_latent_cache_paths:
                if args.keep_cache:
                    logger.info(f"Keep cache file not in the dataset: {cache_file}")
                else:
                    os.remove(cache_file)
                    logger.info(f"Removed old cache file: {cache_file}")


def main():
    parser = setup_parser_common()
    parser = hv_setup_parser(parser)

    args = parser.parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_HUNYUAN_VIDEO)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images)
        return

    assert args.vae is not None, "vae checkpoint is required"

    # Load VAE model: HunyuanVideo VAE model is float16
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    logger.info(f"Loaded VAE: {vae.config}, dtype: {vae.dtype}")

    if args.vae_chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(args.vae_chunk_size)
        logger.info(f"Set chunk_size to {args.vae_chunk_size} for CausalConv3d in VAE")
    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    elif args.vae_tiling:
        vae.enable_spatial_tiling(True)

    # Encode images
    def encode(one_batch: list[ItemInfo]):
        encode_and_save_batch(vae, one_batch)

    encode_datasets(datasets, encode, args)


def setup_parser_common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_config", type=str, required=True, help="path to dataset config .toml file")
    parser.add_argument("--vae", type=str, required=False, default=None, help="path to vae checkpoint")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default depends on model, e.g., float16")
    parser.add_argument("--device", type=str, default=None, help="device to use, default is cuda if available")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="batch size, override dataset config if dataset batch size > this"
    )
    parser.add_argument("--num_workers", type=int, default=None, help="number of workers for dataset. default is cpu count-1")
    parser.add_argument("--skip_existing", action="store_true", help="skip existing cache files")
    parser.add_argument("--keep_cache", action="store_true", help="keep cache files not in dataset")
    parser.add_argument("--debug_mode", type=str, default=None, choices=["image", "console", "video"], help="debug mode")
    parser.add_argument("--console_width", type=int, default=80, help="debug mode: console width")
    parser.add_argument(
        "--console_back", type=str, default=None, help="debug mode: console background color, one of ascii_magic.Back"
    )
    parser.add_argument(
        "--console_num_images",
        type=int,
        default=None,
        help="debug mode: not interactive, number of images to show for each dataset",
    )
    parser.add_argument(
        "--disable_cudnn_backend", action="store_true", help="Disable CUDNN PyTorch backend. May be useful for AMD GPUs."
    )
    return parser


def hv_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--vae_tiling",
        action="store_true",
        help="enable spatial tiling for VAE, default is False. If vae_spatial_tile_sample_min_size is set, this is automatically enabled",
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    return parser


if __name__ == "__main__":
    main()
