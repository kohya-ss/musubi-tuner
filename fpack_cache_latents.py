import argparse
import logging
import math
import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import SiglipImageProcessor, SiglipVisionModel

from dataset import config_utils
from dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from dataset.image_video_dataset import BaseDataset, ItemInfo, save_latent_cache_framepack, ARCHITECTURE_FRAMEPACK
from frame_pack import hunyuan
from frame_pack.framepack_utils import load_image_encoders, load_vae
from hunyuan_model.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from frame_pack.clip_vision import hf_clip_vision_encode
import cache_latents

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(
    vae: AutoencoderKLCausal3D,
    feature_extractor: SiglipImageProcessor,
    image_encoder: SiglipVisionModel,
    batch: List[ItemInfo],
    latent_window_size: int,
    vanilla_sampling: bool = False,
):
    """Encode a batch of original RGB videos and save FramePack section caches."""

    # Stack batch into tensor (B,C,F,H,W) in RGB order
    contents = torch.stack([torch.from_numpy(item.content) for item in batch])
    if len(contents.shape) == 4:
        contents = contents.unsqueeze(1)  # B, H, W, C -> B, F, H, W, C

    contents = contents.permute(0, 4, 1, 2, 3).contiguous()  # B, C, F, H, W
    contents = contents.to(vae.device, dtype=vae.dtype)
    contents = contents / 127.5 - 1.0  # normalize to [-1, 1]

    height, width = contents.shape[3], contents.shape[4]
    if height < 8 or width < 8:
        item = batch[0]  # other items should have the same size
        raise ValueError(f"Image or video size too small: {item.item_key} and {len(batch) - 1} more, size: {item.original_size}")

    # calculate latent frame count from original frame count (4n+1)
    latent_f = (batch[0].frame_count - 1) // 4 + 1

    # calculate the total number of sections (excluding the first frame, divided by window size)
    total_latent_sections = math.floor((latent_f - 1) / latent_window_size)
    if total_latent_sections < 1:
        min_frames_needed = latent_window_size * 4 + 1
        raise ValueError(
            f"Not enough frames for FramePack: {batch[0].frame_count} frames ({latent_f} latent frames), minimum required: {min_frames_needed} frames ({latent_window_size+1} latent frames)"
        )

    # 実際に処理する潜在変数のフレーム数 (セクション境界に合わせる)
    latent_f_aligned = total_latent_sections * latent_window_size + 1
    # 実際に処理する元のフレーム数
    frame_count_aligned = (latent_f_aligned - 1) * 4 + 1
    if frame_count_aligned != batch[0].frame_count:
        logger.info(
            f"Frame count mismatch: required={frame_count_aligned} != actual={batch[0].frame_count}, trimming to {frame_count_aligned}"
        )
        contents = contents[:, :, :frame_count_aligned, :, :]

    latent_f = latent_f_aligned  # Update to the aligned value

    # VAE encode (list of tensor -> stack)
    latents = hunyuan.vae_encode(contents, vae)  # include scaling factor
    latents = latents.to("cpu")  # (B, C, latent_f, H/8, W/8)

    # Vision encoding per‑item (once)
    images = np.stack([item.content[0] for item in batch], axis=0)  # B, H, W, C

    # encode image with image encoder
    image_embeddings = []
    with torch.no_grad():
        for image in images:
            image_encoder_output = hf_clip_vision_encode(image, feature_extractor, image_encoder)
            image_embeddings.append(image_encoder_output.last_hidden_state)
    image_embeddings = torch.cat(image_embeddings, dim=0)  # B, LEN, 1152
    image_embeddings = image_embeddings.to("cpu")  # Save memory

    if not vanilla_sampling:
        # padding is reversed for inference (future to past)
        latent_paddings = list(reversed(range(total_latent_sections)))
        # Note: The padding trick for inference. See the paper for details.
        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        for b, item in enumerate(batch):
            original_latent_cache_path = item.latent_cache_path
            video_lat = latents[b : b + 1]  # keep batch dim, 1, C, F, H, W

            # emulate inference step (history latents)
            # Note: In inference, history_latents stores *generated* future latents.
            # Here, for caching, we just need its shape and type for clean_* tensors.
            # The actual content doesn't matter much as clean_* will be overwritten.
            history_latents = torch.zeros(
                (1, video_lat.shape[1], 1 + 2 + 16, video_lat.shape[3], video_lat.shape[4]), dtype=video_lat.dtype
            )  # C=16 for HY

            latent_f_index = latent_f - latent_window_size  # Start from the last section
            section_index = total_latent_sections - 1

            for latent_padding in latent_paddings:
                is_last_section = section_index == 0  # the last section in inference order == the first section in time
                latent_padding_size = latent_padding * latent_window_size
                if is_last_section:
                    assert latent_f_index == 1, "Last section should be starting from frame 1"

                # indices generation (same as inference)
                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                (
                    clean_latent_indices_pre,  # Index for start_latent
                    blank_indices,  # Indices for padding (future context in inference)
                    latent_indices,  # Indices for the target latents to predict
                    clean_latent_indices_post,  # Index for the most recent history frame
                    clean_latent_2x_indices,  # Indices for the next 2 history frames
                    clean_latent_4x_indices,  # Indices for the next 16 history frames
                ) = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)

                # Indices for clean_latents (start + recent history)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # clean latents preparation (emulating inference)
                clean_latents_pre = video_lat[:, :, 0:1, :, :]  # Always the first frame (start_latent)
                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, : 1 + 2 + 16, :, :].split(
                    [1, 2, 16], dim=2
                )
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)  # Combine start frame + placeholder

                # Target latents for this section (ground truth)
                target_latents = video_lat[:, :, latent_f_index : latent_f_index + latent_window_size, :, :]

                # save cache (file path is inside item.latent_cache_path pattern), remove batch dim
                item.latent_cache_path = append_section_idx_to_latent_cache_path(original_latent_cache_path, section_index)
                save_latent_cache_framepack(
                    item_info=item,
                    latent=target_latents.squeeze(0),  # Ground truth for this section
                    latent_indices=latent_indices.squeeze(0),  # Indices for the ground truth section
                    clean_latents=clean_latents.squeeze(0),  # Start frame + history placeholder
                    clean_latent_indices=clean_latent_indices.squeeze(0),  # Indices for start frame + history placeholder
                    clean_latents_2x=clean_latents_2x.squeeze(0),  # History placeholder
                    clean_latent_2x_indices=clean_latent_2x_indices.squeeze(0),  # Indices for history placeholder
                    clean_latents_4x=clean_latents_4x.squeeze(0),  # History placeholder
                    clean_latent_4x_indices=clean_latent_4x_indices.squeeze(0),  # Indices for history placeholder
                    image_embeddings=image_embeddings[b],
                )

                if is_last_section:  # If this was the first section generated in inference (time=0)
                    # History gets the start frame + the generated first section
                    generated_latents_for_history = video_lat[:, :, : latent_window_size + 1, :, :]
                else:
                    # History gets the generated current section
                    generated_latents_for_history = target_latents  # Use true latents as stand-in for generated

                history_latents = torch.cat([generated_latents_for_history, history_latents], dim=2)

                section_index -= 1
                latent_f_index -= latent_window_size

    else:
        # Vanilla Sampling Logic
        for b, item in enumerate(batch):
            original_latent_cache_path = item.latent_cache_path
            video_lat = latents[b : b + 1]  # Keep batch dim: 1, C, F_aligned, H, W
            img_emb = image_embeddings[b]  # LEN, 1152

            for section_index in range(total_latent_sections):
                target_start_f = section_index * latent_window_size + 1
                target_end_f = target_start_f + latent_window_size
                target_latents = video_lat[:, :, target_start_f:target_end_f, :, :]

                # Clean latents preparation (Vanilla)

                # Get clean_latents_pre (Always frame 0)
                clean_latents_pre = video_lat[:, :, 0:1, :, :]

                # Frame indices for past context (relative to anchor)
                idx_post_frame = target_start_f - 1  # Frame index of the last frame of section i-1
                idx_2x_frame_1 = idx_post_frame - 1
                idx_2x_frame_2 = idx_post_frame - 2
                idx_4x_start_frame = idx_post_frame - idx_2x_frame_2 - 16

                # Helper function to get frame or zeros if index is out of bounds
                def get_frame_or_zeros(frame_idx):
                    if frame_idx >= 0:
                        # Ensure frame_idx doesn't exceed the actual length
                        if frame_idx < video_lat.shape[2]:
                            return video_lat[:, :, frame_idx : frame_idx + 1, :, :]
                        else:
                            # This case should ideally not happen if indexing is correct
                            logger.warning(
                                f"Attempted to access frame {frame_idx} beyond latent length {video_lat.shape[2]}. Returning zeros."
                            )
                            return torch.zeros_like(clean_latents_pre)
                    else:
                        return torch.zeros_like(clean_latents_pre)

                # Get clean_latents_post (frame at idx_post_frame)
                clean_latents_post = get_frame_or_zeros(idx_post_frame)

                # Get clean_latents_2x (frames at idx_2x_frame_1, idx_2x_frame_2)
                frame_2x_1 = get_frame_or_zeros(idx_2x_frame_1)
                frame_2x_2 = get_frame_or_zeros(idx_2x_frame_2)
                clean_latents_2x = torch.cat(
                    [frame_2x_2, frame_2x_1], dim=2
                )  # Order might matter (older first?) - assuming order [..., t-2, t-1]

                # Get clean_latents_4x (16 frames ending at idx_4x_start_frame)
                clean_latents_4x_list = []
                for i in range(16):
                    frame_idx = idx_4x_start_frame + i
                    clean_latents_4x_list.append(get_frame_or_zeros(frame_idx))
                clean_latents_4x = torch.cat(clean_latents_4x_list, dim=2)  # Ensure correct temporal order [..., t-18, ..., t-3]

                # Combine pre and post for the main clean_latents input
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)  # (1, C, 2, H, W)

                # Indices generation (Vanilla with Offset)
                vanilla_offset_size = section_index * latent_window_size  # Offset based on section index
                # print(f"Vanilla offset size: {vanilla_offset_size}")

                # Calculate total length including the offset
                total_length = sum([1, vanilla_offset_size, latent_window_size, 1, 2, 16])
                indices = torch.arange(0, total_length).unsqueeze(0)

                # Split indices including the offset part
                (
                    clean_latent_indices_pre,  # Index for frame 0
                    past_offset_indices,  # Indices representing the time offset *before* section i
                    latent_indices,  # Indices for the target latents (section i)
                    clean_latent_indices_post,  # Index for frame from end of section i-1
                    clean_latent_2x_indices,  # Indices for frames from end of section i-2, i-3
                    clean_latent_4x_indices,  # Indices for the 16 past frames
                ) = indices.split([1, vanilla_offset_size, latent_window_size, 1, 2, 16], dim=1)

                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # Save cache
                item.latent_cache_path = append_section_idx_to_latent_cache_path(original_latent_cache_path, section_index)
                save_latent_cache_framepack(
                    item_info=item,
                    latent=target_latents.squeeze(0),
                    latent_indices=latent_indices.squeeze(0),  # Indices for target section i
                    clean_latents=clean_latents.squeeze(0),  # Past clean frames
                    clean_latent_indices=clean_latent_indices.squeeze(0),  # Indices for clean_latents_pre/post
                    clean_latents_2x=clean_latents_2x.squeeze(0),  # Past clean frames (2x)
                    clean_latent_2x_indices=clean_latent_2x_indices.squeeze(0),  # Indices for clean_latents_2x
                    clean_latents_4x=clean_latents_4x.squeeze(0),  # Past clean frames (4x)
                    clean_latent_4x_indices=clean_latent_4x_indices.squeeze(0),  # Indices for clean_latents_4x
                    image_embeddings=img_emb,
                    # Note: We don't explicitly save past_offset_indices,
                    # but its size influences the absolute values in other indices.
                )


def framepack_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--image_encoder", type=str, required=True, help="Image encoder (CLIP) checkpoint path or directory")
    parser.add_argument("--latent_window_size", type=int, default=9, help="FramePack latent window size (default 9)")
    parser.add_argument(
        "--vanilla_sampling",
        action="store_true",
        help="Generate cache for vanilla (autoregressive) sampling instead of inference emulation",
    )
    return parser


def main(args: argparse.Namespace):
    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    # Load dataset config
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FRAMEPACK)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)

    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16
        )
        return

    assert args.vae is not None, "vae checkpoint is required"

    logger.info(f"Loading VAE model from {args.vae}")
    vae = load_vae(args.vae, args.vae_chunk_size, args.vae_spatial_tile_sample_min_size, device=device)
    vae.to(device)

    logger.info(f"Loading image encoder from {args.image_encoder}")
    feature_extractor, image_encoder = load_image_encoders(args)
    image_encoder.eval()
    image_encoder.to(device)

    logger.info(f"Cache generation mode: {'Vanilla Sampling' if args.vanilla_sampling else 'Inference Emulation'}")

    # encoding closure
    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, feature_extractor, image_encoder, batch, args.latent_window_size, args.vanilla_sampling)

    # reuse core loop from cache_latents with no change
    encode_datasets_framepack(datasets, encode, args)


def append_section_idx_to_latent_cache_path(latent_cache_path: str, section_idx: int) -> str:
    tokens = latent_cache_path.split("_")
    tokens[-3] = f"{tokens[-3]}-{section_idx:04d}"  # append section index to "frame_pos-count"
    return "_".join(tokens)


def encode_datasets_framepack(datasets: list[BaseDataset], encode: callable, args: argparse.Namespace):
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() - 1)
    for i, dataset in enumerate(datasets):
        logger.info(f"Encoding dataset [{i}]")
        all_latent_cache_paths = []
        for _, batch in tqdm(dataset.retrieve_latent_cache_batches(num_workers)):
            batch: list[ItemInfo] = batch  # type: ignore

            # latent_cache_path is "{basename}_{w:04d}x{h:04d}_{self.architecture}.safetensors"
            # we expand it to "{basename}_{section_idx:04d}_{w:04d}x{h:04d}_{self.architecture}.safetensors"
            filtered_batch = []
            for item in batch:
                latent_f = (item.frame_count - 1) // 4 + 1
                num_sections = math.floor((latent_f - 1) / args.latent_window_size)
                all_existing = True
                for sec in range(num_sections):
                    p = append_section_idx_to_latent_cache_path(item.latent_cache_path, sec)
                    all_latent_cache_paths.append(p)
                    all_existing = all_existing and os.path.exists(p)

                if all_existing:
                    filtered_batch.append(item)

            if args.skip_existing:
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


if __name__ == "__main__":
    parser = cache_latents.setup_parser_common()
    parser = cache_latents.hv_setup_parser(parser)  # VAE
    parser = framepack_setup_parser(parser)

    args = parser.parse_args()

    if args.vae_dtype is not None:
        raise ValueError("VAE dtype is not supported in FramePack")
    # if args.batch_size != 1:
    #     args.batch_size = 1
    #     logger.info("Batch size is set to 1 for FramePack.")

    main(args)
