from __future__ import annotations

import math
import random
from typing import Any, Optional, Tuple, TYPE_CHECKING

import numpy as np
import torch
from safetensors.torch import load_file

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_FRAMEPACK,
    ARCHITECTURE_FLUX_2_DEV,
    ARCHITECTURE_FLUX_2_KLEIN_4B,
    ARCHITECTURE_FLUX_2_KLEIN_9B,
    ARCHITECTURE_FLUX_KONTEXT,
    ARCHITECTURE_HIDREAM_O1,
    ARCHITECTURE_HUNYUAN_VIDEO,
    ARCHITECTURE_HUNYUAN_VIDEO_1_5,
    ARCHITECTURE_IDEOGRAM4,
    ARCHITECTURE_KANDINSKY5,
    ARCHITECTURE_KREA2,
    ARCHITECTURE_QWEN_IMAGE,
    ARCHITECTURE_QWEN_IMAGE_EDIT,
    ARCHITECTURE_QWEN_IMAGE_LAYERED,
    ARCHITECTURE_WAN,
    ARCHITECTURE_Z_IMAGE,
)
from musubi_tuner.dataset.media_utils import divisible_by
from musubi_tuner.utils.model_utils import remove_dtype_suffix

if TYPE_CHECKING:
    from musubi_tuner.dataset.image_video_dataset import ItemInfo

import logging

logger = logging.getLogger(__name__)


class BucketSelector:
    RESOLUTION_STEPS_HUNYUAN = 16
    RESOLUTION_STEPS_WAN = 16
    RESOLUTION_STEPS_FRAMEPACK = 16
    RESOLUTION_STEPS_FLUX_KONTEXT = 16
    RESOLUTION_STEPS_FLUX_2 = 16
    RESOLUTION_STEPS_QWEN_IMAGE = 16
    RESOLUTION_STEPS_QWEN_IMAGE_EDIT = 16
    RESOLUTION_STEPS_KANDINSKY5 = 16
    RESOLUTION_STEPS_HUNYUAN_VIDEO_1_5 = 16
    RESOLUTION_STEPS_Z_IMAGE = 16
    RESOLUTION_STEPS_HIDREAM_O1 = 32  # pixel patch tokens require divisibility by PATCH_SIZE (32)
    RESOLUTION_STEPS_IDEOGRAM4 = 16
    RESOLUTION_STEPS_KREA2 = 16  # latent f8 (VAE compression 8) x patch 2 = align 16

    ARCHITECTURE_STEPS_MAP = {
        ARCHITECTURE_HUNYUAN_VIDEO: RESOLUTION_STEPS_HUNYUAN,
        ARCHITECTURE_WAN: RESOLUTION_STEPS_WAN,
        ARCHITECTURE_FRAMEPACK: RESOLUTION_STEPS_FRAMEPACK,
        ARCHITECTURE_FLUX_KONTEXT: RESOLUTION_STEPS_FLUX_KONTEXT,
        ARCHITECTURE_FLUX_2_DEV: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_FLUX_2_KLEIN_4B: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_FLUX_2_KLEIN_9B: RESOLUTION_STEPS_FLUX_2,
        ARCHITECTURE_QWEN_IMAGE: RESOLUTION_STEPS_QWEN_IMAGE,
        ARCHITECTURE_QWEN_IMAGE_EDIT: RESOLUTION_STEPS_QWEN_IMAGE_EDIT,
        ARCHITECTURE_QWEN_IMAGE_LAYERED: RESOLUTION_STEPS_QWEN_IMAGE,  # use same steps as Qwen-Image
        ARCHITECTURE_KANDINSKY5: RESOLUTION_STEPS_KANDINSKY5,
        ARCHITECTURE_HUNYUAN_VIDEO_1_5: RESOLUTION_STEPS_HUNYUAN_VIDEO_1_5,
        ARCHITECTURE_Z_IMAGE: RESOLUTION_STEPS_Z_IMAGE,
        ARCHITECTURE_HIDREAM_O1: RESOLUTION_STEPS_HIDREAM_O1,
        ARCHITECTURE_IDEOGRAM4: RESOLUTION_STEPS_IDEOGRAM4,
        ARCHITECTURE_KREA2: RESOLUTION_STEPS_KREA2,
    }

    def __init__(
        self, resolution: Tuple[int, int], enable_bucket: bool = True, no_upscale: bool = False, architecture: str = "no_default"
    ):
        self.resolution = resolution
        self.bucket_area = resolution[0] * resolution[1]
        self.architecture = architecture

        if architecture in BucketSelector.ARCHITECTURE_STEPS_MAP:
            self.reso_steps = BucketSelector.ARCHITECTURE_STEPS_MAP[architecture]
        else:
            raise ValueError(f"Invalid architecture: {architecture}")

        if not enable_bucket:
            # only define one bucket
            self.bucket_resolutions = [resolution]
            self.no_upscale = False
        else:
            # prepare bucket resolution
            self.no_upscale = no_upscale
            sqrt_size = int(math.sqrt(self.bucket_area))
            min_size = divisible_by(sqrt_size // 2, self.reso_steps)
            self.bucket_resolutions = []
            for w in range(min_size, sqrt_size + self.reso_steps, self.reso_steps):
                h = divisible_by(self.bucket_area // w, self.reso_steps)
                self.bucket_resolutions.append((w, h))
                self.bucket_resolutions.append((h, w))

            self.bucket_resolutions = list(set(self.bucket_resolutions))
            self.bucket_resolutions.sort()

        # calculate aspect ratio to find the nearest resolution
        self.aspect_ratios = np.array([w / h for w, h in self.bucket_resolutions])

    def get_bucket_resolution(self, image_size: tuple[int, int]) -> tuple[int, int]:
        """
        return the bucket resolution for the given image size, (width, height)
        """
        area = image_size[0] * image_size[1]
        if self.no_upscale and area <= self.bucket_area:
            w, h = image_size
            w = divisible_by(w, self.reso_steps)
            h = divisible_by(h, self.reso_steps)
            return w, h

        aspect_ratio = image_size[0] / image_size[1]
        ar_errors = self.aspect_ratios - aspect_ratio
        bucket_id = np.abs(ar_errors).argmin()
        return self.bucket_resolutions[bucket_id]

    @classmethod
    def calculate_bucket_resolution(
        cls,
        image_size: tuple[int, int],
        resolution: tuple[int, int],
        reso_steps: Optional[int] = None,
        architecture: Optional[str] = None,
    ) -> tuple[int, int]:
        """
        Get the bucket resolution for the given image size, resolution and resolution steps.
        Return (width, height).
        """
        if reso_steps is None and architecture is None:
            raise ValueError("resolution steps or architecture must be provided")
        if reso_steps is None and architecture is not None:
            if architecture in BucketSelector.ARCHITECTURE_STEPS_MAP:
                reso_steps = BucketSelector.ARCHITECTURE_STEPS_MAP[architecture]
            else:
                raise ValueError(f"Invalid architecture: {architecture}")

        max_area = resolution[0] * resolution[1]
        width, height = image_size
        aspect_ratio = width / height
        bucket_width = int(math.sqrt(max_area * aspect_ratio))
        bucket_height = int(math.sqrt(max_area / aspect_ratio))
        bucket_width = divisible_by(bucket_width, reso_steps)
        bucket_height = divisible_by(bucket_height, reso_steps)

        # find appropriate resolutions
        best_resolution = None
        best_aspect_ratio_diff = float("inf")
        for i in range(-2, 3):
            w = bucket_width + i * reso_steps
            h = divisible_by(max_area // w, reso_steps)
            current_aspect_ratio_diff = abs((w / h) - aspect_ratio)
            if current_aspect_ratio_diff < best_aspect_ratio_diff:
                best_aspect_ratio_diff = current_aspect_ratio_diff
                best_resolution = (w, h)

        if best_resolution is not None:
            return best_resolution

        return bucket_width, bucket_height


class BucketBatchManager:
    def __init__(
        self, bucketed_item_info: dict[tuple[Any], list[ItemInfo]], batch_size: int, num_timestep_buckets: Optional[int] = None
    ):
        self.batch_size = batch_size
        self.buckets = bucketed_item_info
        self.bucket_resos = list(self.buckets.keys())
        self.bucket_resos.sort()
        self.num_timestep_buckets = num_timestep_buckets
        self.timestep_pool = None

        # indices for enumerating batches. each batch is reso + batch_idx. reso is (width, height) or (width, height, frames)
        self.bucket_batch_indices: list[tuple[tuple[Any], int]] = []
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            num_batches = math.ceil(len(bucket) / self.batch_size)
            for i in range(num_batches):
                self.bucket_batch_indices.append((bucket_reso, i))

        # do no shuffle here to avoid multiple datasets have different order
        # self.shuffle()

    def show_bucket_info(self):
        for bucket_reso in self.bucket_resos:
            bucket = self.buckets[bucket_reso]
            logger.info(f"bucket: {bucket_reso}, count: {len(bucket)}")

        logger.info(f"total batches: {len(self)}")

    def shuffle(self):
        # shuffle each bucket
        for bucket in self.buckets.values():
            random.shuffle(bucket)

        # shuffle the order of batches
        random.shuffle(self.bucket_batch_indices)

        if self.num_timestep_buckets is not None and self.num_timestep_buckets > 1:
            # prepare timesteps for each timestep buckets

            # 1. Calculate total number of timesteps needed for the entire epoch
            num_batches = len(self.bucket_batch_indices)
            total_timesteps_needed = num_batches * self.batch_size

            # 2. Generate a single large pool of stratified timesteps
            all_timesteps = []
            samples_per_bucket = math.ceil(total_timesteps_needed / self.num_timestep_buckets)

            for i in range(self.num_timestep_buckets):
                min_t = i / self.num_timestep_buckets
                max_t = (i + 1) / self.num_timestep_buckets
                for _ in range(samples_per_bucket):
                    all_timesteps.append(random.uniform(min_t, max_t))

            # 3. Shuffle the entire pool thoroughly
            random.shuffle(all_timesteps)

            # Trim the excess timesteps to match the exact number needed
            all_timesteps = all_timesteps[:total_timesteps_needed]

            # 4. Create the final timestep pool by chunking the shuffled list
            self.timestep_pool = []
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                self.timestep_pool.append(all_timesteps[start_idx:end_idx])
                # print(f"timestep pool {i}: {self.timestep_pool[-1]}")

    def __len__(self):
        return len(self.bucket_batch_indices)

    def __getitem__(self, idx):
        bucket_reso, batch_idx = self.bucket_batch_indices[idx]
        bucket = self.buckets[bucket_reso]
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, len(bucket))

        batch_tensor_data = {}
        varlen_keys = set()
        for item_info in bucket[start:end]:
            sd_latent = load_file(item_info.latent_cache_path)
            sd_te = load_file(item_info.text_encoder_output_cache_path)
            sd = {**sd_latent, **sd_te}

            # TODO refactor this
            for key in sd.keys():
                is_varlen_key = key.startswith("varlen_")  # varlen keys are not stacked
                content_key = key

                if is_varlen_key:
                    content_key = content_key.replace("varlen_", "")

                if content_key.endswith("_mask"):
                    pass
                else:
                    content_key = remove_dtype_suffix(content_key)  # remove dtype (handles e.g. float8_e4m3fn)
                    if content_key.startswith("latents_"):
                        content_key = content_key.rsplit("_", 1)[0]  # remove FxHxW

                if content_key not in batch_tensor_data:
                    batch_tensor_data[content_key] = []
                batch_tensor_data[content_key].append(sd[key])

                if is_varlen_key:
                    varlen_keys.add(content_key)

        for key in batch_tensor_data.keys():
            if key not in varlen_keys:
                batch_tensor_data[key] = torch.stack(batch_tensor_data[key])

        if self.timestep_pool is not None:
            batch_tensor_data["timesteps"] = self.timestep_pool[idx][: end - start]  # use the pre-generated timesteps
        else:
            batch_tensor_data["timesteps"] = None

        return batch_tensor_data
