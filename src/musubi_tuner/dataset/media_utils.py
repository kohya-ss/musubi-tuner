from __future__ import annotations

from bisect import bisect_left
import glob
from importlib.util import find_spec
import math
import os
from typing import Optional, Union, TYPE_CHECKING

import numpy as np
from PIL import Image
import cv2
import av

if TYPE_CHECKING:
    from musubi_tuner.dataset.bucket import BucketSelector

import logging

logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP", ".avif", ".AVIF"]


if find_spec("jxlpy") is not None:  # JPEG-XL on Linux
    from jxlpy import JXLImagePlugin  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

if find_spec("pillow_jxl") is not None:  # JPEG-XL on Windows
    import pillow_jxl  # noqa: F401 # type: ignore

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])

VIDEO_EXTENSIONS = [
    ".mp4",
    ".webm",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".MP4",
    ".WEBM",
    ".AVI",
    ".MKV",
    ".MOV",
    ".FLV",
    ".WMV",
    ".M4V",
    ".MPG",
    ".MPEG",
]  # some of them are not tested


def glob_images(directory, base="*", caption_extension=None):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # remove duplicates

    # check for caption files and only keep images with captions
    if caption_extension is not None:
        caption_paths = glob.glob(os.path.join(glob.escape(directory), "*" + caption_extension))
        caption_bases = set()
        for caption_path in caption_paths:
            caption_base = os.path.splitext(os.path.basename(caption_path))[0]
            caption_bases.add(caption_base)
        filtered_img_paths = []
        for img_path in img_paths:
            img_base = os.path.splitext(os.path.basename(img_path))[0]
            if img_base in caption_bases:
                filtered_img_paths.append(img_path)
        img_paths = filtered_img_paths

    img_paths.sort()
    return img_paths


def glob_videos(directory, base="*"):
    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        if base == "*":
            video_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            video_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    video_paths = list(set(video_paths))  # remove duplicates
    video_paths.sort()
    return video_paths


def divisible_by(num: int, divisor: int) -> int:
    return num - num % divisor


def resize_image_to_bucket(image: Union[Image.Image, np.ndarray], bucket_reso: tuple[int, int]) -> np.ndarray:
    """
    Resize the image to the bucket resolution.

    bucket_reso: **(width, height)**
    """
    is_pil_image = isinstance(image, Image.Image)
    if is_pil_image:
        image_width, image_height = image.size
    else:
        image_height, image_width = image.shape[:2]

    if bucket_reso == (image_width, image_height):
        return np.array(image) if is_pil_image else image

    bucket_width, bucket_height = bucket_reso

    # resize the image to the bucket resolution to match the short side
    scale_width = bucket_width / image_width
    scale_height = bucket_height / image_height
    scale = max(scale_width, scale_height)
    image_width = int(image_width * scale + 0.5)
    image_height = int(image_height * scale + 0.5)

    if scale > 1:
        image = Image.fromarray(image) if not is_pil_image else image
        image = image.resize((image_width, image_height), Image.LANCZOS)
        image = np.array(image)
    else:
        image = np.array(image) if is_pil_image else image
        image = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_AREA)

    # crop the image to the bucket resolution
    crop_left = (image_width - bucket_width) // 2
    crop_top = (image_height - bucket_height) // 2
    image = image[crop_top : crop_top + bucket_height, crop_left : crop_left + bucket_width]
    return image


def resample_frame_indices(
    timestamps: list[float],
    *,
    source_frame_duration: float,
    target_fps: float,
) -> list[int]:
    """Maps decoded frame timestamps to nearest-frame indices on a fixed target-fps grid.

    Used by fps_resample_mode="timestamps" to normalize videos of any (possibly variable)
    frame rate to exactly target_fps, so that audio/video alignment is deterministic.
    """
    if not timestamps:
        return []
    if source_frame_duration <= 0 or target_fps <= 0:
        raise ValueError("Video frame durations and target FPS must be positive")
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Video timestamps must be nondecreasing")

    origin = timestamps[0]
    normalized = [timestamp - origin for timestamp in timestamps]
    duration = normalized[-1] + source_frame_duration
    target_count = max(1, math.ceil(duration * target_fps - 1e-9))
    indices = []
    for target_index in range(target_count):
        target_time = target_index / target_fps
        right = bisect_left(normalized, target_time)
        if right == 0:
            source_index = 0
        elif right == len(normalized):
            source_index = len(normalized) - 1
        else:
            left = right - 1
            left_distance = target_time - normalized[left]
            right_distance = normalized[right] - target_time
            source_index = left if left_distance <= right_distance + 1e-12 else right
        indices.append(source_index)
    return indices


def _load_video_timestamp_resampled(
    video_path: str,
    target_fps: float,
    start_frame: Optional[int],
    end_frame: Optional[int],
    bucket_selector: Optional[BucketSelector],
    bucket_reso: Optional[tuple[int, int]],
) -> list[np.ndarray]:
    if not os.path.isfile(video_path):
        raise ValueError(f"fps_resample_mode='timestamps' requires a video file, not a directory: {video_path}")

    with av.open(video_path) as container:
        if not container.streams.video:
            raise ValueError(f"Video source has no video stream: {video_path}")
        stream = container.streams.video[0]
        average_rate = float(stream.average_rate) if stream.average_rate is not None else target_fps
        source_frame_duration = 1.0 / average_rate if average_rate > 0 else 1.0 / target_fps
        frames = []
        timestamps = []
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                timestamp = float(frame.pts * frame.time_base)
            else:
                timestamp = index * source_frame_duration
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(timestamp)
    if not frames:
        raise ValueError(f"Video source decoded no frames: {video_path}")

    indices = resample_frame_indices(timestamps, source_frame_duration=source_frame_duration, target_fps=target_fps)
    indices = indices[slice(start_frame, end_frame)]

    video = []
    for index in indices:
        frame = frames[index]
        if bucket_selector is not None and bucket_reso is None:
            bucket_reso = bucket_selector.get_bucket_resolution((frame.shape[1], frame.shape[0]))
        if bucket_reso is not None:
            frame = resize_image_to_bucket(frame, bucket_reso)
        video.append(frame)
    return video


def load_video(
    video_path: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    bucket_selector: Optional[BucketSelector] = None,
    bucket_reso: Optional[tuple[int, int]] = None,
    source_fps: Optional[float] = None,
    target_fps: Optional[float] = None,
    fps_resample_mode: Optional[str] = None,
) -> list[np.ndarray]:
    """
    bucket_reso: if given, resize the video to the bucket resolution, (width, height)
    fps_resample_mode: None (legacy source_fps/target_fps frame dropping) or "timestamps"
        (PTS-based nearest-frame resampling to target_fps regardless of source fps)
    """
    if fps_resample_mode is not None:
        if fps_resample_mode != "timestamps":
            raise ValueError(f"Unsupported fps_resample_mode: {fps_resample_mode}")
        if target_fps is None:
            raise ValueError("fps_resample_mode='timestamps' requires target_fps")
        if source_fps is not None:
            raise ValueError("fps_resample_mode='timestamps' does not use source_fps")
        return _load_video_timestamp_resampled(video_path, target_fps, start_frame, end_frame, bucket_selector, bucket_reso)

    if source_fps is None or target_fps is None:
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            for i, frame in enumerate(container.decode(video=0)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            for i in range(len(image_files)):
                if start_frame is not None and i < start_frame:
                    continue
                if end_frame is not None and i >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)
    else:
        # drop frames to match the target fps TODO commonize this code with the above if this works
        frame_index_delta = target_fps / source_fps  # example: 16 / 30 = 0.5333
        if os.path.isfile(video_path):
            container = av.open(video_path)
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i, frame in enumerate(container.decode(video=0)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break
                frame = frame.to_image()

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(frame.size)  # calc resolution from first frame

                if bucket_reso is not None:
                    frame = resize_image_to_bucket(frame, bucket_reso)
                else:
                    frame = np.array(frame)

                video.append(frame)
            container.close()
        else:
            # load images in the directory
            image_files = glob_images(video_path)
            image_files.sort()
            video = []
            frame_index_with_fraction = 0.0
            previous_frame_index = -1
            for i in range(len(image_files)):
                target_frame_index = int(frame_index_with_fraction)
                frame_index_with_fraction += frame_index_delta

                if target_frame_index == previous_frame_index:  # drop this frame
                    continue

                # accept this frame
                previous_frame_index = target_frame_index

                if start_frame is not None and target_frame_index < start_frame:
                    continue
                if end_frame is not None and target_frame_index >= end_frame:
                    break

                image_file = image_files[i]
                image = Image.open(image_file).convert("RGB")

                if bucket_selector is not None and bucket_reso is None:
                    bucket_reso = bucket_selector.get_bucket_resolution(image.size)  # calc resolution from first frame
                image = np.array(image)
                if bucket_reso is not None:
                    image = resize_image_to_bucket(image, bucket_reso)

                video.append(image)

    return video
