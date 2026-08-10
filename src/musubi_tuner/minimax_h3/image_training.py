from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Mapping, Sequence

import numpy as np
from PIL import Image

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.bucket import BucketSelector
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo, VideoDataset
from musubi_tuner.minimax_h3.constants import H3_IMAGE_MODES, validate_h3_frame_count


logger = logging.getLogger(__name__)


def _round_down_to_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(value // multiple) * multiple)


def fit_size_to_max_pixels(width: int, height: int, max_pixels: int | None) -> tuple[int, int]:
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / float(width * height))
    return _round_down_to_multiple(width * scale), _round_down_to_multiple(height * scale)


def read_h3_text_image(path: str | Path, max_pixels: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        target_size = fit_size_to_max_pixels(image.width, image.height, max_pixels)
        if target_size != image.size:
            logger.info("Downscaling MiniMax-H3 text visual %s from %sx%s to %sx%s", path, image.width, image.height, *target_size)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


def h3_image_frame_count_for_item(item: ItemInfo, cli_frame_count: int | None) -> int:
    requested = cli_frame_count if cli_frame_count is not None else item.h3_image_frame_count
    return validate_h3_frame_count(5 if requested is None else requested)


def make_h3_image_cache_paths(
    item: ItemInfo,
    frame_count: int,
    architecture: str = ARCHITECTURE_MINIMAX_H3,
) -> tuple[str, str]:
    frame_count = validate_h3_frame_count(frame_count)
    existing_cache_path = item.latent_cache_path or item.text_encoder_output_cache_path
    if existing_cache_path is None:
        raise ValueError("MiniMax-H3 image cache item is missing its cache directory")
    cache_directory = Path(existing_cache_path).parent
    base = re.sub(r"_\d{5}-\d{3}$", "", Path(item.item_key).stem)
    width, height = item.original_size
    if width <= 0 or height <= 0:
        raise ValueError(f"MiniMax-H3 image cache item has invalid original size {item.original_size}")
    frame_token = f"00000-{frame_count:03d}"
    image_size = f"{width:04d}x{height:04d}"
    latent_path = cache_directory / f"{base}_{frame_token}_{image_size}_{architecture}.safetensors"
    text_path = cache_directory / f"{base}_{frame_token}_{architecture}_te.safetensors"
    return str(latent_path), str(text_path)


def configure_h3_image_item(item: ItemInfo, frame_count: int) -> None:
    frame_count = validate_h3_frame_count(frame_count)
    item.frame_count = frame_count
    if len(item.bucket_size) == 2:
        item.bucket_size = (*item.bucket_size, frame_count)
    item.latent_cache_path, item.text_encoder_output_cache_path = make_h3_image_cache_paths(item, frame_count)
    path = Path(item.item_key)
    if not re.fullmatch(r".*_\d{5}-\d{3}", path.stem):
        item.item_key = str(path.with_name(f"{path.stem}_00000-{frame_count:03d}{path.suffix}"))


def validate_h3_cache_datasets(datasets: Sequence, image_mode: str, *, operation: str) -> None:
    if image_mode not in H3_IMAGE_MODES:
        raise ValueError(f"Unsupported MiniMax-H3 image mode: {image_mode}")
    expected_type = ImageDataset if image_mode != "none" else VideoDataset
    if not all(isinstance(dataset, expected_type) for dataset in datasets):
        expected_label = "image" if image_mode != "none" else "video"
        raise ValueError(f"MiniMax-H3 {operation} with image mode {image_mode!r} requires only {expected_label} datasets")
    if image_mode == "none":
        return

    expected_controls = 1 if image_mode == "first" else 2
    for dataset in datasets:
        if dataset.no_resize_control or dataset.control_resolution is not None:
            raise ValueError(
                "MiniMax-H3 image training requires controls to use the target bucket geometry; "
                "no_resize_control and control_resolution are not supported"
            )
        for index in range(len(dataset.datasource)):
            primary_path, _caption = dataset.datasource.get_caption(index)
            _targets, controls = dataset.datasource.get_media_paths(primary_path)
            if len(controls) != expected_controls:
                raise ValueError(
                    f"MiniMax-H3 image mode {image_mode!r} requires {expected_controls} control image(s) per sample; "
                    f"{primary_path} has {len(controls)}"
                )


@dataclass(frozen=True)
class H3ImageSampleDescriptor:
    primary_path: Path
    target_paths: tuple[Path, ...]
    control_paths: tuple[Path, ...]
    original_size: tuple[int, int]
    target_size: tuple[int, int]
    frame_count: int
    mode: str
    dataset: ImageDataset = field(repr=False, compare=False)

    def fingerprint(self, media_fingerprints: Mapping[Path, str]) -> str:
        def media_entry(role: str, index: int, path: Path) -> dict[str, object]:
            resolved = path.resolve()
            try:
                identity = media_fingerprints[resolved]
            except KeyError as error:
                raise ValueError(f"MiniMax-H3 image sample is missing a media fingerprint for {resolved}") from error
            return {
                "role": role,
                "index": index,
                "path": str(resolved),
                "fingerprint": identity,
            }

        payload = {
            "format": "minimax-h3-image-sample-v1",
            "mode": self.mode,
            "frame_count": self.frame_count,
            "original_size": list(self.original_size),
            "target_size": list(self.target_size),
            "dataset_geometry": {
                "resolution": list(self.dataset.resolution),
                "enable_bucket": bool(self.dataset.enable_bucket),
                "bucket_no_upscale": bool(self.dataset.bucket_no_upscale),
                "control_resize": "target_bucket",
            },
            "targets": [media_entry("target", index, path) for index, path in enumerate(self.target_paths)],
            "controls": [media_entry("control", index, path) for index, path in enumerate(self.control_paths)],
        }
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def describe_h3_image_samples(
    dataset: ImageDataset,
    mode: str,
    cli_frame_count: int | None,
) -> dict[Path, H3ImageSampleDescriptor]:
    frame_count = validate_h3_frame_count(cli_frame_count if cli_frame_count is not None else (dataset.h3_image_frame_count or 5))
    bucket_selector = BucketSelector(dataset.resolution, dataset.enable_bucket, dataset.bucket_no_upscale, dataset.architecture)
    descriptors: dict[Path, H3ImageSampleDescriptor] = {}
    for index in range(len(dataset.datasource)):
        primary, _caption = dataset.datasource.get_caption(index)
        targets, controls = dataset.datasource.get_media_paths(primary)
        target_paths = tuple(Path(path).resolve() for path in targets)
        control_paths = tuple(Path(path).resolve() for path in controls)
        with Image.open(target_paths[0]) as image:
            original_size = image.size
        target_size = bucket_selector.get_bucket_resolution(original_size)
        primary_path = target_paths[0]
        if primary_path in descriptors:
            raise ValueError(f"MiniMax-H3 image dataset contains a duplicate primary path: {primary_path}")
        descriptors[primary_path] = H3ImageSampleDescriptor(
            primary_path=primary_path,
            target_paths=target_paths,
            control_paths=control_paths,
            original_size=original_size,
            target_size=target_size,
            frame_count=frame_count,
            mode=mode,
            dataset=dataset,
        )
    return descriptors


def prepare_h3_image_text_item(
    item: ItemInfo,
    descriptor: H3ImageSampleDescriptor,
    *,
    text_visual_max_pixels: int | None,
) -> None:
    item_path = Path(item.item_key)
    if not item_path.is_absolute() or item_path.resolve() != descriptor.primary_path:
        raise ValueError(f"MiniMax-H3 image text-cache item has a noncanonical key: {item.item_key}")
    item.original_size = descriptor.original_size
    item.bucket_size = descriptor.target_size
    item.h3_image_frame_count = descriptor.frame_count
    item.latent_cache_path = descriptor.dataset.get_latent_cache_path(item)
    item.control_content = [read_h3_text_image(path, text_visual_max_pixels) for path in descriptor.control_paths]
    configure_h3_image_item(item, descriptor.frame_count)
    item.h3_image_mode = descriptor.mode
