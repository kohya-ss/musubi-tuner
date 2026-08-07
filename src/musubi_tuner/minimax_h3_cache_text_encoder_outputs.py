from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_minimax_h3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ImageDataset, ItemInfo, VideoDataset
from musubi_tuner.minimax_h3.text_encoder import (
    DEFAULT_PROCESSOR_ID,
    H3TextVisual,
    MAX_TEXT_ROWS,
    TEXT_WIDTH,
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    presentation_fingerprint,
    processor_fingerprint,
)
from musubi_tuner.minimax_h3_cache_latents import (
    PyAVH3MediaDecoder,
    cache_metadata_matches,
    configure_h3_image_item,
    fingerprint_checkpoint,
    fingerprint_file,
    h3_image_frame_count_for_item,
    image_condition_set,
    install_h3_video_decoder,
    record_for_item,
    records_for_dataset,
    target_frames_from_image,
    _validate_h3_image_mode,
)
from musubi_tuner.utils.model_utils import dtype_to_str


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _round_down_to_multiple(value: float, multiple: int = 32) -> int:
    return max(multiple, int(value // multiple) * multiple)


def _fit_size_to_max_pixels(width: int, height: int, max_pixels: int | None) -> tuple[int, int]:
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return width, height
    scale = math.sqrt(max_pixels / float(width * height))
    return _round_down_to_multiple(width * scale), _round_down_to_multiple(height * scale)


def _read_rgb_image(path: str | Path, max_pixels: int | None = None) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        target_size = _fit_size_to_max_pixels(image.width, image.height, max_pixels)
        if target_size != image.size:
            logger.info("Downscaling MiniMax-H3 text visual %s from %sx%s to %sx%s", path, image.width, image.height, *target_size)
            image = image.resize(target_size, Image.Resampling.LANCZOS)
        return np.asarray(image).copy()


def _image_dataset_info_map(datasets) -> dict[Path, tuple[list[Path], ImageDataset]]:
    mapping: dict[Path, tuple[list[Path], ImageDataset]] = {}
    for dataset in datasets:
        datasource = getattr(dataset, "datasource", None)
        for image_path, control_paths in getattr(datasource, "control_paths", {}).items():
            mapping[Path(image_path).resolve()] = ([Path(path).resolve() for path in control_paths], dataset)
    return mapping


def _target_image_paths_for_item(image_path: Path, dataset: ImageDataset) -> list[Path]:
    datasource = getattr(dataset, "datasource", None)
    target_paths = []
    for source_path, paths in getattr(datasource, "target_paths", {}).items():
        if Path(source_path).resolve() == image_path:
            target_paths = paths
            break
    return [image_path, *[Path(path).resolve() for path in target_paths]]


def _load_h3_image_item_pixels(
    item: ItemInfo,
    image_info_by_path: dict[Path, tuple[list[Path], ImageDataset]],
    *,
    text_visual_max_pixels: int | None = None,
) -> None:
    image_path = Path(item.item_key).resolve()
    image_info = image_info_by_path.get(image_path)
    if image_info is None:
        raise ValueError(f"MiniMax-H3 image mode could not find control image(s) for {image_path}")
    control_paths, dataset = image_info
    target_paths = _target_image_paths_for_item(image_path, dataset)
    targets = [_read_rgb_image(path, text_visual_max_pixels) for path in target_paths]
    item.content = targets if len(targets) > 1 else targets[0]
    item.original_size = (int(targets[0].shape[1]), int(targets[0].shape[0]))
    item.latent_cache_path = dataset.get_latent_cache_path(item)
    item.control_content = [_read_rgb_image(path, text_visual_max_pixels) for path in control_paths]


def _text_media_paths(record, task: str) -> set[Path]:
    if task == "fl2va":
        return {record.video_path}
    if task == "ref2va":
        return {reference.path for reference in record.references if reference.type in {"image", "video"}}
    return set()


def _build_visuals(
    record,
    task: str,
    item: ItemInfo,
    decoder: PyAVH3MediaDecoder,
    decoded_reference_cache: dict[tuple, torch.Tensor],
) -> dict[object, H3TextVisual]:
    if task == "t2va":
        return {}
    if getattr(item, "h3_image_mode", "none") != "none":
        conditions = image_condition_set(item, item.h3_image_mode)
        return {
            "first": H3TextVisual(conditions.first.unsqueeze(0)),
            "last": H3TextVisual(conditions.last.unsqueeze(0)),
        }
    target_frames = torch.as_tensor(item.content)
    if target_frames.ndim != 4:
        raise ValueError(f"MiniMax-H3 target frames must be [F,H,W,C], got {tuple(target_frames.shape)}")
    if task == "fl2va":
        return {
            "first": H3TextVisual(target_frames[:1]),
            "last": H3TextVisual(target_frames[-1:]),
        }

    target_size = (int(target_frames.shape[2]), int(target_frames.shape[1]))
    visuals = {}
    for reference in record.references:
        if reference.type not in {"image", "video"}:
            continue
        cache_key = (reference.path, reference.type, item.frame_count, target_size)
        frames = decoded_reference_cache.get(cache_key)
        if frames is None:
            frames = decoder.decode_reference_visual(
                reference,
                target_frame_count=item.frame_count,
                target_size=target_size,
            )
            decoded_reference_cache[cache_key] = frames
        if reference.type == "image":
            visuals[reference.path] = H3TextVisual(frames)
        else:
            sampled = frames[::12]
            timestamps = tuple(index / 2.0 for index in range(sampled.shape[0]))
            visuals[reference.path] = H3TextVisual(sampled, timestamps)
    return visuals


def _text_cache_metadata(
    *,
    task: str,
    crop_start: int,
    frame_count: int,
    processor_identity: str,
    text_encoder_identity: str,
    presentation_identity: str,
    cache_dtype: str,
) -> dict[str, str]:
    return {
        "task": task,
        "crop_start_frame": str(crop_start),
        "frame_count": str(frame_count),
        "text_encoder_fingerprint": text_encoder_identity,
        "processor_fingerprint": processor_identity,
        "presentation_fingerprint": presentation_identity,
        "cache_dtype": cache_dtype,
        "hidden_state_convention": "index50-after-50-layers-pre-final-norm",
        "token_tag_algorithm": "expanded-vision-span-with-flanks-v1",
        "text_width": str(TEXT_WIDTH),
        "max_text_rows": str(MAX_TEXT_ROWS),
    }


def _cache_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported MiniMax-H3 text cache dtype: {name}")


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser.add_argument("--text_encoder", type=str, required=True, help="released MiniMax-H3 Qwen3-VL BF16 safetensors")
    parser.add_argument(
        "--processor",
        type=str,
        default=DEFAULT_PROCESSOR_ID,
        help="Qwen3-VL-32B processor repo or local directory",
    )
    parser.add_argument("--processor_revision", type=str, default=None, help="optional processor revision")
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument(
        "--h3_image_mode",
        choices=("none", "first", "first_last"),
        default="none",
        help="cache image datasets as MiniMax-H3 FL2VA text conditioning",
    )
    parser.add_argument(
        "--h3_image_frame_count",
        type=int,
        default=None,
        help="override frame count for MiniMax-H3 image training modes; otherwise use dataset h3_image_frame_count or 5",
    )
    parser.add_argument(
        "--h3_text_visual_max_pixels",
        type=int,
        default=1024 * 1024,
        help=(
            "maximum pixel count for images/videos passed to the MiniMax-H3 Qwen3-VL text encoder; "
            "large visuals are downscaled to 32-pixel multiples before text caching. Use 0 to disable"
        ),
    )
    parser.add_argument("--text_cache_dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--disable_mmap", action="store_true", help="disable memory-mapped safetensors loading")
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    args.h3_image_mode = _validate_h3_image_mode(args.h3_image_mode)
    if args.h3_image_mode != "none" and args.task != "fl2va":
        raise ValueError("MiniMax-H3 image training modes require --task fl2va")
    if args.h3_text_visual_max_pixels < 0:
        raise ValueError("--h3_text_visual_max_pixels must be non-negative")
    text_visual_max_pixels = args.h3_text_visual_max_pixels or None
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Loading dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_H3)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets
    allowed_dataset_types = (ImageDataset, VideoDataset) if args.h3_image_mode != "none" else (VideoDataset,)
    if not all(isinstance(dataset, allowed_dataset_types) for dataset in datasets):
        raise ValueError("MiniMax-H3 text caching accepts image datasets only with --h3_image_mode")

    decoder = PyAVH3MediaDecoder()
    records = []
    for dataset in datasets:
        dataset_records = records_for_dataset(dataset, args.task, video_only=args.h3_image_mode != "none")
        if isinstance(dataset, VideoDataset):
            install_h3_video_decoder(dataset, decoder)
        records.extend(dataset_records)

    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    text_paths = {path for record in records for path in _text_media_paths(record, args.task)}
    media_fingerprints = {path: fingerprint_file(path) for path in text_paths}

    logger.info("Loading MiniMax-H3 Qwen3-VL processor from %s", args.processor)
    processor = load_h3_processor(args.processor, revision=args.processor_revision)
    processor_identity = processor_fingerprint(processor)
    logger.info("Fingerprinting MiniMax-H3 text encoder checkpoint")
    text_encoder_identity = fingerprint_checkpoint(args.text_encoder)
    logger.info("Loading MiniMax-H3 text encoder from %s", args.text_encoder)
    text_encoder = load_h3_text_encoder(
        args.text_encoder,
        processor_path=args.processor,
        revision=args.processor_revision,
        device=device,
        dtype=torch.bfloat16,
        disable_mmap=args.disable_mmap,
    )

    decoded_reference_cache = {}
    image_info_by_path = _image_dataset_info_map(datasets)
    skip_matching_cache = args.skip_existing
    args.skip_existing = False
    if args.h3_image_mode != "none" and not args.keep_cache:
        logger.info("MiniMax-H3 image mode keeps existing text cache files because it rewrites image dataset cache names")
        args.keep_cache = True

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            if args.h3_image_mode != "none":
                image_frame_count = h3_image_frame_count_for_item(item, args.h3_image_frame_count)
                _load_h3_image_item_pixels(item, image_info_by_path, text_visual_max_pixels=text_visual_max_pixels)
                configure_h3_image_item(item, image_frame_count)
                item.h3_image_mode = args.h3_image_mode
                item.content = target_frames_from_image(item, image_frame_count)
            record, crop_start = record_for_item(item, records)
            visuals = _build_visuals(record, args.task, item, decoder, decoded_reference_cache)
            presentation = build_presentation(record, args.task, visuals)
            record_media_fingerprints = {path: media_fingerprints[path] for path in _text_media_paths(record, args.task)}
            presentation_identity = presentation_fingerprint(presentation, record_media_fingerprints)
            metadata = _text_cache_metadata(
                task=args.task,
                crop_start=crop_start,
                frame_count=item.frame_count,
                processor_identity=processor_identity,
                text_encoder_identity=text_encoder_identity,
                presentation_identity=presentation_identity,
                cache_dtype=args.text_cache_dtype,
            )
            if args.h3_image_mode != "none":
                metadata["image_training_mode"] = args.h3_image_mode
                metadata["text_visual_max_pixels"] = str(args.h3_text_visual_max_pixels)
            if skip_matching_cache and Path(item.text_encoder_output_cache_path).is_file():
                if cache_metadata_matches(item.text_encoder_output_cache_path, metadata):
                    logger.info("Skipping matching MiniMax-H3 text cache: %s", item.text_encoder_output_cache_path)
                    continue
                logger.info("Rebuilding stale MiniMax-H3 text cache: %s", item.text_encoder_output_cache_path)

            hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
            hidden_states = hidden_states.to(_cache_dtype(args.text_cache_dtype))
            payload_mib = hidden_states.numel() * hidden_states.element_size() / (1024**2)
            logger.info(
                "Saving MiniMax-H3 text cache for %s: rows=%d, vision_rows=%d, payload=%.1f MiB",
                item.item_key,
                hidden_states.shape[0],
                int((token_tags == 0).sum().item()),
                payload_mib,
            )
            tensors = {
                f"varlen_mmh3_hidden_states_{dtype_to_str(hidden_states.dtype)}": hidden_states,
                "varlen_mmh3_token_tags_int64": token_tags,
            }
            save_text_encoder_output_cache_minimax_h3(item, tensors, metadata)

    cache_text_encoder_outputs.process_text_encoder_batches(
        args.num_workers,
        args.skip_existing,
        args.batch_size,
        datasets,
        all_cache_files,
        all_cache_paths,
        encode,
        requires_content=True,
    )
    cache_text_encoder_outputs.post_process_cache_files(datasets, all_cache_files, all_cache_paths, args.keep_cache)


if __name__ == "__main__":
    main()
