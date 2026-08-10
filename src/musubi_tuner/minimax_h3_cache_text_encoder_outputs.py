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
    H3TextVisual,
    TEXT_CACHE_FORMAT,
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    presentation_fingerprint,
    processor_fingerprint,
)
from musubi_tuner.minimax_h3.media import h3_records_from_datasource
from musubi_tuner.minimax_h3_cache_latents import (
    PyAVH3MediaDecoder,
    _validate_h3_image_mode,
    cache_metadata_matches,
    configure_h3_image_item,
    dataset_cache_dir_key,
    fingerprint_checkpoint,
    fingerprint_file,
    h3_image_frame_count_for_item,
    image_record_for_item,
    image_condition_set,
    item_record_inputs,
    target_frames_from_image,
    validate_h3_dataset,
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
        if not isinstance(dataset, ImageDataset) or datasource is None:
            continue
        for index in range(len(datasource)):
            image_path, _caption = datasource.get_caption(index)
            _target_paths, control_paths = datasource.get_media_paths(image_path)
            mapping[Path(image_path).resolve()] = ([Path(path).resolve() for path in control_paths], dataset)
    return mapping


def _target_image_paths_for_item(image_path: Path, dataset: ImageDataset) -> list[Path]:
    datasource = getattr(dataset, "datasource", None)
    if datasource is None:
        raise ValueError("MiniMax-H3 image dataset is missing its datasource")
    target_paths, _control_paths = datasource.get_media_paths(str(image_path))
    return [Path(path).resolve() for path in target_paths]


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
    processor_identity: str,
    text_encoder_identity: str,
    presentation_identity: str,
    cache_dtype: str,
) -> dict[str, str]:
    # cache_dtype and crop_start_frame stay so --skip_existing rebuilds when --text_cache_dtype or the
    # FL2VA crop window changes; frame_count is folded into the presentation fingerprint and the
    # behavior tags into TEXT_CACHE_FORMAT.
    return {
        "task": task,
        "crop_start_frame": str(crop_start),
        "cache_format": TEXT_CACHE_FORMAT,
        "text_encoder_fingerprint": text_encoder_identity,
        "processor_fingerprint": processor_identity,
        "presentation_fingerprint": presentation_identity,
        "cache_dtype": cache_dtype,
    }


def _cache_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported MiniMax-H3 text cache dtype: {name}")


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_text_encoder_outputs.setup_parser_common()
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="MiniMax-H3 Qwen3-VL safetensors (BF16, ConvRot INT8 or NVFP4, auto-detected)",
    )
    parser.add_argument(
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use W4A4 scaled_mm for an NVFP4 text encoder (requires PyTorch 2.10+ and Blackwell; default is weight-only dequantization)",
    )
    parser.add_argument(
        "--text_encoder_blocks_to_swap",
        type=int,
        default=0,
        help="number of the 50 Qwen3-VL decoder layers to stream from CPU instead of keeping them on the GPU"
        " (0 = disabled, 50 = minimum VRAM; requires CUDA)",
    )
    parser.add_argument(
        "--text_encoder_attn_mode",
        choices=("sdpa", "flash_attention_2", "eager"),
        default=None,
        help="attention implementation for the text encoder (default: transformers default, sdpa)."
        " Use flash_attention_2 for long presentations: sdpa falls back to the O(L^2) math kernel and can OOM",
    )
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
    records_by_dir = {}
    for dataset in datasets:
        if isinstance(dataset, VideoDataset):
            validate_h3_dataset(dataset)
        records_by_dir[dataset_cache_dir_key(dataset.cache_directory)] = h3_records_from_datasource(dataset.datasource, args.task)

    all_cache_files, all_cache_paths = cache_text_encoder_outputs.prepare_cache_files_and_paths(datasets)
    image_info_by_path = _image_dataset_info_map(datasets)
    text_paths = {
        path for records in records_by_dir.values() for record in records for path in _text_media_paths(record, args.task)
    }
    for image_path, (control_paths, dataset) in image_info_by_path.items():
        text_paths.update(_target_image_paths_for_item(image_path, dataset))
        text_paths.update(control_paths)
    media_fingerprints = {path: fingerprint_file(path) for path in text_paths}

    logger.info("Loading MiniMax-H3 Qwen3-VL processor")
    processor = load_h3_processor()
    processor_identity = processor_fingerprint(processor)
    text_encoder_identity = fingerprint_checkpoint(args.text_encoder)
    logger.info("Loading MiniMax-H3 text encoder from %s", args.text_encoder)
    text_encoder = load_h3_text_encoder(
        args.text_encoder,
        device=device,
        dtype=torch.bfloat16,
        disable_mmap=args.disable_mmap,
        nvfp4_scaled_mm=args.nvfp4_scaled_mm,
        blocks_to_swap=args.text_encoder_blocks_to_swap,
        attn_mode=args.text_encoder_attn_mode,
    )

    decoded_reference_cache = {}
    skip_matching_cache = args.skip_existing
    args.skip_existing = False

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            records = records_by_dir[dataset_cache_dir_key(str(Path(item.text_encoder_output_cache_path).parent))]
            if args.h3_image_mode != "none":
                record = image_record_for_item(item, records)
                crop_start = 0
                image_frame_count = h3_image_frame_count_for_item(item, args.h3_image_frame_count)
                _load_h3_image_item_pixels(item, image_info_by_path, text_visual_max_pixels=text_visual_max_pixels)
                configure_h3_image_item(item, image_frame_count)
                item.h3_image_mode = args.h3_image_mode
                item.content = target_frames_from_image(item, image_frame_count)
            else:
                datasource_index, crop_start = item_record_inputs(item)
                record = records[datasource_index]
            visuals = _build_visuals(record, args.task, item, decoder, decoded_reference_cache)
            presentation = build_presentation(record, args.task, visuals)
            record_media_paths = _text_media_paths(record, args.task)
            if args.h3_image_mode != "none":
                control_paths, dataset = image_info_by_path[record.video_path]
                record_media_paths = set(_target_image_paths_for_item(record.video_path, dataset)) | set(control_paths)
            record_media_fingerprints = {path: media_fingerprints[path] for path in record_media_paths}
            presentation_identity = presentation_fingerprint(
                presentation,
                record_media_fingerprints,
                frame_count=item.frame_count,
            )
            metadata = _text_cache_metadata(
                task=args.task,
                crop_start=crop_start,
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
