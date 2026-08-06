from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

import musubi_tuner.cache_text_encoder_outputs as cache_text_encoder_outputs
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_text_encoder_output_cache_minimax_h3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, VideoDataset
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
    fingerprint_checkpoint,
    fingerprint_file,
    install_h3_video_decoder,
    record_for_item,
    records_for_dataset,
)
from musubi_tuner.utils.model_utils import dtype_to_str


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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
    parser.add_argument("--text_cache_dtype", choices=("bf16", "float32"), default="bf16")
    parser.add_argument("--disable_mmap", action="store_true", help="disable memory-mapped safetensors loading")
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Loading dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_H3)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = dataset_group.datasets
    if not all(isinstance(dataset, VideoDataset) for dataset in datasets):
        raise ValueError("MiniMax-H3 text caching accepts only video datasets")

    decoder = PyAVH3MediaDecoder()
    records = []
    for dataset in datasets:
        dataset_records = records_for_dataset(dataset, args.task)
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
    skip_matching_cache = args.skip_existing

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
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
        False,
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
