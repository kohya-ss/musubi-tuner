from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.audio_utils import decode_audio as decode_audio_waveform
from musubi_tuner.dataset.audio_utils import slice_audio_window
from musubi_tuner.dataset.cache_io import append_audio_present_entry, save_latent_cache_minimax_h3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo, VideoDataset
from musubi_tuner.dataset.media_utils import load_video
from musubi_tuner.minimax_h3.audio_vae import encode_audio_mode, load_audio_vae
from musubi_tuner.minimax_h3.checkpoint import resolve_safetensors_files
from musubi_tuner.minimax_h3.media import (
    AUDIO_SAMPLE_RATE,
    AUDIO_TERMINAL_TOLERANCE_SAMPLES,
    H3_AUDIO_SPEC,
    H3AudioSource,
    H3Record,
    H3Reference,
    H3Task,
    TARGET_FPS,
    audio_latent_frames,
    h3_records_from_datasource,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3.video_vae import (
    VIDEO_VAE_ENCODE_DTYPE,
    encode_video_condition,
    encode_video_target,
    load_video_vae,
)
from musubi_tuner.utils.model_utils import dtype_to_str


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


@dataclass(frozen=True)
class H3LatentCachePayload:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str]


class H3MediaDecoder(Protocol):
    def decode_audio(
        self,
        source: H3AudioSource,
        *,
        start_sample: int,
        sample_count: int,
        require_exact: bool,
    ) -> torch.Tensor: ...

    def decode_reference_visual(
        self,
        reference: H3Reference,
        *,
        target_frame_count: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor: ...


def _round_to_multiple(value: float, multiple: int = CANVAS_MULTIPLE) -> int:
    return max(multiple, round(value / multiple) * multiple)


def _adapt_canvas(width: int, height: int) -> tuple[int, int]:
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale
    return _round_to_multiple(nominal_width), _round_to_multiple(nominal_height)


def _resize_frames(frames: Sequence[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    width, height = size
    resized = [
        torch.from_numpy(np.asarray(Image.fromarray(frame[..., :3]).resize((width, height), Image.Resampling.LANCZOS)).copy())
        for frame in frames
    ]
    return torch.stack(resized)


class PyAVH3MediaDecoder:
    """Decodes MiniMax-H3 reference media (target media is decoded by the shared dataset layer)."""

    def __init__(self, terminal_tolerance_samples: int = AUDIO_TERMINAL_TOLERANCE_SAMPLES):
        self.terminal_tolerance_samples = terminal_tolerance_samples

    def decode_audio(
        self,
        source: H3AudioSource,
        *,
        start_sample: int,
        sample_count: int,
        require_exact: bool,
    ) -> torch.Tensor:
        if start_sample < 0 or sample_count <= 0:
            raise ValueError("MiniMax-H3 audio window must have a nonnegative start and positive length")
        waveform = decode_audio_waveform(source, sample_rate=AUDIO_SAMPLE_RATE, channels=2)
        return slice_audio_window(
            waveform,
            start_sample=start_sample,
            sample_count=sample_count,
            pad_tolerance=self.terminal_tolerance_samples,
            require_exact=require_exact,
            context=str(source.path),
        )

    def decode_reference_visual(
        self,
        reference: H3Reference,
        *,
        target_frame_count: int,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        if reference.type == "image":
            with Image.open(reference.path) as image:
                frame = np.asarray(image.convert("RGB"))
            height, width = frame.shape[:2]
            target_area = target_size[0] * target_size[1]
            scale = min(1.0, math.sqrt(target_area / (width * height)))
            size = _round_to_multiple(width * scale), _round_to_multiple(height * scale)
            return _resize_frames([frame], size)

        if reference.type != "video":
            raise ValueError(f"Reference type {reference.type!r} has no visual stream")
        frames = load_video(str(reference.path), target_fps=TARGET_FPS, fps_resample_mode="timestamps")
        usable_frames = min(len(frames), target_frame_count)
        if usable_frames < 5:
            raise ValueError(f"MiniMax-H3 reference video requires at least 5 frames: {reference.path}")
        usable_frames = 5 + ((usable_frames - 5) // 17) * 17
        frames = frames[:usable_frames]
        source_height, source_width = frames[0].shape[:2]
        width, height = _adapt_canvas(source_width, source_height)
        if source_width * source_height < width * height:
            width = _round_to_multiple(source_width)
            height = _round_to_multiple(source_height)
        return _resize_frames(frames, (width, height))


def _validate_task_record(record: H3Record, task: H3Task) -> None:
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
    references = record.references
    if task != "ref2va":
        if references:
            raise ValueError(f"MiniMax-H3 task {task} does not accept references")
        return

    if len(references) > 12:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 12 reference items")
    image_count = sum(reference.type == "image" for reference in references)
    video_count = sum(reference.type == "video" for reference in references)
    audio_bearing_count = sum(reference.audio is not None for reference in references)
    if image_count > 9:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 9 image references")
    if video_count > 3:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 3 video references")
    if audio_bearing_count > 3:
        raise ValueError("MiniMax-H3 Ref2VA allows at most 3 audio-bearing references")
    if image_count + video_count == 0:
        raise ValueError("MiniMax-H3 Ref2VA requires at least one visual reference")


def _model_device_dtype(model: torch.nn.Module, fallback_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), fallback_dtype


def _prepare_pixels(frames: torch.Tensor | np.ndarray) -> torch.Tensor:
    frames = torch.as_tensor(frames)
    if frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError(f"MiniMax-H3 decoded video must be [F,H,W,C], got {tuple(frames.shape)}")
    frames = frames[..., :3]
    if frames.dtype == torch.uint8:
        frames = frames.float().div_(127.5).sub_(1.0)
    elif frames.is_floating_point():
        if not torch.all((frames >= 0) & (frames <= 1)):
            raise ValueError("Floating MiniMax-H3 decoded pixels must be in [0,1]")
        frames = frames.float().mul_(2.0).sub_(1.0)
    else:
        raise ValueError(f"Unsupported MiniMax-H3 decoded pixel dtype: {frames.dtype}")
    return frames.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def _encode_target_video(video_vae, pixels: torch.Tensor, cache_seed: int, item_key: str) -> torch.Tensor:
    device, dtype = _model_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    return encode_video_target(video_vae, pixels.to(device=device, dtype=dtype), cache_seed, item_key)


def _encode_condition_video(video_vae, pixels: torch.Tensor) -> torch.Tensor:
    device, dtype = _model_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    return encode_video_condition(video_vae, pixels.to(device=device, dtype=dtype))


def _encode_audio(audio_vae, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] != 2:
        raise ValueError(f"MiniMax-H3 decoded audio must be stereo [2,L], got {tuple(waveform.shape)}")
    device, dtype = _model_device_dtype(audio_vae, torch.float32)
    return encode_audio_mode(audio_vae, waveform.unsqueeze(0).to(device=device, dtype=dtype))


def _visual_key(role: str, latent: torch.Tensor) -> str:
    if latent.ndim != 4 or latent.shape[0] != 24:
        raise ValueError(f"MiniMax-H3 visual latent must be [24,F,H,W], got {tuple(latent.shape)}")
    _, frames, height, width = latent.shape
    dtype_name = dtype_to_str(latent.dtype)
    return f"latents_{role + '_' if role else ''}{frames}x{height}x{width}_{dtype_name}"


def _audio_key(role: str, latent: torch.Tensor) -> str:
    if latent.ndim != 3 or latent.shape[:2] != (32, 2):
        raise ValueError(f"MiniMax-H3 audio latent must be [32,2,A], got {tuple(latent.shape)}")
    dtype_name = dtype_to_str(latent.dtype)
    return f"latents_{role + '_' if role else 'audio_'}32x2x{latent.shape[2]}_{dtype_name}"


def _media_fingerprint_metadata(fingerprints: Mapping[Path, str]) -> str:
    normalized = {str(Path(path).resolve()): value for path, value in fingerprints.items()}
    return json.dumps(dict(sorted(normalized.items())), ensure_ascii=True, separators=(",", ":"))


# Bump whenever the cached tensor semantics change (posterior policy, normalization constants, key
# layout, or the fingerprint formats) so --skip_existing rebuilds stale caches.
LATENT_CACHE_FORMAT = "minimax-h3-latent-v2"


def build_latent_metadata(
    *,
    task: H3Task,
    crop_start_frame: int,
    cache_seed: int,
    video_vae_fingerprint: str,
    audio_vae_fingerprint: str,
    media_fingerprints: Mapping[Path, str],
) -> dict[str, str]:
    return {
        "task": task,
        "cache_seed": str(cache_seed),
        "crop_start_frame": str(crop_start_frame),
        "cache_format": LATENT_CACHE_FORMAT,
        "video_vae_fingerprint": video_vae_fingerprint,
        "audio_vae_fingerprint": audio_vae_fingerprint,
        "media_fingerprints": _media_fingerprint_metadata(media_fingerprints),
    }


def cache_metadata_matches(path: str | Path, expected: Mapping[str, str]) -> bool:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            actual = handle.metadata() or {}
    except Exception as error:
        logger.warning("Unable to read MiniMax-H3 cache metadata from %s: %s", path, error)
        return False
    return all(actual.get(key) == value for key, value in expected.items())


def build_latent_tensors(
    *,
    record: H3Record,
    task: H3Task,
    target_frames: torch.Tensor | np.ndarray,
    target_waveform: torch.Tensor,
    audio_present: bool,
    crop_start_frame: int,
    video_vae,
    audio_vae,
    cache_seed: int,
    media_decoder: H3MediaDecoder,
    video_vae_fingerprint: str,
    audio_vae_fingerprint: str,
    media_fingerprints: Mapping[Path, str],
    allow_experimental_duration: bool = False,
) -> H3LatentCachePayload:
    _validate_task_record(record, task)
    if crop_start_frame < 0:
        raise ValueError(f"MiniMax-H3 crop start must be nonnegative, got {crop_start_frame}")

    target_frames = torch.as_tensor(target_frames)
    if target_frames.ndim != 4:
        raise ValueError(f"MiniMax-H3 target frames must be [F,H,W,C], got {tuple(target_frames.shape)}")
    frame_count, height, width = target_frames.shape[:3]
    expected_video_frames = video_latent_frames(frame_count)
    expected_audio_frames = audio_latent_frames(frame_count)
    if width % 32 or height % 32:
        raise ValueError(f"MiniMax-H3 target axes must be divisible by 32, got {width}x{height}")
    duration = Fraction(frame_count, TARGET_FPS)
    if not allow_experimental_duration and not (Fraction(5, 1) <= duration <= Fraction(15, 1)):
        raise ValueError(
            f"MiniMax-H3 target duration {float(duration):.3f}s is outside the released 5-15s range; "
            "pass --allow_experimental_duration to proceed"
        )

    target_samples = waveform_samples(expected_audio_frames)
    target_waveform = torch.as_tensor(target_waveform, dtype=torch.float32)
    if tuple(target_waveform.shape) != (2, target_samples):
        raise ValueError(
            f"MiniMax-H3 target waveform must be [2,{target_samples}] for {frame_count} frames, got {tuple(target_waveform.shape)}"
        )
    if not audio_present and torch.any(target_waveform != 0):
        raise ValueError("MiniMax-H3 silence placeholder waveform must be all zeros when audio_present is False")

    target_pixels = _prepare_pixels(target_frames)
    canonical_item_key = f"{record.video_path}#{crop_start_frame}:{frame_count}"
    target_video = _encode_target_video(video_vae, target_pixels, cache_seed, canonical_item_key)[0]
    if target_video.shape[1] != expected_video_frames:
        raise ValueError(f"MiniMax-H3 video VAE returned {target_video.shape[1]} frames, expected {expected_video_frames}")

    target_audio = _encode_audio(audio_vae, target_waveform)[0]
    if target_audio.shape[2] != expected_audio_frames:
        raise ValueError(f"MiniMax-H3 audio VAE returned {target_audio.shape[2]} frames, expected {expected_audio_frames}")

    tensors = {
        _visual_key("", target_video): target_video,
        _audio_key("", target_audio): target_audio,
    }
    append_audio_present_entry(tensors, audio_present)
    if task == "fl2va":
        for role, frame in (("first", target_frames[:1]), ("last", target_frames[-1:])):
            condition = _encode_condition_video(video_vae, _prepare_pixels(frame))[0]
            tensors[_visual_key(role, condition)] = condition
    elif task == "ref2va":
        for index, reference in enumerate(record.references):
            role_prefix = f"ref_{index:03d}"
            visual_frames = None
            if reference.type in {"image", "video"}:
                visual_frames = media_decoder.decode_reference_visual(
                    reference,
                    target_frame_count=frame_count,
                    target_size=(width, height),
                )
                if reference.type == "video":
                    video_latent_frames(visual_frames.shape[0])
                condition = _encode_condition_video(video_vae, _prepare_pixels(visual_frames))[0]
                tensors[_visual_key(f"{role_prefix}_{reference.type}", condition)] = condition

            if reference.audio is not None:
                if reference.type == "video":
                    reference_audio_frames = audio_latent_frames(visual_frames.shape[0])
                    reference_samples = waveform_samples(reference_audio_frames)
                    require_exact = True
                else:
                    reference_samples = target_samples
                    require_exact = False
                waveform = media_decoder.decode_audio(
                    reference.audio,
                    start_sample=0,
                    sample_count=reference_samples,
                    require_exact=require_exact,
                )
                audio_latent = _encode_audio(audio_vae, waveform)[0]
                tensors[_audio_key(f"{role_prefix}_audio", audio_latent)] = audio_latent

    metadata = build_latent_metadata(
        task=task,
        crop_start_frame=crop_start_frame,
        cache_seed=cache_seed,
        video_vae_fingerprint=video_vae_fingerprint,
        audio_vae_fingerprint=audio_vae_fingerprint,
        media_fingerprints=media_fingerprints,
    )
    return H3LatentCachePayload(tensors=tensors, metadata=metadata)


def fingerprint_file(path: str | Path) -> str:
    """Lightweight file identity (size + mtime) for cache-staleness checks; deliberately not a content hash."""
    stat = Path(path).resolve().stat()
    return f"stat:{stat.st_size}:{stat.st_mtime_ns}"


def fingerprint_checkpoint(path: str | Path) -> str:
    files = resolve_safetensors_files(Path(path).resolve())
    digest = hashlib.sha256()
    for file in files:
        digest.update(file.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprint_file(file).encode("ascii"))
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def record_media_paths(record: H3Record) -> set[Path]:
    paths = {record.video_path}
    for reference in record.references:
        paths.add(reference.path)
        if reference.audio is not None:
            paths.add(reference.audio.path)
    return paths


def validate_h3_dataset(dataset: VideoDataset) -> None:
    if dataset.control_directory is not None or dataset.has_control:
        raise ValueError("MiniMax-H3 does not use the shared control-video fields")


def dataset_cache_dir_key(cache_directory: str) -> str:
    return os.path.normpath(os.path.abspath(cache_directory))


def item_cache_dir_key(item: ItemInfo) -> str:
    return dataset_cache_dir_key(os.path.dirname(item.latent_cache_path))


def item_record_inputs(item: ItemInfo) -> tuple[int, int]:
    """Returns (datasource_index, crop_start_frame) with presence validation."""
    if item.datasource_index is None or item.frame_pos is None:
        raise ValueError(f"MiniMax-H3 cache item is missing datasource provenance: {item.item_key}")
    return item.datasource_index, item.frame_pos


def log_audio_presence_summary(presence_counts: Mapping[bool, int]) -> None:
    real_audio = presence_counts.get(True, 0)
    missing_audio = presence_counts.get(False, 0)
    total = real_audio + missing_audio
    fraction = real_audio / total if total else 0.0
    logger.info(
        "MiniMax-H3 target-audio cache summary: real_audio=%d missing_audio=%d supervised_audio_fraction=%.6f",
        real_audio,
        missing_audio,
        fraction,
    )
    if total and real_audio == 0:
        logger.warning(
            "No cached item has real audio: training with these caches keeps the audio loss at 0; "
            "if this is intended, pass --video_only to the trainer explicitly"
        )


def setup_parser() -> argparse.ArgumentParser:
    parser = cache_latents.setup_parser_common(include_vae=False)
    parser.add_argument("--video_vae", type=str, required=True, help="MiniMax-H3 video VAE safetensors path or directory")
    parser.add_argument("--audio_vae", type=str, required=True, help="MiniMax-H3 audio VAE safetensors path or directory")
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument("--cache_seed", type=int, default=0, help="seed used for reproducible target-video posterior samples")
    parser.add_argument(
        "--allow_experimental_duration",
        action="store_true",
        help="allow target crops outside the released 5-15 second duration range",
    )
    parser.add_argument("--disable_mmap", action="store_true", help="disable memory-mapped safetensors loading")
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    if args.disable_cudnn_backend:
        torch.backends.cudnn.enabled = False

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info("Loading dataset config from %s", args.dataset_config)
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_MINIMAX_H3)
    dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group, audio_spec=H3_AUDIO_SPEC)
    datasets = dataset_group.datasets
    if not all(isinstance(dataset, VideoDataset) for dataset in datasets):
        raise ValueError("MiniMax-H3 latent caching accepts only video datasets")

    records_by_dir: dict[str, list[H3Record]] = {}
    audio_sources_by_dir: dict[str, list] = {}
    for dataset_index, dataset in enumerate(datasets):
        validate_h3_dataset(dataset)
        if int(dataset.batch_size) != 1:
            logger.warning(
                "MiniMax-H3 dataset %d has batch_size=%d in the dataset config; training requires batch_size=1 "
                "(use gradient accumulation for a larger effective batch) and will stop on the first training batch",
                dataset_index,
                int(dataset.batch_size),
            )
        key = dataset_cache_dir_key(dataset.cache_directory)
        records_by_dir[key] = h3_records_from_datasource(dataset.datasource, args.task)
        audio_sources_by_dir[key] = dataset.datasource.audio_sources

    if args.debug_mode is not None:
        cache_latents.show_datasets(
            datasets,
            args.debug_mode,
            args.console_width,
            args.console_back,
            args.console_num_images,
            fps=TARGET_FPS,
        )
        return

    video_vae_fingerprint = fingerprint_checkpoint(args.video_vae)
    audio_vae_fingerprint = fingerprint_checkpoint(args.audio_vae)
    media_fingerprints: dict[Path, str] = {}
    for key, records in records_by_dir.items():
        for record in records:
            for path in record_media_paths(record):
                media_fingerprints[path] = fingerprint_file(path)
        for source in audio_sources_by_dir[key]:
            if source is not None:
                media_fingerprints[source.path] = fingerprint_file(source.path)

    logger.info("Loading MiniMax-H3 video VAE from %s", args.video_vae)
    video_vae = load_video_vae(
        args.video_vae,
        device=device,
        dtype=VIDEO_VAE_ENCODE_DTYPE,
        disable_mmap=args.disable_mmap,
    )
    logger.info("Loading MiniMax-H3 audio VAE from %s", args.audio_vae)
    audio_vae = load_audio_vae(args.audio_vae, device=device, dtype=torch.float32, disable_mmap=args.disable_mmap)

    decoder = PyAVH3MediaDecoder()
    skip_matching_cache = args.skip_existing
    args.skip_existing = False
    presence_counts: Counter[bool] = Counter()

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            key = item_cache_dir_key(item)
            datasource_index, crop_start = item_record_inputs(item)
            record = records_by_dir[key][datasource_index]
            audio_source = audio_sources_by_dir[key][datasource_index]
            if item.audio_content is None or item.audio_present is None:
                raise ValueError(f"MiniMax-H3 cache item is missing its audio window: {item.item_key}")
            presence_counts[item.audio_present] += 1

            record_fingerprints = {path: media_fingerprints[path] for path in record_media_paths(record)}
            if audio_source is not None:
                record_fingerprints[audio_source.path] = media_fingerprints[audio_source.path]
            expected_metadata = build_latent_metadata(
                task=args.task,
                crop_start_frame=crop_start,
                cache_seed=args.cache_seed,
                video_vae_fingerprint=video_vae_fingerprint,
                audio_vae_fingerprint=audio_vae_fingerprint,
                media_fingerprints=record_fingerprints,
            )
            if skip_matching_cache and Path(item.latent_cache_path).is_file():
                if cache_metadata_matches(item.latent_cache_path, expected_metadata):
                    logger.info("Skipping matching MiniMax-H3 latent cache: %s", item.latent_cache_path)
                    continue
                logger.info("Rebuilding stale MiniMax-H3 latent cache: %s", item.latent_cache_path)
            payload = build_latent_tensors(
                record=record,
                task=args.task,
                target_frames=item.content,
                target_waveform=item.audio_content,
                audio_present=item.audio_present,
                crop_start_frame=crop_start,
                video_vae=video_vae,
                audio_vae=audio_vae,
                cache_seed=args.cache_seed,
                media_decoder=decoder,
                video_vae_fingerprint=video_vae_fingerprint,
                audio_vae_fingerprint=audio_vae_fingerprint,
                media_fingerprints=record_fingerprints,
                allow_experimental_duration=args.allow_experimental_duration,
            )
            logger.info("Saving MiniMax-H3 latent cache for %s to %s", item.item_key, item.latent_cache_path)
            save_latent_cache_minimax_h3(item, payload.tensors, payload.metadata)

    cache_latents.encode_datasets(datasets, encode, args)
    log_audio_presence_summary(presence_counts)


if __name__ == "__main__":
    main()
