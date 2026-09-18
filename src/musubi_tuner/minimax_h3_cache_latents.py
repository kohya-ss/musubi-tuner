from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import json
import logging
from pathlib import Path

import numpy as np
import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3
from musubi_tuner.dataset.cache_io import save_latent_cache_minimax_h3
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.minimax_h3.audio_vae import encode_audio_mode, load_audio_vae
from musubi_tuner.minimax_h3.cache_plan import (
    H3DatasetPlan,
    cache_metadata_matches,
    item_audio_source,
    item_control_paths,
    item_crop_start,
    item_plan,
    item_record,
    plan_h3_datasets,
)
from musubi_tuner.minimax_h3.checkpoint import fingerprint_checkpoint
from musubi_tuner.minimax_h3.packing import (
    FL_CONDITION_ROLES,
    ONE_FRAME_AUDIO_LATENT_FRAMES,
    ONE_FRAME_VIDEO_LATENT_FRAMES,
    one_frame_condition_role,
    reference_condition_role,
)
from musubi_tuner.minimax_h3.args import add_h3_vae_args
from musubi_tuner.minimax_h3.media import (
    H3_AUDIO_SPEC,
    H3_TASKS,
    ONE_FRAME_REFERENCE_FRAME_CAP,
    H3MediaDecoder,
    H3Record,
    H3Task,
    PyAVH3MediaDecoder,
    TARGET_FPS,
    audio_latent_frames,
    fingerprint_file,
    module_device_dtype,
    prepare_pixels,
    reject_one_frame_audio_references,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3.video_vae import (
    VIDEO_VAE_ENCODE_DTYPE,
    encode_video_condition,
    encode_video_target,
    load_video_vae,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass(frozen=True)
class H3LatentCachePayload:
    """Everything one MiniMax-H3 latent cache stores, before the writer names the entries.

    The conditions are keyed by their layout role (`first`/`last`, `cond_{i}`,
    `ref_{i}_{image|video|audio}`); `save_latent_cache_minimax_h3` turns each field into the
    `latents[_<role>]_<shape>_<dtype>` entry the trainer reads back as `latents[_<role>]`.
    """

    target_video: torch.Tensor  # [24,F,H,W]
    target_audio: torch.Tensor  # [32,2,A]
    audio_present: bool
    metadata: dict[str, str]
    visual_conditions: dict[str, torch.Tensor] = field(default_factory=dict)  # role -> [24,F,H,W]
    audio_conditions: dict[str, torch.Tensor] = field(default_factory=dict)  # role -> [32,2,A]
    one_frame_target_index: int | None = None
    one_frame_control_indices: tuple[int, ...] | None = None

    def save(self, item: ItemInfo) -> None:
        save_latent_cache_minimax_h3(
            item,
            target_video=self.target_video,
            target_audio=self.target_audio,
            audio_present=self.audio_present,
            visual_conditions=self.visual_conditions,
            audio_conditions=self.audio_conditions,
            one_frame_target_index=self.one_frame_target_index,
            one_frame_control_indices=self.one_frame_control_indices,
            metadata=self.metadata,
        )


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


def _encode_target_video(video_vae, pixels: torch.Tensor, cache_seed: int, item_key: str) -> torch.Tensor:
    device, dtype = module_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    return encode_video_target(video_vae, pixels.to(device=device, dtype=dtype), cache_seed, item_key)


def _encode_condition_video(video_vae, pixels: torch.Tensor) -> torch.Tensor:
    device, dtype = module_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    return encode_video_condition(video_vae, pixels.to(device=device, dtype=dtype))


def _encode_audio(audio_vae, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] != 2:
        raise ValueError(f"MiniMax-H3 decoded audio must be stereo [2,L], got {tuple(waveform.shape)}")
    device, dtype = module_device_dtype(audio_vae, torch.float32)
    return encode_audio_mode(audio_vae, waveform.unsqueeze(0).to(device=device, dtype=dtype))


def _media_fingerprint_metadata(fingerprints: Mapping[Path, str]) -> str:
    normalized = {str(Path(path).resolve()): value for path, value in fingerprints.items()}
    return json.dumps(dict(sorted(normalized.items())), ensure_ascii=True, separators=(",", ":"))


# Bump whenever the cached tensor semantics change (posterior policy, normalization constants, key
# layout, or the fingerprint formats) so --skip_existing rebuilds stale caches.
LATENT_CACHE_FORMAT = "minimax-h3-latent-v2"
# The one-frame counterpart, bumped for one-frame-only layout changes so that video caches are not
# rebuilt along: v2 packs the conditions under the ordered cond_{i} roles (was first/last).
ONE_FRAME_CACHE_FORMAT = "minimax-h3-one-frame-v2"


def build_latent_metadata(
    *,
    task: H3Task,
    crop_start_frame: int,
    cache_seed: int,
    video_vae_fingerprint: str,
    audio_vae_fingerprint: str,
    media_fingerprints: Mapping[Path, str],
    one_frame_target_index: int | None = None,
    one_frame_control_indices: Sequence[int] | None = None,
) -> dict[str, str]:
    metadata = {
        "task": task,
        "cache_seed": str(cache_seed),
        "crop_start_frame": str(crop_start_frame),
        "cache_format": LATENT_CACHE_FORMAT,
        "video_vae_fingerprint": video_vae_fingerprint,
        "audio_vae_fingerprint": audio_vae_fingerprint,
        "media_fingerprints": _media_fingerprint_metadata(media_fingerprints),
    }
    if one_frame_target_index is not None:
        # duplicated from the tensor entries so --skip_existing rebuilds when the dataset's
        # fp_1f_target_index / fp_1f_clean_indices change (runtime reads the tensors)
        metadata["one_frame"] = "1"
        metadata["one_frame_format"] = ONE_FRAME_CACHE_FORMAT
        metadata["one_frame_target_index"] = str(one_frame_target_index)
        if one_frame_control_indices is not None:
            metadata["one_frame_control_indices"] = ";".join(str(index) for index in one_frame_control_indices)
    return metadata


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

    target_pixels = prepare_pixels(target_frames)
    canonical_item_key = f"{record.video_path}#{crop_start_frame}:{frame_count}"
    target_video = _encode_target_video(video_vae, target_pixels, cache_seed, canonical_item_key)[0]
    if target_video.shape[1] != expected_video_frames:
        raise ValueError(f"MiniMax-H3 video VAE returned {target_video.shape[1]} frames, expected {expected_video_frames}")

    target_audio = _encode_audio(audio_vae, target_waveform)[0]
    if target_audio.shape[2] != expected_audio_frames:
        raise ValueError(f"MiniMax-H3 audio VAE returned {target_audio.shape[2]} frames, expected {expected_audio_frames}")

    visual_conditions: dict[str, torch.Tensor] = {}
    audio_conditions: dict[str, torch.Tensor] = {}
    if task == "fl2va":
        for role, frame in zip(FL_CONDITION_ROLES, (target_frames[:1], target_frames[-1:])):
            visual_conditions[role] = _encode_condition_video(video_vae, prepare_pixels(frame))[0]
    elif task == "ref2va":
        _encode_reference_conditions(
            visual_conditions,
            audio_conditions,
            record=record,
            reference_frame_cap=frame_count,
            target_size=(width, height),
            target_samples=target_samples,
            video_vae=video_vae,
            audio_vae=audio_vae,
            media_decoder=media_decoder,
        )

    metadata = build_latent_metadata(
        task=task,
        crop_start_frame=crop_start_frame,
        cache_seed=cache_seed,
        video_vae_fingerprint=video_vae_fingerprint,
        audio_vae_fingerprint=audio_vae_fingerprint,
        media_fingerprints=media_fingerprints,
    )
    return H3LatentCachePayload(
        target_video=target_video,
        target_audio=target_audio,
        audio_present=audio_present,
        metadata=metadata,
        visual_conditions=visual_conditions,
        audio_conditions=audio_conditions,
    )


def _encode_reference_conditions(
    visual_conditions: dict[str, torch.Tensor],
    audio_conditions: dict[str, torch.Tensor],
    *,
    record: H3Record,
    reference_frame_cap: int,
    target_size: tuple[int, int],
    target_samples: int | None,
    video_vae,
    audio_vae,
    media_decoder: H3MediaDecoder,
) -> None:
    """Encodes the record's ordered references under the numbered ``ref_{i:03d}_{type}`` roles.

    Reference videos are capped at ``reference_frame_cap`` pixel frames (the target duration for
    video targets, the released 15 s span for one-frame targets) and keep their own audio
    duration; standalone audio references take the target's window (``target_samples``), which
    one-frame targets do not have (callers reject them first).
    """
    for index, reference in enumerate(record.references):
        visual_frames = None
        if reference.type in {"image", "video"}:
            visual_frames = media_decoder.decode_reference_visual(
                reference,
                target_frame_count=reference_frame_cap,
                target_size=target_size,
            )
            if reference.type == "video":
                video_latent_frames(visual_frames.shape[0])
            role = reference_condition_role(index, reference.type)
            visual_conditions[role] = _encode_condition_video(video_vae, prepare_pixels(visual_frames))[0]

        if reference.audio is not None:
            if reference.type == "video":
                reference_audio_frames = audio_latent_frames(visual_frames.shape[0])
                reference_samples = waveform_samples(reference_audio_frames)
                require_exact = True
            else:
                if target_samples is None:
                    raise ValueError("MiniMax-H3 standalone audio references require a target audio window")
                reference_samples = target_samples
                require_exact = False
            waveform = media_decoder.decode_audio(
                reference.audio,
                start_sample=0,
                sample_count=reference_samples,
                require_exact=require_exact,
            )
            audio_conditions[reference_condition_role(index, "audio")] = _encode_audio(audio_vae, waveform)[0]


def encode_one_frame_silence_latent(audio_vae) -> torch.Tensor:
    """The [32,2,2] silence placeholder shared by every one-frame item (a constant per audio VAE)."""
    silence = torch.zeros(2, waveform_samples(ONE_FRAME_AUDIO_LATENT_FRAMES), dtype=torch.float32)
    latent = _encode_audio(audio_vae, silence)[0]
    if latent.shape[2] != ONE_FRAME_AUDIO_LATENT_FRAMES:
        raise ValueError(
            f"MiniMax-H3 audio VAE returned {latent.shape[2]} silence frames, expected {ONE_FRAME_AUDIO_LATENT_FRAMES}"
        )
    return latent


def build_one_frame_latent_tensors(
    *,
    image_frames: torch.Tensor | np.ndarray,
    target_index: int,
    video_vae,
    silence_audio_latent: torch.Tensor,
    cache_seed: int,
    item_key: str,
    video_vae_fingerprint: str,
    audio_vae_fingerprint: str,
    media_fingerprints: Mapping[Path, str],
    control_frames: Sequence[torch.Tensor | np.ndarray] | None = None,
    control_indices: Sequence[int] | None = None,
    record: H3Record | None = None,
    audio_vae=None,
    media_decoder: H3MediaDecoder | None = None,
) -> H3LatentCachePayload:
    """One-frame (image) target: a single video latent token, the silence audio placeholder,
    and the target's 24 fps pixel-frame index as a tensor entry for the trainer's RoPE override.

    With control_frames/control_indices (K>=1, fl2va editing/inbetween), each bucket-resized
    control image becomes a condition latent under the ordered ``cond_{i:03d}`` role keys, and
    the indices ride along as an int64 tensor for the trainer's condition-time overrides.

    With a record carrying references (ref2va), the ordered references become numbered
    ``ref_{i:03d}`` condition latents exactly like video Ref2VA caches (image references are
    canvas-capped to the target area, video references keep their released span and audio);
    references are untimed, only the target index enters the time overrides."""
    if target_index < 0:
        raise ValueError(f"MiniMax-H3 one-frame target index must be nonnegative, got {target_index}")
    if (control_frames is None) != (control_indices is None):
        raise ValueError("MiniMax-H3 one-frame control frames and control indices must be provided together")
    references = () if record is None else record.references
    if references:
        if control_frames is not None:
            raise ValueError("MiniMax-H3 one-frame caching cannot combine control images with references")
        if media_decoder is None:
            raise ValueError("MiniMax-H3 one-frame references require a media decoder")
        _validate_task_record(record, "ref2va")
        reject_one_frame_audio_references(record)
        if audio_vae is None and any(reference.audio is not None for reference in references):
            raise ValueError("MiniMax-H3 one-frame audio-bearing references require the audio VAE")
    image_frames = torch.as_tensor(image_frames)
    if image_frames.ndim == 3:
        image_frames = image_frames.unsqueeze(0)
    if image_frames.ndim != 4 or image_frames.shape[0] != 1:
        raise ValueError(f"MiniMax-H3 one-frame target must be a single [H,W,C] image, got {tuple(image_frames.shape)}")
    height, width = image_frames.shape[1:3]
    if width % 32 or height % 32:
        raise ValueError(f"MiniMax-H3 target axes must be divisible by 32, got {width}x{height}")
    if silence_audio_latent.shape != (32, 2, ONE_FRAME_AUDIO_LATENT_FRAMES):
        raise ValueError(
            f"MiniMax-H3 one-frame silence latent must be [32,2,{ONE_FRAME_AUDIO_LATENT_FRAMES}],"
            f" got {tuple(silence_audio_latent.shape)}"
        )
    if control_frames is not None:
        if len(control_frames) < 1:
            raise ValueError("MiniMax-H3 one-frame caching requires at least one control image when controls are given")
        if len(control_frames) != len(control_indices):
            raise ValueError(
                f"MiniMax-H3 one-frame control count {len(control_frames)} does not match {len(control_indices)} control indices"
            )
        if any(int(index) < 0 for index in control_indices):
            raise ValueError(f"MiniMax-H3 one-frame control indices must be nonnegative, got {list(control_indices)}")

    target_pixels = prepare_pixels(image_frames)
    canonical_item_key = f"{item_key}#1f"
    target_video = _encode_target_video(video_vae, target_pixels, cache_seed, canonical_item_key)[0]
    if target_video.shape[1] != ONE_FRAME_VIDEO_LATENT_FRAMES:
        raise ValueError(f"MiniMax-H3 video VAE returned {target_video.shape[1]} frames, expected {ONE_FRAME_VIDEO_LATENT_FRAMES}")

    visual_conditions: dict[str, torch.Tensor] = {}
    audio_conditions: dict[str, torch.Tensor] = {}
    if control_frames is not None:
        for index, control in enumerate(control_frames):
            control = torch.as_tensor(control)
            if control.ndim != 3:
                raise ValueError(f"MiniMax-H3 one-frame control must be [H,W,C], got {tuple(control.shape)}")
            if tuple(control.shape[:2]) != (int(height), int(width)):
                raise ValueError(
                    f"MiniMax-H3 one-frame control size {control.shape[1]}x{control.shape[0]} does not match"
                    f" the target {width}x{height} (controls are resized to the bucket resolution)"
                )
            role = one_frame_condition_role(index)
            visual_conditions[role] = _encode_condition_video(video_vae, prepare_pixels(control.unsqueeze(0)))[0]
    if references:
        _encode_reference_conditions(
            visual_conditions,
            audio_conditions,
            record=record,
            reference_frame_cap=ONE_FRAME_REFERENCE_FRAME_CAP,
            target_size=(int(width), int(height)),
            target_samples=None,
            video_vae=video_vae,
            audio_vae=audio_vae,
            media_decoder=media_decoder,
        )

    if references:
        task: H3Task = "ref2va"
    elif control_frames is not None:
        task = "fl2va"
    else:
        task = "t2va"
    metadata = build_latent_metadata(
        task=task,
        crop_start_frame=0,
        cache_seed=cache_seed,
        video_vae_fingerprint=video_vae_fingerprint,
        audio_vae_fingerprint=audio_vae_fingerprint,
        media_fingerprints=media_fingerprints,
        one_frame_target_index=target_index,
        one_frame_control_indices=control_indices,
    )
    return H3LatentCachePayload(
        target_video=target_video,
        target_audio=silence_audio_latent,
        audio_present=False,
        metadata=metadata,
        visual_conditions=visual_conditions,
        audio_conditions=audio_conditions,
        one_frame_target_index=target_index,
        one_frame_control_indices=None if control_indices is None else tuple(int(index) for index in control_indices),
    )


def record_media_paths(record: H3Record) -> set[Path]:
    paths = {record.video_path}
    for reference in record.references:
        paths.add(reference.path)
        if reference.audio is not None:
            paths.add(reference.audio.path)
    return paths


@dataclass(frozen=True)
class _OneFrameCacheInputs:
    record: H3Record
    fingerprints: dict[Path, str]
    control_frames: Sequence[np.ndarray] | None
    control_indices: list[int] | None
    target_index: int
    metadata: dict[str, str]


@dataclass(frozen=True)
class _VideoCacheInputs:
    record: H3Record
    fingerprints: dict[Path, str]
    crop_start: int
    metadata: dict[str, str]


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
    add_h3_vae_args(parser, required=True)
    parser.add_argument("--task", choices=H3_TASKS, required=True)
    parser.add_argument(
        "--one_frame",
        action="store_true",
        help="experimental one-frame (image) training caches: accept image datasets whose items become single-token"
        " video targets with a silence audio placeholder. --task t2va caches plain image targets; --task fl2va"
        " additionally encodes the control images as time-annotated conditions (fp_1f_clean_indices); --task ref2va"
        " encodes the per-item references (image_jsonl_file references, or control images without"
        " fp_1f_clean_indices) as untimed Ref2VA conditions",
    )
    parser.add_argument("--cache_seed", type=int, default=0, help="seed used for reproducible target-video posterior samples")
    parser.add_argument(
        "--allow_experimental_duration",
        action="store_true",
        help="allow target crops outside the released 5-15 second duration range",
    )
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable numpy memmap while loading safetensors")
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

    plans = plan_h3_datasets(datasets, task=args.task, one_frame=args.one_frame)
    for dataset in datasets:
        if int(dataset.batch_size) != 1:
            logger.warning(
                "MiniMax-H3 dataset %d has batch_size=%d in the dataset config; training requires batch_size=1 "
                "(use gradient accumulation for a larger effective batch) and will stop on the first training batch",
                dataset.dataset_index,
                int(dataset.batch_size),
            )

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
    # every media file behind the records, fingerprinted before the models load so a missing
    # file fails fast; control images are fingerprinted per item (they are per-dataset lookups)
    media_fingerprints: dict[Path, str] = {}
    for plan in plans:
        for record in plan.records:
            for path in record_media_paths(record):
                media_fingerprints[path] = fingerprint_file(path)
        for source in plan.audio_sources or ():
            if source is not None:
                media_fingerprints[source.path] = fingerprint_file(source.path)

    logger.info("Loading MiniMax-H3 video VAE from %s", args.video_vae)
    video_vae = load_video_vae(
        args.video_vae,
        device=device,
        dtype=VIDEO_VAE_ENCODE_DTYPE,
        disable_numpy_memmap=args.disable_numpy_memmap,
    )
    logger.info("Loading MiniMax-H3 audio VAE from %s", args.audio_vae)
    audio_vae = load_audio_vae(args.audio_vae, device=device, dtype=torch.float32, disable_numpy_memmap=args.disable_numpy_memmap)

    silence_audio_latent: torch.Tensor | None = None
    if any(plan.is_image for plan in plans):
        # the silence placeholder is a constant per audio VAE, so encode it once for every item
        silence_audio_latent = encode_one_frame_silence_latent(audio_vae)

    decoder = PyAVH3MediaDecoder()
    presence_counts: Counter[bool] = Counter()
    one_frame_item_count = 0

    def one_frame_inputs(item: ItemInfo, plan: H3DatasetPlan) -> _OneFrameCacheInputs:
        record = item_record(plan, item)
        # the target image and, for ref2va, the references the cache encodes form the identity
        fingerprints = {path: media_fingerprints[path] for path in record_media_paths(record)}
        control_frames = None
        control_indices = None
        if args.task == "fl2va":
            control_frames = item.control_content
            control_indices = item.fp_1f_clean_indices
            if not control_indices or control_frames is None or len(control_frames) != len(control_indices):
                raise ValueError(f"MiniMax-H3 fl2va one-frame item is missing its control images: {item.item_key}")
            control_indices = [int(index) for index in control_indices]
            for control_path in item_control_paths(plan, item, len(control_indices)):
                fingerprints[control_path] = media_fingerprints.setdefault(control_path, fingerprint_file(control_path))
        target_index = 0 if item.fp_1f_target_index is None else int(item.fp_1f_target_index)
        metadata = build_latent_metadata(
            task=args.task,
            crop_start_frame=0,
            cache_seed=args.cache_seed,
            video_vae_fingerprint=video_vae_fingerprint,
            audio_vae_fingerprint=audio_vae_fingerprint,
            media_fingerprints=fingerprints,
            one_frame_target_index=target_index,
            one_frame_control_indices=control_indices,
        )
        return _OneFrameCacheInputs(record, fingerprints, control_frames, control_indices, target_index, metadata)

    def video_inputs(item: ItemInfo, plan: H3DatasetPlan) -> _VideoCacheInputs:
        record = item_record(plan, item)
        crop_start = item_crop_start(item)
        if item.audio_content is None or item.audio_present is None:
            raise ValueError(f"MiniMax-H3 cache item is missing its audio window: {item.item_key}")
        fingerprints = {path: media_fingerprints[path] for path in record_media_paths(record)}
        audio_source = item_audio_source(plan, item)
        if audio_source is not None:
            fingerprints[audio_source.path] = media_fingerprints[audio_source.path]
        metadata = build_latent_metadata(
            task=args.task,
            crop_start_frame=crop_start,
            cache_seed=args.cache_seed,
            video_vae_fingerprint=video_vae_fingerprint,
            audio_vae_fingerprint=audio_vae_fingerprint,
            media_fingerprints=fingerprints,
        )
        return _VideoCacheInputs(record, fingerprints, crop_start, metadata)

    def expected_metadata(item: ItemInfo) -> dict[str, str]:
        plan = item_plan(plans, item)
        if item.frame_count is None:
            return one_frame_inputs(item, plan).metadata
        return video_inputs(item, plan).metadata

    def cache_is_current(item: ItemInfo) -> bool:
        # --skip_existing: a cache counts as existing only when its identity metadata matches
        if not Path(item.latent_cache_path).is_file():
            return False
        if cache_metadata_matches(item.latent_cache_path, expected_metadata(item)):
            logger.info("Skipping matching MiniMax-H3 latent cache: %s", item.latent_cache_path)
            return True
        logger.info("Rebuilding stale MiniMax-H3 latent cache: %s", item.latent_cache_path)
        return False

    def encode_one_frame(item: ItemInfo, plan: H3DatasetPlan) -> None:
        nonlocal one_frame_item_count
        one_frame_item_count += 1
        inputs = one_frame_inputs(item, plan)
        payload = build_one_frame_latent_tensors(
            image_frames=item.content,
            target_index=inputs.target_index,
            video_vae=video_vae,
            silence_audio_latent=silence_audio_latent,
            cache_seed=args.cache_seed,
            item_key=str(inputs.record.video_path),
            video_vae_fingerprint=video_vae_fingerprint,
            audio_vae_fingerprint=audio_vae_fingerprint,
            media_fingerprints=inputs.fingerprints,
            control_frames=inputs.control_frames,
            control_indices=inputs.control_indices,
            record=inputs.record,
            audio_vae=audio_vae,
            media_decoder=decoder,
        )
        logger.info("Saving MiniMax-H3 one-frame latent cache for %s to %s", item.item_key, item.latent_cache_path)
        payload.save(item)

    def encode_video(item: ItemInfo, plan: H3DatasetPlan) -> None:
        inputs = video_inputs(item, plan)
        presence_counts[item.audio_present] += 1
        payload = build_latent_tensors(
            record=inputs.record,
            task=args.task,
            target_frames=item.content,
            target_waveform=item.audio_content,
            audio_present=item.audio_present,
            crop_start_frame=inputs.crop_start,
            video_vae=video_vae,
            audio_vae=audio_vae,
            cache_seed=args.cache_seed,
            media_decoder=decoder,
            video_vae_fingerprint=video_vae_fingerprint,
            audio_vae_fingerprint=audio_vae_fingerprint,
            media_fingerprints=inputs.fingerprints,
            allow_experimental_duration=args.allow_experimental_duration,
        )
        logger.info("Saving MiniMax-H3 latent cache for %s to %s", item.item_key, item.latent_cache_path)
        payload.save(item)

    def encode(batch: list[ItemInfo]) -> None:
        for item in batch:
            plan = item_plan(plans, item)
            if item.frame_count is None:
                encode_one_frame(item, plan)
            else:
                encode_video(item, plan)

    cache_latents.encode_datasets(datasets, encode, args, cache_is_current=cache_is_current)
    if one_frame_item_count:
        logger.info(
            "MiniMax-H3 one-frame cache summary: %d image items (silence audio placeholder, excluded from audio supervision)",
            one_frame_item_count,
        )
    if presence_counts:
        log_audio_presence_summary(presence_counts)


if __name__ == "__main__":
    main()
