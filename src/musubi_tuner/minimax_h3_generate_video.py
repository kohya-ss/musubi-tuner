from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
import torch
from tqdm.auto import tqdm

from musubi_tuner.minimax_h3.audio_vae import encode_audio_mode, load_audio_vae
from musubi_tuner.minimax_h3.media import (
    H3AudioSource,
    H3Record,
    audio_latent_frames,
    load_h3_jsonl_records,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import H3ReferenceGeometry, H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3.sampling import (
    augment_condition_latents,
    initialize_target_latents,
    sample_joint_av,
    synchronize_decoded_av,
    write_joint_av,
)
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
    validate_text_rows,
)
from musubi_tuner.minimax_h3.video_vae import encode_video_condition, load_video_vae
from musubi_tuner.minimax_h3_cache_latents import PyAVH3MediaDecoder, fingerprint_file
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.lora_utils import filter_lora_state_dict


logger = logging.getLogger(__name__)
VIDEO_VAE_SPATIAL_RATIO = 16


def _require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"MiniMax-H3 generation requires --{label}")
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"MiniMax-H3 --{label} does not exist: {path}")
    return path


def validate_generation_args(args: argparse.Namespace) -> None:
    if args.task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError("MiniMax-H3 --task must be t2va, fl2va, or ref2va")
    for label in ("dit", "video_vae", "audio_vae"):
        _require_path(getattr(args, label, None), label)
    using_text_cache = getattr(args, "text_cache", None) is not None
    if not using_text_cache:
        _require_path(getattr(args, "text_encoder", None), "text_encoder")
    else:
        _require_path(args.text_cache, "text_cache")

    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError(f"MiniMax-H3 width and height must be positive and divisible by 32, got {args.width}x{args.height}")
    video_latent_frames(args.frame_count)
    duration = args.frame_count / 24.0
    if not args.allow_experimental_duration and not 5.0 <= duration <= 15.0:
        raise ValueError(
            f"MiniMax-H3 duration {duration:.3f}s is outside the released 5-15s range; "
            "pass --allow_experimental_duration to proceed"
        )
    if args.steps <= 0:
        raise ValueError("MiniMax-H3 --steps must be positive")
    if not 0 <= args.blocks_to_swap <= 48:
        raise ValueError("MiniMax-H3 --blocks_to_swap must be between 0 and 48")
    for label in ("h3_shift_video", "h3_shift_audio"):
        value = float(getattr(args, label))
        if not 0.01 <= value <= 100.0:
            raise ValueError(f"MiniMax-H3 --{label} must be in [0.01,100.0], got {value}")
    for label in ("h3_visual_cond_clean", "h3_audio_cond_clean"):
        value = float(getattr(args, label))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"MiniMax-H3 --{label} must be in [0.0,1.0], got {value}")
    if Path(args.output).suffix.lower() not in {".mp4", ".mkv", ".mov"}:
        raise ValueError("MiniMax-H3 --output must use .mp4, .mkv, or .mov")

    if args.task == "t2va":
        if not args.prompt:
            raise ValueError("MiniMax-H3 T2VA requires --prompt")
        if args.first_frame or args.last_frame or args.reference_jsonl:
            raise ValueError("MiniMax-H3 T2VA does not accept first/last/reference inputs")
    elif args.task == "fl2va":
        if using_text_cache:
            raise ValueError("MiniMax-H3 FL2VA generation does not accept --text_cache")
        if not args.prompt:
            raise ValueError("MiniMax-H3 FL2VA requires --prompt")
        _require_path(args.first_frame, "first_frame")
        _require_path(args.last_frame, "last_frame")
        if args.reference_jsonl:
            raise ValueError("MiniMax-H3 FL2VA does not accept --reference_jsonl")
    else:
        _require_path(args.reference_jsonl, "reference_jsonl")
        if args.first_frame or args.last_frame:
            raise ValueError("MiniMax-H3 Ref2VA does not accept --first_frame or --last_frame")
        if args.reference_index < 0:
            raise ValueError("MiniMax-H3 --reference_index must be nonnegative")

    lora_weights = args.lora_weight or []
    for path in lora_weights:
        _require_path(path, "lora_weight")
    if args.lora_multiplier and len(args.lora_multiplier) > len(lora_weights):
        raise ValueError("MiniMax-H3 has more --lora_multiplier values than --lora_weight files")


def load_cached_text_conditioning(
    path: str | Path,
    *,
    task: str,
    frame_count: int | None = None,
    presentation_identity: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        cached_task = metadata.get("task")
        if cached_task != task:
            raise ValueError(f"MiniMax-H3 requested task {task} conflicts with text-cache task {cached_task}")
        cached_frame_count = metadata.get("frame_count")
        if frame_count is not None and cached_frame_count != str(frame_count):
            raise ValueError(
                f"MiniMax-H3 requested frame count {frame_count} conflicts with text-cache frame count {cached_frame_count}"
            )
        expected_metadata = {
            "hidden_state_convention": "index50-after-50-layers-pre-final-norm",
            "token_tag_algorithm": "expanded-vision-span-with-flanks-v1",
            "text_width": str(TEXT_WIDTH),
            "max_text_rows": str(MAX_TEXT_ROWS),
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                raise ValueError(f"MiniMax-H3 text cache metadata {key} must be {expected!r}, got {metadata.get(key)!r}")
        cached_presentation = metadata.get("presentation_fingerprint")
        if not cached_presentation:
            raise ValueError("MiniMax-H3 text cache is missing its presentation fingerprint")
        if presentation_identity is not None and cached_presentation != presentation_identity:
            raise ValueError(
                "MiniMax-H3 requested presentation fingerprint "
                f"{presentation_identity} conflicts with text-cache presentation fingerprint {cached_presentation}"
            )
        hidden_keys = [key for key in handle.keys() if key.startswith("varlen_mmh3_hidden_states_")]
        if len(hidden_keys) != 1 or set(handle.keys()) != {hidden_keys[0], "varlen_mmh3_token_tags_int64"}:
            raise ValueError("MiniMax-H3 text cache has an invalid tensor-key set")
        hidden_states = handle.get_tensor(hidden_keys[0])
        token_tags = handle.get_tensor("varlen_mmh3_token_tags_int64")
    validate_text_rows(hidden_states, token_tags)
    return hidden_states.unsqueeze(0), token_tags


def _dummy_record(prompt: str) -> H3Record:
    placeholder = Path(".")
    return H3Record(
        video_path=placeholder,
        caption=prompt,
        target_audio=H3AudioSource(placeholder, embedded=False),
        references=(),
        jsonl_line=0,
    )


def _load_image_frames(path: str | Path, *, width: int, height: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        pixels = torch.from_numpy(np.asarray(image).copy())
    return pixels.unsqueeze(0)


def _prepare_pixels(frames: torch.Tensor) -> torch.Tensor:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"MiniMax-H3 condition pixels must be [F,H,W,3], got {tuple(frames.shape)}")
    if frames.dtype == torch.uint8:
        frames = frames.float().div(127.5).sub(1.0)
    else:
        frames = frames.float().mul(2.0).sub(1.0)
    return frames.permute(3, 0, 1, 2).unsqueeze(0).contiguous()


def _load_record_and_visuals(args, decoder: PyAVH3MediaDecoder):
    raw_visuals = {}
    text_visuals = {}
    if args.task in {"t2va", "fl2va"}:
        record = _dummy_record(args.prompt or "")
        if args.task == "fl2va":
            for role, path in (("first", args.first_frame), ("last", args.last_frame)):
                frames = _load_image_frames(path, width=args.width, height=args.height)
                raw_visuals[role] = frames
                text_visuals[role] = H3TextVisual(frames)
        return record, raw_visuals, text_visuals

    records = load_h3_jsonl_records(args.reference_jsonl, "ref2va")
    if args.reference_index >= len(records):
        raise ValueError(f"MiniMax-H3 --reference_index {args.reference_index} is outside {len(records)} JSONL records")
    record = records[args.reference_index]
    if args.prompt is not None:
        record = replace(record, caption=args.prompt)
    for reference in record.references:
        if reference.type not in {"image", "video"}:
            continue
        frames = decoder.decode_reference_visual(
            reference,
            target_frame_count=args.frame_count,
            target_size=(args.width, args.height),
        )
        raw_visuals[reference.path] = frames
        if reference.type == "image":
            text_visuals[reference.path] = H3TextVisual(frames)
        else:
            sampled = frames[::12]
            text_visuals[reference.path] = H3TextVisual(
                sampled,
                tuple(index / 2.0 for index in range(sampled.shape[0])),
            )
    return record, raw_visuals, text_visuals


def _encode_text(args, record: H3Record, text_visuals, device: torch.device):
    presentation = build_presentation(record, args.task, text_visuals)
    if args.text_cache:
        media_fingerprints = {
            reference.path: fingerprint_file(reference.path)
            for reference in record.references
            if reference.type in {"image", "video"}
        }
        presentation_identity = presentation_fingerprint(presentation, media_fingerprints)
        return load_cached_text_conditioning(
            args.text_cache,
            task=args.task,
            frame_count=args.frame_count,
            presentation_identity=presentation_identity,
        )
    logger.info("Loading MiniMax-H3 Qwen3-VL text encoder")
    processor = load_h3_processor(args.processor, revision=args.processor_revision)
    text_encoder = load_h3_text_encoder(
        args.text_encoder,
        processor_path=args.processor,
        revision=args.processor_revision,
        device=device,
        dtype=torch.bfloat16,
        disable_mmap=args.disable_numpy_memmap,
    )
    hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
    del processor, text_encoder
    gc.collect()
    clean_memory_on_device(device)
    return hidden_states.to(torch.bfloat16).unsqueeze(0).cpu(), token_tags.cpu()


def _module_device_dtype(module, fallback_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    for tensor in (*module.parameters(), *module.buffers()):
        if tensor.is_floating_point():
            return tensor.device, tensor.dtype
    return torch.device("cpu"), fallback_dtype


@torch.no_grad()
def _encode_visual_conditions(args, record, raw_visuals, video_vae):
    video_device, video_dtype = _module_device_dtype(video_vae, torch.float16)
    visual_latents = []
    visual_geometries = []
    reference_visual_geometries = {}

    def encode_visual(frames):
        latent = encode_video_condition(video_vae, _prepare_pixels(frames).to(video_device, video_dtype)).cpu()
        visual_latents.append(latent)
        return H3VideoGeometry(*latent.shape[2:])

    if args.task == "fl2va":
        for role in ("first", "last"):
            visual_geometries.append(encode_visual(raw_visuals[role]))
    elif args.task == "ref2va":
        for index, reference in enumerate(record.references):
            if reference.type in {"image", "video"}:
                reference_visual_geometries[index] = encode_visual(raw_visuals[reference.path])
    return tuple(visual_latents), tuple(visual_geometries), reference_visual_geometries


@torch.no_grad()
def _encode_audio_conditions(args, record, raw_visuals, decoder, audio_vae):
    audio_device, audio_dtype = _module_device_dtype(audio_vae, torch.float32)
    audio_latents = []
    reference_audio_frames = {}
    target_audio_frames = audio_latent_frames(args.frame_count)
    for index, reference in enumerate(record.references):
        if reference.audio is None:
            continue
        if reference.type == "video":
            frames = audio_latent_frames(raw_visuals[reference.path].shape[0])
            require_exact = True
        else:
            frames = target_audio_frames
            require_exact = False
        waveform = decoder.decode_audio(
            reference.audio,
            start_sample=0,
            sample_count=waveform_samples(frames),
            require_exact=require_exact,
        )
        latent = encode_audio_mode(audio_vae, waveform.unsqueeze(0).to(audio_device, audio_dtype)).cpu()
        audio_latents.append(latent)
        reference_audio_frames[index] = latent.shape[-1]
    return tuple(audio_latents), reference_audio_frames


def _build_reference_geometries(record, visual_geometries, audio_frames):
    references = []
    for index, reference in enumerate(record.references):
        if reference.type == "image":
            references.append(H3ReferenceGeometry("image", video=visual_geometries[index]))
        elif reference.type == "audio":
            references.append(H3ReferenceGeometry("audio", audio_frames=audio_frames[index]))
        else:
            references.append(
                H3ReferenceGeometry(
                    "video",
                    video=visual_geometries[index],
                    audio_frames=audio_frames.get(index, 0),
                )
            )
    return tuple(references)


def _merge_lora_weights(transformer, args) -> None:
    weights = args.lora_weight or []
    multipliers = args.lora_multiplier or []
    includes = args.include_patterns or []
    excludes = args.exclude_patterns or []
    for index, path in enumerate(weights):
        multiplier = multipliers[index] if index < len(multipliers) else 1.0
        include = includes[index] if index < len(includes) else None
        exclude = excludes[index] if index < len(excludes) else None
        logger.info("Merging MiniMax-H3 LoRA %s with multiplier %s", path, multiplier)
        state = filter_lora_state_dict(load_file(path), include, exclude)
        network = lora_minimax_h3.create_arch_network_from_weights(
            multiplier,
            state,
            unet=transformer,
            for_inference=True,
        )
        if not network.unet_loras:
            raise ValueError(f"MiniMax-H3 LoRA {path} contains no compatible target modules")
        network.merge_to(None, transformer, state, dtype=torch.bfloat16, device="cpu")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument("--dit", required=True, help="MiniMax-H3 BF16 transformer safetensors path or directory")
    parser.add_argument("--video_vae", required=True, help="MiniMax-H3 video VAE safetensors path or directory")
    parser.add_argument("--audio_vae", required=True, help="MiniMax-H3 audio VAE safetensors path or directory")
    parser.add_argument("--text_encoder", default=None, help="MiniMax-H3 Qwen3-VL BF16 safetensors path")
    parser.add_argument("--text_cache", default=None, help="optional precomputed mmh3 text cache")
    parser.add_argument("--processor", default=DEFAULT_PROCESSOR_ID, help="Qwen3-VL processor repo or directory")
    parser.add_argument("--processor_revision", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--first_frame", default=None)
    parser.add_argument("--last_frame", default=None)
    parser.add_argument("--reference_jsonl", default=None)
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1344)
    parser.add_argument("--frame_count", type=int, default=124)
    parser.add_argument("--allow_experimental_duration", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--attn_mode",
        choices=("torch", "sdpa", "flash", "flash3", "sageattn", "xformers"),
        default="torch",
    )
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    parser.add_argument("--h3_shift_video", type=float, default=12.0)
    parser.add_argument("--h3_shift_audio", type=float, default=3.0)
    parser.add_argument("--h3_visual_cond_clean", type=float, default=0.999)
    parser.add_argument("--h3_audio_cond_clean", type=float, default=1.0)
    parser.add_argument("--lora_weight", nargs="*", default=None)
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None)
    parser.add_argument("--include_patterns", nargs="*", default=None)
    parser.add_argument("--exclude_patterns", nargs="*", default=None)
    parser.add_argument("--disable_numpy_memmap", action="store_true")
    return parser


def run_generation(args: argparse.Namespace) -> Path:
    validate_generation_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    decoder = PyAVH3MediaDecoder()
    record, raw_visuals, text_visuals = _load_record_and_visuals(args, decoder)
    text_hidden_states, text_token_tags = _encode_text(args, record, text_visuals, device)

    visual_conditions = ()
    visual_geometries = ()
    reference_visual_geometries = {}
    if args.task != "t2va":
        logger.info("Loading MiniMax-H3 video VAE for visual conditions")
        condition_video_vae = load_video_vae(
            args.video_vae,
            device=device,
            dtype=torch.float16,
            disable_mmap=args.disable_numpy_memmap,
        )
        if condition_video_vae.vae_ratio != VIDEO_VAE_SPATIAL_RATIO:
            raise ValueError(
                f"MiniMax-H3 video VAE spatial ratio must be {VIDEO_VAE_SPATIAL_RATIO}, got {condition_video_vae.vae_ratio}"
            )
        visual_conditions, visual_geometries, reference_visual_geometries = _encode_visual_conditions(
            args,
            record,
            raw_visuals,
            condition_video_vae,
        )
        del condition_video_vae
        gc.collect()
        clean_memory_on_device(device)

    audio_conditions = ()
    reference_audio_frames = {}
    if args.task == "ref2va" and any(reference.audio is not None for reference in record.references):
        logger.info("Loading MiniMax-H3 audio VAE for audio conditions")
        condition_audio_vae = load_audio_vae(
            args.audio_vae,
            device=device,
            dtype=torch.float32,
            disable_mmap=args.disable_numpy_memmap,
        )
        audio_conditions, reference_audio_frames = _encode_audio_conditions(
            args,
            record,
            raw_visuals,
            decoder,
            condition_audio_vae,
        )
        del condition_audio_vae
        gc.collect()
        clean_memory_on_device(device)
    reference_geometries = (
        _build_reference_geometries(record, reference_visual_geometries, reference_audio_frames) if args.task == "ref2va" else ()
    )
    del raw_visuals, text_visuals
    clean_memory_on_device(device)

    layout = build_h3_layout(
        task=args.task,
        text_length=text_hidden_states.shape[1],
        target_video=H3VideoGeometry(
            video_latent_frames(args.frame_count),
            args.height // VIDEO_VAE_SPATIAL_RATIO,
            args.width // VIDEO_VAE_SPATIAL_RATIO,
        ),
        target_audio_frames=audio_latent_frames(args.frame_count),
        visual_conditions=visual_geometries,
        references=reference_geometries,
    )
    logger.info(
        "MiniMax-H3 layout: task=%s video=%s audio_frames=%d text_rows=%d packed_rows=%d",
        args.task,
        layout.target_video,
        layout.target_audio_frames,
        layout.text_length,
        layout.row_count,
    )
    initial_video, initial_audio = initialize_target_latents(
        video_shape=(
            1,
            24,
            layout.target_video.frames,
            layout.target_video.height,
            layout.target_video.width,
        ),
        audio_shape=(1, 32, 2, layout.target_audio_frames),
        seed=args.seed,
        device=device,
        video_dtype=torch.float32,
        audio_dtype=torch.float32,
    )
    visual_conditions, audio_conditions = augment_condition_latents(
        visual_conditions,
        audio_conditions,
        seed=args.seed,
        visual_clean=args.h3_visual_cond_clean,
        audio_clean=args.h3_audio_cond_clean,
        device=device,
    )

    load_on_cpu = bool(args.blocks_to_swap or args.lora_weight)
    logger.info("Loading MiniMax-H3 BF16 transformer")
    transformer = load_h3_transformer(
        args.dit,
        device="cpu" if load_on_cpu else device,
        dtype=torch.bfloat16,
        attn_mode="torch" if args.attn_mode == "sdpa" else args.attn_mode,
        split_attn=args.split_attn,
        disable_mmap=args.disable_numpy_memmap,
    )
    if args.lora_weight:
        _merge_lora_weights(transformer, args)
    if args.blocks_to_swap:
        swap_config = BlockSwapConfig(
            device=device,
            supports_backward=False,
            use_pinned_memory=args.use_pinned_memory_for_block_swap,
        )
        transformer.enable_block_swap(args.blocks_to_swap, swap_config)
        transformer.move_to_device_except_swap_blocks(device)
        transformer.prepare_block_swap_before_forward()
        transformer.switch_block_swap_for_inference()
    else:
        transformer.to(device)
    transformer.eval().requires_grad_(False)

    text_hidden_states = text_hidden_states.to(device=device, dtype=torch.bfloat16)
    text_token_tags = text_token_tags.unsqueeze(0).to(device)
    with tqdm(total=args.steps, desc="MiniMax-H3", unit="step") as progress:
        sample = sample_joint_av(
            transformer,
            layout=layout,
            text_hidden_states=text_hidden_states,
            text_token_tags=text_token_tags,
            initial_video=initial_video,
            initial_audio=initial_audio,
            steps=args.steps,
            video_shift=args.h3_shift_video,
            audio_shift=args.h3_shift_audio,
            visual_condition_latents=visual_conditions,
            audio_condition_latents=audio_conditions,
            visual_condition_clean=args.h3_visual_cond_clean,
            audio_condition_clean=args.h3_audio_cond_clean,
            step_callback=lambda completed, total: progress.update(1),
        )
    if transformer.offloader is not None:
        transformer.offloader.set_forward_only(True)
    del transformer, text_hidden_states, text_token_tags, visual_conditions, audio_conditions
    gc.collect()
    clean_memory_on_device(device)

    video_latents = sample.video
    audio_latents = sample.audio
    del sample

    logger.info("Decoding MiniMax-H3 video")
    video_vae = load_video_vae(
        args.video_vae,
        device=device,
        dtype=torch.float16,
        disable_mmap=args.disable_numpy_memmap,
    )
    with torch.no_grad():
        decoded_video = video_vae.decode(video_latents.to(device)).cpu()
    del video_vae, video_latents
    gc.collect()
    clean_memory_on_device(device)

    logger.info("Decoding MiniMax-H3 audio")
    audio_vae = load_audio_vae(
        args.audio_vae,
        device=device,
        dtype=torch.float32,
        disable_mmap=args.disable_numpy_memmap,
    )
    with torch.no_grad():
        decoded_audio = audio_vae.decode(audio_latents.to(device)).cpu()
    del audio_vae, audio_latents
    gc.collect()
    clean_memory_on_device(device)

    decoded = synchronize_decoded_av(
        decoded_video,
        decoded_audio,
        frame_count=args.frame_count,
    )
    write_joint_av(decoded, args.output)
    logger.info("Saved MiniMax-H3 output: %s", args.output)
    return Path(args.output)


def main() -> None:
    run_generation(setup_parser().parse_args())


if __name__ == "__main__":
    main()
