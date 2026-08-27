from __future__ import annotations

import argparse
import copy
import gc
import logging
import random
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from tqdm.auto import tqdm

from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
from musubi_tuner.minimax_h3.generation_inputs import (
    VIDEO_VAE_SPATIAL_RATIO,
    build_reference_geometries,
    decode_generation_visuals,
    encode_audio_conditions,
    encode_visual_conditions,
    load_generation_record,
    parse_one_frame_options,
)
from musubi_tuner.minimax_h3.media import (
    TARGET_FPS,
    H3Record,
    audio_latent_frames,
    video_latent_frames,
)
from musubi_tuner.minimax_h3.checkpoint import resolve_safetensors_files
from musubi_tuner.modules.convrot_int8_utils import has_comfy_quant_tensors
from musubi_tuner.minimax_h3.model import MiniMaxH3Config, load_h3_transformer
from musubi_tuner.minimax_h3.packing import (
    FRAME_RESCALE,
    ONE_FRAME_AUDIO_LATENT_FRAMES,
    ONE_FRAME_VIDEO_LATENT_FRAMES,
    H3TimeOverrides,
    H3VideoGeometry,
    build_h3_layout,
)
from musubi_tuner.minimax_h3.sampling import (
    augment_condition_latents,
    build_shifted_schedule,
    create_sampling_generator,
    decoded_video_to_uint8,
    initialize_target_latents,
    sample_joint_av,
    synchronize_decoded_av,
    write_image,
    write_joint_av,
    write_video_only,
)
from musubi_tuner.minimax_h3.text_encoder import (
    TEXT_CACHE_FORMAT,
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    presentation_fingerprint,
    validate_text_rows,
)
from musubi_tuner.minimax_h3.video_vae import VIDEO_VAE_DECODE_DTYPE, VIDEO_VAE_ENCODE_DTYPE, load_video_vae
from musubi_tuner.minimax_h3_cache_latents import PyAVH3MediaDecoder, fingerprint_file
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.networks import lora_minimax_h3
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.model_utils import compile_transformer, setup_parser_compile


logger = logging.getLogger(__name__)

VIDEO_OUTPUT_SUFFIXES = (".mp4", ".mkv", ".mov")
LATENT_FILE_FORMAT = "minimax-h3-latents-v1"
# each cached entry can reach ~100 MB for long Ref2VA presentations, so keep the LRU small
TEXT_CONDITIONING_CACHE_ENTRIES = 16


def _time_flag() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]


def _one_frame_time_overrides(args: argparse.Namespace) -> H3TimeOverrides | None:
    if args.frame_count != 1:
        return None
    target_index, control_indices = parse_one_frame_options(args.one_frame) if args.one_frame else (0, None)
    return H3TimeOverrides(
        condition_times=tuple(FRAME_RESCALE * index for index in (control_indices or ())),
        target_time=FRAME_RESCALE * target_index,
    )


def _require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"MiniMax-H3 generation requires --{label}")
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"MiniMax-H3 --{label} does not exist: {path}")
    return path


def validate_session_args(args: argparse.Namespace) -> None:
    """Validate arguments that hold for the whole invocation (model paths, mode selection)."""
    mode_flags = [
        bool(getattr(args, "interactive", False)),
        bool(getattr(args, "from_file", None)),
        bool(getattr(args, "latent_path", None)),
    ]
    if sum(mode_flags) > 1:
        raise ValueError("MiniMax-H3 --interactive, --from_file, and --latent_path are mutually exclusive")

    if getattr(args, "latent_path", None):
        for path in args.latent_path:
            _require_path(path, "latent_path")
        _require_path(getattr(args, "video_vae", None), "video_vae")
        return

    if not args.task:
        raise ValueError("MiniMax-H3 generation requires --task")
    if args.task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError("MiniMax-H3 --task must be t2va, fl2va, or ref2va")
    for label in ("dit", "video_vae", "audio_vae"):
        _require_path(getattr(args, label, None), label)

    multi_prompt = bool(getattr(args, "interactive", False)) or bool(getattr(args, "from_file", None))
    if multi_prompt:
        if getattr(args, "text_cache", None):
            raise ValueError("MiniMax-H3 --interactive and --from_file do not accept --text_cache")
        if getattr(args, "trajectory_dir", None):
            raise ValueError("MiniMax-H3 --interactive and --from_file do not accept --trajectory_dir")
        _require_path(getattr(args, "text_encoder", None), "text_encoder")
    else:
        if getattr(args, "text_cache", None) is not None:
            _require_path(args.text_cache, "text_cache")
        else:
            _require_path(getattr(args, "text_encoder", None), "text_encoder")
    if getattr(args, "from_file", None):
        _require_path(args.from_file, "from_file")

    if not 0 <= args.blocks_to_swap <= 48:
        raise ValueError("MiniMax-H3 --blocks_to_swap must be between 0 and 48")

    lora_weights = args.lora_weight or []
    for path in lora_weights:
        _require_path(path, "lora_weight")
    if args.lora_multiplier and len(args.lora_multiplier) > len(lora_weights):
        raise ValueError("MiniMax-H3 has more --lora_multiplier values than --lora_weight files")


def validate_prompt_args(args: argparse.Namespace, *, directory_output: bool = False) -> None:
    """Validate per-prompt arguments; with directory_output the output path is an auto-named directory."""
    if args.width <= 0 or args.height <= 0 or args.width % 32 or args.height % 32:
        raise ValueError(f"MiniMax-H3 width and height must be positive and divisible by 32, got {args.width}x{args.height}")
    one_frame = args.frame_count == 1
    # fps above the native rate would let the duration gate admit packed sequences far past
    # the released maximum (and desynchronize the floored audio count), so the squeeze
    # direction stays closed until it is validated
    if not 1 <= args.output_fps <= TARGET_FPS:
        raise ValueError(f"MiniMax-H3 --output_fps must be in [1,{TARGET_FPS}], got {args.output_fps}")
    if one_frame and args.output_fps != TARGET_FPS:
        raise ValueError(f"MiniMax-H3 one-frame generation has no timeline to stretch; --output_fps must stay {TARGET_FPS}")
    # at least one band must stay on the stretched clock, or the video RoPE silently reverts
    # to the native timeline while the audio still covers the stretched duration
    max_keep_bands = MiniMaxH3Config.rope_inv_freq_len - 1
    if not 0 <= args.stretch_keep_bands <= max_keep_bands:
        raise ValueError(f"MiniMax-H3 --stretch_keep_bands must be in [0,{max_keep_bands}], got {args.stretch_keep_bands}")
    if args.stretch_keep_bands and args.output_fps == TARGET_FPS:
        raise ValueError(f"MiniMax-H3 --stretch_keep_bands requires an --output_fps below {TARGET_FPS}")
    if one_frame:
        _, control_indices = parse_one_frame_options(args.one_frame) if args.one_frame else (0, None)
        if args.task == "fl2va":
            provided_frames = int(bool(args.first_frame)) + int(bool(args.last_frame))
            # a missing-frames error is raised by the task input checks below
            if provided_frames and (control_indices is None or len(control_indices) != provided_frames):
                given = 0 if control_indices is None else len(control_indices)
                provided = " and ".join(label for label in ("first_frame", "last_frame") if getattr(args, label))
                raise ValueError(
                    "MiniMax-H3 one-frame FL2VA requires --one_frame control_index with one entry per provided frame:"
                    f" got {given} control_index entries for {provided_frames} condition frames ({provided}), "
                    'e.g. --one_frame "target_index=24,control_index=0" for a first frame at index 0'
                )
        elif control_indices is not None:
            raise ValueError("MiniMax-H3 --one_frame control_index applies only to FL2VA conditions")
    else:
        if args.one_frame is not None:
            raise ValueError("MiniMax-H3 --one_frame options require --frame_count 1")
        video_latent_frames(args.frame_count)
        # with a temporal stretch the rotary timeline spans the real (stretched) duration,
        # so that is the quantity to hold inside the released range
        duration = args.frame_count / args.output_fps
        if not args.allow_experimental_duration and not 5.0 <= duration <= 15.0:
            raise ValueError(
                f"MiniMax-H3 duration {duration:.3f}s is outside the released 5-15s range; "
                "pass --allow_experimental_duration to proceed"
            )
    if args.steps <= 0:
        raise ValueError("MiniMax-H3 --steps must be positive")
    for label in ("h3_shift_video", "h3_shift_audio"):
        value = float(getattr(args, label))
        if not 0.01 <= value <= 100.0:
            raise ValueError(f"MiniMax-H3 --{label} must be in [0.01,100.0], got {value}")
    for label in ("h3_visual_cond_clean", "h3_audio_cond_clean"):
        value = float(getattr(args, label))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"MiniMax-H3 --{label} must be in [0.0,1.0], got {value}")
    output_name = Path(getattr(args, "output_name", None)) if directory_output and getattr(args, "output_name", None) else None
    checked_output = output_name if directory_output else Path(args.output)
    if checked_output is not None:
        if one_frame:
            if checked_output.suffix.lower() != ".png":
                raise ValueError("MiniMax-H3 one-frame generation writes an image; the output name must use .png")
        elif checked_output.suffix.lower() not in VIDEO_OUTPUT_SUFFIXES:
            raise ValueError("MiniMax-H3 output names must use .mp4, .mkv, or .mov")
    if args.trajectory_stride < 1:
        raise ValueError(f"MiniMax-H3 --trajectory_stride must be at least 1, got {args.trajectory_stride}")

    if args.task == "t2va":
        if not args.prompt:
            raise ValueError("MiniMax-H3 T2VA requires --prompt")
        if args.first_frame or args.last_frame or args.reference_jsonl or args.ref:
            raise ValueError("MiniMax-H3 T2VA does not accept first/last/reference inputs")
    elif args.task == "fl2va":
        if getattr(args, "text_cache", None) is not None:
            raise ValueError("MiniMax-H3 FL2VA generation does not accept --text_cache")
        if not args.prompt:
            raise ValueError("MiniMax-H3 FL2VA requires --prompt")
        if args.reference_jsonl or args.ref:
            raise ValueError("MiniMax-H3 FL2VA does not accept --reference_jsonl or --ref")
        if one_frame:
            if not args.first_frame and not args.last_frame:
                raise ValueError("MiniMax-H3 one-frame FL2VA requires --first_frame and/or --last_frame")
            for label in ("first_frame", "last_frame"):
                if getattr(args, label):
                    _require_path(getattr(args, label), label)
        else:
            _require_path(args.first_frame, "first_frame")
            _require_path(args.last_frame, "last_frame")
    else:
        if bool(args.reference_jsonl) == bool(args.ref):
            raise ValueError("MiniMax-H3 Ref2VA requires exactly one of --reference_jsonl or --ref")
        if args.first_frame or args.last_frame:
            raise ValueError("MiniMax-H3 Ref2VA does not accept --first_frame or --last_frame")
        if args.ref:
            if not args.prompt:
                raise ValueError("MiniMax-H3 Ref2VA with --ref requires --prompt")
            if args.reference_index:
                raise ValueError("MiniMax-H3 --reference_index selects a --reference_jsonl record and does not apply to --ref")
        else:
            _require_path(args.reference_jsonl, "reference_jsonl")
            if args.reference_index < 0:
                raise ValueError("MiniMax-H3 --reference_index must be nonnegative")


def validate_generation_args(args: argparse.Namespace) -> None:
    validate_session_args(args)
    validate_prompt_args(args)


def load_cached_text_conditioning(
    path: str | Path,
    *,
    task: str,
    presentation_identity: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        cached_task = metadata.get("task")
        if cached_task != task:
            raise ValueError(f"MiniMax-H3 requested task {task} conflicts with text-cache task {cached_task}")
        cached_format = metadata.get("cache_format")
        if cached_format != TEXT_CACHE_FORMAT:
            raise ValueError(f"MiniMax-H3 text cache format must be {TEXT_CACHE_FORMAT!r}, got {cached_format!r}")
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


@dataclass
class H3SharedModels:
    """Session-resident models for --interactive and --from_file.

    The text encoder and transformer keep whatever placement their load flags chose
    (GPU-resident, or CPU-resident with layer/block streaming); the VAEs idle on the
    CPU and are borrowed onto the device per use. Stage helpers that receive no
    container reproduce the single-shot load-use-free behavior instead.
    """

    device: torch.device
    processor: object | None = None
    text_encoder: object | None = None
    video_vaes: dict[torch.dtype, torch.nn.Module] = field(default_factory=dict)
    audio_vae: torch.nn.Module | None = None
    transformer: torch.nn.Module | None = None
    lora_networks: list[torch.nn.Module] = field(default_factory=list)
    text_conditioning_cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=OrderedDict)

    def release_text_encoder(self) -> None:
        if self.processor is None and self.text_encoder is None:
            return
        self.processor = None
        self.text_encoder = None
        gc.collect()
        clean_memory_on_device(self.device)

    def release_transformer(self) -> None:
        transformer = self.transformer
        if transformer is None:
            return
        if transformer.offloader is not None:
            transformer.offloader.set_forward_only(True)
        self.transformer = None
        self.lora_networks = []
        del transformer
        gc.collect()
        clean_memory_on_device(self.device)


@contextmanager
def _borrowed_video_vae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, shared: H3SharedModels | None):
    if shared is None:
        vae = load_video_vae(args.video_vae, device=device, dtype=dtype, disable_mmap=args.disable_numpy_memmap)
        try:
            yield vae
        finally:
            del vae
            gc.collect()
            clean_memory_on_device(device)
        return
    vae = shared.video_vaes.get(dtype)
    if vae is None:
        vae = load_video_vae(args.video_vae, device="cpu", dtype=dtype, disable_mmap=args.disable_numpy_memmap)
        shared.video_vaes[dtype] = vae
    vae.to(device)
    try:
        yield vae
    finally:
        vae.to("cpu")
        clean_memory_on_device(device)


@contextmanager
def _borrowed_audio_vae(args: argparse.Namespace, device: torch.device, shared: H3SharedModels | None):
    if shared is None:
        vae = load_audio_vae(args.audio_vae, device=device, dtype=torch.float32, disable_mmap=args.disable_numpy_memmap)
        try:
            yield vae
        finally:
            del vae
            gc.collect()
            clean_memory_on_device(device)
        return
    if shared.audio_vae is None:
        shared.audio_vae = load_audio_vae(args.audio_vae, device="cpu", dtype=torch.float32, disable_mmap=args.disable_numpy_memmap)
    shared.audio_vae.to(device)
    try:
        yield shared.audio_vae
    finally:
        shared.audio_vae.to("cpu")
        clean_memory_on_device(device)


def _text_conditioning_cache_key(args: argparse.Namespace, record: H3Record, presentation) -> str:
    # the presentation fingerprint hashes text and media shapes; media contents enter through
    # per-file fingerprints. FL2VA frames are not record references, so they are added here.
    if args.task == "fl2va":
        media_fingerprints = {Path(path): fingerprint_file(path) for path in (args.first_frame, args.last_frame) if path}
    else:
        media_fingerprints = {
            reference.path: fingerprint_file(reference.path)
            for reference in record.references
            if reference.type in {"image", "video"}
        }
    return presentation_fingerprint(presentation, media_fingerprints, frame_count=args.frame_count)


def _encode_text(
    args: argparse.Namespace,
    record: H3Record,
    text_visuals,
    device: torch.device,
    shared: H3SharedModels | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    presentation = build_presentation(record, args.task, text_visuals)
    if args.text_cache:
        media_fingerprints = {
            reference.path: fingerprint_file(reference.path)
            for reference in record.references
            if reference.type in {"image", "video"}
        }
        presentation_identity = presentation_fingerprint(
            presentation,
            media_fingerprints,
            frame_count=args.frame_count,
        )
        return load_cached_text_conditioning(
            args.text_cache,
            task=args.task,
            presentation_identity=presentation_identity,
        )
    cache_key = None
    if shared is not None:
        cache_key = _text_conditioning_cache_key(args, record, presentation)
        cached = shared.text_conditioning_cache.get(cache_key)
        if cached is not None:
            shared.text_conditioning_cache.move_to_end(cache_key)
            logger.info("Reusing cached MiniMax-H3 text conditioning")
            return cached
    if shared is not None and shared.text_encoder is not None:
        processor = shared.processor
        text_encoder = shared.text_encoder
    else:
        logger.info("Loading MiniMax-H3 Qwen3-VL text encoder")
        processor = load_h3_processor()
        text_encoder = load_h3_text_encoder(
            args.text_encoder,
            device=device,
            dtype=torch.bfloat16,
            disable_mmap=args.disable_numpy_memmap,
            nvfp4_scaled_mm=args.nvfp4_scaled_mm,
            blocks_to_swap=args.text_encoder_blocks_to_swap,
            attn_mode=args.text_encoder_attn_mode,
        )
        if shared is not None:
            shared.processor = processor
            shared.text_encoder = text_encoder
    hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
    if shared is None:
        del processor, text_encoder
        gc.collect()
    clean_memory_on_device(device)
    hidden_states = hidden_states.to(torch.bfloat16).unsqueeze(0).cpu()
    token_tags = token_tags.cpu()
    if cache_key is not None:
        shared.text_conditioning_cache[cache_key] = (hidden_states, token_tags)
        while len(shared.text_conditioning_cache) > TEXT_CONDITIONING_CACHE_ENTRIES:
            shared.text_conditioning_cache.popitem(last=False)
    return hidden_states, token_tags


def _load_lora_state_dicts(args) -> list[dict]:
    """Load and filter LoRA state dicts for the load-time merge (ConvRot INT8 path)."""
    includes = args.include_patterns or []
    excludes = args.exclude_patterns or []
    state_dicts = []
    for index, path in enumerate(args.lora_weight or []):
        include = includes[index] if index < len(includes) else None
        exclude = excludes[index] if index < len(excludes) else None
        state_dicts.append(filter_lora_state_dict(load_file(path), include, exclude))
    return state_dicts


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


def _apply_lora_weights(transformer, args, device: torch.device) -> list[torch.nn.Module]:
    """Attach LoRAs as runtime additive branches (pre-quantized INT8 bases).

    The INT8 base tensors are never modified or requantized; each LoRA stays a separate
    branch with its own multiplier for the sampling lifetime.
    """
    weights = args.lora_weight or []
    multipliers = args.lora_multiplier or []
    includes = args.include_patterns or []
    excludes = args.exclude_patterns or []
    networks = []
    for index, path in enumerate(weights):
        multiplier = multipliers[index] if index < len(multipliers) else 1.0
        include = includes[index] if index < len(includes) else None
        exclude = excludes[index] if index < len(excludes) else None
        logger.info("Attaching MiniMax-H3 LoRA %s with multiplier %s", path, multiplier)
        state = filter_lora_state_dict(load_file(path), include, exclude)
        network = lora_minimax_h3.create_arch_network_from_weights(
            multiplier,
            state,
            unet=transformer,
            for_inference=True,
        )
        if not network.unet_loras:
            raise ValueError(f"MiniMax-H3 LoRA {path} contains no compatible target modules")
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        network.load_state_dict(state, strict=True)
        network.eval().requires_grad_(False).to(device)
        networks.append(network)
    return networks


def _configure_lora_weights(transformer, args, device: torch.device, *, prequantized: bool) -> list[torch.nn.Module]:
    """Route LoRA application by base artifact.

    Pre-quantized INT8 bases get runtime additive branches; a BF16 base with
    --convrot_int8 was already merged during the streaming load (no-op here); a plain
    BF16 base gets the one-time destructive CPU merge. --lora_runtime_attach forces the
    runtime-branch route on any base: merging rounds the fused weights to the base
    storage grid (BF16 mantissa step, or the INT8 quantization grid), which silently
    erases LoRAs whose per-element deltas sit below it -- small-magnitude adapters such
    as teacher-matching LoRAs. The runtime branch keeps the LoRA in its own precision,
    matching how it ran during training.
    """
    if not args.lora_weight:
        return []
    if prequantized or getattr(args, "lora_runtime_attach", False):
        return _apply_lora_weights(transformer, args, device)
    if not args.convrot_int8:
        _merge_lora_weights(transformer, args)
    return []


def _load_transformer(args: argparse.Namespace, device: torch.device) -> tuple[torch.nn.Module, list[torch.nn.Module]]:
    # Three LoRA routes, keyed on the base artifact:
    # - BF16 base + --convrot_int8: merge into BF16 during the streaming load, then quantize.
    # - Pre-quantized INT8 base (auto-detected): attach LoRAs as runtime additive branches;
    #   the INT8 tensors cannot be merged into.
    # - Plain BF16 base: one-time destructive CPU merge after loading (fastest inference).
    # --lora_runtime_attach overrides the two merge routes with runtime branches, for
    # small-magnitude LoRAs whose deltas would be rounded away by the merge.
    prequantized = has_comfy_quant_tensors(resolve_safetensors_files(args.dit), disable_numpy_memmap=args.disable_numpy_memmap)
    convrot_int8 = args.convrot_int8 or prequantized
    merge_at_load = bool(args.lora_weight) and args.convrot_int8 and not prequantized and not args.lora_runtime_attach
    load_on_cpu = bool(args.blocks_to_swap or (args.lora_weight and not convrot_int8 and not args.lora_runtime_attach))
    lora_weights, lora_multipliers = (_load_lora_state_dicts(args), args.lora_multiplier) if merge_at_load else (None, None)
    logger.info("Loading MiniMax-H3 transformer%s", " (ConvRot INT8)" if convrot_int8 else "")
    transformer = load_h3_transformer(
        args.dit,
        device="cpu" if load_on_cpu else device,
        dtype=torch.bfloat16,
        attn_mode="torch" if args.attn_mode == "sdpa" else args.attn_mode,
        split_attn=args.split_attn,
        disable_mmap=args.disable_numpy_memmap,
        convrot_int8=args.convrot_int8,
        quant_device=device,
        lora_weights=lora_weights,
        lora_multipliers=lora_multipliers,
        prune_adaln=args.prune_adaln,
    )
    attached_lora_networks = _configure_lora_weights(transformer, args, device, prequantized=prequantized)
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
    if getattr(args, "compile", False):
        # mirrors minimax_h3_train_network.compile_transformer: ConvRot INT8 Linears are
        # excluded (custom autograd.Function + autotuned Triton kernels are not
        # dynamo-traceable), as are the Linears of swapped blocks
        transformer = compile_transformer(
            args,
            transformer,
            [transformer.blocks],
            disable_linear=bool(args.blocks_to_swap) or bool(getattr(transformer, "is_convrot_int8", False)),
        )
    return transformer, attached_lora_networks


def _acquire_transformer(
    args: argparse.Namespace, device: torch.device, shared: H3SharedModels | None
) -> tuple[torch.nn.Module, list[torch.nn.Module]]:
    if shared is not None and shared.transformer is not None:
        shared.transformer.prepare_block_swap_before_forward()
        return shared.transformer, shared.lora_networks
    transformer, lora_networks = _load_transformer(args, device)
    if shared is not None:
        shared.transformer = transformer
        shared.lora_networks = lora_networks
    return transformer, lora_networks


def _reject_one_frame_audio_references(args: argparse.Namespace, record: H3Record) -> None:
    if args.frame_count == 1 and any(reference.type == "audio" for reference in record.references):
        raise ValueError(
            "MiniMax-H3 one-frame generation does not accept standalone audio references"
            " (their window is defined by the target duration); video references keep their embedded audio"
        )


def _encode_conditions(
    args: argparse.Namespace,
    record: H3Record,
    raw_visuals,
    decoder: PyAVH3MediaDecoder,
    device: torch.device,
    shared: H3SharedModels | None = None,
):
    visual_conditions = ()
    visual_geometries = ()
    reference_visual_geometries = {}
    if args.task != "t2va":
        logger.info("Encoding MiniMax-H3 visual conditions")
        with _borrowed_video_vae(args, device, VIDEO_VAE_ENCODE_DTYPE, shared) as condition_video_vae:
            if condition_video_vae.vae_ratio != VIDEO_VAE_SPATIAL_RATIO:
                raise ValueError(
                    f"MiniMax-H3 video VAE spatial ratio must be {VIDEO_VAE_SPATIAL_RATIO}, got {condition_video_vae.vae_ratio}"
                )
            visual_conditions, visual_geometries, reference_visual_geometries = encode_visual_conditions(
                args,
                record,
                raw_visuals,
                condition_video_vae,
            )

    audio_conditions = ()
    reference_audio_frames = {}
    if args.task == "ref2va" and any(reference.audio is not None for reference in record.references):
        logger.info("Encoding MiniMax-H3 audio conditions")
        with _borrowed_audio_vae(args, device, shared) as condition_audio_vae:
            audio_conditions, reference_audio_frames = encode_audio_conditions(
                args,
                record,
                decoder,
                condition_audio_vae,
                reference_video_frame_counts={
                    index: int(raw_visuals[reference.path].shape[0])
                    for index, reference in enumerate(record.references)
                    if reference.type == "video"
                },
            )
    reference_geometries = (
        build_reference_geometries(record, reference_visual_geometries, reference_audio_frames) if args.task == "ref2va" else ()
    )
    return visual_conditions, visual_geometries, reference_geometries, audio_conditions


def _build_layout(args: argparse.Namespace, text_length: int, visual_geometries, reference_geometries):
    one_frame = args.frame_count == 1
    condition_roles = None
    if args.task == "fl2va":
        condition_roles = tuple(role for role, path in (("first", args.first_frame), ("last", args.last_frame)) if path)
    layout = build_h3_layout(
        task=args.task,
        text_length=text_length,
        target_video=H3VideoGeometry(
            ONE_FRAME_VIDEO_LATENT_FRAMES if one_frame else video_latent_frames(args.frame_count),
            args.height // VIDEO_VAE_SPATIAL_RATIO,
            args.width // VIDEO_VAE_SPATIAL_RATIO,
        ),
        target_audio_frames=(
            ONE_FRAME_AUDIO_LATENT_FRAMES if one_frame else audio_latent_frames(args.frame_count, output_fps=args.output_fps)
        ),
        visual_conditions=visual_geometries,
        references=reference_geometries,
        one_frame=one_frame,
        condition_roles=condition_roles,
        time_overrides=_one_frame_time_overrides(args),
        output_fps=args.output_fps,
        temporal_fine_bands=args.stretch_keep_bands,
    )
    logger.info(
        "MiniMax-H3 layout: task=%s video=%s audio_frames=%d text_rows=%d packed_rows=%d temporal_stretch=%.4f fine_bands=%d",
        args.task,
        layout.target_video,
        layout.target_audio_frames,
        layout.text_length,
        layout.row_count,
        layout.temporal_stretch,
        layout.temporal_fine_bands,
    )
    return layout


def _setup_trajectory(args: argparse.Namespace):
    if not args.trajectory_dir:
        return None, None, [], None
    trajectory_dir = Path(args.trajectory_dir).expanduser()
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_schedule = build_shifted_schedule(
        args.steps,
        video_shift=args.h3_shift_video,
        audio_shift=args.h3_shift_audio,
    )
    with open(trajectory_dir / "sigma_schedule.csv", "w", encoding="utf-8", newline="") as handle:
        handle.write("step,base_sigma,sigma_video,sigma_audio\n")
        for index in range(args.steps):
            handle.write(
                f"{index},{trajectory_schedule.base[index]:.6f},"
                f"{trajectory_schedule.video[index]:.6f},{trajectory_schedule.audio[index]:.6f}\n"
            )
    for index in range(args.steps):
        logger.info(
            "MiniMax-H3 step %d/%d: base sigma %.4f, video sigma %.4f, audio sigma %.4f",
            index,
            args.steps,
            trajectory_schedule.base[index],
            trajectory_schedule.video[index],
            trajectory_schedule.audio[index],
        )
    trajectory: list[tuple[int, torch.Tensor]] = []

    def x0_callback(index: int, x0_video: torch.Tensor, x0_audio: torch.Tensor) -> None:
        del x0_audio  # the diagnostic decodes video only
        if index % args.trajectory_stride == 0 or index == args.steps - 1:
            trajectory.append((index, x0_video.detach().to(device="cpu", dtype=torch.float32)))

    return trajectory_dir, trajectory_schedule, trajectory, x0_callback


def _sample_latents(
    args: argparse.Namespace,
    *,
    layout,
    seed: int,
    text_hidden_states: torch.Tensor,
    text_token_tags: torch.Tensor,
    visual_conditions,
    audio_conditions,
    device: torch.device,
    shared: H3SharedModels | None = None,
    x0_callback=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = create_sampling_generator(seed)
    initial_video, initial_audio = initialize_target_latents(
        video_shape=(
            1,
            24,
            layout.target_video.frames,
            layout.target_video.height,
            layout.target_video.width,
        ),
        audio_shape=(1, 32, 2, layout.target_audio_frames),
        generator=generator,
        device=device,
        video_dtype=torch.float32,
        audio_dtype=torch.float32,
    )
    visual_conditions, audio_conditions = augment_condition_latents(
        visual_conditions,
        audio_conditions,
        generator=generator,
        visual_clean=args.h3_visual_cond_clean,
        audio_clean=args.h3_audio_cond_clean,
        device=device,
    )
    transformer, lora_networks = _acquire_transformer(args, device, shared)
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
            x0_callback=x0_callback,
        )
    video_latents = sample.video.detach().cpu()
    audio_latents = sample.audio.detach().cpu()
    if shared is None and transformer.offloader is not None:
        transformer.offloader.set_forward_only(True)
    del transformer, lora_networks, sample, text_hidden_states, text_token_tags
    del visual_conditions, audio_conditions, initial_video, initial_audio
    gc.collect()
    clean_memory_on_device(device)
    return video_latents, audio_latents


def _decode_and_save(
    args: argparse.Namespace,
    video_latents: torch.Tensor,
    audio_latents: torch.Tensor | None,
    output_path: str | Path,
    device: torch.device,
    shared: H3SharedModels | None = None,
    *,
    trajectory=None,
    trajectory_dir: Path | None = None,
    trajectory_schedule=None,
) -> Path:
    one_frame = args.frame_count == 1
    logger.info("Decoding MiniMax-H3 video")
    with _borrowed_video_vae(args, device, VIDEO_VAE_DECODE_DTYPE, shared) as video_vae:
        with torch.no_grad():
            decoded_video = video_vae.decode(video_latents.to(device=device, dtype=VIDEO_VAE_DECODE_DTYPE)).cpu()
        if trajectory_dir is not None and trajectory:
            logger.info("Decoding MiniMax-H3 trajectory (%d of %d steps)", len(trajectory), args.steps)
            for index, x0_latents in trajectory:
                with torch.no_grad():
                    step_video = video_vae.decode(x0_latents.to(device=device, dtype=VIDEO_VAE_DECODE_DTYPE)).cpu()
                step_stem = f"step{index:03d}_base{trajectory_schedule.base[index]:.4f}_sigv{trajectory_schedule.video[index]:.4f}"
                if one_frame:
                    step_path = trajectory_dir / f"{step_stem}.png"
                    write_image(decoded_video_to_uint8(step_video, frame_limit=1)[0], step_path)
                else:
                    step_path = trajectory_dir / f"{step_stem}.mp4"
                    write_video_only(
                        decoded_video_to_uint8(step_video, frame_limit=args.frame_count), step_path, fps=args.output_fps
                    )
                del step_video
                clean_memory_on_device(device)
                logger.info("Saved MiniMax-H3 trajectory step: %s", step_path)
            trajectory.clear()
    del video_latents
    gc.collect()
    clean_memory_on_device(device)

    if one_frame:
        write_image(decoded_video_to_uint8(decoded_video, frame_limit=1)[0], output_path)
        logger.info("Saved MiniMax-H3 output: %s", output_path)
        return Path(output_path)

    if audio_latents is None:
        raise ValueError("MiniMax-H3 video decoding requires audio latents")
    logger.info("Decoding MiniMax-H3 audio")
    with _borrowed_audio_vae(args, device, shared) as audio_vae:
        with torch.no_grad():
            decoded_audio = audio_vae.decode(audio_latents.to(device=device, dtype=torch.float32)).cpu()
    del audio_latents
    gc.collect()
    clean_memory_on_device(device)

    decoded = synchronize_decoded_av(
        decoded_video,
        decoded_audio,
        frame_count=args.frame_count,
        fps=args.output_fps,
    )
    write_joint_av(decoded, output_path)
    logger.info("Saved MiniMax-H3 output: %s", output_path)
    return Path(output_path)


def _resolve_seed(args: argparse.Namespace) -> int:
    if args.seed is not None:
        return int(args.seed)
    seed = random.randint(0, 2**32 - 1)
    logger.info("MiniMax-H3 using random seed %d", seed)
    return seed


def _resolve_output_path(args: argparse.Namespace, seed: int, *, directory_mode: bool) -> Path:
    if not directory_mode:
        return Path(args.output)
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = getattr(args, "output_name", None)
    if output_name:
        return output_dir / output_name
    suffix = ".png" if args.frame_count == 1 else ".mp4"
    return output_dir / f"{_time_flag()}_{seed}{suffix}"


def _save_latent_file(
    path: Path, video_latents: torch.Tensor, audio_latents: torch.Tensor | None, args: argparse.Namespace, seed: int
) -> Path:
    tensors = {"latent_video": video_latents.contiguous()}
    if audio_latents is not None:
        tensors["latent_audio"] = audio_latents.contiguous()
    metadata = {
        "format": LATENT_FILE_FORMAT,
        "seeds": str(seed),
        "prompt": args.prompt or "",
        "task": args.task,
        "width": str(args.width),
        "height": str(args.height),
        "frame_count": str(args.frame_count),
        "output_fps": str(args.output_fps),
        "steps": str(args.steps),
        "h3_shift_video": str(args.h3_shift_video),
        "h3_shift_audio": str(args.h3_shift_audio),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)
    logger.info("Saved MiniMax-H3 intermediate latents: %s", path)
    return path


def _load_latent_file(path: Path) -> tuple[torch.Tensor, torch.Tensor | None, int, dict]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        if "latent_video" not in keys or not keys <= {"latent_video", "latent_audio"}:
            raise ValueError(f"MiniMax-H3 latent file {path} has unexpected tensors {sorted(keys)}")
        video_latents = handle.get_tensor("latent_video")
        audio_latents = handle.get_tensor("latent_audio") if "latent_audio" in keys else None
    if metadata.get("format") != LATENT_FILE_FORMAT:
        raise ValueError(f"MiniMax-H3 latent file {path} format must be {LATENT_FILE_FORMAT!r}, got {metadata.get('format')!r}")
    frame_count = metadata.get("frame_count")
    if frame_count is None:
        raise ValueError(f"MiniMax-H3 latent file {path} is missing its frame_count metadata")
    frame_count = int(frame_count)
    if frame_count > 1 and audio_latents is None:
        raise ValueError(f"MiniMax-H3 latent file {path} is missing latent_audio for a {frame_count}-frame video")
    return video_latents, audio_latents, frame_count, metadata


def parse_prompt_line(line: str) -> dict:
    """Parse an interactive/from-file prompt line into argument overrides.

    Format: "prompt text --w 768 --h 1344 --f 1 --d 42 --s 30 --fs 12.0 --fsa 3.0
    --ofps 12 --skb 3 --i first.png --ei last.png --ref face.png --of target_index=24 --o name.png".
    --ref is repeatable and replaces any session-level --ref list. A line starting
    with "--" carries only options; without prompt text the command-line --prompt
    (when given) stays in effect. The literal string "\\n" in the prompt text becomes
    a newline, for the multi-line official prompt format.
    """
    line = line.strip()
    parts = ["", *line[2:].split(" --")] if line.startswith("--") else line.split(" --")
    overrides: dict = {}
    if parts[0].strip():
        overrides["prompt"] = parts[0].strip().replace("\\n", "\n")
    refs: list[str] = []
    for part in parts[1:]:
        part = part.strip()
        if not part:
            continue
        option, _, value = part.partition(" ")
        value = value.strip()
        if option == "w":
            overrides["width"] = int(value)
        elif option == "h":
            overrides["height"] = int(value)
        elif option == "f":
            overrides["frame_count"] = int(value)
        elif option == "ofps":
            overrides["output_fps"] = int(value)
        elif option == "skb":
            overrides["stretch_keep_bands"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["steps"] = int(value)
        elif option == "fs":
            overrides["h3_shift_video"] = float(value)
        elif option == "fsa":
            overrides["h3_shift_audio"] = float(value)
        elif option == "i":
            overrides["first_frame"] = value
        elif option == "ei":
            overrides["last_frame"] = value
        elif option == "ref":
            refs.append(value)
        elif option == "of":
            overrides["one_frame"] = value
        elif option == "o":
            overrides["output_name"] = value
        else:
            raise ValueError(f"MiniMax-H3 prompt line has unknown option --{option}")
    if refs:
        overrides["ref"] = refs
    return overrides


def apply_overrides(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    prompt_args = copy.deepcopy(args)
    prompt_args.output_name = None
    for key, value in overrides.items():
        setattr(prompt_args, key, value)
    return prompt_args


def run_generation(
    args: argparse.Namespace,
    device: torch.device | None = None,
    *,
    shared: H3SharedModels | None = None,
    decoder: PyAVH3MediaDecoder | None = None,
    directory_output: bool = False,
) -> Path:
    validate_prompt_args(args, directory_output=directory_output)
    one_frame = args.frame_count == 1
    if device is None:
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    decoder = decoder or PyAVH3MediaDecoder()
    seed = _resolve_seed(args)
    args.seed = seed

    record = load_generation_record(args)
    _reject_one_frame_audio_references(args, record)
    raw_visuals, text_visuals = decode_generation_visuals(args, record, decoder)
    text_hidden_states, text_token_tags = _encode_text(args, record, text_visuals, device, shared)
    visual_conditions, visual_geometries, reference_geometries, audio_conditions = _encode_conditions(
        args, record, raw_visuals, decoder, device, shared
    )
    del raw_visuals, text_visuals
    clean_memory_on_device(device)

    layout = _build_layout(args, text_hidden_states.shape[1], visual_geometries, reference_geometries)
    trajectory_dir, trajectory_schedule, trajectory, x0_callback = _setup_trajectory(args)
    video_latents, audio_latents = _sample_latents(
        args,
        layout=layout,
        seed=seed,
        text_hidden_states=text_hidden_states,
        text_token_tags=text_token_tags,
        visual_conditions=visual_conditions,
        audio_conditions=audio_conditions,
        device=device,
        shared=shared,
        x0_callback=x0_callback,
    )
    if one_frame:
        # the 2-frame audio target is a byproduct of the joint layout, not an output
        audio_latents = None
    output_path = _resolve_output_path(args, seed, directory_mode=directory_output)
    return _decode_and_save(
        args,
        video_latents,
        audio_latents,
        output_path,
        device,
        shared,
        trajectory=trajectory,
        trajectory_dir=trajectory_dir,
        trajectory_schedule=trajectory_schedule,
    )


@dataclass
class _BatchItem:
    index: int
    args: argparse.Namespace
    seed: int = 0
    record: H3Record | None = None
    text_visuals: dict | None = None
    text_hidden_states: torch.Tensor | None = None
    text_token_tags: torch.Tensor | None = None
    visual_conditions: tuple = ()
    visual_geometries: tuple = ()
    reference_geometries: tuple = ()
    audio_conditions: tuple = ()
    video_latents: torch.Tensor | None = None
    audio_latents: torch.Tensor | None = None
    latent_file: Path | None = None
    error: str | None = None


def _mark_failed(item: _BatchItem, stage: str, error: Exception) -> None:
    item.error = f"{stage}: {error}"
    logger.error("MiniMax-H3 prompt %d failed during %s: %s", item.index + 1, stage, error, exc_info=True)


def process_from_file(args: argparse.Namespace, device: torch.device) -> None:
    """Phased batch: each model family is loaded once and serves every prompt, so the
    peak VRAM matches single-shot generation. Sampled latents are written to disk
    immediately; a crash before decoding loses nothing (--latent_path decodes them)."""
    with open(args.from_file, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    items: list[_BatchItem] = []
    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            prompt_args = apply_overrides(args, parse_prompt_line(line))
            validate_prompt_args(prompt_args, directory_output=True)
        except ValueError as error:
            # an invalid line must not abort the batch: record it as a failed item so the
            # remaining prompts still run and the summary reports it
            failed = _BatchItem(index=len(items), args=copy.deepcopy(args))
            _mark_failed(failed, f"line {line_number} validation", error)
            items.append(failed)
            continue
        items.append(_BatchItem(index=len(items), args=prompt_args))
    if not items:
        logger.warning("MiniMax-H3 --from_file %s contains no prompts", args.from_file)
        return
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    shared = H3SharedModels(device=device)
    decoder = PyAVH3MediaDecoder()

    logger.info("MiniMax-H3 batch phase 1/4: preparing inputs for %d prompts", len(items))
    for item in items:
        try:
            item.seed = _resolve_seed(item.args)
            item.args.seed = item.seed
            item.record = load_generation_record(item.args)
            _reject_one_frame_audio_references(item.args, item.record)
            raw_visuals, item.text_visuals = decode_generation_visuals(item.args, item.record, decoder)
            (
                item.visual_conditions,
                item.visual_geometries,
                item.reference_geometries,
                item.audio_conditions,
            ) = _encode_conditions(item.args, item.record, raw_visuals, decoder, device, shared)
            del raw_visuals
        except Exception as error:
            _mark_failed(item, "input preparation", error)
    clean_memory_on_device(device)

    logger.info("MiniMax-H3 batch phase 2/4: text encoding")
    for item in items:
        if item.error:
            continue
        try:
            item.text_hidden_states, item.text_token_tags = _encode_text(item.args, item.record, item.text_visuals, device, shared)
            item.text_visuals = None
        except Exception as error:
            _mark_failed(item, "text encoding", error)
    shared.release_text_encoder()

    logger.info("MiniMax-H3 batch phase 3/4: sampling")
    for item in items:
        if item.error:
            continue
        try:
            layout = _build_layout(item.args, item.text_hidden_states.shape[1], item.visual_geometries, item.reference_geometries)
            item.video_latents, item.audio_latents = _sample_latents(
                item.args,
                layout=layout,
                seed=item.seed,
                text_hidden_states=item.text_hidden_states,
                text_token_tags=item.text_token_tags,
                visual_conditions=item.visual_conditions,
                audio_conditions=item.audio_conditions,
                device=device,
                shared=shared,
            )
            if item.args.frame_count == 1:
                # the 2-frame audio target is a byproduct of the joint layout, not an output
                item.audio_latents = None
            item.latent_file = _save_latent_file(
                output_dir / f"{_time_flag()}_{item.index:03d}_{item.seed}_latent.safetensors",
                item.video_latents,
                item.audio_latents,
                item.args,
                item.seed,
            )
            item.text_hidden_states = None
            item.text_token_tags = None
            item.visual_conditions = ()
            item.audio_conditions = ()
        except Exception as error:
            _mark_failed(item, "sampling", error)
    shared.release_transformer()

    logger.info("MiniMax-H3 batch phase 4/4: decoding")
    for item in items:
        if item.error:
            continue
        try:
            output_path = _resolve_output_path(item.args, item.seed, directory_mode=True)
            _decode_and_save(item.args, item.video_latents, item.audio_latents, output_path, device, shared)
            item.video_latents = None
            item.audio_latents = None
            if item.latent_file is not None:
                item.latent_file.unlink(missing_ok=True)
                item.latent_file = None
        except Exception as error:
            _mark_failed(item, "decoding", error)
            if item.latent_file is not None:
                logger.info("MiniMax-H3 intermediate latents kept for --latent_path decoding: %s", item.latent_file)

    failed = [item for item in items if item.error]
    logger.info("MiniMax-H3 batch finished: %d/%d prompts succeeded", len(items) - len(failed), len(items))
    for item in failed:
        logger.error("MiniMax-H3 prompt %d (%s) failed: %s", item.index + 1, item.args.prompt, item.error)


def process_interactive(args: argparse.Namespace, device: torch.device) -> None:
    """Interactive loop with all models session-resident. The text encoder and the
    transformer coexist on the accelerator, so VRAM-limited setups should pass
    --text_encoder_blocks_to_swap 50 and a generous --blocks_to_swap."""
    shared = H3SharedModels(device=device)
    decoder = PyAVH3MediaDecoder()
    Path(args.output).expanduser().mkdir(parents=True, exist_ok=True)

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")
    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError
                prompt_args = apply_overrides(args, parse_prompt_line(line))
                run_generation(prompt_args, device, shared=shared, decoder=decoder, directory_output=True)
                if args.bell:
                    print("\a")
            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue
            except EOFError:
                raise
            except Exception as error:
                logger.error("MiniMax-H3 generation failed: %s", error, exc_info=True)
    except EOFError:
        print("\nExiting interactive mode")


def _parse_output_fps_metadata(source: Path, metadata: dict, requested_fps: int) -> int:
    """The stored rate is authoritative: the latents were sampled on its rotary timeline,
    so decoding at any other rate would desynchronize audio and video."""
    raw = metadata.get("output_fps", str(TARGET_FPS))
    try:
        output_fps = int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"MiniMax-H3 latent file {source} has invalid output_fps metadata {raw!r}") from error
    if not 1 <= output_fps <= TARGET_FPS:
        raise ValueError(f"MiniMax-H3 latent file {source} output_fps metadata {output_fps} is outside [1,{TARGET_FPS}]")
    if requested_fps not in (TARGET_FPS, output_fps):
        logger.warning(
            "MiniMax-H3 latent file %s was sampled at %d fps; decoding at that rate and ignoring --output_fps %d",
            source,
            output_fps,
            requested_fps,
        )
    return output_fps


def process_latent_decode(args: argparse.Namespace, device: torch.device) -> None:
    """Decode-only mode for --from_file intermediate latents; only the VAEs are loaded."""
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for path in args.latent_path:
        source = Path(path).expanduser()
        loaded.append((source, *_load_latent_file(source)))
    if any(audio_latents is not None for _, _, audio_latents, _, _ in loaded):
        _require_path(getattr(args, "audio_vae", None), "audio_vae")
    shared = H3SharedModels(device=device)
    for source, video_latents, audio_latents, frame_count, metadata in loaded:
        logger.info("Decoding MiniMax-H3 latents from %s", source)
        try:
            item_args = copy.deepcopy(args)
            item_args.frame_count = frame_count
            item_args.output_fps = _parse_output_fps_metadata(source, metadata, args.output_fps)
            seed = metadata.get("seeds", "0")
            suffix = ".png" if frame_count == 1 else ".mp4"
            output_path = output_dir / f"{_time_flag()}_{seed}_{source.stem}{suffix}"
            _decode_and_save(item_args, video_latents, audio_latents, output_path, device, shared)
        except Exception as error:
            logger.error("MiniMax-H3 latent decode failed for %s: %s", source, error, exc_info=True)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("t2va", "fl2va", "ref2va"),
        default=None,
        help="generation task; required except with --latent_path",
    )
    parser.add_argument(
        "--dit",
        default=None,
        help="MiniMax-H3 transformer safetensors path or directory (BF16 or ConvRot INT8, each full or pruned; "
        "pre-quantized and pruned checkpoints are detected automatically). Required except with --latent_path",
    )
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="quantize BF16 DiT base weights to ConvRot INT8 at load time (requires triton for the fused kernels; "
        "falls back to slower dequantized bf16 matmul without it). ComfyUI pre-quantized ConvRot INT8 checkpoints "
        "are detected automatically and do not need this flag. With a BF16 base, LoRA weights are merged before "
        "quantization; with a pre-quantized base, LoRAs are attached as runtime branches instead.",
    )
    parser.add_argument(
        "--prune_adaln",
        action="store_true",
        help="prune the AdaLN projections of a full BF16 DiT at load time (mean-centered rank-8 basis, time "
        "embedder retained). Published pruned checkpoints do not need this flag; pre-quantized ConvRot INT8 "
        "checkpoints are rejected. Combines with --convrot_int8.",
    )
    parser.add_argument("--video_vae", default=None, help="MiniMax-H3 video VAE safetensors path or directory")
    parser.add_argument(
        "--audio_vae",
        default=None,
        help="MiniMax-H3 audio VAE safetensors path or directory; required except for --latent_path files without audio",
    )
    parser.add_argument(
        "--text_encoder", default=None, help="MiniMax-H3 Qwen3-VL safetensors path (BF16, ConvRot INT8 or NVFP4, auto-detected)"
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
    parser.add_argument("--text_cache", default=None, help="optional precomputed mmh3 text cache (single generation only)")
    parser.add_argument(
        "--prompt",
        default=None,
        help='the literal string "\\n" becomes a newline, for the multi-line official prompt format',
    )
    parser.add_argument("--first_frame", default=None)
    parser.add_argument("--last_frame", default=None)
    parser.add_argument("--reference_jsonl", default=None)
    parser.add_argument("--reference_index", type=int, default=0)
    parser.add_argument(
        "--ref",
        action="append",
        default=None,
        metavar="PATH[;type=image|video|audio][;audio=AUDIO_PATH]",
        help="Ref2VA inline reference, repeatable; occurrence order is the reference order and the caption comes from"
        " --prompt, so no JSONL (and no dummy target video_path) is needed. The type is inferred from the file"
        " extension unless ;type= overrides it; ;audio= attaches an external audio track to a video reference."
        " Validation matches the JSONL references schema exactly. Mutually exclusive with --reference_jsonl.",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1344)
    parser.add_argument(
        "--frame_count",
        type=int,
        default=124,
        help="pixel frame count, 17*n+5 for video; 1 enables the experimental one-frame (image) mode, which writes"
        " a PNG and skips audio decoding",
    )
    parser.add_argument(
        "--one_frame",
        default=None,
        metavar="target_index=N,control_index=A;B",
        help="one-frame mode time options (requires --frame_count 1): 0-based 24 fps pixel-frame indices on the"
        " nominal timeline, converted to RoPE times relative to the target-block cursor. target_index (default 0)"
        " places the generated frame; control_index places the FL2VA condition frames in --first_frame/--last_frame"
        " order and is required when conditions are present. The base model reads these as trainable time inputs;"
        " see docs/minimax_h3_1f.md",
    )
    parser.add_argument(
        "--output_fps",
        type=int,
        default=TARGET_FPS,
        help="experimental temporal stretch: sample the generated timeline at this rate (1-24) instead of"
        " 24 fps. The --frame_count frames then cover frame_count/fps seconds -- target RoPE spans scale"
        " by 24/fps, the audio track covers the stretched duration, and the output container is written"
        " at this rate. The model was trained at 24 fps only, so lower rates trade temporal resolution"
        " (and possibly quality) for compute; pair with --stretch_keep_bands, see docs. 24 disables the"
        " stretch",
    )
    parser.add_argument(
        "--stretch_keep_bands",
        type=int,
        default=0,
        help="with --output_fps below 24: rotate this many leading (highest-frequency) temporal RoPE bands"
        " by the unstretched grid. Those bands have periods at or below the latent token spacing and carry"
        " a per-token lattice phase rather than time; stretching them scrambles that phase with the"
        " 17-pixel-frame VAE group period (periodic fading/stripes). Recommended: 3-4 at 12 fps, 2 at"
        " 16 fps, 1 at 20 fps (at most 15 -- at least one band must stay on the stretched clock)."
        " 0 stretches all bands (default)",
    )
    parser.add_argument("--allow_experimental_duration", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None, help="random when omitted")
    parser.add_argument(
        "--output",
        required=True,
        help="output file for a single generation (.png for one-frame, .mp4/.mkv/.mov otherwise);"
        " output directory with --interactive, --from_file, and --latent_path (auto-named files)",
    )
    parser.add_argument(
        "--from_file",
        default=None,
        help="batch mode: read prompt lines (with inline --w/--h/--f/--d/--s/--fs/--fsa/--ofps/--skb/--i/--ei/--ref/--of/--o"
        " options) from a file and run them in phases, loading each model family once. Sampled latents are saved"
        " to the --output directory before decoding so a crash loses nothing; the files are removed after their"
        " output is written. See docs/minimax_h3.md",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="interactive mode: read prompt lines from the console with all models kept resident."
        " VRAM-limited setups should pass --text_encoder_blocks_to_swap 50 and a generous --blocks_to_swap",
    )
    parser.add_argument(
        "--latent_path",
        nargs="*",
        default=None,
        help="decode-only mode: decode intermediate latents safetensors saved by --from_file into the --output"
        " directory (only the VAEs are loaded)",
    )
    parser.add_argument(
        "--bell",
        action="store_true",
        help="ring a terminal bell when done (after each generation in interactive mode)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--attn_mode",
        choices=("torch", "sdpa", "flash", "flash3", "sageattn", "xformers"),
        default="torch",
    )
    parser.add_argument("--split_attn", action="store_true")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")
    setup_parser_compile(parser)  # torch.compile for the DiT, same flags as training
    parser.add_argument("--h3_shift_video", type=float, default=12.0)
    parser.add_argument("--h3_shift_audio", type=float, default=3.0)
    parser.add_argument("--h3_visual_cond_clean", type=float, default=0.999)
    parser.add_argument("--h3_audio_cond_clean", type=float, default=1.0)
    parser.add_argument("--lora_weight", nargs="*", default=None)
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None)
    parser.add_argument(
        "--lora_runtime_attach",
        action="store_true",
        help="attach LoRAs as runtime additive branches instead of merging them into the base weights"
        " (always the case for pre-quantized INT8 bases). Merging rounds the fused weights to the base"
        " storage grid, which silently erases LoRAs whose deltas are below the BF16 mantissa step --"
        " small-magnitude adapters such as teacher-matching LoRAs. Slightly slower, exact.",
    )
    parser.add_argument("--include_patterns", nargs="*", default=None)
    parser.add_argument("--exclude_patterns", nargs="*", default=None)
    parser.add_argument("--disable_numpy_memmap", action="store_true")
    parser.add_argument(
        "--trajectory_dir",
        default=None,
        help="diagnostic: decode each denoising step's clean estimate (x0_hat = x_t + sigma*v) to a"
        " video-only mp4 in this directory and write the per-step sigma schedule to sigma_schedule.csv,"
        " showing at which step the video content settles. The per-step latents are held on the CPU and"
        " decoded after the normal output, so peak VRAM is unchanged; decode time grows with the step count",
    )
    parser.add_argument(
        "--trajectory_stride",
        type=int,
        default=1,
        help="decode every N-th step into --trajectory_dir (the last step is always included)",
    )
    return parser


def main() -> None:
    args = setup_parser().parse_args()
    args.output_name = None
    if args.prompt:
        args.prompt = args.prompt.replace("\\n", "\n")
    validate_session_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.latent_path:
        process_latent_decode(args, device)
    elif args.from_file:
        process_from_file(args, device)
    elif args.interactive:
        process_interactive(args, device)
    else:
        run_generation(args, device)
    if args.bell and not args.interactive:
        print("\a")


if __name__ == "__main__":
    main()
