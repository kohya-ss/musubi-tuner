from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file
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
)
from musubi_tuner.minimax_h3.media import (
    H3Record,
    audio_latent_frames,
    video_latent_frames,
)
from musubi_tuner.minimax_h3.checkpoint import resolve_safetensors_files
from musubi_tuner.modules.convrot_int8_utils import has_comfy_quant_tensors
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import H3VideoGeometry, build_h3_layout
from musubi_tuner.minimax_h3.sampling import (
    augment_condition_latents,
    build_shifted_schedule,
    create_sampling_generator,
    decoded_video_to_uint8,
    initialize_target_latents,
    sample_joint_av,
    synchronize_decoded_av,
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


logger = logging.getLogger(__name__)


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
    if args.trajectory_stride < 1:
        raise ValueError(f"MiniMax-H3 --trajectory_stride must be at least 1, got {args.trajectory_stride}")

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


def _encode_text(args, record: H3Record, text_visuals, device: torch.device):
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
    hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
    del processor, text_encoder
    gc.collect()
    clean_memory_on_device(device)
    return hidden_states.to(torch.bfloat16).unsqueeze(0).cpu(), token_tags.cpu()


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


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), required=True)
    parser.add_argument(
        "--dit",
        required=True,
        help="MiniMax-H3 transformer safetensors path or directory (BF16 or ConvRot INT8, each full or pruned; "
        "pre-quantized and pruned checkpoints are detected automatically)",
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
    parser.add_argument("--video_vae", required=True, help="MiniMax-H3 video VAE safetensors path or directory")
    parser.add_argument("--audio_vae", required=True, help="MiniMax-H3 audio VAE safetensors path or directory")
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
    parser.add_argument("--text_cache", default=None, help="optional precomputed mmh3 text cache")
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


def run_generation(args: argparse.Namespace) -> Path:
    validate_generation_args(args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    decoder = PyAVH3MediaDecoder()
    record = load_generation_record(args)
    raw_visuals, text_visuals = decode_generation_visuals(args, record, decoder)
    text_hidden_states, text_token_tags = _encode_text(args, record, text_visuals, device)

    visual_conditions = ()
    visual_geometries = ()
    reference_visual_geometries = {}
    if args.task != "t2va":
        logger.info("Loading MiniMax-H3 video VAE for visual conditions")
        condition_video_vae = load_video_vae(
            args.video_vae,
            device=device,
            dtype=VIDEO_VAE_ENCODE_DTYPE,
            disable_mmap=args.disable_numpy_memmap,
        )
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
        del condition_audio_vae
        gc.collect()
        clean_memory_on_device(device)
    reference_geometries = (
        build_reference_geometries(record, reference_visual_geometries, reference_audio_frames) if args.task == "ref2va" else ()
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
    generator = create_sampling_generator(args.seed)
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

    trajectory_dir = Path(args.trajectory_dir).expanduser() if args.trajectory_dir else None
    trajectory_schedule = None
    trajectory: list[tuple[int, torch.Tensor]] = []
    x0_callback = None
    if trajectory_dir is not None:
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

        def x0_callback(index: int, x0_video: torch.Tensor, x0_audio: torch.Tensor) -> None:
            del x0_audio  # the diagnostic decodes video only
            if index % args.trajectory_stride == 0 or index == args.steps - 1:
                trajectory.append((index, x0_video.detach().to(device="cpu", dtype=torch.float32)))

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
    if transformer.offloader is not None:
        transformer.offloader.set_forward_only(True)
    del transformer, attached_lora_networks, text_hidden_states, text_token_tags, visual_conditions, audio_conditions
    gc.collect()
    clean_memory_on_device(device)

    video_latents = sample.video
    audio_latents = sample.audio
    del sample

    logger.info("Decoding MiniMax-H3 video")
    video_vae = load_video_vae(
        args.video_vae,
        device=device,
        dtype=VIDEO_VAE_DECODE_DTYPE,
        disable_mmap=args.disable_numpy_memmap,
    )
    with torch.no_grad():
        decoded_video = video_vae.decode(video_latents.to(device=device, dtype=VIDEO_VAE_DECODE_DTYPE)).cpu()
    if trajectory_dir is not None:
        logger.info("Decoding MiniMax-H3 trajectory (%d of %d steps)", len(trajectory), args.steps)
        for index, x0_latents in trajectory:
            with torch.no_grad():
                step_video = video_vae.decode(x0_latents.to(device=device, dtype=VIDEO_VAE_DECODE_DTYPE)).cpu()
            step_path = trajectory_dir / (
                f"step{index:03d}_base{trajectory_schedule.base[index]:.4f}_sigv{trajectory_schedule.video[index]:.4f}.mp4"
            )
            write_video_only(decoded_video_to_uint8(step_video, frame_limit=args.frame_count), step_path)
            del step_video
            clean_memory_on_device(device)
            logger.info("Saved MiniMax-H3 trajectory step: %s", step_path)
    trajectory.clear()
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
        decoded_audio = audio_vae.decode(audio_latents.to(device=device, dtype=torch.float32)).cpu()
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
