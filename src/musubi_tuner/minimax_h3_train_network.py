from __future__ import annotations

import argparse
import gc
import logging
import re
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_MINIMAX_H3,
    ARCHITECTURE_MINIMAX_H3_FULL,
    round_down_frame_count,
)
from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
from musubi_tuner.minimax_h3.generation_inputs import (
    VIDEO_VAE_SPATIAL_RATIO,
    build_reference_geometries,
    decode_generation_visuals,
    encode_audio_conditions,
    encode_visual_conditions,
    load_generation_record,
    module_device_dtype,
)
from musubi_tuner.minimax_h3.media import H3_AUDIO_SPEC, audio_latent_frames, video_latent_frames
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import (
    H3PackedLayout,
    H3ReferenceGeometry,
    H3VideoGeometry,
    build_h3_layout,
)
from musubi_tuner.minimax_h3.sampling import (
    augment_condition_latents,
    initialize_target_latents,
    sample_joint_av,
    synchronize_decoded_av,
    write_joint_av,
)
from musubi_tuner.minimax_h3.text_encoder import (
    DEFAULT_PROCESSOR_ID,
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
)
from musubi_tuner.minimax_h3.video_vae import VIDEO_VAE_DECODE_DTYPE, VIDEO_VAE_ENCODE_DTYPE, load_video_vae
from musubi_tuner.minimax_h3_cache_latents import PyAVH3MediaDecoder
from musubi_tuner.training.audio_loss import add_audio_train_args, effective_audio_loss_weights
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)


_RUNTIME_REF_KEY = re.compile(r"^latents_ref_(\d{3})_(image|video|audio)$")


def _require_sampling_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"MiniMax-H3 training-time sampling requires --{label}")
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"MiniMax-H3 --{label} does not exist: {path}")
    return path


def _normalize_h3_sample_parameter(args: argparse.Namespace, parameter: dict[str, Any]) -> dict[str, Any]:
    sample = parameter.copy()
    sample_task = sample.get("task", args.task)
    if sample_task != args.task:
        raise ValueError(f"MiniMax-H3 sample prompt task {sample_task!r} does not match the training --task {args.task!r}")
    if sample.get("negative_prompt") not in {None, ""}:
        raise ValueError("MiniMax-H3 training-time sampling does not support negative prompts or CFG")
    if sample.get("cfg_scale") not in {None, 1, 1.0}:
        raise ValueError("MiniMax-H3 training-time sampling does not support --cfg_scale")
    if sample.get("guidance_scale") not in {None, 1, 1.0}:
        raise ValueError("MiniMax-H3 training-time sampling does not support --guidance_scale")
    if sample.get("discrete_flow_shift") not in {None, 1, 1.0}:
        raise ValueError("MiniMax-H3 sample prompts use --h3_shift_video and --h3_shift_audio, not discrete_flow_shift")

    width = int(sample.get("width", 768))
    height = int(sample.get("height", 1344))
    requested_frame_count = int(sample.get("frame_count", 124))
    frame_count = round_down_frame_count(requested_frame_count, ARCHITECTURE_MINIMAX_H3, 17)
    if frame_count != requested_frame_count:
        logger.warning(
            "MiniMax-H3 sample frame count %d was rounded down to %d (17*n+5)",
            requested_frame_count,
            frame_count,
        )
    sample_steps = int(sample.get("sample_steps", 30))
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError(f"MiniMax-H3 sample width and height must be positive and divisible by 32, got {width}x{height}")
    video_latent_frames(frame_count)
    duration = frame_count / 24.0
    allow_experimental = bool(
        getattr(args, "h3_allow_experimental_sample_duration", False) or sample.get("allow_experimental_duration", False)
    )
    if not allow_experimental and not 5.0 <= duration <= 15.0:
        raise ValueError(
            f"MiniMax-H3 sample duration {duration:.3f}s is outside the released 5-15s range; "
            "pass --h3_allow_experimental_sample_duration to proceed"
        )
    if sample_steps <= 0:
        raise ValueError("MiniMax-H3 sample_steps must be positive")

    prompt = sample.get("prompt")
    first_frame = sample.get("first_frame") or sample.get("image_path")
    last_frame = sample.get("last_frame") or sample.get("end_image_path")
    reference_jsonl = sample.get("reference_jsonl")
    reference_index = int(sample.get("reference_index", 0))
    if args.task == "t2va":
        if not prompt:
            raise ValueError("MiniMax-H3 T2VA training sample requires a prompt")
        if first_frame or last_frame or reference_jsonl:
            raise ValueError("MiniMax-H3 T2VA training sample does not accept first/last/reference inputs")
    elif args.task == "fl2va":
        if not prompt:
            raise ValueError("MiniMax-H3 FL2VA training sample requires a prompt")
        _require_sampling_path(first_frame, "first_frame")
        _require_sampling_path(last_frame, "last_frame")
        if reference_jsonl:
            raise ValueError("MiniMax-H3 FL2VA training sample does not accept reference_jsonl")
    else:
        _require_sampling_path(reference_jsonl, "reference_jsonl")
        if first_frame or last_frame:
            raise ValueError("MiniMax-H3 Ref2VA training sample does not accept first/last frames")
        if reference_index < 0:
            raise ValueError("MiniMax-H3 reference_index must be nonnegative")

    seed = sample.get("seed")
    sample.update(
        task=args.task,
        prompt=prompt,
        first_frame=first_frame,
        last_frame=last_frame,
        reference_jsonl=reference_jsonl,
        reference_index=reference_index,
        width=width,
        height=height,
        frame_count=frame_count,
        sample_steps=sample_steps,
        seed=None if seed is None else int(seed),
    )
    return sample


def _validate_audio_present(value: Any, batch_size: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.shape != (batch_size,) or value.dtype != torch.float32:
        raise ValueError("MiniMax-H3 batch requires a float32 audio_present tensor with shape [B]; re-run latent caching")
    if not torch.isfinite(value).all().item() or not ((value == 0.0) | (value == 1.0)).all().item():
        raise ValueError("MiniMax-H3 audio_present must be exactly 0.0 or 1.0 per sample")
    return value


@dataclass(frozen=True)
class _H3RuntimeBatch:
    layout: H3PackedLayout
    text_hidden_states: torch.Tensor
    text_token_tags: torch.Tensor
    visual_conditions: tuple[torch.Tensor, ...]
    audio_conditions: tuple[torch.Tensor, ...]
    audio_present: torch.Tensor


def _stack_single_text_rows(value, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 {label} must keep a leading batch axis of size 1")
        return value
    if not isinstance(value, Sequence) or len(value) != 1 or not isinstance(value[0], torch.Tensor):
        raise ValueError(f"MiniMax-H3 {label} must contain exactly one tensor")
    return value[0].unsqueeze(0)


def _runtime_batch_plan(batch: dict[str, Any], video_latents: torch.Tensor) -> _H3RuntimeBatch:
    if video_latents.ndim != 5 or video_latents.shape[1] != 24:
        raise ValueError(f"MiniMax-H3 target video latents must be [B,24,F,H,W], got {tuple(video_latents.shape)}")
    batch_size = video_latents.shape[0]
    if batch_size != 1:
        raise ValueError(f"MiniMax-H3 R1 requires batch_size=1, got {batch_size}; use gradient accumulation")
    audio_latents = batch.get("latents_audio")
    if not isinstance(audio_latents, torch.Tensor) or audio_latents.ndim != 4 or tuple(audio_latents.shape[1:3]) != (32, 2):
        shape = None if not isinstance(audio_latents, torch.Tensor) else tuple(audio_latents.shape)
        raise ValueError(f"MiniMax-H3 target audio latents must be [B,32,2,A], got {shape}")
    if audio_latents.shape[0] != batch_size:
        raise ValueError("MiniMax-H3 target video and audio batch sizes differ")
    audio_present = _validate_audio_present(batch.get("audio_present"), batch_size)

    hidden_states = _stack_single_text_rows(batch.get("mmh3_hidden_states"), "text hidden states")
    token_tags = _stack_single_text_rows(batch.get("mmh3_token_tags"), "text token tags")
    if hidden_states.ndim != 3 or token_tags.ndim != 2 or hidden_states.shape[:2] != token_tags.shape:
        raise ValueError("MiniMax-H3 hidden states and token tags must share [B,L]")
    if token_tags.dtype != torch.int64 or not torch.all((token_tags == 0) | (token_tags == 1)):
        raise ValueError("MiniMax-H3 text token tags must be int64 values 0 or 1")
    has_fl_condition = "latents_first" in batch or "latents_last" in batch
    reference_roles = {}
    for key, value in batch.items():
        match = _RUNTIME_REF_KEY.fullmatch(key)
        if match is not None:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"MiniMax-H3 condition {key} must be a tensor")
            reference_roles.setdefault(int(match.group(1)), {})[match.group(2)] = value
    if has_fl_condition and reference_roles:
        raise ValueError("MiniMax-H3 batch cannot mix FL2VA and Ref2VA condition roles")

    visual_conditions = []
    audio_conditions = []
    condition_geometries = []
    references = []
    if has_fl_condition:
        if "latents_first" not in batch or "latents_last" not in batch:
            raise ValueError("MiniMax-H3 FL2VA batch requires both first and last conditions")
        task = "fl2va"
        for role in ("latents_first", "latents_last"):
            tensor = batch[role]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5 or tensor.shape[1] != 24:
                raise ValueError(f"MiniMax-H3 {role} must be [B,24,F,H,W]")
            if tensor.shape[0] != batch_size:
                raise ValueError(f"MiniMax-H3 {role} batch size does not match the targets")
            visual_conditions.append(tensor)
            condition_geometries.append(H3VideoGeometry(*tensor.shape[2:]))
    elif reference_roles:
        task = "ref2va"
        if set(reference_roles) != set(range(len(reference_roles))):
            raise ValueError("MiniMax-H3 reference indices must be contiguous from 000")
        for index in range(len(reference_roles)):
            roles = reference_roles[index]
            image = roles.get("image")
            video = roles.get("video")
            audio = roles.get("audio")
            if image is not None:
                if video is not None or audio is not None:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} image cannot share video/audio roles")
                if image.ndim != 5 or image.shape[1] != 24 or image.shape[0] != batch_size:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} image must be [B,24,1,H,W]")
                geometry = H3VideoGeometry(*image.shape[2:])
                references.append(H3ReferenceGeometry("image", video=geometry))
                visual_conditions.append(image)
            elif video is not None:
                if video.ndim != 5 or video.shape[1] != 24 or video.shape[0] != batch_size:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} video must be [B,24,F,H,W]")
                geometry = H3VideoGeometry(*video.shape[2:])
                audio_frames = 0
                if audio is not None:
                    if audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2) or audio.shape[0] != batch_size:
                        raise ValueError(f"MiniMax-H3 reference {index:03d} audio must be [B,32,2,A]")
                    audio_frames = audio.shape[-1]
                    audio_conditions.append(audio)
                references.append(H3ReferenceGeometry("video", video=geometry, audio_frames=audio_frames))
                visual_conditions.append(video)
            elif audio is not None:
                if audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2) or audio.shape[0] != batch_size:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} audio must be [B,32,2,A]")
                references.append(H3ReferenceGeometry("audio", audio_frames=audio.shape[-1]))
                audio_conditions.append(audio)
            else:
                raise ValueError(f"MiniMax-H3 reference {index:03d} has no supported role")
    else:
        task = "t2va"

    layout = build_h3_layout(
        task=task,
        text_length=hidden_states.shape[1],
        target_video=H3VideoGeometry(*video_latents.shape[2:]),
        target_audio_frames=audio_latents.shape[-1],
        visual_conditions=tuple(condition_geometries),
        references=tuple(references),
    )
    return _H3RuntimeBatch(
        layout=layout,
        text_hidden_states=hidden_states,
        text_token_tags=token_tags,
        visual_conditions=tuple(visual_conditions),
        audio_conditions=tuple(audio_conditions),
        audio_present=audio_present,
    )


def _shift_noise_amount(base: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * base / (1.0 + (shift - 1.0) * base)


def _sample_base_time(args, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    lower = (0.0 if args.min_timestep is None else float(args.min_timestep)) / 1000.0
    upper = (1000.0 if args.max_timestep is None else float(args.max_timestep)) / 1000.0
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("MiniMax-H3 min_timestep/max_timestep must define a range inside [0,1000]")
    pool = batch.get("timesteps")
    if pool is None:
        base = torch.rand((1,), device=device, dtype=torch.float32)[0]
    else:
        pool = torch.as_tensor(pool, device=device, dtype=torch.float32)
        if pool.numel() != 1:
            raise ValueError("MiniMax-H3 R1 requires exactly one timestep value for its single-item batch")
        base = pool.reshape(())
    return lower + base * (upper - lower)


def _augment_conditions(
    tensors: tuple[torch.Tensor, ...],
    clean: float,
    seeds: torch.Tensor,
    *,
    seed_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    moved = tuple(tensor.to(device) for tensor in tensors)
    if clean == 1.0:
        return moved
    augmented = []
    for tensor in moved:
        samples = []
        for sample, seed in zip(tensor, seeds):
            generator = torch.Generator(device="cpu").manual_seed(int(seed.item()) + seed_offset)
            noise = torch.randn(tuple(sample.shape), generator=generator, dtype=torch.float32, device="cpu").to(
                device=sample.device, dtype=sample.dtype
            )
            samples.append(clean * sample + (1.0 - clean) * noise)
        augmented.append(torch.stack(samples))
    return tuple(augmented)


@dataclass(frozen=True)
class H3SamplingResources:
    """Training-time sampling payload: H3 decodes samples with two separate VAEs."""

    video_vae: torch.nn.Module
    audio_vae: torch.nn.Module


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    audio_spec = H3_AUDIO_SPEC

    def __init__(self):
        super().__init__()
        # per-rank audio-supervision accounting, fed by process_batch; drives the
        # first-epoch warning and the observed fraction saved in metadata
        self._audio_items_seen = 0
        self._audio_supervised_seen = 0

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 1.0
        if getattr(args, "task", None) not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("MiniMax-H3 requires --task t2va, fl2va, or ref2va")
        if args.timestep_sampling != "uniform":
            raise ValueError("MiniMax-H3 supports --timestep_sampling uniform only")
        if args.weighting_scheme != "none":
            raise ValueError("MiniMax-H3 supports --weighting_scheme none only")
        if float(args.discrete_flow_shift) != 1.0:
            raise ValueError("MiniMax-H3 requires --discrete_flow_shift 1.0; use the two H3 shifts instead")
        for name in ("h3_shift_video", "h3_shift_audio"):
            value = float(getattr(args, name))
            if not 0.01 <= value <= 100.0:
                raise ValueError(f"--{name} must be in [0.01,100.0], got {value}")
        for name in ("h3_visual_cond_clean", "h3_audio_cond_clean"):
            value = float(getattr(args, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name} must be in [0.0,1.0], got {value}")
        if args.audio_loss_weight < 0:
            raise ValueError(f"--audio_loss_weight must be nonnegative, got {args.audio_loss_weight}")
        if args.blocks_to_swap is not None and args.blocks_to_swap > 48:
            raise ValueError("--blocks_to_swap for MiniMax-H3 must be <= 48")
        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            raise ValueError("MiniMax-H3 does not support fp8 transformer bases; use --convrot_int8 for a quantized base")
        if getattr(args, "dit_dtype", None) not in {None, "bfloat16", "bf16"}:
            raise ValueError("MiniMax-H3 R1 requires --dit_dtype bfloat16")
        if (
            getattr(args, "block_swap_h2d_only", False)
            and bool(args.blocks_to_swap)
            and not getattr(args, "gradient_checkpointing", False)
        ):
            raise ValueError("MiniMax-H3 --block_swap_h2d_only training requires --gradient_checkpointing")

    def on_transformer_loaded(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
    ) -> None:
        # pre-quantized ConvRot INT8 checkpoints are detected during loading, so these
        # guards can only run once the effective base quantization is known
        is_convrot_int8 = bool(getattr(transformer, "is_convrot_int8", False))
        if args.convrot_int8_bwd == "int8":
            if not is_convrot_int8:
                raise ValueError(
                    "--convrot_int8_bwd int8 requires a ConvRot INT8 base"
                    " (--convrot_int8 or a pre-quantized ConvRot INT8 checkpoint)"
                )
            if torch.device(accelerator.device).type != "cuda":
                raise ValueError("--convrot_int8_bwd int8 requires a CUDA training device")
        if is_convrot_int8 and getattr(args, "base_weights", None):
            raise ValueError("MiniMax-H3 --base_weights cannot be merged into a ConvRot INT8 transformer base")

    def process_sample_prompts(self, args, accelerator, sample_prompts):
        # only the default prepare_sampling needs this seam; guard against future
        # base-side callers silently getting the base NotImplementedError instead
        raise NotImplementedError(
            "MiniMax-H3 prepares sample prompts inside prepare_sampling, which returns joint AV sampling resources"
        )

    def prepare_sampling(self, args, accelerator, vae_dtype):
        del vae_dtype  # the H3 video/audio VAE dtypes are fixed per stage
        if not args.sample_prompts:
            return None, None
        for label in ("video_vae", "audio_vae", "text_encoder"):
            _require_sampling_path(getattr(args, label, None), label)
        sample_prompts = args.sample_prompts
        logger.info("Preparing MiniMax-H3 joint AV training samples from %s", sample_prompts)
        parameters = [_normalize_h3_sample_parameter(args, item) for item in load_prompts(sample_prompts)]
        if not parameters:
            raise ValueError("MiniMax-H3 sample prompt file is empty")
        device = accelerator.device
        decoder = PyAVH3MediaDecoder()

        logger.info("Loading MiniMax-H3 Qwen3-VL text encoder for training samples")
        processor = load_h3_processor(args.processor, revision=args.processor_revision)
        text_encoder = load_h3_text_encoder(
            args.text_encoder,
            processor_path=args.processor,
            revision=args.processor_revision,
            device=device,
            dtype=torch.bfloat16,
            disable_mmap=getattr(args, "disable_numpy_memmap", False),
        )
        text_encoder.eval().requires_grad_(False)
        try:
            for parameter in parameters:
                request = SimpleNamespace(**parameter)
                record = load_generation_record(request)
                raw_visuals, text_visuals = decode_generation_visuals(request, record, decoder)
                presentation = build_presentation(record, args.task, text_visuals)
                hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
                parameter["h3_text_hidden_states"] = hidden_states.to(torch.bfloat16).unsqueeze(0).cpu()
                parameter["h3_text_token_tags"] = token_tags.unsqueeze(0).cpu()
                parameter["_h3_record"] = record
                del raw_visuals, text_visuals, presentation
        finally:
            del processor, text_encoder
            gc.collect()
            clean_memory_on_device(device)

        logger.info("Loading MiniMax-H3 video VAE for training samples")
        has_visual_conditions = args.task != "t2va"
        video_vae_device = device if has_visual_conditions else torch.device("cpu")
        video_vae = load_video_vae(
            args.video_vae,
            device=video_vae_device,
            dtype=VIDEO_VAE_ENCODE_DTYPE if has_visual_conditions else VIDEO_VAE_DECODE_DTYPE,
            disable_mmap=getattr(args, "disable_numpy_memmap", False),
        )
        video_vae.eval().requires_grad_(False)
        try:
            if video_vae.vae_ratio != VIDEO_VAE_SPATIAL_RATIO:
                raise ValueError(f"MiniMax-H3 video VAE spatial ratio must be {VIDEO_VAE_SPATIAL_RATIO}, got {video_vae.vae_ratio}")
            for parameter in parameters:
                if args.task == "t2va":
                    parameter["h3_visual_conditions"] = ()
                    parameter["_h3_visual_geometries"] = ()
                    parameter["_h3_reference_visual_geometries"] = {}
                    parameter["_h3_reference_video_frame_counts"] = {}
                    parameter["_h3_has_audio_conditions"] = False
                    continue
                request = SimpleNamespace(**parameter)
                record = parameter["_h3_record"]
                # Re-decode here instead of retaining hundreds of MB of pixels across model teardown.
                raw_visuals, text_visuals = decode_generation_visuals(request, record, decoder)
                visual_conditions, visual_geometries, reference_visual_geometries = encode_visual_conditions(
                    request,
                    record,
                    raw_visuals,
                    video_vae,
                )
                parameter["h3_visual_conditions"] = visual_conditions
                parameter["_h3_visual_geometries"] = visual_geometries
                parameter["_h3_reference_visual_geometries"] = reference_visual_geometries
                parameter["_h3_reference_video_frame_counts"] = {
                    index: int(raw_visuals[reference.path].shape[0])
                    for index, reference in enumerate(record.references)
                    if reference.type == "video"
                }
                parameter["_h3_has_audio_conditions"] = any(reference.audio is not None for reference in record.references)
                parameter["_h3_record"] = record
                del raw_visuals, text_visuals
        finally:
            video_vae.to(device="cpu", dtype=VIDEO_VAE_DECODE_DTYPE)
            gc.collect()
            clean_memory_on_device(device)

        logger.info("Loading MiniMax-H3 audio VAE for training samples")
        has_audio_conditions = any(parameter["_h3_has_audio_conditions"] for parameter in parameters)
        audio_vae = load_audio_vae(
            args.audio_vae,
            device=device if has_audio_conditions else torch.device("cpu"),
            dtype=torch.float32,
            disable_mmap=getattr(args, "disable_numpy_memmap", False),
        )
        audio_vae.eval().requires_grad_(False)
        try:
            for parameter in parameters:
                if not parameter["_h3_has_audio_conditions"]:
                    parameter["h3_audio_conditions"] = ()
                    parameter["_h3_reference_audio_frames"] = {}
                    continue
                request = SimpleNamespace(**parameter)
                audio_conditions, reference_audio_frames = encode_audio_conditions(
                    request,
                    parameter["_h3_record"],
                    decoder,
                    audio_vae,
                    reference_video_frame_counts=parameter["_h3_reference_video_frame_counts"],
                )
                parameter["h3_audio_conditions"] = audio_conditions
                parameter["_h3_reference_audio_frames"] = reference_audio_frames
        finally:
            audio_vae.to("cpu")
            gc.collect()
            clean_memory_on_device(device)

        for parameter in parameters:
            references = (
                build_reference_geometries(
                    parameter["_h3_record"],
                    parameter["_h3_reference_visual_geometries"],
                    parameter["_h3_reference_audio_frames"],
                )
                if args.task == "ref2va"
                else ()
            )
            parameter["h3_layout"] = build_h3_layout(
                task=args.task,
                text_length=parameter["h3_text_hidden_states"].shape[1],
                target_video=H3VideoGeometry(
                    video_latent_frames(parameter["frame_count"]),
                    parameter["height"] // VIDEO_VAE_SPATIAL_RATIO,
                    parameter["width"] // VIDEO_VAE_SPATIAL_RATIO,
                ),
                target_audio_frames=audio_latent_frames(parameter["frame_count"]),
                visual_conditions=parameter["_h3_visual_geometries"],
                references=references,
            )
            layout = parameter["h3_layout"]
            logger.info(
                "MiniMax-H3 training sample %d: task=%s video=%s audio_frames=%d text_rows=%d packed_rows=%d",
                parameter["enum"],
                layout.task,
                layout.target_video,
                layout.target_audio_frames,
                layout.text_length,
                layout.row_count,
            )
            for key in (
                "_h3_record",
                "_h3_visual_geometries",
                "_h3_reference_visual_geometries",
                "_h3_reference_video_frame_counts",
                "_h3_reference_audio_frames",
                "_h3_has_audio_conditions",
            ):
                parameter.pop(key)
        return parameters, H3SamplingResources(video_vae=video_vae, audio_vae=audio_vae)

    def sample_image_inference(
        self,
        accelerator,
        args,
        transformer,
        dit_dtype,
        sample_resources,
        save_dir,
        sample_parameter,
        epoch,
        steps,
    ):
        del dit_dtype
        if not isinstance(sample_resources, H3SamplingResources):
            raise RuntimeError("MiniMax-H3 training sample VAEs were not prepared")
        video_vae = sample_resources.video_vae
        audio_vae = sample_resources.audio_vae
        layout = sample_parameter["h3_layout"]
        sample_steps = sample_parameter["sample_steps"]
        frame_count = sample_parameter["frame_count"]
        seed = sample_parameter.get("seed")
        if seed is None:
            seed = torch.seed()
            if torch.cuda.is_available():
                torch.cuda.seed()
        else:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        logger.info(
            "MiniMax-H3 joint sample: prompt=%r size=%dx%d frames=%d steps=%d seed=%d",
            sample_parameter.get("prompt", ""),
            sample_parameter["width"],
            sample_parameter["height"],
            frame_count,
            sample_steps,
            seed,
        )

        device = accelerator.device
        has_self_ref_orig_mod = getattr(transformer, "_orig_mod", None) is transformer
        was_training = transformer.training if not has_self_ref_orig_mod else True
        if not has_self_ref_orig_mod:
            transformer.eval()
        try:
            initial_video, initial_audio = initialize_target_latents(
                video_shape=(
                    1,
                    24,
                    layout.target_video.frames,
                    layout.target_video.height,
                    layout.target_video.width,
                ),
                audio_shape=(1, 32, 2, layout.target_audio_frames),
                seed=seed,
                device=device,
                video_dtype=torch.float32,
                audio_dtype=torch.float32,
            )
            visual_conditions, audio_conditions = augment_condition_latents(
                sample_parameter["h3_visual_conditions"],
                sample_parameter["h3_audio_conditions"],
                seed=seed,
                visual_clean=args.h3_visual_cond_clean,
                audio_clean=args.h3_audio_cond_clean,
                device=device,
            )
            text_hidden_states = sample_parameter["h3_text_hidden_states"].to(device=device, dtype=torch.bfloat16)
            text_token_tags = sample_parameter["h3_text_token_tags"].to(device)
            with tqdm(
                total=sample_steps,
                desc=f"MiniMax-H3 sample {sample_parameter.get('enum', 0)}",
                unit="step",
                disable=not getattr(accelerator, "is_local_main_process", True),
            ) as progress:
                sample = sample_joint_av(
                    transformer,
                    layout=layout,
                    text_hidden_states=text_hidden_states,
                    text_token_tags=text_token_tags,
                    initial_video=initial_video,
                    initial_audio=initial_audio,
                    steps=sample_steps,
                    video_shift=args.h3_shift_video,
                    audio_shift=args.h3_shift_audio,
                    visual_condition_latents=visual_conditions,
                    audio_condition_latents=audio_conditions,
                    visual_condition_clean=args.h3_visual_cond_clean,
                    audio_condition_clean=args.h3_audio_cond_clean,
                    step_callback=lambda completed, total: progress.update(1),
                )
            video_latents = sample.video.cpu()
            audio_latents = sample.audio.cpu()
            del sample, initial_video, initial_audio, visual_conditions, audio_conditions
            del text_hidden_states, text_token_tags
            synchronize_device(device)
            clean_memory_on_device(device)

            logger.info("Decoding MiniMax-H3 training sample video")
            video_vae.to(device).eval()
            _, video_dtype = module_device_dtype(video_vae, VIDEO_VAE_DECODE_DTYPE)
            decoded_video = video_vae.decode(video_latents.to(device=device, dtype=video_dtype)).cpu()
            video_vae.to("cpu")
            del video_latents
            clean_memory_on_device(device)

            logger.info("Decoding MiniMax-H3 training sample audio")
            audio_vae.to(device).eval()
            _, audio_dtype = module_device_dtype(audio_vae, torch.float32)
            decoded_audio = audio_vae.decode(audio_latents.to(device=device, dtype=audio_dtype)).cpu()
            audio_vae.to("cpu")
            del audio_latents
            clean_memory_on_device(device)

            decoded = synchronize_decoded_av(decoded_video, decoded_audio, frame_count=frame_count)
            timestamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
            number = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
            original_seed = sample_parameter.get("seed")
            seed_suffix = "" if original_seed is None else f"_{original_seed}"
            prompt_index = sample_parameter.get("enum", 0)
            prefix = "" if args.output_name is None else f"{args.output_name}_"
            output_path = Path(save_dir) / f"{prefix}{number}_{prompt_index:02d}_{timestamp}{seed_suffix}.mp4"
            write_joint_av(decoded, output_path)
            logger.info("Saved MiniMax-H3 joint training sample: %s", output_path)

            try:
                wandb_tracker = accelerator.get_tracker("wandb")
            except (AttributeError, ValueError):
                wandb_tracker = None
            if wandb_tracker is not None:
                try:
                    import wandb
                except ImportError:
                    logger.warning("wandb tracker is active but wandb is not installed")
                else:
                    wandb_tracker.log({f"sample_{prompt_index}": wandb.Video(str(output_path), fps=24)}, step=steps)
            return output_path
        finally:
            video_vae.to("cpu")
            audio_vae.to("cpu")
            if not has_self_ref_orig_mod:
                transformer.train(was_training)
            gc.collect()
            clean_memory_on_device(device)

    def on_epoch_end(self, args: argparse.Namespace, accelerator: Accelerator, network, transformer, epoch: int) -> None:
        del network, transformer
        if epoch != 1 or args.video_only or args.audio_loss_weight <= 0:
            return
        # per-rank observation: under DDP each process only sees its own shard
        if accelerator.is_main_process and self._audio_items_seen > 0 and self._audio_supervised_seen == 0:
            logger.warning(
                "No training item with real audio was seen during the first epoch, so the audio loss is always 0; "
                "if this is intended, consider passing --video_only explicitly"
            )

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        metadata = {
            "ss_minimax_h3_task": args.task,
            "ss_minimax_h3_base_family": "ref2va" if args.task == "ref2va" else "fl2va",
            "ss_minimax_h3_shift_video": args.h3_shift_video,
            "ss_minimax_h3_shift_audio": args.h3_shift_audio,
            "ss_minimax_h3_visual_cond_clean": args.h3_visual_cond_clean,
            "ss_minimax_h3_audio_cond_clean": args.h3_audio_cond_clean,
            "ss_minimax_h3_loss_policy": "video_mean_plus_weighted_audio_mean",
            "ss_minimax_h3_audio_supervision": "presence_gated_training_weight",
            "ss_minimax_h3_audio_loss_weight": args.audio_loss_weight,
            "ss_minimax_h3_video_only": args.video_only,
            "ss_minimax_h3_target_modules": "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2",
            "ss_minimax_h3_convrot_int8": getattr(self, "_convrot_int8_active", args.convrot_int8),
            "ss_minimax_h3_latent_cache_version": "2",
            "ss_minimax_h3_text_cache_version": "1",
        }
        if self._audio_items_seen > 0:
            # fraction observed on this rank so far (exact once a full epoch has run)
            metadata["ss_minimax_h3_supervised_audio_fraction"] = round(self._audio_supervised_seen / self._audio_items_seen, 6)
        return metadata

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        if dit_weight_dtype not in {None, torch.bfloat16}:
            raise ValueError("MiniMax-H3 transformer weights must stay BF16")
        transformer = load_h3_transformer(
            dit_path,
            device=loading_device,
            dtype=torch.bfloat16,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_mmap=getattr(args, "disable_numpy_memmap", False),
            convrot_int8=args.convrot_int8,
            convrot_int8_bwd=args.convrot_int8_bwd,
            # quantization runs on the accelerator device even when the weights load to CPU
            # for block swap (cf. the Krea 2 calc-device fix in #1008)
            quant_device=accelerator.device,
        )
        # pre-quantized ConvRot INT8 checkpoints are detected during loading, so the
        # effective base quantization can differ from the --convrot_int8 flag
        self._convrot_int8_active = bool(getattr(transformer, "is_convrot_int8", False))
        return transformer

    def compile_transformer(self, args, transformer):
        # ConvRot int8 Linears are excluded from compile: the custom autograd.Function +
        # autotuned Triton kernels are not dynamo-traceable (cf. krea2_train_network).
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.blocks],
            disable_linear=bool(self.blocks_to_swap) or bool(getattr(transformer, "is_convrot_int8", False)),
        )

    def scale_shift_latents(self, latents):
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        del batch, timesteps
        audio_latents = kwargs.pop("audio_latents")
        audio_noise = kwargs.pop("audio_noise")
        noisy_audio_input = kwargs.pop("noisy_audio_input")
        runtime = kwargs.pop("runtime")
        model_t_video = kwargs.pop("model_t_video")
        model_t_audio = kwargs.pop("model_t_audio")
        visual_conditions = kwargs.pop("visual_conditions")
        audio_conditions = kwargs.pop("audio_conditions")
        audio_loss_weight = kwargs.pop("audio_loss_weight")
        if kwargs:
            raise TypeError(f"Unexpected MiniMax-H3 call_dit arguments: {sorted(kwargs)}")

        text_hidden_states = runtime.text_hidden_states.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(accelerator.device)
        noisy_audio_input = noisy_audio_input.to(accelerator.device)
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            noisy_audio_input.requires_grad_(True)
            text_hidden_states.requires_grad_(True)
        autocast = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext
        with autocast():
            prediction = transformer(
                video_latents=noisy_model_input,
                audio_latents=noisy_audio_input,
                text_hidden_states=text_hidden_states,
                text_token_tags=runtime.text_token_tags.to(accelerator.device),
                layout=runtime.layout,
                model_t_video=model_t_video,
                model_t_audio=model_t_audio,
                visual_condition_latents=visual_conditions,
                audio_condition_latents=audio_conditions,
                visual_condition_clean=args.h3_visual_cond_clean,
                audio_condition_clean=args.h3_audio_cond_clean,
            )
        return DiTOutput(
            pred=prediction.video,
            target=latents - noise,
            extra={
                "audio_pred": prediction.audio,
                "audio_target": audio_latents - audio_noise,
                "audio_loss_weight": audio_loss_weight,
            },
        )

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        sample_resources,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del network, sample_resources
        # _runtime_batch_plan rejects batches larger than one item (batch_size=1 rule)
        runtime = _runtime_batch_plan(batch, latents)
        self._audio_items_seen += int(runtime.audio_present.numel())
        self._audio_supervised_seen += int(runtime.audio_present.sum().item())
        if runtime.layout.task != args.task:
            raise ValueError(f"MiniMax-H3 --task {args.task} cannot train a {runtime.layout.task.upper()} cache batch")
        device = latents.device
        audio_latents = batch["latents_audio"].to(device=device)
        audio_noise = torch.randn_like(audio_latents)
        base = _sample_base_time(args, batch, device)
        sigma_video = _shift_noise_amount(base, args.h3_shift_video)
        sigma_audio = _shift_noise_amount(base, args.h3_shift_audio)
        model_t_video = 1.0 - sigma_video
        model_t_audio = 1.0 - sigma_audio
        noisy_video = (1.0 - sigma_video) * latents + sigma_video * noise
        noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

        needs_condition_noise = (bool(runtime.visual_conditions) and args.h3_visual_cond_clean != 1.0) or (
            bool(runtime.audio_conditions) and args.h3_audio_cond_clean != 1.0
        )
        condition_seeds = (
            torch.randint(0, 2**63 - 2, (latents.shape[0],), dtype=torch.int64, device="cpu")
            if needs_condition_noise
            else torch.empty(0, dtype=torch.int64)
        )
        visual_conditions = _augment_conditions(
            runtime.visual_conditions,
            args.h3_visual_cond_clean,
            condition_seeds,
            seed_offset=0,
            device=device,
        )
        audio_conditions = _augment_conditions(
            runtime.audio_conditions,
            args.h3_audio_cond_clean,
            condition_seeds,
            seed_offset=1,
            device=device,
        )
        output = self.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            batch,
            noise,
            noisy_video,
            base,
            network_dtype,
            audio_latents=audio_latents,
            audio_noise=audio_noise,
            noisy_audio_input=noisy_audio,
            runtime=runtime,
            model_t_video=model_t_video,
            model_t_audio=model_t_audio,
            visual_conditions=visual_conditions,
            audio_conditions=audio_conditions,
            audio_loss_weight=effective_audio_loss_weights(runtime.audio_present, args),
        )
        return self.compute_loss(args, output, base, noise_scheduler, dit_dtype, network_dtype, global_step)

    def compute_loss(
        self,
        args: argparse.Namespace,
        output: DiTOutput,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del args, timesteps, noise_scheduler, dit_dtype, global_step
        video_loss = torch.nn.functional.mse_loss(
            output.pred.to(network_dtype),
            output.target.to(network_dtype),
            reduction="mean",
        )
        audio_loss_weight = output.extra.get("audio_loss_weight")
        if (
            not isinstance(audio_loss_weight, torch.Tensor)
            or audio_loss_weight.shape != (1,)
            or not torch.isfinite(audio_loss_weight).all().item()
            or audio_loss_weight.item() < 0.0
        ):
            raise ValueError("MiniMax-H3 audio loss weight must be a finite nonnegative float32 tensor with shape [1]")
        weight = audio_loss_weight.item()
        if weight == 0.0:
            audio_loss = video_loss.detach().new_zeros(())
        else:
            audio_loss = torch.nn.functional.mse_loss(
                output.extra["audio_pred"].to(network_dtype),
                output.extra["audio_target"].to(network_dtype),
                reduction="mean",
            )
        return video_loss + weight * audio_loss, {
            "loss/video": video_loss.detach(),
            "loss/audio": audio_loss.detach(),
        }


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(
        timestep_sampling="uniform",
        weighting_scheme="none",
        discrete_flow_shift=1.0,
        network_module="networks.lora_minimax_h3",
    )
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), default=None, help="MiniMax-H3 training task")
    add_audio_train_args(parser)
    parser.add_argument("--h3_shift_video", type=float, default=12.0, help="MiniMax-H3 target-video flow shift")
    parser.add_argument("--h3_shift_audio", type=float, default=3.0, help="MiniMax-H3 target-audio flow shift")
    parser.add_argument(
        "--h3_visual_cond_clean",
        type=float,
        default=0.999,
        help="clean coefficient used to augment MiniMax-H3 visual conditions",
    )
    parser.add_argument(
        "--h3_audio_cond_clean",
        type=float,
        default=1.0,
        help="clean coefficient used to augment MiniMax-H3 audio conditions",
    )
    parser.add_argument(
        "--video_vae",
        type=str,
        default=None,
        help="MiniMax-H3 video VAE checkpoint used for training-time joint AV samples",
    )
    parser.add_argument(
        "--audio_vae",
        type=str,
        default=None,
        help="MiniMax-H3 audio VAE checkpoint used for training-time joint AV samples",
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="MiniMax-H3 Qwen3-VL checkpoint used to encode training sample prompts",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default=DEFAULT_PROCESSOR_ID,
        help="Qwen3-VL processor repository or directory for training samples",
    )
    parser.add_argument("--processor_revision", type=str, default=None)
    parser.add_argument(
        "--h3_allow_experimental_sample_duration",
        action="store_true",
        help="allow training samples outside the released 5-15 second duration range",
    )
    parser.add_argument("--dit_dtype", type=str, default=None, help="MiniMax-H3 DiT dtype; R1 requires bfloat16")
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="quantize the BF16 DiT base weights to ConvRot INT8 at load time (Hadamard rotation + int8 on the "
        "per-block Linears: attn/mlp/adaln_proj). ComfyUI pre-quantized ConvRot INT8 checkpoints (full or pruned) "
        "are detected automatically and do not need this flag. Forward runs fused Triton int8 GEMM (requires "
        "triton / triton-windows; falls back to slower dequantized bf16 matmul without it).",
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward mode for a ConvRot INT8 base. bf16 (default): transient dequantized matmul, most accurate. "
        "int8: reuse the fused int8 GEMM for grad_x (faster, quantizes gradients slightly, requires triton and CUDA).",
    )
    return parser


def main() -> None:
    parser = minimax_h3_setup_parser(setup_parser_common())
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    args.dit_dtype = "bfloat16" if args.dit_dtype is None else args.dit_dtype
    args.vae_dtype = "bfloat16" if args.vae_dtype is None else args.vae_dtype
    MiniMaxH3NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
