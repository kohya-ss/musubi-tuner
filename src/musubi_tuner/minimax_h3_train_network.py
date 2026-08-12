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
    create_sampling_generator,
    initialize_target_latents,
    sample_joint_av,
    synchronize_decoded_av,
    write_joint_av,
)
from musubi_tuner.minimax_h3.text_encoder import (
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    load_h3_uncond_cache,
    normalize_teacher_conditions,
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
    # teacher-matching extras: the FL2VA teacher runs on its own layout with the first/last
    # condition latents and the Picture-prefixed text rows, none of which reach the student
    teacher_layout: H3PackedLayout | None = None
    teacher_text_hidden_states: torch.Tensor | None = None
    teacher_text_token_tags: torch.Tensor | None = None
    teacher_visual_conditions: tuple[torch.Tensor, ...] = ()


def _stack_single_text_rows(value, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 {label} must keep a leading batch axis of size 1")
        return value
    if not isinstance(value, Sequence) or len(value) != 1 or not isinstance(value[0], torch.Tensor):
        raise ValueError(f"MiniMax-H3 {label} must contain exactly one tensor")
    return value[0].unsqueeze(0)


def _collect_fl_conditions(
    batch: dict[str, Any],
    batch_size: int,
    visual_conditions: list[torch.Tensor],
    condition_geometries: list[H3VideoGeometry],
) -> None:
    if "latents_first" not in batch or "latents_last" not in batch:
        raise ValueError("MiniMax-H3 FL2VA batch requires both first and last conditions")
    for role in ("latents_first", "latents_last"):
        tensor = batch[role]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5 or tensor.shape[1] != 24:
            raise ValueError(f"MiniMax-H3 {role} must be [B,24,F,H,W]")
        if tensor.shape[0] != batch_size:
            raise ValueError(f"MiniMax-H3 {role} batch size does not match the targets")
        visual_conditions.append(tensor)
        condition_geometries.append(H3VideoGeometry(*tensor.shape[2:]))


def _runtime_batch_plan(
    batch: dict[str, Any],
    video_latents: torch.Tensor,
    *,
    teacher_matching: bool = False,
) -> _H3RuntimeBatch:
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
    has_teacher_text = "mmh3_teacher_hidden_states" in batch or "mmh3_teacher_token_tags" in batch
    if has_teacher_text and not teacher_matching:
        raise ValueError(
            "MiniMax-H3 text cache contains teacher rows (--teacher_conditions); pass --h3_teacher_matching"
            " or rebuild the text cache without --teacher_conditions"
        )
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
    teacher_layout = None
    teacher_hidden_states = None
    teacher_token_tags = None
    teacher_visual_conditions: list[torch.Tensor] = []
    if teacher_matching:
        # the student trains as T2VA; the first/last latents and the Picture-prefixed text rows
        # feed only the no-grad FL2VA teacher forward
        if reference_roles:
            raise ValueError("MiniMax-H3 teacher matching does not accept Ref2VA condition roles")
        if not has_fl_condition:
            raise ValueError(
                "MiniMax-H3 teacher matching requires FL2VA-style latent caches with first/last conditions;"
                " re-run minimax_h3_cache_latents.py --task fl2va"
            )
        if not has_teacher_text:
            raise ValueError(
                "MiniMax-H3 teacher matching requires teacher text rows;"
                " re-run minimax_h3_cache_text_encoder_outputs.py --task t2va --teacher_conditions first,last"
            )
        task = "t2va"
        teacher_geometries: list[H3VideoGeometry] = []
        _collect_fl_conditions(batch, batch_size, teacher_visual_conditions, teacher_geometries)
        teacher_hidden_states = _stack_single_text_rows(batch.get("mmh3_teacher_hidden_states"), "teacher text hidden states")
        teacher_token_tags = _stack_single_text_rows(batch.get("mmh3_teacher_token_tags"), "teacher text token tags")
        if (
            teacher_hidden_states.ndim != 3
            or teacher_token_tags.ndim != 2
            or teacher_hidden_states.shape[:2] != teacher_token_tags.shape
        ):
            raise ValueError("MiniMax-H3 teacher hidden states and token tags must share [B,L]")
        if teacher_token_tags.dtype != torch.int64 or not torch.all((teacher_token_tags == 0) | (teacher_token_tags == 1)):
            raise ValueError("MiniMax-H3 teacher text token tags must be int64 values 0 or 1")
        if teacher_hidden_states.shape[2] != hidden_states.shape[2]:
            raise ValueError(
                f"MiniMax-H3 teacher text width {teacher_hidden_states.shape[2]} does not match"
                f" the student text width {hidden_states.shape[2]}"
            )
        teacher_layout = build_h3_layout(
            task="fl2va",
            text_length=teacher_hidden_states.shape[1],
            target_video=H3VideoGeometry(*video_latents.shape[2:]),
            target_audio_frames=audio_latents.shape[-1],
            visual_conditions=tuple(teacher_geometries),
        )
    elif has_fl_condition:
        task = "fl2va"
        _collect_fl_conditions(batch, batch_size, visual_conditions, condition_geometries)
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
        teacher_layout=teacher_layout,
        teacher_text_hidden_states=teacher_hidden_states,
        teacher_text_token_tags=teacher_token_tags,
        teacher_visual_conditions=tuple(teacher_visual_conditions),
    )


def _shift_noise_amount(base: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * base / (1.0 + (shift - 1.0) * base)


def _apply_timestep_focus(base: torch.Tensor, low: float, high: float, prob: float) -> torch.Tensor:
    """Deterministic remap of a uniform [0,1) draw that concentrates sampling on a band.

    With probability ``prob`` the sample lands uniformly in [low, high); otherwise it stays
    uniform over [0, 1). The band density becomes prob + (1-prob)*(high-low), so the rest of
    the range (including a base-preservation anchor band) keeps nonzero coverage. The shift
    family s*u/(1+(s-1)u) can only pile mass onto an endpoint, which is why an interior
    decision band needs this mixture form instead.
    """
    if prob <= 0.0:
        return base
    focused = low + (high - low) * (base / prob)
    passthrough = (base - prob) / max(1.0 - prob, 1e-8)
    return torch.where(base < prob, focused, passthrough)


def _dc_attenuated_prediction(pred: torch.Tensor, target: torch.Tensor, dc_weight: float) -> torch.Tensor:
    """Scale the residual's per-channel DC so that mse(pred', target) = mse_ac + dc_weight*mse_dc.

    The DC of the residual is a global color/tone cast (the style axis); attenuating it in the
    loss stops the coherent palette absorption without touching the spatially structured AC
    content. Implemented as a linear map of the residual, so gradients stay exact.
    """
    residual = pred - target
    residual_dc = residual.mean(dim=tuple(range(2, residual.ndim)), keepdim=True)
    return pred - (1.0 - dc_weight**0.5) * residual_dc


def _decomposed_flow_loss(pred: torch.Tensor, target: torch.Tensor, mag_weight: float, dir_weight: float) -> torch.Tensor:
    """Magnitude/direction split of the MSE with the norm-shrinkage coupling removed.

    Exact identity: ||p - t||^2 = (||p|| - ||t||)^2 + 2*||p||*||t||*(1 - cos). In plain MSE the
    direction term's ||p|| factor couples the two components: hedging the direction pays off by
    shrinking the norm, which drives the prediction toward the conditional mean's reduced
    magnitude (the wash-out). Detaching ||p|| in the direction term makes its gradient purely
    rotational, so the magnitude optimum becomes E[||t||] (full per-sample commitment) instead
    of ||E[t]||. At unit weights the loss VALUE still equals the MSE exactly (detaching changes
    gradients only), so loss curves stay comparable across the switch.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    pred_norm = pred_flat.norm()
    target_norm = target_flat.norm()
    eps = 1e-12
    cos = torch.dot(pred_flat, target_flat) / (pred_norm * target_norm + eps)
    magnitude_term = (pred_norm - target_norm).pow(2)
    direction_term = 2.0 * pred_norm.detach() * target_norm * (1.0 - cos)
    return (mag_weight * magnitude_term + dir_weight * direction_term) / pred_flat.numel()


def _preservation_density_compensation(sigma_max: float, focus_min: float, focus_max: float, focus_prob: float) -> float:
    """Loss-weight correction that keeps the preservation anchor's expected gradient share
    invariant under timestep focus.

    Focus concentrates the base-sigma draw on the teaching band and thins the anchor band
    (base sigma > sigma_max) from its uniform share ``1 - sigma_max`` to
    ``(1-p)*(1-sigma_max) + p*overlap/(max-min)``; multiplying each anchor step's loss by
    uniform/focused restores the anchor's per-unit-time pull, so raising the focus does not
    silently weaken the drift protection.
    """
    anchor_width = 1.0 - sigma_max
    if anchor_width <= 0.0 or focus_prob <= 0.0:
        return 1.0
    overlap = max(0.0, focus_max - max(focus_min, sigma_max))
    focused_share = (1.0 - focus_prob) * anchor_width + focus_prob * overlap / (focus_max - focus_min)
    if focused_share <= 0.0:
        # the anchor band is never sampled, so the multiplier is never applied
        return 1.0
    return anchor_width / focused_share


def _prediction_geometry_log(label: str, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cosine similarity, norm ratio, and residual DC/AC split between prediction and target.

    cos isolates the direction component of the residual; norm_ratio (student/target, 1 =
    matched) isolates the magnitude component and drifting above 1 is an early warning for
    burn-style amplification. At the conditional-mean optimum of MSE teacher matching the
    per-bin averages of cos and norm_ratio coincide, so their gap reads as remaining
    training distance and their common limit as the band's irreducible share.

    The residual (student - target) is additionally split into its per-channel mean over
    all non-batch/channel axes (DC: a global color/tone cast, the style component) and the
    remainder (AC: spatially structured content). rms(residual)^2 = dc_rms^2 + ac_rms^2,
    so the split shows whether a shrinking gap is style or content being learned.
    """
    student = prediction.detach().float()
    reference = target.detach().float()
    student_norm = student.flatten().norm()
    reference_norm = reference.flatten().norm()
    eps = 1e-12
    residual = student - reference
    residual_dc = residual.mean(dim=tuple(range(2, residual.ndim)), keepdim=True)
    residual_ac = residual - residual_dc
    return {
        f"teacher/{label}_cos": torch.dot(student.flatten(), reference.flatten()) / (student_norm * reference_norm + eps),
        f"teacher/{label}_norm_ratio": student_norm / (reference_norm + eps),
        f"teacher/{label}_residual_dc_rms": residual_dc.pow(2).mean().sqrt(),
        f"teacher/{label}_residual_ac_rms": residual_ac.pow(2).mean().sqrt(),
    }


def _augment_conditions(tensors: tuple[torch.Tensor, ...], clean: float) -> tuple[torch.Tensor, ...]:
    """Blend independent Gaussian noise into condition latents: clean*x + (1-clean)*eps.

    Training draws fresh noise from the global RNG, like the target noise; only the
    sampling path (minimax_h3.sampling) needs seed-reproducible condition noise.
    """
    if clean == 1.0:
        return tensors
    augmented = []
    for tensor in tensors:
        noise = torch.randn(tuple(tensor.shape), dtype=torch.float32, device=tensor.device).to(tensor.dtype)
        augmented.append(clean * tensor + (1.0 - clean) * noise)
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
        # guidance-loss uncond probe (CPU hidden rows + tags), loaded when
        # --h3_guidance_loss_scale is active
        self._guidance_uncond: tuple[torch.Tensor, torch.Tensor] | None = None

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
        lower = 0.0 if args.min_timestep is None else float(args.min_timestep)
        upper = 1000.0 if args.max_timestep is None else float(args.max_timestep)
        if not 0.0 <= lower <= upper <= 1000.0:
            raise ValueError("MiniMax-H3 min_timestep/max_timestep must define a range inside [0,1000]")
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

        focus_prob = float(args.h3_timestep_focus_prob)
        if not 0.0 <= focus_prob <= 1.0:
            raise ValueError(f"--h3_timestep_focus_prob must be in [0.0,1.0], got {focus_prob}")
        if focus_prob > 0.0:
            focus_min = float(args.h3_timestep_focus_min)
            focus_max = float(args.h3_timestep_focus_max)
            if not 0.0 <= focus_min < focus_max <= 1.0:
                raise ValueError(f"--h3_timestep_focus_min/max must satisfy 0 <= min < max <= 1, got {focus_min}/{focus_max}")
            if args.min_timestep is not None or args.max_timestep is not None:
                raise ValueError("--h3_timestep_focus_prob does not compose with --min_timestep/--max_timestep")
            logger.info(
                "MiniMax-H3 timestep focus: base sigma band [%s,%s) sampled with density %.3f (uniform elsewhere)",
                focus_min,
                focus_max,
                focus_prob + (1.0 - focus_prob) * (focus_max - focus_min),
            )

        guidance_scale = float(args.h3_guidance_loss_scale)
        if guidance_scale < 0.0:
            raise ValueError(f"--h3_guidance_loss_scale must be nonnegative, got {guidance_scale}")
        for name in ("h3_teacher_loss_dc_weight", "h3_teacher_loss_mag_weight", "h3_teacher_preservation_weight"):
            value = float(getattr(args, name))
            if value < 0.0:
                raise ValueError(f"--{name} must be nonnegative, got {value}")
            if value != 1.0 and not getattr(args, "h3_teacher_matching", False):
                raise ValueError(f"--{name} shapes the teacher-matching loss and requires --h3_teacher_matching")
        if getattr(args, "h3_teacher_matching", False):
            if args.task != "t2va":
                raise ValueError("MiniMax-H3 --h3_teacher_matching trains a T2VA student and requires --task t2va")
            if guidance_scale > 0.0:
                raise ValueError(
                    "--h3_teacher_matching and --h3_guidance_loss_scale are mutually exclusive:"
                    " the teacher target already lives in the distilled guided space"
                )
            conditions = normalize_teacher_conditions(args.h3_teacher_conditions)
            sigma_max = float(args.h3_teacher_condition_sigma_max)
            if not 0.0 <= sigma_max <= 1.0:
                raise ValueError(f"--h3_teacher_condition_sigma_max must be in [0.0,1.0], got {sigma_max}")
            logger.info(
                "MiniMax-H3 teacher matching: FL2VA teacher conditioned on %s up to base sigma %s (base-preservation anchor above)",
                conditions,
                sigma_max,
            )
        if args.h3_guidance_loss_scale_audio is not None and float(args.h3_guidance_loss_scale_audio) < 0.0:
            raise ValueError(f"--h3_guidance_loss_scale_audio must be nonnegative, got {args.h3_guidance_loss_scale_audio}")
        if not 0.0 <= float(args.h3_guidance_loss_sigma_min) <= 1.0:
            raise ValueError(f"--h3_guidance_loss_sigma_min must be in [0.0,1.0], got {args.h3_guidance_loss_sigma_min}")
        self._guidance_uncond = None
        if guidance_scale > 0.0:
            if not args.h3_guidance_loss_uncond_cache:
                raise ValueError(
                    "--h3_guidance_loss_scale requires --h3_guidance_loss_uncond_cache"
                    " (write one with minimax_h3_cache_text_encoder_outputs.py --uncond_output)"
                )
            hidden_states, token_tags, metadata = load_h3_uncond_cache(args.h3_guidance_loss_uncond_cache)
            self._guidance_uncond = (hidden_states, token_tags)
            logger.info(
                "MiniMax-H3 guidance loss: scale=%s scale_audio=%s sigma_min=%s uncond=%r (%d rows)",
                guidance_scale,
                self._guidance_audio_scale(args),
                args.h3_guidance_loss_sigma_min,
                metadata.get("text", "?"),
                hidden_states.shape[0],
            )
        elif args.h3_guidance_loss_uncond_cache:
            logger.warning("--h3_guidance_loss_uncond_cache is ignored because --h3_guidance_loss_scale is 0")

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
        processor = load_h3_processor()
        text_encoder = load_h3_text_encoder(
            args.text_encoder,
            device=device,
            dtype=torch.bfloat16,
            disable_mmap=getattr(args, "disable_numpy_memmap", False),
            nvfp4_scaled_mm=getattr(args, "nvfp4_scaled_mm", False),
            blocks_to_swap=getattr(args, "text_encoder_blocks_to_swap", 0),
            attn_mode=getattr(args, "text_encoder_attn_mode", None),
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
                sample_parameter["h3_visual_conditions"],
                sample_parameter["h3_audio_conditions"],
                generator=generator,
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
        if float(args.h3_guidance_loss_scale) > 0.0:
            metadata["ss_minimax_h3_guidance_loss_scale"] = args.h3_guidance_loss_scale
            metadata["ss_minimax_h3_guidance_loss_scale_audio"] = self._guidance_audio_scale(args)
            metadata["ss_minimax_h3_guidance_loss_sigma_min"] = args.h3_guidance_loss_sigma_min
        if getattr(args, "h3_teacher_matching", False):
            metadata["ss_minimax_h3_teacher_matching"] = True
            metadata["ss_minimax_h3_teacher_conditions"] = normalize_teacher_conditions(args.h3_teacher_conditions)
            metadata["ss_minimax_h3_teacher_condition_sigma_max"] = args.h3_teacher_condition_sigma_max
            metadata["ss_minimax_h3_teacher_loss"] = "decomposed_mag_dir"
            metadata["ss_minimax_h3_teacher_loss_mag_weight"] = args.h3_teacher_loss_mag_weight
            metadata["ss_minimax_h3_teacher_loss_dc_weight"] = args.h3_teacher_loss_dc_weight
            metadata["ss_minimax_h3_teacher_preservation_weight"] = args.h3_teacher_preservation_weight
        if float(args.h3_timestep_focus_prob) > 0.0:
            metadata["ss_minimax_h3_timestep_focus_min"] = args.h3_timestep_focus_min
            metadata["ss_minimax_h3_timestep_focus_max"] = args.h3_timestep_focus_max
            metadata["ss_minimax_h3_timestep_focus_prob"] = args.h3_timestep_focus_prob
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
            prune_adaln=args.prune_adaln,
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
        del batch  # timesteps (the pre-shift base sigma) gates the guidance-loss forward
        audio_latents = kwargs.pop("audio_latents")
        audio_noise = kwargs.pop("audio_noise")
        noisy_audio_input = kwargs.pop("noisy_audio_input")
        runtime = kwargs.pop("runtime")
        model_t_video = kwargs.pop("model_t_video")
        model_t_audio = kwargs.pop("model_t_audio")
        visual_conditions = kwargs.pop("visual_conditions")
        audio_conditions = kwargs.pop("audio_conditions")
        audio_loss_weight = kwargs.pop("audio_loss_weight")
        network = kwargs.pop("network", None)
        teacher_visual_conditions = kwargs.pop("teacher_visual_conditions", ())
        if kwargs:
            raise TypeError(f"Unexpected MiniMax-H3 call_dit arguments: {sorted(kwargs)}")

        text_hidden_states = runtime.text_hidden_states.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(accelerator.device)
        noisy_audio_input = noisy_audio_input.to(accelerator.device)
        autocast = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext

        video_target = latents - noise
        audio_target = audio_latents - audio_noise
        guidance_log: dict[str, torch.Tensor] = {}
        if self._guidance_uncond is not None:
            # the uncond forward runs before the grad forward so the block-swap offloader
            # keeps its forward->backward alternation and no autograd graph is live yet
            applied = float(timesteps) >= float(args.h3_guidance_loss_sigma_min)
            # the drawn pre-shift sigma, so the logged gap magnitudes can be binned by noise level
            guidance_log["guidance/base_sigma"] = torch.as_tensor(float(timesteps))
            guidance_log["guidance/applied"] = torch.tensor(1.0 if applied else 0.0)
            if applied:
                video_target, audio_target, gap_log = self._apply_guidance_loss_targets(
                    args,
                    accelerator,
                    transformer,
                    runtime,
                    noisy_model_input,
                    noisy_audio_input,
                    model_t_video,
                    model_t_audio,
                    visual_conditions,
                    audio_conditions,
                    video_target,
                    audio_target,
                    network_dtype,
                )
                guidance_log.update(gap_log)
        if getattr(args, "h3_teacher_matching", False):
            video_target, audio_target, teacher_log = self._apply_teacher_matching_targets(
                args,
                accelerator,
                transformer,
                network,
                runtime,
                noisy_model_input,
                noisy_audio_input,
                model_t_video,
                model_t_audio,
                teacher_visual_conditions,
                video_target,
                audio_target,
                network_dtype,
                timesteps,
            )
            guidance_log.update(teacher_log)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            noisy_audio_input.requires_grad_(True)
            text_hidden_states.requires_grad_(True)
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
        if getattr(args, "h3_teacher_matching", False):
            # direction/magnitude decomposition of the student-teacher residual (observation
            # only): MSE mixes both, but content errors are direction-flavored while
            # burn/wash-out drift is magnitude-flavored, so the split (binned by
            # teacher/base_sigma) shows which one dominates in each noise band
            guidance_log.update(_prediction_geometry_log("video", prediction.video, video_target))
            guidance_log.update(_prediction_geometry_log("audio", prediction.audio, audio_target))
        return DiTOutput(
            pred=prediction.video,
            target=video_target,
            extra={
                "audio_pred": prediction.audio,
                "audio_target": audio_target,
                "audio_loss_weight": audio_loss_weight,
                "guidance_log": guidance_log,
            },
        )

    def _guidance_audio_scale(self, args: argparse.Namespace) -> float:
        if args.h3_guidance_loss_scale_audio is not None:
            return float(args.h3_guidance_loss_scale_audio)
        return float(args.h3_guidance_loss_scale)

    def _apply_guidance_loss_targets(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        runtime: _H3RuntimeBatch,
        noisy_model_input: torch.Tensor,
        noisy_audio_input: torch.Tensor,
        model_t_video,
        model_t_audio,
        visual_conditions: tuple[torch.Tensor, ...],
        audio_conditions: tuple[torch.Tensor, ...],
        video_target: torch.Tensor,
        audio_target: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Re-anchor the flow targets in the CFG-amplified space of the distilled base.

        The released H3 weights are CFG-distilled: ``g(c) = u + s*(c(c) - u)``. Training
        on the plain velocity target would pull the student out of that amplified space
        (de-distillation drift). Instead the target is rebuilt as
        ``u + scale*(v - u)`` where ``u`` is the model's own prediction under the uncond
        probe -- the true velocity slots into the CFG identity where the conditional
        prediction would go. The probe (a single space by default) was screened against
        the released checkpoint: see docs/minimax_h3.md.

        The uncond forward is no_grad but keeps the LoRA active, matching how the
        adapted model will be run at inference; only the text condition is swapped, all
        visual/audio conditions stay (the same augmented tensors as the main forward).
        """
        uncond_hidden, uncond_tags = self._guidance_uncond
        if uncond_hidden.shape[1] != runtime.text_hidden_states.shape[2]:
            raise ValueError(
                f"MiniMax-H3 guidance-loss uncond cache width {uncond_hidden.shape[1]} does not match"
                f" the text cache width {runtime.text_hidden_states.shape[2]}"
            )
        uncond_layout = build_h3_layout(
            task=runtime.layout.task,
            text_length=uncond_hidden.shape[0],
            target_video=runtime.layout.target_video,
            target_audio_frames=runtime.layout.target_audio_frames,
            visual_conditions=runtime.layout.visual_conditions,
            references=runtime.layout.references,
        )
        autocast = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext
        with torch.no_grad(), autocast():
            uncond = transformer(
                video_latents=noisy_model_input,
                audio_latents=noisy_audio_input,
                text_hidden_states=uncond_hidden.to(device=accelerator.device, dtype=network_dtype).unsqueeze(0),
                text_token_tags=uncond_tags.to(accelerator.device).unsqueeze(0),
                layout=uncond_layout,
                model_t_video=model_t_video,
                model_t_audio=model_t_audio,
                visual_condition_latents=visual_conditions,
                audio_condition_latents=audio_conditions,
                visual_condition_clean=args.h3_visual_cond_clean,
                audio_condition_clean=args.h3_audio_cond_clean,
            )
        uncond_video = uncond.video.detach().float()
        uncond_audio = uncond.audio.detach().float()
        video_gap = video_target.float() - uncond_video
        audio_gap = audio_target.float() - uncond_audio
        # the sigma-binned gap magnitudes are the measured guidance signal; they feed the
        # sigma_min gate and any future scale schedule
        gap_log = {
            "guidance/video_gap_rms": video_gap.pow(2).mean().sqrt().detach(),
            "guidance/audio_gap_rms": audio_gap.pow(2).mean().sqrt().detach(),
        }
        video_target = uncond_video + float(args.h3_guidance_loss_scale) * video_gap
        audio_target = uncond_audio + self._guidance_audio_scale(args) * audio_gap
        return video_target, audio_target, gap_log

    def _apply_teacher_matching_targets(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        runtime: _H3RuntimeBatch,
        noisy_model_input: torch.Tensor,
        noisy_audio_input: torch.Tensor,
        model_t_video,
        model_t_audio,
        teacher_visual_conditions: tuple[torch.Tensor, ...],
        video_target: torch.Tensor,
        audio_target: torch.Tensor,
        network_dtype: torch.dtype,
        base_sigma,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Replace both flow targets with the frozen base model's predictions.

        The teacher shares weights with the student: the same transformer runs once with the
        LoRA disabled, conditioned on the real first/last frames and the Picture-prefixed text
        rows -- privileged information the T2VA student never sees. The teacher prediction
        lives in the distilled guided space, so no guidance scale or uncond probe is needed
        and the de-distillation drift of plain flow targets is structurally avoided; this is
        why the loss is mutually exclusive with the contrastive guidance loss. The loss keeps
        an irreducible floor (endpoint content the text alone cannot determine), so read the
        sigma-binned teacher/*_flow_gap_rms logs rather than expecting it to reach zero. The
        audio target degenerates to a base-preservation anchor: the visual endpoints carry
        almost no audio information, so real audio content is not learned in this mode.

        Above --h3_teacher_condition_sigma_max the teacher instead runs on the student's own
        text and layout with no endpoint conditions, turning the target into a pure
        base-preservation anchor. The teacher target is a noiseless regression label
        (deterministic per x_t), and near pure noise the endpoint content is unpredictable
        from the text, so unrestricted teaching there rapidly overwrites the base composition
        prior with the dataset mean; anchoring that band to the base also counters the
        collateral drift of the LoRA's shared weights.
        """
        if runtime.teacher_layout is None or runtime.teacher_text_hidden_states is None:
            raise RuntimeError("MiniMax-H3 teacher matching batch plan is missing the teacher layout")
        if network is None:
            raise RuntimeError("MiniMax-H3 teacher matching requires the LoRA network to disable it for the teacher forward")
        unwrap = getattr(accelerator, "unwrap_model", None)
        base_network = unwrap(network) if callable(unwrap) else network
        if not hasattr(base_network, "set_enabled"):
            raise RuntimeError(
                "MiniMax-H3 teacher matching requires a network exposing set_enabled (e.g. networks.lora_minimax_h3)"
            )
        conditioned = float(base_sigma) <= float(args.h3_teacher_condition_sigma_max)
        if conditioned:
            teacher_text = runtime.teacher_text_hidden_states
            teacher_tags = runtime.teacher_text_token_tags
            teacher_layout = runtime.teacher_layout
        else:
            # base-preservation anchor: same text and layout as the student, LoRA off
            teacher_text = runtime.text_hidden_states
            teacher_tags = runtime.text_token_tags
            teacher_layout = runtime.layout
            teacher_visual_conditions = ()
        autocast = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext
        # the teacher forward runs before the grad forward so the block-swap offloader keeps
        # its forward->backward alternation and no autograd graph is live yet
        base_network.set_enabled(False)
        try:
            with torch.no_grad(), autocast():
                teacher = transformer(
                    video_latents=noisy_model_input,
                    audio_latents=noisy_audio_input,
                    text_hidden_states=teacher_text.to(device=accelerator.device, dtype=network_dtype),
                    text_token_tags=teacher_tags.to(accelerator.device),
                    layout=teacher_layout,
                    model_t_video=model_t_video,
                    model_t_audio=model_t_audio,
                    visual_condition_latents=teacher_visual_conditions,
                    audio_condition_latents=(),
                    visual_condition_clean=args.h3_visual_cond_clean,
                    audio_condition_clean=args.h3_audio_cond_clean,
                )
        finally:
            base_network.set_enabled(True)
        teacher_video = teacher.video.detach().float()
        teacher_audio = teacher.audio.detach().float()
        # the flow-gap magnitudes measure how far the teacher deviates from the raw velocity
        # target (guidance amplification + endpoint information), binned by base sigma
        teacher_log = {
            "teacher/base_sigma": torch.as_tensor(float(base_sigma)),
            "teacher/conditioned": torch.tensor(1.0 if conditioned else 0.0),
            "teacher/video_flow_gap_rms": (teacher_video - video_target.float()).pow(2).mean().sqrt().detach(),
            "teacher/audio_flow_gap_rms": (teacher_audio - audio_target.float()).pow(2).mean().sqrt().detach(),
        }
        return teacher_video, teacher_audio, teacher_log

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
        del sample_resources
        # _runtime_batch_plan rejects batches larger than one item (batch_size=1 rule)
        runtime = _runtime_batch_plan(batch, latents, teacher_matching=bool(getattr(args, "h3_teacher_matching", False)))
        self._audio_items_seen += int(runtime.audio_present.numel())
        self._audio_supervised_seen += int(runtime.audio_present.sum().item())
        if runtime.layout.task != args.task:
            raise ValueError(f"MiniMax-H3 --task {args.task} cannot train a {runtime.layout.task.upper()} cache batch")
        device = latents.device
        audio_latents = batch["latents_audio"].to(device=device)
        audio_noise = torch.randn_like(audio_latents)
        pool = batch.get("timesteps")
        if pool is not None and len(pool) != 1:
            raise ValueError("MiniMax-H3 R1 requires exactly one timestep value for its single-item batch")
        base = self.sample_timesteps(args, 1, pool, latents, device)[0]
        base = _apply_timestep_focus(
            base,
            float(getattr(args, "h3_timestep_focus_min", 0.4)),
            float(getattr(args, "h3_timestep_focus_max", 0.8)),
            float(getattr(args, "h3_timestep_focus_prob", 0.0)),
        )
        sigma_video = _shift_noise_amount(base, args.h3_shift_video)
        sigma_audio = _shift_noise_amount(base, args.h3_shift_audio)
        model_t_video = 1.0 - sigma_video
        model_t_audio = 1.0 - sigma_audio
        noisy_video = (1.0 - sigma_video) * latents + sigma_video * noise
        noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

        visual_conditions = _augment_conditions(
            tuple(tensor.to(device) for tensor in runtime.visual_conditions), args.h3_visual_cond_clean
        )
        audio_conditions = _augment_conditions(
            tuple(tensor.to(device) for tensor in runtime.audio_conditions), args.h3_audio_cond_clean
        )
        # the teacher's first/last conditions get the same per-step augmentation as FL2VA training
        teacher_visual_conditions = _augment_conditions(
            tuple(tensor.to(device) for tensor in runtime.teacher_visual_conditions), args.h3_visual_cond_clean
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
            network=network,
            teacher_visual_conditions=teacher_visual_conditions,
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
        del timesteps, noise_scheduler, dit_dtype, global_step
        teacher_matching = bool(getattr(args, "h3_teacher_matching", False))
        guidance_log = output.extra.get("guidance_log") or {}
        conditioned_flag = guidance_log.get("teacher/conditioned")
        conditioned = bool(conditioned_flag.item() > 0.5) if isinstance(conditioned_flag, torch.Tensor) else True
        dc_weight = float(getattr(args, "h3_teacher_loss_dc_weight", 1.0))
        mag_weight = float(getattr(args, "h3_teacher_loss_mag_weight", 1.0))

        def flow_loss(pred: torch.Tensor, target: torch.Tensor, *, attenuate_dc: bool) -> torch.Tensor:
            if not teacher_matching:
                return torch.nn.functional.mse_loss(pred, target, reduction="mean")
            # the DC attenuation applies only to conditioned teaching steps: on preservation
            # steps the DC penalty is exactly what pulls palette drift back to the base
            if attenuate_dc and conditioned and dc_weight != 1.0:
                pred = _dc_attenuated_prediction(pred, target, dc_weight)
            return _decomposed_flow_loss(pred, target, mag_weight, 1.0)

        # the DC attenuation targets the video palette axis; the audio anchor keeps its full DC
        video_loss = flow_loss(output.pred.to(network_dtype), output.target.to(network_dtype), attenuate_dc=True)
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
            audio_loss = flow_loss(
                output.extra["audio_pred"].to(network_dtype),
                output.extra["audio_target"].to(network_dtype),
                attenuate_dc=False,
            )
        logs = {
            "loss/video": video_loss.detach(),
            "loss/audio": audio_loss.detach(),
        }
        logs.update(guidance_log)
        total_loss = video_loss + weight * audio_loss
        if teacher_matching and not conditioned:
            # preservation-anchor step: user weight on top of the automatic focus compensation,
            # so raising the timestep focus does not silently weaken the drift protection.
            # loss/video and loss/audio are logged unweighted to keep sigma-binned reads comparable
            multiplier = float(getattr(args, "h3_teacher_preservation_weight", 1.0)) * _preservation_density_compensation(
                float(args.h3_teacher_condition_sigma_max),
                float(getattr(args, "h3_timestep_focus_min", 0.4)),
                float(getattr(args, "h3_timestep_focus_max", 0.8)),
                float(getattr(args, "h3_timestep_focus_prob", 0.0)),
            )
            if multiplier != 1.0:
                total_loss = total_loss * multiplier
        return total_loss, logs


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
        "--nvfp4_scaled_mm",
        action="store_true",
        help="use W4A4 scaled_mm for an NVFP4 text encoder (requires PyTorch 2.10+ and Blackwell; default is weight-only dequantization)",
    )
    parser.add_argument(
        "--text_encoder_blocks_to_swap",
        type=int,
        default=0,
        help="number of the 50 Qwen3-VL decoder layers to stream from CPU while encoding training sample prompts"
        " (0 = disabled, 50 = minimum VRAM; requires CUDA; unrelated to the transformer's --blocks_to_swap)",
    )
    parser.add_argument(
        "--text_encoder_attn_mode",
        choices=("sdpa", "flash_attention_2", "eager"),
        default=None,
        help="attention implementation for the sample-prompt text encoder (default: transformers default, sdpa)."
        " Use flash_attention_2 for long presentations: sdpa falls back to the O(L^2) math kernel and can OOM",
    )
    parser.add_argument(
        "--h3_allow_experimental_sample_duration",
        action="store_true",
        help="allow training samples outside the released 5-15 second duration range",
    )
    parser.add_argument(
        "--h3_guidance_loss_scale",
        type=float,
        default=0.0,
        help="guidance-distillation countermeasure: rebuild the flow target as uncond + scale*(target - uncond) using a"
        " no-grad uncond forward per step (0 = disabled; field reports suggest 3-4). Requires"
        " --h3_guidance_loss_uncond_cache.",
    )
    parser.add_argument(
        "--h3_guidance_loss_scale_audio",
        type=float,
        default=None,
        help="separate guidance-loss scale for the audio target (default: same as --h3_guidance_loss_scale)",
    )
    parser.add_argument(
        "--h3_guidance_loss_sigma_min",
        type=float,
        default=0.0,
        help="skip the guidance-loss forward when the drawn pre-shift base sigma is below this threshold"
        " (0 = always on; the text-guidance signal concentrates at high sigma, so gating saves the extra"
        " forward where the correction is negligible)",
    )
    parser.add_argument(
        "--h3_guidance_loss_uncond_cache",
        type=str,
        default=None,
        help="uncond probe embedding for the guidance loss, written by minimax_h3_cache_text_encoder_outputs.py --uncond_output",
    )
    parser.add_argument(
        "--h3_teacher_matching",
        action="store_true",
        help="teacher-matching training (--task t2va only): replace both flow targets with the frozen base model's"
        " FL2VA predictions conditioned on the real first/last frames (one extra no-grad forward per step)."
        " Requires FL2VA-style latent caches and a text cache written with --teacher_conditions first,last."
        " The teacher targets live in the distilled guided space, so this replaces (and is mutually exclusive"
        " with) --h3_guidance_loss_scale; audio degenerates to a base-preservation anchor, so real audio content"
        " is not learned in this mode.",
    )
    parser.add_argument(
        "--h3_teacher_conditions",
        type=str,
        default="first,last",
        help="visual conditions handed to the FL2VA teacher (currently only 'first,last' is supported;"
        " the flag exists so single-sided or anchored teachers can be added later)",
    )
    parser.add_argument(
        "--h3_teacher_condition_sigma_max",
        type=float,
        default=0.75,
        help="teacher matching only: above this drawn base sigma (pre-shift, 1 = pure noise) the teacher drops the"
        " endpoint conditions and runs on the student's own text, turning the target into a pure base-preservation"
        " anchor. Near pure noise the endpoint content is unpredictable from the text, so unrestricted teaching"
        " there rapidly overwrites the base composition prior. The identity-decision band was measured at base"
        " sigma 0.6-0.75 on diverse character data, so the default keeps it in the teaching band; lower toward"
        " 0.4-0.5 for low-diversity data (1.0 = always conditioned, unprotected)",
    )
    parser.add_argument(
        "--h3_teacher_loss_dc_weight",
        type=float,
        default=1.0,
        help="teacher matching only: weight of the video residual's per-channel DC component on conditioned"
        " teaching steps (1.0 = unchanged). The DC axis is a global color/tone cast, so lowering it (e.g. 0.0-0.3)"
        " stops the coherent absorption of the dataset's palette while leaving the spatially structured content"
        " signal untouched. Preservation-anchor steps and the audio anchor always keep their full DC penalty --"
        " there it is what pulls palette drift back to the base",
    )
    parser.add_argument(
        "--h3_teacher_loss_mag_weight",
        type=float,
        default=1.0,
        help="teacher matching only: weight of the magnitude term of the decomposed loss, relative to the"
        " direction term fixed at 1.0. At 1.0 the loss value equals the plain MSE (only the gradient geometry"
        " differs); lower it to prioritize direction matching (0 = pure direction)",
    )
    parser.add_argument(
        "--h3_teacher_preservation_weight",
        type=float,
        default=1.0,
        help="teacher matching only: loss weight of preservation-anchor steps (base sigma above"
        " --h3_teacher_condition_sigma_max), applied on top of an automatic correction that keeps the anchor's"
        " expected gradient share invariant under --h3_timestep_focus_prob. Raise it if the anchor-band drift"
        " (teacher/*_residual_dc_rms on unconditioned steps) keeps growing",
    )
    parser.add_argument(
        "--h3_timestep_focus_min",
        type=float,
        default=0.4,
        help="lower edge of the base-sigma focus band for --h3_timestep_focus_prob",
    )
    parser.add_argument(
        "--h3_timestep_focus_max",
        type=float,
        default=0.8,
        help="upper edge of the base-sigma focus band for --h3_timestep_focus_prob",
    )
    parser.add_argument(
        "--h3_timestep_focus_prob",
        type=float,
        default=0.0,
        help="probability of drawing the training base sigma uniformly from the focus band instead of [0,1)"
        " (0 = uniform sampling, unchanged). Concentrates training on the band where content is decided while the"
        " rest of the range, including the base-preservation anchor band, keeps (1-prob) of the samples. The band"
        " density becomes prob + (1-prob)*(max-min)",
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
    parser.add_argument(
        "--prune_adaln",
        action="store_true",
        help="prune the AdaLN projections of a full BF16 DiT at load time (mean-centered rank-8 basis of the "
        "time-embedding curve, computed on the fly; the time embedder is retained, so timesteps stay exact and "
        "continuous). Cuts the AdaLN weights from ~26 GB to a few MB with near-identical outputs. Published pruned "
        "checkpoints are already pruned and do not need this flag; pre-quantized ConvRot INT8 checkpoints are "
        "rejected. Combines with --convrot_int8 to reproduce the published pruned INT8 scope from a full BF16 file.",
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
