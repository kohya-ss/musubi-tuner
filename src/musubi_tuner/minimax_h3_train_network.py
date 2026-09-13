from __future__ import annotations

import argparse
import gc
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_MINIMAX_H3,
    ARCHITECTURE_MINIMAX_H3_FULL,
    round_down_frame_count,
)
from musubi_tuner.minimax_h3.args import add_h3_sampling_args, add_h3_text_encoder_args, add_h3_vae_args
from musubi_tuner.minimax_h3.audio_vae import load_audio_vae
from musubi_tuner.minimax_h3.generation_inputs import (
    DEFAULT_FRAME_COUNT,
    VIDEO_VAE_SPATIAL_RATIO,
    H3GenerationRequest,
    build_generation_layout,
    build_reference_geometries,
    decode_generation_visuals,
    encode_audio_conditions,
    encode_visual_conditions,
    load_generation_record,
    reference_video_frame_counts,
    request_overrides,
    require_path,
    validate_generation_request,
)
from musubi_tuner.minimax_h3.media import (
    H3_AUDIO_SPEC,
    H3_TASKS,
    H3Record,
    PyAVH3MediaDecoder,
    module_device_dtype,
    reject_one_frame_audio_references,
)
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import (
    FL_CONDITION_ROLES,
    FRAME_RESCALE,
    H3ConditionRole,
    H3PackedLayout,
    H3ReferenceGeometry,
    H3TimeOverrides,
    H3VideoGeometry,
    ONE_FRAME_VIDEO_LATENT_FRAMES,
    build_h3_layout,
    one_frame_condition_roles,
    parse_condition_role,
    validate_clean_coefficient,
)
from musubi_tuner.minimax_h3.sampling import (
    H3DecodedAV,
    augment_condition_latents,
    sample_joint_av_latents,
    shift_sigma,
    synchronize_decoded_av,
    validate_shift,
    write_joint_av,
)
from musubi_tuner.minimax_h3.text_encoder import (
    TEACHER_CONDITIONS_REF,
    TEACHER_CONDITIONS_SUBJECT_REF,
    TEACHER_TEXT_CACHE_PREFIXES,
    build_presentation,
    encode_h3_presentation,
    load_h3_processor,
    load_h3_text_encoder,
    load_h3_uncond_cache,
    normalize_teacher_conditions,
)
from musubi_tuner.minimax_h3.video_vae import VIDEO_VAE_DECODE_DTYPE, VIDEO_VAE_ENCODE_DTYPE, load_video_vae
from musubi_tuner.training.audio_loss import add_audio_train_args, effective_audio_loss_weights
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer, wandb_tracker_and_module
from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)


# validated --h3_teacher_condition_sigma_max for the endpoint (first,last) and clip (ref) teachers:
# above it the conditioned content is unpredictable from the text (and the FL2VA weights stop
# aligning a reference near ~0.85), so the band is better spent as a base-preservation anchor
SIGMA_MAX_RECOMMENDED_COMPLETE_INFORMATION = 0.75


def _sample_frame_count(frame_count: int) -> int:
    """The house sample convention on the H3 frame grid: a one-frame sample stays 1, an off-grid
    video frame count rounds down onto 17*n+5 instead of failing."""
    if frame_count == 1:
        return 1
    # the H3 grid is not a stride, so the stride argument is unused for this architecture
    return round_down_frame_count(frame_count, ARCHITECTURE_MINIMAX_H3, 0)


def _sample_request(args: argparse.Namespace, parameter: Mapping[str, Any]) -> H3GenerationRequest:
    """The generation request of one ``--sample_prompts`` entry: the same contract as the
    generation CLI (generation_inputs), with the training run's task and sampler flags as the
    base and the prompt file's directory as the root of relative reference paths."""
    sample_task = parameter.get("task", args.task)
    if sample_task != args.task:
        raise ValueError(f"MiniMax-H3 sample prompt task {sample_task!r} does not match the training --task {args.task!r}")
    prompt_directory = Path(args.sample_prompts).expanduser().resolve().parent if args.sample_prompts else Path.cwd()
    overrides = request_overrides(parameter)
    if args.h3_allow_experimental_sample_duration:
        overrides["allow_experimental_duration"] = True
    requested_frame_count = overrides.get("frame_count", DEFAULT_FRAME_COUNT)
    frame_count = _sample_frame_count(requested_frame_count)
    if frame_count != requested_frame_count:
        logger.warning("MiniMax-H3 sample frame count %d was rounded down to %d (17*n+5)", requested_frame_count, frame_count)
    overrides["frame_count"] = frame_count
    reference_jsonl = overrides.get("reference_jsonl")
    if reference_jsonl and not Path(reference_jsonl).expanduser().is_absolute():
        # relative reference_jsonl paths resolve from the prompt file's directory, falling
        # back to the historical CWD-relative behavior
        prompt_relative = prompt_directory / Path(reference_jsonl).expanduser()
        if prompt_relative.exists():
            overrides["reference_jsonl"] = str(prompt_relative)
    request = H3GenerationRequest(
        task=args.task,
        ref_base_directory=prompt_directory,
        h3_shift_video=args.h3_shift_video,
        h3_shift_audio=args.h3_shift_audio,
        h3_visual_cond_clean=args.h3_visual_cond_clean,
        h3_audio_cond_clean=args.h3_audio_cond_clean,
    )
    request = replace(request, **overrides)
    validate_generation_request(request)
    return request


@dataclass(frozen=True)
class _H3RuntimeBatch:
    """One training batch as the transformer consumes it: the packed layout, the text rows, the
    condition latents in layout order, and the audio-presence flags. The batch's cache entries
    are read by the training --task (t2va uses none, fl2va the first/last or timed cond_
    latents, ref2va the numbered references); a configured teacher additionally reads its own
    text rows and conditions, which never reach the student."""

    layout: H3PackedLayout
    text_hidden_states: torch.Tensor
    text_token_tags: torch.Tensor
    visual_conditions: tuple[torch.Tensor, ...]
    audio_conditions: tuple[torch.Tensor, ...]
    audio_present: torch.Tensor
    teacher_layout: H3PackedLayout | None = None
    teacher_text_hidden_states: torch.Tensor | None = None
    teacher_text_token_tags: torch.Tensor | None = None
    teacher_visual_conditions: tuple[torch.Tensor, ...] = ()
    teacher_audio_conditions: tuple[torch.Tensor, ...] = ()
    # warn-once observations about the batch's cache entries (see MiniMaxH3NetworkTrainer._notice)
    notices: tuple[str, ...] = ()


_REQUIRED_BATCH_KEYS = ("latents_audio", "audio_present", "mmh3_hidden_states", "mmh3_token_tags")


class _BatchEntries:
    """The H3 cache entries of one batch with consumption tracking: the condition latents
    (``latents_<role>``), the teacher text rows and the one-frame index tensors. Whatever the
    task and the teacher leave unread is reported, since it usually means the caches were
    written for another task."""

    def __init__(self, batch: Mapping[str, Any]):
        self._batch = batch
        self.unread = {
            key
            for key in batch
            if (key.startswith("latents_") and key != "latents_audio")
            or key.startswith("mmh3_teacher_")
            or key.startswith("one_frame_")
        }

    def __contains__(self, key: str) -> bool:
        return key in self._batch

    def take(self, key: str) -> Any:
        self.unread.discard(key)
        return self._batch[key]

    def take_index_tensor(self, key: str, batch_size: int, *, ndim: int, hint: str) -> torch.Tensor:
        value = self._batch.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim != ndim or value.shape[0] != batch_size:
            shape = "[B]" if ndim == 1 else "[B,K]"
            raise ValueError(
                f"MiniMax-H3 one-frame batch requires a {shape} {key} tensor; re-run minimax_h3_cache_latents.py --one_frame{hint}"
            )
        return self.take(key)


def _teacher_text_keys(teacher_conditions: str) -> tuple[str, str]:
    """The batch keys of one teacher kind's text rows: the cache stems without the collator's
    varlen_ marker."""
    stem = TEACHER_TEXT_CACHE_PREFIXES[teacher_conditions].removeprefix("varlen_")
    return f"{stem}_hidden_states", f"{stem}_token_tags"


def _stack_single_text_rows(value, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 {label} must keep a leading batch axis of size 1")
        return value
    if not isinstance(value, Sequence) or len(value) != 1 or not isinstance(value[0], torch.Tensor):
        raise ValueError(f"MiniMax-H3 {label} must contain exactly one tensor")
    return value[0].unsqueeze(0)


def _condition_roles_in(batch: Mapping[str, Any]) -> dict[str, H3ConditionRole]:
    """The condition latents the batch carries, keyed by role: every `latents_<role>` entry other
    than the targets (`latents`, `latents_audio`), parsed with the packing vocabulary."""
    roles = {}
    for key in batch:
        if key.startswith("latents_") and key != "latents_audio":
            role = key.removeprefix("latents_")
            try:
                roles[role] = parse_condition_role(role)
            except ValueError as error:
                raise ValueError(f"MiniMax-H3 batch entry {key} is neither a target nor a known condition latent") from error
    return roles


def _one_frame_condition_roles_in(condition_roles: Mapping[str, H3ConditionRole]) -> tuple[str, ...]:
    """The ordered cond_{i} roles among the batch's conditions (empty for video FL2VA caches)."""
    indices = sorted(role.index for role in condition_roles.values() if role.family == "one_frame")
    if indices and indices != list(range(len(indices))):
        raise ValueError(f"MiniMax-H3 one-frame FL2VA condition latents must be the contiguous cond_000..., got {indices}")
    return one_frame_condition_roles(len(indices))


def _collect_fl_conditions(
    entries: _BatchEntries,
    condition_roles: Mapping[str, H3ConditionRole],
    visual_conditions: list[torch.Tensor],
    condition_geometries: list[H3VideoGeometry],
    *,
    one_frame: bool = False,
) -> tuple[str, ...]:
    """Reads the FL2VA condition latents in layout order and returns their roles: first/last for
    video targets, the ordered cond_{i} slots (timed by one_frame_control_indices) for one-frame
    targets. The layout builder and the transformer validate the tensors themselves."""
    if one_frame:
        roles = _one_frame_condition_roles_in(condition_roles)
        if not roles:
            raise ValueError(
                "MiniMax-H3 one-frame FL2VA batch requires latents_cond_000... condition latents (first/last keys are the"
                " video layout); re-run minimax_h3_cache_latents.py --one_frame --task fl2va"
            )
    else:
        roles = tuple(role for role in FL_CONDITION_ROLES if role in condition_roles)
        if not roles:
            raise ValueError(
                "MiniMax-H3 FL2VA batch requires latents_first/latents_last condition latents;"
                " re-run minimax_h3_cache_latents.py --task fl2va"
            )
    for role in roles:
        tensor = entries.take(f"latents_{role}")
        visual_conditions.append(tensor)
        condition_geometries.append(H3VideoGeometry(*tensor.shape[2:]))
    return roles


def _collect_reference_conditions(
    entries: _BatchEntries, condition_roles: Mapping[str, H3ConditionRole]
) -> tuple[list[H3ReferenceGeometry], list[torch.Tensor], list[torch.Tensor]]:
    """Turns the numbered ``latents_ref_{i}_{image|video|audio}`` batch entries into ordered
    reference geometries plus the visual/audio condition tensors in layout order."""
    by_index: dict[int, dict[str, torch.Tensor]] = {}
    for name, role in condition_roles.items():
        if role.family == "reference":
            by_index.setdefault(role.index, {})[role.reference_kind] = entries.take(f"latents_{name}")
    if set(by_index) != set(range(len(by_index))):
        raise ValueError("MiniMax-H3 reference indices must be contiguous from 000")
    references: list[H3ReferenceGeometry] = []
    visual_conditions: list[torch.Tensor] = []
    audio_conditions: list[torch.Tensor] = []
    for index in range(len(by_index)):
        roles = by_index[index]
        image = roles.get("image")
        video = roles.get("video")
        audio = roles.get("audio")
        if image is not None:
            if video is not None or audio is not None:
                raise ValueError(f"MiniMax-H3 reference {index:03d} image cannot share video/audio roles")
            references.append(H3ReferenceGeometry("image", video=H3VideoGeometry(*image.shape[2:])))
            visual_conditions.append(image)
        elif video is not None:
            audio_frames = 0
            if audio is not None:
                audio_frames = audio.shape[-1]
                audio_conditions.append(audio)
            references.append(H3ReferenceGeometry("video", video=H3VideoGeometry(*video.shape[2:]), audio_frames=audio_frames))
            visual_conditions.append(video)
        else:
            references.append(H3ReferenceGeometry("audio", audio_frames=audio.shape[-1]))
            audio_conditions.append(audio)
    return references, visual_conditions, audio_conditions


def _runtime_batch_plan(
    batch: Mapping[str, Any],
    video_latents: torch.Tensor,
    *,
    task: str,
    teacher_conditions: str | None = None,
    one_frame: bool = False,
) -> _H3RuntimeBatch:
    missing = [key for key in _REQUIRED_BATCH_KEYS if key not in batch]
    if missing:
        raise ValueError(f"MiniMax-H3 batch is missing {', '.join(missing)}; re-run latent caching")
    batch_size = video_latents.shape[0]
    if batch_size != 1:
        raise ValueError(f"MiniMax-H3 R1 requires batch_size=1, got {batch_size}; use gradient accumulation")
    audio_latents = batch["latents_audio"]
    audio_present = batch["audio_present"]
    if not isinstance(audio_present, torch.Tensor) or audio_present.shape != (batch_size,):
        raise ValueError("MiniMax-H3 batch requires an audio_present tensor with shape [B]; re-run latent caching")
    hidden_states = _stack_single_text_rows(batch["mmh3_hidden_states"], "text hidden states")
    token_tags = _stack_single_text_rows(batch["mmh3_token_tags"], "text token tags")
    entries = _BatchEntries(batch)
    condition_roles = _condition_roles_in(batch)
    notices: list[str] = []

    is_one_frame = video_latents.shape[2] == ONE_FRAME_VIDEO_LATENT_FRAMES
    target_index = None
    if is_one_frame:
        if not one_frame:
            raise ValueError("MiniMax-H3 batch carries a one-frame latent cache; pass --one_frame to train on image targets")
        target_index = int(entries.take_index_tensor("one_frame_target_index", batch_size, ndim=1, hint="")[0].item())

    # the student's conditions, by the authoritative --task
    visual_conditions: list[torch.Tensor] = []
    audio_conditions: list[torch.Tensor] = []
    condition_geometries: list[H3VideoGeometry] = []
    references: list[H3ReferenceGeometry] = []
    fl_condition_roles: tuple[str, ...] | None = None
    condition_times: tuple[float, ...] = ()
    if task == "fl2va":
        fl_condition_roles = _collect_fl_conditions(
            entries, condition_roles, visual_conditions, condition_geometries, one_frame=is_one_frame
        )
        if is_one_frame:
            # one-frame conditions are timed by their control indices, not by role names
            control_value = entries.take_index_tensor("one_frame_control_indices", batch_size, ndim=2, hint=" --task fl2va")
            control_indices = [int(index) for index in control_value[0].tolist()]
            if target_index in control_indices:
                notices.append(
                    f"MiniMax-H3 one-frame FL2VA data places a control at the target index ({target_index}): the base"
                    " model's prior at coinciding timestamps is verbatim anchor copying, so make sure that is the"
                    " intended training signal (see docs/minimax_h3_1f.md)"
                )
            condition_times = tuple(FRAME_RESCALE * index for index in control_indices)
    elif task == "ref2va":
        references, visual_conditions, audio_conditions = _collect_reference_conditions(entries, condition_roles)
        if not references:
            raise ValueError(
                "MiniMax-H3 Ref2VA batch requires latents_ref_000_* reference latents; re-run minimax_h3_cache_latents.py --task ref2va"
            )
    # references are untimed: only FL2VA controls carry condition times
    time_overrides = H3TimeOverrides(condition_times, FRAME_RESCALE * target_index) if is_one_frame else None

    target_geometry = H3VideoGeometry(*video_latents.shape[2:])
    teacher_layout = None
    teacher_hidden_states = None
    teacher_token_tags = None
    teacher_visual_conditions: list[torch.Tensor] = []
    teacher_audio_conditions: list[torch.Tensor] = []
    if teacher_conditions is not None:
        hidden_key, tags_key = _teacher_text_keys(teacher_conditions)
        if hidden_key not in entries or tags_key not in entries:
            raise ValueError(
                f"MiniMax-H3 {teacher_conditions} teacher matching requires {teacher_conditions} teacher text rows;"
                f" re-run minimax_h3_cache_text_encoder_outputs.py --task t2va --teacher_conditions {teacher_conditions}"
            )
        teacher_hidden_states = _stack_single_text_rows(entries.take(hidden_key), "teacher text hidden states")
        teacher_token_tags = _stack_single_text_rows(entries.take(tags_key), "teacher text token tags")
        if teacher_conditions == TEACHER_CONDITIONS_REF:
            # the teacher runs on the Ref2VA layout with the cached target latents themselves
            # (video + audio) as the reference condition, so it sees complete information at every
            # sigma; FL2VA first/last latents, if present in the caches, are simply unused
            teacher_visual_conditions.append(video_latents)
            teacher_audio_conditions.append(audio_latents)
            teacher_layout = build_h3_layout(
                task="ref2va",
                text_length=teacher_hidden_states.shape[1],
                target_video=target_geometry,
                target_audio_frames=audio_latents.shape[-1],
                references=(H3ReferenceGeometry("video", video=target_geometry, audio_frames=audio_latents.shape[-1]),),
            )
        elif teacher_conditions == TEACHER_CONDITIONS_SUBJECT_REF:
            # the item's own reference latents (Ref2VA cache) and the subject-declaration text
            # rows feed only the no-grad Ref2VA teacher forward: other pictures of the subject
            # supply the concept without the complete-information degeneration of the
            # self-reference teacher
            teacher_references, teacher_visual_conditions, teacher_audio_conditions = _collect_reference_conditions(
                entries, condition_roles
            )
            if not teacher_references:
                raise ValueError(
                    "MiniMax-H3 subject-reference teacher matching requires the item's reference latents;"
                    " re-run minimax_h3_cache_latents.py --task ref2va (with --one_frame for image datasets)"
                )
            if any(reference.kind != "image" for reference in teacher_references):
                raise ValueError("MiniMax-H3 subject-reference teacher matching supports image references only (v1)")
            # a one-frame teacher layout must carry the one-frame flag and the target-time override
            # (the same rebuild trap as the guidance-loss uncond layout); references are untimed
            teacher_layout = build_h3_layout(
                task="ref2va",
                text_length=teacher_hidden_states.shape[1],
                target_video=target_geometry,
                target_audio_frames=audio_latents.shape[-1],
                references=tuple(teacher_references),
                one_frame=is_one_frame,
                time_overrides=time_overrides,
            )
        else:
            # the first/last latents and the Picture-prefixed text rows feed only the no-grad
            # FL2VA teacher forward
            teacher_geometries: list[H3VideoGeometry] = []
            if not any(role.family == "fl" for role in condition_roles.values()):
                raise ValueError(
                    "MiniMax-H3 teacher matching requires FL2VA-style latent caches with first/last conditions;"
                    " re-run minimax_h3_cache_latents.py --task fl2va"
                )
            _collect_fl_conditions(entries, condition_roles, teacher_visual_conditions, teacher_geometries)
            teacher_layout = build_h3_layout(
                task="fl2va",
                text_length=teacher_hidden_states.shape[1],
                target_video=target_geometry,
                target_audio_frames=audio_latents.shape[-1],
                visual_conditions=tuple(teacher_geometries),
            )

    if entries.unread:
        consumer = f"--task {task}" if teacher_conditions is None else f"--task {task} with the {teacher_conditions} teacher"
        notices.append(
            f"MiniMax-H3 batch entries {', '.join(sorted(entries.unread))} are not used by {consumer} and are ignored;"
            " check that the caches were written for this task"
        )

    layout = build_h3_layout(
        task=task,
        text_length=hidden_states.shape[1],
        target_video=target_geometry,
        target_audio_frames=audio_latents.shape[-1],
        visual_conditions=tuple(condition_geometries),
        references=tuple(references),
        one_frame=is_one_frame,
        condition_roles=fl_condition_roles,
        time_overrides=time_overrides,
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
        teacher_audio_conditions=tuple(teacher_audio_conditions),
        notices=tuple(notices),
    )


def _base_sigma_of(timesteps: torch.Tensor) -> float:
    """The drawn pre-shift base sigma behind the trainer's 1..1000 timestep convention."""
    return float((timesteps.reshape(-1)[0].item() - 1.0) / 1000.0)


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


def _preservation_density_compensation(
    sigma_max: float, focus_min: float, focus_max: float, focus_prob: float, sigma_min: float = 0.0
) -> float:
    """Loss-weight correction that keeps the preservation anchor's expected gradient share
    invariant under timestep focus.

    Focus concentrates the base-sigma draw on the teaching band and thins the anchor bands
    (base sigma > sigma_max, and < sigma_min when a lower gate is set) from their uniform
    share ``(1 - sigma_max) + sigma_min`` to ``(1-p)*uniform + p*overlap/(max-min)``;
    multiplying each anchor step's loss by uniform/focused restores the anchor's per-unit-time
    pull, so raising the focus does not silently weaken the drift protection.
    """
    anchor_width = (1.0 - sigma_max) + max(0.0, sigma_min)
    if anchor_width <= 0.0 or focus_prob <= 0.0:
        return 1.0
    overlap = max(0.0, focus_max - max(focus_min, sigma_max)) + max(0.0, min(focus_max, sigma_min) - focus_min)
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


def _scalar_logs(logs: Mapping[str, torch.Tensor | float]) -> dict[str, float]:
    """The per-step metrics as plain floats (the ``process_batch`` contract), fetched from the
    device in one transfer instead of one sync per entry."""
    if not logs:
        return {}
    scalars = [torch.as_tensor(value).detach().float().reshape(()) for value in logs.values()]
    device = next((scalar.device for scalar in scalars if scalar.device.type != "cpu"), torch.device("cpu"))
    values = torch.stack([scalar.to(device) for scalar in scalars]).cpu().tolist()
    return dict(zip(logs.keys(), values))


class H3SamplingResources(torch.nn.Module):
    """Training-time sampling payload: H3 decodes samples with two separate VAEs. A Module, so the
    base trainer's resource handling (device moves after each sample) covers both."""

    def __init__(self, video_vae: torch.nn.Module, audio_vae: torch.nn.Module):
        super().__init__()
        self.video_vae = video_vae
        self.audio_vae = audio_vae


@dataclass
class _SamplePreparation:
    """Per-prompt state of prepare_sampling between its model phases (text encoder, video VAE,
    audio VAE); the tensors that survive into the sample dict are copied out at the end."""

    request: H3GenerationRequest
    record: H3Record
    visual_conditions: tuple[torch.Tensor, ...] = ()
    visual_geometries: tuple[H3VideoGeometry, ...] = ()
    reference_visual_geometries: dict[int, H3VideoGeometry] = field(default_factory=dict)
    reference_video_frame_counts: dict[int, int] = field(default_factory=dict)
    audio_conditions: tuple[torch.Tensor, ...] = ()
    reference_audio_frames: dict[int, int] = field(default_factory=dict)

    @property
    def has_audio_conditions(self) -> bool:
        return any(reference.audio is not None for reference in self.record.references)


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    audio_spec = H3_AUDIO_SPEC

    def __init__(self):
        super().__init__()
        # per-rank audio-supervision accounting, fed by process_batch; drives the
        # first-epoch warning and the observed fraction saved in metadata
        self._audio_items_seen = 0
        self._audio_supervised_seen = 0
        # guidance-loss uncond probe (CPU hidden rows + tags), loaded by on_train_start when
        # --h3_guidance_loss_scale is active
        self._guidance_uncond: tuple[torch.Tensor, torch.Tensor] | None = None
        # effective base quantization, known once load_transformer has seen the checkpoint
        # (pre-quantized ConvRot INT8 files are detected there, independent of --convrot_int8)
        self._convrot_int8_active: bool | None = None
        # batch observations already warned about (each one is logged once per run)
        self._warned_notices: set[str] = set()

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
        if args.task not in H3_TASKS:
            raise ValueError(f"MiniMax-H3 requires --task {', '.join(H3_TASKS)}")
        if args.one_frame:
            if (
                args.h3_teacher_matching
                and normalize_teacher_conditions(args.h3_teacher_conditions) != TEACHER_CONDITIONS_SUBJECT_REF
            ):
                raise ValueError("--h3_teacher_matching supports --one_frame with --h3_teacher_conditions subject_ref only")
            logger.info(
                "MiniMax-H3 one-frame training: image batches carry a silence audio placeholder that presence"
                " gating excludes from the audio loss; pass --video_only for image-only runs to skip the"
                " audio-loss bookkeeping entirely"
            )
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
        validate_shift(args.h3_shift_video, "--h3_shift_video")
        validate_shift(args.h3_shift_audio, "--h3_shift_audio")
        validate_clean_coefficient(args.h3_visual_cond_clean, "--h3_visual_cond_clean")
        validate_clean_coefficient(args.h3_audio_cond_clean, "--h3_audio_cond_clean")
        if args.blocks_to_swap is not None and args.blocks_to_swap > 48:
            raise ValueError("--blocks_to_swap for MiniMax-H3 must be <= 48")
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("MiniMax-H3 does not support fp8 transformer bases; use --convrot_int8 for a quantized base")
        if args.dit_dtype not in {None, "bfloat16", "bf16"}:
            raise ValueError("MiniMax-H3 R1 requires --dit_dtype bfloat16")
        if args.block_swap_h2d_only and bool(args.blocks_to_swap) and not args.gradient_checkpointing:
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
        for name, value in (
            ("h3_teacher_loss_dc_weight", args.h3_teacher_loss_dc_weight),
            ("h3_teacher_loss_mag_weight", args.h3_teacher_loss_mag_weight),
            ("h3_teacher_preservation_weight", args.h3_teacher_preservation_weight),
        ):
            value = float(value)
            if value < 0.0:
                raise ValueError(f"--{name} must be nonnegative, got {value}")
            if value != 1.0 and not args.h3_teacher_matching:
                raise ValueError(f"--{name} shapes the teacher-matching loss and requires --h3_teacher_matching")
        if args.h3_teacher_matching:
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
            sigma_min = float(args.h3_teacher_condition_sigma_min)
            if not 0.0 <= sigma_min <= sigma_max:
                raise ValueError(
                    f"--h3_teacher_condition_sigma_min must be in [0.0, --h3_teacher_condition_sigma_max], got {sigma_min}"
                )
            if conditions == TEACHER_CONDITIONS_SUBJECT_REF:
                logger.info(
                    "MiniMax-H3 teacher matching: Ref2VA teacher conditioned on the item's own subject-reference"
                    " pictures for base sigma in [%s, %s] (base-preservation anchor outside)",
                    sigma_min,
                    sigma_max,
                )
                if sigma_max < 1.0:
                    logger.warning(
                        "MiniMax-H3 subject_ref teacher: --h3_teacher_condition_sigma_max %s anchors base sigma > %s to"
                        " the text-only base, but the identity decisions (hair shape, eye color) are made at base sigma"
                        " 0.92-1.0 and this teacher keeps its amplification there; the validated recipe is 1.0 (the"
                        " default). Expect the student to learn composition but not identity below ~0.95",
                        sigma_max,
                        sigma_max,
                    )
            elif sigma_max > SIGMA_MAX_RECOMMENDED_COMPLETE_INFORMATION:
                logger.warning(
                    "MiniMax-H3 %s teacher: --h3_teacher_condition_sigma_max %s teaches above base sigma %s, where the"
                    " conditioned content is unpredictable from the text and the teaching overwrites the base"
                    " composition prior (for ref the FL2VA weights also fail to align the reference above ~0.85);"
                    " the validated recipe for this teacher is %s",
                    conditions,
                    sigma_max,
                    SIGMA_MAX_RECOMMENDED_COMPLETE_INFORMATION,
                    SIGMA_MAX_RECOMMENDED_COMPLETE_INFORMATION,
                )
            elif conditions == TEACHER_CONDITIONS_REF:
                logger.info(
                    "MiniMax-H3 teacher matching: Ref2VA teacher conditioned on the training clip itself (video+audio)"
                    " up to base sigma %s (base-preservation anchor above)",
                    sigma_max,
                )
            else:
                logger.info(
                    "MiniMax-H3 teacher matching: FL2VA teacher conditioned on %s up to base sigma %s"
                    " (base-preservation anchor above)",
                    conditions,
                    sigma_max,
                )
        if args.h3_guidance_loss_scale_audio is not None and float(args.h3_guidance_loss_scale_audio) < 0.0:
            raise ValueError(f"--h3_guidance_loss_scale_audio must be nonnegative, got {args.h3_guidance_loss_scale_audio}")
        if not 0.0 <= float(args.h3_guidance_loss_sigma_min) <= 1.0:
            raise ValueError(f"--h3_guidance_loss_sigma_min must be in [0.0,1.0], got {args.h3_guidance_loss_sigma_min}")
        if guidance_scale > 0.0:
            if not args.h3_guidance_loss_uncond_cache:
                raise ValueError(
                    "--h3_guidance_loss_scale requires --h3_guidance_loss_uncond_cache"
                    " (write one with minimax_h3_cache_text_encoder_outputs.py --uncond_output)"
                )
            require_path(args.h3_guidance_loss_uncond_cache, "h3_guidance_loss_uncond_cache")
        elif args.h3_guidance_loss_uncond_cache:
            logger.warning("--h3_guidance_loss_uncond_cache is ignored because --h3_guidance_loss_scale is 0")

    def on_train_start(self, args: argparse.Namespace, accelerator: Accelerator, network, transformer, optimizer) -> None:
        del accelerator, network, transformer, optimizer
        self._guidance_uncond = None
        if float(args.h3_guidance_loss_scale) > 0.0:
            hidden_states, token_tags, metadata = load_h3_uncond_cache(args.h3_guidance_loss_uncond_cache)
            self._guidance_uncond = (hidden_states, token_tags)
            logger.info(
                "MiniMax-H3 guidance loss: scale=%s scale_audio=%s sigma_min=%s uncond=%r (%d rows)",
                args.h3_guidance_loss_scale,
                self._guidance_audio_scale(args),
                args.h3_guidance_loss_sigma_min,
                metadata.get("text", "?"),
                hidden_states.shape[0],
            )

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
        if is_convrot_int8 and args.base_weights:
            raise ValueError("MiniMax-H3 --base_weights cannot be merged into a ConvRot INT8 transformer base")

    def prepare_sampling(self, args, accelerator, vae_dtype):
        del vae_dtype  # the H3 video/audio VAE dtypes are fixed per stage
        if not args.sample_prompts:
            return None, None
        for label, value in (("video_vae", args.video_vae), ("audio_vae", args.audio_vae), ("text_encoder", args.text_encoder)):
            require_path(value, label)
        sample_prompts = args.sample_prompts
        logger.info("Preparing MiniMax-H3 joint AV training samples from %s", sample_prompts)
        parameters = load_prompts(sample_prompts)
        if not parameters:
            raise ValueError("MiniMax-H3 sample prompt file is empty")
        # every request and record is resolved before the first model loads, so a bad prompt
        # line fails fast instead of after the text encoder
        preparations = []
        for parameter in parameters:
            request = _sample_request(args, parameter)
            record = load_generation_record(request)
            if request.one_frame:
                reject_one_frame_audio_references(record)
            preparations.append(_SamplePreparation(request=request, record=record))
        device = accelerator.device
        decoder = PyAVH3MediaDecoder()

        logger.info("Loading MiniMax-H3 Qwen3-VL text encoder for training samples")
        processor = load_h3_processor()
        text_encoder = load_h3_text_encoder(
            args.text_encoder,
            device=device,
            dtype=torch.bfloat16,
            disable_numpy_memmap=args.disable_numpy_memmap,
            nvfp4_scaled_mm=args.nvfp4_scaled_mm,
            blocks_to_swap=args.text_encoder_blocks_to_swap,
            attn_mode=args.text_encoder_attn_mode,
        )
        text_encoder.eval().requires_grad_(False)
        try:
            for parameter, preparation in zip(parameters, preparations):
                raw_visuals, text_visuals = decode_generation_visuals(preparation.request, preparation.record, decoder)
                presentation = build_presentation(preparation.record, args.task, text_visuals)
                hidden_states, token_tags = encode_h3_presentation(processor, text_encoder, presentation)
                parameter["h3_text_hidden_states"] = hidden_states.to(torch.bfloat16).unsqueeze(0).cpu()
                parameter["h3_text_token_tags"] = token_tags.unsqueeze(0).cpu()
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
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        video_vae.eval().requires_grad_(False)
        try:
            if video_vae.vae_ratio != VIDEO_VAE_SPATIAL_RATIO:
                raise ValueError(f"MiniMax-H3 video VAE spatial ratio must be {VIDEO_VAE_SPATIAL_RATIO}, got {video_vae.vae_ratio}")
            if has_visual_conditions:
                for preparation in preparations:
                    # Re-decode here instead of retaining hundreds of MB of pixels across model teardown.
                    raw_visuals, text_visuals = decode_generation_visuals(preparation.request, preparation.record, decoder)
                    (
                        preparation.visual_conditions,
                        preparation.visual_geometries,
                        preparation.reference_visual_geometries,
                    ) = encode_visual_conditions(preparation.request, preparation.record, raw_visuals, video_vae)
                    preparation.reference_video_frame_counts = reference_video_frame_counts(preparation.record, raw_visuals)
                    del raw_visuals, text_visuals
        finally:
            video_vae.to(device="cpu", dtype=VIDEO_VAE_DECODE_DTYPE)
            gc.collect()
            clean_memory_on_device(device)

        logger.info("Loading MiniMax-H3 audio VAE for training samples")
        has_audio_conditions = any(preparation.has_audio_conditions for preparation in preparations)
        audio_vae = load_audio_vae(
            args.audio_vae,
            device=device if has_audio_conditions else torch.device("cpu"),
            dtype=torch.float32,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        audio_vae.eval().requires_grad_(False)
        try:
            for preparation in preparations:
                if not preparation.has_audio_conditions:
                    continue
                preparation.audio_conditions, preparation.reference_audio_frames = encode_audio_conditions(
                    preparation.request,
                    preparation.record,
                    decoder,
                    audio_vae,
                    reference_video_frame_counts=preparation.reference_video_frame_counts,
                )
        finally:
            audio_vae.to("cpu")
            gc.collect()
            clean_memory_on_device(device)

        for parameter, preparation in zip(parameters, preparations):
            references = (
                build_reference_geometries(
                    preparation.record, preparation.reference_visual_geometries, preparation.reference_audio_frames
                )
                if args.task == "ref2va"
                else ()
            )
            logger.info("MiniMax-H3 training sample %d: %r", parameter["enum"], preparation.request.prompt)
            # the resolved coordinates, so the base sampler's log lines describe the actual sample
            parameter["width"] = preparation.request.width
            parameter["height"] = preparation.request.height
            parameter["frame_count"] = preparation.request.frame_count
            parameter["sample_steps"] = preparation.request.steps
            parameter["seed"] = preparation.request.seed
            parameter["h3_request"] = preparation.request
            parameter["h3_layout"] = build_generation_layout(
                preparation.request,
                text_length=parameter["h3_text_hidden_states"].shape[1],
                visual_geometries=preparation.visual_geometries,
                reference_geometries=references,
            )
            parameter["h3_visual_conditions"] = preparation.visual_conditions
            parameter["h3_audio_conditions"] = preparation.audio_conditions
        return parameters, H3SamplingResources(video_vae=video_vae, audio_vae=audio_vae)

    def round_sample_frame_count(self, frame_count: int) -> int:
        return _sample_frame_count(frame_count)

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        # the validated request prepared by prepare_sampling carries every sample coordinate
        del args, dit_dtype, discrete_flow_shift, sample_steps, width, height, frame_count
        del do_classifier_free_guidance, guidance_scale, cfg_scale, image_path, control_video_path
        request: H3GenerationRequest = sample_parameter["h3_request"]
        layout = sample_parameter["h3_layout"]
        device = accelerator.device
        logger.info(
            "MiniMax-H3 joint sample: shift video=%s audio=%s, condition clean visual=%s audio=%s, output fps=%d",
            request.h3_shift_video,
            request.h3_shift_audio,
            request.h3_visual_cond_clean,
            request.h3_audio_cond_clean,
            request.output_fps,
        )
        with tqdm(
            total=request.steps,
            desc=f"MiniMax-H3 sample {sample_parameter.get('enum', 0)}",
            unit="step",
            disable=not accelerator.is_local_main_process,
        ) as progress:
            sample = sample_joint_av_latents(
                transformer,
                layout=layout,
                # the base seeds the generator from the prompt's seed, or from a fresh random one
                seed=generator.initial_seed(),
                text_hidden_states=sample_parameter["h3_text_hidden_states"],
                text_token_tags=sample_parameter["h3_text_token_tags"],
                visual_conditions=sample_parameter["h3_visual_conditions"],
                audio_conditions=sample_parameter["h3_audio_conditions"],
                steps=request.steps,
                video_shift=request.h3_shift_video,
                audio_shift=request.h3_shift_audio,
                visual_condition_clean=request.h3_visual_cond_clean,
                audio_condition_clean=request.h3_audio_cond_clean,
                device=device,
                step_callback=lambda completed, total: progress.update(1),
            )
        synchronize_device(device)
        clean_memory_on_device(device)

        logger.info("Decoding MiniMax-H3 training sample video")
        video_vae = vae.video_vae
        video_vae.to(device)
        _, video_dtype = module_device_dtype(video_vae, VIDEO_VAE_DECODE_DTYPE)
        decoded_video = video_vae.decode(sample.video.to(device=device, dtype=video_dtype)).cpu()
        video_vae.to("cpu")
        clean_memory_on_device(device)
        if request.one_frame:
            # one-frame sample: the audio rows are a byproduct and are never decoded; the single
            # frame is saved by the base like any image sample ([1,3,1,H,W] in [0,1])
            return (decoded_video[:, :, :1].float().clamp(-1.0, 1.0) + 1.0) / 2.0

        logger.info("Decoding MiniMax-H3 training sample audio")
        audio_vae = vae.audio_vae
        audio_vae.to(device)
        _, audio_dtype = module_device_dtype(audio_vae, torch.float32)
        decoded_audio = audio_vae.decode(sample.audio.to(device=device, dtype=audio_dtype)).cpu()
        audio_vae.to("cpu")
        clean_memory_on_device(device)
        # a stretched sample (--ofps) plays its frame_count frames over the stretched real
        # duration, like the generation CLI: the container rate and the audio trim follow it
        return synchronize_decoded_av(decoded_video, decoded_audio, frame_count=request.frame_count, fps=request.output_fps)

    def save_sample(self, accelerator, args, sample_parameter, sample, save_dir: str, save_path: str, steps: int) -> None:
        if not isinstance(sample, H3DecodedAV):
            # a one-frame sample is a plain image tensor
            return super().save_sample(accelerator, args, sample_parameter, sample, save_dir, save_path, steps)
        output_path = Path(save_dir) / f"{save_path}.mp4"
        write_joint_av(sample, output_path)
        logger.info("Saved MiniMax-H3 joint training sample: %s", output_path)
        wandb_tracker, wandb = wandb_tracker_and_module(accelerator)
        if wandb_tracker is not None:
            wandb_tracker.log(
                {f"sample_{sample_parameter.get('enum', 0)}": wandb.Video(str(output_path), fps=sample.fps)}, step=steps
            )

    def _notice(self, message: str) -> None:
        """Logs a batch observation once per run (the same cache condition recurs every step)."""
        if message not in self._warned_notices:
            self._warned_notices.add(message)
            logger.warning(message)

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
            "ss_minimax_h3_convrot_int8": args.convrot_int8 if self._convrot_int8_active is None else self._convrot_int8_active,
            "ss_minimax_h3_latent_cache_version": "2",
            "ss_minimax_h3_text_cache_version": "1",
        }
        if args.one_frame:
            metadata["ss_minimax_h3_one_frame"] = True
        if float(args.h3_guidance_loss_scale) > 0.0:
            metadata["ss_minimax_h3_guidance_loss_scale"] = args.h3_guidance_loss_scale
            metadata["ss_minimax_h3_guidance_loss_scale_audio"] = self._guidance_audio_scale(args)
            metadata["ss_minimax_h3_guidance_loss_sigma_min"] = args.h3_guidance_loss_sigma_min
        if args.h3_teacher_matching:
            metadata["ss_minimax_h3_teacher_matching"] = True
            metadata["ss_minimax_h3_teacher_conditions"] = normalize_teacher_conditions(args.h3_teacher_conditions)
            metadata["ss_minimax_h3_teacher_condition_sigma_max"] = args.h3_teacher_condition_sigma_max
            metadata["ss_minimax_h3_teacher_condition_sigma_min"] = args.h3_teacher_condition_sigma_min
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
            disable_numpy_memmap=args.disable_numpy_memmap,
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
        del batch
        base_sigma = _base_sigma_of(timesteps)  # the pre-shift draw gates the guidance/teacher forwards
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
        teacher_audio_conditions = kwargs.pop("teacher_audio_conditions", ())
        if kwargs:
            raise TypeError(f"Unexpected MiniMax-H3 call_dit arguments: {sorted(kwargs)}")

        text_hidden_states = runtime.text_hidden_states.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(accelerator.device)
        noisy_audio_input = noisy_audio_input.to(accelerator.device)

        video_target = latents - noise
        audio_target = audio_latents - audio_noise
        guidance_log: dict[str, torch.Tensor | float] = {}
        teacher_conditioned = True
        if self._guidance_uncond is not None:
            # the uncond forward runs before the grad forward so the block-swap offloader
            # keeps its forward->backward alternation and no autograd graph is live yet
            applied = base_sigma >= float(args.h3_guidance_loss_sigma_min)
            # the drawn pre-shift sigma, so the logged gap magnitudes can be binned by noise level
            guidance_log["guidance/base_sigma"] = base_sigma
            guidance_log["guidance/applied"] = 1.0 if applied else 0.0
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
        if args.h3_teacher_matching:
            video_target, audio_target, teacher_conditioned, teacher_log = self._apply_teacher_matching_targets(
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
                teacher_audio_conditions,
                video_target,
                audio_target,
                network_dtype,
                base_sigma,
            )
            guidance_log.update(teacher_log)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            noisy_audio_input.requires_grad_(True)
            text_hidden_states.requires_grad_(True)
        with accelerator.autocast():
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
        if args.h3_teacher_matching:
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
                "teacher_conditioned": teacher_conditioned,
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
        # the probe swaps only the text rows; the one-frame times stay valid because they
        # are relative to the target-block cursor, which moves with the text length. The
        # FL2VA condition roles are recovered from the segments so a one-frame FL2VA layout
        # with a single condition rebuilds identically (roles are required for K=1); Ref2VA
        # reference blocks share the segment kind but are addressed by the references tuple
        uncond_condition_roles = ()
        if runtime.layout.task == "fl2va":
            uncond_condition_roles = tuple(
                segment.role for segment in runtime.layout.segments if segment.kind == "visual_condition"
            )
        uncond_layout = build_h3_layout(
            task=runtime.layout.task,
            text_length=uncond_hidden.shape[0],
            target_video=runtime.layout.target_video,
            target_audio_frames=runtime.layout.target_audio_frames,
            visual_conditions=runtime.layout.visual_conditions,
            references=runtime.layout.references,
            one_frame=runtime.layout.target_video.frames == ONE_FRAME_VIDEO_LATENT_FRAMES,
            condition_roles=uncond_condition_roles or None,
            time_overrides=runtime.layout.time_overrides,
        )
        with torch.no_grad(), accelerator.autocast():
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
        teacher_audio_conditions: tuple[torch.Tensor, ...],
        video_target: torch.Tensor,
        audio_target: torch.Tensor,
        network_dtype: torch.dtype,
        base_sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor, bool, dict[str, torch.Tensor | float]]:
        """Replace both flow targets with the frozen base model's predictions.

        The teacher shares weights with the student: the same transformer runs once with the
        LoRA disabled, conditioned on privileged information the T2VA student never sees.
        With the default first,last conditions that is the real first/last frames and the
        Picture-prefixed FL2VA text rows; with --h3_teacher_conditions ref it is the Ref2VA
        layout carrying the training clip itself (target video and audio latents) as the
        copy-source reference. The teacher prediction lives in the distilled guided space, so
        no guidance scale or uncond probe is needed and the de-distillation drift of plain
        flow targets is structurally avoided; this is why the loss is mutually exclusive with
        the contrastive guidance loss.

        With endpoint conditions the loss keeps an irreducible floor (endpoint content the
        text alone cannot determine), so read the sigma-binned teacher/*_flow_gap_rms logs
        rather than expecting it to reach zero, and the audio target degenerates to a
        base-preservation anchor (the visual endpoints carry almost no audio information).
        The ref teacher collapses that floor to the model's copy error and turns the audio
        target into a real teaching signal (the reference audio is declared fully_copy) --
        but with a complete-information teacher the guided-space safety margin inside the
        teaching band shrinks, so the anchor band, the decomposed loss, and the norm-ratio
        logs carry the de-distillation protection there.

        Above --h3_teacher_condition_sigma_max the teacher instead runs on the student's own
        text and layout with no conditions, turning the target into a pure base-preservation
        anchor. The teacher target is a noiseless regression label (deterministic per x_t),
        and near pure noise the content is unpredictable from the text, so unrestricted
        teaching there rapidly overwrites the base composition prior with the dataset mean
        (for the ref teacher the same band is also where the FL2VA weights fail to align the
        reference against a footing-less x_t); anchoring that band to the base also counters
        the collateral drift of the LoRA's shared weights.
        """
        if network is None:
            raise RuntimeError("MiniMax-H3 teacher matching requires the LoRA network to disable it for the teacher forward")
        base_network = accelerator.unwrap_model(network)
        # the lower gate is the mirror of the upper one: near sigma 0 the noised target itself
        # reveals x_0, so every teacher's prediction collapses toward the raw velocity (the
        # complete-information de-amplification channel entering from below)
        sigma_min = float(args.h3_teacher_condition_sigma_min)
        conditioned = sigma_min <= base_sigma <= float(args.h3_teacher_condition_sigma_max)
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
            teacher_audio_conditions = ()
        # the teacher forward runs before the grad forward so the block-swap offloader keeps
        # its forward->backward alternation and no autograd graph is live yet
        base_network.set_enabled(False)
        try:
            with torch.no_grad(), accelerator.autocast():
                teacher = transformer(
                    video_latents=noisy_model_input,
                    audio_latents=noisy_audio_input,
                    text_hidden_states=teacher_text.to(device=accelerator.device, dtype=network_dtype),
                    text_token_tags=teacher_tags.to(accelerator.device),
                    layout=teacher_layout,
                    model_t_video=model_t_video,
                    model_t_audio=model_t_audio,
                    visual_condition_latents=teacher_visual_conditions,
                    audio_condition_latents=teacher_audio_conditions,
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
            "teacher/base_sigma": base_sigma,
            "teacher/conditioned": 1.0 if conditioned else 0.0,
            "teacher/video_flow_gap_rms": (teacher_video - video_target.float()).pow(2).mean().sqrt().detach(),
            "teacher/audio_flow_gap_rms": (teacher_audio - audio_target.float()).pow(2).mean().sqrt().detach(),
        }
        return teacher_video, teacher_audio, conditioned, teacher_log

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
        teacher_conditions = normalize_teacher_conditions(args.h3_teacher_conditions) if args.h3_teacher_matching else None
        runtime = _runtime_batch_plan(
            batch, latents, task=args.task, teacher_conditions=teacher_conditions, one_frame=bool(args.one_frame)
        )
        for notice in runtime.notices:
            self._notice(notice)
        # the shared audio-loss policy validates the cached presence flags
        audio_loss_weight = effective_audio_loss_weights(runtime.audio_present, args)
        self._audio_items_seen += int(runtime.audio_present.numel())
        self._audio_supervised_seen += int(runtime.audio_present.sum().item())
        device = latents.device
        noisy_video, timesteps = self.get_noisy_model_input_and_timesteps(
            args, noise, latents, batch["timesteps"], noise_scheduler, device, dit_dtype
        )
        # the audio shares the drawn base sigma under its own shift
        base = (timesteps[0] - 1.0) / 1000.0
        sigma_video = shift_sigma(base, args.h3_shift_video)
        sigma_audio = shift_sigma(base, args.h3_shift_audio)
        audio_latents = batch["latents_audio"].to(device=device)
        audio_noise = torch.randn_like(audio_latents)
        noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

        visual_conditions, audio_conditions = augment_condition_latents(
            runtime.visual_conditions,
            runtime.audio_conditions,
            generator=None,
            visual_clean=args.h3_visual_cond_clean,
            audio_clean=args.h3_audio_cond_clean,
            device=device,
        )
        # the teacher's conditions get the same per-step augmentation as FL2VA/Ref2VA training
        teacher_visual_conditions, teacher_audio_conditions = augment_condition_latents(
            runtime.teacher_visual_conditions,
            runtime.teacher_audio_conditions,
            generator=None,
            visual_clean=args.h3_visual_cond_clean,
            audio_clean=args.h3_audio_cond_clean,
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
            timesteps,
            network_dtype,
            audio_latents=audio_latents,
            audio_noise=audio_noise,
            noisy_audio_input=noisy_audio,
            runtime=runtime,
            model_t_video=1.0 - sigma_video,
            model_t_audio=1.0 - sigma_audio,
            visual_conditions=visual_conditions,
            audio_conditions=audio_conditions,
            audio_loss_weight=audio_loss_weight,
            network=network,
            teacher_visual_conditions=teacher_visual_conditions,
            teacher_audio_conditions=teacher_audio_conditions,
        )
        return self.compute_loss(args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step)

    def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
        """The video half of the H3 noising: one base sigma per item from the shared draw (uniform,
        then the optional timestep focus), shifted by --h3_shift_video. The returned timesteps
        follow the trainer's 1..1000 convention; process_batch derives the audio noising from the
        same base sigma under --h3_shift_audio."""
        del noise_scheduler, dtype
        base = self.sample_timesteps(args, noise.shape[0], timesteps, latents, device)
        base = _apply_timestep_focus(
            base, float(args.h3_timestep_focus_min), float(args.h3_timestep_focus_max), float(args.h3_timestep_focus_prob)
        )
        sigma_video = shift_sigma(base, args.h3_shift_video).view(-1, 1, 1, 1, 1)
        # blended in fp32, stored in the cache dtype (the released FP16 video latents stay FP16)
        noisy_model_input = ((1.0 - sigma_video) * latents.float() + sigma_video * noise.float()).to(latents.dtype)
        return noisy_model_input, base * 1000.0 + 1.0

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
        teacher_matching = bool(args.h3_teacher_matching)
        conditioned = bool(output.extra.get("teacher_conditioned", True))
        dc_weight = float(args.h3_teacher_loss_dc_weight)
        mag_weight = float(args.h3_teacher_loss_mag_weight)

        def flow_loss(pred: torch.Tensor, target: torch.Tensor, *, attenuate_dc: bool) -> torch.Tensor:
            if not teacher_matching:
                return torch.nn.functional.mse_loss(pred, target, reduction="mean")
            # the DC attenuation applies only to conditioned teaching steps: on preservation
            # steps the DC penalty is exactly what pulls palette drift back to the base
            if attenuate_dc and conditioned and dc_weight != 1.0:
                pred = _dc_attenuated_prediction(pred, target, dc_weight)
            # the magnitude down-weight is likewise education-only: on anchor steps the
            # magnitude term is what pulls learned de-amplification back to the base norm
            # (measured on a one-frame TM A/B: ungated mag 0.25 sank the anchor norm ratio)
            return _decomposed_flow_loss(pred, target, mag_weight if conditioned else 1.0, 1.0)

        # the DC attenuation targets the video palette axis; the audio anchor keeps its full DC
        video_loss = flow_loss(output.pred.to(network_dtype), output.target.to(network_dtype), attenuate_dc=True)
        weight = float(output.extra["audio_loss_weight"].item())
        if weight == 0.0:
            audio_loss = video_loss.detach().new_zeros(())
        else:
            audio_loss = flow_loss(
                output.extra["audio_pred"].to(network_dtype),
                output.extra["audio_target"].to(network_dtype),
                attenuate_dc=False,
            )
        logs: dict[str, torch.Tensor | float] = {
            "loss/video": video_loss.detach(),
            "loss/audio": audio_loss.detach(),
            **output.extra.get("guidance_log", {}),
        }
        total_loss = video_loss + weight * audio_loss
        if teacher_matching and not conditioned:
            # preservation-anchor step: user weight on top of the automatic focus compensation,
            # so raising the timestep focus does not silently weaken the drift protection.
            # loss/video and loss/audio are logged unweighted to keep sigma-binned reads comparable
            multiplier = float(args.h3_teacher_preservation_weight) * _preservation_density_compensation(
                float(args.h3_teacher_condition_sigma_max),
                float(args.h3_timestep_focus_min),
                float(args.h3_timestep_focus_max),
                float(args.h3_timestep_focus_prob),
                float(args.h3_teacher_condition_sigma_min),
            )
            if multiplier != 1.0:
                total_loss = total_loss * multiplier
        return total_loss, _scalar_logs(logs)


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(
        timestep_sampling="uniform",
        weighting_scheme="none",
        discrete_flow_shift=1.0,
        network_module="networks.lora_minimax_h3",
    )
    parser.add_argument("--task", choices=H3_TASKS, default=None, help="MiniMax-H3 training task")
    parser.add_argument(
        "--one_frame",
        action="store_true",
        help="experimental one-frame (image) training: accept single-token latent caches written by"
        " minimax_h3_cache_latents.py --one_frame — plain image targets (t2va), editing/inbetween targets"
        " with 1-2 time-annotated control images (fl2va), or reference-conditioned targets (ref2va)."
        " Video batches are unaffected, so image and video datasets can mix in one run",
    )
    add_audio_train_args(parser)
    add_h3_sampling_args(parser)
    add_h3_vae_args(parser, note="training-time joint AV samples")
    add_h3_text_encoder_args(parser, note="training-time sample prompts")
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
        " predictions conditioned on privileged information from the training clip (one extra no-grad forward per"
        " step); the condition set is chosen by --h3_teacher_conditions and the matching text cache must be written"
        " with the same --teacher_conditions value. The teacher targets live in the distilled guided space, so this"
        " replaces (and is mutually exclusive with) --h3_guidance_loss_scale. With 'first,last' conditions audio"
        " degenerates to a base-preservation anchor (real audio content is not learned); with 'ref' the reference"
        " audio is a real teaching target; with 'subject_ref' the teacher sees other pictures of the subject"
        " (partial information) and image targets (--one_frame) are supported.",
    )
    parser.add_argument(
        "--h3_teacher_conditions",
        type=str,
        default="first,last",
        help="conditions handed to the teacher forward. 'first,last' (default): FL2VA teacher on the real first/last"
        " frames, requires FL2VA-style latent caches and a text cache written with --teacher_conditions first,last."
        " 'ref': Ref2VA teacher on the training clip itself (cached target video+audio latents as the reference),"
        " complete information at every sigma; requires a text cache written with --teacher_conditions ref, works"
        " with FL2VA or T2VA latent caches (first/last latents are unused)."
        " 'subject_ref': Ref2VA teacher on the item's own JSONL image references (other pictures of the subject),"
        " partial information: the concept plus the base's own amplification without the complete-information"
        " degeneration; requires --task ref2va latent caches (image datasets with --one_frame) and a text cache"
        " written with --teacher_conditions subject_ref; the only teacher available with --one_frame",
    )
    parser.add_argument(
        "--h3_teacher_condition_sigma_min",
        type=float,
        default=0.0,
        help="teacher matching only: below this drawn base sigma the teacher likewise drops its conditions and the"
        " step becomes a base-preservation anchor (0 = off). Near sigma 0 the noised target itself reveals the"
        " clean latent, so any teacher's prediction collapses toward the raw velocity target and stops carrying the"
        " base's amplification; the mirror of --h3_guidance_loss_sigma_min. Recommended 0.15 for image targets"
        " with the subject_ref teacher",
    )
    parser.add_argument(
        "--h3_teacher_condition_sigma_max",
        type=float,
        default=1.0,
        help="teacher matching only: above this drawn base sigma (pre-shift, 1 = pure noise) the teacher drops its"
        " conditions and runs on the student's own text, turning the target into a pure base-preservation"
        " anchor (1.0 = always conditioned). Recommended per teacher: 1.0 (the default) for subject_ref, whose"
        " teacher keeps the base's amplification at every sigma and whose identity decisions live at base sigma"
        " 0.92-1.0; 0.75 for first,last and ref, where near pure noise the conditioned content is unpredictable"
        " from the text and unrestricted teaching overwrites the base composition prior (for ref the FL2VA weights"
        " also fail to align the reference above base sigma ~0.85). Lower toward 0.4-0.5 for low-diversity data",
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
        help="teacher matching only: on conditioned teaching steps, weight of the magnitude term of the decomposed"
        " loss, relative to the direction term fixed at 1.0. At 1.0 the loss value equals the plain MSE (only the"
        " gradient geometry differs); lower it to prioritize direction matching (0 = pure direction)."
        " Preservation-anchor steps always keep the full magnitude term — it is what pulls learned"
        " de-amplification back to the base norm",
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
    MiniMaxH3NetworkTrainer().train(args)


if __name__ == "__main__":
    main()
