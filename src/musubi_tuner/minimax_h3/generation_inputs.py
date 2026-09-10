"""The MiniMax-H3 generation contract shared by ``minimax_h3_generate_video.py`` and the
trainer's training-time samples: one validated request (``H3GenerationRequest``), the prompt-line
vocabulary that fills it, its media and condition inputs, and its packed layout. The two callers
differ only in where a request comes from (an argparse namespace, or a sample prompt dict) and in
how they load the models around these helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from musubi_tuner.dataset.media_utils import resize_image_to_bucket
from musubi_tuner.minimax_h3.audio_vae import encode_audio_mode
from musubi_tuner.minimax_h3.media import (
    CANVAS_MULTIPLE,
    H3_TASKS,
    ONE_FRAME_REFERENCE_FRAME_CAP,
    RELEASED_DURATION_SECONDS,
    TARGET_FPS,
    TEXT_VISUAL_FPS,
    TEXT_VISUAL_FRAME_STRIDE,
    H3MediaDecoder,
    H3Record,
    H3Task,
    audio_latent_frames,
    load_h3_jsonl_records,
    module_device_dtype,
    parse_inline_references,
    prepare_pixels,
    video_latent_frames,
    waveform_samples,
)
from musubi_tuner.minimax_h3.model import MiniMaxH3Config
from musubi_tuner.minimax_h3.packing import (
    FRAME_RESCALE,
    ONE_FRAME_AUDIO_LATENT_FRAMES,
    ONE_FRAME_VIDEO_LATENT_FRAMES,
    H3PackedLayout,
    H3ReferenceGeometry,
    H3TimeOverrides,
    H3VideoGeometry,
    build_h3_layout,
    one_frame_condition_role,
    validate_clean_coefficient,
)
from musubi_tuner.minimax_h3.sampling import (
    DEFAULT_AUDIO_CONDITION_CLEAN,
    DEFAULT_AUDIO_SHIFT,
    DEFAULT_VIDEO_SHIFT,
    DEFAULT_VISUAL_CONDITION_CLEAN,
    validate_shift,
)
from musubi_tuner.minimax_h3.text_encoder import H3TextVisual
from musubi_tuner.minimax_h3.video_vae import VIDEO_VAE_ENCODE_DTYPE, encode_video_condition


logger = logging.getLogger(__name__)

VIDEO_VAE_SPATIAL_RATIO = 16
# released canvas and schedule defaults (the generation CLI's argparse defaults, and what a
# training sample prompt gets when it leaves them out)
DEFAULT_WIDTH = 768
DEFAULT_HEIGHT = 1344
DEFAULT_FRAME_COUNT = 124
DEFAULT_STEPS = 30


def parse_one_frame_options(spec: str) -> tuple[int, tuple[int, ...] | None]:
    """Parses --one_frame_inference "target_index=N,control_index=A;B" into 24 fps pixel-frame indices."""

    def nonnegative_index(value: str, label: str) -> int:
        try:
            index = int(value)
        except ValueError as error:
            raise ValueError(f"MiniMax-H3 --one_frame_inference {label} must be an integer, got {value!r}") from error
        if index < 0:
            raise ValueError(f"MiniMax-H3 --one_frame_inference {label} must be nonnegative, got {index}")
        return index

    target_index = 0
    control_indices = None
    seen = set()
    for part in spec.split(","):
        key, separator, value = part.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise ValueError(f"MiniMax-H3 --one_frame_inference options must be key=value, got {part!r}")
        if key not in {"target_index", "control_index"}:
            raise ValueError(f"MiniMax-H3 --one_frame_inference has unknown option {key!r} (allowed: target_index, control_index)")
        if key in seen:
            raise ValueError(f"MiniMax-H3 --one_frame_inference has duplicate option {key!r}")
        seen.add(key)
        if key == "target_index":
            target_index = nonnegative_index(value, "target_index")
        else:
            control_indices = tuple(nonnegative_index(item, "control_index") for item in value.split(";"))
    return target_index, control_indices


@dataclass(frozen=True)
class H3GenerationRequest:
    """One generation, as the shared helpers read it.

    The fields are named like the generation CLI's arguments (``request_from_args`` copies them
    one to one); a training sample prompt dict reaches the same shape through
    ``request_overrides``. A request is only trusted after ``validate_generation_request``.
    """

    task: H3Task
    prompt: str | None = None
    first_frame: str | None = None
    last_frame: str | None = None
    condition_image: Sequence[str] | None = None
    reference_jsonl: str | None = None
    reference_index: int = 0
    ref: Sequence[str] | None = None
    # where relative --ref paths resolve from: the working directory for the CLI (None), the
    # prompt file's directory for training samples
    ref_base_directory: Path | None = None
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    frame_count: int = DEFAULT_FRAME_COUNT
    one_frame_inference: str | None = None
    output_fps: int = TARGET_FPS
    stretch_keep_bands: int = 0
    allow_experimental_duration: bool = False
    steps: int = DEFAULT_STEPS
    seed: int | None = None
    h3_shift_video: float = DEFAULT_VIDEO_SHIFT
    h3_shift_audio: float = DEFAULT_AUDIO_SHIFT
    h3_visual_cond_clean: float = DEFAULT_VISUAL_CONDITION_CLEAN
    h3_audio_cond_clean: float = DEFAULT_AUDIO_CONDITION_CLEAN

    @property
    def one_frame(self) -> bool:
        return self.frame_count == 1

    def one_frame_indices(self) -> tuple[int, tuple[int, ...] | None]:
        """(target_index, control_indices) of a one-frame request on the 24 fps pixel-frame timeline."""
        if self.one_frame_inference is None:
            return 0, None
        return parse_one_frame_options(self.one_frame_inference)


def unescape_prompt(prompt: str | None) -> str | None:
    """The literal string "\\n" becomes a newline, so a one-line prompt (a CLI argument or a
    prompt-file line) can carry the multi-line official caption format."""
    return None if prompt is None else prompt.replace("\\n", "\n")


def request_from_args(args) -> H3GenerationRequest:
    """The request of a generation CLI namespace (the parser in minimax_h3_generate_video.py defines every field)."""
    return H3GenerationRequest(
        task=args.task,
        prompt=unescape_prompt(args.prompt),
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        condition_image=args.condition_image,
        reference_jsonl=args.reference_jsonl,
        reference_index=args.reference_index,
        ref=args.ref,
        width=args.width,
        height=args.height,
        frame_count=args.frame_count,
        one_frame_inference=args.one_frame_inference,
        output_fps=args.output_fps,
        stretch_keep_bands=args.stretch_keep_bands,
        allow_experimental_duration=args.allow_experimental_duration,
        steps=args.steps,
        seed=args.seed,
        h3_shift_video=args.h3_shift_video,
        h3_shift_audio=args.h3_shift_audio,
        h3_visual_cond_clean=args.h3_visual_cond_clean,
        h3_audio_cond_clean=args.h3_audio_cond_clean,
    )


# prompt dict keys (training/sampling_prompts.line_to_prompt_dict, or a JSON/TOML prompt file)
# -> request fields. The generic keys keep their house meaning; the H3 target-audio shift and the
# temporal-stretch options are H3-only additions to the line vocabulary.
PROMPT_DICT_FIELDS: Mapping[str, str] = {
    "prompt": "prompt",
    "width": "width",
    "height": "height",
    "frame_count": "frame_count",
    "seed": "seed",
    "sample_steps": "steps",
    "discrete_flow_shift": "h3_shift_video",
    "discrete_flow_shift_audio": "h3_shift_audio",
    "output_fps": "output_fps",
    "stretch_keep_bands": "stretch_keep_bands",
    "image_path": "first_frame",
    "end_image_path": "last_frame",
    "control_image_path": "condition_image",
    "one_frame": "one_frame_inference",
    "reference_jsonl": "reference_jsonl",
    "reference_index": "reference_index",
    "ref": "ref",
    "allow_experimental_duration": "allow_experimental_duration",
}
# generic prompt options with no MiniMax-H3 meaning: the sampler has no CFG, negative prompt or control video
UNSUPPORTED_PROMPT_KEYS: Mapping[str, str] = {
    "negative_prompt": "--n",
    "cfg_scale": "--l",
    "guidance_scale": "--g",
    "control_video_path": "--cn",
}
# the caller's fields, never taken from a prompt dict
_CALLER_FIELDS = frozenset({"task", "ref_base_directory"})
_INT_FIELDS = frozenset({"width", "height", "frame_count", "seed", "steps", "output_fps", "stretch_keep_bands", "reference_index"})
_FLOAT_FIELDS = frozenset({"h3_shift_video", "h3_shift_audio", "h3_visual_cond_clean", "h3_audio_cond_clean"})


def request_overrides(prompt_dict: Mapping[str, Any]) -> dict[str, Any]:
    """The request fields a prompt dict sets, keyed by field name (for ``dataclasses.replace`` on a
    base request, or ``setattr`` on a CLI namespace).

    Prompt-line keys are translated with ``PROMPT_DICT_FIELDS``; request field names are accepted
    as they are (prompt files written in the request vocabulary); unrelated keys (``enum`` and the
    like) are ignored; the generic options H3 cannot honor are rejected.
    """
    field_names = {field.name for field in fields(H3GenerationRequest)} - _CALLER_FIELDS
    overrides: dict[str, Any] = {}
    for key, value in prompt_dict.items():
        if key in UNSUPPORTED_PROMPT_KEYS:
            raise ValueError(
                f"MiniMax-H3 prompts do not support {key} ({UNSUPPORTED_PROMPT_KEYS[key]}): the sampler has no CFG,"
                " negative prompt or control video"
            )
        field = PROMPT_DICT_FIELDS.get(key, key if key in field_names else None)
        if field is None:
            continue
        if value is not None and field in _INT_FIELDS:
            value = int(value)
        elif value is not None and field in _FLOAT_FIELDS:
            value = float(value)
        elif field == "prompt":
            value = unescape_prompt(value)
        overrides[field] = value
    return overrides


def require_path(value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"MiniMax-H3 generation requires --{label}")
    path = Path(value).expanduser()
    if not path.exists():
        raise ValueError(f"MiniMax-H3 --{label} does not exist: {path}")
    return path


def _validate_string_list(values: Sequence[str] | None, label: str) -> None:
    if values is None:
        return
    if isinstance(values, str) or not all(isinstance(value, str) and value.strip() for value in values):
        raise ValueError(f"MiniMax-H3 --{label} entries must be non-empty strings")


def validate_generation_request(request: H3GenerationRequest) -> None:
    """Rejects a request the pipeline cannot run: the canvas, timeline and schedule ranges, and the
    per-task inputs (condition images and the reference JSONL must exist; ``--ref`` specs are
    parsed with the JSONL reference rules). Everything past this point assumes a validated request."""
    if request.task not in H3_TASKS:
        raise ValueError(f"MiniMax-H3 --task must be one of {', '.join(H3_TASKS)}, got {request.task!r}")
    if request.width <= 0 or request.height <= 0 or request.width % CANVAS_MULTIPLE or request.height % CANVAS_MULTIPLE:
        raise ValueError(
            f"MiniMax-H3 width and height must be positive and divisible by {CANVAS_MULTIPLE}, got {request.width}x{request.height}"
        )
    one_frame = request.one_frame
    # fps above the native rate would let the duration gate admit packed sequences far past
    # the released maximum (and desynchronize the floored audio count), so the squeeze
    # direction stays closed until it is validated
    if not 1 <= request.output_fps <= TARGET_FPS:
        raise ValueError(f"MiniMax-H3 --output_fps must be in [1,{TARGET_FPS}], got {request.output_fps}")
    if one_frame and request.output_fps != TARGET_FPS:
        raise ValueError(f"MiniMax-H3 one-frame generation has no timeline to stretch; --output_fps must stay {TARGET_FPS}")
    # at least one band must stay on the stretched clock, or the video RoPE silently reverts
    # to the native timeline while the audio still covers the stretched duration
    max_keep_bands = MiniMaxH3Config.rope_inv_freq_len - 1
    if not 0 <= request.stretch_keep_bands <= max_keep_bands:
        raise ValueError(f"MiniMax-H3 --stretch_keep_bands must be in [0,{max_keep_bands}], got {request.stretch_keep_bands}")
    if request.stretch_keep_bands and request.output_fps == TARGET_FPS:
        raise ValueError(f"MiniMax-H3 --stretch_keep_bands requires an --output_fps below {TARGET_FPS}")
    if one_frame:
        _, control_indices = request.one_frame_indices()
        if request.task == "fl2va":
            entries = fl_condition_entries(request)
            # a missing-images error is raised by the task input checks below
            if entries and (control_indices is None or len(control_indices) != len(entries)):
                given = 0 if control_indices is None else len(control_indices)
                provided = ", ".join(path for _, path in entries)
                raise ValueError(
                    "MiniMax-H3 one-frame FL2VA requires --one_frame_inference control_index with one entry per condition"
                    f" image: got {given} control_index entries for {len(entries)} condition images ({provided}), "
                    'e.g. --one_frame_inference "target_index=24,control_index=0" for one condition at index 0'
                )
        elif control_indices is not None:
            raise ValueError("MiniMax-H3 --one_frame_inference control_index applies only to FL2VA conditions")
    else:
        if request.one_frame_inference is not None:
            raise ValueError("MiniMax-H3 --one_frame_inference options require --frame_count 1 (--f 1 in prompt lines)")
        video_latent_frames(request.frame_count)
        # with a temporal stretch the rotary timeline spans the real (stretched) duration,
        # so that is the quantity to hold inside the released range
        duration = request.frame_count / request.output_fps
        shortest, longest = RELEASED_DURATION_SECONDS
        if not request.allow_experimental_duration and not shortest <= duration <= longest:
            raise ValueError(
                f"MiniMax-H3 duration {duration:.3f}s is outside the released {shortest:g}-{longest:g}s range; "
                "pass --allow_experimental_duration to proceed"
            )
    if request.steps <= 0:
        raise ValueError("MiniMax-H3 --steps must be positive")
    validate_shift(request.h3_shift_video, "--h3_shift_video")
    validate_shift(request.h3_shift_audio, "--h3_shift_audio")
    validate_clean_coefficient(request.h3_visual_cond_clean, "--h3_visual_cond_clean")
    validate_clean_coefficient(request.h3_audio_cond_clean, "--h3_audio_cond_clean")
    _validate_string_list(request.condition_image, "condition_image")
    _validate_string_list(request.ref, "ref")

    if request.task == "t2va":
        if not request.prompt:
            raise ValueError("MiniMax-H3 T2VA requires --prompt")
        if request.first_frame or request.last_frame or request.condition_image or request.reference_jsonl or request.ref:
            raise ValueError("MiniMax-H3 T2VA does not accept condition/first/last/reference inputs")
    elif request.task == "fl2va":
        if not request.prompt:
            raise ValueError("MiniMax-H3 FL2VA requires --prompt")
        if request.reference_jsonl or request.ref:
            raise ValueError("MiniMax-H3 FL2VA does not accept --reference_jsonl or --ref")
        entries = fl_condition_entries(request)  # rejects --condition_image for video targets and mixed one-frame inputs
        if not entries:
            raise ValueError(
                "MiniMax-H3 FL2VA requires --first_frame and/or --last_frame"
                " (first only = I2VA, last only = L2VA; use --task t2va to condition on neither;"
                " one-frame targets may also take the ordered --condition_image list)"
            )
        for label, path in entries:
            require_path(path, label)
    else:
        if bool(request.reference_jsonl) == bool(request.ref):
            raise ValueError("MiniMax-H3 Ref2VA requires exactly one of --reference_jsonl (--rj) or --ref")
        if request.first_frame or request.last_frame or request.condition_image:
            raise ValueError("MiniMax-H3 Ref2VA does not accept --first_frame, --last_frame or --condition_image")
        if request.ref:
            if not request.prompt:
                raise ValueError("MiniMax-H3 Ref2VA with --ref requires --prompt")
            if request.reference_index:
                raise ValueError("MiniMax-H3 --reference_index selects a --reference_jsonl record and does not apply to --ref")
            # existence, probes and count limits are checked here, before any model is loaded;
            # load_generation_record parses the specs again into the record
            parse_inline_references(request.ref, Path(request.ref_base_directory or Path.cwd()))
        else:
            require_path(request.reference_jsonl, "reference_jsonl")
            if request.reference_index < 0:
                raise ValueError("MiniMax-H3 --reference_index must be nonnegative")


def dummy_record(prompt: str) -> H3Record:
    return H3Record(video_path=Path("."), caption=prompt, references=(), label="--prompt")


def load_image_frames(path: str | Path, *, width: int, height: int) -> torch.Tensor:
    """An FL2VA condition image as a uint8 [1,H,W,3] frame on the target canvas.

    The image is fitted the way the dataset layer fits training targets and control images
    (``resize_image_to_bucket``: scale to cover the canvas, then center crop), so a LoRA sees
    its conditions preprocessed exactly as during training. The released Diffusers pipeline
    instead stretches the first picture onto the canvas and cover-crops the rest; training
    parity is preferred here, the stretch never being a no-op with an explicit canvas.
    """
    with Image.open(path) as image:
        pixels = resize_image_to_bucket(np.asarray(image.convert("RGB")), (width, height))
    return torch.from_numpy(np.ascontiguousarray(pixels)).unsqueeze(0)


def load_generation_record(request: H3GenerationRequest) -> H3Record:
    """The H3 record of a generation request; ``--ref`` paths resolve from ``ref_base_directory``."""
    if request.task in {"t2va", "fl2va"}:
        return dummy_record(request.prompt or "")

    if request.ref:
        references = parse_inline_references(request.ref, Path(request.ref_base_directory or Path.cwd()))
        return H3Record(video_path=Path("."), caption=request.prompt or "", references=references, label="--ref")

    records = load_h3_jsonl_records(request.reference_jsonl, "ref2va")
    if request.reference_index >= len(records):
        raise ValueError(f"MiniMax-H3 --reference_index {request.reference_index} is outside {len(records)} JSONL records")
    record = records[request.reference_index]
    if request.prompt is not None:
        record = replace(record, caption=request.prompt)
    return record


def fl_condition_entries(request: H3GenerationRequest) -> tuple[tuple[str, str], ...]:
    """The FL2VA condition images of a generation request as ordered (role, path) pairs.

    Video targets take the released ``first``/``last`` anchors (``--first_frame`` /
    ``--last_frame``). One-frame targets take an ordered list of any length: the repeatable
    ``--condition_image`` (``--ci`` in prompt lines), or, as aliases for the first two slots,
    ``--first_frame`` / ``--last_frame``; the roles are the ``cond_{i}`` slots and the times come
    from ``--one_frame_inference control_index`` in the same order.
    """
    first_frame = request.first_frame
    last_frame = request.last_frame
    condition_images = request.condition_image or ()
    if not request.one_frame:
        if condition_images:
            raise ValueError(
                "MiniMax-H3 --condition_image applies to one-frame targets (--frame_count 1); video FL2VA takes"
                " --first_frame and/or --last_frame"
            )
        return tuple((role, path) for role, path in (("first", first_frame), ("last", last_frame)) if path)
    if condition_images and (first_frame or last_frame):
        raise ValueError(
            "MiniMax-H3 one-frame FL2VA takes either --condition_image entries or --first_frame/--last_frame, not both"
        )
    paths = list(condition_images) if condition_images else [path for path in (first_frame, last_frame) if path]
    return tuple((one_frame_condition_role(index), path) for index, path in enumerate(paths))


def one_frame_time_overrides(request: H3GenerationRequest) -> H3TimeOverrides | None:
    """The rotary times of a one-frame request (None for video targets)."""
    if not request.one_frame:
        return None
    target_index, control_indices = request.one_frame_indices()
    return H3TimeOverrides(
        condition_times=tuple(FRAME_RESCALE * index for index in (control_indices or ())),
        target_time=FRAME_RESCALE * target_index,
    )


def decode_generation_visuals(request: H3GenerationRequest, record: H3Record, decoder: H3MediaDecoder):
    raw_visuals = {}
    text_visuals = {}
    if request.task == "t2va":
        return raw_visuals, text_visuals
    if request.task == "fl2va":
        for role, path in fl_condition_entries(request):
            frames = load_image_frames(path, width=request.width, height=request.height)
            raw_visuals[role] = frames
            text_visuals[role] = H3TextVisual(frames)
        return raw_visuals, text_visuals

    if request.one_frame:
        reference_frame_cap = ONE_FRAME_REFERENCE_FRAME_CAP
    else:
        # cap reference videos by the real target duration in native 24 fps frames; a
        # temporal stretch makes that duration exceed frame_count generated frames
        reference_frame_cap = request.frame_count * TARGET_FPS // request.output_fps
    for reference in record.references:
        if reference.type not in {"image", "video"}:
            continue
        frames = decoder.decode_reference_visual(
            reference,
            target_frame_count=reference_frame_cap,
            target_size=(request.width, request.height),
        )
        raw_visuals[reference.path] = frames
        if reference.type == "image":
            text_visuals[reference.path] = H3TextVisual(frames)
        else:
            sampled = frames[::TEXT_VISUAL_FRAME_STRIDE]
            text_visuals[reference.path] = H3TextVisual(
                sampled,
                tuple(index / TEXT_VISUAL_FPS for index in range(sampled.shape[0])),
            )
    return raw_visuals, text_visuals


def reference_video_frame_counts(record: H3Record, raw_visuals: Mapping[Any, torch.Tensor]) -> dict[int, int]:
    """Decoded frame count per video reference (by reference index), the audio window of its track."""
    return {
        index: int(raw_visuals[reference.path].shape[0])
        for index, reference in enumerate(record.references)
        if reference.type == "video"
    }


@torch.no_grad()
def encode_visual_conditions(request: H3GenerationRequest, record: H3Record, raw_visuals, video_vae):
    video_device, video_dtype = module_device_dtype(video_vae, VIDEO_VAE_ENCODE_DTYPE)
    visual_latents = []
    visual_geometries = []
    reference_visual_geometries = {}

    def encode_visual(frames):
        latent = encode_video_condition(video_vae, prepare_pixels(frames).to(video_device, video_dtype)).cpu()
        visual_latents.append(latent)
        return H3VideoGeometry(*latent.shape[2:])

    if request.task == "fl2va":
        for role, _ in fl_condition_entries(request):
            visual_geometries.append(encode_visual(raw_visuals[role]))
    elif request.task == "ref2va":
        for index, reference in enumerate(record.references):
            if reference.type in {"image", "video"}:
                reference_visual_geometries[index] = encode_visual(raw_visuals[reference.path])
    return tuple(visual_latents), tuple(visual_geometries), reference_visual_geometries


@torch.no_grad()
def encode_audio_conditions(
    request: H3GenerationRequest,
    record: H3Record,
    decoder: H3MediaDecoder,
    audio_vae,
    *,
    reference_video_frame_counts: Mapping[int, int],
):
    audio_device, audio_dtype = module_device_dtype(audio_vae, torch.float32)
    audio_latents = []
    reference_audio_frames = {}
    for index, reference in enumerate(record.references):
        if reference.audio is None:
            continue
        if reference.type == "video":
            if index not in reference_video_frame_counts:
                raise ValueError(f"MiniMax-H3 reference video {index:03d} is missing its decoded frame count")
            frame_count = reference_video_frame_counts[index]
            frames = audio_latent_frames(frame_count)
            require_exact = True
        else:
            # standalone audio spans the target duration (stretched when --output_fps lowers the
            # sampling rate); one-frame generation rejects it upstream
            frames = audio_latent_frames(request.frame_count, output_fps=request.output_fps)
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


def build_reference_geometries(record: H3Record, visual_geometries, audio_frames) -> tuple[H3ReferenceGeometry, ...]:
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


def build_generation_layout(
    request: H3GenerationRequest,
    *,
    text_length: int,
    visual_geometries: Sequence[H3VideoGeometry] = (),
    reference_geometries: Sequence[H3ReferenceGeometry] = (),
) -> H3PackedLayout:
    """The packed layout of a validated request: the target geometry from the canvas and frame
    count, the FL2VA anchor roles from the given condition images, the one-frame times from
    ``--one_frame_inference``, and the temporal-stretch options."""
    one_frame = request.one_frame
    condition_roles = None
    if request.task == "fl2va" and not one_frame:
        # video FL2VA roles select the anchor times; one-frame layouts derive their ordered cond_{i} roles
        condition_roles = tuple(role for role, _ in fl_condition_entries(request))
    layout = build_h3_layout(
        task=request.task,
        text_length=text_length,
        target_video=H3VideoGeometry(
            ONE_FRAME_VIDEO_LATENT_FRAMES if one_frame else video_latent_frames(request.frame_count),
            request.height // VIDEO_VAE_SPATIAL_RATIO,
            request.width // VIDEO_VAE_SPATIAL_RATIO,
        ),
        target_audio_frames=(
            ONE_FRAME_AUDIO_LATENT_FRAMES if one_frame else audio_latent_frames(request.frame_count, output_fps=request.output_fps)
        ),
        visual_conditions=tuple(visual_geometries),
        references=tuple(reference_geometries),
        one_frame=one_frame,
        condition_roles=condition_roles,
        time_overrides=one_frame_time_overrides(request),
        output_fps=request.output_fps,
        temporal_fine_bands=request.stretch_keep_bands,
    )
    logger.info(
        "MiniMax-H3 layout: task=%s video=%s audio_frames=%d text_rows=%d packed_rows=%d temporal_stretch=%.4f fine_bands=%d",
        layout.task,
        layout.target_video,
        layout.target_audio_frames,
        layout.text_length,
        layout.row_count,
        layout.temporal_stretch,
        layout.temporal_fine_bands,
    )
    return layout
