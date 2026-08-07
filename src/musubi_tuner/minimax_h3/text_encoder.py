# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted for Musubi from Hugging Face Diffusers PR #14355 at commit
# abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc
# (modular_pipelines/minimax_h3/encoders.py and packing_ref2va.py).
# ComfyUI is used only as an independent numerical reference.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from musubi_tuner.minimax_h3.checkpoint import (
    inspect_safetensors_convrot_int8,
    load_safetensors_module,
    resolve_safetensors_files,
)
from musubi_tuner.minimax_h3.media import H3Record, H3Task


TEXT_WIDTH = 5120
MAX_TEXT_ROWS = 32768
LAYER_50_HIDDEN_STATE_INDEX = 50
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
DEFAULT_PROCESSOR_ID = "Qwen/Qwen3-VL-32B-Instruct"


@dataclass(frozen=True)
class H3TextVisual:
    frames: torch.Tensor
    timestamps: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.frames.ndim != 4 or self.frames.shape[-1] < 3 or self.frames.shape[0] == 0:
            raise ValueError(f"MiniMax-H3 text visual must be [T,H,W,C], got {tuple(self.frames.shape)}")
        if self.timestamps is not None and len(self.timestamps) != self.frames.shape[0]:
            raise ValueError("MiniMax-H3 text visual timestamps must match its frame count")


@dataclass(frozen=True)
class H3Presentation:
    text: str
    images: tuple[torch.Tensor, ...] = ()
    videos: tuple[torch.Tensor, ...] = ()
    processor_text: str | None = None


def _require_visual(visuals: Mapping[object, H3TextVisual], key: object, label: str) -> H3TextVisual:
    try:
        return visuals[key]
    except KeyError as error:
        raise ValueError(f"MiniMax-H3 presentation is missing {label} visual data") from error


def build_presentation(
    record: H3Record,
    task: H3Task,
    visuals: Mapping[object, H3TextVisual] | None = None,
) -> H3Presentation:
    if task not in {"t2va", "fl2va", "ref2va"}:
        raise ValueError(f"Unsupported MiniMax-H3 task: {task}")
    visuals = visuals or {}
    if task != "ref2va" and record.references:
        raise ValueError(f"MiniMax-H3 task {task} does not accept references")
    if task == "t2va":
        return H3Presentation(text=record.caption, processor_text=record.caption)

    parts = []
    processor_parts = []
    images = []
    videos = []
    if task == "fl2va":
        for index, key in enumerate(("first", "last"), start=1):
            visual = _require_visual(visuals, key, f"FL2VA {key}")
            if visual.frames.shape[0] != 1:
                raise ValueError(f"MiniMax-H3 FL2VA {key} visual must contain exactly one frame")
            part = f"<Picture {index}>: {IMAGE_PLACEHOLDER}"
            parts.append(part)
            processor_parts.append(part)
            images.append(visual.frames[0])
        parts.append(record.caption)
        processor_parts.append(record.caption)
        return H3Presentation(
            text="".join(parts),
            images=tuple(images),
            processor_text="".join(processor_parts),
        )

    counters = {"image": 0, "audio": 0, "video": 0}
    for reference in record.references:
        if reference.type == "image":
            counters["image"] += 1
            visual = _require_visual(visuals, reference.path, f"reference image {reference.path}")
            if visual.frames.shape[0] != 1:
                raise ValueError(f"MiniMax-H3 reference image must contain one frame: {reference.path}")
            part = f"<Picture {counters['image']}>: {IMAGE_PLACEHOLDER}"
            parts.append(part)
            processor_parts.append(part)
            images.append(visual.frames[0])
            continue

        if reference.type == "audio":
            counters["audio"] += 1
            part = f"<Audio {counters['audio']}>: "
            parts.append(part)
            processor_parts.append(part)
            continue

        if reference.type != "video":
            raise ValueError(f"Unsupported MiniMax-H3 reference type: {reference.type}")
        if reference.audio is not None:
            counters["audio"] += 1
            part = f"<Audio {counters['audio']}>: "
            parts.append(part)
            processor_parts.append(part)
        counters["video"] += 1
        part = f"<Video {counters['video']}>: "
        parts.append(part)
        processor_parts.append(part)
        processor_parts.append(VIDEO_PLACEHOLDER)

        visual = _require_visual(visuals, reference.path, f"reference video {reference.path}")
        frames = visual.frames
        timestamps = list(visual.timestamps) if visual.timestamps is not None else [index / 2.0 for index in range(len(frames))]
        if len(frames) % 2:
            frames = torch.cat((frames, frames[-1:]), dim=0)
            timestamps.append(timestamps[-1])
        for index in range(0, len(frames), 2):
            block_timestamp = (timestamps[index] + timestamps[index + 1]) / 2.0
            parts.append(f"<{block_timestamp:.1f} seconds>{VIDEO_PLACEHOLDER}")
        videos.append(visual.frames)

    parts.append(record.caption)
    processor_parts.append(record.caption)
    return H3Presentation(
        text="".join(parts),
        images=tuple(images),
        videos=tuple(videos),
        processor_text="".join(processor_parts),
    )


def build_token_tags(processed: Mapping[str, Any]) -> torch.Tensor:
    if "input_ids" not in processed:
        raise ValueError("MiniMax-H3 processor output has no input_ids")
    input_ids = torch.as_tensor(processed["input_ids"])
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("MiniMax-H3 text caching processes one presentation at a time")
        input_ids = input_ids[0]
    if input_ids.ndim != 1:
        raise ValueError(f"MiniMax-H3 input_ids must be [L] or [1,L], got {tuple(input_ids.shape)}")

    tags = torch.ones(input_ids.shape[0], dtype=torch.int64, device=input_ids.device)
    open_start = None
    for index, token_id in enumerate(input_ids.tolist()):
        if token_id == VISION_START_TOKEN_ID:
            if open_start is not None:
                raise ValueError("MiniMax-H3 processor produced nested vision-start tokens")
            open_start = index
        elif token_id == VISION_END_TOKEN_ID:
            if open_start is None:
                raise ValueError("MiniMax-H3 processor produced an unmatched vision-end token")
            interior = input_ids[open_start + 1 : index]
            if not torch.any((interior == IMAGE_TOKEN_ID) | (interior == VIDEO_TOKEN_ID)):
                raise ValueError("MiniMax-H3 vision span contains no expanded image or video rows")
            tags[open_start : index + 1] = 0
            open_start = None
    if open_start is not None:
        raise ValueError("MiniMax-H3 processor produced an unmatched vision-start token")
    return tags.cpu()


def _text_size_error(row_count: int, token_tags: torch.Tensor) -> ValueError:
    vision_rows = int((token_tags == 0).sum().item())
    text_rows = row_count - vision_rows
    payload_mib = row_count * TEXT_WIDTH * 2 / (1024**2)
    return ValueError(
        f"MiniMax-H3 text presentation has {row_count} rows, exceeds {MAX_TEXT_ROWS}; "
        f"vision_rows={vision_rows}, text_rows={text_rows}, estimated BF16 payload={payload_mib:.1f} MiB"
    )


def validate_text_rows(hidden_states: torch.Tensor, token_tags: torch.Tensor) -> None:
    if hidden_states.ndim != 2 or hidden_states.shape[1] != TEXT_WIDTH:
        raise ValueError(f"Expected MiniMax-H3 hidden states [L,{TEXT_WIDTH}], got {tuple(hidden_states.shape)}")
    if token_tags.dtype != torch.int64 or token_tags.shape != (hidden_states.shape[0],):
        raise ValueError("MiniMax-H3 token tags must be int64 [L]")
    if hidden_states.shape[0] > MAX_TEXT_ROWS:
        raise _text_size_error(hidden_states.shape[0], token_tags)
    if not torch.all((token_tags == 0) | (token_tags == 1)):
        raise ValueError("MiniMax-H3 token tags may contain only 0 and 1")


def _language_layers(model) -> list:
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError("MiniMax-H3 Qwen3-VL model has no language_model.layers")
    return layers


def extract_layer_50_pre_norm(output, model) -> torch.Tensor:
    layers = _language_layers(model)
    if len(layers) < LAYER_50_HIDDEN_STATE_INDEX:
        raise ValueError(f"MiniMax-H3 Qwen3-VL needs at least 50 layers, got {len(layers)}")

    captured = getattr(output, "h3_layer_50_pre_norm", None)
    if captured is None:
        captured = getattr(model, "_h3_layer_50_pre_norm", None)
    if captured is not None:
        return captured

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is None or len(hidden_states) <= LAYER_50_HIDDEN_STATE_INDEX:
        raise ValueError("MiniMax-H3 layer-50 pre-norm state was not captured")
    return hidden_states[LAYER_50_HIDDEN_STATE_INDEX]


def normalize_h3_text_encoder_key(key: str) -> str:
    if key.startswith("model."):
        return "language_model." + key.removeprefix("model.")
    return key


def load_h3_processor(path: str = DEFAULT_PROCESSOR_ID, *, revision: str | None = None):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(path, revision=revision)


def load_h3_text_encoder(
    checkpoint_path: str | Path,
    *,
    processor_path: str = DEFAULT_PROCESSOR_ID,
    revision: str | None = None,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    disable_mmap: bool = False,
):
    from transformers import Qwen3VLConfig, Qwen3VLModel

    config = Qwen3VLConfig.from_pretrained(processor_path, revision=revision)
    if config.text_config.hidden_size != TEXT_WIDTH:
        raise ValueError(f"MiniMax-H3 Qwen3-VL hidden size must be {TEXT_WIDTH}, got {config.text_config.hidden_size}")
    config.text_config.num_hidden_layers = LAYER_50_HIDDEN_STATE_INDEX
    config.text_config.use_cache = False

    def factory():
        model = Qwen3VLModel(config)
        del model.language_model.norm
        return model

    files = resolve_safetensors_files(checkpoint_path)
    convrot_artifact = inspect_safetensors_convrot_int8(
        files,
        key_transform=normalize_h3_text_encoder_key,
        disable_mmap=disable_mmap,
    )
    return load_safetensors_module(
        factory,
        files,
        device=device,
        dtype=dtype,
        key_transform=normalize_h3_text_encoder_key,
        strict_dtype=False,
        convrot_artifact=convrot_artifact,
        disable_mmap=disable_mmap,
    )


class _Layer50Captured(Exception):
    pass


def _processor_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def encode_h3_presentation(processor, model, presentation: H3Presentation) -> tuple[torch.Tensor, torch.Tensor]:
    processor_args = {
        "text": [presentation.processor_text or presentation.text],
        "return_tensors": "pt",
        "padding": False,
    }
    if presentation.images:
        processor_args["images"] = [_processor_value(image) for image in presentation.images]
    if presentation.videos:
        processor_args["videos"] = [_processor_value(video) for video in presentation.videos]
        processor_args["video_metadata"] = [
            {
                "total_num_frames": int(video.shape[0]),
                "fps": 2.0,
                "duration": float(video.shape[0]) / 2.0,
                "frames_indices": list(range(video.shape[0])),
                "height": int(video.shape[1]),
                "width": int(video.shape[2]),
            }
            for video in presentation.videos
        ]
        processor_args["do_sample_frames"] = False
    processed = processor(**processor_args)
    token_tags = build_token_tags(processed)
    row_count = token_tags.shape[0]
    if row_count > MAX_TEXT_ROWS:
        raise _text_size_error(row_count, token_tags)

    layers = _language_layers(model)
    if len(layers) != LAYER_50_HIDDEN_STATE_INDEX:
        raise ValueError(f"Released MiniMax-H3 text encoder must contain exactly 50 layers, got {len(layers)}")
    captured = {}

    def capture_layer_50(module, args, output):
        del module, args
        captured["hidden"] = output[0] if isinstance(output, tuple) else output
        raise _Layer50Captured

    handle = layers[LAYER_50_HIDDEN_STATE_INDEX - 1].register_forward_hook(capture_layer_50)
    device = next(model.parameters()).device
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
        if key != "token_type_ids"
    }
    try:
        model(**model_inputs, use_cache=False)
    except _Layer50Captured:
        pass
    finally:
        handle.remove()
    if "hidden" not in captured:
        raise RuntimeError("MiniMax-H3 text encoder did not reach layer 50")

    output = SimpleNamespace(h3_layer_50_pre_norm=captured["hidden"], hidden_states=None, last_hidden_state=None)
    hidden_states = extract_layer_50_pre_norm(output, model)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"MiniMax-H3 layer-50 output must be [1,L,{TEXT_WIDTH}], got {tuple(hidden_states.shape)}")
    hidden_states = hidden_states[0]
    validate_text_rows(hidden_states, token_tags)
    return hidden_states, token_tags


def processor_fingerprint(processor) -> str:
    tokenizer = getattr(processor, "tokenizer", None)
    payload = {
        "processor_class": type(processor).__name__,
        "processor_name": getattr(processor, "name_or_path", None),
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "tokenizer_name": getattr(tokenizer, "name_or_path", None),
        "tokenizer_size": len(tokenizer) if tokenizer is not None else None,
        "commit": getattr(processor, "_commit_hash", None) or (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash"),
        "vision_tokens": [VISION_START_TOKEN_ID, VISION_END_TOKEN_ID, IMAGE_TOKEN_ID, VIDEO_TOKEN_ID],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


# Bump whenever the cached hidden-state semantics change (hidden-state convention, token-tag
# algorithm, text layout constants, or the fingerprint formats) so stale caches are rebuilt/rejected.
TEXT_CACHE_FORMAT = "minimax-h3-text-v2"


def presentation_fingerprint(
    presentation: H3Presentation,
    media_fingerprints: Mapping[Path, str],
    *,
    frame_count: int,
) -> str:
    # frame_count is part of the identity because Ref2VA reference videos are resampled to the
    # target frame count before presentation. The target crop start is deliberately excluded: it
    # only affects FL2VA visuals, and is guarded by explicit cache metadata instead.
    payload = {
        "text": presentation.text,
        "processor_text": presentation.processor_text,
        "image_shapes": [list(image.shape) for image in presentation.images],
        "video_shapes": [list(video.shape) for video in presentation.videos],
        "media": dict(sorted((str(Path(path).resolve()), value) for path, value in media_fingerprints.items())),
        "frame_count": frame_count,
        "format": "minimax-h3-non-chat-v2",
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
