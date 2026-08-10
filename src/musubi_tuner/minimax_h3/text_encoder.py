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
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from musubi_tuner.minimax_h3.checkpoint import resolve_safetensors_files
from musubi_tuner.minimax_h3.media import H3Record, H3Task

logger = logging.getLogger(__name__)


TEXT_WIDTH = 5120
MAX_TEXT_ROWS = 32768
LAYER_50_HIDDEN_STATE_INDEX = 50
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
IMAGE_TOKEN_ID = 151655
VIDEO_TOKEN_ID = 151656
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
# the H3 encoder is Qwen3-VL-32B, but the released repository ships its own processor and config:
# the tokenizer adds <d>, </d>, <|cutoff|>, <|lyrics_start|>, <|lyrics_end|>, <|caption_start|> and
# <|caption_end|> as special tokens (ids 151669-151675), which Qwen/Qwen3-VL-32B-Instruct splits into
# ordinary tokens instead. The released prompt format writes dialogue and lyrics as <d>[Language] ...</d>,
# so the H3 files are required, not interchangeable with the upstream Qwen ones.
H3_REPO_ID = "MiniMaxAI/MiniMax-H3"
PROCESSOR_SUBFOLDER = "processor"
TEXT_ENCODER_CONFIG_SUBFOLDER = "text_encoder"


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


def normalize_h3_text_encoder_key(key: str) -> str:
    if key.startswith("model."):
        return "language_model." + key.removeprefix("model.")
    return key


def load_h3_processor():
    # the concrete class, not AutoProcessor: the auto class resolves the processor type from the
    # repository root, where the H3 release keeps a diffusers model_index.json instead of a config
    from transformers import Qwen3VLProcessor

    return Qwen3VLProcessor.from_pretrained(H3_REPO_ID, subfolder=PROCESSOR_SUBFOLDER)


def load_h3_text_encoder(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype = torch.bfloat16,
    disable_mmap: bool = False,
    nvfp4_scaled_mm: bool = False,
    blocks_to_swap: int = 0,
    attn_mode: str | None = None,
):
    from transformers import Qwen3VLConfig, Qwen3VLModel

    device = torch.device(device)
    if blocks_to_swap > 0 and device.type != "cuda":
        raise ValueError(
            "--text_encoder_blocks_to_swap requires a CUDA device. / --text_encoder_blocks_to_swap には CUDA デバイスが必要です。"
        )

    config = Qwen3VLConfig.from_pretrained(H3_REPO_ID, subfolder=TEXT_ENCODER_CONFIG_SUBFOLDER)
    if config.text_config.hidden_size != TEXT_WIDTH:
        raise ValueError(f"MiniMax-H3 Qwen3-VL hidden size must be {TEXT_WIDTH}, got {config.text_config.hidden_size}")
    config.text_config.num_hidden_layers = LAYER_50_HIDDEN_STATE_INDEX
    config.text_config.use_cache = False
    if attn_mode is not None:
        # sdpa falls back to the O(L^2) math kernel at long context; flash_attention_2 is
        # recommended for presentations beyond ~3k rows
        config._attn_implementation = attn_mode

    from accelerate import init_empty_weights

    from musubi_tuner.modules.comfy_quant_utils import (
        FORMAT_CONVROT_INT8,
        FORMAT_INT8_TENSORWISE,
        FORMAT_NVFP4,
        detect_comfy_quant_formats,
    )
    from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8Quantizer, apply_convrot_int8_monkey_patch
    from musubi_tuner.modules.nvfp4_utils import NvFp4Quantizer, apply_nvfp4_monkey_patch
    from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
    from musubi_tuner.utils.safetensors_utils import load_safetensors

    with init_empty_weights():
        model = Qwen3VLModel(config)
        # the layer-50 pre-norm convention: the model is truncated to 50 layers and the final
        # norm is replaced by Identity, so last_hidden_state IS the layer-50 pre-norm state
        model.language_model.norm = nn.Identity()

    # with block swap the state dict stays on the CPU and the streamed layers never fully
    # reside on the device; otherwise loading straight to the target device avoids a
    # resident full-model CPU copy
    streaming = blocks_to_swap > 0
    load_device = torch.device("cpu") if streaming else device

    def _load_with_quantizer(quantizer):
        # the same streaming loader as the transformer; the file dictates the quantized layers
        sd = load_safetensors_with_lora_and_fp8(
            model_files=[str(checkpoint_path)],  # unexpanded: the loader expands split shards itself
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=False,
            calc_device=device,
            move_to_device=not streaming,
            disable_numpy_memmap=disable_mmap,
            quantizer=quantizer,
        )
        return {normalize_h3_text_encoder_key(key): value for key, value in sd.items()}

    files = resolve_safetensors_files(checkpoint_path)
    formats = detect_comfy_quant_formats(files, disable_numpy_memmap=disable_mmap)
    if formats == {FORMAT_CONVROT_INT8}:
        # pre-quantized ConvRot INT8 artifact; an empty target list disables dynamic quantization
        quantizer = ConvRotInt8Quantizer(target_layer_keys=[])
        sd = _load_with_quantizer(quantizer)
        groupsize_map = {normalize_h3_text_encoder_key(key): value for key, value in quantizer.module_groupsizes.items()}
        apply_convrot_int8_monkey_patch(model, sd, groupsize_map=groupsize_map)
        # int8 tensors cannot require grad, and load_state_dict(assign=True) re-wraps incoming
        # tensors with the meta params' requires_grad; the text encoder is frozen anyway
        model.requires_grad_(False)
    elif FORMAT_NVFP4 in formats and formats <= {FORMAT_NVFP4, FORMAT_INT8_TENSORWISE}:
        # pre-quantized ComfyUI NVFP4 (+AWQ) artifact with an INT8 per-row embedding
        quantizer = NvFp4Quantizer()
        sd = _load_with_quantizer(quantizer)
        nvfp4_module_shapes = {normalize_h3_text_encoder_key(key): value for key, value in quantizer.nvfp4_module_shapes.items()}
        int8_embedding_modules = [normalize_h3_text_encoder_key(key) for key in quantizer.int8_embedding_modules]
        apply_nvfp4_monkey_patch(
            model, sd, nvfp4_module_shapes, int8_embedding_modules, use_scaled_mm=nvfp4_scaled_mm, embedding_dtype=dtype
        )
        model.requires_grad_(False)
    elif formats:
        raise ValueError(
            f"Unsupported ComfyUI quantization format combination in text encoder checkpoint: {sorted(formats)}."
            " Supported: ConvRot INT8, or NVFP4 with an INT8 embedding."
            f" / テキストエンコーダの量子化形式の組み合わせ {sorted(formats)} はサポートされていません。"
            "ConvRot INT8、または NVFP4(+INT8 embedding) のみ対応しています。"
        )
    else:
        sd = {}
        for file in files:
            shard = load_safetensors(str(file), device=load_device, disable_mmap=True, disable_numpy_memmap=disable_mmap)
            sd.update({normalize_h3_text_encoder_key(key): value for key, value in shard.items()})

    # quantization scale tensors keep their own dtypes (fp32 row scales, fp8 block scales)
    _KEEP_DTYPE_SUFFIXES = (".scale_weight", ".nvfp4_scale", ".nvfp4_block_scale")
    for key in sd.keys():
        if sd[key].is_floating_point() and not key.endswith(_KEEP_DTYPE_SUFFIXES):
            sd[key] = sd[key].to(dtype)
    model.load_state_dict(sd, strict=True, assign=True)
    if streaming:
        # the offloader requires frozen weights (the text encoder is frozen by contract anyway;
        # the quantized branches have already dropped requires_grad, BF16 has not)
        model.requires_grad_(False)
        _enable_h3_text_encoder_streaming(model, device, blocks_to_swap)
    else:
        model.to(device)
    model.eval()
    return model


# quantization tensors that the ConvRot INT8 / NVFP4 monkey patches hang off the patched Linear
# modules; streamed together with the weight. An explicit allowlist keeps unrelated (tiny)
# buffers resident instead of silently streaming whatever a future patch registers.
_TE_STREAM_QUANT_BUFFER_NAMES = ("scale_weight", "nvfp4_block_scale", "nvfp4_scale", "pre_quant_scale")


def _te_swap_tensor_selector(block: nn.Module) -> list[tuple[nn.Module, str]]:
    jobs: list[tuple[nn.Module, str]] = []
    for _, module in block.named_modules():
        if isinstance(module, nn.Linear) and module.weight is not None:
            jobs.append((module, "weight"))
            for name in _TE_STREAM_QUANT_BUFFER_NAMES:
                if name in module._buffers:
                    jobs.append((module, name))
    return jobs


def _enable_h3_text_encoder_streaming(model, device: torch.device, blocks_to_swap: int) -> None:
    """Stream ``blocks_to_swap`` of the 50 Qwen3-VL decoder layers from CPU masters (H2D only).

    The text encoder is frozen and forward-only, so ``LoRAStreamOffloader`` applies as-is;
    the transformers forward loop does not know about the offloader, so it is driven from
    forward hooks. Everything outside the decoder layers (embeddings, vision tower, norms)
    stays resident on the device. Masters are pageable (staged copier): no giant pinned
    allocation, which matters on Windows.
    """
    from musubi_tuner.modules.custom_offloading_utils import LoRAStreamOffloader, attach_forward_streaming_hooks

    layers = list(_language_layers(model))
    num_layers = len(layers)
    if blocks_to_swap > num_layers:
        raise ValueError(
            f"--text_encoder_blocks_to_swap must be at most the number of text encoder layers ({num_layers}), got {blocks_to_swap}"
        )

    for name, child in model.named_children():
        if name != "language_model":
            child.to(device)
    for name, child in model.language_model.named_children():
        if name != "layers":
            child.to(device)

    offloader = LoRAStreamOffloader(
        "mmh3-te",
        layers,
        num_layers,
        blocks_to_swap,
        supports_backward=False,
        device=device,
        ring_size=2,
        use_pinned_memory=False,
        swap_tensor_selector=_te_swap_tensor_selector,
    )
    # the handles are kept on the model so the hooks live exactly as long as the model does
    model._h3_te_stream_hook_handles = attach_forward_streaming_hooks(offloader, layers)
    offloader.prepare_block_devices_before_forward(layers)
    model.h3_te_offloader = offloader


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
    norm = getattr(model.language_model, "norm", None)
    if not isinstance(norm, nn.Identity):
        raise ValueError("MiniMax-H3 text encoder must have an Identity final norm (load via load_h3_text_encoder)")

    # with block swap the layer weights live on CPU masters, so the input device is taken
    # from the always-resident embedding
    device = model.language_model.embed_tokens.weight.device
    model_inputs = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in processed.items()
        if key != "token_type_ids"
    }
    # the model is truncated to 50 layers and the final norm is Identity, so
    # last_hidden_state is the layer-50 pre-norm hidden state
    hidden_states = model(**model_inputs, use_cache=False).last_hidden_state
    if hidden_states is None:
        raise RuntimeError("MiniMax-H3 text encoder returned no last_hidden_state")
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError(f"MiniMax-H3 layer-50 output must be [1,L,{TEXT_WIDTH}], got {tuple(hidden_states.shape)}")
    hidden_states = hidden_states[0]
    validate_text_rows(hidden_states, token_tags)
    return hidden_states, token_tags


# The guidance-loss uncond cache: one text-only probe embedding (layer-50 hidden rows +
# token tags), shared between the cache script that writes it and the trainer that reads
# it. The format id matches the uncond-probe screening harness so screened probes load
# directly. Bump on any semantic change so stale caches are rejected.
UNCOND_CACHE_FORMAT = "h3-uncond-probe-v1"


def _validate_uncond_cache_tensors(hidden_states: torch.Tensor, token_tags: torch.Tensor, label: str) -> None:
    if hidden_states.ndim != 2 or hidden_states.shape[0] < 1:
        raise ValueError(f"MiniMax-H3 uncond cache {label} hidden states must be [L>=1,width], got {tuple(hidden_states.shape)}")
    if hidden_states.shape[0] > MAX_TEXT_ROWS:
        raise _text_size_error(hidden_states.shape[0], token_tags)
    if token_tags.dtype != torch.int64 or token_tags.shape != (hidden_states.shape[0],):
        raise ValueError(f"MiniMax-H3 uncond cache {label} token tags must be int64 [L]")
    if not torch.all((token_tags == 0) | (token_tags == 1)):
        raise ValueError(f"MiniMax-H3 uncond cache {label} token tags may contain only 0 and 1")


def save_h3_uncond_cache(
    path: str | Path,
    hidden_states: torch.Tensor,
    token_tags: torch.Tensor,
    *,
    metadata: Mapping[str, str] | None = None,
) -> None:
    from safetensors.torch import save_file

    _validate_uncond_cache_tensors(hidden_states, token_tags, str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"hidden_states": hidden_states.contiguous(), "token_tags": token_tags.contiguous()},
        str(path),
        metadata={"cache_format": UNCOND_CACHE_FORMAT, **dict(metadata or {})},
    )


def load_h3_uncond_cache(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, str]]:
    from safetensors import safe_open

    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        cached_format = metadata.get("cache_format")
        if cached_format != UNCOND_CACHE_FORMAT:
            raise ValueError(f"MiniMax-H3 uncond cache format must be {UNCOND_CACHE_FORMAT!r}, got {cached_format!r}: {path}")
        if set(handle.keys()) != {"hidden_states", "token_tags"}:
            raise ValueError(f"MiniMax-H3 uncond cache has an invalid tensor-key set: {path}")
        hidden_states = handle.get_tensor("hidden_states")
        token_tags = handle.get_tensor("token_tags")
    _validate_uncond_cache_tensors(hidden_states, token_tags, str(path))
    return hidden_states, token_tags, metadata


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

# The teacher-matching initial scope fixes the teacher conditioning to both endpoints: full
# training videos always provide first and last frames, and the FL2VA base is most
# in-distribution with both anchors. The value is validated through this seam so single-sided
# or anchored teachers can slot in later without changing the cache or trainer interfaces.
TEACHER_CONDITIONS_FIRST_LAST = "first,last"


def normalize_teacher_conditions(value: str) -> str:
    parts = [part.strip() for part in str(value).split(",")]
    if parts != ["first", "last"]:
        raise ValueError(
            f"MiniMax-H3 teacher matching supports only teacher conditions '{TEACHER_CONDITIONS_FIRST_LAST}', got {value!r}"
        )
    return TEACHER_CONDITIONS_FIRST_LAST


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
