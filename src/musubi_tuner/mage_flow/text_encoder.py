from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from PIL import Image
from safetensors import safe_open
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLModel

from .utils import ComponentValidationError


QWEN3_VL_4B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_VL_4B_INSTRUCT_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

QWEN3_VL_4B_INSTRUCT_CONFIG = {
    "architectures": ["Qwen3VLForConditionalGeneration"],
    "image_token_id": 151655,
    "model_type": "qwen3_vl",
    "text_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2560,
        "initializer_range": 0.02,
        "intermediate_size": 9728,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-6,
        "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        "rope_theta": 5000000,
        "tie_word_embeddings": True,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": True,
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}

MAGE_FLOW_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
MAGE_FLOW_EDIT_PROMPT_TEMPLATE = (
    "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture,"
    " objects, background), then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency with the original "
    "input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
)
EDIT_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def _qwen_config() -> Qwen3VLConfig:
    return Qwen3VLConfig.from_dict(QWEN3_VL_4B_INSTRUCT_CONFIG)


def _detect_qwen_layout(keys: Sequence[str]) -> tuple[str, str]:
    meaningful = [key for key in keys if key != "lm_head.weight"]
    if not meaningful:
        raise ComponentValidationError("Qwen3-VL checkpoint has no backbone keys")
    official = all(key.startswith(("model.language_model.", "model.visual.")) for key in meaningful)
    canonical = all(key.startswith(("language_model.", "visual.")) for key in meaningful)
    if official:
        return "official_hf", "model."
    if canonical:
        return "canonical", ""
    known_official = any(key.startswith(("model.language_model.", "model.visual.")) for key in meaningful)
    known_canonical = any(key.startswith(("language_model.", "visual.")) for key in meaningful)
    if known_official or known_canonical:
        raise ComponentValidationError("Qwen3-VL checkpoint uses a mixed key layout")
    raise ComponentValidationError(
        "Qwen3-VL checkpoint uses an unknown key layout; future Comfy-Org key mappings are not guessed"
    )


def normalize_qwen_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    _, prefix = _detect_qwen_layout(list(state_dict))
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key != "lm_head.weight"
    }


@lru_cache(maxsize=1)
def _expected_qwen_shapes() -> dict[str, tuple[int, ...]]:
    with init_empty_weights():
        model = Qwen3VLModel._from_config(_qwen_config())
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _inspect_qwen_header(path: str | Path) -> tuple[Path, str, dict[str, tuple[int, ...]]]:
    component_path = Path(path)
    if not component_path.exists():
        raise ComponentValidationError(f"text_encoder component file does not exist: {component_path}")
    if not component_path.is_file():
        raise ComponentValidationError(f"text_encoder component path must be one regular file: {component_path}")
    if component_path.suffix.lower() != ".safetensors":
        raise ComponentValidationError(f"text_encoder component must be one .safetensors file: {component_path}")

    with safe_open(component_path, framework="pt", device="cpu") as handle:
        raw_keys = list(handle.keys())
        layout, prefix = _detect_qwen_layout(raw_keys)
        canonical_keys = [key[len(prefix) :] for key in raw_keys if key != "lm_head.weight"]
        shapes = {
            key[len(prefix) :]: tuple(handle.get_slice(key).get_shape())
            for key in raw_keys
            if key != "lm_head.weight"
        }
        dtypes = {
            key[len(prefix) :]: handle.get_slice(key).get_dtype()
            for key in raw_keys
            if key != "lm_head.weight"
        }

    language_layers = sorted(
        {
            int(match.group(1))
            for key in canonical_keys
            if (match := re.match(r"^language_model\.layers\.(\d+)\.", key))
        }
    )
    visual_layers = sorted(
        {
            int(match.group(1))
            for key in canonical_keys
            if (match := re.match(r"^visual\.blocks\.(\d+)\.", key))
        }
    )
    if language_layers != list(range(36)) or visual_layers != list(range(24)):
        raise ComponentValidationError(
            "Qwen3-VL-4B layer signature mismatch: "
            f"language expected 0..35 actual={language_layers[:40]}, visual expected 0..23 actual={visual_layers[:30]}"
        )
    anchors = {
        "language_model.embed_tokens.weight": (151936, 2560),
        "language_model.layers.0.self_attn.q_proj.weight": (4096, 2560),
        "language_model.layers.0.self_attn.k_proj.weight": (1024, 2560),
        "language_model.layers.0.mlp.gate_proj.weight": (9728, 2560),
        "visual.patch_embed.proj.weight": (1024, 3, 2, 16, 16),
    }
    anchor_errors = [
        f"{key}: expected {shape}, actual {shapes.get(key)}" for key, shape in anchors.items() if shapes.get(key) != shape
    ]
    if anchor_errors:
        raise ComponentValidationError(
            f"Qwen3-VL-4B pinned dimensional signature mismatch (layout={layout}): {anchor_errors[:10]}"
        )

    expected = _expected_qwen_shapes()
    missing = sorted(set(expected) - set(shapes))
    unexpected = sorted(set(shapes) - set(expected))
    mismatched = sorted(
        f"{key}: expected {expected[key]}, actual {shapes[key]}"
        for key in set(expected) & set(shapes)
        if expected[key] != shapes[key]
    )
    invalid_dtypes = sorted(f"{key}:{dtype}" for key, dtype in dtypes.items() if dtype not in {"BF16", "F16", "F32"})
    if missing or unexpected or mismatched or invalid_dtypes:
        raise ComponentValidationError(
            f"Qwen3-VL component structural mismatch (layout={layout}); missing={missing[:10]}; "
            f"unexpected={unexpected[:10]}; shapes={mismatched[:10]}; dtypes={invalid_dtypes[:10]}"
        )
    return component_path, prefix, expected


class MageFlowTextEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, processor):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.processor = processor
        if hasattr(processor, "tokenizer"):
            processor.tokenizer.padding_side = "right"

    @property
    def device(self) -> torch.device:
        try:
            return next(self.backbone.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.backbone.parameters()).dtype
        except StopIteration:
            return torch.float32


def load_mage_flow_text_encoder(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    processor_source: str | Path = QWEN3_VL_4B_INSTRUCT_REPO_ID,
) -> MageFlowTextEncoder:
    component_path, prefix, expected = _inspect_qwen_header(path)
    with safe_open(component_path, framework="pt", device="cpu") as handle:
        state_dict = {key: handle.get_tensor(prefix + key) for key in expected}
    with init_empty_weights():
        backbone = Qwen3VLModel._from_config(_qwen_config())
    backbone.load_state_dict(state_dict, strict=True, assign=True)
    backbone.to(device=device, dtype=dtype).eval().requires_grad_(False)
    source = str(processor_source)
    processor_kwargs = {}
    if source == QWEN3_VL_4B_INSTRUCT_REPO_ID:
        processor_kwargs["revision"] = QWEN3_VL_4B_INSTRUCT_REVISION
    processor = AutoProcessor.from_pretrained(source, **processor_kwargs)
    return MageFlowTextEncoder(backbone, processor)


def _resize_long_edge(image, max_long_edge: int = 384) -> Image.Image:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image[..., :3].astype(np.uint8))
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Edit reference must be a PIL image or HWC numpy array, got {type(image).__name__}")
    image = image.convert("RGB")
    width, height = image.size
    longest = max(width, height)
    if longest <= max_long_edge:
        return image
    scale = max_long_edge / longest
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, Image.Resampling.LANCZOS)


def _edit_prompt_body(instruction: str, reference_count: int) -> str:
    prefix = "".join(f"Image {index}: {EDIT_IMAGE_PLACEHOLDER}" for index in range(1, reference_count + 1))
    return prefix + instruction


def _move_processor_inputs(inputs, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif key in {"pixel_values", "pixel_values_videos"}:
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def encode_conditioning(
    encoder: MageFlowTextEncoder,
    prompts: Sequence[str],
    *,
    references: Sequence[Sequence[Image.Image | np.ndarray]] | None = None,
    is_edit: bool,
) -> list[torch.Tensor]:
    prompt_list = list(prompts)
    if not prompt_list:
        return []
    if is_edit:
        if references is None or len(references) != len(prompt_list):
            raise ValueError("Mage-Flow-Edit requires one ordered reference list per prompt")
        reference_lists = [list(sample_references) for sample_references in references]
        for index, sample_references in enumerate(reference_lists):
            if not 1 <= len(sample_references) <= 3:
                raise ValueError(
                    f"Mage-Flow-Edit prompt {index} requires between 1 and 3 ordered references, "
                    f"got {len(sample_references)}"
                )
    else:
        if references is not None and any(references):
            raise ValueError("Mage-Flow T2I conditioning cannot contain reference images")
        reference_lists = [[] for _ in prompt_list]

    outputs = []
    drop = 64 if is_edit else 34
    template = MAGE_FLOW_EDIT_PROMPT_TEMPLATE if is_edit else MAGE_FLOW_PROMPT_TEMPLATE
    for prompt_index, (prompt, sample_references) in enumerate(zip(prompt_list, reference_lists)):
        if is_edit:
            body = _edit_prompt_body(prompt, len(sample_references))
            images = [_resize_long_edge(image) for image in sample_references]
        else:
            body = prompt
            images = None
        processor_kwargs = {
            "text": [template.format(body)],
            "padding": True,
            "truncation": True,
            "max_length": drop + 2048,
            "return_tensors": "pt",
        }
        if images is not None:
            processor_kwargs["images"] = images
        inputs = encoder.processor(**processor_kwargs)
        inputs = _move_processor_inputs(inputs, encoder.device, encoder.dtype)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("Qwen3-VL processor did not return an attention_mask")
        with torch.no_grad():
            model_outputs = encoder.backbone(
                **inputs,
                output_hidden_states=False,
                return_dict=True,
            )
        hidden = model_outputs.last_hidden_state
        valid_length = int(attention_mask[0].sum().item())
        valid = hidden[0, :valid_length]
        conditioned = valid[drop:]
        if not 1 <= conditioned.shape[0] <= 2048:
            raise ValueError(
                f"Mage-Flow prompt {prompt_index} effective length must be between 1 and 2048, "
                f"got {conditioned.shape[0]}"
            )
        if conditioned.ndim != 2 or conditioned.shape[1] != 2560:
            raise ValueError(f"Qwen3-VL final hidden state must have shape [L,2560], got {tuple(conditioned.shape)}")
        if not torch.isfinite(conditioned).all():
            raise ValueError(f"Qwen3-VL final hidden state for prompt {prompt_index} contains non-finite values")
        outputs.append(conditioned)
    return outputs


__all__ = [
    "EDIT_IMAGE_PLACEHOLDER",
    "MAGE_FLOW_EDIT_PROMPT_TEMPLATE",
    "MAGE_FLOW_PROMPT_TEMPLATE",
    "MageFlowTextEncoder",
    "encode_conditioning",
    "load_mage_flow_text_encoder",
    "normalize_qwen_state_dict",
]
