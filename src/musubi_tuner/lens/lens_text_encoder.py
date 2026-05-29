"""GPT-OSS text encoder utilities for Lens."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
from accelerate import init_empty_weights
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM

from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, load_split_weights

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CHAT_SYSTEM = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background."
CHAT_ASSISTANT_THINKING = "Need to generate one image according to the description."
DEFAULT_TXT_OFFSET = 97
DEFAULT_SELECTED_LAYERS = (5, 11, 17, 23)
DEFAULT_MAX_SEQUENCE_LENGTH = 512
DEFAULT_LENS_TEXT_REPO = "microsoft/Lens"
DEFAULT_LENS_CONFIG_SUBFOLDER = "text_encoder"
DEFAULT_LENS_TOKENIZER_SUBFOLDER = "tokenizer"
COMFY_FP4_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
COMFY_EXPERT_WEIGHT_SUFFIXES = (".mlp.experts.gate_up_proj.weight", ".mlp.experts.down_proj.weight")
COMFY_EXPERT_AUX_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_scale_2")


class LensGptOssEncoder(GptOssForCausalLM):
    def set_selected_layers(self, layer_indices: Sequence[int]) -> None:
        layers = [int(i) for i in layer_indices]
        if not layers:
            raise ValueError("layer_indices must be non-empty")
        if len(set(layers)) != len(layers):
            raise ValueError(f"layer_indices must be unique; got {layers}")
        if min(layers) < 0 or max(layers) >= len(self.model.layers):
            raise ValueError(f"layer_indices out of range; got {layers}, model has {len(self.model.layers)} layers")
        self._lens_selected_layers = layers
        self._lens_max_layer = max(layers)

    @torch.no_grad()
    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None, *args, **kwargs):
        is_lens_feature_call = (
            input_ids is not None
            and attention_mask is not None
            and hasattr(self, "_lens_selected_layers")
            and not args
            and not kwargs
        )

        target_device = self.model.embed_tokens.weight.device
        if input_ids is not None and input_ids.device != target_device:
            input_ids = input_ids.to(target_device)
        if attention_mask is not None and attention_mask.device != target_device:
            attention_mask = attention_mask.to(target_device)

        if not is_lens_feature_call:
            return super().forward(input_ids, attention_mask, *args, **kwargs)

        model = self.model
        inputs_embeds = model.embed_tokens(input_ids)
        cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        position_ids = cache_position.unsqueeze(0).expand_as(input_ids)

        mask_kwargs = {
            "config": model.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
        }

        hidden_states = inputs_embeds
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        captured: List[torch.Tensor] = [None] * len(self._lens_selected_layers)
        index_lookup = {idx: pos for pos, idx in enumerate(self._lens_selected_layers)}

        for i, decoder_layer in enumerate(model.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[model.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=None,
                use_cache=False,
            )
            if i in index_lookup:
                captured[index_lookup[i]] = hidden_states
            if i == self._lens_max_layer:
                break

        for pos, layer_idx in enumerate(self._lens_selected_layers):
            if captured[pos] is None:
                raise RuntimeError(f"Failed to capture hidden state for layer {layer_idx}")
        return captured

    def encode_layers(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
        if not hasattr(self, "_lens_selected_layers"):
            raise RuntimeError("Call set_selected_layers(...) before encode_layers().")
        return self(input_ids=input_ids, attention_mask=attention_mask)


def _strip_prefixes(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("text_encoder.", "model.text_encoder.", "module.")
    out = {}
    for key, value in sd.items():
        new_key = _strip_lens_text_prefixes(key)
        out[new_key] = value
    return out


def _strip_lens_text_prefixes(key: str) -> str:
    prefixes = ("text_encoder.", "model.text_encoder.", "module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def infer_lens_text_paths(
    text_encoder: Union[str, Path],
) -> tuple[Path, Path, Path]:
    text_encoder = Path(text_encoder)
    if text_encoder.is_file():
        weights_path = text_encoder
        root = text_encoder.parent.parent if text_encoder.parent.name == "text_encoders" else text_encoder.parent
    else:
        if text_encoder.name == "text_encoders":
            candidates = [text_encoder / "gpt_oss_20b_nvfp4.safetensors"]
        else:
            candidates = [
                text_encoder / "text_encoders" / "gpt_oss_20b_nvfp4.safetensors",
                text_encoder / "gpt_oss_20b_nvfp4.safetensors",
            ]
        weights_path = next((p for p in candidates if p.exists()), candidates[0])
        root = text_encoder.parent if text_encoder.name == "text_encoders" else text_encoder

    return weights_path, root / "text_encoder", root / "tokenizer"


def _resolve_local_or_hf_source(
    default_local_path: Path,
    hf_subfolder: str,
    description: str,
) -> tuple[Union[str, Path], Optional[str]]:
    if default_local_path.exists():
        return default_local_path, None

    logger.info(f"Lens {description} not found at {default_local_path}; falling back to {DEFAULT_LENS_TEXT_REPO}/{hf_subfolder}")
    return DEFAULT_LENS_TEXT_REPO, hf_subfolder


def _single_file_has_comfy_quant(path: Path) -> bool:
    if not path.is_file():
        return False
    with MemoryEfficientSafeOpen(str(path), disable_numpy_memmap=True) as f:
        metadata = f.metadata()
        return (
            metadata.get("lens_te_mode") == "nvfp4"
            or "tokenizer_json" in f.keys()
            or any(key.endswith(COMFY_EXPERT_AUX_SUFFIXES) for key in f.keys())
        )


def _normalize_lens_text_key(key: str) -> Optional[str]:
    key = _strip_lens_text_prefixes(key)
    if key == "tokenizer_json" or key.startswith("__") or key.startswith("lm_head.") or key.endswith(COMFY_EXPERT_AUX_SUFFIXES):
        return None
    if key.endswith(".mlp.experts.gate_up_proj.weight"):
        key = key[: -len(".weight")]
    elif key.endswith(".mlp.experts.gate_up_proj.bias"):
        key = f"{key[: -len('.bias')]}_bias"
    elif key.endswith(".mlp.experts.down_proj.weight"):
        key = key[: -len(".weight")]
    elif key.endswith(".mlp.experts.down_proj.bias"):
        key = f"{key[: -len('.bias')]}_bias"
    if not key.startswith(("model.", "lm_head.")):
        key = f"model.{key}"
    return key


def _normalize_lens_text_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        new_key = _normalize_lens_text_key(key)
        if new_key is not None:
            out[new_key] = value
    return out


def _unpack_comfy_fp4_e2m1(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.uint8:
        raise TypeError(f"Comfy NVFP4 packed weight must be uint8, got {weight.dtype}")
    high = weight >> 4
    low = weight & 0x0F
    indices = torch.stack((high, low), dim=-1).reshape(*weight.shape[:-1], weight.shape[-1] * 2).to(torch.long)
    lut = torch.tensor(COMFY_FP4_E2M1_VALUES, dtype=torch.float32, device=indices.device)
    return lut[indices]


def _from_comfy_blocked_scale(scale: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    if scale.ndim != 2:
        raise ValueError(f"Comfy NVFP4 block scale must be 2D per expert, got shape {tuple(scale.shape)}")
    padded_rows, padded_cols = scale.shape
    if padded_rows % 128 != 0 or padded_cols % 4 != 0:
        raise ValueError(f"Unexpected Comfy NVFP4 scale layout shape {tuple(scale.shape)}")
    if rows > padded_rows or cols > padded_cols:
        raise ValueError(f"Comfy NVFP4 scale shape {tuple(scale.shape)} is too small for unblocked shape {(rows, cols)}")

    n_row_blocks = padded_rows // 128
    n_col_blocks = padded_cols // 4
    blocked = scale.reshape(-1, 32, 16)
    blocked = blocked.reshape(-1, 32, 4, 4).transpose(1, 2)
    blocked = blocked.reshape(n_row_blocks, n_col_blocks, 128, 4)
    unblocked = blocked.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return unblocked[:rows, :cols]


def _dequant_comfy_nvfp4_weight(
    packed_weight: torch.Tensor,
    block_scale: torch.Tensor,
    tensor_scale: torch.Tensor,
    *,
    target_dtype: torch.dtype,
    device: Union[str, torch.device],
    transpose: bool,
) -> torch.Tensor:
    if packed_weight.ndim != 3:
        raise ValueError(f"Comfy NVFP4 expert weight must be 3D, got shape {tuple(packed_weight.shape)}")
    if block_scale.ndim != 3:
        raise ValueError(f"Comfy NVFP4 expert block scale must be 3D, got shape {tuple(block_scale.shape)}")
    if tensor_scale.ndim != 1 or tensor_scale.shape[0] != packed_weight.shape[0]:
        raise ValueError(
            f"Comfy NVFP4 tensor scale shape {tuple(tensor_scale.shape)} does not match expert count {packed_weight.shape[0]}"
        )

    expert_count, rows, packed_cols = packed_weight.shape
    cols = packed_cols * 2
    scale_cols = (cols + 15) // 16
    out = []
    device = torch.device(device)
    for expert_idx in range(expert_count):
        values = _unpack_comfy_fp4_e2m1(packed_weight[expert_idx]).to(torch.float32)
        scale = _from_comfy_blocked_scale(block_scale[expert_idx].to(torch.float32), rows, scale_cols)
        scale = scale.repeat_interleave(16, dim=1)[:, :cols]
        dequant = values * scale * tensor_scale[expert_idx].to(torch.float32)
        if transpose:
            dequant = dequant.transpose(0, 1).contiguous()
        out.append(dequant.to(device=device, dtype=target_dtype))
    return torch.stack(out, dim=0)


def _load_comfy_lens_text_state_dict(
    weights_path: Path,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool,
) -> dict[str, torch.Tensor]:
    target_dtype = dtype if dtype is not None and dtype.itemsize > 1 else torch.bfloat16
    device = torch.device(device)
    state_dict: dict[str, torch.Tensor] = {}

    with MemoryEfficientSafeOpen(str(weights_path), disable_numpy_memmap=disable_mmap) as f:
        keys = set(f.keys())
        for key in sorted(keys):
            if key == "tokenizer_json" or key.startswith("__") or key.endswith(COMFY_EXPERT_AUX_SUFFIXES):
                continue

            if key.endswith(COMFY_EXPERT_WEIGHT_SUFFIXES):
                scale_key = f"{key[: -len('.weight')]}.weight_scale"
                scale2_key = f"{key[: -len('.weight')]}.weight_scale_2"
                if scale_key not in keys or scale2_key not in keys:
                    raise KeyError(f"Missing Comfy NVFP4 scales for {key}")
                packed_weight = f.get_tensor(key, device=torch.device("cpu"), dtype=None)
                block_scale = f.get_tensor(scale_key, device=torch.device("cpu"), dtype=None)
                tensor_scale = f.get_tensor(scale2_key, device=torch.device("cpu"), dtype=None)
                new_key = _normalize_lens_text_key(key)
                if new_key is None:
                    continue
                state_dict[new_key] = _dequant_comfy_nvfp4_weight(
                    packed_weight,
                    block_scale,
                    tensor_scale,
                    target_dtype=target_dtype,
                    device=device,
                    transpose=True,
                )
                continue

            new_key = _normalize_lens_text_key(key)
            if new_key is None:
                continue
            state_dict[new_key] = f.get_tensor(key, device=device, dtype=target_dtype)

    return state_dict


def _assert_no_meta_parameters(model: torch.nn.Module) -> None:
    meta = [name for name, param in model.named_parameters() if param.is_meta]
    if meta:
        preview = ", ".join(meta[:8])
        suffix = "..." if len(meta) > 8 else ""
        raise RuntimeError(f"Lens GPT-OSS text encoder still has meta tensors after loading: {preview}{suffix}")


def _resolve_tokenizer_file(source: Union[str, Path], subfolder: Optional[str], filename: str) -> Path:
    if subfolder is None:
        path = Path(source) / filename
        if not path.exists():
            raise FileNotFoundError(f"Lens tokenizer file not found: {path}")
        return path
    return Path(hf_hub_download(str(source), filename=filename, subfolder=subfolder))


def _load_lens_tokenizer(source: Union[str, Path], subfolder: Optional[str]):
    tokenizer_kwargs = {"subfolder": subfolder} if subfolder is not None else {}
    try:
        tokenizer_obj = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
    except ValueError as e:
        if "TokenizersBackend" not in str(e):
            raise
        logger.info("Falling back to PreTrainedTokenizerFast for Lens TokenizersBackend metadata")
        config_path = _resolve_tokenizer_file(source, subfolder, "tokenizer_config.json")
        tokenizer_path = _resolve_tokenizer_file(source, subfolder, "tokenizer.json")
        chat_template_path = _resolve_tokenizer_file(source, subfolder, "chat_template.jinja")
        with config_path.open("r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        chat_template = chat_template_path.read_text(encoding="utf-8")
        tokenizer_obj = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_path),
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            pad_token=tokenizer_config.get("pad_token"),
            model_max_length=tokenizer_config.get("model_max_length", int(1e30)),
            clean_up_tokenization_spaces=tokenizer_config.get("clean_up_tokenization_spaces", False),
            chat_template=chat_template,
        )
    if tokenizer_obj.pad_token_id is None:
        tokenizer_obj.pad_token = tokenizer_obj.eos_token
    tokenizer_obj.padding_side = "right"
    return tokenizer_obj


def _load_lens_text_encoder_from_single_file(
    weights_path: Path,
    config: GptOssConfig,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool,
) -> LensGptOssEncoder:
    with init_empty_weights():
        model = LensGptOssEncoder(config)
        model.lm_head = torch.nn.Identity()
    model.set_selected_layers(DEFAULT_SELECTED_LAYERS)

    logger.info(f"Loading Lens GPT-OSS weights from {weights_path}")
    if _single_file_has_comfy_quant(weights_path):
        sd = _load_comfy_lens_text_state_dict(weights_path, dtype, device, disable_mmap)
    else:
        sd = load_split_weights(str(weights_path), device=str(device), disable_mmap=disable_mmap, dtype=dtype)
        sd = _normalize_lens_text_state_dict(sd)
    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Lens GPT-OSS text encoder: {info}")
    _assert_no_meta_parameters(model)
    model.to(device)
    if dtype is not None and dtype.itemsize > 1:
        model.to(dtype)
    model.eval()
    return model


class LensTextEmbedder(torch.nn.Module):
    def __init__(
        self,
        text_encoder: LensGptOssEncoder,
        tokenizer,
        max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
        txt_offset: int = DEFAULT_TXT_OFFSET,
    ) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.txt_offset = txt_offset

    @property
    def dtype(self) -> torch.dtype:
        return next(self.text_encoder.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.text_encoder.parameters()).device

    def _build_chat_inputs(self, prompts: Sequence[str], device: torch.device):
        rendered: List[str] = []
        for prompt in prompts:
            conversation = [
                {"role": "system", "content": CHAT_SYSTEM, "thinking": None},
                {"role": "user", "content": prompt, "thinking": None},
                {"role": "assistant", "thinking": CHAT_ASSISTANT_THINKING, "content": ""},
            ]
            text = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
            text = text.split("<|return|>")[0]
            rendered.append(text)

        encoded = self.tokenizer(
            rendered,
            padding=True,
            truncation=True,
            max_length=self.max_sequence_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

    @torch.no_grad()
    def forward(self, prompts: Union[str, Sequence[str]]) -> tuple[list[torch.Tensor], torch.Tensor]:
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        input_ids, attn_mask = self._build_chat_inputs(prompts, self.device)
        layer_outputs = self.text_encoder.encode_layers(input_ids, attn_mask)

        offset = self.txt_offset
        if input_ids.shape[1] > offset:
            features = [feat[:, offset:, :].contiguous() for feat in layer_outputs]
            mask = attn_mask[:, offset:].bool()
        else:
            zero_shape = (input_ids.shape[0], 0, layer_outputs[0].shape[-1])
            features = [layer_outputs[0].new_zeros(zero_shape) for _ in layer_outputs]
            mask = torch.zeros((input_ids.shape[0], 0), dtype=torch.bool, device=self.device)
        return features, mask


def load_lens_text_embedder(
    text_encoder: Union[str, Path],
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
) -> LensTextEmbedder:
    weights_path, config_path, tokenizer_path = infer_lens_text_paths(text_encoder)

    config_source, config_subfolder = _resolve_local_or_hf_source(
        config_path,
        DEFAULT_LENS_CONFIG_SUBFOLDER,
        "text encoder config",
    )
    tokenizer_source, tokenizer_subfolder = _resolve_local_or_hf_source(
        tokenizer_path,
        DEFAULT_LENS_TOKENIZER_SUBFOLDER,
        "tokenizer",
    )

    config_kwargs = {"subfolder": config_subfolder} if config_subfolder is not None else {}
    logger.info(f"Loading Lens GPT-OSS config from {config_source}")
    config = GptOssConfig.from_pretrained(config_source, **config_kwargs)
    if not weights_path.exists():
        raise FileNotFoundError(f"Lens text encoder weights not found: {weights_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Lens text encoder must be a Comfy safetensors file or a directory containing it: {weights_path}")
    model = _load_lens_text_encoder_from_single_file(weights_path, config, dtype, device, disable_mmap)

    tokenizer_obj = _load_lens_tokenizer(tokenizer_source, tokenizer_subfolder)
    return LensTextEmbedder(model, tokenizer_obj)
