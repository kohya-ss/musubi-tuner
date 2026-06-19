from __future__ import annotations

import inspect
import logging
import os
from typing import Optional

import torch
from accelerate import init_empty_weights
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, Qwen3VLConfig
from transformers.masking_utils import create_causal_mask

from musubi_tuner.ideogram4.caption_verifier import CaptionVerifier
from musubi_tuner.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)
from musubi_tuner.ideogram4.ideogram4_autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from musubi_tuner.ideogram4.ideogram4_model import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.ideogram4_scheduler import get_schedule_for_resolution, make_step_intervals
from musubi_tuner.ideogram4.ideogram4_quantized_loading import (
    COMFY_FP8_MARKER_SUFFIX,
    FP8_SCALE_SUFFIX,
    FP8_WEIGHT_DTYPE,
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    load_bnb4bit_state_dict,
    require_fp8_dtype,
    swap_linears_to_bnb4bit,
)
from musubi_tuner.ideogram4.latent_norm import get_latent_norm
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils import safetensors_utils
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8

logger = logging.getLogger(__name__)

QWEN3_VL_8B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Vendored copy of the Qwen3-VL-8B-Instruct config.json so the text encoder is built without
# fetching the config from the Hugging Face Hub. Only the tokenizer is still pulled by repo id
# (it is small and HF-cached after first use); the config and model architecture are now fully
# self-owned. Qwen3-VL is natively supported by transformers (no auto_map / remote code), so
# Qwen3VLConfig.from_dict reproduces AutoConfig.from_pretrained exactly (only the cosmetic
# _name_or_path differs). Mirror upstream config.json if Qwen ever revises it.
QWEN3_VL_8B_INSTRUCT_CONFIG = {
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
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 12288,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 32,
        "num_hidden_layers": 36,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default",
        },
        "rope_theta": 5000000,
        "use_cache": True,
        "vocab_size": 151936,
    },
    "tie_word_embeddings": False,
    "video_token_id": 151656,
    "vision_config": {
        "deepstack_visual_indexes": [8, 16, 24],
        "depth": 27,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1152,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4304,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 4096,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
    },
    "vision_end_token_id": 151653,
    "vision_start_token_id": 151652,
}
IDEOGRAM4_MAX_TEXT_TOKENS = 2048
IDEOGRAM4_PATCH_SIZE = 2
IDEOGRAM4_AE_SCALE_FACTOR = 8
IDEOGRAM4_IMAGE_PATCH = IDEOGRAM4_PATCH_SIZE * IDEOGRAM4_AE_SCALE_FACTOR
IDEOGRAM4_COND_MODEL_TYPE = "ideogram4_cond"
IDEOGRAM4_UNCOND_MODEL_TYPE = "ideogram4_uncond"

# FP8 optimization scope for the DiT: the per-block attention and feed-forward Linear weights
# (qkv / o / w1 / w2 / w3). Norm and adaLN modulation weights stay in the compute dtype.
IDEOGRAM4_FP8_OPTIMIZATION_TARGET_KEYS = ["layers."]
IDEOGRAM4_FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "adaln"]


def validate_local_safetensors(path: str, expected_model_type: Optional[str] = None) -> dict[str, str]:
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"Ideogram 4 component file not found: {path}")
    with safetensors_utils.MemoryEfficientSafeOpen(path) as f:
        metadata = f.metadata()
    if expected_model_type is not None and metadata.get("model_type") != expected_model_type:
        raise ValueError(f"{path} has safetensors model_type={metadata.get('model_type')!r}, expected {expected_model_type!r}")
    return metadata


def _load_state_dict(path: str, device: str | torch.device = "cpu", disable_mmap: bool = False) -> dict[str, torch.Tensor]:
    return safetensors_utils.load_safetensors(path, device=device, disable_mmap=disable_mmap, dtype=None)


def _reshape_prequant_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Reshape a pre-quantized FP8 weight scale to a layout that broadcasts against ``[out, in]``.

    Handles both layouts seen in the wild: per-channel ``[out]`` (official) -> ``[out, 1]`` and
    per-tensor scalar ``[]`` (ComfyUI) -> ``[1]``. Any other shape is left as-is.
    """
    if scale.ndim == 1:
        return scale.unsqueeze(1)  # per-channel [out] -> [out, 1]
    if scale.ndim == 0:
        return scale.reshape(1)  # per-tensor [] -> [1]
    return scale


def _make_ideogram4_comfy_fp8_split_hook(compute_dtype: torch.dtype):
    """Build a split hook that normalizes pre-quantized FP8 keys to Musubi's monkey-patch layout.

    - ``<name>.weight_scale`` -> ``<name>.scale_weight``, reshaped to broadcast against ``[out, in]``
      and cast to ``compute_dtype`` (dequantization is done in the compute dtype to match the
      official pipeline and the bf16 activations feeding each Linear).
    - ``<name>.comfy_quant`` marker keys are dropped.
    - every other key passes through unchanged.
    """

    def split_hook(key: str, value: Optional[torch.Tensor]):
        if key.endswith(COMFY_FP8_MARKER_SUFFIX):
            return [], None  # drop the Comfy marker key
        if key.endswith(FP8_SCALE_SUFFIX):
            new_key = key[: -len(FP8_SCALE_SUFFIX)] + ".scale_weight"
            if value is None:
                return [new_key], None
            return [new_key], [_reshape_prequant_fp8_scale(value).to(compute_dtype)]
        return None, None  # direct mapping

    return split_hook


def _prepare_qwen3_vl_fp8_state_dict(
    state_dict: dict[str, torch.Tensor], *, device: torch.device, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Normalize a pre-quantized FP8 Qwen3-VL state dict to Musubi's monkey-patch layout.

    Mirrors the FP8 split hook used for the DiT, but operates on an already-loaded (and
    key-normalized / vision-tower-stripped) state dict instead of a file:

    - FP8 weights stay FP8 (moved to ``device``);
    - ``<name>.weight_scale`` -> ``<name>.scale_weight``, reshaped to broadcast and cast to
      ``dtype`` so :func:`fp8_linear_forward_patch` dequantizes in the same compute dtype as the
      bf16 activations (it does not cast its input);
    - ``<name>.comfy_quant`` markers are dropped;
    - every other floating tensor is cast to ``dtype`` (matching the non-FP8 params/activations);
    - non-floating tensors keep their dtype.
    """
    fp8_dtype = require_fp8_dtype()
    prepared: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(COMFY_FP8_MARKER_SUFFIX):
            continue
        if key.endswith(FP8_SCALE_SUFFIX):
            new_key = key[: -len(FP8_SCALE_SUFFIX)] + ".scale_weight"
            prepared[new_key] = _reshape_prequant_fp8_scale(value).to(device=device, dtype=dtype)
        elif value.dtype == fp8_dtype:
            prepared[key] = value.to(device=device)
        elif value.is_floating_point():
            prepared[key] = value.to(device=device, dtype=dtype)
        else:
            prepared[key] = value.to(device=device)
    return prepared


def load_ideogram4_transformer(
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    expected_model_type: Optional[str],
    disable_mmap: bool = False,
    config: Optional[Ideogram4Config] = None,
    attn_mode: str = "torch",
    split_attn: bool = False,
) -> Ideogram4Transformer:
    validate_local_safetensors(path, expected_model_type)
    device = torch.device(device)

    # Cheap header peek (no tensor data loaded) to detect the checkpoint format.
    with safetensors_utils.MemoryEfficientSafeOpen(path) as f:
        keys = f.keys()
    is_bnb4bit = is_bnb4bit_state_dict(keys)
    is_prequant_fp8 = any(k.endswith(FP8_SCALE_SUFFIX) for k in keys)

    if is_bnb4bit:
        # bnb 4-bit checkpoints stay on the dedicated quantized loader for now.
        if device.type != "cuda":
            raise ValueError(f"bnb 4-bit weights require a CUDA device, got device={device}")
        state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
        model = Ideogram4Transformer(config or Ideogram4Config(), attn_mode=attn_mode, split_attn=split_attn)
        swap_linears_to_bnb4bit(model, compute_dtype=dtype)
        load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
        model.eval()
        return model

    if is_prequant_fp8:
        # Pre-quantized (ComfyUI) FP8 weights -> Musubi's shared monkey-patch FP8 path (as used by
        # Z-Image etc.). The weights are kept as FP8; a local split hook normalizes the Comfy scale
        # layout, and apply_fp8_monkey_patch installs the dequantizing forward on each Linear.
        require_fp8_dtype()
        with init_empty_weights():
            model = Ideogram4Transformer(config or Ideogram4Config(), attn_mode=attn_mode, split_attn=split_attn)
        hooks = safetensors_utils.WeightTransformHooks(split_hook=_make_ideogram4_comfy_fp8_split_hook(dtype))
        sd = load_safetensors_with_lora_and_fp8(
            model_files=path,
            lora_weights_list=None,
            lora_multipliers=None,
            fp8_optimization=True,
            calc_device=device,
            move_to_device=True,
            dit_weight_dtype=None,
            target_keys=IDEOGRAM4_FP8_OPTIMIZATION_TARGET_KEYS,
            exclude_keys=IDEOGRAM4_FP8_OPTIMIZATION_EXCLUDE_KEYS,
            disable_numpy_memmap=disable_mmap,
            weight_transform_hooks=hooks,
            allow_prequantized_fp8=True,
        )
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        model.load_state_dict(sd, strict=True, assign=True)
        model.to(device)
        model.eval()
        return model

    # plain (bf16/fp16/fp32) weights
    state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
    with init_empty_weights():
        model = Ideogram4Transformer(config or Ideogram4Config(), attn_mode=attn_mode, split_attn=split_attn)
    model.load_state_dict(state_dict, strict=True, assign=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


def load_ideogram4_autoencoder(
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    disable_mmap: bool = False,
) -> AutoEncoder:
    validate_local_safetensors(path)
    state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
    # The VAE ships in two key layouts: this repo's native layout and the diffusers layout
    # (down_blocks/up_blocks/conv_norm_out). Detect the format up front so a wrong checkpoint
    # fails the strict load below with a clear missing/unexpected-keys error, instead of being
    # masked by a native-then-diffusers try/except fallback.
    if _is_diffusers_vae_state_dict(state_dict):
        state_dict = convert_diffusers_state_dict(state_dict)
    ae = AutoEncoder(AutoEncoderParams())
    ae.load_state_dict(state_dict, strict=True)
    ae.to(device=torch.device(device), dtype=dtype)
    ae.eval()
    return ae


def _is_diffusers_vae_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    """True if the VAE checkpoint uses the diffusers key layout (vs this repo's native layout)."""
    return any(".down_blocks." in k or ".up_blocks." in k or k.endswith("conv_norm_out.weight") for k in state_dict)


def load_ideogram4_tokenizer():
    # Qwen3-VL ships a native (Qwen2TokenizerFast) tokenizer with no remote code, so
    # trust_remote_code is unnecessary. The tokenizer is the only artifact still fetched by
    # repo id; it is small and retained in the Hugging Face local cache after first use.
    return AutoTokenizer.from_pretrained(QWEN3_VL_8B_INSTRUCT_REPO_ID)


def _has_meta_tensors(model: torch.nn.Module) -> bool:
    return any(param.is_meta for param in model.parameters()) or any(buffer.is_meta for buffer in model.buffers())


def _materialize_meta_tensors(model: torch.nn.Module) -> None:
    if _has_meta_tensors(model):
        model.to_empty(device=torch.device("cpu"))


def _normalize_qwen3_vl_state_dict_for_automodel(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        if key in ("lm_head.weight", "model.lm_head.weight", "language_model.lm_head.weight"):
            continue
        if key.startswith("model.visual."):
            key = "visual." + key[len("model.visual.") :]
        elif key.startswith("model.language_model."):
            key = key[len("model.") :]
        elif key.startswith("model."):
            key = "language_model." + key[len("model.") :]
        normalized[key] = tensor
    return normalized


def _raise_on_text_encoder_load_mismatch(missing: list[str], unexpected: list[str]) -> None:
    if missing or unexpected:
        raise RuntimeError(
            "Qwen3-VL text encoder checkpoint did not match AutoModel after key normalization: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )


def load_ideogram4_text_encoder(
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    disable_mmap: bool = False,
):
    validate_local_safetensors(path)
    # Build the config from the vendored dict instead of AutoConfig.from_pretrained, so no config
    # is fetched from the Hub. Qwen3-VL is natively supported (no auto_map / remote code), hence
    # no trust_remote_code here or on AutoModel.from_config below.
    config = Qwen3VLConfig.from_dict(QWEN3_VL_8B_INSTRUCT_CONFIG)
    state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
    state_dict = _normalize_qwen3_vl_state_dict_for_automodel(state_dict)
    # Ideogram 4 conditioning only consumes the language model (see _get_qwen3_vl_embeddings), so
    # drop the Qwen3-VL vision tower weights and remove the submodule to avoid materializing it.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("visual.")}
    device = torch.device(device)
    with init_empty_weights():
        model = AutoModel.from_config(config)
    if hasattr(model, "visual"):
        del model.visual
    _materialize_meta_tensors(model)
    if is_fp8_state_dict(state_dict):
        # Pre-quantized FP8 Qwen3-VL (official or ComfyUI) -> Musubi's shared monkey-patch FP8 path,
        # the same mechanism the DiT uses. The dedicated Fp8Linear class is no longer needed here.
        state_dict = _prepare_qwen3_vl_fp8_state_dict(state_dict, device=device, dtype=dtype)
        apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        _raise_on_text_encoder_load_mismatch(missing, unexpected)
        model.to(device)
    else:
        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        _raise_on_text_encoder_load_mismatch(missing, unexpected)
        model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _create_qwen_causal_mask(language_model, inputs_embeds, attention_mask, text_position_ids, cache_position):
    signature = inspect.signature(create_causal_mask)
    kwargs = {
        "config": language_model.config,
        "attention_mask": attention_mask,
        "past_key_values": None,
        "position_ids": text_position_ids,
    }
    if "input_embeds" in signature.parameters:
        kwargs["input_embeds"] = inputs_embeds
    else:
        kwargs["inputs_embeds"] = inputs_embeds
    if "cache_position" in signature.parameters:
        kwargs["cache_position"] = cache_position
    return create_causal_mask(**kwargs)


def tokenize_prompt(tokenizer, prompt: str, max_text_tokens: int = IDEOGRAM4_MAX_TEXT_TOKENS) -> tuple[torch.Tensor, int]:
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > max_text_tokens:
        raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={max_text_tokens}")
    return token_ids, num_text_tokens


def _get_qwen3_vl_embeddings(text_encoder, token_ids: torch.Tensor, attention_mask: torch.Tensor, pos_2d: torch.Tensor):
    # NOTE: This manually drives Qwen3-VL's text decoder stack (embed_tokens -> rotary_emb -> layers)
    # instead of calling text_encoder.forward(), so it depends on transformers internals: the submodule
    # layout (language_model.embed_tokens / .rotary_emb / .layers), the Qwen3VLDecoderLayer call
    # signature, and transformers.masking_utils.create_causal_mask. This is validated against the pinned
    # transformers==4.57.6 (see pyproject.toml); a transformers upgrade may require updating this path.
    language_model = text_encoder.language_model
    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    cache_position = torch.arange(token_ids.shape[1], dtype=torch.long, device=token_ids.device)
    causal_mask = _create_qwen_causal_mask(
        language_model,
        inputs_embeds,
        attention_mask,
        text_position_ids,
        cache_position,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        if layer_idx in tap_set:
            captured[layer_idx] = hidden_states

    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]


@torch.no_grad()
def encode_prompt_to_features(tokenizer, text_encoder, prompt: str, device: torch.device) -> torch.Tensor:
    token_ids, num_text_tokens = tokenize_prompt(tokenizer, prompt)
    token_ids = token_ids.unsqueeze(0).to(device)
    attention_mask = torch.ones(1, num_text_tokens, dtype=torch.long, device=device)
    pos_2d = torch.arange(num_text_tokens, dtype=torch.long, device=device).unsqueeze(0)
    selected = _get_qwen3_vl_embeddings(text_encoder, token_ids, attention_mask, pos_2d)
    stacked = torch.stack(selected, dim=0)  # taps, B, L, H
    stacked = torch.permute(stacked, (1, 2, 3, 0)).reshape(1, num_text_tokens, -1)
    return stacked[0].to(torch.float32)


def build_sequence_inputs_from_features(
    text_features: list[torch.Tensor],
    height: int,
    width: int,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor | int]:
    if height % IDEOGRAM4_IMAGE_PATCH != 0 or width % IDEOGRAM4_IMAGE_PATCH != 0:
        raise ValueError(f"height/width must be divisible by {IDEOGRAM4_IMAGE_PATCH}")
    grid_h = height // IDEOGRAM4_IMAGE_PATCH
    grid_w = width // IDEOGRAM4_IMAGE_PATCH
    num_image_tokens = grid_h * grid_w
    max_text_tokens = max(int(x.shape[0]) for x in text_features)
    total_seq_len = max_text_tokens + num_image_tokens
    batch_size = len(text_features)
    feature_dim = int(text_features[0].shape[-1])

    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    # Sequence layout is [image][text][right-padding]: image tokens lead (Qwen-Image / Z-Image
    # convention) so the shared attention() can build its mask from the text-only attention_mask.
    llm_features = torch.zeros(batch_size, total_seq_len, feature_dim, dtype=text_features[0].dtype)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_text_tokens, dtype=torch.long)
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    position_ids[:, :num_image_tokens] = image_pos
    indicator[:, :num_image_tokens] = OUTPUT_IMAGE_INDICATOR

    for b, features in enumerate(text_features):
        num_text = int(features.shape[0])
        text_start = num_image_tokens
        text_pos = torch.arange(num_text)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        llm_features[b, text_start : text_start + num_text] = features.cpu()
        position_ids[b, text_start : text_start + num_text] = text_pos_3d
        indicator[b, text_start : text_start + num_text] = LLM_TOKEN_INDICATOR
        attention_mask[b, :num_text] = 1

    return {
        "llm_features": llm_features.to(device),
        "position_ids": position_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "indicator": indicator.to(device),
        "num_image_tokens": num_image_tokens,
        "grid_h": grid_h,
        "grid_w": grid_w,
        "max_text_tokens": max_text_tokens,
    }


def build_negative_image_inputs(
    sequence_inputs: dict[str, torch.Tensor | int],
    *,
    feature_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    num_image_tokens = int(sequence_inputs["num_image_tokens"])
    # Image tokens lead the sequence, so the image-only negative is just the leading slice.
    position_ids = sequence_inputs["position_ids"][:, :num_image_tokens]
    indicator = sequence_inputs["indicator"][:, :num_image_tokens]
    batch_size = int(position_ids.shape[0])
    # No text tokens: an empty text mask makes every (image) token valid.
    attention_mask = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
    llm_features = torch.zeros(batch_size, num_image_tokens, feature_dim, dtype=dtype, device=device)
    return {
        "llm_features": llm_features,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "indicator": indicator,
    }


def patchify_vae_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4:
        raise ValueError(f"expected VAE latents as B,C,H,W, got {tuple(latents.shape)}")
    b, c, h, w = latents.shape
    if h % IDEOGRAM4_PATCH_SIZE != 0 or w % IDEOGRAM4_PATCH_SIZE != 0:
        raise ValueError(f"latent height/width must be divisible by {IDEOGRAM4_PATCH_SIZE}: {tuple(latents.shape)}")
    p = IDEOGRAM4_PATCH_SIZE
    latents = latents.reshape(b, c, h // p, p, w // p, p)
    latents = latents.permute(0, 3, 5, 1, 2, 4).contiguous()
    return latents.reshape(b, c * p * p, h // p, w // p)


def unpatchify_vae_latents(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    if tokens.ndim == 4:
        b, c, h, w = tokens.shape
        if h != grid_h or w != grid_w:
            raise ValueError(f"token grid mismatch: got {(h, w)}, expected {(grid_h, grid_w)}")
        tokens = tokens.permute(0, 2, 3, 1).reshape(b, grid_h * grid_w, c)
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens as B,L,C or B,C,H,W, got {tuple(tokens.shape)}")
    b = tokens.shape[0]
    p = IDEOGRAM4_PATCH_SIZE
    ae_channels = tokens.shape[-1] // (p * p)
    z = tokens.view(b, grid_h, grid_w, p, p, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    return z.view(b, ae_channels, grid_h * p, grid_w * p)


def flatten_token_grid(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 4:
        raise ValueError(f"expected token grid as B,C,H,W, got {tuple(tokens.shape)}")
    return tokens.permute(0, 2, 3, 1).reshape(tokens.shape[0], tokens.shape[2] * tokens.shape[3], tokens.shape[1])


def unflatten_token_grid(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"expected tokens as B,L,C, got {tuple(tokens.shape)}")
    return tokens.reshape(tokens.shape[0], grid_h, grid_w, tokens.shape[2]).permute(0, 3, 1, 2).contiguous()


def normalize_token_grid(token_grid: torch.Tensor) -> torch.Tensor:
    shift, scale = get_latent_norm(device=token_grid.device, dtype=token_grid.dtype)
    shift = shift.view(1, -1, 1, 1)
    scale = scale.view(1, -1, 1, 1)
    return (token_grid - shift) / scale


def denormalize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    shift, scale = get_latent_norm(device=tokens.device, dtype=tokens.dtype)
    if tokens.ndim == 4:
        shift = shift.view(1, -1, 1, 1)
        scale = scale.view(1, -1, 1, 1)
    return tokens * scale + shift


def encode_pixels_to_vae_latents(autoencoder: AutoEncoder, pixels: torch.Tensor) -> torch.Tensor:
    pixels = pixels * 2.0 - 1.0
    return autoencoder.encode(pixels)


def decode_tokens_to_images(autoencoder: AutoEncoder, tokens: torch.Tensor, grid_h: int, grid_w: int) -> list[Image.Image]:
    tokens = denormalize_tokens(tokens)
    z = unpatchify_vae_latents(tokens, grid_h, grid_w).to(autoencoder.dtype)
    decoded = autoencoder.decoder(z)
    decoded = decoded.float().clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]


def _iter_denoising_steps(params, schedule, step_intervals: torch.Tensor):
    for step in range(params.num_steps):
        official_index = params.num_steps - 1 - step
        t_val = float(schedule(step_intervals[official_index + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[official_index].unsqueeze(0)).item())
        guidance = params.guidance_schedule[params.num_steps - 1 - step]
        yield t_val, s_val, guidance


@torch.no_grad()
def generate_images(
    *,
    conditional_transformer: Ideogram4Transformer,
    autoencoder: AutoEncoder,
    text_features: list[torch.Tensor],
    height: int,
    width: int,
    sampler_preset: str,
    device: torch.device,
    unconditional_transformer: Optional[Ideogram4Transformer] = None,
    unconditional_text_features: Optional[list[torch.Tensor]] = None,
    seed: Optional[int] = None,
    show_progress: bool = True,
    initial_sigma: Optional[float] = None,
) -> list[Image.Image]:
    if sampler_preset not in PRESETS:
        raise ValueError(f"unknown Ideogram 4 sampler preset {sampler_preset!r}; choices: {sorted(PRESETS)}")
    params = PRESETS[sampler_preset]
    schedule = get_schedule_for_resolution((height, width), known_mean=params.mu, std=params.std)
    step_intervals = make_step_intervals(params.num_steps).to(device)

    inputs = build_sequence_inputs_from_features(text_features, height, width, device=device)
    num_image_tokens = int(inputs["num_image_tokens"])
    grid_h = int(inputs["grid_h"])
    grid_w = int(inputs["grid_w"])
    max_text_tokens = int(inputs["max_text_tokens"])
    batch_size = len(text_features)
    latent_dim = conditional_transformer.config.in_channels

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    z = torch.randn(batch_size, num_image_tokens, latent_dim, dtype=torch.float32, device=device, generator=generator)
    text_z_padding = torch.zeros(batch_size, max_text_tokens, latent_dim, dtype=torch.float32, device=device)

    cond_features = inputs["llm_features"].to(device=device, dtype=torch.float32)
    if unconditional_transformer is None:
        if unconditional_text_features is None:
            unconditional_text_features = [torch.zeros_like(features) for features in text_features]
        if len(unconditional_text_features) != len(text_features):
            raise ValueError("unconditional_text_features must have the same batch size as text_features")
        negative_inputs = build_sequence_inputs_from_features(unconditional_text_features, height, width, device=device)
        # Match the negative llm_features dtype to cond_features (float32), mirroring the
        # build_negative_image_inputs branch so both CFG paths feed the DiT the same dtype.
        negative_inputs["llm_features"] = negative_inputs["llm_features"].to(device=device, dtype=cond_features.dtype)
        neg_max_text_tokens = int(negative_inputs["max_text_tokens"])
        neg_text_z_padding = torch.zeros(batch_size, neg_max_text_tokens, latent_dim, dtype=torch.float32, device=device)
    else:
        negative_inputs = build_negative_image_inputs(
            inputs,
            feature_dim=cond_features.shape[-1],
            dtype=cond_features.dtype,
            device=device,
        )
        neg_max_text_tokens = 0
        neg_text_z_padding = None

    denoising_steps = list(_iter_denoising_steps(params, schedule, step_intervals))
    iterator = range(params.num_steps)
    if show_progress:
        iterator = tqdm(iterator, total=params.num_steps, desc="Denoising steps")

    for i in iterator:
        t_val, s_val, gw_i = denoising_steps[i]
        if i == 0 and initial_sigma is not None:
            t_val = 1.0 - float(initial_sigma)
        t = torch.full((batch_size,), t_val, dtype=torch.float32, device=device)

        pos_z = torch.cat([z, text_z_padding], dim=1)
        pos_out = conditional_transformer(
            llm_features=cond_features,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            attention_mask=inputs["attention_mask"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, :num_image_tokens]

        if unconditional_transformer is None:
            neg_z = torch.cat([z, neg_text_z_padding], dim=1)
            neg_out = conditional_transformer(
                llm_features=negative_inputs["llm_features"],
                x=neg_z,
                t=t,
                position_ids=negative_inputs["position_ids"],
                attention_mask=negative_inputs["attention_mask"],
                indicator=negative_inputs["indicator"],
            )
            neg_v = neg_out[:, :num_image_tokens]
        else:
            neg_v = unconditional_transformer(
                llm_features=negative_inputs["llm_features"],
                x=z,
                t=t,
                position_ids=negative_inputs["position_ids"],
                attention_mask=negative_inputs["attention_mask"],
                indicator=negative_inputs["indicator"],
            )

        v = gw_i * pos_v + (1.0 - gw_i) * neg_v
        z = z + v * (s_val - t_val)

    return decode_tokens_to_images(autoencoder, z, grid_h=grid_h, grid_w=grid_w)


def validate_prompt(prompt: str, *, warn_only: bool) -> None:
    issues = CaptionVerifier().verify_raw(prompt)
    if not issues:
        return
    message = "caption verifier flagged prompt:\n" + "\n".join(issues)
    if warn_only:
        logger.warning(message)
    else:
        raise ValueError(message)


def dtype_cache_cost(num_tokens: int, dtype: torch.dtype) -> tuple[float, float]:
    element_size = torch.empty((), dtype=dtype).element_size()
    bytes_per_image = num_tokens * 53248 * element_size
    mb_per_image = bytes_per_image / 1_000_000
    gb_per_1k_images = bytes_per_image * 1_000 / 1_000_000_000
    return mb_per_image, gb_per_1k_images


def fp8_cache_dtype() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is not available in this environment")
    return FP8_WEIGHT_DTYPE
