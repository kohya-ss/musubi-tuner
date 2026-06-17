from __future__ import annotations

import inspect
import logging
import os
from typing import Optional

import torch
from accelerate import init_empty_weights
from PIL import Image
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from musubi_tuner.ideogram4.caption_verifier import CaptionVerifier
from musubi_tuner.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
    SEQUENCE_PADDING_INDICATOR,
)
from musubi_tuner.ideogram4.ideogram4_autoencoder import AutoEncoder, AutoEncoderParams, convert_diffusers_state_dict
from musubi_tuner.ideogram4.ideogram4_model import Ideogram4Config, Ideogram4Transformer
from musubi_tuner.ideogram4.ideogram4_scheduler import get_schedule_for_resolution, make_step_intervals
from musubi_tuner.ideogram4.ideogram4_quantized_loading import (
    FP8_WEIGHT_DTYPE,
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    load_bnb4bit_state_dict,
    load_fp8_state_dict,
    require_fp8_dtype,
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
)
from musubi_tuner.ideogram4.latent_norm import get_latent_norm
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.utils import safetensors_utils

logger = logging.getLogger(__name__)

QWEN3_VL_8B_INSTRUCT_REPO_ID = "Qwen/Qwen3-VL-8B-Instruct"
IDEOGRAM4_MAX_TEXT_TOKENS = 2048
IDEOGRAM4_PATCH_SIZE = 2
IDEOGRAM4_AE_SCALE_FACTOR = 8
IDEOGRAM4_IMAGE_PATCH = IDEOGRAM4_PATCH_SIZE * IDEOGRAM4_AE_SCALE_FACTOR
IDEOGRAM4_COND_MODEL_TYPE = "ideogram4_cond"
IDEOGRAM4_UNCOND_MODEL_TYPE = "ideogram4_uncond"


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


def load_ideogram4_transformer(
    path: str,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    expected_model_type: Optional[str],
    disable_mmap: bool = False,
    config: Optional[Ideogram4Config] = None,
) -> Ideogram4Transformer:
    validate_local_safetensors(path, expected_model_type)
    state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
    device = torch.device(device)
    is_bnb4bit = is_bnb4bit_state_dict(state_dict)
    is_fp8 = is_fp8_state_dict(state_dict)

    if is_fp8 or not is_bnb4bit:
        with init_empty_weights():
            model = Ideogram4Transformer(config or Ideogram4Config())
    else:
        model = Ideogram4Transformer(config or Ideogram4Config())

    if is_bnb4bit:
        if device.type != "cuda":
            raise ValueError(f"bnb 4-bit weights require a CUDA device, got device={device}")
        swap_linears_to_bnb4bit(model, compute_dtype=dtype)
        load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
    elif is_fp8:
        require_fp8_dtype()
        swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
        load_fp8_state_dict(model, state_dict, device=device, dtype=dtype, assign=True)
    else:
        model.load_state_dict(state_dict, assign=True)
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
    ae = AutoEncoder(AutoEncoderParams())
    try:
        ae.load_state_dict(state_dict)
    except RuntimeError:
        ae.load_state_dict(convert_diffusers_state_dict(state_dict))
    ae.to(device=torch.device(device), dtype=dtype)
    ae.eval()
    return ae


def load_ideogram4_tokenizer():
    return AutoTokenizer.from_pretrained(
        QWEN3_VL_8B_INSTRUCT_REPO_ID,
        trust_remote_code=True,
    )


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
    config = AutoConfig.from_pretrained(
        QWEN3_VL_8B_INSTRUCT_REPO_ID,
        trust_remote_code=True,
    )
    state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
    state_dict = _normalize_qwen3_vl_state_dict_for_automodel(state_dict)
    # Ideogram 4 conditioning only consumes the language model (see _get_qwen3_vl_embeddings), so
    # drop the Qwen3-VL vision tower weights and remove the submodule to avoid materializing it.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("visual.")}
    device = torch.device(device)
    with init_empty_weights():
        model = AutoModel.from_config(config, trust_remote_code=True)
    if hasattr(model, "visual"):
        del model.visual
    _materialize_meta_tensors(model)
    if is_fp8_state_dict(state_dict):
        require_fp8_dtype()
        swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
        load_fp8_state_dict(model, state_dict, device=device, dtype=dtype, assign=True)
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

    llm_features = torch.zeros(batch_size, total_seq_len, feature_dim, dtype=text_features[0].dtype)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long)
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, features in enumerate(text_features):
        num_text = int(features.shape[0])
        pad_len = max_text_tokens - num_text
        total_unpadded = num_text + num_image_tokens
        offset = pad_len
        text_pos = torch.arange(num_text)
        text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
        llm_features[b, offset : offset + num_text] = features.cpu()
        position_ids[b, offset : offset + num_text] = text_pos_3d
        position_ids[b, offset + num_text :] = image_pos
        indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
        indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
        segment_ids[b, offset : offset + total_unpadded] = 1

    return {
        "llm_features": llm_features.to(device),
        "position_ids": position_ids.to(device),
        "segment_ids": segment_ids.to(device),
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
    max_text_tokens = int(sequence_inputs["max_text_tokens"])
    num_image_tokens = int(sequence_inputs["num_image_tokens"])
    position_ids = sequence_inputs["position_ids"][:, max_text_tokens:]
    segment_ids = sequence_inputs["segment_ids"][:, max_text_tokens:]
    indicator = sequence_inputs["indicator"][:, max_text_tokens:]
    batch_size = int(position_ids.shape[0])
    llm_features = torch.zeros(batch_size, num_image_tokens, feature_dim, dtype=dtype, device=device)
    return {
        "llm_features": llm_features,
        "position_ids": position_ids,
        "segment_ids": segment_ids,
        "indicator": indicator,
    }


def build_unconditional_sequence_inputs(
    text_features: list[torch.Tensor],
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor | int]:
    inputs = build_sequence_inputs_from_features(text_features, height, width, device=device)
    inputs["llm_features"] = inputs["llm_features"].to(device=device, dtype=dtype)
    return inputs


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
    shift, scale = get_latent_norm()
    shift = shift.to(device=token_grid.device, dtype=token_grid.dtype).view(1, -1, 1, 1)
    scale = scale.to(device=token_grid.device, dtype=token_grid.dtype).view(1, -1, 1, 1)
    return (token_grid - shift) / scale


def denormalize_tokens(tokens: torch.Tensor) -> torch.Tensor:
    shift, scale = get_latent_norm()
    shift = shift.to(device=tokens.device, dtype=tokens.dtype)
    scale = scale.to(device=tokens.device, dtype=tokens.dtype)
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
        negative_inputs = build_unconditional_sequence_inputs(
            unconditional_text_features,
            height,
            width,
            device=device,
            dtype=cond_features.dtype,
        )
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

        pos_z = torch.cat([text_z_padding, z], dim=1)
        pos_out = conditional_transformer(
            llm_features=cond_features,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, max_text_tokens:]

        if unconditional_transformer is None:
            neg_z = torch.cat([neg_text_z_padding, z], dim=1)
            neg_out = conditional_transformer(
                llm_features=negative_inputs["llm_features"],
                x=neg_z,
                t=t,
                position_ids=negative_inputs["position_ids"],
                segment_ids=negative_inputs["segment_ids"],
                indicator=negative_inputs["indicator"],
            )
            neg_v = neg_out[:, neg_max_text_tokens:]
        else:
            neg_v = unconditional_transformer(
                llm_features=negative_inputs["llm_features"],
                x=z,
                t=t,
                position_ids=negative_inputs["position_ids"],
                segment_ids=negative_inputs["segment_ids"],
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
