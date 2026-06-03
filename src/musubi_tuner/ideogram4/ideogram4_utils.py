from __future__ import annotations

import logging
import os
from typing import Optional

import torch
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
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
)
from musubi_tuner.ideogram4.latent_norm import get_latent_norm
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.utils import safetensors_utils

logger = logging.getLogger(__name__)

IDEOGRAM4_OFFICIAL_REPO_ID = "ideogram-ai/ideogram-4-fp8"
IDEOGRAM4_TOKENIZER_SUBFOLDER = "tokenizer"
IDEOGRAM4_TEXT_ENCODER_CONFIG_SUBFOLDER = "text_encoder"
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
        raise ValueError(
            f"{path} has safetensors model_type={metadata.get('model_type')!r}, expected {expected_model_type!r}"
        )
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
    model = Ideogram4Transformer(config or Ideogram4Config())
    device = torch.device(device)

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("Ideogram 4 FP8 weights require torch.float8_e4m3fn support")

    if is_bnb4bit_state_dict(state_dict):
        if device.type != "cuda":
            raise ValueError(f"bnb 4-bit weights require a CUDA device, got device={device}")
        swap_linears_to_bnb4bit(model, compute_dtype=dtype)
        load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
    elif is_fp8_state_dict(state_dict):
        model.to(dtype)
        swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
        load_fp8_state_dict(model, state_dict, device=device, dtype=dtype)
    else:
        model.load_state_dict(state_dict)
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
        IDEOGRAM4_OFFICIAL_REPO_ID,
        subfolder=IDEOGRAM4_TOKENIZER_SUBFOLDER,
        trust_remote_code=True,
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
        IDEOGRAM4_OFFICIAL_REPO_ID,
        subfolder=IDEOGRAM4_TEXT_ENCODER_CONFIG_SUBFOLDER,
        trust_remote_code=True,
    )
    model = AutoModel.from_config(config, trust_remote_code=True)
    state_dict = _load_state_dict(path, device="cpu", disable_mmap=disable_mmap)
    device = torch.device(device)
    if is_fp8_state_dict(state_dict):
        swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
        load_fp8_state_dict(model, state_dict, device=device, dtype=dtype, assign=True, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)
        model.to(device=device, dtype=dtype)
    model.eval()
    return model


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
    language_model = text_encoder.language_model
    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
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
    decoded = autoencoder.decode(z)
    decoded = decoded.float().clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]


@torch.no_grad()
def generate_images(
    *,
    conditional_transformer: Ideogram4Transformer,
    unconditional_transformer: Ideogram4Transformer,
    autoencoder: AutoEncoder,
    text_features: list[torch.Tensor],
    height: int,
    width: int,
    sampler_preset: str,
    device: torch.device,
    seed: Optional[int] = None,
    show_progress: bool = True,
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

    cond_features = inputs["llm_features"].to(device=device, dtype=conditional_transformer.dtype)
    negative_inputs = build_negative_image_inputs(
        inputs,
        feature_dim=cond_features.shape[-1],
        dtype=cond_features.dtype,
        device=device,
    )

    iterator = range(params.num_steps - 1, -1, -1)
    if show_progress:
        iterator = tqdm(iterator, total=params.num_steps, desc="Denoising steps")

    for i in iterator:
        t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
        s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
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

        neg_v = unconditional_transformer(
            llm_features=negative_inputs["llm_features"],
            x=z,
            t=t,
            position_ids=negative_inputs["position_ids"],
            segment_ids=negative_inputs["segment_ids"],
            indicator=negative_inputs["indicator"],
        )

        gw_i = params.guidance_schedule[i]
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
    mb_per_image = num_tokens * 53248 * element_size / 1_000_000
    return mb_per_image, mb_per_image * 1000 / 1000


def fp8_cache_dtype() -> torch.dtype:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is not available in this environment")
    return FP8_WEIGHT_DTYPE
