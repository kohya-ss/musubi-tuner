import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from accelerate import init_empty_weights
from transformers import AutoModel, AutoTokenizer, AutoConfig

from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.safetensors_utils import load_split_weights

from .ernie_image_model import ErnieImageTransformer2DModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ERNIE_IMAGE_ID = "baidu/ERNIE-Image"

FP8_OPTIMIZATION_TARGET_KEYS = ["layers."]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["adaLN_modulation", ".norm", "time_", "x_embedder", "text_proj", "final_"]

ERNIE_IMAGE_TEXT_ENCODER_CONFIG_JSON = """\
{
  "architectures": [
    "Mistral3Model"
  ],
  "dtype": "bfloat16",
  "image_token_index": 10,
  "model_type": "mistral3",
  "multimodal_projector_bias": false,
  "projector_hidden_act": "gelu",
  "spatial_merge_size": 2,
  "text_config": {
    "attention_dropout": 0.0,
    "bos_token_id": 1,
    "dtype": "bfloat16",
    "eos_token_id": 2,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 3072,
    "initializer_range": 0.02,
    "intermediate_size": 9216,
    "max_position_embeddings": 262144,
    "model_type": "ministral3",
    "num_attention_heads": 32,
    "num_hidden_layers": 26,
    "num_key_value_heads": 8,
    "pad_token_id": 11,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {
      "beta_fast": 32.0,
      "beta_slow": 1.0,
      "factor": 16.0,
      "llama_4_scaling_beta": 0.1,
      "mscale": 1.0,
      "mscale_all_dim": 1.0,
      "original_max_position_embeddings": 16384,
      "rope_theta": 1000000.0,
      "rope_type": "yarn",
      "type": "yarn"
    },
    "sliding_window": null,
    "tie_word_embeddings": true,
    "use_cache": true,
    "vocab_size": 131072
  },
  "tie_word_embeddings": true,
  "transformers_version": "5.2.0",
  "vision_config": {
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "head_dim": 64,
    "hidden_act": "silu",
    "hidden_size": 1024,
    "image_size": 1540,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "pixtral",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 24,
    "patch_size": 14,
    "rope_parameters": {
      "rope_theta": 10000.0,
      "rope_type": "default"
    }
  },
  "vision_feature_layer": -1
}
"""

# Default model params matching HuggingFace config
DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_NUM_ATTENTION_HEADS = 32
DEFAULT_NUM_LAYERS = 36
DEFAULT_FFN_HIDDEN_SIZE = 12288
DEFAULT_IN_CHANNELS = 128
DEFAULT_OUT_CHANNELS = 128
DEFAULT_PATCH_SIZE = 1
DEFAULT_TEXT_IN_DIM = 3072
DEFAULT_ROPE_THETA = 256
DEFAULT_ROPE_AXES_DIM = (32, 48, 48)
DEFAULT_EPS = 1e-6
DEFAULT_QK_LAYERNORM = True

MAX_SEQ_LENGTH = 512


def create_dit(
    attn_mode: str = "torch",
    split_attn: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> ErnieImageTransformer2DModel:
    with init_empty_weights():
        model = ErnieImageTransformer2DModel(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            num_attention_heads=DEFAULT_NUM_ATTENTION_HEADS,
            num_layers=DEFAULT_NUM_LAYERS,
            ffn_hidden_size=DEFAULT_FFN_HIDDEN_SIZE,
            in_channels=DEFAULT_IN_CHANNELS,
            out_channels=DEFAULT_OUT_CHANNELS,
            patch_size=DEFAULT_PATCH_SIZE,
            text_in_dim=DEFAULT_TEXT_IN_DIM,
            rope_theta=DEFAULT_ROPE_THETA,
            rope_axes_dim=DEFAULT_ROPE_AXES_DIM,
            eps=DEFAULT_EPS,
            qk_layernorm=DEFAULT_QK_LAYERNORM,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )
        if dtype is not None:
            model.to(dtype)
    return model


def load_dit(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype] = None,
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[List[float]] = None,
    disable_numpy_memmap: bool = False,
) -> ErnieImageTransformer2DModel:
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    model = create_dit(attn_mode, split_attn, dit_weight_dtype)

    logger.info(f"Loading ERNIE-Image DiT from {dit_path}, device={loading_device}")
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded ERNIE-Image DiT: {info}")
    return model


def load_text_encoder(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    tokenizer_id: Optional[str] = None,
) -> Tuple[AutoTokenizer, nn.Module]:
    config_dict = json.loads(ERNIE_IMAGE_TEXT_ENCODER_CONFIG_JSON)
    config = AutoConfig.for_model(**config_dict)
    with init_empty_weights():
        text_encoder = AutoModel.from_config(config)

    logger.info(f"Loading text encoder from {ckpt_path}")
    sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    # Replace "language_model.model." prefix with "language_model."
    sd = {k.replace("language_model.model.", "language_model."): v for k, v in sd.items()}

    info = text_encoder.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded text encoder: {info}")
    text_encoder.to(device)

    if dtype is not None:
        text_encoder.to(dtype)

    tok_path = tokenizer_id if tokenizer_id else ERNIE_IMAGE_ID
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=False, subfolder="tokenizer")
    return tokenizer, text_encoder


@torch.no_grad()
def encode_text(
    tokenizer: AutoTokenizer,
    text_encoder: nn.Module,
    prompt: Union[str, List[str]],
) -> List[torch.Tensor]:
    if isinstance(prompt, str):
        prompt = [prompt]

    text_hiddens = []
    for p in prompt:
        ids = tokenizer(
            p,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )["input_ids"]

        if len(ids) == 0:
            if tokenizer.bos_token_id is not None:
                ids = [tokenizer.bos_token_id]
            else:
                ids = [0]

        input_ids = torch.tensor([ids], device=text_encoder.device)
        outputs = text_encoder(input_ids=input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-2][0]  # [T, hidden_size]
        text_hiddens.append(hidden)

    return text_hiddens


def pad_text(
    text_hiddens: List[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    text_in_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(text_hiddens)
    if B == 0:
        return (
            torch.zeros((0, 0, text_in_dim), device=device, dtype=dtype),
            torch.zeros((0,), device=device, dtype=torch.long),
        )

    normalized = [th.squeeze(1).to(device).to(dtype) if th.dim() == 3 else th.to(device).to(dtype) for th in text_hiddens]
    lens = torch.tensor([t.shape[0] for t in normalized], device=device, dtype=torch.long)
    Tmax = int(lens.max().item())
    text_bth = torch.zeros((B, Tmax, text_in_dim), device=device, dtype=dtype)
    for i, t in enumerate(normalized):
        text_bth[i, : t.shape[0], :] = t
    return text_bth, lens


def patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """2x2 patchify: [B, 32, H, W] -> [B, 128, H/2, W/2]"""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    """Reverse patchify: [B, 128, H/2, W/2] -> [B, 32, H, W]"""
    b, c, h, w = latents.shape
    latents = latents.reshape(b, c // 4, 2, 2, h, w)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(b, c // 4, h * 2, w * 2)


def get_sigmas(num_steps: int, device: torch.device) -> torch.Tensor:
    return torch.linspace(1.0, 0.0, num_steps + 1, device=device)
