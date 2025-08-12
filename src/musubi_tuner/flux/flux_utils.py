import json
import math
from PIL import Image
from typing import Callable, Optional, Union
import einops
import numpy as np
import torch
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel, CLIPTokenizer, T5Tokenizer
from accelerate import init_empty_weights

from musubi_tuner.flux import flux_models
from musubi_tuner.utils import image_utils
from musubi_tuner.utils.safetensors_utils import load_safetensors
from musubi_tuner.utils.train_utils import get_lin_function

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"

# copy from https://github.com/black-forest-labs/flux/blob/main/src/flux/util.py
PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def preprocess_control_image(
    control_image_path: str, resize_to_prefered: bool = True
) -> tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]:
    """
    Preprocess the control image for the model. See `preprocess_image` for details.
    Args:
        control_image_path (str): Path to the control image.
    Returns:
        Tuple[torch.Tensor, np.ndarray, Optional[np.ndarray]]: same as `preprocess_image`.
    """
    # find appropriate bucket for the image size. reference: https://github.com/black-forest-labs/flux/blob/main/src/flux/sampling.py
    control_image = Image.open(control_image_path)
    width, height = control_image.size
    aspect_ratio = width / height

    if resize_to_prefered:
        # Kontext is trained on specific resolutions, using one of them is recommended
        _, bucket_width, bucket_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
        control_latent_width = int(bucket_width / 16)
        control_latent_height = int(bucket_height / 16)
        bucket_width = control_latent_width * 16
        bucket_height = control_latent_height * 16
    else:
        # use the original image size, but make sure it's divisible by 16
        control_latent_width = int(math.floor(width / 16))
        control_latent_height = int(math.floor(height / 16))
        bucket_width = control_latent_width * 16
        bucket_height = control_latent_height * 16
        control_image = control_image.crop((0, 0, bucket_width, bucket_height))

    return image_utils.preprocess_image(control_image, bucket_width, bucket_height)


def is_fp8(dt):
    return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]


def prepare_img_ids(batch_size: int, packed_latent_height: int, packed_latent_width: int, is_ctrl: bool = False) -> torch.Tensor:
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    if is_ctrl:
        img_ids[..., 0] = 1
    img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift_value: Optional[float] = None,
) -> list[float]:

    # shifting the schedule to favor high timesteps for higher signal images
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    if shift_value is None:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    else:
        # logits_norm = torch.randn((1,), device=device)
        # timesteps = logits_norm.sigmoid()
        # timesteps = (timesteps * shift_value) / (1 + (shift_value - 1) * timesteps)

        timesteps = torch.linspace(1, 0, num_steps + 1)
        sigmas = timesteps
        sigmas = shift_value * sigmas / (1 + (shift_value - 1) * sigmas)
        timesteps = sigmas

    return timesteps.tolist()


def load_flow_model(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    attn_mode: str = "torch",
    split_attn: bool = False,
    loading_device: Optional[Union[str, torch.device]] = None,
    fp8_scaled: bool = False,
) -> flux_models.Flux:
    if loading_device is None:
        loading_device = device

    device = torch.device(device)
    loading_device = torch.device(loading_device) if loading_device is not None else device
    flux_kontext_loading_device = loading_device if not fp8_scaled else torch.device("cpu")

    # build model
    with init_empty_weights():
        params = flux_models.configs_flux_dev_context.params

        model = flux_models.Flux(params, attn_mode, split_attn)
        if dtype is not None:
            model = model.to(dtype)

    # load_sft doesn't support torch.device
    logger.info(f"Loading state dict from {ckpt_path} to {flux_kontext_loading_device}")
    sd = load_safetensors(ckpt_path, device=flux_kontext_loading_device, disable_mmap=disable_mmap, dtype=dtype)

    # if the key has annoying prefix, remove it
    for key in list(sd.keys()):
        new_key = key.replace("model.diffusion_model.", "")
        if new_key == key:
            break  # the model doesn't have annoying prefix
        sd[new_key] = sd.pop(key)

    # if fp8_scaled is True, convert the model to fp8
    if fp8_scaled:
        # fp8 optimization: calculate on CUDA, move back to CPU if loading_device is CPU (block swap)
        logger.info(f"Optimizing model weights to fp8. This may take a while.")
        sd = model.fp8_optimization(sd, device, move_to_device=loading_device.type == "cpu")

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Flux: {info}")
    return model


def load_ae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> flux_models.AutoEncoder:
    logger.info("Building AutoEncoder")
    with init_empty_weights():
        # dev and schnell have the same AE params
        ae = flux_models.AutoEncoder(flux_models.configs_flux_dev_context.ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae


def load_clip_l(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[CLIPTokenizer, CLIPTextModel]:
    logger.info("Building CLIP-L")
    CLIPL_CONFIG = {
        "_name_or_path": "clip-vit-large-patch14/",
        "architectures": ["CLIPModel"],
        "initializer_factor": 1.0,
        "logit_scale_init_value": 2.6592,
        "model_type": "clip",
        "projection_dim": 768,
        # "text_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "bos_token_id": 0,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0.0,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_norm_eps": 1e-05,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 77,
        "min_length": 0,
        "model_type": "clip_text_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 1,
        "prefix": None,
        "problem_type": None,
        "projection_dim": 768,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": None,
        "torchscript": False,
        "transformers_version": "4.16.0.dev0",
        "use_bfloat16": False,
        "vocab_size": 49408,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        # },
        # "text_config_dict": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "projection_dim": 768,
        # },
        # "torch_dtype": "float32",
        # "transformers_version": None,
    }
    config = CLIPConfig(**CLIPL_CONFIG)
    with init_empty_weights():
        clip = CLIPTextModel._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = clip.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded CLIP-L: {info}")
    clip.to(device)

    if dtype is not None:
        if is_fp8(dtype):
            logger.info(f"prepare CLIP-L for fp8: set to {dtype}, set embeddings to {torch.bfloat16}")
            clip.to(dtype)  # fp8
            clip.text_model.embeddings.to(dtype=torch.bfloat16)
        else:
            logger.info(f"Setting CLIP-L to dtype: {dtype}")
            clip.to(dtype)

    tokenizer = CLIPTokenizer.from_pretrained(CLIP_L_TOKENIZER_ID)
    return tokenizer, clip


def load_t5xxl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[T5Tokenizer, T5EncoderModel]:
    T5_CONFIG_JSON = """
{
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    config = json.loads(T5_CONFIG_JSON)
    config = T5Config(**config)
    with init_empty_weights():
        t5xxl = T5EncoderModel._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = t5xxl.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded T5xxl: {info}")
    t5xxl.to(device)

    if dtype is not None:
        if is_fp8(dtype):
            logger.info(f"prepare T5xxl for fp8: set to {dtype}")

            def prepare_fp8(text_encoder, target_dtype):
                def forward_hook(module):
                    def forward(hidden_states):
                        hidden_gelu = module.act(module.wi_0(hidden_states))
                        hidden_linear = module.wi_1(hidden_states)
                        hidden_states = hidden_gelu * hidden_linear
                        hidden_states = module.dropout(hidden_states)

                        hidden_states = module.wo(hidden_states)
                        return hidden_states

                    return forward

                for module in text_encoder.modules():
                    if module.__class__.__name__ in ["T5LayerNorm", "Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["T5DenseGatedActDense"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = forward_hook(module)

            t5xxl.to(dtype)
            prepare_fp8(t5xxl.encoder, torch.bfloat16)
        else:
            logger.info(f"Setting T5xxl to dtype: {dtype}")
            t5xxl.to(dtype)

    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained(T5_XXL_TOKENIZER_ID)
    return tokenizer, t5xxl


def get_t5xxl_actual_dtype(t5xxl: T5EncoderModel) -> torch.dtype:
    # nn.Embedding is the first layer, but it could be casted to bfloat16 or float32
    return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype
