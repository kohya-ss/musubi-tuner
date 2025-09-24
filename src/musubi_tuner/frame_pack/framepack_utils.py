import os
import logging
from typing import Optional, Union

from accelerate import init_empty_weights
import torch
from safetensors.torch import load_file
from transformers import (
    LlamaTokenizerFast,
    LlamaConfig,
    LlamaModel,
    CLIPTokenizer,
    CLIPTextModel,
    CLIPConfig,
    SiglipImageProcessor,
    SiglipVisionModel,
    SiglipVisionConfig,
)

from musubi_tuner.utils.safetensors_utils import load_split_weights
from musubi_tuner.hunyuan_model.vae import load_vae as hunyuan_load_vae


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_vae(
    vae_path: str,
    vae_chunk_size: Optional[int],
    vae_spatial_tile_sample_min_size: Optional[int],
    vae_tiling: bool,
    device: Union[str, torch.device],
):
    # single file and directory (contains 'vae') support
    if os.path.isdir(vae_path):
        vae_path = os.path.join(vae_path, "vae", "diffusion_pytorch_model.safetensors")
    else:
        vae_path = vae_path

    vae_dtype = torch.float16  # if vae_dtype is None else str_to_dtype(vae_dtype)
    vae, _, s_ratio, t_ratio = hunyuan_load_vae(vae_dtype=vae_dtype, device=device, vae_path=vae_path)
    vae.eval()
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # set chunk_size to CausalConv3d recursively
    chunk_size = vae_chunk_size
    if chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(chunk_size)
        logger.info(f"Set chunk_size to {chunk_size} for CausalConv3d")

    if vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = vae_spatial_tile_sample_min_size // 8
        logger.info(f"Enabled spatial tiling with min size {vae_spatial_tile_sample_min_size}")
    elif vae_tiling:
        vae.enable_spatial_tiling(True)

    return vae


# region Text Encoders

# Text Encoder configs are copied from HunyuanVideo repo

LLAMA_CONFIG = {
    "architectures": ["LlamaModel"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 8192,
    "mlp_bias": False,
    "model_type": "llama",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 500000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float16",
    "transformers_version": "4.46.3",
    "use_cache": True,
    "vocab_size": 128320,
}

CLIP_CONFIG = {
    #   "_name_or_path": "/raid/aryan/llava-llama-3-8b-v1_1-extracted/text_encoder_2",
    "architectures": ["CLIPTextModel"],
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "dropout": 0.0,
    "eos_token_id": 2,
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "initializer_factor": 1.0,
    "initializer_range": 0.02,
    "intermediate_size": 3072,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 77,
    "model_type": "clip_text_model",
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "pad_token_id": 1,
    "projection_dim": 768,
    "torch_dtype": "float16",
    "transformers_version": "4.48.0.dev0",
    "vocab_size": 49408,
}


def load_text_encoder1(
    args, fp8_llm: Optional[bool] = False, device: Optional[Union[str, torch.device]] = None
) -> tuple[LlamaTokenizerFast, LlamaModel]:
    # single file, split file and directory (contains 'text_encoder') support
    logger.info("Loading text encoder 1 tokenizer")
    tokenizer1 = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer")

    logger.info(f"Loading text encoder 1 from {args.text_encoder1}")
    if os.path.isdir(args.text_encoder1):
        # load from directory, configs are in the directory
        text_encoder1 = LlamaModel.from_pretrained(args.text_encoder1, subfolder="text_encoder", torch_dtype=torch.float16)
    else:
        # load from file, we create the model with the appropriate config
        config = LlamaConfig(**LLAMA_CONFIG)
        with init_empty_weights():
            text_encoder1 = LlamaModel._from_config(config, torch_dtype=torch.float16)

        state_dict = load_split_weights(args.text_encoder1)

        # support weights from ComfyUI
        if "model.embed_tokens.weight" in state_dict:
            for key in list(state_dict.keys()):
                if key.startswith("model."):
                    new_key = key.replace("model.", "")
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        if "tokenizer" in state_dict:
            state_dict.pop("tokenizer")
        if "lm_head.weight" in state_dict:
            state_dict.pop("lm_head.weight")

        # # support weights from ComfyUI
        # if "tokenizer" in state_dict:
        #     state_dict.pop("tokenizer")

        text_encoder1.load_state_dict(state_dict, strict=True, assign=True)

    if fp8_llm:
        org_dtype = text_encoder1.dtype
        logger.info(f"Moving and casting text encoder to {device} and torch.float8_e4m3fn")
        text_encoder1.to(device=device, dtype=torch.float8_e4m3fn)

        # prepare LLM for fp8
        def prepare_fp8(llama_model: LlamaModel, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                    return module.weight.to(input_dtype) * hidden_states.to(input_dtype)

                return forward

            for module in llama_model.modules():
                if module.__class__.__name__ in ["Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["LlamaRMSNorm"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        prepare_fp8(text_encoder1, org_dtype)
    else:
        text_encoder1.to(device)

    text_encoder1.eval()
    return tokenizer1, text_encoder1


def load_text_encoder2(args) -> tuple[CLIPTokenizer, CLIPTextModel]:
    # single file and directory (contains 'text_encoder_2') support
    logger.info("Loading text encoder 2 tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder="tokenizer_2")

    logger.info(f"Loading text encoder 2 from {args.text_encoder2}")
    if os.path.isdir(args.text_encoder2):
        # load from directory, configs are in the directory
        text_encoder2 = CLIPTextModel.from_pretrained(args.text_encoder2, subfolder="text_encoder_2", torch_dtype=torch.float16)
    else:
        # we only have one file, so we can load it directly
        config = CLIPConfig(**CLIP_CONFIG)
        with init_empty_weights():
            text_encoder2 = CLIPTextModel._from_config(config, torch_dtype=torch.float16)

        state_dict = load_file(args.text_encoder2)

        text_encoder2.load_state_dict(state_dict, strict=True, assign=True)

    text_encoder2.eval()
    return tokenizer2, text_encoder2


# endregion

# region image encoder

# Siglip configs are copied from FramePack repo
FEATURE_EXTRACTOR_CONFIG = {
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "SiglipImageProcessor",
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "SiglipProcessor",
    "resample": 3,
    "rescale_factor": 0.00392156862745098,
    "size": {"height": 384, "width": 384},
}
IMAGE_ENCODER_CONFIG = {
    "_name_or_path": "/home/lvmin/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-Redux-dev/snapshots/1282f955f706b5240161278f2ef261d2a29ad649/image_encoder",
    "architectures": ["SiglipVisionModel"],
    "attention_dropout": 0.0,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1152,
    "image_size": 384,
    "intermediate_size": 4304,
    "layer_norm_eps": 1e-06,
    "model_type": "siglip_vision_model",
    "num_attention_heads": 16,
    "num_channels": 3,
    "num_hidden_layers": 27,
    "patch_size": 14,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.46.2",
}


def load_image_encoders(args):
    logger.info("Loading image encoder feature extractor")
    feature_extractor = SiglipImageProcessor(**FEATURE_EXTRACTOR_CONFIG)

    # single file, split file and directory (contains 'image_encoder') support
    logger.info(f"Loading image encoder from {args.image_encoder}")
    if os.path.isdir(args.image_encoder):
        # load from directory, configs are in the directory
        image_encoder = SiglipVisionModel.from_pretrained(args.image_encoder, subfolder="image_encoder", torch_dtype=torch.float16)
    else:
        # load from file, we create the model with the appropriate config
        config = SiglipVisionConfig(**IMAGE_ENCODER_CONFIG)
        with init_empty_weights():
            image_encoder = SiglipVisionModel._from_config(config, torch_dtype=torch.float16)

        state_dict = load_file(args.image_encoder)

        image_encoder.load_state_dict(state_dict, strict=True, assign=True)

    image_encoder.eval()
    return feature_extractor, image_encoder


# endregion
