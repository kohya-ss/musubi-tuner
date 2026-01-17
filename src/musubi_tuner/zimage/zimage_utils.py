import json
import logging
import math
import os
from typing import Optional, Union

import numpy as np
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3ForCausalLM

from musubi_tuner.utils.safetensors_utils import load_split_weights
from musubi_tuner.zimage import zimage_config

try:
    from transformers import (
        Siglip2ImageProcessorFast,
        Siglip2Processor,
        Siglip2VisionConfig,
        Siglip2VisionModel,
    )  # type: ignore
except Exception:
    Siglip2VisionModel = None
    Siglip2Processor = None
    Siglip2ImageProcessorFast = None
    Siglip2VisionConfig = None

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


ZIMAGE_ID = "Tongyi-MAI/Z-Image-Turbo"


SIGLIP2_IMAGE_PROCESSOR_CONFIG = {
    "data_format": "channels_first",
    "default_to_square": True,
    "device": None,
    "disable_grouping": None,
    "do_convert_rgb": None,
    "do_normalize": True,
    "do_pad": None,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "Siglip2ImageProcessorFast",
    "image_std": [0.5, 0.5, 0.5],
    "input_data_format": None,
    "max_num_patches": 256,
    "pad_size": None,
    "patch_size": 16,
    "processor_class": "Siglip2Processor",
    "resample": 2,
    "rescale_factor": 0.00392156862745098,
    "return_tensors": None,
}


SIGLIP2_VISION_CONFIG = {
    "architectures": ["Siglip2VisionModel"],
    "hidden_size": 1152,
    "intermediate_size": 4608,
    "num_hidden_layers": 27,
    "num_attention_heads": 18,
    "image_size": 256,
    "patch_size": 16,
    "num_channels": 3,
    "attention_dropout": 0.0,
    "layer_norm_eps": 1e-6,
    "hidden_act": "gelu_pytorch_tanh",
}


def shift_scale_latents_for_decode(latents: torch.Tensor) -> torch.Tensor:
    """Shift and scale latents before decoding with the VAE. latents should be casted to float32 before calling this function."""
    latents = (latents / zimage_config.ZIMAGE_VAE_SCALING_FACTOR) + zimage_config.ZIMAGE_VAE_SHIFT_FACTOR
    return latents


def load_image_encoders(args):
    if args.image_encoder is None:
        raise ValueError("--image_encoder is required")

    logger.info("Loading image encoder feature extractor")
    if Siglip2Processor is None and Siglip2ImageProcessorFast is None:
        raise RuntimeError(
            "SigLIP2 processor classes are not available in this transformers version. "
            "Please upgrade transformers to a version that provides Siglip2Processor/Siglip2ImageProcessorFast."
        )

    if Siglip2VisionModel is None:
        raise RuntimeError(
            "Siglip2VisionModel is not available in this transformers version. "
            "Please upgrade transformers to a version that provides Siglip2VisionModel."
        )

    if os.path.isdir(args.image_encoder):
        # directory checkpoint
        if Siglip2Processor is not None:
            feature_extractor = Siglip2Processor.from_pretrained(args.image_encoder, subfolder="image_encoder")
        else:
            assert Siglip2ImageProcessorFast is not None
            feature_extractor = Siglip2ImageProcessorFast.from_pretrained(args.image_encoder, subfolder="image_encoder")

        logger.info(f"Loading image encoder from {args.image_encoder}")
        image_encoder = Siglip2VisionModel.from_pretrained(args.image_encoder, subfolder="image_encoder", torch_dtype=torch.float16)
        image_encoder.eval()
        return feature_extractor, image_encoder

    # single-file checkpoint
    if Siglip2ImageProcessorFast is None:
        raise RuntimeError(
            "Siglip2ImageProcessorFast is not available in this transformers version. "
            "Single-file --image_encoder requires Siglip2ImageProcessorFast."
        )
    if Siglip2VisionConfig is None:
        raise RuntimeError(
            "Siglip2VisionConfig is not available in this transformers version. "
            "Single-file --image_encoder requires Siglip2VisionConfig."
        )

    feature_extractor = Siglip2ImageProcessorFast(**SIGLIP2_IMAGE_PROCESSOR_CONFIG)

    cfg = Siglip2VisionConfig(**SIGLIP2_VISION_CONFIG)
    with init_empty_weights():
        image_encoder = Siglip2VisionModel._from_config(cfg, torch_dtype=torch.float16)

    sd = load_file(args.image_encoder)
    image_encoder.load_state_dict(sd, strict=True, assign=True)
    image_encoder.eval()
    return feature_extractor, image_encoder


def load_qwen3(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> tuple[Qwen2Tokenizer, Qwen3ForCausalLM]:
    QWEN3_CONFIG_JSON = """
{
  "architectures": [
    "Qwen3ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 9728,
  "max_position_embeddings": 40960,
  "max_window_layers": 36,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 36,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 1000000,
  "sliding_window": null,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 151936
}
"""
    config = json.loads(QWEN3_CONFIG_JSON)
    config = Qwen3Config(**config)
    with init_empty_weights():
        qwen3 = Qwen3ForCausalLM._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_split_weights(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    sd["lm_head.weight"] = sd["model.embed_tokens.weight"]  # tie weights

    info = qwen3.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Qwen3: {info}")
    qwen3.to(device)

    if dtype is not None:
        if dtype.itemsize == 1:  # torch.float8
            # prepare Qwen3 for fp8
            org_dtype = torch.bfloat16  # model weight is fp8 in loading, but original dtype is bfloat16
            logger.info(f"prepare Qwen3 for fp8: set to {dtype} from {org_dtype}")
            qwen3.to(dtype)

            # prepare LLM for fp8
            def prepare_fp8(vl_model: Qwen3ForCausalLM, target_dtype):
                def rms_norm_forward_hook(module):
                    def forward(hidden_states):
                        input_dtype = hidden_states.dtype
                        hidden_states = hidden_states.to(torch.float32)
                        variance = hidden_states.pow(2).mean(-1, keepdim=True)
                        hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                        # return module.weight.to(input_dtype) * hidden_states.to(input_dtype)
                        return (module.weight.to(torch.float32) * hidden_states.to(torch.float32)).to(input_dtype)

                    return forward

                for module in vl_model.modules():
                    if module.__class__.__name__ in ["Embedding"]:
                        # print("set", module.__class__.__name__, "to", target_dtype)
                        module.to(target_dtype)
                    if module.__class__.__name__ in ["Qwen3RMSNorm"]:
                        # print("set", module.__class__.__name__, "hooks")
                        module.forward = rms_norm_forward_hook(module)

            prepare_fp8(qwen3, org_dtype)

        else:
            logger.info(f"Setting Qwen3 to dtype: {dtype}")
            qwen3.to(dtype)
    # Load tokenizer
    # TODO change to specific tokenizer class
    logger.info(f"Loading tokenizer from {ZIMAGE_ID}")
    tokenizer = Qwen2Tokenizer.from_pretrained(ZIMAGE_ID, subfolder="tokenizer")
    return tokenizer, qwen3


def get_text_embeds(
    tokenizer: Qwen2Tokenizer,
    text_encoder: Qwen3ForCausalLM,
    prompt: Union[list[str], str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Get text embeddings from the text encoder.
    Applies chat template to each prompt before encoding.

    Args:
        tokenizer (Qwen2Tokenizer): The tokenizer to use.
        text_encoder (Qwen3ForCausalLM): The text encoder model.
        prompt (list[str] | str): The input prompt(s).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the prompt embeddings and attention masks.
    """
    prompt = [prompt] if isinstance(prompt, str) else prompt

    # logger.info(f"Encoding prompts: {prompt}. Applying chat template.")
    formatted_prompts = []
    for p in prompt:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(formatted_prompt)

    text_inputs = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=zimage_config.DEFAULT_MAX_SEQUENCE_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(text_encoder.device)
    prompt_masks = text_inputs.attention_mask.to(text_encoder.device).bool()

    with torch.no_grad():
        text_encoder_params = text_encoder.parameters()
        text_encoder_params.__next__()  # skip first param (embedding)
        second_param = text_encoder_params.__next__()
        if second_param.dtype.itemsize == 1:  # torch.float8
            with torch.autocast(device_type=text_encoder.device.type, dtype=torch.bfloat16):
                prompt_embeds = text_encoder(
                    input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True
                ).hidden_states[-2]
        else:
            prompt_embeds = text_encoder(
                input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True
            ).hidden_states[-2]
    return prompt_embeds, prompt_masks


def trim_pad_embeds_and_mask(
    image_length: int, prompt_embeds: torch.Tensor, prompt_masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Trim and pad embeddings and masks to the divisible of SEQ_MULTI_OF according to the maximum image and text length.
    If the batch size is 1, this function will trim the embeddings and masks to the actual text length without padding.
    """
    if prompt_embeds.shape[0] == 1:
        actual_text_length = int(prompt_masks.sum(dim=1).item())
        prompt_embeds = prompt_embeds[:, :actual_text_length, :]
        prompt_masks = prompt_masks[:, :actual_text_length]
        return prompt_embeds, prompt_masks

    max_text_length = prompt_masks.sum(dim=1).max().item()
    total_length = image_length + max_text_length
    padded_total_length = math.ceil(total_length / zimage_config.SEQ_MULTI_OF) * zimage_config.SEQ_MULTI_OF
    pad_length = padded_total_length - total_length
    max_text_length += pad_length
    if max_text_length > prompt_embeds.shape[1]:
        # pad
        pad_size = max_text_length - prompt_embeds.shape[1]
        pad_embeds = torch.zeros(
            (prompt_embeds.shape[0], pad_size, prompt_embeds.shape[2]), dtype=prompt_embeds.dtype, device=prompt_embeds.device
        )
        prompt_embeds = torch.cat([prompt_embeds, pad_embeds], dim=1)
        pad_masks = torch.zeros((prompt_masks.shape[0], pad_size), dtype=prompt_masks.dtype, device=prompt_masks.device)
        prompt_masks = torch.cat([prompt_masks, pad_masks], dim=1)
    else:
        # trim
        prompt_embeds = prompt_embeds[:, :max_text_length, :]
        prompt_masks = prompt_masks[:, :max_text_length]
    return prompt_embeds, prompt_masks


def get_timesteps_sigmas(num_inference_steps: int, shift: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Retrieve timesteps based on Z-Image's sigma schedule with shift."""
    num_train_timesteps = zimage_config.DEFAULT_SCHEDULER_NUM_TRAIN_TIMESTEPS
    timesteps = np.linspace(num_train_timesteps, 1, num_inference_steps + 1)[:-1]
    sigmas = timesteps / num_train_timesteps  # 0-1 range

    sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
    timesteps = sigmas * num_train_timesteps

    timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)
    sigmas = torch.cat([sigmas, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)], dim=0)  # add final sigma 0

    return timesteps, sigmas


def step(model_output: torch.Tensor, sample: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    """Predict the sample at the previous timestep."""
    sample = sample.to(torch.float32)
    sigma_idx = step_index
    sigma = sigmas[sigma_idx]
    sigma_next = sigmas[sigma_idx + 1]

    dt = sigma_next - sigma
    prev_sample = sample + dt * model_output
    prev_sample = prev_sample.to(model_output.dtype)
    return prev_sample
