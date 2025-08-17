import argparse
from datetime import datetime
import gc
import json
import random
import os
import re
import time
import math
import copy
from typing import Tuple, Optional, List, Union, Any, Dict

from einops import rearrange, repeat
import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import numpy as np
from tqdm import tqdm

from musubi_tuner.flux import flux_utils
from musubi_tuner.flux.flux_utils import load_flow_model
from musubi_tuner.flux import flux_models
from musubi_tuner.dataset import image_video_dataset

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from musubi_tuner.networks import lora_flux
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.hv_generate_video import get_time_flag, save_images_grid, synchronize_device
from musubi_tuner.wan_generate_video import merge_lora_weights

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype  # not used currently because model may be optimized


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="FLUX.1 Kontext inference script")

    # WAN arguments
    # parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    # parser.add_argument(
    #     "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    # )

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="AE directory or path")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 (T5) directory or path")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 (CLIP-L) directory or path")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="image size, height and width")
    parser.add_argument("--control_image_path", type=str, default=None, help="path to control (reference) image for Kontext.")
    parser.add_argument("--no_resize_control", action="store_true", help="do not resize control image")
    parser.add_argument("--infer_steps", type=int, default=25, help="number of inference steps, default is 25")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    # parser.add_argument(
    #     "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    # )
    parser.add_argument(
        "--embedded_cfg_scale", type=float, default=2.5, help="Embeded CFG scale (distilled CFG Scale), default is 2.5"
    )
    # parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    # parser.add_argument(
    #     "--image_path",
    #     type=str,
    #     default=None,
    #     help="path to image for image2video inference. If `;;;` is used, it will be used as section images. The notation is same as `--prompt`.",
    # )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default is None (FLUX.1 default).",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    # parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled")

    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder 1 (T5)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  #  "flash2", "flash3",
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    # parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    # parser.add_argument(
    #     "--compile_args",
    #     nargs=4,
    #     metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
    #     default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
    #     help="Torch.compile settings",
    # )

    # New arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Create dictionary of overrides
    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        # Map options to argument names
        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        # elif option == "g" or option == "l":
        #     overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        # elif option == "im":
        #     overrides["image_mask_path"] = value
        # elif option == "cn":
        #     overrides["control_path"] = value
        # elif option == "n":
        #     overrides["negative_prompt"] = value
        elif option == "ci":  # control_image_path
            overrides["control_image_path"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Apply overrides to args

    Args:
        args: Original arguments
        overrides: Dictionary of overrides

    Returns:
        argparse.Namespace: New arguments with overrides applied
    """
    args_copy = copy.deepcopy(args)

    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int]: (height, width)
    """
    height = args.image_size[0]
    width = args.image_size[1]

    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    return height, width


# region DiT model


def load_dit_model(args: argparse.Namespace, device: torch.device) -> flux_models.Flux:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        flux_models.Flux: DiT model
    """
    loading_device = "cpu"
    if args.blocks_to_swap == 0 and not args.fp8_scaled and args.lora_weight is None:
        loading_device = device

    # do not fp8 optimize because we will merge LoRA weights
    model = load_flow_model(
        ckpt_path=args.dit,
        dtype=None,
        device=loading_device,
        disable_mmap=True,
        attn_mode=args.attn_mode,
        split_attn=False,
    )

    return model


def optimize_model(model: flux_models.Flux, args: argparse.Namespace, device: torch.device) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        args: command line arguments
        device: device to use
    """
    if args.fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=False)  # args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if args.blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if args.fp8:
            target_dtype = torch.float8e4m3fn

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        if target_device is not None and target_dtype is not None:
            model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    # if args.compile:
    #     compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
    #     logger.info(
    #         f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
    #     )
    #     torch._dynamo.config.cache_size_limit = 32
    #     for i in range(len(model.blocks)):
    #         model.blocks[i] = torch.compile(
    #             model.blocks[i],
    #             backend=compile_backend,
    #             mode=compile_mode,
    #             dynamic=compile_dynamic.lower() in "true",
    #             fullgraph=compile_fullgraph.lower() in "true",
    #         )

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(args.blocks_to_swap, device, supports_backward=False)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)


# endregion


def decode_latent(ae: flux_models.AutoEncoder, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    logger.info(f"Decoding image...")
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)  # add batch dimension

    ae.to(device)
    with torch.no_grad():
        pixels = ae.decode(latent.to(device))  # decode to pixels
    pixels = pixels.to("cpu")
    ae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def prepare_image_inputs(args: argparse.Namespace, device: torch.device, ae: flux_models.AutoEncoder) -> Dict[str, Any]:
    """Prepare image-related inputs for Kontext: AE encoding."""
    height, width = check_inputs(args)

    from musubi_tuner.utils.image_utils import preprocess_image

    if args.control_image_path is not None:
        control_image_tensor, _, _ = flux_utils.preprocess_control_image(args.control_image_path, not args.no_resize_control)

        # AE encoding
        logger.info(f"Encoding control image to latent space with AE")
        ae_original_device = ae.device
        ae.to(device)

        with torch.no_grad():
            control_latent = ae.encode(control_image_tensor.to(device, dtype=ae.dtype))
        control_latent = control_latent.to(torch.bfloat16).to("cpu")

        ae.to(ae_original_device)  # Move VAE back to its original device
        clean_memory_on_device(device)

        control_latent = control_latent.cpu()
    else:
        control_latent = None

    return {"height": height, "width": width, "control_latent": control_latent}


def prepare_text_inputs(
    args: argparse.Namespace,
    device: torch.device,
    shared_models: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Prepare text-related inputs for I2V: LLM and TextEncoder encoding."""

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    if shared_models is not None:
        tokenizer1, text_encoder1 = shared_models.get("tokenizer1"), shared_models.get("text_encoder1")
        tokenizer2, text_encoder2 = shared_models.get("tokenizer2"), shared_models.get("text_encoder2")
        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]
        # text_encoder1 and text_encoder2 are on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        # T5XXL is float16 by default, but it causes NaN values in some cases, so we use bfloat16 (or fp8 if specified)
        t5_dtype = torch.float8e4m3fn if args.fp8_t5 else torch.bfloat16
        tokenizer1, text_encoder1 = flux_utils.load_t5xxl(args.text_encoder1, dtype=t5_dtype, device=device, disable_mmap=True)
        tokenizer2, text_encoder2 = flux_utils.load_clip_l(
            args.text_encoder2, dtype=torch.bfloat16, device=device, disable_mmap=True
        )

    # Store original devices to move back later if they were shared. This does nothing if shared_models is None
    text_encoder1_original_device = text_encoder1.device if text_encoder1 else None
    text_encoder2_original_device = text_encoder2.device if text_encoder2 else None

    logger.info(f"Encoding prompt with Text Encoders")

    # Ensure text_encoder1 and text_encoder2 are not None before proceeding
    if not text_encoder1 or not text_encoder2 or not tokenizer1 or not tokenizer2:
        raise ValueError("Text encoders or tokenizers are not loaded properly.")

    # Define a function to move models to device if needed
    # This is to avoid moving models if not needed, especially in interactive mode
    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        nonlocal shared_models

        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving DiT and Text Encoders to appropriate device: {device} or CPU")
        if shared_models and "model" in shared_models:  # DiT model is shared
            if args.blocks_to_swap > 0:
                logger.info("Waiting for 5 seconds to finish block swap")
                time.sleep(5)
            model = shared_models["model"]
            model.to("cpu")
            clean_memory_on_device(device)  # clean memory on device before moving models

        text_encoder1.to(device)
        text_encoder2.to(device)

    prompt = args.prompt
    if prompt in conds_cache:
        t5_vec, clip_l_pooler = conds_cache[prompt]
    else:
        move_models_to_device_if_needed()

        t5_tokens = tokenizer1(
            prompt,
            max_length=flux_models.T5XXL_MAX_LENGTH,
            padding="max_length",
            return_length=False,
            return_overflowing_tokens=False,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
        l_tokens = tokenizer2(prompt, max_length=77, padding="max_length", truncation=True, return_tensors="pt")["input_ids"]

        with torch.autocast(device_type=device.type, dtype=text_encoder1.dtype), torch.no_grad():
            t5_vec = text_encoder1(input_ids=t5_tokens.to(text_encoder1.device), attention_mask=None, output_hidden_states=False)[
                "last_hidden_state"
            ]
            assert torch.isnan(t5_vec).any() == False, "T5 vector contains NaN values"
            t5_vec = t5_vec.cpu()

        with torch.autocast(device_type=device.type, dtype=text_encoder2.dtype), torch.no_grad():
            clip_l_pooler = text_encoder2(l_tokens.to(text_encoder2.device))["pooler_output"]
            clip_l_pooler = clip_l_pooler.cpu()

        conds_cache[prompt] = (t5_vec, clip_l_pooler)

    if not (shared_models and "text_encoder1" in shared_models):  # if loaded locally
        del tokenizer1, text_encoder1, tokenizer2, text_encoder2
    else:  # if shared, move back to original device (likely CPU)
        if text_encoder1:
            text_encoder1.to(text_encoder1_original_device)
        if text_encoder2:
            text_encoder2.to(text_encoder2_original_device)

    clean_memory_on_device(device)

    arg_c = {"t5_vec": t5_vec, "clip_l_pooler": clip_l_pooler, "prompt": prompt}

    return arg_c


def prepare_i2v_inputs(
    args: argparse.Namespace,
    device: torch.device,
    ae: flux_models.AutoEncoder,
    shared_models: Optional[Dict] = None,
) -> Tuple[int, int, Dict[str, Any], Optional[torch.Tensor]]:
    """Prepare inputs for image2video generation: image encoding, text encoding, and AE encoding.

    Args:
        args: command line arguments
        device: device to use
        ae: AE model instance
        shared_models: dictionary containing pre-loaded models (mainly for DiT)

    Returns:
        Tuple[int, int, Dict[str, Any], Optional[torch.Tensor]]: (height, width, context, end_latent)
    """
    # prepare image inputs
    image_inputs = prepare_image_inputs(args, device, ae)
    control_latent = image_inputs["control_latent"]

    # prepare text inputs
    context = prepare_text_inputs(args, device, shared_models)

    return image_inputs["height"], image_inputs["width"], context, control_latent


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_image_data: Optional[Dict] = None,
    precomputed_text_data: Optional[Dict] = None,
) -> tuple[Optional[flux_models.AutoEncoder], torch.Tensor]:  # AE can be Optional
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models (mainly for DiT)
        precomputed_image_data: Optional dictionary with precomputed image data
        precomputed_text_data: Optional dictionary with precomputed text data

    Returns:
        tuple: (flux_models.AutoEncoder model (vae) or None, torch.Tensor generated latent)
    """
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)
    vae_instance_for_return = None

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if precomputed_image_data is not None and precomputed_text_data is not None:
        logger.info("Using precomputed image and text data.")
        height = precomputed_image_data["height"]
        width = precomputed_image_data["width"]
        control_latent = precomputed_image_data["control_latent"]

        context = precomputed_text_data

        # VAE is not loaded here if data is precomputed; decoding VAE is handled by caller (e.g., process_batch_prompts)
        # vae_instance_for_return remains None
    else:
        # Load VAE if not precomputed (for single/interactive mode)
        # shared_models for single/interactive might contain text/image encoders, but not VAE after `load_shared_models` change.
        # So, VAE will be loaded here for single/interactive.
        logger.info("No precomputed data. Preparing image and text inputs.")
        if shared_models and "ae" in shared_models:  # Should not happen with new load_shared_models
            vae_instance_for_return = shared_models["ae"]
        else:
            # the dtype of VAE weights is float32, but we can load it as bfloat16 for better performance in future
            vae_instance_for_return = flux_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)

        height, width, context, control_latent = prepare_i2v_inputs(args, device, vae_instance_for_return, shared_models)

    if shared_models is None or "model" not in shared_models:
        # load DiT model
        model = load_dit_model(args, device)

        # merge LoRA weights
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            merge_lora_weights(
                lora_flux,
                model,
                args.lora_weight,
                args.lora_multiplier,
                args.include_patterns,
                args.exclude_patterns,
                device,
                args.lycoris,
                args.save_merged_model,
            )

            # if we only want to save the model, we can skip the rest
            if args.save_merged_model:
                return None, None

        # optimize model: fp8 conversion, block swap etc.
        optimize_model(model, args, device)

        if shared_models is not None:
            shared_models["model"] = model
    else:
        # use shared model
        model: flux_models.Flux = shared_models["model"]
        model.move_to_device_except_swap_blocks(device)  # Handles block swap correctly
        model.prepare_block_swap_before_forward()

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######

    # def get_latent_mask(mask_image: Image.Image) -> torch.Tensor:
    #     if mask_image.mode != "L":
    #         mask_image = mask_image.convert("L")
    #     mask_image = mask_image.resize((width // 8, height // 8), Image.LANCZOS)
    #     mask_image = np.array(mask_image)  # PIL to numpy, HWC
    #     mask_image = torch.from_numpy(mask_image).float() / 255.0  # 0 to 1.0, HWC
    #     mask_image = mask_image.squeeze(-1)  # HWC -> HW
    #     mask_image = mask_image.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # HW -> 111HW (BCFHW)
    #     mask_image = mask_image.to(torch.float32)
    #     return mask_image

    logger.info(f"Prompt: {context['prompt']}")

    t5_vec = context["t5_vec"].to(device, dtype=torch.bfloat16)
    clip_l_pooler = context["clip_l_pooler"].to(device, dtype=torch.bfloat16)

    txt_ids = torch.zeros(t5_vec.shape[0], t5_vec.shape[1], 3, device=t5_vec.device)

    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = height // 16, width // 16
    noise_dtype = torch.float32
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        dtype=noise_dtype,
        generator=seed_g,
        device="cpu",
    ).to(device, dtype=torch.bfloat16)

    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(device)

    # image ids are the same as base image with the first dimension set to 1 instead of 0
    if control_latent is not None:
        # pack control_latent
        ctrl_packed_height = control_latent.shape[2] // 2
        ctrl_packed_width = control_latent.shape[3] // 2
        control_latent = rearrange(control_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        control_latent_ids = flux_utils.prepare_img_ids(1, ctrl_packed_height, ctrl_packed_width, is_ctrl=True).to(device)

        control_latent = control_latent.to(device, dtype=torch.bfloat16)
    else:
        control_latent_ids = None

    # denoise
    timesteps = flux_utils.get_schedule(
        num_steps=args.infer_steps, image_seq_len=packed_latent_height * packed_latent_width, shift_value=args.flow_shift
    )

    guidance = args.embedded_cfg_scale
    x = noise

    # logger.info(f"guidance: {guidance}, timesteps: {timesteps}")
    guidance_vec = torch.full((x.shape[0],), guidance, device=x.device, dtype=x.dtype)

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((x.shape[0],), t_curr, dtype=x.dtype, device=x.device)

        img_input = x
        img_input_ids = img_ids
        if control_latent is not None:
            # if control_latent is provided, concatenate it to the input
            img_input = torch.cat((img_input, control_latent), dim=1)
            img_input_ids = torch.cat((img_input_ids, control_latent_ids), dim=1)

        with torch.no_grad():
            pred = model(
                img=img_input,
                img_ids=img_input_ids,
                txt=t5_vec,
                txt_ids=txt_ids,
                y=clip_l_pooler,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
        pred = pred[:, : x.shape[1]]

        x = x + (t_prev - t_curr) * pred

    # unpack
    x = x.float()
    x = rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

    return vae_instance_for_return, x


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: Latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame

    Returns:
        str: Path to saved latent file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed

    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
        }
        # if args.negative_prompt is not None:
        #     metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save images to directory

    Args:
        sample: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved images directory
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0).unsqueeze(2)  # C,HW -> BCTHW, where B=1, C=3, T=1
    save_images_grid(sample, save_path, image_name, rescale=True, create_subdir=False)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    ae: flux_models.AutoEncoder,  # Expect a VAE instance for decoding
    latent: torch.Tensor,
    device: torch.device,
    original_base_names: Optional[List[str]] = None,
) -> None:
    """save output

    Args:
        args: command line arguments
        vae: VAE model
        latent: latent tensor
        device: device to use
        original_base_names: original base names (if latents are loaded from files)
    """
    height, width = latent.shape[-2], latent.shape[-1]  # BCTHW
    height *= 8
    width *= 8
    # print(f"Saving output. Latent shape {latent.shape}; pixel shape {height}x{width}")
    if args.output_type == "latent" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if ae is None:
        logger.error("AE is None, cannot decode latents for saving video/images.")
        return

    video = decode_latent(ae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        # save images
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_images(video, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    """Process multiple prompts for batch mode

    Args:
        prompt_lines: List of prompt lines
        base_args: Base command line arguments

    Returns:
        List[Dict]: List of prompt data dictionaries
    """
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):  # Skip empty lines and comments
            continue

        # Parse prompt line and create override dictionary
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    """Load shared models for batch processing or interactive mode.
    Models are loaded to CPU to save memory. VAE is NOT loaded here.
    DiT model is also NOT loaded here, handled by process_batch_prompts or generate.

    Args:
        args: Base command line arguments

    Returns:
        Dict: Dictionary of shared models (text/image encoders)
    """
    shared_models = {}
    # Load text encoders to CPU
    t5_dtype = torch.float8e4m3fn if args.fp8_t5 else torch.bfloat16
    tokenizer1, text_encoder1 = flux_utils.load_t5xxl(args.text_encoder1, dtype=t5_dtype, device="cpu", disable_mmap=True)
    tokenizer2, text_encoder2 = flux_utils.load_clip_l(args.text_encoder2, dtype=torch.bfloat16, device="cpu", disable_mmap=True)

    shared_models["tokenizer1"] = tokenizer1
    shared_models["text_encoder1"] = text_encoder1
    shared_models["tokenizer2"] = tokenizer2
    shared_models["text_encoder2"] = text_encoder2

    return shared_models


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse and batched precomputation

    Args:
        prompts_data: List of prompt data dictionaries
        args: Base command line arguments
    """
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    device = gen_settings.device

    # 1. Precompute Image Data (AE and Image Encoders)
    logger.info("Loading AE and Image Encoders for batch image preprocessing...")
    ae_for_batch = flux_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)

    all_precomputed_image_data = []
    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]  # Create all arg instances first

    logger.info("Preprocessing images and AE encoding for all prompts...")

    # AE and Image Encoder to device for this phase, because we do not want to offload them to CPU
    ae_for_batch.to(device)

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Image preprocessing for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
        # prepare_image_inputs will move ae/image_encoder to device temporarily
        image_data = prepare_image_inputs(prompt_args_item, device, ae_for_batch)
        all_precomputed_image_data.append(image_data)

    # Models should be back on GPU because prepare_image_inputs moved them to the original device
    ae_for_batch.to("cpu")  # Move AE back to CPU
    clean_memory_on_device(device)

    # 2. Precompute Text Data (Text Encoders)
    logger.info("Loading Text Encoders for batch text preprocessing...")

    # Text Encoders loaded to CPU by load_text_encoder1/2
    t5_dtype = torch.float8e4m3fn if args.fp8_t5 else torch.bfloat16
    tokenizer1_batch, text_encoder1_batch = flux_utils.load_t5xxl(
        args.text_encoder1, dtype=t5_dtype, device=device, disable_mmap=True
    )
    tokenizer2_batch, text_encoder2_batch = flux_utils.load_clip_l(
        args.text_encoder2, dtype=torch.bfloat16, device=device, disable_mmap=True
    )

    # Text Encoders to device for this phase
    text_encoder2_batch.to(device)  # Moved into prepare_text_inputs logic

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "tokenizer1": tokenizer1_batch,
        "text_encoder1": text_encoder1_batch,  # on GPU
        "tokenizer2": tokenizer2_batch,
        "text_encoder2": text_encoder2_batch,  # on GPU
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
        # prepare_text_inputs will move text_encoders to device temporarily
        text_data = prepare_text_inputs(prompt_args_item, device, temp_shared_models_txt)
        all_precomputed_text_data.append(text_data)

    # Models should be removed from device after prepare_text_inputs
    del tokenizer1_batch, text_encoder1_batch, tokenizer2_batch, text_encoder2_batch, temp_shared_models_txt, conds_cache_batch
    clean_memory_on_device(device)

    # 3. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    # Use args from the first prompt for DiT loading (LoRA etc. should be consistent for a batch)
    first_prompt_args = all_prompt_args_list[0]
    dit_model = load_dit_model(first_prompt_args, device)  # Load directly to target device if possible

    if first_prompt_args.lora_weight is not None and len(first_prompt_args.lora_weight) > 0:
        logger.info("Merging LoRA weights into DiT model...")
        merge_lora_weights(
            lora_flux,
            dit_model,
            first_prompt_args.lora_weight,
            first_prompt_args.lora_multiplier,
            first_prompt_args.include_patterns,
            first_prompt_args.exclude_patterns,
            device,
            first_prompt_args.lycoris,
            first_prompt_args.save_merged_model,
        )
        if first_prompt_args.save_merged_model:
            logger.info("Merged DiT model saved. Skipping generation.")
            del dit_model
            clean_memory_on_device(device)
            return

    logger.info("Optimizing DiT model...")
    optimize_model(dit_model, first_prompt_args, device)  # Handles device placement, fp8 etc.

    shared_models_for_generate = {"model": dit_model}  # Pass DiT via shared_models

    all_latents = []

    logger.info("Generating latents for all prompts...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_image_data = all_precomputed_image_data[i]
            current_text_data = all_precomputed_text_data[i]

            logger.info(f"Generating latent for prompt {i+1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
            try:
                # generate is called with precomputed data, so it won't load VAE/Text/Image encoders.
                # It will use the DiT model from shared_models_for_generate.
                # The VAE instance returned by generate will be None here.
                _, latent = generate(
                    prompt_args_item, gen_settings, shared_models_for_generate, current_image_data, current_text_data
                )

                if latent is None:  # and prompt_args_item.save_merged_model:  # Should be caught earlier
                    continue

                # Save latent if needed (using data from precomputed_image_data for H/W)
                if prompt_args_item.output_type in ["latent", "latent_images"]:
                    height = current_image_data["height"]
                    width = current_image_data["width"]
                    save_latent(latent, prompt_args_item, height, width)

                all_latents.append(latent)
            except Exception as e:
                logger.error(f"Error generating latent for prompt: {prompt_args_item.prompt}. Error: {e}", exc_info=True)
                all_latents.append(None)  # Add placeholder for failed generations
                continue

    # Free DiT model
    logger.info("Releasing DiT model from memory...")
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    del shared_models_for_generate["model"]
    del dit_model
    clean_memory_on_device(device)
    synchronize_device(device)  # Ensure memory is freed before loading VAE for decoding

    # 4. Decode latents and save outputs (using vae_for_batch)
    if args.output_type != "latent":
        logger.info("Decoding latents to videos/images using batched VAE...")
        ae_for_batch.to(device)  # Move VAE to device for decoding

        for i, latent in enumerate(all_latents):
            if latent is None:  # Skip failed generations
                logger.warning(f"Skipping decoding for prompt {i+1} due to previous error.")
                continue

            current_args = all_prompt_args_list[i]
            logger.info(f"Decoding output {i+1}/{len(all_latents)} for prompt: {current_args.prompt}")

            # if args.output_type is "latent_images", we already saved latent above.
            # so we skip saving latent here.
            if current_args.output_type == "latent_images":
                current_args.output_type = "images"

            # save_output expects latent to be [BCTHW] or [CTHW]. generate returns [BCTHW] (batch size 1).
            # latent[0] is correct if generate returns it with batch dim.
            # The latent from generate is (1, C, T, H, W)
            save_output(current_args, ae_for_batch, latent[0], device)  # Pass vae_for_batch

        ae_for_batch.to("cpu")  # Move VAE back to CPU

    del ae_for_batch
    clean_memory_on_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode

    Args:
        args: Base command line arguments
    """
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}  # Initialize empty cache for interactive mode

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:  # Ctrl+D or Ctrl+Z with prompt_toolkit
                    raise EOFError  # Exit on Ctrl+D or Ctrl+Z

                # Parse prompt
                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                # Generate latent
                # For interactive, precomputed data is None. shared_models contains text/image encoders.
                # generate will load VAE internally.
                returned_vae, latent = generate(prompt_args, gen_settings, shared_models)

                # # If not one_frame_inference, move DiT model to CPU after generation
                # if prompt_args.blocks_to_swap > 0:
                #     logger.info("Waiting for 5 seconds to finish block swap")
                #     time.sleep(5)
                # model = shared_models.get("model")
                # model.to("cpu")  # Move DiT model to CPU after generation

                # Save latent and video
                # returned_vae from generate will be used for decoding here.
                save_output(prompt_args, returned_vae, latent[0], device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = None  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def main():
    # Parse arguments
    args = parse_args()

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
        # Original latent decode mode
        original_base_names = []
        latents_list = []
        seeds = []

        # assert len(args.latent_path) == 1, "Only one latent path is supported for now"

        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 5:  # [BCTHW]
                latents = latents.squeeze(0)  # [CTHW]

            latents_list.append(latents)

        # latent = torch.stack(latents_list, dim=0)  # [N, ...], must be same shape

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]

            ae = flux_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)
            save_output(args, ae, latent, device, original_base_names)

    elif args.from_file:
        # Batch mode from file

        # Read prompts from file
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_lines = f.readlines()

        # Process prompts
        prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        process_batch_prompts(prompts_data, args)

    elif args.interactive:
        # Interactive mode
        process_interactive(args)

    else:
        # Single prompt mode (original behavior)

        # Generate latent
        gen_settings = get_generation_settings(args)
        # For single mode, precomputed data is None, shared_models is None.
        # generate will load all necessary models (VAE, Text/Image Encoders, DiT).
        returned_vae, latent = generate(args, gen_settings)
        # print(f"Generated latent shape: {latent.shape}")
        # if args.save_merged_model:
        #     return

        # Save latent and video
        # returned_vae from generate will be used for decoding here.
        save_output(args, returned_vae, latent[0], device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
