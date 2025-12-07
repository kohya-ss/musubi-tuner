import argparse
import gc
from importlib.util import find_spec
import random
import os
import time
import copy
from typing import Tuple, Optional, List, Any, Dict

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, SiglipImageProcessor, SiglipVisionModel, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Stack

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.frame_pack.clip_vision import hf_clip_vision_encode
from musubi_tuner.hunyuan_video_1_5 import hunyuan_video_1_5_text_encoder, hunyuan_video_1_5_utils, hunyuan_video_1_5_vae
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_models import (
    detect_hunyuan_video_1_5_sd_dtype,
    load_hunyuan_video_1_5_model,
    HunyuanVideo_1_5_DiffusionTransformer,
)
from musubi_tuner.hunyuan_video_1_5.hunyuan_video_1_5_vae import AutoencoderKLConv3D
from musubi_tuner.wan_generate_video import merge_lora_weights
from musubi_tuner.frame_pack.framepack_utils import load_image_encoders
from musubi_tuner.networks import lora_wan
from musubi_tuner.qwen_image import qwen_image_utils
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.lora_utils import filter_lora_state_dict

lycoris_available = find_spec("lycoris") is not None

from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device
from musubi_tuner.hv_generate_video import get_time_flag, save_images_grid, save_videos_grid, setup_parser_compile

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(
        self, device: torch.device, dit_dtype: torch.dtype, dit_weight_dtype: Optional[torch.dtype], vae_dtype: torch.dtype
    ):
        self.device = device
        self.dit_dtype = dit_dtype
        self.dit_weight_dtype = dit_weight_dtype  # may be None if fp8_scaled, may be float8 if fp8 not scaled
        self.vae_dtype = vae_dtype


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="HunyuanVideo-1.5 inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument(
        "--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors. Default is False."
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument(
        "--vae_dtype",
        type=str,
        default=None,
        help="data type for VAE, default is float16. If VRAM is sufficient, use float32 for better quality.",
    )
    parser.add_argument(
        "--vae_sample_size",
        type=int,
        default=128,
        help="VAE sample size (height and width). Default is 128. If VRAM is sufficient, set to 256 for better quality. Set to 0 to disable tiling.",
    )
    parser.add_argument(
        "--vae_enable_patch_conv", action="store_true", help="Enable patch-based convolution in VAE for memory optimization"
    )
    parser.add_argument("--text_encoder", type=str, default=None, help="Text encoder checkpoint path (Qwen2.5-VL)")
    parser.add_argument("--text_encoder_cpu", action="store_true", help="Load text encoder on CPU to save GPU memory")
    parser.add_argument("--byt5", type=str, default=None, help="BYT5 text encoder checkpoint path")
    parser.add_argument("--image_encoder", type=str, default=None, help="Image Encoder directory or path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier")
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
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="video length, required")
    parser.add_argument("--fps", type=int, default=24, help="video fps, Default is 24")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=6.0, help="Guidance scale for classifier free guidance. Default is 6.0."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to image for image2video inference. If not specified, text2video is used.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=7.0,
        help="Shift factor for flow matching schedulers. Default ",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    # parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled")
    # parser.add_argument("--fp8_vlm", action="store_true", help="use fp8 for vision-language model")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping, which may speed up data transfer between CPU and GPU but uses more shared GPU memory on Windows",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="video",
        choices=["video", "images", "latent", "both", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument(
        "--lycoris", action="store_true", help=f"use lycoris for inference{'' if lycoris_available else ' (not available)'}"
    )
    setup_parser_compile(parser)

    # New arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.prompt is None and not args.from_file and not args.interactive and args.latent_path is None:
        raise ValueError("Either --prompt, --from_file, --interactive, or --latent_path must be specified")

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

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
            overrides["video_size_width"] = int(value)
        elif option == "h":
            overrides["video_size_height"] = int(value)
        elif option == "f":
            overrides["video_length"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value

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
        if key == "video_size_width":
            args_copy.video_size[1] = value
        elif key == "video_size_height":
            args_copy.video_size[0] = value
        else:
            setattr(args_copy, key, value)

    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, int]: (height, width, video_length)
    """
    height = args.video_size[0]
    width = args.video_size[1]

    video_length = args.video_length

    if height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

    return height, width, video_length


def load_vae(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> AutoencoderKLConv3D:
    """load VAE model

    Args:
        args: command line arguments
        device: device to use
        dtype: data type for the model

    Returns:
        AutoencoderKLConv3D: loaded VAE model
    """
    vae_path = args.vae

    logger.info(f"Loading VAE model from {vae_path}")
    vae = hunyuan_video_1_5_vae.load_vae_from_checkpoint(
        vae_path, device, dtype, sample_size=args.vae_sample_size, enable_patch_conv=args.vae_enable_patch_conv
    )
    vae.eval()
    return vae


def load_vision_encoder(args: argparse.Namespace, config, device: torch.device) -> tuple[SiglipImageProcessor, SiglipVisionModel]:
    return load_image_encoders(args)


def load_text_encoders(args: argparse.Namespace) -> tuple[Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, T5Tokenizer, T5Stack]:
    """load text encoder (T5) model

    Args:
        args: command line arguments
    Returns:
        tuple[Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration, T5Tokenizer, T5Stack]: loaded text encoder models
    """
    vl_dtype = torch.bfloat16
    tokenizer_vlm, text_encoder_vlm = qwen_image_utils.load_qwen2_5_vl(
        args.text_encoder, dtype=vl_dtype, device="cpu", disable_mmap=True
    )
    tokenizer_byt5, text_encoder_byt5 = hunyuan_video_1_5_text_encoder.load_byt5(
        args.byt5, dtype=torch.float16, device="cpu", disable_mmap=True
    )
    return tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5


def load_dit_model(
    args: argparse.Namespace,
    dit_path: str,
    lora_weights: List[str],
    lora_multipliers: List[float],
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> HunyuanVideo_1_5_DiffusionTransformer:
    """load DiT model

    Args:
        args: command line arguments
        dit_path: path to the DiT checkpoint
        lora_weights: path to the LoRA weights
        lora_multipliers: multiplier for the LoRA weights
        device: device to use
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        HunyuanVideo_1_5_DiffusionTransformer: loaded DiT model
    """

    # If LyCORIS is enabled, we will load the model to CPU and then merge LoRA weights (static method)

    loading_device = "cpu"
    if args.blocks_to_swap == 0 and not args.lycoris:
        loading_device = device

    # load LoRA weights
    if not args.lycoris and lora_weights is not None and len(lora_weights) > 0:
        lora_weights_list = []
        for lora_weight in lora_weights:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled and not args.lycoris:
        loading_weight_dtype = None  # we will load weights as-is and then optimize to fp8

    model = load_hunyuan_video_1_5_model(
        device,
        "i2v" if args.image_path is not None else "t2v",
        dit_path,
        args.attn_mode,
        False,
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.lycoris,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
    )

    # merge LoRA weights
    if args.lycoris:
        if lora_weights is not None and len(lora_weights) > 0:
            merge_lora_weights(
                lora_wan,
                model,
                lora_weights,
                lora_multipliers,
                args.include_patterns,
                args.exclude_patterns,
                device,
                lycoris=True,
                save_merged_model=args.save_merged_model,
            )

        if args.fp8_scaled:
            # load state dict as-is and optimize to fp8
            state_dict = model.state_dict()

            # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
            move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
            state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

            info = model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Loaded FP8 optimized weights: {info}")

    # if we only want to save the model, we can skip the rest
    if args.save_merged_model:
        return None

    if not args.fp8_scaled:
        # simple cast to dit_weight_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(
            args.blocks_to_swap, device, supports_backward=False, use_pinned_memory=args.use_pinned_memory_for_block_swap
        )
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        # make sure the model is on the right device
        model.to(device)

    if args.compile:
        model = model_utils.compile_transformer(args, model, [model.double_blocks], disable_linear=args.blocks_to_swap > 0)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


def prepare_i2v_or_t2v_inputs(
    args: argparse.Namespace,
    device: torch.device,
    vae: Optional[AutoencoderKLConv3D] = None,
    shared_models: Optional[dict[str, torch.nn.Module]] = None,
) -> Tuple[torch.Tensor, dict, dict]:
    """Prepare inputs for I2V or T2V tasks

    Args:
        args: command line arguments
        device: device to use
        vae: VAE model, used for image encoding
        encoded_context: Pre-encoded text context

    Returns:
        Tuple[torch.Tensor, Tuple[dict, dict]]:
            (cond_latents, (arg_c, arg_null))
    """
    is_i2v = args.image_path is not None

    # get video dimensions
    height, width, frames = check_inputs(args)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else ""

    if shared_models is None:
        # load text encoder
        tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5 = load_text_encoders(args)
        vlm_original_device = None
        byt5_original_device = None
    else:
        tokenizer_vlm = shared_models.get("tokenizer_vlm")
        text_encoder_vlm = shared_models.get("text_encoder_vlm")
        vlm_original_device = text_encoder_vlm.device
        tokenizer_byt5 = shared_models.get("tokenizer_byt5")
        text_encoder_byt5 = shared_models.get("text_encoder_byt5")
        byt5_original_device = text_encoder_byt5.device

    vl_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_vlm.to(vl_device)
    text_encoder_byt5.to(device)

    with torch.no_grad():
        embed, mask = hunyuan_video_1_5_text_encoder.get_qwen_prompt_embeds(tokenizer_vlm, text_encoder_vlm, args.prompt)
        embed_byt5, mask_byt5 = hunyuan_video_1_5_text_encoder.get_glyph_prompt_embeds(
            tokenizer_byt5, text_encoder_byt5, args.prompt
        )
        negative_embed, negative_mask = hunyuan_video_1_5_text_encoder.get_qwen_prompt_embeds(
            tokenizer_vlm, text_encoder_vlm, n_prompt
        )
        # use empty negative prompt for BYT5 as in official code
        negative_embed_byt5, negative_mask_byt5 = hunyuan_video_1_5_text_encoder.get_glyph_prompt_embeds(
            tokenizer_byt5, text_encoder_byt5, ""
        )

        # move to CPU to free GPU memory
        embed = embed.to("cpu")
        mask = mask.to("cpu")
        embed_byt5 = embed_byt5.to("cpu")
        mask_byt5 = mask_byt5.to("cpu")
        negative_embed = negative_embed.to("cpu")
        negative_mask = negative_mask.to("cpu")
        negative_embed_byt5 = negative_embed_byt5.to("cpu")
        negative_mask_byt5 = negative_mask_byt5.to("cpu")

    # free text encoder and clean memory. if shared_models is not None, we need to move the models back to original device
    if shared_models is not None:
        text_encoder_vlm.to(vlm_original_device)
        text_encoder_byt5.to(byt5_original_device)
    # remove references but do not free if shared_models is not None
    del tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5

    if shared_models is None or vlm_original_device != device:
        clean_memory_on_device(device)

    # calculate latent dimensions
    lat_h = height // 16
    lat_w = width // 16
    lat_f = 1 + (frames - 1) // 4  # number of latent frames

    if is_i2v:
        # load image
        img = Image.open(args.image_path).convert("RGB")

        # resize image keeping aspect ratio
        img_np = image_video_dataset.resize_image_to_bucket(img, (width, height))  # numpy HWC

        # convert to tensor (-1 to 1)
        img_tensor = TF.to_tensor(img_np).sub_(0.5).div_(0.5).to(device)
        img_tensor = img_tensor[None, :, None, :, :]  # BCFHW, B=1, F=1

        if shared_models is not None:
            feature_extractor = shared_models.get("feature_extractor")
            image_encoder = shared_models.get("image_encoder")
            image_encoder_original_device = image_encoder.device
        else:
            # load vision model
            feature_extractor, image_encoder = load_image_encoders(args)
            image_encoder_original_device = None
        image_encoder.to(device)

        with torch.no_grad():
            image_encoder_output = hf_clip_vision_encode(img_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state  # float16
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to("cpu")

        logger.info("Encoding complete")

        # free vision model and clean memory. if shared_models is not None, we need to move the model back to original device
        if shared_models is not None:
            image_encoder.to(image_encoder_original_device)
        del feature_extractor, image_encoder

        if shared_models is None or image_encoder_original_device != device:
            clean_memory_on_device(device)

        # encode image to latent space with VAE
        logger.info("Encoding image to latent space")
        vae_original_device = vae.device
        vae.to(device)

        # encode image to latent space
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=True), torch.no_grad():
            cond_latents = vae.encode(img_tensor)[0].mode()
            cond_latents = cond_latents * vae.scaling_factor
            cond_latents = cond_latents.to("cpu")

        logger.info("Encoding complete")

        # prepare mask for image latent
        latent_mask = torch.zeros(1, 1, lat_f, lat_h, lat_w, device="cpu")
        latent_mask[0, 0, 0, :, :] = 1.0  # first frame is image

        latents_concat = torch.zeros(
            1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS, lat_f, lat_h, lat_w, dtype=torch.float32, device="cpu"
        )
        latents_concat[:, :, 0:1, :, :] = cond_latents

        cond_latents = torch.concat([latents_concat, latent_mask], dim=1)

        vae.to(vae_original_device)
        if vae_original_device != device:
            clean_memory_on_device(device)
    else:
        # T2V mode
        image_encoder_last_hidden_state = None
        cond_latents = torch.zeros(
            1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS + 1, lat_f, lat_h, lat_w, dtype=torch.float32, device="cpu"
        )

    context = (embed, mask, embed_byt5, mask_byt5, image_encoder_last_hidden_state, cond_latents)
    context_null = (
        negative_embed,
        negative_mask,
        negative_embed_byt5,
        negative_mask_byt5,
        image_encoder_last_hidden_state,
        cond_latents,
    )

    # prepare model input arguments
    max_seq_len = lat_f * lat_h * lat_w
    arg_c = {
        "context": context,
        "seq_len": max_seq_len,
        "cond_latents": cond_latents,
    }
    arg_null = {
        "context": context_null,
        "seq_len": max_seq_len,
        "cond_latents": cond_latents,
    }
    return arg_c, arg_null


def generate(
    args: argparse.Namespace, gen_settings: GenerationSettings, shared_models: Optional[Dict] = None
) -> tuple[torch.Tensor, Optional[int]]:
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models and encoded data

    Returns:
        tuple[torch.Tensor, Optional[int]]: (latent tensor, one frame inference index)
    """
    device, dit_dtype, dit_weight_dtype, vae_dtype = (
        gen_settings.device,
        gen_settings.dit_dtype,
        gen_settings.dit_weight_dtype,
        gen_settings.vae_dtype,
    )

    # I2V or T2V
    is_i2v = args.image_path is not None

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    # Check if we have shared models
    if shared_models is not None:
        # Use shared models and encoded data if available
        vae = shared_models.get("vae")  # may be None for T2V
        if "encoded_context" in shared_models:
            arg_c, arg_null = shared_models["encoded_context"]
            logger.info("Using pre-encoded context from shared models")
        else:
            # prepare inputs
            if is_i2v:
                arg_c, arg_null = prepare_i2v_or_t2v_inputs(args, device, vae, shared_models)
            else:
                arg_c, arg_null = prepare_i2v_or_t2v_inputs(args, device, vae, shared_models)
    else:
        # prepare inputs without shared models
        if is_i2v:
            vae = load_vae(args, device, vae_dtype)
        else:
            vae = None
        arg_c, arg_null = prepare_i2v_or_t2v_inputs(args, device, vae)

    if vae is not None:
        vae.to("cpu")

    # load DiT models
    if shared_models is not None and "model" in shared_models:
        model = shared_models["model"]
        logger.info("Using pre-loaded DiT model from shared models")
    else:
        model = load_dit_model(args, args.dit, args.lora_weight, args.lora_multiplier, device, dit_weight_dtype)

    # if we only want to save the model, we can skip the rest
    if args.save_merged_model:
        return None

    # setup timesteps
    timesteps, sigmas = hunyuan_video_1_5_utils.get_timesteps_sigmas(args.infer_steps, args.flow_shift, device)

    # set random generator
    seed_g = torch.Generator(device=device if not args.cpu_noise else "cpu")
    seed_g.manual_seed(seed)

    # prepare noise
    height, width, video_length = check_inputs(args)
    lat_f = 1 + (video_length - 1) // 4  # number of latent frames
    lat_h = height // 16
    lat_w = width // 16
    noise_shape = (1, hunyuan_video_1_5_vae.VAE_LATENT_CHANNELS, lat_f, lat_h, lat_w)

    if not args.cpu_noise:
        noise = torch.randn(noise_shape, generator=seed_g, device=device, dtype=dit_dtype)
    else:
        noise = torch.randn(noise_shape, generator=seed_g, device="cpu", dtype=dit_dtype)
    noise = noise.to(device)

    # run sampling
    logger.info("Starting generation...")

    # Unpack arguments
    embed, mask, embed_byt5, mask_byt5, image_encoder_last_hidden_state, cond_latents = arg_c["context"]
    cond_latents = arg_c["cond_latents"]
    (
        negative_embed,
        negative_mask,
        negative_embed_byt5,
        negative_mask_byt5,
        image_encoder_last_hidden_state_null,
        cond_latents_null,
    ) = arg_null["context"]
    cond_latents_null = arg_null["cond_latents"]

    # 6. Denoising loop
    do_cfg = args.guidance_scale != 1.0

    latents = noise.to(dit_dtype)
    cond_latents = cond_latents.to(device, dit_dtype)
    cond_latents_null = cond_latents_null.to(device, dit_dtype)

    embed = embed.to(device, dit_dtype)
    mask = mask.to(device)
    embed_byt5 = embed_byt5.to(device, dit_dtype)
    mask_byt5 = mask_byt5.to(device)
    if image_encoder_last_hidden_state is not None:
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(device, dit_dtype)

    negative_embed = negative_embed.to(device, dit_dtype)
    negative_mask = negative_mask.to(device)
    negative_embed_byt5 = negative_embed_byt5.to(device, dit_dtype)
    negative_mask_byt5 = negative_mask_byt5.to(device)
    if image_encoder_last_hidden_state_null is not None:
        image_encoder_last_hidden_state_null = image_encoder_last_hidden_state_null.to(device, dit_dtype)

    with tqdm(total=len(timesteps), desc="Denoising steps") as pbar:
        for i, t in enumerate(timesteps):
            timestep = t.expand(latents.shape[0])  # keep dtype as float32 for better precision; avoid bfloat16 precision issues
            latents_concat = torch.cat([latents, cond_latents], dim=1)
            with torch.autocast(device_type=device.type, dtype=dit_dtype), torch.no_grad():
                noise_pred = model(
                    hidden_states=latents_concat,
                    timestep=timestep,
                    text_states=embed,
                    encoder_attention_mask=mask,
                    vision_states=image_encoder_last_hidden_state,
                    byt5_text_states=embed_byt5,
                    byt5_text_mask=mask_byt5,
                    rotary_pos_emb_cache=None,
                )

            if do_cfg:
                latents_concat = torch.cat([latents, cond_latents_null], dim=1)
                with torch.autocast(device_type=device.type, dtype=dit_dtype), torch.no_grad():
                    neg_noise_pred = model(
                        hidden_states=latents_concat,
                        timestep=timestep,
                        text_states=negative_embed,
                        encoder_attention_mask=negative_mask,
                        vision_states=image_encoder_last_hidden_state_null,
                        byt5_text_states=negative_embed_byt5,
                        byt5_text_mask=negative_mask_byt5,
                        rotary_pos_emb_cache=None,
                    )

                noise_pred = neg_noise_pred + args.guidance_scale * (noise_pred - neg_noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            # latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = hunyuan_video_1_5_utils.step(latents, noise_pred, sigmas, i)

            pbar.update()

    # Only clean up shared models if they were created within this function
    if shared_models is None:
        # free memory
        del model
        synchronize_device(device)

        # wait for 5 seconds until block swap is done
        if args.blocks_to_swap > 0:
            logger.info("Waiting for 5 seconds to finish block swap")
            time.sleep(5)

        gc.collect()
        clean_memory_on_device(device)

        # save VAE model for decoding
        if vae is None:
            args._vae = None
        else:
            args._vae = vae

    return latents


def decode_latent(latent: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """decode latent

    Args:
        latent: latent tensor
        args: command line arguments

    Returns:
        torch.Tensor: decoded video or image
    """
    device = torch.device(args.device)

    # load VAE model or use the one from the generation
    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else torch.float16
    if hasattr(args, "_vae") and args._vae is not None:
        vae = args._vae
    else:
        vae = load_vae(args, device, vae_dtype)

    vae.to(device)

    x0 = latent.to(device)
    x0 = x0 / vae.scaling_factor

    logger.info(f"Decoding video from latents: {latent.shape}")
    with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
        videos = vae.decode(x0)[0]

    video = videos[0]
    del videos
    video = video.to(torch.float32).cpu()

    logger.info("Decoding complete")
    return video


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    """Save latent to file

    Args:
        latent: latent tensor
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
    video_length = args.video_length
    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "video_length": f"{video_length}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_video(video: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    """Save video to file

    Args:
        video: Video tensor
        args: command line arguments
        original_base_name: Original base name (if latents are loaded from files)

    Returns:
        str: Path to saved video file
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    video_path = f"{save_path}/{time_flag}_{seed}{original_name}.mp4"

    video = video.unsqueeze(0)
    save_videos_grid(video, video_path, fps=args.fps, rescale=True)
    logger.info(f"Video saved to: {video_path}")

    return video_path


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
    sample = sample.unsqueeze(0)
    one_frame_inference = sample.shape[2] == 1  # check if one frame inference is used
    save_images_grid(sample, save_path, image_name, rescale=True, create_subdir=not one_frame_inference)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    latent: torch.Tensor, args: argparse.Namespace, height: int, width: int, original_base_names: Optional[List[str]] = None
) -> None:
    """save output

    Args:
        latent: latent tensor
        args: command line arguments
        height: height of frame
        width: width of frame
        original_base_names: original base names (if latents are loaded from files)
    """
    if args.output_type == "latent" or args.output_type == "both" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)

    if args.output_type == "video" or args.output_type == "both":
        # save video
        sample = decode_latent(latent.unsqueeze(0), args)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_video(sample, args, original_name)

    elif args.output_type == "images":
        # save images
        sample = decode_latent(latent.unsqueeze(0), args)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_images(sample, args, original_name)


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


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    """Process multiple prompts with model reuse

    Args:
        prompts_data: List of prompt data dictionaries
        args: Base command line arguments
    """
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    # 1. Load configuration
    gen_settings = get_generation_settings(args)
    device, dit_dtype, dit_weight_dtype, vae_dtype = (
        gen_settings.device,
        gen_settings.dit_dtype,
        gen_settings.dit_weight_dtype,
        gen_settings.vae_dtype,
    )
    is_i2v = args.image_path is not None

    # 2. Encode all prompts, and 3. Process I2V additional encodings if needed
    logger.info("Loading VAE, text encoders and image processors to encode all prompts/images")
    if is_i2v:
        vae = load_vae(args, device, vae_dtype)
        feature_extractor, image_encoder = load_image_encoders(args)
        image_encoder.to(device)
    else:
        vae = None
        feature_extractor = None
        image_encoder = None
    tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5 = load_text_encoders(args)
    vl_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_vlm.to(vl_device)
    text_encoder_byt5.to(device)

    shared_models = {
        "tokenizer_vlm": tokenizer_vlm,
        "text_encoder_vlm": text_encoder_vlm,
        "tokenizer_byt5": tokenizer_byt5,
        "text_encoder_byt5": text_encoder_byt5,
        "vae": vae,
        "feature_extractor": feature_extractor if is_i2v else None,
        "image_encoder": image_encoder if is_i2v else None,
    }
    encoded_contexts = {}

    with torch.no_grad():
        for prompt_data in prompts_data:
            prompt = prompt_data["prompt"]
            prompt_args = apply_overrides(args, prompt_data)
            arg_c, arg_null = prepare_i2v_or_t2v_inputs(prompt_args, device, vae, shared_models)
            encoded_contexts[prompt] = {"context": arg_c, "context_null": arg_null}

    # Free tokenizers and text encoders and clean memory
    del tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5
    del feature_extractor, image_encoder

    if vae is not None:
        vae.to("cpu")
    synchronize_device(device)

    gc.collect()
    clean_memory_on_device(device)

    # 4. Load DiT model, 5. Merge LoRA weights if needed, and 6. Optimize model
    logger.info("Loading DiT model(s)")
    model = load_dit_model(args, args.dit, args.lora_weight, args.lora_multiplier, device, dit_weight_dtype)

    if args.save_merged_model:
        logger.info("Model merged and saved. Exiting.")
        return

    # Create shared models dict for generate function
    shared_models.update({"vae": vae, "model": model, "encoded_contexts": encoded_contexts})

    # 7. Generate for each prompt
    all_latents = []
    all_prompt_args = []

    for i, prompt_data in enumerate(prompts_data):
        logger.info(f"Processing prompt {i + 1}/{len(prompts_data)}: {prompt_data['prompt'][:50]}...")

        # Apply overrides for this prompt
        prompt_args = apply_overrides(args, prompt_data)

        # Generate latent
        latent = generate(prompt_args, gen_settings, shared_models)

        # Save latent if needed
        height, width, _ = check_inputs(prompt_args)
        if prompt_args.output_type == "latent" or prompt_args.output_type == "both" or prompt_args.output_type == "latent_images":
            save_latent(latent, prompt_args, height, width)

        all_latents.append(latent)
        all_prompt_args.append(prompt_args)

    # 8. Free DiT model
    del model, shared_models
    synchronize_device(device)

    # wait for 5 seconds until block swap is done
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    # 9. Decode latents if needed
    if args.output_type != "latent":
        logger.info("Decoding latents to videos/images")
        if vae is None:
            vae = load_vae(args, device, vae_dtype)
        vae.to(device)

        for i, (latent, prompt_args) in enumerate(zip(all_latents, all_prompt_args)):
            logger.info(f"Decoding output {i + 1}/{len(all_latents)}")

            # Decode latent
            video = decode_latent(latent, prompt_args)

            # Save as video or images
            if prompt_args.output_type == "video" or prompt_args.output_type == "both":
                save_video(video, prompt_args)
            elif prompt_args.output_type == "images" or prompt_args.output_type == "latent_images":
                save_images(video, prompt_args)

        # Free VAE
        del vae

    clean_memory_on_device(device)
    gc.collect()


def process_interactive(args: argparse.Namespace) -> None:
    """Process prompts in interactive mode

    Args:
        args: Base command line arguments
    """
    gen_settings = get_generation_settings(args)
    device, dit_dtype, dit_weight_dtype, vae_dtype = (
        gen_settings.device,
        gen_settings.dit_dtype,
        gen_settings.dit_weight_dtype,
        gen_settings.vae_dtype,
    )
    is_i2v = args.image_path is not None

    # Initialize models to None
    shared_models = None
    vae = None
    model = None

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

                # Ensure we have all the models we need

                # 1. Load text encoder if not already loaded. All models except DiT are kept in CPU after use
                if shared_models is None:
                    logger.info("Loading VAE and text encoders")
                    vae = load_vae(args, "cpu", vae_dtype)

                    logger.info("Loading text encoders")
                    tokenizer_vlm, text_encoder_vlm, tokenizer_byt5, text_encoder_byt5 = load_text_encoders(args)
                    text_encoder_vlm.to("cpu")
                    text_encoder_byt5.to("cpu")

                    if is_i2v:
                        logger.info("Loading image encoders")
                        feature_extractor, image_encoder = load_image_encoders(args)
                        image_encoder.to("cpu")
                    else:
                        feature_extractor = None
                        image_encoder = None
                    shared_models = {
                        "tokenizer_vlm": tokenizer_vlm,
                        "text_encoder_vlm": text_encoder_vlm,
                        "tokenizer_byt5": tokenizer_byt5,
                        "text_encoder_byt5": text_encoder_byt5,
                        "feature_extractor": feature_extractor,
                        "image_encoder": image_encoder,
                    }

                # 2. Encode prompt. Models are moved back to original device after encoding
                with torch.no_grad():
                    arg_c, arg_null = prepare_i2v_or_t2v_inputs(prompt_args, device, vae, shared_models)

                # 3. Load DiT model if not already loaded
                if model is None:
                    logger.info("Loading DiT model")
                    model = load_dit_model(args, args.dit, args.lora_weight, args.lora_multiplier, device, dit_weight_dtype)
                else:
                    # Move model to GPU if it was offloaded
                    if args.blocks_to_swap > 0:
                        model.move_to_device_except_swap_blocks(device)
                        model.prepare_block_swap_before_forward()
                    else:
                        model.to(device)

                # Create shared models dict
                shared_models.update({"vae": vae, "model": model, "encoded_context": (arg_c, arg_null)})

                # Generate latent
                latent = generate(prompt_args, gen_settings, shared_models)

                # Move model to CPU after generation
                model.to("cpu")
                clean_memory_on_device(device)

                # Save latent if needed
                height, width, _ = check_inputs(prompt_args)
                if (
                    prompt_args.output_type == "latent"
                    or prompt_args.output_type == "both"
                    or prompt_args.output_type == "latent_images"
                ):
                    save_latent(latent, prompt_args, height, width)

                # Decode and save output
                if prompt_args.output_type != "latent":
                    if vae is None:
                        vae = load_vae(args, device)

                    vae.to(device)
                    video = decode_latent(latent, prompt_args)

                    if prompt_args.output_type == "video" or prompt_args.output_type == "both":
                        save_video(video, prompt_args)
                    elif prompt_args.output_type == "images" or prompt_args.output_type == "latent_images":
                        save_images(video, prompt_args)

                    # Move VAE to CPU after use
                    vae.to("cpu")

                clean_memory_on_device(device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")

    # Clean up all models
    if model is not None:
        del model
    if shared_models is not None:
        del shared_models
    if vae is not None:
        del vae

    gc.collect()
    clean_memory_on_device(device)


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    # select dtype: auto-detect from DiT model
    dit_dtype = detect_hunyuan_video_1_5_sd_dtype(args.dit)
    if dit_dtype == torch.float32:
        dit_dtype = torch.bfloat16  # use bfloat16 instead of float32 for better performance

    dit_weight_dtype = dit_dtype  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else torch.float16
    logger.info(
        f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}, VAE precision: {vae_dtype}"
    )

    gen_settings = GenerationSettings(device=device, dit_dtype=dit_dtype, dit_weight_dtype=dit_weight_dtype, vae_dtype=vae_dtype)
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

        assert len(args.latent_path) == 1, "Only one latent path is supported for now"

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
                    args.video_size = [height, width]
                if "video_length" in metadata:
                    args.video_length = int(metadata["video_length"])

            seeds.append(seed)
            latents_list.append(latents)

            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

        latent = torch.stack(latents_list, dim=0)  # [N, ...], must be same shape

        height = latents.shape[-2]
        width = latents.shape[-1]
        height *= 16
        width *= 16
        args.seed = seeds[0]

        save_output(latent[0], args, height, width, original_base_names)

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
        height, width, video_length = check_inputs(args)

        logger.info(
            f"Video size: {height}x{width}@{video_length} (HxW@F), fps: {args.fps}, "
            f"infer_steps: {args.infer_steps}, flow_shift: {args.flow_shift}"
        )

        # Generate latent
        gen_settings = get_generation_settings(args)
        latent = generate(args, gen_settings)

        # Make sure the model is freed from GPU memory
        gc.collect()
        clean_memory_on_device(args.device)

        # Save latent and video
        if args.save_merged_model:
            return

        save_output(latent[0], args, height, width)

    logger.info("Done!")


if __name__ == "__main__":
    main()
