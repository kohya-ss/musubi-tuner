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

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.flux_2 import flux2_models
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.lora_utils import filter_lora_state_dict

lycoris_available = find_spec("lycoris") is not None

from musubi_tuner.networks import lora_flux_2
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.hv_generate_video import get_time_flag, save_images_grid, setup_parser_compile, synchronize_device
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
    parser = argparse.ArgumentParser(description="FLUX.2 inference script")

    # WAN arguments
    # parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    # parser.add_argument(
    #     "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    # )

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument(
        "--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors. Default is False."
    )
    parser.add_argument("--vae", type=str, default=None, help="AE directory or path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text Encoder Mistral 3/Qwen 3 directory or path")

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
    parser.add_argument(
        "--guidance_scale", type=float, default=4.0, help="Guidance scale for classifier free guidance. Default is 4.0."
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, default is None (` ` for non-distilled model)",
    )

    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width")
    parser.add_argument(
        "--control_image_path",
        nargs="*",
        type=str,
        default=None,
        help="path to control (reference) image(s) for Flux 2 image edit",
    )
    parser.add_argument(
        "--no_resize_control", action="store_true", help="Do not resize control image (default is to resize if too large)"
    )
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 25")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    # parser.add_argument(
    #     "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    # )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=4.0,
        help="Embeded CFG scale (distilled CFG Scale), default is 4.0. All klein models ignore this.",
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
        help="Shift factor for flow matching schedulers. Default is None (FLUX.2 default).",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")

    parser.add_argument("--fp8_text_encoder", action="store_true", help="use fp8 for Text Encoder (Mistral 3)")
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
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping, which may speed up data transfer between CPU and GPU but uses more shared GPU memory on Windows",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
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

    flux2_utils.add_model_version_args(parser)

    args = parser.parse_args()

    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")

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
    if line.strip().startswith("--"):  # No prompt
        parts = (" " + line.strip()).split(" --")
        prompt = None
    else:
        parts = line.split(" --")
        prompt = parts[0].strip()
        parts = parts[1:]

    # Create dictionary of overrides
    overrides = {} if prompt is None else {"prompt": prompt}
    overrides["control_image_path"] = []

    for part in parts:
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
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        # elif option == "im":
        #     overrides["image_mask_path"] = value
        # elif option == "cn":
        #     overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "ci":  # control_image_path
            overrides["control_image_path"].append(value)

    # If no control_image_path was provided, remove the empty list
    if not overrides["control_image_path"]:
        del overrides["control_image_path"]

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


def load_dit_model(
    args: argparse.Namespace, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
) -> flux2_models.Flux2:
    """load DiT model

    Args:
        args: command line arguments
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is

    Returns:
        flux2_models.Flux2: DiT model
    """
    # If LyCORIS is enabled, we will load the model to CPU and then merge LoRA weights (static method)

    loading_device = "cpu"
    if args.blocks_to_swap == 0 and not args.lycoris:
        loading_device = device

    # load LoRA weights
    if not args.lycoris and args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
            lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled and not args.lycoris:
        loading_weight_dtype = None  # we will load weights as-is and then optimize to fp8
    elif args.lycoris:
        loading_weight_dtype = torch.bfloat16  # lycoris requires bfloat16 or float16, because it merges weights

    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
    model = flux2_utils.load_flow_model(
        device,
        model_version_info,
        args.dit,
        args.attn_mode,
        False,
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.lycoris,
        lora_weights_list,
        args.lora_multiplier,
        args.disable_numpy_memmap,
    )

    # merge LoRA weights
    if args.lycoris:
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            merge_lora_weights(
                lora_flux_2,
                model,
                args.lora_weight,
                args.lora_multiplier,
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
            # state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

            from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

            # inplace optimization
            state_dict = optimize_state_dict_with_fp8(
                state_dict,
                device,
                flux2_models.FP8_OPTIMIZATION_TARGET_KEYS,
                flux2_models.FP8_OPTIMIZATION_EXCLUDE_KEYS,
                move_to_device=move_to_device,
            )
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)  # args.scaled_mm)

            info = model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Loaded FP8 optimized weights: {info}")

    # if we only want to save the model, we can skip the rest of the setup but still return the model
    if args.save_merged_model:
        return model

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
        model = model_utils.compile_transformer(
            args, model, [model.double_blocks, model.single_blocks], disable_linear=args.blocks_to_swap > 0
        )

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


def decode_latent(ae: flux2_models.AutoEncoder, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    logger.info("Decoding image...")
    if latent.ndim == 3:
        latent = latent.unsqueeze(0)  # add batch dimension

    ae.to(device)
    with torch.no_grad():
        pixels = ae.decode(latent.to(device, ae.dtype))  # decode to pixels
    pixels = pixels.to("cpu")
    ae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def prepare_image_inputs(
    args: argparse.Namespace, device: torch.device, ae: flux2_models.AutoEncoder
) -> Tuple[int, int, Optional[List[torch.Tensor]]]:
    """Prepare image-related inputs for FLUX.2: AE encoding."""
    height, width = check_inputs(args)

    if args.control_image_path is not None and len(args.control_image_path):
        limit_size = (1024, 1024) if len(args.control_image_path) > 1 else (2024, 2024)
        if args.no_resize_control:
            limit_size = None

        img_ctx_prep = []
        for image_path in args.control_image_path:
            image_tensor, _, _ = flux2_utils.preprocess_control_image(image_path, limit_size)
            img_ctx_prep.append(image_tensor)

        # AE encoding
        logger.info("Encoding control image to latent space with AE")
        ae_original_device = ae.device
        ae.to(device)

        control_latent = []
        with torch.no_grad():
            # Encode each reference image
            for img in img_ctx_prep:
                encoded = ae.encode(img.to(device, dtype=ae.dtype))[0]  # C, H, W
                control_latent.append(encoded.to(torch.bfloat16).to("cpu"))

        ae.to(ae_original_device)  # Move VAE back to its original device
        clean_memory_on_device(device)
    else:
        control_latent = None

    return height, width, control_latent


def prepare_text_inputs(
    args: argparse.Namespace, device: torch.device, shared_models: Optional[Dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare text-related inputs for I2V: LLM and TextEncoder encoding."""
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

    # load text encoder: conds_cache holds cached encodings for prompts without padding
    conds_cache = {}
    if shared_models is not None:
        text_embedder = shared_models.get("text_embedder")
        if "conds_cache" in shared_models:  # Use shared cache if available
            conds_cache = shared_models["conds_cache"]
        # text_encoder is on device (batched inference) or CPU (interactive inference)
    else:  # Load if not in shared_models
        te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
        text_embedder = flux2_utils.load_text_embedder(
            model_version_info, args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True
        )

    # Store original devices to move back later if they were shared. This does nothing if shared_models is None
    text_encoder_original_device = text_embedder.device if text_embedder else None

    logger.info("Encoding prompt with Text Encoders")

    # Ensure text_encoder is not None before proceeding
    if not text_embedder:
        raise ValueError("Text embedder is not loaded properly.")

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

        text_embedder.to(device)

    prompt = args.prompt
    if prompt in conds_cache:
        ctx_vec = conds_cache[prompt]
    else:
        move_models_to_device_if_needed()

        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            ctx_vec = text_embedder([prompt])  # [1, 512, 15360]
        ctx_vec = ctx_vec.cpu()
        conds_cache[prompt] = ctx_vec

    negative_prompt = args.negative_prompt
    negative_ctx_vec = None
    if not model_version_info.guidance_distilled:
        if negative_prompt is None:
            negative_prompt = " "  # for non-distilled model, use empty string as negative prompt
        if negative_prompt in conds_cache:
            negative_ctx_vec = conds_cache[negative_prompt]
        else:
            move_models_to_device_if_needed()

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                negative_ctx_vec = text_embedder([negative_prompt])  # [1, 512, 15360]
            negative_ctx_vec = negative_ctx_vec.cpu()
            conds_cache[negative_prompt] = negative_ctx_vec

    if not (shared_models and "text_embedder" in shared_models):  # if loaded locally
        del text_embedder
    else:  # if shared, move back to original device (likely CPU)
        if text_embedder:
            text_embedder.to(text_encoder_original_device)

    gc.collect()  # Force cleanup of Text Encoder from GPU memory
    clean_memory_on_device(device)

    arg_c = {"ctx_vec": ctx_vec, "prompt": prompt}
    if negative_ctx_vec is None:
        arg_null = None
    else:
        arg_null = {"ctx_vec": negative_ctx_vec, "prompt": negative_prompt}

    return arg_c, arg_null


def prepare_i2v_inputs(
    args: argparse.Namespace, device: torch.device, ae: flux2_models.AutoEncoder, shared_models: Optional[Dict] = None
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
    height, width, control_latent = prepare_image_inputs(args, device, ae)

    # prepare text inputs
    ctx_nctx = prepare_text_inputs(args, device, shared_models)

    return height, width, ctx_nctx, control_latent


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_image_data: Optional[tuple[int, int, Optional[torch.Tensor]]] = None,
    precomputed_text_data: Optional[tuple[Dict, Dict]] = None,
) -> tuple[Optional[flux2_models.AutoEncoder], torch.Tensor]:  # AE can be Optional
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models (mainly for DiT)
        precomputed_image_data: Optional tuple with precomputed image data (height, width, control_latent)
        precomputed_text_data: Optional tuple with precomputed text data (context, context_null)

    Returns:
        tuple: (flux2_models.AutoEncoder model (vae) or None, torch.Tensor generated latent)
    """
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)
    vae_instance_for_return = None

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    if precomputed_image_data is not None and precomputed_text_data is not None:
        logger.info("Using precomputed image and text data.")
        height, width, control_latent = precomputed_image_data
        ctx_nctx = precomputed_text_data

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
            vae_instance_for_return = flux2_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)

        height, width, ctx_nctx, control_latent = prepare_i2v_inputs(args, device, vae_instance_for_return, shared_models)

        vae_instance_for_return.to("cpu")

    context, context_null = ctx_nctx  # unpack
    if shared_models is None or "model" not in shared_models:
        # load DiT model
        model = load_dit_model(args, device, dit_weight_dtype)

        if args.save_merged_model:
            return None, None

        if shared_models is not None:
            shared_models["model"] = model
    else:
        # use shared model
        model: flux2_models.Flux = shared_models["model"]
        model.move_to_device_except_swap_blocks(device)  # Handles block swap correctly
        model.prepare_block_swap_before_forward()

    # set random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")

    # image generation ######
    logger.info(f"Prompt: {context['prompt']}, Negative Prompt: {context_null['prompt'] if context_null is not None else 'N/A'}")
    ctx_vec = context["ctx_vec"].to(device, dtype=torch.bfloat16)
    ctx, ctx_ids = flux2_utils.prc_txt(ctx_vec)
    if context_null is None:
        negative_ctx_vec = None
        ctx_null, ctx_null_ids = None, None
    else:
        negative_ctx_vec = context_null["ctx_vec"].to(device, dtype=torch.bfloat16)
        ctx_null, ctx_null_ids = flux2_utils.prc_txt(negative_ctx_vec)

    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = height // 16, width // 16
    noise_dtype = torch.float32
    noise = torch.randn(1, 128, packed_latent_height, packed_latent_width, dtype=noise_dtype, generator=seed_g, device="cpu").to(
        device, dtype=torch.bfloat16
    )
    x, x_ids = flux2_utils.prc_img(noise)

    # prompt upsampling is not supported

    if control_latent is not None:
        ref_tokens, ref_ids = flux2_utils.pack_control_latent(control_latent)
        del control_latent  # free memory
        ref_tokens = ref_tokens.to(device, dtype=torch.bfloat16)
        ref_ids = ref_ids.to(device)
    else:
        ref_tokens = None
        ref_ids = None

    # denoise
    timesteps = flux2_utils.get_schedule(args.infer_steps, x.shape[1], args.flow_shift)
    if model_version_info.guidance_distilled:
        x = flux2_utils.denoise(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            timesteps=timesteps,
            guidance=args.embedded_cfg_scale,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )
    else:
        x = flux2_utils.denoise_cfg(
            model,
            x,
            x_ids,
            ctx,
            ctx_ids,
            ctx_null,
            ctx_null_ids,
            timesteps=timesteps,
            guidance=args.guidance_scale,
            img_cond_seq=ref_tokens,
            img_cond_seq_ids=ref_ids,
        )
    x = torch.cat(flux2_utils.scatter_ids(x, x_ids)).squeeze(2)
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
            "guidance_scale": f"{args.guidance_scale}",
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
    sample = sample.to(torch.float32)  # convert to float32 for numpy conversion
    save_images_grid(sample, save_path, image_name, rescale=True, create_subdir=False)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    ae: flux2_models.AutoEncoder,  # Expect a VAE instance for decoding
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
    height *= 16
    width *= 16
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
    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]

    # Load text encoders to CPU
    m3_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    text_embedder = flux2_utils.load_text_embedder(
        model_version_info, args.text_encoder, dtype=m3_dtype, device="cpu", disable_mmap=True
    )
    shared_models["text_embedder"] = text_embedder

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

    model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
    gen_settings = get_generation_settings(args)
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Precompute Image Data (AE and Image Encoders)
    logger.info("Loading AE and Image Encoders for batch image preprocessing...")
    ae_for_batch = flux2_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)

    all_precomputed_image_data = []
    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]  # Create all arg instances first

    logger.info("Preprocessing images and AE encoding for all prompts...")

    # AE and Image Encoder to device for this phase, because we do not want to offload them to CPU
    ae_for_batch.to(device)

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Image preprocessing for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
        # prepare_image_inputs will move ae/image_encoder to device temporarily
        image_data = prepare_image_inputs(prompt_args_item, device, ae_for_batch)
        all_precomputed_image_data.append(image_data)

    # Models should be back on GPU because prepare_image_inputs moved them to the original device
    ae_for_batch.to("cpu")  # Move AE back to CPU
    clean_memory_on_device(device)

    # 2. Precompute Text Data (Text Encoder)
    logger.info("Loading Text Encoder for batch text preprocessing...")

    # Text Encoders loaded to CPU by load_text_encoder
    m3_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    text_embedder_batch = flux2_utils.load_text_embedder(
        model_version_info, args.text_encoder, dtype=m3_dtype, device=device, disable_mmap=True
    )

    # Text Encoders to device for this phase
    text_embedder_batch.to(device)  # Moved into prepare_text_inputs logic

    all_precomputed_text_data = []
    conds_cache_batch = {}

    logger.info("Preprocessing text and LLM/TextEncoder encoding for all prompts...")
    temp_shared_models_txt = {
        "text_embedder": text_embedder_batch,  # on GPU
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
        # prepare_text_inputs will move text_encoders to device temporarily
        ctx_nctx = prepare_text_inputs(prompt_args_item, device, temp_shared_models_txt)
        all_precomputed_text_data.append(ctx_nctx)

    # Models should be removed from device after prepare_text_inputs
    del text_embedder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()  # Force cleanup of Text Encoder from GPU memory
    clean_memory_on_device(device)

    # 3. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    # Use args from the first prompt for DiT loading (LoRA etc. should be consistent for a batch)
    first_prompt_args = all_prompt_args_list[0]
    dit_model = load_dit_model(first_prompt_args, device, dit_weight_dtype)  # Load directly to target device if possible

    if first_prompt_args.lora_weight is not None and len(first_prompt_args.lora_weight) > 0:
        logger.info("Merging LoRA weights into DiT model...")
        merge_lora_weights(
            lora_flux_2,
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
            gc.collect()  # Force cleanup of DiT from GPU memory
            clean_memory_on_device(device)
            return

    shared_models_for_generate = {"model": dit_model}  # Pass DiT via shared_models

    all_latents = []

    logger.info("Generating latents for all prompts...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_image_data = all_precomputed_image_data[i]
            current_text_data = all_precomputed_text_data[i]

            logger.info(f"Generating latent for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
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
                    height, width, _ = current_image_data
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
    gc.collect()  # Force cleanup of DiT from GPU memory
    clean_memory_on_device(device)
    synchronize_device(device)  # Ensure memory is freed before loading VAE for decoding

    # 4. Decode latents and save outputs (using vae_for_batch)
    if args.output_type != "latent":
        logger.info("Decoding latents to videos/images using batched VAE...")
        ae_for_batch.to(device)  # Move VAE to device for decoding

        for i, latent in enumerate(all_latents):
            if latent is None:  # Skip failed generations
                logger.warning(f"Skipping decoding for prompt {i + 1} due to previous error.")
                continue

            current_args = all_prompt_args_list[i]
            logger.info(f"Decoding output {i + 1}/{len(all_latents)} for prompt: {current_args.prompt}")

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

    dit_weight_dtype = torch.bfloat16  # default
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

            ae = flux2_utils.load_ae(args.vae, dtype=torch.float32, device=device, disable_mmap=True)
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

        if args.blocks_to_swap > 0:
            logger.info("Waiting for 5 seconds to finish block swap")
            time.sleep(5)
        gc.collect()  # Force cleanup of DiT from GPU memory
        clean_memory_on_device(device)  # clean memory on device before moving models

        # Save latent and video
        # returned_vae from generate will be used for decoding here.
        save_output(args, returned_vae, latent[0], device)

    logger.info("Done!")


if __name__ == "__main__":
    main()
