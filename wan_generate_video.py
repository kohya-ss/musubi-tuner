import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
from typing import Tuple, Optional, List, Union, Any

import torch
import accelerate
from accelerate import Accelerator
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file, load_safetensors
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
from hv_generate_video import save_images_grid, save_videos_grid, synchronize_device

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--exclude_single_blocks", action="store_true", help="Exclude single blocks when loading LoRA weights")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="video length, Default depends on task")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance. Default is 5.0.",
    )
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument("--strength", type=float, default=0.75, help="Strength for video2video inference (0.0-1.0)")
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    parser.add_argument("--trim_tail_frames", type=int, default=0, help="trim tail N frames from the video before saving")
    parser.add_argument(
        "--cfg_skip_mode",
        type=str,
        default="none",
        choices=["early", "late", "middle", "early_late", "alternate", "none"],
        help="CFG skip mode. each mode skips different parts of the CFG. "
        " early: initial steps, late: later steps, middle: middle steps, early_late: both early and late, alternate: alternate, none: no skip (default)",
    )
    parser.add_argument(
        "--cfg_apply_ratio",
        type=float,
        default=None,
        help="The ratio of steps to apply CFG (0.0 to 1.0). Default is None (apply all steps).",
    )
    parser.add_argument(
        "--slg_layers", type=str, default=None, help="Skip block (layer) indices for SLG (Skip Layer Guidance), comma separated"
    )
    parser.add_argument(
        "--slg_scale",
        type=float,
        default=3.0,
        help="scale for SLG classifier free guidance. Default is 3.0. Ignored if slg_mode is None or uncond",
    )
    parser.add_argument("--slg_start", type=float, default=0.0, help="start ratio for inference steps for SLG. Default is 0.0.")
    parser.add_argument("--slg_end", type=float, default=0.3, help="end ratio for inference steps for SLG. Default is 0.3.")
    parser.add_argument(
        "--slg_mode",
        type=str,
        default=None,
        choices=["original", "uncond"],
        help="SLG mode. original: same as SD3, uncond: replace uncond pred with SLG pred",
    )

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default depends on task.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arithmetic (RTX 4XXX+), only for fp8_scaled")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
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
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        help="Torch.compile settings",
    )

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    return args


def get_task_defaults(task: str, size: Optional[Tuple[int, int]] = None) -> Tuple[int, float, int, bool]:
    """Return default values for each task

    Args:
        task: task name (t2v, t2i, i2v etc.)
        size: size of the video (width, height)

    Returns:
        Tuple[int, float, int, bool]: (infer_steps, flow_shift, video_length, needs_clip)
    """
    width, height = size if size else (0, 0)

    if "t2i" in task:
        return 50, 5.0, 1, False
    elif "i2v" in task:
        flow_shift = 3.0 if (width == 832 and height == 480) or (width == 480 and height == 832) else 5.0
        return 40, flow_shift, 81, True
    else:  # t2v or default
        return 50, 5.0, 81, False


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and set default values for optional arguments

    Args:
        args: command line arguments

    Returns:
        argparse.Namespace: updated arguments
    """
    # Get default values for the task
    infer_steps, flow_shift, video_length, _ = get_task_defaults(args.task, tuple(args.video_size))

    # Apply default values to unset arguments
    if args.infer_steps is None:
        args.infer_steps = infer_steps
    if args.flow_shift is None:
        args.flow_shift = flow_shift
    if args.video_length is None:
        args.video_length = video_length

    # Force video_length to 1 for t2i tasks
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"

    # parse slg_layers
    if args.slg_layers is not None:
        args.slg_layers = list(map(int, args.slg_layers.split(",")))

    return args


def check_inputs(args: argparse.Namespace) -> Tuple[int, int, int]:
    """Validate video size and length

    Args:
        args: command line arguments

    Returns:
        Tuple[int, int, int]: (height, width, video_length)
    """
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    if size not in SUPPORTED_SIZES[args.task]:
        logger.warning(f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")

    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

    return height, width, video_length


def calculate_dimensions(video_size: Tuple[int, int], video_length: int, config) -> Tuple[Tuple[int, int, int, int], int]:
    """calculate dimensions for the generation

    Args:
        video_size: video frame size (height, width)
        video_length: number of frames in the video
        config: model configuration

    Returns:
        Tuple[Tuple[int, int, int, int], int]:
            ((channels, frames, height, width), seq_len)
    """
    height, width = video_size
    frames = video_length

    # calculate latent space dimensions
    lat_f = (frames - 1) // config.vae_stride[0] + 1
    lat_h = height // config.vae_stride[1]
    lat_w = width // config.vae_stride[2]

    # calculate sequence length
    seq_len = math.ceil((lat_h * lat_w) / (config.patch_size[1] * config.patch_size[2]) * lat_f)

    return ((16, lat_f, lat_h, lat_w), seq_len)


def load_vae(args: argparse.Namespace, config, device: torch.device, dtype: torch.dtype) -> WanVAE:
    """load VAE model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dtype: data type for the model

    Returns:
        WanVAE: loaded VAE model
    """
    vae_path = args.vae if args.vae is not None else os.path.join(args.ckpt_dir, config.vae_checkpoint)

    logger.info(f"Loading VAE model from {vae_path}")
    cache_device = torch.device("cpu") if args.vae_cache_cpu else None
    vae = WanVAE(vae_path=vae_path, device=device, dtype=dtype, cache_device=cache_device)
    return vae


def load_text_encoder(args: argparse.Namespace, config, device: torch.device) -> T5EncoderModel:
    """load text encoder (T5) model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        T5EncoderModel: loaded text encoder model
    """
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.t5_tokenizer)

    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.t5,
        fp8=args.fp8_t5,
    )

    return text_encoder


def load_clip_model(args: argparse.Namespace, config, device: torch.device) -> CLIPModel:
    """load CLIP model (for I2V only)

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        CLIPModel: loaded CLIP model
    """
    checkpoint_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.clip_checkpoint)
    tokenizer_path = None if args.ckpt_dir is None else os.path.join(args.ckpt_dir, config.clip_tokenizer)

    clip = CLIPModel(
        dtype=config.clip_dtype,
        device=device,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        weight_path=args.clip,
    )

    return clip


def load_dit_model(
    args: argparse.Namespace,
    config,
    device: torch.device,
    dit_dtype: torch.dtype,
    dit_weight_dtype: Optional[torch.dtype] = None,
    is_i2v: bool = False,
) -> WanModel:
    """load DiT model

    Args:
        args: command line arguments
        config: model configuration
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is
        is_i2v: I2V mode

    Returns:
        WanModel: loaded DiT model
    """
    loading_device = "cpu"
    if args.blocks_to_swap == 0 and args.lora_weight is None and not args.fp8_scaled:
        loading_device = device

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled or args.lora_weight is not None:
        loading_weight_dtype = dit_dtype  # load as-is

    # do not fp8 optimize because we will merge LoRA weights
    model = load_wan_model(config, device, args.dit, args.attn_mode, False, loading_device, loading_weight_dtype, False)

    return model


def merge_lora_weights(model: WanModel, args: argparse.Namespace, device: torch.device) -> None:
    """merge LoRA weights to the model

    Args:
        model: DiT model
        args: command line arguments
        device: device to use
    """
    if args.lora_weight is None or len(args.lora_weight) == 0:
        return

    for i, lora_weight in enumerate(args.lora_weight):
        if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
            lora_multiplier = args.lora_multiplier[i]
        else:
            lora_multiplier = 1.0

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)

        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if args.include_patterns is not None and len(args.include_patterns) > i:
            include_pattern = args.include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(
                f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}"
            )
        if args.exclude_patterns is not None and len(args.exclude_patterns) > i:
            original_key_count_ex = len(weights_sd.keys())
            exclude_pattern = args.exclude_patterns[i]
            regex_exclude = re.compile(exclude_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if not regex_exclude.search(k)}
            logger.info(
                f"Filtered keys with exclude pattern {exclude_pattern}: {original_key_count_ex} -> {len(weights_sd.keys())}"
            )
        if len(weights_sd) != original_key_count:
            remaining_keys = list(set([k.split(".", 1)[0] for k in weights_sd.keys()]))
            remaining_keys.sort()
            logger.info(f"Remaining LoRA modules after filtering: {remaining_keys}")
            if len(weights_sd) == 0:
                logger.warning(f"No keys left after filtering.")

        if args.lycoris:
            lycoris_net, _ = create_network_from_weights(
                multiplier=lora_multiplier,
                file=None,
                weights_sd=weights_sd,
                unet=model,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            lycoris_net.merge_to(None, model, weights_sd, dtype=None, device=device)
        else:
            network = lora_wan.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
            network.merge_to(None, model, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info("LoRA weights loaded")

    # save model here before casting to dit_weight_dtype
    if args.save_merged_model:
        logger.info(f"Saving merged model to {args.save_merged_model}")
        mem_eff_save_file(model.state_dict(), args.save_merged_model)  # save_file needs a lot of memory
        logger.info("Merged model saved")


def optimize_model(
    model: WanModel, args: argparse.Namespace, device: torch.device, dit_dtype: torch.dtype, dit_weight_dtype: torch.dtype
) -> None:
    """optimize the model (FP8 conversion, device move etc.)

    Args:
        model: dit model
        args: command line arguments
        device: device to use
        dit_dtype: dtype for the model
        dit_weight_dtype: dtype for the model weights
    """
    if args.fp8_scaled:
        # load state dict as-is and optimize to fp8
        state_dict = model.state_dict()

        # if no blocks to swap, we can move the weights to GPU after optimization on GPU (omit redundant CPU->GPU copy)
        move_to_device = args.blocks_to_swap == 0  # if blocks_to_swap > 0, we will keep the model on CPU
        state_dict = model.fp8_optimization(state_dict, device, move_to_device, use_scaled_mm=args.fp8_fast)

        info = model.load_state_dict(state_dict, strict=True, assign=True)
        logger.info(f"Loaded FP8 optimized weights: {info}")

        if args.blocks_to_swap == 0:
            model.to(device)  # make sure all parameters are on the right device (e.g. RoPE etc.)
    else:
        # simple cast to dit_dtype
        target_dtype = None  # load as-is (dit_weight_dtype == dtype of the weights in state_dict)
        target_device = None

        if dit_weight_dtype is not None:  # in case of args.fp8 and not args.fp8_scaled
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)  # move and cast  at the same time. this reduces redundant copy operations

    if args.compile:
        compile_backend, compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
        logger.info(
            f"Torch Compiling[Backend: {compile_backend}; Mode: {compile_mode}; Dynamic: {compile_dynamic}; Fullgraph: {compile_fullgraph}]"
        )
        torch._dynamo.config.cache_size_limit = 32
        for i in range(len(model.blocks)):
            model.blocks[i] = torch.compile(
                model.blocks[i],
                backend=compile_backend,
                mode=compile_mode,
                dynamic=compile_dynamic.lower() in "true",
                fullgraph=compile_fullgraph.lower() in "true",
            )

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


def prepare_t2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for T2V

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, (arg_c, arg_null))
    """
    # Prepare inputs for T2V
    # calculate dimensions and sequence length
    (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config)
    target_shape = (16, lat_f, lat_h, lat_w)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # load text encoder
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)

    # encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)

    # free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)

    # generate noise
    noise = torch.randn(
        target_shape[0], target_shape[1], target_shape[2], target_shape[3], dtype=torch.float32, device=device, generator=seed_g
    )

    # prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}

    return noise, context, context_null, (arg_c, arg_null)


def prepare_i2v_inputs(
    args: argparse.Namespace, config, accelerator: Accelerator, device: torch.device, vae: WanVAE
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for I2V

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, used for image encoding

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, y, (arg_c, arg_null))
    """
    # get video dimensions
    height, width = args.video_size
    frames = args.video_length
    max_area = width * height

    # load image
    img = Image.open(args.image_path).convert("RGB")

    # convert to numpy
    img_cv2 = np.array(img)  # PIL to numpy

    # convert to tensor (-1 to 1)
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

    # end frame image
    if args.end_image_path is not None:
        end_img = Image.open(args.end_image_path).convert("RGB")
        end_img_cv2 = np.array(end_img)  # PIL to numpy
    else:
        end_img = None
        end_img_cv2 = None
    has_end_image = end_img is not None

    # calculate latent dimensions: keep aspect ratio
    h, w = img_tensor.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
    h = lat_h * config.vae_stride[1]
    w = lat_w * config.vae_stride[2]
    lat_f = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
    max_seq_len = (lat_f + (1 if has_end_image else 0)) * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # generate noise
    noise = torch.randn(16, lat_f + (1 if has_end_image else 0), lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=device)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # load text encoder
    text_encoder = load_text_encoder(args, config, device)
    text_encoder.model.to(device)

    # encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)

    # free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)

    # load CLIP model
    clip = load_clip_model(args, config, device)
    clip.model.to(device)

    # encode image to CLIP context
    logger.info(f"Encoding image to CLIP context")
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
        clip_context = clip.visual([img_tensor[:, None, :, :]])
    logger.info(f"Encoding complete")

    # free CLIP model and clean memory
    del clip
    clean_memory_on_device(device)

    # encode image to latent space with VAE
    logger.info(f"Encoding image to latent space")
    vae.to_device(device)

    # resize image
    interpolation = cv2.INTER_AREA if h < img_cv2.shape[0] else cv2.INTER_CUBIC
    img_resized = cv2.resize(img_cv2, (w, h), interpolation=interpolation)
    img_resized = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
    img_resized = img_resized.unsqueeze(1)  # CFHW

    if has_end_image:
        interpolation = cv2.INTER_AREA if h < end_img_cv2.shape[1] else cv2.INTER_CUBIC
        end_img_resized = cv2.resize(end_img_cv2, (w, h), interpolation=interpolation)
        end_img_resized = TF.to_tensor(end_img_resized).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
        end_img_resized = end_img_resized.unsqueeze(1)  # CFHW

    # create mask for the first frame
    # msk = torch.ones(1, frames, lat_h, lat_w, device=device)
    # msk[:, 1:] = 0
    # msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    # msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    # msk = msk.transpose(1, 2)[0]

    # rewrite to simpler version
    msk = torch.zeros(4, lat_f + (1 if has_end_image else 0), lat_h, lat_w, device=device)
    msk[:, 0] = 1
    if has_end_image:
        msk[:, -1] = 1

    # encode image to latent space
    with accelerator.autocast(), torch.no_grad():
        # padding to match the required number of frames
        padding_frames = frames - 1  # the first frame is image
        img_resized = torch.concat([img_resized, torch.zeros(3, padding_frames, h, w, device=device)], dim=1)
        y = vae.encode([img_resized])[0]

        if has_end_image:
            y_end = vae.encode([end_img_resized])[0]
            y = torch.concat([y, y_end], dim=1)  # add end frame

    y = torch.concat([msk, y])
    logger.info(f"Encoding complete")

    # move VAE to CPU
    vae.to_device("cpu")
    clean_memory_on_device(device)

    # prepare model input arguments
    arg_c = {
        "context": [context[0]],
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y],
    }

    arg_null = {
        "context": context_null,
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y],
    }

    return noise, context, context_null, y, (arg_c, arg_null)


def setup_scheduler(args: argparse.Namespace, config, device: torch.device) -> Tuple[Any, torch.Tensor]:
    """setup scheduler for sampling

    Args:
        args: command line arguments
        config: model configuration
        device: device to use

    Returns:
        Tuple[Any, torch.Tensor]: (scheduler, timesteps)
    """
    if args.sample_solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False)
        scheduler.set_timesteps(args.infer_steps, device=device, shift=args.flow_shift)
        timesteps = scheduler.timesteps
    elif args.sample_solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas = get_sampling_sigmas(args.infer_steps, args.flow_shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    elif args.sample_solver == "vanilla":
        scheduler = FlowMatchDiscreteScheduler(num_train_timesteps=config.num_train_timesteps, shift=args.flow_shift)
        scheduler.set_timesteps(args.infer_steps, device=device)
        timesteps = scheduler.timesteps

        # FlowMatchDiscreteScheduler does not support generator argument in step method
        org_step = scheduler.step

        def step_wrapper(
            model_output: torch.Tensor,
            timestep: Union[int, torch.Tensor],
            sample: torch.Tensor,
            return_dict: bool = True,
            generator=None,
        ):
            return org_step(model_output, timestep, sample, return_dict=return_dict)

        scheduler.step = step_wrapper
    else:
        raise NotImplementedError("Unsupported solver.")

    return scheduler, timesteps


# Modify the run_sampling function in wan_generate_video.py
# Replace the entire function with this implementation that properly handles tensor dimensions

def run_sampling(
    model: WanModel,
    noise: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
    args: argparse.Namespace,
    inputs: Tuple[dict, dict],
    device: torch.device,
    seed_g: torch.Generator,
    accelerator: Accelerator,
    is_i2v: bool = False,
    use_cpu_offload: bool = True,
) -> torch.Tensor:
    """run sampling
    Args:
        model: dit model
        noise: initial noise
        scheduler: scheduler for sampling
        timesteps: time steps for sampling
        args: command line arguments
        inputs: model input (arg_c, arg_null)
        device: device to use
        seed_g: random generator
        accelerator: Accelerator instance
        is_i2v: I2V mode (False means T2V mode)
        use_cpu_offload: Whether to offload tensors to CPU during processing
    Returns:
        torch.Tensor: generated latent
    """
    arg_c, arg_null = inputs

    latent = noise
    latent_storage_device = device if not use_cpu_offload else "cpu"
    latent = latent.to(latent_storage_device)

    # Get SLG parameters if available
    slg_list = getattr(args, 'slg_layers', None)
    slg_start = getattr(args, 'slg_start', 0.0)
    slg_end = getattr(args, 'slg_end', 0.3)
    total_steps = len(timesteps)

    # cfg skip
    apply_cfg_array = []
    num_timesteps = len(timesteps)

    if hasattr(args, 'cfg_skip_mode') and args.cfg_skip_mode != "none" and hasattr(args, 'cfg_apply_ratio') and args.cfg_apply_ratio is not None:
        # Calculate thresholds based on cfg_apply_ratio
        apply_steps = int(num_timesteps * args.cfg_apply_ratio)

        if args.cfg_skip_mode == "early":
            # Skip CFG in early steps, apply in late steps
            start_index = num_timesteps - apply_steps
            end_index = num_timesteps
        elif args.cfg_skip_mode == "late":
            # Skip CFG in late steps, apply in early steps
            start_index = 0
            end_index = apply_steps
        elif args.cfg_skip_mode == "early_late":
            # Skip CFG in early and late steps, apply in middle steps
            start_index = (num_timesteps - apply_steps) // 2
            end_index = start_index + apply_steps
        elif args.cfg_skip_mode == "middle":
            # Skip CFG in middle steps, apply in early and late steps
            skip_steps = num_timesteps - apply_steps
            middle_start = (num_timesteps - skip_steps) // 2
            middle_end = middle_start + skip_steps

        w = 0.0
        for step_idx in range(num_timesteps):
            if args.cfg_skip_mode == "alternate":
                # accumulate w and apply CFG when w >= 1.0
                w += args.cfg_apply_ratio
                apply = w >= 1.0
                if apply:
                    w -= 1.0
            elif args.cfg_skip_mode == "middle":
                # Skip CFG in early and late steps, apply in middle steps
                apply = step_idx < middle_start or step_idx >= middle_end
            else:
                # Apply CFG on some steps based on ratio
                apply = step_idx >= start_index and step_idx < end_index

            apply_cfg_array.append(apply)

        pattern = ["A" if apply else "S" for apply in apply_cfg_array]
        pattern = "".join(pattern)
        logger.info(f"CFG skip mode: {args.cfg_skip_mode}, apply ratio: {args.cfg_apply_ratio}, pattern: {pattern}")
    else:
        # Apply CFG on all steps
        apply_cfg_array = [True] * num_timesteps

    for i, t in enumerate(tqdm(timesteps)):
        # Determine if SLG should be applied for this step
        slg_layers_local = None
        if slg_list is not None:
            if int(slg_start * total_steps) <= i < int(slg_end * total_steps):
                slg_layers_local = slg_list

        # Process latent tensor to ensure correct dimensionality
        # Check if latent needs batch dimension adjustment
        if len(latent.shape) == 5 and latent.shape[0] == 1:  # [1, C, F, H, W]
            # Remove batch dimension to get [C, F, H, W]
            latent_for_model = latent.squeeze(0)
        else:
            latent_for_model = latent
            
        # Move to device
        latent_for_model = latent_for_model.to(device)
        
        # Provide as a list with a single tensor
        latent_model_input = [latent_for_model]
        timestep = torch.stack([t]).to(device)

        with accelerator.autocast(), torch.no_grad():
            # Get conditional prediction
            noise_pred_cond = model(
                latent_model_input, 
                t=timestep, 
                **arg_c
            )[0]
            
            if use_cpu_offload:
                noise_pred_cond = noise_pred_cond.to(latent_storage_device)

            apply_cfg = apply_cfg_array[i]  # apply CFG or not
            
            if apply_cfg:
                # Get unconditional prediction
                noise_pred_uncond = model(
                    latent_model_input, 
                    t=timestep, 
                    skip_block_indices=slg_layers_local,
                    **arg_null
                )[0]
                
                if use_cpu_offload:
                    noise_pred_uncond = noise_pred_uncond.to(latent_storage_device)
                
                # Apply guidance
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            # Check if we need to add a batch dimension for the scheduler
            if len(latent.shape) == 4:  # [C, F, H, W]
                latent_input = latent.unsqueeze(0)  # [1, C, F, H, W]
            else:
                latent_input = latent
                
            # Step
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent_input, return_dict=False, generator=seed_g)[0]

            # Update latent - preserve shape
            if len(latent.shape) == 4:  # [C, F, H, W]
                latent = temp_x0.squeeze(0)
            else:
                latent = temp_x0

    return latent


def generate(args: argparse.Namespace) -> torch.Tensor:
    """main function for generation

    Args:
        args: command line arguments

    Returns:
        torch.Tensor: generated latent
    """
    device = torch.device(args.device)

    cfg = WAN_CONFIGS[args.task]

    # select dtype
    dit_dtype = detect_wan_sd_dtype(args.dit) if args.dit is not None else torch.bfloat16
    if dit_dtype.itemsize == 1:
        # if weight is in fp8, use bfloat16 for DiT (input/output)
        dit_dtype = torch.bfloat16
        if args.fp8_scaled:
            raise ValueError(
                "DiT weights is already in fp8 format, cannot scale to fp8. Please use fp16/bf16 weights / DiTの重みはすでにfp8形式です。fp8にスケーリングできません。fp16/bf16の重みを使用してください"
            )

    dit_weight_dtype = dit_dtype  # default
    if args.fp8_scaled:
        dit_weight_dtype = None  # various precision weights, so don't cast to specific dtype
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else dit_dtype
    logger.info(
        f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}, VAE precision: {vae_dtype}"
    )

    # prepare accelerator
    mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

    # I2V or T2V or V2V
    is_i2v = "i2v" in args.task
    is_v2v = args.video_path is not None

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    # Load VAE early if we need it for video or image processing
    vae = None
    if is_v2v or is_i2v or args.image_path is not None:
        # Load VAE for encoding
        vae = load_vae(args, cfg, device, vae_dtype)
    
    # Process video for V2V if provided
    video_latents = None
    if is_v2v:
        logger.info(f"Video2Video inference: {args.video_path}")
        
        # For V2V, don't automatically extend - use the specified video_length if provided
        # Otherwise, use the natural length of the video
        if args.video_length is None:
            logger.info("No video_length specified, will use the natural length of the video")
            # Just load the video without trying to extend it
            video = load_video(args.video_path, 0, None, bucket_reso=tuple(args.video_size))
            video = np.stack(video, axis=0)  # [F, H, W, C]
            video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()  # [1, C, F, H, W]
            video = video / 255.0
            
            # Update args.video_length with the actual length
            args.video_length = video.shape[2]
            logger.info(f"Set video_length to match input video: {args.video_length} frames")
        else:
            # User specified a video_length, so use it
            logger.info(f"Using specified video_length: {args.video_length} frames")
            
            # Load the first args.video_length frames of the video
            video = load_video(args.video_path, 0, args.video_length, bucket_reso=tuple(args.video_size))
            video = np.stack(video, axis=0)  # [F, H, W, C]
            video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()  # [1, C, F, H, W]
            video = video / 255.0
            
            # Double-check that we got the right number of frames
            if video.shape[2] < args.video_length:
                logger.warning(f"Could only load {video.shape[2]} frames, less than requested {args.video_length}")
                args.video_length = video.shape[2]

        # Log the video shape before encoding
        logger.info(f"Video tensor shape before encoding: {video.shape}")

        # Encode video to latents
        logger.info(f"Encoding video to latents")
        video_latents = encode_video_to_latents(video, vae, device, dit_dtype)

        # Free memory
        del video
        clean_memory_on_device(device)

        # Prepare inputs for V2V 
        noise, context, context_null, inputs, seed_g = prepare_v2v_inputs(args, cfg, accelerator, device, vae, video_latents)
    elif is_i2v:
        # I2V: need text encoder, VAE and CLIP
        # vae was already loaded above
        noise, context, context_null, y, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
        # vae is on CPU
    else:
        # T2V: need text encoder
        noise, context, context_null, inputs = prepare_t2v_inputs(args, cfg, accelerator, device)
        vae = None

    # load DiT model
    model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)

    # merge LoRA weights
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(model, args, device)

        # if we only want to save the model, we can skip the rest
        if args.save_merged_model:
            return None

    # optimize model: fp8 conversion, block swap etc.
    optimize_model(model, args, device, dit_dtype, dit_weight_dtype)

    # setup scheduler
    scheduler, timesteps = setup_scheduler(args, cfg, device)

    # set random generator
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # Adjust timesteps for V2V if needed
    if is_v2v and args.strength < 1.0:
        # Calculate number of inference steps based on strength
        num_inference_steps = max(1, int(args.infer_steps * args.strength))
        logger.info(f"Strength: {args.strength}, adjusted inference steps: {num_inference_steps}")
        
        # Get starting timestep based on strength
        t_start_idx = len(timesteps) - num_inference_steps
        t_start = timesteps[t_start_idx]
        
        # Mix noise and video latents based on starting timestep
        t_value = t_start / 1000.0  # Normalize to 0-1 range
        logger.info(f"Mixing noise and video latents with t_value: {t_value}")
        noise = noise * t_value + video_latents * (1.0 - t_value)
        
        # Use only the required timesteps
        timesteps = timesteps[-num_inference_steps:]
        
        latent = noise  # Initialize latent with the noise+video mixture
    else:
        latent = noise  # Use regular noise for standard generation

    # run sampling
    latent = run_sampling(model, latent, scheduler, timesteps, args, inputs, device, seed_g, accelerator, is_i2v)

    # free memory
    del model
    del scheduler
    synchronize_device(device)

    # wait for 5 seconds until block swap is done
    logger.info("Waiting for 5 seconds to finish block swap")
    time.sleep(5)

    gc.collect()
    clean_memory_on_device(device)

    # save VAE model for decoding
    if vae is None:
        args._vae = None
    else:
        args._vae = vae

    return latent


def decode_latent(latent: torch.Tensor, args: argparse.Namespace, cfg) -> torch.Tensor:
    """decode latent

    Args:
        latent: latent tensor
        args: command line arguments
        cfg: model configuration

    Returns:
        torch.Tensor: decoded video or image
    """
    device = torch.device(args.device)

    # load VAE model or use the one from the generation
    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else torch.bfloat16
    if hasattr(args, "_vae") and args._vae is not None:
        vae = args._vae
    else:
        vae = load_vae(args, cfg, device, vae_dtype)

    vae.to_device(device)

    logger.info(f"Decoding video from latents: {latent.shape}")
    x0 = latent.to(device)
    
    # Ensure correct dimensionality - VAE decoder expects [C, F, H, W] or [B, C, F, H, W]
    # Reshape if needed
    if len(x0.shape) > 5:  # Too many dimensions
        x0 = x0.squeeze(0)  # Remove extra batch dimension
    elif len(x0.shape) < 5:  # Too few dimensions (shouldn't happen)
        x0 = x0.unsqueeze(0)  # Add batch dimension
        
    with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
        videos = vae.decode(x0)

    # some tail frames may be corrupted when end frame is used, we add an option to remove them
    if args.trim_tail_frames:
        videos[0] = videos[0][:, : -args.trim_tail_frames]

    logger.info(f"Decoding complete")
    video = videos[0]
    del videos
    video = video.to(torch.float32).cpu()

    return video


def save_output(
    latent: torch.Tensor, args: argparse.Namespace, cfg, height: int, width: int, original_base_names: Optional[List[str]] = None
) -> None:
    """save output

    Args:
        latent: latent tensor
        args: command line arguments
        cfg: model configuration
        height: height of frame
        width: width of frame
        original_base_names: original base names (if latents are loaded from files)
    """
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    video_length = args.video_length

    if args.output_type == "latent" or args.output_type == "both":
        # save latent
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
        logger.info(f"Latent save to: {latent_path}")

    if args.output_type == "video" or args.output_type == "both":
        # save video
        sample = decode_latent(latent.unsqueeze(0), args, cfg)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        video_path = f"{save_path}/{time_flag}_{seed}{original_name}.mp4"
        save_videos_grid(sample, video_path, fps=args.fps, rescale=True)
        logger.info(f"Sample save to: {video_path}")

    elif args.output_type == "images":
        # save images
        sample = decode_latent(latent.unsqueeze(0), args, cfg)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        image_name = f"{time_flag}_{seed}{original_name}"
        save_images_grid(sample, save_path, image_name, rescale=True)
        logger.info(f"Sample images save to: {save_path}/{image_name}")

def prepare_v2v_inputs(args, cfg, accelerator, device, vae, video_latents):
    """Prepare inputs for Video2Video inference
    
    Args:
        args (argparse.Namespace): Command line arguments
        cfg: Model configuration
        accelerator: Accelerator instance
        device (torch.device): Device to use
        vae: VAE model
        video_latents (torch.Tensor): Encoded latent representation of input video
        
    Returns:
        Tuple containing noise, context, context_null, (arg_c, arg_null), and seed_g
    """
    # Get dimensions directly from the latent
    if len(video_latents.shape) == 5:  # [B, C, F, H, W]
        batch_size, channels, lat_f, lat_h, lat_w = video_latents.shape
    else:  # [C, F, H, W]
        channels, lat_f, lat_h, lat_w = video_latents.shape
        batch_size = 1
    
    # Calculate spatial tokens per frame using the model's patch size
    patch_h, patch_w = cfg.patch_size[1], cfg.patch_size[2]
    spatial_tokens = (lat_h * lat_w) // (patch_h * patch_w)
    
    # Calculate the sequence length based on actual latent dimensions
    seq_len = spatial_tokens * lat_f
    
    # Log detailed information about latent dimensions and sequence length
    logger.info(f"Video latent dimensions: channels={channels}, frames={lat_f}, height={lat_h}, width={lat_w}")
    logger.info(f"Using patch size: {patch_h}x{patch_w}, spatial tokens per frame: {spatial_tokens}")
    logger.info(f"Calculated sequence length: {seq_len}")
    
    # Configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else cfg.sample_neg_prompt
    
    # Set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # Update args for consistency
    
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    
    # Load text encoder
    text_encoder = load_text_encoder(args, cfg, device)
    text_encoder.model.to(device)
    
    # Encode prompt
    with torch.no_grad():
        if args.fp8_t5:
            with torch.amp.autocast(device_type=device.type, dtype=cfg.t5_dtype):
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
        else:
            context = text_encoder([args.prompt], device)
            context_null = text_encoder([n_prompt], device)
    
    # Free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)
    
    # Generate noise with the same shape as video_latents
    noise = torch.randn(
        video_latents.shape,
        dtype=torch.float32, 
        device=device, 
        generator=seed_g
    )
    
    # Prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    
    return noise, context, context_null, (arg_c, arg_null), seed_g

def load_video(video_path, start_frame=0, num_frames=None, bucket_reso=(256, 256)):
    """Load video frames and resize them to the target resolution.
    
    Args:
        video_path (str): Path to the video file
        start_frame (int): First frame to load (0-indexed)
        num_frames (int, optional): Number of frames to load. If None, load all frames.
        bucket_reso (tuple): Target resolution (height, width)
        
    Returns:
        list: List of numpy arrays containing video frames in RGB format
    """
    logger.info(f"Loading video from {video_path}, frames {start_frame}-{start_frame+num_frames if num_frames else 'end'}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    
    # Get total frame count and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Video has {total_frames} frames, {fps} FPS")
    
    # Calculate how many frames to load
    if num_frames is None:
        num_frames = total_frames - start_frame
    else:
        # Make sure we don't try to load more frames than exist
        num_frames = min(num_frames, total_frames - start_frame)
    
    # Skip to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read frames
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Reached end of video after {len(frames)} frames")
            break
        
        # Convert from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize the frame
        h, w = bucket_reso
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        
        frames.append(frame)
    
    cap.release()
    logger.info(f"Successfully loaded {len(frames)} frames")
    
    return frames

def glob_images(image_dir):
    """Get list of image files in directory."""
    extensions = ['jpg', 'jpeg', 'png', 'webp', 'bmp']
    image_files = []
    
    for ext in extensions:
        image_files.extend(list(Path(image_dir).glob(f"*.{ext}")))
        image_files.extend(list(Path(image_dir).glob(f"*.{ext.upper()}")))
    
    return image_files

def extend_video_frames(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Extend video to target number of frames by repeating frames."""
    current_frames = video.shape[2]
    if current_frames >= target_frames:
        return video
    
    # Create repeat pattern with evenly distributed repeats
    base_repeats = target_frames // current_frames
    extra = target_frames % current_frames
    
    repeats = torch.full((current_frames,), base_repeats, 
                         dtype=torch.int64, device=video.device)
    repeats[:extra] += 1  # Distribute extra repeats to early frames
    
    # Create index pattern
    indices = torch.arange(current_frames, device=video.device)
    indices = indices.repeat_interleave(repeats)
    
    extended_video = torch.index_select(video, 2, indices)
    return extended_video

def load_and_extend_video(video_path, video_length, bucket_reso=(256, 256), extend=True):
    """Load video and extend it to match target length if needed and requested.
    
    Args:
        video_path: Path to input video file or directory of images
        video_length: Target video length
        bucket_reso: Resolution to resize to (height, width)
        extend: Whether to extend the video to match video_length
               False means keep original length, True means extend if needed
    
    Returns:
        torch.Tensor: Video tensor with shape [1, C, F, H, W]
    """
    # Load video or image sequence
    if os.path.isfile(video_path):
        # Load video file
        video = load_video(video_path, 0, None, bucket_reso=bucket_reso)  # Pass None to load all frames
    else:
        # Load images
        image_files = glob_images(video_path)
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {video_path}")
        
        image_files.sort()
        video = []
        for image_file in image_files:  # Don't limit to video_length yet
            image = Image.open(image_file)
            image = resize_image_to_bucket(image, bucket_reso)
            video.append(image)
    
    # Get the actual frame count
    actual_frame_count = len(video)
    logger.info(f"Loaded video with {actual_frame_count} frames")
    
    # Only extend if requested and needed
    if extend and actual_frame_count < video_length:
        logger.info(f"Video length ({actual_frame_count}) is less than target length ({video_length}). Extending video...")
        # Convert list of frames to tensor
        video_tensor = torch.from_numpy(np.stack(video, axis=0))  # [F, H, W, C]
        video_tensor = video_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1, C, F, H, W]
        
        # Extend the video
        extended_tensor = extend_video_frames(video_tensor, video_length)
        
        # Convert back to list of frames
        extended_tensor = extended_tensor.squeeze(0).permute(1, 2, 3, 0)  # [F, H, W, C]
        video = [frame.numpy() for frame in extended_tensor]
        
        logger.info(f"Video extended to {len(video)} frames")
    else:
        # If we're not extending, ensure we don't have too many frames
        if not extend and video_length is not None and actual_frame_count > video_length:
            logger.info(f"Limiting video to first {video_length} frames from {actual_frame_count} total frames")
            video = video[:video_length]
    
    # Stack video frames
    video = np.stack(video, axis=0)  # [F, H, W, C]
    video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()  # [1, C, F, H, W]
    video = video / 255.0  # Normalize to [0, 1]
    
    logger.info(f"Final video tensor shape: {video.shape}")
    return video

def encode_video_to_latents(video, vae, device, vae_dtype):
    """Encode video to latent space using VAE.
    
    Args:
        video (torch.Tensor): Video tensor with shape [B, C, F, H, W]
        vae (WanVAE): VAE model
        device (torch.device): Device to use for encoding
        vae_dtype (torch.dtype): Dtype to use for the latents
    
    Returns:
        torch.Tensor: Encoded latents with shape [B, C', F', H', W']
    """
    logger.info(f"Encoding video to latents: {video.shape}")
    
    # Move video to device with the right dtype
    video = video.to(device=device, dtype=torch.float32)
    
    # Convert from [0, 1] to [-1, 1] range expected by VAE
    video = video * 2 - 1
    
    # Move VAE to device
    vae.to_device(device)
    
    # Get video dimensions
    batch_size, channels, num_frames, height, width = video.shape
    
    # Calculate expected latent dimensions based on VAE version
    # This logic mirrors what's in hv_generate_video.py
    vae_ver = vae.model.vae_ver if hasattr(vae.model, 'vae_ver') else ""
    
    if "884" in vae_ver:
        expected_lat_f = (num_frames - 1) // 4 + 1
        logger.info(f"Using VAE version with 884 temporal stride, expecting {expected_lat_f} latent frames")
    elif "888" in vae_ver:
        expected_lat_f = (num_frames - 1) // 8 + 1
        logger.info(f"Using VAE version with 888 temporal stride, expecting {expected_lat_f} latent frames")
    else:
        expected_lat_f = num_frames
        logger.info(f"Using VAE version with no temporal stride, expecting {expected_lat_f} latent frames")
    
    # The WanVAE.encode method expects input in format [C, T, H, W] (no batch dimension)
    # We need to extract the video without batch dimension
    video_no_batch = video[0]  # Shape [C, F, H, W]
    
    with torch.no_grad():
        try:
            # Encode the video - WanVAE.encode expects a list containing the video tensor
            latents = vae.encode([video_no_batch])[0]
            
            # Log the actual latent shape for debugging
            logger.info(f"Encoded latent shape: {latents.shape}")
        except RuntimeError as e:
            logger.error(f"Error encoding video: {str(e)}")
            raise  # Re-raise the error after logging
    
    # Add batch dimension back
    latents = latents.unsqueeze(0)  # [1, C, F', H', W']
    
    # Convert to the desired dtype
    latents = latents.to(dtype=vae_dtype)
    
    logger.info(f"Final latent shape: {latents.shape}")
    
    return latents

def main():
    # 引数解析
    args = parse_args()

    # check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if not latents_mode:
        # generation mode
        # setup arguments
        args = setup_args(args)
        height, width, video_length = check_inputs(args)

        # Log generation type
        if args.video_path is not None:
            logger.info(f"Running video-to-video inference with strength {args.strength}")
        elif args.image_path is not None:
            logger.info(f"Running image-to-video inference")
        else:
            logger.info(f"Running text-to-video inference")

        logger.info(
            f"video size: {height}x{width}@{video_length} (HxW@F), fps: {args.fps}, "
            f"infer_steps: {args.infer_steps}, flow_shift: {args.flow_shift}"
        )

        # generate latent
        latent = generate(args)

        # make sure the model is freed from GPU memory
        gc.collect()
        clean_memory_on_device(args.device)

        # save latent and video
        if args.save_merged_model:
            return

        # add batch dimension
        latent = latent.unsqueeze(0)
        original_base_names = None
    else:
        # latents mode
        cfg = WAN_CONFIGS[args.task]  # any task is fine
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

        # # use the arguments TODO get from latent shape
        # height, width = args.video_size
        # video_length = args.video_length
        cfg = WAN_CONFIGS[args.task]
        height = latents.shape[-2]
        width = latents.shape[-1]
        height *= cfg.patch_size[1] * cfg.vae_stride[1]
        width *= cfg.patch_size[2] * cfg.vae_stride[2]
        video_length = latents.shape[1]
        video_length = (video_length - 1) * cfg.vae_stride[0] + 1
        args.seed = seeds[0]

    # decode and save
    cfg = WAN_CONFIGS[args.task]
    save_output(latent[0], args, cfg, height, width, original_base_names)

    logger.info("Done!")


if __name__ == "__main__":
    main()
