import argparse
import gc
from importlib.util import find_spec
import random
import os
import re
import time
import math
import copy
from types import ModuleType
from typing import Tuple, Optional, List, Union, Any, Dict

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

from musubi_tuner.dataset import image_video_dataset
from musubi_tuner.networks import lora_wan
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
from musubi_tuner.wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from musubi_tuner.wan.modules.vae import WanVAE
from musubi_tuner.wan.modules.t5 import T5EncoderModel
from musubi_tuner.wan.modules.clip import CLIPModel
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from musubi_tuner.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    from lycoris.kohya import create_network_from_weights

from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device
from musubi_tuner.hv_generate_video import get_time_flag, save_images_grid, save_videos_grid, setup_parser_compile
from musubi_tuner.dataset.image_video_dataset import load_video

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GenerationSettings:
    def __init__(
        self, device: torch.device, cfg, dit_dtype: torch.dtype, dit_weight_dtype: Optional[torch.dtype], vae_dtype: torch.dtype
    ):
        self.device = device
        self.cfg = cfg
        self.dit_dtype = dit_dtype
        self.dit_weight_dtype = dit_weight_dtype  # may be None if fp8_scaled, may be float8 if fp8 not scaled
        self.vae_dtype = vae_dtype


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
    parser.add_argument("--dit_high_noise", type=str, default=None, help="DiT checkpoint path for high noise (optional)")
    parser.add_argument(
        "--force_v2_1_time_embedding", action="store_true", help="Force using 2.1 style time embedding even for Wan 2.2"
    )
    parser.add_argument("--offload_inactive_dit", action="store_true", help="Offload DiT model to CPU")
    parser.add_argument("--lazy_loading", action="store_true", help="Enable lazy loading for DiT models")
    parser.add_argument(
        "--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors. Default is False."
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier")
    parser.add_argument("--lora_weight_high_noise", type=str, nargs="*", default=None, help="LoRA weight path for high noise")
    parser.add_argument("--lora_multiplier_high_noise", type=float, nargs="*", default=None, help="LoRA multiplier for high noise")
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
    parser.add_argument("--video_length", type=int, default=None, help="video length, Default depends on task")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps, default depends on task")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    )
    parser.add_argument(
        "--timestep_boundary",
        type=float,
        default=None,
        help="Timestep boundary for guidance (0.0 to 1.0). Default depends on task.",
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=None, help="Guidance scale for classifier free guidance. Default depends on task."
    )
    parser.add_argument(
        "--guidance_scale_high_noise",
        type=float,
        default=None,
        help="Guidance scale for classifier free guidance in high noise model. Default depends on task.",
    )
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    parser.add_argument("--end_image_path", type=str, default=None, help="path to end image for image2video inference")
    parser.add_argument(
        "--control_path",
        type=str,
        default=None,
        help="path to control video for inference with controlnet. video file or directory with images",
    )
    parser.add_argument(
        "--one_frame_inference",
        type=str,
        default=None,
        help="one frame inference, default is None, comma separated values from 'no_2x', 'no_4x', 'no_post', 'control_indices' and 'target_index'.",
    )
    parser.add_argument(
        "--control_image_path", type=str, default=None, nargs="*", help="path to control (reference) image for one frame inference."
    )
    parser.add_argument(
        "--control_image_mask_path",
        type=str,
        default=None,
        nargs="*",
        help="path to control (reference) image mask for one frame inference.",
    )
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
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        # default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        default=None,
        help="[Deprecated] Torch.compile settings. Use individual arguments instead.",
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
    # Initialize control_image_path and control_image_mask_path as a list to accommodate multiple paths
    overrides["control_image_path"] = []
    overrides["control_image_mask_path"] = []

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
        elif option == "ei":
            overrides["end_image_path"] = value
        elif option == "cn":
            overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value
        # one frame inference options
        elif option == "ci":  # control_image_path
            overrides["control_image_path"].append(value)
        elif option == "cim":  # control_image_mask_path
            overrides["control_image_mask_path"].append(value)
        elif option == "of":  # one_frame_inference
            overrides["one_frame_inference"] = value

    # If no control_image_path was provided, remove the empty list
    if not overrides["control_image_path"]:
        del overrides["control_image_path"]
    if not overrides["control_image_mask_path"]:
        del overrides["control_image_mask_path"]

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


def get_task_defaults(task: str, size: Optional[Tuple[int, int]] = None) -> Tuple[int, float, int, bool]:
    """Return default values for each task

    Args:
        task: task name (t2v, t2i, i2v etc.)
        size: size of the video (width, height)

    Returns:
        Tuple[int, Optional[float], float, float, float, int, bool]: (infer_steps, boundary, flow_shift, guidance_scale, guidance_scale_high_noise, video_length, needs_clip)
    """
    width, height = size if size else (0, 0)

    cfg = WAN_CONFIGS[task]

    infer_steps = cfg.sample_steps
    boundary = cfg.boundary  # may be None
    flow_shift = cfg.sample_shift
    guidance_scale = cfg.sample_guide_scale[0]
    guidance_scale_high_noise = cfg.sample_guide_scale[1] if len(cfg.sample_guide_scale) > 1 else guidance_scale

    video_length = 1 if "t2i" in task else 81  # default video length for t2i is 1, for others is 81

    if not cfg.v2_2:
        # Wan2.1
        needs_clip = "i2v" in task
        if "i2v" in task and ((width == 832 and height == 480) or (width == 480 and height == 832)):
            #  I2V
            flow_shift = 3.0
    else:
        # Wan2.2
        needs_clip = False

    return infer_steps, boundary, flow_shift, guidance_scale, guidance_scale_high_noise, video_length, needs_clip


def setup_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate and set default values for optional arguments

    Args:
        args: command line arguments

    Returns:
        argparse.Namespace: updated arguments
    """
    # Get default values for the task
    infer_steps, boundary, flow_shift, guidance_scale, guidance_scale_high_noise, video_length, needs_clip = get_task_defaults(
        args.task, tuple(args.video_size)
    )

    # Apply default values to unset arguments
    if args.infer_steps is None:
        args.infer_steps = infer_steps
    if args.timestep_boundary is None:
        args.timestep_boundary = boundary
    if args.flow_shift is None:
        args.flow_shift = flow_shift
    if args.guidance_scale is None:
        args.guidance_scale = guidance_scale
    if args.guidance_scale_high_noise is None:
        args.guidance_scale_high_noise = guidance_scale_high_noise
    if args.video_length is None:
        args.video_length = video_length

    # Force video_length to 1 for t2i tasks
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"
    if args.timestep_boundary is not None:
        if args.timestep_boundary > 1.0:
            logger.warning(
                f"timestep_boundary {args.timestep_boundary} is greater than 1.0, setting to {args.timestep_boundary / 1000.0}"
            )
            args.timestep_boundary = args.timestep_boundary / 1000.0

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


def load_dit_models(
    args: argparse.Namespace,
    config,
    device: torch.device,
    dit_dtype: torch.dtype,
    dit_weight_dtype: Optional[torch.dtype] = None,
    is_i2v: bool = False,
) -> List[WanModel]:
    """load DiT models

    Returns:
        List[WanModel]: loaded DiT models. If args.dit_high_noise is specified, returns a list with two models: high noise and low noise.
    """
    use_high_model = args.dit_high_noise is not None and len(args.dit_high_noise) > 0
    if use_high_model and args.lazy_loading:
        logger.info("Using lazy loading")
        return [None, None]  # lazy loading will load models on demand

    model = load_dit_model(args, args.dit, args.lora_weight, args.lora_multiplier, config, device, dit_weight_dtype)

    if use_high_model:
        if args.offload_inactive_dit:
            logger.warning("Offloading low noise DiT model to CPU, high noise DiT will be loaded on GPU")
            model.to("cpu")

        logger.info(f"Loading high noise DiT model from {args.dit_high_noise}")
        dit_high_noise = load_dit_model(
            args,
            args.dit_high_noise,
            args.lora_weight_high_noise,
            args.lora_multiplier_high_noise,
            config,
            device,
            dit_weight_dtype,
        )
        return [dit_high_noise, model]

    return [model]


def load_dit_model(
    args: argparse.Namespace,
    dit_path: str,
    lora_weights: List[str],
    lora_multipliers: List[float],
    config,
    device: torch.device,
    dit_weight_dtype: Optional[torch.dtype] = None,
) -> WanModel:
    """load DiT model

    Args:
        args: command line arguments
        dit_path: path to the DiT checkpoint
        lora_weights: path to the LoRA weights
        lora_multipliers: multiplier for the LoRA weights
        config: model configuration
        device: device to use
        dit_dtype: data type for the model
        dit_weight_dtype: data type for the model weights. None for as-is
        is_i2v: I2V mode

    Returns:
        WanModel: loaded DiT model
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

    model = load_wan_model(
        config,
        device,
        dit_path,
        args.attn_mode,
        False,
        loading_device,
        loading_weight_dtype,
        args.fp8_scaled and not args.lycoris,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        use_scaled_mm=args.fp8_fast,
        disable_numpy_memmap=args.disable_numpy_memmap,
    )
    if args.force_v2_1_time_embedding:
        model.set_time_embedding_v2_1(True)

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
        if args.compile_args is not None:
            # deprecated
            args.compile_backend, args.compile_mode, compile_dynamic, compile_fullgraph = args.compile_args
            args.compile_dynamic = compile_dynamic.lower()
            args.compile_fullgraph = compile_fullgraph.lower() in "true"
            args.compile_cache_size_limit = 32  # old default value

        model = model_utils.compile_transformer(args, model, [model.blocks], disable_linear=args.blocks_to_swap > 0)

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


def merge_lora_weights(
    lora_module: ModuleType,
    model: torch.nn.Module,
    lora_weights: List[str],
    lora_multipliers: List[float],
    include_patterns: Optional[List[str]],
    exclude_patterns: Optional[List[str]],
    device: torch.device,
    lycoris: bool = False,
    save_merged_model: Optional[str] = None,
    converter: Optional[callable] = None,
) -> None:
    """merge LoRA weights to the model

    Args:
        lora_module: LoRA module, e.g. lora_wan
        model: DiT model
        lora_weights: paths to LoRA weights
        lora_multipliers: multipliers for LoRA weights
        include_patterns: regex patterns to include LoRA modules
        exclude_patterns: regex patterns to exclude LoRA modules
        device: torch.device
        lycoris: use LyCORIS
        save_merged_model: path to save merged model, if specified, no inference will be performed
        converter: Optional[callable] = None
    """
    if lora_weights is None or len(lora_weights) == 0:
        return

    for i, lora_weight in enumerate(lora_weights):
        if lora_multipliers is not None and len(lora_multipliers) > i:
            lora_multiplier = lora_multipliers[i]
        else:
            lora_multiplier = 1.0

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
        weights_sd = load_file(lora_weight)
        if converter is not None:
            weights_sd = converter(weights_sd)

        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if include_patterns is not None and len(include_patterns) > i:
            include_pattern = include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")
        if exclude_patterns is not None and len(exclude_patterns) > i:
            original_key_count_ex = len(weights_sd.keys())
            exclude_pattern = exclude_patterns[i]
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
                logger.warning("No keys left after filtering.")

        if lycoris:
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
            network = lora_module.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
            network.merge_to(None, model, weights_sd, device=device, non_blocking=True)

        synchronize_device(device)
        logger.info("LoRA weights loaded")

    # save model here before casting to dit_weight_dtype
    if save_merged_model:
        logger.info(f"Saving merged model to {save_merged_model}")
        mem_eff_save_file(model.state_dict(), save_merged_model)  # save_file needs a lot of memory
        logger.info("Merged model saved")


def prepare_t2v_inputs(
    args: argparse.Namespace,
    config,
    accelerator: Accelerator,
    device: torch.device,
    vae: Optional[WanVAE] = None,
    encoded_context: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Tuple[dict, dict]]:
    """Prepare inputs for T2V

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model for control video encoding
        encoded_context: Pre-encoded text context

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
            (noise, context, context_null, (arg_c, arg_null))
    """
    # Prepare inputs for T2V
    # calculate dimensions and sequence length
    height, width = args.video_size
    frames = args.video_length
    (_, lat_f, lat_h, lat_w), seq_len = calculate_dimensions(args.video_size, args.video_length, config)
    target_shape = (16, lat_f, lat_h, lat_w)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
        seed_g = torch.manual_seed(seed)

    if encoded_context is None:
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
    else:
        # Use pre-encoded context
        context = encoded_context["context"]
        context_null = encoded_context["context_null"]

    # Fun-Control: encode control video to latent space
    if config.is_fun_control:
        # TODO use same resizing as for image
        logger.info("Encoding control video to latent space")
        # C, F, H, W
        control_video = load_control_video(args.control_path, frames, height, width).to(device)
        vae.to_device(device)
        with torch.autocast(device_type=device.type, dtype=vae.dtype), torch.no_grad():
            control_latent = vae.encode([control_video])[0]
        y = torch.concat([control_latent, torch.zeros_like(control_latent)], dim=0)  # add control video latent
        vae.to_device("cpu")
    else:
        y = None

    # generate noise
    noise = torch.randn(target_shape, dtype=torch.float32, generator=seed_g, device=device if not args.cpu_noise else "cpu")
    noise = noise.to(device)

    # prepare model input arguments
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    if y is not None:
        arg_c["y"] = [y]
        arg_null["y"] = [y]

    return noise, (arg_c, arg_null)


def parse_one_frame_inference_args(one_frame_inference_arg: str) -> Tuple[int, List[int], List[int], int]:
    """Parse one frame inference arguments"""
    one_frame_inference = set()
    for mode in one_frame_inference_arg.split(","):
        one_frame_inference.add(mode.strip())

    target_index = 0
    control_indices = []
    for one_frame_param in one_frame_inference:
        if one_frame_param.startswith("target_index="):
            target_index = int(one_frame_param.split("=")[1])
            logger.info(f"Set index for target: {target_index}")
        elif one_frame_param.startswith("control_index="):
            control_indices = one_frame_param.split("=")[1].split(";")
            control_indices = [int(idx) for idx in control_indices]
            logger.info(f"Set control indices: {control_indices}")

    target_and_control_latent_indices = control_indices + [target_index]
    f_indices = sorted(target_and_control_latent_indices)

    one_frame_inference_index = f_indices.index(target_index)

    return target_index, control_indices, f_indices, one_frame_inference_index


def prepare_one_frame_inference(
    args: argparse.Namespace,
    accelerator: Accelerator,
    vae: WanVAE,
    device: torch.device,
    lat_h: int,
    lat_w: int,
    height: int,
    width: int,
) -> Tuple[int, torch.Tensor, List[int]]:
    target_index, _, f_indices, one_frame_inference_index = parse_one_frame_inference_args(args.one_frame_inference)

    # prepare image
    def preprocess_image(image_path: str):
        image = Image.open(image_path)
        if image.mode == "RGBA":
            alpha = image.split()[-1]
        else:
            alpha = None
        image = image.convert("RGB")

        image_np = np.array(image)  # PIL to numpy, HWC

        image_np = image_video_dataset.resize_image_to_bucket(image_np, (width, height))
        image_tensor = torch.from_numpy(image_np).float() / 127.5 - 1.0  # -1 to 1.0, HWC
        image_tensor = image_tensor.permute(2, 0, 1)[:, None]  # HWC -> CHW -> CFHW, C=3, F=1
        return image_tensor, image_np, alpha

    # check control images
    control_image_tensors = []
    control_mask_images = []
    if args.control_image_path is not None and len(args.control_image_path) > 0:
        for ctrl_image_path in args.control_image_path:
            control_image_tensor, _, control_mask = preprocess_image(ctrl_image_path)
            control_image_tensors.append(control_image_tensor)
            control_mask_images.append(control_mask)

    # TODO mask is not supported yet

    vae.to_device(device)

    with accelerator.autocast(), torch.no_grad():
        black_image_latent = vae.encode([torch.zeros((3, 1, height, width), dtype=torch.float32, device=device)])[0]

    control_latents = []
    if control_image_tensors is not None:
        # encode image to latent space with VAE
        logger.info("Encoding image to latent space")

        for ctrl_image_tensor in control_image_tensors:
            # encode image one by one
            with accelerator.autocast(), torch.no_grad():
                control_latent = vae.encode([ctrl_image_tensor.to(device)])[0]
            control_latents.append(control_latent)

    vae.to_device("cpu")

    lat_f = 1 + (len(control_latents) if control_latents is not None else 0)

    # Create latent and mask for the required number of frames
    y = torch.zeros(4 + 16, lat_f, lat_h, lat_w, dtype=torch.float32, device=device)
    ci = 0
    for j, index in enumerate(f_indices):
        if index == target_index:
            y[4:, j : j + 1, :, :] = black_image_latent  # set target latent to black image
        else:
            y[:4, j, :, :] = 1.0  # set mask to 1.0 for the clean latent frames
            y[4:, j : j + 1, :, :] = control_latents[ci]  # set control latent
            ci += 1

    return one_frame_inference_index, y, f_indices


def prepare_i2v_inputs(
    args: argparse.Namespace,
    config,
    accelerator: Accelerator,
    device: torch.device,
    vae: WanVAE,
    encoded_context: Optional[Dict] = None,
) -> Tuple[torch.Tensor, Tuple[dict, dict], Optional[int]]:
    """Prepare inputs for I2V

    Args:
        args: command line arguments
        config: model configuration
        accelerator: Accelerator instance
        device: device to use
        vae: VAE model, used for image encoding
        encoded_context: Pre-encoded text context

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
    additional_frames = 1 if has_end_image and not config.flf2v else 0

    # calculate latent dimensions: keep aspect ratio
    height, width = img_tensor.shape[1:]
    aspect_ratio = height / width
    lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
    height = lat_h * config.vae_stride[1]
    width = lat_w * config.vae_stride[2]
    lat_f = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
    max_seq_len = (lat_f + additional_frames) * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
        seed_g = torch.manual_seed(seed)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    if encoded_context is None:
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
        clip_context = None
        if not config.v2_2:
            clip = load_clip_model(args, config, device)
            clip.model.to(device)

            # encode image to CLIP context
            logger.info("Encoding image to CLIP context")
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                clip_context = clip.visual([img_tensor[:, None, :, :]])
                # I2V end image is not officially supported, so no additional CLIP context
                if end_img is not None and config.flf2v:
                    end_img_tensor = TF.to_tensor(end_img).sub_(0.5).div_(0.5).to(device)
                    end_clip_context = clip.visual([end_img_tensor[:, None, :, :]])
                    clip_context = torch.concat([clip_context, end_clip_context], dim=0)
            logger.info("Encoding complete")

            # free CLIP model and clean memory
            del clip
            clean_memory_on_device(device)
    else:
        # Use pre-encoded context
        context = encoded_context["context"]
        context_null = encoded_context["context_null"]
        clip_context = encoded_context["clip_context"]

    # check if one frame inference is enabled
    if args.one_frame_inference is not None:
        if has_end_image and not config.flf2v:
            logger.warning("One frame inference with end image is not supported other than FLF2V")
        one_frame_inference_index, y, f_indices = prepare_one_frame_inference(
            args, accelerator, vae, device, lat_h, lat_w, height, width
        )
        max_seq_len = len(f_indices) * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])
    else:
        one_frame_inference_index, f_indices = None, None

        # encode image to latent space with VAE
        logger.info("Encoding image to latent space")
        vae.to_device(device)

        # resize image
        interpolation = cv2.INTER_AREA if height < img_cv2.shape[0] else cv2.INTER_CUBIC
        img_resized = cv2.resize(img_cv2, (width, height), interpolation=interpolation)
        img_resized = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
        img_resized = img_resized.unsqueeze(1)  # CFHW

        if has_end_image:
            interpolation = cv2.INTER_AREA if height < end_img_cv2.shape[1] else cv2.INTER_CUBIC
            end_img_resized = cv2.resize(end_img_cv2, (width, height), interpolation=interpolation)
            end_img_resized = TF.to_tensor(end_img_resized).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
            end_img_resized = end_img_resized.unsqueeze(1)  # CFHW

        # create mask for the first frame
        # if unofficial end image is used, we need to add an additional frame for it
        msk = torch.zeros(4, lat_f + additional_frames, lat_h, lat_w, device=device)
        msk[:, 0] = 1
        if has_end_image:
            # this process is confirmed by official code for FLF2V
            if config.flf2v:
                msk[-1, -1] = 1
            else:
                msk[:, -1] = 1

        # encode image to latent space
        with accelerator.autocast(), torch.no_grad():
            if not config.flf2v or not has_end_image:
                # padding to match the required number of frames
                padding_frames = frames - 1  # the first frame is image
                img_resized = torch.concat([img_resized, torch.zeros(3, padding_frames, height, width, device=device)], dim=1)
                y = vae.encode([img_resized])[0]

                if has_end_image:
                    y_end = vae.encode([end_img_resized])[0]
                    y = torch.concat([y, y_end], dim=1)  # add end frame
            else:
                # FLF2V: encode image and end image together
                padding_frames = frames - 2  # first and last frames are images
                img_resized = torch.concat(
                    [img_resized, torch.zeros(3, padding_frames, height, width, device=device), end_img_resized], dim=1
                )
                y = vae.encode([img_resized])[0]

        y = torch.concat([msk, y])
        logger.info("Encoding complete")

        # Fun-Control: encode control video to latent space
        if config.is_fun_control:
            # TODO use same resizing as for image
            logger.info("Encoding control video to latent space")
            # C, F, H, W
            control_video = load_control_video(args.control_path, frames + (1 if has_end_image else 0), height, width).to(device)
            with accelerator.autocast(), torch.no_grad():
                control_latent = vae.encode([control_video])[0]
            y = y[msk.shape[0] :]  # remove mask because Fun-Control does not need it
            if has_end_image:
                y[:, 1:-1] = 0  # remove image latent except first and last frame. according to WanVideoWrapper, this doesn't work
            else:
                y[:, 1:] = 0  # remove image latent except first frame
            y = torch.concat([control_latent, y], dim=0)  # add control video latent

    # generate noise
    noise = torch.randn(
        16,
        y.shape[1],  # number of frames in latent space
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=device if not args.cpu_noise else "cpu",
    )
    noise = noise.to(device)

    # prepare model input arguments
    arg_c = {
        "context": [context[0]],
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y],
        "f_indices": [f_indices] if f_indices is not None else None,
    }

    arg_null = {
        "context": context_null,
        "clip_fea": clip_context,
        "seq_len": max_seq_len,
        "y": [y],
        "f_indices": [f_indices] if f_indices is not None else None,
    }

    vae.to_device("cpu")  # move VAE to CPU to save memory
    clean_memory_on_device(device)

    return noise, (arg_c, arg_null), one_frame_inference_index


def load_control_video(control_path: str, frames: int, height: int, width: int) -> torch.Tensor:
    """load control video to latent space

    Args:
        control_path: path to control video
        frames: number of frames in the video
        height: height of the video
        width: width of the video

    Returns:
        torch.Tensor: control video latent, CFHW
    """
    logger.info(f"Load control video from {control_path}")
    video = load_video(control_path, 0, frames, bucket_reso=(width, height))  # list of frames
    if len(video) < frames:
        raise ValueError(f"Video length is less than {frames}")
    # video = np.stack(video, axis=0)  # F, H, W, C
    video = torch.stack([TF.to_tensor(frame).sub_(0.5).div_(0.5) for frame in video], dim=0)  # F, C, H, W, -1 to 1
    video = video.permute(1, 0, 2, 3)  # C, F, H, W
    return video


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


def run_sampling(
    models: List[WanModel],
    noise: torch.Tensor,
    scheduler: Any,
    timesteps: torch.Tensor,
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    inputs: Tuple[dict, dict],
    device: torch.device,
    seed_g: torch.Generator,
    accelerator: Accelerator,
    is_i2v: bool = False,
    use_cpu_offload: bool = True,
) -> torch.Tensor:
    """run sampling
    Args:
        models: dit models
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

    # cfg skip
    apply_cfg_array = []
    num_timesteps = len(timesteps)

    if args.cfg_skip_mode != "none" and args.cfg_apply_ratio is not None:
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

    # SLG original implementation is based on https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py
    slg_start_step = int(args.slg_start * num_timesteps)
    slg_end_step = int(args.slg_end * num_timesteps)

    prev_high_noise = args.timestep_boundary is not None
    if prev_high_noise:
        model = models[0]
        guidance_scale = args.guidance_scale_high_noise
    else:
        model = models[-1]  # use low noise model for low noise steps
        guidance_scale = args.guidance_scale

    logger.info(
        f"Starting sampling (high noise: {prev_high_noise}). Models: {len(models)}, timestep boundary: {args.timestep_boundary}, flow shift: {args.flow_shift}, guidance scale: {guidance_scale}"
    )

    for i, t in enumerate(tqdm(timesteps)):
        is_high_noise = (t / 1000.0) >= args.timestep_boundary if args.timestep_boundary is not None else False

        if not is_high_noise and prev_high_noise:
            guidance_scale = args.guidance_scale
            logger.info(f"Switching to low noise at step {i}, t={t}, guidance_scale={guidance_scale}")

            del model
            gc.collect()

            if len(models) > 1 and (args.offload_inactive_dit or args.lazy_loading):
                if args.blocks_to_swap > 0:
                    # prepare block swap for low noise model
                    logger.info("Waiting for 5 seconds to finish block swap")
                    time.sleep(5)

                if args.offload_inactive_dit:
                    logger.info("Switching model to CPU/GPU for both low and high noise models")
                    models[0].to("cpu")

                    if args.blocks_to_swap > 0:
                        # prepare block swap for low noise model
                        models[-1].move_to_device_except_swap_blocks(device)
                        models[-1].prepare_block_swap_before_forward()

                else:  # lazy loading
                    pass

                gc.collect()
                clean_memory_on_device(device)

            model = models[-1]  # use low noise model for low noise steps

        if model is None:
            # lazy loading
            dit_path = args.dit_high_noise if is_high_noise else args.dit
            lora_weight = args.lora_weight_high_noise if is_high_noise else args.lora_weight
            lora_multiplier = args.lora_multiplier_high_noise if is_high_noise else args.lora_multiplier
            model = load_dit_model(
                args, dit_path, lora_weight, lora_multiplier, gen_settings.cfg, device, gen_settings.dit_weight_dtype
            )

        prev_high_noise = is_high_noise

        # latent is on CPU if use_cpu_offload is True
        latent_model_input = [latent.to(device)]
        timestep = torch.stack([t]).to(device)

        with accelerator.autocast(), torch.no_grad():
            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0].to(latent_storage_device)

            apply_cfg = apply_cfg_array[i]  # apply CFG or not
            if apply_cfg:
                apply_slg = i >= slg_start_step and i < slg_end_step
                # print(f"Applying SLG: {apply_slg}, i: {i}, slg_start_step: {slg_start_step}, slg_end_step: {slg_end_step}")
                if args.slg_mode == "original" and apply_slg:
                    noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0].to(latent_storage_device)

                    # apply guidance
                    # SD3 formula: scaled = neg_out + (pos_out - neg_out) * cond_scale
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # calculate skip layer out
                    skip_layer_out = model(latent_model_input, t=timestep, skip_block_indices=args.slg_layers, **arg_null)[0].to(
                        latent_storage_device
                    )

                    # apply skip layer guidance
                    # SD3 formula: scaled = scaled + (pos_out - skip_layer_out) * self.slg
                    noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)
                elif args.slg_mode == "uncond" and apply_slg:
                    # noise_pred_uncond is skip layer out
                    noise_pred_uncond = model(latent_model_input, t=timestep, skip_block_indices=args.slg_layers, **arg_null)[0].to(
                        latent_storage_device
                    )

                    # apply guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                else:
                    # normal guidance
                    noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0].to(latent_storage_device)

                    # apply guidance
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                noise_pred = noise_pred_cond

            # step
            latent_input = latent.unsqueeze(0)
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent_input, return_dict=False, generator=seed_g)[0]

            # update latent
            latent = temp_x0.squeeze(0)

    if len(models) > 1 and args.lazy_loading:  # lazy loading
        del model
        gc.collect()
        clean_memory_on_device(device)

    return latent


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
    device, cfg, dit_dtype, dit_weight_dtype, vae_dtype = (
        gen_settings.device,
        gen_settings.cfg,
        gen_settings.dit_dtype,
        gen_settings.dit_weight_dtype,
        gen_settings.vae_dtype,
    )

    # prepare accelerator
    mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
    accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

    # I2V or T2V
    is_i2v = "i2v" in args.task or "flf2v" in args.task

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    # Check if we have shared models
    one_frame_inference_index = None
    if shared_models is not None:
        # Use shared models and encoded data
        vae = shared_models.get("vae")
        models = shared_models.get("models", [])
        encoded_context = shared_models.get("encoded_contexts", {}).get(args.prompt)

        # prepare inputs
        if is_i2v:
            # I2V
            noise, inputs, one_frame_inference_index = prepare_i2v_inputs(args, cfg, accelerator, device, vae, encoded_context)
        else:
            # T2V
            noise, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae, encoded_context)
    else:
        # prepare inputs without shared models
        if is_i2v:
            # I2V: need text encoder, VAE and CLIP
            vae = load_vae(args, cfg, device, vae_dtype)
            noise, inputs, one_frame_inference_index = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
            # vae is on CPU after prepare_i2v_inputs
        else:
            # T2V: need text encoder
            vae = None
            if cfg.is_fun_control:
                # Fun-Control: need VAE for encoding control video
                vae = load_vae(args, cfg, device, vae_dtype)
            noise, inputs = prepare_t2v_inputs(args, cfg, accelerator, device, vae)

        # load DiT models
        models = load_dit_models(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)

        # if we only want to save the model, we can skip the rest
        if args.save_merged_model:
            return None

    # setup scheduler
    scheduler, timesteps = setup_scheduler(args, cfg, device)

    # set random generator
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # run sampling
    latent = run_sampling(models, noise, scheduler, timesteps, args, gen_settings, inputs, device, seed_g, accelerator, is_i2v)
    if one_frame_inference_index is not None:
        latent = latent[:, one_frame_inference_index : one_frame_inference_index + 1, :]
        latent = latent.contiguous()  # safetensors requires contiguous tensors :(

    # Only clean up shared models if they were created within this function
    if shared_models is None:
        # free memory
        del models
        del scheduler
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

    x0 = latent.to(device)

    logger.info(f"Decoding video from latents: {latent.shape}")
    with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
        videos = vae.decode(x0)

    # some tail frames may be corrupted when end frame is used, we add an option to remove them
    if args.trim_tail_frames:
        videos[0] = videos[0][:, : -args.trim_tail_frames]

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
    if args.output_type == "latent" or args.output_type == "both" or args.output_type == "latent_images":
        # save latent
        save_latent(latent, args, height, width)

    if args.output_type == "video" or args.output_type == "both":
        # save video
        sample = decode_latent(latent.unsqueeze(0), args, cfg)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_video(sample, args, original_name)

    elif args.output_type == "images":
        # save images
        sample = decode_latent(latent.unsqueeze(0), args, cfg)
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
    device, cfg, dit_dtype, dit_weight_dtype, vae_dtype = (
        gen_settings.device,
        gen_settings.cfg,
        gen_settings.dit_dtype,
        gen_settings.dit_weight_dtype,
        gen_settings.vae_dtype,
    )
    is_i2v = "i2v" in args.task

    # 2. Encode all prompts
    logger.info("Loading text encoder to encode all prompts")
    text_encoder = load_text_encoder(args, cfg, device)
    text_encoder.model.to(device)

    encoded_contexts = {}

    with torch.no_grad():
        for prompt_data in prompts_data:
            prompt = prompt_data["prompt"]
            prompt_args = apply_overrides(args, prompt_data)
            n_prompt = prompt_data.get(
                "negative_prompt", prompt_args.negative_prompt if prompt_args.negative_prompt else cfg.sample_neg_prompt
            )

            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=cfg.t5_dtype):
                    context = text_encoder([prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([prompt], device)
                context_null = text_encoder([n_prompt], device)

            encoded_contexts[prompt] = {"context": context, "context_null": context_null}

    # Free text encoder and clean memory
    del text_encoder
    clean_memory_on_device(device)

    # 3. Process I2V additional encodings if needed
    vae = None
    if is_i2v:
        logger.info("Loading VAE and CLIP for I2V preprocessing")
        vae = load_vae(args, cfg, device, vae_dtype)
        vae.to_device(device)

        clip = None
        if not cfg.v2_2:
            clip = load_clip_model(args, cfg, device)
            clip.model.to(device)

        # Process each image and encode with CLIP
        for prompt_data in prompts_data:
            if "image_path" not in prompt_data and args.image_path is None:
                continue

            if not cfg.v2_2:
                prompt_args = apply_overrides(args, prompt_data)
                if not os.path.exists(prompt_args.image_path):
                    logger.warning(f"Image path not found: {prompt_args.image_path}")
                    continue

                # Load and encode image with CLIP
                img = Image.open(prompt_args.image_path).convert("RGB")
                img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

                with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                    clip_context = clip.visual([img_tensor[:, None, :, :]])

                if prompt_args.end_image_path is not None and os.path.exists(prompt_args.end_image_path):
                    end_img = Image.open(prompt_args.end_image_path).convert("RGB")
                    end_img_tensor = TF.to_tensor(end_img).sub_(0.5).div_(0.5).to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                        end_clip_context = clip.visual([end_img_tensor[:, None, :, :]])
                    clip_context = torch.concat([clip_context, end_clip_context], dim=0)
            else:
                clip_context = None

            encoded_contexts[prompt_data["prompt"]]["clip_context"] = clip_context

        # Free CLIP and clean memory
        del clip
        clean_memory_on_device(device)

        # Keep VAE in CPU memory for later use
        vae.to_device("cpu")
    elif cfg.is_fun_control:
        # For Fun-Control, we need VAE but keep it on CPU
        vae = load_vae(args, cfg, device, vae_dtype)
        vae.to_device("cpu")

    # 4. Load DiT model, 5. Merge LoRA weights if needed, and 6. Optimize model
    logger.info("Loading DiT model(s)")
    models = load_dit_models(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)

    if args.save_merged_model:
        logger.info("Model merged and saved. Exiting.")
        return

    # Create shared models dict for generate function
    shared_models = {"vae": vae, "models": models, "encoded_contexts": encoded_contexts}

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
    del models
    clean_memory_on_device(device)
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
            vae = load_vae(args, cfg, device, vae_dtype)

        vae.to_device(device)

        for i, (latent, prompt_args) in enumerate(zip(all_latents, all_prompt_args)):
            logger.info(f"Decoding output {i + 1}/{len(all_latents)}")

            # Decode latent
            video = decode_latent(latent.unsqueeze(0), prompt_args, cfg)

            # Save as video or images
            if prompt_args.output_type == "video" or prompt_args.output_type == "both":
                save_video(video, prompt_args)
            elif prompt_args.output_type == "images":
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
    device, cfg, dit_dtype, dit_weight_dtype, vae_dtype = (
        gen_settings.device,
        gen_settings.cfg,
        gen_settings.dit_dtype,
        gen_settings.dit_weight_dtype,
        gen_settings.vae_dtype,
    )
    is_i2v = "i2v" in args.task or "flf2v" in args.task

    # Initialize models to None
    text_encoder = None
    vae = None
    models = None
    clip = None

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

                # 1. Load text encoder if not already loaded
                if text_encoder is None:
                    logger.info("Loading text encoder")
                    text_encoder = load_text_encoder(args, cfg, device)

                text_encoder.model.to(device)

                # Encode prompt
                n_prompt = prompt_data.get(
                    "negative_prompt", prompt_args.negative_prompt if prompt_args.negative_prompt else cfg.sample_neg_prompt
                )

                with torch.no_grad():
                    if args.fp8_t5:
                        with torch.amp.autocast(device_type=device.type, dtype=cfg.t5_dtype):
                            context = text_encoder([prompt_data["prompt"]], device)
                            context_null = text_encoder([n_prompt], device)
                    else:
                        context = text_encoder([prompt_data["prompt"]], device)
                        context_null = text_encoder([n_prompt], device)

                encoded_context = {"context": context, "context_null": context_null}

                # Move text encoder to CPU after use
                text_encoder.model.to("cpu")

                # 2. For I2V, we need CLIP and VAE
                if is_i2v:
                    if not cfg.v2_2:
                        if clip is None:
                            logger.info("Loading CLIP model")
                            clip = load_clip_model(args, cfg, device)

                        clip.model.to(device)

                        # Encode image with CLIP if there's an image path
                        if prompt_args.image_path and os.path.exists(prompt_args.image_path):
                            img = Image.open(prompt_args.image_path).convert("RGB")
                            img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

                            with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                                clip_context = clip.visual([img_tensor[:, None, :, :]])

                            if prompt_args.end_image_path is not None and os.path.exists(prompt_args.end_image_path):
                                end_img = Image.open(prompt_args.end_image_path).convert("RGB")
                                end_img_tensor = TF.to_tensor(end_img).sub_(0.5).div_(0.5).to(device)
                                with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                                    end_clip_context = clip.visual([end_img_tensor[:, None, :, :]])
                                clip_context = torch.concat([clip_context, end_clip_context], dim=0)

                            encoded_context["clip_context"] = clip_context

                        # Move CLIP to CPU after use
                        clip.model.to("cpu")
                    else:
                        encoded_context["clip_context"] = None

                    # Load VAE if needed
                    if vae is None:
                        logger.info("Loading VAE model")
                        vae = load_vae(args, cfg, device, vae_dtype)
                elif cfg.is_fun_control and vae is None:
                    # For Fun-Control, we need VAE
                    logger.info("Loading VAE model for Fun-Control")
                    vae = load_vae(args, cfg, device, vae_dtype)

                # 3. Load DiT model if not already loaded
                if models is None:
                    logger.info("Loading DiT model")
                    models = load_dit_models(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)
                else:
                    # Move model to GPU if it was offloaded
                    if args.blocks_to_swap > 0:
                        if len(models) > 1:
                            if args.offload_inactive_dit:
                                models[-1].to("cpu")  # Move last model to CPU if multiple models
                            else:
                                models[-1].move_to_device_except_swap_blocks(device)
                                models[-1].prepare_block_swap_before_forward()
                        models[0].move_to_device_except_swap_blocks(device)
                        models[0].prepare_block_swap_before_forward()
                    else:
                        for model in models:
                            model.to(device)

                # Create shared models dict
                shared_models = {"vae": vae, "models": models, "encoded_contexts": {prompt_data["prompt"]: encoded_context}}

                # Generate latent
                latent = generate(prompt_args, gen_settings, shared_models)

                # Move model to CPU after generation
                for model in models:
                    if model is not None:  # not lazy loading
                        model.to("cpu")

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
                        vae = load_vae(args, cfg, device, vae_dtype)

                    vae.to_device(device)
                    video = decode_latent(latent.unsqueeze(0), prompt_args, cfg)

                    if prompt_args.output_type == "video" or prompt_args.output_type == "both":
                        save_video(video, prompt_args)
                    elif prompt_args.output_type == "images":
                        save_images(video, prompt_args)

                    # Move VAE to CPU after use
                    vae.to_device("cpu")

                clean_memory_on_device(device)

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")

    # Clean up all models
    if text_encoder is not None:
        del text_encoder
    if clip is not None:
        del clip
    if vae is not None:
        del vae
    if models is not None:
        del models

    clean_memory_on_device(device)
    gc.collect()


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
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

    gen_settings = GenerationSettings(
        device=device,
        cfg=cfg,
        dit_dtype=dit_dtype,
        dit_weight_dtype=dit_weight_dtype,
        vae_dtype=vae_dtype,
    )
    return gen_settings


def main():
    # Parse arguments
    args = parse_args()

    assert not (args.offload_inactive_dit and args.lazy_loading), (
        "--offload_inactive_dit and --lazy_loading cannot be used together"
    )

    # Check if latents are provided
    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    # Set device
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
        # Original latent decode mode
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

        height = latents.shape[-2]
        width = latents.shape[-1]
        height *= cfg.patch_size[1] * cfg.vae_stride[1]
        width *= cfg.patch_size[2] * cfg.vae_stride[2]
        video_length = latents.shape[1]
        video_length = (video_length - 1) * cfg.vae_stride[0] + 1
        args.seed = seeds[0]

        save_output(latent[0], args, cfg, height, width, original_base_names)

    elif args.from_file:
        # Batch mode from file
        args = setup_args(args)

        # Read prompts from file
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_lines = f.readlines()

        # Process prompts
        prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        process_batch_prompts(prompts_data, args)

    elif args.interactive:
        # Interactive mode
        args = setup_args(args)
        process_interactive(args)

    else:
        # Single prompt mode (original behavior)
        args = setup_args(args)
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

        # Add batch dimension
        latent = latent.unsqueeze(0)
        save_output(latent[0], args, WAN_CONFIGS[args.task], height, width)

    logger.info("Done!")


if __name__ == "__main__":
    main()
