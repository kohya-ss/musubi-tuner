import argparse
from datetime import datetime
import gc
import random
import os
import re
import time
import math
import copy
from types import ModuleType
from typing import Tuple, Optional, List, Union, Any, Dict
from rich_argparse import RichHelpFormatter
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

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
from wan.modules.model import WanModel, load_wan_model, detect_wan_sd_dtype
from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel
from modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from convert_lora import convert_from_diffusers
try:
    from lycoris.kohya import create_network_from_weights
except ImportError:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
from hv_generate_video import synchronize_device, save_images_grid
from blissful_tuner.latent_preview import LatentPreviewer
from blissful_tuner.cfg import apply_zerostar_scaling, perpendicular_negative_cfg, parse_scheduled_cfg
from blissful_tuner.utils import string_to_seed
from blissful_tuner.blissful_logger import BlissfulLogger
from blissful_tuner.prompt_management import MiniT5Wrapper, process_wildcards
from blissful_tuner.blissful_args import add_blissful_args, parse_blissful_args
from blissful_tuner.common_extensions import save_videos_grid_advanced, prepare_v2v_noise, prepare_metadata, BlissfulKeyboardManager
from dataset.image_video_dataset import load_video

logger = BlissfulLogger(__name__, "green")


class GenerationSettings:
    def __init__(
        self, device: torch.device, cfg, dit_dtype: torch.dtype, dit_weight_dtype: Optional[torch.dtype], vae_dtype: torch.dtype
    ):
        self.device = device
        self.cfg = cfg
        self.dit_dtype = dit_dtype
        self.dit_weight_dtype = dit_weight_dtype
        self.vae_dtype = vae_dtype


def parse_args() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script", formatter_class=RichHelpFormatter)

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
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=str, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise (compatible with ComfyUI). Default is False."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance. Default is 5.0.",
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
        default=["inductor", "default", "False", "False"],
        help="Torch.compile settings",
    )

    # New arguments for batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")
    parser = add_blissful_args(parser)
    args = parser.parse_args()
    args = parse_blissful_args(args)
    # Validate arguments
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.prompt is None and not args.from_file and not args.interactive and args.latent_path is None:
        raise ValueError("Either --prompt, --from_file, --interactive, or --latent_path must be specified")

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    return args


def parse_prompt_line(line: str, prompt_wildcards: Optional[str] = None) -> Dict[str, Any]:
    """Parse a prompt line into a dictionary of argument overrides

    Args:
        line: Prompt line with options

    Returns:
        Dict[str, Any]: Dictionary of argument overrides
    """
    # TODO common function with hv_train_network.line_to_prompt_dict
    parts = line.split(" --")
    prompt = parts[0].strip()
    if prompt_wildcards is not None:
        prompt = process_wildcards(prompt, prompt_wildcards)
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
            try:
                overrides["seed"] = int(value)
            except ValueError:
                overrides["seed"] = string_to_seed(value, bits=32)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "i":
            overrides["image_path"] = value
        elif option == "cn":
            overrides["control_path"] = value
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "cs":
            overrides["cfg_schedule"] = value

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
        logger.info(f"Size {size} is not (officially) supported for task {args.task}.\n Officially supported sizes are {SUPPORTED_SIZES[args.task]}.")

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
    if args.mixed_precision_transformer:
        loading_weight_dtype = None

    # do not fp8 optimize because we will merge LoRA weights
    model = load_wan_model(config, device, args.dit, args.attn_mode, False, loading_device, loading_weight_dtype, False, rope_func=args.rope_func, riflex_index=args.riflex_index, num_frames=args.video_length)

    return model


def merge_lora_weights(
    lora_module: ModuleType,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    converter: Optional[callable] = None,
) -> None:
    """merge LoRA weights to the model

    Args:
        lora_module: LoRA module, e.g. lora_wan
        model: DiT model
        args: command line arguments
        device: device to use
        converter: Optional callable to convert weights
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
        conversion_needed = False
        if converter is not None:
            weights_sd = converter(weights_sd)
        else:  # Kohya still hasn't implemented conversion for Wan/Hunyuan so this else handles that
            for key, weight in weights_sd.items():
                prefix, key_body = key.split(".", 1)
                if prefix == "diffusion_model" or prefix == "transformer":
                    conversion_needed = True
                    break
                elif "lora_unet" in prefix:
                    conversion_needed = False
                    break

        if conversion_needed:
            logger.info("Converting LoRA from diffusers format")
            weights_sd = convert_from_diffusers("lora_unet_", weights_sd)
        # apply include/exclude patterns
        original_key_count = len(weights_sd.keys())
        if args.include_patterns is not None and len(args.include_patterns) > i:
            include_pattern = args.include_patterns[i]
            regex_include = re.compile(include_pattern)
            weights_sd = {k: v for k, v in weights_sd.items() if regex_include.search(k)}
            logger.info(f"Filtered keys with include pattern {include_pattern}: {original_key_count} -> {len(weights_sd.keys())}")
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
                logger.warning("No keys left after filtering.")

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
            network = lora_module.create_arch_network_from_weights(lora_multiplier, weights_sd, unet=model, for_inference=True)
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
        state_dict = model.fp8_optimization(state_dict, device, use_scaled_mm=args.fp8_fast, upcast_linear=args.upcast_linear, quant_dtype=torch.float32 if args.upcast_quantization else None)
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
    args: argparse.Namespace,
    config,
    accelerator: Accelerator,
    device: torch.device,
    vae: Optional[WanVAE] = None,
    encoded_context: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
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
        if args.prompt_weighting:
            text_encoder = MiniT5Wrapper(device, config.t5_dtype, text_encoder)

        # encode prompt
        with torch.no_grad():
            if args.fp8_t5:
                with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                    context = text_encoder([args.prompt], device)
                    context_null = text_encoder([n_prompt], device)
            else:
                context = text_encoder([args.prompt], device)
                context_null = text_encoder([n_prompt], device)
            if args.perp_neg is not None:
                if args.fp8_t5:
                    with torch.amp.autocast(device_type=device.type, dtype=config.t5_dtype):
                        context_nocond = text_encoder("", device)
                else:
                    context_nocond = text_encoder("", device)

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
    arg_nocond = {"context": context_nocond, "seq_len": seq_len} if args.perp_neg is not None else None
    if y is not None:
        arg_c["y"] = [y]
        arg_null["y"] = [y]
        if args.perp_neg is not None:
            arg_nocond["y"] = [y]

    return noise, context, context_null, (arg_c, arg_null), arg_nocond


def prepare_i2v_inputs(
    args: argparse.Namespace,
    config,
    accelerator: Accelerator,
    device: torch.device,
    vae: WanVAE,
    encoded_context: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[dict, dict]]:
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
    if args.image_path is not None:
        # classic I2V path
        img = Image.open(args.image_path).convert("RGB")
        img_cv2 = np.array(img)  # for the VAE path later
        if args.video_path is not None:
            logger.info(f"Loaded still image `{args.image_path}` as I2V anchor for IV2V.")
    elif args.video_path is not None:
        # V2V‐with‐I2V‐anchor path
        logger.info("Using first frame of video as I2V anchor for IV2V.")
        cap = cv2.VideoCapture(args.video_path)
        success, frame_bgr = cap.read()
        cap.release()
        if not success or frame_bgr is None:
            raise RuntimeError(f"Failed to read first frame from `{args.video_path}`")
        img_cv2 = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_cv2)
    else:
        raise ValueError(f"Must supply --image_path for I2V or --video_path for IV2V when task is {args.task}!")

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
    height, width = img_tensor.shape[1:]
    aspect_ratio = height / width
    lat_h = int(round(np.sqrt(max_area * aspect_ratio) / config.vae_stride[1] / config.patch_size[1] * config.patch_size[1]))
    lat_w = int(round(np.sqrt(max_area / aspect_ratio) / config.vae_stride[2] / config.patch_size[2] * config.patch_size[2]))
    height = lat_h * config.vae_stride[1]
    width = lat_w * config.vae_stride[2]
    lat_f = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
    max_seq_len = (lat_f + (1 if has_end_image else 0)) * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if not args.cpu_noise:
        seed_g = torch.Generator(device=device)
        seed_g.manual_seed(seed)
    else:
        # ComfyUI compatible noise
        seed_g = torch.manual_seed(seed)

    # generate noise
    noise = torch.randn(
        16,
        lat_f + (1 if has_end_image else 0),
        lat_h,
        lat_w,
        dtype=torch.float32,
        generator=seed_g,
        device=device if not args.cpu_noise else "cpu",
    )
    noise = noise.to(device)

    # configure negative prompt
    n_prompt = args.negative_prompt if args.negative_prompt else config.sample_neg_prompt

    if encoded_context is None:
        # load text encoder
        text_encoder = load_text_encoder(args, config, device)
        text_encoder.model.to(device)
        if args.prompt_weighting:
            text_encoder = MiniT5Wrapper(device, config.t5_dtype, text_encoder)

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
        logger.info("Encoding image to CLIP context")
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
            clip_context = clip.visual([img_tensor[:, None, :, :]])
        logger.info("Encoding complete")

        # free CLIP model and clean memory
        del clip
        clean_memory_on_device(device)
    else:
        # Use pre-encoded context
        context = encoded_context["context"]
        context_null = encoded_context["context_null"]
        clip_context = encoded_context["clip_context"]

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
    msk = torch.zeros(4, lat_f + (1 if has_end_image else 0), lat_h, lat_w, device=device)
    msk[:, 0] = 1
    if has_end_image:
        msk[:, -1] = 1

    # encode image to latent space
    with accelerator.autocast(), torch.no_grad():
        # padding to match the required number of frames
        padding_frames = frames - 1  # the first frame is image
        img_resized = torch.concat([img_resized, torch.zeros(3, padding_frames, height, width, device=device)], dim=1)
        y = vae.encode([img_resized])[0]

        if has_end_image:
            y_end = vae.encode([end_img_resized])[0]
            y = torch.concat([y, y_end], dim=1)  # add end frame

    y = torch.concat([msk, y])
    logger.info("Encoding complete")

    if args.i2v_extra_noise not in (None, 0.0):
        logger.info(f"Adding {100 * args.i2v_extra_noise:.1f}% extra noise to I2V conditioning latents")
        extra_noise = torch.randn(
            y.shape,
            generator=seed_g,
            device=y.device if not args.cpu_noise else "cpu",
            dtype=y.dtype
        ) * args.i2v_extra_noise
        y = y + extra_noise.to(y.device)

    # Fun-Control: encode control video to latent space
    if config.is_fun_control:
        # TODO use same resizing as for image
        logger.info("Encoding control video to latent space")
        # C, F, H, W
        control_video = load_control_video(args.control_path, frames + (1 if has_end_image else 0), height, width).to(device)
        with accelerator.autocast(), torch.no_grad():
            control_latent = vae.encode([control_video])[0]
        y = y[msk.shape[0]:]  # remove mask because Fun-Control does not need it
        if has_end_image:
            y[:, 1:-1] = 0  # remove image latent except first and last frame. according to WanVideoWrapper, this doesn't work
        else:
            y[:, 1:] = 0  # remove image latent except first frame
        y = torch.concat([control_latent, y], dim=0)  # add control video latent

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

    vae.to_device("cpu")  # move VAE to CPU to save memory
    clean_memory_on_device(device)

    return noise, context, context_null, y, (arg_c, arg_null)


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
    arg_nocond: dict = None
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
    if args.preview_latent_every:
        previewer = LatentPreviewer(args, noise, timesteps, model.device, model.dtype, model_type="wan")
    km = BlissfulKeyboardManager()
    # cfg skip
    apply_cfg_array = []
    num_timesteps = len(timesteps)

    if args.cfg_skip_mode != "none" and args.cfg_apply_ratio is not None:
        # Kohya's method
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
    elif args.cfg_schedule is not None:
        scale_per_step = parse_scheduled_cfg(args.cfg_schedule, num_timesteps, args.guidance_scale)
        apply_cfg_array = [(i + 1) in scale_per_step for i in range(num_timesteps)]
        included_steps = sorted(scale_per_step.keys())
        step_str = ", ".join(f"{step}:{scale_per_step[step]}" for step in included_steps)
        logger.info(f"CFG Schedule: {step_str}")
        logger.info(f"Total CFG steps: {len(included_steps)}")

    else:
        # Apply CFG on all steps
        apply_cfg_array = [True] * num_timesteps

    if args.cfgzerostar_scaling or args.cfgzerostar_init_steps != -1:
        logger.info(f"Using CFGZero* - Scaling: {args.cfgzerostar_scaling}; Zero init steps: {'None' if args.cfgzerostar_init_steps == -1 else args.cfgzerostar_init_steps}")
    # SLG original implementation is based on https://github.com/Stability-AI/sd3.5/blob/main/sd3_impls.py
    slg_start_step = int(args.slg_start * num_timesteps)
    slg_end_step = int(args.slg_end * num_timesteps)
    for i, t in enumerate(tqdm(timesteps)):
        # latent is on CPU if use_cpu_offload is True
        latent_model_input = [latent.to(device)]
        timestep = torch.stack([t]).to(device)

        with accelerator.autocast(), torch.no_grad():
            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0].to(latent_storage_device)  # Cond is always the same
            apply_cfg = apply_cfg_array[i]  # Will we do any CFG or just proceed with cond?
            if apply_cfg:  # We will do CFG this step
                if args.cfg_schedule is not None:  # Is it scheduled CFG? If so and it's off we won't come here but if it's on...
                    args.guidance_scale = scale_per_step[i + 1]  # Then reset the guidance scale for this step

                apply_slg = args.slg_mode is not None and i >= slg_start_step and i < slg_end_step  # Do we do SLG this step
                skip_block_indices = None if not apply_slg else args.slg_layers  # If so set slg else None(implicit none anyway)
                noise_pred_uncond = model(latent_model_input, timestep, skip_block_indices=skip_block_indices, **arg_null)[0].to(latent_storage_device)  # uncond

                if args.perp_neg is not None and not km.early_exit_requested:  # Are we using perpendicular negative?
                    noise_pred_nocond = model(latent_model_input, t=timestep, **arg_nocond)[0].to(latent_storage_device)  # Then calculate nocond
                    noise_pred = perpendicular_negative_cfg(noise_pred_cond, noise_pred_uncond, noise_pred_nocond, args.perp_neg, args.guidance_scale)  # And update the noise_pred
                elif args.cfgzerostar_scaling:  # No perp_neg, so CFGZero* scaling instead? (mutually exclusive)
                    noise_pred = apply_zerostar_scaling(noise_pred_cond, noise_pred_uncond, args.guidance_scale)  # does CFG with scaling inside, returns noise_pred after CFG formula
                else:  # Neither perp_neg nor scaling so normal CFG formula
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                if i + 1 <= args.cfgzerostar_init_steps:  # Do zero init? User provides init_steps as 1 based but i is 0 based
                    noise_pred *= 0  # Zero this step
                elif apply_slg and args.slg_mode == "original" and not km.eearly_exit_requested:  # No need do traditional slg on zeroed steps so make it an elif
                    # SLG original mode uses 3 model passes, cond, uncond, and uncond with layer skip
                    skip_layer_out = model(latent_model_input, t=timestep, skip_block_indices=args.slg_layers, **arg_null)[0].to(latent_storage_device)
                    noise_pred = noise_pred + args.slg_scale * (noise_pred_cond - skip_layer_out)  # SD3 SLG formula: scaled = scaled + (pos_out - skip_layer_out) * self.slg

            else:  # No CFG shenanigans at all
                noise_pred = noise_pred_cond

            # step
            latent_input = latent.unsqueeze(0)
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent_input, return_dict=False, generator=seed_g)[0]

            # update latent
            latent = temp_x0.squeeze(0)
            if km.early_exit_requested:
                latent = None
                break
            if args.preview_latent_every is not None and (i + 1) % args.preview_latent_every == 0 and i + 1 != len(timesteps):
                previewer.preview(latent, i)
    km.terminate()
    return latent


def generate(args: argparse.Namespace, gen_settings: GenerationSettings, shared_models: Optional[Dict] = None) -> torch.Tensor:
    """main function for generation

    Args:
        args: command line arguments
        shared_models: dictionary containing pre-loaded models and encoded data

    Returns:
        torch.Tensor: generated latent
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
    is_i2v = "i2v" in args.task

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving
    # setup scheduler
    scheduler, timesteps = setup_scheduler(args, cfg, device)
    # Check if we have shared models
    arg_nocond = None  # Perp neg is only available for T2V currently
    if shared_models is not None:
        # Use shared models and encoded data
        vae = shared_models.get("vae")
        model = shared_models.get("model")
        encoded_context = shared_models.get("encoded_contexts", {}).get(args.prompt)

        # prepare inputs
        if is_i2v:
            # I2V
            noise, context, context_null, y, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae, encoded_context)
        else:
            # T2V
            noise, context, context_null, inputs, arg_nocond = prepare_t2v_inputs(args, cfg, accelerator, device, vae, encoded_context)

    else:
        # prepare inputs without shared models
        if is_i2v:
            # I2V: need text encoder, VAE and CLIP
            vae = load_vae(args, cfg, device, vae_dtype)
            noise, context, context_null, y, inputs = prepare_i2v_inputs(args, cfg, accelerator, device, vae)
            # vae is on CPU after prepare_i2v_inputs
        else:
            # T2V: need text encoder
            vae = None
            if cfg.is_fun_control or args.video_path is not None:
                # Fun-Control: need VAE for encoding control video
                vae = load_vae(args, cfg, device, vae_dtype)
            noise, context, context_null, inputs, arg_nocond = prepare_t2v_inputs(args, cfg, accelerator, device, vae)

        if args.video_path is not None:
            noise, timesteps = prepare_v2v_noise(noise, args, cfg, timesteps, device, vae)
        # load DiT model
        model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)

        # merge LoRA weights
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            merge_lora_weights(lora_wan, model, args, device)

            # if we only want to save the model, we can skip the rest
            if args.save_merged_model:
                return None

        # optimize model: fp8 conversion, block swap etc.
        optimize_model(model, args, device, dit_dtype, dit_weight_dtype)

    # set random generator
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # run sampling
    latent = run_sampling(model, noise, scheduler, timesteps, args, inputs, device, seed_g, accelerator, is_i2v, arg_nocond=arg_nocond)

    # Only clean up shared models if they were created within this function
    if shared_models is None:
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

    with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
        videos = vae.decode(x0)

    # some tail frames may be corrupted when end frame is used, we add an option to remove them
    if args.trim_tail_frames:
        videos[0] = videos[0][:, : -args.trim_tail_frames]

    logger.info("Decoding complete")
    video = videos[0]
    del videos
    video = video.to(torch.float32).cpu()

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
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"
    metadata = prepare_metadata(args)  # Will be set to None inside if args.no_metadata
    sd = {"latent": latent}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_video(video: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None, metadata: Optional[dict] = None) -> str:
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
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    video_path = f"{save_path}/{time_flag}_{seed}{original_name}.mp4"
    video = video.unsqueeze(0)
    metadata = prepare_metadata(args) if metadata is None else metadata
    save_videos_grid_advanced(video, video_path, args, rescale=True, metadata=metadata)

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
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0)
    save_images_grid(sample, save_path, image_name, rescale=True)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    latent: torch.Tensor,
    args: argparse.Namespace,
    cfg,
    height: int,
    width: int,
    original_base_names: Optional[List[str]] = None,
    metadata: Optional[dict] = None
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
    if args.output_type == "latent" or args.output_type == "both":
        # save latent
        save_latent(latent, args, height, width)

    if args.output_type != "latent":
        # save video
        sample = decode_latent(latent.unsqueeze(0), args, cfg)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        save_video(sample, args, original_name, metadata)


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
        prompt_data = parse_prompt_line(line, base_args.prompt_wildcards)
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
    if args.prompt_weighting:
        text_encoder = MiniT5Wrapper(device, cfg.t5_dtype, text_encoder)

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

        clip = load_clip_model(args, cfg, device)
        clip.model.to(device)

        # Process each image and encode with CLIP
        for prompt_data in prompts_data:
            if "image_path" not in prompt_data:
                continue

            prompt_args = apply_overrides(args, prompt_data)
            if not os.path.exists(prompt_args.image_path):
                logger.warning(f"Image path not found: {prompt_args.image_path}")
                continue

            # Load and encode image with CLIP
            img = Image.open(prompt_args.image_path).convert("RGB")
            img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                clip_context = clip.visual([img_tensor[:, None, :, :]])

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

    # 4. Load DiT model
    logger.info("Loading DiT model")
    model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)

    # 5. Merge LoRA weights if needed
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        merge_lora_weights(lora_wan, model, args, device)
        if args.save_merged_model:
            logger.info("Model merged and saved. Exiting.")
            return

    # 6. Optimize model
    optimize_model(model, args, device, dit_dtype, dit_weight_dtype)

    # Create shared models dict for generate function
    shared_models = {"vae": vae, "model": model, "encoded_contexts": encoded_contexts}

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
        if prompt_args.output_type == "latent" or prompt_args.output_type == "both":
            save_latent(latent, prompt_args, height, width)

        all_latents.append(latent)
        all_prompt_args.append(prompt_args)

    # 8. Free DiT model
    del model
    clean_memory_on_device(device)
    synchronize_device(device)

    # wait for 5 seconds until block swap is done
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
    is_i2v = "i2v" in args.task

    # Initialize models to None
    text_encoder = None
    vae = None
    model = None
    clip = None

    print("Interactive mode. Enter prompts (Ctrl+D to exit):")

    try:
        while True:
            try:
                line = input("> ")
                if not line.strip():
                    continue

                # Parse prompt
                prompt_data = parse_prompt_line(line, args.prompt_wildcards)
                prompt_args = apply_overrides(args, prompt_data)

                # Ensure we have all the models we need

                # 1. Load text encoder if not already loaded
                if text_encoder is None:
                    logger.info("Loading text encoder")
                    text_encoder = load_text_encoder(args, cfg, device)
                    if args.prompt_weighting:
                        text_encoder = MiniT5Wrapper(device, cfg.t5_dtype, text_encoder)

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
                    else:  # original behavior
                        context = text_encoder([prompt_data["prompt"]], device)
                        context_null = text_encoder([n_prompt], device)

                encoded_context = {"context": context, "context_null": context_null}

                # Move text encoder to CPU after use
                text_encoder.model.to("cpu")

                # 2. For I2V, we need CLIP and VAE
                if is_i2v:
                    if clip is None:
                        logger.info("Loading CLIP model")
                        clip = load_clip_model(args, cfg, device)

                    clip.model.to(device)

                    # Encode image with CLIP if there's an image path
                    if prompt_args.image_path and os.path.exists(prompt_args.image_path):
                        img = Image.open(prompt_args.image_path).convert("RGB")
                        img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)
                        img_tensor = img_tensor.unsqueeze(1)
                        with torch.amp.autocast(device_type=device.type, dtype=torch.float16), torch.no_grad():
                            clip_context = clip.visual([img_tensor])

                        encoded_context["clip_context"] = clip_context

                    # Move CLIP to CPU after use
                    clip.model.to("cpu")

                    # Load VAE if needed
                    if vae is None:
                        logger.info("Loading VAE model")
                        vae = load_vae(args, cfg, device, vae_dtype)
                elif cfg.is_fun_control and vae is None:
                    # For Fun-Control, we need VAE
                    logger.info("Loading VAE model for Fun-Control")
                    vae = load_vae(args, cfg, device, vae_dtype)

                # 3. Load DiT model if not already loaded
                if model is None:
                    logger.info("Loading DiT model")
                    model = load_dit_model(args, cfg, device, dit_dtype, dit_weight_dtype, is_i2v)

                    # Merge LoRA weights if needed
                    if args.lora_weight is not None and len(args.lora_weight) > 0:
                        merge_lora_weights(lora_wan, model, args, device)

                    # Optimize model
                    optimize_model(model, args, device, dit_dtype, dit_weight_dtype)
                else:
                    # Move model to GPU if it was offloaded
                    model.to(device)

                # Create shared models dict
                shared_models = {"vae": vae, "model": model, "encoded_contexts": {prompt_data["prompt"]: encoded_context}}

                # Generate latent
                latent = generate(prompt_args, gen_settings, shared_models)

                # Move model to CPU after generation
                model.to("cpu")
                if latent is None:  # None is sentinel for early exit
                    raise KeyboardInterrupt
                # Save latent if needed
                height, width, _ = check_inputs(prompt_args)
                if prompt_args.output_type == "latent" or prompt_args.output_type == "both":
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
    if model is not None:
        del model

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
    if args.fp8_scaled or args.mixed_precision_transformer:
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

                if "bt_seeds" in metadata:
                    seed = int(metadata["bt_seeds"])
                if "bt_height" in metadata and "bt_width" in metadata:
                    height = int(metadata["bt_height"])
                    width = int(metadata["bt_width"])
                    args.video_size = [height, width]
                if "bt_video_length" in metadata:
                    args.video_length = int(metadata["bt_video_length"])

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

        # Decode and save
        save_output(latent[0], args, cfg, height, width, original_base_names, metadata=metadata)

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

        if latent is not None:
            # Save latent and video
            if args.save_merged_model:
                return
            # Add batch dimension
            latent = latent.unsqueeze(0)
            save_output(latent[0], args, WAN_CONFIGS[args.task], height, width)

    logger.info("Done!")


if __name__ == "__main__":
    main()
