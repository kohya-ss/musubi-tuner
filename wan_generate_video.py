import argparse
from datetime import datetime
import gc
import random
import os
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
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")

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
    model = load_wan_model(config, is_i2v, device, args.dit, args.attn_mode, False, loading_device, loading_weight_dtype, False)

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
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

    # convert to tensor (-1 to 1)
    img_tensor = TF.to_tensor(img).sub_(0.5).div_(0.5).to(device)

    # calculate latent dimensions: keep aspect ratio
    h, w = img_tensor.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // config.vae_stride[1] // config.patch_size[1] * config.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // config.vae_stride[2] // config.patch_size[2] * config.patch_size[2])
    h = lat_h * config.vae_stride[1]
    w = lat_w * config.vae_stride[2]
    lat_f = (frames - 1) // config.vae_stride[0] + 1  # size of latent frames
    max_seq_len = lat_f * lat_h * lat_w // (config.patch_size[1] * config.patch_size[2])

    # set seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    # generate noise
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=device)

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
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_resized = TF.to_tensor(img_resized).sub_(0.5).div_(0.5).to(device)  # -1 to 1, CHW
    img_resized = img_resized.unsqueeze(1)  # CFHW

    # create mask for the first frame
    # msk = torch.ones(1, frames, lat_h, lat_w, device=device)
    # msk[:, 1:] = 0
    # msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    # msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    # msk = msk.transpose(1, 2)[0]

    # rewrite to simpler version
    msk = torch.zeros(4, lat_f, lat_h, lat_w, device=device)
    msk[:, 0] = 1

    # encode image to latent space
    with accelerator.autocast(), torch.no_grad():
        # padding to match the required number of frames
        padding_frames = frames - 1  # the first frame is image
        img_resized = torch.concat([img_resized, torch.zeros(3, padding_frames, h, w, device=device)], dim=1)
        y = vae.encode([img_resized])[0]

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
    if use_cpu_offload:
        latent = latent.to("cpu")

    for _, t in enumerate(tqdm(timesteps)):
        # latent is on CPU if use_cpu_offload is True
        latent_model_input = [latent.to(device)]
        timestep = torch.stack([t]).to(device)

        with accelerator.autocast(), torch.no_grad():
            noise_pred_cond = model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = model(latent_model_input, t=timestep, **arg_null)[0]
            del latent_model_input

            if use_cpu_offload:
                noise_pred_cond = noise_pred_cond.to("cpu")
                noise_pred_uncond = noise_pred_uncond.to("cpu")

            # apply guidance
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # step
            latent_input = latent.unsqueeze(0)
            temp_x0 = scheduler.step(noise_pred.unsqueeze(0), t, latent_input, return_dict=False, generator=seed_g)[0]

            # update latent
            latent = temp_x0.squeeze(0)

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

    # I2V or T2V
    is_i2v = "i2v" in args.task

    # prepare seed
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed  # set seed to args for saving

    # prepare inputs
    if is_i2v:
        # I2V: need text encoder, VAE and CLIP
        vae = load_vae(args, cfg, device, vae_dtype)
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

    # run sampling
    latent = run_sampling(model, noise, scheduler, timesteps, args, inputs, device, seed_g, accelerator, is_i2v)

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
