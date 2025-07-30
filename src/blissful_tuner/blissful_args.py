#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:11:58 2025

@author: blyss
"""
import sys
import os
import gc
import argparse
import random
import torch
from rich.traceback import install as install_rich_tracebacks
from blissful_tuner.utils import string_to_seed, error_out
from blissful_tuner.blissful_logger import BlissfulLogger
from blissful_tuner.prompt_management import process_wildcards
logger = BlissfulLogger(__name__, "#8e00ed")

BLISSFUL_VERSION = "0.9.66"

CFG_SCHEDULE_HELP = """
Comma-separated list of steps/ranges where CFG should be applied.

You can specify:
- Single steps (e.g., '5')
- Ranges (e.g., '1-10')
- Modulus patterns (e.g., 'e~2' for every 2 steps)
- Guidance scale overrides (e.g., '1-10:5.0')

Example schedule:
  'e~2:6.4, 1-10, 46-50'

This would apply:
- Default CFG scale for steps 1-10 and 46-50
- 6.4 CFG scale every 2 steps outside that range
- No CFG otherwise

You can exclude steps using '!', e.g., '!32' skips step 32.
Note: The list is processed left to right, so modulus ranges should come first and exclusions at the end!
"""

ROOT_SCRIPT = os.path.basename(sys.argv[0]).lower()
DIFFUSION_MODEL = None
if "hv_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "hunyuan"
elif "wan_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "wan"
elif "fpack_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "framepack"

MODE = None
if "generate" in ROOT_SCRIPT:
    MODE = "generate"
elif "train" in ROOT_SCRIPT:
    MODE = "train"


def get_current_model_type():
    return DIFFUSION_MODEL


def get_current_version():
    return BLISSFUL_VERSION


def blissful_prefunc(args: argparse.Namespace):
    """Simple function to print about version, environment, and things"""
    cuda_list = [f"Python: {sys.version.split(" ")[0]}"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocator = torch.cuda.get_allocator_backend()
        cuda = torch.cuda.get_device_properties(0)
        cuda_list[0] += f", CUDA: {torch.version.cuda} CC: {cuda.major}.{cuda.minor}"
        cuda_list.append(f"Device: '{cuda.name}', VRAM: '{cuda.total_memory // 1024 ** 2}MB'")
    logger.info(f"Blissful Tuner version {BLISSFUL_VERSION} extended from Musubi Tuner!")
    logger.info(f"PyTorch: {torch.__version__}, Memory allocation: '{allocator}'")
    for string in cuda_list:
        logger.info(string)
    if args.optimized and MODE == "generate":
        logger.info("Optimized arguments enabled!")
        args.fp16_accumulation = True
        args.attn_mode = "sageattn"
        args.compile = True
        args.fp8_scaled = True
        if DIFFUSION_MODEL == "wan":
            args.rope_func = "comfy"
        elif DIFFUSION_MODEL in ["hunyuan", "framepack"]:
            args.fp16_accumulation = False  # Disable this for hunyuan and framepack b/c we enable fp8_fast which offsets it anyway and torch 2.7.0 has issues with compiling hunyuan sometimes
            args.fp8_fast = True
    if args.fp16_accumulation and MODE == "generate":
        logger.info("Enabling FP16 accumulation")
        if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
        else:
            logger.warning("FP16 accumulation not available! Requires at least PyTorch 2.7.0")
    if DIFFUSION_MODEL == "wan" and args.video_path is not None:
        if "i2v" in args.task:
            logger.info("V2V operating in IV2V mode!")
        else:
            logger.info("V2V operating in normal mode!")


def add_blissful_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    install_rich_tracebacks()
    if DIFFUSION_MODEL == "wan":
        parser.add_argument("--i2i_path", type=str, default=None, help="path to an image for image2image inference. Use with T2V model and T2I task.")
        parser.add_argument("--v2_extra_noise", type=float, default=None, help="Extra latent noise for v2v. Low values are best e.g. 0.015. Can help add fine details, especially when upscaling(output res > input res)")
        parser.add_argument("--i2_extra_noise", type=float, default=None, help="Extra latent noise for i2v. Low values are best e.g. 0.025. Can help add fine details, especially when upscaling(output res > input res)")
        parser.add_argument("--denoise_strength", type=float, default=0.5, help="Amount of denoising to do for V2V or I2I, 0.0-1.0")
        parser.add_argument(
            "--noise_mode", choices=["traditional", "direct"], default="traditional",
            help="Controls how --denoise_strength value works. Traditional is like usual and controls what percent of the timestep schedule to run. "
            "Direct allows you to directly control how much noise will be added."
        )
        parser.add_argument(
            "--v2v_pad_mode", type=str, choices=["front", "end"], default="end",
            help="Padding mode for when V2V input is shorter than requested output"
        )
        parser.add_argument("--prompt_weighting", action="store_true", help="Enable (prompt weighting:1.2)")
        parser.add_argument(
            "--rope_func", type=str, choices=["default", "comfy"], default="default",
            help="Function to use for ROPE. Choose from 'default' or 'comfy' the latter of which uses ComfyUI implementation and is compilable with torch.compile to enable BIG VRAM savings"
        )
        parser.add_argument("--mixed_precision_transformer", action="store_true", help="Allow loading mixed precision transformer such as a combination of float16 weights / float32 everything else")
    elif DIFFUSION_MODEL == "hunyuan":
        parser.add_argument("--from_latent", type=str, default=None, help="Input path for latent.safetensors to use for V2V")
        parser.add_argument("--scheduler", type=str, choices=["default", "dpm++", "dpm++sde"], default="default")
        parser.add_argument("--disable_embedded_for_cfg", action="store_true", help="Fully disable embedded guidance when doing CFG. Allows higher CFG scale and better control.")
        parser.add_argument("--hidden_state_skip_layer", type=int, default=2, help="Hidden state skip layer for LLM. Default is 2. Think 'clip skip' for the LLM")
        parser.add_argument("--apply_final_norm", action="store_true", help="Apply final norm for LLM. Default is False. Usually makes things worse.")
        parser.add_argument("--reproduce", action="store_true", help="Enable reproducible output(Same seed = same result. Default is False.")
        parser.add_argument("--fp8_scaled", action="store_true", help="Scaled FP8 quantization. Better quality/accuracy with slightly more VRAM usage.")
        parser.add_argument("--prompt_2", type=str, default=None, help="Optional different prompt for CLIP")
        parser.add_argument("--te_multiplier", nargs=2, metavar=("llm_multiplier", "clip_multiplier"), help="Scale clip and llm influence")
    elif DIFFUSION_MODEL == "framepack":
        parser.add_argument("--preview_latent_every", type=int, default=None, help="Enable latent preview every N sections. If --preview_vae is not specified it will use latent2rgb")
        parser.add_argument("--te_multiplier", nargs=2, metavar=("llm_multiplier", "clip_multiplier"), help="Scale clip and llm influence")

    if DIFFUSION_MODEL in ["wan", "hunyuan"]:
        parser.add_argument("--riflex_index", type=int, default=0, help="Frequency for RifleX extension. 4 is good for Hunyuan, 6 is good for Wan. Only 'comfy' rope_func supports this with Wan!")
        parser.add_argument("--cfgzerostar_scaling", action="store_true", help="Enables CFG-Zero* scaling - https://github.com/WeichenFan/CFG-Zero-star")
        parser.add_argument("--cfgzerostar_multiplier", type=float, default=0, help="Multiplier used for cfgzerostar_init. Default is 0 which zeroes the step. 1 would be like not using zero init.")
        parser.add_argument("--cfgzerostar_init_steps", type=int, default=-1, help="Enables CFGZero* zeroing out the first N steps. 2 is good for Wan T2V, 1 for I2V")
        parser.add_argument("--preview_latent_every", type=int, default=None, help="Enable latent preview every N steps. If --preview_vae is not specified it will use latent2rgb")
        parser.add_argument("--cfg_schedule", type=str, help=CFG_SCHEDULE_HELP)
        parser.add_argument(
            "--perp_neg", type=float, default=None,
            help="Enable and set scale for perpendicular negative guidance. Start with like 1.5 - 2.0. This is a stronger, more precise form of CFG."
            "It requires a third modeling pass per step so it will slow you down by 30 percent but can definitely help isolating concepts. An example would be "
            "prompting for 'nature photography' when using 'tree' as a negative. Normal CFG will mostly avoid trees but because 'nature' and 'tree' are associated "
            "there will likely still be some. With perp neg, the entire concept of tree is subtracted from the result.")
    # Common
    parser.add_argument(
        "--prompt_wildcards", type=str, default=None,
        help="Path to a directory of wildcard.txt files to enable wildcards in prompts and negative prompts. For instance __color__ will look for wildcards in color.txt in that directory. "
        "Wildcard files should have one possible replacement per line, optionally with a relative weight attached like red:2.0 or yellow:0.5, and wildcards can be nested.")
    parser.add_argument("--preview_vae", type=str, help="Path to TAE vae for taehv previews")
    parser.add_argument("--keep_pngs", action="store_true", help="Save frames as PNGs in addition to output video")
    parser.add_argument(
        "--codec", choices=["h264", "h265", "prores"], default="h264",
        help="Codec to use when saving videos, choose from 'prores', 'h264', or 'h265'. Default is 'h264'"
    )
    parser.add_argument(
        "--container", choices=["mkv", "mp4"], default="mp4",
        help="Container format to use for output, choose from 'mkv' or 'mp4'. Default is 'mp4' and note that 'prores' can only go in 'mkv'! Ignored for images."
    )
    parser.add_argument(
        "--upcast_linear", action="store_true", help="If supplied, upcast linear transformations to fp32."
        "Only for fp8_scaled and not active during mm_scaled. Can potentially increase accuracy at little cost to speed."
    )
    parser.add_argument(
        "--upcast_quantization", action="store_true", help="If supplied, upcast quantization steps to fp32 for better accuracy."
        "Will improve quantization accuracy a bit at a small VRAM cost. Only for fp8_scaled"
    )
    parser.add_argument("--fp16_accumulation", action="store_true", help="Enable full FP16 Accmumulation in FP16 GEMMs, requires Pytorch 2.7.0 or higher")
    parser.add_argument(
        "--optimized", action="store_true",
        help="Overrides the default values of several command line args to provide an optimized but quality experience. "
        "Enables fp16_accumulation, fp8_scaled, sageattn and torch.compile. For Wan additionally enables 'rope_func comfy'. "
        "For Hunyuan/Fpack additionally enables fp8_fast. Requires SageAttention and Triton to be installed in addition to PyTorch 2.7.0 or higher!"
    )
    return parser


def parse_blissful_args(args: argparse.Namespace) -> argparse.Namespace:
    if DIFFUSION_MODEL != "framepack":
        if args.cfgzerostar_scaling and args.perp_neg is not None:
            error_out(argparse.ArgumentTypeError, "Cannot use '--cfgzerostar_scaling' with '--perp_neg'!")
    blissful_prefunc(args)
    args.seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    try:
        args.seed = int(args.seed)
    except ValueError:
        args.seed = string_to_seed(args.seed, bits=32)

    if args.prompt_wildcards is not None:
        args.prompt = process_wildcards(args.prompt, args.prompt_wildcards) if args.prompt is not None else None
        args.negative_prompt = process_wildcards(args.negative_prompt, args.prompt_wildcards) if args.negative_prompt is not None else None
        if hasattr(args, "prompt_2"):
            args.prompt_2 = process_wildcards(args.prompt_2, args.prompt_wildcards) if args.prompt2 is not None else None
    if DIFFUSION_MODEL == "wan":
        if args.riflex_index != 0 and args.rope_func.lower() != "comfy":
            logger.error("RIFLEx can only be used with rope_func == 'comfy'!")
            raise ValueError("RIFLEx can only be used with rope_func =='comfy'!")
    if DIFFUSION_MODEL in ["wan", "hunyuan"]:
        if args.upcast_linear and not args.fp8_scaled:
            error_out(argparse.ArgumentTypeError, "--upcast_linear is only for --fp8_scaled")
    return args
