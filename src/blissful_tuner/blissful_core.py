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
import torch
from rich.traceback import install as install_rich_tracebacks
from blissful_tuner.utils import error_out
from blissful_tuner.blissful_logger import BlissfulLogger
from blissful_tuner.prompt_management import process_wildcards
from blissful_tuner.utils import power_seed

logger = BlissfulLogger(__name__, "#8e00ed")

BLISSFUL_VERSION = "0.10.66"

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
elif "flux_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "flux"
elif "qwen_" in ROOT_SCRIPT:
    DIFFUSION_MODEL = "qwen"

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
    cuda_list = [f"Python: {sys.version.split(' ')[0]}"]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocator = torch.cuda.get_allocator_backend()
        cuda = torch.cuda.get_device_properties(0)
        cuda_list[0] += f", CUDA: {torch.version.cuda} CC: {cuda.major}.{cuda.minor}"
        cuda_list.append(f"Device: '{cuda.name}', VRAM: '{cuda.total_memory // 1024**2}MB'")
    logger.info(f"Blissful Tuner version {BLISSFUL_VERSION} extended from Musubi Tuner!")
    logger.info(f"PyTorch: {torch.__version__}, Memory allocation: '{allocator}'")
    for string in cuda_list:
        logger.info(string)

    if hasattr(args, "optimized") and args.optimized and MODE == "generate":
        logger.info("Optimized arguments enabled!")
        args.fp16_accumulation = True
        args.attn_mode = "sageattn"
        args.compile = True
        args.fp8_scaled = True
        if DIFFUSION_MODEL == "wan":
            args.rope_func = "comfy"
            args.simple_modulation = True
        elif DIFFUSION_MODEL in ["hunyuan", "framepack"]:
            args.fp16_accumulation = False  # Disable this for hunyuan and framepack b/c we enable fp8_fast which offsets it anyway and torch 2.7.0 has issues with compiling hunyuan sometimes
            args.fp8_fast = True
        elif DIFFUSION_MODEL == "flux":
            args.compile = False
            args.fp16_accumulation = False
            args.fp8_fast = True
    if hasattr(args, "fp16_accumulation") and args.fp16_accumulation and MODE == "generate":
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
        parser.add_argument(
            "--nag_scale",
            type=float,
            default=None,
            help="Enable Normalized Attention Guidance (NAG) and set scale. The scale for attention feature extrapolation. Higher values result in stronger negative guidance.",
        )
        parser.add_argument(
            "--nag_tau",
            type=float,
            default=3.5,
            help="The normalisation threshold. Higher values result in stronger negative guidance.",
        )
        parser.add_argument(
            "--nag_alpha",
            type=float,
            default=0.5,
            help="0 to 1, Blending factor between original and extrapolated attention. Higher values result in stronger negative guidance.",
        )
        parser.add_argument(
            "--nag_prompt",
            type=str,
            default=None,
            help="Allows to specify a separate negative prompt for NAG to avoid overbroad guidance. If not specified and NAG is enabled it will use the regular negative prompt.",
        )
        parser.add_argument(
            "--optimized_compile",
            action="store_true",
            help="Enable optimized torch.compile of just the most crucial blocks. Exclusive of --compile. Works best with --rope_func comfy",
        )
        parser.add_argument(
            "--simple_modulation",
            action="store_true",
            help="Use Wan 2.1 style modulation even for Wan 2.2 to save lots of VRAM. With this and --lazy_loading, 2.2 should use same VRAM as 2.1 ceteris paribus",
        )
        parser.add_argument(
            "--lower_precision_attention",
            action="store_true",
            help="Do parts of attention calculation in and maintain e tensor in float16 to save some VRAM at small cost to quality.",
        )
        parser.add_argument(
            "--i2i_path",
            type=str,
            default=None,
            help="path to an image for image2image inference. Use with T2V model and T2I task.",
        )
        parser.add_argument(
            "--v2_extra_noise",
            type=float,
            default=None,
            help="Extra latent noise for v2v. Low values are best e.g. 0.015. Can help add fine details, especially when upscaling(output res > input res)",
        )
        parser.add_argument(
            "--i2_extra_noise",
            type=float,
            default=None,
            help="Extra latent noise for i2v. Low values are best e.g. 0.025. Can help add fine details, especially when upscaling(output res > input res)",
        )
        parser.add_argument("--denoise_strength", type=float, default=0.5, help="Amount of denoising to do for V2V or I2I, 0.0-1.0")
        parser.add_argument(
            "--noise_mode",
            choices=["traditional", "direct"],
            default="traditional",
            help="Controls how --denoise_strength value works. Traditional is like usual and controls what percent of the timestep schedule to run. "
            "Direct allows you to directly control how much noise will be added.",
        )
        parser.add_argument(
            "--v2v_pad_mode",
            type=str,
            choices=["front", "end"],
            default="end",
            help="Padding mode for when V2V input is shorter than requested output",
        )
        parser.add_argument("--prompt_weighting", action="store_true", help="Enable (prompt weighting:1.2)")
        parser.add_argument(
            "--rope_func",
            type=str,
            choices=["default", "comfy"],
            default="default",
            help="Function to use for ROPE. Choose from 'default' or 'comfy' the latter of which uses ComfyUI implementation and is compilable with torch.compile to enable BIG VRAM savings",
        )
        parser.add_argument(
            "--mixed_precision_transformer",
            action="store_true",
            help="Allow loading mixed precision transformer such as a combination of float16 weights / float32 everything else",
        )
    elif DIFFUSION_MODEL == "hunyuan":
        parser.add_argument("--from_latent", type=str, default=None, help="Input path for latent.safetensors to use for V2V")
        parser.add_argument("--scheduler", type=str, choices=["default", "dpm++", "dpm++sde"], default="default")
        parser.add_argument(
            "--disable_embedded_for_cfg",
            action="store_true",
            help="Fully disable embedded guidance when doing CFG. Allows higher CFG scale and better control.",
        )
        parser.add_argument(
            "--hidden_state_skip_layer",
            type=int,
            default=2,
            help="Hidden state skip layer for LLM. Default is 2. Think 'clip skip' for the LLM",
        )
        parser.add_argument(
            "--apply_final_norm",
            action="store_true",
            help="Apply final norm for LLM. Default is False. Usually makes things worse.",
        )
        parser.add_argument(
            "--reproduce", action="store_true", help="Enable reproducible output(Same seed = same result. Default is False."
        )
        parser.add_argument(
            "--fp8_scaled",
            action="store_true",
            help="Scaled FP8 quantization. Better quality/accuracy with slightly more VRAM usage.",
        )
        parser.add_argument("--prompt_2", type=str, default=None, help="Optional different prompt for CLIP")
        parser.add_argument(
            "--te_multiplier", nargs=2, metavar=("llm_multiplier", "clip_multiplier"), help="Scale clip and llm influence"
        )
    elif DIFFUSION_MODEL == "framepack":
        parser.add_argument(
            "--preview_latent_every",
            type=int,
            default=None,
            help="Enable latent preview every N sections. If --preview_vae is not specified it will use latent2rgb",
        )
        parser.add_argument(
            "--te_multiplier", nargs=2, metavar=("llm_multiplier", "clip_multiplier"), help="Scale clip and llm influence"
        )

    if DIFFUSION_MODEL in ["wan", "hunyuan"]:
        parser.add_argument(
            "--riflex_index",
            type=int,
            default=0,
            help="Frequency for RifleX extension. 4 is good for Hunyuan, 6 is good for Wan. Only 'comfy' rope_func supports this with Wan!",
        )
        parser.add_argument(
            "--cfgzerostar_scaling",
            action="store_true",
            help="Enables CFG-Zero* scaling - https://github.com/WeichenFan/CFG-Zero-star",
        )
        parser.add_argument(
            "--cfgzerostar_multiplier",
            type=float,
            default=0,
            help="Multiplier used for cfgzerostar_init. Default is 0 which zeroes the step. 1 would be like not using zero init.",
        )
        parser.add_argument(
            "--cfgzerostar_init_steps",
            type=int,
            default=-1,
            help="Enables CFGZero* zeroing out the first N steps. 2 is good for Wan T2V, 1 for I2V",
        )
        parser.add_argument(
            "--preview_latent_every",
            type=int,
            default=None,
            help="Enable latent preview every N steps. If --preview_vae is not specified it will use latent2rgb",
        )
        parser.add_argument("--cfg_schedule", type=str, default=None, help=CFG_SCHEDULE_HELP)
        parser.add_argument(
            "--perp_neg",
            type=float,
            default=None,
            help="Enable and set scale for perpendicular negative guidance. Start with like 1.5 - 2.0. This is a stronger, more precise form of CFG."
            "It requires a third modeling pass per step so it will slow you down by 30 percent but can definitely help isolating concepts. An example would be "
            "prompting for 'nature photography' when using 'tree' as a negative. Normal CFG will mostly avoid trees but because 'nature' and 'tree' are associated "
            "there will likely still be some. With perp neg, the entire concept of tree is subtracted from the result.",
        )
    # Common
    parser.add_argument(
        "--prompt_wildcards",
        type=str,
        default=None,
        help="Path to a directory of wildcard.txt files to enable wildcards in prompts and negative prompts. For instance __color__ will look for wildcards in color.txt in that directory. "
        "Wildcard files should have one possible replacement per line, optionally with a relative weight attached like red:2.0 or yellow:0.5, and wildcards can be nested.",
    )
    parser.add_argument("--preview_vae", type=str, help="Path to TAE vae for taehv previews")
    parser.add_argument("--keep_pngs", action="store_true", help="Save frames as PNGs in addition to output video")
    parser.add_argument(
        "--codec",
        choices=["h264", "h265", "prores"],
        default="h264",
        help="Codec to use when saving videos, choose from 'prores', 'h264', or 'h265'. Default is 'h264'",
    )
    parser.add_argument(
        "--container",
        choices=["mkv", "mp4"],
        default="mp4",
        help="Container format to use for output, choose from 'mkv' or 'mp4'. Default is 'mp4' and note that 'prores' can only go in 'mkv'! Ignored for images.",
    )
    parser.add_argument(
        "--fp16_accumulation",
        action="store_true",
        help="Enable full FP16 Accmumulation in FP16 GEMMs, requires Pytorch 2.7.0 or higher",
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Overrides the default values of several command line args to provide an optimized but quality experience. "
        "Enables fp16_accumulation, fp8_scaled, sageattn and torch.compile. For Wan additionally enables 'rope_func comfy'. "
        "For Hunyuan/Fpack additionally enables fp8_fast. Requires SageAttention and Triton to be installed in addition to PyTorch 2.7.0 or higher!",
    )
    return parser


def parse_blissful_args(args: argparse.Namespace) -> argparse.Namespace:
    blissful_prefunc(args)
    if DIFFUSION_MODEL in ["wan", "hunyuan"]:
        if args.cfgzerostar_scaling and args.perp_neg is not None:
            error_out(argparse.ArgumentTypeError, "Cannot use '--cfgzerostar_scaling' with '--perp_neg'!")
    args.seed = power_seed(args.seed)  # Save it back because it might have been a STR before
    if args.prompt_wildcards is not None:
        args.prompt = process_wildcards(args.prompt, args.prompt_wildcards) if args.prompt is not None else None
        args.negative_prompt = (
            process_wildcards(args.negative_prompt, args.prompt_wildcards) if args.negative_prompt is not None else None
        )
        if hasattr(args, "prompt_2"):
            args.prompt_2 = process_wildcards(args.prompt_2, args.prompt_wildcards) if args.prompt2 is not None else None
    if DIFFUSION_MODEL == "wan":
        if args.nag_scale and args.nag_alpha > 1:
            logger.warning(f"NAG alpha requested is {args.nag_alpha} which is greater than 1. Results will be unpredictablee!")
        if args.compile and args.optimized_compile:
            error_out(argparse.ArgumentTypeError, "Only one of --compile and --optimized compile may be used.")
        if args.perp_neg is not None and args.slg_mode == "original":
            error_out(argparse.ArgumentTypeError, "--perp_neg cannot be used with --slg_mode 'original'")
        if args.riflex_index != 0 and args.rope_func.lower() != "comfy":
            error_out(argparse.ArgumentTypeError, "RIFLEx can only be used with rope_func =='comfy'!")
    return args


# =====================================================Region Flux============================================================================


def add_blissful_flux_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:  # Todo: individualize the others to avoid the clusterf*** above
    parser.add_argument(
        "--offload_transformer_for_decode", action="store_true", help="Free VRAM for decode by offloading transformer"
    )
    parser.add_argument(
        "--scheduler", type=str, choices=["dpm++", "euler"], default="euler", help="Scheduler/sampler to use for denoise"
    )
    parser.add_argument(
        "--fp32_cpu_te",
        action="store_true",
        help="Load and run text encoders on CPU in FP32 for extra precision. Takes just a few seconds extra but you should start with fp32 model as well!",
    )
    parser.add_argument(
        "--fp32_working_dtype",
        action="store_true",
        help="Input contexts, noise etc in torch.float32 and autocast to torch.float32 - slow but max quality. Best with --fp32_cpu_te",
    )
    parser.add_argument("--preview_vae", type=str, help="Path to TAE vae for taehv previews")
    parser.add_argument(
        "--preview_latent_every",
        type=int,
        default=None,
        help="Enable latent preview every N steps. If --preview_vae is not specified it will use latent2rgb",
    )
    parser.add_argument(
        "--cfgzerostar_scaling", action="store_true", help="Enables CFG-Zero* scaling - https://github.com/WeichenFan/CFG-Zero-star"
    )
    parser.add_argument(
        "--cfgzerostar_multiplier",
        type=float,
        default=0,
        help="Multiplier used for cfgzerostar_init. Default is 0 which zeroes the step. 1 would be like not using zero init.",
    )
    parser.add_argument(
        "--cfgzerostar_init_steps",
        type=int,
        default=-1,
        help="Enables CFGZero* zeroing out the first N steps. 2 is good for Wan T2V, 1 for I2V",
    )
    parser.add_argument("--cfg_schedule", type=str, default=None, help=CFG_SCHEDULE_HELP)
    parser.add_argument(
        "--fp16_accumulation", action="store_true", help="Enable FP16 accumulation in GEMMS for speed. Requires Pytorch 2.7+"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="If > 1.0, enables and sets guidance scale for CFG(Classifier Free Guidance)",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None, help="Specify a negative prompt for CFG(Classifier Free Guidance)"
    )
    parser.add_argument(
        "--optimized",
        action="store_true",
        help="Overrides the default values of several command line args to provide an optimized but quality experience. "
        "Enables fp16_accumulation, fp8_scaled, sageattn and torch.compile. For Wan additionally enables 'rope_func comfy'. "
        "For Hunyuan/Fpack additionally enables fp8_fast. Requires SageAttention and Triton to be installed in addition to PyTorch 2.7.0 or higher!",
    )
    parser.add_argument(
        "--prompt_wildcards",
        type=str,
        default=None,
        help="Path to a directory of wildcard.txt files to enable wildcards in prompts and negative prompts. For instance __color__ will look for wildcards in color.txt in that directory. "
        "Wildcard files should have one possible replacement per line, optionally with a relative weight attached like red:2.0 or yellow:0.5, and wildcards can be nested.",
    )
    return parser


# =====================================================Region Qwen============================================================================


def add_blissful_qwen_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        default=["inductor", "default", None, "False"],
        help="Torch.compile settings",
    )
    parser.add_argument(
        "--prompt_wildcards",
        type=str,
        default=None,
        help="Path to a directory of wildcard.txt files to enable wildcards in prompts and negative prompts. For instance __color__ will look for wildcards in color.txt in that directory. "
        "Wildcard files should have one possible replacement per line, optionally with a relative weight attached like red:2.0 or yellow:0.5, and wildcards can be nested.",
    )
    return parser
