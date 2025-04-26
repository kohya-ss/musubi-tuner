#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 15:11:58 2025

@author: blyss
"""
import argparse
from blissful_tuner.utils import BlissfulLogger, string_to_seed, parse_scheduled_cfg
logger = BlissfulLogger(__name__, "#8e00ed")


def add_blissful_args(parser: argparse.ArgumentParser, mode: str = "wan") -> argparse.ArgumentParser:
    if mode == "wan":
        parser.add_argument("--noise_aug_strength", type=float, default=0.0, help="Additional multiplier for i2v noise, higher might help motion/quality")
        parser.add_argument("--prompt_weighting", action="store_true", help="Enable (AUTOMATIC1111) [style] (prompt weighting:1.2)")
        parser.add_argument("--cfgzerostar_scaling", action="store_true", help="Enables CFG-Zero* scaling - https://github.com/WeichenFan/CFG-Zero-star")
        parser.add_argument("--cfgzerostar_init_steps", type=int, default=-1, help="Enables CFGZero* zeroing out the first N steps.")
        parser.add_argument("--rope_func", type=str, default="default", help="Function to use for ROPE. Choose from 'default' or 'comfy' the latter of which uses ComfyUI implementation and is compilable with torch.compile")
        parser.add_argument("--riflex_index", type=int, default=0, help="Frequency for RifleX extension. 6 is good for Wan. Only 'comfy' rope_func supports this!")

    elif mode == "hunyuan":
        parser.add_argument("--hidden_state_skip_layer", type=int, default=2, help="Hidden state skip layer for LLM. Default is 2.")
        parser.add_argument("--apply_final_norm", type=bool, default=False, help="Apply final norm for LLM. Default is False.")
        parser.add_argument("--reproduce", action="store_true", help="Enable reproducible output(Same seed = same result. Default is False.")
        parser.add_argument("--fp8_scaled", action="store_true", help="Scaled FP8 quantization")
        parser.add_argument("--prompt_2", type=str, required=False, help="Optional different prompt for CLIP")
        parser.add_argument(
            "--te_multiplier",
            nargs=2,
            metavar=("llm_multiplier", "clip_multiplier"),
            help="Scale clip and llm influence"
        )
    # Common
    parser.add_argument("--fp16_accumulation", action="store_true", help="Enable full FP16 Accmumulation in FP16 GEMMs, requires Pytorch Nightly")
    parser.add_argument("--preview_latent_every", type=int, default=None, help="Enable latent preview every N steps")
    parser.add_argument("--preview_vae", type=str, help="Path to TAE vae for taehv previews")
    parser.add_argument("--keep_pngs", action="store_true", help="Also keep individual frames as PNGs")
    parser.add_argument(
        "--cfg_schedule",
        type=str,
        help="Comma-separated list of steps/ranges where CFG should be applied (e.g. '1-10,20,40-50')."
    )
    parser.add_argument(
        "--codec", choices=["prores", "h264", "h265"], default=None,
        help="Codec to use, choose from 'prores', 'h264', or 'h265'"
    )
    parser.add_argument(
        "--container", choices=["mkv", "mp4"], default="mkv",
        help="Container format to use, choose from 'mkv' or 'mp4'. Note prores can only go in MKV!"
    )
    return parser


def parse_blissful_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.seed is not None:
        try:
            args.seed = int(args.seed)
        except ValueError:
            string_seed = args.seed
            args.seed = string_to_seed(args.seed)
            logger.info(f"Seed {args.seed} was generated from string '{string_seed}'!")
    if args.cfg_schedule:
        args.cfg_schedule = parse_scheduled_cfg(args.cfg_schedule, args.infer_steps, args.guidance_scale)
    return args
