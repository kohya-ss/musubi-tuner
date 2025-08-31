e#!/usr/bin/env python3
"""
WAN 2.2 T2V Inference Script for Character LoRA
This script generates videos using trained high noise and low noise LoRA weights.
"""

import os
import sys
import argparse
from pathlib import Path

def create_inference_command(
    # Model paths
    dit_model_path: str,
    dit_high_noise_path: str = None,
    vae_path: str = None,
    t5_path: str = None,

    # LoRA paths
    lora_low_noise_path: str = None,
    lora_high_noise_path: str = None,
    lora_multiplier: float = 1.0,
    lora_multiplier_high_noise: float = 1.0,

    # Generation parameters
    prompt: str = "A video of CHARACTER_VANRAJ_V1 walking in a garden",
    negative_prompt: str = None,
    video_size: tuple = (512, 512),  # (height, width)
    video_length: int = 81,  # frames
    fps: int = 16,
    infer_steps: int = None,  # Will use task default if None
    seed: int = 42,

    # Quality and guidance parameters
    guidance_scale: float = None,  # Will use task default if None
    guidance_scale_high_noise: float = None,  # Will use task default if None
    flow_shift: float = None,  # Will use task default if None
    timestep_boundary: float = None,  # For switching between high/low noise models

    # Sampling parameters
    sample_solver: str = "unipc",  # "unipc", "dpm++", "vanilla"
    cfg_skip_mode: str = "none",  # "early", "late", "middle", "early_late", "alternate", "none"
    cfg_apply_ratio: float = None,

    # Skip Layer Guidance (SLG) parameters
    slg_layers: str = None,  # e.g., "0,1,2" for layers to skip
    slg_scale: float = 3.0,
    slg_start: float = 0.0,
    slg_end: float = 0.3,
    slg_mode: str = None,  # "original", "uncond"

    # Performance parameters
    task: str = "t2v-14B",  # "t2v-14B", "t2v-1.3B", "i2v-14B", "t2i-14B"
    device: str = "cuda",
    vae_dtype: str = "bfloat16",
    fp8: bool = False,
    fp8_scaled: bool = False,
    fp8_fast: bool = False,
    fp8_t5: bool = False,
    blocks_to_swap: int = 0,
    offload_inactive_dit: bool = False,
    vae_cache_cpu: bool = False,
    cpu_noise: bool = False,
    compile_model: bool = False,

    # Output parameters
    save_path: str = "./outputs",
    output_type: str = "video",  # "video", "images", "latent", "both", "latent_images"
    trim_tail_frames: int = 0,
    no_metadata: bool = False,
):
    """
    Create the inference command for WAN video generation.

    Args:
        dit_model_path: Path to the main DiT model (low noise)
        dit_high_noise_path: Path to high noise DiT model (optional)
        vae_path: Path to VAE model
        t5_path: Path to T5 text encoder
        lora_low_noise_path: Path to low noise LoRA weights
        lora_high_noise_path: Path to high noise LoRA weights
        lora_multiplier: Multiplier for low noise LoRA
        lora_multiplier_high_noise: Multiplier for high noise LoRA
        prompt: Text prompt for generation (should include CHARACTER_VANRAJ_V1)
        negative_prompt: Negative prompt
        video_size: (height, width) tuple
        video_length: Number of frames
        fps: Frames per second
        infer_steps: Number of inference steps
        seed: Random seed
        guidance_scale: CFG scale for low noise
        guidance_scale_high_noise: CFG scale for high noise
        flow_shift: Flow matching shift parameter
        timestep_boundary: Boundary for switching models (0.0-1.0)
        sample_solver: Sampling solver to use
        cfg_skip_mode: CFG skip strategy
        cfg_apply_ratio: Ratio of steps to apply CFG
        slg_layers: Skip layer guidance layers
        slg_scale: SLG scale factor
        slg_start: SLG start ratio
        slg_end: SLG end ratio
        slg_mode: SLG mode
        task: WAN task configuration
        device: Device to use
        vae_dtype: VAE data type
        fp8: Use FP8 for DiT
        fp8_scaled: Use scaled FP8
        fp8_fast: Enable fast FP8
        fp8_t5: Use FP8 for T5
        blocks_to_swap: Number of blocks to swap
        offload_inactive_dit: Offload inactive DiT to CPU
        vae_cache_cpu: Cache VAE features on CPU
        cpu_noise: Generate noise on CPU
        compile_model: Enable torch.compile
        save_path: Output directory
        output_type: Type of output to save
        trim_tail_frames: Frames to trim from end
        no_metadata: Skip saving metadata

    Returns:
        str: Complete command line for inference
    """

    cmd = ["python", "wan_generate_video.py"]

    # Required parameters
    cmd.extend(["--task", task])
    cmd.extend(["--save_path", save_path])
    cmd.extend(["--prompt", f'"{prompt}"'])

    # Model paths
    if dit_model_path:
        cmd.extend(["--dit", dit_model_path])
    if dit_high_noise_path:
        cmd.extend(["--dit_high_noise", dit_high_noise_path])
    if vae_path:
        cmd.extend(["--vae", vae_path])
    if t5_path:
        cmd.extend(["--t5", t5_path])

    # LoRA parameters
    if lora_low_noise_path:
        cmd.extend(["--lora_weight", lora_low_noise_path])
        cmd.extend(["--lora_multiplier", str(lora_multiplier)])

    if lora_high_noise_path:
        cmd.extend(["--lora_weight_high_noise", lora_high_noise_path])
        cmd.extend(["--lora_multiplier_high_noise", str(lora_multiplier_high_noise)])

    # Generation parameters
    if negative_prompt:
        cmd.extend(["--negative_prompt", f'"{negative_prompt}"'])

    cmd.extend(["--video_size", str(video_size[0]), str(video_size[1])])
    cmd.extend(["--video_length", str(video_length)])
    cmd.extend(["--fps", str(fps)])
    cmd.extend(["--seed", str(seed)])

    if infer_steps:
        cmd.extend(["--infer_steps", str(infer_steps)])

    # Quality and guidance parameters
    if guidance_scale is not None:
        cmd.extend(["--guidance_scale", str(guidance_scale)])
    if guidance_scale_high_noise is not None:
        cmd.extend(["--guidance_scale_high_noise", str(guidance_scale_high_noise)])
    if flow_shift is not None:
        cmd.extend(["--flow_shift", str(flow_shift)])
    if timestep_boundary is not None:
        cmd.extend(["--timestep_boundary", str(timestep_boundary)])

    # Sampling parameters
    cmd.extend(["--sample_solver", sample_solver])

    if cfg_skip_mode != "none":
        cmd.extend(["--cfg_skip_mode", cfg_skip_mode])
    if cfg_apply_ratio is not None:
        cmd.extend(["--cfg_apply_ratio", str(cfg_apply_ratio)])

    # Skip Layer Guidance
    if slg_layers:
        cmd.extend(["--slg_layers", slg_layers])
        cmd.extend(["--slg_scale", str(slg_scale)])
        cmd.extend(["--slg_start", str(slg_start)])
        cmd.extend(["--slg_end", str(slg_end)])
        if slg_mode:
            cmd.extend(["--slg_mode", slg_mode])

    # Performance parameters
    cmd.extend(["--device", device])
    cmd.extend(["--vae_dtype", vae_dtype])

    if fp8:
        cmd.append("--fp8")
    if fp8_scaled:
        cmd.append("--fp8_scaled")
    if fp8_fast:
        cmd.append("--fp8_fast")
    if fp8_t5:
        cmd.append("--fp8_t5")

    if blocks_to_swap > 0:
        cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
    if offload_inactive_dit:
        cmd.append("--offload_inactive_dit")
    if vae_cache_cpu:
        cmd.append("--vae_cache_cpu")
    if cpu_noise:
        cmd.append("--cpu_noise")
    if compile_model:
        cmd.append("--compile")

    # Output parameters
    cmd.extend(["--output_type", output_type])
    if trim_tail_frames > 0:
        cmd.extend(["--trim_tail_frames", str(trim_tail_frames)])
    if no_metadata:
        cmd.append("--no_metadata")

    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="WAN 2.2 T2V Character LoRA Inference Script")

    # Model paths
    parser.add_argument("--dit_model", type=str, required=True,
                       help="Path to the main DiT model (low noise)")
    parser.add_argument("--dit_high_noise", type=str, default=None,
                       help="Path to high noise DiT model (optional)")
    parser.add_argument("--vae", type=str, default=None,
                       help="Path to VAE model")
    parser.add_argument("--t5", type=str, default=None,
                       help="Path to T5 text encoder")

    # LoRA paths
    parser.add_argument("--lora_low_noise", type=str, required=True,
                       help="Path to low noise LoRA weights")
    parser.add_argument("--lora_high_noise", type=str, default=None,
                       help="Path to high noise LoRA weights")
    parser.add_argument("--lora_multiplier", type=float, default=1.0,
                       help="Multiplier for low noise LoRA")
    parser.add_argument("--lora_multiplier_high_noise", type=float, default=1.0,
                       help="Multiplier for high noise LoRA")

    # Generation parameters
    parser.add_argument("--prompt", type=str,
                       default="A video of CHARACTER_VANRAJ_V1 walking in a beautiful garden during sunset",
                       help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt")
    parser.add_argument("--video_size", type=int, nargs=2, default=[512, 512],
                       help="Video size (height width)")
    parser.add_argument("--video_length", type=int, default=81,
                       help="Number of frames")
    parser.add_argument("--fps", type=int, default=16,
                       help="Frames per second")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    # Quality parameters
    parser.add_argument("--guidance_scale", type=float, default=None,
                       help="CFG scale for low noise")
    parser.add_argument("--guidance_scale_high_noise", type=float, default=None,
                       help="CFG scale for high noise")
    parser.add_argument("--infer_steps", type=int, default=None,
                       help="Number of inference steps")

    # Output
    parser.add_argument("--save_path", type=str, default="./outputs",
                       help="Output directory")
    parser.add_argument("--output_type", type=str, default="video",
                       choices=["video", "images", "latent", "both", "latent_images"],
                       help="Output type")

    # Performance
    parser.add_argument("--task", type=str, default="t2v-14B",
                       choices=["t2v-14B", "t2v-1.3B", "i2v-14B", "t2i-14B"],
                       help="WAN task configuration")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    parser.add_argument("--fp8", action="store_true",
                       help="Use FP8 for DiT")
    parser.add_argument("--blocks_to_swap", type=int, default=0,
                       help="Number of blocks to swap to CPU")
    parser.add_argument("--compile", action="store_true",
                       help="Enable torch.compile")

    # Advanced options
    parser.add_argument("--dry_run", action="store_true",
                       help="Print command without executing")
    parser.add_argument("--execute", action="store_true",
                       help="Execute the command directly")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)

    # Generate the command
    cmd = create_inference_command(
        dit_model_path=args.dit_model,
        dit_high_noise_path=args.dit_high_noise,
        vae_path=args.vae,
        t5_path=args.t5,
        lora_low_noise_path=args.lora_low_noise,
        lora_high_noise_path=args.lora_high_noise,
        lora_multiplier=args.lora_multiplier,
        lora_multiplier_high_noise=args.lora_multiplier_high_noise,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        video_size=tuple(args.video_size),
        video_length=args.video_length,
        fps=args.fps,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        guidance_scale_high_noise=args.guidance_scale_high_noise,
        infer_steps=args.infer_steps,
        task=args.task,
        device=args.device,
        fp8=args.fp8,
        blocks_to_swap=args.blocks_to_swap,
        compile_model=args.compile,
        save_path=args.save_path,
        output_type=args.output_type,
    )

    print("Generated inference command:")
    print(cmd)
    print()

    if args.dry_run:
        print("Dry run mode - command not executed")
        return

    if args.execute:
        print("Executing command...")
        os.system(cmd)
    else:
        print("To execute this command, run with --execute flag or copy the command above")


if __name__ == "__main__":
    main()
