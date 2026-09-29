import argparse
import gc
import logging
import random
from datetime import datetime

import torch
from safetensors.torch import load_file

from musubi_tuner.hv_generate_video import save_images_grid
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.qwen_image_21 import qwen_image_21_model, qwen_image_21_sampling, qwen_image_21_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    """Create the Qwen-Image 2.1 inference parser."""
    parser = argparse.ArgumentParser(description="Qwen-Image 2.1 inference script")
    parser.add_argument("--dit", type=str, required=True, help="DiT directory or safetensors file")
    parser.add_argument("--dit_config", type=str, default=None, help="DiT config.json for a single-file checkpoint")
    parser.add_argument("--vae", type=str, required=True, help="VAE directory or safetensors file")
    parser.add_argument("--vae_tiling", action="store_true", help="Enable VAE spatial tiling")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL directory or safetensors file")
    parser.add_argument("--text_encoder_cpu", action="store_true", help="Run the text encoder on CPU")
    parser.add_argument("--fp8_vl", action="store_true", help="Use FP8 storage for the text encoder language blocks")
    parser.add_argument("--lora_weight", type=str, nargs="+", default=None, help="LoRA weight paths")
    parser.add_argument("--lora_multiplier", type=float, nargs="+", default=None, help="LoRA multipliers, default is 1.0")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for CFG")
    parser.add_argument("--control_image_path", type=str, nargs="+", default=None, help="Ordered control image paths")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="Image size: height width")
    parser.add_argument("--infer_steps", type=int, default=40, help="Number of denoising steps")
    parser.add_argument(
        "--guidance_scale", type=float, default=1.0, help="CFG scale, requires a negative prompt when greater than 1"
    )
    parser.add_argument("--flow_shift", type=float, default=None, help="Fixed flow shift; default uses dynamic shifting")
    parser.add_argument("--save_path", type=str, required=True, help="Directory for generated PNG images")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use, default is CUDA if available")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Model compute dtype")
    parser.add_argument("--attn_mode", choices=["sdpa", "torch", "flash", "flash3", "xformers", "sageattn"], default="sdpa")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled FP8 for DiT weights")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of DiT blocks to swap to CPU")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable numpy memory mapping when loading weights")
    return parser


def generate(args: argparse.Namespace) -> list[str]:
    """Load the models and save a generated RGBA image."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)
    if args.infer_steps < 1 or min(args.image_size) < 32 or any(size % 32 for size in args.image_size):
        raise ValueError("Image dimensions must be positive multiples of 32 and --infer_steps must be positive")
    if args.flow_shift is not None and args.flow_shift <= 0:
        raise ValueError("--flow_shift must be positive")
    if args.guidance_scale > 1 and args.negative_prompt is None:
        raise ValueError("--guidance_scale greater than 1 requires --negative_prompt")
    multipliers = args.lora_multiplier
    if multipliers is not None:
        if not args.lora_weight or len(multipliers) not in (1, len(args.lora_weight)):
            raise ValueError("Specify one --lora_multiplier or one per LoRA weight")
        if len(multipliers) == 1:
            multipliers = multipliers * len(args.lora_weight)
    if args.blocks_to_swap < 0:
        raise ValueError("--blocks_to_swap must be non-negative")

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")
    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "cfg_scale": args.guidance_scale,
        "control_image_path": args.control_image_path,
    }
    te_device = torch.device("cpu") if args.text_encoder_cpu else device
    processor, encoder = qwen_image_21_utils.load_text_encoder(args.text_encoder, te_device, dtype, args.fp8_vl)
    prompt = qwen_image_21_sampling.encode_sample_prompts(processor, encoder, [prompt])[0]
    del processor, encoder
    gc.collect()
    clean_memory_on_device(device)

    vae = qwen_image_21_utils.load_vae(args.vae, "cpu", dtype, args.vae_tiling)
    transformer = qwen_image_21_model.load_model(
        args.dit,
        device=device,
        loading_device="cpu" if args.blocks_to_swap else device,
        dtype=dtype,
        fp8_scaled=args.fp8_scaled,
        config_path=args.dit_config,
        disable_numpy_memmap=args.disable_numpy_memmap,
        attn_mode="torch" if args.attn_mode == "sdpa" else args.attn_mode,
        lora_weights_list=[load_file(path) for path in args.lora_weight] if args.lora_weight else None,
        lora_multipliers=multipliers,
    )
    transformer.eval().requires_grad_(False)
    if args.blocks_to_swap:
        transformer.enable_block_swap(args.blocks_to_swap, BlockSwapConfig(device, supports_backward=False))
        transformer.move_to_device_except_swap_blocks(device)
        transformer.switch_block_swap_for_inference()

    height, width = args.image_size
    with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
        pixels = qwen_image_21_sampling.sample_image(
            transformer,
            vae,
            sample_parameter=prompt,
            device=device,
            dit_dtype=dtype,
            width=width,
            height=height,
            sample_steps=args.infer_steps,
            generator=torch.Generator(device=device).manual_seed(seed),
            discrete_flow_shift=args.flow_shift,
            cfg_scale=args.guidance_scale,
            do_classifier_free_guidance=args.negative_prompt is not None,
        )
    vae.to("cpu")
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{seed}"
    paths = save_images_grid(pixels.unsqueeze(2), args.save_path, name, create_subdir=False)
    for path in paths:
        logger.info(f"Saved image to {path}")
    return paths


def main():
    args = setup_parser().parse_args()
    generate(args)


if __name__ == "__main__":
    main()
