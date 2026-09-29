import argparse
import copy
import gc
import logging
import random
from datetime import datetime
from importlib.util import find_spec

import torch
from safetensors.torch import load_file

from musubi_tuner.hv_generate_video import save_images_grid
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.qwen_image_21 import qwen_image_21_model, qwen_image_21_sampling, qwen_image_21_utils
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.lora_utils import filter_lora_state_dict

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
    parser.add_argument(
        "--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns, one per weight"
    )
    parser.add_argument(
        "--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns, one per weight"
    )
    parser.add_argument("--lycoris", action="store_true", help="Use LyCORIS for inference")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for generation")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--from_file", type=str, default=None, help="Read prompts from a text file")
    modes.add_argument("--interactive", action="store_true", help="Read prompts from the console")
    parser.add_argument("--negative_prompt", type=str, default=None, help="Negative prompt for CFG")
    parser.add_argument("--control_image_path", type=str, nargs="+", default=None, help="Ordered control image paths")
    resize = parser.add_mutually_exclusive_group()
    resize.add_argument("--resize_control_to_image_size", action="store_true", help="Resize and crop controls to the output size")
    resize.add_argument(
        "--resize_control_to_official_size", action="store_true", help="Resize controls to approximately 1M pixels (default)"
    )
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
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true", help="Use pinned memory for block swapping")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable numpy memory mapping when loading weights")
    model_utils.setup_parser_compile(parser)
    return parser


def load_dit_model(args: argparse.Namespace, device: torch.device, dtype: torch.dtype, multipliers: list[float] | None):
    """Load the DiT and merge filtered adapters before quantization."""
    lora_weights_list = []
    for index, path in enumerate(args.lora_weight or []):
        include = args.include_patterns[index] if args.include_patterns and len(args.include_patterns) > index else None
        exclude = args.exclude_patterns[index] if args.exclude_patterns and len(args.exclude_patterns) > index else None
        lora_weights_list.append(filter_lora_state_dict(load_file(path), include, exclude))
    transformer = qwen_image_21_model.load_model(
        args.dit,
        device=device,
        loading_device="cpu" if args.blocks_to_swap or args.lycoris else device,
        dtype=dtype,
        fp8_scaled=args.fp8_scaled and not args.lycoris,
        config_path=args.dit_config,
        disable_numpy_memmap=args.disable_numpy_memmap,
        attn_mode="torch" if args.attn_mode == "sdpa" else args.attn_mode,
        lora_weights_list=None if args.lycoris else lora_weights_list,
        lora_multipliers=multipliers,
    )
    if args.lycoris:
        from lycoris.kohya import create_network_from_weights

        for index, weights_sd in enumerate(lora_weights_list):
            network, _ = create_network_from_weights(
                multiplier=multipliers[index] if multipliers else 1.0,
                file=None,
                weights_sd=weights_sd,
                unet=transformer,
                text_encoder=None,
                vae=None,
                for_inference=True,
            )
            if not network.unet_loras:
                raise ValueError("LyCORIS weights contain no modules that match the Qwen-Image 2.1 model")
            # Pass the multiplier to each module; the Kohya wrapper merges with 1.0.
            for module in network.unet_loras:
                module.to(device="cpu", dtype=dtype)
                module.merge_to(multipliers[index] if multipliers else 1.0)
        if args.fp8_scaled:
            from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

            state_dict = optimize_state_dict_with_fp8(
                transformer.state_dict(), device, ["transformer_blocks"], ["norm"], move_to_device=args.blocks_to_swap == 0
            )
            apply_fp8_monkey_patch(transformer, state_dict, use_scaled_mm=False)
            transformer.load_state_dict(state_dict, strict=True, assign=True)
    transformer.eval().requires_grad_(False)
    if args.blocks_to_swap:
        transformer.enable_block_swap(
            args.blocks_to_swap,
            BlockSwapConfig(device, supports_backward=False, use_pinned_memory=args.use_pinned_memory_for_block_swap),
        )
        transformer.move_to_device_except_swap_blocks(device)
        transformer.switch_block_swap_for_inference()
    else:
        transformer.to(device)
    if args.compile:
        transformer = model_utils.compile_transformer(
            args, transformer, [transformer.transformer_blocks], disable_linear=args.blocks_to_swap > 0
        )
    return transformer


def parse_prompt_line(line: str) -> dict:
    """Parse a prompt and the short options used by Qwen-Image inference."""
    parts = (" " + line.strip()).split(" --")
    overrides = {"prompt": parts[0].strip()} if parts[0].strip() else {}
    options = {
        "w": ("image_size_width", int),
        "h": ("image_size_height", int),
        "d": ("seed", int),
        "s": ("infer_steps", int),
        "g": ("guidance_scale", float),
        "l": ("guidance_scale", float),
        "fs": ("flow_shift", float),
        "n": ("negative_prompt", str),
    }
    for part in parts[1:]:
        option, _, value = part.strip().partition(" ")
        if option == "ci":
            overrides.setdefault("control_image_path", []).append(value.strip())
        elif option in options:
            name, convert = options[option]
            overrides[name] = convert(value.strip())
        elif option:
            raise ValueError(f"Unknown prompt option: --{option}")
    return overrides


def apply_overrides(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    """Apply prompt options without changing the command-line defaults."""
    args = copy.deepcopy(args)
    for key, value in overrides.items():
        if key == "image_size_width":
            args.image_size[1] = value
        elif key == "image_size_height":
            args.image_size[0] = value
        else:
            setattr(args, key, value)
    return args


def generate(args: argparse.Namespace, shared_models: dict | None = None) -> list[str]:
    """Load the models and save a generated RGBA image."""
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)
    if args.prompt is None:
        raise ValueError("A prompt is required for generation")
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
    if args.lycoris and find_spec("lycoris") is None:
        raise ImportError("LyCORIS is not installed. Install lycoris-lora to use --lycoris")

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    logger.info(f"Seed: {seed}")
    prompt = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "cfg_scale": args.guidance_scale,
        "control_image_path": args.control_image_path,
    }
    te_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None and "encoder" in shared_models:
        processor, encoder = shared_models["processor"], shared_models["encoder"]
        encoder.to(te_device)
    else:
        processor, encoder = qwen_image_21_utils.load_text_encoder(args.text_encoder, te_device, dtype, args.fp8_vl)
    control_image_size = tuple(reversed(args.image_size)) if args.resize_control_to_image_size else None
    prompt = qwen_image_21_sampling.encode_sample_prompts(processor, encoder, [prompt], control_image_size=control_image_size)[0]
    if shared_models is not None:
        encoder.to("cpu")
        shared_models.update(processor=processor, encoder=encoder)
    del processor, encoder
    gc.collect()
    clean_memory_on_device(device)

    vae = shared_models.get("vae") if shared_models is not None else None
    if vae is None:
        vae = qwen_image_21_utils.load_vae(args.vae, "cpu", dtype, args.vae_tiling)
    transformer = shared_models.get("transformer") if shared_models is not None else None
    if transformer is None:
        transformer = load_dit_model(args, device, dtype, multipliers)
    elif args.blocks_to_swap:
        transformer.move_to_device_except_swap_blocks(device)
        transformer.switch_block_swap_for_inference()
    else:
        transformer.to(device)

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
    if shared_models is not None:
        transformer.to("cpu")
        shared_models.update(vae=vae, transformer=transformer)
        clean_memory_on_device(device)
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}_{seed}"
    paths = save_images_grid(pixels.unsqueeze(2), args.save_path, name, create_subdir=False)
    for path in paths:
        logger.info(f"Saved image to {path}")
    return paths


def main():
    args = setup_parser().parse_args()
    if args.from_file:
        with open(args.from_file, encoding="utf-8") as file:
            prompts = [parse_prompt_line(line) for line in file if line.strip() and not line.lstrip().startswith("#")]
        shared_models = {}
        for index, prompt in enumerate(prompts):
            logger.info(f"Generating image {index + 1}/{len(prompts)}")
            generate(apply_overrides(args, prompt), shared_models)
    elif args.interactive:
        shared_models = {}
        while True:
            try:
                line = input("Enter prompt (q to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line.lower() in ("q", "quit", "exit"):
                break
            if line:
                generate(apply_overrides(args, parse_prompt_line(line)), shared_models)
    else:
        generate(args)


if __name__ == "__main__":
    main()
