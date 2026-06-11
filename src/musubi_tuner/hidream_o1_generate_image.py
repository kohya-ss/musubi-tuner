import argparse
import logging
import os
import re

import torch
from safetensors.torch import load_file

from musubi_tuner.hidream_o1 import hidream_o1_utils
from musubi_tuner.hidream_o1.pipeline import generate_image
from musubi_tuner.networks import lora_hidream_o1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HiDream-O1-Image inference script")
    parser.add_argument(
        "--dit", type=str, required=True, help="HiDream-O1 single checkpoint (.safetensors) or model weights directory"
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for generation")
    parser.add_argument("--ref_images", nargs="*", default=[], help="Reference image path(s) for editing or subject guidance")
    parser.add_argument("--save_path", "--output_image", dest="save_path", type=str, required=True, help="Path to save image")
    parser.add_argument("--image_size", type=int, nargs=2, default=[2048, 2048], help="Image size as height width")
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "dev"], help="HiDream-O1 model variant")
    parser.add_argument("--infer_steps", type=int, default=None, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=32, help="Seed")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="CFG guidance scale for full model")
    parser.add_argument("--flow_shift", type=float, default=3.0, help="Flow shift for full model")
    parser.add_argument("--noise_scale_start", type=float, default=7.5, help="Dev scheduler start noise scale")
    parser.add_argument("--noise_scale_end", type=float, default=7.5, help="Dev scheduler end noise scale")
    parser.add_argument("--noise_clip_std", type=float, default=2.5, help="Dev scheduler noise clipping std")
    parser.add_argument(
        "--editing_scheduler",
        type=str,
        default="flow_match",
        choices=["flow_match", "flash"],
        help="Scheduler for dev-model editing with exactly one reference image",
    )
    parser.add_argument("--keep_original_aspect", action="store_true", help="Preserve one reference image aspect ratio")
    parser.add_argument("--layout_bboxes", type=str, default=None, help="Layout boxes as a JSON string or JSON file path")
    parser.add_argument("--flash_attn", action="store_true", help="Use HiDream-O1 two-pass flash attention")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of Qwen3VL decoder blocks to swap to CPU")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="Use pinned memory for block swapping. Faster transfer, higher shared GPU memory use on Windows.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device. Defaults to cuda if available, otherwise cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Model dtype")
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s)")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s)")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA include regex per weight")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA exclude regex per weight")
    return parser.parse_args()


def str_to_dtype(dtype: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]


def merge_lora_weights(model: torch.nn.Module, args: argparse.Namespace, device: torch.device):
    if not args.lora_weight:
        return

    for i, lora_weight in enumerate(args.lora_weight):
        multiplier = 1.0
        if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
            multiplier = args.lora_multiplier[i]

        logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {multiplier}")
        weights_sd = load_file(lora_weight)

        if args.include_patterns is not None and len(args.include_patterns) > i:
            include_re = re.compile(args.include_patterns[i])
            weights_sd = {k: v for k, v in weights_sd.items() if include_re.search(k)}
        if args.exclude_patterns is not None and len(args.exclude_patterns) > i:
            exclude_re = re.compile(args.exclude_patterns[i])
            weights_sd = {k: v for k, v in weights_sd.items() if not exclude_re.search(k)}

        network = lora_hidream_o1.create_arch_network_from_weights(multiplier, weights_sd, unet=model, for_inference=True)
        network.merge_to(None, model, weights_sd, device=device, non_blocking=True)
        logger.info("LoRA weights loaded")


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = str_to_dtype(args.dtype)

    logger.info(f"Loading HiDream-O1 processor for model_type={args.model_type}")
    processor = hidream_o1_utils.load_processor(model_type=args.model_type)
    loading_device = "cpu" if args.blocks_to_swap else device
    logger.info(f"Loading HiDream-O1 model from {args.dit}")
    model = hidream_o1_utils.load_model(args.dit, dtype=dtype, device=loading_device, model_type=args.model_type)
    if args.blocks_to_swap:
        model.enable_block_swap(
            args.blocks_to_swap,
            device,
            supports_backward=False,
            use_pinned_memory=args.use_pinned_memory_for_block_swap,
        )
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()

    merge_lora_weights(model, args, device)

    num_inference_steps, guidance_scale, shift, timesteps_list, scheduler_name = hidream_o1_utils.select_inference_schedule(
        args.model_type,
        len(args.ref_images),
        args.infer_steps,
        args.guidance_scale,
        args.flow_shift,
        args.editing_scheduler,
    )
    if scheduler_name == "flash":
        extra_kwargs = {
            "noise_scale_start": args.noise_scale_start,
            "noise_scale_end": args.noise_scale_end,
            "noise_clip_std": args.noise_clip_std,
        }
    else:
        extra_kwargs = {}

    height, width = args.image_size
    image = generate_image(
        model=model,
        processor=processor,
        prompt=args.prompt,
        ref_image_paths=args.ref_images,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        shift=shift,
        timesteps_list=timesteps_list,
        scheduler_name=scheduler_name,
        seed=args.seed,
        keep_original_aspect=args.keep_original_aspect,
        layout_bboxes=args.layout_bboxes,
        use_flash_attn=args.flash_attn,
        **extra_kwargs,
    )

    save_dir = os.path.dirname(os.path.abspath(args.save_path))
    os.makedirs(save_dir, exist_ok=True)
    image.save(args.save_path)
    logger.info(f"Saved image to {args.save_path}")


if __name__ == "__main__":
    main()
