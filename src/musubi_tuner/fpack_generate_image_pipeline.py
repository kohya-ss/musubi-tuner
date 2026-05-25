# CLI script for FramePack one-frame image generation using the Pipeline class.
#
# Usage examples:
#   # Single prompt
#   python src/musubi_tuner/fpack_generate_image_pipeline.py \
#     --dit path/to/dit --vae path/to/vae \
#     --text_encoder1 path/to/te1 --text_encoder2 path/to/te2 \
#     --image_encoder path/to/ie \
#     --prompt "a cat sitting on a couch" --save_path ./output
#
#   # Batch from file
#   python src/musubi_tuner/fpack_generate_image_pipeline.py \
#     --dit path/to/dit --vae path/to/vae \
#     --text_encoder1 path/to/te1 --text_encoder2 path/to/te2 \
#     --image_encoder path/to/ie \
#     --from_file prompts.txt --save_path ./output --batch_size 4
#
# Prompt file format (one per line, options after --):
#   a beautiful sunset --w 512 --h 384 --s 30 --d 42
#   a cat --ci control.png --cim mask.png

import argparse
import os
import logging
import time
from typing import List, Optional

from PIL import Image

from musubi_tuner.frame_pack.framepack_image_pipeline import (
    FramePackImagePipeline,
    ImageInput,
    OneFrameSettings,
    PipelineOutput,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FramePack image generation via Pipeline")

    # Model paths
    parser.add_argument("--dit", type=str, required=True, help="DiT model directory or path")
    parser.add_argument("--vae", type=str, required=True, help="VAE directory or path")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 (LLM) directory or path")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 (CLIP) directory or path")
    parser.add_argument("--image_encoder", type=str, required=True, help="Image Encoder (CLIP Vision) directory or path")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s)")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s)")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")

    # Precision / optimization
    parser.add_argument("--fp8", action="store_true", help="Use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use scaled fp8 for DiT")
    parser.add_argument("--fp8_fast_quantization_mode", type=str, default=None, choices=["tensor", "block", "channel"])
    parser.add_argument("--fp8_block_size", type=int, default=64)
    parser.add_argument("--fp8_llm", action="store_true", help="Use fp8 for Text Encoder 1")
    parser.add_argument("--nvfp4", action="store_true", help="Use NVFP4 for DiT model")
    parser.add_argument("--nvfp4_compile", action="store_true", help="Enable torch.compile for NVFP4")
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="Number of DiT blocks to swap to CPU")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true")

    # Attention
    parser.add_argument("--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "xformers", "sdpa"])

    # VAE
    parser.add_argument("--vae_tiling", action="store_true")
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_spatial_tile_sample_min_size", type=int, default=None)

    # RoPE
    parser.add_argument("--rope_scaling_factor", type=float, default=0.5)
    parser.add_argument("--rope_scaling_timestep_threshold", type=int, default=None)

    # Generation defaults (can be overridden per-prompt in batch file)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for single generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size: height width")
    parser.add_argument("--infer_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument(
        "--strength",
        type=float,
        default=None,
        help="Strength for image-to-image generation (0.0-1.0), uses `image_path` as an initial image",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--embedded_cfg_scale", type=float, default=10.0, help="Distilled CFG scale")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="CFG scale (1.0 = no CFG)")
    parser.add_argument("--guidance_rescale", type=float, default=0.0)
    parser.add_argument("--flow_shift", type=float, default=None)
    parser.add_argument("--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"])
    parser.add_argument("--latent_window_size", type=int, default=9)
    parser.add_argument("--image_path", type=str, default=None, help="Image for CLIP embedding (I2V)")
    parser.add_argument("--control_image_path", type=str, nargs="*", default=None, help="Control image path(s)")
    parser.add_argument("--control_image_mask_path", type=str, nargs="*", default=None, help="Control image mask path(s)")
    parser.add_argument(
        "--one_frame_inference",
        type=str,
        default=None,
        help="One-frame mode options, comma separated: no_2x, no_4x, no_post, target_index=N, control_index=N;N",
    )
    parser.add_argument("--custom_system_prompt", type=str, default=None)

    # Batch / output
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file (one per line)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for DiT generation")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save outputs")
    parser.add_argument(
        "--output_type",
        type=str,
        default="pil",
        choices=["pil", "latent", "both"],
        help="Output type: pil (images), latent (safetensors), both",
    )

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--disable_numpy_memmap", action="store_true")

    return parser.parse_args()


def parse_one_frame_string(s: Optional[str]) -> OneFrameSettings:
    """Parse a comma-separated one_frame_inference string into OneFrameSettings."""
    if s is None:
        return OneFrameSettings()

    settings = OneFrameSettings()
    for token in s.split(","):
        token = token.strip()
        if token == "no_2x":
            settings.no_2x = True
        elif token == "no_4x":
            settings.no_4x = True
        elif token == "no_post":
            settings.no_post = True
        elif token.startswith("target_index="):
            settings.target_index = int(token.split("=")[1])
        elif token.startswith("control_index="):
            indices_str = token.split("=")[1]
            settings.control_indices = [int(x) for x in indices_str.split(";")]
        elif token == "default":
            pass
    return settings


def load_images_if_paths(
    image_path: Optional[str],
    control_paths: Optional[List[str]],
    mask_paths: Optional[List[str]],
) -> tuple:
    """Load PIL images from file paths."""
    image = None
    if image_path is not None:
        image = Image.open(image_path)

    control_images = None
    if control_paths:
        control_images = [Image.open(p) for p in control_paths]

    control_masks = None
    if mask_paths:
        control_masks = [Image.open(p) for p in mask_paths]

    return image, control_images, control_masks


def parse_prompt_line(line: str, defaults: argparse.Namespace) -> ImageInput:
    """Parse a prompt line with inline overrides into an ImageInput.

    Format: prompt text --option value --option value ...
    Supported short options: --w, --h, --s (steps), --d (seed), --g (guidance),
    --fs (flow_shift), --i (image), --ci (control image), --cim (control mask),
    --n (negative prompt), --of (one_frame_inference)
    """
    parts = line.split(" --")
    prompt = parts[0].strip()

    # Start with defaults
    height, width = defaults.image_size
    seed = defaults.seed
    infer_steps = defaults.infer_steps
    strength = defaults.strength
    guidance_scale = defaults.guidance_scale
    embedded_cfg_scale = defaults.embedded_cfg_scale
    guidance_rescale = defaults.guidance_rescale
    flow_shift = defaults.flow_shift
    negative_prompt = defaults.negative_prompt
    image_path = defaults.image_path
    control_paths = list(defaults.control_image_path) if defaults.control_image_path else []
    mask_paths = list(defaults.control_image_mask_path) if defaults.control_image_mask_path else []
    one_frame_str = defaults.one_frame_inference
    custom_system_prompt = defaults.custom_system_prompt

    for part in parts[1:]:
        if not part.strip():
            continue
        tokens = part.split(" ", 1)
        opt = tokens[0].strip()
        val = tokens[1].strip() if len(tokens) > 1 else ""

        if opt == "w":
            width = int(val)
        elif opt == "h":
            height = int(val)
        elif opt == "s":
            infer_steps = int(val)
        elif opt == "t":
            strength = float(val)
        elif opt == "d":
            seed = int(val)
        elif opt in ("g", "l"):
            guidance_scale = float(val)
        elif opt == "cfg":
            embedded_cfg_scale = float(val)
        elif opt == "fs":
            flow_shift = float(val)
        elif opt == "n":
            negative_prompt = val
        elif opt == "i":
            image_path = val
        elif opt == "ci":
            control_paths.append(val)
        elif opt == "cim":
            mask_paths.append(val)
        elif opt == "of":
            one_frame_str = val

    image, control_images, control_masks = load_images_if_paths(image_path, control_paths or None, mask_paths or None)

    return ImageInput(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        control_images=control_images,
        control_masks=control_masks,
        image_size=(height, width),
        seed=seed,
        infer_steps=infer_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        embedded_cfg_scale=embedded_cfg_scale,
        guidance_rescale=guidance_rescale,
        flow_shift=flow_shift,
        one_frame_settings=parse_one_frame_string(one_frame_str),
        custom_system_prompt=custom_system_prompt,
    )


def save_results(results: List[PipelineOutput], save_path: str, output_type: str) -> None:
    """Save pipeline results to disk."""
    os.makedirs(save_path, exist_ok=True)
    time_flag = time.strftime("%Y%m%d_%H%M%S")

    for i, result in enumerate(results):
        base_name = f"{time_flag}_{result.seed}_{i:04d}"

        if result.image is not None and output_type in ("pil", "both"):
            img_path = os.path.join(save_path, f"{base_name}.png")
            result.image.save(img_path)
            logger.info(f"Saved image: {img_path}")

        if result.latent is not None and output_type in ("latent", "both"):
            from safetensors.torch import save_file

            latent_path = os.path.join(save_path, f"{base_name}_latent.safetensors")
            save_file({"latent": result.latent.contiguous()}, latent_path, metadata={"seed": str(result.seed)})
            logger.info(f"Saved latent: {latent_path}")


def main():
    args = parse_args()

    # --- Build pipeline ---
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading pipeline models...")
    pipeline = FramePackImagePipeline.from_pretrained(
        dit=args.dit,
        vae=args.vae,
        text_encoder1=args.text_encoder1,
        text_encoder2=args.text_encoder2,
        image_encoder=args.image_encoder,
        device=device,
        attn_mode=args.attn_mode,
        fp8=args.fp8,
        fp8_scaled=args.fp8_scaled,
        fp8_fast_quantization_mode=args.fp8_fast_quantization_mode,
        fp8_block_size=args.fp8_block_size,
        fp8_llm=args.fp8_llm,
        nvfp4=args.nvfp4,
        nvfp4_compile=args.nvfp4_compile,
        blocks_to_swap=args.blocks_to_swap,
        use_pinned_memory_for_block_swap=args.use_pinned_memory_for_block_swap,
        lora_weights=args.lora_weight,
        lora_multipliers=args.lora_multiplier,
        include_patterns=args.include_patterns,
        exclude_patterns=args.exclude_patterns,
        latent_window_size=args.latent_window_size,
        sample_solver=args.sample_solver,
        rope_scaling_factor=args.rope_scaling_factor,
        rope_scaling_timestep_threshold=args.rope_scaling_timestep_threshold,
        disable_numpy_memmap=args.disable_numpy_memmap,
        vae_chunk_size=args.vae_chunk_size,
        vae_spatial_tile_sample_min_size=args.vae_spatial_tile_sample_min_size,
        vae_tiling=args.vae_tiling,
    )

    # --- Build inputs ---
    if args.from_file:
        logger.info(f"Reading prompts from: {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        inputs = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            inputs.append(parse_prompt_line(line, args))

        logger.info(f"Parsed {len(inputs)} prompts from file")
    elif args.prompt is not None:
        image, control_images, control_masks = load_images_if_paths(
            args.image_path, args.control_image_path, args.control_image_mask_path
        )
        inputs = [
            ImageInput(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=image,
                control_images=control_images,
                control_masks=control_masks,
                image_size=tuple(args.image_size),
                seed=args.seed,
                infer_steps=args.infer_steps,
                guidance_scale=args.guidance_scale,
                embedded_cfg_scale=args.embedded_cfg_scale,
                guidance_rescale=args.guidance_rescale,
                flow_shift=args.flow_shift,
                one_frame_settings=parse_one_frame_string(args.one_frame_inference),
                custom_system_prompt=args.custom_system_prompt,
            )
        ]
    else:
        raise ValueError("Either --prompt or --from_file must be specified")

    # --- Generate ---
    logger.info(f"Generating {len(inputs)} image(s) with batch_size={args.batch_size}...")
    results = pipeline(
        inputs=inputs,
        batch_size=args.batch_size,
        output_type=args.output_type,
    )

    # --- Save ---
    save_results(results, args.save_path, args.output_type)
    logger.info("Done!")


if __name__ == "__main__":
    main()
