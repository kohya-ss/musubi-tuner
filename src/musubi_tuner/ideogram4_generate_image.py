import argparse
import logging
import os

import torch
from safetensors.torch import load_file

from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.networks import lora_ideogram4
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dit", type=str, required=True, help="conditional Ideogram 4 DiT safetensors path")
    parser.add_argument(
        "--unconditional_dit",
        type=str,
        default=None,
        help="optional unconditional Ideogram 4 DiT safetensors path; omitted uses single-DiT unconditional embeds CFG",
    )
    parser.add_argument(
        "--lora_weight", type=str, nargs="*", default=None, help="LoRA weight safetensors path(s), applied to the conditional DiT"
    )
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier per --lora_weight (default 1.0 each)"
    )
    parser.add_argument(
        "--include_patterns", type=str, nargs="*", default=None, help="regex include pattern per --lora_weight (optional)"
    )
    parser.add_argument(
        "--exclude_patterns", type=str, nargs="*", default=None, help="regex exclude pattern per --lora_weight (optional)"
    )
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3-VL BF16 text encoder safetensors path")
    parser.add_argument("--vae", type=str, required=True, help="Flux2 VAE safetensors path")
    parser.add_argument("--prompt", type=str, required=True, help="prompt")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative/unconditional prompt used only when --unconditional_dit is omitted",
    )
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size as height width")
    parser.add_argument("--sampler_preset", type=str, default="V4_DEFAULT_20", choices=sorted(PRESETS.keys()))
    parser.add_argument("--initial_sigma", type=float, default=1.004, help="override the first denoising sigma")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_path", type=str, required=True, help="output image path or directory")
    parser.add_argument("--device", type=str, default=None, help="device, default cuda if available")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="compute dtype for models")
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["torch", "sdpa", "flash", "sageattn", "xformers"],
        help="attention mode for the DiT ('sdpa' is an alias for 'torch')",
    )
    parser.add_argument("--split_attn", action="store_true", help="process each sample's attention separately")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="disable numpy memmap while loading safetensors")
    parser.add_argument("--warn_on_caption_issues", action="store_true", help="warn instead of failing on caption verifier issues")
    return parser


def _apply_lora_weights(transformer, args, device):
    """Apply one or more LoRA weights to the conditional DiT as non-merged forward hooks.

    Ideogram 4's DiT is a frozen pre-quantized FP8 base that cannot have LoRA merged into it
    (re-quantizing the merged delta would weaken it), so we attach the LoRA via apply_to like
    the training-time sampler does. Multiple LoRAs stack (each wraps the previous forward).
    """
    multipliers = args.lora_multiplier or []
    includes = args.include_patterns or []
    excludes = args.exclude_patterns or []
    for i, lora_weight in enumerate(args.lora_weight):
        multiplier = multipliers[i] if i < len(multipliers) else 1.0
        include = includes[i] if i < len(includes) else None
        exclude = excludes[i] if i < len(excludes) else None
        logger.info(f"Loading LoRA from {lora_weight} (multiplier={multiplier})")
        weights_sd = load_file(lora_weight)
        weights_sd = filter_lora_state_dict(weights_sd, include, exclude)
        network = lora_ideogram4.create_arch_network_from_weights(multiplier, weights_sd, unet=transformer, for_inference=True)
        network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
        info = network.load_state_dict(weights_sd, strict=True)
        logger.info(f"Applied LoRA: {info}")
        network.eval()
        network.to(device)


def _save_images(images, save_path: str):
    root, ext = os.path.splitext(save_path)
    if len(images) == 1 and ext:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        images[0].save(save_path)
        logger.info(f"Saved image: {save_path}")
        return

    os.makedirs(save_path, exist_ok=True)
    for i, image in enumerate(images):
        path = os.path.join(save_path, f"ideogram4_{i:02d}.png")
        image.save(path)
        logger.info(f"Saved image: {path}")


def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.unconditional_dit and args.negative_prompt is not None:
        logger.warning("Ideogram 4 v1 uses official asymmetric CFG; --negative_prompt is ignored.")

    height, width = args.image_size
    device = torch.device(args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype)
    # "sdpa" is a user-facing alias for the shared attention()'s "torch" mode.
    attn_mode = "torch" if args.attn_mode == "sdpa" else args.attn_mode

    ideogram4_utils.validate_prompt(args.prompt, warn_only=args.warn_on_caption_issues)

    logger.info("Loading Ideogram 4 tokenizer and text encoder")
    tokenizer = ideogram4_utils.load_ideogram4_tokenizer()
    text_encoder = ideogram4_utils.load_ideogram4_text_encoder(
        args.text_encoder,
        device=device,
        dtype=dtype,
        disable_mmap=args.disable_numpy_memmap,
    )
    text_features = [ideogram4_utils.encode_prompt_to_features(tokenizer, text_encoder, args.prompt, device)]
    unconditional_text_features = None
    if not args.unconditional_dit:
        unconditional_text_features = [
            ideogram4_utils.encode_prompt_to_features(tokenizer, text_encoder, args.negative_prompt or "", device)
        ]
    del tokenizer, text_encoder
    clean_memory_on_device(device)

    logger.info("Loading conditional DiT")
    conditional_transformer = ideogram4_utils.load_ideogram4_transformer(
        args.dit,
        device=device,
        dtype=dtype,
        expected_model_type=ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE,
        disable_mmap=args.disable_numpy_memmap,
        attn_mode=attn_mode,
        split_attn=args.split_attn,
    )
    if args.lora_weight:
        _apply_lora_weights(conditional_transformer, args, device)
    unconditional_transformer = None
    if args.unconditional_dit:
        logger.info("Loading unconditional DiT")
        unconditional_transformer = ideogram4_utils.load_ideogram4_transformer(
            args.unconditional_dit,
            device=device,
            dtype=dtype,
            expected_model_type=ideogram4_utils.IDEOGRAM4_UNCOND_MODEL_TYPE,
            disable_mmap=args.disable_numpy_memmap,
            attn_mode=attn_mode,
            split_attn=args.split_attn,
        )
    else:
        logger.info("Using conditional DiT for unconditional embeds CFG")
    logger.info("Loading VAE")
    autoencoder = ideogram4_utils.load_ideogram4_autoencoder(
        args.vae,
        device=device,
        dtype=dtype,
        disable_mmap=args.disable_numpy_memmap,
    )

    images = ideogram4_utils.generate_images(
        conditional_transformer=conditional_transformer,
        unconditional_transformer=unconditional_transformer,
        autoencoder=autoencoder,
        text_features=text_features,
        unconditional_text_features=unconditional_text_features,
        height=height,
        width=width,
        sampler_preset=args.sampler_preset,
        device=device,
        seed=args.seed,
        initial_sigma=args.initial_sigma,
    )
    _save_images(images, args.save_path)


if __name__ == "__main__":
    main()
