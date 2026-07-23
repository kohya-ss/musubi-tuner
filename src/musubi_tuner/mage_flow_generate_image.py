from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch

from musubi_tuner.mage_flow.mage_vae import load_mage_vae
from musubi_tuner.mage_flow.model import load_mage_flow_transformer
from musubi_tuner.mage_flow.sampling import resolve_output_size, sample_latents
from musubi_tuner.mage_flow.text_encoder import (
    QWEN3_VL_4B_INSTRUCT_REPO_ID,
    encode_conditioning,
    load_mage_flow_text_encoder,
)
from musubi_tuner.networks import lora_mage_flow
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.utils.model_utils import str_to_dtype


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _align_down(value: int) -> int:
    if value <= 0:
        raise ValueError("image dimensions must be positive")
    return max(16, value // 16 * 16)


def image_to_tensor(
    image: Image.Image,
    *,
    width: int,
    height: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    resized = image.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32).copy()
    return torch.from_numpy(array).permute(2, 0, 1).div(127.5).sub(1.0).to(device=device, dtype=dtype)


def latent_to_pil(vae, latent: torch.Tensor) -> Image.Image:
    with torch.no_grad():
        pixels = vae.decode(latent.unsqueeze(0).to(device=vae.device, dtype=vae.dtype))
    pixels = pixels[0].float().clamp(-1, 1).add(1).mul(127.5).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(pixels, mode="RGB")


def _free_model(model, device: torch.device) -> None:
    model.to("cpu")
    del model
    gc.collect()
    clean_memory_on_device(device)


def _apply_loras(
    transformer,
    paths: list[str] | None,
    multipliers: list[float] | None,
    device: torch.device,
    *,
    expected_architecture: str,
    allow_architecture_mismatch: bool,
) -> None:
    if not paths:
        return
    if multipliers is not None and len(multipliers) > len(paths):
        raise ValueError("--lora_multiplier cannot contain more values than --lora_weight")
    for index, path in enumerate(paths):
        lora_mage_flow.validate_adapter_architecture(
            path,
            expected=expected_architecture,
            allow_mismatch=allow_architecture_mismatch,
        )
        multiplier = multipliers[index] if multipliers is not None and index < len(multipliers) else 1.0
        logger.info("Loading Mage-Flow LoRA %s with multiplier %s", path, multiplier)
        weights = load_file(path, device="cpu")
        network = lora_mage_flow.create_arch_network_from_weights(
            multiplier,
            weights,
            unet=transformer,
            for_inference=True,
        )
        network.merge_to(None, transformer, weights, device=device, non_blocking=True)


def _validate_mode(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[Image.Image]:
    control_paths = list(args.control_image or [])
    if args.is_edit:
        if not 1 <= len(control_paths) <= 3:
            parser.error("--is_edit requires one to three ordered --control_image arguments")
    elif control_paths:
        parser.error("--control_image requires explicit --is_edit")
    if (args.width is None) != (args.height is None):
        parser.error("--width and --height must be provided together")
    if not args.is_edit and args.max_size is not None:
        parser.error("--max_size is only valid with --is_edit")
    return [Image.open(path).convert("RGB") for path in control_paths]


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an image with Mage-Flow or Mage-Flow-Edit")
    parser.add_argument("--dit", required=True, help="single Mage-Flow DiT safetensors file")
    parser.add_argument("--vae", required=True, help="single MageVAE safetensors file")
    parser.add_argument("--text_encoder", required=True, help="single Qwen3-VL-4B safetensors file")
    parser.add_argument(
        "--processor",
        default=QWEN3_VL_4B_INSTRUCT_REPO_ID,
        help="Qwen3-VL processor directory or pinned Hugging Face repository",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", default=" ")
    parser.add_argument("--output", default="mage_flow.png")
    parser.add_argument("--is_edit", action="store_true")
    parser.add_argument("--control_image", action="append", default=[], help="ordered Edit reference; repeat one to three times")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--flow_shift", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn_mode", choices=["sdpa", "flash2"], default="sdpa")
    parser.add_argument("--renormalize_cfg", action="store_true")
    parser.add_argument("--lora_weight", nargs="*", default=None)
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=None)
    parser.add_argument("--allow_mage_architecture_mismatch", action="store_true")
    return parser


def generate(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> Image.Image:
    parser = setup_parser() if parser is None else parser
    references = _validate_mode(args, parser)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = str_to_dtype(args.dtype)

    if args.is_edit:
        width, height = resolve_output_size(
            references[0].size,
            width=args.width,
            height=args.height,
            max_size=args.max_size,
        )
    else:
        width = _align_down(args.width if args.width is not None else 1024)
        height = _align_down(args.height if args.height is not None else 1024)

    logger.info("Encoding Mage-Flow conditioning")
    text_encoder = load_mage_flow_text_encoder(
        args.text_encoder,
        device=device,
        dtype=dtype,
        processor_source=args.processor,
    )
    prompts = [args.prompt]
    conditioning_references = [references] if args.is_edit else None
    positive = encode_conditioning(
        text_encoder,
        prompts,
        references=conditioning_references,
        is_edit=args.is_edit,
    )[0].cpu()
    negative = None
    if args.cfg_scale > 1.0:
        negative = encode_conditioning(
            text_encoder,
            [args.negative_prompt],
            references=conditioning_references,
            is_edit=args.is_edit,
        )[0].cpu()
    _free_model(text_encoder, device)
    del text_encoder

    control_latents = None
    if args.is_edit:
        logger.info("Encoding %d ordered Edit reference(s)", len(references))
        encoder_vae = load_mage_vae(args.vae, device=device, dtype=dtype, require_decoder=False)
        reference_pixels = torch.stack(
            [image_to_tensor(image, width=width, height=height, device=device, dtype=dtype) for image in references]
        )
        generators = [torch.Generator(device=device).manual_seed(args.seed + 10_000 + index) for index in range(len(references))]
        control_latents = [[latent.cpu() for latent in encoder_vae.encode(reference_pixels, generators=generators)]]
        _free_model(encoder_vae, device)
        del encoder_vae

    logger.info("Loading Mage-Flow transformer")
    transformer = load_mage_flow_transformer(
        args.dit,
        device=device,
        dtype=dtype,
        attention_backend=args.attn_mode,
    )
    _apply_loras(
        transformer,
        args.lora_weight,
        args.lora_multiplier,
        device,
        expected_architecture="mage_flow_edit" if args.is_edit else "mage_flow",
        allow_architecture_mismatch=args.allow_mage_architecture_mismatch,
    )
    positive = positive.to(device=device, dtype=dtype)
    negative_tokens = [negative.to(device=device, dtype=dtype)] if negative is not None else None
    if control_latents is not None:
        control_latents = [[latent.to(device=device, dtype=dtype) for latent in control_latents[0]]]

    latents = sample_latents(
        transformer,
        [positive],
        [(height // 16, width // 16)],
        steps=args.steps,
        seeds=[args.seed],
        channels=128,
        device=device,
        dtype=dtype,
        controls=control_latents,
        negative_text_tokens=negative_tokens,
        cfg_scale=args.cfg_scale,
        shift=args.flow_shift,
        renormalize_cfg=args.renormalize_cfg,
    )
    latent = latents[0].cpu()
    _free_model(transformer, device)
    del transformer

    logger.info("Decoding MageVAE latent")
    vae = load_mage_vae(args.vae, device=device, dtype=dtype, require_decoder=True)
    image = latent_to_pil(vae, latent)
    _free_model(vae, device)
    del vae
    return image


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    image = generate(args, parser)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    logger.info("Saved %s", output)


if __name__ == "__main__":
    main()
