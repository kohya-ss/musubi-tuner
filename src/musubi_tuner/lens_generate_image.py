import argparse
import logging
import os
from typing import Any

import torch
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image, PngImagePlugin

from musubi_tuner.lens import lens_text_encoder, lens_utils
from musubi_tuner.lens.resolution import SUPPORTED_ASPECT_RATIOS, SUPPORTED_BASE_RESOLUTIONS, resolve_resolution
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lens text-to-image inference")
    parser.add_argument("--dit", type=str, required=True, help="Lens DiT safetensors path")
    parser.add_argument("--vae", type=str, required=True, help="Lens/FLUX.2 VAE safetensors path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Lens Comfy GPT-OSS text encoder safetensors path")
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, help="image size as height width")
    parser.add_argument("--base_resolution", type=int, default=1024, choices=SUPPORTED_BASE_RESOLUTIONS)
    parser.add_argument("--aspect_ratio", type=str, default="1:1", choices=SUPPORTED_ASPECT_RATIOS)
    parser.add_argument("--infer_steps", type=int, default=20, help="number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="classifier-free guidance scale")
    parser.add_argument("--seed", type=int, default=None, help="seed")
    parser.add_argument("--save_path", type=str, required=True, help="path to output PNG")
    parser.add_argument("--device", type=str, default=None, help="device to use")
    parser.add_argument("--dit_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vae_dtype", type=str, default="float32", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--text_encoder_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors")
    parser.add_argument("--no_metadata", action="store_true", help="do not save PNG metadata")
    return parser.parse_args()


def _to_pil(image: torch.Tensor) -> list[Image.Image]:
    image = image.clamp(-1.0, 1.0)
    image = (image + 1.0) * (255.0 / 2.0)
    image = image.permute(0, 2, 3, 1).to(device="cpu", dtype=torch.uint8).numpy()
    return [Image.fromarray(im) for im in image]


def _encode_prompt(text_embedder, prompt: str, negative_prompt: str, device: torch.device, dtype: torch.dtype):
    prompt_features, prompt_mask = text_embedder([prompt])
    prompt_features = [feat.to(device=device, dtype=dtype) for feat in prompt_features]
    prompt_mask = prompt_mask.to(device=device, dtype=torch.bool)

    if negative_prompt.strip():
        negative_features, negative_mask = text_embedder([negative_prompt])
        negative_features = [feat.to(device=device, dtype=dtype) for feat in negative_features]
        negative_mask = negative_mask.to(device=device, dtype=torch.bool)
    else:
        negative_features = [feat.new_zeros(feat.shape) for feat in prompt_features]
        negative_mask = torch.zeros_like(prompt_mask, dtype=torch.bool)

    return lens_utils.align_text_feature_lists(prompt_features, prompt_mask, negative_features, negative_mask)


@torch.no_grad()
def generate(args: argparse.Namespace) -> list[Image.Image]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dit_dtype = model_utils.str_to_dtype(args.dit_dtype)
    vae_dtype = model_utils.str_to_dtype(args.vae_dtype)
    te_dtype = model_utils.str_to_dtype(args.text_encoder_dtype)

    if args.image_size is None:
        height, width = resolve_resolution(args.base_resolution, args.aspect_ratio)
    else:
        height, width = args.image_size
        if height % 16 or width % 16:
            raise ValueError("height and width must be divisible by 16")

    text_embedder = lens_text_encoder.load_lens_text_embedder(
        args.text_encoder,
        dtype=te_dtype,
        device=device,
        disable_mmap=args.disable_numpy_memmap,
    )
    transformer = lens_utils.load_lens_transformer(args.dit, dtype=dit_dtype, device=device, disable_mmap=args.disable_numpy_memmap)
    transformer.eval()
    vae = lens_utils.load_lens_vae(args.vae, dtype=vae_dtype, device=device)
    vae.eval()

    prompt_features, prompt_mask, negative_features, negative_mask = _encode_prompt(
        text_embedder, args.prompt, args.negative_prompt, device, dit_dtype
    )
    del text_embedder

    encoder_features = [torch.cat([pf, nf], dim=0) for pf, nf in zip(prompt_features, negative_features)]
    encoder_mask = torch.cat([prompt_mask, negative_mask], dim=0)

    latent_h, latent_w = height // 16, width // 16
    generator = torch.Generator(device=device).manual_seed(int(args.seed)) if args.seed is not None else None
    latents = randn_tensor((1, latent_h * latent_w, 128), generator=generator, device=device, dtype=dit_dtype)
    sigmas = lens_utils.get_lens_sigmas(args.infer_steps, latent_h * latent_w, device)
    img_shapes = [(1, latent_h, latent_w)]

    for i in range(args.infer_steps):
        timestep = (sigmas[i] * 1000.0).expand(2).to(dtype=dit_dtype)
        hidden_states = latents.repeat(2, 1, 1)
        noise = transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_features,
            encoder_hidden_states_mask=encoder_mask,
            timestep=timestep / 1000.0,
            img_shapes=img_shapes,
        )
        cond, uncond = noise.chunk(2)
        combined = uncond + args.guidance_scale * (cond - uncond)
        cond_norm = torch.norm(cond, dim=-1, keepdim=True)
        combined_norm = torch.norm(combined, dim=-1, keepdim=True)
        scale = torch.where(combined_norm > 0, cond_norm / combined_norm.clamp_min(1e-12), torch.ones_like(combined_norm))
        latents = lens_utils.euler_step(combined * scale, latents, sigmas, i)

    latents = lens_utils.unpack_latents(latents, latent_h, latent_w).to(vae.dtype)
    decoded = vae.decode(latents)
    return _to_pil(decoded)


def save_image(image: Image.Image, path: str, metadata: dict[str, Any] | None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pnginfo = None
    if metadata:
        pnginfo = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            pnginfo.add_text(key, str(value))
    image.save(path, pnginfo=pnginfo)


def main():
    args = parse_args()
    images = generate(args)
    metadata = None
    if not args.no_metadata:
        metadata = {
            "architecture": "lens",
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "steps": args.infer_steps,
            "guidance_scale": args.guidance_scale,
        }
    save_image(images[0], args.save_path, metadata)
    logger.info(f"Saved image to {args.save_path}")


if __name__ == "__main__":
    main()
