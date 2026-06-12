import argparse
import os
from pathlib import Path

import torch
import toml
from PIL import Image
from safetensors.torch import save_file
from tqdm import tqdm


from musubi_tuner.ideogram4.ideogram4_utils import load_ideogram4_vae
from musubi_tuner.ideogram4.latent_utils import encode_images_to_ideogram4_latents
from musubi_tuner.dataset.architectures import ARCHITECTURE_IDEOGRAM4, ARCHITECTURE_IDEOGRAM4_FULL
from musubi_tuner.utils.model_utils import dtype_to_str


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Cache Ideogram4 VAE latents for image datasets")

    parser.add_argument("--vae", type=str, required=True, help="Path to Ideogram4 VAE folder, model folder, or VAE safetensors")
    parser.add_argument("--dataset_config", type=str, default=None, help="Musubi dataset config TOML")

    # Backward-compatible standalone mode.
    parser.add_argument("--image_dir", type=str, default=None, help="Folder containing images")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save latent .safetensors files")
    parser.add_argument("--resolution", type=str, default=None, help="Output image resolution as width,height")

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--recursive", action="store_true", help="Search images recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")

    return parser.parse_args()


def dtype_from_string(s):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32


def parse_resolution(resolution):
    if isinstance(resolution, (list, tuple)):
        if len(resolution) != 2:
            raise ValueError("resolution list must have two values: [width, height]")
        width, height = int(resolution[0]), int(resolution[1])
    else:
        if resolution is None:
            raise ValueError("resolution is required")
        parts = str(resolution).replace("x", ",").split(",")
        if len(parts) != 2:
            raise ValueError("resolution must look like 1024,1024 or [1024, 1024]")
        width = int(parts[0])
        height = int(parts[1])

    if width % 16 != 0 or height % 16 != 0:
        raise ValueError("Ideogram4 cache resolution must be divisible by 16")

    return width, height


def list_images(image_dir: Path, recursive: bool):
    pattern = "**/*" if recursive else "*"
    return sorted([p for p in image_dir.glob(pattern) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS])


def center_crop_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    w, h = img.size
    target_ratio = target_w / target_h
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_h = h
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_w = w
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img.resize((target_w, target_h), Image.Resampling.LANCZOS)


def image_to_tensor(img: Image.Image, dtype: torch.dtype):
    import numpy as np

    arr = np.array(img).astype("float32") / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor * 2.0 - 1.0
    return tensor.to(dtype=dtype)


def get_jobs_from_dataset_config(path: str):
    cfg = toml.load(path)
    general = cfg.get("general", {})
    datasets = cfg.get("datasets", [])

    if not datasets:
        raise ValueError(f"No [[datasets]] entries found in {path}")

    jobs = []
    for i, ds in enumerate(datasets):
        image_dir = ds.get("image_directory")
        output_dir = ds.get("cache_directory")
        resolution = ds.get("resolution", general.get("resolution"))

        if image_dir is None:
            raise ValueError(f"Dataset {i} missing image_directory")
        if output_dir is None:
            raise ValueError(f"Dataset {i} missing cache_directory")
        if resolution is None:
            raise ValueError(f"Dataset {i} missing resolution and [general].resolution")

        target_w, target_h = parse_resolution(resolution)

        jobs.append({
            "image_dir": Path(image_dir),
            "output_dir": Path(output_dir),
            "target_w": target_w,
            "target_h": target_h,
        })

    return jobs


def get_jobs(args):
    if args.dataset_config is not None:
        return get_jobs_from_dataset_config(args.dataset_config)

    if args.image_dir is None or args.output_dir is None or args.resolution is None:
        raise ValueError("Either --dataset_config or all of --image_dir, --output_dir, --resolution must be provided")

    target_w, target_h = parse_resolution(args.resolution)
    return [{
        "image_dir": Path(args.image_dir),
        "output_dir": Path(args.output_dir),
        "target_w": target_w,
        "target_h": target_h,
    }]


def cache_job(job, vae, dtype, recursive, overwrite):
    image_dir = job["image_dir"]
    output_dir = job["output_dir"]
    target_w = job["target_w"]
    target_h = job["target_h"]

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir, recursive)
    print(f"Found {len(images)} images in {image_dir}")
    print(f"Target resolution: {target_w}x{target_h}")
    print(f"Output cache: {output_dir}")

    if len(images) == 0:
        raise RuntimeError(f"No images found in {image_dir}")

    for image_path in tqdm(images, desc=f"Caching latents: {image_dir}"):
        rel = image_path.relative_to(image_dir)
        item_key = rel.with_suffix("").as_posix().replace("/", "_")
        out_name = f"{item_key}_{target_w:04d}x{target_h:04d}_{ARCHITECTURE_IDEOGRAM4}.safetensors"
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            continue

        img = Image.open(image_path).convert("RGB")
        img = center_crop_resize(img, target_w, target_h)
        image_tensor = image_to_tensor(img, dtype=dtype).to("cuda")

        with torch.no_grad():
            latents = encode_images_to_ideogram4_latents(
                vae,
                image_tensor,
                dtype=dtype,
                device="cuda",
            )

        latents_cpu = latents.detach().cpu().contiguous()

        # Ideogram latent from encoder is (1, 128, H, W).
        # Musubi-native cache tensor layout is (C, F, H, W), so store (128, 1, H, W).
        _, _, latent_h, latent_w = latents_cpu.shape
        latents_native = latents_cpu[0].unsqueeze(1).contiguous()

        dtype_str = dtype_to_str(latents_native.dtype)
        musubi_latent_key = f"latents_1x{latent_h}x{latent_w}_{dtype_str}"

        metadata = {
            "source_image": str(image_path),
            "resolution": f"{target_w}x{target_h}",
            "latent_shape": str(tuple(latents_native.shape)),
            "format": "ideogram4_patchified_normalized_latent",
            "architecture": ARCHITECTURE_IDEOGRAM4_FULL,
            "architecture_short": ARCHITECTURE_IDEOGRAM4,
        }

        save_file(
            {
                musubi_latent_key: latents_native,
            },
            str(out_path),
            metadata=metadata,
        )


def main():
    args = parse_args()
    dtype = dtype_from_string(args.dtype)
    jobs = get_jobs(args)

    print("Loading Ideogram4 VAE...")
    vae = load_ideogram4_vae(
        args.vae,
        dtype=dtype,
        device="cuda",
    )

    for job in jobs:
        cache_job(job, vae, dtype, args.recursive, args.overwrite)

    print("Ideogram4 latent caching finished.")


if __name__ == "__main__":
    main()
