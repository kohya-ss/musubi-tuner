import argparse
import os
from pathlib import Path

import torch
import toml
from safetensors.torch import save_file
from tqdm import tqdm


from musubi_tuner.ideogram4.text_encoder_utils import (
    load_qwen3_vl_text_encoder,
    encode_caption_to_features,
)
from musubi_tuner.dataset.architectures import ARCHITECTURE_IDEOGRAM4, ARCHITECTURE_IDEOGRAM4_FULL
from musubi_tuner.dataset.cache_io import dtype_to_str


def parse_args():
    parser = argparse.ArgumentParser(description="Cache Ideogram4 Qwen3-VL text encoder outputs")

    parser.add_argument("--text_encoder", type=str, required=True, help="Path to Qwen3-VL-8B-Instruct folder")
    parser.add_argument("--dataset_config", type=str, default=None, help="Musubi dataset config TOML")

    # Backward-compatible standalone mode.
    parser.add_argument("--caption_dir", type=str, default=None, help="Folder containing .txt captions")
    parser.add_argument("--output_dir", type=str, default=None, help="Folder to save text cache .safetensors files")
    parser.add_argument("--caption_extension", type=str, default=None)

    parser.add_argument("--max_text_length", type=int, default=3072)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--recursive", action="store_true", help="Search captions recursively")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache files")

    return parser.parse_args()


def dtype_from_string(s):
    if s == "bf16":
        return torch.bfloat16
    if s == "fp16":
        return torch.float16
    return torch.float32


def normalize_caption_extension(ext):
    ext = ext or ".txt"
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def list_caption_files(caption_dir: Path, caption_extension: str, recursive: bool):
    caption_extension = normalize_caption_extension(caption_extension)
    pattern = "**/*" + caption_extension if recursive else "*" + caption_extension
    return sorted([p for p in caption_dir.glob(pattern) if p.is_file()])


def get_jobs_from_dataset_config(path: str):
    cfg = toml.load(path)
    general = cfg.get("general", {})
    datasets = cfg.get("datasets", [])

    if not datasets:
        raise ValueError(f"No [[datasets]] entries found in {path}")

    jobs = []
    for i, ds in enumerate(datasets):
        # Musubi normally keeps captions beside images, so default caption_dir=image_directory.
        caption_dir = ds.get("caption_directory", ds.get("image_directory"))
        output_dir = ds.get("cache_directory")
        caption_extension = ds.get("caption_extension", general.get("caption_extension", ".txt"))

        if caption_dir is None:
            raise ValueError(f"Dataset {i} missing image_directory/caption_directory")
        if output_dir is None:
            raise ValueError(f"Dataset {i} missing cache_directory")

        jobs.append({
            "caption_dir": Path(caption_dir),
            "output_dir": Path(output_dir),
            "caption_extension": normalize_caption_extension(caption_extension),
        })

    return jobs


def get_jobs(args):
    if args.dataset_config is not None:
        return get_jobs_from_dataset_config(args.dataset_config)

    if args.caption_dir is None or args.output_dir is None:
        raise ValueError("Either --dataset_config or both --caption_dir and --output_dir must be provided")

    return [{
        "caption_dir": Path(args.caption_dir),
        "output_dir": Path(args.output_dir),
        "caption_extension": normalize_caption_extension(args.caption_extension),
    }]


def cache_job(job, tokenizer, text_encoder, dtype, max_text_length, recursive, overwrite):
    caption_dir = job["caption_dir"]
    output_dir = job["output_dir"]
    caption_extension = job["caption_extension"]

    output_dir.mkdir(parents=True, exist_ok=True)

    caption_files = list_caption_files(caption_dir, caption_extension, recursive)

    print(f"Found {len(caption_files)} caption files in {caption_dir}")
    print(f"Caption extension: {caption_extension}")
    print(f"Output cache: {output_dir}")

    if len(caption_files) == 0:
        raise RuntimeError(f"No caption files found in {caption_dir}")

    for caption_path in tqdm(caption_files, desc=f"Caching text features: {caption_dir}"):
        rel = caption_path.relative_to(caption_dir)
        item_key = rel.with_suffix("").as_posix().replace("/", "_")
        out_name = f"{item_key}_{ARCHITECTURE_IDEOGRAM4}_te.safetensors"
        out_path = output_dir / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            continue

        caption = caption_path.read_text(encoding="utf-8").strip()

        if len(caption) == 0:
            print(f"Warning: empty caption, using blank JSON placeholder: {caption_path}")
            caption = '{"high_level_description":"","key_action":"","subject_count":0}'

        with torch.no_grad():
            features = encode_caption_to_features(
                tokenizer,
                text_encoder,
                caption,
                max_text_length=max_text_length,
                dtype=dtype,
            )

        metadata = {
            "source_caption": str(caption_path),
            "num_tokens": str(features.shape[0]),
            "feature_dim": str(features.shape[1]),
            "format": "ideogram4_qwen3_vl_text_features",
            "architecture": ARCHITECTURE_IDEOGRAM4_FULL,
            "architecture_short": ARCHITECTURE_IDEOGRAM4,
        }

        dtype_str = dtype_to_str(features.dtype)

        save_file(
            {
                f"varlen_vl_embed_{dtype_str}": features.detach().cpu().contiguous(),
            },
            str(out_path),
            metadata=metadata,
        )


def main():
    args = parse_args()
    dtype = dtype_from_string(args.dtype)
    jobs = get_jobs(args)

    print("Loading Qwen3-VL text encoder...")
    tokenizer, text_encoder = load_qwen3_vl_text_encoder(
        args.text_encoder,
        dtype=dtype,
        device_map="auto",
    )

    for job in jobs:
        cache_job(
            job,
            tokenizer,
            text_encoder,
            dtype,
            args.max_text_length,
            args.recursive,
            args.overwrite,
        )

    print("Ideogram4 text encoder output caching finished.")


if __name__ == "__main__":
    main()
