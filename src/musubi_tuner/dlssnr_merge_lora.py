"""Merge a DLSS-NR LoRA into a copy of a canonical checkpoint."""

from __future__ import annotations

import argparse

from musubi_tuner.networks.lora_dlssnr import merge_to_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter onto a canonical DLSS-NR base.")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_multiplier", type=float, default=1.0, help="Scale only the LoRA delta (default 1)")
    parser.add_argument("--save_precision", choices=("native", "fp32"), default="native")
    args = parser.parse_args()
    merge_to_directory(
        args.base_model_dir, args.adapter, args.output_dir, multiplier=args.lora_multiplier, save_precision=args.save_precision
    )
    print(f"wrote merged checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
