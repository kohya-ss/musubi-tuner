"""Merge a DLSS-NR LoRA into a copy of a canonical checkpoint."""

from __future__ import annotations

import argparse

from musubi_tuner.networks.lora_dlssnr import merge_to_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter onto a canonical DLSS-NR base.")
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    merge_to_directory(args.base_model_dir, args.adapter, args.output_dir)
    print(f"wrote merged checkpoint to {args.output_dir}")


if __name__ == "__main__":
    main()
