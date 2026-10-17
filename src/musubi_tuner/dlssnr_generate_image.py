"""Render one still frame per manifest row with the FP32 surrogate."""

from __future__ import annotations

import argparse

from musubi_tuner.dlssnr.infer import add_runtime_arguments, generate_stills, load_model, runtime_overrides_from_args


def main() -> None:
    parser = argparse.ArgumentParser(description="DLSS-NR still inference. Target images are not required.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sample_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    add_runtime_arguments(parser)
    parser.add_argument("--bucket_width", type=int, required=True)
    parser.add_argument("--bucket_height", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    written = generate_stills(
        load_model(args.model_dir, args.device, runtime_overrides=runtime_overrides_from_args(args)),
        args.sample_manifest,
        args.bucket_width,
        args.bucket_height,
        args.output_dir,
        args.seed,
    )
    print(f"wrote {len(written)} stills to {args.output_dir}")


if __name__ == "__main__":
    main()
