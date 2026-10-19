"""Scan tone/structure responses with fixed input frames and paired noise seeds."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from musubi_tuner.dlssnr.control_scan import build_scan_config, scan_controls
from musubi_tuner.dlssnr.content_metrics import create_content_metric
from musubi_tuner.dlssnr.infer import add_runtime_arguments, load_model, runtime_overrides_from_args


def setup_parser():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model_dir", type=Path, required=True, help="Canonical model directory; merge LoRA before scanning.")
    parser.add_argument(
        "--sample_manifest", type=Path, required=True, help="Single-frame JSONL; targets and controls are not required."
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="New directory for PNGs, scan_report.json and scan_metrics.csv."
    )
    parser.add_argument("--bucket_width", type=int, required=True, help="Input width; no implicit crop or resize.")
    parser.add_argument("--bucket_height", type=int, required=True, help="Input height; no implicit crop or resize.")
    parser.add_argument("--tone_values", nargs="+", type=float, default=[0, 0.5, 1], help="Tone grid in [0,1] (default: 0 0.5 1).")
    parser.add_argument("--structure_values", nargs="+", type=float, default=[0, 0.5, 1], help="Structure grid in [0,1].")
    parser.add_argument("--nr_style", type=int, default=0)
    parser.add_argument("--nr_skin", type=float, default=-1, help="Skin strength in [0,1], or -1 to follow structure.")
    parser.add_argument("--nr_auto_mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=0, help="Same sample-derived noise seed at every control point and runtime.")
    parser.add_argument(
        "--lowpass_sigma", type=float, default=6, help="Gaussian sigma for output-minus-input bands; 0 < sigma <= 32."
    )
    parser.add_argument("--eval_native", action="store_true", help="Also scan native-quantized weights with FP32 native attention.")
    parser.add_argument(
        "--eval_content_preservation", action="store_true", help="Add frozen DINOv3 spatial patch MSE against each input."
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    add_runtime_arguments(parser)
    return parser


def run(args):
    config = build_scan_config(
        args.bucket_width,
        args.bucket_height,
        tone_values=args.tone_values,
        structure_values=args.structure_values,
        style=args.nr_style,
        skin=args.nr_skin,
        auto_mask=args.nr_auto_mask,
        seed=args.seed,
        lowpass_sigma=args.lowpass_sigma,
        compare_native=args.eval_native,
        content_preservation=args.eval_content_preservation,
    )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("control scanning is a single-process tool; run without a distributed launcher")
    if args.output_dir.exists():
        raise FileExistsError(f"scan output directory already exists: {args.output_dir}")
    if not args.sample_manifest.is_file():
        raise FileNotFoundError(f"missing sample manifest: {args.sample_manifest}")
    model = load_model(args.model_dir, args.device, runtime_overrides=runtime_overrides_from_args(args))
    content_metric = create_content_metric().to(next(model.parameters()).device) if args.eval_content_preservation else None
    return scan_controls(model, args.sample_manifest, args.output_dir, config, content_metric=content_metric)


def main(argv=None):
    args = setup_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = run(args)
    print(f"wrote {len(report['measurements'])} measurements to {args.output_dir / 'scan_report.json'}")


if __name__ == "__main__":
    main()
