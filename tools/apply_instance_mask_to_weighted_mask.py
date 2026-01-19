#!/usr/bin/env python3
"""
Apply a binary instance mask to an existing weighted mask.

This is intentionally simple: it lets you keep your Sapiens v4.1/v4.x weighted
masks, then "erase" everyone except the chosen subject from group photos by
multiplying with an instance mask (0/255).

Inputs
------
- weighted_mask_dir: grayscale weighted masks (0-255)
- instance_mask_dir: binary instance masks (0/255), same stems

Output
------
- output_dir: final weighted masks (0-255), outside-instance pixels set to 0

Example
-------
python tools/apply_instance_mask_to_weighted_mask.py \
  --weighted-masks /path/to/weighted_masks \
  --instance-masks /path/to/instance_masks \
  --output /path/to/weighted_masks_subject_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 is required.")


def _require_deps():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit("ERROR: numpy is required. Install with: pip install numpy") from exc
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("ERROR: opencv-python is required. Install with: pip install opencv-python") from exc
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("ERROR: Pillow is required. Install with: pip install Pillow") from exc
    return np, cv2, Image


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply binary instance masks to weighted masks.")
    parser.add_argument("--weighted-masks", required=True, help="Directory with weighted masks (0-255)")
    parser.add_argument("--instance-masks", required=True, help="Directory with instance masks (0/255)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--dilate", type=int, default=0, help="Dilate instance mask by N pixels before applying")
    parser.add_argument(
        "--missing",
        choices=["copy", "zero", "skip", "error"],
        default="copy",
        help="What to do when an instance mask is missing for a stem (default: copy).",
    )

    args = parser.parse_args()
    np, cv2, Image = _require_deps()

    weighted_dir = Path(args.weighted_masks)
    instance_dir = Path(args.instance_masks)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    weighted_paths = sorted([p for p in weighted_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not weighted_paths:
        print("No weighted masks found.")
        return

    kernel = None
    if args.dilate > 0:
        k = args.dilate * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)

    for wpath in weighted_paths:
        ipath = instance_dir / f"{wpath.stem}.png"

        wmask = np.array(Image.open(wpath).convert("L"))
        if not ipath.exists():
            if args.missing == "skip":
                continue
            if args.missing == "error":
                raise SystemExit(f"ERROR: missing instance mask for {wpath.stem}: {ipath}")
            if args.missing == "zero":
                out = np.zeros_like(wmask, dtype=np.uint8)
                Image.fromarray(out, mode="L").save(output_dir / f"{wpath.stem}.png", "PNG")
                continue
            # copy (default)
            Image.fromarray(wmask, mode="L").save(output_dir / f"{wpath.stem}.png", "PNG")
            continue

        imask = np.array(Image.open(ipath).convert("L"))

        if imask.shape != wmask.shape:
            imask = cv2.resize(imask, (wmask.shape[1], wmask.shape[0]), interpolation=cv2.INTER_NEAREST)

        if kernel is not None:
            imask = cv2.dilate((imask > 127).astype(np.uint8) * 255, kernel, iterations=1)

        out = wmask.copy()
        out[imask <= 127] = 0
        Image.fromarray(out, mode="L").save(output_dir / f"{wpath.stem}.png", "PNG")


if __name__ == "__main__":
    main()
