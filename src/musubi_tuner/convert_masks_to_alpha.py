"""
Convert separate image and mask files into single RGBA PNG files.

This utility merges existing image+mask pairs into RGBA PNGs where the alpha channel
contains the mask data. This eliminates filename mismatch bugs that occur with separate
mask directories.

The conversion uses the same resampling policy as the trainer:
- NEAREST interpolation for upscaling (preserves discrete mask levels)
- AREA interpolation for downscaling (preserves average weight)

Usage:
    python convert_masks_to_alpha.py \
        --image_directory /path/to/images \
        --mask_directory /path/to/masks \
        --output_directory /path/to/rgba_output \
        --caption_directory /path/to/captions \  # Optional: copy captions
        --dry_run  # Preview without writing
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def resize_mask_to_match_image(mask: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """
    Resize mask using trainer-consistent interpolation.

    Uses NEAREST for upscaling (preserves discrete levels) and AREA for downscaling
    (preserves average weight), matching the behavior in resize_mask_to_bucket().

    Args:
        mask: Grayscale PIL Image (mode "L")
        target_size: (width, height) to resize to

    Returns:
        Resized grayscale PIL Image
    """
    mask_w, mask_h = mask.size
    target_w, target_h = target_size

    if (mask_w, mask_h) == (target_w, target_h):
        return mask

    # Determine if upscaling or downscaling based on area ratio
    mask_area = mask_w * mask_h
    target_area = target_w * target_h

    if target_area > mask_area:
        # Upscaling: NEAREST (preserve discrete levels)
        return mask.resize(target_size, Image.NEAREST)
    else:
        # Downscaling: AREA interpolation via cv2
        mask_np = np.array(mask)
        resized = cv2.resize(mask_np, target_size, interpolation=cv2.INTER_AREA)
        return Image.fromarray(resized)


def merge_image_and_mask(image_path: str, mask_path: str, output_path: str) -> bool:
    """
    Merge an image and mask into a single RGBA PNG.

    The image's RGB values are preserved exactly (no compositing), and the mask
    becomes the alpha channel.

    Args:
        image_path: Path to source image (RGB or RGBA)
        mask_path: Path to mask image (grayscale)
        output_path: Path for output RGBA PNG

    Returns:
        True if successful

    Raises:
        ValueError: If aspect ratios don't match within 1% tolerance
    """
    # Use context managers to ensure file handles are closed properly
    with Image.open(image_path) as img:
        # Strip existing alpha by slicing, NOT convert("RGB") which composites against black
        if img.mode == "RGBA":
            img_rgb = np.array(img)[..., :3]
        else:
            img_rgb = np.array(img.convert("RGB"))

    img_h, img_w = img_rgb.shape[:2]

    with Image.open(mask_path) as mask_img:
        mask = mask_img.convert("L")
        mask_w, mask_h = mask.size

        if (mask_w, mask_h) != (img_w, img_h):
            # Check aspect ratio - error if significantly different (1% tolerance)
            # This prevents silently distorting masks when resize would misalign them.
            img_aspect = img_w / img_h
            mask_aspect = mask_w / mask_h
            if abs(img_aspect - mask_aspect) > 0.01:
                raise ValueError(
                    f"Aspect ratio mismatch: "
                    f"image={img_w}x{img_h} (aspect={img_aspect:.3f}), "
                    f"mask={mask_w}x{mask_h} (aspect={mask_aspect:.3f}). "
                    f"Cannot resize without distortion. Fix mask dimensions first."
                )
            mask = resize_mask_to_match_image(mask, (img_w, img_h))

        mask_np = np.array(mask)

    # Build RGBA by stacking (preserves exact RGB values)
    rgba = np.dstack([img_rgb, mask_np])
    Image.fromarray(rgba, mode="RGBA").save(output_path, "PNG")
    return True


def glob_images(directory: str) -> list[str]:
    """Find all image files in a directory, sorted for deterministic output."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(Path(directory).glob(f"*{ext}"))
        images.extend(Path(directory).glob(f"*{ext.upper()}"))
    # Sort for deterministic results across filesystems
    return sorted(set(str(p) for p in images))


def main():
    parser = argparse.ArgumentParser(
        description="Convert separate image and mask files into single RGBA PNG files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion
  python convert_masks_to_alpha.py \\
      --image_directory /path/to/images \\
      --mask_directory /path/to/masks \\
      --output_directory /path/to/rgba_output

  # With caption copying
  python convert_masks_to_alpha.py \\
      --image_directory /path/to/images \\
      --mask_directory /path/to/masks \\
      --output_directory /path/to/rgba_output \\
      --caption_directory /path/to/captions

  # Preview without writing
  python convert_masks_to_alpha.py \\
      --image_directory /path/to/images \\
      --mask_directory /path/to/masks \\
      --output_directory /path/to/rgba_output \\
      --dry_run
""",
    )
    parser.add_argument("--image_directory", type=str, required=True, help="Directory containing source images")
    parser.add_argument("--mask_directory", type=str, required=True, help="Directory containing mask images")
    parser.add_argument("--output_directory", type=str, required=True, help="Directory for output RGBA PNGs")
    parser.add_argument(
        "--caption_directory",
        type=str,
        default=None,
        help="Optional directory containing captions to copy (default: same as image_directory)",
    )
    parser.add_argument("--caption_extension", type=str, default=".txt", help="Caption file extension (default: .txt)")
    parser.add_argument("--dry_run", action="store_true", help="Preview what would be done without actually writing files")
    parser.add_argument("--skip_existing", action="store_true", help="Skip files that already exist in output directory")

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.image_directory):
        raise ValueError(f"Image directory not found: {args.image_directory}")
    if not os.path.isdir(args.mask_directory):
        raise ValueError(f"Mask directory not found: {args.mask_directory}")

    # Create output directory if needed
    if not args.dry_run:
        os.makedirs(args.output_directory, exist_ok=True)

    # If caption_directory not specified, default to image_directory
    caption_directory = args.caption_directory if args.caption_directory else args.image_directory

    # Glob images and masks
    logger.info(f"Finding images in {args.image_directory}")
    image_paths = glob_images(args.image_directory)
    logger.info(f"Found {len(image_paths)} images")

    logger.info(f"Finding masks in {args.mask_directory}")
    mask_paths = glob_images(args.mask_directory)
    logger.info(f"Found {len(mask_paths)} masks")

    # Build mask lookup dict (basename_no_ext -> full_path)
    mask_by_basename: dict[str, str] = {}
    for mask_path in mask_paths:
        basename_no_ext = os.path.splitext(os.path.basename(mask_path))[0]
        if basename_no_ext in mask_by_basename:
            logger.warning(f"Duplicate mask basename '{basename_no_ext}', keeping first")
        else:
            mask_by_basename[basename_no_ext] = mask_path

    # Match images to masks
    matched_pairs: list[tuple[str, str, str]] = []  # (image_path, mask_path, output_path)
    unmatched_images: list[str] = []

    for image_path in image_paths:
        image_basename = os.path.basename(image_path)
        image_basename_no_ext = os.path.splitext(image_basename)[0]
        output_path = os.path.join(args.output_directory, f"{image_basename_no_ext}.png")

        if args.skip_existing and os.path.exists(output_path):
            continue

        if image_basename_no_ext in mask_by_basename:
            matched_pairs.append((image_path, mask_by_basename[image_basename_no_ext], output_path))
        else:
            unmatched_images.append(image_basename_no_ext)

    logger.info(f"Matched {len(matched_pairs)} image+mask pairs")
    if unmatched_images:
        logger.warning(f"{len(unmatched_images)} images have no matching mask and will be skipped")
        if len(unmatched_images) <= 10:
            for name in unmatched_images:
                logger.warning(f"  - {name}")

    if args.dry_run:
        logger.info("DRY RUN - no files will be written")
        for image_path, mask_path, output_path in matched_pairs[:5]:
            logger.info(f"  Would merge: {os.path.basename(image_path)} + {os.path.basename(mask_path)}")
        if len(matched_pairs) > 5:
            logger.info(f"  ... and {len(matched_pairs) - 5} more")
        return

    # Process matched pairs
    success_count = 0
    error_count = 0

    for image_path, mask_path, output_path in tqdm(matched_pairs, desc="Converting"):
        try:
            merge_image_and_mask(image_path, mask_path, output_path)
            success_count += 1

            # Copy caption if it exists
            image_basename_no_ext = os.path.splitext(os.path.basename(image_path))[0]
            caption_path = os.path.join(caption_directory, f"{image_basename_no_ext}{args.caption_extension}")
            if os.path.exists(caption_path):
                output_caption_path = os.path.join(args.output_directory, f"{image_basename_no_ext}{args.caption_extension}")
                shutil.copy2(caption_path, output_caption_path)

        except Exception as e:
            logger.error(f"Error processing {os.path.basename(image_path)}: {e}")
            error_count += 1

    logger.info(f"Conversion complete: {success_count} succeeded, {error_count} failed")
    if success_count > 0:
        logger.info(f"Output directory: {args.output_directory}")
        logger.info("IMPORTANT: Use a fresh cache_directory when training with these RGBA images!")


if __name__ == "__main__":
    main()
