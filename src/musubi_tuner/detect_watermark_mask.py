"""Detect a static watermark / logo overlay in videos and write a loss mask for it.

The detector assumes only that the overlay is *static*: it sits at the same place whenever
it is on screen. It may be opaque or semi-transparent, and it need not be on screen the
whole time. Two complementary signals are unioned, because neither covers that whole range
alone (see ``detect_watermark_region``):

* low temporal deviation — an overlay that hides the content behind it, see
  ``trimmed_temporal_std``. Discarding the most-deviating frames per pixel, up to
  ``--frame_tolerance`` of them, keeps a watermark that briefly disappears detectable.
* a frame-to-frame-consistent image gradient — an overlay that only tints the content, see
  ``consistent_gradient``.

What both signals mark is turned into a bounding box per region, because the mask has to
cover what the watermark bleeds into once the VAE encodes the frame, not just the pixels
it literally occupies.

The written PNG follows the convention used by the training scripts:

* ``255`` – train on this pixel
* ``0``   – ignore this pixel (watermark)

Only ``opencv-python`` and ``numpy`` are required.
"""

import argparse
from collections import Counter
import glob
import os
from typing import Optional

import cv2
import numpy as np

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


VIDEO_EXTENSIONS = [".mp4", ".webm", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg"]

DEFAULT_MASK_SUFFIX = "_wmask.png"


def sample_frames(video_path: str, n_frames: int) -> Optional[np.ndarray]:
    """Read up to ``n_frames`` grayscale frames spread evenly over the video.

    Returns a ``(N, H, W)`` float32 array, or ``None`` if the video could not be read.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []

    if total_frames > 0:
        # evenly spaced indices, seeking to each one
        indices = np.unique(np.linspace(0, total_frames - 1, num=min(n_frames, total_frames)).astype(int))
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    else:
        # frame count unknown (some containers): fall back to a sequential read
        while len(frames) < n_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    if len(frames) < 2:
        return None

    # frames can differ in size if the container lies about the resolution; keep the first size
    height, width = frames[0].shape
    frames = [f for f in frames if f.shape == (height, width)]
    if len(frames) < 2:
        return None

    return np.stack(frames).astype(np.float32)


def make_border_mask(height: int, width: int, margin: float) -> np.ndarray:
    """Boolean mask that is True only inside a border band of ``margin`` (fraction of the frame)."""
    band_y = max(1, int(round(height * margin)))
    band_x = max(1, int(round(width * margin)))
    border = np.ones((height, width), dtype=bool)
    border[band_y : height - band_y, band_x : width - band_x] = False
    return border


def trimmed_temporal_std(frames: np.ndarray, frame_tolerance: float) -> np.ndarray:
    """Temporal standard deviation per pixel, ignoring the most deviating frames.

    A watermark is rarely present in literally every frame: it can be briefly occluded by
    bright content, fade in over the first frames, or be absent from a title card. Those
    frames alone would blow up a plain ``frames.std(axis=0)`` and hide the watermark.

    So per pixel, the ``frame_tolerance`` fraction of frames furthest from that pixel's
    temporal median is discarded, and the standard deviation is taken over what remains.
    The median is used as the reference because it is itself unaffected by a minority of
    watermark-free frames. With ``frame_tolerance=0`` this is exactly ``frames.std(axis=0)``.
    """
    n_frames = frames.shape[0]
    n_keep = max(2, int(np.ceil(n_frames * (1.0 - frame_tolerance))))
    if n_keep >= n_frames:
        return frames.std(axis=0)

    deviation = np.abs(frames - np.median(frames, axis=0))

    # Per-pixel deviation of the n_keep-th closest frame, then keep everything within it.
    # Ties are kept rather than broken arbitrarily, so a perfectly static pixel keeps all
    # of its frames instead of an arbitrary subset of identical ones.
    cutoff = np.partition(deviation, n_keep - 1, axis=0)[n_keep - 1]
    kept = deviation <= cutoff
    del deviation

    count = kept.sum(axis=0)
    mean = (frames * kept).sum(axis=0) / count
    variance = ((frames - mean) ** 2 * kept).sum(axis=0) / count
    return np.sqrt(variance)


def _temporal_median_sobel(frames: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Per-pixel temporal median of one Sobel derivative, computed one axis at a time."""
    derivatives = np.empty_like(frames)
    for index, frame in enumerate(frames):
        derivatives[index] = cv2.Sobel(frame, cv2.CV_32F, dx, dy, ksize=3)
    return np.median(derivatives, axis=0)


def consistent_gradient(frames: np.ndarray, gradient_threshold: float) -> np.ndarray:
    """Pixels whose image gradient is the same in every sampled frame.

    This is what finds a *semi-transparent* watermark. Blended as
    ``alpha * W + (1 - alpha) * content``, such a watermark keeps varying with the content
    underneath, so its temporal deviation is merely scaled by ``1 - alpha``: at 30% opacity
    it is still 70% of the content's, far above any usable ``--threshold``.

    The gradient does separate it. Content gradients change sign and position from frame to
    frame and cancel in a temporal median, while the watermark contributes the same
    ``alpha * grad(W)`` in every frame and survives. The median also tolerates a watermark
    missing from a minority of frames on its own, without any trimming.

    (The same observation underpins Dekel et al., "On the Effectiveness of Visible
    Watermarks", CVPR 2017.)
    """
    gx = _temporal_median_sobel(frames, 1, 0)
    gy = _temporal_median_sobel(frames, 0, 1)
    edges = np.hypot(gx, gy)

    # The watermark covers a small part of the frame, so the median of the map is a good
    # stand-in for "how much consistent gradient this footage has anyway" — the threshold is
    # expressed relative to it. The floor keeps footage with almost no texture from making
    # the threshold collapse to zero.
    baseline = max(float(np.median(edges)), 1.0)
    return edges > gradient_threshold * baseline


def bounding_boxes(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Replace each connected region of ``mask`` with its bounding box.

    The detectors return the watermark's *outline* (gradient) or its solid pixels
    (deviation), neither of which is the region that has to be excluded from the loss. That
    region is broader than the glyphs themselves: the VAE encoder mixes a neighbourhood of
    pixels into every latent cell, so a watermark bleeds into latents that its own pixels do
    not cover, and a mask traced tightly around thin strokes under-masks once it is
    downsampled to latent resolution.

    A bounding box covers that neighbourhood, needs no morphological reconstruction of the
    interior, and errs in the safe direction — losing a little clean content around a logo
    costs far less than leaving watermark in the loss.
    """
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = np.zeros_like(mask)
    for label in range(1, num_labels):  # 0 is the background
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        width, height = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        boxes[y : y + height, x : x + width] = 1
    return boxes


def detect_watermark_region(
    frames: np.ndarray,
    threshold: float = 8.0,
    corner_only: bool = False,
    corner_margin: float = 0.25,
    dilate: int = 3,
    min_area: int = 32,
    frame_tolerance: float = 0.1,
    gradient_threshold: float = 3.0,
) -> np.ndarray:
    """Return a boolean ``(H, W)`` array that is True where the watermark is.

    Two complementary signals are unioned, because neither covers the whole range on its
    own: low temporal deviation finds an overlay that hides the content behind it, and a
    frame-to-frame-consistent gradient finds one that only tints it. An opaque logo trips
    both; a 30% opacity logo trips only the second; a large flat opaque patch trips the
    first strongly and the second only along its border.
    """
    candidate = trimmed_temporal_std(frames, frame_tolerance) < threshold
    if gradient_threshold > 0:  # non-positive disables the signal; the CLI rejects it
        candidate |= consistent_gradient(frames, gradient_threshold)

    if corner_only:
        candidate &= make_border_mask(frames.shape[1], frames.shape[2], corner_margin)

    candidate = candidate.astype(np.uint8)

    # Bridge the two edges of a stroke and any single-pixel holes, so that one logo becomes
    # one region rather than a scatter of fragments with a bounding box each.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)

    candidate = bounding_boxes(candidate, min_area)

    if dilate > 0:
        # Margin around each box: anti-aliased watermark edges blend into the content and
        # still carry watermark signal.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        candidate = cv2.dilate(candidate, kernel)

    return candidate.astype(bool)


def list_videos(video_dir: str, recursive: bool) -> list[str]:
    pattern = os.path.join(video_dir, "**", "*") if recursive else os.path.join(video_dir, "*")
    paths = sorted(p for p in glob.glob(pattern, recursive=recursive) if os.path.isfile(p))
    return [p for p in paths if os.path.splitext(p)[1].lower() in VIDEO_EXTENSIONS]


def process_video(video_path: str, args: argparse.Namespace) -> str:
    """Detect and write the mask for one video.

    Returns the outcome as one of "written", "exists", "unreadable", "not_detected" or
    "over_max_coverage", so that `main` can summarize a whole directory at the end.
    """
    mask_path = os.path.splitext(video_path)[0] + args.suffix
    if os.path.exists(mask_path) and not args.overwrite:
        logger.info(f"Mask already exists, skipping (use --overwrite to regenerate): {mask_path}")
        return "exists"

    frames = sample_frames(video_path, args.n_frames)
    if frames is None:
        logger.warning(f"Could not read enough frames, skipping: {video_path}")
        return "unreadable"

    watermark = detect_watermark_region(
        frames,
        args.threshold,
        args.corner_only,
        args.corner_margin,
        args.dilate,
        args.min_area,
        args.frame_tolerance,
        args.gradient_threshold,
    )

    coverage = float(watermark.mean())
    if coverage == 0.0:
        logger.info(f"No static watermark detected, no mask written: {video_path}")
        return "not_detected"

    if coverage > args.max_coverage:
        # Usually a locked-off camera: everything is temporally static, so the detection is
        # meaningless. Extreme letterboxing can reach this too — the bars are a correct
        # detection, there are just so many of them that cropping beats masking.
        logger.warning(
            f"Detected watermark covers {coverage:.1%} of the frame (> --max_coverage {args.max_coverage:.1%}),"
            f" no mask written. The camera is probably static, or the clip is heavily letterboxed;"
            f" try --corner_only, crop the clip, or drop it: {video_path}"
        )
        return "over_max_coverage"

    mask = np.where(watermark, 0, 255).astype(np.uint8)
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        mask_path = os.path.join(args.output_dir, os.path.basename(mask_path))
    cv2.imwrite(mask_path, mask)
    logger.info(f"Wrote mask covering {coverage:.2%} of the frame: {mask_path}")
    return "written"


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a static watermark in videos and write a per-video loss mask (255 = train, 0 = ignore)."
    )
    parser.add_argument("--video_dir", type=str, required=True, help="directory containing the videos")
    parser.add_argument("--recursive", action="store_true", help="search --video_dir recursively")
    parser.add_argument(
        "--n_frames",
        type=int,
        default=30,
        help="number of frames sampled per video (default: 30). More frames make the detection more reliable on"
        " slow-moving footage, and let content gradients cancel out better, which is what makes a"
        " semi-transparent watermark stand out",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=8.0,
        help="temporal standard deviation below which a pixel is considered static (0-255 scale, default: 8.0)",
    )
    parser.add_argument(
        "--frame_tolerance",
        type=float,
        default=0.1,
        help="fraction of sampled frames allowed to not show the watermark (default: 0.1)."
        " Per pixel, that many of the most deviating frames are discarded before measuring the deviation,"
        " so a watermark that briefly disappears, is occluded, or fades in and out is still detected."
        " Set to 0 to require the watermark in every sampled frame",
    )
    parser.add_argument(
        "--gradient_threshold",
        type=float,
        default=3.0,
        help="how far above the frame's own median a pixel's frame-to-frame-consistent gradient must be"
        " to count as a semi-transparent watermark edge (default: 3.0, i.e. 3x that median)."
        " Ordinary content and static background sit around 1x, a watermark at 20-50%% opacity lands"
        " between 3x and 14x depending on opacity and --n_frames."
        " Lower it if a faint watermark is missed, raise it if unrelated static structure is picked up",
    )
    parser.add_argument(
        "--corner_only",
        action="store_true",
        help="only look for the watermark in a band along the frame border (see --corner_margin)",
    )
    parser.add_argument(
        "--corner_margin",
        type=float,
        default=0.25,
        help="width of the border band for --corner_only, as a fraction of the frame size (default: 0.25)",
    )
    parser.add_argument("--dilate", type=int, default=3, help="dilate the detected region by N pixels (default: 3)")
    parser.add_argument("--min_area", type=int, default=32, help="drop detected regions smaller than N pixels (default: 32)")
    parser.add_argument(
        "--max_coverage",
        type=float,
        default=0.4,
        help="skip the video if the detected region covers more than this fraction of the frame (default: 0.4)."
        " The bound exists for locked-off shots, where the whole frame is temporally static and the detection"
        " stops meaning anything; those measure around 0.8. Letterbox bars are a legitimate detection and eat"
        " up to a third of the frame on their own, which is why the default sits between the two",
    )
    parser.add_argument(
        "--suffix", type=str, default=DEFAULT_MASK_SUFFIX, help=f"mask file suffix (default: {DEFAULT_MASK_SUFFIX})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="write masks here instead of next to the videos (default: next to the video)"
    )
    parser.add_argument("--overwrite", action="store_true", help="regenerate masks that already exist")
    return parser


def main() -> None:
    args = setup_parser().parse_args()

    if not os.path.isdir(args.video_dir):
        raise ValueError(f"--video_dir is not a directory: {args.video_dir}")
    if not 0.0 <= args.frame_tolerance < 1.0:
        raise ValueError(f"--frame_tolerance must be in [0, 1): {args.frame_tolerance}")
    if args.gradient_threshold <= 0.0:
        raise ValueError(f"--gradient_threshold must be positive: {args.gradient_threshold}")

    video_paths = list_videos(args.video_dir, args.recursive)
    if not video_paths:
        logger.warning(f"No videos found in {args.video_dir}")
        return

    logger.info(f"Found {len(video_paths)} videos")
    outcomes = Counter(process_video(video_path, args) for video_path in video_paths)

    logger.info(f"Done. Wrote {outcomes['written']} mask(s) for {len(video_paths)} video(s).")
    for outcome, message in [
        ("exists", "already had a mask (use --overwrite to regenerate)"),
        ("not_detected", "had no detectable static watermark"),
        ("unreadable", "could not be read"),
    ]:
        if outcomes[outcome]:
            logger.info(f"  {outcomes[outcome]} video(s) {message}")

    # Surfaced separately: a large batch of these means the threshold or --corner_only needs
    # revisiting, and the individual warnings are easy to lose in a long log.
    if outcomes["over_max_coverage"]:
        logger.warning(
            f"  {outcomes['over_max_coverage']} video(s) exceeded --max_coverage"
            f" ({args.max_coverage:.1%} of the frame) and were left without a mask."
            " Re-run with --corner_only, crop letterboxed clips, or drop those clips."
        )


if __name__ == "__main__":
    main()
