#!/usr/bin/env python3
"""
Create binary instance masks for a specific person (or best-guess) in each image.

Primary use-case: group photos for mask-weighted LoRA training.
You can keep the original image untouched and later zero-out weighted masks outside
the selected subject.

Selection strategies
--------------------
1) FaceID + Instance Segmentation (recommended for group photos)
   - Use InsightFace to match the subject's face to a reference photo
   - Then use either:
       - YOLOv8-seg (fast) OR
       - SAM 3 (more flexible; requires separate environment/checkpoints)
     to get person instance masks and pick the one that overlaps the matched face.

2) Instance Segmentation only (fallback)
   - If no FaceID reference is provided, we pick the largest detected "person".

Output
------
Writes 8-bit PNG masks (mode "L") with values {0,255}, one per input image, matched
by file stem.

Example
-------
python tools/create_instance_masks.py \
  --input /path/to/images \
  --output /path/to/instance_masks \
  --reference /path/to/reference_face.jpg \
  --backend yolo \
  --id-threshold 0.45
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

if sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 is required.")


# ---------------------------------------------------------------------------
# Lazy deps
# ---------------------------------------------------------------------------

_np = None
_cv2 = None
_PIL = None
_tqdm = None


def _require_base_deps() -> None:
    global _np, _cv2, _PIL, _tqdm
    if _np is not None:
        return
    try:
        import numpy as np

        _np = np
    except ImportError as exc:
        raise SystemExit("ERROR: numpy is required. Install with: pip install numpy") from exc
    try:
        import cv2

        _cv2 = cv2
    except ImportError as exc:
        raise SystemExit("ERROR: opencv-python is required. Install with: pip install opencv-python") from exc
    try:
        from PIL import Image, ExifTags

        _PIL = (Image, ExifTags)
    except ImportError as exc:
        raise SystemExit("ERROR: Pillow is required. Install with: pip install Pillow") from exc
    try:
        from tqdm import tqdm

        _tqdm = tqdm
    except ImportError:
        _tqdm = lambda it, **_: it  # type: ignore[assignment]


def _require_ultralytics():
    try:
        from ultralytics import YOLO

        return YOLO
    except ImportError as exc:
        raise SystemExit("ERROR: ultralytics is required. Install with: pip install ultralytics") from exc


def _require_insightface():
    try:
        from insightface.app import FaceAnalysis

        return FaceAnalysis
    except ImportError as exc:
        raise SystemExit(
            "ERROR: insightface is required. Install with: pip install insightface onnxruntime-gpu (or onnxruntime)"
        ) from exc


def _require_sam3():
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        return build_sam3_image_model, Sam3Processor
    except ImportError as exc:
        raise SystemExit(
            "ERROR: sam3 is not installed/available in this environment.\n"
            "Install in a separate env per the SAM 3 README, or use --backend yolo."
        ) from exc


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def _exif_transpose(image):
    Image, ExifTags = _PIL
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = image._getexif()
        if exif is None:
            return image
        orientation_value = exif.get(orientation)
        if orientation_value == 3:
            return image.rotate(180, expand=True)
        if orientation_value == 6:
            return image.rotate(270, expand=True)
        if orientation_value == 8:
            return image.rotate(90, expand=True)
    except Exception:
        return image
    return image


def _iter_images(input_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def _load_rgb(path: Path):
    Image, _ = _PIL
    img = Image.open(path)
    img = _exif_transpose(img).convert("RGB")
    return _np.array(img)


def _rgb_to_bgr(rgb):
    return rgb[..., ::-1].copy()


def _clamp_box_xyxy(box: Sequence[int], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


@dataclass(frozen=True)
class CandidateMask:
    mask: "object"  # np.ndarray uint8 {0,255}
    score: float
    box_xyxy: Tuple[int, int, int, int]


def _select_best_candidate(
    candidates: Sequence[CandidateMask],
    face_box_xyxy: Optional[Tuple[int, int, int, int]],
) -> Optional[CandidateMask]:
    if not candidates:
        return None

    if face_box_xyxy is None:
        return max(candidates, key=lambda c: (int((c.mask > 0).sum()), c.score))

    x1, y1, x2, y2 = face_box_xyxy
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # 1) Prefer masks containing the face center.
    containing = [c for c in candidates if c.mask[cy, cx] > 0]
    if containing:
        return max(containing, key=lambda c: (c.score, int((c.mask > 0).sum())))

    # 2) Fallback: max overlap with face box.
    best = None
    best_overlap = 0
    for c in candidates:
        crop = c.mask[y1:y2, x1:x2]
        overlap = int((crop > 0).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best = c
    return best if best_overlap > 0 else None


# ---------------------------------------------------------------------------
# FaceID
# ---------------------------------------------------------------------------


class FaceIdMatcher:
    def __init__(self, reference_bgr, providers: list[str], ctx_id: int, det_size: Tuple[int, int]):
        FaceAnalysis = _require_insightface()
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

        faces = self.app.get(reference_bgr)
        if not faces:
            raise ValueError("No face detected in reference image.")
        self.ref_face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )

        emb = self.ref_face.embedding
        norm = float((_np.linalg.norm(emb) + 1e-12))
        self.ref_emb = emb / norm

    def match_face(self, image_bgr, threshold: float) -> Optional[Tuple[Tuple[int, int, int, int], float]]:
        faces = self.app.get(image_bgr)
        if not faces:
            return None

        best_score = -1.0
        best_box = None
        for face in faces:
            emb = face.embedding
            emb = emb / float((_np.linalg.norm(emb) + 1e-12))
            score = float(_np.dot(self.ref_emb, emb))
            if score > best_score:
                best_score = score
                best_box = tuple(int(v) for v in face.bbox)

        if best_box is None or best_score < threshold:
            return None
        return best_box, best_score


def _default_ort_providers() -> list[str]:
    # Try to prefer GPU when possible, but fall back to CPU.
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:
        available = set()

    providers: list[str] = []
    for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"):
        if not available or p in available:
            providers.append(p)
    return providers or ["CPUExecutionProvider"]


def _default_insightface_ctx_id(providers: Sequence[str]) -> int:
    return 0 if "CUDAExecutionProvider" in providers else -1


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class YoloSegBackend:
    def __init__(self, model_name: str, device: str, conf: float):
        YOLO = _require_ultralytics()
        self.model = YOLO(model_name)
        self.device = device
        self.conf = conf

    def get_person_candidates(self, image_bgr) -> list[CandidateMask]:
        h, w = image_bgr.shape[:2]
        results = self.model.predict(source=image_bgr, device=self.device, conf=self.conf, classes=[0], verbose=False)
        if not results or results[0].masks is None or results[0].boxes is None:
            return []

        r = results[0]
        masks = r.masks.data  # (N, mh, mw)
        boxes = r.boxes.xyxy
        scores = r.boxes.conf if hasattr(r.boxes, "conf") else None

        candidates: list[CandidateMask] = []
        for i in range(masks.shape[0]):
            m = masks[i].detach().cpu().numpy()
            m = _cv2.resize(m, (w, h), interpolation=_cv2.INTER_LINEAR)
            m = (m > 0.5).astype(_np.uint8) * 255

            b = boxes[i].detach().cpu().numpy()
            b = _clamp_box_xyxy(b, w, h)

            s = float(scores[i].item()) if scores is not None else 0.0
            candidates.append(CandidateMask(mask=m, score=s, box_xyxy=b))
        return candidates


class Sam3Backend:
    def __init__(self, device: str, confidence_threshold: float, prompt: str):
        build_sam3_image_model, Sam3Processor = _require_sam3()
        self.prompt = prompt
        self.model = build_sam3_image_model(device=device)
        self.processor = Sam3Processor(self.model, device=device, confidence_threshold=confidence_threshold)

    def get_person_candidates(self, image_rgb) -> list[CandidateMask]:
        Image, _ = _PIL
        image = Image.fromarray(image_rgb)
        state = self.processor.set_image(image)
        state = self.processor.set_text_prompt(prompt=self.prompt, state=state)

        masks = state.get("masks", None)
        boxes = state.get("boxes", None)
        scores = state.get("scores", None)
        if masks is None or boxes is None or scores is None:
            return []

        masks = masks.squeeze(1)  # (N,H,W)
        h, w = image_rgb.shape[:2]
        candidates: list[CandidateMask] = []
        for i in range(masks.shape[0]):
            m = masks[i].detach().cpu().numpy().astype(_np.uint8) * 255
            b = boxes[i].detach().cpu().numpy()
            b = _clamp_box_xyxy(b, w, h)
            s = float(scores[i].item())
            candidates.append(CandidateMask(mask=m, score=s, box_xyxy=b))
        return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Create per-image binary instance masks for a target subject.")
    parser.add_argument("--input", required=True, help="Input image directory")
    parser.add_argument("--output", required=True, help="Output mask directory")
    parser.add_argument("--files", help="Comma-separated file stems to process (optional)")

    # FaceID
    parser.add_argument("--reference", help="Reference face photo (enables FaceID selection)")
    parser.add_argument("--id-threshold", type=float, default=0.45, help="Cosine similarity threshold (typical: 0.4-0.6)")
    parser.add_argument("--id-det-size", default="640,640", help="Face detector input size, e.g. 640,640")

    # Instance backend
    parser.add_argument("--backend", choices=["yolo", "sam3"], default="yolo", help="Instance segmentation backend")
    parser.add_argument("--fallback-largest", action="store_true", help="If FaceID fails, pick the largest person")
    parser.add_argument(
        "--write-empty-on-fail",
        action="store_true",
        help="Write an all-black mask when selection fails (instead of skipping the file).",
    )

    # YOLO
    parser.add_argument("--yolo-model", default="yolov8l-seg.pt", help="Ultralytics segmentation model")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--yolo-device", default="cuda:0", help="YOLO device, e.g. cuda:0 or cpu")

    # SAM3
    parser.add_argument("--sam3-device", default="cuda", help="SAM3 device: cuda or cpu")
    parser.add_argument("--sam3-prompt", default="person", help="SAM3 text prompt (default: person)")
    parser.add_argument("--sam3-confidence", type=float, default=0.5, help="SAM3 confidence threshold")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    _require_base_deps()

    det_w, det_h = (int(x) for x in args.id_det_size.split(","))

    # Optional FaceID matcher
    matcher: Optional[FaceIdMatcher] = None
    if args.reference:
        ref_rgb = _load_rgb(Path(args.reference))
        ref_bgr = _rgb_to_bgr(ref_rgb)
        providers = _default_ort_providers()
        ctx_id = _default_insightface_ctx_id(providers)
        matcher = FaceIdMatcher(ref_bgr, providers=providers, ctx_id=ctx_id, det_size=(det_w, det_h))

    # Backend
    if args.backend == "sam3":
        backend = Sam3Backend(device=args.sam3_device, confidence_threshold=args.sam3_confidence, prompt=args.sam3_prompt)
    else:
        backend = YoloSegBackend(model_name=args.yolo_model, device=args.yolo_device, conf=args.yolo_conf)

    files = _iter_images(input_dir)
    if args.files:
        stems = {s.strip() for s in args.files.split(",") if s.strip()}
        files = [p for p in files if p.stem in stems]

    if not files:
        print("No images found to process.")
        return

    Image, _ = _PIL
    tqdm = _tqdm

    for path in tqdm(files, desc="Instance masks"):
        out_path = output_dir / f"{path.stem}.png"
        if out_path.exists():
            continue

        rgb = _load_rgb(path)
        h, w = rgb.shape[:2]
        face_box = None

        if matcher is not None:
            bgr = _rgb_to_bgr(rgb)
            match = matcher.match_face(bgr, threshold=args.id_threshold)
            if match is not None:
                raw_box, _score = match
                face_box = _clamp_box_xyxy(raw_box, w, h)
            elif not args.fallback_largest:
                # Skip if we require a match and cannot find one.
                if args.write_empty_on_fail:
                    Image.fromarray(_np.zeros((h, w), dtype=_np.uint8), mode="L").save(out_path, "PNG")
                continue

        if args.backend == "sam3":
            candidates = backend.get_person_candidates(rgb)
        else:
            candidates = backend.get_person_candidates(_rgb_to_bgr(rgb))

        best = _select_best_candidate(candidates, face_box_xyxy=face_box)
        if best is None:
            if args.write_empty_on_fail:
                Image.fromarray(_np.zeros((h, w), dtype=_np.uint8), mode="L").save(out_path, "PNG")
            continue

        Image.fromarray(best.mask, mode="L").save(out_path, "PNG")


if __name__ == "__main__":
    main()
