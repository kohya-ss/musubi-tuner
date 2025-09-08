#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for blurring faces with YOLO for training LoRA with Blissful Tuner
Created on Sat Mar  1 10:56:20 2025

@author: Blyss Sarania
License: Apache 2.0
"""

import cv2
from tqdm import tqdm
from rich.traceback import install as install_rich_tracebacks
from ultralytics import YOLO
from blissful_tuner.video_processing_common import BlissfulVideoProcessor, setup_parser_video_common
from blissful_tuner.utils import setup_compute_context
from blissful_tuner.blissful_logger import BlissfulLogger


logger = BlissfulLogger(__name__, "#8e00ed")
install_rich_tracebacks()

parser = setup_parser_video_common(
    description="Blur faces in images/videos with LoRA for training face agnostic LoRA",
    model_help="Path to yolo face model e.g. yolov8x-face-lindevs.pt",
)
parser.add_argument(
    "--strength",
    required=False,
    type=int,
    default=51,
    help="Strength of blur. Higher values make the blur more intense. Default is 51",
)
parser.add_argument(
    "--sigmax",
    required=False,
    type=int,
    default=40,
    help="SigmaX of blur. The standard deviation in the X direction. Default is 40",
)
parser.add_argument(
    "--conf",
    type=float,
    default=0.5,
    help="Confidence threshold for face detection. Higher means less false positives, but might miss hard to detect faces. Default is 0.5",
)
args = parser.parse_args()
if args.strength % 2 == 0:
    args.strength += 1  # Kernel size must be odd
device, dtype = setup_compute_context(None, args.dtype)
vp = BlissfulVideoProcessor(device, dtype)
vp.prepare_files_and_path(args.input, args.output, "blur", args.codec, args.container, overwrite_all=args.yes)
frames, fps, width, height = vp.load_frames()
model = YOLO(args.model).to(device, dtype)
for frame in tqdm(frames):
    results = model.predict(source=frame, conf=args.conf, verbose=False)

    if len(results) > 0:
        # We only have one frame in the list, so take the first result
        dets = results[0].boxes
        if dets is not None:
            for box in dets:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                face_roi = frame[y1:y2, x1:x2]  # Blur it
                blurred_roi = cv2.GaussianBlur(face_roi, (args.strength, args.strength), args.sigmax)
                frame[y1:y2, x1:x2] = blurred_roi
    vp.write_np_or_tensor_to_png(frame)
vp.write_buffered_frames_to_output(fps=fps, keep_frames=args.keep_pngs)
