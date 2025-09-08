#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple video frames -> png converter
Apache 2.0
Created on Mon May  5 14:21:21 2025

@author: blyss
"""

import os
import argparse
from rich_argparse import RichHelpFormatter
from video_processing_common import BlissfulVideoProcessor

parser = argparse.ArgumentParser(
    description="Extract N frames from a video and save to png in specified directory", formatter_class=RichHelpFormatter
)
parser.add_argument("--input", required=True, help="Input video to process")
parser.add_argument(
    "--output", type=str, default=None, help="Output path for directory that will contain PNGs. Default is same path as input"
)
parser.add_argument("--num_frames", type=int, default=None, help="Number of frames to extract, default is all of them")
args = parser.parse_args()
if args.output is None:
    args.output = os.path.dirname(args.input)
vp = BlissfulVideoProcessor(will_write_video=False)
vp.prepare_files_and_path(args.input, args.output)
frames, _, _, _ = vp.load_frames()
num_do = len(frames) if args.num_frames is None else args.num_frames
num_do = len(frames) if num_do > len(frames) else num_do
for i in range(num_do):
    vp.write_np_or_tensor_to_png(frames[i])
