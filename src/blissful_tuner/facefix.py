#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face restoration for Blissful Tuner Extension

License: Apache 2.0
Created on Wed Apr 23 10:19:19 2025
@author: blyss
"""

import sys
import types
import warnings

# Patch basicsr expecting torchvision.transforms.functional_tensor
try:
    import torchvision.transforms.functional as F

    # Create a dummy module to stand in for functional_tensor
    functional_tensor = types.ModuleType("functional_tensor")
    functional_tensor.rgb_to_grayscale = F.rgb_to_grayscale

    # Insert into sys.modules so basicsr can find it
    sys.modules["torchvision.transforms.functional_tensor"] = functional_tensor
except ImportError:
    warnings.warn("Failed to monkeypatch torchvision for basicsr. Face restoration may fail.", stacklevel=1)

from rich.traceback import install as install_rich_tracebacks
from tqdm import tqdm
import torch
from torchvision.transforms.functional import normalize
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from blissful_tuner.codeformer.basicsr.utils.registry import ARCH_REGISTRY
from blissful_tuner.codeformer.basicsr.utils import img2tensor, tensor2img
from blissful_tuner.gfpgan import GFPGANer
from blissful_tuner.video_processing_common import BlissfulVideoProcessor, setup_parser_video_common, get_media_input_list
from blissful_tuner.utils import power_seed
from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "#8e00ed")
install_rich_tracebacks()


def main():
    parser = setup_parser_video_common(description="Restore faces with GFPGAN or CODEFORMER")
    parser.add_argument("--only_center", action="store_true", help="Only process center face")
    parser.add_argument("--weight", type=float, default=0.5, help="Strength of GFPGAN or CodeFormer power")
    parser.add_argument("-s", "--upscale", type=float, default=1, help="The final upsampling scale of the image. Default: 1")
    parser.add_argument(
        "--detection_model", type=str, default="retinaface_resnet50", help="Face detector. Default: retinaface_resnet50"
    )
    parser.add_argument(
        "--mode", choices=["gfpgan", "codeformer"], type=str, default="gfpgan", help="Mode - either gfpgan or codeformer"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = parser.parse_args()
    master_input = get_media_input_list(args.input, ignore_prompts=args.yes)
    if args.mode.lower() == "gfpgan":
        restorer = GFPGANer(model_path=args.model, upscale=args.upscale, arch="clean", channel_multiplier=2, bg_upsampler=None)
    elif args.mode.lower() == "codeformer":
        net = ARCH_REGISTRY.get("CodeFormer")(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=["32", "64", "128", "256"]
        ).to(device)
        checkpoint = torch.load(args.model)["params_ema"]
        net.load_state_dict(checkpoint)
        net.eval()

        face_helper = FaceRestoreHelper(
            args.upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=args.detection_model,
            save_ext="png",
            use_parse=True,
            device=device,
        )
    for input_file in master_input:
        logger.info(f"Processing FaceRestore for {input_file}")
        VideoProcessor = BlissfulVideoProcessor(device, torch.float32)
        VideoProcessor.prepare_files_and_path(input_file, args.output, args.mode.upper(), overwrite_all=args.yes)
        frames, fps, _, _ = VideoProcessor.load_frames()
        power_seed(args.seed)
        if args.mode.lower() == "gfpgan":
            # ------------------------ restore ------------------------
            for frame in tqdm(frames):
                # restore faces and background if necessary
                _, _, restored_frame = restorer.enhance(
                    frame, has_aligned=False, only_center_face=args.only_center, paste_back=True, weight=args.weight
                )
                VideoProcessor.write_np_or_tensor_to_png(restored_frame)
                del restored_frame
        elif args.mode.lower() == "codeformer":
            for frame in tqdm(frames):
                # clean all the intermediate results to process the next image
                face_helper.clean_all()
                face_helper.read_image(frame)
                # get face landmarks for each face
                _ = face_helper.get_face_landmarks_5(only_center_face=args.only_center, resize=640, eye_dist_threshold=5)
                # align and warp each face
                face_helper.align_warp_face()
                # face restoration for each cropped face
                for cropped_face in face_helper.cropped_faces:
                    # prepare data
                    cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
                    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

                    try:
                        with torch.no_grad():
                            output = net(cropped_face_t, w=args.weight, adain=True)[0]
                            restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                        del output
                        torch.cuda.empty_cache()
                    except Exception as error:
                        logger.warning(f"CodeFormer inference failed: {error}. Using original face.")
                        restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

                    restored_face = restored_face.astype("uint8")
                    face_helper.add_restored_face(restored_face)

                face_helper.get_inverse_affine(None)
                restored_img = face_helper.paste_faces_to_input_image()
                VideoProcessor.write_np_or_tensor_to_png(restored_img)
                del restored_img

        VideoProcessor.write_buffered_frames_to_output(fps, args.keep_pngs)
        del VideoProcessor


if __name__ == "__main__":
    main()
