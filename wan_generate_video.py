import argparse
from datetime import datetime
import random
import os
import time

import torch
import accelerate
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image

from networks import lora_wan
from utils.safetensors_utils import mem_eff_save_file
from wan.configs import WAN_CONFIGS, SUPPORTED_SIZES
import wan
from wan.modules.vae import WanVAE

try:
    from lycoris.kohya import create_network_from_weights
except:
    pass

from utils.model_utils import str_to_dtype
from utils.device_utils import clean_memory_on_device
from hv_generate_video import save_images_grid, save_videos_grid, synchronize_device

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Wan 2.1 inference script")

    # WAN arguments
    parser.add_argument("--ckpt_dir", type=str, default=None, help="The path to the checkpoint directory (Wan 2.1 official).")
    parser.add_argument("--task", type=str, default="t2v-14B", choices=list(WAN_CONFIGS.keys()), help="The task to run.")
    # parser.add_argument("--use_prompt_extend", action="store_true", default=False, help="Whether to use prompt extend.")
    # prompt extend is not supported
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "vanilla"], help="The solver used to sample."
    )

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is bfloat16")
    parser.add_argument("--vae_cache_cpu", action="store_true", help="cache features in VAE on CPU")
    parser.add_argument("--t5", type=str, default=None, help="text encoder (T5) checkpoint path")
    parser.add_argument("--clip", type=str, default=None, help="text encoder (CLIP) checkpoint path")
    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="negative prompt for generation, use default negative prompt if not specified",
    )
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size, height and width")
    parser.add_argument("--video_length", type=int, default=None, help="video length, Default is 81 for video inference")
    parser.add_argument("--fps", type=int, default=16, help="video fps, Default is 16")
    parser.add_argument("--infer_steps", type=int, default=None, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=5.0,
        help="Guidance scale for classifier free guidance. Default is 5.0.",
    )
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument("--image_path", type=str, default=None, help="path to image for image2video inference")
    # parser.add_argument(
    #     "--split_uncond",
    #     action="store_true",
    #     help="split unconditional call for classifier free guidance, slower but less memory usage",
    # )
    # parser.add_argument("--strength", type=float, default=0.8, help="strength for video2video inference")

    # Flow Matching
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=None,
        help="Shift factor for flow matching schedulers. Default is 3.0 for I2V with 832*480, 5.0 for others.",
    )

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_t5", action="store_true", help="use fp8 for Text Encoder model")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "flash2", "flash3", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    # parser.add_argument(
    #     "--split_attn", action="store_true", help="use split attention, default is False. if True, --split_uncond becomes True"
    # )
    # parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    # parser.add_argument(
    #     "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    # )
    parser.add_argument("--blocks_to_swap", type=int, default=None, help="number of blocks to swap in the model")
    # parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="offload img_in and txt_in to cpu")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    return args


def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    size = f"{width}*{height}"

    # assert (
    #     size in SUPPORTED_SIZES[args.task]
    # ), f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}."
    if size not in SUPPORTED_SIZES[args.task]:
        logger.warning(f"Size {size} is not supported for task {args.task}. Supported sizes are {SUPPORTED_SIZES[args.task]}.")
    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    return height, width, video_length


def main():
    args = parse_args()

    # validate args
    if args.video_length is None:
        args.video_length = 1 if "t2i" in args.task else 81
    if "t2i" in args.task:
        assert args.video_length == 1, f"video_length should be 1 for task {args.task}"

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0
    if not latents_mode:
        # check inputs: may be height, width, video_length etc will be changed for each generation in future
        height, width, video_length = check_inputs(args)
        size = (width, height)
    else:
        height, width, video_length = None, None, None
        size = None

    if args.infer_steps is None:
        args.infer_steps = 40 if "i2v" in args.task else 50
    if args.flow_shift is None:
        args.flow_shift = 3.0 if "i2v" in args.task and (width == 832 and height == 480 or width == 480 and height == 832) else 5.0

    print(
        f"video size: {height}x{width}@{video_length} (HxW@F), fps: {args.fps}, infer_steps: {args.infer_steps}, flow_shift: {args.flow_shift}"
    )

    cfg = WAN_CONFIGS[args.task]

    # prepare device and dtype
    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dit_dtype = torch.bfloat16
    dit_weight_dtype = torch.float8_e4m3fn if args.fp8 else dit_dtype
    vae_dtype = str_to_dtype(args.vae_dtype) if args.vae_dtype is not None else dit_dtype
    logger.info(
        f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}, VAE precision: {vae_dtype}"
    )

    def load_vae():
        vae_path = args.vae if args.vae is not None else os.path.join(args.ckpt_dir, cfg.vae_checkpoint)

        logger.info(f"Loading VAE model from {vae_path}")
        cache_device = torch.device("cpu") if args.vae_cache_cpu else None
        vae = WanVAE(vae_path=vae_path, device=device, dtype=vae_dtype, cache_device=cache_device)
        return vae

    vae = None

    original_base_names = None
    if latents_mode:
        original_base_names = []
        latents_list = []
        seeds = []
        assert len(args.latent_path) == 1, "Only one latent path is supported for now"
        for latent_path in args.latent_path:
            original_base_names.append(os.path.splitext(os.path.basename(latent_path))[0])
            seed = 0

            if os.path.splitext(latent_path)[1] != ".safetensors":
                latents = torch.load(latent_path, map_location="cpu")
            else:
                latents = load_file(latent_path)["latent"]
                with safe_open(latent_path, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is None:
                    metadata = {}
                logger.info(f"Loaded metadata: {metadata}")

                if "seeds" in metadata:
                    seed = int(metadata["seeds"])

            seeds.append(seed)
            latents_list.append(latents)

            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")
        latents = torch.stack(latents_list, dim=0)  # [N, ...]
    else:
        # prepare accelerator
        mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
        accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

        # load prompt
        prompt = args.prompt  # TODO load prompts from file
        assert prompt is not None, "prompt is required"

        seed = args.seed
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0

        # load LoRA weights
        merge_lora = None
        if args.lora_weight is not None and len(args.lora_weight) > 0:

            def merge_lora(transformer):
                for i, lora_weight in enumerate(args.lora_weight):
                    if type(args.lora_multiplier) == list:
                        if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
                            lora_multiplier = args.lora_multiplier[i]
                        else:
                            lora_multiplier = 1.0
                    else:
                        lora_multiplier = args.lora_multiplier

                    logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
                    weights_sd = load_file(lora_weight)
                    if args.lycoris:
                        lycoris_net, _ = create_network_from_weights(
                            multiplier=lora_multiplier,
                            file=None,
                            weights_sd=weights_sd,
                            unet=transformer,
                            text_encoder=None,
                            vae=None,
                            for_inference=True,
                        )
                    else:
                        network = lora_wan.create_arch_network_from_weights(
                            lora_multiplier, weights_sd, unet=transformer, for_inference=True
                        )
                    logger.info("Merging LoRA weights to DiT model")

                    # try:
                    #     network.apply_to(None, transformer, apply_text_encoder=False, apply_unet=True)
                    #     info = network.load_state_dict(weights_sd, strict=True)
                    #     logger.info(f"Loaded LoRA weights from {weights_file}: {info}")
                    #     network.eval()
                    #     network.to(device)
                    # except Exception as e:
                    if args.lycoris:
                        lycoris_net.merge_to(None, transformer, weights_sd, dtype=None, device=device)
                    else:
                        network.merge_to(None, transformer, weights_sd, device=device, non_blocking=True)

                    synchronize_device(device)

                    logger.info("LoRA weights loaded")

                # save model here before casting to dit_weight_dtype
                if args.save_merged_model:
                    logger.info(f"Saving merged model to {args.save_merged_model}")
                    mem_eff_save_file(transformer.state_dict(), args.save_merged_model)  # save_file needs a lot of memory
                    logger.info("Merged model saved")
                    return

        # create pipeline
        if "t2v" in args.task or "t2i" in args.task:
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device=device,
                dtype=dit_weight_dtype,
                dit_path=args.dit,
                dit_attn_mode=args.attn_mode,
                t5_path=args.t5,
                t5_fp8=args.fp8_t5,
            )

            logging.info(f"Generating {'image' if 't2i' in args.task else 'video'} ...")
            latents = wan_t2v.generate(
                accelerator,
                merge_lora,
                torch.bfloat16 if merge_lora is not None else None,
                prompt,
                size=size,
                frame_num=video_length,
                shift=args.flow_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.infer_steps,
                guide_scale=args.guidance_scale,
                seed=seed,
                blocks_to_swap=blocks_to_swap,
            )
            latents = latents.unsqueeze(0)
            del wan_t2v
        elif "i2v" in args.task:
            wan_i2v = wan.WanI2V(
                config=cfg,
                checkpoint_dir=args.ckpt_dir,
                device=device,
                dtype=dit_weight_dtype,
                dit_path=args.dit,
                dit_attn_mode=args.attn_mode,
                t5_path=args.t5,
                clip_path=args.clip,
                t5_fp8=args.fp8_t5,
            )

            # i2v inference
            logger.info(f"Image2Video inference: {args.image_path}")
            image = Image.open(args.image_path).convert("RGB")

            vae = load_vae()

            logging.info(f"Generating video ...")
            latents = wan_i2v.generate(
                accelerator,
                merge_lora,
                torch.bfloat16 if merge_lora is not None else None,
                prompt,
                img=image,
                size=size,
                frame_num=video_length,
                shift=args.flow_shift,
                sample_solver=args.sample_solver,
                sampling_steps=args.infer_steps,
                guide_scale=args.guidance_scale,
                seed=seed,
                blocks_to_swap=blocks_to_swap,
                vae=vae,
            )
            del wan_i2v
            latents = latents.unsqueeze(0)

        logger.info(f"wait for 5s to clean memory")
        time.sleep(5.0)
        clean_memory_on_device(device)

    # prepare accelerator for decode
    output_type = args.output_type

    def decode_latents(x0):
        nonlocal vae
        if vae is None:
            vae = load_vae()
        vae.to_device(device)

        logger.info(f"Decoding video from latents: {x0.shape}")
        x0 = x0.to(device)  # , dtype=vae_dtype)
        # with accelerator.autocast(), torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=vae_dtype), torch.no_grad():
            videos = vae.decode(x0)
        logger.info(f"Decoding complete")
        video = videos[0]
        del videos
        video = video.to(torch.float32).cpu()
        return video

    # Save samples
    save_path = args.save_path  # if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)
    time_flag = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

    if output_type == "latent" or output_type == "both":
        # save latent
        latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

        if args.no_metadata:
            metadata = None
        else:
            metadata = {
                "seeds": f"{seed}",
                "prompt": f"{args.prompt}",
                "height": f"{height}",
                "width": f"{width}",
                "video_length": f"{video_length}",
                "infer_steps": f"{args.infer_steps}",
                "guidance_scale": f"{args.guidance_scale}",
                # "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
            }
            if args.negative_prompt is not None:
                metadata["negative_prompt"] = f"{args.negative_prompt}"
        sd = {"latent": latents[0]}
        save_file(sd, latent_path, metadata=metadata)

        logger.info(f"Latent save to: {latent_path}")
    if output_type == "video" or output_type == "both":
        # save video
        sample = decode_latents(latents)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        video_path = f"{save_path}/{time_flag}_{seed}{original_name}.mp4"
        save_videos_grid(sample, video_path, fps=args.fps, rescale=True)
        logger.info(f"Sample save to: {video_path}")
    elif output_type == "images":
        # save images
        sample = decode_latents(latents)
        original_name = "" if original_base_names is None else f"_{original_base_names[0]}"
        sample = sample.unsqueeze(0)
        image_name = f"{time_flag}_{seed}{original_name}"
        save_images_grid(sample, save_path, image_name, rescale=True)
        logger.info(f"Sample images save to: {save_path}/{image_name}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
