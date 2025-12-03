import argparse
from datetime import datetime
import gc
from importlib.util import find_spec
import random
import os
import time
from typing import Union

import numpy as np
import torch
import torchvision
import accelerate
from diffusers.utils.torch_utils import randn_tensor
from transformers.models.llama import LlamaModel
from tqdm import tqdm
import av
from einops import rearrange
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from PIL import Image

from musubi_tuner.hunyuan_model import vae
from musubi_tuner.hunyuan_model.text_encoder import TextEncoder
from musubi_tuner.hunyuan_model.text_encoder import PROMPT_TEMPLATE
from musubi_tuner.hunyuan_model.vae import load_vae
from musubi_tuner.hunyuan_model.models import load_transformer, get_rotary_pos_embed
from musubi_tuner.hunyuan_model.fp8_optimization import convert_fp8_linear
from musubi_tuner.modules.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from musubi_tuner.networks import lora

lycoris_available = find_spec("lycoris") is not None
if lycoris_available:
    from lycoris.kohya import create_network_from_weights

from musubi_tuner.utils import model_utils
from musubi_tuner.utils.model_utils import str_to_dtype
from musubi_tuner.utils.device_utils import clean_memory_on_device, synchronize_device
from musubi_tuner.utils.safetensors_utils import mem_eff_save_file
from musubi_tuner.dataset.image_video_dataset import load_video, resize_image_to_bucket

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_time_flag():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # # save video with av
    # container = av.open(path, "w")
    # stream = container.add_stream("libx264", rate=fps)
    # for x in outputs:
    #     frame = av.VideoFrame.from_ndarray(x, format="rgb24")
    #     packet = stream.encode(frame)
    #     container.mux(packet)
    # packet = stream.encode(None)
    # container.mux(packet)
    # container.close()

    height, width, _ = outputs[0].shape

    # create output container
    container = av.open(path, mode="w")

    # create video stream
    codec = "libx264"
    pixel_format = "yuv420p"
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = pixel_format
    stream.bit_rate = 4000000  # 4Mbit/s

    for frame_array in outputs:
        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
        packets = stream.encode(frame)
        for packet in packets:
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()


def save_images_grid(
    videos: torch.Tensor, parent_dir: str, image_name: str, rescale: bool = False, n_rows: int = 1, create_subdir=True
) -> list[str]:
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = torch.clamp(x, 0, 1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    if create_subdir:
        output_dir = os.path.join(parent_dir, image_name)
    else:
        output_dir = parent_dir

    os.makedirs(output_dir, exist_ok=True)
    image_paths = []
    for i, x in enumerate(outputs):
        image_path = os.path.join(output_dir, f"{image_name}_{i:03d}.png")
        image_paths.append(image_path)
        image = Image.fromarray(x)
        image.save(image_path)

    return image_paths


# region Encoding prompt


def encode_prompt(prompt: Union[str, list[str]], device: torch.device, num_videos_per_prompt: int, text_encoder: TextEncoder):
    r"""
    Encodes the prompt into text encoder hidden states.

    Args:
        prompt (`str` or `List[str]`):
            prompt to be encoded
        device: (`torch.device`):
            torch device
        num_videos_per_prompt (`int`):
            number of videos that should be generated per prompt
        text_encoder (TextEncoder):
            text encoder to be used for encoding the prompt
    """
    # LoRA and Textual Inversion are not supported in this script
    # negative prompt and prompt embedding are not supported in this script
    # clip_skip is not supported in this script because it is not used in the original script
    data_type = "video"  # video only, image is not supported

    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

    with torch.no_grad():
        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
    prompt_embeds = prompt_outputs.hidden_state

    attention_mask = prompt_outputs.attention_mask
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
        bs_embed, seq_len = attention_mask.shape
        attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
        attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

    prompt_embeds_dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

    if prompt_embeds.ndim == 2:
        bs_embed, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
    else:
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds, attention_mask


def encode_input_prompt(prompt: Union[str, list[str]], args, device, fp8_llm=False, accelerator=None):
    # constants
    prompt_template_video = "dit-llm-encode-video"
    prompt_template = "dit-llm-encode"
    text_encoder_dtype = torch.float16
    text_encoder_type = "llm"
    text_len = 256
    hidden_state_skip_layer = 2
    apply_final_norm = False
    reproduce = False

    text_encoder_2_type = "clipL"
    text_len_2 = 77

    num_videos = 1

    # if args.prompt_template_video is not None:
    #     crop_start = PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
    # elif args.prompt_template is not None:
    #     crop_start = PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
    # else:
    #     crop_start = 0
    crop_start = PROMPT_TEMPLATE[prompt_template_video].get("crop_start", 0)
    max_length = text_len + crop_start

    # prompt_template
    prompt_template = PROMPT_TEMPLATE[prompt_template]

    # prompt_template_video
    prompt_template_video = PROMPT_TEMPLATE[prompt_template_video]  # if args.prompt_template_video is not None else None

    # load text encoders
    logger.info(f"loading text encoder: {args.text_encoder1}")
    text_encoder = TextEncoder(
        text_encoder_type=text_encoder_type,
        max_length=max_length,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=args.text_encoder1,
        tokenizer_type=text_encoder_type,
        prompt_template=prompt_template,
        prompt_template_video=prompt_template_video,
        hidden_state_skip_layer=hidden_state_skip_layer,
        apply_final_norm=apply_final_norm,
        reproduce=reproduce,
    )
    text_encoder.eval()
    if fp8_llm:
        org_dtype = text_encoder.dtype
        logger.info(f"Moving and casting text encoder to {device} and torch.float8_e4m3fn")
        text_encoder.to(device=device, dtype=torch.float8_e4m3fn)

        # prepare LLM for fp8
        def prepare_fp8(llama_model: LlamaModel, target_dtype):
            def forward_hook(module):
                def forward(hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(variance + module.variance_epsilon)
                    return module.weight.to(input_dtype) * hidden_states.to(input_dtype)

                return forward

            for module in llama_model.modules():
                if module.__class__.__name__ in ["Embedding"]:
                    # print("set", module.__class__.__name__, "to", target_dtype)
                    module.to(target_dtype)
                if module.__class__.__name__ in ["LlamaRMSNorm"]:
                    # print("set", module.__class__.__name__, "hooks")
                    module.forward = forward_hook(module)

        prepare_fp8(text_encoder.model, org_dtype)

    logger.info(f"loading text encoder 2: {args.text_encoder2}")
    text_encoder_2 = TextEncoder(
        text_encoder_type=text_encoder_2_type,
        max_length=text_len_2,
        text_encoder_dtype=text_encoder_dtype,
        text_encoder_path=args.text_encoder2,
        tokenizer_type=text_encoder_2_type,
        reproduce=reproduce,
    )
    text_encoder_2.eval()

    # encode prompt
    logger.info("Encoding prompt with text encoder 1")
    text_encoder.to(device=device)
    if fp8_llm:
        with accelerator.autocast():
            prompt_embeds, prompt_mask = encode_prompt(prompt, device, num_videos, text_encoder)
    else:
        prompt_embeds, prompt_mask = encode_prompt(prompt, device, num_videos, text_encoder)
    text_encoder = None
    gc.collect()  # transformers==4.54.1 needs this
    clean_memory_on_device(device)

    logger.info("Encoding prompt with text encoder 2")
    text_encoder_2.to(device=device)
    prompt_embeds_2, prompt_mask_2 = encode_prompt(prompt, device, num_videos, text_encoder_2)

    prompt_embeds = prompt_embeds.to("cpu")
    prompt_mask = prompt_mask.to("cpu")
    prompt_embeds_2 = prompt_embeds_2.to("cpu")
    prompt_mask_2 = prompt_mask_2.to("cpu")

    text_encoder_2 = None
    gc.collect()
    clean_memory_on_device(device)

    return prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2


# endregion


def prepare_vae(args, device):
    vae_dtype = torch.float16 if args.vae_dtype is None else str_to_dtype(args.vae_dtype)
    vae, _, s_ratio, t_ratio = load_vae(vae_dtype=vae_dtype, device=device, vae_path=args.vae)
    vae.eval()
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}

    # set chunk_size to CausalConv3d recursively
    chunk_size = args.vae_chunk_size
    if chunk_size is not None:
        vae.set_chunk_size_for_causal_conv_3d(chunk_size)
        logger.info(f"Set chunk_size to {chunk_size} for CausalConv3d")

    if args.vae_spatial_tile_sample_min_size is not None:
        vae.enable_spatial_tiling(True)
        vae.tile_sample_min_size = args.vae_spatial_tile_sample_min_size
        vae.tile_latent_min_size = args.vae_spatial_tile_sample_min_size // 8
    # elif args.vae_tiling:
    else:
        vae.enable_spatial_tiling(True)

    return vae, vae_dtype


def encode_to_latents(args, video, device):
    vae, vae_dtype = prepare_vae(args, device)

    video = video.to(device=device, dtype=vae_dtype)
    video = video * 2 - 1  # 0, 1 -> -1, 1
    with torch.no_grad():
        latents = vae.encode(video).latent_dist.sample()

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    else:
        latents = latents * vae.config.scaling_factor

    return latents


def decode_latents(args, latents, device):
    vae, vae_dtype = prepare_vae(args, device)

    expand_temporal_dim = False
    if len(latents.shape) == 4:
        latents = latents.unsqueeze(2)
        expand_temporal_dim = True
    elif len(latents.shape) == 5:
        pass
    else:
        raise ValueError(f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

    if hasattr(vae.config, "shift_factor") and vae.config.shift_factor:
        latents = latents / vae.config.scaling_factor + vae.config.shift_factor
    else:
        latents = latents / vae.config.scaling_factor

    latents = latents.to(device=device, dtype=vae_dtype)
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    if expand_temporal_dim:
        image = image.squeeze(2)

    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.cpu().float()

    return image


def setup_parser_compile(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (requires Triton) / torch.compileを有効にする（Tritonが必要）",
    )
    parser.add_argument(
        "--compile_backend",
        type=str,
        default="inductor",
        help="torch.compile backend (default: inductor) / torch.compileのバックエンド（デフォルト: inductor）",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",  # 学習用のデフォルト
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode (default: default) / torch.compileのモード（デフォルト: default）",
    )
    parser.add_argument(
        "--compile_dynamic",
        type=str,
        default=None,
        choices=["true", "false", "auto"],
        help="Dynamic shapes mode for torch.compile (default: None, same as auto)"
        " / torch.compileの動的形状モード（デフォルト: None、autoと同じ動作）",
    )
    parser.add_argument(
        "--compile_fullgraph",
        action="store_true",
        help="Enable fullgraph mode in torch.compile / torch.compileでフルグラフモードを有効にする",
    )
    parser.add_argument(
        "--compile_cache_size_limit",
        type=int,
        default=None,
        help="Set torch._dynamo.config.cache_size_limit (default: PyTorch default, typically 8-32) / torch._dynamo.config.cache_size_limitを設定（デフォルト: PyTorchのデフォルト、通常8-32）",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint path or directory")
    parser.add_argument(
        "--dit_in_channels",
        type=int,
        default=None,
        help="input channels for DiT, default is None (automatically detect). 32 for SkyReels-I2V, 16 for others",
    )
    parser.add_argument("--vae", type=str, required=True, help="VAE checkpoint path or directory")
    parser.add_argument("--vae_dtype", type=str, default=None, help="data type for VAE, default is float16")
    parser.add_argument("--text_encoder1", type=str, required=True, help="Text Encoder 1 directory")
    parser.add_argument("--text_encoder2", type=str, required=True, help="Text Encoder 2 directory")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )
    parser.add_argument("--exclude_single_blocks", action="store_true", help="Exclude single blocks when loading LoRA weights")

    # inference
    parser.add_argument("--prompt", type=str, required=True, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for generation")
    parser.add_argument("--video_size", type=int, nargs=2, default=[256, 256], help="video size")
    parser.add_argument("--video_length", type=int, default=129, help="video length")
    parser.add_argument("--fps", type=int, default=24, help="video fps")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated video")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Guidance scale for classifier free guidance. Default is 1.0 (means no guidance)",
    )
    parser.add_argument("--embedded_cfg_scale", type=float, default=6.0, help="Embeded classifier free guidance scale.")
    parser.add_argument("--video_path", type=str, default=None, help="path to video for video2video inference")
    parser.add_argument(
        "--image_path", type=str, default=None, help="path to image for image2video inference, only works for SkyReels-I2V model"
    )
    parser.add_argument(
        "--split_uncond",
        action="store_true",
        help="split unconditional call for classifier free guidance, slower but less memory usage",
    )
    parser.add_argument("--strength", type=float, default=0.8, help="strength for video2video inference")

    # Flow Matching
    parser.add_argument("--flow_shift", type=float, default=7.0, help="Shift factor for flow matching schedulers.")

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_llm", action="store_true", help="use fp8 for Text Encoder 1 (LLM)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "xformers", "sdpa"], help="attention mode"
    )
    parser.add_argument(
        "--split_attn", action="store_true", help="use split attention, default is False. if True, --split_uncond becomes True"
    )
    parser.add_argument("--vae_chunk_size", type=int, default=None, help="chunk size for CausalConv3d in VAE")
    parser.add_argument(
        "--vae_spatial_tile_sample_min_size", type=int, default=None, help="spatial tile sample min size for VAE, default 256"
    )
    parser.add_argument("--blocks_to_swap", type=int, default=None, help="number of blocks to swap in the model")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping, which may speed up data transfer between CPU and GPU but uses more shared GPU memory on Windows",
    )
    parser.add_argument("--img_in_txt_in_offloading", action="store_true", help="offload img_in and txt_in to cpu")
    parser.add_argument(
        "--output_type", type=str, default="video", choices=["video", "images", "latent", "both"], help="output type"
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    parser.add_argument(
        "--lycoris", action="store_true", help=f"use lycoris for inference{'' if lycoris_available else ' (not available)'}"
    )
    parser.add_argument("--fp8_fast", action="store_true", help="Enable fast FP8 arthimetic(RTX 4XXX+)")
    parser.add_argument(
        "--compile_args",
        nargs=4,
        metavar=("BACKEND", "MODE", "DYNAMIC", "FULLGRAPH"),
        # default=["inductor", "max-autotune-no-cudagraphs", "False", "False"],
        default=None,
        help="[Deprecated] Torch.compile settings. Use individual args instead.",
    )
    setup_parser_compile(parser)

    args = parser.parse_args()

    assert (args.latent_path is None or len(args.latent_path) == 0) or (
        args.output_type == "images" or args.output_type == "video"
    ), "latent_path is only supported for images or video output"

    # update dit_weight based on model_base if not exists

    if args.fp8_fast and not args.fp8:
        raise ValueError("--fp8_fast requires --fp8")

    if args.lycoris and not lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    return args


def check_inputs(args):
    height = args.video_size[0]
    width = args.video_size[1]
    video_length = args.video_length

    if height % 8 != 0 or width % 8 != 0:
        raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")
    return height, width, video_length


def main():
    args = parse_args()

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dit_dtype = torch.bfloat16
    dit_weight_dtype = torch.float8_e4m3fn if args.fp8 else dit_dtype
    logger.info(f"Using device: {device}, DiT precision: {dit_dtype}, weight precision: {dit_weight_dtype}")

    original_base_names = None
    if args.latent_path is not None and len(args.latent_path) > 0:
        original_base_names = []
        latents_list = []
        seeds = []
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
        latents = torch.stack(latents_list, dim=0)
    else:
        # prepare accelerator
        mixed_precision = "bf16" if dit_dtype == torch.bfloat16 else "fp16"
        accelerator = accelerate.Accelerator(mixed_precision=mixed_precision)

        # load prompt
        prompt = args.prompt  # TODO load prompts from file
        assert prompt is not None, "prompt is required"

        # check inputs: may be height, width, video_length etc will be changed for each generation in future
        height, width, video_length = check_inputs(args)

        # encode prompt with LLM and Text Encoder
        logger.info(f"Encoding prompt: {prompt}")

        do_classifier_free_guidance = args.guidance_scale != 1.0
        if do_classifier_free_guidance:
            negative_prompt = args.negative_prompt
            if negative_prompt is None:
                logger.info("Negative prompt is not provided, using empty prompt")
                negative_prompt = ""
            logger.info(f"Encoding negative prompt: {negative_prompt}")
            prompt = [negative_prompt, prompt]
        else:
            if args.negative_prompt is not None:
                logger.warning("Negative prompt is provided but guidance_scale is 1.0, negative prompt will be ignored.")

        prompt_embeds, prompt_mask, prompt_embeds_2, prompt_mask_2 = encode_input_prompt(
            prompt, args, device, args.fp8_llm, accelerator
        )

        # encode latents for video2video inference
        video_latents = None
        if args.video_path is not None:
            # v2v inference
            logger.info(f"Video2Video inference: {args.video_path}")
            video = load_video(args.video_path, 0, video_length, bucket_reso=(width, height))  # list of frames
            if len(video) < video_length:
                raise ValueError(f"Video length is less than {video_length}")
            video = np.stack(video, axis=0)  # F, H, W, C
            video = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).float()  # 1, C, F, H, W
            video = video / 255.0

            logger.info("Encoding video to latents")
            video_latents = encode_to_latents(args, video, device)
            video_latents = video_latents.to(device=device, dtype=dit_dtype)

            clean_memory_on_device(device)

        # encode latents for image2video inference
        image_latents = None
        if args.image_path is not None:
            # i2v inference
            logger.info(f"Image2Video inference: {args.image_path}")

            image = Image.open(args.image_path)
            image = resize_image_to_bucket(image, (width, height))  # returns a numpy array
            image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float()  # 1, C, 1, H, W
            image = image / 255.0

            logger.info("Encoding image to latents")
            image_latents = encode_to_latents(args, image, device)  # 1, C, 1, H, W
            image_latents = image_latents.to(device=device, dtype=dit_dtype)

            clean_memory_on_device(device)

        # load DiT model
        blocks_to_swap = args.blocks_to_swap if args.blocks_to_swap else 0
        loading_device = "cpu"  # if blocks_to_swap > 0 else device

        logger.info(f"Loading DiT model from {args.dit}")
        if args.attn_mode == "sdpa":
            args.attn_mode = "torch"

        # if image_latents is given, the model should be I2V model, so the in_channels should be 32
        dit_in_channels = args.dit_in_channels if args.dit_in_channels is not None else (32 if image_latents is not None else 16)

        # if we use LoRA, weigths should be bf16 instead of fp8, because merging should be done in bf16
        # the model is too large, so we load the model to cpu. in addition, the .pt file is loaded to cpu anyway
        # on the fly merging will be a solution for this issue for .safetenors files (not implemented yet)
        transformer = load_transformer(
            args.dit, args.attn_mode, args.split_attn, loading_device, dit_dtype, in_channels=dit_in_channels
        )
        transformer.eval()

        # load LoRA weights
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            for i, lora_weight in enumerate(args.lora_weight):
                if args.lora_multiplier is not None and len(args.lora_multiplier) > i:
                    lora_multiplier = args.lora_multiplier[i]
                else:
                    lora_multiplier = 1.0

                logger.info(f"Loading LoRA weights from {lora_weight} with multiplier {lora_multiplier}")
                weights_sd = load_file(lora_weight)

                # Filter to exclude keys that are part of single_blocks
                if args.exclude_single_blocks:
                    filtered_weights = {k: v for k, v in weights_sd.items() if "single_blocks" not in k}
                    weights_sd = filtered_weights

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
                    network = lora.create_arch_network_from_weights(
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

        logger.info(f"Casting model to {dit_weight_dtype}")
        transformer.to(dtype=dit_weight_dtype)

        if args.fp8_fast:
            logger.info("Enabling FP8 acceleration")
            params_to_keep = {"norm", "bias", "time_in", "vector_in", "guidance_in", "txt_in", "img_in"}
            for name, param in transformer.named_parameters():
                dtype_to_use = dit_dtype if any(keyword in name for keyword in params_to_keep) else dit_weight_dtype
                param.to(dtype=dtype_to_use)
            convert_fp8_linear(transformer, dit_dtype, params_to_keep=params_to_keep)

        if blocks_to_swap > 0:
            logger.info(f"Enable swap {blocks_to_swap} blocks to CPU from device: {device}")
            transformer.enable_block_swap(
                blocks_to_swap, device, supports_backward=False, use_pinned_memory=args.use_pinned_memory_for_block_swap
            )
            transformer.move_to_device_except_swap_blocks(device)
            transformer.prepare_block_swap_before_forward()
        else:
            logger.info(f"Moving model to {device}")
            transformer.to(device=device)
        if args.img_in_txt_in_offloading:
            logger.info("Enable offloading img_in and txt_in to CPU")
            transformer.enable_img_in_txt_in_offloading()

        if args.compile:
            if args.compile_args is not None:
                # deprecated
                args.compile_backend, args.compile_mode, args.compile_dynamic, compile_fullgraph = args.compile_args
                args.compile_dynamic = args.compile_dynamic.lower()
                args.compile_fullgraph = compile_fullgraph.lower() in "true"
                args.compile_cache_size_limit = 32  # old default value

            transformer = model_utils.compile_transformer(
                args, transformer, [transformer.double_blocks, transformer.single_blocks], disable_linear=blocks_to_swap > 0
            )

        # load scheduler
        logger.info("Loading scheduler")
        scheduler = FlowMatchDiscreteScheduler(shift=args.flow_shift, reverse=True, solver="euler")

        # Prepare timesteps
        num_inference_steps = args.infer_steps
        scheduler.set_timesteps(num_inference_steps, device=device)  # n_tokens is not used in FlowMatchDiscreteScheduler
        timesteps = scheduler.timesteps

        # Prepare generator
        num_videos_per_prompt = 1  # args.num_videos # currently only support 1 video per prompt, this is a batch size
        seed = args.seed
        if seed is None:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(num_videos_per_prompt)]
        elif isinstance(seed, int):
            seeds = [seed + i for i in range(num_videos_per_prompt)]
        else:
            raise ValueError(f"Seed must be an integer or None, got {seed}.")
        generator = [torch.Generator(device).manual_seed(seed) for seed in seeds]

        # Prepare noisy latents
        num_channels_latents = 16  # transformer.config.in_channels
        vae_scale_factor = 2 ** (4 - 1)  # len(self.vae.config.block_out_channels) == 4

        vae_ver = vae.VAE_VER
        if "884" in vae_ver:
            latent_video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            latent_video_length = (video_length - 1) // 8 + 1
        else:
            latent_video_length = video_length

        # shape = (
        #     num_videos_per_prompt,
        #     num_channels_latents,
        #     latent_video_length,
        #     height // vae_scale_factor,
        #     width // vae_scale_factor,
        # )
        # latents = randn_tensor(shape, generator=generator, device=device, dtype=dit_dtype)

        # make first N frames to be the same if the given seed is same
        shape_of_frame = (num_videos_per_prompt, num_channels_latents, 1, height // vae_scale_factor, width // vae_scale_factor)
        latents = []
        for i in range(latent_video_length):
            latents.append(randn_tensor(shape_of_frame, generator=generator, device=device, dtype=dit_dtype))
        latents = torch.cat(latents, dim=2)

        # pad image_latents to match the length of video_latents
        if image_latents is not None:
            zero_latents = torch.zeros_like(latents)
            zero_latents[:, :, :1, :, :] = image_latents
            image_latents = zero_latents

        if args.video_path is not None:
            # v2v inference
            noise = latents
            assert noise.shape == video_latents.shape, f"noise shape {noise.shape} != video_latents shape {video_latents.shape}"

            num_inference_steps = int(num_inference_steps * args.strength)
            timestep_start = scheduler.timesteps[-num_inference_steps]  # larger strength, less inference steps and more start time
            t = timestep_start / 1000.0
            latents = noise * t + video_latents * (1 - t)

            timesteps = timesteps[-num_inference_steps:]

            logger.info(f"strength: {args.strength}, num_inference_steps: {num_inference_steps}, timestep_start: {timestep_start}")

        # FlowMatchDiscreteScheduler does not have init_noise_sigma

        # Denoising loop
        embedded_guidance_scale = args.embedded_cfg_scale
        if embedded_guidance_scale is not None:
            guidance_expand = torch.tensor([embedded_guidance_scale * 1000.0] * latents.shape[0], dtype=torch.float32, device="cpu")
            guidance_expand = guidance_expand.to(device=device, dtype=dit_dtype)
            if do_classifier_free_guidance:
                guidance_expand = torch.cat([guidance_expand, guidance_expand], dim=0)
        else:
            guidance_expand = None
        freqs_cos, freqs_sin = get_rotary_pos_embed(vae_ver, transformer, video_length, height, width)
        # n_tokens = freqs_cos.shape[0]

        # move and cast all inputs to the correct device and dtype
        prompt_embeds = prompt_embeds.to(device=device, dtype=dit_dtype)
        prompt_mask = prompt_mask.to(device=device)
        prompt_embeds_2 = prompt_embeds_2.to(device=device, dtype=dit_dtype)
        prompt_mask_2 = prompt_mask_2.to(device=device)

        freqs_cos = freqs_cos.to(device=device, dtype=dit_dtype)
        freqs_sin = freqs_sin.to(device=device, dtype=dit_dtype)

        num_warmup_steps = len(timesteps) - num_inference_steps * scheduler.order  # this should be 0 in v2v inference

        # assert split_uncond and split_attn
        if args.split_attn and do_classifier_free_guidance and not args.split_uncond:
            logger.warning("split_attn is enabled, split_uncond will be enabled as well.")
            args.split_uncond = True

        # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as p:
        with tqdm(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latents = scheduler.scale_model_input(latents, t)

                # predict the noise residual
                with torch.no_grad(), accelerator.autocast():
                    latents_input = latents if not do_classifier_free_guidance else torch.cat([latents, latents], dim=0)
                    if image_latents is not None:
                        latents_image_input = (
                            image_latents if not do_classifier_free_guidance else torch.cat([image_latents, image_latents], dim=0)
                        )
                        latents_input = torch.cat([latents_input, latents_image_input], dim=1)  # 1 or 2, C*2, F, H, W

                    batch_size = 1 if args.split_uncond else latents_input.shape[0]

                    noise_pred_list = []
                    for j in range(0, latents_input.shape[0], batch_size):
                        noise_pred = transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                            latents_input[j : j + batch_size],  # [1, 16, 33, 24, 42]
                            t.repeat(batch_size).to(device=device, dtype=dit_dtype),  # [1]
                            text_states=prompt_embeds[j : j + batch_size],  # [1, 256, 4096]
                            text_mask=prompt_mask[j : j + batch_size],  # [1, 256]
                            text_states_2=prompt_embeds_2[j : j + batch_size],  # [1, 768]
                            freqs_cos=freqs_cos,  # [seqlen, head_dim]
                            freqs_sin=freqs_sin,  # [seqlen, head_dim]
                            guidance=guidance_expand[j : j + batch_size],  # [1]
                            return_dict=True,
                        )["x"]
                        noise_pred_list.append(noise_pred)
                    noise_pred = torch.cat(noise_pred_list, dim=0)

                # perform classifier free guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                    # # SkyReels' rescale noise config is omitted for now
                    # if guidance_rescale > 0.0:
                    #     # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    #     noise_pred = rescale_noise_cfg(
                    #         noise_pred,
                    #         noise_pred_cond,
                    #         guidance_rescale=self.guidance_rescale,
                    #     )

                # compute the previous noisy sample x_t -> x_t-1
                latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % scheduler.order == 0):
                    if progress_bar is not None:
                        progress_bar.update()

        # print(p.key_averages().table(sort_by="self_cpu_time_total", row_limit=-1))
        # print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))

        latents = latents.detach().cpu()
        transformer = None
        clean_memory_on_device(device)

    # Save samples
    output_type = args.output_type
    save_path = args.save_path  # if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    if output_type == "latent" or output_type == "both":
        # save latent
        for i, latent in enumerate(latents):
            latent_path = f"{save_path}/{time_flag}_{i}_{seeds[i]}_latent.safetensors"

            if args.no_metadata:
                metadata = None
            else:
                metadata = {
                    "seeds": f"{seeds[i]}",
                    "prompt": f"{args.prompt}",
                    "height": f"{height}",
                    "width": f"{width}",
                    "video_length": f"{video_length}",
                    "infer_steps": f"{num_inference_steps}",
                    "guidance_scale": f"{args.guidance_scale}",
                    "embedded_cfg_scale": f"{args.embedded_cfg_scale}",
                }
                if args.negative_prompt is not None:
                    metadata["negative_prompt"] = f"{args.negative_prompt}"
            sd = {"latent": latent}
            save_file(sd, latent_path, metadata=metadata)

            logger.info(f"Latent save to: {latent_path}")
    if output_type == "video" or output_type == "both":
        # save video
        videos = decode_latents(args, latents, device)
        for i, sample in enumerate(videos):
            original_name = "" if original_base_names is None else f"_{original_base_names[i]}"
            sample = sample.unsqueeze(0)
            video_path = f"{save_path}/{time_flag}_{i}_{seeds[i]}{original_name}.mp4"
            save_videos_grid(sample, video_path, fps=args.fps)
            logger.info(f"Sample save to: {video_path}")
    elif output_type == "images":
        # save images
        videos = decode_latents(args, latents, device)
        for i, sample in enumerate(videos):
            original_name = "" if original_base_names is None else f"_{original_base_names[i]}"
            sample = sample.unsqueeze(0)
            image_name = f"{time_flag}_{i}_{seeds[i]}{original_name}"
            save_images_grid(sample, save_path, image_name)
            logger.info(f"Sample images save to: {save_path}/{image_name}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
