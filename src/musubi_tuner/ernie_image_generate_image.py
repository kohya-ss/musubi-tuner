import argparse
import gc
import random
import os
import time
import copy
from typing import Tuple, Optional, List, Any, Dict

import torch
from safetensors.torch import load_file, save_file
from safetensors import safe_open
from tqdm import tqdm

from musubi_tuner.utils import model_utils
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.ernie_image import ernie_image_model, ernie_image_utils
from musubi_tuner.flux_2 import flux2_utils, flux2_models
from musubi_tuner.hv_generate_video import get_time_flag, save_images_grid, setup_parser_compile, synchronize_device

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ERNIE-Image VAE scale: 8x conv downscale + 2x2 patchify = 16 total
ERNIE_IMAGE_VAE_SCALE = 16


class GenerationSettings:
    def __init__(self, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None):
        self.device = device
        self.dit_weight_dtype = dit_weight_dtype


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ERNIE-Image inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument(
        "--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors. Default is False."
    )
    parser.add_argument("--vae", type=str, default=None, help="VAE (FLUX.2 AutoEncoder) directory or path")
    parser.add_argument("--text_encoder", type=str, required=True, help="Mistral3 text encoder directory or path")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path (defaults to 'baidu/ERNIE-Image')")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument(
        "--save_merged_model",
        type=str,
        default=None,
        help="Save merged model to path. If specified, no inference will be performed.",
    )

    # inference
    parser.add_argument(
        "--cpu_noise", action="store_true", help="Use CPU to generate noise. Default is False."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="Guidance scale for classifier free guidance. Default is 4.0.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for generation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=4.0,
        help="flow matching timestep shift (official scheduler_config.json uses 4.0). 1.0 = no shift.",
    )
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated image(s)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    parser.add_argument("--fp8", action="store_true", help="use fp8 for DiT model")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT, only for fp8")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="use fp8 for Text Encoder (Mistral3)")
    parser.add_argument("--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoder (Mistral3)")
    parser.add_argument(
        "--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap in the model")
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned memory for block swapping",
    )
    parser.add_argument(
        "--output_type",
        type=str,
        default="images",
        choices=["images", "latent", "latent_images"],
        help="output type",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")
    setup_parser_compile(parser)

    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")
    parser.add_argument(
        "--bell",
        action="store_true",
        help="Ring bell when done. For interactive mode, ring bell on each iteration. For other modes, ring bell at the end.",
    )

    args = parser.parse_args()

    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")

    return args


def parse_prompt_line(line: str) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()

    overrides = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    args_copy = copy.deepcopy(args)
    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)
    return args_copy


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    height = args.image_size[0]
    width = args.image_size[1]

    if height % ERNIE_IMAGE_VAE_SCALE != 0 or width % ERNIE_IMAGE_VAE_SCALE != 0:
        raise ValueError(f"`height` and `width` have to be divisible by {ERNIE_IMAGE_VAE_SCALE} but are {height} and {width}.")

    return height, width


# region DiT model


def load_dit_model(
    args: argparse.Namespace, device: torch.device, dit_weight_dtype: Optional[torch.dtype] = None
) -> ernie_image_model.ErnieImageTransformer2DModel:
    loading_device = "cpu"
    if args.blocks_to_swap == 0:
        loading_device = device

    # load LoRA weights
    if args.lora_weight is not None and len(args.lora_weight) > 0:
        lora_weights_list = []
        for lora_weight in args.lora_weight:
            logger.info(f"Loading LoRA weight from: {lora_weight}")
            lora_sd = load_file(lora_weight)
            lora_sd = filter_lora_state_dict(lora_sd, args.include_patterns, args.exclude_patterns)
            lora_weights_list.append(lora_sd)
    else:
        lora_weights_list = None

    loading_weight_dtype = dit_weight_dtype
    if args.fp8_scaled:
        loading_weight_dtype = None  # load as-is and then optimize to fp8

    model = ernie_image_utils.load_dit(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=loading_device,
        dit_weight_dtype=loading_weight_dtype,
        fp8_scaled=args.fp8_scaled,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
        disable_numpy_memmap=args.disable_numpy_memmap,
    )

    if args.save_merged_model:
        return None

    if not args.fp8_scaled:
        target_dtype = None
        target_device = None

        if dit_weight_dtype is not None:
            logger.info(f"Convert model to {dit_weight_dtype}")
            target_dtype = dit_weight_dtype

        if args.blocks_to_swap == 0:
            logger.info(f"Move model to device: {device}")
            target_device = device

        model.to(target_device, target_dtype)

    if args.blocks_to_swap > 0:
        logger.info(f"Enable swap {args.blocks_to_swap} blocks to CPU from device: {device}")
        model.enable_block_swap(
            args.blocks_to_swap, device, supports_backward=False, use_pinned_memory=args.use_pinned_memory_for_block_swap
        )
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        model.to(device)

    if args.compile:
        model = model_utils.compile_transformer(
            args, model, [model.layers], disable_linear=args.blocks_to_swap > 0
        )

    model.eval().requires_grad_(False)
    clean_memory_on_device(device)

    return model


# endregion


def decode_latent(vae: flux2_models.AutoEncoder, latent: torch.Tensor, device: torch.device) -> torch.Tensor:
    logger.info(f"Decoding image. Latent shape {latent.shape}, device {device}")
    if latent.ndim == 3:  # CHW
        latent = latent.unsqueeze(0)

    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode(latent.to(device, dtype=vae.dtype))
    pixels = pixels.to("cpu", dtype=torch.float32)
    vae.to("cpu")

    logger.info(f"Decoded. Pixel shape {pixels.shape}")
    return pixels[0]  # remove batch dimension


def prepare_text_inputs(
    args: argparse.Namespace, device: torch.device, shared_models: Optional[Dict] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Encode prompts with Mistral3 text encoder."""

    conds_cache = {}
    llm_device = torch.device("cpu") if args.text_encoder_cpu else device
    if shared_models is not None:
        tokenizer = shared_models.get("tokenizer")
        text_encoder = shared_models.get("text_encoder")
        if "conds_cache" in shared_models:
            conds_cache = shared_models["conds_cache"]
    else:
        te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
        tokenizer, text_encoder = ernie_image_utils.load_text_encoder(
            args.text_encoder, dtype=te_dtype, device=llm_device, disable_mmap=True, tokenizer_id=args.tokenizer
        )

    text_encoder_original_device = text_encoder.device if text_encoder else None

    if not text_encoder or not tokenizer:
        raise ValueError("Text encoder or tokenizer is not loaded properly.")

    model_is_moved = False

    def move_models_to_device_if_needed():
        nonlocal model_is_moved
        if model_is_moved:
            return
        model_is_moved = True

        logger.info(f"Moving DiT and Text Encoder to appropriate device: {device} or CPU")
        if shared_models and "model" in shared_models:
            if args.blocks_to_swap > 0:
                logger.info("Waiting for 5 seconds to finish block swap")
                time.sleep(5)
            model = shared_models["model"]
            model.to("cpu")
            clean_memory_on_device(device)

        text_encoder.to(llm_device)

    logger.info("Encoding prompt with Mistral3 text encoder.")

    def encode(prompt: str) -> torch.Tensor:
        if prompt in conds_cache:
            return conds_cache[prompt]
        move_models_to_device_if_needed()
        text_hiddens = ernie_image_utils.encode_text(tokenizer, text_encoder, prompt)
        embed = text_hiddens[0].cpu()  # [T, D]
        conds_cache[prompt] = embed
        return embed

    prompt = args.prompt
    embed = encode(prompt)

    negative_prompt = args.negative_prompt
    if negative_prompt is None:
        negative_prompt = ""
    negative_embed = encode(negative_prompt)

    if not (shared_models and "text_encoder" in shared_models):
        del tokenizer, text_encoder
        gc.collect()
    else:
        if text_encoder:
            text_encoder.to(text_encoder_original_device)

    clean_memory_on_device(device)

    arg_c = {"embed": embed, "prompt": prompt}
    arg_null = {"embed": negative_embed, "prompt": negative_prompt}

    return arg_c, arg_null


def generate(
    args: argparse.Namespace,
    gen_settings: GenerationSettings,
    shared_models: Optional[Dict] = None,
    precomputed_text_data: Optional[Tuple[Dict, Dict]] = None,
) -> torch.Tensor:
    device, dit_weight_dtype = (gen_settings.device, gen_settings.dit_weight_dtype)

    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed

    if precomputed_text_data is not None:
        logger.info("Using precomputed text data.")
        context, context_null = precomputed_text_data
    else:
        logger.info("No precomputed data. Preparing text inputs.")
        context, context_null = prepare_text_inputs(args, device, shared_models)

    if shared_models is None or "model" not in shared_models:
        model = load_dit_model(args, device, dit_weight_dtype)
        if args.save_merged_model:
            return None
        if shared_models is not None:
            shared_models["model"] = model
    else:
        model: ernie_image_model.ErnieImageTransformer2DModel = shared_models["model"]
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()

    seed_g = torch.Generator(device="cpu" if args.cpu_noise else device)
    seed_g.manual_seed(seed)

    height, width = check_inputs(args)
    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}")
    logger.info(f"Prompt: {context['prompt']}")
    logger.info(f"Negative prompt: {context_null['prompt']}")

    # Prepare text embeddings
    embed = context["embed"].to(device, dtype=torch.bfloat16)
    if embed.dim() == 2:
        embed = embed.unsqueeze(0)  # [T, D] -> [1, T, D]
    negative_embed = context_null["embed"].to(device, dtype=torch.bfloat16)
    if negative_embed.dim() == 2:
        negative_embed = negative_embed.unsqueeze(0)

    do_cfg = args.guidance_scale > 1.0

    # Prepare latents: [1, 128, H/16, W/16]
    latent_h = height // ERNIE_IMAGE_VAE_SCALE
    latent_w = width // ERNIE_IMAGE_VAE_SCALE
    shape = (1, model.in_channels, latent_h, latent_w)
    latents = torch.randn(
        shape, generator=seed_g, device="cpu" if args.cpu_noise else device, dtype=torch.float32
    ).to(device)

    # Pad text for batch (CFG doubles batch)
    if do_cfg:
        text_hiddens_list = [negative_embed[0], embed[0]]  # [uncond, cond]
    else:
        text_hiddens_list = [embed[0]]

    text_bth, text_lens = ernie_image_utils.pad_text(
        text_hiddens_list, device, torch.bfloat16, model.text_in_dim
    )

    # Flow-matching sigmas: linspace(1.0, 0.0, N+1) with optional shift remap
    sigmas = ernie_image_utils.get_sigmas(args.infer_steps, device, shift=args.flow_shift)

    with tqdm(total=args.infer_steps, desc="Denoising steps") as pbar:
        for i in range(args.infer_steps):
            t = sigmas[i]

            # Model expects timestep in [0, 1000] range (matches training's t * 1000 and
            # Diffusers' FlowMatchEulerDiscreteScheduler which scales sigmas by num_train_timesteps).
            t_scaled = t.item() * 1000.0
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                t_batch = torch.full((2,), t_scaled, device=device, dtype=torch.bfloat16)
            else:
                latent_model_input = latents
                t_batch = torch.full((1,), t_scaled, device=device, dtype=torch.bfloat16)

            latent_model_input = latent_model_input.to(dtype=torch.bfloat16)

            with torch.no_grad():
                pred = model(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                )

            if do_cfg:
                pred_uncond, pred_cond = pred.chunk(2, dim=0)
                pred = pred_uncond + args.guidance_scale * (pred_cond - pred_uncond)

            # Euler step. ERNIE-Image follows standard flow matching (pred ≈ noise - latents = dx/dsigma),
            # so x_new = x + dt * pred, with dt = sigma_next - sigma < 0.
            dt = sigmas[i + 1] - sigmas[i]
            latents = latents + pred.to(torch.float32) * dt

            pbar.update(1)

    if shared_models is None:
        del model
        synchronize_device(device)
        if args.blocks_to_swap > 0:
            logger.info("Waiting for 5 seconds to finish block swap")
            time.sleep(5)
        gc.collect()
        clean_memory_on_device(device)

    return latents


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    latent_path = f"{save_path}/{time_flag}_{seed}_latent.safetensors"

    if args.no_metadata:
        metadata = None
    else:
        metadata = {
            "seeds": f"{seed}",
            "prompt": f"{args.prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
            "flow_shift": f"{args.flow_shift}",
        }
        if args.negative_prompt is not None:
            metadata["negative_prompt"] = f"{args.negative_prompt}"

    sd = {"latent": latent.contiguous()}
    save_file(sd, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")

    return latent_path


def save_images(sample: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()

    seed = args.seed
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_name = f"{time_flag}_{seed}{original_name}"
    sample = sample.unsqueeze(0).unsqueeze(2)  # CHW -> BCFHW, B=1, F=1
    save_images_grid(sample, save_path, image_name, rescale=True, create_subdir=False)
    logger.info(f"Sample images saved to: {save_path}/{image_name}")

    return f"{save_path}/{image_name}"


def save_output(
    args: argparse.Namespace,
    vae: flux2_models.AutoEncoder,
    latent: torch.Tensor,
    device: torch.device,
    original_base_names: Optional[List[str]] = None,
) -> None:
    height, width = latent.shape[-2], latent.shape[-1]
    height *= ERNIE_IMAGE_VAE_SCALE
    width *= ERNIE_IMAGE_VAE_SCALE

    if args.output_type == "latent" or args.output_type == "latent_images":
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return

    if vae is None:
        logger.error("VAE is None, cannot decode latents for saving images.")
        return

    image = decode_latent(vae, latent, device)

    if args.output_type == "images" or args.output_type == "latent_images":
        original_name = None if original_base_names is None else original_base_names[0]
        save_images(image, args, original_name)


def preprocess_prompts_for_batch(prompt_lines: List[str], base_args: argparse.Namespace) -> List[Dict]:
    prompts_data = []

    for line in prompt_lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        prompt_data = parse_prompt_line(line)
        logger.info(f"Parsed prompt data: {prompt_data}")
        prompts_data.append(prompt_data)

    return prompts_data


def load_shared_models(args: argparse.Namespace) -> Dict:
    shared_models = {}
    te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    tokenizer, text_encoder = ernie_image_utils.load_text_encoder(
        args.text_encoder, dtype=te_dtype, device="cpu", disable_mmap=True, tokenizer_id=args.tokenizer
    )
    shared_models["tokenizer"] = tokenizer
    shared_models["text_encoder"] = text_encoder
    return shared_models


def process_batch_prompts(prompts_data: List[Dict], args: argparse.Namespace) -> None:
    if not prompts_data:
        logger.warning("No valid prompts found")
        return

    gen_settings = get_generation_settings(args)
    dit_weight_dtype = gen_settings.dit_weight_dtype
    device = gen_settings.device

    # 1. Prepare VAE (keep on CPU for now)
    logger.info("Loading VAE for batch generation...")
    vae_for_batch = flux2_utils.load_ae(args.vae, dtype=torch.bfloat16, device="cpu", disable_mmap=True)
    vae_for_batch.eval()

    all_prompt_args_list = [apply_overrides(args, pd) for pd in prompts_data]
    for prompt_args in all_prompt_args_list:
        check_inputs(prompt_args)

    # 2. Precompute Text Data
    logger.info("Loading Text Encoder for batch text preprocessing...")
    te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
    tokenizer_batch, text_encoder_batch = ernie_image_utils.load_text_encoder(
        args.text_encoder, dtype=te_dtype, device="cpu", disable_mmap=True, tokenizer_id=args.tokenizer
    )

    llm_device = torch.device("cpu") if args.text_encoder_cpu else device
    text_encoder_batch.to(llm_device)

    all_precomputed_text_data = []
    conds_cache_batch = {}

    temp_shared_models_txt = {
        "tokenizer": tokenizer_batch,
        "text_encoder": text_encoder_batch,
        "conds_cache": conds_cache_batch,
    }

    for i, prompt_args_item in enumerate(all_prompt_args_list):
        logger.info(f"Text preprocessing for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
        context, context_null = prepare_text_inputs(prompt_args_item, device, temp_shared_models_txt)
        all_precomputed_text_data.append((context, context_null))

    del tokenizer_batch, text_encoder_batch, temp_shared_models_txt, conds_cache_batch
    gc.collect()
    clean_memory_on_device(device)

    # 3. Load DiT Model once
    logger.info("Loading DiT model for batch generation...")
    first_prompt_args = all_prompt_args_list[0]
    dit_model = load_dit_model(first_prompt_args, device, dit_weight_dtype)

    if first_prompt_args.save_merged_model:
        logger.info("Merged DiT model saved. Skipping generation.")
        return

    shared_models_for_generate = {"model": dit_model}

    all_latents = []

    logger.info("Generating latents for all prompts...")
    with torch.no_grad():
        for i, prompt_args_item in enumerate(all_prompt_args_list):
            current_text_data = all_precomputed_text_data[i]
            height, width = check_inputs(prompt_args_item)

            logger.info(f"Generating latent for prompt {i + 1}/{len(all_prompt_args_list)}: {prompt_args_item.prompt}")
            try:
                latent = generate(prompt_args_item, gen_settings, shared_models_for_generate, current_text_data)

                if latent is None:
                    continue

                if prompt_args_item.output_type in ["latent", "latent_images"]:
                    save_latent(latent, prompt_args_item, height, width)

                all_latents.append(latent)
            except Exception as e:
                logger.error(f"Error generating latent for prompt: {prompt_args_item.prompt}. Error: {e}", exc_info=True)
                all_latents.append(None)
                continue

    logger.info("Releasing DiT model from memory...")
    if args.blocks_to_swap > 0:
        logger.info("Waiting for 5 seconds to finish block swap")
        time.sleep(5)

    del shared_models_for_generate["model"]
    del dit_model
    gc.collect()
    clean_memory_on_device(device)
    synchronize_device(device)

    # 4. Decode latents and save outputs
    if args.output_type != "latent":
        logger.info("Decoding latents to images...")
        vae_for_batch.to(device)

        for i, latent in enumerate(all_latents):
            if latent is None:
                logger.warning(f"Skipping decoding for prompt {i + 1} due to previous error.")
                continue

            current_args = all_prompt_args_list[i]
            logger.info(f"Decoding output {i + 1}/{len(all_latents)} for prompt: {current_args.prompt}")

            if current_args.output_type == "latent_images":
                current_args.output_type = "images"

            save_output(current_args, vae_for_batch, latent[0], device)

        vae_for_batch.to("cpu")

    del vae_for_batch
    clean_memory_on_device(device)


def process_interactive(args: argparse.Namespace) -> None:
    gen_settings = get_generation_settings(args)
    device = gen_settings.device
    shared_models = load_shared_models(args)
    shared_models["conds_cache"] = {}

    logger.info("Loading VAE for interactive mode...")
    vae = flux2_utils.load_ae(args.vae, dtype=torch.bfloat16, device="cpu", disable_mmap=True)
    vae.eval()

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")

    try:
        import prompt_toolkit
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        prompt_toolkit = None

    if prompt_toolkit:
        session = prompt_toolkit.PromptSession()

        def input_line(prompt: str) -> str:
            return session.prompt(prompt)

    else:

        def input_line(prompt: str) -> str:
            return input(prompt)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:
                    raise EOFError

                prompt_data = parse_prompt_line(line)
                prompt_args = apply_overrides(args, prompt_data)

                latent = generate(prompt_args, gen_settings, shared_models)

                save_output(prompt_args, vae, latent[0], device)

                if args.bell:
                    print("\a")

            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue

    except EOFError:
        print("\nExiting interactive mode")


def get_generation_settings(args: argparse.Namespace) -> GenerationSettings:
    device = torch.device(args.device)

    dit_weight_dtype = torch.bfloat16
    if args.fp8_scaled:
        dit_weight_dtype = None
    elif args.fp8:
        dit_weight_dtype = torch.float8_e4m3fn

    logger.info(f"Using device: {device}, DiT weight precision: {dit_weight_dtype}")

    gen_settings = GenerationSettings(device=device, dit_weight_dtype=dit_weight_dtype)
    return gen_settings


def main():
    args = parse_args()

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    logger.info(f"Using device: {device}")
    args.device = device

    if latents_mode:
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
                if "height" in metadata and "width" in metadata:
                    height = int(metadata["height"])
                    width = int(metadata["width"])
                    args.image_size = [height, width]

            seeds.append(seed)
            logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}")

            if latents.ndim == 4:  # [BCHW]
                latents = latents.squeeze(0)  # [CHW]

            latents_list.append(latents)

        for i, latent in enumerate(latents_list):
            args.seed = seeds[i]

            vae = flux2_utils.load_ae(args.vae, dtype=torch.bfloat16, device=device, disable_mmap=True)
            vae.eval()
            save_output(args, vae, latent, device, original_base_names)

    elif args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_lines = f.readlines()

        prompts_data = preprocess_prompts_for_batch(prompt_lines, args)
        process_batch_prompts(prompts_data, args)

        if args.bell:
            print("\a")

    elif args.interactive:
        process_interactive(args)

    else:
        gen_settings = get_generation_settings(args)
        latent = generate(args, gen_settings)

        vae = flux2_utils.load_ae(args.vae, dtype=torch.bfloat16, device=device, disable_mmap=True)
        vae.eval()
        save_output(args, vae, latent[0], device)

        if args.bell:
            print("\a")

    logger.info("Done!")


if __name__ == "__main__":
    main()
