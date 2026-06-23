import argparse
import gc
import os
import random

import torch
from safetensors.torch import save_file
from tqdm import tqdm
from PIL import Image

from musubi_tuner.krea2 import krea2_model, krea2_utils
from musubi_tuner.qwen_image.qwen_image_autoencoder_kl import AutoencoderKLQwenImage
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.utils.lora_utils import filter_lora_state_dict
from musubi_tuner.utils.device_utils import clean_memory_on_device
from musubi_tuner.hv_generate_video import setup_parser_compile, synchronize_device

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

lycoris_available = False
try:
    import lycoris  # noqa: F401

    lycoris_available = True
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Krea 2 inference script")

    parser.add_argument("--dit", type=str, default=None, help="DiT checkpoint path (.safetensors)")
    parser.add_argument("--disable_numpy_memmap", action="store_true", help="Disable numpy memmap when loading safetensors.")
    parser.add_argument("--vae", type=str, default=None, help="VAE checkpoint path")
    parser.add_argument("--vae_enable_tiling", action="store_true", help="Enable tiling for VAE decoding.")
    parser.add_argument("--text_encoder", type=str, required=True, help="Text Encoder (Qwen3-VL 4B) checkpoint path")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=[1.0], help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None, help="LoRA module include patterns")
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None, help="LoRA module exclude patterns")
    parser.add_argument("--save_merged_model", type=str, default=None, help="Save merged model to path.")

    # inference
    parser.add_argument("--guidance_scale", type=float, default=4.5, help="Guidance scale for CFG. Default is 4.5.")
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for generation")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width")
    parser.add_argument("--infer_steps", type=int, default=28, help="number of inference steps, default is 28")
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated image")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument("--turbo", action="store_true", help="Use turbo mode (fixed mu=1.15, fewer steps).")

    # Flow Matching
    parser.add_argument("--flow_shift", type=float, default=None, help="Override shift factor. Default is None (dynamic).")

    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder")
    parser.add_argument("--text_encoder_cpu", action="store_true", help="Inference on CPU for Text Encoder")
    parser.add_argument("--device", type=str, default=None, help="device to use for inference")
    parser.add_argument(
        "--attn_mode", type=str, default="torch", choices=["flash", "torch", "sageattn", "xformers", "sdpa"], help="attention mode"
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of blocks to swap")
    parser.add_argument("--use_pinned_memory_for_block_swap", action="store_true", help="use pinned memory for block swap")
    parser.add_argument("--output_type", type=str, default="images", choices=["images", "latent"], help="output type")
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode")
    parser.add_argument("--lycoris", action="store_true", help="use lycoris for inference")
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--bell", action="store_true", help="Ring bell when done")
    parser.add_argument("--split_attn", action="store_true", help="Use split attention")

    setup_parser_compile(parser)

    args = parser.parse_args()

    # set defaults
    if args.turbo and args.infer_steps == 28:
        args.infer_steps = 8
    if args.turbo and args.guidance_scale == 4.5:
        args.guidance_scale = 0.0

    return args


def load_vae(args, device) -> AutoencoderKLQwenImage:
    logger.info(f"Loading VAE from {args.vae}")
    vae = krea2_utils.load_vae(args.vae, device="cpu", disable_mmap=True)
    vae.eval()
    return vae


def load_dit_model(args, device, dit_weight_dtype):
    split_attn = args.split_attn

    lora_weights_list = None
    if args.lora_weight is not None:
        lora_weights_list = []
        for lora_weight_path in args.lora_weight:
            lora_weights = filter_lora_state_dict(load_safetensors_or_load_file(lora_weight_path))
            lora_weights_list.append(lora_weights)

    lora_multipliers = args.lora_multiplier
    if lora_multipliers is not None and not isinstance(lora_multipliers, list):
        lora_multipliers = [lora_multipliers]

    model = krea2_model.load_krea2_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=split_attn,
        loading_device="cpu",
        dit_weight_dtype=dit_weight_dtype,
        fp8_scaled=args.fp8_scaled,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        disable_numpy_memmap=args.disable_numpy_memmap,
    )

    if args.save_merged_model:
        sd = model.state_dict()
        save_file(sd, args.save_merged_model)
        logger.info(f"Saved merged model to {args.save_merged_model}")
        return None

    # block swap
    if args.blocks_to_swap > 0:
        config = BlockSwapConfig(
            supports_backward=False,
            supports_hierarchical_offload=False,
            pin_memory=args.use_pinned_memory_for_block_swap,
            h2d_only=False,
        )
        model.enable_block_swap(args.blocks_to_swap, config)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        model.to(device)

    return model


def load_safetensors_or_load_file(path):
    from safetensors.torch import load_file

    return load_file(path)


def prepare_text_inputs(args, device):
    vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
    te_device = "cpu" if args.text_encoder_cpu else device

    logger.info(f"Loading Qwen3-VL text encoder: {args.text_encoder}")
    processor, text_encoder = krea2_utils.load_text_encoder(args.text_encoder, dtype=vl_dtype, device=te_device)

    prompt = args.prompt or ""
    negative_prompt = args.negative_prompt or " "

    with (
        torch.amp.autocast(device_type=te_device.type if hasattr(te_device, "type") else str(te_device), dtype=vl_dtype),
        torch.no_grad(),
    ):
        hidden, mask = krea2_utils.get_krea2_prompt_embeds(processor, text_encoder, [prompt], device=te_device, dtype=vl_dtype)
        neg_hidden, neg_mask = krea2_utils.get_krea2_prompt_embeds(
            processor, text_encoder, [negative_prompt], device=te_device, dtype=vl_dtype
        )

    del processor, text_encoder
    gc.collect()
    clean_memory_on_device(device)

    txt_len = mask[0].to(dtype=torch.bool).sum().item()
    embed = hidden[0, :txt_len].to(device, dtype=torch.bfloat16)
    embed_mask = mask[0, :txt_len].to(device, dtype=torch.bool)

    neg_txt_len = neg_mask[0].to(dtype=torch.bool).sum().item()
    neg_embed = neg_hidden[0, :neg_txt_len].to(device, dtype=torch.bfloat16)
    neg_embed_mask = neg_mask[0, :neg_txt_len].to(device, dtype=torch.bool)

    return {
        "prompt": prompt,
        "embed": embed,
        "mask": embed_mask,
    }, {
        "prompt": negative_prompt,
        "embed": neg_embed,
        "mask": neg_embed_mask,
    }


def generate(args, device, dit_weight_dtype):
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed

    # Load VAE
    vae = load_vae(args, device)

    # Prepare text inputs
    context, context_null = prepare_text_inputs(args, device)

    # Load DiT
    model = load_dit_model(args, device, dit_weight_dtype)
    if model is None:
        return vae, None

    # Random generator
    seed_g = torch.Generator(device="cpu")
    seed_g.manual_seed(seed)

    height, width = args.image_size[0], args.image_size[1]
    # Ensure dimensions are multiples of 16
    height = (height // 16) * 16
    width = (width // 16) * 16
    logger.info(f"Image size: {height}x{width} (HxW), infer_steps: {args.infer_steps}, turbo: {args.turbo}")

    embed = context["embed"]
    embed_mask = context["mask"]
    neg_embed = context_null["embed"]
    neg_embed_mask = context_null["mask"]

    # Prepare latents
    num_channels_latents = krea2_utils.VAE_CHANNELS
    latents = krea2_utils.prepare_latents(1, num_channels_latents, height, width, torch.bfloat16, device, seed_g)

    img_h = height // krea2_utils.VAE_SCALE_FACTOR // krea2_utils.PATCH_SIZE
    img_w = width // krea2_utils.VAE_SCALE_FACTOR // krea2_utils.PATCH_SIZE
    img_shapes = [(1, img_h, img_w)]

    image_seq_len = latents.shape[1]
    logger.info(f"Image seq len: {image_seq_len}, embed shape: {embed.shape}")

    # Get timesteps
    timesteps = krea2_utils.get_timesteps(args.infer_steps, image_seq_len, device, is_turbo=args.turbo)

    do_cfg = args.guidance_scale > 0.0

    # Denoising loop
    with tqdm(total=args.infer_steps, desc="Denoising steps") as pbar:
        for i in range(len(timesteps) - 1):
            t_curr = timesteps[i]
            t_prev = timesteps[i + 1]

            timestep = t_curr.expand(latents.shape[0]).to(latents.dtype)

            with torch.no_grad():
                noise_pred = model(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=embed.unsqueeze(0),
                    encoder_hidden_states_mask=embed_mask.unsqueeze(0),
                    img_shapes=img_shapes,
                    txt_seq_lens=[embed.shape[0]],
                )

            if do_cfg:
                with torch.no_grad():
                    neg_noise_pred = model(
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=neg_embed.unsqueeze(0),
                        encoder_hidden_states_mask=neg_embed_mask.unsqueeze(0),
                        img_shapes=img_shapes,
                        txt_seq_lens=[neg_embed.shape[0]],
                    )
                noise_pred = neg_noise_pred + args.guidance_scale * (noise_pred - neg_noise_pred)

            latents = latents + (t_prev - t_curr) * noise_pred
            pbar.update()

    # Unpack latents
    latents = krea2_utils.unpack_latents(latents, height, width)

    if args.output_type == "latent":
        logger.info(f"Saving latent to {args.save_path}")
        save_file({"latent": latents.cpu()}, args.save_path)
        return vae, latents

    # Decode
    logger.info(f"Decoding image from latents: {latents.shape}")
    vae.to(device)
    vae.eval()
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latents.to(device, vae.dtype))

    vae.to("cpu")
    clean_memory_on_device(device)

    # Save image
    pixels = pixels.to(torch.float32).cpu()
    pixels = pixels[:, :, 0]  # B, C, H, W (remove frame dim)
    pixels = (pixels * 255).clamp(0, 255).to(torch.uint8)
    pixels = pixels.permute(0, 2, 3, 1)  # B, H, W, C

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    for b in range(pixels.shape[0]):
        img = Image.fromarray(pixels[b].numpy())
        img.save(args.save_path)
        logger.info(f"Saved image to {args.save_path}")

    return vae, latents


def main():
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    dit_weight_dtype = None
    if args.fp8_scaled:
        dit_weight_dtype = None  # fp8_scaled uses None
    else:
        dit_weight_dtype = torch.bfloat16

    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
        for prompt in prompts:
            args.prompt = prompt
            generate(args, device, dit_weight_dtype)
            synchronize_device(device)
    elif args.interactive:
        while True:
            try:
                prompt = input("Prompt (Ctrl+C to exit): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not prompt:
                continue
            args.prompt = prompt
            generate(args, device, dit_weight_dtype)
            synchronize_device(device)
            if args.bell:
                print("\a", end="", flush=True)
    else:
        generate(args, device, dit_weight_dtype)

    if args.bell:
        print("\a", end="", flush=True)


if __name__ == "__main__":
    main()
