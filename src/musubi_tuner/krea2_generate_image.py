"""Krea 2 (K2) text-to-image generation.

Mirrors the official `inference.py` end to end, parametrized for musubi: the DiT
checkpoint, Qwen-Image VAE, and Qwen3-VL text encoder are passed via argparse, and
the model code lives under `musubi_tuner.krea2`.

Memory-efficient inference is supported for the DiT: dynamic scaled fp8
(`--fp8_scaled`) and block swap (`--blocks_to_swap`, CPU offloading of the main
SingleStreamBlocks). Trained LoRA(s) are merged into the base weights at load time
(the only correct route under fp8).

Memory model: the DiT stays resident on the GPU (with block swap as needed), while the
Text Encoder and VAE shuttle between CPU and GPU. The encoder is kept on CPU and moved
to the GPU only to encode each prompt (use ``--text_encoder_cpu`` to encode on CPU when
even that does not fit); the VAE is kept on CPU and moved to the GPU only to decode. So
on a 24GB card the headroom for encoding/decoding comes from running the DiT under fp8
and/or block swap, rather than from evacuating the ~24GB DiT to host RAM.

Three input modes (mirroring zimage_generate_image.py):
  * single prompt — positional ``prompt`` argument
  * ``--from_file`` — one prompt per line, with optional per-prompt overrides
  * ``--interactive`` — read prompts from the console

In ``--from_file`` / ``--interactive`` each line may carry per-prompt overrides as
``--<opt> <value>`` after the prompt text, e.g.::

    a cat on a sofa --w 1280 --h 768 --s 32 --g 6.0 --d 42 --n blurry

Supported per-prompt options: ``--w`` (width), ``--h`` (height), ``--s`` (steps),
``--d`` (seed), ``--g`` / ``--l`` (guidance_scale), ``--n`` (negative_prompt),
``--y1``, ``--y2``, ``--mu`` (timestep-shift), ``--i`` (num images).
"""

import argparse
import copy
import gc
import logging
import os
import random
import time
from datetime import datetime
from typing import Any, Dict

import torch
from safetensors.torch import load_file

from musubi_tuner.krea2 import krea2_utils
from musubi_tuner.krea2.krea2_sampling import encode_prompts, sample
from musubi_tuner.krea2.krea2_utils import single_mmdit_large_wide
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig
from musubi_tuner.qwen_image import qwen_image_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_text_encoder(text_encoder: str, dtype: torch.dtype):
    """Load the Qwen3-VL conditioner onto CPU.

    The encoder is kept on CPU between prompts and moved to the GPU only for the brief
    encode of each prompt (see ``encode``), so it does not permanently occupy VRAM next to
    the resident DiT. Loading it to CPU also avoids competing with the DiT for the transient
    GPU headroom fp8 quantization needs at DiT load time.
    """
    logger.info("Loading Qwen3-VL text encoder (on CPU)")
    return krea2_utils.load_krea2_text_encoder(text_encoder, dtype=dtype, device="cpu")


def encode(encoder, prompts, negative_prompts, cfg: bool, te_device: str):
    """Encode prompts with the Qwen3-VL conditioner, shuttling it to ``te_device`` for the call.

    The encoder is moved to ``te_device`` (the GPU, or CPU with ``--text_encoder_cpu``) just for
    the encode, then moved back to CPU so the denoise/decode that follow have the VRAM to
    themselves. The returned embeddings are tiny and ``sample`` moves them to the compute device.
    """
    logger.info("Encoding prompts with Qwen3-VL")
    encoder.to(te_device)
    embeds = encode_prompts(encoder, prompts, negative_prompts, cfg=cfg)
    encoder.to("cpu")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeds  # (txt, txtmask, untxt, untxtmask)


def build_pipeline(
    dit_path: str,
    vae_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    mmdit_config=single_mmdit_large_wide,
    lora_weights=None,
    lora_multipliers=None,
    attn_mode: str = "torch",
    split_attn: bool = False,
    fp8_scaled: bool = False,
    blocks_to_swap: int = 0,
    swap_config=None,
):
    """Build the autoencoder and MMDiT and load weights (the text encoder is loaded separately).

    The DiT is loaded with optional scaled fp8 and/or block swap and stays resident on the
    device. Trained LoRA(s) are merged into the base weights inside the loader (works for fp8
    too). The VAE is kept on CPU here and moved to the device for decoding by ``sample`` (then
    back to CPU). When ``blocks_to_swap > 0`` a prebuilt ``swap_config`` (BlockSwapConfig)
    selects the offloader policy (pinned memory, H2D-only, etc.).
    """
    dev = torch.device(device)

    # K2 reuses the Qwen-Image VAE; load it to CPU (sample() moves it to the device for decode).
    ae = qwen_image_utils.load_vae(vae_path, input_channels=3, device="cpu", disable_mmap=True)
    ae = ae.to(dtype=dtype).eval().requires_grad_(False)

    # LoRA is merged into the base weights at load time (the only correct route under fp8).
    lora_sds = [load_file(p) for p in lora_weights] if lora_weights else None

    # With block swap the DiT must load to CPU first; the offloader then places the resident
    # blocks on the device and keeps the swap blocks on CPU.
    loading_device = "cpu" if blocks_to_swap > 0 else device
    mmdit = krea2_utils.load_krea2_dit(
        dit_path,
        device=device,
        dtype=dtype,
        config=mmdit_config,
        fp8_scaled=fp8_scaled,
        loading_device=loading_device,
        attn_mode=attn_mode,
        split_attn=split_attn,
        lora_weights=lora_sds,
        lora_multipliers=lora_multipliers,
    )

    # Freeze BEFORE enabling block swap: the H2D-only offloader (LoRAStreamOffloader) asserts the
    # streamed base weights are frozen (requires_grad=False) at construction time.
    mmdit.eval().requires_grad_(False)

    if blocks_to_swap > 0:
        logger.info(f"Enabling block swap: offloading {blocks_to_swap} blocks to CPU")
        if swap_config is None:
            swap_config = BlockSwapConfig(dev, supports_backward=False)
        mmdit.enable_block_swap(blocks_to_swap, swap_config)
        mmdit.move_to_device_except_swap_blocks(dev)
        mmdit.switch_block_swap_for_inference()
    else:
        mmdit.to(dev)

    return mmdit, ae


def generate(args: argparse.Namespace, dit, ae, encoder, device: str, dtype: torch.dtype, te_device: str):
    """Run one prompt end to end: encode -> denoise -> decode, returning a list of PIL images.

    The DiT (``dit``) and VAE (``ae``) are reused across calls. Per call the encoder is shuttled
    to ``te_device`` for the encode and the VAE is shuttled to the GPU for the decode (both inside
    the helpers), while the DiT stays resident.

    The base seed is resolved here: if ``args.seed`` is None a random one is drawn (and written
    back to ``args.seed`` so the caller can use it for file names); image *i* uses ``base + i``.
    """
    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
        logger.info(f"No seed specified; using random seed {args.seed}")

    prompts = [args.prompt] * args.num_images
    negative_prompts = [args.negative_prompt] * args.num_images
    cfg = args.guidance_scale > 1.0

    txt, txtmask, untxt, untxtmask = encode(encoder, prompts, negative_prompts, cfg, te_device)

    # Re-arm the block-swap offloader before each forward (no-op when block swap is off).
    if args.blocks_to_swap > 0:
        dit.prepare_block_swap_before_forward()

    images = sample(
        dit,
        ae,
        txt,
        txtmask,
        untxt=untxt,
        untxtmask=untxtmask,
        device=device,
        dtype=dtype,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg_scale=args.guidance_scale,
        seed=args.seed,
        y1=args.y1,
        y2=args.y2,
        mu=args.mu,
    )
    return images


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line into a dict of per-prompt overrides.

    Format: ``<prompt text> --w 1280 --h 768 --s 32 --g 6.0 --d 42 --n <negative prompt>``.
    Anything before the first ``--`` is the prompt; each ``--<opt> <value>`` after it overrides
    a generation parameter. Unknown options are ignored.
    """
    parts = line.split(" --")
    prompt = parts[0].strip()

    overrides: Dict[str, Any] = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        if option == "w":
            overrides["width"] = int(value)
        elif option == "h":
            overrides["height"] = int(value)
        elif option == "s":
            overrides["steps"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "y1":
            overrides["y1"] = float(value)
        elif option == "y2":
            overrides["y2"] = float(value)
        elif option == "mu":
            overrides["mu"] = float(value)
        elif option == "i":
            overrides["num_images"] = int(value)
        else:
            logger.warning(f"Unknown prompt option '--{option}', ignoring")

    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    """Return a copy of ``args`` with ``overrides`` applied (one set per prompt line)."""
    args_copy = copy.deepcopy(args)
    for key, value in overrides.items():
        setattr(args_copy, key, value)
    return args_copy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate images with Krea 2 (K2).")
    parser.add_argument("prompt", type=str, nargs="?", default=None, help="Prompt for image generation (single-prompt mode)")
    parser.add_argument("--dit", type=str, required=True, help="Path to the MMDiT checkpoint (.safetensors)")
    parser.add_argument("--vae", type=str, required=True, help="Path to the Qwen-Image VAE checkpoint (.safetensors)")
    parser.add_argument(
        "--text_encoder",
        type=str,
        required=True,
        help="Qwen3-VL-4B text encoder safetensors path (official or ComfyUI key layout)",
    )
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt (used only when --guidance_scale > 1)")
    parser.add_argument("--steps", type=int, default=28, help="Number of denoising steps")
    parser.add_argument("--guidance_scale", type=float, default=5.5, help="Classifier-free guidance scale (<= 1 disables CFG)")
    parser.add_argument("--y1", type=float, default=0.5, help="Timestep-shift mu at min resolution")
    parser.add_argument("--y2", type=float, default=1.15, help="Timestep-shift mu at max resolution")
    parser.add_argument("--mu", type=float, default=None, help="Pin a constant timestep-shift mu (overrides y1/y2)")
    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image")
    parser.add_argument("--num-images", dest="num_images", type=int, default=1, help="Number of images to generate")
    parser.add_argument(
        "--seed", type=int, default=None, help="Base seed; image i uses seed + i. If omitted, a random seed is used."
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument(
        "--text_encoder_cpu",
        action="store_true",
        help="Encode prompts on CPU instead of moving the text encoder to the GPU "
        "(slower, but avoids the encoder competing with the resident DiT for VRAM)",
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["torch", "flash", "sageattn", "xformers"],
        help="attention backend ('torch'=SDPA, 'flash', 'sageattn', 'xformers')",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="split attention per sample (recommended for non-sdpa backends; required for xformers + GQA)",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use dynamic scaled fp8 for the DiT (per-block Linears quantized at load time). "
        "K2 supports only scaled fp8 (plain fp8 would cast norms and break).",
    )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="number of main SingleStreamBlocks to offload to CPU (block swap). Max 26 (= 28 - 2). "
        "Reduces VRAM at the cost of speed.",
    )
    parser.add_argument(
        "--use_pinned_memory_for_block_swap",
        action="store_true",
        help="use pinned CPU memory for block swap (faster H2D copies, more host RAM)",
    )
    parser.add_argument(
        "--block_swap_h2d_only",
        action="store_true",
        help="H2D-only block swap: keep a CPU master of streamed blocks and copy Host->Device only "
        "(no device->host copy). The DiT base weights are frozen at inference, so this is always safe here.",
    )
    parser.add_argument(
        "--block_swap_ring_size",
        type=int,
        default=2,
        help="(used with --block_swap_h2d_only) number of GPU ring buffers for streamed blocks. "
        "2 = double buffering (transfer/compute overlap); 1 = minimal memory, no overlap.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Directory to save generated images. File names are auto-generated from a timestamp and the seed.",
    )
    parser.add_argument(
        "--lora_weight", type=str, nargs="*", default=None, help="Trained LoRA weight path(s) (.safetensors) to merge"
    )
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier per weight (default 1.0 each)"
    )

    # batch / interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts (one per line) from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from the console")
    parser.add_argument(
        "--bell",
        action="store_true",
        help="Ring the terminal bell after each prompt (interactive) or at the end (other modes)",
    )

    args = parser.parse_args()

    # Validate input mode: exactly one of {prompt, --from_file, --interactive}.
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")
    if not args.from_file and not args.interactive and args.prompt is None:
        raise ValueError("Either a positional prompt, --from_file or --interactive must be specified")

    return args


def get_time_flag() -> str:
    """Millisecond timestamp string used to make output file names unique (e.g. 20260623-143012-123)."""
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_images(images, save_path: str, base_seed: int):
    """Save a list of PIL images into ``save_path`` (created if needed).

    File names are auto-generated as ``<timestamp>_<seed>.png``; image *i* uses ``base_seed + i``,
    so a single timestamp + per-image seed keeps the names unique within and across prompts.
    """
    os.makedirs(save_path, exist_ok=True)
    time_flag = get_time_flag()
    for i, image in enumerate(images):
        out = os.path.join(save_path, f"{time_flag}_{base_seed + i}.png")
        image.save(out)
        logger.info(f"saved {out}")


def main():
    args = parse_args()
    dtype = torch.bfloat16
    device = args.device
    te_device = "cpu" if args.text_encoder_cpu else device

    # Load all three models up front and keep them: the encoder and VAE live on CPU, the DiT
    # lives on the GPU (with block swap as needed). Encoder/VAE shuttle to the GPU per prompt.
    encoder = load_text_encoder(args.text_encoder, dtype)

    swap_config = None
    if args.blocks_to_swap > 0:
        # Inference is forward-only (supports_backward=False); from_args picks up the
        # pinned-memory / h2d-only / ring-size knobs.
        swap_config = BlockSwapConfig.from_args(args, torch.device(device), supports_backward=False)
    dit, ae = build_pipeline(
        args.dit,
        args.vae,
        device=device,
        dtype=dtype,
        lora_weights=args.lora_weight,
        lora_multipliers=args.lora_multiplier,
        attn_mode=args.attn_mode,
        split_attn=args.split_attn,
        fp8_scaled=args.fp8_scaled,
        blocks_to_swap=args.blocks_to_swap,
        swap_config=swap_config,
    )

    if args.from_file:
        with open(args.from_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        index = 0
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):  # skip blank lines and comments
                continue
            prompt_args = apply_overrides(args, parse_prompt_line(line))
            logger.info(f"[{index}] Prompt: {prompt_args.prompt}")
            images = generate(prompt_args, dit, ae, encoder, device, dtype, te_device)
            save_images(images, args.save_path, prompt_args.seed)
            index += 1
        if args.bell:
            print("\a")

    elif args.interactive:
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

        print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")
        try:
            while True:
                try:
                    line = input_line("> ").strip()
                    if not line:
                        continue
                    if len(line) == 1 and line in ["\x04", "\x1a"]:  # Ctrl+D / Ctrl+Z
                        raise EOFError
                    prompt_args = apply_overrides(args, parse_prompt_line(line))
                    logger.info(f"Prompt: {prompt_args.prompt}")
                    images = generate(prompt_args, dit, ae, encoder, device, dtype, te_device)
                    save_images(images, args.save_path, prompt_args.seed)
                    if args.bell:
                        print("\a")
                except KeyboardInterrupt:
                    print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                    continue
        except EOFError:
            print("\nExiting interactive mode")

    else:
        images = generate(args, dit, ae, encoder, device, dtype, te_device)
        save_images(images, args.save_path, args.seed)
        if args.bell:
            print("\a")

    logger.info("Done!")


if __name__ == "__main__":
    main()
