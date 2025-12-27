import argparse
import os
from types import SimpleNamespace
from typing import Optional

import torch
import torchvision.utils as vutils
from safetensors.torch import load_file

from musubi_tuner.kandinsky5.configs import TASK_CONFIGS
from musubi_tuner.kandinsky5.generation_utils import generate_sample_latents_only, decode_latents, get_first_frame_from_image
from musubi_tuner.kandinsky5.models.text_embedders import get_text_embedder
from musubi_tuner.kandinsky5_train_network import Kandinsky5NetworkTrainer
from musubi_tuner.hv_train_network import clean_memory_on_device
from musubi_tuner.hv_generate_video import save_videos_grid
from musubi_tuner.networks import lora_kandinsky


def _get_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kandinsky5 sampling (mirrors training sampler, no training)")
    parser.add_argument("--task", type=str, default="k5-pro-t2v-5s-sd", choices=list(TASK_CONFIGS.keys()))
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument(
        "--i", "--image", dest="image", type=str, default=None, help="Init image path for i2v-style seeding (first frame)"
    )
    parser.add_argument(
        "--image_last", type=str, default=None, help="Optional last-frame image path for i2v first_last conditioning"
    )
    parser.add_argument("--output", type=str, required=True)  # mp4 for video, png for image
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--scheduler_scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dit", type=str, default=None)
    parser.add_argument("--vae", type=str, default=None)
    parser.add_argument("--text_encoder_qwen", type=str, default=None)
    parser.add_argument("--text_encoder_clip", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--offload_dit_during_sampling", action="store_true")
    parser.add_argument("--fp8_base", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--fp8_fast", action="store_true")
    parser.add_argument("--disable_numpy_memmap", action="store_true")
    parser.add_argument("--sdpa", action="store_true", help="use SDPA for visual attention")
    parser.add_argument("--flash_attn", action="store_true", help="use FlashAttention 2 for visual attention")
    parser.add_argument("--flash3", action="store_true", help="use FlashAttention 3 for visual attention")
    parser.add_argument("--sage_attn", action="store_true", help="use SageAttention for visual attention")
    parser.add_argument("--xformers", action="store_true", help="use xformers for visual attention")
    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path(s) to merge for inference")
    parser.add_argument(
        "--lora_multiplier", type=float, nargs="*", default=None, help="LoRA multiplier(s), align with lora_weight order"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task_conf = TASK_CONFIGS[args.task]
    device = _get_device(args.device)
    # keep dtype option for compatibility; actual model dtype follows training loader defaults
    _ = getattr(torch, args.dtype)

    width = args.width or task_conf.resolution
    height = args.height or task_conf.resolution
    frames = args.frames if args.frames is not None else (5 if task_conf.dit_params.visual_cond else 1)
    i2v_mode = "first_last" if args.image_last else "first"
    steps = args.steps or task_conf.num_steps
    guidance = args.guidance if args.guidance is not None else task_conf.guidance_weight
    scheduler_scale = args.scheduler_scale if args.scheduler_scale is not None else (task_conf.scheduler_scale or 1.0)

    latent_h = max(1, height // 8)
    latent_w = max(1, width // 8)
    shape = (1, frames, latent_h, latent_w, task_conf.dit_params.in_visual_dim)

    # Resolve paths
    dit_path = args.dit or task_conf.checkpoint_path
    vae_path = args.vae or task_conf.vae.checkpoint_path
    qwen_path = args.text_encoder_qwen or task_conf.text.qwen_checkpoint
    clip_path = args.text_encoder_clip or task_conf.text.clip_checkpoint

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Build trainer (reuse training loaders)
    trainer = Kandinsky5NetworkTrainer()
    trainer.task_conf = task_conf
    trainer.blocks_to_swap = args.blocks_to_swap
    trainer._text_encoder_qwen_path = args.text_encoder_qwen
    trainer._text_encoder_clip_path = args.text_encoder_clip
    trainer._vae_checkpoint_path = vae_path

    # --- Stage 1: text encoder only ---
    text_embedder_conf = SimpleNamespace(
        qwen=SimpleNamespace(checkpoint_path=qwen_path, max_length=task_conf.text.qwen_max_length),
        clip=SimpleNamespace(checkpoint_path=clip_path, max_length=task_conf.text.clip_max_length),
    )
    text_embedder = get_text_embedder(text_embedder_conf, device=device, quantized_qwen=False)
    neg_text = args.negative_prompt or "low quality, bad quality"
    enc_out, _, attention_mask = text_embedder.encode([args.prompt], type_of_content=("video" if frames > 1 else "image"))
    neg_out, _, neg_attention_mask = text_embedder.encode([neg_text], type_of_content=("video" if frames > 1 else "image"))
    text_embeds = enc_out["text_embeds"].to("cpu")
    pooled_embed = enc_out["pooled_embed"].to("cpu")
    null_text_embeds = neg_out["text_embeds"].to("cpu")
    null_pooled_embed = neg_out["pooled_embed"].to("cpu")
    if attention_mask is not None:
        attention_mask = attention_mask.to("cpu")
    if neg_attention_mask is not None:
        neg_attention_mask = neg_attention_mask.to("cpu")
    if attention_mask is not None:
        mask = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
        mask = mask.bool().flatten()
        if mask.shape[0] != text_embeds.shape[0]:
            # Processor returns padded masks; embeds are packed to valid tokens.
            if mask.sum().item() == text_embeds.shape[0]:
                mask = mask[mask]
            else:
                mask = torch.ones((text_embeds.shape[0],), dtype=torch.bool)
        text_embeds = text_embeds[mask]
        attention_mask = None
    if neg_attention_mask is not None:
        mask = neg_attention_mask[0] if neg_attention_mask.dim() > 1 else neg_attention_mask
        mask = mask.bool().flatten()
        if mask.shape[0] != null_text_embeds.shape[0]:
            # Processor returns padded masks; embeds are packed to valid tokens.
            if mask.sum().item() == null_text_embeds.shape[0]:
                mask = mask[mask]
            else:
                mask = torch.ones((null_text_embeds.shape[0],), dtype=torch.bool)
        null_text_embeds = null_text_embeds[mask]
        neg_attention_mask = None
    try:
        text_embedder.to("cpu")
    except Exception:
        pass
    del text_embedder
    clean_memory_on_device(device)

    conf_ns = SimpleNamespace(model=task_conf, metrics=SimpleNamespace(scale_factor=task_conf.scale_factor))

    with torch.no_grad():
        # --- Stage 2: load DiT, sample latents ---
        loader_args = SimpleNamespace(
            fp8_base=args.fp8_base,
            fp8_scaled=args.fp8_scaled,
            fp8_fast=args.fp8_fast,
            blocks_to_swap=args.blocks_to_swap,
            disable_numpy_memmap=args.disable_numpy_memmap,
            override_dit=None,
            sdpa=args.sdpa,
            flash_attn=args.flash_attn,
            flash3=args.flash3,
            sage_attn=args.sage_attn,
            xformers=args.xformers,
        )
        accel_stub = SimpleNamespace(device=device)
        dit = trainer.load_transformer(
            accelerator=accel_stub,
            args=loader_args,
            dit_path=dit_path,
            attn_mode=task_conf.attention.type,
            split_attn=False,
            loading_device=device,
            dit_weight_dtype=None,
        )
        dit.eval()
        dit.requires_grad_(False)
        # Merge LoRA weights before any casting/offloading.
        if args.lora_weight is not None and len(args.lora_weight) > 0:
            for idx, lora_path in enumerate(args.lora_weight):
                mult = args.lora_multiplier[idx] if args.lora_multiplier and len(args.lora_multiplier) > idx else 1.0
                lora_sd = load_file(lora_path)
                net = lora_kandinsky.create_arch_network_from_weights(mult, lora_sd, unet=dit, for_inference=True)
                net.merge_to(None, dit, lora_sd, device=dit.device if hasattr(dit, "device") else device, non_blocking=True)
            clean_memory_on_device(device)
        if args.blocks_to_swap and args.blocks_to_swap > 0:
            dit.enable_block_swap(args.blocks_to_swap, device, supports_backward=False, use_pinned_memory=False)
            dit.move_to_device_except_swap_blocks(device)
        if hasattr(dit, "switch_block_swap_for_inference"):
            dit.switch_block_swap_for_inference()

        autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
        transformer_offloaded = args.offload_dit_during_sampling and device.type == "cuda"
        original_device = device

        if transformer_offloaded:
            dit.to("cpu")
            torch.cuda.empty_cache()

        first_frames = None
        # Optional init image(s) -> latent first/last frames (i2v-style). Requires temporary VAE load.
        if args.image:
            vae_for_encode = trainer._load_vae_for_sampling(args, device=device)
            try:
                max_area = 512 * 768 if int(task_conf.resolution) == 512 else 1024 * 1024
                divisibility = 16 if int(task_conf.resolution) == 512 else 128
                # Always encode the first image
                _, lat_image_first, _ = get_first_frame_from_image(
                    args.image,
                    vae_for_encode,
                    device,
                    max_area=max_area,
                    divisibility=divisibility,
                )
                frame_list = [lat_image_first[:1]]
                # Optionally encode the last image
                if args.image_last:
                    _, lat_image_last, _ = get_first_frame_from_image(
                        args.image_last,
                        vae_for_encode,
                        device,
                        max_area=max_area,
                        divisibility=divisibility,
                    )
                    frame_list.append(lat_image_last[:1])
                first_frames = torch.cat(frame_list, dim=0)
                # If the init image was resized by the encoder, match sampling shape to it.
                if first_frames is not None:
                    latent_h = int(first_frames.shape[1])
                    latent_w = int(first_frames.shape[2])
                    shape = (1, frames, latent_h, latent_w, task_conf.dit_params.in_visual_dim)
            finally:
                try:
                    vae_for_encode.to("cpu")
                except Exception:
                    pass
                del vae_for_encode
                clean_memory_on_device(device)

        if transformer_offloaded:
            if hasattr(dit, "move_to_device_except_swap_blocks"):
                dit.move_to_device_except_swap_blocks(original_device)
            else:
                dit.to(original_device)
            torch.cuda.empty_cache()
        if hasattr(dit, "prepare_block_swap_before_forward"):
            dit.prepare_block_swap_before_forward()

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            latents = generate_sample_latents_only(
                shape=shape,
                dit=dit,
                text_embeds=text_embeds,
                pooled_embed=pooled_embed,
                attention_mask=attention_mask,
                null_text_embeds=null_text_embeds,
                null_pooled_embed=null_pooled_embed,
                null_attention_mask=neg_attention_mask,
                first_frames=first_frames,
                num_steps=steps,
                guidance_weight=guidance,
                scheduler_scale=scheduler_scale,
                seed=args.seed,
                device=device,
                conf=conf_ns,
                progress=True,
                i2v_mode=i2v_mode,
            )
        # free DiT
        dit.to("cpu")
        del dit
        clean_memory_on_device(device)

        # --- Stage 3: load VAE, decode ---
        vae = trainer._load_vae_for_sampling(args, device=device)
        images = decode_latents(latents, vae, device=device, batch_size=shape[0], num_frames=frames)
        try:
            vae.to("cpu")
        except Exception:
            pass
        del vae
        clean_memory_on_device(device)

    # Save
    if frames > 1:
        video_tensor = images.permute(0, 4, 1, 2, 3).float() / 255.0
        video_tensor = video_tensor.cpu()
        save_videos_grid(video_tensor, args.output, rescale=False, n_rows=1)
        first_frame_path = os.path.splitext(args.output)[0] + ".png"
        frame = images[0, 0].float() / 255.0
        frame = frame.cpu()
        vutils.save_image(frame.permute(2, 0, 1), first_frame_path)
        print(f"Saved video to {args.output} and first frame to {first_frame_path}")
    else:
        frame = images[0].float() / 255.0
        frame = frame.cpu()
        vutils.save_image(frame.permute(2, 0, 1), args.output)
        print(f"Saved image to {args.output}")


if __name__ == "__main__":
    main()
