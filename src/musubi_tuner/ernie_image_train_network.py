import argparse
from typing import Optional

import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_ERNIE_IMAGE, ARCHITECTURE_ERNIE_IMAGE_FULL
from musubi_tuner.ernie_image import ernie_image_model, ernie_image_utils
from musubi_tuner.flux_2 import flux2_utils, flux2_models
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)
from musubi_tuner.utils import model_utils

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ErnieImageNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_ERNIE_IMAGE

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_ERNIE_IMAGE_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 4.0
        self.default_discrete_flow_shift = None

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"Cache text encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        tokenizer, text_encoder = ernie_image_utils.load_text_encoder(
            args.text_encoder, dtype=torch.bfloat16, device=device, disable_mmap=True, tokenizer_id=args.tokenizer
        )
        text_encoder.eval()

        sample_prompts_te_outputs = {}

        for prompt_dict in prompts:
            if "negative_prompt" not in prompt_dict:
                prompt_dict["negative_prompt"] = ""

            for prompt in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                if prompt is None or prompt in sample_prompts_te_outputs:
                    continue

                logger.info(f"Encoding prompt: {prompt}")
                text_hiddens = ernie_image_utils.encode_text(tokenizer, text_encoder, prompt)
                embed = text_hiddens[0].cpu()  # [T, D]
                sample_prompts_te_outputs[prompt] = embed

        del tokenizer, text_encoder
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            prompt = prompt_dict.get("prompt", "")
            prompt_dict_copy["text_embed"] = sample_prompts_te_outputs[prompt]

            negative_prompt = prompt_dict.get("negative_prompt", "")
            prompt_dict_copy["negative_text_embed"] = sample_prompts_te_outputs[negative_prompt]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)
        return sample_parameters

    def do_inference(
        self,
        accelerator,
        args,
        sample_parameter,
        vae,
        dit_dtype,
        transformer,
        discrete_flow_shift,
        sample_steps,
        width,
        height,
        frame_count,
        generator,
        do_classifier_free_guidance,
        guidance_scale,
        cfg_scale,
        image_path=None,
        control_video_path=None,
    ):
        model: ernie_image_model.ErnieImageTransformer2DModel = accelerator.unwrap_model(transformer)
        device = accelerator.device

        embed = sample_parameter["text_embed"].to(device=device, dtype=torch.bfloat16)
        if embed.dim() == 1:
            embed = embed.unsqueeze(0)  # [D] -> [1, D]
        if embed.dim() == 2:
            embed = embed.unsqueeze(0)  # [T, D] -> [1, T, D]

        if cfg_scale is None:
            cfg_scale = 4.0
        do_cfg = cfg_scale > 1.0
        if do_cfg:
            negative_embed = sample_parameter["negative_text_embed"].to(device=device, dtype=torch.bfloat16)
            if negative_embed.dim() == 1:
                negative_embed = negative_embed.unsqueeze(0)
            if negative_embed.dim() == 2:
                negative_embed = negative_embed.unsqueeze(0)

        # Latent dimensions: VAE downscales by 16 (4 blocks of 2x), patchify doubles channels
        vae_scale = 16
        latent_h = int(height) // vae_scale
        latent_w = int(width) // vae_scale
        shape = (1, model.in_channels, latent_h, latent_w)

        latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)

        # Scheduler: linear sigmas from 1.0 to 0.0, optionally remapped by flow shift
        shift = discrete_flow_shift if discrete_flow_shift is not None else 1.0
        sigmas = ernie_image_utils.get_sigmas(sample_steps, device, shift=shift)

        # Pad text for batch (CFG doubles batch)
        if do_cfg:
            text_hiddens_list = [negative_embed[0], embed[0]]  # [uncond, cond]
        else:
            text_hiddens_list = [embed[0]]

        text_bth, text_lens = ernie_image_utils.pad_text(
            text_hiddens_list, device, torch.bfloat16, model.text_in_dim
        )

        for i in tqdm(range(sample_steps), desc="Sampling"):
            t = sigmas[i]

            # Model expects timestep in [0, 1000] range (matches training's t * 1000).
            t_scaled = t.item() * 1000.0
            if do_cfg:
                latent_model_input = torch.cat([latents, latents], dim=0)
                t_batch = torch.full((2,), t_scaled, device=device, dtype=torch.bfloat16)
            else:
                latent_model_input = latents
                t_batch = torch.full((1,), t_scaled, device=device, dtype=torch.bfloat16)

            latent_model_input = latent_model_input.to(model.dtype if hasattr(model, 'dtype') else dit_dtype)

            with accelerator.autocast(), torch.no_grad():
                pred = transformer(
                    hidden_states=latent_model_input,
                    timestep=t_batch,
                    text_bth=text_bth,
                    text_lens=text_lens,
                )

            if do_cfg:
                pred_uncond, pred_cond = pred.chunk(2, dim=0)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

            # Euler step
            dt = sigmas[i + 1] - sigmas[i]
            latents = latents + pred.to(torch.float32) * dt

        # Decode
        vae.to(device)
        vae.eval()

        with torch.no_grad():
            pixels = vae.decode(latents.to(vae.dtype))

        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels.clamp(-1, 1) + 1) / 2

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # B C F H W, F=1
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae
        logger.info(f"Loading VAE from {vae_path}")
        vae = flux2_utils.load_ae(vae_path, dtype=vae_dtype, device="cpu", disable_mmap=True)
        return vae

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = ernie_image_utils.load_dit(
            device=loading_device,
            dit_path=dit_path,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            dit_weight_dtype=dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        return model

    def compile_transformer(self, args, transformer):
        model: ernie_image_model.ErnieImageTransformer2DModel = transformer
        return model_utils.compile_transformer(
            args, model, [model.layers], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
        # ERNIE-Image latents are stored as BN-normalized patchified latents, no additional shift needed
        return latents

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ):
        model: ernie_image_model.ErnieImageTransformer2DModel = accelerator.unwrap_model(transformer)
        bsize = latents.shape[0]

        # Text embeddings from cache (varlen_text_embed -> text_embed as list)
        text_embed_list = batch["text_embed"]  # list[torch.Tensor], each [T_i, D]

        txt_seq_lens = [x.shape[0] for x in text_embed_list]
        max_len = max(txt_seq_lens)

        # Pad text embeddings
        text_embed_padded = [torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0])) for x in text_embed_list]
        text_bth = torch.stack(text_embed_padded, dim=0)  # [B, max_len, D]
        text_lens = torch.tensor(txt_seq_lens, device=latents.device, dtype=torch.long)

        # Prepare inputs
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        text_bth = text_bth.to(device=accelerator.device, dtype=network_dtype)
        text_lens = text_lens.to(device=accelerator.device)

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            text_bth.requires_grad_(True)

        with accelerator.autocast():
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                text_bth=text_bth,
                text_lens=text_lens,
            )

        # Flow matching target: v = noise - latents (dx/dsigma with x = (1-sigma)*latents + sigma*noise).
        # This matches the Diffusers ERNIE-Image pipeline convention where the scheduler applies
        # x_new = x + (sigma_next - sigma) * pred. Note this is the OPPOSITE sign of Z-Image.
        # Cast to network_dtype (typically float32) so backward grad dtypes match,
        # regardless of cached latent dtype (cache is bf16 for size reduction).
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        noise = noise.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        return model_pred, target


def ernie_image_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--text_encoder", type=str, default=None, help="Mistral3 text encoder .safetensors path")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer path (defaults to 'baidu/ERNIE-Image')")
    return parser


def main():
    parser = setup_parser_common()
    parser = ernie_image_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = ErnieImageNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
