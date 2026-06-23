import argparse
import gc
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import (
    ARCHITECTURE_KREA2,
    ARCHITECTURE_KREA2_FULL,
)
from musubi_tuner.krea2 import krea2_model, krea2_utils
from musubi_tuner.qwen_image import qwen_image_autoencoder_kl
from musubi_tuner.hv_train_network import (
    DiTOutput,
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


class Krea2NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_frame_stride = 1

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_KREA2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_KREA2_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        vl_dtype = torch.float8_e4m3fn if args.fp8_vl else torch.bfloat16
        processor, text_encoder = krea2_utils.load_text_encoder(args.text_encoder, dtype=vl_dtype, device=device)

        logger.info("Encoding with Qwen3-VL")

        sample_prompts_te_outputs = {}

        with torch.amp.autocast(device_type=device.type, dtype=vl_dtype), torch.no_grad():
            for prompt_dict in prompts:
                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", " ")]:
                    if p is None or p in sample_prompts_te_outputs:
                        continue

                    logger.info(f"cache Text Encoder outputs for prompt: {p}")
                    hidden, mask = krea2_utils.get_krea2_prompt_embeds(
                        processor, text_encoder, [p], device=device, dtype=vl_dtype
                    )
                    txt_len = mask[0].to(dtype=torch.bool).sum().item()
                    hidden_i = hidden[0, :txt_len]
                    mask_i = mask[0, :txt_len]
                    sample_prompts_te_outputs[p] = (hidden_i, mask_i)

        del processor, text_encoder
        gc.collect()
        clean_memory_on_device(device)

        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()
            p = prompt_dict.get("prompt", "")
            neg_p = prompt_dict.get("negative_prompt", " ")

            hidden, mask = sample_prompts_te_outputs[p]
            prompt_dict_copy["vl_embed"] = hidden
            prompt_dict_copy["vl_mask"] = mask

            neg_hidden, neg_mask = sample_prompts_te_outputs[neg_p]
            prompt_dict_copy["negative_vl_embed"] = neg_hidden
            prompt_dict_copy["negative_vl_mask"] = neg_mask

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
        model: krea2_model.Krea2Transformer2DModel = transformer
        vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage = vae

        device = accelerator.device

        if cfg_scale is None:
            cfg_scale = 4.0

        vl_embed = sample_parameter["vl_embed"].to(device=device, dtype=torch.bfloat16)
        vl_mask = sample_parameter["vl_mask"].to(device=device, dtype=torch.bool)
        negative_vl_embed = sample_parameter["negative_vl_embed"].to(device=device, dtype=torch.bfloat16)
        negative_vl_mask = sample_parameter["negative_vl_mask"].to(device=device, dtype=torch.bool)

        num_channels_latents = krea2_utils.VAE_CHANNELS
        latents = krea2_utils.prepare_latents(
            1, num_channels_latents, height, width, torch.bfloat16, device, generator
        )

        img_h = height // krea2_utils.VAE_SCALE_FACTOR // krea2_utils.PATCH_SIZE
        img_w = width // krea2_utils.VAE_SCALE_FACTOR // krea2_utils.PATCH_SIZE
        img_shapes = [(1, img_h, img_w)]

        image_seq_len = latents.shape[1]
        timesteps = krea2_utils.get_timesteps(sample_steps, image_seq_len, device, is_turbo=False)

        do_cfg = do_classifier_free_guidance and cfg_scale > 0.0

        with tqdm(total=sample_steps, desc="Denoising steps") as pbar:
            for i in range(len(timesteps) - 1):
                t_curr = timesteps[i]
                t_prev = timesteps[i + 1]

                timestep = t_curr.expand(latents.shape[0]).to(latents.dtype)

                with torch.no_grad():
                    noise_pred = model(
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=vl_embed.unsqueeze(0),
                        encoder_hidden_states_mask=vl_mask.unsqueeze(0),
                        img_shapes=img_shapes,
                        txt_seq_lens=[vl_embed.shape[0]],
                    )

                if do_cfg:
                    with torch.no_grad():
                        neg_noise_pred = model(
                            hidden_states=latents,
                            timestep=timestep,
                            encoder_hidden_states=negative_vl_embed.unsqueeze(0),
                            encoder_hidden_states_mask=negative_vl_mask.unsqueeze(0),
                            img_shapes=img_shapes,
                            txt_seq_lens=[negative_vl_embed.shape[0]],
                        )
                    noise_pred = neg_noise_pred + cfg_scale * (noise_pred - neg_noise_pred)

                latents = latents + (t_prev - t_curr) * noise_pred
                pbar.update()

        latents = krea2_utils.unpack_latents(latents, height, width)

        vae.to(device)
        vae.eval()

        logger.info(f"Decoding image from latents: {latents.shape}")
        with torch.no_grad():
            pixels = vae.decode_to_pixels(latents.to(device, vae.dtype))

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.to(torch.float32).cpu()
        pixels = pixels.unsqueeze(2)  # L C 1 H W
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading VAE model from {args.vae}")
        vae = krea2_utils.load_vae(args.vae, device="cpu", disable_mmap=True)
        vae.eval()
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
        model = krea2_model.load_krea2_model(
            accelerator.device,
            dit_path,
            attn_mode,
            split_attn,
            loading_device,
            dit_weight_dtype,
            fp8_scaled=args.fp8_scaled,
            disable_numpy_memmap=args.disable_numpy_memmap,
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: krea2_model.Krea2Transformer2DModel = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.blocks], disable_linear=self.blocks_to_swap > 0
        )

    def scale_shift_latents(self, latents):
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
        **kwargs,
    ) -> DiTOutput:
        model: krea2_model.Krea2Transformer2DModel = transformer

        bsize = latents.shape[0]
        latents = batch["latents"]  # B, C, 1, H, W
        assert latents.shape[2] == 1, "Expected latents shape B, C, 1, H, W for Krea2"

        lat_h = latents.shape[3]
        lat_w = latents.shape[4]
        noisy_model_input = krea2_utils.pack_latents(noisy_model_input)
        img_seq_len = noisy_model_input.shape[1]

        # context: multi-layer hidden states (B, L, 12, 2560)
        vl_embed = batch["vl_embed"]  # list of (L, 12, 2560)
        txt_seq_lens = [x.shape[0] for x in vl_embed]

        max_len = max(txt_seq_lens)
        vl_embed = [torch.nn.functional.pad(x, (0, 0, 0, 0, 0, max_len - x.shape[0])) for x in vl_embed]
        vl_embed = torch.stack(vl_embed, dim=0)  # B, L, 12, 2560

        # attention mask
        vl_mask = torch.zeros(bsize, max_len, dtype=torch.bool, device=vl_embed[0].device)
        for i, x in enumerate(txt_seq_lens):
            vl_mask[i, :x] = True

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            vl_embed.requires_grad_(True)

        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        vl_embed = vl_embed.to(device=accelerator.device, dtype=network_dtype)
        vl_mask = vl_mask.to(device=accelerator.device)

        img_h = lat_h // krea2_utils.PATCH_SIZE
        img_w = lat_w // krea2_utils.PATCH_SIZE
        img_shapes = [(1, img_h, img_w)]

        timesteps = timesteps / 1000.0
        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=vl_embed,
                encoder_hidden_states_mask=vl_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

        model_pred = krea2_utils.unpack_latents(
            model_pred,
            lat_h * krea2_utils.VAE_SCALE_FACTOR,
            lat_w * krea2_utils.VAE_SCALE_FACTOR,
        )

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        return DiTOutput(pred=model_pred, target=target)

    # endregion model specific


def krea2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder (Qwen3-VL 4B) checkpoint path")
    parser.add_argument("--fp8_vl", action="store_true", help="use fp8 for Text Encoder model")
    # Default to the Krea 2 resolution-dependent flow-matching shift so training matches inference
    # (krea2_utils.get_timesteps). Users can still override with --timestep_sampling.
    parser.set_defaults(timestep_sampling="krea2_shift")
    return parser


def main():
    parser = setup_parser_common()
    parser = krea2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16"
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"

    trainer = Krea2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
