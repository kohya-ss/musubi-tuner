import argparse
from typing import Optional
import torch


from accelerate import Accelerator
from einops import rearrange
from diffusers.utils.torch_utils import randn_tensor

from musubi_tuner.flux_2 import flux2_models, flux2_utils
from musubi_tuner.hv_train_network import (
    NetworkTrainer,
    load_prompts,
    clean_memory_on_device,
    setup_parser_common,
    read_config_from_file,
)

import logging

from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Flux2NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    # region model specific

    @property
    def architecture(self) -> str:
        return self.model_version_info.architecture

    @property
    def architecture_full_name(self) -> str:
        return self.model_version_info.architecture_full

    def handle_model_specific_args(self, args):
        self.model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self._i2v_training = False
        self._control_training = False  # this means video training, not control image training
        self.default_guidance_scale = 4.0  # CFG scale for inference for base models
        self.default_discrete_flow_shift = None  # Use FLUX.2 shift as default

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Text Encoder (Mistral 3 or Qwen-3)
        te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
        text_embedder = flux2_utils.load_text_embedder(
            self.model_version_info, args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True
        )

        # Encode with Text Encoder (Mistral 3 or Qwen-3)
        logger.info("Encoding with Text Encoder (Mistral 3 or Qwen-3)...")

        sample_prompts_te_outputs = {}  # prompt -> encoded tensor
        for prompt_dict in prompts:
            # add negative prompt if not present even if the model is guidance distilled for simplicity
            if "negative_prompt" not in prompt_dict:
                prompt_dict["negative_prompt"] = " "

            for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", " ")]:
                if p is None or p in sample_prompts_te_outputs:
                    continue

                # encode prompt
                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                with torch.no_grad():
                    if te_dtype.itemsize == 1:
                        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                            ctx_vec = text_embedder([p])  # [1, 512, 15360]
                    else:
                        ctx_vec = text_embedder([p])  # [1, 512, 15360]
                ctx_vec = ctx_vec.cpu()

                # save prompt cache
                sample_prompts_te_outputs[p] = ctx_vec

        del text_embedder
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            p = prompt_dict.get("prompt", "")
            prompt_dict_copy["ctx_vec"] = sample_prompts_te_outputs[p]
            p = prompt_dict.get("negative_prompt", " ")
            prompt_dict_copy["negative_ctx_vec"] = sample_prompts_te_outputs[p]

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
        """architecture dependent inference"""
        model: flux2_models.Flux2 = transformer
        device = accelerator.device

        # Get embeddings
        ctx = sample_parameter["ctx_vec"].to(device=device, dtype=torch.bfloat16)  # [1, 512, 15360]
        ctx, ctx_ids = flux2_utils.prc_txt(ctx)  # [1, 512, 15360], [1, 512, 4]
        negative_ctx = sample_parameter.get("negative_ctx_vec").to(device=device, dtype=torch.bfloat16)
        negative_ctx, negative_ctx_ids = flux2_utils.prc_txt(negative_ctx)

        # Initialize latents
        packed_latent_height, packed_latent_width = height // 16, width // 16
        latents = randn_tensor(
            (1, 128, packed_latent_height, packed_latent_width),  # [1, 128, 52, 78]
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        x, x_ids = flux2_utils.prc_img(latents)  # [1, 4056, 128], [1, 4056, 4]

        # prepare control latent
        ref_tokens = None
        ref_ids = None
        if "control_image_path" in sample_parameter:
            vae.to(device)
            vae.eval()

            control_image_paths = sample_parameter["control_image_path"]
            limit_size = (2024, 2024) if len(control_image_paths) == 1 else (1024, 1024)
            control_latent_list = []
            with torch.no_grad():
                for image_path in control_image_paths:
                    control_image_tensor, _, _ = flux2_utils.preprocess_control_image(image_path, limit_size)
                    control_latent = vae.encode(control_image_tensor.to(device, vae.dtype))
                    control_latent_list.append(control_latent.squeeze(0))

            ref_tokens, ref_ids = flux2_utils.pack_control_latent(control_latent_list)

            vae.to("cpu")
            clean_memory_on_device(device)

        # denoise
        timesteps = flux2_utils.get_schedule(sample_steps, x.shape[1], discrete_flow_shift)
        if self.model_version_info.guidance_distilled:
            x = flux2_utils.denoise(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=guidance_scale,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
        else:
            x = flux2_utils.denoise_cfg(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                negative_ctx,
                negative_ctx_ids,
                timesteps=timesteps,
                guidance=guidance_scale,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
        x = torch.cat(flux2_utils.scatter_ids(x, x_ids)).squeeze(2)
        latent = x.to(vae.dtype)
        del x

        # Move VAE to the appropriate device for sampling
        vae.to(device)
        vae.eval()

        # Decode latents to video
        logger.info(f"Decoding video from latents: {latent.shape}")
        with torch.no_grad():
            pixels = vae.decode(latent)  # decode to pixels
        del latent

        logger.info("Decoding complete")
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)  # -1 to 1 -> 0 to 1

        vae.to("cpu")
        clean_memory_on_device(device)

        pixels = pixels.unsqueeze(2)  # add a dummy dimension for video frames, B C H W -> B C 1 H W
        return pixels

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading AE model from {vae_path}")
        ae = flux2_utils.load_ae(vae_path, dtype=vae_dtype, device="cpu", disable_mmap=True)
        return ae

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
        model = flux2_utils.load_flow_model(
            accelerator.device,
            model_version_info=self.model_version_info,
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
        transformer: flux2_models.Flux2 = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks, transformer.single_blocks], disable_linear=self.blocks_to_swap > 0
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
    ):
        model: flux2_models.Flux2 = transformer

        bsize = latents.shape[0]
        # pack latents
        packed_latent_height = latents.shape[2]
        packed_latent_width = latents.shape[3]
        noisy_model_input, img_ids = flux2_utils.prc_img(noisy_model_input)  # (B, HW, C), (B, HW, 4)

        # control
        num_control_images = 0
        ref_tokens, ref_ids = None, None
        if "latents_control_0" in batch:
            control_latents: list[torch.Tensor] = []
            while True:
                key = f"latents_control_{num_control_images}"
                if key in batch:
                    control_latents.append(batch[key])  # list of (B, C, H, W)
                    num_control_images += 1
                else:
                    break

            ref_tokens, ref_ids = flux2_utils.pack_control_latent(control_latents)

        # context
        ctx_vec = batch["ctx_vec"]  # B, T, D = B, 512, 15360]
        ctx, ctx_ids = flux2_utils.prc_txt(ctx_vec)  # [B, 512, 15360], [B, 512, 4]

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            ctx.requires_grad_(True)
            if ref_tokens is not None:
                ref_tokens.requires_grad_(True)

        # call DiT
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        img_ids = img_ids.to(device=accelerator.device)
        if ref_tokens is not None:
            ref_tokens = ref_tokens.to(device=accelerator.device, dtype=network_dtype)
            ref_ids = ref_ids.to(device=accelerator.device)
        ctx = ctx.to(device=accelerator.device, dtype=network_dtype)
        ctx_ids = ctx_ids.to(device=accelerator.device)

        # use 1.0 as guidance scale for FLUX.2 non-base training
        guidance_vec = torch.full((bsize,), 1.0, device=accelerator.device, dtype=network_dtype)

        img_input = noisy_model_input  # [B, HW, C]
        img_input_ids = img_ids  # [B, HW, 4]
        if ref_tokens is not None:
            img_input = torch.cat((img_input, ref_tokens), dim=1)
            img_input_ids = torch.cat((img_input_ids, ref_ids), dim=1)

        timesteps = timesteps / 1000.0
        model_pred = model(x=img_input, x_ids=img_input_ids, timesteps=timesteps, ctx=ctx, ctx_ids=ctx_ids, guidance=guidance_vec)
        model_pred = model_pred[:, : noisy_model_input.shape[1]]  # [B, 4096, 128]

        # unpack height/width latents
        model_pred = rearrange(model_pred, "b (h w) c -> b c h w", h=packed_latent_height, w=packed_latent_width)

        # flow matching loss
        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents

        return model_pred, target

    # endregion model specific


def flux2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flux.2-dev specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder checkpoint path")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="use fp8 for Text Encoder model")
    flux2_utils.add_model_version_args(parser)
    return parser


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "float32"  # make float32 as default for VAE

    trainer = Flux2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
