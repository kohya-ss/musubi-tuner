import argparse
import torch

from typing import Optional

from accelerate import Accelerator
from einops import rearrange
from PIL import Image

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_FLUX_2, ARCHITECTURE_FLUX_2_FULL
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
        return ARCHITECTURE_FLUX_2

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_FLUX_2_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        if not args.split_attn:
            logger.info(
                "Split attention will be automatically enabled if the control images are not resized to the same size as the target image."
                + " / 制御画像がターゲット画像と同じサイズにリサイズされていない場合、分割アテンションが自動的に有効になります。"
            )
        self._i2v_training = False
        self._control_training = False  # this means video training, not control image training
        self.default_guidance_scale = 2.5  # embeded guidance scale for inference

    @staticmethod
    def process_sample_prompts(
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        # Load Mistral 3
        m3_dtype = torch.float8e4m3fn if args.fp8_te else torch.bfloat16
        text_embedder = flux2_utils.load_textembedder(
            args.model_version, args.text_encoder, dtype=m3_dtype, device=device, disable_mmap=True
        )

        # Encode with Mistral 3 text encoders
        logger.info("Encoding with Mistral 3 text encoders")

        sample_prompts_te_outputs = {}  # (prompt) -> (t5, clip)
        with torch.amp.autocast(device_type=device.type, dtype=m3_dtype), torch.no_grad():
            for prompt_dict in prompts:
                prompt = prompt_dict.get("prompt", "")
                if prompt is None or prompt in sample_prompts_te_outputs:
                    continue

                # encode prompt
                logger.info(f"cache Text Encoder outputs for prompt: {prompt}")
                if flux2_utils.FLUX2_MODEL_INFO[args.model_version]["guidance_distilled"]:
                    ctx_vec = text_embedder([prompt])  # [1, 512, 15360]
                else:
                    ctx_empty = text_embedder([""]).to(torch.bfloat16)
                    ctx_prompt = text_embedder([prompt]).to(torch.bfloat16)
                    ctx_vec = torch.cat([ctx_empty, ctx_prompt], dim=0)

                ctx_vec = ctx_vec.cpu()

                # save prompt cache
                sample_prompts_te_outputs[prompt] = (ctx_vec,)

        del text_embedder
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            prompt = prompt_dict.get("prompt", "")
            prompt_dict_copy["ctx_vec"] = sample_prompts_te_outputs[prompt][0]

            sample_parameters.append(prompt_dict_copy)

        clean_memory_on_device(accelerator.device)

        return sample_parameters

    @staticmethod
    def do_inference(
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
        ctx, ctx_ids = flux2_utils.batched_prc_txt(ctx)  # [1, 512, 15360], [1, 512, 4]

        # Initialize latents
        packed_latent_height, packed_latent_width = height // 16, width // 16
        randn = torch.randn(
            (1, 128, packed_latent_height, packed_latent_width),  # [1, 128, 52, 78]
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        x, x_ids = flux2_utils.batched_prc_img(randn)  # [1, 4056, 128], [1, 4056, 4]

        vae.to(device)
        vae.eval()

        # prepare control latent
        ref_tokens = None
        ref_ids = None
        if "control_image_path" in sample_parameter:
            img_ctx = [Image.open(input_image) for input_image in sample_parameter["control_image_path"]]
            ref_tokens, ref_ids = flux2_utils.encode_image_refs(vae, img_ctx)

        vae.to("cpu")
        clean_memory_on_device(device)

        # denoise
        timesteps = flux2_utils.get_schedule(sample_steps, x.shape[1])
        if flux2_utils.FLUX2_MODEL_INFO[args.model_version]["guidance_distilled"]:
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

    @staticmethod
    def load_vae(args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        vae_path = args.vae

        logger.info(f"Loading AE model from {vae_path}")
        ae = flux2_utils.load_ae(vae_path, dtype=torch.float32, device="cpu", disable_mmap=True)
        return ae

    @staticmethod
    def load_transformer(
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: Optional[torch.dtype],
    ):
        model = flux2_utils.load_flow_model(
            ckpt_path=args.dit,
            dtype=None,
            device=loading_device,
            disable_mmap=True,
            attn_mode=attn_mode,
            split_attn=split_attn,
            loading_device=loading_device,
            fp8_scaled=args.fp8_scaled,
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: flux2_models.Flux2 = transformer
        return model_utils.compile_transformer(
            args, transformer, [transformer.double_blocks, transformer.single_blocks], disable_linear=self.blocks_to_swap > 0
        )

    @staticmethod
    def scale_shift_latents(latents):
        return latents

    @staticmethod
    def call_dit(
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

        # control
        num_control_images = 0
        control_keys = []
        while True:
            key = f"latents_control_{num_control_images}"
            if key in batch:
                control_keys.append(key)
                num_control_images += 1
            else:
                break

        # pack latents
        latents = batch["latents"]  # B, C, H, W  # same for noisy_model_input (C = 128, H = height//16, ...)
        packed_latent_height = latents.shape[2]
        packed_latent_width = latents.shape[3]
        noisy_model_input, img_ids = flux2_utils.batched_prc_img(noisy_model_input)  # (B, HW, C), (B, HW, 4)

        ref_tokens, ref_ids = None, None
        if num_control_images:
            assert bsize == 1, "Flux 2 can't be trained with higher batch size since ref images may different size and number"
            encoded_refs = [batch[k][0] for k in control_keys]  # list[(C, H, W)]

            scale = 10
            # Create time offsets for each reference
            t_off = [scale + scale * t for t in torch.arange(0, len(encoded_refs))]
            t_off = [t.view(-1) for t in t_off]
            # Process with position IDs
            ref_tokens, ref_ids = flux2_utils.listed_prc_img(encoded_refs, t_coord=t_off)  # list[(HW, C)], list[(HW, 4)]
            # Concatenate all references along sequence dimension
            ref_tokens = torch.cat(ref_tokens, dim=0)  # (total_ref_tokens, C)
            ref_ids = torch.cat(ref_ids, dim=0)  # (total_ref_tokens, 4)
            # Add batch dimension
            ref_tokens = ref_tokens.unsqueeze(0).to(torch.bfloat16)  # (1, total_ref_tokens, C)
            ref_ids = ref_ids.unsqueeze(0)  # (1, total_ref_tokens, 4)

        # context
        ctx_vec = batch["ctx_vec"]  # B, T, D  # [1, 512, 15360]
        ctx, ctx_ids = flux2_utils.batched_prc_txt(ctx_vec)  # [1, 512, 15360], [1, 512, 4]

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            ctx.requires_grad_(True)
            if num_control_images:
                ref_tokens.requires_grad_(True)

        # call DiT
        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        img_ids = img_ids.to(device=accelerator.device)
        if ref_tokens:
            ref_tokens = ref_tokens.to(device=accelerator.device, dtype=network_dtype)
            ref_ids = ref_ids.to(device=accelerator.device)
        ctx = ctx.to(device=accelerator.device, dtype=network_dtype)
        ctx_ids = ctx_ids.to(device=accelerator.device)

        # use 1.0 as guidance scale for FLUX.2 training
        guidance_vec = torch.full((bsize,), 1.0, device=accelerator.device, dtype=network_dtype)

        img_input = noisy_model_input
        img_input_ids = img_ids
        if ref_tokens is not None:
            img_input = torch.cat((img_input, ref_tokens), dim=1)
            img_input_ids = torch.cat((img_input_ids, ref_ids), dim=1)

        timesteps = timesteps / 1000.0
        model_pred = model(
            x=img_input,  # [1, 8192, 128]
            x_ids=img_input_ids,
            timesteps=timesteps,
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=guidance_vec,
        )  # [1, 8192, 128]
        model_pred = model_pred[:, : noisy_model_input.shape[1]]  # [1, 4096, 128]

        # unpack height/width latents
        model_pred = rearrange(model_pred, "b (h w) c -> b c h w", h=packed_latent_height, w=packed_latent_width)

        # flow matching loss
        target = noise - latents

        return model_pred, target

    # endregion model specific


def flux2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flux.2-dev specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder checkpoint path")
    parser.add_argument("--fp8_te", action="store_true", help="use fp8 for Text Encoder model")
    flux2_utils.add_model_version_args(parser)
    return parser


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "bfloat16"  # make bfloat16 as default for VAE

    trainer = Flux2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
