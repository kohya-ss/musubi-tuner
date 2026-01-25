import argparse
import torch

from typing import Optional

from accelerate import Accelerator
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from musubi_tuner.flux_2 import flux2_models, flux2_utils
from musubi_tuner.flux_2.flux2_utils import Flux2ModelInfo
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
        self.model_version_info: Flux2ModelInfo | None = None

    # region model specific

    @property
    def architecture(self) -> str:
        if self.model_version_info is None:
            raise RuntimeError("model_version_info not set - call handle_model_specific_args first")
        return self.model_version_info.architecture

    @property
    def architecture_full_name(self) -> str:
        if self.model_version_info is None:
            raise RuntimeError("model_version_info not set - call handle_model_specific_args first")
        return self.model_version_info.architecture_full

    def handle_model_specific_args(self, args):
        # Get model version info first (dataclass with architecture info)
        self.model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
        logger.info(f"Model version: {args.model_version}, architecture: {self.model_version_info.architecture}")

        if args.fp8_text_encoder and self.model_version_info.qwen_variant is None:
            raise ValueError("--fp8_text_encoder is not supported for FLUX.2 dev (Mistral3). Remove this flag or use a Klein model.")

        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        self.default_discrete_flow_shift = None  # Use model defaults
        if not args.split_attn:
            logger.info(
                "Split attention will be automatically enabled if the control images are not resized to the same size as the target image."
                + " / 制御画像がターゲット画像と同じサイズにリサイズされていない場合、分割アテンションが自動的に有効になります。"
            )
        self._i2v_training = False
        self._control_training = False  # this means video training, not control image training
        self.default_guidance_scale = 4.0  # CFG scale for inference for base models

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        device = accelerator.device

        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompts = load_prompts(sample_prompts)

        if self.model_version_info is None:
            raise RuntimeError("model_version_info not set - call handle_model_specific_args first")
        model_version_info = self.model_version_info

        # Load text encoder (Mistral3 for dev, Qwen3 for Klein)
        te_dtype = torch.float8_e4m3fn if args.fp8_text_encoder else torch.bfloat16
        text_embedder = flux2_utils.load_text_embedder(
            model_version_info, args.text_encoder, dtype=te_dtype, device=device, disable_mmap=True
        )

        # Encode with text encoder
        encoder_name = "Qwen3" if model_version_info.qwen_variant else "Mistral3"
        logger.info(f"Encoding with {encoder_name} text encoder")

        # Use bfloat16 for autocast when text embedder uses FP8 (itemsize == 1 byte)
        autocast_dtype = torch.bfloat16 if text_embedder.dtype.itemsize == 1 else text_embedder.dtype

        sample_prompts_te_outputs = {}  # (prompt) -> ctx_vec
        with torch.amp.autocast(device_type=device.type, dtype=autocast_dtype), torch.no_grad():
            for prompt_dict in prompts:
                # add negative prompt if not present even if the model is guidance distilled for simplicity
                if "negative_prompt" not in prompt_dict:
                    prompt_dict["negative_prompt"] = " "

                for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", " ")]:
                    if p is None or p in sample_prompts_te_outputs:
                        continue

                    logger.info(f"cache Text Encoder outputs for prompt: {p}")
                    ctx_vec = text_embedder([p]).to(torch.bfloat16).cpu()  # [1, 512, 15360]

                    # save prompt cache
                    sample_prompts_te_outputs[p] = ctx_vec

        del text_embedder
        clean_memory_on_device(device)

        # prepare sample parameters
        sample_parameters = []
        for prompt_dict in prompts:
            prompt_dict_copy = prompt_dict.copy()

            prompt = prompt_dict.get("prompt", "")
            prompt_dict_copy["ctx_vec"] = sample_prompts_te_outputs[prompt]

            negative_prompt = prompt_dict.get("negative_prompt", " ")
            prompt_dict_copy["negative_ctx_vec"] = sample_prompts_te_outputs[negative_prompt]

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

        if self.model_version_info is None:
            raise RuntimeError("model_version_info not set - call handle_model_specific_args first")

        if self.model_version_info.guidance_distilled:
            ctx, ctx_ids = flux2_utils.prc_txt(ctx)  # [1, 512, 15360], [1, 512, 4]
        else:
            negative_ctx = sample_parameter.get("negative_ctx_vec")
            if negative_ctx is None:
                raise ValueError("negative_ctx_vec is required for non-guidance-distilled models")
            negative_ctx = negative_ctx.to(device=device, dtype=torch.bfloat16)  # [1, 512, 15360]
            ctx = torch.cat([negative_ctx, ctx], dim=0)
            ctx, ctx_ids = flux2_utils.prc_txt(ctx)  # [2, 512, 15360], [2, 512, 4]

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
                    control_latent = vae.encode(control_image_tensor.to(device, vae.dtype)).squeeze(0).to(torch.bfloat16)
                    control_latent_list.append(control_latent)

            ref_tokens, ref_ids = flux2_utils.pack_control_latent(control_latent_list)

            vae.to("cpu")
            clean_memory_on_device(device)

        # denoise
        flow_shift = sample_parameter.get("discrete_flow_shift", None) if "discrete_flow_shift" in sample_parameter else None
        timesteps = flux2_utils.get_schedule(sample_steps, x.shape[1], flow_shift)
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
        vae_path = vae_path or args.vae

        logger.info(f"Loading AE model from {vae_path}")
        if vae_dtype != torch.float32:
            logger.warning(f"FLUX.2 AE is always loaded in float32 for stability; ignoring vae_dtype={vae_dtype}")
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
        # FLUX.2 only supports torch attention currently (sdpa maps to torch).
        if attn_mode == "sdpa":
            attn_mode = "torch"

        if split_attn:
            raise ValueError("--split_attn is not supported for FLUX.2 training. Remove this flag.")

        if attn_mode != "torch":
            raise ValueError(
                f"Attention mode '{attn_mode}' (from --sdpa/--flash_attn/etc.) is not supported for FLUX.2 training. "
                "Use --sdpa (torch SDPA). Other modes require porting upstream's unified attention module."
            )

        model_version_info = flux2_utils.FLUX2_MODEL_INFO[args.model_version]
        model = flux2_utils.load_flow_model(
            device=accelerator.device,
            model_version_info=model_version_info,
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

        # pack latents - use passed-in latents (which went through scale_shift_latents) for consistency
        # with noisy_model_input computation; shape is (B, C, H, W) where C=128, H=height//16, W=width//16
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
        if ref_tokens is not None:
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

        # flow matching loss: target = v = (z_1 - z_0) = (noise - latents)
        # model_pred and target remain 4D (B, C, H, W) - apply_masked_loss() now handles 4D tensors
        target = noise - latents

        return model_pred, target

    # endregion model specific


def flux2_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Flux.2-dev specific parser setup"""
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled fp8 for DiT / DiTにスケーリングされたfp8を使う")
    parser.add_argument("--text_encoder", type=str, default=None, help="text encoder checkpoint path")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="use fp8 for Text Encoder model (Qwen3 only)")
    parser.add_argument("--fp8_te", action="store_true", help=argparse.SUPPRESS)  # deprecated alias for config/CLI compat
    flux2_utils.add_model_version_args(parser)
    return parser


def main():
    parser = setup_parser_common()
    parser = flux2_setup_parser(parser)

    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None  # set from mixed_precision
    if args.vae_dtype is None:
        args.vae_dtype = "float32"  # match upstream / AE float32 default
    if getattr(args, "fp8_te", False):
        logger.warning("--fp8_te is deprecated; use --fp8_text_encoder instead")
        args.fp8_text_encoder = True

    trainer = Flux2NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
