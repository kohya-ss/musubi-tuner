import argparse
import logging
from typing import Optional

import torch
from accelerate import Accelerator
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_LENS, ARCHITECTURE_LENS_FULL
from musubi_tuner.lens import lens_text_encoder, lens_utils
from musubi_tuner.lens.lens_model import LensTransformer2DModel
from musubi_tuner.training.accelerator_setup import clean_memory_on_device
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LensNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_LENS

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_LENS_FULL

    def handle_model_specific_args(self, args):
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("Lens MVP supports bf16/fp16 DiT training only; fp8/mxfp8 training is out of scope.")
        if args.blocks_to_swap and args.blocks_to_swap > 0:
            raise ValueError("Lens MVP does not support blocks_to_swap yet.")
        self.dit_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 5.0
        self.default_discrete_flow_shift = None

    def _encode_prompts(self, args: argparse.Namespace, device: torch.device, prompts: list[str]):
        te_dtype = torch.bfloat16 if args.text_encoder_dtype is None else model_utils.str_to_dtype(args.text_encoder_dtype)
        text_embedder = lens_text_encoder.load_lens_text_embedder(
            args.text_encoder,
            dtype=te_dtype,
            device=device,
            text_encoder_config=args.text_encoder_config,
            tokenizer=args.tokenizer,
            disable_mmap=args.disable_numpy_memmap,
        )
        outputs = {}
        for prompt in prompts:
            if prompt in outputs:
                continue
            with torch.no_grad():
                features, mask = text_embedder([prompt])
            outputs[prompt] = ([feat.cpu() for feat in features], mask.cpu())
        del text_embedder
        clean_memory_on_device(device)
        return outputs

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        device = accelerator.device
        logger.info(f"cache Text Encoder outputs for sample prompt: {sample_prompts}")
        prompt_dicts = load_prompts(sample_prompts)

        all_prompts = []
        for prompt_dict in prompt_dicts:
            if "negative_prompt" not in prompt_dict:
                prompt_dict["negative_prompt"] = ""
            for prompt in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                if prompt is not None:
                    all_prompts.append(prompt)

        encoded = self._encode_prompts(args, device, all_prompts)
        sample_parameters = []
        for prompt_dict in prompt_dicts:
            prompt_dict_copy = prompt_dict.copy()
            prompt = prompt_dict.get("prompt", "")
            features, mask = encoded[prompt]
            prompt_dict_copy["lens_ctx"] = features
            prompt_dict_copy["lens_ctx_mask"] = mask

            negative_prompt = prompt_dict.get("negative_prompt", "")
            negative_features, negative_mask = encoded[negative_prompt]
            prompt_dict_copy["negative_lens_ctx"] = negative_features
            prompt_dict_copy["negative_lens_ctx_mask"] = negative_mask
            sample_parameters.append(prompt_dict_copy)

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
        del discrete_flow_shift, frame_count, do_classifier_free_guidance, image_path, control_video_path
        model: LensTransformer2DModel = accelerator.unwrap_model(transformer)
        device = accelerator.device
        cfg = cfg_scale if cfg_scale is not None else guidance_scale
        if cfg is None:
            cfg = 5.0
        steps = sample_steps if sample_steps is not None else 20

        prompt_features = [feat.to(device=device, dtype=dit_dtype) for feat in sample_parameter["lens_ctx"]]
        prompt_mask = sample_parameter["lens_ctx_mask"].to(device=device, dtype=torch.bool)
        negative_features = [feat.to(device=device, dtype=dit_dtype) for feat in sample_parameter["negative_lens_ctx"]]
        negative_mask = sample_parameter["negative_lens_ctx_mask"].to(device=device, dtype=torch.bool)
        prompt_features, prompt_mask, negative_features, negative_mask = lens_utils.align_text_feature_lists(
            prompt_features, prompt_mask, negative_features, negative_mask
        )
        encoder_features = [torch.cat([pf, nf], dim=0) for pf, nf in zip(prompt_features, negative_features)]
        encoder_mask = torch.cat([prompt_mask, negative_mask], dim=0)

        latent_h, latent_w = height // 16, width // 16
        latents = randn_tensor(
            (1, latent_h * latent_w, 128),
            generator=generator,
            device=device,
            dtype=dit_dtype,
        )
        sigmas = lens_utils.get_lens_sigmas(steps, latent_h * latent_w, device)
        img_shapes = [(1, latent_h, latent_w)]
        for i in range(steps):
            timestep = (sigmas[i] * 1000.0).expand(2).to(dtype=dit_dtype)
            hidden_states = latents.repeat(2, 1, 1)
            with accelerator.autocast(), torch.no_grad():
                noise = model(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_features,
                    encoder_hidden_states_mask=encoder_mask,
                    timestep=timestep / 1000.0,
                    img_shapes=img_shapes,
                )
            cond, uncond = noise.chunk(2)
            combined = uncond + cfg * (cond - uncond)
            cond_norm = torch.norm(cond, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True)
            scale = torch.where(combined_norm > 0, cond_norm / combined_norm.clamp_min(1e-12), torch.ones_like(combined_norm))
            latents = lens_utils.euler_step(combined * scale, latents, sigmas, i)

        latent = lens_utils.unpack_latents(latents, latent_h, latent_w).to(vae.dtype)
        vae.to(device)
        vae.eval()
        with torch.no_grad():
            pixels = vae.decode(latent)
        pixels = pixels.to(torch.float32).cpu()
        pixels = (pixels / 2 + 0.5).clamp(0, 1)
        vae.to("cpu")
        clean_memory_on_device(device)
        return pixels.unsqueeze(2)

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        logger.info(f"Loading Lens VAE from {vae_path}")
        return lens_utils.load_lens_vae(vae_path, dtype=vae_dtype, device="cpu")

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
        if split_attn:
            raise ValueError("Lens MVP supports SDPA attention only; split_attn is not implemented.")
        if attn_mode not in ("torch", "sdpa"):
            raise ValueError("Lens MVP supports --sdpa only.")
        model = lens_utils.load_lens_transformer(
            dit_path,
            dtype=dit_weight_dtype,
            device=loading_device,
            disable_mmap=args.disable_numpy_memmap,
        )
        return model

    def compile_transformer(self, args, transformer):
        transformer: LensTransformer2DModel = transformer
        return model_utils.compile_transformer(args, transformer, [transformer.transformer_blocks], disable_linear=False)

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
        del kwargs
        model: LensTransformer2DModel = transformer
        latent_h, latent_w = latents.shape[2], latents.shape[3]

        noisy_model_input = rearrange(noisy_model_input, "b c h w -> b (h w) c")
        layer_features = []
        lens_mask = None
        for i in range(4):
            tensor, mask = lens_utils.pad_lens_text_features(batch[f"lens_ctx_{i}"])
            layer_features.append(tensor)
            if lens_mask is None:
                lens_mask = mask

        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for tensor in layer_features:
                tensor.requires_grad_(True)

        noisy_model_input = noisy_model_input.to(device=accelerator.device, dtype=network_dtype)
        layer_features = [tensor.to(device=accelerator.device, dtype=network_dtype) for tensor in layer_features]
        lens_mask = lens_mask.to(device=accelerator.device)
        timesteps = timesteps.to(device=accelerator.device, dtype=network_dtype) / 1000.0

        with accelerator.autocast():
            model_pred = model(
                hidden_states=noisy_model_input,
                encoder_hidden_states=layer_features,
                encoder_hidden_states_mask=lens_mask,
                timestep=timesteps,
                img_shapes=[(1, latent_h, latent_w)],
            )
        model_pred = rearrange(model_pred, "b (h w) c -> b c h w", h=latent_h, w=latent_w)

        latents = latents.to(device=accelerator.device, dtype=network_dtype)
        target = noise - latents
        return DiTOutput(pred=model_pred, target=target)


def lens_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(network_module="networks.lora_lens")
    parser.add_argument("--fp8_scaled", action="store_true", help="not supported for Lens MVP")
    parser.add_argument("--text_encoder", type=str, default=None, help="Lens GPT-OSS text encoder safetensors path")
    parser.add_argument("--text_encoder_config", type=str, default=None, help="directory containing GPT-OSS config.json")
    parser.add_argument("--tokenizer", type=str, default=None, help="directory containing Lens tokenizer files")
    parser.add_argument("--text_encoder_dtype", type=str, default=None, help="text encoder dtype, default bfloat16")
    return parser


def main():
    parser = setup_parser_common()
    parser = lens_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = None
    if args.vae_dtype is None:
        args.vae_dtype = "float32"
    if args.sample_prompts and args.text_encoder is None:
        raise ValueError("--text_encoder is required when --sample_prompts is used")

    trainer = LensNetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
