import argparse
import gc
import logging
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4, ARCHITECTURE_IDEOGRAM4_FULL
from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.ideogram4_scheduler import get_schedule_for_resolution
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Ideogram4NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.unconditional_transformer = None

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_IDEOGRAM4

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_IDEOGRAM4_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 7.0
        self.default_discrete_flow_shift = 1.0
        self.args_ideogram4_timestep_mu = args.ideogram4_timestep_mu
        self.args_ideogram4_timestep_std = args.ideogram4_timestep_std
        if args.blocks_to_swap is not None and args.blocks_to_swap > 33:
            raise ValueError("--blocks_to_swap for Ideogram 4 must be <= 33")
        if args.sample_prompts and (args.unconditional_dit is None or args.text_encoder is None or args.vae is None):
            raise ValueError("--sample_prompts for Ideogram 4 requires --unconditional_dit, --text_encoder, and --vae")

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        prompts = load_prompts(sample_prompts)
        device = accelerator.device

        logger.info("Encoding Ideogram 4 sample prompts")
        tokenizer = ideogram4_utils.load_ideogram4_tokenizer()
        text_encoder = ideogram4_utils.load_ideogram4_text_encoder(
            args.text_encoder,
            device=device,
            dtype=torch.bfloat16,
            disable_mmap=args.disable_numpy_memmap,
        )

        sample_parameters = []
        with torch.no_grad():
            for prompt_dict in prompts:
                prompt = prompt_dict.get("prompt", "")
                if prompt_dict.get("negative_prompt") is not None:
                    logger.warning("Ideogram 4 v1 ignores negative_prompt in sample prompts.")
                if args.validate_caption_structure:
                    ideogram4_utils.validate_prompt(prompt, warn_only=args.warn_on_caption_issues)
                prompt_dict = prompt_dict.copy()
                prompt_dict["i4_llm_features"] = ideogram4_utils.encode_prompt_to_features(
                    tokenizer, text_encoder, prompt, device
                ).cpu()
                sample_parameters.append(prompt_dict)

        del tokenizer, text_encoder
        gc.collect()
        clean_memory_on_device(device)
        return sample_parameters

    def on_before_sample_images(
        self,
        accelerator: Accelerator,
        args,
        epoch,
        steps,
        vae,
        transformer,
        network,
        sample_parameters,
        dit_dtype,
    ) -> None:
        if self.unconditional_transformer is None:
            logger.info(f"Loading Ideogram 4 unconditional DiT from {args.unconditional_dit}")
            self.unconditional_transformer = ideogram4_utils.load_ideogram4_transformer(
                args.unconditional_dit,
                device=accelerator.device,
                dtype=dit_dtype,
                expected_model_type=ideogram4_utils.IDEOGRAM4_UNCOND_MODEL_TYPE,
                disable_mmap=args.disable_numpy_memmap,
            )

    def on_after_sample_images(
        self,
        accelerator: Accelerator,
        args,
        epoch,
        steps,
        vae,
        transformer,
        network,
        sample_parameters,
        dit_dtype,
    ) -> None:
        if self.unconditional_transformer is not None:
            self.unconditional_transformer.to("cpu")
            del self.unconditional_transformer
            self.unconditional_transformer = None
            clean_memory_on_device(accelerator.device)

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
        del discrete_flow_shift, sample_steps, frame_count, generator, do_classifier_free_guidance, guidance_scale, cfg_scale
        height = (height // ideogram4_utils.IDEOGRAM4_IMAGE_PATCH) * ideogram4_utils.IDEOGRAM4_IMAGE_PATCH
        width = (width // ideogram4_utils.IDEOGRAM4_IMAGE_PATCH) * ideogram4_utils.IDEOGRAM4_IMAGE_PATCH
        if self.unconditional_transformer is None:
            raise RuntimeError("unconditional transformer is not loaded for Ideogram 4 sampling")
        sampler_preset = sample_parameter.get("sampler_preset", args.sampler_preset)
        features = sample_parameter["i4_llm_features"].to(torch.float32)
        vae.to(accelerator.device)
        vae.eval()
        images = ideogram4_utils.generate_images(
            conditional_transformer=transformer,
            unconditional_transformer=self.unconditional_transformer,
            autoencoder=vae,
            text_features=[features],
            height=height,
            width=width,
            sampler_preset=sampler_preset,
            device=accelerator.device,
            seed=sample_parameter.get("seed", None),
            show_progress=True,
        )
        arr = np.asarray(images[0]).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        return tensor

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del vae_path
        logger.info(f"Loading Ideogram 4 VAE from {args.vae}")
        vae = ideogram4_utils.load_ideogram4_autoencoder(
            args.vae,
            device="cpu",
            dtype=vae_dtype,
            disable_mmap=args.disable_numpy_memmap,
        )
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
        del accelerator, attn_mode, split_attn, dit_weight_dtype
        return ideogram4_utils.load_ideogram4_transformer(
            dit_path,
            device=loading_device,
            dtype=model_utils.str_to_dtype(args.dit_dtype),
            expected_model_type=ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE,
            disable_mmap=args.disable_numpy_memmap,
        )

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(args, transformer, [transformer.layers], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        return latents

    def _sample_ideogram_timesteps(self, batch_size: int, image_height: int, image_width: int, device: torch.device, dtype: torch.dtype):
        schedule = get_schedule_for_resolution(
            (image_height, image_width),
            known_mean=self.args_ideogram4_timestep_mu,
            std=self.args_ideogram4_timestep_std,
        )
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        t = schedule(u).to(device=device, dtype=dtype)
        return t

    def process_batch(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer,
        network,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        noise: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        vae,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del network, noise, noise_scheduler, vae, global_step
        latents = latents.to(device=accelerator.device, dtype=dit_dtype)
        token_grid = ideogram4_utils.normalize_token_grid(latents)
        noise = torch.randn_like(token_grid)
        image_height = latents.shape[-2] * ideogram4_utils.IDEOGRAM4_AE_SCALE_FACTOR
        image_width = latents.shape[-1] * ideogram4_utils.IDEOGRAM4_AE_SCALE_FACTOR
        t = self._sample_ideogram_timesteps(token_grid.shape[0], image_height, image_width, accelerator.device, dit_dtype)
        noisy_model_input = (1.0 - t.view(-1, 1, 1, 1)) * token_grid + t.view(-1, 1, 1, 1) * noise
        timesteps = t * 1000.0

        output = self.call_dit(
            args,
            accelerator,
            transformer,
            token_grid,
            batch,
            noise,
            noisy_model_input,
            timesteps,
            network_dtype,
        )
        loss = torch.nn.functional.mse_loss(output.pred.to(network_dtype), output.target.to(network_dtype), reduction="mean")
        return loss, {}

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
        model = transformer
        bsize, _, grid_h, grid_w = noisy_model_input.shape
        text_features = [x.to(dtype=network_dtype) for x in batch["i4_llm_features"]]
        image_height = grid_h * ideogram4_utils.IDEOGRAM4_IMAGE_PATCH
        image_width = grid_w * ideogram4_utils.IDEOGRAM4_IMAGE_PATCH
        inputs = ideogram4_utils.build_sequence_inputs_from_features(text_features, image_height, image_width, device=accelerator.device)

        image_tokens = ideogram4_utils.flatten_token_grid(noisy_model_input).to(device=accelerator.device, dtype=network_dtype)
        text_padding = torch.zeros(
            bsize,
            int(inputs["max_text_tokens"]),
            model.config.in_channels,
            dtype=network_dtype,
            device=accelerator.device,
        )
        x = torch.cat([text_padding, image_tokens], dim=1)
        llm_features = inputs["llm_features"].to(dtype=network_dtype)
        if args.gradient_checkpointing:
            x.requires_grad_(True)
            llm_features.requires_grad_(True)

        model_pred = model(
            llm_features=llm_features,
            x=x,
            t=timesteps.to(accelerator.device, dtype=torch.float32) / 1000.0,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        model_pred = model_pred[:, int(inputs["max_text_tokens"]) :]
        model_pred = ideogram4_utils.unflatten_token_grid(model_pred, grid_h, grid_w)
        target = noise - latents
        return DiTOutput(pred=model_pred, target=target)


def ideogram4_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--unconditional_dit", type=str, default=None, help="unconditional Ideogram 4 DiT safetensors path")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL BF16 text encoder safetensors path; only needed for sampling")
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for Ideogram 4 DiT, default is bfloat16")
    parser.add_argument("--sampler_preset", type=str, default="V4_DEFAULT_20", choices=sorted(PRESETS.keys()))
    parser.add_argument("--ideogram4_timestep_mu", type=float, default=0.0, help="known mean for Ideogram 4 logit-normal training timesteps")
    parser.add_argument("--ideogram4_timestep_std", type=float, default=1.0, help="std for Ideogram 4 logit-normal training timesteps")
    parser.add_argument(
        "--validate_caption_structure",
        action="store_true",
        help="validate official structured JSON sample prompts; ordinary prompts are accepted by default",
    )
    parser.add_argument(
        "--warn_on_caption_issues",
        action="store_true",
        help="warn instead of failing on sample caption verifier issues when --validate_caption_structure is enabled",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = ideogram4_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    args.dit_dtype = "bfloat16" if args.dit_dtype is None else args.dit_dtype
    args.vae_dtype = "bfloat16" if args.vae_dtype is None else args.vae_dtype

    trainer = Ideogram4NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
