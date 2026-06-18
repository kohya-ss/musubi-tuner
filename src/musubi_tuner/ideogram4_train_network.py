import argparse
import gc
import logging
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_IDEOGRAM4, ARCHITECTURE_IDEOGRAM4_FULL
from musubi_tuner.ideogram4 import ideogram4_utils
from musubi_tuner.ideogram4.sampling_policy import should_use_unconditional_dit_for_lora_sampling
from musubi_tuner.ideogram4.sampler_configs import PRESETS
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.sampling_prompts import load_prompts
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils import model_utils
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS = {
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "qwen_shift",
    "ideogram4_shift",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
    "flux2_shift",
}


class Ideogram4NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.unconditional_transformer = None
        # Attention backend selected for the conditional DiT, reused for the unconditional DiT so both
        # share the same attention path during asymmetric-CFG LoRA sampling.
        self._attn_mode = "torch"
        self._split_attn = False

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
        if args.blocks_to_swap is not None and args.blocks_to_swap > 33:
            raise ValueError("--blocks_to_swap for Ideogram 4 must be <= 33")
        if args.sample_prompts and (args.text_encoder is None or args.vae is None):
            raise ValueError("--sample_prompts for Ideogram 4 requires --text_encoder and --vae")
        # compute_loss() uses a plain mean MSE and intentionally ignores the SD3 timestep
        # weighting, so reject a non-default --weighting_scheme rather than silently dropping it.
        if args.weighting_scheme != "none":
            raise ValueError("Ideogram 4 currently supports --weighting_scheme none only.")

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
            use_unconditional_dit = should_use_unconditional_dit_for_lora_sampling(args)
            if args.unconditional_dit and not use_unconditional_dit:
                logger.warning(
                    "Ignoring --unconditional_dit for Ideogram 4 LoRA sampling. "
                    "LoRA training only adapts the conditional DiT, so sampling defaults to single-DiT CFG. "
                    "Pass --use_unconditional_dit_for_lora_sampling to opt into the old asymmetric CFG path."
                )
            for prompt_dict in prompts:
                prompt = prompt_dict.get("prompt", "")
                negative_prompt = prompt_dict.get("negative_prompt")
                if use_unconditional_dit and negative_prompt is not None:
                    logger.warning("Ideogram 4 v1 ignores negative_prompt in sample prompts.")
                if args.validate_caption_structure:
                    ideogram4_utils.validate_prompt(prompt, warn_only=args.warn_on_caption_issues)
                prompt_dict = prompt_dict.copy()
                prompt_dict["i4_llm_features"] = ideogram4_utils.encode_prompt_to_features(
                    tokenizer, text_encoder, prompt, device
                ).cpu()
                if not use_unconditional_dit:
                    prompt_dict["i4_unconditional_llm_features"] = ideogram4_utils.encode_prompt_to_features(
                        tokenizer, text_encoder, negative_prompt or "", device
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
        if should_use_unconditional_dit_for_lora_sampling(args) and self.unconditional_transformer is None:
            logger.info(f"Loading Ideogram 4 unconditional DiT from {args.unconditional_dit}")
            self.unconditional_transformer = ideogram4_utils.load_ideogram4_transformer(
                args.unconditional_dit,
                device=accelerator.device,
                dtype=dit_dtype,
                expected_model_type=ideogram4_utils.IDEOGRAM4_UNCOND_MODEL_TYPE,
                disable_mmap=args.disable_numpy_memmap,
                attn_mode=self._attn_mode,
                split_attn=self._split_attn,
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
        sampler_preset = sample_parameter.get("sampler_preset", args.sampler_preset)
        features = sample_parameter["i4_llm_features"].to(torch.float32)
        unconditional_features = sample_parameter.get("i4_unconditional_llm_features")
        if unconditional_features is not None:
            unconditional_features = unconditional_features.to(torch.float32)
        vae.to(accelerator.device)
        vae.eval()
        conditional_transformer = accelerator.unwrap_model(transformer, keep_fp32_wrapper=False)
        # Sampling runs under the caller's accelerator.autocast() (see NetworkTrainer.sample_images), so
        # it follows --mixed_precision exactly like training's call_dit(). This used to force autocast off
        # to dodge a checkerboard, but the real cause was Ideogram4MRoPE losing precision under autocast;
        # now that MRoPE pins its frequency matmul to fp32, autocast sampling is safe and we let the
        # mixed-precision regime match between training and sampling.
        images = ideogram4_utils.generate_images(
            conditional_transformer=conditional_transformer,
            unconditional_transformer=self.unconditional_transformer,
            autoencoder=vae,
            text_features=[features],
            unconditional_text_features=[unconditional_features] if unconditional_features is not None else None,
            height=height,
            width=width,
            sampler_preset=sampler_preset,
            device=accelerator.device,
            seed=sample_parameter.get("seed", None),
            show_progress=True,
            initial_sigma=args.initial_sigma,
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
        del accelerator, dit_weight_dtype
        self._attn_mode = attn_mode
        self._split_attn = split_attn
        return ideogram4_utils.load_ideogram4_transformer(
            dit_path,
            device=loading_device,
            dtype=model_utils.str_to_dtype(args.dit_dtype),
            expected_model_type=ideogram4_utils.IDEOGRAM4_COND_MODEL_TYPE,
            disable_mmap=args.disable_numpy_memmap,
            attn_mode=attn_mode,
            split_attn=split_attn,
        )

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(args, transformer, [transformer.layers], disable_linear=self.blocks_to_swap > 0)

    def scale_shift_latents(self, latents):
        # Transform raw VAE latents into the model's normalized token-grid space
        # (per-channel (latents - shift) / scale). This is the designated latents-transform
        # hook (cf. Z-Image), called by the base training loop before noise sampling, so the
        # base process_batch can build the noisy input directly in model space. Keeping the
        # transform here lets this trainer override only call_dit and compute_loss instead of
        # reimplementing the whole batch flow.
        return ideogram4_utils.normalize_token_grid(latents)

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
        inputs = ideogram4_utils.build_sequence_inputs_from_features(
            text_features, image_height, image_width, device=accelerator.device
        )

        image_tokens = ideogram4_utils.flatten_token_grid(noisy_model_input).to(device=accelerator.device, dtype=network_dtype)
        text_padding = torch.zeros(
            bsize,
            int(inputs["max_text_tokens"]),
            model.config.in_channels,
            dtype=network_dtype,
            device=accelerator.device,
        )
        x = torch.cat([image_tokens, text_padding], dim=1)
        llm_features = inputs["llm_features"].to(dtype=network_dtype)
        if args.gradient_checkpointing:
            x.requires_grad_(True)
            llm_features.requires_grad_(True)

        # Training samples use t=0 as clean and t=1 as noise. The Ideogram 4
        # transformer uses the inverse convention: t=0 is noise and t=1 is clean.
        raw_timestep = timesteps.to(accelerator.device, dtype=torch.float32)
        if getattr(args, "timestep_sampling", "sigma") in NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS:
            model_t = ((1001.0 - raw_timestep) / 1000.0).clamp(0.0, 1.0)
        else:
            model_t = (1.0 - raw_timestep / 1000.0).clamp(0.0, 1.0)
        # Run the DiT under accelerator.autocast() like every other architecture, so --mixed_precision
        # selects the compute regime: bf16 -> autocast mixed precision (LoRA computed in bf16), and the
        # default 'no' -> a true no-op, i.e. fp32 LoRA over the bf16 base (the previous Ideogram 4
        # behaviour, unchanged). This is only safe because Ideogram4MRoPE forces its frequency matmul to
        # fp32 internally; without that guard autocast would collapse the image RoPE positions to a single
        # value (offset 2**16, below bf16 resolution) and produce a flat checkerboard.
        with accelerator.autocast():
            model_pred = model(
                llm_features=llm_features,
                x=x,
                t=model_t,
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
                indicator=inputs["indicator"],
            )
        model_pred = model_pred[:, : int(inputs["num_image_tokens"])]
        model_pred = ideogram4_utils.unflatten_token_grid(model_pred, grid_h, grid_w)
        target = latents - noise
        return DiTOutput(pred=model_pred, target=target)

    def compute_loss(
        self,
        args: argparse.Namespace,
        output: DiTOutput,
        timesteps: torch.Tensor,
        noise_scheduler,
        dit_dtype: torch.dtype,
        network_dtype: torch.dtype,
        global_step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # Plain mean MSE in flow-matching velocity space. Ideogram 4 does not use the
        # SD3 weighting_scheme, so this owns the full loss formulation rather than
        # augmenting super().compute_loss (which would apply that weighting).
        del noise_scheduler, dit_dtype, global_step
        pred = output.pred.to(network_dtype)
        target = output.target.to(network_dtype)
        loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")

        loss_metrics = {}
        if getattr(args, "log_loss_stats", False):
            with torch.no_grad():
                pred_f = pred.detach().float()
                target_f = target.detach().float()
                pred_rms = pred_f.square().mean().sqrt()
                target_rms = target_f.square().mean().sqrt()
                denom = (pred_rms * target_rms).clamp_min(1e-8)
                loss_metrics = {
                    "loss/zero_pred": target_f.square().mean().item(),
                    "loss/flipped_pred": torch.nn.functional.mse_loss(-pred_f, target_f, reduction="mean").item(),
                    "loss/pred_rms": pred_rms.item(),
                    "loss/target_rms": target_rms.item(),
                    "loss/pred_target_cosine": ((pred_f * target_f).mean() / denom).item(),
                    "timestep/mean": timesteps.detach().float().mean().item(),
                    "timestep/min": timesteps.detach().float().min().item(),
                    "timestep/max": timesteps.detach().float().max().item(),
                }
        return loss, loss_metrics


def ideogram4_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(timestep_sampling="ideogram4_shift")
    parser.add_argument("--unconditional_dit", type=str, default=None, help="unconditional Ideogram 4 DiT safetensors path")
    parser.add_argument(
        "--use_unconditional_dit_for_lora_sampling",
        action="store_true",
        help=(
            "opt into official asymmetric CFG during LoRA training samples; by default LoRA sampling uses single-DiT CFG "
            "because only the conditional DiT is trained"
        ),
    )
    parser.add_argument(
        "--text_encoder", type=str, default=None, help="Qwen3-VL BF16 text encoder safetensors path; only needed for sampling"
    )
    parser.add_argument("--dit_dtype", type=str, default=None, help="data type for Ideogram 4 DiT, default is bfloat16")
    parser.add_argument("--sampler_preset", type=str, default="V4_DEFAULT_20", choices=sorted(PRESETS.keys()))
    parser.add_argument("--initial_sigma", type=float, default=1.004, help="override the first denoising sigma for sampling")
    parser.add_argument(
        "--log_loss_stats", action="store_true", help="log Ideogram 4 prediction/target diagnostics during training"
    )
    # Deprecated compatibility knobs. Ideogram 4 now uses the shared
    # --timestep_sampling path, so these legacy values are intentionally ignored.
    parser.add_argument("--ideogram4_timestep_mu", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--ideogram4_timestep_std", type=float, default=1.0, help=argparse.SUPPRESS)
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
