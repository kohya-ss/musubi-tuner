import argparse
import logging
from typing import Any, Optional

import numpy as np
import torch
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HIDREAM_O1, ARCHITECTURE_HIDREAM_O1_FULL
from musubi_tuner.hidream_o1 import hidream_o1_utils
from musubi_tuner.hidream_o1.pipeline import DEFAULT_TIMESTEPS, TIMESTEP_TOKEN_NUM, generate_image
from musubi_tuner.hidream_o1.utils import get_rope_index_fix_point
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8
from musubi_tuner.hv_train_network import NetworkTrainer, get_sigmas, load_prompts, read_config_from_file, setup_parser_common
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FP8_OPTIMIZATION_TARGET_KEYS = [
    "model.language_model.layers",
    "model.visual.blocks",
    "model.visual.merger",
    "model.visual.deepstack_merger_list",
    "model.t_embedder1",
    "model.x_embedder",
    "model.final_layer2",
]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "rotary", "pos_embed", "embed_tokens", "lm_head"]
NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS = {
    "uniform",
    "sigmoid",
    "shift",
    "flux_shift",
    "qwen_shift",
    "logsnr",
    "qinglong_flux",
    "qinglong_qwen",
    "flux2_shift",
}


class _NoVAE(torch.nn.Module):
    pass


class HiDreamO1NetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.use_flash_attn = False
        self.model_type = "full"

    @property
    def architecture(self) -> str:
        return ARCHITECTURE_HIDREAM_O1

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_HIDREAM_O1_FULL

    def handle_model_specific_args(self, args):
        self.dit_dtype = torch.bfloat16
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.use_flash_attn = getattr(args, "flash_attn", False)
        self.model_type = args.model_type
        self.default_guidance_scale = 0.0 if self.model_type == "dev" else 5.0
        self.default_discrete_flow_shift = 1.0 if self.model_type == "dev" else 3.0
        if args.noise_scale_start is None:
            args.noise_scale_start = 7.5 if self.model_type == "dev" else 8.0
        if args.noise_scale_end is None:
            args.noise_scale_end = args.noise_scale_start
        if args.noise_clip_std is None:
            args.noise_clip_std = 2.5 if self.model_type == "dev" else 0.0
        logger.info(
            f"HiDream-O1 {self.model_type} noise settings: "
            f"scale_start={args.noise_scale_start}, scale_end={args.noise_scale_end}, clip_std={args.noise_clip_std}"
        )

        if args.weighting_scheme != "none":
            raise ValueError("HiDream-O1 currently supports --weighting_scheme none only.")
        if args.fp8_base and not getattr(args, "fp8_scaled", False):
            raise ValueError("HiDream-O1 supports --fp8_base only together with --fp8_scaled.")

    def prepare_network_kwargs(
        self,
        args: argparse.Namespace,
        train_dataset_group: Any,
        network_module: Any,
        net_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        has_control = any(getattr(dataset, "has_control", False) for dataset in train_dataset_group.datasets)
        net_kwargs = dict(net_kwargs)
        net_kwargs["hidream_has_control"] = str(has_control)
        target = "I2I/control" if has_control else "T2I"
        logger.info(f"HiDream-O1 LoRA target is selected from dataset: {target}")
        return net_kwargs

    def process_sample_prompts(self, args: argparse.Namespace, accelerator: Accelerator, sample_prompts: str):
        return load_prompts(sample_prompts)

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
        model = accelerator.unwrap_model(transformer)
        processor = self.processor or hidream_o1_utils.load_processor(model_type=self.model_type)

        prompt = sample_parameter.get("prompt", "")
        ref_image_paths = sample_parameter.get("control_image_path", None)
        if ref_image_paths is None:
            ref_image_paths = []
        elif isinstance(ref_image_paths, str):
            ref_image_paths = [ref_image_paths]

        sample_steps = sample_steps if "sample_steps" in sample_parameter else None

        if self.model_type == "dev":
            num_inference_steps = sample_steps if sample_steps is not None else 28
            guidance = 0.0
            shift = 1.0
            timesteps_list = DEFAULT_TIMESTEPS if sample_steps in (None, 28) else None
            scheduler_name = "flash"
        else:
            num_inference_steps = sample_steps if sample_steps is not None else 50
            guidance = cfg_scale if cfg_scale is not None else guidance_scale
            shift = discrete_flow_shift
            timesteps_list = None
            scheduler_name = "default"

        noise_kwargs = {
            "noise_scale_start": args.noise_scale_start,
            "noise_scale_end": args.noise_scale_end,
            "noise_clip_std": args.noise_clip_std,
        }

        seed = int(generator.initial_seed()) if generator is not None else int(sample_parameter.get("seed", 42))
        image = generate_image(
            model=model,
            processor=processor,
            prompt=prompt,
            ref_image_paths=ref_image_paths,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance,
            shift=shift,
            timesteps_list=timesteps_list,
            scheduler_name=scheduler_name,
            seed=seed,
            use_flash_attn=self.use_flash_attn,
            **noise_kwargs,
        )

        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return pixels.unsqueeze(0).unsqueeze(2)

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        return _NoVAE()

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
        self.processor = hidream_o1_utils.load_processor(model_type=args.model_type)
        dtype = dit_weight_dtype or torch.bfloat16
        model = hidream_o1_utils.load_model(dit_path, dtype=dtype, device=loading_device, model_type=args.model_type)

        if getattr(args, "fp8_scaled", False):
            logger.info("Applying HiDream-O1 scaled fp8 optimization.")
            state_dict = model.state_dict()
            quant_device = accelerator.device
            move_to_device = str(loading_device) != "cpu"
            state_dict = optimize_state_dict_with_fp8(
                state_dict,
                quant_device,
                FP8_OPTIMIZATION_TARGET_KEYS,
                FP8_OPTIMIZATION_EXCLUDE_KEYS,
                move_to_device=move_to_device,
            )
            apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
            info = model.load_state_dict(state_dict, strict=True, assign=True)
            logger.info(f"Loaded HiDream-O1 scaled fp8 optimized weights: {info}")

        if not hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing = lambda cpu_offload=False: model.gradient_checkpointing_enable()

        return model

    def compile_transformer(self, args, transformer):
        return transformer

    def scale_shift_latents(self, latents):
        return latents

    def get_noisy_model_input_and_timesteps(
        self,
        args: argparse.Namespace,
        noise: torch.Tensor,
        latents: torch.Tensor,
        timesteps: Optional[list[float]],
        noise_scheduler,
        device: torch.device,
        dtype: torch.dtype,
    ):
        if getattr(args, "show_timesteps", None):
            return super().get_noisy_model_input_and_timesteps(args, noise, latents, timesteps, noise_scheduler, device, dtype)

        if args.timestep_sampling == "uniform" and not getattr(args, "preserve_distribution_shape", False):
            sigma = (
                torch.rand((noise.shape[0],), device=device, dtype=dtype)
                if timesteps is None
                else torch.as_tensor(timesteps, device=device, dtype=dtype)
            )
            t_min = (args.min_timestep if args.min_timestep is not None else 0.0) / 1000.0
            t_max = (args.max_timestep if args.max_timestep is not None else 1000.0) / 1000.0
            sigma = sigma * (t_max - t_min) + t_min
            timesteps = sigma * 1000.0 + 1.0
            sigma_ndim = sigma.view(noise.shape[0], *([1] * (latents.ndim - 1)))
        else:
            _, timesteps = super().get_noisy_model_input_and_timesteps(
                args, noise, latents, timesteps, noise_scheduler, device, dtype
            )
            if args.timestep_sampling not in NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS:
                sigma_ndim = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
                sigma = sigma_ndim.flatten()
            else:
                sigma = ((timesteps.to(device=device, dtype=dtype) - 1.0) / 1000.0).clamp(0.0, 1.0)
                sigma_ndim = sigma.view(noise.shape[0], *([1] * (latents.ndim - 1)))

        noise_clip_std = getattr(args, "noise_clip_std", 0.0) or 0.0
        if noise_clip_std > 0.0:
            noise_std = noise.float().std().to(device=noise.device, dtype=noise.dtype)
            clip_val = noise_clip_std * noise_std
            noise = noise.clamp(min=-clip_val, max=clip_val)

        noise_scale_start = getattr(args, "noise_scale_start", None)
        if noise_scale_start is None:
            noise_scale_start = 7.5 if getattr(args, "model_type", "full") == "dev" else 8.0
        noise_scale_end = getattr(args, "noise_scale_end", None)
        if noise_scale_end is None:
            noise_scale_end = noise_scale_start
        noise_scale = noise_scale_start + (noise_scale_end - noise_scale_start) * (1.0 - sigma)
        noise_scale_ndim = noise_scale.view(noise.shape[0], *([1] * (latents.ndim - 1)))

        noisy_model_input = (1.0 - sigma_ndim) * latents + sigma_ndim * noise * noise_scale_ndim
        return noisy_model_input, timesteps

    def _build_layout_tensors(
        self,
        input_ids: torch.Tensor,
        height_patches: int,
        width_patches: int,
        model_config,
        device: torch.device,
        control_patch_shapes: Optional[list[tuple[int, int]]] = None,
        processor_image_grid_thw: Optional[torch.Tensor] = None,
    ):
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id
        spatial_merge_size = model_config.vision_config.spatial_merge_size

        image_seq_len = height_patches * width_patches
        control_patch_shapes = control_patch_shapes or []

        input_ids = input_ids.to(device=device, dtype=torch.long).unsqueeze(0)

        image_grid_thw_tgt = torch.tensor([1, height_patches, width_patches], dtype=torch.int64, device=device).unsqueeze(0)
        if control_patch_shapes:
            if processor_image_grid_thw is None:
                raise ValueError(
                    "HiDream-O1 control training requires text cache with pixel_values/image_grid_thw. "
                    "Rebuild the text encoder cache."
                )
            processor_image_grid_thw = processor_image_grid_thw.to(device=device, dtype=torch.int64)
            if processor_image_grid_thw.dim() == 3 and processor_image_grid_thw.shape[0] == 1:
                processor_image_grid_thw = processor_image_grid_thw.squeeze(0)
            if processor_image_grid_thw.shape[0] != len(control_patch_shapes):
                raise ValueError(
                    f"HiDream-O1 control count mismatch: text cache has {processor_image_grid_thw.shape[0]} "
                    f"vision inputs, pixel cache has {len(control_patch_shapes)} controls."
                )

            image_grid_thw_cond = processor_image_grid_thw.clone()
            image_grid_thw_cond[:, 1] //= spatial_merge_size
            image_grid_thw_cond[:, 2] //= spatial_merge_size
            image_grid_thw_ref = torch.tensor(
                [[1, height, width] for height, width in control_patch_shapes], dtype=torch.int64, device=device
            )
            image_grid_thw = torch.cat([image_grid_thw_cond, image_grid_thw_tgt, image_grid_thw_ref], dim=0)
        else:
            image_grid_thw = image_grid_thw_tgt

        vision_tokens_list = []
        vision_tokens_tgt = torch.full((1, image_seq_len), image_token_id, dtype=input_ids.dtype, device=device)
        vision_tokens_tgt[0, 0] = vision_start_token_id
        vision_tokens_list.append(vision_tokens_tgt)
        for control_height_patches, control_width_patches in control_patch_shapes:
            control_seq_len = control_height_patches * control_width_patches
            vision_tokens_ref = torch.full((1, control_seq_len), image_token_id, dtype=input_ids.dtype, device=device)
            vision_tokens_ref[0, 0] = vision_start_token_id
            vision_tokens_list.append(vision_tokens_ref)
        vision_tokens = torch.cat(vision_tokens_list, dim=-1)
        input_ids_pad = torch.cat([input_ids, vision_tokens], dim=-1)

        position_ids, _ = get_rope_index_fix_point(
            1,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            input_ids=input_ids_pad,
            image_grid_thw=image_grid_thw,
            video_grid_thw=None,
            attention_mask=None,
            skip_vision_start_token=[0] * len(control_patch_shapes) + [1] + [1] * len(control_patch_shapes),
        )

        txt_seq_len = input_ids.shape[-1]
        token_types_raw = torch.zeros((1, position_ids.shape[-1]), dtype=input_ids.dtype, device=device)
        bgn = txt_seq_len - TIMESTEP_TOKEN_NUM
        end = bgn + image_seq_len + TIMESTEP_TOKEN_NUM
        token_types_raw[0, bgn:end] = 1
        ref_seq_len = sum(height * width for height, width in control_patch_shapes)
        if ref_seq_len > 0:
            token_types_raw[0, end : end + ref_seq_len] = 2
        token_types_raw[0, txt_seq_len - TIMESTEP_TOKEN_NUM : txt_seq_len] = 3
        vinput_mask = torch.logical_or(token_types_raw == 1, token_types_raw == 2)
        token_types = (token_types_raw > 0).to(input_ids.dtype)

        return input_ids, position_ids, token_types, vinput_mask

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
        model_config = accelerator.unwrap_model(transformer).config
        input_ids_list = batch["input_ids"]
        if torch.is_tensor(input_ids_list):
            input_ids_list = list(input_ids_list)

        height_patches, width_patches = latents.shape[1], latents.shape[2]
        latents_seq = latents.reshape(latents.shape[0], height_patches * width_patches, latents.shape[-1])
        noisy_model_input_seq = noisy_model_input.reshape(
            noisy_model_input.shape[0], height_patches * width_patches, noisy_model_input.shape[-1]
        )

        model_preds = []
        input_embeds_list = batch.get("input_embeds", None)
        if torch.is_tensor(input_embeds_list):
            input_embeds_list = list(input_embeds_list)
        pixel_values_list = batch.get("pixel_values", None)
        if torch.is_tensor(pixel_values_list):
            pixel_values_list = list(pixel_values_list)
        image_grid_thw_list = batch.get("image_grid_thw", None)
        if torch.is_tensor(image_grid_thw_list):
            image_grid_thw_list = list(image_grid_thw_list)
        control_keys = sorted(
            [key for key in batch.keys() if key.startswith("latents_control")],
            key=lambda key: int(key.rsplit("_", 1)[-1]) if key.rsplit("_", 1)[-1].isdigit() else 0,
        )
        for i, input_ids in enumerate(input_ids_list):
            control_patch_shapes = []
            control_sequences = []
            for key in control_keys:
                control = batch[key][i].to(device=accelerator.device, dtype=network_dtype)
                control_patch_shapes.append((control.shape[0], control.shape[1]))
                control_sequences.append(control.reshape(1, control.shape[0] * control.shape[1], control.shape[-1]))

            if control_sequences and (pixel_values_list is None or image_grid_thw_list is None):
                raise ValueError(
                    "HiDream-O1 control pixel cache was found, but text cache has no pixel_values/image_grid_thw. "
                    "Rebuild the text encoder cache."
                )
            if not control_sequences and (pixel_values_list is not None or image_grid_thw_list is not None):
                raise ValueError(
                    "HiDream-O1 control text cache was found, but pixel cache has no latents_control_* tensors. "
                    "Rebuild the pixel cache."
                )

            processor_image_grid_thw = None
            if image_grid_thw_list is not None:
                processor_image_grid_thw = image_grid_thw_list[i]

            input_ids, position_ids, token_types, vinput_mask = self._build_layout_tensors(
                input_ids,
                height_patches,
                width_patches,
                model_config,
                accelerator.device,
                control_patch_shapes=control_patch_shapes,
                processor_image_grid_thw=processor_image_grid_thw,
            )
            vinputs = noisy_model_input_seq[i : i + 1].to(device=accelerator.device, dtype=network_dtype)
            if control_sequences:
                vinputs = torch.cat([vinputs, *control_sequences], dim=1)
            raw_timestep = timesteps[i : i + 1].to(device=accelerator.device, dtype=network_dtype)
            if args.timestep_sampling in NOISE_COEFFICIENT_TIMESTEP_SAMPLINGS:
                timestep = ((1001.0 - raw_timestep) / 1000.0).clamp(0.0, 1.0).reshape(-1)
            else:
                timestep = (1.0 - raw_timestep / 1000.0).clamp(0.0, 1.0).reshape(-1)

            if args.gradient_checkpointing:
                vinputs.requires_grad_(True)

            input_embeds = None
            if input_embeds_list is not None:
                input_embeds = input_embeds_list[i].to(device=accelerator.device, dtype=network_dtype).unsqueeze(0)

            pixel_values = None
            image_grid_thw = None
            if pixel_values_list is not None:
                pixel_values = pixel_values_list[i].to(device=accelerator.device, dtype=network_dtype)
            if image_grid_thw_list is not None:
                image_grid_thw = image_grid_thw_list[i].to(device=accelerator.device, dtype=torch.int64)

            with accelerator.autocast():
                outputs = transformer(
                    input_ids=input_ids,
                    inputs_embeds=input_embeds,
                    position_ids=position_ids,
                    vinputs=vinputs,
                    timestep=timestep,
                    token_types=token_types,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    use_flash_attn=self.use_flash_attn,
                )

            pred = outputs.x_pred[0, vinput_mask[0]][: latents_seq.shape[1]].unsqueeze(0)
            model_preds.append(pred)

        model_pred = torch.cat(model_preds, dim=0)
        target = latents_seq.to(device=accelerator.device, dtype=network_dtype)
        return model_pred, target

    def prepare_dino_loss_images(
        self,
        args: argparse.Namespace,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        batch: dict[str, torch.Tensor],
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if model_pred.ndim == 3 and target.ndim == 3 and latents.ndim == 4:
            height = latents.shape[1] * hidream_o1_utils.PATCH_SIZE
            width = latents.shape[2] * hidream_o1_utils.PATCH_SIZE
            pred_images = hidream_o1_utils.unpatchify_pixels(model_pred, height, width)
            target_images = hidream_o1_utils.unpatchify_pixels(target, height, width)
            return pred_images, target_images

        return super().prepare_dino_loss_images(args, model_pred, target, batch, latents, timesteps, network_dtype)


def hidream_o1_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "dev"], help="HiDream-O1 model variant")
    parser.add_argument(
        "--noise_scale_start",
        type=float,
        default=None,
        help="HiDream-O1 noise scale at the first denoising/noisiest step. Defaults to 8.0 for full and 7.5 for dev.",
    )
    parser.add_argument(
        "--noise_scale_end",
        type=float,
        default=None,
        help="HiDream-O1 noise scale at the final denoising/cleanest step. Defaults to --noise_scale_start.",
    )
    parser.add_argument(
        "--noise_clip_std",
        type=float,
        default=None,
        help="Clip Gaussian noise by this many standard deviations before building HiDream-O1 training inputs. Defaults to 0.0 for full and 2.5 for dev.",
    )
    parser.add_argument(
        "--fp8_scaled",
        action="store_true",
        help="use scaled fp8 for HiDream-O1 DiT weights / HiDream-O1 DiTにスケーリングされたfp8を使う",
    )
    return parser


def main():
    parser = setup_parser_common()
    parser = hidream_o1_setup_parser(parser)
    args = parser.parse_args()
    args = read_config_from_file(args, parser)

    trainer = HiDreamO1NetworkTrainer()
    trainer.train(args)


if __name__ == "__main__":
    main()
