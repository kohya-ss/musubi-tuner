import argparse
import logging
from typing import Optional

import numpy as np
import torch
from accelerate import Accelerator

from musubi_tuner.dataset.image_video_dataset import ARCHITECTURE_HIDREAM_O1, ARCHITECTURE_HIDREAM_O1_FULL
from musubi_tuner.hidream_o1 import hidream_o1_utils
from musubi_tuner.hidream_o1.pipeline import DEFAULT_TIMESTEPS, generate_image
from musubi_tuner.hidream_o1.utils import get_rope_index_fix_point
from musubi_tuner.hv_train_network import NetworkTrainer, load_prompts, read_config_from_file, setup_parser_common
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
        self.default_guidance_scale = 5.0
        self.default_discrete_flow_shift = 3.0
        self.use_flash_attn = getattr(args, "flash_attn", False)
        self.model_type = args.model_type

        if args.weighting_scheme != "none":
            raise ValueError("HiDream-O1 currently supports --weighting_scheme none only.")
        if args.fp8_base or args.fp8_scaled:
            raise ValueError("HiDream-O1 fp8 model loading is not supported yet.")

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

        if not hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing = lambda cpu_offload=False: model.gradient_checkpointing_enable()

        return model

    def compile_transformer(self, args, transformer):
        return transformer

    def scale_shift_latents(self, latents):
        return latents

    def _build_layout_tensors(
        self,
        input_ids: torch.Tensor,
        height_patches: int,
        width_patches: int,
        model_config,
        device: torch.device,
    ):
        image_token_id = model_config.image_token_id
        video_token_id = model_config.video_token_id
        vision_start_token_id = model_config.vision_start_token_id

        image_seq_len = height_patches * width_patches

        input_ids = input_ids.to(device=device, dtype=torch.long).unsqueeze(0)
        image_grid_thw = torch.tensor([1, height_patches, width_patches], dtype=torch.int64, device=device).unsqueeze(0)
        vision_tokens = torch.full((1, image_seq_len), image_token_id, dtype=input_ids.dtype, device=device)
        vision_tokens[0, 0] = vision_start_token_id
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
            skip_vision_start_token=[1],
        )

        txt_seq_len = input_ids.shape[-1]
        token_types_raw = torch.zeros((1, position_ids.shape[-1]), dtype=input_ids.dtype, device=device)
        bgn = txt_seq_len - 1
        token_types_raw[0, bgn : bgn + image_seq_len + 1] = 1
        token_types_raw[0, txt_seq_len - 1 : txt_seq_len] = 3
        vinput_mask = token_types_raw == 1
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
        for i, input_ids in enumerate(input_ids_list):
            input_ids, position_ids, token_types, vinput_mask = self._build_layout_tensors(
                input_ids,
                height_patches,
                width_patches,
                model_config,
                accelerator.device,
            )
            vinputs = noisy_model_input_seq[i : i + 1].to(device=accelerator.device, dtype=network_dtype)
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

            with accelerator.autocast():
                outputs = transformer(
                    input_ids=input_ids,
                    inputs_embeds=input_embeds,
                    position_ids=position_ids,
                    vinputs=vinputs,
                    timestep=timestep,
                    token_types=token_types,
                    use_flash_attn=self.use_flash_attn,
                )

            pred = outputs.x_pred[0, vinput_mask[0]].unsqueeze(0)
            model_preds.append(pred)

        model_pred = torch.cat(model_preds, dim=0)
        target = latents_seq.to(device=accelerator.device, dtype=network_dtype)
        return model_pred, target


def hidream_o1_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "dev"], help="HiDream-O1 model variant")
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
