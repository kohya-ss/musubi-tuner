from __future__ import annotations

import argparse
import gc
import logging
import re
from typing import Optional

from PIL import Image
import torch
from accelerate import Accelerator

from musubi_tuner.hv_train_network import (
    DiTOutput,
    NetworkTrainer,
    clean_memory_on_device,
    load_prompts,
    read_config_from_file,
    setup_parser_common,
)
from musubi_tuner.mage_flow_generate_image import image_to_tensor
from musubi_tuner.mage_flow.mage_vae import decode_mage_vae_latents, load_mage_vae
from musubi_tuner.mage_flow.model import MageFlow, load_mage_flow_transformer
from musubi_tuner.mage_flow.sampling import resolve_output_size, sample_latents
from musubi_tuner.mage_flow.text_encoder import encode_conditioning, load_mage_flow_text_encoder
from musubi_tuner.mage_flow.training import (
    sigma_from_training_timesteps,
    stack_bucketed_targets,
    unpack_target_predictions,
)
from musubi_tuner.mage_flow.utils import architecture_for_mode, pack_training_batch
from musubi_tuner.networks import lora_mage_flow
from musubi_tuner.utils import model_utils


_CONTROL_KEY = re.compile(r"^latents_control_(\d+)$")
logger = logging.getLogger(__name__)


def _ordered_controls(batch: dict[str, object]) -> list[torch.Tensor]:
    indexed = sorted((int(match.group(1)), value) for key, value in batch.items() if (match := _CONTROL_KEY.match(key)))
    if indexed and [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError("Mage-Flow Edit reference cache keys must be contiguous from latents_control_0")
    controls = [value for _, value in indexed]
    if any(not isinstance(control, torch.Tensor) or control.ndim != 4 for control in controls):
        raise ValueError("Mage-Flow Edit reference latents must have shape [B,C,H,W]")
    return controls


class MageFlowNetworkTrainer(NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.is_edit = False

    @property
    def architecture(self) -> str:
        return architecture_for_mode(self.is_edit)[0]

    @property
    def architecture_full_name(self) -> str:
        return architecture_for_mode(self.is_edit)[1]

    def handle_model_specific_args(self, args):
        self.is_edit = bool(args.is_edit)
        if bool(getattr(args, "compile", False)) and bool(getattr(args, "compile_fullgraph", False)):
            raise ValueError("Mage-Flow does not support --compile_fullgraph; use --compile without fullgraph")
        if args.fp8_base and not args.fp8_scaled:
            raise ValueError("Mage-Flow --fp8_base requires --fp8_scaled; unscaled FP8 is unsupported")
        if args.network_module != "musubi_tuner.networks.lora_mage_flow":
            raise ValueError("Mage-Flow is LoRA-only and requires --network_module musubi_tuner.networks.lora_mage_flow")
        blocks_to_swap = int(getattr(args, "blocks_to_swap", 0) or 0)
        if not 0 <= blocks_to_swap <= 10:
            raise ValueError(f"Mage-Flow --blocks_to_swap must be from 0 through 10, got {blocks_to_swap}")
        unsupported_attention = [name for name in ("sage_attn", "xformers", "flash3") if bool(getattr(args, name, False))]
        if unsupported_attention:
            raise ValueError(
                f"Mage-Flow supports SDPA and optional FlashAttention 2 only; unsupported flags: {', '.join(unsupported_attention)}"
            )
        if getattr(args, "flash_attn", False):
            args.sdpa = False
        elif not getattr(args, "sdpa", False):
            args.sdpa = True
        network_weights = getattr(args, "network_weights", None)
        if getattr(args, "dim_from_weights", False) and not network_weights:
            raise ValueError("--dim_from_weights requires --network_weights")
        adapter_paths = [network_weights] if network_weights else []
        adapter_paths.extend(getattr(args, "base_weights", None) or [])
        for path in dict.fromkeys(adapter_paths):
            lora_mage_flow.validate_adapter_architecture(
                path,
                expected=self.architecture_full_name,
                allow_mismatch=bool(getattr(args, "allow_mage_architecture_mismatch", False)),
            )
        if getattr(args, "vae_dtype", None) is None:
            args.vae_dtype = "bfloat16"
        self.dit_dtype = (
            torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
        )
        args.dit_dtype = model_utils.dtype_to_str(self.dit_dtype)
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 6.0
        self.vae_frame_stride = 1

    def scale_shift_latents(self, latents):
        return latents

    @staticmethod
    def _sample_reference_paths(prompt_dict: dict) -> list[str]:
        raw = prompt_dict.get("control_image_path", [])
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return list(raw)

    def process_sample_prompts(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        sample_prompts: str,
    ):
        if not args.text_encoder:
            raise ValueError("--text_encoder is required when --sample_prompts is used")
        prompts = load_prompts(sample_prompts)
        reference_lists: list[list[Image.Image]] = []
        for index, prompt_dict in enumerate(prompts):
            paths = self._sample_reference_paths(prompt_dict)
            if self.is_edit and not 1 <= len(paths) <= 3:
                raise ValueError(f"Edit sample prompt {index} requires one to three control_image_path values")
            if not self.is_edit and paths:
                raise ValueError(f"T2I sample prompt {index} must not contain control_image_path")
            reference_lists.append([Image.open(path).convert("RGB") for path in paths])

        encoder = load_mage_flow_text_encoder(
            args.text_encoder,
            device=accelerator.device,
            dtype=torch.bfloat16,
        )
        sample_parameters = []
        for prompt_dict, references in zip(prompts, reference_lists):
            item = prompt_dict.copy()
            prompt = item.get("prompt", "")
            negative_prompt = item.get("negative_prompt", " ")
            item["negative_prompt"] = negative_prompt
            encoded = encode_conditioning(
                encoder,
                [prompt, negative_prompt],
                references=[references, references] if self.is_edit else None,
                is_edit=self.is_edit,
            )
            item["mage_flow_embed"] = encoded[0].cpu()
            item["negative_mage_flow_embed"] = encoded[1].cpu()
            if self.is_edit:
                width, height = resolve_output_size(
                    references[0].size,
                    width=item.get("width"),
                    height=item.get("height"),
                    max_size=item.get("max_size"),
                )
                item["width"] = width
                item["height"] = height
                item["mage_flow_reference_images"] = references
            sample_parameters.append(item)
        encoder.to("cpu")
        del encoder
        gc.collect()
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
        del args, guidance_scale, image_path, control_video_path
        if frame_count != 1:
            raise ValueError("Mage-Flow sampling supports images only")
        width = max(16, width // 16 * 16)
        height = max(16, height // 16 * 16)
        device = accelerator.device
        positive = sample_parameter["mage_flow_embed"].to(device=device, dtype=dit_dtype)
        cfg_scale = 5.0 if cfg_scale is None else cfg_scale
        negative = None
        if do_classifier_free_guidance and cfg_scale > 1.0:
            negative = [sample_parameter["negative_mage_flow_embed"].to(device=device, dtype=dit_dtype)]

        controls = None
        if self.is_edit:
            references = sample_parameter.get("mage_flow_reference_images", [])
            if not 1 <= len(references) <= 3:
                raise ValueError("Mage-Flow-Edit sampling requires one to three ordered references")
            vae.to(device)
            pixels = torch.stack(
                [image_to_tensor(image, width=width, height=height, device=device, dtype=vae.dtype) for image in references]
            )
            base_seed = generator.initial_seed()
            reference_generators = [
                torch.Generator(device=device).manual_seed(base_seed + 10_000 + index) for index in range(len(references))
            ]
            controls = [[latent.to(dit_dtype) for latent in vae.encode(pixels, generators=reference_generators)]]
            vae.to("cpu")
            clean_memory_on_device(device)

        latents = sample_latents(
            transformer,
            [positive],
            [(height // 16, width // 16)],
            steps=sample_steps,
            seeds=[generator.initial_seed()],
            device=device,
            dtype=dit_dtype,
            controls=controls,
            negative_text_tokens=negative,
            cfg_scale=cfg_scale if negative is not None else 1.0,
            shift=discrete_flow_shift,
        )
        vae.to(device)
        pixels = decode_mage_vae_latents(vae, latents[0].unsqueeze(0))
        pixels = pixels.float().cpu().clamp(-1, 1).add(1).div(2)
        vae.to("cpu")
        clean_memory_on_device(device)
        return pixels.unsqueeze(2)

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
        del accelerator, split_attn
        backend = {
            "torch": "sdpa",
            "sdpa": "sdpa",
            "flash": "flash2",
            "flash2": "flash2",
        }.get(attn_mode, attn_mode)
        return load_mage_flow_transformer(
            dit_path,
            device=loading_device,
            dtype=dit_weight_dtype,
            attention_backend=backend,
            fp8_scaled=args.fp8_scaled,
        )

    def load_vae(self, args: argparse.Namespace, vae_dtype: torch.dtype, vae_path: str):
        del vae_path
        return load_mage_vae(args.vae, device="cpu", dtype=vae_dtype, require_decoder=True)

    def compile_transformer(self, args, transformer):
        model: MageFlow = transformer
        return model_utils.compile_transformer(
            args,
            model,
            [model.transformer_blocks],
            disable_linear=bool(self.blocks_to_swap),
        )

    def call_dit(
        self,
        args: argparse.Namespace,
        accelerator: Accelerator,
        transformer_arg,
        latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
        noise: torch.Tensor,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        network_dtype: torch.dtype,
        **kwargs,
    ) -> DiTOutput:
        del kwargs
        if latents.ndim != 4 or noisy_model_input.shape != latents.shape or noise.shape != latents.shape:
            raise ValueError("Mage-Flow target, noise, and noisy target must share shape [B,C,H,W]")
        controls = _ordered_controls(batch)
        if self.is_edit and not 1 <= len(controls) <= 3:
            raise ValueError(f"Mage-Flow Edit requires between 1 and 3 references, got {len(controls)}")
        if not self.is_edit and controls:
            raise ValueError("Mage-Flow T2I batches must not contain Edit reference latents")

        text = batch.get("mage_flow_embed")
        if not isinstance(text, (list, tuple)) or len(text) != latents.shape[0]:
            raise ValueError("mage_flow_embed must contain one unpadded [L,D] tensor per sample")
        device = accelerator.device
        target = noisy_model_input.to(device=device, dtype=network_dtype)
        clean_controls = [control.to(device=device, dtype=network_dtype) for control in controls]
        text_tokens = [item.to(device=device, dtype=network_dtype) for item in text]
        uses_offset = getattr(args, "timestep_sampling", "uniform") != "sigma"
        sigmas = sigma_from_training_timesteps(timesteps.to(device), one_based_offset=uses_offset)

        packed = pack_training_batch(
            target,
            text_tokens,
            sigmas,
            clean_controls if self.is_edit else None,
            image_dim=target.shape[1],
            text_dim=text_tokens[0].shape[-1],
        )
        if args.gradient_checkpointing:
            packed.image_tokens.requires_grad_(True)
            packed.text_tokens.requires_grad_(True)

        with accelerator.autocast():
            packed_prediction = transformer_arg(packed)
        prediction = stack_bucketed_targets(unpack_target_predictions(packed_prediction, packed))
        flow_target = noise.to(device=device, dtype=prediction.dtype) - latents.to(device=device, dtype=prediction.dtype)
        return DiTOutput(pred=prediction, target=flow_target)


def mage_flow_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--is_edit", action="store_true", help="train Mage-Flow-Edit instead of Mage-Flow T2I")
    parser.add_argument("--fp8_scaled", action="store_true", help="use scaled FP8 for the Mage-Flow transformer")
    parser.add_argument("--text_encoder", type=str, default=None, help="Qwen3-VL-4B safetensors checkpoint")
    parser.add_argument(
        "--allow_mage_architecture_mismatch",
        action="store_true",
        help="explicitly allow loading a mage_flow LoRA in mage_flow_edit mode or the reverse",
    )
    parser.set_defaults(
        network_module="musubi_tuner.networks.lora_mage_flow",
        mixed_precision="bf16",
        timestep_sampling="uniform_shift",
        discrete_flow_shift=6.0,
        weighting_scheme="none",
    )
    return parser


def main() -> None:
    parser = mage_flow_setup_parser(setup_parser_common())
    args = read_config_from_file(parser.parse_args(), parser)
    MageFlowNetworkTrainer().train(args)


if __name__ == "__main__":
    main()
