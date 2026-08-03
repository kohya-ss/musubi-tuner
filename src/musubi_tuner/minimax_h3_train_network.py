from __future__ import annotations

import argparse
import logging
import re
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from accelerate import Accelerator

from musubi_tuner.dataset.architectures import ARCHITECTURE_MINIMAX_H3, ARCHITECTURE_MINIMAX_H3_FULL
from musubi_tuner.minimax_h3.model import load_h3_transformer
from musubi_tuner.minimax_h3.packing import (
    H3PackedLayout,
    H3ReferenceGeometry,
    H3VideoGeometry,
    build_h3_layout,
)
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common
from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer
from musubi_tuner.utils import model_utils

logger = logging.getLogger(__name__)


_RUNTIME_REF_KEY = re.compile(r"^latents_ref_(\d{3})_(image|video|audio)$")


def validate_h3_dataset_batch_size(dataset_group) -> None:
    """Keep R1 on one real sample per step; gradient accumulation provides batching."""
    for dataset_index, dataset in enumerate(dataset_group.datasets):
        batch_size = int(dataset.batch_manager.batch_size)
        if batch_size != 1:
            raise ValueError(
                f"MiniMax-H3 R1 dataset {dataset_index} requires batch_size=1, got batch_size={batch_size}; "
                "use gradient accumulation for a larger effective batch"
            )


@dataclass(frozen=True)
class _H3RuntimeBatch:
    layout: H3PackedLayout
    text_hidden_states: torch.Tensor
    text_token_tags: torch.Tensor
    visual_conditions: tuple[torch.Tensor, ...]
    audio_conditions: tuple[torch.Tensor, ...]


def _stack_single_text_rows(value, label: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.shape[0] != 1:
            raise ValueError(f"MiniMax-H3 {label} must keep a leading batch axis of size 1")
        return value
    if not isinstance(value, Sequence) or len(value) != 1 or not isinstance(value[0], torch.Tensor):
        raise ValueError(f"MiniMax-H3 {label} must contain exactly one tensor")
    return value[0].unsqueeze(0)


def _runtime_batch_plan(batch: dict[str, Any], video_latents: torch.Tensor) -> _H3RuntimeBatch:
    if video_latents.ndim != 5 or video_latents.shape[1] != 24:
        raise ValueError(f"MiniMax-H3 target video latents must be [B,24,F,H,W], got {tuple(video_latents.shape)}")
    batch_size = video_latents.shape[0]
    if batch_size != 1:
        raise ValueError(f"MiniMax-H3 R1 requires batch_size=1, got {batch_size}; use gradient accumulation")
    audio_latents = batch.get("latents_audio")
    if not isinstance(audio_latents, torch.Tensor) or audio_latents.ndim != 4 or tuple(audio_latents.shape[1:3]) != (32, 2):
        shape = None if not isinstance(audio_latents, torch.Tensor) else tuple(audio_latents.shape)
        raise ValueError(f"MiniMax-H3 target audio latents must be [B,32,2,A], got {shape}")
    if audio_latents.shape[0] != batch_size:
        raise ValueError("MiniMax-H3 target video and audio batch sizes differ")

    hidden_states = _stack_single_text_rows(batch.get("mmh3_hidden_states"), "text hidden states")
    token_tags = _stack_single_text_rows(batch.get("mmh3_token_tags"), "text token tags")
    if hidden_states.ndim != 3 or token_tags.ndim != 2 or hidden_states.shape[:2] != token_tags.shape:
        raise ValueError("MiniMax-H3 hidden states and token tags must share [B,L]")
    if token_tags.dtype != torch.int64 or not torch.all((token_tags == 0) | (token_tags == 1)):
        raise ValueError("MiniMax-H3 text token tags must be int64 values 0 or 1")
    has_fl_condition = "latents_first" in batch or "latents_last" in batch
    reference_roles = {}
    for key, value in batch.items():
        match = _RUNTIME_REF_KEY.fullmatch(key)
        if match is not None:
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"MiniMax-H3 condition {key} must be a tensor")
            reference_roles.setdefault(int(match.group(1)), {})[match.group(2)] = value
    if has_fl_condition and reference_roles:
        raise ValueError("MiniMax-H3 batch cannot mix FL2VA and Ref2VA condition roles")

    visual_conditions = []
    audio_conditions = []
    condition_geometries = []
    references = []
    if has_fl_condition:
        if "latents_first" not in batch or "latents_last" not in batch:
            raise ValueError("MiniMax-H3 FL2VA batch requires both first and last conditions")
        task = "fl2va"
        for role in ("latents_first", "latents_last"):
            tensor = batch[role]
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 5 or tensor.shape[1] != 24:
                raise ValueError(f"MiniMax-H3 {role} must be [B,24,F,H,W]")
            if tensor.shape[0] != batch_size:
                raise ValueError(f"MiniMax-H3 {role} batch size does not match the targets")
            visual_conditions.append(tensor)
            condition_geometries.append(H3VideoGeometry(*tensor.shape[2:]))
    elif reference_roles:
        task = "ref2va"
        if set(reference_roles) != set(range(len(reference_roles))):
            raise ValueError("MiniMax-H3 reference indices must be contiguous from 000")
        for index in range(len(reference_roles)):
            roles = reference_roles[index]
            image = roles.get("image")
            video = roles.get("video")
            audio = roles.get("audio")
            if image is not None:
                if video is not None or audio is not None:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} image cannot share video/audio roles")
                if image.ndim != 5 or image.shape[1] != 24 or image.shape[0] != batch_size:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} image must be [B,24,1,H,W]")
                geometry = H3VideoGeometry(*image.shape[2:])
                references.append(H3ReferenceGeometry("image", video=geometry))
                visual_conditions.append(image)
            elif video is not None:
                if video.ndim != 5 or video.shape[1] != 24 or video.shape[0] != batch_size:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} video must be [B,24,F,H,W]")
                geometry = H3VideoGeometry(*video.shape[2:])
                audio_frames = 0
                if audio is not None:
                    if audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2) or audio.shape[0] != batch_size:
                        raise ValueError(f"MiniMax-H3 reference {index:03d} audio must be [B,32,2,A]")
                    audio_frames = audio.shape[-1]
                    audio_conditions.append(audio)
                references.append(H3ReferenceGeometry("video", video=geometry, audio_frames=audio_frames))
                visual_conditions.append(video)
            elif audio is not None:
                if audio.ndim != 4 or tuple(audio.shape[1:3]) != (32, 2) or audio.shape[0] != batch_size:
                    raise ValueError(f"MiniMax-H3 reference {index:03d} audio must be [B,32,2,A]")
                references.append(H3ReferenceGeometry("audio", audio_frames=audio.shape[-1]))
                audio_conditions.append(audio)
            else:
                raise ValueError(f"MiniMax-H3 reference {index:03d} has no supported role")
    else:
        task = "t2va"

    layout = build_h3_layout(
        task=task,
        text_length=hidden_states.shape[1],
        target_video=H3VideoGeometry(*video_latents.shape[2:]),
        target_audio_frames=audio_latents.shape[-1],
        visual_conditions=tuple(condition_geometries),
        references=tuple(references),
    )
    return _H3RuntimeBatch(
        layout=layout,
        text_hidden_states=hidden_states,
        text_token_tags=token_tags,
        visual_conditions=tuple(visual_conditions),
        audio_conditions=tuple(audio_conditions),
    )


def _shift_noise_amount(base: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * base / (1.0 + (shift - 1.0) * base)


def _sample_base_time(args, batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    lower = (0.0 if args.min_timestep is None else float(args.min_timestep)) / 1000.0
    upper = (1000.0 if args.max_timestep is None else float(args.max_timestep)) / 1000.0
    if not 0.0 <= lower <= upper <= 1.0:
        raise ValueError("MiniMax-H3 min_timestep/max_timestep must define a range inside [0,1000]")
    pool = batch.get("timesteps")
    if pool is None:
        base = torch.rand((1,), device=device, dtype=torch.float32)[0]
    else:
        pool = torch.as_tensor(pool, device=device, dtype=torch.float32)
        if pool.numel() != 1:
            raise ValueError("MiniMax-H3 R1 requires exactly one timestep value for its single-item batch")
        base = pool.reshape(())
    return lower + base * (upper - lower)


def _augment_conditions(
    tensors: tuple[torch.Tensor, ...],
    clean: float,
    seeds: torch.Tensor,
    *,
    seed_offset: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    moved = tuple(tensor.to(device) for tensor in tensors)
    if clean == 1.0:
        return moved
    augmented = []
    for tensor in moved:
        samples = []
        for sample, seed in zip(tensor, seeds):
            generator = torch.Generator(device="cpu").manual_seed(int(seed.item()) + seed_offset)
            noise = torch.randn(tuple(sample.shape), generator=generator, dtype=torch.float32, device="cpu").to(
                device=sample.device, dtype=sample.dtype
            )
            samples.append(clean * sample + (1.0 - clean) * noise)
        augmented.append(torch.stack(samples))
    return tuple(augmented)


class MiniMaxH3NetworkTrainer(NetworkTrainer):
    @property
    def architecture(self) -> str:
        return ARCHITECTURE_MINIMAX_H3

    @property
    def architecture_full_name(self) -> str:
        return ARCHITECTURE_MINIMAX_H3_FULL

    def handle_model_specific_args(self, args: argparse.Namespace):
        self.dit_dtype = torch.bfloat16
        self._i2v_training = False
        self._control_training = False
        self.default_guidance_scale = 1.0
        self.default_discrete_flow_shift = 1.0
        if getattr(args, "task", None) not in {"t2va", "fl2va", "ref2va"}:
            raise ValueError("MiniMax-H3 requires --task t2va, fl2va, or ref2va")
        if args.timestep_sampling != "uniform":
            raise ValueError("MiniMax-H3 supports --timestep_sampling uniform only")
        if args.weighting_scheme != "none":
            raise ValueError("MiniMax-H3 supports --weighting_scheme none only")
        if float(args.discrete_flow_shift) != 1.0:
            raise ValueError("MiniMax-H3 requires --discrete_flow_shift 1.0; use the two H3 shifts instead")
        for name in ("h3_shift_video", "h3_shift_audio"):
            value = float(getattr(args, name))
            if not 0.01 <= value <= 100.0:
                raise ValueError(f"--{name} must be in [0.01,100.0], got {value}")
        for name in ("h3_visual_cond_clean", "h3_audio_cond_clean"):
            value = float(getattr(args, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{name} must be in [0.0,1.0], got {value}")
        if args.blocks_to_swap is not None and args.blocks_to_swap > 48:
            raise ValueError("--blocks_to_swap for MiniMax-H3 must be <= 48")
        if getattr(args, "fp8_base", False) or getattr(args, "fp8_scaled", False):
            raise ValueError("MiniMax-H3 R1 accepts only a BF16 transformer base; quantized bases are deferred to R2")
        if getattr(args, "dit_dtype", None) not in {None, "bfloat16", "bf16"}:
            raise ValueError("MiniMax-H3 R1 requires --dit_dtype bfloat16")
        if getattr(args, "sample_prompts", None):
            raise ValueError(
                "MiniMax-H3 R1 does not support --sample_prompts during training; "
                "use minimax_h3_generate_video.py with a saved LoRA"
            )
        if (
            getattr(args, "block_swap_h2d_only", False)
            and bool(args.blocks_to_swap)
            and not getattr(args, "gradient_checkpointing", False)
        ):
            raise ValueError("MiniMax-H3 --block_swap_h2d_only training requires --gradient_checkpointing")

    def _build_dataset(self, args):
        dataset_group, collator, current_epoch = super()._build_dataset(args)
        validate_h3_dataset_batch_size(dataset_group)
        return dataset_group, collator, current_epoch

    def extra_metadata(self, args: argparse.Namespace) -> dict:
        return {
            "ss_minimax_h3_task": args.task,
            "ss_minimax_h3_base_family": "ref2va" if args.task == "ref2va" else "fl2va",
            "ss_minimax_h3_shift_video": args.h3_shift_video,
            "ss_minimax_h3_shift_audio": args.h3_shift_audio,
            "ss_minimax_h3_visual_cond_clean": args.h3_visual_cond_clean,
            "ss_minimax_h3_audio_cond_clean": args.h3_audio_cond_clean,
            "ss_minimax_h3_loss_policy": "video_mean_plus_audio_mean",
            "ss_minimax_h3_target_modules": "attn.qkv_proj,attn.out_proj,mlp.fc1,mlp.fc2",
            "ss_minimax_h3_latent_cache_version": "1",
            "ss_minimax_h3_text_cache_version": "1",
        }

    def load_transformer(
        self,
        accelerator: Accelerator,
        args: argparse.Namespace,
        dit_path: str,
        attn_mode: str,
        split_attn: bool,
        loading_device: str,
        dit_weight_dtype: torch.dtype | None,
    ):
        del accelerator
        if dit_weight_dtype not in {None, torch.bfloat16}:
            raise ValueError("MiniMax-H3 R1 transformer weights must stay BF16")
        return load_h3_transformer(
            dit_path,
            device=loading_device,
            dtype=torch.bfloat16,
            attn_mode=attn_mode,
            split_attn=split_attn,
            disable_mmap=getattr(args, "disable_numpy_memmap", False),
        )

    def compile_transformer(self, args, transformer):
        return model_utils.compile_transformer(
            args,
            transformer,
            [transformer.blocks],
            disable_linear=bool(self.blocks_to_swap),
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
        del batch, timesteps
        audio_latents = kwargs.pop("audio_latents")
        audio_noise = kwargs.pop("audio_noise")
        noisy_audio_input = kwargs.pop("noisy_audio_input")
        runtime = kwargs.pop("runtime")
        model_t_video = kwargs.pop("model_t_video")
        model_t_audio = kwargs.pop("model_t_audio")
        visual_conditions = kwargs.pop("visual_conditions")
        audio_conditions = kwargs.pop("audio_conditions")
        if kwargs:
            raise TypeError(f"Unexpected MiniMax-H3 call_dit arguments: {sorted(kwargs)}")

        text_hidden_states = runtime.text_hidden_states.to(device=accelerator.device, dtype=network_dtype)
        noisy_model_input = noisy_model_input.to(accelerator.device)
        noisy_audio_input = noisy_audio_input.to(accelerator.device)
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            noisy_audio_input.requires_grad_(True)
            text_hidden_states.requires_grad_(True)
        autocast = accelerator.autocast if hasattr(accelerator, "autocast") else nullcontext
        with autocast():
            prediction = transformer(
                video_latents=noisy_model_input,
                audio_latents=noisy_audio_input,
                text_hidden_states=text_hidden_states,
                text_token_tags=runtime.text_token_tags.to(accelerator.device),
                layout=runtime.layout,
                model_t_video=model_t_video,
                model_t_audio=model_t_audio,
                visual_condition_latents=visual_conditions,
                audio_condition_latents=audio_conditions,
                visual_condition_clean=args.h3_visual_cond_clean,
                audio_condition_clean=args.h3_audio_cond_clean,
            )
        return DiTOutput(
            pred=prediction.video,
            target=latents - noise,
            extra={"audio_pred": prediction.audio, "audio_target": audio_latents - audio_noise},
        )

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
        del network, vae
        runtime = _runtime_batch_plan(batch, latents)
        if runtime.layout.task != args.task:
            raise ValueError(f"MiniMax-H3 --task {args.task} cannot train a {runtime.layout.task.upper()} cache batch")
        device = latents.device
        audio_latents = batch["latents_audio"].to(device=device)
        audio_noise = torch.randn_like(audio_latents)
        base = _sample_base_time(args, batch, device)
        sigma_video = _shift_noise_amount(base, args.h3_shift_video)
        sigma_audio = _shift_noise_amount(base, args.h3_shift_audio)
        model_t_video = 1.0 - sigma_video
        model_t_audio = 1.0 - sigma_audio
        noisy_video = (1.0 - sigma_video) * latents + sigma_video * noise
        noisy_audio = (1.0 - sigma_audio) * audio_latents + sigma_audio * audio_noise

        needs_condition_noise = (bool(runtime.visual_conditions) and args.h3_visual_cond_clean != 1.0) or (
            bool(runtime.audio_conditions) and args.h3_audio_cond_clean != 1.0
        )
        condition_seeds = (
            torch.randint(0, 2**63 - 2, (latents.shape[0],), dtype=torch.int64, device="cpu")
            if needs_condition_noise
            else torch.empty(0, dtype=torch.int64)
        )
        visual_conditions = _augment_conditions(
            runtime.visual_conditions,
            args.h3_visual_cond_clean,
            condition_seeds,
            seed_offset=0,
            device=device,
        )
        audio_conditions = _augment_conditions(
            runtime.audio_conditions,
            args.h3_audio_cond_clean,
            condition_seeds,
            seed_offset=1,
            device=device,
        )
        output = self.call_dit(
            args,
            accelerator,
            transformer,
            latents,
            batch,
            noise,
            noisy_video,
            base,
            network_dtype,
            audio_latents=audio_latents,
            audio_noise=audio_noise,
            noisy_audio_input=noisy_audio,
            runtime=runtime,
            model_t_video=model_t_video,
            model_t_audio=model_t_audio,
            visual_conditions=visual_conditions,
            audio_conditions=audio_conditions,
        )
        return self.compute_loss(args, output, base, noise_scheduler, dit_dtype, network_dtype, global_step)

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
        del args, timesteps, noise_scheduler, dit_dtype, global_step
        video_loss = torch.nn.functional.mse_loss(
            output.pred.to(network_dtype),
            output.target.to(network_dtype),
            reduction="mean",
        )
        audio_loss = torch.nn.functional.mse_loss(
            output.extra["audio_pred"].to(network_dtype),
            output.extra["audio_target"].to(network_dtype),
            reduction="mean",
        )
        return video_loss + audio_loss, {
            "loss/video": video_loss.detach(),
            "loss/audio": audio_loss.detach(),
        }


def minimax_h3_setup_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(
        timestep_sampling="uniform",
        weighting_scheme="none",
        discrete_flow_shift=1.0,
        network_module="networks.lora_minimax_h3",
    )
    parser.add_argument("--task", choices=("t2va", "fl2va", "ref2va"), default=None, help="MiniMax-H3 training task")
    parser.add_argument("--h3_shift_video", type=float, default=12.0, help="MiniMax-H3 target-video flow shift")
    parser.add_argument("--h3_shift_audio", type=float, default=3.0, help="MiniMax-H3 target-audio flow shift")
    parser.add_argument(
        "--h3_visual_cond_clean",
        type=float,
        default=0.999,
        help="clean coefficient used to augment MiniMax-H3 visual conditions",
    )
    parser.add_argument(
        "--h3_audio_cond_clean",
        type=float,
        default=1.0,
        help="clean coefficient used to augment MiniMax-H3 audio conditions",
    )
    parser.add_argument("--dit_dtype", type=str, default=None, help="MiniMax-H3 DiT dtype; R1 requires bfloat16")
    return parser


def main() -> None:
    parser = minimax_h3_setup_parser(setup_parser_common())
    args = parser.parse_args()
    args = read_config_from_file(args, parser)
    args.dit_dtype = "bfloat16" if args.dit_dtype is None else args.dit_dtype
    args.vae_dtype = "bfloat16" if args.vae_dtype is None else args.vae_dtype
    MiniMaxH3NetworkTrainer().train(args)
