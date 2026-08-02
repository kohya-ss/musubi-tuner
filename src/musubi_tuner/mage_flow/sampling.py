from __future__ import annotations

from collections.abc import Sequence

import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from .training import unpack_target_predictions
from .utils import pack_training_batch


def build_scheduler(
    num_steps: int,
    *,
    device: str | torch.device | None = None,
    shift: float = 6.0,
) -> FlowMatchEulerDiscreteScheduler:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=shift,
        use_dynamic_shifting=False,
    )
    # diffusers 0.32.1 annotates this as a list but performs NumPy arithmetic.
    base_sigmas = torch.linspace(1.0, 1.0 / num_steps, num_steps, dtype=torch.float32).numpy()
    scheduler.set_timesteps(sigmas=base_sigmas, device=device)
    return scheduler


def euler_step(
    latent: torch.Tensor,
    velocity: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
) -> torch.Tensor:
    """Advance ``x`` along Mage-Flow's ``epsilon - z`` velocity field."""
    # Preserve the pinned scheduler's FP32 accumulation without delegating the step.
    sigma_tensor = torch.as_tensor(sigma, device=latent.device, dtype=torch.float32)
    next_sigma_tensor = torch.as_tensor(next_sigma, device=latent.device, dtype=torch.float32)
    stepped = latent.to(torch.float32) + (next_sigma_tensor - sigma_tensor) * velocity
    return stepped.to(velocity.dtype)


def scheduler_step_targets_only(
    image_tokens: torch.Tensor,
    velocity: torch.Tensor,
    target_token_mask: torch.Tensor,
    *,
    sigma: float | torch.Tensor,
    next_sigma: float | torch.Tensor,
) -> torch.Tensor:
    if image_tokens.shape != velocity.shape:
        raise ValueError("image tokens and velocity must have identical shapes")
    if target_token_mask.dtype != torch.bool or target_token_mask.shape != (image_tokens.shape[1],):
        raise ValueError("target_token_mask must be a bool vector matching the packed token count")
    stepped = image_tokens.clone()
    stepped[:, target_token_mask] = euler_step(
        image_tokens[:, target_token_mask],
        velocity[:, target_token_mask],
        sigma=sigma,
        next_sigma=next_sigma,
    )
    return stepped


def _target_list(targets: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(targets, torch.Tensor):
        if targets.ndim != 4:
            raise ValueError("batched target latents must have shape [B,C,H,W]")
        return list(targets.unbind(0))
    items = list(targets)
    if not items:
        raise ValueError("at least one target latent is required")
    return items


def _control_lists(
    controls: Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None,
    batch_size: int,
) -> list[list[torch.Tensor]]:
    if controls is None:
        return [[] for _ in range(batch_size)]
    items = list(controls)
    if items and all(isinstance(item, torch.Tensor) and item.ndim == 4 for item in items):
        if any(item.shape[0] != batch_size for item in items):
            raise ValueError("each reference batch must match the target batch size")
        return [[item[sample_index] for item in items] for sample_index in range(batch_size)]
    if len(items) != batch_size:
        raise ValueError("per-sample reference lists must match the target batch size")
    return [list(item) for item in items]


@torch.no_grad()
def predict_target_velocity(
    transformer,
    targets: torch.Tensor | Sequence[torch.Tensor],
    text_tokens: Sequence[torch.Tensor],
    *,
    sigma: float,
    controls: Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None = None,
    negative_text_tokens: Sequence[torch.Tensor] | None = None,
    cfg_scale: float = 1.0,
    renormalize_cfg: bool = False,
) -> list[torch.Tensor]:
    target_items = _target_list(targets)
    positive_text = list(text_tokens)
    if len(positive_text) != len(target_items):
        raise ValueError("positive conditioning count must match target count")
    control_items = _control_lists(controls, len(target_items))
    use_controls = any(control_items)
    use_cfg = negative_text_tokens is not None and cfg_scale > 1.0

    if use_cfg:
        negative_text = list(negative_text_tokens)
        if len(negative_text) != len(target_items):
            raise ValueError("negative conditioning count must match target count")
        packed_targets = target_items + target_items
        packed_text = positive_text + negative_text
        packed_controls = control_items + control_items if use_controls else None
    else:
        packed_targets = target_items
        packed_text = positive_text
        packed_controls = control_items if use_controls else None

    reference = target_items[0]
    timesteps = torch.full(
        (len(packed_targets),),
        float(sigma),
        device=reference.device,
        dtype=torch.float32,
    )
    packed = pack_training_batch(
        packed_targets,
        packed_text,
        timesteps,
        packed_controls,
        image_dim=reference.shape[0],
        text_dim=positive_text[0].shape[-1],
    )
    predictions = unpack_target_predictions(transformer(packed), packed)
    if not use_cfg:
        return predictions

    batch_size = len(target_items)
    conditional = predictions[:batch_size]
    unconditional = predictions[batch_size:]
    guided = [uncond + cfg_scale * (cond - uncond) for cond, uncond in zip(conditional, unconditional)]
    if renormalize_cfg:
        normalized = []
        for cond, combined in zip(conditional, guided):
            cond_norm = torch.linalg.vector_norm(cond, dim=0, keepdim=True)
            combined_norm = torch.linalg.vector_norm(combined, dim=0, keepdim=True)
            normalized.append(combined * (cond_norm / (combined_norm + 1e-6)))
        guided = normalized
    return guided


def _align_down(value: int) -> int:
    if value <= 0:
        raise ValueError("image dimensions must be positive")
    return max(16, 16 * (int(value) // 16))


def resolve_output_size(
    source_size: tuple[int, int],
    *,
    width: int | None,
    height: int | None,
    max_size: int | None,
) -> tuple[int, int]:
    """Resolve an Edit output size as ``(width, height)`` using official precedence."""
    if (width is None) != (height is None):
        raise ValueError("--width and --height must be provided together")
    source_width, source_height = source_size
    if width is not None and height is not None:
        return _align_down(width), _align_down(height)

    longest = max_size if max_size is not None else max(source_width, source_height)
    if longest <= 0:
        raise ValueError("max_size must be positive")
    if source_height >= source_width:
        resolved_height = longest
        resolved_width = round(source_width * longest / source_height)
    else:
        resolved_width = longest
        resolved_height = round(source_height * longest / source_width)
    return _align_down(resolved_width), _align_down(resolved_height)


@torch.no_grad()
def sample_latents(
    transformer,
    text_tokens: Sequence[torch.Tensor],
    latent_shapes: Sequence[tuple[int, int]],
    *,
    steps: int,
    seeds: Sequence[int],
    channels: int = 128,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    controls: Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None = None,
    negative_text_tokens: Sequence[torch.Tensor] | None = None,
    cfg_scale: float = 1.0,
    shift: float = 6.0,
    renormalize_cfg: bool = False,
) -> list[torch.Tensor]:
    shapes = list(latent_shapes)
    seed_list = list(seeds)
    conditioning = list(text_tokens)
    if not shapes or len(shapes) != len(seed_list) or len(shapes) != len(conditioning):
        raise ValueError("latent shapes, seeds, and conditioning must have the same non-zero length")
    if channels <= 0:
        raise ValueError("latent channels must be positive")
    target_device = torch.device(device)
    latents = []
    for (height, width), seed in zip(shapes, seed_list):
        if height <= 0 or width <= 0:
            raise ValueError("latent spatial dimensions must be positive")
        generator = torch.Generator(device=target_device).manual_seed(int(seed))
        latent = torch.randn(
            channels,
            height,
            width,
            generator=generator,
            device=target_device,
            dtype=torch.float32,
        ).to(dtype=dtype)
        latents.append(latent)

    scheduler = build_scheduler(steps, device=target_device, shift=shift)
    for step_index in range(len(scheduler.timesteps)):
        sigma = float(scheduler.sigmas[step_index].item())
        next_sigma = float(scheduler.sigmas[step_index + 1].item())
        velocities = predict_target_velocity(
            transformer,
            latents,
            conditioning,
            sigma=sigma,
            controls=controls,
            negative_text_tokens=negative_text_tokens,
            cfg_scale=cfg_scale,
            renormalize_cfg=renormalize_cfg,
        )
        latents = [
            euler_step(latent, velocity, sigma=sigma, next_sigma=next_sigma) for latent, velocity in zip(latents, velocities)
        ]
    return latents


__all__ = [
    "build_scheduler",
    "euler_step",
    "predict_target_velocity",
    "resolve_output_size",
    "sample_latents",
    "scheduler_step_targets_only",
]
