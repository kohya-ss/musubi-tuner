from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from musubi_tuner.dataset.architectures import (
    ARCHITECTURE_MAGE_FLOW,
    ARCHITECTURE_MAGE_FLOW_EDIT,
    ARCHITECTURE_MAGE_FLOW_EDIT_FULL,
    ARCHITECTURE_MAGE_FLOW_FULL,
)


ImageShape = tuple[int, int, int]


@dataclass(frozen=True)
class MageFlowConfig:
    in_channels: int = 128
    out_channels: int = 128
    context_in_dim: int = 2560
    hidden_size: int = 3072
    depth: int = 12
    num_heads: int = 24
    axes_dim: tuple[int, ...] = (16, 56, 56)
    patch_size: int = 1
    text_max_length: int = 2048
    static_shift: float = 6.0
    checkpoint: bool = False

    def __post_init__(self) -> None:
        positive = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "context_in_dim": self.context_in_dim,
            "hidden_size": self.hidden_size,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "patch_size": self.patch_size,
            "text_max_length": self.text_max_length,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if sum(self.axes_dim) != self.hidden_size // self.num_heads:
            raise ValueError(
                f"RoPE axes sum ({sum(self.axes_dim)}) must equal head dimension "
                f"({self.hidden_size // self.num_heads})"
            )
        if any(axis <= 0 or axis % 2 for axis in self.axes_dim):
            raise ValueError(f"RoPE axes must contain positive even dimensions, got {self.axes_dim}")

    @classmethod
    def released(cls, *, checkpoint: bool = False) -> "MageFlowConfig":
        return cls(checkpoint=checkpoint)


@dataclass(frozen=True)
class PackedMageFlowInputs:
    image_tokens: torch.Tensor
    image_cu_seqlens: torch.Tensor
    text_tokens: torch.Tensor
    text_cu_seqlens: torch.Tensor
    image_shapes: list[list[ImageShape]]
    timesteps: torch.Tensor
    target_token_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.image_cu_seqlens.numel() - 1)

    def validate(self, image_dim: int, text_dim: int, text_max_length: int = 2048) -> None:
        validate_packed_inputs(self, image_dim=image_dim, text_dim=text_dim, text_max_length=text_max_length)


def architecture_for_mode(is_edit: bool) -> tuple[str, str]:
    if is_edit:
        return ARCHITECTURE_MAGE_FLOW_EDIT, ARCHITECTURE_MAGE_FLOW_EDIT_FULL
    return ARCHITECTURE_MAGE_FLOW, ARCHITECTURE_MAGE_FLOW_FULL


def _validate_cumulative_lengths(name: str, cu: torch.Tensor, token_count: int) -> list[int]:
    if cu.ndim != 1 or cu.numel() < 2:
        raise ValueError(f"{name} must be a rank-1 tensor with at least two entries")
    if cu.dtype != torch.int32:
        raise ValueError(f"{name} must use torch.int32, got {cu.dtype}")
    values = cu.detach().cpu().tolist()
    if values[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if values[-1] != token_count:
        raise ValueError(f"{name} must end at token count {token_count}, got {values[-1]}")
    if any(end <= start for start, end in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def validate_packed_inputs(
    packed: PackedMageFlowInputs,
    *,
    image_dim: int,
    text_dim: int,
    text_max_length: int = 2048,
) -> None:
    if packed.image_tokens.ndim != 3 or packed.image_tokens.shape[0] != 1:
        raise ValueError(f"image_tokens must have shape [1, total, {image_dim}]")
    if packed.image_tokens.shape[-1] != image_dim:
        raise ValueError(f"image feature dimension must be {image_dim}, got {packed.image_tokens.shape[-1]}")
    if packed.text_tokens.ndim != 3 or packed.text_tokens.shape[0] != 1:
        raise ValueError(f"text_tokens must have shape [1, total, {text_dim}]")
    if packed.text_tokens.shape[-1] != text_dim:
        raise ValueError(f"text feature dimension must be {text_dim}, got {packed.text_tokens.shape[-1]}")
    if not torch.isfinite(packed.image_tokens).all() or not torch.isfinite(packed.text_tokens).all():
        raise ValueError("packed image and text tokens must contain only finite values")

    image_cu = _validate_cumulative_lengths(
        "image_cu_seqlens", packed.image_cu_seqlens, packed.image_tokens.shape[1]
    )
    text_cu = _validate_cumulative_lengths("text_cu_seqlens", packed.text_cu_seqlens, packed.text_tokens.shape[1])
    batch_size = len(image_cu) - 1
    if len(text_cu) - 1 != batch_size:
        raise ValueError("image and text cumulative lengths must describe the same batch size")
    if len(packed.image_shapes) != batch_size:
        raise ValueError(f"image_shapes outer length must equal batch size {batch_size}")
    if packed.timesteps.ndim != 1 or packed.timesteps.shape[0] != batch_size:
        raise ValueError(f"timesteps must have shape [{batch_size}]")
    if not torch.isfinite(packed.timesteps).all():
        raise ValueError("timesteps must contain only finite values")
    if torch.any((packed.timesteps < 0) | (packed.timesteps > 1)):
        raise ValueError("timesteps must be denoising sigmas in the inclusive range [0, 1]")
    if packed.target_token_mask.dtype != torch.bool or packed.target_token_mask.ndim != 1:
        raise ValueError("target_token_mask must be a rank-1 bool tensor")
    if packed.target_token_mask.numel() != packed.image_tokens.shape[1]:
        raise ValueError("target_token_mask length must equal the packed image token count")

    for sample_index, (img_start, img_end, txt_start, txt_end, shapes) in enumerate(
        zip(image_cu, image_cu[1:], text_cu, text_cu[1:], packed.image_shapes)
    ):
        text_length = txt_end - txt_start
        if not 1 <= text_length <= text_max_length:
            raise ValueError(
                f"text segment {sample_index} length must be between 1 and {text_max_length}, got {text_length}"
            )
        if not 1 <= len(shapes) <= 4:
            raise ValueError(f"sample {sample_index} must contain one target and at most three references")
        segment_length = 0
        for shape in shapes:
            if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] <= 0:
                raise ValueError(f"invalid image shape {shape!r} for sample {sample_index}; expected (1, H, W)")
            segment_length += shape[0] * shape[1] * shape[2]
        if segment_length != img_end - img_start:
            raise ValueError(
                f"image shapes for sample {sample_index} describe {segment_length} tokens, "
                f"but its segment contains {img_end - img_start}"
            )
        target_length = shapes[0][1] * shapes[0][2]
        expected_mask = torch.zeros(img_end - img_start, dtype=torch.bool, device=packed.target_token_mask.device)
        expected_mask[:target_length] = True
        if not torch.equal(packed.target_token_mask[img_start:img_end], expected_mask):
            raise ValueError(f"target_token_mask does not select only the target frame for sample {sample_index}")


def _normalize_targets(targets: torch.Tensor | Sequence[torch.Tensor]) -> list[torch.Tensor]:
    if isinstance(targets, torch.Tensor):
        if targets.ndim != 4:
            raise ValueError(f"batched targets must have shape [B,C,H,W], got {tuple(targets.shape)}")
        normalized = list(targets.unbind(0))
    else:
        normalized = list(targets)
    if not normalized:
        raise ValueError("targets must contain at least one sample")
    for index, target in enumerate(normalized):
        if not isinstance(target, torch.Tensor) or target.ndim != 3:
            shape = tuple(target.shape) if isinstance(target, torch.Tensor) else type(target).__name__
            raise ValueError(f"target {index} must have shape [C,H,W], got {shape}")
        if target.shape[1] <= 0 or target.shape[2] <= 0:
            raise ValueError(f"target {index} must have positive spatial dimensions")
    return normalized


def _normalize_controls(
    controls: Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]],
    batch_size: int,
) -> list[list[torch.Tensor]]:
    raw = list(controls)
    if raw and all(isinstance(control, torch.Tensor) and control.ndim == 4 for control in raw):
        for control_index, control in enumerate(raw):
            if control.shape[0] != batch_size:
                raise ValueError(
                    f"control batch {control_index} has batch size {control.shape[0]}, expected {batch_size}"
                )
        per_sample = [[control[sample_index] for control in raw] for sample_index in range(batch_size)]
    else:
        if len(raw) != batch_size:
            raise ValueError(f"per-sample controls outer length must equal batch size {batch_size}")
        per_sample = [list(sample_controls) for sample_controls in raw]

    for sample_index, sample_controls in enumerate(per_sample):
        if not 1 <= len(sample_controls) <= 3:
            raise ValueError(
                f"Edit sample {sample_index} must contain between 1 and 3 ordered references, "
                f"got {len(sample_controls)}"
            )
        for control_index, control in enumerate(sample_controls):
            if not isinstance(control, torch.Tensor) or control.ndim != 3:
                raise ValueError(f"control {control_index} for sample {sample_index} must have shape [C,H,W]")
            if control.shape[1] <= 0 or control.shape[2] <= 0:
                raise ValueError(f"control {control_index} for sample {sample_index} has an empty spatial dimension")
    return per_sample


def pack_training_batch(
    targets: torch.Tensor | Sequence[torch.Tensor],
    text_tokens: Sequence[torch.Tensor],
    timesteps: torch.Tensor,
    controls: Sequence[torch.Tensor] | Sequence[Sequence[torch.Tensor]] | None = None,
    *,
    image_dim: int | None = None,
    text_dim: int | None = None,
    text_max_length: int = 2048,
) -> PackedMageFlowInputs:
    target_list = _normalize_targets(targets)
    batch_size = len(target_list)
    text_list = list(text_tokens)
    if len(text_list) != batch_size:
        raise ValueError(f"text batch size {len(text_list)} does not match target batch size {batch_size}")
    if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
        raise ValueError(f"timesteps batch size must be {batch_size}, got shape {tuple(timesteps.shape)}")

    detected_image_dim = int(target_list[0].shape[0])
    image_dim = detected_image_dim if image_dim is None else image_dim
    device = target_list[0].device
    dtype = target_list[0].dtype
    for index, target in enumerate(target_list):
        if target.shape[0] != image_dim:
            raise ValueError(f"target {index} has {target.shape[0]} channels, expected {image_dim}")
        if target.device != device:
            raise ValueError("all target tensors must be on the same device")
        if target.dtype != dtype:
            raise ValueError("all target tensors must use the same dtype")

    per_sample_controls = _normalize_controls(controls, batch_size) if controls is not None else [[] for _ in target_list]
    detected_text_dim = int(text_list[0].shape[-1]) if text_list and text_list[0].ndim == 2 else -1
    text_dim = detected_text_dim if text_dim is None else text_dim

    flat_images: list[torch.Tensor] = []
    flat_text: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    image_shapes: list[list[ImageShape]] = []
    image_lengths: list[int] = []
    text_lengths: list[int] = []

    for sample_index, (target, sample_text, sample_controls) in enumerate(
        zip(target_list, text_list, per_sample_controls)
    ):
        if not isinstance(sample_text, torch.Tensor) or sample_text.ndim != 2:
            raise ValueError(f"text sample {sample_index} must have shape [L,D]")
        if sample_text.shape[0] < 1 or sample_text.shape[0] > text_max_length:
            raise ValueError(
                f"text sample {sample_index} length must be between 1 and {text_max_length}, got {sample_text.shape[0]}"
            )
        if sample_text.shape[1] != text_dim:
            raise ValueError(f"text feature dimension must be {text_dim}, got {sample_text.shape[1]}")

        sample_images = [target]
        for control_index, control in enumerate(sample_controls):
            if control.shape[0] != image_dim:
                raise ValueError(
                    f"control {control_index} for sample {sample_index} has {control.shape[0]} channels, "
                    f"expected {image_dim}"
                )
            sample_images.append(control.to(device=device, dtype=dtype))

        sample_shapes = [(1, int(image.shape[1]), int(image.shape[2])) for image in sample_images]
        sample_flat = [image.permute(1, 2, 0).reshape(-1, image_dim) for image in sample_images]
        target_length = sample_flat[0].shape[0]
        sample_length = sum(image.shape[0] for image in sample_flat)
        sample_mask = torch.zeros(sample_length, device=device, dtype=torch.bool)
        sample_mask[:target_length] = True

        flat_images.extend(sample_flat)
        flat_text.append(sample_text.to(device=device))
        masks.append(sample_mask)
        image_shapes.append(sample_shapes)
        image_lengths.append(sample_length)
        text_lengths.append(int(sample_text.shape[0]))

    image_cu = torch.tensor([0, *torch.tensor(image_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    text_cu = torch.tensor([0, *torch.tensor(text_lengths).cumsum(0).tolist()], device=device, dtype=torch.int32)
    packed = PackedMageFlowInputs(
        image_tokens=torch.cat(flat_images, dim=0).unsqueeze(0),
        image_cu_seqlens=image_cu,
        text_tokens=torch.cat(flat_text, dim=0).unsqueeze(0),
        text_cu_seqlens=text_cu,
        image_shapes=image_shapes,
        timesteps=timesteps.to(device=device),
        target_token_mask=torch.cat(masks, dim=0),
    )
    packed.validate(image_dim=image_dim, text_dim=text_dim, text_max_length=text_max_length)
    return packed
