# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Canonical DLSS-NR 310.8.0 module.

Parameter names match the P0 checkpoint exactly. The default FP32 surrogate follows the block order,
channel widths, head-major QKV, window phases and skip wiring, but not native matrix accumulations.
Experimental compute/attention policies are model-local. See `numerics.SURROGATE_FLAGS`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.dlssnr.geometry import Geometry
from musubi_tuner.dlssnr.arithmetic import cubic_half, e4m3_ste, half_ste
from musubi_tuner.dlssnr.numerics import apply_linear, global_attention, nearest_upsample, pool2x2, window_attention
from musubi_tuner.dlssnr.profiles import build_records


def _phases() -> dict[int, int | None]:
    phases: dict[int, int | None] = {}
    for record in build_records():
        phases[record.block] = record.phase
    return phases


class ChannelLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.compute_dtype = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.materialized_weight(), x)

    def materialized_weight(self) -> torch.Tensor:
        weight = self.weight.float() if self.weight.dtype == torch.float8_e4m3fn else self.weight
        scale = getattr(self, "scale_weight", None)
        if scale is None:
            return weight
        if scale.ndim == 3:
            return (weight.reshape(scale.shape[0], scale.shape[1], 64) * scale).reshape_as(weight)
        return weight * scale

    def project(self, weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, dtype=self.compute_dtype, enabled=self.compute_dtype is not None):
            return apply_linear(weight, x).float()


class DenseFFN(nn.Module):
    def __init__(self, channels: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = ChannelLinear(hidden, channels)
        self.fc2 = ChannelLinear(channels, hidden)
        self.skip_scale = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        x = e4m3_ste(x)
        skip = x if residual is None else half_ste(residual)
        hidden = e4m3_ste(cubic_half(self.fc1(x)))
        result = half_ste(self.fc2(hidden) + half_ste(skip * half_ste(self.skip_scale).view(1, -1, 1, 1)))
        return result if x.shape[1] == 32 else e4m3_ste(result)


class ExpertFFN(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        experts = channels // 32
        self.experts = nn.ModuleList(
            [nn.ModuleDict({"fc1": ChannelLinear(128, channels), "fc2": ChannelLinear(32, 128)}) for _ in range(experts)]
        )
        self.fc3 = ChannelLinear(channels, channels)
        self.skip_scale = nn.Parameter(torch.empty(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = e4m3_ste(x)
        parts = [e4m3_ste(expert["fc2"](e4m3_ste(cubic_half(expert["fc1"](x))))) for expert in self.experts]
        return e4m3_ste(self.fc3(torch.cat(parts, dim=1)) + half_ste(x * half_ste(self.skip_scale).view(1, -1, 1, 1)))


class Split512FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "in_proj": ChannelLinear(64, 512),
                        "fc1": ChannelLinear(256, 64),
                        "fc2": ChannelLinear(64, 256),
                    }
                )
                for _ in range(8)
            ]
        )
        self.contract = ChannelLinear(512, 512)
        self.skip_scale = nn.Parameter(torch.empty(512))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = e4m3_ste(x)
        projected = e4m3_ste(torch.cat([branch["in_proj"](x) for branch in self.branches], dim=1))
        pieces = projected.split(64, dim=1)
        outputs = []
        for branch, piece in zip(self.branches, pieces):
            outputs.append(e4m3_ste(branch["fc2"](e4m3_ste(cubic_half(branch["fc1"](piece))))))
        return e4m3_ste(self.contract(torch.cat(outputs, dim=1)) + half_ste(x * half_ste(self.skip_scale).view(1, -1, 1, 1)))


class WindowAttn(nn.Module):
    def __init__(self, channels: int, phase: int) -> None:
        super().__init__()
        self.phase = phase
        self.compute_dtype = None
        self.attention_backend = "native"
        self.qkv = ChannelLinear(channels * 3, channels)
        self.proj = ChannelLinear(channels, channels)
        self.prior = nn.Parameter(torch.empty(channels // 32, 64, 64))
        self.temperature = nn.Parameter(torch.empty(channels // 32))
        self.skip_scale = nn.Parameter(torch.empty(channels))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return window_attention(
            y,
            self.qkv,
            self.proj,
            self.prior,
            self.temperature,
            self.skip_scale,
            self.phase,
            compute_dtype=self.compute_dtype,
            backend=self.attention_backend,
        )


class GlobalAttn(nn.Module):
    def __init__(self, channels: int = 1024) -> None:
        super().__init__()
        self.compute_dtype = None
        self.attention_backend = "native"
        self.qkv = ChannelLinear(channels * 3, channels)
        self.proj = ChannelLinear(channels, channels)
        self.temperature = nn.Parameter(torch.empty(channels // 32))
        self.skip_scale = nn.Parameter(torch.empty(channels))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return global_attention(
            y,
            self.qkv,
            self.proj,
            self.temperature,
            self.skip_scale,
            compute_dtype=self.compute_dtype,
            backend=self.attention_backend,
        )


class NRModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        phases = _phases()
        blocks = nn.ModuleDict()
        for block in range(71):
            blocks[str(block)] = self._make_block(block, phases[block])
        self.blocks = blocks
        self.gradient_checkpointing = False
        self.reset_parameters()

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    @staticmethod
    def _make_block(block: int, phase: int | None) -> nn.Module:
        module = nn.Module()
        if block == 39:
            module.proj = ChannelLinear(512, 1024)
            module.skip_scale = nn.Parameter(torch.empty(512))
            return module
        if 31 <= block <= 38:
            module.ffn = DenseFFN(1024, 4096)
            module.attn = GlobalAttn()
            return module
        if 23 <= block <= 30 or 40 <= block <= 47:
            module.ffn = Split512FFN()
            module.attn = WindowAttn(512, int(phase))
            if block == 30:
                module.to_vit = ChannelLinear(1024, 512)
            return module
        channels = _window_channels(block)
        if channels == 32:
            module.ffn = DenseFFN(32, 128)
        else:
            module.ffn = ExpertFFN(channels)
        module.attn = WindowAttn(channels, int(phase))
        if block == 0:
            module.input_adapter = ChannelLinear(32, 16)
        if block == 4:
            module.down = ChannelLinear(64, 32)
        elif block == 8:
            module.down = ChannelLinear(128, 64)
        elif block == 14:
            module.down = ChannelLinear(256, 128)
        elif block == 22:
            module.down = ChannelLinear(512, 256)
        if block == 48:
            _attach_up(module, 256, 512)
        elif block == 56:
            _attach_up(module, 128, 256)
        elif block == 62:
            _attach_up(module, 64, 128)
        elif block == 66:
            _attach_up(module, 32, 64)
        if block == 70:
            module.merge = nn.Module()
            module.merge.up_scale = nn.Parameter(torch.empty(32))
            module.merge.adapter_scale = nn.Parameter(torch.empty(32))
            module.head = nn.Module()
            module.head.rgb = ChannelLinear(3, 32)
            module.head.logit = ChannelLinear(1, 32)
            module.blend_scale = nn.Parameter(torch.empty(1))
        return module

    def reset_parameters(self) -> None:
        for name, parameter in self.named_parameters():
            if name.endswith("skip_scale") or name.endswith("up_scale") or name.endswith("adapter_scale"):
                nn.init.ones_(parameter)
            elif name.endswith("temperature"):
                nn.init.ones_(parameter)
            elif name.endswith("prior"):
                nn.init.zeros_(parameter)
            elif name.endswith("blend_scale"):
                nn.init.constant_(parameter, 0.73974609375)
            else:
                nn.init.normal_(parameter, std=1e-3)
        with torch.no_grad():
            self.blocks["0"].input_adapter.weight[:, 15].zero_()

    def freeze_single_frame(self) -> None:
        """Logit and blend scale have no supervision without history."""
        self.blocks["70"].head.logit.weight.requires_grad_(False)
        self.blocks["70"].blend_scale.requires_grad_(False)

    def enforce_lane15(self) -> None:
        weight = self.blocks["0"].input_adapter.weight
        # Frozen LoRA bases must retain even the source's signed-zero bit patterns.
        if weight.requires_grad:
            with torch.no_grad():
                weight[:, 15].zero_()
        if weight.grad is not None:
            weight.grad[:, 15] = 0

    def load_canonical(self, path: str) -> None:
        from safetensors.torch import load_file

        if getattr(self, "base_quantization", None) is not None:
            raise ValueError("load canonical weights into a fresh FP32 model before quantizing its frozen base")
        tensors = load_file(path)
        for block in range(31, 39):
            opaque = tensors.pop(f"blocks.{block}.opaque.layer3", None)
            if opaque is not None and (opaque.dtype != torch.uint8 or opaque.shape != (2,)):
                raise ValueError(f"block {block}: opaque layer3 must contain exactly two bytes")
        for name, value in tensors.items():
            if value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError(f"{name}: canonical parameters must be finite FP32")
        adapter = tensors.get("blocks.0.input_adapter.weight")
        if adapter is not None and torch.count_nonzero(adapter[:, 15]):
            raise ValueError("canonical input adapter lane 15 must remain zero")
        self.load_state_dict(tensors, strict=True)

    def forward(self, features: torch.Tensor, geometry: Geometry) -> torch.Tensor:
        """features [B, 16, full_h, full_w] -> raw head [B, 4, full_h, full_w]."""
        levels = geometry.levels
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        block0 = self.blocks["0"]
        adapted = half_ste(block0.input_adapter(half_ste(features)))
        full = _run_block(block0, adapted, checkpointing, separate_residual=True)
        state = pool2x2(full, levels[0][1], levels[0][0])
        state = _run_window_stage(self.blocks, state, range(1, 5), checkpointing=checkpointing)
        skip32 = state
        state = self.blocks["4"].down(pool2x2(state, levels[1][1], levels[1][0]))

        state = _run_window_stage(self.blocks, state, range(5, 9), checkpointing=checkpointing)
        skip64 = state
        state = self.blocks["8"].down(pool2x2(state, levels[2][1], levels[2][0]))

        state = _run_window_stage(self.blocks, state, range(9, 15), checkpointing=checkpointing)
        skip128 = state
        state = self.blocks["14"].down(pool2x2(state, levels[3][1], levels[3][0]))

        state = _run_window_stage(self.blocks, state, range(15, 23), checkpointing=checkpointing)
        skip256 = state
        state = self.blocks["22"].down(pool2x2(state, levels[4][1], levels[4][0]))

        state = _run_window_stage(self.blocks, state, range(23, 31), checkpointing=checkpointing)
        skip512 = state
        pooled = pool2x2(state, levels[5][1], levels[5][0])
        state = self.blocks["30"].to_vit(pooled)
        state = _run_window_stage(self.blocks, state, range(31, 39), checkpointing=checkpointing)

        block39 = self.blocks["39"]
        projected = half_ste(block39.proj(e4m3_ste(state)))
        state = _merge_skip(projected, skip512, block39.skip_scale, levels[4])
        state = _run_window_stage(self.blocks, state, range(40, 48), checkpointing=checkpointing)

        state = _decode_stage(self.blocks, state, 48, range(48, 56), skip256, levels[3], checkpointing=checkpointing)
        state = _decode_stage(self.blocks, state, 56, range(56, 62), skip128, levels[2], checkpointing=checkpointing)
        state = _decode_stage(self.blocks, state, 62, range(62, 66), skip64, levels[1], checkpointing=checkpointing)
        state = _decode_stage(self.blocks, state, 66, range(66, 70), skip32, levels[0], checkpointing=checkpointing)

        block70 = self.blocks["70"]
        upsampled = nearest_upsample(e4m3_ste(state), geometry.full_height, geometry.full_width)
        low = half_ste(upsampled * half_ste(block70.merge.up_scale).view(1, -1, 1, 1))
        merged = half_ste(low + e4m3_ste(full) * half_ste(block70.merge.adapter_scale).view(1, -1, 1, 1))
        state = _run_block(block70, merged, checkpointing, separate_residual=True)
        rgb = block70.head.rgb(state)
        logit = block70.head.logit(state)
        return torch.cat((rgb, logit), dim=1)


def _window_channels(block: int) -> int:
    if block <= 4 or block >= 66:
        return 32
    if block <= 8 or block >= 62:
        return 64
    if block <= 14 or block >= 56:
        return 128
    return 256


def _attach_up(module: nn.Module, out_channels: int, in_channels: int) -> None:
    module.up = ChannelLinear(out_channels, in_channels)
    module.up.skip_scale = nn.Parameter(torch.empty(out_channels))


def _run_block(block, state, checkpointing, *, separate_residual=False):
    # Bind this block outside the stage loop so backward cannot replay a later block.
    def forward(value):
        hidden = block.ffn(value, residual=value) if separate_residual else block.ffn(value)
        return block.attn(hidden)

    if checkpointing and block.training and torch.is_grad_enabled():
        return checkpoint(forward, state, use_reentrant=False, preserve_rng_state=True)
    return forward(state)


def _run_window_stage(blocks: nn.ModuleDict, state: torch.Tensor, block_ids: range, *, checkpointing=False) -> torch.Tensor:
    for block_id in block_ids:
        block = blocks[str(block_id)]
        state = _run_block(block, state, checkpointing)
    return state


def _merge_skip(projected: torch.Tensor, skip: torch.Tensor, scale: torch.Tensor, high: tuple[int, int]) -> torch.Tensor:
    upsampled = nearest_upsample(projected, high[1], high[0])
    return half_ste(upsampled + e4m3_ste(skip) * half_ste(scale).view(1, -1, 1, 1))


def _decode_stage(
    blocks, state, first: int, block_ids: range, skip: torch.Tensor, high: tuple[int, int], *, checkpointing=False
) -> torch.Tensor:
    block = blocks[str(first)]
    projected = half_ste(block.up(e4m3_ste(state)))
    state = _merge_skip(projected, skip, block.up.skip_scale, high)
    if first == 66:
        state = _run_block(block, state, checkpointing, separate_residual=True)
        return _run_window_stage(blocks, state, range(first + 1, block_ids.stop), checkpointing=checkpointing)
    return _run_window_stage(blocks, state, block_ids, checkpointing=checkpointing)
