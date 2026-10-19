# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""Canonical DLSS-NR HT/CC module.

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
from musubi_tuner.dlssnr.profiles import PROFILE_ID, block_channels, build_records, get_profile
from musubi_tuner.dlssnr.weight_quantization import native_weight_ste


def _phases(profile: str = PROFILE_ID) -> dict[int, int | None]:
    phases: dict[int, int | None] = {}
    for record in build_records(profile):
        phases[record.block] = record.phase
    return phases


class ChannelLinear(nn.Module):
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.compute_dtype = None
        self.native_weight_kind = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.published_weight(self.materialized_weight()), x)

    def published_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if self.native_weight_kind is None:
            return weight
        return native_weight_ste(weight, self.native_weight_kind, validation=getattr(self, "_native_weight_validation", None))

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
    def __init__(self, profile: str = PROFILE_ID) -> None:
        super().__init__()
        self.profile = get_profile(profile).id
        phases = _phases(profile)
        blocks = nn.ModuleDict()
        for block in range(get_profile(profile).block_count):
            blocks[str(block)] = self._make_block(block, phases[block], profile)
        self.blocks = blocks
        self.gradient_checkpointing = False
        self.gradient_checkpointing_min_pixels = 0
        self.reset_parameters()

    def enable_gradient_checkpointing(self, min_pixels: int = 0) -> None:
        if type(min_pixels) is not int or min_pixels < 0:
            raise ValueError("gradient_checkpointing_min_pixels must be a nonnegative integer")
        self.gradient_checkpointing = True
        self.gradient_checkpointing_min_pixels = min_pixels

    @staticmethod
    def _make_block(block: int, phase: int | None, profile: str = PROFILE_ID) -> nn.Module:
        topology = get_profile(profile)
        module = nn.Module()
        if block == topology.bridge_block:
            module.proj = ChannelLinear(512, 1024)
            module.skip_scale = nn.Parameter(torch.empty(512))
            return module
        if block in topology.vit_blocks:
            module.ffn = DenseFFN(1024, 4096)
            module.attn = GlobalAttn()
            return module
        channels = block_channels(block, profile)
        if channels == 512:
            module.ffn = Split512FFN()
            module.attn = WindowAttn(512, int(phase))
            if block == topology.vit_blocks.start - 1:
                module.to_vit = ChannelLinear(1024, 512)
            return module
        if channels == 32:
            module.ffn = DenseFFN(32, 128)
        else:
            module.ffn = ExpertFFN(channels)
        module.attn = WindowAttn(channels, int(phase))
        if block == 0:
            module.input_adapter = ChannelLinear(32, 16)
        if any(block == stage.stop - 1 for stage in topology.encoder_blocks[:-1]):
            module.down = ChannelLinear(channels * 2, channels)
        if any(block == stage.start for stage in topology.decoder_blocks[1:]):
            _attach_up(module, channels, channels * 2)
        if block == topology.head_block:
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
        head = self.blocks[str(get_profile(getattr(self, "profile", PROFILE_ID)).head_block)]
        head.head.logit.weight.requires_grad_(False)
        head.blend_scale.requires_grad_(False)

    def enforce_lane15(self) -> None:
        weight = self.blocks["0"].input_adapter.weight
        # Frozen LoRA bases must retain even the source's signed-zero bit patterns.
        if weight.requires_grad:
            with torch.no_grad():
                weight[:, 15].zero_()
        if weight.grad is not None:
            weight.grad[:, 15] = 0

    def load_canonical(self, path: str) -> None:
        from musubi_tuner.dlssnr.storage import load_canonical_tensors

        if getattr(self, "base_quantization", None) is not None:
            raise ValueError("load canonical weights into a fresh FP32 model before quantizing its frozen base")
        tensors = load_canonical_tensors(path, getattr(self, "profile", PROFILE_ID))
        for block in get_profile(getattr(self, "profile", PROFILE_ID)).vit_blocks:
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
        topology = get_profile(self.profile)
        levels = geometry.levels
        checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        min_pixels = self.gradient_checkpointing_min_pixels
        block0 = self.blocks["0"]
        adapted = half_ste(block0.input_adapter(half_ste(features)))
        full = _run_block(block0, adapted, checkpointing, separate_residual=True, min_pixels=min_pixels)
        state = pool2x2(full, levels[0][1], levels[0][0])
        skips = []
        for level, blocks in enumerate(topology.encoder_blocks):
            state = _run_window_stage(self.blocks, state, blocks, checkpointing=checkpointing, min_pixels=min_pixels)
            skips.append(state)
            pooled = pool2x2(state, levels[level + 1][1], levels[level + 1][0])
            last = self.blocks[str(blocks[-1])]
            state = last.to_vit(pooled) if level == 4 else last.down(pooled)
        state = _run_window_stage(self.blocks, state, topology.vit_blocks, checkpointing=checkpointing, min_pixels=min_pixels)

        bridge = self.blocks[str(topology.bridge_block)]
        projected = half_ste(bridge.proj(e4m3_ste(state)))
        state = _merge_skip(projected, skips[-1], bridge.skip_scale, levels[4])
        state = _run_window_stage(
            self.blocks, state, topology.decoder_blocks[0], checkpointing=checkpointing, min_pixels=min_pixels
        )
        for level, blocks in zip(range(3, -1, -1), topology.decoder_blocks[1:]):
            state = _decode_stage(
                self.blocks,
                state,
                blocks.start,
                blocks,
                skips[level],
                levels[level],
                checkpointing=checkpointing,
                min_pixels=min_pixels,
            )

        head = self.blocks[str(topology.head_block)]
        upsampled = nearest_upsample(e4m3_ste(state), geometry.full_height, geometry.full_width)
        low = half_ste(upsampled * half_ste(head.merge.up_scale).view(1, -1, 1, 1))
        merged = half_ste(low + e4m3_ste(full) * half_ste(head.merge.adapter_scale).view(1, -1, 1, 1))
        state = _run_block(head, merged, checkpointing, separate_residual=True, min_pixels=min_pixels)
        rgb = head.head.rgb(state)
        logit = head.head.logit(state)
        return torch.cat((rgb, logit), dim=1)


def _attach_up(module: nn.Module, out_channels: int, in_channels: int) -> None:
    module.up = ChannelLinear(out_channels, in_channels)
    module.up.skip_scale = nn.Parameter(torch.empty(out_channels))


def _run_block(block, state, checkpointing, *, separate_residual=False, min_pixels=0):
    # Bind this block outside the stage loop so backward cannot replay a later block.
    def forward(value):
        hidden = block.ffn(value, residual=value) if separate_residual else block.ffn(value)
        return block.attn(hidden)

    if checkpointing and block.training and torch.is_grad_enabled() and state.shape[-2] * state.shape[-1] > min_pixels:
        return checkpoint(forward, state, use_reentrant=False, preserve_rng_state=True)
    return forward(state)


def _run_window_stage(
    blocks: nn.ModuleDict, state: torch.Tensor, block_ids: range, *, checkpointing=False, min_pixels=0
) -> torch.Tensor:
    for block_id in block_ids:
        block = blocks[str(block_id)]
        state = _run_block(block, state, checkpointing, min_pixels=min_pixels)
    return state


def _merge_skip(projected: torch.Tensor, skip: torch.Tensor, scale: torch.Tensor, high: tuple[int, int]) -> torch.Tensor:
    upsampled = nearest_upsample(projected, high[1], high[0])
    return half_ste(upsampled + e4m3_ste(skip) * half_ste(scale).view(1, -1, 1, 1))


def _decode_stage(
    blocks, state, first: int, block_ids: range, skip: torch.Tensor, high: tuple[int, int], *, checkpointing=False, min_pixels=0
) -> torch.Tensor:
    block = blocks[str(first)]
    projected = half_ste(block.up(e4m3_ste(state)))
    state = _merge_skip(projected, skip, block.up.skip_scale, high)
    if block.up.weight.shape[0] == 32:
        state = _run_block(block, state, checkpointing, separate_residual=True, min_pixels=min_pixels)
        return _run_window_stage(
            blocks, state, range(first + 1, block_ids.stop), checkpointing=checkpointing, min_pixels=min_pixels
        )
    return _run_window_stage(blocks, state, block_ids, checkpointing=checkpointing, min_pixels=min_pixels)
