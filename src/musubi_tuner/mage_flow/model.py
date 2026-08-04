"""Packed Mage-Flow transformer adapted from Microsoft Mage commit ea7109b.

Copyright (c) 2026 Microsoft. Licensed under the MIT License.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from safetensors.torch import load_file

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

from .layers import AdaLayerNormContinuous, MageFlowEmbedRope, MageFlowTimestepProjEmbeddings, MageFlowTransformerBlock
from .utils import ComponentValidationError, MageFlowConfig, PackedMageFlowInputs, inspect_component, normalize_dit_state_dict


FP8_OPTIMIZATION_TARGET_KEYS = ["transformer_blocks."]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["img_mod", "txt_mod", "norm"]


class MageFlow(nn.Module):
    def __init__(self, config: MageFlowConfig, attention_backend: str = "sdpa"):
        super().__init__()
        self.config = config
        self.params = config
        self.checkpoint = config.checkpoint
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.inner_dim = config.hidden_size
        self.axes_dim = config.axes_dim
        self.num_attention_heads = config.num_heads
        self.attention_head_dim = config.hidden_size // config.num_heads
        self.patch_size = config.patch_size
        self.attention_backend = attention_backend

        self.pos_embed = MageFlowEmbedRope(theta=10000, axes_dim=config.axes_dim, scale_rope=True)
        self.img_in = nn.Linear(config.in_channels, config.hidden_size)
        self.txt_norm = RMSNorm(config.context_in_dim, eps=1e-6)
        self.txt_in = nn.Linear(config.context_in_dim, config.hidden_size)
        self.time_text_embed = MageFlowTimestepProjEmbeddings(embedding_dim=config.hidden_size)
        self.transformer_blocks = nn.ModuleList(
            [
                MageFlowTransformerBlock(
                    dim=config.hidden_size,
                    num_attention_heads=config.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    attention_backend=attention_backend,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(
            config.hidden_size,
            config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.proj_out = nn.Linear(config.hidden_size, config.patch_size * config.patch_size * config.out_channels)
        self.blocks_to_swap = 0
        self.offloader = None

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.checkpoint = enabled

    def enable_gradient_checkpointing(self, cpu_offload: bool = False) -> None:
        if cpu_offload:
            raise ValueError("Mage-Flow activation CPU offload is not implemented; use block swap instead")
        self.set_gradient_checkpointing(True)

    def disable_gradient_checkpointing(self) -> None:
        self.set_gradient_checkpointing(False)

    def enable_block_swap(self, num_blocks: int, config: BlockSwapConfig) -> None:
        maximum = max(0, len(self.transformer_blocks) - 2)
        if not isinstance(num_blocks, int) or not 0 <= num_blocks <= maximum:
            raise ValueError(f"Mage-Flow blocks_to_swap must be from 0 through {maximum}, got {num_blocks}")
        self.blocks_to_swap = num_blocks
        self.offloader = (
            create_offloader(
                "mage-flow",
                self.transformer_blocks,
                len(self.transformer_blocks),
                num_blocks,
                config,
            )
            if num_blocks
            else None
        )

    def move_to_device_except_swap_blocks(self, device: torch.device) -> None:
        if self.blocks_to_swap:
            blocks = self.transformer_blocks
            self.transformer_blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.transformer_blocks = blocks

    def prepare_block_swap_before_forward(self) -> None:
        if self.offloader is not None:
            self.offloader.prepare_block_devices_before_forward(self.transformer_blocks)

    def switch_block_swap_for_inference(self) -> None:
        if self.offloader is not None:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self) -> None:
        if self.offloader is not None:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def set_attention_backend(self, backend: str) -> None:
        normalized = backend.lower().strip()
        if normalized not in {"sdpa", "torch", "torch_sdpa", "flash2", "fa2", "flash_attention_2", "flash_attn_2"}:
            raise ValueError(f"unknown Mage-Flow attention backend {backend!r}")
        self.attention_backend = normalized
        for block in self.transformer_blocks:
            block.attn.backend = normalized

    def forward(self, packed: PackedMageFlowInputs) -> torch.Tensor:
        packed.validate(
            image_dim=self.config.in_channels,
            text_dim=self.config.context_in_dim,
            text_max_length=self.config.text_max_length,
        )
        image_rotary_emb = self.pos_embed(packed.image_shapes, device=packed.image_tokens.device)
        image = self.img_in(packed.image_tokens)
        text = self.txt_in(self.txt_norm(packed.text_tokens))
        timestep_embedding = self.time_text_embed(packed.timesteps.to(image.dtype), image)

        for block_index, block in enumerate(self.transformer_blocks):
            if self.offloader is not None:
                self.offloader.wait_for_block(block_index)
            if self.training and self.checkpoint:
                text, image = torch.utils.checkpoint.checkpoint(
                    block,
                    image,
                    text,
                    timestep_embedding,
                    image_rotary_emb,
                    packed.text_cu_seqlens,
                    packed.image_cu_seqlens,
                    use_reentrant=False,
                )
            else:
                text, image = block(
                    hidden_states=image,
                    encoder_hidden_states=text,
                    temb=timestep_embedding,
                    image_rotary_emb=image_rotary_emb,
                    txt_cu_lens=packed.text_cu_seqlens,
                    img_cu_lens=packed.image_cu_seqlens,
                )
            if self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.transformer_blocks, block_index)

        image = self.norm_out(image, timestep_embedding, cu_seqlens=packed.image_cu_seqlens)
        return self.proj_out(image)


def load_mage_flow_transformer(
    path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = torch.bfloat16,
    attention_backend: str = "sdpa",
    fp8_scaled: bool = False,
    _config: MageFlowConfig | None = None,
) -> MageFlow:
    config = MageFlowConfig.released() if _config is None else _config
    inspection = inspect_component(path, "dit", config=config)
    state_dict = normalize_dit_state_dict(load_file(str(inspection.path), device="cpu"))
    with torch.device("meta"):
        model = MageFlow(config, attention_backend=attention_backend)
    if fp8_scaled:
        compute_dtype = torch.bfloat16 if dtype is None else dtype
        state_dict = {key: value.to(compute_dtype) if value.is_floating_point() else value for key, value in state_dict.items()}
        state_dict = optimize_state_dict_with_fp8(
            state_dict,
            torch.device(device),
            FP8_OPTIMIZATION_TARGET_KEYS,
            FP8_OPTIMIZATION_EXCLUDE_KEYS,
            move_to_device=False,
        )
        apply_fp8_monkey_patch(model, state_dict, use_scaled_mm=False)
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except RuntimeError as exc:
        raise ComponentValidationError(f"DiT strict state-dict load failed after header validation: {exc}") from exc
    if fp8_scaled:
        model.to(device=device)
    elif dtype is not None:
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)
    model.eval()
    return model


__all__ = ["MageFlow", "load_mage_flow_transformer"]
