"""Packed Mage-Flow transformer adapted from Microsoft Mage commit ea7109b.

Copyright (c) 2026 Microsoft. Licensed under the MIT License.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.models.normalization import RMSNorm
from safetensors.torch import load_file

from .layers import AdaLayerNormContinuous, MageFlowEmbedRope, MageFlowTimestepProjEmbeddings, MageFlowTransformerBlock
from .utils import ComponentValidationError, MageFlowConfig, PackedMageFlowInputs, inspect_component, normalize_dit_state_dict


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

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.checkpoint = enabled

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

        for block in self.transformer_blocks:
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
    if fp8_scaled:
        raise NotImplementedError("scaled FP8 conversion is applied by the trainer after strict Mage-Flow loading")
    config = MageFlowConfig.released() if _config is None else _config
    inspection = inspect_component(path, "dit", config=config)
    state_dict = normalize_dit_state_dict(load_file(str(inspection.path), device="cpu"))
    model = MageFlow(config, attention_backend=attention_backend)
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except RuntimeError as exc:
        raise ComponentValidationError(f"DiT strict state-dict load failed after header validation: {exc}") from exc
    if dtype is not None:
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)
    model.eval()
    return model


__all__ = ["MageFlow", "load_mage_flow_transformer"]
