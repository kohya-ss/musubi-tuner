# This file includes code derived from:
# https://github.com/kandinskylab/kandinsky-5
# Copyright (c) 2025 Kandinsky Lab
# Licensed under the MIT License

import logging
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .nn import (
    TimeEmbeddings,
    TextEmbeddings,
    VisualEmbeddings,
    RoPE1D,
    RoPE3D,
    Modulation,
    MultiheadSelfAttentionEnc,
    MultiheadSelfAttentionDec,
    MultiheadCrossAttention,
    FeedForward,
    OutLayer,
    apply_scale_shift_norm,
    apply_gate_sum,
    _maybe_compile,
)
from .utils import fractal_flatten, fractal_unflatten
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader

logger = logging.getLogger(__name__)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, attention_engine="auto"):
        super().__init__()
        self.text_modulation = Modulation(time_dim, model_dim, 6)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionEnc(model_dim, head_dim, attention_engine)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, x, time_embed, rope, attention_mask=None):
        self_attn_params, ff_params = torch.chunk(self.text_modulation(time_embed), 2, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.self_attention_norm, x, scale, shift)
        out = self.self_attention(out, rope, attention_mask)
        x = apply_gate_sum(x, out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        out = apply_scale_shift_norm(self.feed_forward_norm, x, scale, shift)
        out = self.feed_forward(out)
        x = apply_gate_sum(x, out, gate)
        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(self, model_dim, time_dim, ff_dim, head_dim, attention_engine="auto"):
        super().__init__()
        self.visual_modulation = Modulation(time_dim, model_dim, 9)

        self.self_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.self_attention = MultiheadSelfAttentionDec(model_dim, head_dim, attention_engine)

        self.cross_attention_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.cross_attention = MultiheadCrossAttention(model_dim, head_dim, attention_engine)

        self.feed_forward_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.feed_forward = FeedForward(model_dim, ff_dim)

    def forward(self, visual_embed, text_embed, time_embed, rope, sparse_params, attention_mask=None):
        self_attn_params, cross_attn_params, ff_params = torch.chunk(self.visual_modulation(time_embed), 3, dim=-1)
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.self_attention_norm, visual_embed, scale, shift)
        visual_out = self.self_attention(visual_out, rope, sparse_params)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.cross_attention_norm, visual_embed, scale, shift)
        visual_out = self.cross_attention(visual_out, text_embed, attention_mask)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)

        shift, scale, gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = apply_scale_shift_norm(self.feed_forward_norm, visual_embed, scale, shift)
        visual_out = self.feed_forward(visual_out)
        visual_embed = apply_gate_sum(visual_embed, visual_out, gate)
        return visual_embed


class DiffusionTransformer3D(nn.Module):
    def __init__(
        self,
        in_visual_dim=4,
        in_text_dim=3584,
        in_text_dim2=768,
        time_dim=512,
        out_visual_dim=4,
        patch_size=(1, 2, 2),
        model_dim=2048,
        ff_dim=5120,
        num_text_blocks=2,
        num_visual_blocks=32,
        axes_dims=(16, 24, 24),
        visual_cond=False,
        attention_engine="auto",
        instruct_type=None,
    ):
        super().__init__()
        self.instruct_type = instruct_type
        head_dim = sum(axes_dims)
        self.in_visual_dim = in_visual_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.visual_cond = visual_cond

        visual_embed_dim = 2 * in_visual_dim + 1 if visual_cond or instruct_type == "channel" else in_visual_dim
        self.time_embeddings = TimeEmbeddings(model_dim, time_dim)
        self.text_embeddings = TextEmbeddings(in_text_dim, model_dim)
        self.pooled_text_embeddings = TextEmbeddings(in_text_dim2, time_dim)
        self.visual_embeddings = VisualEmbeddings(visual_embed_dim, model_dim, patch_size)

        self.text_rope_embeddings = RoPE1D(head_dim)
        self.text_transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine) for _ in range(num_text_blocks)]
        )

        self.visual_rope_embeddings = RoPE3D(axes_dims)
        self.visual_transformer_blocks = nn.ModuleList(
            [TransformerDecoderBlock(model_dim, time_dim, ff_dim, head_dim, attention_engine) for _ in range(num_visual_blocks)]
        )

        self.out_layer = OutLayer(model_dim, time_dim, out_visual_dim, patch_size)

        # block swap state
        self.blocks_to_swap = None
        self.offloader_visual: ModelOffloader | None = None
        self.offloader_text: ModelOffloader | None = None
        self.num_text_blocks = len(self.text_transformer_blocks)
        self.num_visual_blocks = len(self.visual_transformer_blocks)
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    @_maybe_compile()
    def before_text_transformer_blocks(self, text_embed, time, pooled_text_embed, x, text_rope_pos):
        text_embed = self.text_embeddings(text_embed)
        time_embed = self.time_embeddings(time)
        pooled_time_embed = self.pooled_text_embeddings(pooled_text_embed)
        if pooled_time_embed.dim() > 1 and pooled_time_embed.shape[0] > 1:
            # Reduce any per-frame pooled embeddings to a single vector so modulation can broadcast to all tokens.
            pooled_time_embed = pooled_time_embed.mean(dim=0, keepdim=True)
        time_embed = time_embed + pooled_time_embed

        visual_embed = self.visual_embeddings(x)
        text_rope = self.text_rope_embeddings(text_rope_pos)
        return text_embed, time_embed, text_rope, visual_embed

    @_maybe_compile()
    def before_visual_transformer_blocks(self, visual_embed, visual_rope_pos, scale_factor, sparse_params):
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(visual_shape, visual_rope_pos, scale_factor)
        to_fractal = sparse_params["to_fractal"] if sparse_params is not None else False
        visual_embed, visual_rope = fractal_flatten(visual_embed, visual_rope, visual_shape, block_mask=to_fractal)
        return visual_embed, visual_shape, to_fractal, visual_rope

    def after_blocks(self, visual_embed, visual_shape, to_fractal, text_embed, time_embed):
        visual_embed = fractal_unflatten(visual_embed, visual_shape, block_mask=to_fractal)
        x = self.out_layer(visual_embed, text_embed, time_embed)
        return x

    def forward(
        self,
        x,
        text_embed,
        pooled_text_embed,
        time,
        visual_rope_pos,
        text_rope_pos,
        scale_factor=(1.0, 1.0, 1.0),
        sparse_params=None,
        attention_mask=None,
    ):
        text_embed, time_embed, text_rope, visual_embed = self.before_text_transformer_blocks(
            text_embed, time, pooled_text_embed, x, text_rope_pos
        )

        for block_idx, text_transformer_block in enumerate(self.text_transformer_blocks):
            if self.blocks_to_swap and self.offloader_text:
                self.offloader_text.wait_for_block(block_idx)
            if self.training and self.gradient_checkpointing:
                text_embed = checkpoint(
                    text_transformer_block,
                    text_embed,
                    time_embed,
                    text_rope,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                text_embed = text_transformer_block(text_embed, time_embed, text_rope, attention_mask)
            if self.blocks_to_swap and self.offloader_text:
                self.offloader_text.submit_move_blocks_forward(self.text_transformer_blocks, block_idx)

        visual_embed, visual_shape, to_fractal, visual_rope = self.before_visual_transformer_blocks(
            visual_embed, visual_rope_pos, scale_factor, sparse_params
        )

        for block_idx, visual_transformer_block in enumerate(self.visual_transformer_blocks):
            if self.blocks_to_swap and self.offloader_visual:
                self.offloader_visual.wait_for_block(block_idx)
            if self.training and self.gradient_checkpointing:
                visual_embed = checkpoint(
                    visual_transformer_block,
                    visual_embed,
                    text_embed,
                    time_embed,
                    visual_rope,
                    sparse_params,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                visual_embed = visual_transformer_block(
                    visual_embed, text_embed, time_embed, visual_rope, sparse_params, attention_mask
                )
            if self.blocks_to_swap and self.offloader_visual:
                self.offloader_visual.submit_move_blocks_forward(self.visual_transformer_blocks, block_idx)

        x = self.after_blocks(visual_embed, visual_shape, to_fractal, text_embed, time_embed)
        return x

    # Offloading
    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks
        if num_blocks <= 0:
            return

        text_to_swap = max(0, min(self.num_text_blocks // 2, num_blocks // 4))
        visual_to_swap = max(1, num_blocks - text_to_swap)
        visual_to_swap = min(visual_to_swap, self.num_visual_blocks - 2)

        if text_to_swap > 0:
            self.offloader_text = ModelOffloader(
                "text",
                self.text_transformer_blocks,
                self.num_text_blocks,
                text_to_swap,
                supports_backward,
                device,
                use_pinned_memory,
            )
        self.offloader_visual = ModelOffloader(
            "visual",
            self.visual_transformer_blocks,
            self.num_visual_blocks,
            visual_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        logger.info(f"Kandinsky5: block swap enabled. text={text_to_swap}, visual={visual_to_swap}")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            if self.offloader_text:
                self.offloader_text.set_forward_only(True)
            if self.offloader_visual:
                self.offloader_visual.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            if self.offloader_text:
                self.offloader_text.set_forward_only(False)
            if self.offloader_visual:
                self.offloader_visual.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_text = self.text_transformer_blocks
            save_visual = self.visual_transformer_blocks
            self.text_transformer_blocks = None
            self.visual_transformer_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.text_transformer_blocks = save_text
            self.visual_transformer_blocks = save_visual

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        if self.offloader_text:
            self.offloader_text.prepare_block_devices_before_forward(self.text_transformer_blocks)
        if self.offloader_visual:
            self.offloader_visual.prepare_block_devices_before_forward(self.visual_transformer_blocks)

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False


def get_dit(conf):
    dit = DiffusionTransformer3D(**conf)
    return dit
