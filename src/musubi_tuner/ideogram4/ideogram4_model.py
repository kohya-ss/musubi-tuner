"""Ideogram4 transformer backbone.

The transformer consumes Qwen3-VL embeddings and flow-matching noise tokens to
produce velocity predictions on image latents.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from musubi_tuner.modules.attention import AttentionParams, attention
from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, LoRAStreamOffloader, ModelOffloader, create_offloader
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)


def _linear_compute_dtype(linear: nn.Module) -> torch.dtype:
    """Return the compute dtype of a (possibly FP8) Linear used as a dtype reference.

    A plain ``nn.Linear`` exposes it as ``weight.dtype``. But when the weight is FP8 — either the
    legacy ``Fp8Linear`` (which carries a ``compute_dtype`` attribute) or a monkey-patched FP8
    ``nn.Linear`` (whose ``scale_weight`` buffer holds the compute dtype) — ``weight.dtype`` would be
    float8, so we must read the compute dtype from those instead.
    """
    compute_dtype = getattr(linear, "compute_dtype", None)
    if compute_dtype is not None:
        return compute_dtype
    scale_weight = getattr(linear, "scale_weight", None)
    if scale_weight is not None:
        return scale_weight.dtype
    return linear.weight.dtype


@dataclass
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512

    # Latent dimension after patchification: ae_channels (32) * patch_size**2 (4) = 128.
    in_channels: int = 128

    # Hidden size of Qwen3-VL-8B-Instruct multiplied by the number of layers we extract
    # Qwen3-VL hidden size = 4096
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)

    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)

    norm_eps: float = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, L, num_heads, head_dim); cos/sin: (B, L, head_dim).
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L, 3) of int.
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, seq_len, _ = position_ids.shape

        # The MRoPE frequencies must be computed in real fp32. Image position ids start at
        # IMAGE_POSITION_OFFSET (2**16), where bf16 has a spacing of 512, so under an active
        # autocast context the `inv_freq @ pos` matmul would be downcast to bf16 and every image
        # position would collapse to the same value -- wiping out the spatial RoPE phase and
        # producing a flat checkerboard. The explicit `.to(torch.float32)` below does NOT protect
        # against this because autocast overrides matmul's compute dtype regardless of input dtype,
        # so we force autocast off for the whole computation. This is a no-op when no outer autocast
        # is active (the current training/inference default), so it never changes existing results.
        device_type = position_ids.device.type
        autocast_disabled = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in ("cuda", "cpu", "xpu", "hpu")
            else nullcontext()
        )
        with autocast_disabled:
            # (3, B, inv_freq_size, L)
            pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)  # type: ignore[arg-type]
            inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)  # type: ignore[index]
            freqs = inv_freq @ pos.unsqueeze(2)
            freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

            # interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
            freqs_t = freqs[0].clone()
            for axis, offset in ((1, 1), (2, 2)):
                length = self.mrope_section[axis] * 3
                idx = torch.arange(offset, length, 3, device=freqs_t.device)
                freqs_t[..., idx] = freqs[axis][..., idx]

            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_params: AttentionParams,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, L, num_heads, head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # The shared attention() consumes (B, L, num_heads, head_dim) and returns (B, L, hidden_size).
        # attn_params carries the text padding mask (image tokens are always valid and come first), so
        # this is mathematically equivalent to the previous block-diagonal SDPA mask whose only role was
        # excluding the leading padding -- but it also unlocks the flash/sage/xformers + split_attn paths.
        out = attention(q, k, v, attn_params)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attn_params: AttentionParams,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            attn_params=attn_params,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)  # type: ignore[assignment]
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape (..., 1) or (...,) holding a scalar per token.
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(_linear_compute_dtype(self.mlp_in))
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
    """Ideogram 4 flow-matching transformer."""

    def __init__(self, config: Ideogram4Config, attn_mode: str = "torch", split_attn: bool = False) -> None:
        super().__init__()
        self.config = config
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        # create_offloader returns a LoRAStreamOffloader (not a ModelOffloader subclass) when the
        # BlockSwapConfig requests h2d_only streaming, so the annotation covers both implementations.
        self.offloader: ModelOffloader | LoRAStreamOffloader | None = None

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=config.emb_dim,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    norm_eps=config.norm_eps,
                    adanln_dim=config.adanln_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        print(f"Ideogram4Transformer: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        print("Ideogram4Transformer: Gradient checkpointing disabled.")

    def enable_block_swap(self, blocks_to_swap: int, config: BlockSwapConfig):
        self.blocks_to_swap = blocks_to_swap
        num_blocks = len(self.layers)
        assert self.blocks_to_swap <= num_blocks - 1, (
            f"Cannot swap more than {num_blocks - 1} blocks. Requested {self.blocks_to_swap} blocks to swap."
        )
        self.offloader = create_offloader(
            "ideogram4-block",
            self.layers,
            num_blocks,
            self.blocks_to_swap,
            config,
        )
        print(
            f"Ideogram4Transformer: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {num_blocks} blocks. Supports backward: {config.supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            assert self.offloader is not None
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("Ideogram4Transformer: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            assert self.offloader is not None
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("Ideogram4Transformer: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.layers
            self.layers = None
        self.to(device)
        if self.blocks_to_swap:
            self.layers = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        assert self.offloader is not None
        self.offloader.prepare_block_devices_before_forward(self.layers)

    def _gradient_checkpointing_func(self, block, *args):
        if self.activation_cpu_offloading:
            block = create_cpu_offloading_wrapper(block, self.input_proj.weight.device)
        return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity prediction.

        The sequence layout is ``[image][text][right-padding]`` (image tokens first, valid text next,
        padding last), matching the Qwen-Image / Z-Image convention so the shared attention() can be used.

        Args:
          llm_features: (B, L, llm_features_dim) Qwen3-VL conditioning features.
          x: (B, L, in_channels) noise tokens.
          t: (B,) or (B, L) flow-matching time in [0, 1].
          position_ids: (B, L, 3) (t, h, w) positions for MRoPE.
          attention_mask: (B, max_text_tokens) text-token validity mask (1 = valid, 0 = padding).
            Image tokens are always valid and occupy the first ``L - max_text_tokens`` positions.
          indicator: (B, L) per-token role: LLM_TOKEN_INDICATOR or OUTPUT_IMAGE_INDICATOR.

        Returns:
          (B, L, in_channels) velocity prediction in float32. Only the positions
          with ``indicator == OUTPUT_IMAGE_INDICATOR`` are meaningful.
        """
        batch_size, seq_len, in_channels = x.shape
        assert in_channels == self.config.in_channels

        # Image tokens lead the sequence; the remaining positions are the (right-padded) text tokens.
        img_len = seq_len - attention_mask.shape[1] if attention_mask is not None else seq_len
        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, img_len, attention_mask)

        param_dtype = _linear_compute_dtype(self.input_proj)
        x = x.to(param_dtype)
        # NOTE: t is intentionally NOT cast here. It is consumed only by self.t_embedding
        # (Ideogram4EmbedScalar), which re-casts to float32 internally. Casting t to the bf16
        # param dtype first would quantize the timestep to bf16 resolution for no benefit, losing
        # precision in the --mixed_precision=no path where t arrives as clean float32.
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        # Keep shape (B, 1, ...) when t is per-sample so downstream adaln_modulation
        # projections don't pay for L identical copies.
        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        for index, layer in enumerate(self.layers):
            if self.blocks_to_swap:
                assert self.offloader is not None
                self.offloader.wait_for_block(index)
            if self.gradient_checkpointing and self.training:
                h = self._gradient_checkpointing_func(layer, h, attn_params, cos, sin, adaln_input)
            else:
                h = layer(h, attn_params=attn_params, cos=cos, sin=sin, adaln_input=adaln_input)
            if self.blocks_to_swap:
                assert self.offloader is not None
                self.offloader.submit_move_blocks_forward(self.layers, index)

        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
