"""Lens denoising transformer.

Adapted from microsoft/Lens, commit 5bf0f0c. The implementation is kept as a
plain torch module so Musubi does not need newer diffusers runtime classes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.qwen_image.qwen_image_model import AdaLayerNormContinuous, FeedForward, RMSNorm, TimestepEmbedding, Timesteps


DEFAULT_TRANSFORMER_CONFIG = dict(
    patch_size=2,
    in_channels=128,
    out_channels=32,
    num_layers=48,
    attention_head_dim=64,
    num_attention_heads=24,
    inner_dim=1536,
    enc_hidden_dim=2880,
    axes_dims_rope=(8, 28, 28),
    gate_mlp=True,
    rms_norm=True,
    multi_layer_encoder_feature=True,
    selected_layer_index=(5, 11, 17, 23),
)

FP8_OPTIMIZATION_TARGET_KEYS = ["transformer_blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "pos_embed", "time_text_embed", "img_mod", "txt_mod", "norm_out", "proj_out"]


def apply_rotary_emb_lens(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(1)
    x_out = torch.view_as_real(x_complex * freqs_cis).flatten(3)
    return x_out.type_as(x)


class GateMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LensTimestepProjEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timestep: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        proj = self.time_proj(timestep)
        return self.timestep_embedder(proj.to(dtype=hidden_states.dtype))


class LensEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat([self._rope_params(pos_index, d, theta) for d in axes_dim], dim=1)
        self.neg_freqs = torch.cat([self._rope_params(neg_index, d, theta) for d in axes_dim], dim=1)
        self.rope_cache: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _rope_params(index: torch.Tensor, dim: int, theta: int = 10000) -> torch.Tensor:
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).float().div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)

    def forward(
        self,
        video_fhw: Union[List[Tuple[int, int, int]], Tuple[int, int, int]],
        txt_seq_lens: Union[List[int], int],
        device: torch.device = torch.device("cuda"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]
        if not isinstance(txt_seq_lens, list):
            txt_seq_lens = [txt_seq_lens]
        assert len(video_fhw) == 1, "Lens MVP expects one image shape"

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"
            if rope_key not in self.rope_cache:
                self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx=0).to("cpu")
            video_freq = self.rope_cache[rope_key].to(device)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)
            vid_freqs.append(video_freq)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        return torch.cat(vid_freqs, dim=0), txt_freqs

    def _compute_video_freqs(self, frame: int, height: int, width: int, idx: int = 0) -> torch.Tensor:
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([d // 2 for d in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([d // 2 for d in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class LensJointAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        added_kv_proj_dim: int,
        dim_head: int = 64,
        heads: int = 8,
        out_dim: Optional[int] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.heads = self.inner_dim // dim_head
        self.dim_head = dim_head
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)
        self.norm_added_q = RMSNorm(dim_head, eps=eps)
        self.norm_added_k = RMSNorm(dim_head, eps=eps)

        self.img_qkv = nn.Linear(query_dim, 3 * self.inner_dim, bias=True)
        self.txt_qkv = nn.Linear(added_kv_proj_dim, 3 * self.inner_dim, bias=True)

        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.out_dim, bias=True), nn.Identity()])
        self.to_add_out = nn.Linear(self.inner_dim, query_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_img, _ = hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]

        img_qkv = self.img_qkv(hidden_states).view(bsz, seq_img, 3, self.heads, self.dim_head)
        txt_qkv = self.txt_qkv(encoder_hidden_states).view(bsz, seq_txt, 3, self.heads, self.dim_head)
        img_q, img_k, img_v = img_qkv.unbind(dim=2)
        txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)

        img_q = self.norm_q(img_q)
        img_k = self.norm_k(img_k)
        txt_q = self.norm_added_q(txt_q)
        txt_k = self.norm_added_k(txt_k)

        img_freqs, txt_freqs = image_rotary_emb
        if img_freqs.shape[0] < seq_img:
            raise ValueError(f"Image RoPE length {img_freqs.shape[0]} is shorter than image sequence length {seq_img}.")
        img_freqs = img_freqs[:seq_img]
        img_q = apply_rotary_emb_lens(img_q, img_freqs)
        img_k = apply_rotary_emb_lens(img_k, img_freqs)
        if seq_txt > 0:
            if txt_freqs.shape[0] < seq_txt:
                raise ValueError(f"Text RoPE length {txt_freqs.shape[0]} is shorter than text sequence length {seq_txt}.")
            txt_freqs = txt_freqs[:seq_txt]
            txt_q = apply_rotary_emb_lens(txt_q, txt_freqs)
            txt_k = apply_rotary_emb_lens(txt_k, txt_freqs)

        q = torch.cat([img_q, txt_q], dim=1).transpose(1, 2)
        k = torch.cat([img_k, txt_k], dim=1).transpose(1, 2)
        v = torch.cat([img_v, txt_v], dim=1).transpose(1, 2)

        if attention_mask is not None:
            expected_mask_shape = (bsz, 1, 1, seq_img + seq_txt)
            if attention_mask.shape != expected_mask_shape:
                raise ValueError(f"attention_mask must have shape {expected_mask_shape}, got {tuple(attention_mask.shape)}.")
            attention_mask = attention_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2).reshape(bsz, seq_img + seq_txt, -1)

        img_out = self.to_out[1](self.to_out[0](out[:, :seq_img, :]))
        txt_out = self.to_add_out(out[:, seq_img:, :])
        return img_out, txt_out


class LensTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        eps: float = 1e-6,
        rms_norm: bool = False,
        gate_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.attn = LensJointAttention(
            query_dim=dim,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            eps=eps,
        )

        norm_cls = (lambda d: RMSNorm(d, eps=eps)) if rms_norm else (lambda d: nn.LayerNorm(d, elementwise_affine=False, eps=eps))
        if gate_mlp:
            mlp_cls = lambda: GateMLP(dim, int(dim / 3 * 8))
        else:
            mlp_cls = lambda: FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.img_norm1 = norm_cls(dim)
        self.img_norm2 = norm_cls(dim)
        self.img_mlp = mlp_cls()

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))
        self.txt_norm1 = norm_cls(dim)
        self.txt_norm2 = norm_cls(dim)
        self.txt_mlp = mlp_cls()

    @staticmethod
    def _modulate(x: torch.Tensor, mod_params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        img_mod1, img_mod2 = self.img_mod(temb).chunk(2, dim=-1)
        txt_mod1, txt_mod2 = self.txt_mod(temb).chunk(2, dim=-1)

        img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), img_mod1)
        txt_modulated, txt_gate1 = self._modulate(self.txt_norm1(encoder_hidden_states), txt_mod1)

        img_attn, txt_attn = self.attn(
            hidden_states=img_modulated,
            encoder_hidden_states=txt_modulated,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + img_gate1 * img_attn
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn

        img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), img_mod2)
        hidden_states = hidden_states + img_gate2 * self.img_mlp(img_modulated2)

        txt_modulated2, txt_gate2 = self._modulate(self.txt_norm2(encoder_hidden_states), txt_mod2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * self.txt_mlp(txt_modulated2)

        return encoder_hidden_states, hidden_states


class LensTransformer2DModel(nn.Module):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["LensTransformerBlock"]
    _repeated_blocks = ["LensTransformerBlock"]

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 128,
        out_channels: Optional[int] = 32,
        num_layers: int = 48,
        attention_head_dim: int = 64,
        num_attention_heads: int = 24,
        inner_dim: int = 1536,
        enc_hidden_dim: int = 2880,
        axes_dims_rope: Tuple[int, int, int] = (8, 28, 28),
        gate_mlp: bool = True,
        rms_norm: bool = True,
        multi_layer_encoder_feature: bool = True,
        selected_layer_index: Tuple[int, ...] = (5, 11, 17, 23),
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            inner_dim=inner_dim,
            enc_hidden_dim=enc_hidden_dim,
            axes_dims_rope=axes_dims_rope,
            gate_mlp=gate_mlp,
            rms_norm=rms_norm,
            multi_layer_encoder_feature=multi_layer_encoder_feature,
            selected_layer_index=selected_layer_index,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.multi_layer_encoder_feature = multi_layer_encoder_feature
        self.selected_layer_index = list(selected_layer_index)
        self.gradient_checkpointing = False
        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = num_layers

        self.pos_embed = LensEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)
        self.time_text_embed = LensTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        if self.multi_layer_encoder_feature:
            self.txt_norm = nn.ModuleList([RMSNorm(enc_hidden_dim, eps=1e-5) for _ in self.selected_layer_index])
            self.txt_in = nn.Linear(enc_hidden_dim * len(self.selected_layer_index), self.inner_dim)
        else:
            self.txt_norm = RMSNorm(enc_hidden_dim, eps=1e-5)
            self.txt_in = nn.Linear(enc_hidden_dim, self.inner_dim)

        self.img_in = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                LensTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rms_norm=rms_norm,
                    gate_mlp=gate_mlp,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        if cpu_offload:
            raise ValueError("Lens does not support activation CPU offloading yet.")
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def enable_block_swap(
        self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False
    ):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.transformer_blocks)

        assert self.blocks_to_swap <= self.num_blocks - 1, (
            f"Cannot swap more than {self.num_blocks - 1} Lens blocks. Requested {self.blocks_to_swap} blocks."
        )

        self.offloader = ModelOffloader(
            "lens-block",
            self.transformer_blocks,
            self.num_blocks,
            self.blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        print(
            f"LensTransformer2DModel: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks}. Supports backward: {supports_backward}"
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("LensTransformer2DModel: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("LensTransformer2DModel: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.transformer_blocks
            self.transformer_blocks = nn.ModuleList()

        self.to(device)

        if self.blocks_to_swap:
            self.transformer_blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.transformer_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Union[torch.Tensor, List[torch.Tensor]],
        encoder_hidden_states_mask: torch.Tensor,
        timestep: torch.Tensor,
        img_shapes: List[Tuple[int, int, int]],
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        del attention_kwargs
        bsz, img_len, _ = hidden_states.shape
        if self.multi_layer_encoder_feature:
            if not isinstance(encoder_hidden_states, (list, tuple)):
                raise ValueError("multi_layer_encoder_feature=True expects a list of per-layer text tensors.")
            if len(encoder_hidden_states) != len(self.selected_layer_index):
                raise ValueError(f"Expected {len(self.selected_layer_index)} text feature layers, got {len(encoder_hidden_states)}.")
            text_seq_len = encoder_hidden_states[0].shape[1]
            for i, feat in enumerate(encoder_hidden_states):
                if feat.shape[0] != bsz:
                    raise ValueError(f"Text feature layer {i} batch size {feat.shape[0]} does not match hidden_states batch size {bsz}.")
                if feat.shape[1] != text_seq_len:
                    raise ValueError(f"Text feature layer {i} sequence length {feat.shape[1]} does not match layer 0 length {text_seq_len}.")
        else:
            if not isinstance(encoder_hidden_states, torch.Tensor):
                raise ValueError("multi_layer_encoder_feature=False expects a single text feature tensor.")
            if encoder_hidden_states.shape[0] != bsz:
                raise ValueError(f"Text feature batch size {encoder_hidden_states.shape[0]} does not match hidden_states batch size {bsz}.")
            text_seq_len = encoder_hidden_states.shape[1]
        if encoder_hidden_states_mask.shape != (bsz, text_seq_len):
            raise ValueError(
                f"encoder_hidden_states_mask must have shape {(bsz, text_seq_len)}, got {tuple(encoder_hidden_states_mask.shape)}."
            )
        attention_mask = self._build_joint_attention_mask(encoder_hidden_states_mask, img_len)

        hidden_states = self.img_in(hidden_states)
        timestep = timestep.to(hidden_states.dtype)

        if self.multi_layer_encoder_feature:
            normed = [self.txt_norm[i](encoder_hidden_states[i]) for i in range(len(self.selected_layer_index))]
            encoder_hidden_states = torch.cat(normed, dim=-1)
        else:
            encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        temb = self.time_text_embed(timestep, hidden_states)
        image_rotary_emb = self.pos_embed(img_shapes, [text_seq_len], device=hidden_states.device)

        input_device = hidden_states.device
        for index_block, block in enumerate(self.transformer_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index_block)

            if self.training and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = checkpoint(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                    use_reentrant=False,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                )

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.transformer_blocks, index_block)

        if input_device != hidden_states.device:
            hidden_states = hidden_states.to(input_device)

        hidden_states = self.norm_out(hidden_states, temb)
        return self.proj_out(hidden_states)

    @staticmethod
    def _build_joint_attention_mask(text_mask: torch.Tensor, img_len: int) -> torch.Tensor:
        if text_mask.dtype != torch.bool:
            text_mask = text_mask.bool()
        bsz = text_mask.shape[0]
        img_ones = torch.ones((bsz, img_len), dtype=torch.bool, device=text_mask.device)
        joint = torch.cat([img_ones, text_mask], dim=1)
        additive = torch.zeros_like(joint, dtype=torch.float32)
        additive.masked_fill_(~joint, float("-inf"))
        return additive[:, None, None, :]


def create_default_lens_transformer() -> LensTransformer2DModel:
    return LensTransformer2DModel(**DEFAULT_TRANSFORMER_CONFIG)
