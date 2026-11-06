# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
# Adapted from Diffusers transformer_qwenimage21.py and ComfyUI qwen_image21/model.py.
"""Qwen-Image 2.1 transformer.

Text embeddings exclude vision tokens. image_slots gives the insertion position
of each reference image in the remaining text. Target image tokens are appended last.
"""

import json
import logging
from pathlib import Path

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F

from musubi_tuner.modules import attention as attention_backends
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.qwen_image.qwen_image_model import QwenImageTransformer2DModel, TimestepEmbedding
from musubi_tuner.qwen_image_21.qwen_image_21_utils import store_linears_in_fp8
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.utils.safetensors_utils import WeightTransformHooks

logger = logging.getLogger(__name__)


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * (1 + self.weight.float())).to(x.dtype)


class TextProjection(nn.Module):
    def __init__(self, context_dim, dim, eps):
        super().__init__()
        self.text_norm = ZeroCenteredRMSNorm(context_dim, eps)
        self.in_layer = nn.Linear(context_dim, dim, bias=False)
        self.out_layer = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.gelu(self.in_layer(self.text_norm(x)), approximate="tanh"))


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.timestep_embedder = TimestepEmbedding(256, dim, sample_proj_bias=False)

    def forward(self, timestep, dtype):
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=timestep.device))
            * torch.arange(128, device=timestep.device, dtype=torch.float32)
            / 128
        )
        phase = 1000 * timestep.float()[:, None] * freqs[None]
        return self.timestep_embedder(torch.cat([phase.cos(), phase.sin()], -1).to(dtype))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim, bias=False)
        self.gate_layer = nn.Linear(dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.out(F.silu(self.gate_layer(x)) * self.proj(x))


def rotary_embedding(ids, axes_dims):
    phases = []
    for axis, dim in enumerate(axes_dims):
        inv = 10000 ** (-torch.arange(0, dim, 2, device=ids.device, dtype=torch.float32) / dim)
        phases.append(ids[..., axis, None].float() * inv)
    phase = torch.cat(phases, -1)
    return torch.polar(torch.ones_like(phase), phase)


def apply_rope(x, freqs):
    complex_x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    return torch.view_as_real(complex_x * freqs[:, :, None]).flatten(-2).to(x.dtype)


def segmented_attention(q, k, v, segments, attn_mode="torch"):
    """Compute attention separately for each text or image segment.

    Text tokens attend to preceding tokens. Image tokens attend to preceding
    segments and all tokens in the same image. Padding is excluded per sample.
    """
    rows = []
    for b, runs in enumerate(segments):
        parts = []
        for start, end, is_text in runs:
            mask = None
            causal = is_text and start == 0
            if is_text and start:
                mask = torch.arange(end, device=q.device)[None, :] <= torch.arange(start, end, device=q.device)[:, None]
            # Text segments need an offset causal mask. Image segments use all
            # keys up to the end of the image, so no mask is needed.
            backend = attn_mode
            if is_text or (backend == "sageattn" and torch.is_grad_enabled()):
                backend = "torch"  # SageAttention has no autograd kernel.
            qs, ks, vs = q[b : b + 1, start:end], k[b : b + 1, :end], v[b : b + 1, :end]
            if backend in {"flash", "flash3"}:
                func = (
                    attention_backends.flash_attn_func
                    if backend == "flash"
                    else (
                        None
                        if attention_backends.flash_attn_interface is None
                        else attention_backends.flash_attn_interface.flash_attn_func
                    )
                )
                if func is None:
                    raise ImportError(f"Install the {backend} attention backend, or select --sdpa")
                out = attention_backends._flash3_output(func(qs, ks, vs, causal=False)).flatten(2)
            elif backend == "xformers":
                if attention_backends.xops is None:
                    raise ImportError("Install xformers, or select --sdpa")
                out = attention_backends.xops.memory_efficient_attention(qs, ks, vs).flatten(2)
            elif backend == "sageattn":
                if attention_backends.sageattn is None:
                    raise ImportError("Install sageattention, or select --sdpa")
                out = (
                    attention_backends.sageattn(qs.transpose(1, 2), ks.transpose(1, 2), vs.transpose(1, 2))
                    .transpose(1, 2)
                    .flatten(2)
                )
            elif backend == "torch":
                out = (
                    F.scaled_dot_product_attention(
                        qs.transpose(1, 2),
                        ks.transpose(1, 2),
                        vs.transpose(1, 2),
                        attn_mask=mask,
                        is_causal=causal,
                    )
                    .transpose(1, 2)
                    .flatten(2)
                )
            else:
                raise ValueError(f"Unknown attention backend: {backend}")
            parts.append(out)
        row = torch.cat(parts, 1)
        rows.append(F.pad(row, (0, 0, 0, q.shape[1] - row.shape[1])))
    return torch.cat(rows, 0)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, eps):
        super().__init__()
        self.heads = heads
        self.attn_mode = "torch"
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False)])
        self.norm_q = nn.RMSNorm(head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(head_dim, eps=eps)

    def forward(self, x, rope, segments):
        q = self.norm_q(self.to_q(x).unflatten(-1, (self.heads, -1)))
        k = self.norm_k(self.to_k(x).unflatten(-1, (self.heads, -1)))
        v = self.to_v(x).unflatten(-1, (self.heads, -1))
        # RMSNorm may return FP32 under autocast; attention kernels require matching Q/K/V dtypes.
        q, k = q.to(v.dtype), k.to(v.dtype)
        return self.to_out[0](segmented_attention(apply_rope(q, rope), apply_rope(k, rope), v, segments, self.attn_mode))


def select_rows(params, target_mask):
    return torch.where(target_mask[..., None], params[:-1, None], params[-1:, None])


class QwenImage21TransformerBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_ratio=3, eps=1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(dim, heads, head_dim, eps)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x, modulation, rope, segments, target_mask):
        scale1, gate1, scale2, gate2 = [select_rows(p, target_mask) for p in modulation.chunk(4, -1)]
        x = x + gate1.tanh() * self.attn(self.img_norm1(x) * (1 + scale1), rope, segments)
        x = x + gate2.tanh() * self.img_mlp(self.img_norm2(x) * (1 + scale2))
        return x.clamp(-65504, 65504) if x.dtype == torch.float16 else x


class FinalNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(self, x, temb):
        return self.norm(x) * (1 + self.linear(F.silu(temb))[:, None])


class QwenImage21Transformer2DModel(QwenImageTransformer2DModel):
    """Qwen-Image 2.1 single-stream transformer."""

    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        num_layers=32,
        attention_head_dim=128,
        num_attention_heads=32,
        context_in_dim=4096,
        mlp_ratio=3,
        axes_dims_rope=(16, 56, 56),
        eps=1e-6,
        patch_size=1,
        causal_condition=True,
    ):
        nn.Module.__init__(self)
        self.config = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            context_in_dim=context_in_dim,
            mlp_ratio=mlp_ratio,
            axes_dims_rope=list(axes_dims_rope),
            eps=eps,
            patch_size=patch_size,
            causal_condition=causal_condition,
        )
        if patch_size != 1 or not causal_condition:
            raise ValueError("Qwen-Image 2.1 training requires patch_size=1 and causal_condition=True")
        if sum(axes_dims_rope) != attention_head_dim or any(d <= 0 or d % 2 for d in axes_dims_rope):
            raise ValueError("RoPE axis dimensions must be positive, even, and sum to attention_head_dim")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.inner_dim = attention_head_dim * num_attention_heads
        self.axes_dims_rope = tuple(axes_dims_rope)
        self.context_in_dim = context_in_dim
        self.img_in = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.txt_in = TextProjection(context_in_dim, self.inner_dim, eps)
        self.time_text_embed = TimeEmbedding(self.inner_dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.inner_dim, 4 * self.inner_dim, bias=False))
        self.transformer_blocks = nn.ModuleList(
            [
                QwenImage21TransformerBlock(self.inner_dim, num_attention_heads, attention_head_dim, mlp_ratio, eps)
                for _ in range(num_layers)
            ]
        )
        self.norm_out = FinalNorm(self.inner_dim, eps)
        self.proj_out = nn.Linear(self.inner_dim, out_channels, bias=False)
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None
        self.img_in_txt_in_offloading = False

    def wait_for_pending_block_moves(self):
        """Wait for pending weight swaps without starting H2D-only loads."""
        if self.blocks_to_swap:
            # H2D-only streaming handles pending transfers during its own reset.
            for block_idx in list(getattr(self.offloader, "futures", {})):
                self.offloader.wait_for_block(block_idx)

    def prepare_block_swap_before_forward(self):
        # Finish the previous forward's transfers before resetting block devices.
        self.wait_for_pending_block_moves()
        super().prepare_block_swap_before_forward()

    def _gradient_checkpointing_func(self, block, *args):
        if self.activation_cpu_offloading:
            block = create_cpu_offloading_wrapper(block, self.proj_out.weight.device)
        return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)

    def build_sequence(self, target, context, img_shapes, txt_seq_lens=None, reference_latents=None, image_slots=None):
        batch, target_len, _ = target.shape
        if context.ndim != 3 or context.shape[0] != batch or context.shape[-1] != self.context_in_dim:
            raise ValueError(f"Expected context [B, L, {self.context_in_dim}]; rebuild Qwen-Image 2.1 text caches")
        if len(img_shapes) == 1:
            img_shapes = img_shapes * batch
        if len(img_shapes) != batch:
            raise ValueError("img_shapes must contain one layout or one per sample")
        refs = list(reference_latents or [])
        lengths = txt_seq_lens if txt_seq_lens is not None else [context.shape[1]] * batch
        slots = image_slots if image_slots is not None else [[] for _ in range(batch)]
        if len(lengths) != batch or len(slots) != batch:
            raise ValueError("Text lengths and image slots must match batch size")
        input_device = target.device
        if self.img_in_txt_in_offloading:
            self.img_in.to("cpu")
            self.txt_in.to("cpu")
        projection_device = self.img_in.weight.device
        projection_dtype = self.img_in.weight.dtype
        text = self.txt_in(context.to(device=projection_device, dtype=projection_dtype)).to(input_device)
        target = self.img_in(target.to(device=projection_device, dtype=projection_dtype)).to(input_device)
        refs = [self.img_in(ref.to(device=projection_device, dtype=projection_dtype)).to(input_device) for ref in refs]
        rows, all_ids, all_runs, prefixes = [], [], [], []
        for b in range(batch):
            shapes = img_shapes[b]  # references first, target last
            ntext = int(lengths[b])
            if ntext < 1 or ntext > context.shape[1]:
                raise ValueError("Each sample must have at least one valid text token")
            row_slots = [int(s) for s in slots[b]]
            if len(shapes) != len(refs) + 1 or len(row_slots) != len(refs):
                raise ValueError("Reference latents, shapes and cached image slots must have the same count")
            if row_slots != sorted(row_slots) or any(s < 0 or s > ntext for s in row_slots):
                raise ValueError("Image slots must be ordered positions within the unpadded text")
            parts, ids, runs = [], [], []
            cursor = position = length = 0
            th, tw = shapes[-1][1:]
            for index, ((frames, h, w), end, img) in enumerate(zip(shapes, row_slots + [ntext], refs + [target])):
                if frames != 1 or h <= 0 or w <= 0 or img.shape[1] != h * w:
                    raise ValueError("Each image must be a single frame with h*w latent tokens")
                if end > cursor:
                    count = end - cursor
                    parts.append(text[b, cursor:end])
                    ids.append(torch.arange(position, position + count, device=target.device).float()[:, None].expand(-1, 3))
                    runs.append((length, length + count, True))
                    position += count
                    length += count
                if index == len(refs):
                    prefixes.append(length)
                parts.append(img[b])
                yy = torch.arange(h, device=target.device).float() - (h - h // 2) + 0.5 * (h % 2 - th % 2)
                xx = torch.arange(w, device=target.device).float() - (w - w // 2) + 0.5 * (w % 2 - tw % 2)
                grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
                ids.append(torch.stack([torch.full_like(grid_y, position), grid_y, grid_x], -1).reshape(-1, 3))
                runs.append((length, length + h * w, False))
                position += max(h, w)
                length += h * w
                cursor = end
            rows.append(torch.cat(parts))
            all_ids.append(torch.cat(ids))
            all_runs.append(runs)
        max_len = max(row.shape[0] for row in rows)
        x = torch.stack([F.pad(row, (0, 0, 0, max_len - row.shape[0])) for row in rows])
        ids = torch.stack([F.pad(row, (0, 0, 0, max_len - row.shape[0])) for row in all_ids])
        positions = torch.arange(max_len, device=x.device)[None]
        prefix = torch.tensor(prefixes, device=x.device)[:, None]
        target_mask = (positions >= prefix) & (positions < prefix + target_len)
        return x, rotary_embedding(ids, self.axes_dims_rope), all_runs, target_mask, prefixes

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_shapes,
        txt_seq_lens=None,
        reference_latents=None,
        image_slots=None,
        encoder_hidden_states_mask=None,
    ):
        if encoder_hidden_states_mask is not None:
            mask = encoder_hidden_states_mask.bool()
            lengths = mask.sum(1).tolist()
            expected = torch.arange(mask.shape[1], device=mask.device)[None] < mask.sum(1)[:, None]
            if not torch.equal(mask, expected):
                raise ValueError("Text padding must be on the right")
            if txt_seq_lens is not None and list(txt_seq_lens) != lengths:
                raise ValueError("txt_seq_lens disagrees with encoder_hidden_states_mask")
            txt_seq_lens = lengths
        target_len = hidden_states.shape[1]
        x, rope, segments, target_mask, prefixes = self.build_sequence(
            hidden_states, encoder_hidden_states, img_shapes, txt_seq_lens, reference_latents, image_slots
        )
        t = timestep.reshape(-1).to(device=x.device, dtype=x.dtype)
        if t.numel() != x.shape[0]:
            raise ValueError("Expected one timestep per sample")
        temb = self.time_text_embed(torch.cat([t, t.new_zeros(1)]), x.dtype)
        modulation = self.modulation(temb)
        for index, block in enumerate(self.transformer_blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = self._gradient_checkpointing_func(block, x, modulation, rope, segments, target_mask)
            else:
                x = block(x, modulation, rope, segments, target_mask)
            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.transformer_blocks, index)
        # CPU offload returns the block output on CPU; output projections stay on the compute device.
        x = x.to(hidden_states.device)
        targets = torch.stack([x[b, p : p + target_len] for b, p in enumerate(prefixes)])
        return self.proj_out(self.norm_out(targets, temb[:-1]))


def canonical_weight_hook(key, tensor):
    """Remove checkpoint prefixes and split fused gate/up weights."""
    key = key.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
    if key.endswith(".img_mlp.gate_up.weight"):
        keys = [key.replace("gate_up", "gate_layer"), key.replace("gate_up", "proj")]
        if tensor is None:
            return keys, None
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            raise ValueError("Prequantized fused weights require their quantization metadata; use BF16/FP32 weights")
        return keys, list(tensor.chunk(2, 0))
    return [key], None if tensor is None else [tensor]


def load_model(
    dit_path,
    device="cpu",
    loading_device="cpu",
    dtype=torch.bfloat16,
    fp8_scaled=False,
    config_path=None,
    disable_numpy_memmap=False,
    attn_mode="torch",
    lora_weights_list=None,
    lora_multipliers=None,
    num_layers=None,
):
    """Load DiT weights, merge base adapters, and apply optional FP8 quantization."""
    logger.info("Loading DiT model from %s, device=%s", dit_path, loading_device)
    path = Path(dit_path)
    config_file = Path(config_path) if config_path else (path / "config.json" if path.is_dir() else path.parent / "config.json")
    config = json.loads(config_file.read_text(encoding="utf-8")) if config_file.is_file() else {}
    if config_path is None and path.is_file():
        with safe_open(str(path), framework="pt", device="cpu") as file:
            metadata = file.metadata() or {}
        if "qwen_image_21_config" in metadata:
            config = json.loads(metadata["qwen_image_21_config"])
    if config.get("_class_name", "QwenImage21Transformer2DModel") != "QwenImage21Transformer2DModel":
        raise ValueError("This is not a QwenImage21Transformer2DModel config")
    config = {k: v for k, v in config.items() if not k.startswith("_")}
    if num_layers is not None:
        config["num_layers"] = num_layers
    bare_fp8 = dtype == torch.float8_e4m3fn and not fp8_scaled
    compute_dtype = torch.bfloat16 if bare_fp8 or dtype is None else dtype
    with init_empty_weights():
        model = QwenImage21Transformer2DModel(**config).to(dtype=compute_dtype)
    for block in model.transformer_blocks:
        block.attn.attn_mode = attn_mode
    if path.is_dir():
        candidates = list(path.glob("diffusion_pytorch_model.safetensors")) or list(
            path.glob("diffusion_pytorch_model-00001-of-*.safetensors")
        )
        if not candidates:
            candidates = list(path.glob("*.safetensors"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one checkpoint or first shard in {path}; pass an explicit safetensors file")
        files = str(candidates[0])  # the shared loader expands the remaining shards
    else:
        files = str(path)
    sd = load_safetensors_with_lora_and_fp8(
        files,
        lora_weights_list,
        lora_multipliers,
        fp8_scaled,
        torch.device(device),
        move_to_device=torch.device(loading_device) == torch.device(device),
        dit_weight_dtype=None if fp8_scaled else compute_dtype,
        target_keys=["transformer_blocks"],
        exclude_keys=["norm"],
        disable_numpy_memmap=disable_numpy_memmap,
        weight_transform_hooks=WeightTransformHooks(split_hook=canonical_weight_hook),
    )
    sd = {k.removeprefix("model.diffusion_model.").removeprefix("diffusion_model."): v for k, v in sd.items()}
    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info("Loaded DiT model from %s, info=%s", dit_path, info)
    if bare_fp8:
        store_linears_in_fp8(model.transformer_blocks)
    return model.to(device=loading_device)
