# This file includes code derived from:
# https://github.com/meituan-longcat/LongCat-Video
# Copyright (c) 2025 Meituan
# Licensed under the MIT License

import logging
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn

import numpy as np
from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from safetensors.torch import load_file

from musubi_tuner.modules.custom_offloading_utils import ModelOffloader, weighs_to_device
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch, optimize_state_dict_with_fp8

from .lora_utils import create_lora_network
from .attention import Attention, MultiHeadCrossAttention
from .blocks import (
    TimestepEmbedder,
    CaptionEmbedder,
    PatchEmbed3D,
    FeedForwardSwiGLU,
    FinalLayer_FP32,
    LayerNorm_FP32,
    modulate_fp32,
)

from torch.cuda.amp import autocast
from torch.utils.hooks import RemovableHandle


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # scale and gate modulation
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True))
        for module in self.adaLN_modulation.modules():
            if isinstance(module, nn.Linear):
                module._block_swap_pin_weights = True

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))
        self._gradient_checkpointing = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        if activation_cpu_offloading:
            logger.warning("LongCatSingleStreamBlock ignores activation CPU offloading; running without offload.")

        self._gradient_checkpointing = True

    def forward(self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None, skip_crs_attn=False):
        """
        x: [B, N, C]
        y: [1, N_valid_tokens, C]
        t: [B, T, C_t]
        y_seqlen: [B]; type of a list
        latent_shape: latent shape of a single item
        """
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape  # S != T*H*W in case of CP split on H*W.

        # compute modulation params in fp32
        with autocast(dtype=torch.float32):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
            )  # [B, T, 1, C]

        # self attn with modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            kv_cache = (kv_cache[0].to(x.device), kv_cache[1].to(x.device))
            attn_outputs = self.attn.forward_with_kv_cache(
                x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache
            )
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv)

        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with autocast(dtype=torch.float32):
            x = x + (gate_msa * x_s.view(B, -1, N // T, C)).view(B, -1, C)  # [B, N, C]
        x = x.to(x_dtype)

        # cross attn
        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape)

        # ffn with modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.view(B, -1, N // T, C), shift_mlp, scale_mlp).view(B, -1, C)
        x_s = self.ffn(x_m)
        with autocast(dtype=torch.float32):
            x = x + (gate_mlp * x_s.view(B, -1, N // T, C)).view(B, -1, C)  # [B, N, C]
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        else:
            return x


class LongCatVideoTransformer3DModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        if activation_cpu_offloading:
            logger.warning("LongCatVideoTransformer3DModel ignores activation CPU offloading; running without offload.")

        self.gradient_checkpointing = True

        for block in self.blocks:
            enable_block_checkpoint = getattr(block, "enable_gradient_checkpointing", None)
            if callable(enable_block_checkpoint):
                enable_block_checkpoint(activation_cpu_offloading)

    def _gradient_checkpointing_func(self, block, *inputs):
        return torch.utils.checkpoint.checkpoint(block, *inputs, use_reentrant=False)

    # Restrict FP8 quantization to the large linear projections inside each transformer block.
    # Keeping LayerNorm/adaLN and the final out projection in higher precision prevents NaN overflow.
    FP8_OPTIMIZATION_TARGET_KEYS: List[str] = [
        ".attn.qkv",
        ".attn.proj",
        ".cross_attn.q_linear",
        ".cross_attn.kv_linear",
        ".cross_attn.proj",
        ".ffn.w1",
        ".ffn.w2",
        ".ffn.w3",
    ]
    FP8_OPTIMIZATION_EXCLUDE_KEYS: List[str] = [
        "adaLN_modulation",
        "mod_norm",
        "pre_crs_attn_norm",
        "LayerNorm",
        "norm",
        "final_layer",
        "t_embedder",
        "y_embedder",
        "x_embedder",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        # default params
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        # attention config
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        text_tokens_zero_pad: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
        )

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size,
            np.prod(self.patch_size),
            out_channels,
            adaln_tembed_dim,
        )

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

        self.lora_dict: Dict[str, Any] = {}
        self.active_loras = []
        self.blocks_to_swap: Optional[int] = None
        self.offloader: Optional[ModelOffloader] = None
        self._block_swap_cache: List[nn.Module] = []
        self._block_swap_ready: bool = False
        self._block_swap_hook: Optional[RemovableHandle] = None
        self._block_swap_block_hooks: List[RemovableHandle] = []

    def load_lora(self, lora_path, lora_key, multiplier=1.0, lora_network_dim=128, lora_network_alpha=64):
        lora_network_state_dict_loaded = load_file(lora_path, device="cpu")
        lora_network = create_lora_network(
            transformer=self,
            lora_network_state_dict_loaded=lora_network_state_dict_loaded,
            multiplier=multiplier,
            network_dim=lora_network_dim,
            network_alpha=lora_network_alpha,
        )

        lora_network.load_state_dict(lora_network_state_dict_loaded, strict=True)

        self.lora_dict[lora_key] = lora_network

    def enable_loras(self, lora_key_list=[]):
        self.disable_all_loras()

        module_loras: Dict[str, List[Any]] = {}  # {module_name: [lora1, lora2, ...]}
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        for lora_key in lora_key_list:
            if lora_key in self.lora_dict:
                for lora in self.lora_dict[lora_key].loras:
                    lora.to(model_device, dtype=model_dtype, non_blocking=True)
                    module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
                    if module_name not in module_loras:
                        module_loras[module_name] = []
                    module_loras[module_name].append(lora)
                self.active_loras.append(lora_key)

        for module_name, loras in module_loras.items():
            module = self._get_module_by_name(module_name)
            if not hasattr(module, "org_forward"):
                module.org_forward = module.forward  # type: ignore[attr-defined]
            module.forward = self._create_multi_lora_forward(module, loras)  # type: ignore[assignment]

    def _create_multi_lora_forward(self, module, loras):
        def multi_lora_forward(x, *args, **kwargs):
            weight_dtype = x.dtype
            org_forward = getattr(module, "org_forward")  # type: ignore[attr-defined]
            org_output = org_forward(x, *args, **kwargs)

            total_lora_output = 0
            for lora in loras:
                if lora.use_lora:
                    lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
                    lx = lora.lora_up(lx)
                    lora_output = lx.to(weight_dtype) * lora.multiplier * lora.alpha_scale
                    total_lora_output += lora_output

            return org_output + total_lora_output

        return multi_lora_forward

    def _get_module_by_name(self, module_name):
        try:
            module = self
            for part in module_name.split("."):
                module = getattr(module, part)
            return module
        except AttributeError as e:
            raise ValueError(f"Cannot find module: {module_name}, error: {e}")

    def disable_all_loras(self):
        for name, module in self.named_modules():
            if hasattr(module, "org_forward"):
                module.forward = module.org_forward
                delattr(module, "org_forward")

        for lora_key, lora_network in self.lora_dict.items():
            for lora in lora_network.loras:
                lora.to("cpu")

        self.active_loras.clear()

    def fp8_optimization(
        self,
        state_dict: Dict[str, torch.Tensor],
        device: torch.device,
        move_to_device: bool,
        use_scaled_mm: bool = False,
    ) -> Dict[str, torch.Tensor]:
        state_dict = optimize_state_dict_with_fp8(
            state_dict,
            device,
            self.FP8_OPTIMIZATION_TARGET_KEYS,
            self.FP8_OPTIMIZATION_EXCLUDE_KEYS,
            move_to_device=move_to_device,
        )
        apply_fp8_monkey_patch(self, state_dict, use_scaled_mm=use_scaled_mm)
        return state_dict

    def enable_block_swap(
        self, blocks_to_swap: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False
    ):
        self.blocks_to_swap = blocks_to_swap
        block_list = list(self.blocks)
        num_blocks = len(block_list)
        assert 0 < blocks_to_swap < num_blocks, (
            f"Cannot swap {blocks_to_swap} blocks out of {num_blocks}. Provide a value between 1 and {num_blocks - 1}."
        )
        self._block_swap_cache = block_list
        self.offloader = ModelOffloader(
            "longcat_block",
            block_list,
            num_blocks,
            blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
        )
        self._block_swap_ready = False

        if self._block_swap_hook is not None:
            self._block_swap_hook.remove()
            self._block_swap_hook = None
        for handle in self._block_swap_block_hooks:
            handle.remove()
        self._block_swap_block_hooks.clear()

        def _ensure_block_swap_ready(module, _inputs):
            if module.offloader and module.blocks_to_swap and not module.offloader.forward_only and not module._block_swap_ready:
                module.switch_block_swap_for_training()

        self._block_swap_hook = self.register_forward_pre_hook(_ensure_block_swap_ready)

        for idx, block in enumerate(self.blocks):

            def _make_block_hook(block_index: int):
                def _block_wait_hook(_module, _inputs):
                    if self.offloader and self.blocks_to_swap:
                        self.offloader.wait_for_block(block_index)
                        weighs_to_device(_module, self.offloader.device)

                return _block_wait_hook

            handle = block.register_forward_pre_hook(_make_block_hook(idx))
            self._block_swap_block_hooks.append(handle)

    def switch_block_swap_for_inference(self):
        if self.offloader:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            self._block_swap_ready = False

    def switch_block_swap_for_training(self):
        if self.offloader:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            self._block_swap_ready = True

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap is None or self.blocks_to_swap <= 0:
            nn.Module.to(self, device)
            return

        block_list = self._block_swap_cache or list(self.blocks)
        for block in block_list:
            nn.Module.to(block, "cpu")

        nn.Module.to(self, device)

        for block in block_list:
            nn.Module.to(block, "cpu")
        self._block_swap_ready = False

    def prepare_block_swap_before_forward(self):
        if self.offloader and self.blocks_to_swap:
            cached_blocks = self._block_swap_cache or list(self.blocks)
            if cached_blocks:
                self.offloader.prepare_block_devices_before_forward(cached_blocks)
                self._block_swap_ready = True

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        return_kv=False,
        kv_cache_dict={},
        skip_crs_attn=False,
        offload_kv_cache=False,
    ):
        B, _, T, H, W = hidden_states.shape

        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        assert self.patch_size[0] == 1, "Currently, 3D x_embedder should not compress the temporal dimension."

        # expand the shape of timestep from [B] to [B, T]
        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)  # [B, T]

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        with autocast(dtype=torch.float32):
            t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)  # [B, T, C_t]

        if encoder_hidden_states.dim() == 3:
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )  # [1, N_valid_tokens, C]
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()  # [B]
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

        # blocks
        kv_cache_dict_ret = {}
        swap_blocks = self._block_swap_cache if self.offloader and self.blocks_to_swap else []
        for i, block in enumerate(self.blocks):
            if self.offloader and self.blocks_to_swap and swap_blocks:
                self.offloader.wait_for_block(i)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_outputs = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )

            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

            if self.offloader and self.blocks_to_swap and swap_blocks:
                self.offloader.submit_move_blocks_forward(swap_blocks, i)

        if self.offloader and self.blocks_to_swap and swap_blocks:
            self.offloader.wait_for_block(len(self.blocks) - 1)
            self._block_swap_ready = False

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))  # [B, N, C=T_p*H_p*W_p*C_out]

        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)  # [B, C_out, H, W]
        hidden_states = cast(torch.Tensor, hidden_states)

        # cast to float32 for better accuracy
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        else:
            return hidden_states

    def unpatchify(self, x, N_t, N_h, N_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x
