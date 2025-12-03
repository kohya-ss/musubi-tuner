import os
from typing import List, Tuple, Optional, Union, Dict
import accelerate
from einops import rearrange

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from musubi_tuner.hunyuan_model.activation_layers import get_activation_layer
from musubi_tuner.hunyuan_model.norm_layers import get_norm_layer
from musubi_tuner.hunyuan_model.embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from musubi_tuner.hunyuan_model.attention import attention, parallel_attention, get_cu_seqlens
from musubi_tuner.hunyuan_model.posemb_layers import apply_rotary_emb
from musubi_tuner.hunyuan_model.mlp_layers import MLP, MLPEmbedder, FinalLayer
from musubi_tuner.hunyuan_model.modulate_layers import ModulateDiT, modulate, apply_gate
from musubi_tuner.hunyuan_model.token_refiner import SingleTokenRefiner
from musubi_tuner.modules.custom_offloading_utils import ModelOffloader
from musubi_tuner.utils.device_utils import synchronize_device, clean_memory_on_device
from musubi_tuner.hunyuan_model.posemb_layers import get_nd_rotary_pos_embed

from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper
from musubi_tuner.utils.safetensors_utils import load_safetensors


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_mode: str = "flash",
        split_attn: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        total_len: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (img_mod1_shift, img_mod1_scale, img_mod1_gate, img_mod2_shift, img_mod2_scale, img_mod2_gate) = self.img_mod(vec).chunk(
            6, dim=-1
        )
        (txt_mod1_shift, txt_mod1_scale, txt_mod1_gate, txt_mod2_shift, txt_mod2_scale, txt_mod2_gate) = self.txt_mod(vec).chunk(
            6, dim=-1
        )

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated, shift=img_mod1_shift, scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_modulated = None
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        img_qkv = None
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q_shape = img_q.shape
            img_k_shape = img_k.shape
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_q.shape == img_q_shape and img_k.shape == img_k_shape, (
                f"img_kk: {img_q.shape}, img_q: {img_q_shape}, img_kk: {img_k.shape}, img_k: {img_k_shape}"
            )
            # img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_modulated = None
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        txt_qkv = None
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        img_q_len = img_q.shape[1]
        img_kv_len = img_k.shape[1]
        batch_size = img_k.shape[0]
        q = torch.cat((img_q, txt_q), dim=1)
        img_q = txt_q = None
        k = torch.cat((img_k, txt_k), dim=1)
        img_k = txt_k = None
        v = torch.cat((img_v, txt_v), dim=1)
        img_v = txt_v = None

        assert cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1, (
            f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        )

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            l = [q, k, v]
            q = k = v = None
            attn = attention(
                l,
                mode=self.attn_mode,
                attn_mask=attn_mask,
                total_len=total_len,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=batch_size,
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q_len,
                img_kv_len=img_kv_len,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
            )

        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]
        attn = None

        # Calculate the img bloks.
        # img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = torch.addcmul(img, self.img_attn_proj(img_attn), img_mod1_gate.unsqueeze(1))
        img_attn = None
        # img = img + apply_gate(
        #     self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)),
        #     gate=img_mod2_gate,
        # )
        img = torch.addcmul(
            img, self.img_mlp(modulate(self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale)), img_mod2_gate.unsqueeze(1)
        )

        # Calculate the txt bloks.
        # txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = torch.addcmul(txt, self.txt_attn_proj(txt_attn), txt_mod1_gate.unsqueeze(1))
        txt_attn = None
        # txt = txt + apply_gate(
        #     self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)),
        #     gate=txt_mod2_gate,
        # )
        txt = torch.addcmul(
            txt, self.txt_mlp(modulate(self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale)), txt_mod2_gate.unsqueeze(1)
        )

        return img, txt

    # def forward(
    #     self,
    #     img: torch.Tensor,
    #     txt: torch.Tensor,
    #     vec: torch.Tensor,
    #     attn_mask: Optional[torch.Tensor] = None,
    #     cu_seqlens_q: Optional[torch.Tensor] = None,
    #     cu_seqlens_kv: Optional[torch.Tensor] = None,
    #     max_seqlen_q: Optional[int] = None,
    #     max_seqlen_kv: Optional[int] = None,
    #     freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    # ) -> Tuple[torch.Tensor, torch.Tensor]:
    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.img_attn_qkv.weight.device)
            return checkpoint(forward_fn, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_mode: str = "flash",
        split_attn: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(hidden_size, factor=3, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.hybrid_seq_parallel_attn = None

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def _forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        attn_mask: Optional[torch.Tensor] = None,
        total_len: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)
        x_mod = None
        # mlp = mlp.to("cpu", non_blocking=True)
        # clean_memory_on_device(x.device)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        qkv = None

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            q = k = None
            img_q_shape = img_q.shape
            img_k_shape = img_k.shape
            img_q, img_k = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert img_q.shape == img_q_shape and img_k_shape == img_k.shape, (
                f"img_kk: {img_q.shape}, img_q: {img_q.shape}, img_kk: {img_k.shape}, img_k: {img_k.shape}"
            )
            # img_q, img_k = img_qq, img_kk
            # del img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)
            del img_q, txt_q, img_k, txt_k

        # Compute attention.
        assert cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1, f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            l = [q, k, v]
            q = k = v = None
            attn = attention(
                l,
                mode=self.attn_mode,
                attn_mask=attn_mask,
                total_len=total_len,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
            )
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        # mlp = mlp.to(x.device)
        mlp = self.mlp_act(mlp)
        attn_mlp = torch.cat((attn, mlp), 2)
        attn = None
        mlp = None
        output = self.linear2(attn_mlp)
        attn_mlp = None
        # return x + apply_gate(output, gate=mod_gate)
        return torch.addcmul(x, output, mod_gate.unsqueeze(1))

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     vec: torch.Tensor,
    #     txt_len: int,
    #     attn_mask: Optional[torch.Tensor] = None,
    #     cu_seqlens_q: Optional[torch.Tensor] = None,
    #     cu_seqlens_kv: Optional[torch.Tensor] = None,
    #     max_seqlen_q: Optional[int] = None,
    #     max_seqlen_kv: Optional[int] = None,
    #     freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    # ) -> torch.Tensor:
    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            forward_fn = self._forward
            if self.activation_cpu_offloading:
                forward_fn = create_cpu_offloading_wrapper(forward_fn, self.linear1.weight.device)
            return checkpoint(forward_fn, *args, use_reentrant=False, **kwargs)
        else:
            return self._forward(*args, **kwargs)


class HYVideoDiffusionTransformer(nn.Module):  # ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    attn_mode: str
        The mode of the attention, default is flash.
    split_attn: bool
        Whether to use split attention (make attention as batch size 1).
    """

    # @register_to_config
    def __init__(
        self,
        text_states_dim: int,
        text_states_dim_2: int,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        attn_mode: str = "flash",
        split_attn: bool = False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = text_states_dim
        self.text_states_dim_2 = text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}")
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        self.attn_mode = attn_mode
        self.split_attn = split_attn
        print(f"Using {self.attn_mode} attention mode, split_attn: {self.split_attn}")

        # image projection
        self.img_in = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs)

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)

        # text modulation
        self.vector_in = MLPEmbedder(self.text_states_dim_2, self.hidden_size, **factory_kwargs)

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs) if guidance_embed else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    attn_mode=attn_mode,
                    split_attn=split_attn,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    attn_mode=attn_mode,
                    split_attn=split_attn,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader_double = None
        self.offloader_single = None
        self._enable_img_in_txt_in_offloading = False

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

        self.txt_in.enable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.enable_gradient_checkpointing(activation_cpu_offloading)

        print(
            f"HYVideoDiffusionTransformer: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}"
        )

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

        self.txt_in.disable_gradient_checkpointing()

        for block in self.double_blocks + self.single_blocks:
            block.disable_gradient_checkpointing()

        print("HYVideoDiffusionTransformer: Gradient checkpointing disabled.")

    def enable_img_in_txt_in_offloading(self):
        self._enable_img_in_txt_in_offloading = True

    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool, use_pinned_memory: bool = False):
        self.blocks_to_swap = num_blocks
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)
        double_blocks_to_swap = num_blocks // 2
        single_blocks_to_swap = (num_blocks - double_blocks_to_swap) * 2 + 1

        assert double_blocks_to_swap <= self.num_double_blocks - 1 and single_blocks_to_swap <= self.num_single_blocks - 1, (
            f"Cannot swap more than {self.num_double_blocks - 1} double blocks and {self.num_single_blocks - 1} single blocks. "
            f"Requested {double_blocks_to_swap} double blocks and {single_blocks_to_swap} single blocks."
        )

        self.offloader_double = ModelOffloader(
            "double",
            self.double_blocks,
            self.num_double_blocks,
            double_blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
            # , debug=True
        )
        self.offloader_single = ModelOffloader(
            "single",
            self.single_blocks,
            self.num_single_blocks,
            single_blocks_to_swap,
            supports_backward,
            device,
            use_pinned_memory,
            # , debug=True
        )
        print(
            f"HYVideoDiffusionTransformer: Block swap enabled. Swapping {num_blocks} blocks, double blocks: {double_blocks_to_swap}, single blocks: {single_blocks_to_swap}."
        )

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(True)
            self.offloader_single.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("HYVideoDiffusionTransformer: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader_double.set_forward_only(False)
            self.offloader_single.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("HYVideoDiffusionTransformer: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_double_blocks = self.double_blocks
            save_single_blocks = self.single_blocks
            self.double_blocks = None
            self.single_blocks = None

        self.to(device)

        if self.blocks_to_swap:
            self.double_blocks = save_double_blocks
            self.single_blocks = save_single_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        if self._enable_img_in_txt_in_offloading:
            self.img_in.to(x.device, non_blocking=True)
            self.txt_in.to(x.device, non_blocking=True)
            synchronize_device(x.device)

        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(f"Unsupported text_projection: {self.text_projection}")

        if self._enable_img_in_txt_in_offloading:
            self.img_in.to(torch.device("cpu"), non_blocking=True)
            self.txt_in.to(torch.device("cpu"), non_blocking=True)
            synchronize_device(x.device)
            clean_memory_on_device(x.device)

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        attn_mask = total_len = None
        if self.split_attn or self.attn_mode == "torch":
            # calculate text length and total length
            text_len = text_mask.sum(dim=1)  #  (bs, )
            total_len = img_seq_len + text_len  # (bs, )
        if self.attn_mode == "torch" and not self.split_attn:
            # initialize attention mask: bool tensor for sdpa, (b, 1, n, n)
            bs = img.shape[0]
            attn_mask = torch.zeros((bs, 1, max_seqlen_q, max_seqlen_q), dtype=torch.bool, device=text_mask.device)

            # set attention mask with total_len
            for i in range(bs):
                attn_mask[i, :, : total_len[i], : total_len[i]] = True
            total_len = None  # means we don't use split_attn

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        input_device = img.device
        for block_idx, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                attn_mask,
                total_len,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            if self.blocks_to_swap:
                self.offloader_double.wait_for_block(block_idx)

            img, txt = block(*double_block_args)

            if self.blocks_to_swap:
                self.offloader_double.submit_move_blocks_forward(self.double_blocks, block_idx)

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if self.blocks_to_swap:
            # delete img, txt to reduce memory usage
            del img, txt
            clean_memory_on_device(x.device)

        if len(self.single_blocks) > 0:
            for block_idx, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    attn_mask,
                    total_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                ]
                if self.blocks_to_swap:
                    self.offloader_single.wait_for_block(block_idx)

                x = block(*single_block_args)

                if self.blocks_to_swap:
                    self.offloader_single.submit_move_blocks_forward(self.single_blocks, block_idx)

        img = x[:, :img_seq_len, ...]
        x = None
        if img.device != input_device:
            img = img.to(input_device)

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters()) + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}


def load_dit_model(text_states_dim, text_states_dim_2, in_channels, out_channels, factor_kwargs):
    """load hunyuan video model

    NOTE: Only support HYVideo-T/2-cfgdistill now.

    Args:
        text_state_dim (int): text state dimension
        text_state_dim_2 (int): text state dimension 2
        in_channels (int): input channels number
        out_channels (int): output channels number
        factor_kwargs (dict): factor kwargs

    Returns:
        model (nn.Module): The hunyuan video model
    """
    # if args.model in HUNYUAN_VIDEO_CONFIG.keys():
    model = HYVideoDiffusionTransformer(
        text_states_dim=text_states_dim,
        text_states_dim_2=text_states_dim_2,
        in_channels=in_channels,
        out_channels=out_channels,
        **HUNYUAN_VIDEO_CONFIG["HYVideo-T/2-cfgdistill"],
        **factor_kwargs,
    )
    return model
    # else:
    #     raise NotImplementedError()


def load_state_dict(model, model_path):
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=True)

    load_key = "module"
    if load_key in state_dict:
        state_dict = state_dict[load_key]
    else:
        raise KeyError(
            f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint are: {list(state_dict.keys())}."
        )
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def load_transformer(dit_path, attn_mode, split_attn, device, dtype, in_channels=16) -> HYVideoDiffusionTransformer:
    # =========================== Build main model ===========================
    factor_kwargs = {"device": device, "dtype": dtype, "attn_mode": attn_mode, "split_attn": split_attn}
    latent_channels = 16
    out_channels = latent_channels

    with accelerate.init_empty_weights():
        transformer = load_dit_model(
            text_states_dim=4096,
            text_states_dim_2=768,
            in_channels=in_channels,
            out_channels=out_channels,
            factor_kwargs=factor_kwargs,
        )

    if os.path.splitext(dit_path)[-1] == ".safetensors":
        # loading safetensors: may be already fp8
        device = torch.device(device) if device is not None else None
        state_dict = load_safetensors(dit_path, device=device, disable_mmap=True, dtype=dtype)
        transformer.load_state_dict(state_dict, strict=True, assign=True)
    else:
        transformer = load_state_dict(transformer, dit_path)

    return transformer


def get_rotary_pos_embed_by_shape(model, latents_size):
    target_ndim = 3
    ndim = 5 - 2

    if isinstance(model.patch_size, int):
        assert all(s % model.patch_size == 0 for s in latents_size), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size for s in latents_size]
    elif isinstance(model.patch_size, list):
        assert all(s % model.patch_size[idx] == 0 for idx, s in enumerate(latents_size)), (
            f"Latent size(last {ndim} dimensions) should be divisible by patch size({model.patch_size}), but got {latents_size}."
        )
        rope_sizes = [s // model.patch_size[idx] for idx, s in enumerate(latents_size)]

    if len(rope_sizes) != target_ndim:
        rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
    head_dim = model.hidden_size // model.heads_num
    rope_dim_list = model.rope_dim_list
    if rope_dim_list is None:
        rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"

    rope_theta = 256
    freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
        rope_dim_list, rope_sizes, theta=rope_theta, use_real=True, theta_rescale_factor=1
    )
    return freqs_cos, freqs_sin


def get_rotary_pos_embed(vae_name, model, video_length, height, width):
    # 884
    if "884" in vae_name:
        latents_size = [(video_length - 1) // 4 + 1, height // 8, width // 8]
    elif "888" in vae_name:
        latents_size = [(video_length - 1) // 8 + 1, height // 8, width // 8]
    else:
        latents_size = [video_length, height // 8, width // 8]

    return get_rotary_pos_embed_by_shape(model, latents_size)
