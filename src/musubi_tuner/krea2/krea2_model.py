import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from musubi_tuner.modules.custom_offloading_utils import BlockSwapConfig, create_offloader
from musubi_tuner.modules.fp8_optimization_utils import apply_fp8_monkey_patch
from musubi_tuner.utils.lora_utils import load_safetensors_with_lora_and_fp8
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# region constants

DIM = 6144
NUM_HEADS = 48
NUM_KV_HEADS = 12
HEAD_DIM = 128
NUM_LAYERS = 28
IN_CHANNELS = 64  # 16 * 2 * 2
OUT_CHANNELS = 16
PATCH_SIZE = 2
MLP_DIM = 16384
TXT_DIM = 2560
TXT_HEADS = 20
TXT_KV_HEADS = 20
TXT_LAYERS = 12  # 2 layerwise + 2 refiner = 4 blocks, but txtfusion has 12 sub-layers total
THETA = 1e3
AXES_DIM = [32, 48, 48]  # head_dim=128 split into 3 axes

FP8_OPTIMIZATION_TARGET_KEYS = ["blocks"]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["norm", "qknorm", "mod", "tproj", "tmlp", "txtfusion", "last"]

# endregion constants


# region helpers


def get_timestep_embedding(timesteps, num_channels=256, max_period=10000):
    half = num_channels // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if num_channels % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        # `scale` is always kept in (b)float (norm keys are in FP8_OPTIMIZATION_EXCLUDE_KEYS),
        # so a single cast-and-scale path is correct.
        x = x.to(self.scale.dtype) * self.scale
        return x.to(input_dtype)


def apply_rope(q, k, freqs):
    """Apply rotary embeddings. freqs: (S, D) complex or (S, D//2, 2) real.
    q: (B, S, H, D), k: (B, S, H_kv, D)
    Uses complex multiplication approach matching official mmdit.py.
    """
    # freqs: (S, D) complex
    # Reshape q/k for complex view: (B, S, H, D//2, 2) -> (B, S, H, D//2) complex
    q_complex = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_complex = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))

    # freqs: (S, D//2) complex -> expand to match q/k heads
    freqs = freqs.unsqueeze(1)  # (S, 1, D//2)
    q_out = torch.view_as_real(q_complex * freqs).flatten(-2)
    k_out = torch.view_as_real(k_complex * freqs).flatten(-2)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def compute_rope_freqs(pos, axes_dim, theta=1e3):
    """Compute RoPE frequencies for 3-axis positions.
    pos: (L, 3) tensor with (axis0, H, W) positions.
    Returns: (L, HEAD_DIM) complex tensor.
    """
    freqs = []
    for i, d in enumerate(axes_dim):
        # (L, d//2) frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float32, device=pos.device) / d))
        # pos[:, i]: (L,) -> (L, d//2)
        freqs_i = pos[:, i : i + 1].float() * inv_freq[None]  # (L, d//2)
        freqs_i = torch.polar(torch.ones_like(freqs_i), freqs_i)  # (L, d//2) complex
        freqs.append(freqs_i)
    return torch.cat(freqs, dim=-1)  # (L, HEAD_DIM//2) complex


def build_pos(img_h, img_w, txt_len, device):
    """Build position tensor. Text positions are all zeros, image positions are (0, row, col).
    Returns (L_txt + L_img, 3) tensor.
    """
    txt_pos = torch.zeros(txt_len, 3, device=device)
    img_rows = torch.arange(img_h, device=device).view(-1, 1).expand(img_h, img_w)
    img_cols = torch.arange(img_w, device=device).view(1, -1).expand(img_h, img_w)
    img_pos = torch.stack([torch.zeros(img_h * img_w, device=device), img_rows.reshape(-1), img_cols.reshape(-1)], dim=-1)
    return torch.cat([txt_pos, img_pos], dim=0)


# endregion helpers


# region txtfusion


class TxtFusionAttention(nn.Module):
    def __init__(self, dim, heads, kv_heads, head_dim=128):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.inner_kv_dim = kv_heads * head_dim

        self.wq = nn.Linear(dim, self.inner_dim, bias=False)
        self.wk = nn.Linear(dim, self.inner_kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.inner_kv_dim, bias=False)
        self.wo = nn.Linear(self.inner_dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.qknorm = QKNorm(head_dim)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).reshape(B, L, self.heads, self.head_dim)
        k = self.wk(x).reshape(B, L, self.kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, L, self.kv_heads, self.head_dim)
        gate = torch.sigmoid(self.gate(x))
        q, k = self.qknorm(q, k)
        # no RoPE for txtfusion
        # repeat k/v for GQA
        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)
        q = q.transpose(1, 2)  # B, H, L, D
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if mask is not None:
            # mask: (B, L) bool, True=valid
            mask = mask[:, None, None, :]  # B, 1, 1, L
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(B, L, -1)
        return self.wo(out * gate)


class QKNorm(nn.Module):
    def __init__(self, head_dim, eps=1e-6):
        super().__init__()
        self.qnorm = RMSNorm(head_dim, eps)
        self.knorm = RMSNorm(head_dim, eps)

    def forward(self, q, k):
        return self.qnorm(q), self.knorm(k)


class TxtFusionMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TxtFusionBlock(nn.Module):
    """A transformer block in txtfusion (layerwise or refiner)."""

    def __init__(self, dim, heads, kv_heads, mlp_dim, head_dim=128):
        super().__init__()
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.attn = TxtFusionAttention(dim, heads, kv_heads, head_dim)
        self.mlp = TxtFusionMLP(dim, mlp_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.prenorm(x), mask)
        x = x + self.mlp(self.postnorm(x))
        return x


class TxtFusion(nn.Module):
    """Text fusion module.

    Per spec §6.1 the order is:
      input (B, L, 12, 2560)
        -> reshape (B*L, 12, 2560), run 2 ``layerwise_blocks`` (attention over the LAYER axis, len 12)
        -> ``projector`` Linear(12 -> 1) collapse to (B, L, 2560)
        -> run 2 ``refiner_blocks`` (attention over the TOKEN axis, with the token mask)
    The layerwise blocks attend across the 12 selected hidden layers; the refiner blocks attend
    across text tokens. ``projector`` is a Linear (checkpoint key ``txtfusion.projector.weight [1,12]``),
    not a learned weighted sum.
    """

    def __init__(self, dim=TXT_DIM, heads=TXT_HEADS, kv_heads=TXT_KV_HEADS, head_dim=128, num_select_layers=12):
        super().__init__()
        # Linear(12 -> 1) over the layer axis. Checkpoint key: txtfusion.projector.weight [1, 12].
        self.projector = nn.Linear(num_select_layers, 1, bias=False)

        # 2 layerwise blocks + 2 refiner blocks = 4 blocks
        # mlp_dim for txtfusion: ceil_to_128(int(2*2560/3)*4) = 6912
        txt_mlp_dim = 6912
        self.layerwise_blocks = nn.ModuleList([TxtFusionBlock(dim, heads, kv_heads, txt_mlp_dim, head_dim) for _ in range(2)])
        self.refiner_blocks = nn.ModuleList([TxtFusionBlock(dim, heads, kv_heads, txt_mlp_dim, head_dim) for _ in range(2)])

    def forward(self, context, mask=None):
        """context: (B, L, 12, 2560) -> (B, L, 2560)
        mask: (B, L) bool, used only by the refiner (token-axis) blocks.
        """
        B, L, num_layers, D = context.shape

        # Layerwise blocks: attend over the 12-layer axis, batched over (B*L). No mask here:
        # every one of the 12 layers is valid for a token.
        x = context.reshape(B * L, num_layers, D)  # (B*L, 12, 2560)
        for block in self.layerwise_blocks:
            x = block(x, mask=None)

        # Projector Linear(12 -> 1) collapses the layer axis.
        x = x.transpose(1, 2)  # (B*L, 2560, 12)
        x = self.projector(x)  # (B*L, 2560, 1)
        x = x.squeeze(-1).reshape(B, L, D)  # (B, L, 2560)

        # Refiner blocks: attend over the token axis with the token mask.
        for block in self.refiner_blocks:
            x = block(x, mask)

        return x


# endregion txtfusion


# region main DiT blocks


class SingleStreamAttention(nn.Module):
    def __init__(self, dim=DIM, heads=NUM_HEADS, kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, attn_mode="torch", split_attn=False):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.inner_kv_dim = kv_heads * head_dim
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        self.wq = nn.Linear(dim, self.inner_dim, bias=False)
        self.wk = nn.Linear(dim, self.inner_kv_dim, bias=False)
        self.wv = nn.Linear(dim, self.inner_kv_dim, bias=False)
        self.wo = nn.Linear(self.inner_dim, dim, bias=False)
        self.gate = nn.Linear(dim, dim, bias=False)
        self.qknorm = QKNorm(head_dim)

    def forward(self, x, freqs, mask=None, txt_len=None):
        """x: (B, L, DIM), freqs: (L, HEAD_DIM//2) complex
        mask: (B, L) bool or None
        txt_len: int, length of text prefix (for split_attn)
        """
        B, L, _ = x.shape
        q = self.wq(x).reshape(B, L, self.heads, self.head_dim)
        k = self.wk(x).reshape(B, L, self.kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, L, self.kv_heads, self.head_dim)
        gate = self.gate(x)

        q, k = self.qknorm(q, k)

        # Apply RoPE (text tokens get zero rotation since their pos is all zeros)
        q, k = apply_rope(q, k, freqs)

        # GQA: repeat k/v
        if self.kv_heads != self.heads:
            rep = self.heads // self.kv_heads
            k = k.repeat_interleave(rep, dim=2)
            v = v.repeat_interleave(rep, dim=2)

        # Sigmoid gate
        gate = torch.sigmoid(gate)  # (B, L, DIM)

        if self.split_attn and txt_len is not None:
            # Split attention: text and image separately
            q_txt, q_img = q[:, :txt_len], q[:, txt_len:]
            k_txt, k_img = k[:, :txt_len], k[:, txt_len:]
            v_txt, v_img = v[:, :txt_len], v[:, txt_len:]
            # text attends to text only
            q_t = q_txt.transpose(1, 2)
            k_t = k_txt.transpose(1, 2)
            v_t = v_txt.transpose(1, 2)
            out_txt = F.scaled_dot_product_attention(q_t, k_t, v_t)
            # image attends to all (text + image)
            q_i = q_img.transpose(1, 2)
            k_all = k.transpose(1, 2)
            v_all = v.transpose(1, 2)
            if mask is not None:
                mask_all = mask[:, None, None, :]
            else:
                mask_all = None
            out_img = F.scaled_dot_product_attention(q_i, k_all, v_all, attn_mask=mask_all)
            out = torch.cat([out_txt, out_img], dim=2).transpose(1, 2).reshape(B, L, -1)
        else:
            q = q.transpose(1, 2)  # B, H, L, D
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            if mask is not None:
                mask = mask[:, None, None, :]
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out = out.transpose(1, 2).reshape(B, L, -1)

        out = self.wo(out * gate)
        return out


class SingleStreamMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim=DIM, hidden_dim=MLP_DIM):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Modulation(nn.Module):
    """adaLN-single modulation: produces 6*dim bias from tvec."""

    def __init__(self, dim=DIM):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    def forward(self, tvec):
        return tvec + self.lin


class SingleStreamBlock(nn.Module):
    """Single-stream MMDiT block.

    From spec §12.2:
    prescale,preshift,pregate,postscale,postshift,postgate = mod(tvec)
    x = x + pregate * attn((1+prescale)*prenorm(x)+preshift, freqs, mask)
    x = x + postgate * mlp((1+postscale)*postnorm(x)+postshift)
    """

    def __init__(self, dim=DIM, heads=NUM_HEADS, kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, attn_mode="torch", split_attn=False):
        super().__init__()
        self.prenorm = RMSNorm(dim)
        self.postnorm = RMSNorm(dim)
        self.mod = Modulation(dim)
        self.attn = SingleStreamAttention(dim, heads, kv_heads, head_dim, attn_mode, split_attn)
        self.mlp = SingleStreamMLP(dim, MLP_DIM)

    def forward(self, x, tvec, freqs, mask=None, txt_len=None):
        # tvec: (B, 6*DIM)
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(tvec).chunk(6, dim=-1)

        # Attention
        x_norm = self.prenorm(x)
        x_mod = x_norm * (1 + prescale.unsqueeze(1)) + preshift.unsqueeze(1)
        attn_out = self.attn(x_mod, freqs, mask, txt_len)
        x = x + pregate.unsqueeze(1) * attn_out

        # MLP
        x_norm2 = self.postnorm(x)
        x_mod2 = x_norm2 * (1 + postscale.unsqueeze(1)) + postshift.unsqueeze(1)
        mlp_out = self.mlp(x_mod2)
        x = x + postgate.unsqueeze(1) * mlp_out

        return x


# endregion main DiT blocks


# region last layer


class LastLayer(nn.Module):
    """Final layer: norm + modulation + residual MLP + linear projection.

    From spec §12.2:
    scale,shift = modulation(t)  # t is the timestep embedding (B, DIM), not tvec
    x = (1+scale)*norm(x) + shift + up(down(x))
    x = linear(x)  # 6144 -> 64
    """

    def __init__(self, dim=DIM, out_channels=IN_CHANNELS):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.modulation = nn.Module()
        self.modulation.lin = nn.Parameter(torch.zeros(2, dim))  # (2, DIM) for scale and shift
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.up = nn.Linear(dim, dim, bias=False)
        self.down = nn.Linear(dim, dim, bias=False)

    def forward(self, x, t):
        """x: (B, L, DIM), t: (B, DIM) timestep embedding"""
        scale, shift = self.modulation.lin  # each (DIM,)
        x_norm = self.norm(x)
        x = x_norm * (1 + scale.unsqueeze(0).unsqueeze(0)) + shift.unsqueeze(0).unsqueeze(0)
        x = x + self.up(self.down(x))
        x = self.linear(x)
        return x


# endregion last layer


# region full model


class Krea2Transformer2DModel(nn.Module):
    def __init__(
        self,
        attn_mode: str = "torch",
        split_attn: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.attn_mode = attn_mode
        self.split_attn = split_attn
        self.inner_dim = DIM
        self.in_channels = IN_CHANNELS
        self.out_channels = OUT_CHANNELS
        self.patch_size = PATCH_SIZE
        self.num_layers = NUM_LAYERS

        # Image input: Linear(64 -> 6144)
        self.first = nn.Linear(IN_CHANNELS, DIM, bias=True)

        # Time embedding: temb(256) -> tmlp -> tproj
        # tmlp: Sequential(Linear(256, 6144), GELU, Linear(6144, 6144))
        self.tmlp = nn.Sequential(
            nn.Linear(256, DIM, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(DIM, DIM, bias=True),
        )
        # tproj: Sequential(GELU, Linear(6144, 6*6144))
        self.tproj = nn.Sequential(
            nn.GELU(approximate="tanh"),
            nn.Linear(DIM, 6 * DIM, bias=True),
        )

        # Text fusion + text MLP
        self.txtfusion = TxtFusion()
        # txtmlp: Sequential(RMSNorm(2560), Linear(2560, 6144), GELU, Linear(6144, 6144))
        self.txtmlp = nn.Sequential(
            RMSNorm(TXT_DIM),
            nn.Linear(TXT_DIM, DIM, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(DIM, DIM, bias=True),
        )

        # Main blocks
        self.blocks = nn.ModuleList(
            [SingleStreamBlock(DIM, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, attn_mode, split_attn) for _ in range(NUM_LAYERS)]
        )

        # Last layer
        self.last = LastLayer(DIM, IN_CHANNELS)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        self.blocks_to_swap = None
        self.offloader = None

        if dtype is not None:
            self.to(dtype)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False):
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading
        print(f"Krea2Model: Gradient checkpointing enabled. Activation CPU offloading: {activation_cpu_offloading}")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        print("Krea2Model: Gradient checkpointing disabled.")

    def enable_block_swap(self, blocks_to_swap: int, config: BlockSwapConfig):
        self.blocks_to_swap = blocks_to_swap
        self.num_blocks = len(self.blocks)
        assert self.blocks_to_swap <= self.num_blocks - 1
        self.offloader = create_offloader("krea2-block", self.blocks, self.num_blocks, self.blocks_to_swap, config)
        print(f"Krea2Model: Block swap enabled. Swapping {self.blocks_to_swap} blocks out of {self.num_blocks} blocks.")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()
            print("Krea2Model: Block swap set to forward only.")

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()
            print("Krea2Model: Block swap set to forward and backward.")

    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = None
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def _gradient_checkpointing_func(self, block, *args):
        if self.activation_cpu_offloading:
            block = create_cpu_offloading_wrapper(block, self.first.weight.device)
        return torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.Tensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, N_img, 64) packed image latents
            encoder_hidden_states: (B, L_txt, 12, 2560) multi-layer hidden states
            encoder_hidden_states_mask: (B, L_txt) attention mask
            timestep: (B,) timestep in [0, 1]
            img_shapes: list of (1, H//16, W//16) for image patch grid
            txt_seq_lens: list of int, text sequence lengths
        """
        device = hidden_states.device
        B = hidden_states.shape[0]

        # Image input projection
        img = self.first(hidden_states)  # (B, N_img, DIM)

        # Time embedding
        t_emb = get_timestep_embedding(timestep, num_channels=256)  # (B, 256)
        t = self.tmlp(t_emb.to(img.dtype))  # (B, DIM)
        tvec = self.tproj(t)  # (B, 6*DIM)

        # Text fusion: (B, L_txt, 12, 2560) -> (B, L_txt, 2560) -> (B, L_txt, DIM)
        if encoder_hidden_states_mask is not None and encoder_hidden_states_mask.dtype != torch.bool:
            encoder_hidden_states_mask = encoder_hidden_states_mask.bool()
        _mask = encoder_hidden_states_mask

        context = self.txtfusion(encoder_hidden_states, mask=_mask)  # (B, L_txt, 2560)
        context = self.txtmlp(context)  # (B, L_txt, DIM)

        # Concatenate: text first, then image
        txt_len = context.shape[1]
        combined = torch.cat([context, img], dim=1)  # (B, L_txt + N_img, DIM)

        # Build full mask
        if _mask is not None:
            img_mask = torch.ones(B, img.shape[1], dtype=torch.bool, device=device)
            full_mask = torch.cat([_mask, img_mask], dim=1)  # (B, L_txt + N_img)
        else:
            full_mask = None

        # Pad to multiple of 256
        total_len = combined.shape[1]
        pad_to = math.ceil(total_len / 256) * 256
        if pad_to > total_len:
            pad_len = pad_to - total_len
            combined = F.pad(combined, (0, 0, 0, pad_len))
            if full_mask is not None:
                full_mask = F.pad(full_mask, (0, pad_len), value=False)

        # Build positions: text pos = zeros, image pos = (0, row, col)
        img_h, img_w = img_shapes[0][1], img_shapes[0][2]  # H//16, W//16
        pos = build_pos(img_h, img_w, txt_len, device)  # (txt_len + img_h*img_w, 3)
        # Pad positions
        if pad_to > pos.shape[0]:
            pos = F.pad(pos, (0, 0, 0, pad_to - pos.shape[0]), value=0)

        # Compute RoPE frequencies
        freqs = compute_rope_freqs(pos, AXES_DIM, THETA)  # (pad_to, HEAD_DIM//2) complex

        # Run blocks
        for index_block, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index_block)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                combined = self._gradient_checkpointing_func(block, combined, tvec, freqs, full_mask, txt_len)
            else:
                combined = block(combined, tvec, freqs, full_mask, txt_len)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks_forward(self.blocks, index_block)

        # Last layer: use t (not tvec) as the modulation input
        out = self.last(combined, t)  # (B, pad_to, 64)

        # Extract only the image portion: [txt_len : txt_len + N_img]
        out = out[:, txt_len : txt_len + img.shape[1]]

        return out


# endregion full model


# region loading


def create_model(
    attn_mode: str,
    split_attn: bool,
    dtype: Optional[torch.dtype] = None,
) -> Krea2Transformer2DModel:
    with init_empty_weights():
        logger.info(f"Creating Krea2Transformer2DModel. Attn mode: {attn_mode}, split_attn: {split_attn}")
        model = Krea2Transformer2DModel(
            attn_mode=attn_mode,
            split_attn=split_attn,
            dtype=dtype,
        )
    return model


def load_krea2_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[List[float]] = None,
    disable_numpy_memmap: bool = False,
) -> Krea2Transformer2DModel:
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    model = create_model(attn_mode, split_attn, dit_weight_dtype)

    logger.info(f"Loading Krea2 DiT model from {dit_path}, device={loading_device}")
    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
        disable_numpy_memmap=disable_numpy_memmap,
    )

    # Remove possible prefix
    for key in list(sd.keys()):
        if key.startswith("model.diffusion_model."):
            sd[key[22:]] = sd.pop(key)

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)
        if loading_device.type != "cpu":
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded Krea2 DiT model from {dit_path}, info={info}")

    return model


# endregion loading
