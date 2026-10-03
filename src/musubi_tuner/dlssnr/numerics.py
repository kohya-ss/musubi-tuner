# Portions adapted from OpenDLSS-NR, Copyright (c) 2026 maan (MIT).
# See LICENSE.OpenDLSS-NR and NOTICE.md in this directory.

"""FP32 train_surrogate ops.

These are not the native E4M3/half network. Differences that change results:

- Activations, cosine reductions and attention weights use explicit half/E4 publications with STE.
- Matrix products accumulate in FP32, not native F13/F24 chains or split-K order.
- Residual adds happen after the matmul. Native seeds the skip into the first accumulator.
- 32-channel raw-half residuals are kept separately from E4 matrix operands.
- ViT padding and its half denominator correction are explicit.
- Window tokens outside the field stay in the softmax. Their Q/K/V are zero, so the score is the prior.
"""

from __future__ import annotations

import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from musubi_tuner.dlssnr.arithmetic import cosine_half, e4m3_ste, exp_weight, half_ste, sum64_half

SURROGATE_FLAGS = {
    "profile": "train_surrogate",
    "compute_dtype": "float32",
    "tf32": False,
    "operators_version": 2,
    "activation": "mp_cubic_half_ste",
    "softmax": "native_exp_half_tree_e4_window_and_unnormalized_vit",
    "qk_norm": "half_tree_cosine_ste_zero_rows_defined",
    "vit_q_scale": "half_norm_then_half_sqrt32_then_half_temperature",
    "vit_padding": "multiple_64_half_denominator_correction",
    "residual": "half_scaled_post_add_not_native_accumulator_seed",
    "quantization": "half_then_e4m3_activation_fake_quant",
    "weight_publication": "fp32_master_not_requantized",
    "matmul_accumulator": "fp32_not_native_f13_f24_or_split_k",
    "surrogate_backward": "round_identity_saturation_clamp_exp_linear_mantissa",
    "history_sampler": "bilinear",
    "history_publish": "fp32",
}

# Window origin is (-shift_x, -shift_y). Index by the 4-cycle phase.
PHASE_SHIFTS = ((0, 0), (4, 4), (4, 0), (0, 4))


def mp_cubic_silu(value: torch.Tensor) -> torch.Tensor:
    """Trainable FP32 form of the 310.8.0 magnitude-preserving cubic activation.

    Only the polynomial's argument is clamped. Its positive tail keeps the
    original value and gain; ordinary SiLU suppresses the pretrained FFN paths.
    """
    bounded = value.clamp(-4, 4)
    inner = -0.055908203125 * bounded.abs() + 0.447265625
    return value * (bounded * inner + 0.89453125)


@contextmanager
def fp32_execution():
    """Do not silently replace FP32 products with TF32, or leak backend flags to other models."""
    matmul, cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def pool2x2(x: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:
    """2x2 box pool. A destination whose 2x2 is not fully inside the source is zero."""
    batch, channels, height, width = x.shape
    usable_h = height // 2
    usable_w = width // 2
    out = x.new_zeros(batch, channels, out_height, out_width)
    if usable_h == 0 or usable_w == 0:
        return out
    cropped = x[:, :, : usable_h * 2, : usable_w * 2]
    top = half_ste(cropped[:, :, 0::2, 0::2] + cropped[:, :, 0::2, 1::2])
    bottom = half_ste(cropped[:, :, 1::2, 0::2] + cropped[:, :, 1::2, 1::2])
    pooled = e4m3_ste(half_ste(top + bottom) * 0.25)
    copy_h = min(usable_h, out_height)
    copy_w = min(usable_w, out_width)
    out[:, :, :copy_h, :copy_w] = pooled[:, :, :copy_h, :copy_w]
    return out


def nearest_upsample(x: torch.Tensor, out_height: int, out_width: int) -> torch.Tensor:
    """Nearest upsample used by the decoder: source pixel (y // 2, x // 2)."""
    if out_height < 1 or out_width < 1:
        raise ValueError("upsample output is empty")
    y = torch.arange(out_height, device=x.device) // 2
    x_index = torch.arange(out_width, device=x.device) // 2
    if int(y[-1]) >= x.shape[-2] or int(x_index[-1]) >= x.shape[-1]:
        raise RuntimeError(f"cannot nearest-upsample {tuple(x.shape[-2:])} to {out_height}x{out_width}: source index out of range")
    return x[:, :, y][:, :, :, x_index]


def apply_linear(weight: torch.Tensor | torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Apply a projection module (including adapters), or an explicit [out, in] weight."""
    if isinstance(weight, torch.nn.Module):
        return weight(x)
    if x.ndim == 4:
        return F.conv2d(x, weight.view(weight.shape[0], weight.shape[1], 1, 1))
    return F.linear(x, weight)


def _matmul(left, right, compute_dtype=None):
    with torch.autocast(left.device.type, dtype=compute_dtype, enabled=compute_dtype is not None):
        return torch.matmul(left, right).float()


def _cosine(values: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(values, dim=-1, keepdim=True)
    scaled = values / norm.clamp_min(1e-12)
    return torch.where(norm > 0, scaled, torch.zeros_like(values))


def _as_heads(qkv: torch.Tensor, heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """qkv [..., heads * 96] head-major Q32,K32,V32 -> three [..., heads, 32]."""
    grouped = qkv.reshape(*qkv.shape[:-1], heads, 96)
    query, key, value = grouped.split(32, dim=-1)
    return query, key, value


def window_attention(
    y: torch.Tensor,
    qkv_weight: torch.Tensor | torch.nn.Module,
    proj_weight: torch.Tensor | torch.nn.Module,
    prior: torch.Tensor,
    temperature: torch.Tensor,
    skip_scale: torch.Tensor,
    phase: int,
    *,
    compute_dtype: torch.dtype | None = None,
    backend: str = "native",
) -> torch.Tensor:
    """Shifted 8x8 window attention. `y` is NCHW. Out-of-field tokens stay in the softmax."""
    batch, channels, height, width = y.shape
    heads = channels // 32
    if heads < 1 or channels != heads * 32:
        raise ValueError(f"window attention channels {channels} are not a multiple of 32")
    shift_x, shift_y = PHASE_SHIFTS[phase & 3]
    windows_y = math.ceil((height + shift_y) / 8)
    windows_x = math.ceil((width + shift_x) / 8)
    qkv = half_ste(apply_linear(qkv_weight, e4m3_ste(y)))
    query, key, value = _window_qkv(qkv, height, width, windows_y, windows_x, shift_x, shift_y, heads)
    query = e4m3_ste(cosine_half(query) * half_ste(temperature).view(1, 1, heads, 1, 1))
    key, value = e4m3_ste(cosine_half(key)), e4m3_ste(value)
    if backend == "native":
        scores = half_ste(_matmul(query, key.transpose(-1, -2), compute_dtype) + half_ste(prior).view(1, 1, heads, 64, 64))
        exponential = exp_weight(scores)
        probability = e4m3_ste(exponential * half_ste(sum64_half(exponential, window=True).reciprocal()))
        mixed = e4m3_ste(_matmul(probability, value, compute_dtype))
    else:
        from musubi_tuner.dlssnr.attention import attention

        mixed = e4m3_ste(attention(query, key, value, backend=backend, prior=half_ste(prior), compute_dtype=compute_dtype))
    mixed = mixed.permute(0, 1, 3, 2, 4).reshape(batch, windows_y * windows_x, 64, channels)
    attended = _scatter_windows(mixed, height, width, windows_y, windows_x, shift_x, shift_y)
    skip = half_ste(y) if channels == 32 else e4m3_ste(y)
    return half_ste(apply_linear(proj_weight, attended) + half_ste(skip * half_ste(skip_scale).view(1, -1, 1, 1)))


def _window_qkv(qkv, height, width, windows_y, windows_x, shift_x, shift_y, heads):
    device = qkv.device
    token_y = torch.arange(8, device=device)
    token_x = torch.arange(8, device=device)
    y_img = torch.arange(windows_y, device=device)[:, None] * 8 + token_y[None, :] - shift_y
    x_img = torch.arange(windows_x, device=device)[:, None] * 8 + token_x[None, :] - shift_x
    valid_y = (y_img >= 0) & (y_img < height)
    valid_x = (x_img >= 0) & (x_img < width)
    gathered = qkv[:, :, y_img.clamp(0, height - 1)[:, None, :, None], x_img.clamp(0, width - 1)[None, :, None, :]]
    # The broadcast advanced indices produce [windows_y, windows_x, token_y, token_x].
    gathered = gathered.permute(0, 2, 3, 4, 5, 1).reshape(qkv.shape[0], windows_y * windows_x, 64, qkv.shape[1])
    valid = (valid_y[:, None, :, None] & valid_x[None, :, None, :]).reshape(windows_y * windows_x, 64)
    gathered = gathered * valid.to(gathered.dtype).view(1, -1, 64, 1)
    query, key, value = _as_heads(gathered, heads)
    # [B, windows, 64, heads, 32] -> [B, windows, heads, 64, 32]
    query = query.permute(0, 1, 3, 2, 4)
    key = key.permute(0, 1, 3, 2, 4)
    value = value.permute(0, 1, 3, 2, 4)
    return query, key, value


def _scatter_windows(tokens, height, width, windows_y, windows_x, shift_x, shift_y):
    batch, _, _, channels = tokens.shape
    device = tokens.device
    linear = torch.arange(64, device=device)
    y_img = torch.arange(windows_y, device=device)[:, None] * 8 + (linear // 8)[None, :] - shift_y
    x_img = torch.arange(windows_x, device=device)[:, None] * 8 + (linear % 8)[None, :] - shift_x
    y_grid = y_img[:, None, :].expand(windows_y, windows_x, 64)
    x_grid = x_img[None, :, :].expand(windows_y, windows_x, 64)
    valid = (y_grid >= 0) & (y_grid < height) & (x_grid >= 0) & (x_grid < width)
    flat = (y_grid.clamp(0, height - 1) * width + x_grid.clamp(0, width - 1)).reshape(-1)
    mask = valid.reshape(-1)
    source = tokens.reshape(batch, windows_y, windows_x, 64, channels).permute(0, 4, 1, 2, 3).reshape(batch, channels, -1)
    out = tokens.new_zeros(batch, channels, height * width)
    out[:, :, flat[mask]] = source[:, :, mask]
    return out.view(batch, channels, height, width)


def global_attention(
    y: torch.Tensor,
    qkv_weight: torch.Tensor | torch.nn.Module,
    proj_weight: torch.Tensor | torch.nn.Module,
    temperature: torch.Tensor,
    skip_scale: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
    backend: str = "native",
) -> torch.Tensor:
    """Global attention over every spatial token. Q also gets sqrt(32). There is no prior."""
    batch, channels, height, width = y.shape
    heads = channels // 32
    tokens = y.flatten(2).transpose(1, 2)
    query, key, value = _as_heads(half_ste(apply_linear(qkv_weight, e4m3_ste(tokens))), heads)
    query = half_ste(cosine_half(query) * 5.65625)
    query = e4m3_ste(query * half_ste(temperature).view(1, 1, heads, 1))
    key, value = e4m3_ste(cosine_half(key)), e4m3_ste(value)
    # [B, N, heads, 32] -> [B, heads, N, 32]
    query = query.permute(0, 2, 1, 3)
    key = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)
    if backend == "native":
        count = height * width
        padding = (-count) % 64
        key = F.pad(key, (0, 0, 0, padding))
        value = F.pad(value, (0, 0, 0, padding))
        scores = half_ste(_matmul(query, key.transpose(-1, -2), compute_dtype))
        exponential = exp_weight(scores, vit=True)
        block_totals = sum64_half(exponential.reshape(batch, heads, count, -1, 64))
        total = block_totals[..., 0, :]
        for index in range(1, block_totals.shape[-2]):
            total = half_ste(total + block_totals[..., index, :])
        if padding:
            correction = half_ste(exp_weight(scores.new_zeros(()), vit=True) * padding)
            total = half_ste(total - correction)
        accumulated = half_ste(_matmul(e4m3_ste(exponential), value, compute_dtype))
        mixed = e4m3_ste(accumulated * half_ste(total.reciprocal()))
    else:
        from musubi_tuner.dlssnr.attention import attention

        mixed = e4m3_ste(attention(query, key, value, backend=backend, compute_dtype=compute_dtype))
    mixed = mixed.permute(0, 2, 1, 3).reshape(batch, height * width, channels)
    projected = apply_linear(proj_weight, mixed).transpose(1, 2).reshape(batch, channels, height, width)
    return e4m3_ste(projected + half_ste(e4m3_ste(y) * half_ste(skip_scale).view(1, -1, 1, 1)))
