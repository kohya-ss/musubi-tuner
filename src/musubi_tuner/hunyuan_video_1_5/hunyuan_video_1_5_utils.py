# Original work: https://github.com/Tencent-Hunyuan/HunyuanImage-2.1
# Re-implemented for license compliance for sd-scripts.

import math
from typing import Tuple, Union
import torch

# region model


def _to_tuple(x, dim=2):
    """
    Convert int or sequence to tuple of specified dimension.

    Args:
        x: Int or sequence to convert.
        dim: Target dimension for tuple.

    Returns:
        Tuple of length dim.
    """
    if isinstance(x, int) or isinstance(x, float):
        return (x,) * dim
    elif len(x) == dim:
        return x
    else:
        raise ValueError(f"Expected length {dim} or int, but got {x}")


def get_meshgrid_nd(start, dim=2):
    """
    Generate n-dimensional coordinate meshgrid from 0 to grid_size.

    Creates coordinate grids for each spatial dimension, useful for
    generating position embeddings.

    Args:
        start: Grid size for each dimension (int or tuple).
        dim: Number of spatial dimensions.

    Returns:
        Coordinate grid tensor [dim, *grid_size].
    """
    # Convert start to grid sizes
    num = _to_tuple(start, dim=dim)
    start = (0,) * dim
    stop = num

    # Generate coordinate arrays for each dimension
    axis_grid = []
    for i in range(dim):
        a, b, n = start[i], stop[i], num[i]
        g = torch.linspace(a, b, n + 1, dtype=torch.float32)[:n]
        axis_grid.append(g)
    grid = torch.meshgrid(*axis_grid, indexing="ij")  # dim x [W, H, D]
    grid = torch.stack(grid, dim=0)  # [dim, W, H, D]

    return grid


def get_nd_rotary_pos_embed(rope_dim_list, start, theta=10000.0):
    """
    Generate n-dimensional rotary position embeddings for spatial tokens.

    Creates RoPE embeddings for multi-dimensional positional encoding,
    distributing head dimensions across spatial dimensions.

    Args:
        rope_dim_list: Dimensions allocated to each spatial axis (should sum to head_dim).
        start: Spatial grid size for each dimension.
        theta: Base frequency for RoPE computation.

    Returns:
        Tuple of (cos_freqs, sin_freqs) for rotary embedding [H*W, D/2].
    """

    grid = get_meshgrid_nd(start, dim=len(rope_dim_list))  # [3, W, H, D] / [2, W, H]

    # Generate RoPE embeddings for each spatial dimension
    embs = []
    for i in range(len(rope_dim_list)):
        emb = get_1d_rotary_pos_embed(rope_dim_list[i], grid[i].reshape(-1), theta)  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
    sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
    return cos, sin


def get_1d_rotary_pos_embed(
    dim: int, pos: Union[torch.FloatTensor, int], theta: float = 10000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate 1D rotary position embeddings.

    Args:
        dim: Embedding dimension (must be even).
        pos: Position indices [S] or scalar for sequence length.
        theta: Base frequency for sinusoidal encoding.

    Returns:
        Tuple of (cos_freqs, sin_freqs) tensors [S, D].
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))  # [D/2]
    freqs = torch.outer(pos, freqs)  # [S, D/2]
    freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
    return freqs_cos, freqs_sin


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings for diffusion models.

    Converts scalar timesteps to high-dimensional embeddings using
    sinusoidal encoding at different frequencies.

    Args:
        t: Timestep tensor [N].
        dim: Output embedding dimension.
        max_period: Maximum period for frequency computation.

    Returns:
        Timestep embeddings [N, dim].
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def modulate(x, shift=None, scale=None):
    """
    Apply adaptive layer normalization modulation.

    Applies scale and shift transformations for conditioning
    in adaptive layer normalization.

    Args:
        x: Input tensor to modulate.
        shift: Additive shift parameter (optional).
        scale: Multiplicative scale parameter (optional).

    Returns:
        Modulated tensor x * (1 + scale) + shift.
    """
    if scale is None and shift is None:
        return x
    elif shift is None:
        return x * (1 + scale.unsqueeze(1))
    elif scale is None:
        return x + shift.unsqueeze(1)
    else:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_gate(x, gate=None, tanh=False):
    """
    Apply gating mechanism to tensor.

    Multiplies input by gate values, optionally applying tanh activation.
    Used in residual connections for adaptive control.

    Args:
        x: Input tensor to gate.
        gate: Gating values (optional).
        tanh: Whether to apply tanh to gate values.

    Returns:
        Gated tensor x * gate (with optional tanh).
    """
    if gate is None:
        return x
    if tanh:
        return x * gate.unsqueeze(1).tanh()
    else:
        return x * gate.unsqueeze(1)


def reshape_for_broadcast(
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    x: torch.Tensor,
    head_first=False,
):
    """
    Reshape RoPE frequency tensors for broadcasting with attention tensors.

    Args:
        freqs_cis: Tuple of (cos_freqs, sin_freqs) tensors.
        x: Target tensor for broadcasting compatibility.
        head_first: Must be False (only supported layout).

    Returns:
        Reshaped (cos_freqs, sin_freqs) tensors ready for broadcasting.
    """
    assert not head_first, "Only head_first=False layout supported."
    assert isinstance(freqs_cis, tuple), "Expected tuple of (cos, sin) frequency tensors."
    assert x.ndim > 1, f"x should have at least 2 dimensions, but got {x.ndim}"

    # Validate frequency tensor dimensions match target tensor
    assert freqs_cis[0].shape == (
        x.shape[1],
        x.shape[-1],
    ), f"Frequency tensor shape {freqs_cis[0].shape} incompatible with target shape {x.shape}"

    shape = [d if i == 1 or i == x.ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)


def rotate_half(x):
    """
    Rotate half the dimensions for RoPE computation.

    Splits the last dimension in half and applies a 90-degree rotation
    by swapping and negating components.

    Args:
        x: Input tensor [..., D] where D is even.

    Returns:
        Rotated tensor with same shape as input.
    """
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: Tuple[torch.Tensor, torch.Tensor], head_first: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    Args:
        xq: Query tensor [B, S, H, D].
        xk: Key tensor [B, S, H, D].
        freqs_cis: Tuple of (cos_freqs, sin_freqs) for rotation.
        head_first: Whether head dimension precedes sequence dimension.

    Returns:
        Tuple of rotated (query, key) tensors.
    """
    device = xq.device
    dtype = xq.dtype

    cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)
    cos, sin = cos.to(device), sin.to(device)

    # Apply rotation: x' = x * cos + rotate_half(x) * sin
    xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).to(dtype)
    xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).to(dtype)

    return xq_out, xk_out


# endregion

# region inference


def get_timesteps_sigmas(sampling_steps: int, shift: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate timesteps and sigmas for diffusion sampling.

    Args:
        sampling_steps: Number of sampling steps.
        shift: Sigma shift parameter for schedule modification.
        device: Target device for tensors.

    Returns:
        Tuple of (timesteps, sigmas) tensors.
    """
    sigmas = torch.linspace(1, 0, sampling_steps + 1)
    sigmas = (shift * sigmas) / (1 + (shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    timesteps = (sigmas[:-1] * 1000).to(dtype=torch.float32, device=device)
    return timesteps, sigmas


def step(latents, noise_pred, sigmas, step_i):
    """
    Perform a single diffusion sampling step.

    Args:
        latents: Current latent state.
        noise_pred: Predicted noise.
        sigmas: Noise schedule sigmas.
        step_i: Current step index.

    Returns:
        Updated latents after the step.
    """
    return latents.float() - (sigmas[step_i] - sigmas[step_i + 1]) * noise_pred.float()


# endregion
