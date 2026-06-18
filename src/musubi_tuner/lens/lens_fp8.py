from __future__ import annotations

from types import MethodType

import torch
import torch.nn.functional as F


def require_lens_fp8_dtype(dtype: torch.dtype | None = None) -> torch.dtype:
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("Lens fp8 text encoder requires torch.float8_e4m3fn support.")
    if dtype is not None and dtype != fp8_dtype:
        raise ValueError(f"Lens fp8 text encoder supports only torch.float8_e4m3fn, got {dtype}.")
    return fp8_dtype


def apply_lens_fp8_storage(model: torch.nn.Module, dtype: torch.dtype | None = None) -> dict[str, int]:
    """Store large Lens GPT-OSS weights as FP8 while computing in activation dtype."""
    fp8_dtype = require_lens_fp8_dtype(dtype)
    converted = {"linear": 0, "experts": 0}

    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            _cast_parameter(module, "weight", fp8_dtype)
            module.forward = MethodType(_fp8_linear_forward, module)
            converted["linear"] += 1
        elif module.__class__.__name__ == "GptOssExperts":
            _cast_parameter(module, "gate_up_proj", fp8_dtype)
            _cast_parameter(module, "down_proj", fp8_dtype)
            module.forward = MethodType(_fp8_gpt_oss_experts_forward, module)
            converted["experts"] += 1

    return converted


def _cast_parameter(module: torch.nn.Module, name: str, dtype: torch.dtype) -> None:
    param = getattr(module, name, None)
    if not isinstance(param, torch.nn.Parameter):
        return
    if param.dtype == dtype:
        return
    param.data = param.detach().to(dtype)


def _fp8_linear_forward(self: torch.nn.Linear, input: torch.Tensor) -> torch.Tensor:
    weight = self.weight.to(input.dtype)
    bias = self.bias.to(input.dtype) if self.bias is not None else None
    return F.linear(input, weight, bias)


def _fp8_gpt_oss_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    router_indices: torch.Tensor | None = None,
    routing_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if router_indices is None or routing_weights is None:
        raise ValueError("GptOssExperts fp8 forward requires router_indices and routing_weights.")

    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)
    num_experts = routing_weights.shape[1]
    next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)

    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=num_experts + 1)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx_tensor in expert_hit:
        expert_idx = int(expert_idx_tensor[0].item())
        if expert_idx == num_experts:
            continue

        with torch.no_grad():
            _, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate_up = current_state @ self.gate_up_proj[expert_idx].to(hidden_states.dtype)
        gate_up = gate_up + self.gate_up_proj_bias[expert_idx].to(hidden_states.dtype)
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        out = ((up + 1) * glu) @ self.down_proj[expert_idx].to(hidden_states.dtype)
        out = out + self.down_proj_bias[expert_idx].to(hidden_states.dtype)
        weighted_output = out * routing_weights[token_idx, expert_idx, None].to(hidden_states.dtype)
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states.view(batch_size, -1, self.hidden_size)
