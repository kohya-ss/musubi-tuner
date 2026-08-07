# SPDX-License-Identifier: GPL-3.0-only
#
# Minimal MiniMax-H3 text-encoder loader for ComfyUI-style quantized
# checkpoints. This file is isolated under GPL-3.0-only because it is derived
# from ComfyUI quantized loading concepts and small NVFP4/INT8 decoding
# routines from comfy-kitchen. The rest of the MiniMax-H3 implementation does
# not depend on ComfyUI at runtime.

from __future__ import annotations

from collections.abc import Callable, Iterable
import json
import math
from pathlib import Path

from accelerate import init_empty_weights
from safetensors import safe_open
import torch
import torch.nn.functional as F


QUANT_EXTRA_SUFFIXES = (".comfy_quant", ".weight_scale", ".weight_scale_2", ".input_scale", ".pre_quant_scale")

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def checkpoint_has_comfy_quant(files: Iterable[str | Path]) -> bool:
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if any(key.endswith(".comfy_quant") for key in handle.keys()):
                return True
    return False


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    return step5.reshape(padded_rows, padded_cols)[:num_rows, :num_cols]


def _dequantize_nvfp4(
    qdata: torch.Tensor,
    tensor_scale: torch.Tensor,
    block_scales: torch.Tensor,
    *,
    orig_shape: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    logical_rows, logical_cols = qdata.shape[0], qdata.shape[1] * 2
    lut = E2M1_LUT.to(device=qdata.device, dtype=dtype)
    lo = qdata & 0x0F
    hi = qdata >> 4
    unpacked = torch.stack([hi, lo], dim=-1).view(logical_rows, logical_cols)
    values = lut[unpacked.long()]
    block_size = 16
    values = values.reshape(logical_rows, -1, block_size)
    num_blocks_per_row = logical_cols // block_size
    scales = _from_blocked(block_scales, num_rows=logical_rows, num_cols=num_blocks_per_row)
    total_scale = tensor_scale.to(device=qdata.device, dtype=dtype) * scales.to(dtype)
    dequantized = (values * total_scale.unsqueeze(-1)).reshape(logical_rows, logical_cols)
    rows, cols = orig_shape
    return dequantized[:rows, :cols].contiguous()


def _dequantize_int8(qdata: torch.Tensor, scale: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return qdata.to(dtype=dtype) * scale.to(device=qdata.device, dtype=dtype)


def _build_hadamard(size: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"MiniMax-H3 ConvRot group size must be a power of 4, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=dtype,
        device=device,
    )
    h = h4
    current_size = 4
    while current_size < size:
        h = torch.kron(h, h4)
        current_size *= 4
    h = h / (size**0.5)
    _HADAMARD_CACHE[cache_key] = h
    return h


def _apply_convrot_inverse(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"MiniMax-H3 ConvRot tensor width {cols} is not divisible by group size {group_size}")
    grouped = weight.reshape(rows, cols // group_size, group_size)
    hadamard = _build_hadamard(group_size, weight.device, weight.dtype)
    restored = torch.matmul(grouped, hadamard.T)
    return restored.reshape(rows, cols).contiguous()


def interpolate_adaln_curve(table: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    table = table.to(dtype=torch.float32)
    pos = timesteps.to(dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
    i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
    return torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1).to(dtype=table.dtype))


class ComfyQuantLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, *, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype or torch.bfloat16
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=self.compute_dtype))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, device=device, dtype=self.compute_dtype))
        else:
            self.register_parameter("bias", None)
        self.quant_format: str | None = None
        self.input_scale = None
        self.pre_quant_scale = None

    def _dequant_weight(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.quant_format == "nvfp4":
            return _dequantize_nvfp4(
                self.weight_qdata.to(device=device),
                self.weight_scale_2.to(device=device),
                self.weight_scale.to(device=device),
                orig_shape=(self.out_features, self.in_features),
                dtype=dtype,
            )
        if self.quant_format == "int8_tensorwise":
            weight = _dequantize_int8(self.weight_qdata.to(device=device), self.weight_scale.to(device=device), dtype=dtype)
            if getattr(self, "convrot", False):
                weight = _apply_convrot_inverse(weight, self.convrot_groupsize)
            return weight
        return self.weight.to(device=device, dtype=dtype)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.pre_quant_scale is not None:
            input = input * self.pre_quant_scale.to(device=input.device, dtype=input.dtype)
        weight = self._dequant_weight(input.device, input.dtype)
        bias = None if self.bias is None else self.bias.to(device=input.device, dtype=input.dtype)
        return F.linear(input, weight, bias)


class ComfyQuantEmbedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx=None,
        max_norm=None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        *,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.compute_dtype = dtype or torch.bfloat16
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=self.compute_dtype))
        self.quant_format: str | None = None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.quant_format == "int8_tensorwise":
            qrows = self.weight_qdata.to(device=input.device).index_select(0, input.reshape(-1))
            scales = self.weight_scale.to(device=input.device).index_select(0, input.reshape(-1))
            output = _dequantize_int8(qrows, scales, dtype=self.compute_dtype)
            if getattr(self, "convrot", False):
                output = _apply_convrot_inverse(output, self.convrot_groupsize)
            output = output.reshape(*input.shape, self.embedding_dim)
            return output
        return F.embedding(
            input,
            self.weight.to(input.device),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


def replace_linears_and_embeddings(module: torch.nn.Module, *, device: torch.device, dtype: torch.dtype) -> None:
    for name, child in list(module.named_children()):
        replacement = None
        if isinstance(child, torch.nn.Linear):
            replacement = ComfyQuantLinear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=device,
                dtype=dtype,
            )
        elif isinstance(child, torch.nn.Embedding):
            replacement = ComfyQuantEmbedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                device=device,
                dtype=dtype,
            )
        if replacement is None:
            replace_linears_and_embeddings(child, device=device, dtype=dtype)
        elif isinstance(module, torch.nn.Sequential):
            module[int(name)] = replacement
        else:
            setattr(module, name, replacement)


def _expected_keys(model: torch.nn.Module) -> set[str]:
    expected = set()
    for name, module in model.named_modules():
        if isinstance(module, (ComfyQuantLinear, ComfyQuantEmbedding)):
            expected.add(f"{name}.weight")
            if isinstance(module, ComfyQuantLinear) and module.bias is not None:
                expected.add(f"{name}.bias")
    expected.update(model.state_dict().keys())
    return expected


def load_quantized_text_encoder(
    factory: Callable[[], torch.nn.Module],
    files: Iterable[str | Path],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    key_transform: Callable[[str], str],
    materialize_quantized: bool = False,
) -> torch.nn.Module:
    target_device = torch.device(device)
    with init_empty_weights():
        model = factory()
        replace_linears_and_embeddings(model, device=target_device, dtype=dtype)
    modules = dict(model.named_modules())
    expected = _expected_keys(model)
    seen = set()
    unexpected = set()
    shard = {}

    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            raw_keys = set(handle.keys())
            for raw_key in raw_keys:
                key = key_transform(raw_key)
                if key.endswith(QUANT_EXTRA_SUFFIXES):
                    continue
                module_name, _, param_name = key.rpartition(".")
                module = modules.get(module_name)
                if isinstance(module, (ComfyQuantLinear, ComfyQuantEmbedding)) and param_name == "weight":
                    _load_module_weight(module, handle, raw_key, raw_keys, dtype, target_device)
                    seen.add(key)
                    continue
                if isinstance(module, ComfyQuantLinear) and param_name == "bias":
                    module.bias = torch.nn.Parameter(handle.get_tensor(raw_key).to(device=target_device, dtype=dtype), requires_grad=False)
                    seen.add(key)
                    continue
                if key not in expected:
                    unexpected.add(key)
                    continue
                tensor = handle.get_tensor(raw_key)
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                shard[key] = tensor
                seen.add(key)
    if shard:
        model.load_state_dict(shard, strict=False, assign=True)
    missing = expected - seen
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 Comfy quantized text encoder key mismatch: "
            f"missing={sorted(missing)[:20]}, unexpected={sorted(unexpected)[:20]}"
        )
    model.to(device=target_device)
    if materialize_quantized:
        materialize_quantized_modules(model, dtype=dtype, device=target_device)
    model.eval()
    return model


load_quantized_module = load_quantized_text_encoder


def materialize_quantized_modules(module: torch.nn.Module, *, dtype: torch.dtype, device: torch.device | str) -> None:
    target_device = torch.device(device)
    for name, child in list(module.named_children()):
        if isinstance(child, ComfyQuantLinear):
            replacement = torch.nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=target_device,
                dtype=dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(child._dequant_weight(target_device, dtype))
                if child.bias is not None:
                    replacement.bias.copy_(child.bias.to(device=target_device, dtype=dtype))
            replacement.requires_grad_(False)
            setattr(module, name, replacement)
        elif isinstance(child, ComfyQuantEmbedding):
            replacement = torch.nn.Embedding(
                child.num_embeddings,
                child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                device=target_device,
                dtype=dtype,
            )
            with torch.no_grad():
                if child.quant_format == "int8_tensorwise":
                    weight = _dequantize_int8(
                        child.weight_qdata.to(device=target_device),
                        child.weight_scale.to(device=target_device),
                        dtype=dtype,
                    )
                    if getattr(child, "convrot", False):
                        weight = _apply_convrot_inverse(weight, child.convrot_groupsize)
                else:
                    weight = child.weight.to(device=target_device, dtype=dtype)
                replacement.weight.copy_(weight)
            replacement.requires_grad_(False)
            setattr(module, name, replacement)
        else:
            materialize_quantized_modules(child, dtype=dtype, device=target_device)


def _set_quant_buffer(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    if name in module._parameters:
        module.register_parameter(name, None)
    if name in module._buffers:
        module._buffers[name] = value
    else:
        module.register_buffer(name, value, persistent=False)


def _load_module_weight(
    module: torch.nn.Module,
    handle,
    raw_weight_key: str,
    raw_keys: set[str],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    raw_prefix = raw_weight_key.rsplit(".", 1)[0] + "."
    conf_key = raw_prefix + "comfy_quant"
    layer_conf = json.loads(handle.get_tensor(conf_key).numpy().tobytes()) if conf_key in raw_keys else None
    weight = handle.get_tensor(raw_weight_key)
    if layer_conf is None:
        module.weight = torch.nn.Parameter(weight.to(device=device, dtype=dtype), requires_grad=False)
        return
    module.quant_format = layer_conf.get("format")
    module.register_parameter("weight", None)
    _set_quant_buffer(module, "weight_qdata", weight.to(device=device))
    if module.quant_format == "nvfp4":
        _set_quant_buffer(module, "weight_scale", handle.get_tensor(raw_prefix + "weight_scale").to(device=device))
        _set_quant_buffer(module, "weight_scale_2", handle.get_tensor(raw_prefix + "weight_scale_2").to(device=device))
    elif module.quant_format == "int8_tensorwise":
        _set_quant_buffer(module, "weight_scale", handle.get_tensor(raw_prefix + "weight_scale").to(device=device))
        params_conf = layer_conf.get("params", {})
        if not isinstance(params_conf, dict):
            params_conf = {}
        if layer_conf.get("convrot", params_conf.get("convrot", False)):
            module.convrot = True
            module.convrot_groupsize = int(layer_conf.get("convrot_groupsize", params_conf.get("convrot_groupsize", 256)))
    else:
        raise ValueError(f"Unsupported MiniMax-H3 Comfy quantization format: {module.quant_format}")
    pre_quant_key = raw_prefix + "pre_quant_scale"
    if pre_quant_key in raw_keys:
        module.pre_quant_scale = handle.get_tensor(pre_quant_key).to(device=device)
