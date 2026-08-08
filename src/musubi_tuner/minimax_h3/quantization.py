# Portions adapted from ai-toolkit (Ostris, LLC) at commit
# f4e91305471a3727d52886ef6d410eb570cd484f.
#
# MIT License
# Copyright (c) 2024 Ostris, LLC
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ComfyUI quantized checkpoint support used by MiniMax-H3.

The released compact artifacts store a ConvRot INT8 transformer and an
NVFP4/AWQ Qwen3-VL text encoder.  This module keeps those representations
quantized in memory and only reconstructs the arithmetic required by each
layer's forward pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from musubi_tuner.modules.convrot_int8_utils import ConvRotInt8LinearFn
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, load_safetensors

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


@dataclass(frozen=True)
class SafetensorsTensorSpec:
    shape: tuple[int, ...]
    dtype: torch.dtype


def inspect_safetensors_tensors(
    files: Iterable[str | Path],
    *,
    key_prefixes: tuple[str, ...] = (),
    key_transform: Callable[[str], str] | None = None,
    disable_mmap: bool = False,
) -> dict[str, SafetensorsTensorSpec]:
    """Read names, shapes, and dtypes without materializing checkpoint weights."""

    specs: dict[str, SafetensorsTensorSpec] = {}
    for file in files:
        path = Path(file).resolve()
        with MemoryEfficientSafeOpen(str(path), disable_numpy_memmap=disable_mmap) as handle:
            for raw_key in handle.keys():
                key = _normalize_key(raw_key, key_prefixes, key_transform)
                if key in specs:
                    raise ValueError(f"Duplicate MiniMax-H3 checkpoint key {key!r} in {path}")
                header = handle.header[raw_key]
                tensor_dtype = handle._get_torch_dtype(header["dtype"])
                if tensor_dtype is None:
                    raise ValueError(f"Unsupported safetensors dtype {header['dtype']!r} for {raw_key!r} in {path}")
                specs[key] = SafetensorsTensorSpec(tuple(header["shape"]), tensor_dtype)
    return specs


if _HAS_TRITON:

    @triton.jit
    def _nvfp4_dequant_kernel(q_ptr, scale_ptr, pts_ptr, output_ptr, columns, BLOCK_BYTES: tl.constexpr):
        row = tl.program_id(0)
        block = tl.program_id(1)
        per_row_bytes = columns // 2
        byte_offsets = block * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        byte_mask = byte_offsets < per_row_bytes
        packed = tl.load(q_ptr + row * per_row_bytes + byte_offsets, mask=byte_mask, other=0)
        codes = tl.interleave(packed & 15, packed >> 4)
        magnitude_code = (codes & 7).to(tl.float32)
        magnitude = tl.where(
            magnitude_code < 2,
            magnitude_code * 0.5,
            tl.exp2(tl.floor(magnitude_code / 2) - 1) * (1 + (magnitude_code % 2) * 0.5),
        )
        decoded = tl.where((codes & 8) != 0, -magnitude, magnitude)
        scale_count: tl.constexpr = (2 * BLOCK_BYTES) // 16
        scale_offsets = block * scale_count + tl.arange(0, scale_count)
        scales = tl.load(
            scale_ptr + row * (columns // 16) + scale_offsets,
            mask=scale_offsets < columns // 16,
            other=0.0,
        )
        decoded = tl.reshape(decoded, (scale_count, 16)) * (scales.to(tl.float32) * tl.load(pts_ptr))[:, None]
        decoded = tl.reshape(decoded, (2 * BLOCK_BYTES,))
        output_offsets = block * (2 * BLOCK_BYTES) + tl.arange(0, 2 * BLOCK_BYTES)
        tl.store(
            output_ptr + row * columns + output_offsets,
            decoded.to(output_ptr.dtype.element_ty),
            mask=output_offsets < columns,
        )


@dataclass(frozen=True)
class ComfyQuantLayerSpec:
    config: Mapping[str, object]
    tensors: Mapping[str, SafetensorsTensorSpec]


def parse_comfy_quant_blob(blob: torch.Tensor) -> dict[str, object]:
    try:
        value = json.loads(bytes(blob.detach().cpu().reshape(-1).tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError("Invalid ComfyUI quantization marker") from error
    if not isinstance(value, dict):
        raise ValueError("ComfyUI quantization marker must contain a JSON object")
    return value


def _normalize_key(key: str, prefixes: tuple[str, ...], transform: Callable[[str], str] | None) -> str:
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break
    return transform(key) if transform is not None else key


def inspect_comfy_quantized_layers(
    files: list[str | Path],
    tensor_specs: Mapping[str, SafetensorsTensorSpec],
    *,
    key_prefixes: tuple[str, ...] = (),
    key_transform: Callable[[str], str] | None = None,
    disable_mmap: bool = False,
) -> dict[str, ComfyQuantLayerSpec]:
    """Load only the small JSON markers and describe their companion tensors."""

    markers: dict[str, dict[str, object]] = {}
    for file in files:
        path = Path(file).resolve()
        with MemoryEfficientSafeOpen(str(path), disable_numpy_memmap=disable_mmap) as handle:
            for raw_key in handle.keys():
                if not raw_key.endswith(".comfy_quant"):
                    continue
                key = _normalize_key(raw_key, key_prefixes, key_transform)
                prefix = key[: -len(".comfy_quant")]
                if prefix in markers:
                    raise ValueError(f"Duplicate ComfyUI quantization marker for {prefix!r}")
                markers[prefix] = parse_comfy_quant_blob(handle.get_tensor(raw_key, device=torch.device("cpu")))

    result: dict[str, ComfyQuantLayerSpec] = {}
    companion_suffixes = (
        "weight",
        "weight_scale",
        "weight_scale_2",
        "pre_quant_scale",
        "input_scale",
        "comfy_quant",
    )
    for prefix, config in markers.items():
        companions = {
            suffix: tensor_specs[f"{prefix}.{suffix}"]
            for suffix in companion_suffixes
            if f"{prefix}.{suffix}" in tensor_specs
        }
        fmt = config.get("format")
        required = {"weight", "weight_scale", "comfy_quant"}
        if fmt == "nvfp4":
            required.add("weight_scale_2")
        if missing := sorted(required - companions.keys()):
            raise ValueError(f"Quantized layer {prefix!r} is missing companion tensors: {missing}")
        result[prefix] = ComfyQuantLayerSpec(config=config, tensors=companions)
    return result


def _meta_tensor(spec: SafetensorsTensorSpec) -> torch.Tensor:
    return torch.empty(spec.shape, dtype=spec.dtype, device="meta")


def _register_checkpoint_buffers(
    module: nn.Module,
    specs: Mapping[str, SafetensorsTensorSpec],
    *,
    skip: frozenset[str] = frozenset(),
) -> None:
    for name, spec in specs.items():
        if name not in skip:
            module.register_buffer(name, _meta_tensor(spec), persistent=True)


def _convrot_forward(module: nn.Linear, hidden_states: torch.Tensor) -> torch.Tensor:
    weight_scale = module.weight_scale
    if getattr(module, "_h3_weight_scale_byte_view", False):
        weight_scale = weight_scale.view(torch.float32)
    return ConvRotInt8LinearFn.apply(
        hidden_states,
        module.weight,
        weight_scale,
        module.bias,
        module._convrot_groupsize,
        module._convrot_bwd_mode,
    )


class H3Int8Embedding(nn.Module):
    """Per-row symmetric INT8 embedding without a full-table dequantization."""

    def __init__(self, original: nn.Embedding, specs: Mapping[str, SafetensorsTensorSpec], output_dtype: torch.dtype) -> None:
        super().__init__()
        if specs["weight"].shape != (original.num_embeddings, original.embedding_dim):
            raise ValueError("INT8 embedding weight shape does not match the target module")
        if math.prod(specs["weight_scale"].shape) != original.num_embeddings:
            raise ValueError("INT8 embedding must contain one scale per row")
        self.num_embeddings = original.num_embeddings
        self.embedding_dim = original.embedding_dim
        self.padding_idx = original.padding_idx
        self.output_dtype = output_dtype
        self._h3_weight_scale_byte_view = False
        _register_checkpoint_buffers(self, specs)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        flat = input_ids.reshape(-1).to(self.weight.device)
        rows = self.weight.index_select(0, flat).float()
        scales = self.weight_scale
        if self._h3_weight_scale_byte_view:
            scales = scales.view(torch.float32)
        scales = scales.float().reshape(-1).index_select(0, flat)
        output = (rows * scales.unsqueeze(1)).to(self.output_dtype)
        return output.to(input_ids.device).reshape(*input_ids.shape, self.embedding_dim)


def swap_nvfp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    """Convert ComfyUI's high-nibble-first pairs to low-nibble-first pairs."""

    return ((packed << 4) | (packed >> 4)).contiguous()


def unswizzle_nvfp4_scales(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Undo the cuBLAS 128x4 blocked scale layout used by comfy-kitchen."""

    row_blocks = (rows + 127) // 128
    col_blocks = (cols + 3) // 4
    padded_rows = row_blocks * 128
    padded_cols = col_blocks * 4
    value = scales.reshape(-1, 32, 16)
    value = value.reshape(-1, 32, 4, 4).transpose(1, 2)
    value = value.reshape(row_blocks, col_blocks, 4, 32, 4)
    value = value.reshape(row_blocks, col_blocks, 128, 4)
    value = value.permute(0, 2, 1, 3).reshape(padded_rows, padded_cols)
    return value[:rows, :cols].contiguous()


def dequantize_nvfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    *,
    rows: int,
    columns: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Decode packed E2M1 values with E4M3 block-16 scales."""

    if _HAS_TRITON and packed.is_cuda and torch.cuda.get_device_capability(packed.device) >= (8, 9):
        output = torch.empty(rows, columns, device=packed.device, dtype=dtype)
        block_bytes = 1024
        _nvfp4_dequant_kernel[(rows, triton.cdiv(columns // 2, block_bytes))](
            packed.contiguous(),
            scales.contiguous(),
            per_tensor_scale.contiguous(),
            output,
            columns,
            BLOCK_BYTES=block_bytes,
            num_warps=4,
        )
        return output

    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, columns)
    values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=packed.device)
    magnitude = values.index_select(0, (codes & 7).reshape(-1).to(torch.int64)).reshape(rows, columns)
    decoded = magnitude * torch.where((codes & 8) != 0, -1.0, 1.0)
    decoded = decoded.reshape(rows, columns // 16, 16)
    decoded = decoded * (scales.float() * per_tensor_scale.float().reshape(())).unsqueeze(-1)
    return decoded.reshape(rows, columns).to(dtype)


class H3Nvfp4Linear(nn.Module):
    """NVFP4 weight-only Linear with optional ModelOpt AWQ input scaling."""

    def __init__(self, original: nn.Linear, specs: Mapping[str, SafetensorsTensorSpec], output_dtype: torch.dtype) -> None:
        super().__init__()
        if original.in_features % 16:
            raise ValueError("NVFP4 Linear in_features must be divisible by 16")
        if specs["weight"].shape != (original.out_features, original.in_features // 2):
            raise ValueError("NVFP4 packed weight shape does not match the target Linear")
        if math.prod(specs["weight_scale_2"].shape) != 1:
            raise ValueError("NVFP4 weight_scale_2 must be scalar")
        if "pre_quant_scale" in specs and math.prod(specs["pre_quant_scale"].shape) != original.in_features:
            raise ValueError("NVFP4 AWQ pre_quant_scale width does not match the target Linear")
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.output_dtype = output_dtype
        self._h3_nvfp4_normalized = False
        if original.bias is None:
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(
                torch.empty(original.bias.shape, dtype=original.bias.dtype, device="meta"),
                requires_grad=original.bias.requires_grad,
            )
        _register_checkpoint_buffers(self, specs)

    @torch.no_grad()
    def normalize_checkpoint_storage_(self) -> None:
        if self._h3_nvfp4_normalized:
            return
        self.weight = swap_nvfp4_nibbles(self.weight)
        self.weight_scale = (
            unswizzle_nvfp4_scales(
                self.weight_scale.view(torch.float8_e4m3fn),
                self.out_features,
                self.in_features // 16,
            )
            .view(torch.uint8)
            .contiguous()
        )
        self.weight_scale_2 = self.weight_scale_2.detach().float().reshape(1).contiguous().view(torch.uint8)
        if hasattr(self, "pre_quant_scale"):
            self.pre_quant_scale = self.pre_quant_scale.detach().float().reshape(-1).contiguous().view(torch.uint8)
        self._h3_nvfp4_normalized = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self._h3_nvfp4_normalized:
            raise RuntimeError("NVFP4 checkpoint storage was not normalized after loading")
        pre_scale = getattr(self, "pre_quant_scale", None)
        if pre_scale is not None:
            hidden_states = hidden_states * pre_scale.view(torch.float32).reshape(-1).to(hidden_states.dtype)
        with torch.no_grad():
            weight = dequantize_nvfp4(
                self.weight,
                self.weight_scale.view(torch.float8_e4m3fn),
                self.weight_scale_2.view(torch.float32),
                rows=self.out_features,
                columns=self.in_features,
                dtype=hidden_states.dtype,
            )
        return F.linear(hidden_states, weight, self.bias)


def prepare_comfy_quantized_layers(
    root: nn.Module,
    layer_specs: Mapping[str, ComfyQuantLayerSpec],
    *,
    output_dtype: torch.dtype,
    convrot_bwd_mode: str = "h3_bf16",
) -> None:
    """Replace or patch modules so the streaming loader can assign raw tensors."""

    for path, layer_spec in layer_specs.items():
        module = root.get_submodule(path)
        fmt = layer_spec.config.get("format")
        if isinstance(module, nn.Embedding):
            if fmt != "int8_tensorwise":
                raise ValueError(f"Unsupported quantization format {fmt!r} for embedding {path!r}")
            replacement: nn.Module = H3Int8Embedding(module, layer_spec.tensors, output_dtype)
        elif isinstance(module, nn.Linear) and fmt == "nvfp4":
            replacement = H3Nvfp4Linear(module, layer_spec.tensors, output_dtype)
        elif isinstance(module, nn.Linear) and fmt == "int8_tensorwise":
            if layer_spec.tensors["weight"].dtype != torch.int8:
                raise ValueError(f"INT8 layer {path!r} has {layer_spec.tensors['weight'].dtype} weight storage")
            module.weight.requires_grad_(False)
            if math.prod(layer_spec.tensors["weight_scale"].shape) != module.out_features:
                raise ValueError(f"INT8 layer {path!r} must contain one weight scale per output row")
            _register_checkpoint_buffers(module, layer_spec.tensors, skip=frozenset({"weight"}))
            module._convrot_groupsize = (
                int(layer_spec.config.get("convrot_groupsize", 256)) if layer_spec.config.get("convrot") else 1
            )
            group_size = module._convrot_groupsize
            reduced = group_size
            while reduced > 1 and reduced % 4 == 0:
                reduced //= 4
            if group_size > 1 and (group_size < 4 or reduced != 1 or module.in_features % group_size):
                raise ValueError(
                    f"INT8 ConvRot layer {path!r} has invalid group size {group_size} for in_features={module.in_features}"
                )
            module._convrot_bwd_mode = convrot_bwd_mode
            module._h3_quant_format = "int8_convrot" if module._convrot_groupsize > 1 else "int8_tensorwise"
            module._h3_weight_scale_byte_view = False

            def forward(self, value):
                return _convrot_forward(self, value)

            module.forward = forward.__get__(module, type(module))
            replacement = module
        else:
            raise ValueError(
                f"ComfyUI quantization marker {path!r} points at {type(module).__name__} with unsupported format {fmt!r}"
            )

        if replacement is not module:
            parent_path, _, name = path.rpartition(".")
            parent = root.get_submodule(parent_path) if parent_path else root
            setattr(parent, name, replacement)


def uses_generic_comfy_loader(layer_specs: Mapping[str, ComfyQuantLayerSpec]) -> bool:
    """Return whether an artifact contains formats outside the official ConvRot path."""

    return any(spec.config.get("format") != "int8_tensorwise" or not spec.config.get("convrot") for spec in layer_specs.values())


def load_comfy_quantized_model(
    factory: Callable[[], nn.Module],
    files: Iterable[str | Path],
    layer_specs: Mapping[str, ComfyQuantLayerSpec],
    *,
    device: str | torch.device,
    output_dtype: torch.dtype,
    key_prefixes: tuple[str, ...] = (),
    key_transform: Callable[[str], str] | None = None,
    cast_unquantized_to_output_dtype: bool = False,
    disable_mmap: bool = False,
) -> nn.Module:
    """Stream a mixed ComfyUI INT8/NVFP4 artifact into a prepared meta model.

    The official MiniMax-H3 loader remains authoritative for ordinary BF16 and
    pure ConvRot checkpoints. This loader is limited to mixed tensorwise INT8 or
    NVFP4/AWQ artifacts that the official ConvRot importer cannot represent.
    """

    from accelerate import init_empty_weights

    files = [Path(path) for path in files]
    if not files:
        raise ValueError("MiniMax-H3 quantized checkpoint file list is empty")

    with init_empty_weights():
        model = factory()
        model.requires_grad_(False)
        prepare_comfy_quantized_layers(model, layer_specs, output_dtype=output_dtype)

    expected_state = model.state_dict()
    expected_keys = set(expected_state)
    checkpoint_specs = {
        f"{module_path}.{suffix}": tensor_spec
        for module_path, layer_spec in layer_specs.items()
        for suffix, tensor_spec in layer_spec.tensors.items()
    }
    loaded_keys: set[str] = set()
    unexpected: set[str] = set()
    shape_mismatches: list[str] = []
    dtype_mismatches: list[str] = []
    target_device = torch.device(device)

    for file in files:
        raw_state = load_safetensors(
            str(file),
            device=target_device,
            disable_mmap=True,
            disable_numpy_memmap=disable_mmap,
        )
        shard: dict[str, torch.Tensor] = {}
        for raw_key, tensor in raw_state.items():
            key = _normalize_key(raw_key, key_prefixes, key_transform)
            if key in loaded_keys:
                raise ValueError(f"Duplicate MiniMax-H3 checkpoint key {key!r} in {file}")
            loaded_keys.add(key)
            if key not in expected_keys:
                unexpected.add(key)
                continue
            expected = expected_state[key]
            if tensor.shape != expected.shape:
                shape_mismatches.append(f"{key}: expected {tuple(expected.shape)}, got {tuple(tensor.shape)}")
                continue
            checkpoint_spec = checkpoint_specs.get(key)
            target_dtype = checkpoint_spec.dtype if checkpoint_spec is not None else expected.dtype
            if checkpoint_spec is None and cast_unquantized_to_output_dtype and tensor.is_floating_point():
                target_dtype = output_dtype
            elif checkpoint_spec is None and tensor.is_floating_point():
                # Mixed DiT artifacts carry both compute weights and deliberate
                # FP32 islands; preserve the checkpoint's published dtype.
                target_dtype = tensor.dtype
            if tensor.dtype != target_dtype:
                half_dtypes = {torch.float16, torch.bfloat16}
                if tensor.dtype in half_dtypes and target_dtype in half_dtypes:
                    tensor = tensor.to(dtype=target_dtype)
                else:
                    dtype_mismatches.append(f"{key}: expected {target_dtype}, got {tensor.dtype}")
                    continue
            shard[key] = tensor
        del raw_state
        if shard:
            model.load_state_dict(shard, strict=False, assign=True)
        del shard

    missing = expected_keys - loaded_keys
    if missing or unexpected or shape_mismatches or dtype_mismatches:
        raise ValueError(
            "MiniMax-H3 checkpoint key mismatch: "
            f"missing={sorted(missing)[:20]}, unexpected={sorted(unexpected)[:20]}, "
            f"shape_mismatches={sorted(shape_mismatches)[:20]}, dtype_mismatches={sorted(dtype_mismatches)[:20]}"
        )

    model.requires_grad_(False)
    model.to(target_device)
    finalize_comfy_quantized_layers(model)
    model.checkpoint_quantization = quantization_summary(layer_specs)
    model.eval()
    return model


@torch.no_grad()
def finalize_comfy_quantized_layers(root: nn.Module) -> None:
    for module in root.modules():
        if isinstance(module, H3Nvfp4Linear):
            module.normalize_checkpoint_storage_()
        elif isinstance(module, H3Int8Embedding):
            if not module._h3_weight_scale_byte_view:
                module.weight_scale = module.weight_scale.detach().float().reshape(-1).contiguous().view(torch.uint8)
                module._h3_weight_scale_byte_view = True
        elif getattr(module, "_h3_quant_format", None) in {"int8_convrot", "int8_tensorwise"}:
            if not module._h3_weight_scale_byte_view:
                module.weight_scale = module.weight_scale.detach().float().reshape(-1).contiguous().view(torch.uint8)
                module._h3_weight_scale_byte_view = True


def quantization_summary(layer_specs: Mapping[str, ComfyQuantLayerSpec]) -> str:
    formats = set()
    for spec in layer_specs.values():
        fmt = str(spec.config.get("format"))
        if fmt == "int8_tensorwise" and spec.config.get("convrot"):
            fmt = "int8_convrot"
        elif fmt == "nvfp4" and "pre_quant_scale" in spec.tensors:
            fmt = "nvfp4_awq"
        formats.add(fmt)
    return ",".join(sorted(formats)) if formats else "none"
