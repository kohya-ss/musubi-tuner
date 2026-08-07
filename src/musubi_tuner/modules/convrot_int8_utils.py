"""ConvRot INT8 optimization for frozen base weights during LoRA training.

Mirrors the structure of fp8_optimization_utils.py: quantize target Linear weights at
load time, store the INT8 weight (in the rotated basis) under the original ``.weight``
key and the per-channel scale under a sibling ``.scale_weight`` key, then monkey-patch
the matching ``nn.Linear`` modules' ``forward``.

Unlike the fp8 path, the forward goes through a custom ``torch.autograd.Function``:
the fused Triton kernel (rotation + dynamic row-wise INT8 quantization + INT8 GEMM
with dequantization epilogue) has no autograd support, and the base weight is frozen
so only grad_x is needed in backward: grad_x = rotate(g @ W_rot).

Keeping the module an ``nn.Linear`` (patched forward, INT8 ``.weight``) is load-bearing:
LoRA targets modules by class name "Linear", block swap streams ``module.weight.data``
of Linear-named modules, and compile exclusion also keys on the class name.
"""

import json
import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from musubi_tuner.modules.convrot_int8_kernels import (
    HAS_TRITON,
    _build_hadamard,
    _rotate_activation,
    int8_linear,
    quantize_int8_convrot_weight,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from musubi_tuner.utils.device_utils import clean_memory_on_device

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CONVROT_GROUPSIZE = 256


@dataclass(frozen=True)
class ConvRotInt8LayerSpec:
    module_path: str
    weight_key: str
    scale_key: str
    groupsize: int


@dataclass(frozen=True)
class ConvRotInt8Artifact:
    layers: Mapping[str, ConvRotInt8LayerSpec]
    control_keys: frozenset[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", MappingProxyType(dict(self.layers)))

    @property
    def weight_keys(self) -> frozenset[str]:
        return frozenset(layer.weight_key for layer in self.layers.values())

    @property
    def scale_keys(self) -> frozenset[str]:
        return frozenset(layer.scale_key for layer in self.layers.values())


@dataclass(frozen=True)
class _HeaderTensor:
    checkpoint_path: str
    raw_key: str
    normalized_key: str
    dtype: str
    shape: tuple[int, ...]


def canonicalize_convrot_int8_key(key: str) -> str:
    if key.endswith(".weight_scale"):
        return key.removesuffix(".weight_scale") + ".scale_weight"
    return key


def _is_power_of_four(value: int) -> bool:
    if isinstance(value, bool) or not isinstance(value, int) or value < 4:
        return False
    while value % 4 == 0:
        value //= 4
    return value == 1


def _artifact_error(entry: _HeaderTensor, module_path: str, message: str) -> ValueError:
    return ValueError(f"Invalid ConvRot INT8 artifact {entry.checkpoint_path}, layer {module_path}: {message}")


def _decode_convrot_control(tensor: torch.Tensor, entry: _HeaderTensor, module_path: str) -> int:
    try:
        payload = bytes(tensor.tolist()).decode("utf-8")
        values = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, TypeError) as error:
        raise _artifact_error(entry, module_path, f".comfy_quant is not valid UTF-8 JSON ({error})") from error

    if not isinstance(values, dict):
        raise _artifact_error(entry, module_path, ".comfy_quant JSON must be an object")
    if values.get("format") != "int8_tensorwise":
        raise _artifact_error(entry, module_path, "unsupported .comfy_quant format")
    if values.get("convrot") is not True:
        raise _artifact_error(entry, module_path, ".comfy_quant convrot must be true")

    groupsize = values.get("convrot_groupsize")
    if not _is_power_of_four(groupsize):
        raise _artifact_error(entry, module_path, "convrot_groupsize must be an exact power of four and at least 4")
    return groupsize


def inspect_convrot_int8_artifact(
    files: Iterable[str | Path],
    *,
    key_normalizer: Callable[[str], str] | None = None,
    disable_numpy_memmap: bool = False,
) -> ConvRotInt8Artifact | None:
    """Inspect Comfy ConvRot triples without materializing model weights."""
    entries: dict[str, _HeaderTensor] = {}
    controls: dict[str, torch.Tensor] = {}
    artifact_modules: dict[str, _HeaderTensor] = {}

    for file in files:
        checkpoint_path = str(file)
        with MemoryEfficientSafeOpen(checkpoint_path, disable_numpy_memmap=disable_numpy_memmap) as handle:
            for raw_key in handle.keys():
                transformed_key = key_normalizer(raw_key) if key_normalizer is not None else raw_key
                normalized_key = canonicalize_convrot_int8_key(transformed_key)
                header = handle.header[raw_key]
                entry = _HeaderTensor(
                    checkpoint_path=checkpoint_path,
                    raw_key=raw_key,
                    normalized_key=normalized_key,
                    dtype=header["dtype"],
                    shape=tuple(header["shape"]),
                )
                if normalized_key in entries:
                    previous = entries[normalized_key]
                    raise ValueError(
                        "Duplicate normalized safetensors key "
                        f"{normalized_key}: {previous.checkpoint_path}:{previous.raw_key} and {checkpoint_path}:{raw_key}"
                    )
                entries[normalized_key] = entry

                if normalized_key.endswith(".comfy_quant"):
                    module_path = normalized_key.removesuffix(".comfy_quant")
                    artifact_modules[module_path] = entry
                    if entry.dtype != "U8":
                        raise _artifact_error(entry, module_path, f".comfy_quant must be U8, got {entry.dtype}")
                    if len(entry.shape) != 1:
                        raise _artifact_error(entry, module_path, ".comfy_quant must be one-dimensional")
                    controls[normalized_key] = handle.get_tensor(raw_key)
                elif transformed_key.endswith(".weight_scale"):
                    module_path = normalized_key.removesuffix(".scale_weight")
                    artifact_modules[module_path] = entry
                elif normalized_key.endswith(".weight") and entry.dtype == "I8":
                    module_path = normalized_key.removesuffix(".weight")
                    artifact_modules[module_path] = entry

    if not artifact_modules:
        return None

    layers: dict[str, ConvRotInt8LayerSpec] = {}
    for module_path, context_entry in artifact_modules.items():
        weight_key = f"{module_path}.weight"
        scale_key = f"{module_path}.scale_weight"
        control_key = f"{module_path}.comfy_quant"
        missing = [key for key in (weight_key, scale_key, control_key) if key not in entries]
        if missing:
            raise _artifact_error(context_entry, module_path, f"missing sibling tensors: {', '.join(missing)}")

        weight_entry = entries[weight_key]
        scale_entry = entries[scale_key]
        control_entry = entries[control_key]
        if weight_entry.dtype != "I8":
            raise _artifact_error(weight_entry, module_path, f"weight must be INT8/I8, got {weight_entry.dtype}")
        if len(weight_entry.shape) != 2:
            raise _artifact_error(weight_entry, module_path, f"weight must be two-dimensional, got {weight_entry.shape}")
        if scale_entry.dtype != "F32":
            raise _artifact_error(scale_entry, module_path, f"scale must be FP32/F32, got {scale_entry.dtype}")
        expected_scale_shape = (weight_entry.shape[0], 1)
        if scale_entry.shape != expected_scale_shape:
            raise _artifact_error(
                scale_entry,
                module_path,
                f"scale shape must be {expected_scale_shape}, got {scale_entry.shape}",
            )
        if control_entry.dtype != "U8":
            raise _artifact_error(control_entry, module_path, f".comfy_quant must be U8, got {control_entry.dtype}")

        groupsize = _decode_convrot_control(controls[control_key], control_entry, module_path)
        if weight_entry.shape[1] % groupsize != 0:
            raise _artifact_error(
                control_entry,
                module_path,
                f"convrot_groupsize {groupsize} does not divide input width {weight_entry.shape[1]}",
            )
        layers[module_path] = ConvRotInt8LayerSpec(module_path, weight_key, scale_key, groupsize)

    return ConvRotInt8Artifact(layers, frozenset(controls))


def quantize_weight_convrot(key: str, tensor: torch.Tensor, groupsize: int = CONVROT_GROUPSIZE):
    """Quantize a single weight tensor with ConvRot INT8, or return None if not applicable.

    The quantization is always done with the eager implementation: it is deterministic
    and independent of triton availability, so the resulting state dict is identical
    across environments.

    Returns:
        (quantized int8 [N, K] in rotated basis, scale float32 [N, 1]), or None if the
        tensor is not a 2D weight or its in_features is not divisible by groupsize.
    """
    if tensor.ndim != 2:
        logger.info(f"Skipping ConvRot INT8 for {key}: not a 2D weight (ndim={tensor.ndim})")
        return None
    if tensor.shape[1] % groupsize != 0:
        logger.info(f"Skipping ConvRot INT8 for {key}: in_features {tensor.shape[1]} not divisible by {groupsize}")
        return None
    return quantize_int8_convrot_weight(tensor, groupsize)


class ConvRotInt8Quantizer:
    """Strategy object that streams safetensors files and quantizes target weights.

    Passed as ``quantizer`` to ``load_safetensors_with_lora_and_fp8``; carries its own
    streaming loader so the fp8 path stays untouched.
    """

    def __init__(
        self,
        target_layer_keys: Optional[List[str]] = None,
        exclude_layer_keys: Optional[List[str]] = None,
        groupsize: int = CONVROT_GROUPSIZE,
    ):
        self.target_layer_keys = target_layer_keys
        self.exclude_layer_keys = exclude_layer_keys
        self.groupsize = groupsize

    def is_target_key(self, key: str) -> bool:
        is_target = (self.target_layer_keys is None or any(pattern in key for pattern in self.target_layer_keys)) and key.endswith(
            ".weight"
        )
        is_excluded = self.exclude_layer_keys is not None and any(pattern in key for pattern in self.exclude_layer_keys)
        return is_target and not is_excluded

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
    ) -> dict:
        """Load state dict from safetensors files, quantizing target weights to ConvRot INT8.

        Same streaming contract as load_safetensors_with_fp8_optimization: the LoRA merge
        weight_hook runs on the raw (bf16) weight before quantization.
        """
        optimized_count = 0
        state_dict = {}
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

                keys = f.keys()
                for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                    value = f.get_tensor(key)
                    original_device = value.device  # usually cpu

                    if weight_hook is not None:
                        value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                    if not self.is_target_key(key):
                        target_device = calc_device if (calc_device is not None and move_to_device) else original_device
                        state_dict[key] = value.to(target_device)
                        continue

                    if calc_device is not None:
                        value = value.to(calc_device)

                    if value.dtype.itemsize == 1:
                        raise ValueError(
                            f"Layer {key} is already in {value.dtype} format. Loading pre-quantized weights with"
                            " --convrot_int8 is not supported yet. Please use fp16/bf16/float32 model weights."
                            + f" / レイヤー {key} は既に{value.dtype}形式です。事前量子化済み重みの --convrot_int8 での"
                            "読み込みは未対応です。FP16/BF16/Float32のモデル重みを使用してください。"
                        )

                    result = quantize_weight_convrot(key, value, self.groupsize)
                    if result is None:
                        # leave the layer unquantized (bf16)
                        if not move_to_device:
                            value = value.to(original_device)
                        state_dict[key] = value
                        continue
                    quantized_weight, scale_tensor = result

                    scale_key = key.replace(".weight", ".scale_weight")
                    assert key != scale_key, "weight key and scale key must be different"

                    if not move_to_device:
                        quantized_weight = quantized_weight.to(original_device)

                    # scale stays float32 [N, 1]: the Triton epilogue and the backward want fp32,
                    # and the shape maps 1:1 to ComfyUI's `weight_scale`
                    state_dict[key] = quantized_weight
                    state_dict[scale_key] = scale_tensor.to(device=quantized_weight.device)

                    optimized_count += 1

                    if calc_device is not None and optimized_count % 10 == 0:
                        clean_memory_on_device(calc_device)

        logger.info(f"Number of ConvRot INT8 optimized Linear layers: {optimized_count}")
        return state_dict


class ConvRotInt8LinearFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, wq, w_scale, bias, groupsize, bwd_mode):
        # x: [..., K] bf16/fp16, wq: [N, K] int8 (rotated basis), w_scale: [N, 1] fp32
        # F.linear casts its inputs to the autocast dtype under autocast; the fused kernel
        # bypasses F.linear, so replicate that here. In K2 the fp32 modulation adds promote
        # the activations to fp32, and downstream flash-attn only accepts fp16/bf16.
        if torch.is_autocast_enabled(x.device.type):
            cast_dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(cast_dtype)
            if bias is not None:
                bias = bias.to(cast_dtype)
        if HAS_TRITON and x.is_cuda:
            out = int8_linear(x, wq, w_scale.reshape(-1), bias, x.dtype, True, groupsize)
        else:
            # eager fallback: rotation + transient dequantized matmul, no activation quantization
            h = _build_hadamard(groupsize, device=x.device, dtype=x.dtype)
            x_rot = _rotate_activation(x, h, groupsize)
            w_rot = wq.to(x.dtype) * w_scale.reshape(-1, 1).to(x.dtype)
            out = F.linear(x_rot, w_rot, bias)
        # wq/w_scale are live buffers, so saving them adds no activation memory; x is not
        # saved (base is frozen, no grad_weight needed)
        ctx.save_for_backward(wq, w_scale)
        ctx.groupsize = groupsize
        ctx.bwd_mode = bwd_mode
        ctx.bias_needs_grad = bias is not None and bias.requires_grad
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        wq, w_scale = ctx.saved_tensors
        gs = ctx.groupsize
        g2d = grad_out.reshape(-1, grad_out.shape[-1])  # [M, N]

        grad_x = None
        if ctx.needs_input_grad[0]:
            # grad_x = g @ W = g @ (W_rot R) = rotate(g @ W_rot), R = block-diag Hadamard
            if ctx.bwd_mode == "int8":
                # fold per-channel weight scale into g, then reuse the fused Triton GEMM
                # (row-wise quant of g + int8 GEMM + dequant epilogue in one pipeline).
                # transient int8 transpose of wq: [K, N], ~1 byte/param, freed after mm
                g_scaled = g2d * w_scale.reshape(1, -1).to(g2d.dtype)
                one = torch.ones(1, device=g2d.device, dtype=torch.float32)
                gx_rot = int8_linear(g_scaled, wq.t().contiguous(), one, None, grad_out.dtype, False, gs)
            else:
                # transient bf16 dequant of the rotated weight (stays in rotated basis)
                w_rot = wq.to(grad_out.dtype) * w_scale.reshape(-1, 1).to(grad_out.dtype)
                gx_rot = g2d @ w_rot  # [M, K]
            h = _build_hadamard(gs, device=gx_rot.device, dtype=gx_rot.dtype)
            grad_x = _rotate_activation(gx_rot, h, gs).reshape(*grad_out.shape[:-1], wq.shape[1])

        grad_bias = g2d.sum(dim=0) if ctx.bias_needs_grad else None
        return grad_x, None, None, grad_bias, None, None


def convrot_int8_linear_forward_patch(self: nn.Linear, x):
    return ConvRotInt8LinearFn.apply(x, self.weight, self.scale_weight, self.bias, self._convrot_groupsize, self._convrot_bwd_mode)


def _validate_convrot_bwd_mode(bwd_mode: str) -> None:
    if bwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 backward mode: {bwd_mode}")
    if bwd_mode == "int8" and not HAS_TRITON:
        raise ValueError("ConvRot INT8 backward mode 'int8' requires triton. Install triton (triton-windows on Windows).")
    if not HAS_TRITON:
        logger.warning(
            "triton is not available: ConvRot INT8 falls back to transient dequantization in forward."
            " Weight VRAM is still reduced, but there is no speedup. Install triton (triton-windows on Windows)"
            " for the fused INT8 kernels."
        )


def prepare_convrot_int8_model(
    model: nn.Module,
    artifact: ConvRotInt8Artifact,
    *,
    bwd_mode: str = "bf16",
) -> nn.Module:
    """Prepare exact Linear modules for streamed ConvRot INT8 assignment."""
    _validate_convrot_bwd_mode(bwd_mode)

    modules: list[tuple[ConvRotInt8LayerSpec, nn.Linear]] = []
    for module_path, layer in artifact.layers.items():
        try:
            module = model.get_submodule(module_path)
        except AttributeError as error:
            raise ValueError(f"ConvRot INT8 artifact declares missing module {module_path}") from error
        if type(module) is not nn.Linear:
            raise TypeError(f"ConvRot INT8 module {module_path} must be an exact nn.Linear, got {type(module).__name__}")
        if not _is_power_of_four(layer.groupsize):
            raise ValueError(f"ConvRot INT8 module {module_path} has invalid group size {layer.groupsize}")
        if module.in_features % layer.groupsize != 0:
            raise ValueError(
                f"ConvRot INT8 module {module_path} group size {layer.groupsize} does not divide input width {module.in_features}"
            )
        modules.append((layer, module))

    for layer, module in modules:
        module.weight = nn.Parameter(
            torch.empty(module.weight.shape, device=module.weight.device, dtype=torch.int8),
            requires_grad=False,
        )
        module.register_buffer(
            "scale_weight",
            torch.empty((module.out_features, 1), device=module.weight.device, dtype=torch.float32),
        )
        module._convrot_groupsize = layer.groupsize
        module._convrot_bwd_mode = bwd_mode
        module.forward = convrot_int8_linear_forward_patch.__get__(module, nn.Linear)

    model.is_convrot_int8 = True
    model.convrot_int8_layer_count = len(modules)
    logger.info("Number of ConvRot INT8 monkey-patched Linear layers: %s", len(modules))
    return model


def apply_convrot_int8_monkey_patch(model, optimized_state_dict, bwd_mode: str = "bf16", groupsize: int = CONVROT_GROUPSIZE):
    """Patch a model from a state dict produced by ``ConvRotInt8Quantizer``."""
    layers = {}
    for scale_key in optimized_state_dict:
        if not scale_key.endswith(".scale_weight"):
            continue
        module_path = scale_key.removesuffix(".scale_weight")
        weight_key = f"{module_path}.weight"
        if weight_key not in optimized_state_dict:
            raise ValueError(f"ConvRot INT8 scale {scale_key} has no sibling weight")
        layers[module_path] = ConvRotInt8LayerSpec(module_path, weight_key, scale_key, groupsize)

    artifact = ConvRotInt8Artifact(layers, frozenset())
    return prepare_convrot_int8_model(model, artifact, bwd_mode=bwd_mode)
