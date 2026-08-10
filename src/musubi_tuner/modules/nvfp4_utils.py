# Low-level NVFP4 routines (E2M1 encode/decode, cuBLAS swizzle, scaled_mm call) are
# adapted from the exp-nvfp4-support-for-torch-2-10 branch, which was inspired by
# comfy-kitchen (https://github.com/Comfy-Org/comfy-kitchen, Apache License 2.0).

"""Loading ComfyUI pre-quantized NVFP4 (+AWQ) checkpoints for frozen text encoders.

Layout of a ComfyUI NVFP4 artifact (verified bit-level against the BF16 reference of
the MiniMax-H3 Qwen3-VL text encoder):

- NVFP4 Linear: ``.weight`` U8 [N, K/2] with two E2M1 codes per byte (element 0 in the
  HIGH nibble), ``.weight_scale`` F8_E4M3 per-16-block scales stored in the cuBLAS
  128x4 tiled ("swizzled") layout, ``.weight_scale_2`` F32 per-tensor scale, and a
  ``.comfy_quant`` spec ``{"format": "nvfp4", ...}``. Modules quantized with AWQ carry
  a ``.pre_quant_scale`` [K] tensor that is multiplied into the *input* at runtime;
  for the remaining modules the AWQ scale is folded into the preceding norm weights,
  so the checkpoint is self-consistent and needs no special handling here.
- INT8 embedding: ``.weight`` I8 [V, D] + ``.weight_scale`` F32 [V, 1]
  (``{"format": "int8_tensorwise"}``, effectively per-row); dequant = weight * scale.
- Everything else (norms, vision tower, biases) stays BF16.

The state dict is converted to the Musubi runtime layout: ``.weight_scale`` becomes
``.nvfp4_block_scale`` (kept swizzled: ``torch.nn.functional.scaled_mm`` consumes the
swizzled layout directly, the dequantizing fallback unswizzles on the fly) and
``.weight_scale_2`` becomes ``.nvfp4_scale``; the embedding scale becomes
``.scale_weight`` (same dequant semantics as the ConvRot INT8 layout).

Inference only: NVFP4 modules are frozen, the patched forwards have no autograd
support. Dynamic (on-the-fly) NVFP4 quantization is deliberately not offered — the
published artifacts are AWQ-calibrated, which cannot be reproduced without
calibration data, and dynamically quantizing BF16 weights would silently produce a
lower-quality model than ConvRot INT8.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

from tqdm import tqdm

from musubi_tuner.modules.comfy_quant_utils import (
    COMFY_QUANT_SUFFIX,
    COMFY_WEIGHT_SCALE_SUFFIX,
    FORMAT_INT8_TENSORWISE,
    FORMAT_NVFP4,
    classify_comfy_quant_spec,
    decode_comfy_quant_spec,
)
from musubi_tuner.utils.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks

logger = logging.getLogger(__name__)

NVFP4_BLOCK_SIZE = 16

F4_E2M1_MAX = 6.0
F8_E4M3_MAX = 448.0

COMFY_WEIGHT_SCALE_2_SUFFIX = ".weight_scale_2"
COMFY_PRE_QUANT_SCALE_SUFFIX = ".pre_quant_scale"

# E2M1 code -> value (codes 0..7 positive, 8..15 negative)
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# byte -> (high nibble value, low nibble value), cached per (device, dtype)
_BYTE_PAIR_LUT_CACHE: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _byte_pair_lut(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    lut = _BYTE_PAIR_LUT_CACHE.get((device, dtype))
    if lut is None:
        values = torch.tensor(_E2M1_VALUES, dtype=torch.float32)
        codes = torch.arange(256, dtype=torch.int64)
        lut = torch.stack([values[codes >> 4], values[codes & 0x0F]], dim=1).to(device=device, dtype=dtype)
        _BYTE_PAIR_LUT_CACHE[(device, dtype)] = lut
    return lut


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _roundup(value: int, multiple: int) -> int:
    return _ceil_div(value, multiple) * multiple


# region cuBLAS 128x4 tiled block-scale layout
# https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout


def to_blocked(input_matrix: torch.Tensor) -> torch.Tensor:
    """Rearrange (H, W) block scales into the cuBLAS tiled layout (padded to 128x4)."""
    rows, cols = input_matrix.shape
    n_row_blocks = _ceil_div(rows, 128)
    n_col_blocks = _ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros((padded_rows, padded_cols), device=input_matrix.device, dtype=input_matrix.dtype)
        padded[:rows, :cols] = input_matrix
    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return rearranged.reshape(padded_rows, padded_cols)


def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """Reverse the cuBLAS tiled layout back to a row-major (num_rows, num_cols) matrix."""
    n_row_blocks = _ceil_div(num_rows, 128)
    n_col_blocks = _ceil_div(num_cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4
    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)
    return unblocked[:num_rows, :num_cols]


# endregion


def dequantize_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    orig_shape: Tuple[int, int],
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize packed NVFP4 data (block_scale in the swizzled layout) to ``orig_shape``."""
    stored_rows = packed.shape[0]
    stored_cols = packed.shape[1] * 2
    lut = _byte_pair_lut(packed.device, out_dtype)
    # one embedding lookup decodes both nibbles of each byte: [R, C/2] -> [R, C/2, 2] -> [R, C]
    weight = F.embedding(packed.reshape(-1).int(), lut).reshape(stored_rows, stored_cols)
    scales = from_blocked(block_scale, stored_rows, stored_cols // NVFP4_BLOCK_SIZE)
    total = (per_tensor_scale.float() * scales.float()).to(out_dtype)
    weight = (weight.reshape(stored_rows, -1, NVFP4_BLOCK_SIZE) * total.unsqueeze(-1)).reshape(stored_rows, stored_cols)
    return weight[: orig_shape[0], : orig_shape[1]]


# region runtime activation quantization for the scaled_mm (W4A4) path


def _n_ones(n: int) -> int:
    return (1 << n) - 1


_EBITS_F32, _MBITS_F32 = 8, 23
_F32_EXP_BIAS = _n_ones(_EBITS_F32 - 1)


def _f32_to_e2m1_unpacked(x: torch.Tensor) -> torch.Tensor:
    """Convert FP32 to E2M1 codes stored one-per-byte in uint8 (round-to-nearest-even)."""
    ebits, mbits = 2, 1
    assert x.dtype == torch.float
    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(_MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)
    denorm_exp = (_F32_EXP_BIAS - exp_bias) + (_MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << _MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32, device=x.device).view(torch.float32)

    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = (x ^ sign).view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    denormal_x = (x + denorm_mask_float).view(torch.int32) - denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (_MBITS_F32 - mbits)) & 1
    normal_x = normal_x + (((exp_bias - _F32_EXP_BIAS) << _MBITS_F32) + magic_adder) + mant_odd
    normal_x = (normal_x >> (_MBITS_F32 - mbits)).to(torch.uint8)

    result = torch.full_like(x, max_int, dtype=torch.uint8)
    result = torch.where(denormal_mask, denormal_x, result)
    result = torch.where(normal_mask, normal_x, result)

    sign_lp = (sign >> (_MBITS_F32 + _EBITS_F32 - mbits - ebits)).to(torch.uint8) & sign_mask
    return result | sign_lp


def pack_uint4(codes: torch.Tensor) -> torch.Tensor:
    """Pack pairs of 4-bit codes (one per byte) into bytes, element 0 in the HIGH nibble."""
    shape = codes.shape
    assert shape[-1] % 2 == 0
    codes = codes.contiguous().view(-1)
    return (codes[::2] << 4 | codes[1::2]).view(*shape[:-1], shape[-1] // 2)


def quantize_nvfp4_activation(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Quantize a 2D activation to NVFP4 for scaled_mm.

    Returns (packed uint8 [Mp, K/2], swizzled block scales F8_E4M3, per-tensor scale F32,
    original row count). Rows are padded to a multiple of 16 as scaled_mm requires.
    """
    orig_rows, cols = x.shape
    if cols % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(f"NVFP4 activation width must be a multiple of {NVFP4_BLOCK_SIZE}, got {cols}")
    padded_rows = _roundup(orig_rows, 16)
    if padded_rows != orig_rows:
        x = F.pad(x, (0, 0, 0, padded_rows - orig_rows))

    per_tensor_scale = (torch.amax(x.abs()).float() / (F8_E4M3_MAX * F4_E2M1_MAX)).reshape(())

    blocks = x.reshape(padded_rows, -1, NVFP4_BLOCK_SIZE)
    block_scale = torch.amax(blocks.abs(), dim=-1).float() / F4_E2M1_MAX
    scaled = torch.clamp(block_scale / torch.clamp(per_tensor_scale, min=torch.finfo(torch.float32).tiny), max=F8_E4M3_MAX)
    scaled_f8 = scaled.to(torch.float8_e4m3fn)
    total = per_tensor_scale * scaled_f8.float()
    total_safe = torch.where(total == 0, torch.ones_like(total), total)

    data = blocks.float() / total_safe.unsqueeze(-1)
    data = torch.where((total == 0).unsqueeze(-1), torch.zeros_like(data), data)
    data = torch.clamp(data, -F4_E2M1_MAX, F4_E2M1_MAX).reshape(padded_rows, cols)

    packed = pack_uint4(_f32_to_e2m1_unpacked(data))
    return packed, to_blocked(scaled_f8), per_tensor_scale, orig_rows


def nvfp4_scaled_mm_available() -> bool:
    return hasattr(torch, "float4_e2m1fn_x2") and hasattr(torch.nn.functional, "scaled_mm")


def nvfp4_scaled_mm_linear(
    x: torch.Tensor,
    weight_packed: torch.Tensor,
    weight_block_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    orig_out_features: int,
) -> torch.Tensor:
    """W4A4 linear via torch.nn.functional.scaled_mm (torch 2.10+, Blackwell)."""
    from torch.nn.functional import ScalingType, SwizzleType

    x_packed, x_block_scale, x_scale, orig_rows = quantize_nvfp4_activation(x)
    result = torch.nn.functional.scaled_mm(
        x_packed.view(torch.float4_e2m1fn_x2),
        weight_packed.view(torch.float4_e2m1fn_x2).t(),
        scale_a=[x_block_scale.view(-1), x_scale],
        scale_b=[weight_block_scale.view(-1), weight_scale],
        bias=bias,
        output_dtype=x.dtype,
        scale_recipe_a=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        scale_recipe_b=[ScalingType.BlockWise1x16, ScalingType.TensorWise],
        swizzle_a=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
        swizzle_b=[SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE],
    )
    return result[:orig_rows, :orig_out_features]


# endregion


# region pre-quantized state dict loading


class NvFp4Quantizer:
    """Streams a ComfyUI pre-quantized NVFP4 (+ INT8 embedding) checkpoint.

    Same protocol as ``ConvRotInt8Quantizer``: passed as ``quantizer`` to
    ``load_safetensors_with_lora_and_fp8``. Loading only converts key names and
    validates the tensors — there is no dynamic quantization, the file dictates the
    quantized layers. ``nvfp4_module_shapes`` maps module paths to their original
    (out_features, in_features) after loading; ``int8_embedding_modules`` lists the
    per-row INT8 modules (embeddings). Pass both to ``apply_nvfp4_monkey_patch``.
    """

    def __init__(self):
        self.nvfp4_module_shapes: Dict[str, Tuple[int, int]] = {}
        self.int8_embedding_modules: List[str] = []

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
    ) -> dict:
        state_dict = {}
        module_formats: Dict[str, str] = {}  # spans all shards
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

                keys = f.keys()

                # pre-scan the tiny spec tensors so each module's format is known before
                # the (possibly earlier-iterated) weight/scale keys arrive
                for key in keys:
                    if key.endswith(COMFY_QUANT_SUFFIX):
                        module_path = key[: -len(COMFY_QUANT_SUFFIX)]
                        spec_format = classify_comfy_quant_spec(decode_comfy_quant_spec(key, f.get_tensor(key)))
                        if spec_format not in (FORMAT_NVFP4, FORMAT_INT8_TENSORWISE):
                            raise ValueError(
                                f"Unsupported comfy_quant format for {key}: {spec_format}. The NVFP4 loader supports"
                                ' "nvfp4" Linear layers and "int8_tensorwise" embeddings only.'
                                f" / {key} の comfy_quant 形式 {spec_format} はNVFP4ローダーではサポートされていません。"
                            )
                        module_formats[module_path] = spec_format
                if module_formats and weight_hook is not None:
                    raise ValueError(
                        f"Cannot merge LoRA weights into pre-quantized NVFP4 checkpoint {model_file}."
                        " Use the original BF16 weights instead."
                        f" / 事前量子化済みNVFP4チェックポイント {model_file} にはLoRAをマージできません。"
                        "BF16の元重みを使用してください。"
                    )

                for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key"):
                    if key.endswith(COMFY_QUANT_SUFFIX):
                        continue  # consumed in the pre-scan, not a model tensor

                    value = f.get_tensor(key)
                    original_device = value.device  # usually cpu
                    passthrough_device = calc_device if (calc_device is not None and move_to_device) else original_device
                    converted_key = self._convert_key(key, value, module_formats)
                    state_dict[converted_key] = value.to(passthrough_device)

        self._validate_completeness(state_dict, module_formats)
        logger.info(
            f"Number of pre-quantized layers: {len(self.nvfp4_module_shapes)} NVFP4 Linear,"
            f" {len(self.int8_embedding_modules)} INT8 embedding"
        )
        return state_dict

    def _convert_key(self, key: str, value: torch.Tensor, module_formats: Dict[str, str]) -> str:
        """Validate a tensor against its module's declared format and return the Musubi key."""
        for suffix in (COMFY_WEIGHT_SCALE_SUFFIX, COMFY_WEIGHT_SCALE_2_SUFFIX, COMFY_PRE_QUANT_SCALE_SUFFIX, ".weight"):
            if key.endswith(suffix):
                module_path = key[: -len(suffix)]
                break
        else:
            return key  # bias, norm, etc.: passthrough
        spec_format = module_formats.get(module_path)
        if spec_format is None:
            if key.endswith(".weight") and value.dtype.itemsize == 1:
                raise ValueError(
                    f"Layer {key} is already in {value.dtype} format but has no {COMFY_QUANT_SUFFIX} spec."
                    f" / レイヤー {key} は既に{value.dtype}形式ですが {COMFY_QUANT_SUFFIX} がありません。"
                )
            if key.endswith((COMFY_WEIGHT_SCALE_SUFFIX, COMFY_WEIGHT_SCALE_2_SUFFIX, COMFY_PRE_QUANT_SCALE_SUFFIX)):
                raise ValueError(f"Found {key} without a matching {module_path}{COMFY_QUANT_SUFFIX} spec")
            return key

        if spec_format == FORMAT_NVFP4:
            if key.endswith(COMFY_WEIGHT_SCALE_SUFFIX):
                if value.dtype is not torch.float8_e4m3fn:
                    raise ValueError(f"NVFP4 block scale {key} must be F8_E4M3, got {value.dtype}")
                return module_path + ".nvfp4_block_scale"
            if key.endswith(COMFY_WEIGHT_SCALE_2_SUFFIX):
                if value.dtype is not torch.float32 or value.ndim != 0:
                    raise ValueError(f"NVFP4 per-tensor scale {key} must be a F32 scalar, got {value.dtype} {tuple(value.shape)}")
                return module_path + ".nvfp4_scale"
            if key.endswith(COMFY_PRE_QUANT_SCALE_SUFFIX):
                if not value.is_floating_point() or value.ndim != 1:
                    raise ValueError(f"AWQ pre_quant_scale {key} must be a 1D float tensor, got {value.dtype} {tuple(value.shape)}")
                return key
            # .weight
            if value.dtype is not torch.uint8 or value.ndim != 2:
                raise ValueError(f"NVFP4 weight {key} must be 2D uint8 (packed), got {value.dtype} ndim={value.ndim}")
            in_features = value.shape[1] * 2
            if in_features % NVFP4_BLOCK_SIZE != 0:
                raise ValueError(f"NVFP4 weight {key}: in_features {in_features} not a multiple of {NVFP4_BLOCK_SIZE}")
            self.nvfp4_module_shapes[module_path] = (value.shape[0], in_features)
            return key

        # FORMAT_INT8_TENSORWISE: per-row INT8, embeddings only (validated at patch time)
        if key.endswith(COMFY_WEIGHT_SCALE_SUFFIX):
            if value.dtype is not torch.float32:
                raise ValueError(f"INT8 per-row scale {key} must be F32, got {value.dtype}")
            return module_path + ".scale_weight"
        if key.endswith((COMFY_WEIGHT_SCALE_2_SUFFIX, COMFY_PRE_QUANT_SCALE_SUFFIX)):
            raise ValueError(f"Unexpected tensor {key} for int8_tensorwise module {module_path}")
        if value.dtype is not torch.int8:
            raise ValueError(f"INT8 weight {key} must be int8, got {value.dtype}")
        self.int8_embedding_modules.append(module_path)
        return key

    def _validate_completeness(self, state_dict: dict, module_formats: Dict[str, str]) -> None:
        for module_path, spec_format in module_formats.items():
            if spec_format == FORMAT_NVFP4:
                required = (".weight", ".nvfp4_block_scale", ".nvfp4_scale")
            else:
                required = (".weight", ".scale_weight")
            missing = [module_path + suffix for suffix in required if module_path + suffix not in state_dict]
            if missing:
                raise ValueError(f"Pre-quantized module {module_path} is missing tensors {missing}")

            if spec_format == FORMAT_NVFP4:
                rows, in_features = self.nvfp4_module_shapes[module_path]
                expected_scale_numel = _roundup(rows, 128) * _roundup(in_features // NVFP4_BLOCK_SIZE, 4)
                block_scale = state_dict[module_path + ".nvfp4_block_scale"]
                if block_scale.numel() != expected_scale_numel:
                    raise ValueError(
                        f"NVFP4 module {module_path}: block scale has {block_scale.numel()} elements,"
                        f" expected {expected_scale_numel} for weight shape ({rows}, {in_features})"
                    )
                pre_quant_scale = state_dict.get(module_path + COMFY_PRE_QUANT_SCALE_SUFFIX)
                if pre_quant_scale is not None and pre_quant_scale.shape[0] != in_features:
                    raise ValueError(
                        f"NVFP4 module {module_path}: pre_quant_scale has {pre_quant_scale.shape[0]} elements,"
                        f" expected in_features {in_features}"
                    )
            else:
                weight = state_dict[module_path + ".weight"]
                scale = state_dict[module_path + ".scale_weight"]
                expected_scale_shape = (weight.shape[0], 1)
                if tuple(scale.shape) != expected_scale_shape:
                    raise ValueError(
                        f"INT8 module {module_path}: scale shape must be {expected_scale_shape}, got {tuple(scale.shape)}"
                    )


# endregion


# region monkey patch


def nvfp4_linear_forward_patch(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    pre_quant_scale = getattr(self, "pre_quant_scale", None)
    if pre_quant_scale is not None:
        x = x * pre_quant_scale
    if self._nvfp4_use_scaled_mm:
        x_2d = x.reshape(-1, x.shape[-1])
        out = nvfp4_scaled_mm_linear(
            x_2d, self.weight, self.nvfp4_block_scale, self.nvfp4_scale, self.bias, self._nvfp4_orig_shape[0]
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
    weight = dequantize_nvfp4(self.weight, self.nvfp4_block_scale, self.nvfp4_scale, self._nvfp4_orig_shape, x.dtype)
    return F.linear(x, weight, self.bias)


def int8_embedding_forward_patch(self: nn.Embedding, input: torch.Tensor) -> torch.Tensor:
    rows = self.weight[input]  # index_select works on int8; padding_idx etc. only affect training
    return (rows.float() * self.scale_weight[input]).to(self._int8_dequant_dtype)


def apply_nvfp4_monkey_patch(
    model: nn.Module,
    optimized_state_dict: dict,
    nvfp4_module_shapes: Dict[str, Tuple[int, int]],
    int8_embedding_modules: List[str],
    use_scaled_mm: bool = False,
    embedding_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Patch NVFP4 Linear and INT8 embedding modules so a strict assign load can follow.

    The patched modules get placeholder parameters/buffers with the quantized shapes and
    dtypes (on the meta device); ``model.load_state_dict(state_dict, strict=True,
    assign=True)`` then installs the real tensors. Modules stay ``nn.Linear`` /
    ``nn.Embedding`` (patched forward), mirroring the ConvRot INT8 approach.
    """
    if use_scaled_mm and not nvfp4_scaled_mm_available():
        raise ValueError(
            "NVFP4 scaled_mm requires PyTorch 2.10+ (torch.float4_e2m1fn_x2 and torch.nn.functional.scaled_mm)."
            " Omit the scaled_mm option to use the dequantize fallback."
            " / NVFP4 scaled_mm には PyTorch 2.10 以降が必要です。scaled_mm オプションを外すと dequantize フォールバックで動作します。"
        )

    modules_by_name = dict(model.named_modules())
    patched_count = 0

    for name, (out_features, in_features) in nvfp4_module_shapes.items():
        module = modules_by_name.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"NVFP4 state dict declares {name}, which is not an nn.Linear in the model")
        weight_key = name + ".weight"
        module.weight = nn.Parameter(torch.empty_like(optimized_state_dict[weight_key], device="meta"), requires_grad=False)
        module.register_buffer(
            "nvfp4_block_scale", torch.empty_like(optimized_state_dict[name + ".nvfp4_block_scale"], device="meta")
        )
        module.register_buffer("nvfp4_scale", torch.empty((), dtype=torch.float32, device="meta"))
        pre_quant_key = name + COMFY_PRE_QUANT_SCALE_SUFFIX
        if pre_quant_key in optimized_state_dict:
            module.register_buffer("pre_quant_scale", torch.empty_like(optimized_state_dict[pre_quant_key], device="meta"))
        module._nvfp4_orig_shape = (out_features, in_features)
        module._nvfp4_use_scaled_mm = use_scaled_mm
        module.forward = nvfp4_linear_forward_patch.__get__(module, type(module))
        patched_count += 1

    for name in int8_embedding_modules:
        module = modules_by_name.get(name)
        if not isinstance(module, nn.Embedding):
            raise ValueError(
                f"int8_tensorwise module {name} is not an nn.Embedding; INT8 per-row quantization is only"
                " supported for embeddings (use ConvRot INT8 checkpoints for Linear layers)"
            )
        module.weight = nn.Parameter(torch.empty_like(optimized_state_dict[name + ".weight"], device="meta"), requires_grad=False)
        module.register_buffer("scale_weight", torch.empty_like(optimized_state_dict[name + ".scale_weight"], device="meta"))
        module._int8_dequant_dtype = embedding_dtype
        module.forward = int8_embedding_forward_patch.__get__(module, type(module))
        patched_count += 1

    if not use_scaled_mm and nvfp4_module_shapes:
        logger.info("NVFP4 runs in weight-only mode (transient dequantization per forward)")
    model.is_nvfp4 = True
    model.nvfp4_layer_count = len(nvfp4_module_shapes)
    logger.info(f"Number of NVFP4/INT8 monkey-patched modules: {patched_count}")
    return model


# endregion
