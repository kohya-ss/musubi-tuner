"""Tests for NVFP4 training support: columnwise weight requantization and the
autograd-enabled forward/backward (NvFp4LinearFn).

CPU tests exercise pure-tensor quantization math (no scaled_mm). GPU tests
requiring real FP4 tensor-core execution are marked ``requires_nvfp4_scaled_mm``.
"""

import pytest
import torch

from musubi_tuner.modules.nvfp4_utils import (
    NVFP4_BLOCK_SIZE,
    _quantize_nvfp4_2d,
    dequantize_nvfp4,
    nvfp4_scaled_mm_available,
    quantize_nvfp4_weight_columnwise,
)

requires_nvfp4_scaled_mm = pytest.mark.skipif(
    not (torch.cuda.is_available() and nvfp4_scaled_mm_available()),
    reason="CUDA + torch 2.10+ scaled_mm/float4_e2m1fn_x2 required",
)


def _make_quantized_weight(n, k, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k) * 0.02
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(w)
    return w, packed, block_scale, tensor_scale


def test_quantize_nvfp4_weight_columnwise_shapes():
    n, k = 64, 32
    _w, packed, block_scale, tensor_scale = _make_quantized_weight(n, k)

    packed_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(
        packed, block_scale, tensor_scale, (n, k)
    )

    assert packed_t.dtype is torch.uint8
    assert packed_t.shape == (k, n // 2)
    assert block_scale_t.dtype is torch.float8_e4m3fn
    assert tensor_scale_t.dtype is torch.float32
    assert tensor_scale_t.ndim == 0


def test_quantize_nvfp4_weight_columnwise_roundtrip_matches_rowwise():
    n, k = 64, 32
    w, packed, block_scale, tensor_scale = _make_quantized_weight(n, k)

    packed_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(
        packed, block_scale, tensor_scale, (n, k)
    )

    w_deq = dequantize_nvfp4(packed, block_scale, tensor_scale, (n, k), torch.float32)
    w_t_deq = dequantize_nvfp4(packed_t, block_scale_t, tensor_scale_t, (k, n), torch.float32).t()

    # Both are independent NVFP4 quantizations of the same underlying bf16-scale weight,
    # grouped along different axes -- they should each track the original within normal
    # FP4 quantization noise, and therefore track each other within roughly double that.
    rel_err_to_original = (w_deq - w).norm() / w.norm()
    rel_err_between = (w_deq - w_t_deq).norm() / w_deq.norm()
    assert rel_err_to_original < 0.2
    assert rel_err_between < 0.3


def test_quantize_nvfp4_weight_columnwise_rejects_non_multiple_of_block_size():
    n, k = 48, 32  # n=48 is a multiple of 16, use a bad n to trigger the check
    _w, packed, block_scale, tensor_scale = _make_quantized_weight(64, k)
    with pytest.raises(ValueError, match="out_features"):
        quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (50, k))
