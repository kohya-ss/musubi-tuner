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


def _make_linear_fixture(n, k, m, device, bias=False, seed=0):
    torch.manual_seed(seed)
    w = (torch.randn(n, k, device=device) * 0.02).to(torch.bfloat16)
    x = (torch.randn(m, k, device=device) * 0.5).to(torch.bfloat16)
    b = (torch.randn(n, device=device) * 0.1).to(torch.bfloat16) if bias else None
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(w.float())
    packed_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k))
    return w, x, b, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t


@requires_nvfp4_scaled_mm
def test_nvfp4_linear_fn_forward_matches_scaled_mm_reference():
    from musubi_tuner.modules.nvfp4_utils import NvFp4LinearFn, nvfp4_scaled_mm_linear

    n, k, m = 64, 32, 8
    device = "cuda"
    w, x, b, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t = _make_linear_fixture(
        n, k, m, device, bias=True
    )

    out = NvFp4LinearFn.apply(x, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t, b, n)
    expected = nvfp4_scaled_mm_linear(x, packed, block_scale, tensor_scale, b, n)

    assert torch.equal(out, expected)


@requires_nvfp4_scaled_mm
def test_nvfp4_linear_fn_backward_grad_x_matches_bf16_dequant_reference():
    from musubi_tuner.modules.nvfp4_utils import NvFp4LinearFn, dequantize_nvfp4

    n, k, m = 64, 32, 8
    device = "cuda"
    w, x, b, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t = _make_linear_fixture(
        n, k, m, device, bias=False
    )
    x_fp4 = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    out = NvFp4LinearFn.apply(x_fp4, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t, None, n)
    out.sum().backward()

    w_deq = dequantize_nvfp4(packed, block_scale, tensor_scale, (n, k), torch.bfloat16)
    ref_out = torch.nn.functional.linear(x_ref, w_deq)
    ref_out.sum().backward()

    rel_err = (x_fp4.grad.float() - x_ref.grad.float()).norm() / x_ref.grad.float().norm()
    assert rel_err < 0.3  # two independently FP4-quantized paths (fwd weight vs bwd weight), not exact
