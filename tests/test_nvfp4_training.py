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


import torch.nn as nn

from musubi_tuner.modules.nvfp4_utils import (
    NvFp4Quantizer,
    apply_nvfp4_monkey_patch,
    dequantize_nvfp4,
)


class _TinyDiTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(32, 64, bias=False)

    def forward(self, x):
        return self.proj(x)


def _build_training_patched_model(tmp_path, n=64, k=32):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    weight = torch.randn(n, k) * 0.02
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(weight)
    payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
    tensors = {
        "proj.weight": packed,
        "proj.weight_scale": block_scale,
        "proj.weight_scale_2": tensor_scale,
        "proj.comfy_quant": payload,
    }
    path = tmp_path / "artifact.safetensors"
    save_file(tensors, str(path))

    quantizer = NvFp4Quantizer()
    state_dict = quantizer.load_and_quantize([str(path)], None)
    model = _TinyDiTBlock()
    apply_nvfp4_monkey_patch(
        model, state_dict, quantizer.nvfp4_module_shapes, [], use_scaled_mm=True, training=True,
    )
    model.requires_grad_(False)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model, weight


def test_training_patch_registers_columnwise_buffers():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_path_str:
        from pathlib import Path

        model, _weight = _build_training_patched_model(Path(tmp_path_str))

    assert model.proj.nvfp4_weight_t.dtype is torch.uint8
    assert model.proj.nvfp4_weight_t.shape == (32, 32)  # [K, N/2]
    assert model.proj.nvfp4_block_scale_t.dtype is torch.float8_e4m3fn
    assert model.proj.nvfp4_scale_t.dtype is torch.float32


def test_training_patch_requires_scaled_mm():
    import tempfile
    from pathlib import Path

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        torch.manual_seed(0)
        weight = torch.randn(64, 32) * 0.02
        packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(weight)
        payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
        path = tmp_path / "artifact.safetensors"
        save_file(
            {
                "proj.weight": packed,
                "proj.weight_scale": block_scale,
                "proj.weight_scale_2": tensor_scale,
                "proj.comfy_quant": payload,
            },
            str(path),
        )
        quantizer = NvFp4Quantizer()
        state_dict = quantizer.load_and_quantize([str(path)], None)
        model = _TinyDiTBlock()

        with pytest.raises(ValueError, match="use_scaled_mm=True"):
            apply_nvfp4_monkey_patch(
                model, state_dict, quantizer.nvfp4_module_shapes, [], use_scaled_mm=False, training=True,
            )


@requires_nvfp4_scaled_mm
def test_training_patched_forward_and_backward_run_end_to_end():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_path_str:
        model, _weight = _build_training_patched_model(Path(tmp_path_str))
    model = model.cuda()
    for name, buf in model.named_buffers():
        pass  # buffers already on the right device via .cuda() above

    x = (torch.randn(4, 32, device="cuda") * 0.5).to(torch.bfloat16).requires_grad_(True)
    out = model(x)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@requires_nvfp4_scaled_mm
def test_training_patch_calc_device_computes_on_gpu_but_result_stays_on_original_device():
    """Simulates block swap: the state dict lives on CPU (loading_device="cpu"), but the
    columnwise requant should run on ``calc_device`` (GPU) rather than CPU -- CPU is
    dequant/bit-pack-heavy and impractically slow across a full DiT's worth of modules.
    """
    import tempfile
    from pathlib import Path

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        n, k = 64, 32
        torch.manual_seed(0)
        weight = torch.randn(n, k) * 0.02
        packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(weight)
        payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)
        path = tmp_path / "artifact.safetensors"
        save_file(
            {
                "proj.weight": packed,
                "proj.weight_scale": block_scale,
                "proj.weight_scale_2": tensor_scale,
                "proj.comfy_quant": payload,
            },
            str(path),
        )
        quantizer = NvFp4Quantizer()
        state_dict = quantizer.load_and_quantize([str(path)], None)  # stays on CPU
        assert state_dict["proj.weight"].device.type == "cpu"

        model = _TinyDiTBlock()
        apply_nvfp4_monkey_patch(
            model, state_dict, quantizer.nvfp4_module_shapes, [], use_scaled_mm=True, training=True,
            calc_device="cuda",
        )
        model.requires_grad_(False)
        model.load_state_dict(state_dict, strict=True, assign=True)

    # The computed columnwise buffers must land back on the row-wise weight's original device
    # (CPU), matching block swap's expectation that the resident state dict stays off-GPU.
    assert model.proj.nvfp4_weight_t.device.type == "cpu"
    assert model.proj.nvfp4_block_scale_t.device.type == "cpu"
    assert model.proj.nvfp4_scale_t.device.type == "cpu"
