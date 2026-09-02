"""Tests for NVFP4 training support: columnwise weight requantization and the
autograd-enabled forward/backward (NvFp4LinearFn).

CPU tests exercise pure-tensor quantization math (no scaled_mm). GPU tests
requiring real FP4 tensor-core execution are marked ``requires_nvfp4_scaled_mm``.
"""

import pytest
import torch
from torch import nn

from musubi_tuner.modules.nvfp4_utils import (
    NVFP4_STREAM_QUANT_BUFFER_NAMES,
    NvFp4LinearFn,
    NvFp4Quantizer,
    _quantize_nvfp4_2d,
    _quantize_nvfp4_2d_chunked,
    _roundup,
    apply_nvfp4_monkey_patch,
    block_has_nvfp4_patched_linear,
    dequantize_nvfp4,
    nvfp4_linear_forward_patch,
    nvfp4_linear_forward_patch_autograd,
    nvfp4_scaled_mm_available,
    nvfp4_scaled_mm_linear,
    nvfp4_swap_tensor_selector,
    quantize_nvfp4_activation,
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

    packed_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k))

    assert packed_t.dtype is torch.uint8
    assert packed_t.shape == (k, n // 2)
    assert block_scale_t.dtype is torch.float8_e4m3fn
    assert tensor_scale_t.dtype is torch.float32
    assert tensor_scale_t.ndim == 0


def test_quantize_nvfp4_weight_columnwise_roundtrip_matches_rowwise():
    n, k = 64, 32
    w, packed, block_scale, tensor_scale = _make_quantized_weight(n, k)

    packed_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k))

    w_deq = dequantize_nvfp4(packed, block_scale, tensor_scale, (n, k), torch.float32)
    w_t_deq = dequantize_nvfp4(packed_t, block_scale_t, tensor_scale_t, (k, n), torch.float32).t()

    # Both are independent NVFP4 quantizations of the same underlying bf16-scale weight,
    # grouped along different axes -- they should each track the original within normal
    # FP4 quantization noise, and therefore track each other within roughly double that.
    rel_err_to_original = (w_deq - w).norm() / w.norm()
    rel_err_between = (w_deq - w_t_deq).norm() / w_deq.norm()
    assert rel_err_to_original < 0.2
    assert rel_err_between < 0.3


@pytest.mark.parametrize(
    "n,k",
    [
        pytest.param(1536, 6144, id="N<K-krea2-attn-wk-wv-shape"),
        pytest.param(6144, 1536, id="N>K"),
        pytest.param(2048, 2048, id="N==K"),
        pytest.param(48, 80, id="non-power-of-two-16-multiples"),
    ],
)
def test_quantize_nvfp4_weight_columnwise_roundtrip_matches_rowwise_across_shapes(n, k):
    w, packed, block_scale, tensor_scale = _make_quantized_weight(n, k)

    packed_t, block_scale_t, tensor_scale_t = quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k))

    w_deq = dequantize_nvfp4(packed, block_scale, tensor_scale, (n, k), torch.float32)
    w_t_deq = dequantize_nvfp4(packed_t, block_scale_t, tensor_scale_t, (k, n), torch.float32).t()

    rel_err_to_original = (w_deq - w).norm() / w.norm()
    rel_err_between = (w_deq - w_t_deq).norm() / w_deq.norm()
    assert rel_err_to_original < 0.2
    assert rel_err_between < 0.3


def test_quantize_nvfp4_weight_columnwise_rejects_non_multiple_of_block_size():
    bad_n, k = 50, 32  # bad_n=50 is NOT a multiple of 16 -- this is what triggers the check
    _w, packed, block_scale, tensor_scale = _make_quantized_weight(64, k)
    with pytest.raises(ValueError, match="out_features"):
        quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (bad_n, k))


def test_quantize_nvfp4_2d_chunked_matches_unchunked():
    torch.manual_seed(1)
    # rows deliberately not a multiple of chunk_rows, to exercise the padded tail chunk
    x = torch.randn(550, 64) * 0.03

    packed_ref, scale_ref, tensor_scale_ref, orig_rows_ref = _quantize_nvfp4_2d(x)
    packed_chunked, scale_chunked, tensor_scale_chunked, orig_rows_chunked = _quantize_nvfp4_2d_chunked(x, chunk_rows=128)

    assert orig_rows_chunked == orig_rows_ref == 550
    assert torch.equal(tensor_scale_chunked, tensor_scale_ref)
    assert torch.equal(packed_chunked, packed_ref)
    assert torch.equal(scale_chunked, scale_ref)


def test_quantize_nvfp4_2d_chunked_rejects_non_positive_chunk_rows():
    x = torch.randn(32, 64)
    with pytest.raises(ValueError, match="positive multiple of 128"):
        _quantize_nvfp4_2d_chunked(x, chunk_rows=0)
    with pytest.raises(ValueError, match="positive multiple of 128"):
        _quantize_nvfp4_2d_chunked(x, chunk_rows=-128)


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
    n, k, m = 64, 32, 8
    device = "cuda"
    w, x, b, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t = _make_linear_fixture(
        n, k, m, device, bias=True
    )

    out = NvFp4LinearFn.apply(x, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t, b, n, k)
    expected = nvfp4_scaled_mm_linear(x, packed, block_scale, tensor_scale, b, n)

    assert torch.equal(out, expected)


@requires_nvfp4_scaled_mm
def test_nvfp4_linear_fn_backward_grad_x_matches_bf16_dequant_reference():
    n, k, m = 64, 32, 8
    device = "cuda"
    w, x, b, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t = _make_linear_fixture(
        n, k, m, device, bias=False
    )
    x_fp4 = x.clone().requires_grad_(True)
    x_ref = x.clone().requires_grad_(True)

    out = NvFp4LinearFn.apply(x_fp4, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t, None, n, k)
    out.sum().backward()

    w_deq = dequantize_nvfp4(packed, block_scale, tensor_scale, (n, k), torch.bfloat16)
    ref_out = torch.nn.functional.linear(x_ref, w_deq)
    ref_out.sum().backward()

    rel_err = (x_fp4.grad.float() - x_ref.grad.float()).norm() / x_ref.grad.float().norm()
    assert rel_err < 0.3  # two independently FP4-quantized paths (fwd weight vs bwd weight), not exact


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
        model,
        state_dict,
        quantizer.nvfp4_module_shapes,
        [],
        use_scaled_mm=True,
        training=True,
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
                model,
                state_dict,
                quantizer.nvfp4_module_shapes,
                [],
                use_scaled_mm=False,
                training=True,
            )


@requires_nvfp4_scaled_mm
def test_training_patched_forward_and_backward_run_end_to_end():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_path_str:
        model, _weight = _build_training_patched_model(Path(tmp_path_str))
    model = model.cuda()

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
            model,
            state_dict,
            quantizer.nvfp4_module_shapes,
            [],
            use_scaled_mm=True,
            training=True,
            calc_device="cuda",
        )
        model.requires_grad_(False)
        model.load_state_dict(state_dict, strict=True, assign=True)

    # The computed columnwise buffers must land back on the row-wise weight's original device
    # (CPU), matching block swap's expectation that the resident state dict stays off-GPU.
    assert model.proj.nvfp4_weight_t.device.type == "cpu"
    assert model.proj.nvfp4_block_scale_t.device.type == "cpu"
    assert model.proj.nvfp4_scale_t.device.type == "cpu"


@requires_nvfp4_scaled_mm
def test_quantize_nvfp4_weight_columnwise_chunked_bounds_transient_peak():
    """Reviewer measured ~4.9GB transient peak for a single unchunked call on Krea2's largest
    Linear (24576x6144 mlp gate/up, 151M elements, 72MB packed). The default chunk_rows=1024 is a
    comfortable fixed value (not a computed bound) that keeps this shape's peak well under the
    unchunked baseline -- see test_quantize_nvfp4_weight_columnwise_smaller_chunk_rows_tightens_peak
    for how --nvfp4_columnwise_chunk_rows lets a memory-constrained GPU go even tighter."""
    device = torch.device("cuda")
    n, k = 24576, 6144
    w, packed, block_scale, tensor_scale = _make_quantized_weight(n, k)
    packed, block_scale, tensor_scale = packed.to(device), block_scale.to(device), tensor_scale.to(device)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k), chunk_rows=1024)
    torch.cuda.synchronize()
    chunked_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # chunk_rows >= k (weight_t's row count, [K, N]) makes _quantize_nvfp4_2d_chunked process
    # the whole transposed weight in a single chunk -- equivalent to the old unchunked path.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k), chunk_rows=_roundup(k, 128))
    torch.cuda.synchronize()
    unchunked_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    # Calibrated from an isolated measurement on this shape: chunked (chunk_rows=1024) peaked at
    # ~1446MiB, unchunked (chunk_rows>=k) peaked at ~4986MiB (ratio ~3.45x). Thresholds below carry
    # ~15-20% headroom over those measured numbers.
    assert chunked_peak_mb < 1700, f"chunked columnwise requant peaked at {chunked_peak_mb:.0f}MiB, expected < 1700MiB"
    assert unchunked_peak_mb > chunked_peak_mb * 2.9, (
        f"expected unchunked (chunk_rows>=k, i.e. one chunk covering the whole weight) to peak"
        f" well above chunked ({chunked_peak_mb:.0f}MiB), got {unchunked_peak_mb:.0f}MiB"
    )


@requires_nvfp4_scaled_mm
def test_quantize_nvfp4_weight_columnwise_smaller_chunk_rows_tightens_peak():
    """A configured chunk_rows well below k (e.g. 512, vs k=6144) reaches a tighter peak on
    Krea2's largest Linear shape than the default 1024 (see the sibling test) -- this is what
    --nvfp4_columnwise_chunk_rows exists to let a memory-constrained GPU opt into."""
    device = torch.device("cuda")
    n, k = 24576, 6144
    w, packed, block_scale, tensor_scale = _make_quantized_weight(n, k)
    packed, block_scale, tensor_scale = packed.to(device), block_scale.to(device), tensor_scale.to(device)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    quantize_nvfp4_weight_columnwise(packed, block_scale, tensor_scale, (n, k), chunk_rows=512)
    torch.cuda.synchronize()
    tight_peak_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    assert tight_peak_mb < 1300, f"chunk_rows=512 columnwise requant peaked at {tight_peak_mb:.0f}MiB, expected < 1300MiB"


def test_apply_nvfp4_monkey_patch_forwards_columnwise_chunk_rows(monkeypatch, tmp_path):
    from safetensors.torch import save_file

    from musubi_tuner.modules import nvfp4_utils

    torch.manual_seed(0)
    n, k = 64, 32
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
    state_dict = quantizer.load_and_quantize([str(path)], None)
    model = _TinyDiTBlock()

    seen_chunk_rows = []
    original = nvfp4_utils.quantize_nvfp4_weight_columnwise

    def spy(*args, **kwargs):
        seen_chunk_rows.append(kwargs.get("chunk_rows"))
        return original(*args, **kwargs)

    monkeypatch.setattr(nvfp4_utils, "quantize_nvfp4_weight_columnwise", spy)
    apply_nvfp4_monkey_patch(
        model,
        state_dict,
        quantizer.nvfp4_module_shapes,
        [],
        use_scaled_mm=True,
        training=True,
        columnwise_chunk_rows=128,
    )

    assert seen_chunk_rows == [128]


def test_apply_nvfp4_monkey_patch_rejects_non_positive_columnwise_chunk_rows(tmp_path):
    from safetensors.torch import save_file

    torch.manual_seed(0)
    n, k = 64, 32
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
    state_dict = quantizer.load_and_quantize([str(path)], None)
    model = _TinyDiTBlock()

    with pytest.raises(ValueError, match="columnwise_chunk_rows"):
        apply_nvfp4_monkey_patch(
            model,
            state_dict,
            quantizer.nvfp4_module_shapes,
            [],
            use_scaled_mm=True,
            training=True,
            columnwise_chunk_rows=0,
        )


@requires_nvfp4_scaled_mm
def test_nvfp4_scaled_mm_linear_uses_custom_activation_quantize_fn():
    """nvfp4_scaled_mm_linear must dispatch through activation_quantize_fn when given one
    (needed so NvFp4LinearFn.backward can route grad_out through stochastic rounding), not
    hardcode quantize_nvfp4_activation."""

    torch.manual_seed(0)
    n, k, m = 32, 32, 16
    w = torch.randn(n, k, device="cuda") * 0.02
    _w_packed, w_block_scale, w_tensor_scale, _ = _quantize_nvfp4_2d(w)
    x = torch.randn(m, k, device="cuda")

    calls = []

    def spy(x_arg):
        calls.append(x_arg)
        return quantize_nvfp4_activation(x_arg)

    nvfp4_scaled_mm_linear(x, _w_packed, w_block_scale, w_tensor_scale, None, n, activation_quantize_fn=spy)

    assert len(calls) == 1
    assert calls[0] is x


@requires_nvfp4_scaled_mm
def test_quantize_nvfp4_activation_stochastic_dispatches_to_triton_when_available(monkeypatch):
    """quantize_nvfp4_activation_stochastic must actually call the fused Triton kernel when
    available, not silently keep using the eager row-chunked path."""
    from musubi_tuner.modules import nvfp4_kernels
    from musubi_tuner.modules.nvfp4_utils import quantize_nvfp4_activation_stochastic

    torch.manual_seed(0)
    x = torch.randn(64, 64, device="cuda")

    calls = []
    original_fn = nvfp4_kernels.triton_quantize_nvfp4_stochastic

    def spy(x_arg, scale_arg, seed):
        calls.append((x_arg, scale_arg, seed))
        return original_fn(x_arg, scale_arg, seed)

    monkeypatch.setattr(nvfp4_kernels, "triton_quantize_nvfp4_stochastic", spy)

    quantize_nvfp4_activation_stochastic(x)

    assert len(calls) == 1, "quantize_nvfp4_activation_stochastic did not dispatch to the fused Triton kernel"


@requires_nvfp4_scaled_mm
def test_nvfp4_linear_fn_backward_uses_stochastic_rounding_for_grad_out(monkeypatch):
    """NvFp4LinearFn.backward must quantize grad_out via quantize_nvfp4_activation_stochastic,
    not the deterministic quantize_nvfp4_activation -- per
    docs/superpowers/specs/2026-09-01-nvfp4-dgrad-stochastic-rounding-design.md."""
    from musubi_tuner.modules import nvfp4_utils

    torch.manual_seed(0)
    n, k = 32, 32
    w = torch.randn(n, k, device="cuda") * 0.02
    w_packed, w_block_scale, w_tensor_scale, _ = _quantize_nvfp4_2d(w)
    w_t_packed, w_t_block_scale, w_t_tensor_scale = quantize_nvfp4_weight_columnwise(
        w_packed, w_block_scale, w_tensor_scale, (n, k)
    )

    calls = []
    original_fn = nvfp4_utils.quantize_nvfp4_activation_stochastic

    def spy(x_arg):
        calls.append(x_arg)
        return original_fn(x_arg)

    monkeypatch.setattr(nvfp4_utils, "quantize_nvfp4_activation_stochastic", spy)

    x = torch.randn(8, k, device="cuda", requires_grad=True)
    out = NvFp4LinearFn.apply(x, w_packed, w_block_scale, w_tensor_scale, w_t_packed, w_t_block_scale, w_t_tensor_scale, None, n, k)
    out.sum().backward()

    assert len(calls) == 1, "NvFp4LinearFn.backward did not dispatch grad_out through stochastic rounding"


@requires_nvfp4_scaled_mm
def test_nvfp4_linear_fn_backward_uses_orig_in_features_not_padded_weight_t_shape():
    """backward must use the real K passed in at forward time, not weight_t_packed.shape[0].
    weight_t_packed is constructed here with 64 rows while the real K is 32, so the two values
    can only match if orig_in_features is threaded through and used directly (K itself must be
    a multiple of 32 -- scaled_mm's own GEMM requires that alignment, independent of the
    16-row block padding this test targets)."""

    device = "cuda"
    n, m = 64, 8
    real_k = 32

    torch.manual_seed(2)
    w = (torch.randn(n, real_k, device=device) * 0.02).to(torch.bfloat16)
    x = (torch.randn(m, real_k, device=device) * 0.5).to(torch.bfloat16).requires_grad_(True)
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(w.float())

    # weight_t_packed constructed directly with 64 rows (independent of real_k=32), so
    # weight_t_packed.shape[0] == 64 != real_k.
    w_t = (torch.randn(64, n, device=device) * 0.02).to(torch.bfloat16)
    packed_t, block_scale_t, tensor_scale_t, chunked_orig_rows = _quantize_nvfp4_2d(w_t.float())
    assert chunked_orig_rows == 64

    out = NvFp4LinearFn.apply(
        x, packed, block_scale, tensor_scale, packed_t, block_scale_t, tensor_scale_t, None, n, real_k
    )
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.isfinite(x.grad).all()


def test_nvfp4_stream_quant_buffer_names_is_a_superset_of_both_call_sites():
    from musubi_tuner.minimax_h3.text_encoder import _TE_STREAM_QUANT_BUFFER_NAMES

    assert set(_TE_STREAM_QUANT_BUFFER_NAMES) <= set(NVFP4_STREAM_QUANT_BUFFER_NAMES)
    # Krea2's original inline list (pre-unification), preserved here as the other half of the
    # superset check so a future edit can't silently drop one side's buffer names.
    krea2_names = {"nvfp4_block_scale", "nvfp4_scale", "nvfp4_weight_t", "nvfp4_block_scale_t", "nvfp4_scale_t"}
    assert krea2_names <= set(NVFP4_STREAM_QUANT_BUFFER_NAMES)
    # Both call sites must import the SAME object (identity, not just equal contents) --
    # otherwise a future edit to one list silently doesn't propagate to the other.
    assert _TE_STREAM_QUANT_BUFFER_NAMES is NVFP4_STREAM_QUANT_BUFFER_NAMES


def _build_patched_model_for_training_flag(tmp_path, training: bool, n=64, k=32):
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
        model,
        state_dict,
        quantizer.nvfp4_module_shapes,
        [],
        use_scaled_mm=training,
        training=training,
    )
    model.requires_grad_(False)
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model


def test_apply_nvfp4_monkey_patch_training_false_registers_no_backward_buffers():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_path_str:
        model = _build_patched_model_for_training_flag(Path(tmp_path_str), training=False)

    assert "nvfp4_weight_t" not in model.proj._buffers
    assert "nvfp4_block_scale_t" not in model.proj._buffers
    assert "nvfp4_scale_t" not in model.proj._buffers
    assert model.proj.forward.__func__ is nvfp4_linear_forward_patch


def test_apply_nvfp4_monkey_patch_training_true_registers_backward_buffers_and_autograd_forward():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_path_str:
        model = _build_patched_model_for_training_flag(Path(tmp_path_str), training=True)

    assert "nvfp4_weight_t" in model.proj._buffers
    assert "nvfp4_block_scale_t" in model.proj._buffers
    assert "nvfp4_scale_t" in model.proj._buffers
    assert model.proj.forward.__func__ is nvfp4_linear_forward_patch_autograd


@requires_nvfp4_scaled_mm
def test_apply_nvfp4_monkey_patch_state_dict_keys_available_before_loading_device_move():
    """Regression for the loading_device != 'cpu' ordering invariant relied on by
    krea2_utils.load_krea2_dit: apply_nvfp4_monkey_patch must insert its nvfp4_*_t keys into the
    state dict BEFORE load_krea2_dit's own post-patch move-to-loading_device loop runs, or those
    keys would be silently left on calc_device instead of following the rest of the state dict.
    This test mirrors that loop exactly against the same mutated state dict apply_nvfp4_monkey_patch
    produces, without needing a full SingleStreamDiT checkpoint."""
    import tempfile
    from pathlib import Path

    from safetensors.torch import save_file

    with tempfile.TemporaryDirectory() as tmp_path_str:
        tmp_path = Path(tmp_path_str)
        torch.manual_seed(0)
        n, k = 64, 32
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
        state_dict = quantizer.load_and_quantize([str(path)], None)  # loads to CPU
        model = _TinyDiTBlock()
        apply_nvfp4_monkey_patch(
            model,
            state_dict,
            quantizer.nvfp4_module_shapes,
            [],
            use_scaled_mm=True,
            training=True,
            calc_device="cuda",
        )

        # Mirrors krea2_utils.load_krea2_dit's post-patch move-to-loading_device loop exactly:
        #   if loading_device.type != "cpu":
        #       for key in sd.keys(): sd[key] = sd[key].to(loading_device)
        loading_device = torch.device("cuda")
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(loading_device)

    assert state_dict["proj.nvfp4_weight_t"].device.type == "cuda"
    assert state_dict["proj.nvfp4_block_scale_t"].device.type == "cuda"
    assert state_dict["proj.nvfp4_scale_t"].device.type == "cuda"


@requires_nvfp4_scaled_mm
def test_lora_gradient_flows_through_nvfp4_linear_end_to_end():
    import tempfile
    from pathlib import Path

    from musubi_tuner.networks.lora import LoRAModule

    with tempfile.TemporaryDirectory() as tmp_path_str:
        model, _weight = _build_training_patched_model(Path(tmp_path_str))
    model = model.cuda()

    lora = LoRAModule("lora_proj", model.proj, multiplier=1.0, lora_dim=4, alpha=4).cuda()
    lora.apply_to()

    x = (torch.randn(4, 32, device="cuda") * 0.5).to(torch.bfloat16).requires_grad_(True)
    out = model(x)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert lora.lora_down.weight.grad is not None
    assert torch.isfinite(lora.lora_down.weight.grad).all()
    assert lora.lora_up.weight.grad is not None
    assert torch.isfinite(lora.lora_up.weight.grad).all()
    # The frozen NVFP4 packed weight never requires grad -- accumulating one would mean the
    # base was accidentally left trainable instead of frozen-with-a-LoRA-adapter.
    assert model.proj.weight.grad is None


def test_nvfp4_swap_tensor_selector_follows_only_forward_buffers_under_training_false():
    """Covers block_has_nvfp4_patched_linear and nvfp4_swap_tensor_selector directly (not
    krea2_mmdit.SingleStreamDiT.enable_block_swap's substitution wiring) against a model built
    with training=False. Under training=False the selector must follow only the forward buffers
    (nvfp4_block_scale/nvfp4_scale) -- the training-only columnwise ones were never registered,
    so there's nothing stale to select."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp_path_str:
        model = _build_patched_model_for_training_flag(Path(tmp_path_str), training=False)

    assert block_has_nvfp4_patched_linear(model) is True

    jobs = nvfp4_swap_tensor_selector(model)
    job_names = {name for _module, name in jobs}
    assert "weight" in job_names
    assert "nvfp4_block_scale" in job_names
    assert "nvfp4_scale" in job_names
    assert "nvfp4_weight_t" not in job_names
    assert "nvfp4_block_scale_t" not in job_names
    assert "nvfp4_scale_t" not in job_names


def test_foreign_format_module_is_skipped_not_raised(tmp_path):
    import json

    import torch
    from safetensors.torch import save_file

    from musubi_tuner.modules.comfy_quant_utils import FORMAT_CONVROT_INT8
    from musubi_tuner.modules.nvfp4_utils import NvFp4Quantizer, _quantize_nvfp4_2d

    torch.manual_seed(0)
    nvfp4_weight = torch.randn(64, 32) * 0.02
    packed, block_scale, tensor_scale, _ = _quantize_nvfp4_2d(nvfp4_weight)
    nvfp4_payload = torch.tensor(list(b'{"format":"nvfp4"}'), dtype=torch.uint8)

    convrot_spec = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode("utf-8")
    convrot_payload = torch.tensor(list(convrot_spec), dtype=torch.uint8)

    path = tmp_path / "mixed.safetensors"
    save_file(
        {
            "nvfp4_proj.weight": packed,
            "nvfp4_proj.weight_scale": block_scale,
            "nvfp4_proj.weight_scale_2": tensor_scale,
            "nvfp4_proj.comfy_quant": nvfp4_payload,
            "convrot_proj.weight": torch.zeros(64, 256, dtype=torch.int8),
            "convrot_proj.weight_scale": torch.ones(64, 1, dtype=torch.float32),
            "convrot_proj.comfy_quant": convrot_payload,
        },
        str(path),
    )

    quantizer = NvFp4Quantizer(foreign_formats={FORMAT_CONVROT_INT8})
    state_dict = quantizer.load_and_quantize([str(path)], None)

    assert "nvfp4_proj.weight" in state_dict
    assert "nvfp4_proj.nvfp4_block_scale" in state_dict
    assert quantizer.nvfp4_module_shapes == {"nvfp4_proj": (64, 32)}
    assert "convrot_proj.weight" not in state_dict
    assert "convrot_proj.weight_scale" not in state_dict
    assert "convrot_proj.comfy_quant" not in state_dict
