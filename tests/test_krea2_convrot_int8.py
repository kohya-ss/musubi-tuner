"""Tests for ConvRot int8 base-weight quantization (Krea 2, Phase 1).

Covers the numerical contract of the custom autograd (forward parity vs the
dequantized reference, grad_x in both backward modes, LoRA branch gradient parity),
the eager fallback path, the streaming quantizer + monkey patch + meta-model
``load_state_dict(assign=True)`` round-trip (int8 Parameters cannot require grad),
and non-reentrant gradient checkpointing.

CPU tests exercise the eager fallback (exact math up to rounding); CUDA tests
exercise the fused Triton kernels when available.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from musubi_tuner.modules import convrot_int8_kernels as kernels
from musubi_tuner.modules.convrot_int8_utils import (
    CONVROT_GROUPSIZE,
    ConvRotInt8LinearFn,
    ConvRotInt8Quantizer,
    apply_convrot_int8_monkey_patch,
    quantize_weight_convrot,
)

GS = CONVROT_GROUPSIZE  # 256
K, N, M = 512, 96, 64  # in_features (2 groups), out_features, tokens

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
requires_triton = pytest.mark.skipif(not kernels.HAS_TRITON, reason="triton required")


def _make_quantized_linear(device, dtype, seed=0, bias=False):
    """Random weight -> ConvRot int8 (wq, scale) plus the exact dequantized reference."""
    torch.manual_seed(seed)
    w = torch.randn(N, K, device=device, dtype=dtype) * 0.02
    wq, ws = kernels.quantize_int8_convrot_weight(w, GS)
    w_deq = kernels.dequantize_int8_convrot_weight(wq, ws, GS).to(dtype)
    b = torch.randn(N, device=device, dtype=dtype) * 0.1 if bias else None
    return wq, ws, w_deq, b


def _relerr(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# ---------------------------------------------------------------------------
# quantization scope / applicability
# ---------------------------------------------------------------------------


def test_quantize_weight_convrot_skips_indivisible_and_non_2d():
    assert quantize_weight_convrot("x.weight", torch.randn(8, 300)) is None  # 300 % 256 != 0
    assert quantize_weight_convrot("x.weight", torch.randn(300)) is None  # not 2D
    result = quantize_weight_convrot("x.weight", torch.randn(8, K))
    assert result is not None
    wq, ws = result
    assert wq.dtype == torch.int8 and wq.shape == (8, K)
    assert ws.dtype == torch.float32 and ws.shape == (8, 1)


def test_quantize_dequant_roundtrip_error_is_small():
    torch.manual_seed(0)
    w = torch.randn(N, K) * 0.02
    wq, ws = kernels.quantize_int8_convrot_weight(w, GS)
    w_deq = kernels.dequantize_int8_convrot_weight(wq, ws, GS)
    # row-wise int8 uniform quantization noise: rms ~ absmax/(127*sqrt(3)) per element,
    # relerr ~7e-3 for gaussian weights
    assert _relerr(w_deq, w) < 1e-2


# ---------------------------------------------------------------------------
# eager fallback path (CPU): rotation math is exact up to float rounding
# ---------------------------------------------------------------------------


def test_forward_eager_cpu_matches_dequant_reference():
    wq, ws, w_deq, _ = _make_quantized_linear("cpu", torch.float32)
    x = torch.randn(M, K)
    y = ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
    y_ref = F.linear(x, w_deq)
    # (x H) @ (W H^T)^T == x @ W^T exactly; only float32 rounding remains
    assert _relerr(y, y_ref) < 1e-5


def test_forward_eager_cpu_with_bias():
    wq, ws, w_deq, b = _make_quantized_linear("cpu", torch.float32, bias=True)
    x = torch.randn(M, K)
    y = ConvRotInt8LinearFn.apply(x, wq, ws, b, GS, "bf16")
    assert _relerr(y, F.linear(x, w_deq, b)) < 1e-5


def test_backward_eager_cpu_matches_dequant_reference():
    wq, ws, w_deq, _ = _make_quantized_linear("cpu", torch.float32)
    g = torch.randn(M, N)

    x = torch.randn(M, K, requires_grad=True)
    ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16").backward(g)

    x_ref = x.detach().clone().requires_grad_(True)
    F.linear(x_ref, w_deq).backward(g)

    assert _relerr(x.grad, x_ref.grad) < 1e-5


def test_forward_autocast_casts_fp32_input_like_f_linear():
    # K2's fp32 modulation adds promote activations to fp32; under autocast F.linear
    # casts them back to the autocast dtype. The patched forward must match, or
    # downstream flash-attn (fp16/bf16 only) breaks and attention silently runs fp32.
    wq, ws, w_deq, _ = _make_quantized_linear("cpu", torch.float32)
    x = torch.randn(M, K, dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        y = ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
        y_ref = F.linear(x, w_deq)
    assert y.dtype == torch.bfloat16
    assert y_ref.dtype == torch.bfloat16
    assert _relerr(y, y_ref) < 2e-2  # bf16 rounding


# ---------------------------------------------------------------------------
# fused Triton path (CUDA)
# ---------------------------------------------------------------------------


@requires_cuda
@requires_triton
def test_forward_triton_matches_dequant_reference():
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    y = ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
    y_ref = F.linear(x, w_deq)
    # differs from the reference only by dynamic activation quantization
    assert y.dtype == torch.bfloat16
    assert _relerr(y, y_ref) < 3e-2


@requires_cuda
@requires_triton
@pytest.mark.parametrize("bwd_mode,tol", [("bf16", 2e-2), ("int8", 5e-2)])
def test_grad_x_matches_dequant_reference(bwd_mode, tol):
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, bwd_mode).backward(g)

    x_ref = x.detach().clone().requires_grad_(True)
    F.linear(x_ref, w_deq).backward(g)

    assert _relerr(x.grad, x_ref.grad) < tol


@requires_cuda
@requires_triton
def test_lora_branch_grads_match_dequant_reference():
    # LoRA A/B gradients do not depend on the base-branch numerics (the base is a
    # separate additive branch), so they must match the dequantized reference exactly.
    # B is random-init: with the standard B=0 init, grad_A would be mathematically 0.
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(1)
    lora_a = nn.Linear(K, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    lora_b = nn.Linear(8, N, bias=False, device="cuda", dtype=torch.bfloat16)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    (ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16") + lora_b(lora_a(x))).backward(g)
    ga, gb = lora_a.weight.grad.clone(), lora_b.weight.grad.clone()
    lora_a.weight.grad = lora_b.weight.grad = None

    x_ref = x.detach().clone().requires_grad_(True)
    (F.linear(x_ref, w_deq) + lora_b(lora_a(x_ref))).backward(g)

    assert _relerr(ga, lora_a.weight.grad) < 1e-6
    assert _relerr(gb, lora_b.weight.grad) < 1e-6


@requires_cuda
@requires_triton
def test_forward_autocast_cuda_fp32_input_returns_bf16():
    # same as the CPU autocast test but through the fused Triton path
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.float32, requires_grad=True)
    g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
    assert y.dtype == torch.bfloat16
    y.backward(g)
    assert x.grad is not None and x.grad.isfinite().all()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y_ref = F.linear(x.detach(), w_deq)
    assert _relerr(y, y_ref) < 3e-2


@requires_cuda
def test_triton_and_eager_forward_parity(monkeypatch):
    # The eager fallback skips activation quantization, so parity is bounded by the
    # activation quantization error of the triton path, not exact.
    if not kernels.HAS_TRITON:
        pytest.skip("triton required for the comparison")
    wq, ws, w_deq, _ = _make_quantized_linear("cuda", torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    y_triton = ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")
    monkeypatch.setattr("musubi_tuner.modules.convrot_int8_utils.HAS_TRITON", False)
    y_eager = ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")

    assert _relerr(y_triton, y_eager) < 3e-2


# ---------------------------------------------------------------------------
# streaming quantizer + monkey patch + meta-model load (the load_krea2_dit path)
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Mimics the Krea 2 layout: quantized block Linears, excluded norm-ish Linear,
    an indivisible Linear, and a non-target head."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.ModuleDict({"attn": nn.Linear(K, N, bias=False)}) for _ in range(2)])
        self.blocks_norm_proj = nn.Linear(K, N, bias=False)  # matches exclude pattern "norm"
        self.first = nn.Linear(48, K, bias=False)  # 48 % 256 != 0 -> left unquantized
        self.head = nn.Linear(K, 8, bias=False)  # not under "blocks." -> not targeted


def test_quantizer_patch_and_meta_load_state_dict_roundtrip(tmp_path):
    torch.manual_seed(0)
    model = _TinyModel()
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    path = str(tmp_path / "tiny.safetensors")
    save_file(sd, path)

    quantizer = ConvRotInt8Quantizer(target_layer_keys=["blocks."], exclude_layer_keys=["norm"])
    qsd = quantizer.load_and_quantize([path], calc_device=None)

    # scope: block Linears quantized, excluded/indivisible/non-target left as-is
    assert qsd["blocks.0.attn.weight"].dtype == torch.int8
    assert qsd["blocks.0.attn.scale_weight"].dtype == torch.float32
    assert qsd["blocks.0.attn.scale_weight"].shape == (N, 1)
    assert qsd["blocks_norm_proj.weight"].dtype == torch.bfloat16
    assert "blocks_norm_proj.scale_weight" not in qsd
    assert qsd["first.weight"].dtype == torch.bfloat16  # indivisible in_features
    assert qsd["head.weight"].dtype == torch.bfloat16

    # meta build -> patch -> requires_grad_(False) -> load_state_dict(assign=True),
    # mirroring load_krea2_dit. Without requires_grad_(False) the int8 Parameter
    # re-wrap raises "Only Tensors of floating point ... can require gradients".
    with torch.device("meta"):
        fresh = _TinyModel()
    apply_convrot_int8_monkey_patch(fresh, qsd, bwd_mode="bf16")
    fresh.requires_grad_(False)
    fresh.load_state_dict(qsd, strict=True, assign=True)

    assert fresh.blocks[0]["attn"].weight.dtype == torch.int8
    assert not fresh.blocks[0]["attn"].weight.requires_grad

    # patched forward matches the dequantized reference (eager CPU path)
    x = torch.randn(4, K, dtype=torch.bfloat16)
    y = fresh.blocks[0]["attn"](x)
    w_deq = kernels.dequantize_int8_convrot_weight(
        qsd["blocks.0.attn.weight"], qsd["blocks.0.attn.scale_weight"], GS
    ).to(torch.bfloat16)
    assert _relerr(y, F.linear(x, w_deq)) < 2e-2  # bf16 rounding only

    # unpatched layers still behave as plain Linears
    assert _relerr(fresh.head(x), F.linear(x, qsd["head.weight"])) < 1e-2


def test_quantizer_rejects_prequantized_weights(tmp_path):
    sd = {"blocks.0.attn.weight": torch.zeros(N, K, dtype=torch.float8_e4m3fn)}
    path = str(tmp_path / "prequant.safetensors")
    save_file(sd, path)
    quantizer = ConvRotInt8Quantizer(target_layer_keys=["blocks."])
    with pytest.raises(ValueError, match="already"):
        quantizer.load_and_quantize([path], calc_device=None)


def test_monkey_patch_int8_bwd_requires_triton(monkeypatch):
    monkeypatch.setattr("musubi_tuner.modules.convrot_int8_utils.HAS_TRITON", False)
    with pytest.raises(ValueError, match="triton"):
        apply_convrot_int8_monkey_patch(nn.Module(), {}, bwd_mode="int8")


# ---------------------------------------------------------------------------
# gradient checkpointing (non-reentrant, as used by the Krea 2 blocks)
# ---------------------------------------------------------------------------


def test_nonreentrant_checkpoint_backward():
    wq, ws, w_deq, _ = _make_quantized_linear("cpu", torch.float32)

    def fn(x):
        return ConvRotInt8LinearFn.apply(x, wq, ws, None, GS, "bf16")

    g = torch.randn(M, N)
    x = torch.randn(M, K, requires_grad=True)
    y = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
    y.backward(g)

    x_ref = x.detach().clone().requires_grad_(True)
    fn(x_ref).backward(g)

    assert _relerr(x.grad, x_ref.grad) < 1e-6


# ---------------------------------------------------------------------------
# trainer flag validation
# ---------------------------------------------------------------------------


def _trainer_args(**overrides):
    base = dict(
        fp8_base=False,
        fp8_scaled=False,
        convrot_int8=False,
        convrot_int8_bwd="bf16",
        turbo_dit=None,
        turbo_dit_cache=False,
        blocks_to_swap=0,
        sample_prompts=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _handle_args(args):
    from musubi_tuner.krea2_train_network import Krea2NetworkTrainer

    trainer = Krea2NetworkTrainer()
    trainer.handle_model_specific_args(args)


def test_trainer_rejects_convrot_with_fp8():
    with pytest.raises(ValueError, match="convrot_int8"):
        _handle_args(_trainer_args(convrot_int8=True, fp8_base=True, fp8_scaled=True))


def test_trainer_rejects_convrot_with_turbo():
    with pytest.raises(ValueError, match="turbo"):
        _handle_args(_trainer_args(convrot_int8=True, turbo_dit="turbo.safetensors"))


def test_trainer_rejects_int8_bwd_without_convrot():
    with pytest.raises(ValueError, match="convrot_int8_bwd"):
        _handle_args(_trainer_args(convrot_int8_bwd="int8"))


def test_trainer_accepts_convrot_alone():
    _handle_args(_trainer_args(convrot_int8=True))
    _handle_args(_trainer_args(convrot_int8=True, convrot_int8_bwd="int8", blocks_to_swap=16))
