"""Experimental softmax attention is explicit and retains NR's inputs and priors."""

import importlib.util

import pytest
import torch

from musubi_tuner.dlssnr.config import build_train_config
from musubi_tuner.dlssnr.model import GlobalAttn, WindowAttn
from musubi_tuner.dlssnr.numerics import global_attention, window_attention
from test_dlssnr_config import make_args


def test_sdpa_core_and_shared_prior_gradients_match_explicit_softmax():
    from musubi_tuner.dlssnr.attention import attention

    torch.manual_seed(27)
    query, key, value = [(torch.randn(2, 3, 2, 64, 32) * 0.1).requires_grad_() for _ in range(3)]
    prior = (torch.randn(2, 64, 64) * 0.2).requires_grad_()
    expected = torch.softmax(query @ key.transpose(-1, -2) + prior, dim=-1) @ value
    actual = attention(query, key, value, backend="sdpa", prior=prior)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    first = torch.autograd.grad(expected.square().sum(), (query, key, value, prior), retain_graph=True)
    second = torch.autograd.grad(actual.square().sum(), (query, key, value, prior))
    torch.testing.assert_close(second, first, rtol=2e-5, atol=2e-6)
    assert torch.count_nonzero(second[-1])


def _identity_value_projections():
    qkv = torch.zeros(96, 32)
    qkv[64:] = torch.eye(32)
    return qkv, torch.eye(32)


def test_window_softmax_keeps_prior_and_out_of_field_tokens():
    qkv, proj = _identity_value_projections()
    prior = torch.zeros(1, 64, 64)
    prior[:, :, 0] = torch.log(torch.tensor(63.0))
    prior.requires_grad_()
    output = window_attention(torch.ones(1, 32, 1, 1), qkv, proj, prior, torch.ones(1), torch.zeros(32), 0, backend="sdpa")
    # One valid value gets half the mass; 63 outside zero values remain in normalization.
    torch.testing.assert_close(output, torch.full_like(output, 0.5), rtol=0, atol=0)
    output.sum().backward()
    assert prior.grad[0, 0, 0] > 0
    assert torch.all(prior.grad[0, 0, 1:] < 0)


def test_global_softmax_does_not_treat_alignment_padding_as_keys():
    qkv, proj = _identity_value_projections()
    source = torch.ones(1, 32, 3, 5)
    output = global_attention(source, qkv, proj, torch.ones(1), torch.zeros(32), backend="sdpa")
    # 15 real keys average to one. Treating 49 alignment zeros as keys gives 15/64 instead.
    torch.testing.assert_close(output, source, rtol=0, atol=0)


def test_global_scope_changes_global_attention_but_not_windows(tmp_path):
    from musubi_tuner.dlssnr.runtime import configure_model_runtime, runtime_policy

    torch.manual_seed(37)
    model = torch.nn.ModuleDict({"window": WindowAttn(32, 0), "global": GlobalAttn(32)})
    for name, parameter in model.named_parameters():
        if name.endswith(("temperature", "skip_scale")):
            torch.nn.init.ones_(parameter)
        else:
            torch.nn.init.normal_(parameter, std=0.1)
    source = torch.rand(1, 32, 3, 5)
    before = {name: module(source) for name, module in model.items()}
    config = build_train_config(
        make_args(tmp_path, ["--numerics_profile", "train_experimental", "--sdpa", "--attention_scope", "global"])
    )
    configure_model_runtime(model, runtime_policy(config), training=True)
    torch.testing.assert_close(model["window"](source), before["window"], rtol=0, atol=0)
    assert not torch.equal(model["global"](source), before["global"])


@pytest.mark.parametrize(
    "options,match",
    [
        (["--sdpa"], "train_experimental"),
        (["--numerics_profile", "train_experimental", "--mixed_precision", "bf16", "--flash_attn"], "global"),
        (["--numerics_profile", "train_experimental", "--flash_attn", "--attention_scope", "global"], "fp16|bf16"),
        (
            ["--numerics_profile", "train_experimental", "--mixed_precision", "bf16", "--sage_attn", "--attention_scope", "global"],
            "inference|backward",
        ),
    ],
)
def test_backend_policy_rejects_unsupported_training_combinations(tmp_path, options, match):
    with pytest.raises(ValueError, match=match):
        build_train_config(make_args(tmp_path, options))


def test_backend_switches_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        make_args(tmp_path, ["--sdpa", "--xformers"])


def test_optional_extension_failure_does_not_break_sdpa(monkeypatch):
    from musubi_tuner.dlssnr import attention as backend

    def unavailable(name):
        raise OSError("incompatible optional extension ABI")

    monkeypatch.setattr(backend.importlib, "import_module", unavailable)
    query = torch.randn(1, 2, 7, 32)
    actual = backend.attention(query, query, query, backend="sdpa")
    assert actual.shape == query.shape and torch.isfinite(actual).all()
    with pytest.raises(RuntimeError, match="xformers.*unavailable|xformers.*ABI"):
        backend.load_backend("xformers")


def test_sage_cannot_be_used_for_backward_even_outside_the_trainer():
    from musubi_tuner.dlssnr.attention import attention

    query = torch.ones(1, 2, 7, 32, requires_grad=True)
    with pytest.raises(RuntimeError, match="inference|backward"):
        attention(query, query, query, backend="sage_attn")


@pytest.mark.parametrize(
    "backend,package", [("flash_attn", "flash_attn"), ("xformers", "xformers"), ("sage_attn", "sageattention")]
)
def test_available_optional_cuda_kernel_forward_and_real_backward(backend, package):
    if not torch.cuda.is_available() or importlib.util.find_spec(package) is None:
        pytest.skip(f"optional {package} CUDA kernel is not available")
    from musubi_tuner.dlssnr.attention import attention

    values = [torch.randn(2, 2, 17, 32, device="cuda", dtype=torch.float16, requires_grad=backend != "sage_attn") for _ in range(3)]
    if backend == "sage_attn":
        with torch.no_grad():
            actual = attention(*values, backend=backend)
    else:
        actual = attention(*values, backend=backend)
        actual.square().mean().backward()
        assert all(
            value.grad is not None and torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad) for value in values
        )
    assert actual.dtype == torch.float32 and torch.isfinite(actual).all()
