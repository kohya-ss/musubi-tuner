"""Tests for the pluggable loss function resolver (training/loss.py)."""

import sys
import types
from dataclasses import dataclass, field

import pytest
import torch

from musubi_tuner.training.loss import (
    BUILTIN_LOSS_FNS,
    LossContext,
    mse_loss,
    normalize_loss_output,
    parse_loss_fn_args,
    resolve_loss_fn,
)


@dataclass
class _FakeOutput:
    pred: torch.Tensor
    target: torch.Tensor
    extra: dict = field(default_factory=dict)


def _make_args(**overrides):
    ns = types.SimpleNamespace(weighting_scheme="none", loss_fn="mse", loss_fn_args=None)
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def _make_ctx(output, args=None, timesteps=None):
    return LossContext(
        args=args if args is not None else _make_args(),
        output=output,
        timesteps=timesteps if timesteps is not None else torch.tensor([100.0, 500.0]),
        noise_scheduler=None,
        dit_dtype=torch.float32,
        network_dtype=torch.float32,
        global_step=0,
    )


# --- parse_loss_fn_args ---


def test_parse_none_and_empty():
    assert parse_loss_fn_args(None) == {}
    assert parse_loss_fn_args([]) == {}


def test_parse_literal_values():
    kwargs = parse_loss_fn_args(["weight=0.1", "level=2", "normalize=True"])
    assert kwargs == {"weight": 0.1, "level": 2, "normalize": True}


def test_parse_bare_string_raises_with_quoting_hint():
    # transform=swt: "swt" is not a Python literal -> hard error at startup,
    # message shows exactly how to quote it (optimizer_args semantics)
    with pytest.raises(ValueError, match=r"transform='swt'"):
        parse_loss_fn_args(["transform=swt"])


def test_parse_quoted_string_value():
    assert parse_loss_fn_args(["transform='swt'"]) == {"transform": "swt"}


def test_parse_dict_value():
    kwargs = parse_loss_fn_args(["band_weights={'ll0': 1.0, 'lh2': 0.2}"])
    assert kwargs == {"band_weights": {"ll0": 1.0, "lh2": 0.2}}


def test_parse_numeric_typo_raises():
    # the whole point of strict parsing: a typo'd float must fail loudly,
    # not silently become the string "0.1."
    with pytest.raises(ValueError, match="not a Python literal"):
        parse_loss_fn_args(["alpha=0.1."])


def test_parse_missing_equals_raises():
    with pytest.raises(ValueError, match="key=value"):
        parse_loss_fn_args(["justakey"])


# --- built-in mse ---


def test_mse_matches_manual_computation():
    torch.manual_seed(0)
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    ctx = _make_ctx(_FakeOutput(pred=pred, target=target))

    loss, metrics = mse_loss(ctx)

    expected = torch.nn.functional.mse_loss(pred, target)
    assert torch.allclose(loss, expected)
    assert metrics == {}


# --- resolve_loss_fn ---


def test_resolve_builtin_mse():
    fn = resolve_loss_fn("mse")
    assert fn is BUILTIN_LOSS_FNS["mse"]


def test_resolve_unknown_builtin_lists_valid_names():
    with pytest.raises(ValueError, match="mse"):
        resolve_loss_fn("nope")


def test_resolve_bad_import_path():
    with pytest.raises(ImportError, match="no_such_module_xyz"):
        resolve_loss_fn("no_such_module_xyz.Loss")


def test_resolve_missing_attribute():
    with pytest.raises(ImportError, match="no_such_attr"):
        resolve_loss_fn("musubi_tuner.training.loss.no_such_attr")


def test_resolve_dotted_function_binds_kwargs():
    mod = types.ModuleType("fake_losses")

    def fn_loss(ctx, scale=1.0):
        return torch.nn.functional.mse_loss(ctx.output.pred, ctx.output.target) * scale

    mod.fn_loss = fn_loss
    sys.modules["fake_losses"] = mod
    try:
        fn = resolve_loss_fn("fake_losses.fn_loss", ["scale=2.0"])
        output = _FakeOutput(pred=torch.ones(2, 2), target=torch.zeros(2, 2))
        loss = fn(_make_ctx(output, timesteps=torch.tensor([1.0, 2.0])))
        assert torch.allclose(loss, torch.tensor(2.0))
    finally:
        del sys.modules["fake_losses"]


def test_resolve_dotted_class_instantiates_with_kwargs():
    mod = types.ModuleType("fake_losses_cls")

    class ClassLoss(torch.nn.Module):
        def __init__(self, scale=1.0):
            super().__init__()
            self.scale = scale

        def forward(self, ctx):
            loss = torch.nn.functional.mse_loss(ctx.output.pred, ctx.output.target) * self.scale
            return loss, {"loss/scaled": float(loss)}

    mod.ClassLoss = ClassLoss
    sys.modules["fake_losses_cls"] = mod
    try:
        fn = resolve_loss_fn("fake_losses_cls.ClassLoss", ["scale=3.0"])
        assert isinstance(fn, torch.nn.Module)
        assert fn.scale == 3.0
    finally:
        del sys.modules["fake_losses_cls"]


# --- normalize_loss_output ---


def test_normalize_bare_tensor():
    t = torch.tensor(1.5)
    loss, metrics = normalize_loss_output(t)
    assert loss is t
    assert metrics == {}


def test_normalize_tuple():
    t = torch.tensor(1.5)
    loss, metrics = normalize_loss_output((t, {"loss/aux": 0.5}))
    assert loss is t
    assert metrics == {"loss/aux": 0.5}


# --- CLI args ---


def test_parser_defaults_and_values():
    from musubi_tuner.training.parser_common import setup_parser_common

    parser = setup_parser_common()
    args, _ = parser.parse_known_args([])
    assert args.loss_fn == "mse"
    assert args.loss_fn_args is None

    args, _ = parser.parse_known_args(
        ["--loss_fn", "wavelet_loss.musubi.WaveletPlusMSE", "--loss_fn_args", "alpha=0.1", "transform='swt'"]
    )
    assert args.loss_fn == "wavelet_loss.musubi.WaveletPlusMSE"
    assert args.loss_fn_args == ["alpha=0.1", "transform='swt'"]


# --- trainer integration ---


def _make_trainer():
    from musubi_tuner.training.trainer_base import NetworkTrainer

    return NetworkTrainer()


def test_default_compute_loss_matches_legacy_mse():
    torch.manual_seed(0)
    from musubi_tuner.training.trainer_base import DiTOutput

    trainer = _make_trainer()
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)
    args = _make_args()  # loss_fn="mse", weighting_scheme="none"
    trainer._resolved_loss_fn = None  # simulate startup resolution not yet run
    loss, metrics = trainer.compute_loss(
        args, DiTOutput(pred=pred, target=target), torch.tensor([100.0, 500.0]), None, torch.float32, torch.float32, 0
    )
    assert torch.allclose(loss, torch.nn.functional.mse_loss(pred, target))
    assert metrics == {}


def test_compute_loss_uses_custom_function_and_normalizes_bare_tensor():
    from musubi_tuner.training.trainer_base import DiTOutput

    mod = types.ModuleType("fake_losses_trainer")

    def bare_loss(ctx):
        return torch.nn.functional.l1_loss(ctx.output.pred, ctx.output.target)  # bare tensor, no metrics

    mod.bare_loss = bare_loss
    sys.modules["fake_losses_trainer"] = mod
    try:
        trainer = _make_trainer()
        args = _make_args(loss_fn="fake_losses_trainer.bare_loss")
        pred, target = torch.ones(2, 2), torch.zeros(2, 2)
        loss, metrics = trainer.compute_loss(
            args, DiTOutput(pred=pred, target=target), torch.tensor([1.0, 2.0]), None, torch.float32, torch.float32, 0
        )
        assert torch.allclose(loss, torch.tensor(1.0))
        assert metrics == {}
    finally:
        del sys.modules["fake_losses_trainer"]


def test_compute_loss_resolves_once_and_caches():
    trainer = _make_trainer()
    from musubi_tuner.training.trainer_base import DiTOutput

    calls = []
    trainer._resolved_loss_fn = lambda *a: (calls.append(1) or torch.tensor(0.0), {})
    trainer.compute_loss(
        _make_args(), DiTOutput(pred=torch.zeros(1), target=torch.zeros(1)), torch.tensor([1.0]), None, torch.float32, torch.float32, 0
    )
    assert calls == [1]  # pre-set callable used, not re-resolved


def test_process_batch_passes_call_dit_extras_to_loss():
    # Trainers that support extra-input losses (wavelet x0 recovery, DiNO/PO)
    # stash tensors into output.extra themselves (e.g. in call_dit); the base
    # process_batch must hand that dict through to compute_loss untouched.
    from musubi_tuner.training.trainer_base import DiTOutput, NetworkTrainer

    captured = {}

    class StubTrainer(NetworkTrainer):
        def get_noisy_model_input_and_timesteps(self, args, noise, latents, timesteps, noise_scheduler, device, dtype):
            return latents + noise, torch.tensor([500.0])

        def call_dit(self, args, accelerator, transformer, latents, batch, noise, noisy_model_input, timesteps, network_dtype, **kwargs):
            output = DiTOutput(pred=torch.zeros_like(latents), target=torch.zeros_like(latents))
            output.extra["noisy_model_input"] = noisy_model_input
            return output

        def compute_loss(self, args, output, timesteps, noise_scheduler, dit_dtype, network_dtype, global_step):
            captured.update(output.extra)
            return torch.tensor(0.0), {}

    class _FakeAccelerator:
        device = torch.device("cpu")

    latents = torch.ones(1, 4)
    noise = torch.full((1, 4), 2.0)
    StubTrainer().process_batch(
        _make_args(), _FakeAccelerator(), None, None, {"timesteps": None}, latents, noise, None,
        torch.float32, torch.float32, None, 0,
    )
    assert set(captured) == {"noisy_model_input"}  # only what the trainer stashed
    assert torch.equal(captured["noisy_model_input"], latents + noise)


def test_validate_args_resolves_loss_fn_at_startup():
    # NetworkTrainer.handle_model_specific_args (called later in
    # _validate_args_and_init) is an abstract hook that unconditionally raises
    # NotImplementedError on the base class -- there's no reasonable minimal
    # args namespace that gets a bare NetworkTrainer instance all the way
    # through _validate_args_and_init. Per the task note, resolution is
    # factored into `_resolve_loss_fn_from_args`, called eagerly (before
    # `handle_model_specific_args`) from `_validate_args_and_init`; test the
    # helper directly, which is what actually performs the startup resolution.
    trainer = _make_trainer()
    args = _make_args(loss_fn="mse")
    trainer._resolve_loss_fn_from_args(args)
    assert trainer._resolved_loss_fn is BUILTIN_LOSS_FNS["mse"]

    trainer2 = _make_trainer()
    args.loss_fn = "nonexistent_module_xyz.Loss"
    with pytest.raises(ImportError):
        trainer2._resolve_loss_fn_from_args(args)

    # confirm the real call site: _validate_args_and_init resolves before it
    # reaches the abstract handle_model_specific_args hook (proves ordering,
    # not just that the helper works in isolation).
    trainer3 = _make_trainer()
    args3 = _make_args(
        loss_fn="mse",
        cuda_allow_tf32=False,
        cuda_cudnn_benchmark=False,
        dataset_config="dummy.toml",
        dit="dummy.safetensors",
        fp8_scaled=False,
        fp8_base=False,
        sage_attn=False,
        disable_numpy_memmap=False,
    )
    with pytest.raises(NotImplementedError, match="handle_model_specific_args"):
        trainer3._validate_args_and_init(args3)
    assert trainer3._resolved_loss_fn is BUILTIN_LOSS_FNS["mse"]


def test_trainer_builds_loss_context():
    from musubi_tuner.training.trainer_base import DiTOutput

    trainer = _make_trainer()
    seen = {}

    def capture(ctx):
        assert isinstance(ctx, LossContext)
        seen.update(vars(ctx))
        return torch.tensor(0.0)

    trainer._resolved_loss_fn = capture
    args = _make_args()
    output = DiTOutput(pred=torch.zeros(1), target=torch.zeros(1))
    ts = torch.tensor([42.0])
    trainer.compute_loss(args, output, ts, "sched", torch.float16, torch.bfloat16, 7)

    assert seen["args"] is args
    assert seen["output"] is output
    assert seen["timesteps"] is ts
    assert seen["noise_scheduler"] == "sched"
    assert seen["dit_dtype"] is torch.float16
    assert seen["network_dtype"] is torch.bfloat16
    assert seen["global_step"] == 7
