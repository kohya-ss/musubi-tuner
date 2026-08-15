import argparse
import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from musubi_tuner.krea2_train_network import Krea2NetworkTrainer, krea2_setup_parser
from musubi_tuner.training.parser_common import read_config_from_file, setup_parser_common


class _FakeVAE:
    dtype = torch.bfloat16

    def to(self, _device):
        return self

    def eval(self):
        return self

    def decode_to_pixels(self, latent):
        return torch.zeros(
            (latent.shape[0], 3, latent.shape[-2] * 8, latent.shape[-1] * 8),
            device=latent.device,
            dtype=self.dtype,
        )


class _PreparedTransformer:
    def __init__(self, fill_value=0.0):
        self.calls = []
        self.fill_value = fill_value

    def __call__(self, **inputs):
        self.calls.append(inputs)
        return torch.full_like(inputs["img"], self.fill_value)


class _SamplingTransformer(_PreparedTransformer):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(patch=2, channels=1)


class _WrapperAwareAccelerator:
    device = torch.device("cpu")

    def __init__(self, prepared_transformer, raw_transformer):
        self.prepared_transformer = prepared_transformer
        self.raw_transformer = raw_transformer
        self.unwrap_calls = []

    def unwrap_model(self, model, keep_fp32_wrapper=False):
        self.unwrap_calls.append((model, keep_fp32_wrapper))
        assert model is self.prepared_transformer
        return self.raw_transformer

    @staticmethod
    def autocast():
        return nullcontext()


def _run_sample(monkeypatch, *, args=None, trainer=None, turbo_dit=None):
    network_module = importlib.import_module("musubi_tuner.krea2_train_network")
    observed_schedule = {}

    def fake_timesteps(seqlen, steps, x1, x2, y1, y2, mu=None):
        observed_schedule.update(
            seqlen=seqlen,
            steps=steps,
            x1=x1,
            x2=x2,
            y1=y1,
            y2=y2,
            mu=mu,
        )
        return [1.0, 0.0]

    monkeypatch.setattr(network_module.krea2_sampling, "timesteps", fake_timesteps)
    monkeypatch.setattr(network_module, "clean_memory_on_device", lambda _device: None)
    monkeypatch.setattr(network_module, "tqdm", lambda values, **_kwargs: values)

    model = _SamplingTransformer()
    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        unwrap_model=lambda transformer, keep_fp32_wrapper=False: transformer,
    )
    trainer = Krea2NetworkTrainer() if trainer is None else trainer
    args = SimpleNamespace(turbo_dit=turbo_dit, mixed_precision="no") if args is None else args

    pixels = trainer.do_inference(
        accelerator,
        args,
        {"krea2_vl_embed": torch.ones((2, 1, 4), dtype=torch.bfloat16)},
        _FakeVAE(),
        torch.float32,
        model,
        discrete_flow_shift=0,
        sample_steps=1,
        width=16,
        height=16,
        frame_count=1,
        generator=torch.Generator().manual_seed(1),
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        cfg_scale=1.0,
    )

    return observed_schedule, model, pixels


def test_lora_parser_does_not_expose_full_only_dit_variant():
    parser = krea2_setup_parser(argparse.ArgumentParser())

    assert not hasattr(parser.parse_args([]), "dit_variant")
    with pytest.raises(SystemExit):
        parser.parse_args(["--dit_variant", "turbo"])


def test_raw_primary_uses_resolution_aware_sampling_schedule(monkeypatch):
    schedule, _, _ = _run_sample(monkeypatch)

    assert schedule == {
        "seqlen": 1,
        "steps": 1,
        "x1": 256,
        "x2": 6400,
        "y1": 0.5,
        "y2": 1.15,
        "mu": None,
    }


def test_lora_injected_dit_variant_does_not_change_raw_primary_schedule(monkeypatch):
    args = SimpleNamespace(dit_variant="turbo", turbo_dit=None, mixed_precision="no")

    schedule, _, _ = _run_sample(monkeypatch, args=args)

    assert schedule["mu"] is None


def test_lora_toml_injected_dit_variant_does_not_change_raw_primary_schedule(tmp_path, monkeypatch):
    config_path = tmp_path / "lora.toml"
    config_path.write_text('[model]\ndit_variant = "turbo"\n', encoding="utf-8")
    parser = krea2_setup_parser(setup_parser_common())
    monkeypatch.setattr(
        "sys.argv",
        ["krea2_train_network.py", "--config_file", str(config_path)],
    )
    args = read_config_from_file(parser.parse_args(), parser)

    schedule, _, _ = _run_sample(monkeypatch, args=args)

    assert args.dit_variant == "turbo"
    assert schedule["mu"] is None


def test_sampling_schedule_uses_explicit_trainer_policy_hook(monkeypatch):
    class TurboScheduleTrainer(Krea2NetworkTrainer):
        def use_turbo_sampling_schedule(self, args):
            return True

    schedule, _, _ = _run_sample(monkeypatch, trainer=TurboScheduleTrainer())

    assert schedule["mu"] == 1.15


@pytest.mark.parametrize(("dit_variant", "expected_mu"), [("raw", None), ("turbo", 1.15)])
def test_full_primary_variant_selects_matching_sampling_schedule(monkeypatch, dit_variant, expected_mu):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    args = SimpleNamespace(dit_variant=dit_variant, turbo_dit=None, mixed_precision="no")

    schedule, _, _ = _run_sample(monkeypatch, args=args, trainer=full_train.Krea2Trainer())

    assert schedule["mu"] == expected_mu


def test_lora_raw_to_turbo_sample_path_keeps_fixed_sampling_mu(monkeypatch):
    schedule, _, _ = _run_sample(monkeypatch, turbo_dit="turbo.safetensors")

    assert schedule["mu"] == 1.15


def test_fp32_sampling_uses_raw_model_config_and_dit_dtype(monkeypatch):
    schedule, model, pixels = _run_sample(monkeypatch)

    assert schedule["mu"] is None
    assert len(model.calls) == 1
    assert model.calls[0]["img"].dtype is torch.float32
    assert model.calls[0]["context"].dtype is torch.float32
    assert pixels.dtype is torch.float32


def test_call_dit_reads_raw_config_but_forwards_through_prepared_wrapper():
    trainer = Krea2NetworkTrainer()
    prepared = _PreparedTransformer(fill_value=3.0)
    raw = SimpleNamespace(config=SimpleNamespace(patch=2))
    accelerator = _WrapperAwareAccelerator(prepared, raw)
    latents = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    noise = torch.ones_like(latents)

    output = trainer.call_dit(
        SimpleNamespace(gradient_checkpointing=False),
        accelerator,
        prepared,
        latents,
        {"latents": latents, "krea2_vl_embed": [torch.zeros((1, 1, 4), dtype=torch.float32)]},
        noise,
        torch.zeros_like(latents),
        torch.tensor([500.0]),
        torch.float32,
    )

    assert accelerator.unwrap_calls == [(prepared, False)]
    assert len(prepared.calls) == 1
    assert torch.equal(output.pred, torch.full_like(latents, 3.0))
    assert torch.equal(output.target, noise - latents)


@pytest.mark.parametrize("dit_variant", ["raw", "turbo"])
def test_full_parser_accepts_dit_variants(dit_variant):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    parser = full_train.krea2_full_setup_parser(krea2_setup_parser(argparse.ArgumentParser()))

    assert parser.parse_args(["--dit_variant", dit_variant]).dit_variant == dit_variant


def test_full_validator_rejects_unknown_dit_variant():
    full_train = importlib.import_module("musubi_tuner.krea2_train")

    with pytest.raises(ValueError, match="--dit_variant"):
        full_train.Krea2Trainer().validate_full_finetune_model_args(
            SimpleNamespace(dit_variant="typo", turbo_dit=None, turbo_dit_cache=False)
        )


def test_full_toml_injected_dit_variant_is_semantically_validated(tmp_path, monkeypatch):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    config_path = tmp_path / "full.toml"
    config_path.write_text('[model]\ndit_variant = "typo"\n', encoding="utf-8")
    parser = full_train.krea2_full_setup_parser(krea2_setup_parser(setup_parser_common()))
    monkeypatch.setattr(
        "sys.argv",
        ["krea2_train.py", "--config_file", str(config_path)],
    )
    args = read_config_from_file(parser.parse_args(), parser)

    assert args.dit_variant == "typo"
    with pytest.raises(ValueError, match="--dit_variant"):
        full_train.Krea2Trainer().validate_full_finetune_model_args(args)


@pytest.mark.parametrize(
    ("turbo_dit", "turbo_dit_cache", "rejected_option"),
    [
        ("turbo.safetensors", False, "--turbo_dit"),
        (None, True, "--turbo_dit_cache"),
    ],
)
def test_full_trainer_rejects_lora_only_turbo_swapping(turbo_dit, turbo_dit_cache, rejected_option):
    full_train = importlib.import_module("musubi_tuner.krea2_train")

    with pytest.raises(ValueError, match=rejected_option):
        full_train.Krea2Trainer().validate_full_finetune_model_args(
            SimpleNamespace(
                dit_variant="raw",
                turbo_dit=turbo_dit,
                turbo_dit_cache=turbo_dit_cache,
            )
        )


def test_full_loader_passes_trainable_dtype_directly(monkeypatch):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    loaded_model = object()
    loader_call = {}

    def fake_load_krea2_dit(path, **kwargs):
        loader_call.update(path=path, **kwargs)
        return loaded_model

    monkeypatch.setattr(full_train.krea2_utils, "load_krea2_dit", fake_load_krea2_dit)

    result = full_train.Krea2Trainer().load_full_finetune_transformer(
        SimpleNamespace(device=torch.device("cpu")),
        SimpleNamespace(fp8_scaled=False),
        "turbo.safetensors",
        "torch",
        False,
        torch.device("cpu"),
        torch.float32,
    )

    assert result is loaded_model
    assert loader_call == {
        "path": "turbo.safetensors",
        "device": torch.device("cpu"),
        "dtype": torch.float32,
        "fp8_scaled": False,
        "loading_device": torch.device("cpu"),
        "attn_mode": "torch",
        "split_attn": False,
    }


@pytest.mark.parametrize("dit_variant", ["raw", "turbo"])
def test_full_primary_variant_is_accepted_and_recorded_as_string_metadata(dit_variant):
    full_train = importlib.import_module("musubi_tuner.krea2_train")
    trainer = full_train.Krea2Trainer()
    args = SimpleNamespace(dit_variant=dit_variant, turbo_dit=None, turbo_dit_cache=False)

    trainer.validate_full_finetune_model_args(args)

    assert trainer.full_finetune_metadata(args) == {"ss_krea2_dit_variant": dit_variant}
