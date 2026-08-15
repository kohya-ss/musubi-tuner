import argparse
import importlib
import sys

import pytest
import torch
from accelerate.utils import DistributedType

import musubi_tuner.training.full_finetune as full_finetune
from musubi_tuner.training.full_finetune import (
    TrainingProgress,
    add_full_finetune_args,
    normalize_compiled_state_dict,
    resolve_trainable_dtype,
    validate_full_finetune_args,
)


@pytest.mark.parametrize(
    "distributed_type",
    [
        DistributedType.DEEPSPEED,
        DistributedType.FSDP,
        DistributedType.TP,
        DistributedType.MEGATRON_LM,
        DistributedType.XLA,
    ],
)
def test_validate_full_finetune_distributed_type_rejects_non_replica_backends(distributed_type):
    with pytest.raises(ValueError) as error:
        full_finetune.validate_full_finetune_distributed_type(distributed_type)

    assert distributed_type.value in str(error.value)
    assert "DDP" in str(error.value)


@pytest.mark.parametrize(
    "distributed_type",
    [
        DistributedType.NO,
        DistributedType.MULTI_CPU,
        DistributedType.MULTI_GPU,
        DistributedType.MULTI_NPU,
        DistributedType.MULTI_MLU,
        DistributedType.MULTI_SDAA,
        DistributedType.MULTI_MUSA,
        DistributedType.MULTI_XPU,
        DistributedType.MULTI_HPU,
    ],
)
def test_validate_full_finetune_distributed_type_accepts_replica_backends(distributed_type):
    full_finetune.validate_full_finetune_distributed_type(distributed_type)


def test_training_progress_round_trip():
    progress = TrainingProgress(global_step=7, epoch=2, next_batch=3)
    restored = TrainingProgress()
    restored.load_state_dict(progress.state_dict())
    assert restored == progress


def test_normalize_compiled_state_dict_rejects_collision():
    state = {"blocks.0.weight": torch.ones(1), "blocks.0._orig_mod.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="collision"):
        normalize_compiled_state_dict(state)


def test_normalize_compiled_state_dict_removes_compile_wrappers():
    value = torch.ones(1)
    state = {"_orig_mod.blocks.0._orig_mod.weight": value}

    assert normalize_compiled_state_dict(state) == {"blocks.0.weight": value}


@pytest.mark.parametrize(
    ("full_bf16", "expected"),
    [(False, torch.float32), (True, torch.bfloat16)],
)
def test_resolve_trainable_dtype(full_bf16, expected):
    args = argparse.Namespace(full_bf16=full_bf16)

    assert resolve_trainable_dtype(args) == expected


def test_add_full_finetune_args_defaults_and_flags():
    parser = add_full_finetune_args(argparse.ArgumentParser())

    defaults = parser.parse_args([])
    assert defaults.full_bf16 is False
    assert defaults.fused_backward_pass is False
    assert defaults.mem_eff_save is False
    assert defaults.block_swap_optimizer_patch_params is False

    enabled = parser.parse_args(["--full_bf16", "--fused_backward_pass", "--mem_eff_save", "--block_swap_optimizer_patch_params"])
    assert enabled.full_bf16 is True
    assert enabled.fused_backward_pass is True
    assert enabled.mem_eff_save is True
    assert enabled.block_swap_optimizer_patch_params is True


def make_full_finetune_args(**overrides):
    defaults = {
        "network_weights": None,
        "network_module": None,
        "network_dim": None,
        "network_alpha": 1,
        "network_dropout": None,
        "network_args": None,
        "dim_from_weights": False,
        "scale_weight_norms": None,
        "base_weights": None,
        "base_weights_multiplier": None,
        "fp8_base": False,
        "fp8_scaled": False,
        "block_swap_h2d_only": False,
        "blocks_to_swap": None,
        "fused_backward_pass": False,
        "optimizer_type": "AdamW",
        "gradient_accumulation_steps": 1,
        "save_precision": None,
        "full_bf16": False,
        "full_fp16": False,
        "mixed_precision": None,
        "dit_dtype": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("network_weights", "lora.safetensors"),
        ("network_module", "networks.lora"),
        ("network_dim", 16),
        ("network_alpha", 2),
        ("network_dropout", 0.1),
        ("network_args", ["rank_dropout=0.1"]),
        ("dim_from_weights", True),
        ("scale_weight_norms", 1.0),
        ("base_weights", ["base.safetensors"]),
        ("base_weights_multiplier", [0.5]),
    ],
)
def test_validate_full_finetune_args_rejects_non_default_lora_options(name, value):
    args = make_full_finetune_args(**{name: value})

    with pytest.raises(ValueError, match=f"--{name}"):
        validate_full_finetune_args(args, num_processes=1)


@pytest.mark.parametrize("name", ["fp8_base", "fp8_scaled"])
def test_validate_full_finetune_args_rejects_fp8_flags(name):
    args = make_full_finetune_args(**{name: True})

    with pytest.raises(ValueError, match=f"--{name}"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_h2d_only_swap():
    args = make_full_finetune_args(block_swap_h2d_only=True)

    with pytest.raises(ValueError, match="--block_swap_h2d_only"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_multi_process_block_swap():
    args = make_full_finetune_args(blocks_to_swap=1)

    with pytest.raises(ValueError, match="--blocks_to_swap"):
        validate_full_finetune_args(args, num_processes=2)


def test_validate_full_finetune_args_rejects_fused_backward_for_non_adafactor():
    args = make_full_finetune_args(fused_backward_pass=True, optimizer_type="AdamW")

    with pytest.raises(ValueError, match="--fused_backward_pass"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_multi_process_fused_backward():
    args = make_full_finetune_args(fused_backward_pass=True, optimizer_type="Adafactor")

    with pytest.raises(ValueError, match="--fused_backward_pass"):
        validate_full_finetune_args(args, num_processes=2)


def test_validate_full_finetune_args_rejects_accumulated_fused_backward():
    args = make_full_finetune_args(
        fused_backward_pass=True,
        optimizer_type="Adafactor",
        gradient_accumulation_steps=2,
    )

    with pytest.raises(ValueError, match="--fused_backward_pass"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_fp16_fused_backward():
    args = make_full_finetune_args(
        fused_backward_pass=True,
        optimizer_type="Adafactor",
        mixed_precision="fp16",
    )

    with pytest.raises(ValueError, match=r"--fused_backward_pass.*--mixed_precision fp16"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_negative_block_swap_count():
    args = make_full_finetune_args(blocks_to_swap=-1)

    with pytest.raises(ValueError, match="--blocks_to_swap"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_mismatched_save_precision():
    args = make_full_finetune_args(save_precision="bf16")

    with pytest.raises(ValueError, match="--save_precision"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_full_bf16_without_bf16_mixed_precision():
    args = make_full_finetune_args(full_bf16=True, mixed_precision="no")

    with pytest.raises(ValueError, match="--full_bf16"):
        validate_full_finetune_args(args, num_processes=1)


def test_validate_full_finetune_args_rejects_full_fp16_injected_namespace():
    args = make_full_finetune_args(full_fp16=True)

    with pytest.raises(ValueError, match="--full_fp16"):
        validate_full_finetune_args(args, num_processes=1)


@pytest.mark.parametrize("dit_dtype", ["float16", "bfloat16"])
def test_validate_full_finetune_args_rejects_mismatched_dit_dtype(dit_dtype):
    args = make_full_finetune_args(dit_dtype=dit_dtype)

    with pytest.raises(ValueError, match=r"--dit_dtype.*--full_bf16"):
        validate_full_finetune_args(args, num_processes=1)


@pytest.mark.parametrize(
    ("dit_dtype", "full_bf16", "mixed_precision"),
    [
        (None, False, None),
        ("float32", False, "no"),
        ("fp32", False, "no"),
        ("bfloat16", True, "bf16"),
        ("bf16", True, "bf16"),
    ],
)
def test_validate_full_finetune_args_accepts_matching_dit_dtype(dit_dtype, full_bf16, mixed_precision):
    args = make_full_finetune_args(
        dit_dtype=dit_dtype,
        full_bf16=full_bf16,
        mixed_precision=mixed_precision,
    )

    validate_full_finetune_args(args, num_processes=1)


@pytest.mark.parametrize(
    ("overrides", "num_processes"),
    [
        ({}, 1),
        ({"full_bf16": True, "mixed_precision": "bf16"}, 1),
        ({"blocks_to_swap": 1}, 1),
        ({"fused_backward_pass": True, "optimizer_type": "Adafactor"}, 1),
    ],
)
def test_validate_full_finetune_args_accepts_supported_combinations(overrides, num_processes):
    args = make_full_finetune_args(**overrides)

    validate_full_finetune_args(args, num_processes)


def _run_validation_only_entrypoint(monkeypatch, module_name, trainer_name, argv):
    module = importlib.import_module(module_name)
    captured = []

    class ValidationOnlyTrainer:
        def train(self, args):
            captured.append(args)
            validate_full_finetune_args(args, num_processes=1)

    monkeypatch.setattr(module, trainer_name, ValidationOnlyTrainer)
    monkeypatch.setattr(sys, "argv", [module_name, *argv])
    module.main()
    return captured[0]


@pytest.mark.parametrize(
    ("module_name", "trainer_name"),
    [
        ("musubi_tuner.flux_kontext_train", "FluxKontextTrainer"),
        ("musubi_tuner.flux_2_train", "Flux2Trainer"),
        ("musubi_tuner.ideogram4_train", "Ideogram4Trainer"),
        ("musubi_tuner.krea2_train", "Krea2Trainer"),
    ],
)
def test_full_entrypoints_preserve_matching_toml_dit_dtype(
    tmp_path,
    monkeypatch,
    module_name,
    trainer_name,
):
    config = tmp_path / "matching.toml"
    config.write_text(
        'dit_dtype = "bfloat16"\nfull_bf16 = true\nmixed_precision = "bf16"\n',
        encoding="utf-8",
    )

    args = _run_validation_only_entrypoint(
        monkeypatch,
        module_name,
        trainer_name,
        ["--config_file", str(config)],
    )

    assert args.dit_dtype == "bfloat16"


def test_full_entrypoint_rejects_mismatched_toml_dit_dtype(tmp_path, monkeypatch):
    config = tmp_path / "mismatched.toml"
    config.write_text('dit_dtype = "bfloat16"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"--dit_dtype.*--full_bf16"):
        _run_validation_only_entrypoint(
            monkeypatch,
            "musubi_tuner.flux_kontext_train",
            "FluxKontextTrainer",
            ["--config_file", str(config)],
        )


def test_full_entrypoint_rejects_toml_injected_full_fp16(tmp_path, monkeypatch):
    config = tmp_path / "full-fp16.toml"
    config.write_text("full_fp16 = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="--full_fp16"):
        _run_validation_only_entrypoint(
            monkeypatch,
            "musubi_tuner.flux_kontext_train",
            "FluxKontextTrainer",
            ["--config_file", str(config)],
        )


def test_ideogram_full_entrypoint_rejects_mismatched_cli_dit_dtype(monkeypatch):
    with pytest.raises(ValueError, match=r"--dit_dtype.*--full_bf16"):
        _run_validation_only_entrypoint(
            monkeypatch,
            "musubi_tuner.ideogram4_train",
            "Ideogram4Trainer",
            ["--dit_dtype", "float16"],
        )


def test_ideogram_full_entrypoint_preserves_matching_cli_dit_dtype(monkeypatch):
    args = _run_validation_only_entrypoint(
        monkeypatch,
        "musubi_tuner.ideogram4_train",
        "Ideogram4Trainer",
        ["--dit_dtype", "bfloat16", "--full_bf16", "--mixed_precision", "bf16"],
    )

    assert args.dit_dtype == "bfloat16"
