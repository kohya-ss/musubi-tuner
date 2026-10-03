import importlib
import json
import sys
from pathlib import Path

import pytest
import toml
import torch


def dataset_config(tmp_path, *, extra=None):
    value = {
        "general": {"resolution": [48, 48], "batch_size": 1},
        "datasets": [{"train_manifest": "data.jsonl"}],
    }
    if extra:
        value.update(extra)
    path = tmp_path / "dataset.toml"
    path.write_text(toml.dumps(value), encoding="utf-8")
    return path


def parse_config(tmp_path, *options, lora=False):
    from musubi_tuner.dlssnr.config import build_train_config

    entry = importlib.import_module("musubi_tuner.dlssnr_train_network" if lora else "musubi_tuner.dlssnr_train")
    args = entry.setup_parser().parse_args(
        [
            "--dataset_config",
            str(dataset_config(tmp_path)),
            "--development_smoke",
            "--output_dir",
            "output",
            "--output_name",
            "run",
            *options,
        ]
    )
    return build_train_config(args, lora=lora)


@pytest.mark.parametrize("module", ["dlssnr_train", "dlssnr_train_network"])
def test_help_exposes_training_arguments_not_a_required_training_toml(module, monkeypatch, capsys):
    entry = importlib.import_module(f"musubi_tuner.{module}")
    monkeypatch.setattr(sys, "argv", [module, "--help"])
    with pytest.raises(SystemExit) as result:
        entry.main()
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for option in ("--dataset_config", "--model_dir", "--optimizer_type", "--optimizer_args", "--learning_rate", "--output_name"):
        assert option in help_text
    assert "--config_file" not in help_text
    if module.endswith("train_network"):
        assert "--network_dim" in help_text and "--network_alpha" in help_text


def test_training_and_network_arguments_are_resolved_from_cli(tmp_path):
    config = parse_config(
        tmp_path,
        "--optimizer_type",
        "SGD",
        "--optimizer_args",
        "momentum=0.9",
        "weight_decay=0.02",
        "--learning_rate",
        "0.003",
        "--network_dim",
        "8",
        "--network_alpha",
        "4",
        "--network_dropout",
        "0.1",
        "--max_train_steps",
        "12",
        "--gradient_accumulation_steps",
        "3",
        lora=True,
    )
    assert config["optimizer"]["type"] == "SGD"
    assert config["optimizer"]["learning_rate"] == 0.003
    assert config["lora"]["rank"] == 8
    assert config["lora"]["alpha"] == 4
    assert config["lora"]["dropout"] == 0.1
    assert config["training"]["max_train_steps"] == 12
    assert config["training"]["gradient_accumulation_steps"] == 3
    assert config["training"]["batch_size"] == 1


def test_dataset_paths_and_cli_paths_use_their_own_base_directories(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    config = parse_config(tmp_path, "--model_dir", "models/base", "--forward_validation_report", "reports/forward.json")
    assert Path(config["data"]["train_manifest"]) == tmp_path / "data.jsonl"
    assert Path(config["model"]["model_dir"]) == work / "models/base"
    assert Path(config["model"]["forward_validation_report"]) == work / "reports/forward.json"
    assert Path(config["output"]["output_dir"]) == work / "output"


@pytest.mark.parametrize("section", ["training", "optimizer", "model", "lora", "output", "loss"])
def test_dataset_toml_rejects_training_tables(tmp_path, section):
    from musubi_tuner.dlssnr.config import load_dataset_config

    path = dataset_config(tmp_path, extra={section: {"learning_rate": 0.2}})
    with pytest.raises(ValueError, match="dataset|Dataset"):
        load_dataset_config(path)


@pytest.mark.parametrize("lora", [False, True])
def test_cli_runs_selected_optimizer_and_saves_effective_arguments(tmp_path, monkeypatch, lora):
    from accelerate.state import AcceleratorState
    from musubi_tuner.networks import lora_dlssnr
    from musubi_tuner.training import dlssnr_trainer
    from test_dlssnr_dataset import write_manifest
    from test_dlssnr_training import SmallNR, small_inject

    AcceleratorState._reset_state(reset_partial_state=True)
    monkeypatch.setattr(dlssnr_trainer, "NRModel", SmallNR)
    monkeypatch.setattr(lora_dlssnr, "inject", small_inject)
    write_manifest(tmp_path)
    module = "dlssnr_train_network" if lora else "dlssnr_train"
    argv = [
        module,
        "--dataset_config",
        str(dataset_config(tmp_path)),
        "--development_smoke",
        "--device",
        "cpu",
        "--output_dir",
        str(tmp_path / "output"),
        "--output_name",
        "run",
        "--max_train_steps",
        "1",
        "--save_state",
        "--optimizer_type",
        "SGD",
        "--learning_rate",
        "0.01",
        "--optimizer_args",
        "momentum=0.9",
        "weight_decay=0.0",
    ]
    if lora:
        argv.extend(["--network_dim", "2", "--network_alpha", "2"])
    monkeypatch.setattr(sys, "argv", argv)
    try:
        importlib.import_module(f"musubi_tuner.{module}").main()
        run = tmp_path / "output" / "run"
        state = torch.load(run / "state-step000001/trainer_state.pt", weights_only=True)
        assert all(group["momentum"] == 0.9 for group in state["optimizer"]["param_groups"])
        assert any("momentum_buffer" in value for value in state["optimizer"]["state"].values())
        config = json.loads((run / "run_config.json").read_text())["config"]
        assert config["optimizer"]["type"] == "SGD"
        assert config["optimizer"]["learning_rate"] == 0.01
        if lora:
            assert config["lora"]["rank"] == 2
        assert (run / "final" / ("adapter.safetensors" if lora else "model.safetensors")).is_file()
    finally:
        AcceleratorState._reset_state(reset_partial_state=True)
