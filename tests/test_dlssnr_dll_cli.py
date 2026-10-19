import importlib
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", ["dlssnr_unpack_model.py", "dlssnr_pack_model.py"])
def test_root_dll_scripts_are_runnable(script):
    result = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output" in result.stdout


def test_pack_parser_keeps_mix_strength_and_lora_multiplier_independent():
    module = importlib.import_module("musubi_tuner.dlssnr_pack_model")
    args = module.setup_parser().parse_args(
        [
            "--template_dll",
            "source.dll",
            "--input",
            "base",
            "--output",
            "export.dll",
            "--mix",
            "0.3",
            "--strength",
            "2",
            "--merge_lora",
            "adapter.safetensors",
            "--lora_multiplier",
            "0.7",
        ]
    )
    assert args.model_dir == "base" and args.output_dll == "export.dll"
    assert (args.mix, args.strength, args.lora_multiplier) == (0.3, 2, 0.7)
    assert args.merge_lora == "adapter.safetensors"


def test_pack_parser_defaults_preserve_full_checkpoint_values():
    module = importlib.import_module("musubi_tuner.dlssnr_pack_model")
    args = module.setup_parser().parse_args(["--template_dll", "source.dll", "--model_dir", "trained", "--output_dll", "out.dll"])
    assert (args.mix, args.strength, args.lora_multiplier) == (1, 1, 1)
    assert args.merge_lora is None


def test_unpack_cli_reports_invalid_input_without_writing_output(tmp_path):
    bad = tmp_path / "bad.dll"
    bad.write_bytes(b"not a DLL")
    output = tmp_path / "model"
    result = subprocess.run(
        [sys.executable, str(ROOT / "dlssnr_unpack_model.py"), "--input", str(bad), "--output", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "PE DLL" in result.stderr
    assert not output.exists()


def test_standalone_lora_merge_exposes_the_same_multiplier():
    result = subprocess.run([sys.executable, str(ROOT / "dlssnr_merge_lora.py"), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--lora_multiplier" in result.stdout
